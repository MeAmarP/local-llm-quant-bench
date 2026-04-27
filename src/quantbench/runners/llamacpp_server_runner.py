# Objective: Implement a llama.cpp server-mode backend runner.
#
# The runner starts `llama-server` as a subprocess, waits until its /health
# endpoint returns 200 OK, then serves each PromptCase via HTTP POST to
# /completion.  On unload() the subprocess is terminated and the port is freed.
#
# Why server mode?
# - Avoids model reload between prompt runs → faster multi-prompt benchmarks.
# - Exposes a richer JSON response with per-request timing breakdown.
# - Compatible with the OpenAI-compatible /v1/chat/completions endpoint.

from __future__ import annotations

import json
import logging
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional

from ..metrics import MetricsHelper
from ..models import PromptCase, RunResult, RunSpec
from .base import BaseRunner

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080
_HEALTH_TIMEOUT_SEC = 120.0
_HEALTH_POLL_INTERVAL = 0.5


def _find_free_port(preferred: int = _DEFAULT_PORT) -> int:
    """Return an available TCP port, preferring *preferred*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((_DEFAULT_HOST, preferred))
            return preferred
        except OSError:
            s.bind((_DEFAULT_HOST, 0))
            return s.getsockname()[1]


class LlamaCppServerRunner(BaseRunner):
    """Run prompt cases through a locally managed llama-server HTTP process.

    The runner lifecycle:

    1. ``load()``   — spawns ``llama-server``, waits for /health
    2. ``run_case()`` — POSTs each prompt to /completion, reads timings
    3. ``unload()``  — terminates the server process

    Timing fields extracted from llama-server's ``timings`` response block:

    * ``ttft_ms``         → ``timings.prompt_ms``  (prompt-eval wall time ≈ TTFT)
    * ``tokens_per_sec``  → ``timings.predicted_per_second``
    * ``prompt_tokens``   → ``timings.prompt_n``
    * ``output_tokens``   → ``timings.predicted_n``
    """

    def __init__(
        self,
        run_spec: Optional[RunSpec],
        *,
        generation_config: Mapping[str, Any] | None = None,
        model_path: Optional[str] = None,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        n_gpu_layers: Optional[int] = None,
        n_ctx: Optional[int] = None,
        executable: Optional[str] = None,
        server_extra_args: Optional[list[str]] = None,
        request_timeout_sec: float = 300.0,
    ) -> None:
        """Initialize LlamaCppServerRunner.

        Args:
            run_spec: RunSpec metadata for this benchmark run.
            generation_config: Generation parameters (max_new_tokens, temperature, …).
            model_path: Path to the GGUF model file.
            host: Interface the server listens on (default 127.0.0.1).
            port: Preferred port (an unused port is chosen if this one is busy).
            n_gpu_layers: Number of model layers to offload to GPU (-1 = all).
            n_ctx: Context window size in tokens.
            executable: Override the server binary name/path.
            server_extra_args: Additional raw CLI arguments forwarded to llama-server.
            request_timeout_sec: Per-request HTTP timeout in seconds.
        """
        super().__init__(run_spec, generation_config=generation_config)
        self.model_path = model_path
        self.host = host
        self._preferred_port = port
        self.port: int = port  # resolved on load()
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.executable = executable
        self.server_extra_args = server_extra_args or []
        self.request_timeout_sec = request_timeout_sec

        self._process: subprocess.Popen | None = None
        self._load_time_sec: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Start the llama-server subprocess and wait until it is ready."""
        if not self.model_path:
            raise ValueError("LlamaCppServerRunner requires model_path to be set")

        binary = self._resolve_executable()
        self.port = _find_free_port(self._preferred_port)

        cmd = self._build_server_cmd(binary)
        logger.info("Starting llama-server: %s", " ".join(cmd))

        t0 = time.perf_counter()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if not self._wait_for_health():
            stdout_tail = self._drain_stdout(lines=40)
            self._terminate()
            raise RuntimeError(
                f"llama-server did not become healthy within {_HEALTH_TIMEOUT_SEC}s.\n"
                f"Last output:\n{stdout_tail}"
            )

        self._load_time_sec = time.perf_counter() - t0
        logger.info(
            "llama-server ready on %s:%d (load %.2fs)", self.host, self.port, self._load_time_sec
        )

    def unload(self) -> None:
        """Terminate the server subprocess."""
        self._terminate()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_case(self, prompt_case: PromptCase) -> RunResult:
        """POST a completion request and return a populated RunResult.

        Args:
            prompt_case: The prompt to run.

        Returns:
            RunResult with latency, throughput, TTFT, and generated text.
        """
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Server is not running; call load() first")

        payload = self._build_request_payload(prompt_case.prompt)
        url = f"http://{self.host}:{self.port}/completion"

        server_pid = self._process.pid if self._process is not None else None
        metrics = MetricsHelper(
            use_smi_for_gpu_memory=True,  # llama-server is a subprocess; torch cannot track it
            target_pid=server_pid,  # track llama-server's own RSS, not quantbench's
        )
        metrics.start()

        raw_response = self._post_json(url, payload)

        captured = metrics.capture(
            prompt_text=prompt_case.prompt,
            generated_text=raw_response.get("content", ""),
            generated_tokens=raw_response.get("tokens_predicted"),
        )
        ttft_ms = self._extract_ttft_ms(raw_response)

        timings = raw_response.get("timings") or {}
        tokens_per_sec = (
            timings.get("predicted_per_second")
            or captured.get("tokens_per_sec")
        )
        prompt_tokens = timings.get("prompt_n") or captured.get("prompt_tokens")
        output_tokens = (
            timings.get("predicted_n")
            or raw_response.get("tokens_predicted")
            or captured.get("output_tokens")
        )
        latency_ms = captured["wall_clock_latency_ms"] or 0.0

        return RunResult(
            run_name=self.run_spec.name if self.run_spec else "llamacpp_server",
            backend="llamacpp_server",
            quantization=self.run_spec.quantization if self.run_spec else "unknown",
            prompt_id=prompt_case.id,
            task=prompt_case.task,
            model_ref=self.model_path or "unknown",
            prompt_chars=len(prompt_case.prompt),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_sec=latency_ms / 1000.0,
            tokens_per_sec=tokens_per_sec,
            load_time_sec=self._load_time_sec,
            peak_gpu_mem_mb=captured.get("peak_gpu_memory_mb"),
            generated_text=raw_response.get("content", ""),
            ttft_ms=ttft_ms,
            peak_ram_mb=captured.get("peak_ram_mb"),
            avg_power_w=captured.get("avg_power_w"),
            energy_per_token_j=captured.get("energy_per_token_j"),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_executable(self) -> str:
        """Find the llama-server binary, trying common names."""
        if self.executable:
            return self.executable
        for candidate in ("llama-server", "llama-server-cuda", "server"):
            if shutil.which(candidate):
                return candidate
        # Fall back; will fail at subprocess.Popen if missing
        return "llama-server"

    def _build_server_cmd(self, binary: str) -> list[str]:
        gen = self.generation_config
        cmd = [
            binary,
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.n_ctx or gen.get("n_ctx", 2048)),
            "--seed", str(gen.get("seed", 42)),
            "--log-disable",  # quieter; we check /health instead
        ]
        if self.n_gpu_layers is not None:
            cmd += ["--n-gpu-layers", str(self.n_gpu_layers)]
        cmd += self.server_extra_args
        return cmd

    def _build_request_payload(self, prompt: str) -> dict[str, Any]:
        gen = self.generation_config
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": gen.get("max_new_tokens", 128),
            "temperature": gen.get("temperature", 0.0),
            "top_p": gen.get("top_p", 1.0),
            "repeat_penalty": gen.get("repetition_penalty", 1.0),
            "seed": gen.get("seed", 42),
            "stream": False,
            "timings_per_token": False,
        }
        return payload

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON POST request and return the parsed response body."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server returned HTTP {exc.code}: {body}") from exc

    @staticmethod
    def _extract_ttft_ms(response: dict[str, Any]) -> float | None:
        """Extract TTFT from the timings block (prompt_ms ≈ time-to-first-token)."""
        timings = response.get("timings")
        if not isinstance(timings, dict):
            return None
        val = timings.get("prompt_ms")
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def _wait_for_health(self) -> bool:
        """Poll GET /health until 200 OK or timeout."""
        health_url = f"http://{self.host}:{self.port}/health"
        deadline = time.perf_counter() + _HEALTH_TIMEOUT_SEC
        while time.perf_counter() < deadline:
            if self._process is not None and self._process.poll() is not None:
                return False  # process exited early
            try:
                with urllib.request.urlopen(health_url, timeout=2.0) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL)
        return False

    def _terminate(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=10.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            logger.info("llama-server terminated")

    def _drain_stdout(self, lines: int = 40) -> str:
        """Read buffered stdout lines without blocking (best-effort)."""
        if self._process is None or self._process.stdout is None:
            return ""
        collected: list[str] = []
        try:
            for _ in range(lines):
                line = self._process.stdout.readline()
                if not line:
                    break
                collected.append(line.rstrip())
        except Exception:
            pass
        return "\n".join(collected)
