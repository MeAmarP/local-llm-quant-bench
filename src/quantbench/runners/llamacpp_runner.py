# Objective: Implement a llama.cpp backend runner.

from __future__ import annotations

import re
import subprocess
from typing import Any, Mapping

from ..metrics import MetricsHelper
from ..models import PromptCase, RunResult, RunSpec
from ..utils.logging_utils import get_logger
from .base import BaseRunner

DEFAULT_GENERATION_CONFIG: dict[str, Any] = {
    "max_new_tokens": 128,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,  # Informational in llama.cpp mode; temp/top_p govern decoding behavior.
    "repetition_penalty": 1.0,
    "seed": 42,
}


class LlamaCppRunner(BaseRunner):
    """Run prompt cases through a local llama.cpp CLI binary."""

    def __init__(
        self,
        run_spec: RunSpec,
        *,
        generation_config: Mapping[str, Any] | None = None,
        # ! TODO - I have 2 versions of llama-cli on my machine: llama-cli-cpu and llama-cli-cuda
        executable: str = "llama-cli",
        timeout_sec: float | None = None,
        measure_gpu_memory: bool = False,
    ) -> None:
        super().__init__(run_spec, generation_config=generation_config)
        self.executable = executable
        self.timeout_sec = timeout_sec
        self.logger = get_logger(__name__)
        self.metrics = MetricsHelper(measure_gpu_memory=measure_gpu_memory)
        self._effective_generation = self._build_generation_config(self.generation_config)

        if not self.run_spec.model_path:
            raise ValueError("LlamaCppRunner requires run_spec.model_path to point to a GGUF model.")

    def run_case(self, prompt_case: PromptCase) -> RunResult:
        """Execute one prompt case using llama.cpp and return standardized result payload."""
        prompt_text = prompt_case.prompt
        model_ref = self.run_spec.model_path or self.run_spec.model_id or "unknown_model_ref"
        started = self.metrics.start()
        cmd = self._build_command(prompt_text)

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except FileNotFoundError as exc:
            return self._error_result(
                prompt_case=prompt_case,
                model_ref=model_ref,
                started=started,
                error=f"llama.cpp executable not found: {self.executable} ({exc})",
            )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "").strip()
            return self._error_result(
                prompt_case=prompt_case,
                model_ref=model_ref,
                started=started,
                generated_text=partial,
                error=f"llama.cpp execution timed out after {self.timeout_sec}s",
            )
        except Exception as exc:  # defensive guard for unexpected subprocess failures
            return self._error_result(
                prompt_case=prompt_case,
                model_ref=model_ref,
                started=started,
                error=f"llama.cpp execution failed: {exc}",
            )

        generated_text = self._extract_generated_text(prompt_text, completed.stdout)
        load_time_ms = self._parse_model_load_time_ms(completed.stdout, completed.stderr)
        metrics = self.metrics.capture(
            prompt_text=prompt_text,
            generated_text=generated_text,
            generated_tokens=None,
            model_load_time_ms=load_time_ms,
        )

        error: str | None = None
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            details = stderr or stdout or "unknown llama.cpp error"
            error = f"llama.cpp exited with code {completed.returncode}: {details}"

        return RunResult(
            run_name=self.run_spec.name,
            backend=self.run_spec.backend,
            quantization=self.run_spec.quantization,
            prompt_id=prompt_case.id,
            task=prompt_case.task,
            model_ref=model_ref,
            prompt_chars=len(prompt_text),
            prompt_tokens=metrics["prompt_tokens"],
            output_tokens=metrics["output_tokens"],
            latency_sec=(metrics["wall_clock_latency_ms"] or 0.001) / 1000.0,
            tokens_per_sec=metrics["tokens_per_sec"],
            load_time_sec=(metrics["model_load_time_ms"] or 0.0) / 1000.0 if metrics["model_load_time_ms"] is not None else None,
            peak_gpu_mem_mb=metrics["peak_gpu_memory_mb"],
            generated_text=generated_text,
            error=error,
            extra={
                "command": cmd,
                "returncode": completed.returncode,
                "stderr": (completed.stderr or "").strip(),
            },
        )

    def _build_generation_config(self, user_config: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(DEFAULT_GENERATION_CONFIG)
        merged.update(dict(user_config))
        return merged

    def _build_command(self, prompt: str) -> list[str]:
        cfg = self._effective_generation
        model_path = self.run_spec.model_path
        assert model_path is not None

        cmd = [
            self.executable,
            "-m",
            model_path,
            "-p",
            prompt,
            "-n",
            str(int(cfg["max_new_tokens"])),
            "--temp",
            str(float(cfg["temperature"])),
            "--top-p",
            str(float(cfg["top_p"])),
            "--repeat-penalty",
            str(float(cfg["repetition_penalty"])),
            "--seed",
            str(int(cfg["seed"])),
            "--simple-io",
            "--no-display-prompt",
        ]

        stop_sequences = cfg.get("stop") or cfg.get("stop_sequences")
        if isinstance(stop_sequences, list):
            for stop in stop_sequences:
                if isinstance(stop, str) and stop:
                    # Newer llama.cpp uses `--stop`.
                    cmd.extend(["--stop", stop])

        return cmd

    @staticmethod
    def _extract_generated_text(prompt: str, stdout: str | None) -> str:
        text = (stdout or "").strip()
        if text.startswith(prompt):
            return text[len(prompt) :].lstrip()
        return text

    @staticmethod
    def _parse_model_load_time_ms(stdout: str | None, stderr: str | None) -> float | None:
        combined = "\n".join([(stdout or ""), (stderr or "")])
        # Typical llama.cpp log snippet: "load time = 1234.56 ms"
        match = re.search(r"load\s+time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", combined, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _error_result(
        self,
        *,
        prompt_case: PromptCase,
        model_ref: str,
        started: float,
        error: str,
        generated_text: str = "",
    ) -> RunResult:
        del started  # already tracked by self.metrics.start()
        metrics = self.metrics.capture(
            prompt_text=prompt_case.prompt,
            generated_text=generated_text,
            generated_tokens=None,
            model_load_time_ms=None,
        )
        output_tokens = metrics["output_tokens"] or 0

        return RunResult(
            run_name=self.run_spec.name,
            backend=self.run_spec.backend,
            quantization=self.run_spec.quantization,
            prompt_id=prompt_case.id,
            task=prompt_case.task,
            model_ref=model_ref,
            prompt_chars=len(prompt_case.prompt),
            prompt_tokens=metrics["prompt_tokens"],
            output_tokens=output_tokens,
            latency_sec=(metrics["wall_clock_latency_ms"] or 0.001) / 1000.0,
            tokens_per_sec=0.0 if output_tokens == 0 else None,
            load_time_sec=None,
            peak_gpu_mem_mb=metrics["peak_gpu_memory_mb"],
            generated_text=generated_text,
            error=error,
            extra={"command": self.executable},
        )
