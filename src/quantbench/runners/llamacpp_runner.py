# Objective: Implement a llama.cpp backend runner.

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any, Mapping, Optional

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
    """Run prompt cases through a local llama.cpp CLI binary.
    
    Supports both CPU and CUDA variants:
    - llama-cli or llama-cli-cpu for CPU inference
    - llama-cli-cuda for CUDA GPU inference
    
    Automatically selects the appropriate executable based on device configuration.
    """

    def __init__(
        self,
        run_spec: Optional[RunSpec],
        *,
        generation_config: Mapping[str, Any] | None = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        executable: Optional[str] = None,
        timeout_sec: float | None = None,
        n_gpu_layers: Optional[int] = None,
        n_ctx: Optional[int] = None,
    ) -> None:
        """Initialize LlamaCppRunner.
        
        Args:
            run_spec: RunSpec with model path and backend info (can be None if model_path provided)
            generation_config: Generation parameters (temperature, max_tokens, etc.)
            model_path: Override model path from run_spec
            device: Device selection ("auto", "cpu", "cuda"). Determines executable variant.
            executable: Explicit executable name. If provided, overrides device-based selection.
            timeout_sec: Timeout for subprocess execution in seconds
        """
        super().__init__(run_spec or RunSpec(name="llamacpp", backend="llamacpp", quantization="unknown"), generation_config=generation_config)
        
        self.device = device.lower() if device else "auto"
        self.timeout_sec = timeout_sec
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.logger = get_logger(__name__)
        self.metrics = MetricsHelper()
        self._effective_generation = self._build_generation_config(self.generation_config)

        # Determine model path
        self.model_path = model_path or (self.run_spec.model_path if self.run_spec else None)
        if not self.model_path:
            raise ValueError("LlamaCppRunner requires model_path (either from run_spec or parameter).")

        # Determine and validate executable
        self.executable = self._resolve_executable(executable)
        self.logger.info(f"Using llama.cpp executable: {self.executable}")

    def _resolve_executable(self, explicit_executable: Optional[str]) -> str:
        """Resolve which llama-cli executable to use.
        
        Priority:
        1. Explicit executable parameter
        2. Device-based selection (cuda → llama-cli-cuda, cpu → llama-cli-cpu)
        3. Auto-detection: try CUDA first, fallback to CPU
        
        Args:
            explicit_executable: User-provided executable name
            
        Returns:
            Resolved executable name
            
        Raises:
            FileNotFoundError: If no suitable executable found
        """
        # Option 1: Use explicit executable if provided
        if explicit_executable:
            if self._executable_exists(explicit_executable):
                return explicit_executable
            raise FileNotFoundError(
                f"Specified llama-cli executable not found in PATH: {explicit_executable}"
            )

        # Option 2: Device-based selection
        if self.device == "cuda":
            if self._executable_exists("llama-cli-cuda"):
                return "llama-cli-cuda"
            self.logger.warning(
                "Device=cuda but llama-cli-cuda not found in PATH; trying llama-cli"
            )
            if self._executable_exists("llama-cli"):
                return "llama-cli"
            raise FileNotFoundError(
                "Neither llama-cli-cuda nor llama-cli found in PATH for CUDA device"
            )

        elif self.device == "cpu":
            candidates = ["llama-cli-cpu", "llama-cli"]
            for candidate in candidates:
                if self._executable_exists(candidate):
                    return candidate
            raise FileNotFoundError(
                f"No llama-cli CPU variant found in PATH. Tried: {candidates}"
            )

        # Option 3: Auto-detection
        else:  # device == "auto"
            # Try CUDA first (more likely to be available if GPU present)
            for candidate in ["llama-cli-cuda", "llama-cli-cpu", "llama-cli"]:
                if self._executable_exists(candidate):
                    self.logger.info(f"Auto-detected llama.cpp executable: {candidate}")
                    return candidate

            raise FileNotFoundError(
                "No llama-cli executable found in PATH. "
                "Tried: llama-cli-cuda, llama-cli-cpu, llama-cli. "
                "Please install llama.cpp or ensure it is in your PATH."
            )

    @staticmethod
    def _executable_exists(executable_name: str) -> bool:
        """Check if executable exists in PATH.
        
        Args:
            executable_name: Name of executable to check
            
        Returns:
            True if executable is found in PATH, False otherwise
        """
        return shutil.which(executable_name) is not None

    def load(self) -> None:
        """Validate model path and executable readiness."""
        if not self.model_path:
            raise RuntimeError("Model path not set")
        
        # Check model file exists
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.logger.info(f"LlamaCppRunner ready: {self.executable} with model {self.model_path}")

    def unload(self) -> None:
        """Clean up resources."""
        pass  # llama.cpp subprocess is ephemeral; nothing to clean up

    def run_case(self, prompt_case: PromptCase) -> RunResult:
        """Execute one prompt case using llama.cpp and return standardized result payload."""
        prompt_text = prompt_case.prompt
        model_ref = self.model_path or self.run_spec.model_path or "unknown_model_ref"
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
        # Prefer the native t/s reported by llama.cpp over the wall-clock estimate
        native_tps = self._parse_generation_tps(completed.stdout)
        # TTFT: llama.cpp logs "prompt eval time = X ms" which is the time to process the prompt
        ttft_ms = self._parse_prompt_eval_time_ms(completed.stdout, completed.stderr)
        metrics = self.metrics.capture(
            prompt_text=prompt_text,
            generated_text=generated_text,
            generated_tokens=None,
            model_load_time_ms=load_time_ms,
            ttft_ms=ttft_ms,
        )
        if native_tps is not None:
            metrics["tokens_per_sec"] = native_tps

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
            ttft_ms=metrics.get("ttft_ms"),
            peak_ram_mb=metrics.get("peak_ram_mb"),
            avg_power_w=metrics.get("avg_power_w"),
            energy_per_token_j=metrics.get("energy_per_token_j"),
            extra={
                "command": cmd,
                "returncode": completed.returncode,
                "stderr": (completed.stderr or "").strip(),
                "executable": self.executable,
                "device": self.device,
            },
        )

    def _build_generation_config(self, user_config: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(DEFAULT_GENERATION_CONFIG)
        merged.update(dict(user_config))
        return merged

    def _build_command(self, prompt: str) -> list[str]:
        cfg = self._effective_generation
        model_path = self.model_path
        if model_path is None:
            raise RuntimeError("LlamaCppRunner: model_path is not set")

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
            "--single-turn",
            "--no-display-prompt",
        ]

        if self.n_gpu_layers is not None:
            cmd.extend(["--n-gpu-layers", str(self.n_gpu_layers)])
        if self.n_ctx is not None:
            cmd.extend(["--ctx-size", str(self.n_ctx)])

        stop_sequences = cfg.get("stop") or cfg.get("stop_sequences")
        if isinstance(stop_sequences, list):
            for stop in stop_sequences:
                if isinstance(stop, str) and stop:
                    # Newer llama.cpp uses `--stop`.
                    cmd.extend(["--stop", stop])

        return cmd

    @staticmethod
    def _extract_generated_text(prompt: str, stdout: str | None) -> str:
        text = stdout or ""
        # New llama.cpp (b4000+) chat UI: generated text appears after "> <prompt>\n\n"
        # and before the stats line "[ Prompt: X t/s | Generation: Y t/s ]"
        chat_match = re.search(r"> [^\n]*\n\n(.*?)(?:\n\n\[|\nExiting)", text, re.DOTALL)
        if chat_match:
            return chat_match.group(1).strip()
        # Fallback: old single-shot format where stdout starts with the prompt
        stripped = text.strip()
        if stripped.startswith(prompt):
            return stripped[len(prompt):].lstrip()
        return stripped

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

    @staticmethod
    def _parse_prompt_eval_time_ms(stdout: str | None, stderr: str | None) -> float | None:
        """Parse prompt evaluation time (TTFT proxy) from llama.cpp timing logs.

        e.g. "llama_print_timings: prompt eval time =  1234.56 ms /  16 tokens"
        """
        combined = "\n".join([(stdout or ""), (stderr or "")])
        match = re.search(
            r"prompt\s+eval\s+time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
            combined,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _parse_generation_tps(stdout: str | None) -> float | None:
        """Parse generation tokens/sec from the new llama.cpp stats line.

        e.g. "[ Prompt: 173.3 t/s | Generation: 54.3 t/s ]"
        """
        if not stdout:
            return None
        match = re.search(r"Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s", stdout)
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
            extra={"command": self.executable, "device": self.device},
        )

