# Objective: Provide helpers to compute and capture benchmark metrics.

from __future__ import annotations

import importlib
import time
from typing import Any, Callable


METRIC_NAMES = (
    "wall_clock_latency_ms",
    "generated_tokens",
    "tokens_per_sec",
    "prompt_tokens",
    "output_tokens",
    "peak_gpu_memory_mb",
    "model_load_time_ms",
)


def supported_metrics() -> list[str]:
    """Return the Phase 1 benchmark metric names."""
    return list(METRIC_NAMES)


class MetricsHelper:
    """Compute/capture benchmark metrics for one generation call."""

    def __init__(
        self,
        tokenizer: Any | Callable[[str], Any] | None = None,
        measure_gpu_memory: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.measure_gpu_memory = measure_gpu_memory
        self._step_start: float | None = None
        self._torch_module: Any | None = None
        self._torch_checked = False

    def start(self) -> float:
        """
        Mark the beginning of an inference step.

        Returns the high-resolution timestamp so callers can store it if needed.
        """
        self._step_start = time.perf_counter()
        self._reset_peak_gpu_memory()
        return self._step_start

    def capture(
        self,
        prompt_text: str,
        generated_text: str,
        *,
        generated_tokens: int | None = None,
        model_load_time_ms: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, int | float | None]:
        """
        Capture a standard metric payload for the current step.

        Call `start()` before `capture()`.
        """
        if self._step_start is None:
            raise RuntimeError("MetricsHelper.start() must be called before capture().")

        finish = end_time if end_time is not None else time.perf_counter()
        wall_clock_latency_ms = self.compute_wall_clock_latency_ms(self._step_start, finish)

        prompt_tokens = self.count_tokens(prompt_text)
        output_tokens = self.count_tokens(generated_text)

        if generated_tokens is None:
            generated_tokens = output_tokens

        tokens_per_sec = self.compute_tokens_per_sec(
            generated_tokens=generated_tokens,
            wall_clock_latency_ms=wall_clock_latency_ms,
        )

        return {
            "wall_clock_latency_ms": round(wall_clock_latency_ms, 4),
            "generated_tokens": generated_tokens,
            "tokens_per_sec": round(tokens_per_sec, 4) if tokens_per_sec is not None else None,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb(),
            "model_load_time_ms": model_load_time_ms,
        }

    @staticmethod
    def compute_wall_clock_latency_ms(start_time: float, end_time: float) -> float:
        """Return wall-clock latency in milliseconds."""
        latency_ms = (end_time - start_time) * 1000.0
        if latency_ms <= 0:
            raise ValueError("Computed wall-clock latency must be > 0.")
        return latency_ms

    @staticmethod
    def compute_tokens_per_sec(
        *,
        generated_tokens: int,
        wall_clock_latency_ms: float,
    ) -> float | None:
        """Compute generation throughput from generated token count and latency."""
        if generated_tokens < 0:
            raise ValueError("generated_tokens must be >= 0.")
        if generated_tokens == 0:
            return 0.0
        if wall_clock_latency_ms <= 0:
            return None
        return generated_tokens / (wall_clock_latency_ms / 1000.0)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using a provided tokenizer when possible.

        Fallback when tokenizer is unavailable:
        - whitespace token estimate using `str.split()`
        """
        if not text:
            return 0

        tokenizer = self.tokenizer
        if tokenizer is None:
            return len(text.split())

        # Hugging Face-style tokenizer API.
        if hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return len(tokenizer.encode(text))

        # Callable tokenizer support.
        if callable(tokenizer):
            output = tokenizer(text)
            if isinstance(output, dict):
                ids = output.get("input_ids")
                if isinstance(ids, list):
                    return len(ids)
            if isinstance(output, list):
                return len(output)

        # Last-resort estimate.
        return len(text.split())

    def peak_gpu_memory_mb(self) -> float | None:
        """
        Return peak GPU memory in MB since last `start()`, when CUDA is available.
        """
        if not self.measure_gpu_memory:
            return None
        torch = self._get_torch()
        if torch is None or not torch.cuda.is_available():
            return None
        try:
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            return round(peak_bytes / (1024.0 * 1024.0), 4)
        except Exception:
            return None

    def _reset_peak_gpu_memory(self) -> None:
        if not self.measure_gpu_memory:
            return
        torch = self._get_torch()
        if torch is None or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            # Non-fatal for CPU-only or partially initialized CUDA setups.
            return

    def _get_torch(self) -> Any | None:
        if self._torch_checked:
            return self._torch_module
        self._torch_checked = True
        try:
            self._torch_module = importlib.import_module("torch")
        except Exception:
            self._torch_module = None
        return self._torch_module

