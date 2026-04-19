# Objective: Provide helpers to compute and capture benchmark metrics.

from __future__ import annotations

import importlib
import subprocess
import threading
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
    # Extended metrics (feature/extended-metrics)
    "ttft_ms",
    "peak_ram_mb",
    "avg_power_w",
    "energy_per_token_j",
)


def supported_metrics() -> list[str]:
    """Return the full set of benchmark metric names."""
    return list(METRIC_NAMES)


# ---------------------------------------------------------------------------
# Background GPU power sampler
# ---------------------------------------------------------------------------

class _PowerSampler:
    """Sample GPU power draw in a background thread via nvidia-smi.

    Usage::

        sampler = _PowerSampler(interval_s=0.2)
        sampler.start()
        ... # run inference
        sampler.stop()
        watts = sampler.mean_watts  # float | None
    """

    def __init__(self, interval_s: float = 0.2) -> None:
        self._interval = interval_s
        self._samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="PowerSampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def mean_watts(self) -> float | None:
        return sum(self._samples) / len(self._samples) if self._samples else None

    @property
    def total_samples(self) -> int:
        return len(self._samples)

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                )
                if result.returncode == 0:
                    line = result.stdout.strip().splitlines()[0]
                    self._samples.append(float(line.strip()))
            except Exception:
                pass  # silently skip failed samples (no GPU, no nvidia-smi, etc.)


class MetricsHelper:
    """Compute/capture benchmark metrics for one generation call."""

    def __init__(
        self,
        tokenizer: Any | Callable[[str], Any] | None = None,
        measure_gpu_memory: bool = True,
        measure_ram: bool = False,
        measure_power: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.measure_gpu_memory = measure_gpu_memory
        self.measure_ram = measure_ram
        self.measure_power = measure_power
        self._step_start: float | None = None
        self._torch_module: Any | None = None
        self._torch_checked = False
        self._ram_start_mb: float | None = None
        self._power_sampler: _PowerSampler | None = (
            _PowerSampler() if measure_power else None
        )

    def start(self) -> float:
        """
        Mark the beginning of an inference step.

        Returns the high-resolution timestamp so callers can store it if needed.
        """
        self._step_start = time.perf_counter()
        self._reset_peak_gpu_memory()
        if self.measure_ram:
            self._ram_start_mb = self._get_process_rss_mb()
        if self._power_sampler is not None:
            self._power_sampler.start()
        return self._step_start

    def capture(
        self,
        prompt_text: str,
        generated_text: str,
        *,
        generated_tokens: int | None = None,
        model_load_time_ms: float | None = None,
        end_time: float | None = None,
        ttft_ms: float | None = None,
    ) -> dict[str, int | float | None]:
        """
        Capture a standard metric payload for the current step.

        Call `start()` before `capture()`.

        Args:
            ttft_ms: Time-to-first-token in milliseconds, provided by the runner
                     (parsed from logs or measured via a LogitsProcessor hook).
        """
        if self._step_start is None:
            raise RuntimeError("MetricsHelper.start() must be called before capture().")

        # Stop power sampler before measuring end time for accuracy
        if self._power_sampler is not None:
            self._power_sampler.stop()

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

        # Peak RAM delta since start()
        peak_ram_mb: float | None = None
        if self.measure_ram and self._ram_start_mb is not None:
            current_rss = self._get_process_rss_mb()
            if current_rss is not None:
                peak_ram_mb = round(max(0.0, current_rss - self._ram_start_mb), 2)

        # Power and energy
        avg_power_w: float | None = None
        energy_per_token_j: float | None = None
        if self._power_sampler is not None:
            avg_power_w = self._power_sampler.mean_watts
            if avg_power_w is not None and tokens_per_sec and tokens_per_sec > 0:
                energy_per_token_j = round(avg_power_w / tokens_per_sec, 6)

        return {
            "wall_clock_latency_ms": round(wall_clock_latency_ms, 4),
            "generated_tokens": generated_tokens,
            "tokens_per_sec": round(tokens_per_sec, 4) if tokens_per_sec is not None else None,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb(),
            "model_load_time_ms": model_load_time_ms,
            # Extended metrics
            "ttft_ms": round(ttft_ms, 4) if ttft_ms is not None else None,
            "peak_ram_mb": peak_ram_mb,
            "avg_power_w": round(avg_power_w, 2) if avg_power_w is not None else None,
            "energy_per_token_j": energy_per_token_j,
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

    @staticmethod
    def _get_process_rss_mb() -> float | None:
        """Return current process RSS memory in MB via psutil, or None if unavailable."""
        try:
            import psutil  # type: ignore[import]
            return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return None

