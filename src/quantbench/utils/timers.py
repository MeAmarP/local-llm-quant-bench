# Objective: Provide timing utilities for wall-clock latency and throughput measurements.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import time
from contextlib import contextmanager


def now() -> float:
    """Return a high-resolution timestamp."""
    return time.perf_counter()


def elapsed_seconds(start: float, end: float | None = None) -> float:
    """Return elapsed time in seconds since `start`."""
    stop = now() if end is None else end
    elapsed = stop - start
    if elapsed < 0:
        raise ValueError("end must be >= start")
    return elapsed


def elapsed_milliseconds(start: float, end: float | None = None) -> float:
    """Return elapsed time in milliseconds since `start`."""
    return elapsed_seconds(start, end) * 1000.0


@dataclass
class Timer:
    """Small helper class for start/stop timing in runners and scripts."""

    _start: float | None = None
    _end: float | None = None

    def start(self) -> float:
        self._start = now()
        self._end = None
        return self._start

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer.start() must be called before stop().")
        self._end = now()
        return self._end

    def reset(self) -> None:
        self._start = None
        self._end = None

    @property
    def is_running(self) -> bool:
        return self._start is not None and self._end is None

    @property
    def elapsed_sec(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer has not been started.")
        stop = self._end if self._end is not None else now()
        return elapsed_seconds(self._start, stop)

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_sec * 1000.0

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False


@contextmanager
def time_block() -> Iterator[Timer]:
    """
    Context manager wrapper around Timer.

    Example:
        with time_block() as t:
            run_inference()
        print(t.elapsed_ms)
    """
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
