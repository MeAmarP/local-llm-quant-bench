# Objective: Provide timing utilities for wall-clock latency and throughput measurements.

import time


def now() -> float:
    """Return a high-resolution timestamp."""
    return time.perf_counter()
