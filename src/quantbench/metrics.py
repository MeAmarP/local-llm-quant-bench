# Objective: Define benchmark metrics and utility helpers for metric calculations.


def supported_metrics() -> list[str]:
    """Return the Phase 1 benchmark metric names."""
    return [
        "wall_clock_latency_ms",
        "generated_tokens",
        "tokens_per_sec",
        "prompt_tokens",
        "output_tokens",
        "peak_gpu_memory_mb",
        "model_load_time_ms",
    ]
