import pytest

from src.quantbench.metrics import MetricsHelper, supported_metrics


def test_supported_metrics_contains_phase1_fields() -> None:
    assert supported_metrics() == [
        "wall_clock_latency_ms",
        "generated_tokens",
        "tokens_per_sec",
        "prompt_tokens",
        "output_tokens",
        "peak_gpu_memory_mb",
        "model_load_time_ms",
    ]


def test_compute_wall_clock_latency_ms_positive() -> None:
    assert MetricsHelper.compute_wall_clock_latency_ms(1.0, 1.5) == 500.0


def test_compute_wall_clock_latency_ms_invalid() -> None:
    with pytest.raises(ValueError):
        MetricsHelper.compute_wall_clock_latency_ms(2.0, 2.0)


def test_compute_tokens_per_sec_variants() -> None:
    assert MetricsHelper.compute_tokens_per_sec(generated_tokens=0, wall_clock_latency_ms=1000) == 0.0
    assert MetricsHelper.compute_tokens_per_sec(generated_tokens=20, wall_clock_latency_ms=2000) == 10.0
    assert MetricsHelper.compute_tokens_per_sec(generated_tokens=20, wall_clock_latency_ms=0) is None
    with pytest.raises(ValueError):
        MetricsHelper.compute_tokens_per_sec(generated_tokens=-1, wall_clock_latency_ms=1000)


def test_capture_requires_start() -> None:
    helper = MetricsHelper(measure_gpu_memory=False)
    with pytest.raises(RuntimeError):
        helper.capture(prompt_text="a", generated_text="b")


def test_capture_uses_output_tokens_when_generated_tokens_missing() -> None:
    helper = MetricsHelper(measure_gpu_memory=False)
    helper.start()
    payload = helper.capture(
        prompt_text="one two three",
        generated_text="x y",
        generated_tokens=None,
        model_load_time_ms=111.0,
    )
    assert payload["generated_tokens"] == 2
    assert payload["prompt_tokens"] == 3
    assert payload["output_tokens"] == 2
    assert payload["peak_gpu_memory_mb"] is None
    assert payload["model_load_time_ms"] == 111.0
    assert payload["tokens_per_sec"] is not None


class _EncodeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1, 2, 3] if text else []


def test_count_tokens_with_encode_tokenizer() -> None:
    helper = MetricsHelper(tokenizer=_EncodeTokenizer(), measure_gpu_memory=False)
    assert helper.count_tokens("hello world") == 3


def test_count_tokens_with_callable_tokenizer_dict() -> None:
    helper = MetricsHelper(tokenizer=lambda _: {"input_ids": [5, 6]}, measure_gpu_memory=False)
    assert helper.count_tokens("anything") == 2


def test_count_tokens_with_callable_tokenizer_list() -> None:
    helper = MetricsHelper(tokenizer=lambda _: [9, 8, 7, 6], measure_gpu_memory=False)
    assert helper.count_tokens("anything") == 4
