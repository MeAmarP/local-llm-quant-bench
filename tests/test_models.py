import pytest


pydantic = pytest.importorskip("pydantic")
ValidationError = pydantic.ValidationError

from src.quantbench.models import PromptCase, RunResult, RunSpec


def test_prompt_case_valid() -> None:
    prompt = PromptCase(id="p01", task="reasoning", prompt="Solve 2+2")
    assert prompt.id == "p01"
    assert prompt.task == "reasoning"


def test_prompt_case_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PromptCase(id="p01", task="reasoning", prompt="x", unexpected="nope")


def test_run_spec_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RunSpec(name="int4", backend="transformers", quantization="bitsandbytes_int4", foo="bar")


def test_run_result_allows_extra_fields() -> None:
    result = RunResult(
        run_name="run_1",
        backend="transformers",
        quantization="int8",
        prompt_id="p01",
        task="reasoning",
        model_ref="model-id",
        prompt_chars=12,
        prompt_tokens=3,
        output_tokens=4,
        latency_sec=0.4,
        tokens_per_sec=10.0,
        load_time_sec=1.2,
        peak_gpu_mem_mb=2048.0,
        generated_text="ok",
        custom_flag=True,
    )
    data = result.to_dict()
    assert data["custom_flag"] is True
    assert data["latency_sec"] == 0.4


def test_run_result_validates_non_negative_fields() -> None:
    with pytest.raises(ValidationError):
        RunResult(
            run_name="run_1",
            backend="transformers",
            quantization="int8",
            prompt_id="p01",
            task="reasoning",
            model_ref="model-id",
            prompt_chars=-1,
            prompt_tokens=3,
            output_tokens=4,
            latency_sec=0.4,
            tokens_per_sec=10.0,
            load_time_sec=1.2,
            peak_gpu_mem_mb=2048.0,
            generated_text="ok",
        )
