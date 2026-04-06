from __future__ import annotations

from src.quantbench.models import PromptCase, RunResult, RunSpec
from src.quantbench.runners.base import BaseRunner


class _DummyRunner(BaseRunner):
    def run_case(self, prompt_case: PromptCase) -> RunResult:
        return RunResult(
            run_name=self.run_spec.name,
            backend=self.run_spec.backend,
            quantization=self.run_spec.quantization,
            prompt_id=prompt_case.id,
            task=prompt_case.task,
            model_ref=self.run_spec.model_id or "dummy-model",
            prompt_chars=len(prompt_case.prompt),
            prompt_tokens=1,
            output_tokens=1,
            latency_sec=0.1,
            tokens_per_sec=10.0,
            load_time_sec=0.01,
            peak_gpu_mem_mb=None,
            generated_text="ok",
        )


def test_run_wrapper_returns_dict_with_expected_shape() -> None:
    spec = RunSpec(name="dummy", backend="dummy-backend", quantization="none", model_id="m")
    runner = _DummyRunner(spec)

    out = runner.run("hello", prompt_id="p123", task="unit_test")
    assert isinstance(out, dict)
    assert out["prompt_id"] == "p123"
    assert out["task"] == "unit_test"
    assert out["generated_text"] == "ok"
