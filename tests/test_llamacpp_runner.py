from __future__ import annotations

import subprocess

import pytest

from src.quantbench.models import PromptCase, RunSpec
import src.quantbench.runners.llamacpp_runner as runner_module
from src.quantbench.runners.llamacpp_runner import LlamaCppRunner


def _run_spec(model_path: str | None = "models/gguf/model.gguf") -> RunSpec:
    return RunSpec(
        name="gguf_q4",
        backend="llamacpp",
        quantization="gguf_q4",
        model_path=model_path,
    )


def test_requires_model_path() -> None:
    with pytest.raises(ValueError):
        LlamaCppRunner(_run_spec(model_path=None))


def test_build_command_uses_generation_config() -> None:
    runner = LlamaCppRunner(
        _run_spec(),
        executable="llama-cli-cuda",
        generation_config={
            "max_new_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "seed": 7,
        },
    )
    cmd = runner._build_command("hello")
    assert cmd[0] == "llama-cli-cuda"
    assert "-m" in cmd and "models/gguf/model.gguf" in cmd
    assert "-n" in cmd and "64" in cmd
    assert "--temp" in cmd and "0.2" in cmd
    assert "--top-p" in cmd and "0.95" in cmd
    assert "--repeat-penalty" in cmd and "1.05" in cmd
    assert "--seed" in cmd and "7" in cmd


def test_run_case_success(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LlamaCppRunner(_run_spec(), executable="llama-cli")
    prompt = PromptCase(id="p01", task="reasoning", prompt="Question?")

    def _fake_run(*args, **kwargs):
        cmd = args[0]
        assert "llama-cli" in cmd[0]
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="Answer text",
            stderr="load time = 250.0 ms\n",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)
    result = runner.run_case(prompt)

    assert result.error is None
    assert result.generated_text == "Answer text"
    assert result.load_time_sec == 0.25
    assert result.prompt_id == "p01"
    assert result.task == "reasoning"
    assert result.extra is not None
    assert result.extra["returncode"] == 0


def test_run_case_strips_prompt_echo_from_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LlamaCppRunner(_run_spec(), executable="llama-cli")
    prompt = PromptCase(id="p01", task="reasoning", prompt="Prompt echo")

    def _fake_run(*args, **kwargs):
        cmd = args[0]
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="Prompt echo generated output",
            stderr="",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)
    result = runner.run_case(prompt)
    assert result.generated_text == "generated output"


def test_run_case_non_zero_exit_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LlamaCppRunner(_run_spec(), executable="llama-cli")
    prompt = PromptCase(id="p01", task="reasoning", prompt="Question?")

    def _fake_run(*args, **kwargs):
        cmd = args[0]
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=2,
            stdout="",
            stderr="bad argument",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)
    result = runner.run_case(prompt)

    assert result.error is not None
    assert "code 2" in result.error
    assert result.extra is not None
    assert result.extra["returncode"] == 2


def test_run_case_handles_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LlamaCppRunner(_run_spec(), executable="llama-cli")
    prompt = PromptCase(id="p01", task="reasoning", prompt="Question?")

    def _fake_run(*args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)
    result = runner.run_case(prompt)

    assert result.error is not None
    assert "not found" in result.error.lower()
