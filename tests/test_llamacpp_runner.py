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
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli"

    monkeypatch.setattr(runner_module.LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

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
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli"

    monkeypatch.setattr(runner_module.LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

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
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli"

    monkeypatch.setattr(runner_module.LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

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
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli"

    monkeypatch.setattr(runner_module.LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), executable="llama-cli")
    prompt = PromptCase(id="p01", task="reasoning", prompt="Question?")

    def _fake_run(*args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)
    result = runner.run_case(prompt)

    assert result.error is not None
    assert "not found" in result.error.lower()


# ============================================================================
# NEW TESTS: Executable resolution (CPU/CUDA dual-variant support)
# ============================================================================


def test_executable_exists_returns_true_for_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _executable_exists returns True for a common binary."""
    # python3 is reliably present even in virtualenvs where 'python' may not be
    assert LlamaCppRunner._executable_exists("python3") is True


def test_executable_exists_returns_false_for_nonexistent() -> None:
    """Test that _executable_exists returns False for non-existent binary."""
    assert LlamaCppRunner._executable_exists("definitely_not_a_real_binary_xyz") is False


def test_explicit_executable_parameter_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """When explicit executable is provided, it should be used if it exists."""
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli-custom"

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(
        _run_spec(),
        executable="llama-cli-custom",
    )
    assert runner.executable == "llama-cli-custom"


def test_explicit_executable_raises_if_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """When explicit executable is provided but not found, should raise FileNotFoundError."""
    def _mock_exists(name: str) -> bool:
        return False

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    with pytest.raises(FileNotFoundError, match="not found in PATH"):
        LlamaCppRunner(_run_spec(), executable="nonexistent-binary")


def test_device_cuda_prefers_llama_cli_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=cuda, should prefer llama-cli-cuda."""
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli-cuda"

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), device="cuda")
    assert runner.executable == "llama-cli-cuda"


def test_device_cuda_fallback_to_llama_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=cuda but llama-cli-cuda not found, should fallback to llama-cli."""
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli"

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), device="cuda")
    assert runner.executable == "llama-cli"


def test_device_cuda_raises_if_no_variants_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=cuda but neither llama-cli-cuda nor llama-cli found, should raise."""
    def _mock_exists(name: str) -> bool:
        return False

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    with pytest.raises(FileNotFoundError, match="CUDA device"):
        LlamaCppRunner(_run_spec(), device="cuda")


def test_device_cpu_prefers_llama_cli_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=cpu, should prefer llama-cli-cpu."""
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli-cpu"

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), device="cpu")
    assert runner.executable == "llama-cli-cpu"


def test_device_cpu_fallback_to_llama_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=cpu but llama-cli-cpu not found, should fallback to llama-cli."""
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli"

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), device="cpu")
    assert runner.executable == "llama-cli"


def test_device_cpu_raises_if_no_variants_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=cpu but neither llama-cli-cpu nor llama-cli found, should raise."""
    def _mock_exists(name: str) -> bool:
        return False

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    with pytest.raises(FileNotFoundError, match="CPU variant"):
        LlamaCppRunner(_run_spec(), device="cpu")


def test_device_auto_tries_cuda_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=auto, should try CUDA first."""
    call_count = {"count": 0}

    def _mock_exists(name: str) -> bool:
        call_count["count"] += 1
        # Return True on first call (llama-cli-cuda check)
        if name == "llama-cli-cuda" and call_count["count"] == 1:
            return True
        return False

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), device="auto")
    assert runner.executable == "llama-cli-cuda"


def test_device_auto_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=auto and CUDA not available, should fallback to CPU."""
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli-cpu"

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), device="auto")
    assert runner.executable == "llama-cli-cpu"


def test_device_auto_fallback_to_generic_llama_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=auto and neither variant found, should try generic llama-cli."""
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli"

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), device="auto")
    assert runner.executable == "llama-cli"


def test_device_auto_raises_if_none_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """When device=auto but no variants found, should raise."""
    def _mock_exists(name: str) -> bool:
        return False

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    with pytest.raises(FileNotFoundError, match="found in PATH"):
        LlamaCppRunner(_run_spec(), device="auto")


def test_device_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device parameter should be case-insensitive."""
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli-cuda"

    monkeypatch.setattr(LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), device="CUDA")
    assert runner.device == "cuda"
    assert runner.executable == "llama-cli-cuda"


def test_executable_recorded_in_result_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that the selected executable and device are recorded in result.extra."""
    def _mock_exists(name: str) -> bool:
        return name == "llama-cli-cuda"

    monkeypatch.setattr(runner_module.LlamaCppRunner, "_executable_exists", staticmethod(_mock_exists))

    runner = LlamaCppRunner(_run_spec(), device="cuda", executable="llama-cli-cuda")
    prompt = PromptCase(id="p01", task="reasoning", prompt="Question?")

    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="Answer",
            stderr="",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)
    result = runner.run_case(prompt)

    assert result.extra is not None
    assert result.extra.get("executable") == "llama-cli-cuda"
    assert result.extra.get("device") == "cuda"
