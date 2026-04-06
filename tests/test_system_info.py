from __future__ import annotations

from typing import Any

import pytest

from src.quantbench.utils import system_info


def test_capture_system_info_contains_expected_sections() -> None:
    info = system_info.capture_system_info()
    assert "timestamp_utc" in info
    assert "os" in info
    assert "python" in info
    assert "cpu" in info
    assert "ram" in info
    assert "gpu" in info
    assert "versions" in info


def test_package_version_missing_returns_none() -> None:
    assert system_info._package_version("definitely-not-a-real-package-name") is None


def test_collect_nvidia_smi_info_returns_none_when_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_info.shutil, "which", lambda _: None)
    assert system_info._collect_nvidia_smi_info() is None


def test_collect_nvidia_smi_info_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_info.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    output = "NVIDIA RTX 4090, 24564 MiB, 550.54.14\n"
    monkeypatch.setattr(system_info.subprocess, "check_output", lambda *args, **kwargs: output)

    info = system_info._collect_nvidia_smi_info()
    assert info is not None
    assert info["available"] is True
    assert info["devices"][0]["name"] == "NVIDIA RTX 4090"
    assert info["devices"][0]["memory_total"] == "24564 MiB"
    assert info["devices"][0]["driver_version"] == "550.54.14"


def test_collect_torch_info_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(_: str) -> Any:
        raise ImportError

    monkeypatch.setattr(system_info.importlib, "import_module", _raise_import_error)
    info = system_info._collect_torch_info()
    assert info["available"] is False
    assert info["cuda_available"] is False
