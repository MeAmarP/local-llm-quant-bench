# Objective: Collect CPU, RAM, GPU, and environment metadata for benchmark runs.

from __future__ import annotations

import datetime as dt
import importlib
import importlib.metadata as importlib_metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def capture_system_info() -> dict[str, Any]:
    """
    Collect machine and runtime metadata for reproducible benchmarks.

    Includes:
    - OS and Python details
    - CPU and RAM information
    - GPU/CUDA details where available
    - key library versions (PyTorch, transformers)
    """
    torch_info = _collect_torch_info()
    nvidia_smi = _collect_nvidia_smi_info()

    versions = {
        "pytorch": _package_version("torch"),
        "transformers": _package_version("transformers"),
    }
    if torch_info.get("cuda_version") is not None:
        versions["cuda"] = torch_info["cuda_version"]

    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": {
            "name": _cpu_name(),
            "logical_cores": os.cpu_count(),
        },
        "ram": {
            "total_mb": _read_ram_total_mb(),
        },
        "gpu": {
            "torch": torch_info,
            "nvidia_smi": nvidia_smi,
        },
        "versions": versions,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _cpu_name() -> str | None:
    cpu = platform.processor()
    if cpu:
        return cpu
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
    return None


def _read_ram_total_mb() -> float | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    for line in meminfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            # MemTotal is in kB on Linux.
            kb = float(line.split()[1])
            return round(kb / 1024.0, 2)
    return None


def _collect_torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": [],
    }

    try:
        torch = importlib.import_module("torch")
    except Exception:
        return info

    info["available"] = True
    info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)

    try:
        cuda_available = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda_available
        if not cuda_available:
            return info

        device_count = int(torch.cuda.device_count())
        info["gpu_count"] = device_count

        gpus: list[dict[str, Any]] = []
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            total_mem_mb = round(props.total_memory / (1024.0 * 1024.0), 2)
            capability = f"{props.major}.{props.minor}"
            gpus.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_mb": total_mem_mb,
                    "capability": capability,
                }
            )
        info["gpus"] = gpus
        return info
    except Exception:
        return info


def _collect_nvidia_smi_info() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        return {"available": True, "error": "failed_to_query"}

    devices: list[dict[str, Any]] = []
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        devices.append(
            {
                "name": parts[0],
                "memory_total": parts[1],  # Unit string comes from nvidia-smi (e.g., "24564 MiB")
                "driver_version": parts[2],
            }
        )
    return {"available": True, "devices": devices}
