#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
from pathlib import Path


def read_mem_total_mb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            kb = float(line.split()[1])
            return round(kb / 1024, 2)
    return None


def run_cmd(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture system metadata for a run.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    gpu_info = None
    if shutil.which("nvidia-smi"):
        gpu_info = run_cmd([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ])

    snapshot = {
        "timestamp": dt.datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or None,
        "ram_total_mb": read_mem_total_mb(),
        "gpu": gpu_info,
    }

    out_path = Path(args.run_dir) / "system_snapshot.json"
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
