#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new benchmark run directory.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL")
    parser.add_argument("--results-root", default="results/runs", help="Root directory for runs")
    args = parser.parse_args()

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(args.config, run_dir / "config_snapshot.yaml")
    shutil.copy2(args.prompts, run_dir / "prompts_snapshot.jsonl")

    (run_dir / "observations.jsonl").write_text("", encoding="utf-8")
    (run_dir / "notes.md").write_text("# Run Notes\n\n", encoding="utf-8")

    meta = {
        "run_id": run_id,
        "created_at": dt.datetime.now().isoformat(),
        "config": str(Path(args.config).resolve()),
        "prompts": str(Path(args.prompts).resolve()),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(str(run_dir))


if __name__ == "__main__":
    main()
