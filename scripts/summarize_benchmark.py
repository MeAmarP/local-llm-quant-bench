#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

NUMERIC_FIELDS = [
    "wall_clock_latency_ms",
    "generated_tokens",
    "tokens_per_sec",
    "prompt_tokens",
    "output_tokens",
    "peak_gpu_memory_mb",
    "model_load_time_ms",
]


def mean(values):
    return round(statistics.mean(values), 3) if values else None


def median(values):
    return round(statistics.median(values), 3) if values else None


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value):
    return "-" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize quantization benchmark observations.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    obs_path = run_dir / "observations.jsonl"
    rows = load_rows(obs_path)
    if not rows:
        raise SystemExit("No observations found.")

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["variant_id"]].append(r)

    report_lines = []
    report_lines.append("# Benchmark Summary")
    report_lines.append("")
    report_lines.append("| variant | n | latency_ms (median) | generated_tokens (mean) | tok/s (mean) | prompt_tokens (mean) | output_tokens (mean) | peak_gpu_mb (median) | load_ms (median) |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    summary = {}
    for variant, items in sorted(grouped.items()):
        cols = {k: [x[k] for x in items if x.get(k) is not None] for k in NUMERIC_FIELDS}

        row = {
            "n": len(items),
            "wall_clock_latency_ms_median": median(cols["wall_clock_latency_ms"]),
            "generated_tokens_mean": mean(cols["generated_tokens"]),
            "tokens_per_sec_mean": mean(cols["tokens_per_sec"]),
            "prompt_tokens_mean": mean(cols["prompt_tokens"]),
            "output_tokens_mean": mean(cols["output_tokens"]),
            "peak_gpu_memory_mb_median": median(cols["peak_gpu_memory_mb"]),
            "model_load_time_ms_median": median(cols["model_load_time_ms"]),
        }
        summary[variant] = row

        report_lines.append(
            f"| {variant} | {row['n']} | {fmt(row['wall_clock_latency_ms_median'])} | {fmt(row['generated_tokens_mean'])} | "
            f"{fmt(row['tokens_per_sec_mean'])} | {fmt(row['prompt_tokens_mean'])} | {fmt(row['output_tokens_mean'])} | "
            f"{fmt(row['peak_gpu_memory_mb_median'])} | {fmt(row['model_load_time_ms_median'])} |"
        )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(str(run_dir / "summary.md"))


if __name__ == "__main__":
    main()
