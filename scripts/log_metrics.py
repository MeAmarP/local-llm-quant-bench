#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one benchmark observation.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--wall-clock-latency-ms", type=float, required=True)
    parser.add_argument("--generated-tokens", type=int, required=True)
    parser.add_argument("--tokens-per-sec", type=float)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--peak-gpu-memory-mb", type=float)
    parser.add_argument("--model-load-time-ms", type=float, required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    tokens_per_sec = args.tokens_per_sec
    if tokens_per_sec is None:
        # Fallback when runtime does not provide throughput directly.
        if args.wall_clock_latency_ms <= 0:
            raise ValueError("--wall-clock-latency-ms must be > 0 to derive --tokens-per-sec")
        tokens_per_sec = args.generated_tokens / (args.wall_clock_latency_ms / 1000.0)

    row = {
        "timestamp": dt.datetime.now().isoformat(),
        "variant_id": args.variant_id,
        "prompt_id": args.prompt_id,
        "wall_clock_latency_ms": args.wall_clock_latency_ms,
        "generated_tokens": args.generated_tokens,
        "tokens_per_sec": round(tokens_per_sec, 4),
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "peak_gpu_memory_mb": args.peak_gpu_memory_mb,
        "model_load_time_ms": args.model_load_time_ms,
        "notes": args.notes,
    }

    obs_path = Path(args.run_dir) / "observations.jsonl"
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing observations file: {obs_path}")

    with obs_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    print("ok")


if __name__ == "__main__":
    main()
