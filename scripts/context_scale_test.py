"""Context scaling test — measures how latency and throughput degrade as
input context grows from 512 to 8K tokens.

Usage::

    python scripts/context_scale_test.py \\
        --config configs/qwen3_5_9b_config.yaml \\
        --variant gguf_q4 \\
        --sizes 512 1024 2048 4096 8192 \\
        --prompt-id p03

For each (variant, context_size): a synthetic filler text is prepended to
the chosen prompt so the total input token count reaches approximately
``context_size``.  The runner is invoked once per size and the results are
written to ``results/context_scale_<timestamp>.jsonl`` + printed as a table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — support direct execution without installing the package
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
for _p in [str(_SRC_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantbench.config import load_config
from quantbench.models import PromptCase, RunSpec
from quantbench.prompts import load_prompts
from quantbench.runners.llamacpp_runner import LlamaCppRunner
from quantbench.runners.llamacpp_server_runner import LlamaCppServerRunner
from quantbench.runners.transformers_runner import TransformersRunner

# ---------------------------------------------------------------------------
# Filler word list (no external corpus needed)
# ---------------------------------------------------------------------------
_FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "a stitch in time saves nine all that glitters is not gold "
    "to be or not to be that is the question whether tis nobler in the mind "
    "to suffer the slings and arrows of outrageous fortune "
    "in the beginning was the word and the word was with god "
    "we hold these truths to be self evident that all men are created equal "
).split()


def _build_filler(target_tokens: int, words: list[str] = _FILLER_WORDS) -> str:
    """Return a string of approximately *target_tokens* whitespace-split tokens."""
    cycle = (words * ((target_tokens // len(words)) + 2))[:target_tokens]
    return " ".join(cycle)


def _approx_token_count(text: str) -> int:
    """Rough token count using whitespace splitting (1 word ≈ 1.3 tokens heuristic)."""
    return int(len(text.split()) * 1.3)


def _build_padded_prompt(base_prompt: str, target_context_tokens: int) -> str:
    """Build a prompt padded to approximately *target_context_tokens*."""
    base_tokens = _approx_token_count(base_prompt)
    filler_tokens_needed = max(0, target_context_tokens - base_tokens)
    if filler_tokens_needed == 0:
        return base_prompt
    filler = _build_filler(filler_tokens_needed)
    return f"[Context]:\n{filler}\n\n[Question]:\n{base_prompt}"


def _make_runner(variant_id: str, config_path: Path):
    """Instantiate the appropriate runner for the given variant."""
    cfg = load_config(config_path)
    variant_spec, model_spec, generation_config = cfg.get_variant_config(variant_id)
    backend = variant_spec.backend.lower()

    exp = cfg.experiment_config
    device = exp.device.lower() if exp else "auto"
    gen_dict = generation_config.model_dump()
    _extra = model_spec.model_extra or {}

    if backend == "llamacpp":
        runner = LlamaCppRunner(
            run_spec=RunSpec(
                name=variant_id,
                backend=backend,
                quantization=variant_spec.precision,
                model_path=model_spec.model_path,
            ),
            generation_config=gen_dict,
            model_path=model_spec.model_path,
            device=device,
            timeout_sec=600,
            measure_gpu_memory=exp.measure_gpu_memory if exp else True,
            n_gpu_layers=_extra.get("n_gpu_layers"),
            n_ctx=_extra.get("n_ctx"),
        )
    elif backend == "llamacpp_server":
        runner = LlamaCppServerRunner(
            run_spec=RunSpec(
                name=variant_id,
                backend=backend,
                quantization=variant_spec.precision,
                model_path=model_spec.model_path,
            ),
            generation_config=gen_dict,
            model_path=model_spec.model_path,
            n_gpu_layers=_extra.get("n_gpu_layers"),
            n_ctx=_extra.get("n_ctx"),
        )
    elif backend == "transformers":
        runner = TransformersRunner(
            run_spec=RunSpec(
                name=variant_id,
                backend=backend,
                quantization=variant_spec.precision,
                model_id=model_spec.model_id,
            ),
            generation_config=gen_dict,
            model_id=model_spec.model_id,
            model_path=model_spec.model_path,
            device=device,
            dtype=exp.dtype.lower() if exp else "auto",
            load_in_8bit=model_spec.load_in_8bit or False,
            load_in_4bit=model_spec.load_in_4bit or False,
            bnb_4bit_compute_dtype=model_spec.bnb_4bit_compute_dtype,
            measure_gpu_memory=exp.measure_gpu_memory if exp else True,
        )
    else:
        raise ValueError(f"context_scale_test.py: unsupported backend '{backend}'")

    return runner


def _print_table(rows: list[dict]) -> None:
    print(
        f"\n{'Context (tokens)':>18} | {'Variant':>24} | {'Latency (ms)':>14} | "
        f"{'tok/s':>8} | {'Peak GPU (MB)':>14} | {'TTFT (ms)':>10}"
    )
    print("-" * 100)
    for r in rows:
        ttft = f"{r['ttft_ms']:.1f}" if r["ttft_ms"] is not None else "N/A"
        gpu = f"{r['peak_gpu_memory_mb']:.0f}" if r["peak_gpu_memory_mb"] else "N/A"
        print(
            f"{r['context_size']:>18} | {r['variant_id']:>24} | "
            f"{r['wall_clock_latency_ms']:>14.1f} | {r['tokens_per_sec']:>8.1f} | "
            f"{gpu:>14} | {ttft:>10}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Context scaling benchmark")
    parser.add_argument("--config", type=Path, required=True, help="Unified config YAML")
    parser.add_argument(
        "--variant",
        required=True,
        help="Variant ID to test (must be present in config)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Context window sizes in tokens (default: 512 1024 2048 4096)",
    )
    parser.add_argument(
        "--prompt-id",
        default="p03",
        help="Prompt ID from the configured prompt file to use as the base (default: p03)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results",
        help="Directory to write output JSONL (default: results/)",
    )
    args = parser.parse_args()

    # Load base prompt
    cfg = load_config(args.config)
    exp = cfg.experiment_config
    prompt_file = Path(exp.prompt_file) if exp else _REPO_ROOT / "prompts/prompts.jsonl"
    if not prompt_file.is_absolute():
        prompt_file = _REPO_ROOT / prompt_file
    prompts_by_id = {p.id: p for p in load_prompts(prompt_file)}
    base_prompt_case = prompts_by_id.get(args.prompt_id)
    if base_prompt_case is None:
        print(f"ERROR: prompt_id '{args.prompt_id}' not found in {prompt_file}")
        sys.exit(1)

    runner = _make_runner(args.variant, args.config)
    try:
        runner.load()
    except Exception as exc:
        print(f"ERROR: failed to load runner for variant '{args.variant}': {exc}")
        sys.exit(1)

    rows: list[dict] = []
    sizes_sorted = sorted(set(args.sizes))

    print(f"\nRunning context scaling test for variant='{args.variant}', prompt='{args.prompt_id}'")
    print(f"Context sizes: {sizes_sorted} tokens\n")

    for ctx_size in sizes_sorted:
        padded_prompt = _build_padded_prompt(base_prompt_case.prompt, ctx_size)
        padded_case = PromptCase(
            id=f"{args.prompt_id}_ctx{ctx_size}",
            task=base_prompt_case.task,
            prompt=padded_prompt,
        )
        print(f"  ctx={ctx_size:5d} tokens  (actual ~{_approx_token_count(padded_prompt)} tokens)...", end=" ", flush=True)
        try:
            result = runner.run_case(padded_case)
            row = {
                "context_size": ctx_size,
                "variant_id": args.variant,
                "prompt_id": args.prompt_id,
                "wall_clock_latency_ms": result.latency_sec * 1000 if result.latency_sec else 0,
                "tokens_per_sec": result.tokens_per_sec or 0.0,
                "peak_gpu_memory_mb": result.peak_gpu_mem_mb,
                "ttft_ms": result.ttft_ms,
                "output_tokens": result.output_tokens,
                "error": result.error,
            }
            rows.append(row)
            tps = f"{result.tokens_per_sec:.1f}" if result.tokens_per_sec else "n/a"
            print(f"latency={result.latency_sec*1000:.0f}ms  tok/s={tps}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            rows.append({
                "context_size": ctx_size,
                "variant_id": args.variant,
                "prompt_id": args.prompt_id,
                "error": str(exc),
                "wall_clock_latency_ms": 0,
                "tokens_per_sec": 0,
                "peak_gpu_memory_mb": None,
                "ttft_ms": None,
                "output_tokens": None,
            })

    runner.unload()

    _print_table(rows)

    # Write output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"context_scale_{timestamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
