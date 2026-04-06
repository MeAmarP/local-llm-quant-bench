# local-llm-quant-bench

A practical project to learn how quantization affects:
- model size
- RAM/VRAM usage
- latency and generation speed
- output quality
- long-context stability
- ease of use in a real app

## Learning Plan

### Phase 1: Build Intuition

Keep **one model family fixed** (e.g., 1B-3B or 7B-8B) and only vary quantization.

Phase 1 guardrails:
- same prompt text
- same `max_new_tokens`
- same decoding params
- same stop conditions
- same hardware
- same framework versions where possible
- deterministic-ish decoding: `temperature=0.0`, `do_sample=False`

Suggested variants:
- baseline (FP16/BF16 or closest available)
- INT8 (bitsandbytes)
- INT4 (bitsandbytes)
- one GGUF quant (llama.cpp)
- one GPTQ or AWQ

Artifact policy:
- keep the same model family/instruction variant across runtimes
- exact binary artifact can differ (HF checkpoints vs GGUF artifacts)

## Repo Layout

- `configs/` benchmark matrix and metric schema
- `prompts/` benchmark prompt data
  - `prompts/prompts.jsonl` primary prompt list used by runs
  - `prompts/prompt_sets/` task-focused subsets (`summarization`, `extraction`, `reasoning`, `coding`)
- `models/` local model artifacts and caches
  - `models/gguf/` GGUF files for llama.cpp workflows
  - `models/cache/` local model cache space
- `src/quantbench/` package skeleton for reusable benchmark logic
  - `config.py`, `prompts.py`, `metrics.py`, `models.py`
  - `runners/` backend runner interfaces and implementations
  - `utils/` shared utilities (logging, timers, system info)
  - `reporting/` summary/report writer utilities
- `scripts/` helper scripts for run setup, logging, and summaries
- `results/` run outputs
  - `results/runs/<run_id>/` per-run snapshots, observations, and summaries
- `notebooks/` exploratory analysis
- `reports/` human-readable conclusions

## Quick Start

1. Create a run folder:
```bash
python3 scripts/init_run.py --config configs/benchmark.yaml --prompts prompts/prompts.jsonl
```

2. Execute your model/backend manually for each prompt+variant, and log observations:
```bash
python3 scripts/log_metrics.py \
  --run-dir results/runs/<run_id> \
  --variant-id bnb_int8 \
  --prompt-id p01 \
  --wall-clock-latency-ms 1420 \
  --generated-tokens 80 \
  --tokens-per-sec 56.2 \
  --prompt-tokens 42 \
  --output-tokens 80 \
  --peak-gpu-memory-mb 6120 \
  --model-load-time-ms 950 \
  --notes "good speed, slight quality drop"
```

3. Build a summary report:
```bash
python3 scripts/summarize_benchmark.py --run-dir results/runs/<run_id>
```

4. Capture environment metadata (optional):
```bash
python3 scripts/system_snapshot.py --run-dir results/runs/<run_id>
```

## Phase 1 Metrics

Record these first:
- wall-clock latency
- number of generated tokens
- tokens/sec
- prompt length (tokens)
- output length (tokens)
- peak GPU memory (if CUDA is available)
- model load time

## Optional: Quality Rubric (after core metrics)

Use a 1-5 subjective quality score:
- 5: excellent and reliable
- 4: good, minor issues
- 3: usable with noticeable errors
- 2: frequent issues
- 1: poor/unusable

## Next Step

After 1-2 runs, compare variants by:
- median wall-clock latency
- mean tokens/sec
- median peak GPU memory
- median model load time
- generated/output token consistency

Then write conclusions in `reports/`.
