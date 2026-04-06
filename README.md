# local-llm-quant-bench

A reproducible benchmark for comparing local LLM quantization variants.

## What this project measures

- model footprint and memory behavior
- latency and throughput
- output behavior under fixed decoding
- long-context stability
- practical runtime tradeoffs across backends

## Phase 1 benchmark contract

Use one model family and one instruction variant, then vary only quantization/runtime.

| Control | Requirement |
|---|---|
| Prompts | Same prompt text and order |
| Generation | Same `max_new_tokens`, sampling params, and stop conditions |
| Hardware | Same machine/GPU |
| Software | Pin framework/runtime versions where possible |
| Decoding mode | `temperature=0.0`, `do_sample=false` |

Artifact policy: backend-specific artifacts are allowed (HF checkpoints vs GGUF), but family + instruction variant must stay aligned.

## Configuration

- `configs/benchmark.yaml`: benchmark constraints and metric schema
- `configs/models.yaml`: model mapping by variant (`baseline`, `int8`, `int4`, `gptq`, `awq`, `gguf`)
- `configs/generation.yaml`: decoding defaults
- `configs/experiment.yaml`: runtime knobs (`device`, `dtype`, `repetitions`, `warmup_runs`, memory capture)

## Repository layout

```text
.
├── configs/
├── docs/
├── models/
│   ├── cache/
│   └── gguf/
├── notebooks/
├── prompts/
│   ├── prompts.jsonl
│   └── prompt_sets/
├── reports/
├── results/
│   └── runs/<run_id>/
├── scripts/
│   ├── init_run.py
│   ├── log_metrics.py
│   ├── summarize_benchmark.py
│   └── system_snapshot.py
└── src/
    └── quantbench/
```

## Run workflow

1. Initialize a run directory.

```bash
python3 scripts/init_run.py --config configs/benchmark.yaml --prompts prompts/prompts.jsonl
```

2. Log one prompt/variant observation.

```bash
python3 scripts/log_metrics.py \
  --run-dir results/runs/<run_id> \
  --variant-id int8 \
  --prompt-id p01 \
  --wall-clock-latency-ms 1420 \
  --generated-tokens 80 \
  --tokens-per-sec 56.2 \
  --prompt-tokens 42 \
  --output-tokens 80 \
  --peak-gpu-memory-mb 6120 \
  --model-load-time-ms 950 \
  --notes "sample"
```

3. Build summary artifacts.

```bash
python3 scripts/summarize_benchmark.py --run-dir results/runs/<run_id>
```

4. Capture system metadata (recommended).

```bash
python3 scripts/system_snapshot.py --run-dir results/runs/<run_id>
```

## Metrics (Phase 1)

- `wall_clock_latency_ms`
- `generated_tokens`
- `tokens_per_sec`
- `prompt_tokens`
- `output_tokens`
- `peak_gpu_memory_mb` (if available)
- `model_load_time_ms`

## Output artifacts per run

Each `results/runs/<run_id>/` directory contains:

- `config_snapshot.yaml`
- `prompts_snapshot.jsonl`
- `run_meta.json`
- `observations.jsonl`
- `summary.json`
- `summary.md`
- `system_snapshot.json` (if captured)
