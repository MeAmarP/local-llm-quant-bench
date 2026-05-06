# local-llm-quant-bench

A local benchmark harness for comparing LLM quantization/runtime variants under fixed prompts and decoding settings.

## What it measures

- Wall-clock latency and throughput
- Prompt and output token counts
- Peak GPU memory when available
- Model load time
- Extended metrics: TTFT, peak RAM, average power draw, energy per token
- Optional quality scoring against prompt-specific rubrics or golden answers

## Current supported execution paths

- `transformers`: Hugging Face causal LM inference
- `llamacpp`: local `llama-cli` / `llama-cli-cuda` / `llama-cli-cpu`
- `llamacpp_server`: local `llama-server` subprocess with HTTP completion calls

## Benchmark contract

Use one model family and one instruction variant, then vary only quantization/runtime.

| Control | Requirement |
|---|---|
| Prompts | Same prompt text and order |
| Generation | Same `max_new_tokens`, sampling params, and stop conditions |
| Hardware | Same machine/GPU |
| Software | Pin framework/runtime versions where possible |
| Decoding mode | `temperature=0.0`, `do_sample=false` |

Artifact policy: backend-specific artifacts are allowed, but the family and instruction variant should stay aligned across variants.

## Installation

Python 3.10+ is recommended for the current codebase.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[transformers,llama-cpp,quality,extended-metrics]"
```

Optional extras in `pyproject.toml`:

- `transformers`: installs `transformers` and `torch`
- `llama-cpp`: installs `llama-cpp-python`
- `quality`: installs scoring dependencies such as `rouge-score`
- `extended-metrics`: installs `psutil` for RAM metrics

## Recommended workflow

The main entrypoint is the installed CLI:

```bash
quantbench --config configs/config.yaml
```

Useful overrides:

```bash
quantbench --config configs/config.yaml --prompts prompts/prompt_sets/coding.jsonl
quantbench --config configs/config.yaml --output-dir results/experiments
quantbench --config configs/config.yaml --log-level DEBUG --log-file quantbench.log
quantbench --config configs/config.yaml --golden-answers prompts/golden_answers.jsonl
```

On success, the CLI prints the run directory path, for example `results/runs/20260506_143022`.

## Configuration formats

Recommended:

- Unified single-file config: [configs/config.yaml](/home/mighty/Documents/workspace/local-llm-quant-bench/configs/config.yaml)

Also supported:

- A legacy config directory containing `benchmark.yaml`, `models.yaml`, `generation.yaml`, and `experiment.yaml`

Key runtime fields in the unified config:

- `variants.<name>`: model/backend-specific settings
- `generation`: `max_new_tokens`, `temperature`, `top_p`, `do_sample`, `repetition_penalty`
- `experiment`: `prompt_file`, `output_dir`, `device`, `dtype`, `repetitions`, `warmup_runs`

Notes:

- `warmup_runs` and `repetitions` must both be `>= 1`
- Prompt paths may point to either a single `.jsonl` file or a directory of `.jsonl` files
- For `llamacpp` variants, `model_path` must point to a GGUF file
- For `llamacpp_server`, model-specific extras such as `host`, `port`, `n_ctx`, and `n_gpu_layers` are passed through the variant config

## Prompt files

Prompt files are JSONL. Each line must contain:

```json
{"id":"p01","task":"reasoning","prompt":"Explain quantization to a beginner."}
```

Required fields:

- `id`
- `task`
- `prompt`

Example prompt sets live under:

- `prompts/prompts.jsonl`
- `prompts/prompt_sets/coding.jsonl`
- `prompts/prompt_sets/reasoning.jsonl`
- `prompts/prompt_sets/summarization.jsonl`
- `prompts/prompt_sets/extraction.jsonl`

## Output artifacts

Each benchmark run writes a timestamped directory under `results/runs/` unless overridden.

Files produced by the orchestrated CLI run:

- `config_snapshot.yaml` containing the normalized config payload
- `prompts_snapshot.jsonl`
- `observations.jsonl`
- `run_meta.json`
- `summary.json`
- `summary.md`
- `system_snapshot.json`
- `notes.md`

Quick inspection commands:

```bash
cat results/runs/*/summary.md
cat results/runs/*/observations.jsonl | jq '.'
```

## Metrics

Core metrics:

- `wall_clock_latency_ms`
- `generated_tokens`
- `tokens_per_sec`
- `prompt_tokens`
- `output_tokens`
- `peak_gpu_memory_mb`
- `model_load_time_ms`

Extended metrics:

- `ttft_ms`
- `peak_ram_mb`
- `avg_power_w`
- `energy_per_token_j`

When `--golden-answers` is used, observations can also include:

- `quality_pass`
- `quality_score`
- `quality_method`
- `quality_details`

## Quality scoring

Quality scoring is optional and uses either:

- Prompt-specific entries from `prompts/golden_answers.jsonl`
- Task heuristics when no golden answer entry exists

Supported scoring methods include `numeric`, `code_parse`, `json_format`, `safety_refusal`, `word_count`, `rouge_l`, and `token_f1`.

## Lower-level helper scripts

These scripts still exist for manual pipelines or ad hoc workflows, but they are not the primary benchmark entrypoint:

- [scripts/init_run.py](/home/mighty/Documents/workspace/local-llm-quant-bench/scripts/init_run.py): create a run directory and snapshots
- [scripts/log_metrics.py](/home/mighty/Documents/workspace/local-llm-quant-bench/scripts/log_metrics.py): append one observation row
- [scripts/summarize_benchmark.py](/home/mighty/Documents/workspace/local-llm-quant-bench/scripts/summarize_benchmark.py): build `summary.json` and `summary.md` from `observations.jsonl`
- [scripts/system_snapshot.py](/home/mighty/Documents/workspace/local-llm-quant-bench/scripts/system_snapshot.py): capture system metadata into `system_snapshot.json`

Example manual flow:

```bash
python3 scripts/init_run.py --config configs/config.yaml --prompts prompts/prompts.jsonl
python3 scripts/log_metrics.py --run-dir results/runs/<run_id> --variant-id baseline --prompt-id p01 --wall-clock-latency-ms 1420 --generated-tokens 80 --prompt-tokens 42 --output-tokens 80 --model-load-time-ms 950
python3 scripts/summarize_benchmark.py --run-dir results/runs/<run_id>
python3 scripts/system_snapshot.py --run-dir results/runs/<run_id>
```

## Repository layout

- `README.md`: project overview, installation, and usage guidance
- `pyproject.toml`: package metadata, optional dependency groups, and the `quantbench` CLI entrypoint
- `uv.lock`: locked dependency graph for reproducible `uv` installs
- `configs/`: shipped benchmark configuration templates, including the recommended unified config
- `docs/`: user-facing guides such as the quickstart and llama.cpp setup notes
- `prompts/`: default benchmark prompts, prompt subsets, and golden-answer scoring data
- `results/runs/`: timestamped benchmark outputs produced by the orchestrated CLI workflow
- `scripts/`: lower-level helper utilities for manual run setup, metric logging, summarization, and system snapshots
- `src/quantbench/`: main Python package containing the benchmark implementation
- `src/quantbench/cli.py`: installed `quantbench` command and CLI argument handling
- `src/quantbench/config.py`: unified and legacy config loading plus validation
- `src/quantbench/orchestrator.py`: end-to-end benchmark execution, observation logging, and summary generation
- `src/quantbench/runners/`: backend adapters for `transformers`, `llamacpp`, and `llamacpp_server`
- `src/quantbench/metrics.py`: latency, throughput, memory, TTFT, power, and energy metric capture helpers
- `src/quantbench/quality/`: rubric-based and reference-based response scoring logic
- `src/quantbench/reporting/`: report-writing helpers used for benchmark output formatting
- `src/quantbench/utils/`: shared logging, timing, and system-information helpers
- `tests/`: unit tests covering config loading, runners, metrics, orchestration, and utilities

## Docs

- [docs/QUICKSTART.md](/home/mighty/Documents/workspace/local-llm-quant-bench/docs/QUICKSTART.md)
- [docs/llama_cpp_quick_start.md](/home/mighty/Documents/workspace/local-llm-quant-bench/docs/llama_cpp_quick_start.md)