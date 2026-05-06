# Quick Start: local-llm-quant-bench

This guide is aligned with the current CLI and code paths in `src/quantbench`.

## Requirements

- Python 3.10+
- Linux/macOS/WSL recommended
- For `transformers` variants: model access to Hugging Face
- For `llamacpp`/`llamacpp_server` variants: a GGUF file and `llama.cpp` binaries in `PATH`

## 1) Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[transformers,llama-cpp,quality,extended-metrics]"
```

Notes:
- `quality` is needed for `--golden-answers` scoring.
- `extended-metrics` enables RAM metrics via `psutil`.

## 2) Prepare Config

Create a copy and edit it:

```bash
cp configs/config.yaml my_benchmark.yaml
```

Minimum fields that affect execution are in `variants`, `generation`, and `experiment`.

Working baseline-only example:

```yaml
project: local-llm-quant-bench

benchmark:
  model_family: "Qwen2.5-3B-Instruct"
  instruction_variant: "Instruct"
  hardware: "NVIDIA GPU"

variants:
  baseline:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
    backend: transformers
    quantization: baseline
    quant_family: fp16_or_bf16
    precision: fp16_or_bf16

generation:
  max_new_tokens: 128
  temperature: 0.0
  top_p: 1.0
  do_sample: false
  repetition_penalty: 1.0

experiment:
  prompt_file: "prompts/prompts.jsonl"
  output_dir: "results/runs"
  device: "auto"
  dtype: "auto"
  repetitions: 1
  warmup_runs: 1
```

Important:
- `warmup_runs` must be `>= 1`.
- Supported backends in config validation: `transformers`, `llamacpp`, `llamacpp_server`, `vllm`, `onnx`.
- Current orchestrator implements runners for: `transformers`, `llamacpp`, `llamacpp_server`.

## 3) Run

```bash
quantbench --config my_benchmark.yaml
```

Useful overrides:

```bash
quantbench --config my_benchmark.yaml --prompts prompts/prompt_sets/coding.jsonl
quantbench --config my_benchmark.yaml --output-dir results/experiments
quantbench --config my_benchmark.yaml --log-level DEBUG --log-file quantbench.log
quantbench --config my_benchmark.yaml --golden-answers prompts/golden_answers.jsonl
```

The command prints the run directory path on success, e.g.:

```text
results/runs/20260506_143022
```

## 4) Prompt File Schema

Prompt files are JSONL. Each line must contain:

```json
{"id":"p01","task":"reasoning","prompt":"..."}
```

Required keys:
- `id`
- `task`
- `prompt`

## 5) Output Artifacts

Each run writes to `results/runs/<timestamp>/` (or your `--output-dir`):

- `config_snapshot.yaml` (contains normalized config payload)
- `prompts_snapshot.jsonl`
- `observations.jsonl`
- `run_meta.json`
- `summary.json`
- `summary.md`
- `system_snapshot.json` (best effort)
- `notes.md`

Quick checks:

```bash
cat results/runs/*/summary.md
cat results/runs/*/observations.jsonl | jq '.'
```

## Troubleshooting

### `Config path not found`

Use a valid file path:

```bash
quantbench --config configs/config.yaml
```

### `must be >= 1` for `warmup_runs` or `repetitions`

Set both to at least `1` in `experiment`.

### `Path does not exist` for prompts

Fix `experiment.prompt_file` or pass `--prompts <file-or-dir>`.

### `llama.cpp executable not found`

Install `llama.cpp` binaries and ensure one of these is in `PATH`:
- `llama-cli-cuda`
- `llama-cli-cpu`
- `llama-cli`

### GGUF model not found

Verify `variants.<name>.model_path` exists and is readable.

## Related Docs

- [README.md](../README.md)
- [configs/config.yaml](../configs/config.yaml)
- [llama_cpp_quick_start.md](./llama_cpp_quick_start.md)
