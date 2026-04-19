# Feature Branches — Development Reference

This document records the three feature branches developed for `local-llm-quant-bench`,
covering their objectives, the files changed, and the capabilities introduced.

---

## Branch 1 — `feature/quality-scoring`

**Commit:** `db15863`
**Base:** `master`

### Objective

Extend the benchmark pipeline with per-prompt output quality evaluation so that
raw performance metrics (latency, throughput) can be paired with a quality signal.
Quality is assessed either against a golden reference answer or via task-level
heuristic rubrics, depending on what ground truth is available.

### Feature Overview

| Feature | Description |
|---|---|
| Rubric scoring | Checks output against format, code parseability, numeric answer, safety refusal, and word-count rules |
| Reference scoring | Token F1, ROUGE-L, and exact-match against a golden answer string |
| `QualityScorer` | Dispatcher that selects the right method per prompt based on `golden_answers.jsonl` entries |
| CLI integration | `--golden-answers` flag passes a ground-truth file to the orchestrator |
| JSONL observations | `quality_pass`, `quality_score`, `quality_method`, `quality_details` emitted per observation |

### Files Changed / Created

#### New files

| File | Description |
|---|---|
| `src/quantbench/quality/__init__.py` | Re-exports `QualityScorer` as the public API of the module |
| `src/quantbench/quality/rubrics.py` | Stateless rubric functions: `check_json_format`, `check_code_parseable`, `check_numeric_answer`, `check_safety_refusal`, `check_word_count` |
| `src/quantbench/quality/reference.py` | Reference-based scorers: `token_f1`, `rouge_l_score`, `exact_match` |
| `src/quantbench/quality/scorer.py` | `QualityScorer` class — loads `golden_answers.jsonl` and dispatches to the appropriate scorer per prompt ID or task |
| `prompts/golden_answers.jsonl` | Ground-truth entries for prompts `p02` (numeric), `p03` (code_parse), `p06` (safety_refusal), `p07` (json_format), `p09` (word_count) |
| `tests/test_quality.py` | Unit tests for all rubric functions, reference scorers, and `QualityScorer` dispatch logic |

#### Modified files

| File | Change |
|---|---|
| `src/quantbench/models.py` | Added `quality_pass: bool\|None`, `quality_score: float\|None`, `quality_method: str\|None`, `quality_details: dict\|None` to `RunResult` |
| `src/quantbench/orchestrator.py` | Added `golden_answers_path` parameter; instantiates `QualityScorer` and calls it inside `log_observation()`, appending quality fields to each JSONL record |
| `src/quantbench/cli.py` | Added `--golden-answers` CLI flag; resolves the path and passes it to `BenchmarkOrchestrator` |
| `pyproject.toml` | Added `quality = ["rouge-score>=0.1"]` optional dependency group |

### Quality Methods Reference

| Method key | Logic |
|---|---|
| `numeric` | Parses a float from the output and compares within a tolerance |
| `code_parse` | Attempts to `ast.parse()` the output; pass = no `SyntaxError` |
| `json_format` | `json.loads()` the output; optionally checks for required top-level keys |
| `safety_refusal` | Checks for refusal keywords ("I cannot", "I'm unable to", etc.) |
| `word_count` | Counts occurrences of a target word and compares to an expected count |
| `token_f1` | Token-level F1 between prediction and reference (whitespace split) |
| `rouge_l` | ROUGE-L F1 using `rouge-score` library |
| `exact_match` | Case-insensitive exact string match after stripping whitespace |

---

## Branch 2 — `feature/extended-metrics`

**Commit:** `2cf0891`
**Base:** `master`

### Objective

Capture a richer set of hardware and timing metrics beyond the Phase 1 baseline.
Specifically: time-to-first-token (TTFT), peak process RAM delta, average GPU
power draw, and energy consumption per output token. These metrics expose
efficiency differences between quantization levels that latency alone cannot.

### Feature Overview

| Feature | Description |
|---|---|
| TTFT measurement | Time elapsed from generation start to the first output token |
| Peak RAM tracking | Process RSS memory delta during inference, via `psutil` |
| GPU power sampling | Background thread polls `nvidia-smi` every 200 ms during generation |
| Energy per token | Derived as `(avg_power_w × latency_sec) / output_tokens` |
| Context scale script | `scripts/context_scale_test.py` sweeps context window sizes and reports degradation |

### Files Changed / Created

#### New files

| File | Description |
|---|---|
| `scripts/context_scale_test.py` | CLI script that pads a base prompt to target context sizes (e.g. 512 → 8192 tokens) using synthetic filler text, runs one inference per size, and writes a Markdown table + JSONL of latency/throughput results to `results/` |

#### Modified files

| File | Change |
|---|---|
| `src/quantbench/metrics.py` | Added `_PowerSampler` inner class (daemon thread, polls `nvidia-smi --query-gpu=power.draw`); added `measure_ram: bool`, `measure_power: bool` init params; `start()` records `_ram_start_mb` via `psutil`; `capture()` now accepts `ttft_ms` and returns `ttft_ms`, `peak_ram_mb`, `avg_power_w`, `energy_per_token_j`; `METRIC_NAMES` extended with the four new names |
| `src/quantbench/models.py` | Added `ttft_ms: float\|None`, `peak_ram_mb: float\|None`, `avg_power_w: float\|None`, `energy_per_token_j: float\|None` to `RunResult` |
| `src/quantbench/runners/llamacpp_runner.py` | Added `measure_ram: bool`, `measure_power: bool` init params; added `_parse_prompt_eval_time_ms(stdout, stderr)` static method (regex against llama.cpp log line `prompt eval time = X ms`) to extract TTFT; sets all four extended fields on `RunResult` |
| `src/quantbench/runners/transformers_runner.py` | Added `_FirstTokenTimer` class (`LogitsProcessor`-compatible hook) that records the timestamp of the first token's logits call; added `measure_ram`, `measure_power` init params; attaches the timer via `LogitsProcessorList` in `generate()`; sets all four extended fields on `RunResult` |
| `src/quantbench/config.py` | Added `measure_ram: bool = False` and `measure_power: bool = False` to `ExperimentConfig` |
| `src/quantbench/orchestrator.py` | `get_runner()` forwards `measure_ram`/`measure_power` to both `LlamaCppRunner` and `TransformersRunner`; `log_observation()` includes `ttft_ms`, `peak_ram_mb`, `avg_power_w`, `energy_per_token_j` in each JSONL observation |
| `pyproject.toml` | Added `extended-metrics = ["psutil>=5.9"]` optional dependency group |

### New Metrics Reference

| Field | Unit | Source |
|---|---|---|
| `ttft_ms` | milliseconds | `_FirstTokenTimer` (transformers) or `prompt eval time` log line (llama.cpp) |
| `peak_ram_mb` | megabytes | Process RSS delta: `psutil.Process().memory_info().rss` at start vs. capture |
| `avg_power_w` | watts | Mean of `nvidia-smi --query-gpu=power.draw` samples taken every 200 ms |
| `energy_per_token_j` | joules | `(avg_power_w × latency_sec) / output_tokens` |

### context_scale_test.py Usage

```
python scripts/context_scale_test.py \
    --config configs/qwen3_5_9b_config.yaml \
    --variant gguf_q4 \
    --sizes 512 1024 2048 4096 8192 \
    --prompt-id p03
```

Outputs:
- Printed Markdown table with columns: context size, latency (ms), tok/s, peak GPU (MB), TTFT (ms)
- `results/context_scale_<timestamp>.jsonl` with one record per context size

---

## Branch 3 — `feature/llamacpp-server`

**Commit:** `ee6dc37`
**Base:** `master`

### Objective

Add a server-mode inference backend that keeps the model loaded in memory across
all prompt runs. The existing `LlamaCppRunner` re-initialises the process for
every prompt, which means the model is loaded and unloaded repeatedly.
`LlamaCppServerRunner` starts `llama-server` once in `load()`, routes all
`run_case()` calls to it via HTTP, and tears it down in `unload()`. This is
significantly faster for multi-prompt benchmark runs and opens the door to
concurrent request batching in future work.

### Feature Overview

| Feature | Description |
|---|---|
| Persistent server process | `llama-server` is started once and shared across all prompts in a run |
| Health-check startup | `load()` polls `GET /health` until 200 OK (up to 120 s) before accepting requests |
| Port auto-selection | If the preferred port is busy, an OS-assigned free port is used automatically |
| stdlib-only HTTP | Requests are made with `urllib.request` — no `httpx`/`requests` dependency |
| Structured timings | `timings` block from the JSON response provides TTFT, predicted tokens/s, and token counts directly |
| Config wiring | `llamacpp_server` is a first-class backend in `ModelSpec` and `BenchmarkOrchestrator` |

### Files Changed / Created

#### New files

| File | Description |
|---|---|
| `src/quantbench/runners/llamacpp_server_runner.py` | `LlamaCppServerRunner(BaseRunner)` — manages the full `llama-server` subprocess lifecycle; builds the server CLI command from `generation_config` and `model_extra`; sends JSON POST requests to `/completion`; extracts `timings.prompt_ms` as `ttft_ms`, `timings.predicted_per_second` as `tokens_per_sec`, `timings.predicted_n`/`timings.prompt_n` for token counts |

#### Modified files

| File | Change |
|---|---|
| `src/quantbench/config.py` | Added `"llamacpp_server"` to the valid set in `ModelSpec.valid_backend` validator |
| `src/quantbench/orchestrator.py` | Imported `LlamaCppServerRunner`; added `elif backend == "llamacpp_server":` branch in `get_runner()` that reads `host`, `port`, `n_gpu_layers`, `n_ctx` from `model_extra` and constructs a `LlamaCppServerRunner` |

### LlamaCppServerRunner — Key Design Decisions

| Decision | Rationale |
|---|---|
| `urllib.request` over third-party clients | Keeps the core package dependency-free; no additional install required |
| `--log-disable` server flag | Reduces subprocess stdout noise; readiness is detected via `/health` rather than log parsing |
| Port auto-fallback via `socket.bind(("", 0))` | Prevents port conflicts when running multiple benchmark variants concurrently |
| `timings` block for metrics | More accurate than wall-clock because it reflects server-side token generation time, not round-trip HTTP overhead |

### Configuration Example (`models.yaml`)

```yaml
gguf_server:
  model_path: models/gguf/Qwen3.5-9B-Q4_K_M.gguf
  backend: llamacpp_server
  quantization: Q4_K_M
  n_gpu_layers: -1
  n_ctx: 4096
  host: "127.0.0.1"
  port: 8080
```

---

## Summary Table

| Branch | Commit | New Files | Modified Files | Optional Extra |
|---|---|---|---|---|
| `feature/quality-scoring` | `db15863` | 6 | 4 | `quality` (`rouge-score`) |
| `feature/extended-metrics` | `2cf0891` | 1 | 7 | `extended-metrics` (`psutil`) |
| `feature/llamacpp-server` | `ee6dc37` | 1 | 2 | — |
