# Quick-Start Guide: Running local-llm-quant-bench

This guide walks you through configuring and running the `quantbench` suite step-by-step.

> **Recommended:** Use the **unified `config.yaml` file** (simpler, single file) instead of 4 separate config files.

---

## Table of Contents

1. [Before You Begin](#before-you-begin)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Pre-Run Checklist](#pre-run-checklist)
5. [Running the Benchmark](#running-the-benchmark)
6. [Understanding Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)

---

## Before You Begin

### System Requirements

- **Python**: 3.9 or higher
- **GPU** (recommended): CUDA-capable NVIDIA GPU (optional for CPU-only testing)
- **Disk Space**: 10-50GB for models depending on quantization variants
- **Memory**: 8GB+ RAM (16GB+ for larger models)

### Supported Models

This benchmark works with any HuggingFace and llama.cpp (gguf)-compatible model. Popular choices:
- Qwen2.5-3B-Instruct
- Llama-2-7B-Chat
- Mistral-7B-Instruct-v0.2
- Phi-3.5-mini-instruct

---

## Quick Start

### In 5 Steps:

**1. Create virtual environment and install dependencies**

Using `uv` (recommended for fast Python package management):

```bash
# Create a virtual environment
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with all backends
uv pip install -e ".[transformers,llama-cpp]"
```

**2. Configure a single file**
```bash
# Copy and edit the template
cp configs/config.yaml my_benchmark.yaml
nano my_benchmark.yaml

# Set these values:
# - model_family: "Qwen2.5-3B-Instruct"
# - baseline model_id: "Qwen/Qwen2.5-3B-Instruct"
# - hardware: your GPU model
```

**3. Run the benchmark**
```bash
quantbench --config my_benchmark.yaml
```

**4. View results**
```bash
cat results/runs/*/summary.md
```

**5. Done!**

Results are saved to: `results/runs/<timestamp>/`

---

## Configuration

Edit `configs/config.yaml` with your model and settings:

```bash
# Copy the template config
cp configs/config.yaml my_benchmark.yaml

# Edit with your settings
nano my_benchmark.yaml
```

**Key settings to update in `my_benchmark.yaml`:**

```yaml
benchmark:
  model_family: "Qwen2.5-3B-Instruct"     # ← Change this
  hardware: "NVIDIA RTX 4090"              # ← Change this

variants:
  baseline:
    model_id: "Qwen/Qwen2.5-3B-Instruct"  # ← Change this
```

**Example configurations:**

<details>
<summary><b>Example 1: Qwen 3B on RTX 4090</b></summary>

```yaml
benchmark:
  model_family: "Qwen2.5-3B-Instruct"
  hardware: "NVIDIA RTX 4090"

variants:
  baseline:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
  bnb_int8:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
  gguf_q4:
    model_path: "models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf"
```

</details>

<details>
<summary><b>Example 2: Llama 7B on CPU (testing only)</b></summary>

```yaml
benchmark:
  model_family: "Llama-2-7B-Chat"
  hardware: "CPU (Intel Xeon)"

variants:
  baseline:
    model_id: "meta-llama/Llama-2-7b-chat-hf"

experiment:
  device: "cpu"
  repetitions: 1
  warmup_runs: 0
```

</details>

<details>
<summary><b>Example 3: Multiple variants for comparison</b></summary>

```yaml
benchmark:
  model_family: "Qwen2.5-3B-Instruct"
  hardware: "NVIDIA RTX 4090"
  max_new_tokens: 256

variants:
  baseline:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
  bnb_int8:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
    load_in_8bit: true
  bnb_int4:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: true
  gguf_q4:
    model_path: "models/gguf/qwen2.5-3b-q4_k_m.gguf"
```

</details>

---

## Running the Benchmark

```bash
# Basic run
quantbench --config configs/config.yaml

# With custom prompts
quantbench --config configs/config.yaml --prompts prompts/prompt_sets/coding.jsonl

# With custom output directory
quantbench --config configs/config.yaml --output-dir my_results/

# Full example
quantbench \
  --config configs/config.yaml \
  --prompts prompts/prompt_sets/reasoning.jsonl \
  --output-dir results/exp1 \
  --log-level DEBUG
```

### Logging Options

```bash
# INFO level (default)
quantbench --config configs/config.yaml --log-level INFO

# DEBUG level (verbose)
quantbench --config configs/config.yaml --log-level DEBUG

# Save logs to file
quantbench --config configs/config.yaml --log-file benchmark.log
```

### Expected Output

```
INFO - Loading configuration from unified config file: configs/config.yaml
INFO - Configuration loaded successfully
INFO - Loading prompts from: prompts/prompts.jsonl
INFO - Loaded 10 prompts
INFO - Starting benchmark with 10 prompts
INFO - Initialized run directory: results/runs/20260412_143022
INFO - Running variant: baseline
...
INFO - Benchmark completed: results/runs/20260412_143022
20260412_143022
```

The output prints the **run directory path** at the end. Results are saved there automatically.

---

## Understanding Results

After running, check the timestamped directory (printed at completion):

```
results/runs/20260412_143022/
├── config_snapshot.yaml          # Full config used for this run
├── observations.jsonl            # All raw metrics (one JSON per line)
├── prompts_snapshot.jsonl        # Prompts used
├── run_meta.json                 # Execution metadata
├── summary.json                  # Aggregated metrics (JSON)
├── summary.md                    # Human-readable report
└── system_snapshot.json          # Hardware/software info
```

### Reading `summary.md`

```markdown
# Benchmark Report
## Fixed Parameters
- Model: Qwen2.5-3B-Instruct
- Hardware: NVIDIA RTX 4090
- Prompts: 10

## Variant Results
### baseline (fp16)
- Avg Latency: 523.4 ms
- Tokens/sec: 189.2
- Avg GPU Memory: 6120 MB

### bnb_int8
- Avg Latency: 412.1 ms
- Tokens/sec: 247.8
- Avg GPU Memory: 3200 MB

### gguf_q4
- Avg Latency: 289.5 ms
- Tokens/sec: 353.2
- Avg GPU Memory: 1820 MB
```

### Analyzing `observations.jsonl`

```bash
# View all raw observations
cat results/runs/*/observations.jsonl | jq '.'

# Find slowest observations
cat results/runs/*/observations.jsonl | jq 'sort_by(.wall_clock_latency_ms) | .[-5:]'

# Compare tokens/sec by variant
cat results/runs/*/observations.jsonl | jq -r '[.variant_id, .tokens_per_sec] | @csv'
```

---

## Troubleshooting

### Issue 1: "Out of Memory" Error

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce model size: edit `benchmark.model_family` in `configs/config.yaml`
2. Lower `max_new_tokens` in `configs/config.yaml`
3. Reduce `repetitions` in `configs/config.yaml` experiment section
4. Switch to CPU: set `device: cpu` in `configs/config.yaml` experiment section
5. Use quantized variants (int8, int4) for larger models

### Issue 2: Model Not Found

**Symptom:**
```
OSError: Can't load model from huggingface.co/...
```

**Solutions:**
1. Check `model_id` in `variants` section of `configs/config.yaml`
2. Verify HuggingFace model exists: https://huggingface.co/search
3. Login to HuggingFace if model is private: `huggingface-cli login`
4. Check internet connection

### Issue 3: GGUF File Not Found

**Symptom:**
```
FileNotFoundError: models/gguf/model.gguf
```

**Solutions:**
1. Check `model_path` in `variants.gguf_q4` section of `configs/config.yaml`
2. Verify file exists: `ls -lh models/gguf/`
3. Download if missing:
   ```bash
   mkdir -p models/gguf
   huggingface-cli download <repo> <filename> \
     --local-dir models/gguf --local-dir-use-symlinks False
   ```

### Issue 4: Configuration Invalid

**Symptom:**
```
ValueError: Variant 'baseline' not found in models config
# OR
FileNotFoundError: Config file not found: config.yaml
```

**Solutions:**
1. Ensure all variant IDs in `benchmark.variants` appear in `variants` section of `configs/config.yaml`
2. Check file path is correct
3. Verify no typos in variant IDs
4. Verify config file exists at specified path

## Next Steps

After completing a benchmark run:

1. **Review Results**
   ```bash
   cat results/runs/*/summary.md
   ```

2. **Run Multiple Experiments**
   - Change prompts: `quantbench --config configs/config.yaml --prompts my_prompts.jsonl`
   - Test new model: Edit `configs/config.yaml` `benchmark.model_family` → Re-run
   - Adjust generation params: Edit `configs/config.yaml` `generation` section → Re-run
   - Adjust runtime: Edit `configs/config.yaml` `experiment` section → Re-run

3. **Create Visualizations**
   ```bash
   python scripts/visualize_results.py \
     --run-dir results/runs/<timestamp> \
     --output reports/
   ```

4. **Compare Multiple Runs**
   ```bash
   python scripts/aggregate_runs.py \
     --run-dirs results/runs/20260412_143022 results/runs/20260412_150000 \
     --output results/comparison.json
   ```

---

## Additional Resources

- **Main README**: [README.md](../README.md)
- **Config Template**: [config.yaml](../configs/config.yaml)
- **llama.cpp Setup**: [llama_cpp_quick_start.md](./llama_cpp_quick_start.md)
