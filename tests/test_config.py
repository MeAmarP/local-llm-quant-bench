import pytest
import tempfile
from pathlib import Path

pydantic = pytest.importorskip("pydantic")
ValidationError = pydantic.ValidationError

from src.quantbench.config import (
    BenchmarkConfig,
    ConfigManager,
    ExperimentConfig,
    GenerationConfig,
    ModelSpec,
    VariantSpec,
    load_config,
)

UNIFIED_CONFIG_YAML = """
project: test-benchmark

benchmark:
  model_family: "Qwen2.5-3B"
  instruction_variant: "Instruct"
  hardware: "RTX 3090"
  max_new_tokens: 256
  temperature: 0.0
  do_sample: false

generation:
  max_new_tokens: 128
  temperature: 0.0
  top_p: 1.0
  do_sample: false
  repetition_penalty: 1.0

variants:
  baseline:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
    backend: transformers
    quantization: baseline
    quant_family: fp16
    precision: fp16
    notes: "baseline"
  int8:
    model_id: "Qwen/Qwen2.5-3B-Instruct"
    backend: transformers
    quantization: bitsandbytes_int8
    quant_family: bitsandbytes
    precision: int8
    load_in_8bit: true
    notes: "8-bit weights"
  gguf_q4:
    model_path: models/gguf/qwen-q4.gguf
    backend: llamacpp
    quantization: gguf_q4
    quant_family: gguf
    precision: q4
    notes: "GGUF quantized"

experiment:
  prompt_file: prompts/prompts.jsonl
  output_dir: results/raw
  device: auto
  dtype: auto
  repetitions: 3
  warmup_runs: 1
"""


@pytest.fixture
def temp_config_file():
    """Create a temporary unified config YAML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.yaml"
        config_file.write_text(UNIFIED_CONFIG_YAML)
        yield config_file


class TestGenerationConfig:
    def test_default_values(self):
        cfg = GenerationConfig()
        assert cfg.max_new_tokens == 128
        assert cfg.temperature == 0.0
        assert cfg.do_sample is False

    def test_custom_values(self):
        cfg = GenerationConfig(max_new_tokens=256, temperature=0.7)
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.7

    def test_extra_fields_allowed(self):
        cfg = GenerationConfig(custom_param=42)
        assert cfg.custom_param == 42  # type: ignore


class TestVariantSpec:
    def test_valid_variant(self):
        variant = VariantSpec(
            variant_id="int8",
            quant_family="bitsandbytes",
            precision="int8",
            backend="transformers",
        )
        assert variant.variant_id == "int8"
        assert variant.backend == "transformers"

    def test_empty_variant_id_rejected(self):
        with pytest.raises(ValidationError):
            VariantSpec(
                variant_id="",
                quant_family="bitsandbytes",
                precision="int8",
                backend="transformers",
            )

    def test_empty_backend_rejected(self):
        with pytest.raises(ValidationError):
            VariantSpec(
                variant_id="int8",
                quant_family="bitsandbytes",
                precision="int8",
                backend="",
            )


class TestBenchmarkConfig:
    def test_valid_benchmark_config(self):
        data = {
            "project": "test-bench",
            "model_family": "Qwen",
            "variants": [
                {
                    "variant_id": "baseline",
                    "quant_family": "fp16",
                    "precision": "fp16",
                    "backend": "transformers",
                }
            ],
        }
        cfg = BenchmarkConfig(**data)
        assert cfg.project == "test-bench"
        assert len(cfg.variants) == 1

    def test_at_least_one_variant_required(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(
                project="test",
                variants=[],
            )

    def test_extra_fields_preserved(self):
        cfg = BenchmarkConfig(
            project="test",
            model_family="Qwen",
            hardware="RTX 4090",
            variants=[
                {
                    "variant_id": "baseline",
                    "quant_family": "fp16",
                    "precision": "fp16",
                    "backend": "transformers",
                }
            ],
        )
        assert cfg.model_family == "Qwen"  # type: ignore
        assert cfg.hardware == "RTX 4090"  # type: ignore


class TestModelSpec:
    def test_transformers_model(self):
        spec = ModelSpec(
            variant_id="baseline",
            model_id="Qwen/Qwen2.5-3B",
            backend="transformers",
            quantization="baseline",
        )
        assert spec.model_id == "Qwen/Qwen2.5-3B"

    def test_llamacpp_model(self):
        spec = ModelSpec(
            variant_id="gguf",
            model_path="models/qwen.gguf",
            backend="llamacpp",
            quantization="gguf_q4",
        )
        assert spec.model_path == "models/qwen.gguf"

    def test_invalid_backend(self):
        with pytest.raises(ValidationError):
            ModelSpec(
                variant_id="x",
                model_id="model",
                backend="invalid_backend",
                quantization="something",
            )

    def test_bitsandbytes_config(self):
        spec = ModelSpec(
            variant_id="int8",
            model_id="model",
            backend="transformers",
            quantization="bitsandbytes_int8",
            load_in_8bit=True,
        )
        assert spec.load_in_8bit is True

    def test_extra_fields_allowed(self):
        spec = ModelSpec(
            variant_id="x",
            model_id="m",
            backend="transformers",
            quantization="q",
            custom_flag=True,
        )
        assert spec.custom_flag is True  # type: ignore


class TestExperimentConfig:
    def test_valid_experiment_config(self):
        cfg = ExperimentConfig(
            prompt_file="prompts.jsonl",
            output_dir="results/",
            repetitions=5,
            warmup_runs=2,
        )
        assert cfg.repetitions == 5
        assert cfg.warmup_runs == 2

    def test_invalid_repetitions(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(
                prompt_file="p.jsonl",
                output_dir="r/",
                repetitions=0,
            )

    def test_invalid_warmup_runs(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(
                prompt_file="p.jsonl",
                output_dir="r/",
                warmup_runs=-1,
            )


class TestConfigManager:
    def test_load_yaml_file_missing(self):
        manager = ConfigManager()
        with pytest.raises(FileNotFoundError):
            manager.load_yaml("/nonexistent/path.yaml")

    def test_load_yaml_valid(self, temp_config_file):
        manager = ConfigManager()
        data = manager.load_yaml(temp_config_file)
        assert data["project"] == "test-benchmark"
        assert "variants" in data
        assert len(data["variants"]) == 3

    def test_load_from_unified_file_valid(self, temp_config_file):
        manager = ConfigManager()
        manager.load_from_unified_file(temp_config_file)

        assert manager.benchmark_config is not None
        assert manager.benchmark_config.project == "test-benchmark"
        assert len(manager.benchmark_config.variants) == 3
        assert len(manager.models_config) == 3
        assert manager.generation_config.max_new_tokens == 128
        assert manager.experiment_config is not None

    def test_load_from_unified_file_empty_variants_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("""
project: test
benchmark:
  model_family: "Qwen"
generation:
  max_new_tokens: 128
variants: {}
experiment:
  prompt_file: p.jsonl
  output_dir: r/
""")
            manager = ConfigManager()
            with pytest.raises(ValueError, match="At least one variant"):
                manager.load_from_unified_file(config_file)

    def test_get_variant_config(self, temp_config_file):
        manager = ConfigManager()
        manager.load_from_unified_file(temp_config_file)

        variant, model, generation = manager.get_variant_config("int8")
        assert variant.variant_id == "int8"
        assert model.quantization == "bitsandbytes_int8"
        assert model.load_in_8bit is True
        assert generation.max_new_tokens == 128

    def test_get_variant_config_not_found(self, temp_config_file):
        manager = ConfigManager()
        manager.load_from_unified_file(temp_config_file)

        with pytest.raises(ValueError, match="not found"):
            manager.get_variant_config("nonexistent")

    def test_get_variant_config_not_initialized(self):
        manager = ConfigManager()
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_variant_config("baseline")

    def test_to_dict(self, temp_config_file):
        manager = ConfigManager()
        manager.load_from_unified_file(temp_config_file)

        data = manager.to_dict()
        assert "benchmark" in data
        assert "models" in data
        assert "generation" in data
        assert "experiment" in data
        assert len(data["models"]) == 3


class TestLoadConfigFunction:
    def test_load_config_from_file(self, temp_config_file):
        manager = load_config(temp_config_file)
        assert isinstance(manager, ConfigManager)
        assert manager.benchmark_config is not None
        assert len(manager.models_config) == 3

    def test_load_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_config(tmpdir)
