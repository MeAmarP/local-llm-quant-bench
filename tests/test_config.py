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


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory with valid config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # benchmark.yaml
        benchmark_yaml = tmpdir / "benchmark.yaml"
        benchmark_yaml.write_text("""
project: test-benchmark

fixed:
  model_family: "Qwen2.5-3B"
  instruction_variant: "Instruct"
  hardware: "RTX 3090"
  max_new_tokens: 256
  temperature: 0.0
  do_sample: false

variants:
  - variant_id: baseline
    quant_family: fp16
    precision: fp16
    backend: transformers
    notes: "baseline"
  - variant_id: int8
    quant_family: bitsandbytes
    precision: int8
    backend: transformers
    notes: "8-bit weights"
  - variant_id: gguf_q4
    quant_family: gguf
    precision: q4
    backend: llamacpp
    notes: "GGUF quantized"
""")

        # models.yaml
        models_yaml = tmpdir / "models.yaml"
        models_yaml.write_text("""
baseline:
  model_id: "Qwen/Qwen2.5-3B-Instruct"
  backend: transformers
  quantization: baseline

int8:
  model_id: "Qwen/Qwen2.5-3B-Instruct"
  backend: transformers
  quantization: bitsandbytes_int8
  load_in_8bit: true

gguf_q4:
  model_path: models/gguf/qwen-q4.gguf
  backend: llamacpp
  quantization: gguf_q4
""")

        # generation.yaml
        generation_yaml = tmpdir / "generation.yaml"
        generation_yaml.write_text("""
max_new_tokens: 128
temperature: 0.0
top_p: 1.0
do_sample: false
repetition_penalty: 1.0
""")

        # experiment.yaml
        experiment_yaml = tmpdir / "experiment.yaml"
        experiment_yaml.write_text("""
prompt_file: prompts/prompts.jsonl
output_dir: results/raw
device: auto
dtype: auto
repetitions: 3
warmup_runs: 1
measure_gpu_memory: true
""")

        yield tmpdir


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
        # GenerationConfig.Config.extra = "allow"
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
    def test_valid_benchmark_config(self, temp_config_dir):
        data = {
            "project": "test-bench",
            "fixed": {"model_family": "Qwen"},
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
                fixed={},
                variants=[],
            )


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
        assert cfg.measure_gpu_memory is True

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

    def test_load_yaml_valid(self, temp_config_dir):
        manager = ConfigManager()
        data = manager.load_yaml(temp_config_dir / "benchmark.yaml")
        assert data["project"] == "test-benchmark"
        assert len(data["variants"]) == 3

    def test_load_from_dir_valid(self, temp_config_dir):
        manager = ConfigManager()
        manager.load_from_dir(temp_config_dir)

        assert manager.benchmark_config is not None
        assert manager.benchmark_config.project == "test-benchmark"
        assert len(manager.benchmark_config.variants) == 3
        assert len(manager.models_config) == 3
        assert manager.generation_config.max_new_tokens == 128
        assert manager.experiment_config is not None

    def test_variant_id_consistency_checked(self):
        """Variant IDs in benchmark.yaml must match models.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Mismatched variant IDs
            (tmpdir / "benchmark.yaml").write_text("""
project: test
fixed: {}
variants:
  - variant_id: baseline
    quant_family: fp16
    precision: fp16
    backend: transformers
  - variant_id: int8
    quant_family: bitsandbytes
    precision: int8
    backend: transformers
""")

            (tmpdir / "models.yaml").write_text("""
baseline:
  model_id: model
  backend: transformers
  quantization: baseline
int4:
  model_id: model
  backend: transformers
  quantization: int4
""")

            (tmpdir / "generation.yaml").write_text("max_new_tokens: 128")
            (tmpdir / "experiment.yaml").write_text(
                "prompt_file: p.jsonl\noutput_dir: r/"
            )

            manager = ConfigManager()
            with pytest.raises(ValueError, match="missing in models.yaml"):
                manager.load_from_dir(tmpdir)

    def test_variant_id_extra_in_models(self):
        """Extra variant IDs in models.yaml should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "benchmark.yaml").write_text("""
project: test
fixed: {}
variants:
  - variant_id: baseline
    quant_family: fp16
    precision: fp16
    backend: transformers
""")

            (tmpdir / "models.yaml").write_text("""
baseline:
  model_id: model
  backend: transformers
  quantization: baseline
extra_variant:
  model_id: other
  backend: transformers
  quantization: int8
""")

            (tmpdir / "generation.yaml").write_text("max_new_tokens: 128")
            (tmpdir / "experiment.yaml").write_text(
                "prompt_file: p.jsonl\noutput_dir: r/"
            )

            manager = ConfigManager()
            with pytest.raises(ValueError, match="missing in benchmark.yaml"):
                manager.load_from_dir(tmpdir)

    def test_get_variant_config(self, temp_config_dir):
        manager = ConfigManager()
        manager.load_from_dir(temp_config_dir)

        variant, model, generation = manager.get_variant_config("int8")
        assert variant.variant_id == "int8"
        assert model.quantization == "bitsandbytes_int8"
        assert model.load_in_8bit is True
        assert generation.max_new_tokens == 128

    def test_get_variant_config_not_found(self, temp_config_dir):
        manager = ConfigManager()
        manager.load_from_dir(temp_config_dir)

        with pytest.raises(ValueError, match="not found"):
            manager.get_variant_config("nonexistent")

    def test_get_variant_config_not_initialized(self):
        manager = ConfigManager()
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.get_variant_config("baseline")

    def test_to_dict(self, temp_config_dir):
        manager = ConfigManager()
        manager.load_from_dir(temp_config_dir)

        data = manager.to_dict()
        assert "benchmark" in data
        assert "models" in data
        assert "generation" in data
        assert "experiment" in data
        assert len(data["models"]) == 3


class TestLoadConfigFunction:
    def test_load_config(self, temp_config_dir):
        manager = load_config(temp_config_dir)
        assert isinstance(manager, ConfigManager)
        assert manager.benchmark_config is not None
        assert len(manager.models_config) == 3

    def test_load_config_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/dir")
