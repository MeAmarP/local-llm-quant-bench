# Objective: Load and validate benchmark configuration files.

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class GenerationConfig(BaseModel):
    """Decoding parameters for model generation (generation.yaml)."""

    model_config = ConfigDict(extra="allow")

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.0


class VariantSpec(BaseModel):
    """A single quantization variant specification."""

    variant_id: str
    quant_family: str
    precision: str
    backend: str  # "transformers", "llamacpp", etc.
    notes: Optional[str] = None

    @field_validator("variant_id", "backend")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty string")
        return v.lower() if v == "backend" else v


class BenchmarkConfig(BaseModel):
    """Benchmark metadata and variant definitions from the unified config file."""

    model_config = ConfigDict(extra="allow")

    project: str

    variants: list[VariantSpec] = Field(
        default_factory=list,
        description="List of quantization variants to measure"
    )

    @field_validator("variants")
    @classmethod
    def at_least_one_variant(cls, v: list[VariantSpec]) -> list[VariantSpec]:
        if len(v) < 1:
            raise ValueError("at least one variant must be specified")
        return v


class ModelSpec(BaseModel):
    """A single model configuration per variant (models.yaml)."""

    model_config = ConfigDict(extra="allow")

    variant_id: Optional[str] = None  # Used as key in models.yaml; filled during parsing
    model_id: Optional[str] = None  # For transformers backend
    model_path: Optional[str] = None  # For llamacpp/gguf
    backend: str
    quantization: str
    load_in_8bit: Optional[bool] = None
    load_in_4bit: Optional[bool] = None
    bnb_4bit_compute_dtype: Optional[str] = None
    gptq_quantization_config: Optional[dict] = None

    @field_validator("backend")
    @classmethod
    def valid_backend(cls, v: str) -> str:
        valid = {"transformers", "llamacpp", "llamacpp_server", "vllm", "onnx"}
        if v not in valid:
            raise ValueError(f"backend must be one of {valid}")
        return v


class ExperimentConfig(BaseModel):
    """Runtime experiment knobs (experiment.yaml)."""

    prompt_file: str
    output_dir: str
    device: str = "auto"
    dtype: str = "auto"
    repetitions: int = 3
    warmup_runs: int = 1

    @field_validator("repetitions", "warmup_runs")
    @classmethod
    def positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v


class ConfigManager:
    """Loads and validates configuration files for benchmarking."""

    def __init__(self):
        self.benchmark_config: Optional[BenchmarkConfig] = None
        self.models_config: dict[str, ModelSpec] = {}
        self.generation_config: GenerationConfig = GenerationConfig()
        self.experiment_config: Optional[ExperimentConfig] = None

    def load_yaml(self, path: str | Path) -> dict[str, Any]:
        """Load a single YAML file."""
        path = Path(path)
        logger.debug(f"Loading YAML file: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        logger.debug(f"Loaded YAML file: {path}")
        return data

    def get_variant_config(self, variant_id: str) -> tuple[VariantSpec, ModelSpec, GenerationConfig]:
        """Get combined config for a specific variant.
        
        Args:
            variant_id: Variant identifier from variants list
        
        Returns:
            Tuple of (VariantSpec, ModelSpec, GenerationConfig)
        
        Raises:
            ValueError: If variant_id not found
        """
        if not self.benchmark_config:
            raise RuntimeError("ConfigManager not initialized; call load_config() first")

        variant = next(
            (v for v in self.benchmark_config.variants if v.variant_id == variant_id),
            None,
        )
        if not variant:
            raise ValueError(f"Variant '{variant_id}' not found in benchmark config")

        model = self.models_config.get(variant_id)
        if not model:
            raise ValueError(f"Model config for variant '{variant_id}' not found")

        return variant, model, self.generation_config

    def load_from_unified_file(self, config_path: str | Path) -> "ConfigManager":
        """Load unified config from a single YAML file.

        Expected format:
        ```yaml
        project: local-llm-quant-bench
        benchmark:
          model_family: "Qwen2.5-3B-Instruct"
          max_new_tokens: 256
          metrics: [...]
        generation:
          max_new_tokens: 256
          temperature: 0.0
        variants:
          baseline:
            model_id: "..."
            backend: "transformers"
          gguf_q4:
            model_path: "..."
            backend: "llamacpp"
        experiment:
          prompt_file: "prompts/prompts.jsonl"
          output_dir: "results/runs"
        ```

        Args:
            config_path: Path to unified config YAML file

        Returns:
            self (for chaining)

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If validation fails
        """
        config_path = Path(config_path)
        logger.info(f"Loading config file: {config_path}")
        data = self.load_yaml(config_path)

        project = data.get("project", "local-llm-quant-bench")

        benchmark_data = data.get("benchmark", {})
        benchmark_data["project"] = project

        variants_data = data.get("variants", {})
        if not isinstance(variants_data, dict):
            raise ValueError("'variants' must be a dictionary in config file")

        benchmark_data["variants"] = []
        self.models_config = {}

        for variant_id, variant_spec in variants_data.items():
            if not isinstance(variant_spec, dict):
                raise ValueError(f"Variant '{variant_id}' must be a dict")

            variant_spec_copy = dict(variant_spec)
            variant_spec_copy["variant_id"] = variant_id
            variant_spec_copy.setdefault("quant_family", "unknown")
            variant_spec_copy.setdefault("precision", "unknown")
            variant_spec_copy.setdefault("backend", "unknown")

            benchmark_data["variants"].append(VariantSpec(**variant_spec_copy))

            model_spec_copy = dict(variant_spec)
            model_spec_copy["variant_id"] = variant_id
            self.models_config[variant_id] = ModelSpec(**model_spec_copy)

        if not benchmark_data["variants"]:
            raise ValueError("At least one variant must be specified in 'variants' section")

        self.benchmark_config = BenchmarkConfig(**benchmark_data)

        generation_data = data.get("generation", {})
        self.generation_config = GenerationConfig(**generation_data)

        experiment_data = data.get("experiment", {})
        self.experiment_config = ExperimentConfig(**experiment_data)

        logger.info(f"Loaded config successfully: {len(self.models_config)} variant(s)")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Export config as nested dict."""
        return {
            "benchmark": self.benchmark_config.model_dump() if self.benchmark_config else None,
            "models": {k: v.model_dump() for k, v in self.models_config.items()},
            "generation": self.generation_config.model_dump(),
            "experiment": self.experiment_config.model_dump() if self.experiment_config else None,
        }


def load_config(config_path: str | Path) -> ConfigManager:
    """Load and validate benchmark configuration from a unified config YAML file.

    Args:
        config_path: Path to a unified config YAML file (e.g. configs/config.yaml)

    Returns:
        ConfigManager instance with all configs loaded and validated

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If validation fails
    """
    config_path = Path(config_path)
    manager = ConfigManager()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Provide a unified config YAML file (e.g. configs/config.yaml)"
        )

    manager.load_from_unified_file(config_path)
    logger.info("Configuration loaded and validated")
    return manager
