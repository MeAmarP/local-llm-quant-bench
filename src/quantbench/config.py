# Objective: Load and validate benchmark configuration files.

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    """Phase 1 benchmark constraints and variant definitions (benchmark.yaml)."""

    project: str

    fixed: dict[str, Any] = Field(
        default_factory=dict,
        description="Fixed constraints (model_family, instruction_variant, hardware, framework_versions, generation params)"
    )

    variants: list[VariantSpec] = Field(
        default_factory=list,
        description="List of quantization variants to measure"
    )

    comparison_rules: dict[str, Any] = Field(
        default_factory=dict,
        description="Rules for valid comparison (e.g., model_artifact, model_scope)"
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
        valid = {"transformers", "llamacpp", "vllm", "onnx"}
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
    measure_gpu_memory: bool = True

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
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return data

    def load_from_dir(
        self,
        config_dir: str | Path,
        benchmark_name: str = "benchmark.yaml",
        models_name: str = "models.yaml",
        generation_name: str = "generation.yaml",
        experiment_name: str = "experiment.yaml",
    ) -> "ConfigManager":
        """Load all 4 config files from a directory.
        
        Args:
            config_dir: Directory containing benchmark.yaml, models.yaml, generation.yaml, experiment.yaml
            benchmark_name: Name of benchmark config file (default: benchmark.yaml)
            models_name: Name of models config file (default: models.yaml)
            generation_name: Name of generation config file (default: generation.yaml)
            experiment_name: Name of experiment config file (default: experiment.yaml)
        
        Returns:
            self (for chaining)
        
        Raises:
            FileNotFoundError: If required config files are missing
            ValueError: If validation fails (e.g., variant ID mismatch)
        """
        config_dir = Path(config_dir)

        # Load benchmark config
        benchmark_path = config_dir / benchmark_name
        benchmark_data = self.load_yaml(benchmark_path)
        self.benchmark_config = BenchmarkConfig(**benchmark_data)

        # Load models config
        models_path = config_dir / models_name
        models_data = self.load_yaml(models_path)
        for variant_id, spec_data in models_data.items():
            if not isinstance(spec_data, dict):
                raise ValueError(f"models.yaml: variant '{variant_id}' must be a dict")
            spec_data_copy = dict(spec_data)
            spec_data_copy["variant_id"] = variant_id
            self.models_config[variant_id] = ModelSpec(**spec_data_copy)

        # Validate variant ID consistency
        benchmark_variant_ids = {v.variant_id for v in self.benchmark_config.variants}
        models_variant_ids = set(self.models_config.keys())
        missing_in_models = benchmark_variant_ids - models_variant_ids
        extra_in_models = models_variant_ids - benchmark_variant_ids

        if missing_in_models:
            raise ValueError(
                f"Variant IDs in benchmark.yaml but missing in models.yaml: {missing_in_models}"
            )
        if extra_in_models:
            raise ValueError(
                f"Variant IDs in models.yaml but missing in benchmark.yaml: {extra_in_models}"
            )

        # Load generation config (optional; defaults provided)
        generation_path = config_dir / generation_name
        if generation_path.exists():
            generation_data = self.load_yaml(generation_path)
            self.generation_config = GenerationConfig(**generation_data)

        # Load experiment config
        experiment_path = config_dir / experiment_name
        experiment_data = self.load_yaml(experiment_path)
        self.experiment_config = ExperimentConfig(**experiment_data)

        return self

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
            raise RuntimeError("ConfigManager not initialized; call load_from_dir() first")

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
        
        Unified file format:
        ```yaml
        project: local-llm-quant-bench        
        benchmark:
          model_family: "Qwen2.5-3B-Instruct"
          instruction_variant: "Instruct"
          hardware: "NVIDIA RTX 4090"
          # ... other fixed params ...
          max_new_tokens: 256
          stop_conditions: [...]
          metrics: [...]
        
        generation:
          max_new_tokens: 256
          temperature: 0.0
          # ... other generation params ...
        
        variants:
          baseline:
            model_id: "..."
            backend: "transformers"
            # ...
          int8:
            # ...
        
        experiment:
          prompt_file: "prompts/prompts.jsonl"
          output_dir: "results/runs"
          # ...
        ```
        
        Args:
            config_path: Path to unified config.yaml file
        
        Returns:
            self (for chaining)
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If validation fails
        """
        config_path = Path(config_path)
        data = self.load_yaml(config_path)
        
        # Extract top-level metadata
        project = data.get("project", "local-llm-quant-bench")
            
        
        # Extract benchmark config
        benchmark_data = data.get("benchmark", {})
        benchmark_data["project"] = project
        
        # Extract variants from the unified format
        variants_data = data.get("variants", {})
        if not isinstance(variants_data, dict):
            raise ValueError("'variants' must be a dictionary in unified config file")
        
        benchmark_data["variants"] = []
        self.models_config = {}
        
        # Process each variant
        for variant_id, variant_spec in variants_data.items():
            if not isinstance(variant_spec, dict):
                raise ValueError(f"Variant '{variant_id}' must be a dict")
            
            # Extract variant-level metadata for BenchmarkConfig.variants list
            variant_spec_copy = dict(variant_spec)
            variant_spec_copy["variant_id"] = variant_id
            variant_spec_copy.setdefault("quant_family", variant_spec.get("quant_family", "unknown"))
            variant_spec_copy.setdefault("precision", variant_spec.get("precision", "unknown"))
            variant_spec_copy.setdefault("backend", variant_spec.get("backend", "unknown"))
            
            benchmark_data["variants"].append(VariantSpec(**variant_spec_copy))
            
            # Build ModelSpec for this variant
            model_spec_copy = dict(variant_spec)
            model_spec_copy["variant_id"] = variant_id
            self.models_config[variant_id] = ModelSpec(**model_spec_copy)
        
        # Validate we have at least one variant
        if not benchmark_data["variants"]:
            raise ValueError("At least one variant must be specified in 'variants' section")
        
        # Create BenchmarkConfig
        self.benchmark_config = BenchmarkConfig(**benchmark_data)
        
        # Load generation config (optional; uses defaults)
        generation_data = data.get("generation", {})
        self.generation_config = GenerationConfig(**generation_data)
        
        # Load experiment config
        experiment_data = data.get("experiment", {})
        self.experiment_config = ExperimentConfig(**experiment_data)
        
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
    """Load and validate benchmark configuration.
    
    Supports both single unified file (config.yaml) and multi-file directory approach.
    
    Args:
        config_path: Either:
            - Path to unified config.yaml file (recommended)
            - Path to directory containing benchmark.yaml, models.yaml, generation.yaml, experiment.yaml
    
    Returns:
        ConfigManager instance with all configs loaded and validated
    
    Raises:
        FileNotFoundError: If required files missing
        ValueError: If validation fails
    """
    config_path = Path(config_path)
    manager = ConfigManager()
    
    # Detect if it's a file or directory
    if config_path.is_file():
        # Single unified config file
        manager.load_from_unified_file(config_path)
    elif config_path.is_dir():
        # Multi-file directory approach (backward compatibility)
        manager.load_from_dir(config_path)
    else:
        raise FileNotFoundError(
            f"Config path does not exist: {config_path}\n"
            "Provide either:\n"
            "  - Path to config.yaml file\n"
            "  - Path to directory containing benchmark.yaml, models.yaml, etc."
        )
    
    return manager

