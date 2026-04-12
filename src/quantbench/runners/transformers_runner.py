# Objective: Implement a Hugging Face Transformers backend runner.

from __future__ import annotations

import logging
from typing import Optional

from ..metrics import MetricsHelper
from ..models import PromptCase, RunResult, RunSpec
from .base import BaseRunner

logger = logging.getLogger(__name__)


class TransformersRunner(BaseRunner):
    """Execute prompts via Hugging Face Transformers pipeline API."""

    def __init__(
        self,
        run_spec: Optional[RunSpec],
        generation_config: dict,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: Optional[str] = None,
        measure_gpu_memory: bool = True,
    ):
        """Initialize TransformersRunner.

        Args:
            run_spec: RunSpec (may be None; set during first run_case if needed)
            generation_config: Generation parameters as dict (max_new_tokens, temperature, etc.)
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B")
            model_path: Local path to model (if not on HF hub)
            device: Device for inference ("cpu", "cuda", "cuda:0", "auto")
            dtype: Data type ("float32", "float16", "bfloat16", "auto")
            load_in_8bit: Load model in 8-bit (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit (requires bitsandbytes)
            bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
            measure_gpu_memory: Track GPU memory usage
        """
        super().__init__(run_spec, generation_config=generation_config)
        self.model_id = model_id or model_path
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.measure_gpu_memory = measure_gpu_memory

        # Lazy-loaded Transformers components
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def _resolve_device(self) -> str:
        """Resolve device string to actual device."""
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def _resolve_dtype(self):
        """Resolve dtype string to torch dtype."""
        try:
            import torch

            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "auto": None,
            }
            return dtype_map.get(self.dtype.lower(), None)
        except ImportError:
            return None

    def load(self) -> None:
        """Load model, tokenizer, and pipeline."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"transformers not installed; install with: pip install transformers torch"
            ) from e

        if not self.model_id:
            raise ValueError("Either model_id or model_path must be specified")

        device = self._resolve_device()
        dtype = self._resolve_dtype()

        logger.info(f"Loading model: {self.model_id} (device={device}, dtype={self.dtype})")

        # Prepare model kwargs for quantization
        model_kwargs = {}

        if self.load_in_8bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "bitsandbytes not installed; install with: "
                    "pip install bitsandbytes"
                ) from e
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
            logger.info("Using 8-bit quantization (bitsandbytes)")

        elif self.load_in_4bit:
            try:
                from bitsandbytes.integrations.bnb_4bit_compute_dtype_config import (
                    BitsAndBytesConfig,
                )
            except ImportError:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as e:
                    raise ImportError(
                        "bitsandbytes not installed; install with: pip install bitsandbytes"
                    ) from e

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_bnb_compute_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
            logger.info(
                f"Using 4-bit quantization (bitsandbytes, compute_dtype={self.bnb_4bit_compute_dtype})"
            )

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                **model_kwargs,
            )
            if device not in model_kwargs and device != "auto":
                self.model = self.model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_id}': {e}") from e

        logger.info(f"Model loaded successfully")

    def unload(self) -> None:
        """Unload model and free memory."""
        try:
            import torch

            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            torch.cuda.empty_cache()
            logger.info("Model unloaded and GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to unload model: {e}")

    def run_case(self, prompt_case: PromptCase) -> RunResult:
        """Execute a single prompt.

        Args:
            prompt_case: Prompt to execute

        Returns:
            RunResult with metrics and output
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded; call load() first")

        import torch

        # Prepare metrics collection
        metrics = MetricsHelper()
        metrics.start()

        try:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt_case.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            prompt_token_count = inputs["input_ids"].shape[1]

            device = self._resolve_device()
            if device not in ["auto", "cuda", "cpu"] or "cuda" in device:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.get("max_new_tokens", 128),
                    temperature=self.generation_config.get("temperature", 0.0),
                    top_p=self.generation_config.get("top_p", 1.0),
                    do_sample=self.generation_config.get("do_sample", False),
                    repetition_penalty=self.generation_config.get("repetition_penalty", 1.0),
                )

            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][prompt_token_count:],
                skip_special_tokens=True,
            )

            # Count generated tokens
            generated_token_count = outputs.shape[1] - prompt_token_count

            # Capture metrics
            metrics.capture()
            latency_sec = metrics.elapsed_seconds()
            tokens_per_sec = metrics.compute_tokens_per_sec(generated_token_count, latency_sec)

            # GPU memory if available
            peak_gpu_memory = None
            if self.measure_gpu_memory and torch.cuda.is_available():
                peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

            load_time_sec = None  # Not tracked separately for transformers

            result = RunResult(
                run_name=self.run_spec.name if self.run_spec else "transformers",
                backend="transformers",
                quantization=(
                    self.run_spec.quantization if self.run_spec else "unknown"
                ),
                prompt_id=prompt_case.id,
                task=prompt_case.task,
                model_ref=self.model_id or "unknown",
                prompt_chars=len(prompt_case.prompt),
                prompt_tokens=prompt_token_count,
                output_tokens=generated_token_count,
                latency_sec=latency_sec,
                tokens_per_sec=tokens_per_sec,
                load_time_sec=load_time_sec,
                peak_gpu_mem_mb=peak_gpu_memory,
                generated_text=generated_text,
            )

            return result

        except Exception as e:
            logger.error(f"Error during generation for prompt '{prompt_case.id}': {e}")
            raise

    def _get_bnb_compute_dtype(self):
        """Get bitsandbytes compute dtype."""
        try:
            import torch

            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                None: torch.float32,
            }
            return dtype_map.get(self.bnb_4bit_compute_dtype, torch.float32)
        except ImportError:
            return None
