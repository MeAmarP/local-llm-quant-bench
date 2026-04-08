# Objective: Orchestrate end-to-end benchmark workflow across variants and prompts.

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

from src.quantbench.config import ConfigManager, ExperimentConfig
from src.quantbench.models import PromptCase, RunResult
from src.quantbench.prompts import load_prompts
from src.quantbench.runners.base import BaseRunner
from src.quantbench.runners.llamacpp_runner import LlamaCppRunner
from src.quantbench.runners.transformers_runner import TransformersRunner
from src.quantbench.utils.system_info import capture_system_info


logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """Orchestrates full benchmark workflow: init → variants → summarize → system info."""

    def __init__(
        self,
        config_manager: ConfigManager,
        prompts: list[PromptCase],
        output_dir: str | Path = "results/runs",
    ):
        """Initialize orchestrator.

        Args:
            config_manager: Loaded ConfigManager with all config files
            prompts: List of prompts to benchmark
            output_dir: Root directory for run artifacts
        """
        self.config = config_manager
        self.prompts = prompts
        self.output_dir = Path(output_dir)
        self.run_dir: Optional[Path] = None
        self.runners: dict[str, BaseRunner] = {}
        self.observations: list[dict] = []

    def initialize_run(self) -> Path:
        """Create run directory with config and prompt snapshots.

        Returns:
            Path to the created run directory
        """
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized run directory: {self.run_dir}")

        # Save config and prompts snapshots
        if self.config.benchmark_config:
            config_snapshot = self.run_dir / "config_snapshot.yaml"
            config_snapshot.write_text(
                json.dumps(self.config.to_dict(), indent=2),
                encoding="utf-8",
            )

        prompts_snapshot = self.run_dir / "prompts_snapshot.jsonl"
        with prompts_snapshot.open("w", encoding="utf-8") as f:
            for prompt in self.prompts:
                f.write(json.dumps(prompt.model_dump()) + "\n")

        # Initialize observations file
        obs_path = self.run_dir / "observations.jsonl"
        obs_path.write_text("", encoding="utf-8")

        # Create notes file
        notes_path = self.run_dir / "notes.md"
        notes_path.write_text("# Run Notes\n\n", encoding="utf-8")

        # Save run metadata
        meta = {
            "run_id": run_id,
            "created_at": dt.datetime.now().isoformat(),
            "num_prompts": len(self.prompts),
            "num_variants": len(self.config.benchmark_config.variants)
            if self.config.benchmark_config
            else 0,
        }
        (self.run_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        return self.run_dir

    def get_runner(self, variant_id: str) -> Optional[BaseRunner]:
        """Get or create runner for a variant.

        Args:
            variant_id: Variant identifier

        Returns:
            Initialized BaseRunner subclass, or None if initialization fails

        Raises:
            ValueError: If variant config not found
        """
        if variant_id in self.runners:
            return self.runners[variant_id]

        variant_spec, model_spec, generation_config = self.config.get_variant_config(variant_id)

        backend = variant_spec.backend.lower()
        runner: Optional[BaseRunner] = None

        try:
            if backend == "llamacpp":
                runner = LlamaCppRunner(
                    run_spec=None,  # Will be built per-batch
                    generation_config=generation_config.model_dump(),
                    model_path=model_spec.model_path,
                    device=self.config.experiment_config.device.lower()
                    if self.config.experiment_config
                    else "auto",
                    timeout_sec=300,
                    measure_gpu_memory=self.config.experiment_config.measure_gpu_memory
                    if self.config.experiment_config
                    else True,
                )
                logger.info(f"Initialized LlamaCppRunner for variant '{variant_id}'")

            elif backend == "transformers":
                runner = TransformersRunner(
                    run_spec=None,
                    generation_config=generation_config.model_dump(),
                    model_id=model_spec.model_id,
                    model_path=model_spec.model_path,
                    device=self.config.experiment_config.device.lower()
                    if self.config.experiment_config
                    else "auto",
                    dtype=self.config.experiment_config.dtype.lower()
                    if self.config.experiment_config
                    else "auto",
                    load_in_8bit=model_spec.load_in_8bit,
                    load_in_4bit=model_spec.load_in_4bit,
                    bnb_4bit_compute_dtype=model_spec.bnb_4bit_compute_dtype,
                    measure_gpu_memory=self.config.experiment_config.measure_gpu_memory
                    if self.config.experiment_config
                    else True,
                )
                logger.info(f"Initialized TransformersRunner for variant '{variant_id}'")

            else:
                logger.warning(f"Unsupported backend '{backend}' for variant '{variant_id}'")
                return None

            if runner:
                self.runners[variant_id] = runner
                return runner

        except Exception as e:
            logger.error(f"Failed to initialize runner for variant '{variant_id}': {e}")
            return None

        return None

    def log_observation(
        self,
        variant_id: str,
        prompt_id: str,
        result: RunResult,
    ) -> None:
        """Append an observation to observations.jsonl.

        Args:
            variant_id: Variant identifier
            prompt_id: Prompt identifier
            result: RunResult from runner
        """
        if not self.run_dir:
            raise RuntimeError("Run not initialized; call initialize_run() first")

        obs = {
            "timestamp": dt.datetime.now().isoformat(),
            "variant_id": variant_id,
            "prompt_id": prompt_id,
            "wall_clock_latency_ms": result.latency_sec * 1000 if result.latency_sec else 0,
            "generated_tokens": result.output_tokens,
            "tokens_per_sec": result.tokens_per_sec or 0.0,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
            "peak_gpu_memory_mb": result.peak_gpu_mem_mb,
            "model_load_time_ms": (result.load_time_sec * 1000) if result.load_time_sec else 0,
            "notes": "",
        }

        obs_path = self.run_dir / "observations.jsonl"
        with obs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obs) + "\n")

        self.observations.append(obs)

    def run_benchmark(self) -> None:
        """Execute full benchmark: iterate variants → prompts → repetitions.

        For each variant:
            - Initialize runner
            - For each prompt:
                - For each repetition:
                    - Execute run_case()
                    - Log observation
        """
        if not self.run_dir:
            raise RuntimeError("Run not initialized; call initialize_run() first")

        if not self.config.benchmark_config:
            raise RuntimeError("Benchmark config not loaded")

        if not self.config.experiment_config:
            raise RuntimeError("Experiment config not loaded")

        num_repetitions = self.config.experiment_config.repetitions
        num_warmup = self.config.experiment_config.warmup_runs

        failed_variants = []

        for variant_spec in self.config.benchmark_config.variants:
            variant_id = variant_spec.variant_id

            logger.info(f"Starting variant: {variant_id} (backend={variant_spec.backend})")

            runner = self.get_runner(variant_id)
            if not runner:
                logger.error(f"Skipping variant '{variant_id}' due to initialization failure")
                failed_variants.append(variant_id)
                continue

            try:
                runner.load()

                for prompt in self.prompts:
                    # Warmup runs (not logged)
                    for warmup_idx in range(num_warmup):
                        try:
                            runner.run_case(prompt)
                        except Exception as e:
                            logger.warning(
                                f"Warmup {warmup_idx + 1}/{num_warmup} failed for "
                                f"variant={variant_id}, prompt={prompt.id}: {e}"
                            )

                    # Timed repetitions (logged)
                    for rep_idx in range(num_repetitions):
                        try:
                            result = runner.run_case(prompt)
                            self.log_observation(variant_id, prompt.id, result)
                            logger.debug(
                                f"✓ variant={variant_id}, prompt={prompt.id}, "
                                f"rep={rep_idx + 1}/{num_repetitions}: "
                                f"{result.output_tokens} tokens @ {result.tokens_per_sec:.1f} tok/s"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to run variant='{variant_id}', prompt='{prompt.id}', "
                                f"rep={rep_idx + 1}/{num_repetitions}: {e}"
                            )

                runner.unload()
                logger.info(f"Completed variant: {variant_id}")

            except Exception as e:
                logger.error(f"Unexpected error in variant '{variant_id}': {e}")
                failed_variants.append(variant_id)
                try:
                    runner.unload()
                except Exception as unload_err:
                    logger.warning(f"Failed to unload runner for '{variant_id}': {unload_err}")

        if failed_variants:
            logger.warning(
                f"Benchmark completed with {len(failed_variants)} failed variants: {failed_variants}"
            )

    def finalize_run(self) -> None:
        """Generate summary artifacts and capture system info.

        Produces:
        - summary.json: Aggregated statistics per variant
        - summary.md: Markdown table of results
        - system_snapshot.json: System metadata
        """
        if not self.run_dir:
            raise RuntimeError("Run not initialized; call initialize_run() first")

        logger.info("Finalizing run: generating summaries and system snapshot")

        # Build summary from observations
        summary = self._build_summary()
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        logger.info("Wrote summary.json")

        # Write summary markdown
        summary_md = self._build_summary_markdown(summary)
        (self.run_dir / "summary.md").write_text(summary_md, encoding="utf-8")
        logger.info("Wrote summary.md")

        # Capture system info
        try:
            system_info = capture_system_info()
            (self.run_dir / "system_snapshot.json").write_text(
                json.dumps(system_info, indent=2), encoding="utf-8"
            )
            logger.info("Wrote system_snapshot.json")
        except Exception as e:
            logger.error(f"Failed to capture system info: {e}")

    def _build_summary(self) -> dict:
        """Build aggregated statistics from observations.

        Returns:
            Dict with per-variant statistics (latency, throughput, memory)
        """
        from collections import defaultdict
        from statistics import mean, median, stdev

        summary = {}
        variant_data = defaultdict(list)

        # Group observations by variant
        for obs in self.observations:
            variant_id = obs["variant_id"]
            variant_data[variant_id].append(obs)

        # Compute stats per variant
        for variant_id, obs_list in variant_data.items():
            if not obs_list:
                continue

            latencies = [o["wall_clock_latency_ms"] for o in obs_list]
            throughputs = [o["tokens_per_sec"] for o in obs_list if o["tokens_per_sec"] > 0]
            memories = [o["peak_gpu_memory_mb"] for o in obs_list if o["peak_gpu_memory_mb"]]

            summary[variant_id] = {
                "num_runs": len(obs_list),
                "latency_ms": {
                    "median": median(latencies),
                    "mean": mean(latencies),
                    "stdev": stdev(latencies) if len(latencies) > 1 else 0,
                    "min": min(latencies),
                    "max": max(latencies),
                },
                "tokens_per_sec": {
                    "median": median(throughputs) if throughputs else 0,
                    "mean": mean(throughputs) if throughputs else 0,
                    "min": min(throughputs) if throughputs else 0,
                    "max": max(throughputs) if throughputs else 0,
                },
                "peak_gpu_memory_mb": {
                    "median": median(memories) if memories else None,
                    "mean": mean(memories) if memories else None,
                    "max": max(memories) if memories else None,
                }
                if memories
                else None,
            }

        return summary

    def _build_summary_markdown(self, summary: dict) -> str:
        """Build markdown table from summary statistics.

        Args:
            summary: Output from _build_summary()

        Returns:
            Markdown string with formatted table
        """
        lines = ["# Benchmark Summary\n"]

        if not summary:
            lines.append("No results to summarize.\n")
            return "\n".join(lines)

        lines.append("| Variant | Latency (ms) | Throughput (tok/s) | Peak GPU Mem (MB) |")
        lines.append("|---------|--------------|-------------------|------------------|")

        for variant_id in sorted(summary.keys()):
            stats = summary[variant_id]
            latency_median = stats["latency_ms"]["median"]
            throughput_median = stats["tokens_per_sec"]["median"]
            memory_max = stats["peak_gpu_memory_mb"]["max"] if stats["peak_gpu_memory_mb"] else "N/A"

            lines.append(
                f"| {variant_id} | {latency_median:.1f} | {throughput_median:.1f} | {memory_max} |"
            )

        return "\n".join(lines) + "\n"

    def run(self) -> Path:
        """Execute full benchmark workflow: init → benchmark → finalize.

        Returns:
            Path to the run directory

        Raises:
            RuntimeError: If workflow fails
        """
        try:
            logger.info("Starting benchmark workflow")
            self.initialize_run()
            self.run_benchmark()
            self.finalize_run()
            logger.info(f"Benchmark completed successfully: {self.run_dir}")
            return self.run_dir
        except Exception as e:
            logger.error(f"Benchmark workflow failed: {e}")
            raise
