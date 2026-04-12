# Objective: Command-line interface for benchmark orchestration.

import argparse
import logging
import sys
from pathlib import Path

from src.quantbench.config import load_config
from src.quantbench.orchestrator import BenchmarkOrchestrator
from src.quantbench.prompts import load_prompts
from src.quantbench.utils.logging_utils import setup_logging


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="local-llm-quant-bench: Unified benchmark runner for AI model quantization.",
        epilog=(
            "Examples:\n"
            "  # Using unified config file (recommended)\n"
            "  quantbench --config configs/config.yaml\n\n"
            "  # Using multi-file directory (legacy)\n"
            "  quantbench --config-dir configs/\n\n"
            "  # With custom prompts and output\n"
            "  quantbench --config configs/config.yaml --prompts my_prompts.jsonl --output-dir results/my_exp"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config source: mutually exclusive group (at least one required)
    config_group = parser.add_mutually_exclusive_group(required=True)
    
    config_group.add_argument(
        "--config",
        type=Path,
        help="Path to unified config.yaml file (recommended approach)",
    )

    config_group.add_argument(
        "--config-dir",
        type=Path,
        help="Directory containing benchmark.yaml, models.yaml, generation.yaml, experiment.yaml (legacy approach)",
    )

    parser.add_argument(
        "--prompts",
        required=False,
        type=Path,
        help="Path to prompts JSONL file (overrides prompt_file in config)",
    )

    parser.add_argument(
        "--output-dir",
        default="results/runs",
        type=Path,
        help="Root directory for run artifacts (default: results/runs)",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional log file path",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = logging.getLogger(__name__)

    try:
        # Determine config path (either unified file or directory)
        config_path = args.config if args.config else args.config_dir
        config_type = "unified config file" if args.config else "config directory"
        
        # Load configuration
        logger.info(f"Loading configuration from {config_type}: {config_path}")
        config_manager = load_config(config_path)
        logger.info("Configuration loaded successfully")

        # Load prompts
        prompts_path = (
            args.prompts
            if args.prompts
            else (
                Path(config_manager.experiment_config.prompt_file)
                if config_manager.experiment_config
                else None
            )
        )

        if not prompts_path:
            parser.error("No prompts file specified; provide --prompts or check experiment.yaml")

        logger.info(f"Loading prompts from: {prompts_path}")
        prompts = load_prompts(prompts_path)
        logger.info(f"Loaded {len(prompts)} prompts")

        # Run benchmark
        logger.info(f"Starting benchmark with {len(prompts)} prompts")
        orchestrator = BenchmarkOrchestrator(
            config_manager=config_manager,
            prompts=prompts,
            output_dir=args.output_dir,
        )

        run_dir = orchestrator.run()
        logger.info(f"Benchmark completed: {run_dir}")
        print(str(run_dir))

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
