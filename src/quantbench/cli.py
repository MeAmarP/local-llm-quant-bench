# Objective: Command-line interface for benchmark orchestration.

import argparse
import logging
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return repository root inferred from this file location."""
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw_path: Path, *, config_path: Path | None = None) -> Path:
    """Resolve a potentially relative path across common project anchors."""
    expanded = raw_path.expanduser()
    if expanded.is_absolute():
        return expanded

    candidates: list[Path] = [Path.cwd() / expanded]

    if config_path is not None:
        cfg = config_path.expanduser()
        cfg_abs = cfg if cfg.is_absolute() else (Path.cwd() / cfg)
        candidates.append(cfg_abs.parent / expanded)
        candidates.append(cfg_abs.parent.parent / expanded)

    candidates.append(_repo_root() / expanded)

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            return normalized

    # Keep original relative path for clear downstream error messages.
    return expanded


def _load_runtime_dependencies():
    """Import runtime modules in a way that supports both package and script execution."""
    if __package__:
        from .config import load_config
        from .orchestrator import BenchmarkOrchestrator
        from .prompts import load_prompts
        from .utils.logging_utils import setup_logging
        return load_config, BenchmarkOrchestrator, load_prompts, setup_logging

    # Support direct execution, e.g. `python quantbench/cli.py --help` from `src/`.
    current_file = Path(__file__).resolve()
    src_root = current_file.parents[1]
    repo_root = _repo_root()

    for candidate in (str(src_root), str(repo_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    from quantbench.config import load_config
    from quantbench.orchestrator import BenchmarkOrchestrator
    from quantbench.prompts import load_prompts
    from quantbench.utils.logging_utils import setup_logging
    return load_config, BenchmarkOrchestrator, load_prompts, setup_logging


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
        help="Log file path (default: ./quantbench.log)",
    )

    args = parser.parse_args()

    load_config, BenchmarkOrchestrator, load_prompts, setup_logging = _load_runtime_dependencies()

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
        configured_prompt_path = (
            Path(config_manager.experiment_config.prompt_file)
            if config_manager.experiment_config
            else None
        )
        prompts_path = args.prompts or configured_prompt_path or Path("prompts/prompts.jsonl")

        if not prompts_path:
            parser.error("No prompts file specified; provide --prompts or check experiment.yaml")

        prompts_path = _resolve_path(prompts_path, config_path=config_path)
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
