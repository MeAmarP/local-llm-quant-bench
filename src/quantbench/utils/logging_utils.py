# Objective: Centralize logging setup and logger helpers for benchmark scripts.

from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_LOG_FILE = Path("quantbench.log")
DEFAULT_LOG_FMT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)


def setup_logging(
    level: str | int = "INFO",
    *,
    logger_name: str | None = None,
    log_file: str | Path | None = None,
    fmt: str = DEFAULT_LOG_FMT,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    force: bool = False,
) -> logging.Logger:
    """
    Configure a consistent logger for benchmark modules and scripts.

    - Adds a console handler by default.
    - Adds a file handler (`quantbench.log`) by default.
    - Avoids duplicate handlers across repeated calls unless `force=True`.
    """
    # Use root logger by default so all module loggers share one global configuration.
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(_normalize_level(level))

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if not _has_stream_handler(logger):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    log_path = Path(log_file) if log_file is not None else DEFAULT_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not _has_file_handler(logger, log_path):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid duplicate messages via root propagation when using named loggers.
    if logger_name:
        logger.propagate = False

    return logger


def _normalize_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    if not isinstance(level, str):
        raise TypeError("level must be str or int")

    # Accept standard level names (e.g. "INFO", "debug") and custom registered names.
    normalized = level.strip().upper()
    resolved = logging.getLevelNamesMapping().get(normalized)
    if resolved is None:
        raise ValueError(f"Unknown log level: {level}")
    return resolved


def _has_stream_handler(logger: logging.Logger) -> bool:
    return any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    target = str(path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve().as_posix() == Path(target).as_posix():
                    return True
            except Exception:
                continue
    return False
