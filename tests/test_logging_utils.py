from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.quantbench.utils.logging_utils import get_logger, setup_logging


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("quantbench.test.logger")
    assert logger.name == "quantbench.test.logger"


def test_setup_logging_adds_stream_handler() -> None:
    logger = setup_logging(logger_name="quantbench.test.stream", force=True)
    stream_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert stream_handlers


def test_setup_logging_adds_file_handler(tmp_path: Path) -> None:
    log_path = tmp_path / "bench.log"
    logger = setup_logging(logger_name="quantbench.test.file", log_file=log_path, force=True)
    logger.info("hello file")

    assert log_path.exists()
    assert "hello file" in log_path.read_text(encoding="utf-8")


def test_setup_logging_does_not_duplicate_handlers(tmp_path: Path) -> None:
    log_path = tmp_path / "dup.log"
    name = "quantbench.test.dup"
    logger = setup_logging(logger_name=name, log_file=log_path, force=True)
    initial_handlers = len(logger.handlers)

    logger = setup_logging(logger_name=name, log_file=log_path, force=False)
    assert len(logger.handlers) == initial_handlers


def test_setup_logging_invalid_level_raises() -> None:
    with pytest.raises(ValueError):
        setup_logging(level="NOT_A_LEVEL", logger_name="quantbench.test.badlevel", force=True)
