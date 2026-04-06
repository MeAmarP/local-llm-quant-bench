import time

import pytest

from src.quantbench.utils.timers import Timer, elapsed_milliseconds, elapsed_seconds, time_block


def test_elapsed_seconds_and_milliseconds() -> None:
    sec = elapsed_seconds(1.0, 2.5)
    ms = elapsed_milliseconds(1.0, 2.5)
    assert sec == 1.5
    assert ms == 1500.0


def test_elapsed_seconds_invalid_range() -> None:
    with pytest.raises(ValueError):
        elapsed_seconds(3.0, 2.0)


def test_timer_lifecycle() -> None:
    timer = Timer()
    timer.start()
    assert timer.is_running is True
    time.sleep(0.001)
    timer.stop()
    assert timer.is_running is False
    assert timer.elapsed_sec >= 0
    assert timer.elapsed_ms >= 0


def test_timer_stop_without_start_fails() -> None:
    timer = Timer()
    with pytest.raises(RuntimeError):
        timer.stop()


def test_timer_elapsed_without_start_fails() -> None:
    timer = Timer()
    with pytest.raises(RuntimeError):
        _ = timer.elapsed_sec


def test_timer_context_manager() -> None:
    with Timer() as timer:
        time.sleep(0.001)
    assert timer.elapsed_ms >= 0


def test_time_block_context_manager() -> None:
    with time_block() as timer:
        time.sleep(0.001)
    assert timer.elapsed_sec >= 0
