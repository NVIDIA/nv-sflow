# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from sflow.core.task_logging import (
    BoundedRotatingTaskLogHandler,
    TaskLogPolicy,
    TaskOutputSink,
    create_task_log_handler,
)


def _make_logger(name: str, handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def test_bounded_task_log_handler_suppresses_and_summarizes(tmp_path):
    log_path = tmp_path / "task.log"
    policy = TaskLogPolicy(
        mode="bounded",
        keep_lines_per_second=0,
        keep_first_lines=1,
        max_bytes=1024 * 1024,
        backup_count=1,
    )
    handler = create_task_log_handler(log_path, policy)
    logger = _make_logger("sflow.tests.task_logging.suppression", handler)

    logger.info("line one kept")
    logger.info("line two dropped")
    logger.info("line three dropped")
    handler.close()

    contents = log_path.read_text()
    assert "line one kept" in contents
    assert "line two dropped" not in contents
    assert "line three dropped" not in contents
    assert "sflow: suppressed 2 task log lines due to rate limit" in contents


def test_bounded_task_log_handler_rotates_task_log_files(tmp_path):
    log_path = tmp_path / "task.log"
    policy = TaskLogPolicy(
        mode="bounded",
        keep_lines_per_second=1000,
        keep_first_lines=1000,
        max_bytes=180,
        backup_count=2,
    )
    handler = create_task_log_handler(log_path, policy)
    logger = _make_logger("sflow.tests.task_logging.rotation", handler)

    for idx in range(20):
        logger.info("rotation line %02d with enough content to exceed max bytes", idx)
    handler.close()

    assert log_path.exists()
    assert (tmp_path / "task.log.1").exists()
    assert log_path.stat().st_size <= policy.max_bytes


def test_full_task_log_policy_uses_unbounded_file_handler(tmp_path):
    log_path = tmp_path / "task.log"
    policy = TaskLogPolicy(mode="full")
    handler = create_task_log_handler(log_path, policy)
    logger = _make_logger("sflow.tests.task_logging.full", handler)

    logger.info("full line")
    handler.close()

    assert "full line" in log_path.read_text()
    assert not (tmp_path / "task.log.1").exists()


def test_bounded_task_log_handler_refills_tokens_over_time(tmp_path):
    log_path = tmp_path / "task.log"
    now = 0.0

    def clock() -> float:
        return now

    policy = TaskLogPolicy(
        mode="bounded",
        keep_lines_per_second=2,
        keep_first_lines=1,
        max_bytes=1024 * 1024,
        backup_count=1,
    )
    handler = BoundedRotatingTaskLogHandler(log_path, policy, clock=clock)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = _make_logger("sflow.tests.task_logging.refill", handler)

    logger.info("initial kept")
    logger.info("initial dropped")
    now = 0.5
    logger.info("refilled kept")
    handler.close()

    contents = log_path.read_text()
    assert "initial kept" in contents
    assert "initial dropped" not in contents
    assert "sflow: suppressed 1 task log lines due to rate limit" in contents
    assert "refilled kept" in contents


def test_bounded_task_log_handler_refills_when_keep_first_lines_is_zero(tmp_path):
    log_path = tmp_path / "task.log"
    now = 0.0

    def clock() -> float:
        return now

    policy = TaskLogPolicy(
        mode="bounded",
        keep_lines_per_second=2,
        keep_first_lines=0,
        max_bytes=1024 * 1024,
        backup_count=1,
    )
    handler = BoundedRotatingTaskLogHandler(log_path, policy, clock=clock)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = _make_logger("sflow.tests.task_logging.zero_initial_refill", handler)

    logger.info("initial dropped")
    now = 0.5
    logger.info("refilled kept")
    handler.close()

    contents = log_path.read_text()
    assert "initial dropped" not in contents
    assert "sflow: suppressed 1 task log lines due to rate limit" in contents
    assert "refilled kept" in contents


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "invalid"}, "Invalid task log mode"),
        ({"keep_lines_per_second": -1}, "keep_lines_per_second"),
        ({"keep_first_lines": -1}, "keep_first_lines"),
        ({"max_bytes": -1}, "max_bytes"),
        ({"backup_count": -1}, "backup_count"),
    ],
)
def test_task_log_policy_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TaskLogPolicy(**kwargs)


class _CountingLogger:
    name = "sflow.task.counting"

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_full_task_output_sink_uses_logger_info():
    logger = _CountingLogger()
    sink = TaskOutputSink(logger=logger, policy=TaskLogPolicy(mode="full"))

    sink.emit_line("full line")

    assert logger.messages == ["full line"]


def test_bounded_task_output_sink_suppresses_before_logger_info(tmp_path):
    log_path = tmp_path / "task.log"
    handler = create_task_log_handler(
        log_path,
        TaskLogPolicy(
            mode="bounded",
            keep_lines_per_second=0,
            keep_first_lines=1,
            max_bytes=1024 * 1024,
            backup_count=1,
        ),
    )
    logger = _make_logger("sflow.tests.task_logging.sink", handler)
    sink = TaskOutputSink(
        logger=logger,
        policy=TaskLogPolicy(
            mode="bounded",
            keep_lines_per_second=0,
            keep_first_lines=1,
            max_bytes=1024 * 1024,
            backup_count=1,
        ),
    )
    original_info = logger.info
    info_calls = 0

    def counting_info(message: str) -> None:
        nonlocal info_calls
        info_calls += 1
        original_info(message)

    logger.info = counting_info  # type: ignore[method-assign]

    sink.emit_line("kept")
    sink.emit_line("dropped one")
    sink.emit_line("dropped two")
    sink.close()

    contents = log_path.read_text()
    assert info_calls == 0
    assert "kept" in contents
    assert "dropped one" not in contents
    assert "dropped two" not in contents
    assert "sflow: suppressed 2 task log lines due to rate limit" in contents
