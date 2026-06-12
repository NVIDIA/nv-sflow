# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for SubprocessLauncher log routing.

In-task (subprocess/server) output is high volume and must live only in the per-task
log (``output_logger`` -> ``<task>.log``). It is echoed to the shared ``sflow`` logger
(for the interactive console / TUI pane) only when stdout is a real TTY, and it must
never be persisted to ``sflow.log`` at any log level. Command banners and orchestration
messages stay on the shared logger as "hint and command logs".
"""

import logging

import pytest

from sflow.core.launcher import SubprocessLauncher
from sflow.logging import IN_TASK_OUTPUT_ATTR, add_log_file

_SENTINEL = "SENTINEL_TASK_OUTPUT_LINE"


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


@pytest.fixture
def shared_logger_capture():
    """Capture records reaching the shared ``sflow`` logger (console + sflow.log)."""
    logger = logging.getLogger("sflow")
    handler = _ListHandler()
    prev_level = logger.level
    logger.addHandler(handler)
    try:
        yield logger, handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def _make_task_logger(name: str) -> tuple[logging.Logger, _ListHandler]:
    """Mimic the per-task logger configured in SflowApp.run (INFO, non-propagating)."""
    task_logger = logging.getLogger(f"sflow.task.{name}")
    task_logger.handlers = []
    task_logger.setLevel(logging.INFO)
    task_logger.propagate = False
    handler = _ListHandler()
    handler.setLevel(logging.INFO)
    task_logger.addHandler(handler)
    return task_logger, handler


def test_in_task_output_not_echoed_when_not_tty(shared_logger_capture):
    """Non-TTY (sflow batch / redirected stdout): per-task log only, even at DEBUG."""
    shared_logger, shared_handler = shared_logger_capture
    shared_logger.setLevel(logging.DEBUG)
    task_logger, task_handler = _make_task_logger("notty_unit")

    SubprocessLauncher(echo_to_console=False)._emit_subprocess_line(
        _SENTINEL, prefix="[notty_unit] ", output_logger=task_logger
    )

    # Full in-task output lands in the per-task log...
    assert task_handler.messages == [_SENTINEL]
    # ...and is never echoed to the shared sflow logger, regardless of log level.
    assert not any(_SENTINEL in m for m in shared_handler.messages)


def test_in_task_output_echoed_to_console_when_tty(shared_logger_capture):
    """Interactive TTY: per-task log AND a marked echo to the shared logger."""
    shared_logger, shared_handler = shared_logger_capture
    shared_logger.setLevel(logging.INFO)
    task_logger, task_handler = _make_task_logger("tty_unit")

    SubprocessLauncher(echo_to_console=True)._emit_subprocess_line(
        _SENTINEL, prefix="[tty_unit] ", output_logger=task_logger
    )

    # Per-task log gets the bare line.
    assert task_handler.messages == [_SENTINEL]
    # Shared logger gets the prefixed echo, carrying the in-task marker so the
    # sflow.log file handler can filter it out.
    echoed = [r for r in shared_handler.records if _SENTINEL in r.getMessage()]
    assert echoed, "expected an in-task echo on the shared logger"
    assert all(getattr(r, IN_TASK_OUTPUT_ATTR, False) for r in echoed)
    assert any("[tty_unit] " in r.getMessage() for r in echoed)


def test_no_extra_echo_when_output_logger_propagates(shared_logger_capture):
    """Slurm allocation path: a propagating output_logger already reaches the shared
    logger, so the launcher must not add a second (marked) echo."""
    shared_logger, shared_handler = shared_logger_capture
    shared_logger.setLevel(logging.INFO)
    slurm_logger = logging.getLogger("sflow.fake_slurm")
    slurm_logger.handlers = []
    slurm_logger.setLevel(logging.INFO)
    slurm_logger.propagate = True

    SubprocessLauncher(echo_to_console=True)._emit_subprocess_line(
        _SENTINEL, prefix="[fake_slurm] ", output_logger=slurm_logger
    )

    # No marked in-task echo was added by the launcher.
    assert [r for r in shared_handler.records if getattr(r, IN_TASK_OUTPUT_ATTR, False)] == []
    # The bare line still reaches the shared logger via the propagating output_logger
    # (kept as a command log), without the console prefix.
    assert any(_SENTINEL in m for m in shared_handler.messages)
    assert not any("[fake_slurm] " in m for m in shared_handler.messages)


def test_empty_line_is_dropped(shared_logger_capture):
    shared_logger, shared_handler = shared_logger_capture
    shared_logger.setLevel(logging.DEBUG)
    task_logger, task_handler = _make_task_logger("empty_unit")

    SubprocessLauncher(echo_to_console=True)._emit_subprocess_line(
        "", prefix="[empty_unit] ", output_logger=task_logger
    )

    assert task_handler.messages == []
    assert shared_handler.messages == []


def test_add_log_file_excludes_in_task_output(tmp_path):
    """sflow.log keeps command/hint logs but drops marked in-task output records."""
    log_path = tmp_path / "sflow.log"
    sflow_logger = logging.getLogger("sflow")
    prev_level = sflow_logger.level

    add_log_file(str(log_path))
    # configure_logging sets the sflow logger to INFO in production; mirror that here
    # so INFO records reach the file handler under test.
    sflow_logger.setLevel(logging.INFO)
    added = [
        h
        for h in sflow_logger.handlers
        if isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == str(log_path)
    ]
    assert added, "add_log_file did not attach a file handler"

    try:
        sflow_logger.info("KEEP_THIS_COMMAND_LOG")
        sflow_logger.info(
            "DROP_THIS_TASK_OUTPUT", extra={IN_TASK_OUTPUT_ATTR: True}
        )
        for h in added:
            h.flush()
        contents = log_path.read_text()
        assert "KEEP_THIS_COMMAND_LOG" in contents
        assert "DROP_THIS_TASK_OUTPUT" not in contents
    finally:
        for h in added:
            sflow_logger.removeHandler(h)
            h.close()
        sflow_logger.setLevel(prev_level)
