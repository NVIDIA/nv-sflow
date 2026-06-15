# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for how SubprocessLauncher routes per-task output to loggers.

Per-task subprocess output must always be written to the per-task logger
(``output_logger``), but only additionally echoed to the root ``sflow`` logger
(console / slurm stdout) for interactive TTY sessions, and only when tagged with
``SFLOW_TASK_STREAM_ATTR`` so sflow.log's file handler can drop it.
"""

import logging
from contextlib import contextmanager

import sflow.core.launcher as launcher_mod
from sflow.core.launcher import (
    _console_streams_task_output,
    _emit_task_output_line,
)
from sflow.logging import SFLOW_TASK_STREAM_ATTR


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _capture_root_logger():
    """Temporarily replace the ``sflow.core.launcher`` logger handlers with a
    capture handler to observe exactly what reaches the root (console/sflow.log).
    """
    logger = launcher_mod._logger
    saved = (logger.handlers, logger.level, logger.propagate)
    cap = _CaptureHandler()
    logger.handlers = [cap]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield cap
    finally:
        logger.handlers, logger.level, logger.propagate = saved


def _make_output_logger() -> tuple[logging.Logger, _CaptureHandler]:
    out = logging.getLogger("sflow.task._test_routing")
    out.handlers = []
    cap = _CaptureHandler()
    out.addHandler(cap)
    out.setLevel(logging.INFO)
    out.propagate = False
    return out, cap


def test_task_line_not_echoed_to_root_when_not_streaming():
    out_logger, task_cap = _make_output_logger()

    with _capture_root_logger() as root_cap:
        _emit_task_output_line(
            "TASK_OUTPUT_LINE",
            pfx="[t] ",
            output_logger=out_logger,
            stream_console=False,
        )

    # Per-task content is always written to the per-task logger.
    assert [r.getMessage() for r in task_cap.records] == ["TASK_OUTPUT_LINE"]
    # Non-TTY (batch): nothing echoed to the root logger.
    assert root_cap.records == []


def test_task_line_echoed_to_root_with_marker_when_streaming():
    out_logger, task_cap = _make_output_logger()

    with _capture_root_logger() as root_cap:
        _emit_task_output_line(
            "TASK_OUTPUT_LINE",
            pfx="[t] ",
            output_logger=out_logger,
            stream_console=True,
        )

    # Per-task content still written to the per-task logger.
    assert any("TASK_OUTPUT_LINE" in r.getMessage() for r in task_cap.records)
    # TTY: echoed to the root logger and tagged as task-stream (so file handlers
    # drop it), with the console task prefix applied.
    assert len(root_cap.records) == 1
    rec = root_cap.records[0]
    assert getattr(rec, SFLOW_TASK_STREAM_ATTR, False) is True
    assert rec.getMessage() == "[t] TASK_OUTPUT_LINE"


def test_emit_without_output_logger_is_safe():
    # Should not raise when there is no per-task logger.
    with _capture_root_logger() as root_cap:
        _emit_task_output_line(
            "X", pfx="", output_logger=None, stream_console=True
        )
    assert len(root_cap.records) == 1


def test_console_streams_task_output_follows_tty(monkeypatch):
    class _FakeStdout:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(launcher_mod.sys, "stdout", _FakeStdout(True))
    assert _console_streams_task_output() is True

    monkeypatch.setattr(launcher_mod.sys, "stdout", _FakeStdout(False))
    assert _console_streams_task_output() is False
