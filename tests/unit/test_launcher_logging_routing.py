# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for how SubprocessLauncher routes per-task output to loggers.

Per-task subprocess output must always be written to the per-task logger
(``output_logger``), but only additionally echoed to the root ``sflow`` logger
(console / slurm stdout) for interactive TTY sessions, and only when tagged with
``SFLOW_TASK_STREAM_ATTR`` so sflow.log's file handler can drop it.
"""

import asyncio
import logging
import os
from contextlib import contextmanager

import pytest

import sflow.core.launcher as launcher_mod
from sflow.core.launcher import (
    SubprocessLauncher,
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


def test_emit_to_file_false_streams_console_but_not_per_task_log():
    """Transient progress snapshots stream to the console but not <task>.log."""
    out_logger, task_cap = _make_output_logger()

    with _capture_root_logger() as root_cap:
        _emit_task_output_line(
            "SNAPSHOT",
            pfx="[t] ",
            output_logger=out_logger,
            stream_console=True,
            to_file=False,
        )

    # Not written to the per-task log (file stays final-state-only)...
    assert task_cap.records == []
    # ...but streamed to the console/TUI, tagged + prefixed.
    assert len(root_cap.records) == 1
    assert root_cap.records[0].getMessage() == "[t] SNAPSHOT"


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


# ---------------------------------------------------------------------------
# Integration: SubprocessLauncher.run_async wires subprocess output through
# _emit_task_output_line (per-task log always; root only when streaming).
#
# The repo blocks real subprocesses in unit tests (pytest_subprocess), so we
# fake pty/Popen to "run" a process that emits a single pre-written line --
# the same technique used in tests/unit/test_core_command_log.py.
# ---------------------------------------------------------------------------


class _ExitedProcess:
    returncode = 0

    def poll(self) -> int:
        return 0


def _fake_one_line_subprocess(monkeypatch, line: bytes = b"ROUTEDLINE\n") -> None:
    """Patch the launcher so run_async reads a single line without a real exec.

    The pty "master" is the read end of a pipe pre-loaded with ``line`` and then
    closed (so the launcher sees the line followed by EOF). run_async owns and
    closes both fds it is handed.
    """
    read_fd, write_fd = os.pipe()
    os.write(write_fd, line)
    os.close(write_fd)
    slave_fd = os.open(os.devnull, os.O_WRONLY)

    monkeypatch.setattr(launcher_mod.pty, "openpty", lambda: (read_fd, slave_fd))
    monkeypatch.setattr(
        launcher_mod.subprocess, "Popen", lambda *a, **k: _ExitedProcess()
    )
    monkeypatch.setattr(launcher_mod, "record_active_command", lambda *a, **k: None)


@pytest.mark.skipif(
    launcher_mod.pty is None, reason="PTY-based launching unavailable on this platform"
)
def test_run_async_streams_to_root_with_marker_and_per_task_log_when_tty(monkeypatch):
    # Force the interactive-TTY decision so output is also echoed to the root logger.
    monkeypatch.setattr(launcher_mod, "_console_streams_task_output", lambda: True)
    _fake_one_line_subprocess(monkeypatch)

    out_logger, task_cap = _make_output_logger()
    with _capture_root_logger() as root_cap:
        rc = asyncio.run(
            SubprocessLauncher().run_async(
                ["bash", "-c", "ignored-fake"],
                output_logger=out_logger,
                task_name="t",
            )
        )

    assert rc == 0
    # Per-task log always receives the subprocess output line.
    assert any("ROUTEDLINE" in r.getMessage() for r in task_cap.records)
    # TTY: it is also streamed to the root logger, tagged so file handlers drop it.
    streamed = [r for r in root_cap.records if getattr(r, SFLOW_TASK_STREAM_ATTR, False)]
    assert any(r.getMessage() == "[t] ROUTEDLINE" for r in streamed)


@pytest.mark.skipif(
    launcher_mod.pty is None, reason="PTY-based launching unavailable on this platform"
)
def test_run_async_collapses_carriage_return_redraws_in_per_task_log(monkeypatch):
    """A \\r progress line is recorded in <task>.log as its final state only."""
    monkeypatch.setattr(launcher_mod, "_console_streams_task_output", lambda: False)
    _fake_one_line_subprocess(
        monkeypatch,
        line=b"downloading 10%\rdownloading 50%\rdownloading 100%\ndone\n",
    )

    out_logger, task_cap = _make_output_logger()
    with _capture_root_logger():
        rc = asyncio.run(
            SubprocessLauncher().run_async(
                ["bash", "-c", "ignored-fake"],
                output_logger=out_logger,
                task_name="t",
            )
        )

    assert rc == 0
    msgs = [r.getMessage() for r in task_cap.records]
    # Only the final state of the redrawn line is kept, plus the next line.
    assert msgs == ["downloading 100%", "done"]
    assert not any("10%" in m or "50%" in m for m in msgs)


@pytest.mark.skipif(
    launcher_mod.pty is None, reason="PTY-based launching unavailable on this platform"
)
def test_run_async_streams_in_progress_progress_snapshot_without_newline(monkeypatch):
    """An unterminated \\r progress line is surfaced live to the console, and its
    final state is recorded once in <task>.log (no flood, no duplicate)."""
    monkeypatch.setattr(launcher_mod, "_console_streams_task_output", lambda: True)
    _fake_one_line_subprocess(
        monkeypatch,
        line=b"downloading 10%\rdownloading 99%",  # no trailing newline
    )

    out_logger, task_cap = _make_output_logger()
    with _capture_root_logger() as root_cap:
        rc = asyncio.run(
            SubprocessLauncher().run_async(
                ["bash", "-c", "ignored-fake"],
                output_logger=out_logger,
                task_name="t",
            )
        )

    assert rc == 0
    # Console got the live progress snapshot (final collapsed frame).
    streamed = [
        r.getMessage()
        for r in root_cap.records
        if getattr(r, SFLOW_TASK_STREAM_ATTR, False)
    ]
    assert "[t] downloading 99%" in streamed
    assert not any("10%" in m for m in streamed)
    # File recorded the final state exactly once (no intermediate frames, no dup).
    file_msgs = [r.getMessage() for r in task_cap.records]
    assert file_msgs == ["downloading 99%"]


@pytest.mark.skipif(
    launcher_mod.pty is None, reason="PTY-based launching unavailable on this platform"
)
def test_run_async_keeps_output_off_root_when_not_tty(monkeypatch):
    # Headless/batch: the per-task output must not reach the root logger at all.
    monkeypatch.setattr(launcher_mod, "_console_streams_task_output", lambda: False)
    _fake_one_line_subprocess(monkeypatch)

    out_logger, task_cap = _make_output_logger()
    with _capture_root_logger() as root_cap:
        rc = asyncio.run(
            SubprocessLauncher().run_async(
                ["bash", "-c", "ignored-fake"],
                output_logger=out_logger,
                task_name="t",
            )
        )

    assert rc == 0
    # Per-task log still captures the line ...
    assert any("ROUTEDLINE" in r.getMessage() for r in task_cap.records)
    # ... but nothing per-task is streamed to the root logger.
    streamed = [r for r in root_cap.records if getattr(r, SFLOW_TASK_STREAM_ATTR, False)]
    assert streamed == []
