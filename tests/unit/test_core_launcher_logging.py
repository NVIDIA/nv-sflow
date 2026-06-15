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
from sflow.core.task_logging import TaskLogPolicy, TaskOutputSink, create_task_log_handler
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


def test_echo_to_console_none_autodetects_from_tty(monkeypatch):
    """``echo_to_console=None`` (the SflowApp.run default) must defer to stdout's
    TTY state -- never silently coerce to False -- so redirected/piped ``sflow run``
    stays quiet while an interactive terminal still echoes live output."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert SubprocessLauncher(echo_to_console=None)._echo_to_console is False

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert SubprocessLauncher(echo_to_console=None)._echo_to_console is True
    # No explicit argument behaves the same as an explicit None.
    assert SubprocessLauncher()._echo_to_console is True


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


def test_tty_echo_survives_bounded_task_log_suppression(
    shared_logger_capture, tmp_path
):
    """Interactive/TUI stream is independent from bounded per-task persistence."""
    shared_logger, shared_handler = shared_logger_capture
    shared_logger.setLevel(logging.INFO)

    task_logger = logging.getLogger("sflow.task.bounded_tty_unit")
    task_logger.handlers = []
    task_logger.setLevel(logging.INFO)
    task_logger.propagate = False
    handler = create_task_log_handler(
        tmp_path / "bounded_tty_unit.log",
        TaskLogPolicy(
            mode="bounded",
            keep_lines_per_second=0,
            keep_first_lines=1,
            max_bytes=1024 * 1024,
            backup_count=1,
        ),
    )
    task_logger.addHandler(handler)

    launcher = SubprocessLauncher(echo_to_console=True)
    launcher._emit_subprocess_line(
        "kept on disk", prefix="[bounded_tty_unit] ", output_logger=task_logger
    )
    launcher._emit_subprocess_line(
        "dropped on disk but live", prefix="[bounded_tty_unit] ", output_logger=task_logger
    )
    handler.close()

    disk_contents = (tmp_path / "bounded_tty_unit.log").read_text()
    assert "kept on disk" in disk_contents
    assert "dropped on disk but live" not in disk_contents
    assert any("kept on disk" in m for m in shared_handler.messages)
    assert any("dropped on disk but live" in m for m in shared_handler.messages)
    assert all(
        getattr(r, IN_TASK_OUTPUT_ATTR, False)
        for r in shared_handler.records
        if "disk" in r.getMessage()
    )


def test_non_tty_does_not_echo_when_bounded_task_log_suppresses(
    shared_logger_capture, tmp_path
):
    shared_logger, shared_handler = shared_logger_capture
    shared_logger.setLevel(logging.INFO)

    task_logger = logging.getLogger("sflow.task.bounded_notty_unit")
    task_logger.handlers = []
    task_logger.setLevel(logging.INFO)
    task_logger.propagate = False
    handler = create_task_log_handler(
        tmp_path / "bounded_notty_unit.log",
        TaskLogPolicy(
            mode="bounded",
            keep_lines_per_second=0,
            keep_first_lines=1,
            max_bytes=1024 * 1024,
            backup_count=1,
        ),
    )
    task_logger.addHandler(handler)

    launcher = SubprocessLauncher(echo_to_console=False)
    launcher._emit_subprocess_line(
        "kept only in task log", prefix="[bounded_notty_unit] ", output_logger=task_logger
    )
    launcher._emit_subprocess_line(
        "dropped everywhere shared", prefix="[bounded_notty_unit] ", output_logger=task_logger
    )
    handler.close()

    disk_contents = (tmp_path / "bounded_notty_unit.log").read_text()
    assert "kept only in task log" in disk_contents
    assert "dropped everywhere shared" not in disk_contents
    assert not any("kept only in task log" in m for m in shared_handler.messages)
    assert not any("dropped everywhere shared" in m for m in shared_handler.messages)


def test_on_output_line_callback_runs_before_task_log_suppression(tmp_path):
    task_logger = logging.getLogger("sflow.task.callback_unit")
    task_logger.handlers = []
    task_logger.setLevel(logging.INFO)
    task_logger.propagate = False
    handler = create_task_log_handler(
        tmp_path / "callback_unit.log",
        TaskLogPolicy(
            mode="bounded",
            keep_lines_per_second=0,
            keep_first_lines=0,
            max_bytes=1024 * 1024,
            backup_count=1,
        ),
    )
    task_logger.addHandler(handler)
    seen: list[tuple[str | None, str]] = []

    SubprocessLauncher(echo_to_console=False)._emit_subprocess_line(
        "callback-only line",
        prefix="[callback_unit] ",
        output_logger=task_logger,
        on_output_line=lambda task_name, line: seen.append((task_name, line)),
        task_name="callback_unit",
    )
    handler.close()

    assert seen == [("callback_unit", "callback-only line")]
    assert "callback-only line" not in (tmp_path / "callback_unit.log").read_text()


def test_task_output_sink_replaces_output_logger_info_for_task_persistence(tmp_path):
    task_logger = logging.getLogger("sflow.task.sink_unit")
    task_logger.handlers = []
    task_logger.setLevel(logging.INFO)
    task_logger.propagate = False
    handler = create_task_log_handler(
        tmp_path / "sink_unit.log",
        TaskLogPolicy(
            mode="bounded",
            keep_lines_per_second=0,
            keep_first_lines=1,
            max_bytes=1024 * 1024,
            backup_count=1,
        ),
    )
    task_logger.addHandler(handler)
    sink = TaskOutputSink(
        logger=task_logger,
        policy=TaskLogPolicy(
            mode="bounded",
            keep_lines_per_second=0,
            keep_first_lines=1,
            max_bytes=1024 * 1024,
            backup_count=1,
        ),
    )
    original_info = task_logger.info
    info_calls = 0

    def counting_info(message: str) -> None:
        nonlocal info_calls
        info_calls += 1
        original_info(message)

    task_logger.info = counting_info  # type: ignore[method-assign]

    launcher = SubprocessLauncher(echo_to_console=False)
    launcher._emit_subprocess_line(
        "kept",
        prefix="[sink_unit] ",
        output_logger=task_logger,
        task_output_sink=sink,
        task_name="sink_unit",
    )
    launcher._emit_subprocess_line(
        "dropped",
        prefix="[sink_unit] ",
        output_logger=task_logger,
        task_output_sink=sink,
        task_name="sink_unit",
    )
    sink.close()

    contents = (tmp_path / "sink_unit.log").read_text()
    assert info_calls == 0
    assert "kept" in contents
    assert "dropped" not in contents
