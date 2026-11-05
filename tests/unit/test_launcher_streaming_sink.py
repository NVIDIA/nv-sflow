# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SubprocessLauncher's streaming sink: a data channel, not a log channel.

The default path runs commands on a PTY and collapses carriage-return frames so
progress bars don't grow without bound -- correct for human-readable output, and
destructive for machine-readable rows. Collectors whose samples come back over
stdout therefore opt into a plain pipe via ``stdout_sink``.
"""

import asyncio
import logging
import sys

import pytest

from sflow.core.launcher import SubprocessLauncher


@pytest.fixture(autouse=True)
def _allow_real_subprocesses(fake_process):
    """These tests exercise real pipe plumbing, so they need real children.

    The unit conftest enables pytest-subprocess for every test, which rejects
    any unregistered command; a faked process would not prove anything about
    how bytes actually traverse the pipe.
    """
    fake_process.allow_unregistered(True)
    return fake_process


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _logger_with_capture(name: str) -> tuple[logging.Logger, _CaptureHandler]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = _CaptureHandler()
    logger.addHandler(handler)
    return logger, handler


def test_stdout_sink_receives_every_line_and_stderr_still_logs():
    lines: list[str] = []
    logger, handler = _logger_with_capture("test.streaming.basic")
    script = (
        "import sys\n"
        "for i in range(3):\n"
        "    print('cpu\\t%d,1,2' % i, flush=True)\n"
        "print('status text', file=sys.stderr, flush=True)\n"
    )
    try:
        rc = asyncio.run(
            SubprocessLauncher().run_async(
                [sys.executable, "-c", script],
                output_logger=logger,
                stdout_sink=lines.append,
                task_name="collector",
            )
        )
    finally:
        logger.removeHandler(handler)

    assert rc == 0
    assert lines == ["cpu\t0,1,2", "cpu\t1,1,2", "cpu\t2,1,2"]
    # stderr is the collector's status channel and still belongs in the log.
    assert any("status text" in m for m in handler.messages)
    # ...and must never be mistaken for sample data.
    assert not any("status text" in line for line in lines)


def test_carriage_returns_reach_the_sink_intact():
    """The PTY path rewrites/collapses \\r frames. A data channel must not:
    a row containing one is still exactly one row."""
    lines: list[str] = []
    script = "print('cpu\\t1,a\\rb', flush=True)\nprint('cpu\\t2,x', flush=True)\n"
    rc = asyncio.run(
        SubprocessLauncher().run_async(
            [sys.executable, "-c", script], stdout_sink=lines.append
        )
    )
    assert rc == 0
    assert lines == ["cpu\t1,a\rb", "cpu\t2,x"]


def test_a_long_burst_is_not_truncated():
    """Bounded per-wakeup reads must not drop rows on a chatty collector."""
    lines: list[str] = []
    script = (
        "import sys\n"
        "for i in range(5000):\n"
        "    sys.stdout.write('cpu\\t%d\\n' % i)\n"
        "sys.stdout.flush()\n"
    )
    rc = asyncio.run(
        SubprocessLauncher().run_async(
            [sys.executable, "-c", script], stdout_sink=lines.append
        )
    )
    assert rc == 0
    assert len(lines) == 5000
    assert lines[0] == "cpu\t0" and lines[-1] == "cpu\t4999"


def test_cancelling_stops_the_process():
    """The monitor registry cancels a collector's task on the last release; the
    child must not survive as an orphan sampling forever."""
    script = (
        "import time, sys\n"
        "print('cpu\\t0', flush=True)\n"
        "while True:\n"
        "    time.sleep(0.05)\n"
    )
    lines: list[str] = []

    async def _run_then_cancel():
        launcher = SubprocessLauncher()
        task = asyncio.create_task(
            launcher.run_async(
                [sys.executable, "-c", script], stdout_sink=lines.append
            )
        )
        # Wait until the child has actually produced output, so we are
        # cancelling a running process rather than racing its startup.
        for _ in range(200):
            if lines:
                break
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run_then_cancel())
    assert lines == ["cpu\t0"]


def test_a_partial_final_line_is_still_delivered():
    """A collector killed mid-row should not silently lose its last sample."""
    lines: list[str] = []
    script = "import sys\nsys.stdout.write('cpu\\t1,2')\nsys.stdout.flush()\n"
    asyncio.run(
        SubprocessLauncher().run_async(
            [sys.executable, "-c", script], stdout_sink=lines.append
        )
    )
    assert lines == ["cpu\t1,2"]


def test_a_command_object_with_env_streams_too():
    """The exact shape MonitorRegistry._start_collector passes.

    Every other test here hands in a plain list and no env, so the Command
    unwrapping and the env merge -- the two things production actually uses --
    were the only untaken branches of the streaming path.
    """
    from sflow.core.command import Command

    lines: list[str] = []
    command = Command(exec=sys.executable)
    command.add_arg("-c")
    command.add_arg(
        "import os,sys;print('cpu\\t%s' % os.environ['SFLOW_PROBE'], flush=True)"
    )

    rc = asyncio.run(
        SubprocessLauncher().run_async(
            command,
            env={"SFLOW_PROBE": "from-env"},
            task_name="collector",
            stdout_sink=lines.append,
        )
    )

    assert rc == 0
    assert lines == ["cpu\tfrom-env"]


def test_a_cancelled_collector_that_ignores_sigterm_is_killed():
    """The 1s grace then SIGKILL. A `docker run` client does not exit on TERM --
    it is waiting on its container -- so without the escalation every node's
    teardown would block until the process felt like leaving."""
    import time

    lines: list[str] = []
    script = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('cpu\\t1', flush=True)\n"
        "time.sleep(3600)\n"
    )

    async def scenario():
        task = asyncio.ensure_future(
            SubprocessLauncher().run_async(
                [sys.executable, "-c", script],
                task_name="stubborn",
                stdout_sink=lines.append,
            )
        )
        while not lines:
            await asyncio.sleep(0.01)
        task.cancel()
        started = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    # Grace is 1s; the kill must follow promptly rather than waiting out sleep(3600).
    assert elapsed < 10, elapsed
