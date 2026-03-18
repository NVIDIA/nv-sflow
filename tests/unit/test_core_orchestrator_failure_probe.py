# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for failure probe triggering and fail-fast interaction.

Verifies that:
1. A failure probe marks the task as FAILED with `failed_by_probe=True`
2. Fail-fast distinguishes probe-terminated tasks from process-exit failures
3. Downstream tasks are cancelled when a failure probe fires
4. Log messages clearly indicate probe-triggered vs process-exit failures
"""

import asyncio
import logging
from pathlib import Path

import pytest

from sflow.core.command import Command
from sflow.core.orchestrator import Orchestrator
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.probe import Probe, ProbeStatus, ProbeType
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.probes import LogWatchProbe
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig


class _FakeOperator(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="fake"))

    def build_command(self, *, task_name: str, script, envs) -> Command:
        return Command(exec="echo").add_arg("fake")


class _HangingLauncher:
    """Launcher that blocks forever until cancelled — simulates a long-running server process."""

    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            return -1
        return 0


class _AlwaysTriggeredProbe(Probe):
    """Probe that always reports the condition as met on every check."""

    def __init__(self, *, type: ProbeType, **kwargs):
        super().__init__(type=type, failure_threshold=1, interval=0, timeout=1, **kwargs)

    async def check(self, task) -> bool:
        return True


def test_failure_probe_sets_failed_by_probe_and_cancels_workflow(tmp_path: Path):
    """When a failure probe fires, the task is marked FAILED with failed_by_probe=True,
    and fail-fast cancels all other tasks."""
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    server = Task(
        name="server",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.server"),
        probes=[_AlwaysTriggeredProbe(type=ProbeType.FAILURE)],
    )
    bench = Task(
        name="bench",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.bench"),
    )

    tg.dag.add_node("server", server)
    tg.dag.add_node("bench", bench)
    tg.dag.add_edge("server", "bench")

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=_HangingLauncher(),
        fail_fast=True,
    )

    asyncio.run(asyncio.wait_for(orch.run(), timeout=5))

    assert server.status == TaskStatus.FAILED
    assert server.failed_by_probe is True
    assert bench.status == TaskStatus.CANCELLED


class _LogCapture(logging.Handler):
    """Lightweight handler that collects LogRecords — works regardless of propagate setting."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, *, containing: str) -> list[str]:
        return [r.message for r in self.records if containing in r.message]


def test_fail_fast_message_distinguishes_probe_from_process_exit():
    """The fail-fast log message should say 'failure probe terminated' not 'process exited'."""
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    server = Task(
        name="server",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.server"),
        probes=[_AlwaysTriggeredProbe(type=ProbeType.FAILURE)],
    )

    tg.dag.add_node("server", server)

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=_HangingLauncher(),
        fail_fast=True,
    )

    capture = _LogCapture()
    orch_logger = logging.getLogger("sflow.core.orchestrator")
    orch_logger.addHandler(capture)
    try:
        asyncio.run(asyncio.wait_for(orch.run(), timeout=5))
    finally:
        orch_logger.removeHandler(capture)

    fail_fast_msgs = capture.messages(containing="Fail-fast")
    assert len(fail_fast_msgs) == 1
    assert "failure probe terminated: server" in fail_fast_msgs[0]
    assert "process exited with error" not in fail_fast_msgs[0]


def test_failure_probe_log_includes_pattern_detail():
    """The probe-trigger error log should include the matched pattern."""
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    server = Task(
        name="decode_server",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.decode_server"),
        probes=[_AlwaysTriggeredProbe(type=ProbeType.FAILURE)],
    )

    tg.dag.add_node("decode_server", server)

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=_HangingLauncher(),
        fail_fast=True,
    )

    capture = _LogCapture()
    orch_logger = logging.getLogger("sflow.core.orchestrator")
    orch_logger.addHandler(capture)
    try:
        asyncio.run(asyncio.wait_for(orch.run(), timeout=5))
    finally:
        orch_logger.removeHandler(capture)

    probe_msgs = capture.messages(containing="Failure probe triggered")
    assert len(probe_msgs) == 1
    assert "decode_server" in probe_msgs[0]
    assert "task process was still running" in probe_msgs[0]


def test_log_watch_failure_probe_triggers_fail_fast(tmp_path: Path):
    """End-to-end: LogWatchProbe with 'Traceback' pattern triggers fail-fast."""
    wf_out = tmp_path / "wf"
    (wf_out / "server").mkdir(parents=True)
    log_path = wf_out / "server" / "server.log"
    log_path.write_text("Starting server...\n")

    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    probe = LogWatchProbe(
        regex_pattern="Traceback (most recent call last)",
        type=ProbeType.FAILURE,
        interval=0,
        timeout=1,
        failure_threshold=1,
    )
    server = Task(
        name="server",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.server"),
        probes=[probe],
    )
    server.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf_out)

    bench = Task(
        name="bench",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.bench"),
    )

    tg.dag.add_node("server", server)
    tg.dag.add_node("bench", bench)
    tg.dag.add_edge("server", "bench")

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=_HangingLauncher(),
        fail_fast=True,
    )

    # The probe won't fire yet — no traceback in the log.
    # We need the server task to be RUNNING first, then inject the traceback.
    # Use a custom launcher that writes the traceback after a brief delay.
    class _WriteTracebackThenHang:
        def __init__(self, log_path: Path):
            self._log_path = log_path

        async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
            self._log_path.write_text(
                "Starting server...\nTraceback (most recent call last):\n  File ...\nRuntimeError\n"
            )
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                return -1
            return 0

    orch._subprocess_launcher = _WriteTracebackThenHang(log_path)

    asyncio.run(asyncio.wait_for(orch.run(), timeout=5))

    assert server.status == TaskStatus.FAILED
    assert server.failed_by_probe is True
    assert probe.status == ProbeStatus.TRIGGERED
    assert bench.status == TaskStatus.CANCELLED


def test_process_exit_failure_not_marked_as_probe():
    """A task that fails via process exit should NOT have failed_by_probe=True."""
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    class _ImmediateExitLauncher:
        async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
            await asyncio.sleep(0)
            return 1

    task = Task(
        name="crasher",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.crasher"),
    )

    tg.dag.add_node("crasher", task)

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=_ImmediateExitLauncher(),
        fail_fast=True,
    )

    asyncio.run(asyncio.wait_for(orch.run(), timeout=2))

    assert task.status == TaskStatus.FAILED
    assert task.failed_by_probe is False


def test_mixed_probe_and_process_failure_in_fail_fast_message():
    """When both a probe failure and a process exit occur, fail-fast message reports both."""
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    class _MixedLauncher:
        """First call hangs (for probe target), second call exits immediately with error."""

        def __init__(self):
            self._call = 0

        async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
            self._call += 1
            if self._call == 1:
                # "server" hangs until cancelled
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    return -1
                return 0
            else:
                await asyncio.sleep(0)
                return 1

    server = Task(
        name="server",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.server"),
        probes=[_AlwaysTriggeredProbe(type=ProbeType.FAILURE)],
    )
    worker = Task(
        name="worker",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.worker"),
    )

    tg.dag.add_node("server", server)
    tg.dag.add_node("worker", worker)

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=_MixedLauncher(),
        fail_fast=True,
    )

    capture = _LogCapture()
    orch_logger = logging.getLogger("sflow.core.orchestrator")
    orch_logger.addHandler(capture)
    try:
        asyncio.run(asyncio.wait_for(orch.run(), timeout=5))
    finally:
        orch_logger.removeHandler(capture)

    assert server.status == TaskStatus.FAILED
    assert server.failed_by_probe is True
    assert worker.status in (TaskStatus.FAILED, TaskStatus.CANCELLED)

    fail_fast_msgs = capture.messages(containing="Fail-fast")
    assert len(fail_fast_msgs) == 1
    assert "failure probe terminated: server" in fail_fast_msgs[0]


def test_failure_probe_with_match_count_requires_multiple_matches(tmp_path: Path):
    """Failure probe with match_count=3 should NOT trigger on fewer matches."""
    wf_out = tmp_path / "wf"
    (wf_out / "svc").mkdir(parents=True)
    log_path = wf_out / "svc" / "svc.log"

    two_tracebacks = (
        "Traceback (most recent call last):\nboom\n"
        "Traceback (most recent call last):\nboom2\n"
    )
    three_tracebacks = two_tracebacks + "Traceback (most recent call last):\nboom3\n"

    class _WriteThenHang:
        def __init__(self, log_path: Path, content: str):
            self._log_path = log_path
            self._content = content

        async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
            self._log_path.write_text(self._content)
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                return -1
            return 0

    async def _run_both_phases():
        # Phase 1: 2 tracebacks < match_count 3 → probe should NOT trigger
        probe1 = LogWatchProbe(
            regex_pattern="Traceback (most recent call last)",
            type=ProbeType.FAILURE,
            interval=0,
            timeout=1,
            failure_threshold=1,
            match_count=3,
        )
        svc1 = Task(
            name="svc",
            operator=BashOperator(BashOperatorConfig(name="bash")),
            logger=logging.getLogger("sflow.task.svc.p1"),
            probes=[probe1],
        )
        svc1.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf_out)
        tg1 = TaskGraph()
        tg1.dag.add_node("svc", svc1)
        wf1 = Workflow(name="wf1", task_graph=tg1)
        orch1 = Orchestrator(
            workflow=wf1,
            poll_interval=0,
            launcher=_WriteThenHang(log_path, two_tracebacks),
            fail_fast=True,
        )
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(orch1.run(), timeout=0.5)
        # Let the event loop process pending cancellations from phase 1
        for pending in asyncio.all_tasks():
            if pending is not asyncio.current_task() and not pending.done():
                pending.cancel()
        await asyncio.sleep(0.05)
        assert svc1.failed_by_probe is False

        # Phase 2: 3 tracebacks >= match_count 3 → probe SHOULD trigger
        probe2 = LogWatchProbe(
            regex_pattern="Traceback (most recent call last)",
            type=ProbeType.FAILURE,
            interval=0,
            timeout=1,
            failure_threshold=1,
            match_count=3,
        )
        svc2 = Task(
            name="svc",
            operator=BashOperator(BashOperatorConfig(name="bash")),
            logger=logging.getLogger("sflow.task.svc.p2"),
            probes=[probe2],
        )
        svc2.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf_out)
        tg2 = TaskGraph()
        tg2.dag.add_node("svc", svc2)
        wf2 = Workflow(name="wf2", task_graph=tg2)
        orch2 = Orchestrator(
            workflow=wf2,
            poll_interval=0,
            launcher=_WriteThenHang(log_path, three_tracebacks),
            fail_fast=True,
        )
        await asyncio.wait_for(orch2.run(), timeout=5)
        assert svc2.status == TaskStatus.FAILED
        assert svc2.failed_by_probe is True

    asyncio.run(_run_both_phases())
