# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

from sflow.core.command import Command
from sflow.core.orchestrator import Orchestrator
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.probe import Probe, ProbeType
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


class _OperatorExitCode(Operator):
    def __init__(self, code: int):
        super().__init__(OperatorConfig(type="fake"))
        self._code = int(code)

    def build_command(self, *, task_name: str, script, envs):  # pragma: no cover
        # Orchestrator passes this into launcher, but our injected launcher ignores it.
        return Command(exec="echo").add_arg("fake")


class _LauncherByTaskName:
    def __init__(self, codes: dict[str, int]):
        self._codes = dict(codes)

    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:  # noqa: ARG002
        await asyncio.sleep(0)
        if output_logger is None:
            return 0
        name = output_logger.name.split(".")[-1]
        return int(self._codes.get(name, 0))


class _ExplodingProbe(Probe):
    async def check(self, task: Task) -> bool:
        raise RuntimeError("probe exploded")


class _RecordingSummary:
    def __init__(self):
        self.completed: list[str] = []
        self.failed: list[tuple[str, int | None]] = []
        self.cancelled: list[tuple[str, str | None]] = []
        self.workflow_statuses: list[str | None] = []
        self.workflow_details: list[str | None] = []

    def task_unblocked(self, task, **kwargs):
        pass

    def task_submitted(self, task, **kwargs):
        pass

    def task_completed(self, task, **kwargs):
        self.completed.append(task.name)

    def task_failed(self, task, *, exit_code=None, **kwargs):
        self.failed.append((task.name, exit_code))

    def task_cancelled(self, task, *, reason=None, **kwargs):
        self.cancelled.append((task.name, reason))

    def workflow_finished(self, *, status=None, **kwargs):
        self.workflow_statuses.append(status)
        self.workflow_details.append(kwargs.get("detail"))


def test_orchestrator_fail_fast_cancels_blocked_tasks_and_returns():
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    # Match the common logger naming used by assembly: "sflow.task.<task_name>"
    logger_up = logging.getLogger("sflow.task.up")
    logger_down = logging.getLogger("sflow.task.down")

    up = Task(name="up", operator=_OperatorExitCode(1), logger=logger_up)
    down = Task(name="down", operator=_OperatorExitCode(0), logger=logger_down)

    tg.dag.add_node("up", up)
    tg.dag.add_node("down", down)
    tg.dag.add_edge("up", "down")

    summary = _RecordingSummary()
    orch = Orchestrator(
        workflow=wf,
        poll_interval=0.01,
        launcher=_LauncherByTaskName({"up": 1}),
        fail_fast=True,
        execution_summary=summary,
    )

    asyncio.run(asyncio.wait_for(orch.run(), timeout=5))

    assert tg.get_task("up").status == TaskStatus.FAILED
    # Without fail-fast, 'down' would remain INITIATED forever and the workflow would hang.
    assert tg.get_task("down").status == TaskStatus.CANCELLED
    assert summary.failed == [("up", 1)]
    assert summary.cancelled == [("down", "fail-fast")]


def test_orchestrator_summary_marks_workflow_failed_on_launcher_exception():
    class _ExplodingLauncher:
        async def run_async(self, command, output_logger=None, env=None, **kwargs):
            raise RuntimeError("launcher exploded")

    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    task = Task(
        name="boom",
        operator=_OperatorExitCode(0),
        logger=logging.getLogger("sflow.task.boom"),
    )
    tg.dag.add_node("boom", task)

    summary = _RecordingSummary()
    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=_ExplodingLauncher(),
        execution_summary=summary,
    )

    try:
        asyncio.run(orch.run())
        raise AssertionError("expected launcher exception")
    except RuntimeError as exc:
        assert "launcher exploded" in str(exc)

    assert task.status == TaskStatus.FAILED
    assert summary.failed == [("boom", None)]
    assert summary.workflow_statuses == ["FAILED"]
    assert summary.workflow_details == [
        "Workflow 'wf' failed: 1 task(s) failed (boom)"
    ]


def test_orchestrator_launcher_exception_cancels_running_siblings():
    class _ExplodingWhilePeerHangs:
        def __init__(self):
            self.cancelled: list[str] = []

        async def run_async(self, command, output_logger=None, env=None, **kwargs):
            name = kwargs.get("task_name") or output_logger.name.split(".")[-1]
            if name == "boom":
                await asyncio.sleep(0)
                raise RuntimeError("launcher exploded")
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise

    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    boom = Task(
        name="boom",
        operator=_OperatorExitCode(0),
        logger=logging.getLogger("sflow.task.boom"),
    )
    peer = Task(
        name="peer",
        operator=_OperatorExitCode(0),
        logger=logging.getLogger("sflow.task.peer"),
    )
    tg.dag.add_node("boom", boom)
    tg.dag.add_node("peer", peer)

    summary = _RecordingSummary()
    launcher = _ExplodingWhilePeerHangs()
    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=launcher,
        execution_summary=summary,
    )

    try:
        asyncio.run(asyncio.wait_for(orch.run(), timeout=1))
        raise AssertionError("expected launcher exception")
    except RuntimeError as exc:
        assert "launcher exploded" in str(exc)

    assert boom.status == TaskStatus.FAILED
    assert peer.status == TaskStatus.CANCELLED
    assert launcher.cancelled == ["peer"]
    assert summary.cancelled == [
        ("peer", "cancelled after task 'boom' failed: launcher error: launcher exploded")
    ]


def test_orchestrator_launcher_exception_preserves_completed_siblings():
    class _ExplodingAfterPeerCompletes:
        def __init__(self):
            self.cancelled: list[str] = []

        async def run_async(self, command, output_logger=None, env=None, **kwargs):
            name = kwargs.get("task_name") or output_logger.name.split(".")[-1]
            if name == "boom":
                await asyncio.sleep(0)
                raise RuntimeError("launcher exploded")
            if name == "done":
                await asyncio.sleep(0)
                return 0
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise

    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    boom = Task(
        name="boom",
        operator=_OperatorExitCode(0),
        logger=logging.getLogger("sflow.task.boom"),
    )
    done = Task(
        name="done",
        operator=_OperatorExitCode(0),
        logger=logging.getLogger("sflow.task.done"),
    )
    peer = Task(
        name="peer",
        operator=_OperatorExitCode(0),
        logger=logging.getLogger("sflow.task.peer"),
    )
    tg.dag.add_node("boom", boom)
    tg.dag.add_node("done", done)
    tg.dag.add_node("peer", peer)

    summary = _RecordingSummary()
    launcher = _ExplodingAfterPeerCompletes()
    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=launcher,
        execution_summary=summary,
    )

    try:
        asyncio.run(asyncio.wait_for(orch.run(), timeout=1))
        raise AssertionError("expected launcher exception")
    except RuntimeError as exc:
        assert "launcher exploded" in str(exc)

    assert boom.status == TaskStatus.FAILED
    assert done.status == TaskStatus.COMPLETED
    assert done.exit_code == 0
    assert peer.status == TaskStatus.CANCELLED
    assert launcher.cancelled == ["peer"]
    assert summary.completed == ["done"]
    assert summary.cancelled == [
        ("peer", "cancelled after task 'boom' failed: launcher error: launcher exploded")
    ]


def test_orchestrator_probe_exception_cancels_running_siblings():
    class _HangingLauncher:
        def __init__(self):
            self.cancelled: list[str] = []

        async def run_async(self, command, output_logger=None, env=None, **kwargs):
            name = kwargs.get("task_name") or output_logger.name.split(".")[-1]
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise

    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    server = Task(
        name="server",
        operator=_OperatorExitCode(0),
        logger=logging.getLogger("sflow.task.server"),
        probes=[
            _ExplodingProbe(
                type=ProbeType.READINESS,
                interval=0,
                each_check_timeout=1,
            )
        ],
    )
    peer = Task(
        name="peer",
        operator=_OperatorExitCode(0),
        logger=logging.getLogger("sflow.task.peer"),
    )
    tg.dag.add_node("server", server)
    tg.dag.add_node("peer", peer)

    summary = _RecordingSummary()
    launcher = _HangingLauncher()
    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=launcher,
        execution_summary=summary,
    )

    try:
        asyncio.run(asyncio.wait_for(orch.run(), timeout=1))
        raise AssertionError("expected probe exception")
    except RuntimeError as exc:
        assert "probe exploded" in str(exc)

    assert server.status == TaskStatus.FAILED
    assert peer.status == TaskStatus.CANCELLED
    assert "peer" not in orch._subprocess_tasks
    assert summary.cancelled == [
        ("peer", "cancelled after task 'server' failed: probe error: probe exploded")
    ]
