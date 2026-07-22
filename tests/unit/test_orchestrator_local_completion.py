# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import sflow.core.orchestrator as orchestrator_mod
from sflow.core.orchestrator import Orchestrator
from sflow.core.task import ResultConfigRuntime, Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig


class _FakeLauncher:
    async def run_async(
        self, command, shell: bool = False, output_logger=None, env=None, **kwargs
    ) -> int:
        await asyncio.sleep(0)
        return 0


class _FailingLauncher:
    async def run_async(
        self, command, shell: bool = False, output_logger=None, env=None, **kwargs
    ) -> int:
        await asyncio.sleep(0)
        raise OSError("out of pty devices")


class _RecordingSummary:
    def __init__(self):
        self.events: list[tuple[str, str, int | None, int | None, str | None]] = []

    def task_unblocked(self, task, **kwargs):
        self.events.append(("UNBLOCKED", task.name, task.attempts, None, None))

    def task_submitted(self, task, **kwargs):
        self.events.append(("SUBMITTED", task.name, task.attempts, None, None))

    def task_completed(self, task, **kwargs):
        self.events.append(("COMPLETED", task.name, task.attempts, task.exit_code, None))

    def task_failed(self, task, **kwargs):
        self.events.append(
            (
                "FAILED",
                task.name,
                task.attempts,
                task.exit_code,
                kwargs.get("reason"),
            )
        )

    def workflow_finished(self, **kwargs):
        self.events.append(("WORKFLOW_FINISHED", "", None, None, kwargs.get("status")))


def test_orchestrator_marks_completed_on_exit_code_zero():
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    logger = logging.getLogger("sflow.tests.orchestrator")
    logger.handlers = []
    logger.propagate = False

    op = BashOperator(BashOperatorConfig(name="bash"))
    task = Task(name="t1", logger=logger, operator=op, script=["echo hi"])
    tg.dag.add_node("t1", task)

    summary = _RecordingSummary()
    orch = Orchestrator(workflow=wf, poll_interval=0, execution_summary=summary)
    orch._subprocess_launcher = _FakeLauncher()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=1.0))

    assert task.status == TaskStatus.COMPLETED
    assert summary.events[:3] == [
        ("UNBLOCKED", "t1", 0, None, None),
        ("SUBMITTED", "t1", 1, None, None),
        ("COMPLETED", "t1", 1, 0, None),
    ]
    assert summary.events[-1] == ("WORKFLOW_FINISHED", "", None, None, "COMPLETED")


def test_orchestrator_marks_task_failed_when_launcher_raises():
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    logger = logging.getLogger("sflow.tests.orchestrator.launch_error")
    logger.handlers = []
    logger.propagate = False

    op = BashOperator(BashOperatorConfig(name="bash"))
    task = Task(name="t1", logger=logger, operator=op, script=["echo hi"])
    tg.dag.add_node("t1", task)

    summary = _RecordingSummary()
    orch = Orchestrator(workflow=wf, poll_interval=0, execution_summary=summary)
    orch._subprocess_launcher = _FailingLauncher()

    try:
        asyncio.run(asyncio.wait_for(orch.run(), timeout=1.0))
        raise AssertionError("Expected launcher error to propagate")
    except OSError as exc:
        assert "out of pty devices" in str(exc)

    assert task.status == TaskStatus.FAILED
    assert summary.events[-2] == (
        "FAILED",
        "t1",
        1,
        None,
        "launcher error: out of pty devices",
    )
    assert summary.events[-1] == ("WORKFLOW_FINISHED", "", None, None, "FAILED")


def test_orchestrator_keeps_dependency_blocked_while_collecting_result(monkeypatch):
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    logger = logging.getLogger("sflow.tests.orchestrator.finalizing")
    logger.handlers = []
    logger.propagate = False

    op = BashOperator(BashOperatorConfig(name="bash"))
    up = Task(name="up", logger=logger, operator=op, script=["echo up"])
    up.result_config = ResultConfigRuntime()
    down = Task(name="down", logger=logger, operator=op, script=["echo down"])
    tg.dag.add_node("up", up)
    tg.dag.add_node("down", down)
    tg.dag.add_edge("up", "down")

    events: list[tuple[str, object]] = []

    class _RecordingLauncher:
        async def run_async(
            self, command, shell: bool = False, output_logger=None, env=None, **kwargs
        ) -> int:
            name = kwargs.get("task_name")
            events.append(("submit", name))
            await asyncio.sleep(0)
            return 0

    async def _collect(task: Task) -> dict:
        events.append(
            (
                "collect-start",
                task.status,
                down.status,
                [name for kind, name in events if kind == "submit"],
            )
        )
        await asyncio.sleep(0)
        events.append(("collect-end", task.name))
        return {"ok": True}

    monkeypatch.setattr(orchestrator_mod, "collect_task_result", _collect)

    orch = Orchestrator(workflow=wf, poll_interval=0, launcher=_RecordingLauncher())

    asyncio.run(asyncio.wait_for(orch.run(), timeout=1.0))

    assert (
        "collect-start",
        TaskStatus.FINALIZING,
        TaskStatus.INITIATED,
        ["up"],
    ) in events
    assert events.index(("collect-end", "up")) < events.index(("submit", "down"))
    assert up.status == TaskStatus.COMPLETED
    assert down.status == TaskStatus.COMPLETED
