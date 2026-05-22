# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

from sflow.core.orchestrator import Orchestrator
from sflow.core.task import Task, TaskStatus
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
