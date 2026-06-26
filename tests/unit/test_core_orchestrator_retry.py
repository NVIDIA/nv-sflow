# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

from sflow.core.command import Command
from sflow.core.orchestrator import Orchestrator
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.task import RetryPolicy, Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


class _FakeOperator(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="fake"))

    def build_command(self, *, task_name: str, script, envs) -> Command:  # type: ignore[override]
        return Command(exec="echo").add_arg("hi")


class _FakeLauncher:
    def __init__(self, exit_codes: list[int]):
        self._exit_codes = list(exit_codes)
        self.calls: int = 0

    async def run_async(
        self, command, shell: bool = False, output_logger=None, env=None, **kwargs
    ) -> int:
        self.calls += 1
        await asyncio.sleep(0)
        if not self._exit_codes:
            return 0
        return self._exit_codes.pop(0)


class _RecordingSummary:
    def __init__(self):
        self.submitted_attempts: list[int] = []
        self.retries: list[tuple[str, int, int, float]] = []
        self.failures: list[tuple[str, int | None]] = []
        self.completed: list[tuple[str, int | None]] = []

    def task_unblocked(self, task, **kwargs):
        pass

    def task_submitted(self, task, **kwargs):
        self.submitted_attempts.append(task.attempts)

    def task_retry(self, task, *, exit_code=None, delay=None, **kwargs):
        self.retries.append((task.name, task.attempts, exit_code, delay))

    def task_failed(self, task, *, exit_code=None, **kwargs):
        self.failures.append((task.name, exit_code))

    def task_completed(self, task, **kwargs):
        self.completed.append((task.name, task.exit_code))

    def workflow_finished(self, **kwargs):
        pass


def _workflow_with_single_task(task: Task) -> Workflow:
    tg = TaskGraph()
    tg.dag.add_node(task.name, task)
    return Workflow(name="wf", task_graph=tg)


def test_orchestrator_retries_until_success():
    task = Task(name="t1", operator=_FakeOperator(), logger=logging.getLogger("t1"))
    task.retries = RetryPolicy(count=3, interval=0, backoff=2)

    launcher = _FakeLauncher([1, 1, 0])
    wf = _workflow_with_single_task(task)
    summary = _RecordingSummary()
    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=launcher,  # type: ignore[arg-type]
        execution_summary=summary,
    )

    asyncio.run(orch.run())

    assert launcher.calls == 3
    assert task.attempts == 3
    assert task.status == TaskStatus.COMPLETED
    assert summary.submitted_attempts == [1, 2, 3]
    assert [(name, attempt, exit_code) for name, attempt, exit_code, _ in summary.retries] == [
        ("t1", 1, 1),
        ("t1", 2, 1),
    ]
    assert summary.completed == [("t1", 0)]


def test_orchestrator_stops_after_max_retries():
    task = Task(name="t1", operator=_FakeOperator(), logger=logging.getLogger("t1"))
    task.retries = RetryPolicy(count=2, interval=0, backoff=2)

    launcher = _FakeLauncher([1, 1, 1])
    wf = _workflow_with_single_task(task)
    summary = _RecordingSummary()
    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=launcher,  # type: ignore[arg-type]
        execution_summary=summary,
    )

    asyncio.run(orch.run())

    # total attempts = 1 + count
    assert launcher.calls == 3
    assert task.attempts == 3
    assert task.status == TaskStatus.FAILED
    assert summary.failures == [("t1", 1)]
