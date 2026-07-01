# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator support for operator-managed execution.

Operators that return ``manages_own_execution() is True`` are awaited via
``execute(...)`` instead of launching one subprocess from ``build_command``; the
returned exit code drives status exactly like a subprocess. Long-running managed
tasks that reach a terminal state (e.g. READY services) are stopped when the
workflow finishes, rather than lingering until interpreter shutdown.
"""

import asyncio
import logging

import pytest

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.orchestrator import Orchestrator
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


class _NoopLauncher:
    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
        return 0


class _ManagedOp(Operator):
    """Managed operator whose execute() just returns a fixed exit code."""

    def __init__(self, rc: int):
        super().__init__(OperatorConfig(type="managed"))
        self._rc = rc
        self.executed = False

    def build_command(self, *, task_name, script, envs) -> Command:
        return Command(exec="true")

    def manages_own_execution(self) -> bool:
        return True

    async def execute(
        self, *, launcher, output_logger, env, task_name, script, status_note=None
    ) -> int:
        self.executed = True
        return self._rc


def _single_task_orch(op: _ManagedOp) -> tuple[Orchestrator, Task]:
    task = Task(name="t", operator=op, logger=logging.getLogger("sflow.task._managed"))
    tg = TaskGraph()
    tg.dag.add_node("t", task)
    wf = Workflow(name="wf", task_graph=tg)
    return Orchestrator(workflow=wf, poll_interval=0.01, launcher=_NoopLauncher()), task


def test_managed_execution_success_maps_to_completed():
    op = _ManagedOp(0)
    orch, task = _single_task_orch(op)
    asyncio.run(asyncio.wait_for(orch.run(), timeout=5))
    assert op.executed is True
    assert task.status == TaskStatus.COMPLETED


def test_managed_execution_nonzero_maps_to_failed():
    op = _ManagedOp(3)
    orch, task = _single_task_orch(op)
    asyncio.run(asyncio.wait_for(orch.run(), timeout=5))
    assert task.status == TaskStatus.FAILED
    assert task.exit_code == 3


def test_stop_remaining_subprocess_tasks_cancels_and_clears():
    """A long-running managed task (READY service) is cancelled and cleared."""
    cancelled = {"v": False}

    async def _server():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled["v"] = True
            raise

    async def _drive():
        wf = Workflow(name="wf", task_graph=TaskGraph())
        orch = Orchestrator(workflow=wf, poll_interval=0.01, launcher=_NoopLauncher())
        orch._subprocess_tasks["srv"] = asyncio.ensure_future(_server())
        await asyncio.sleep(0.05)  # let it start
        await orch._stop_remaining_subprocess_tasks(reason="workflow finished")
        return orch

    orch = asyncio.run(_drive())
    assert cancelled["v"] is True
    assert orch._subprocess_tasks == {}
