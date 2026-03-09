# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

from sflow.core.command import Command
from sflow.core.orchestrator import Orchestrator
from sflow.core.operator import Operator, OperatorConfig
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
        # `output_logger` is task.logger, so we can use its name to determine which task.
        if output_logger is None:
            return 0
        name = output_logger.name.split(".")[-1]
        return int(self._codes.get(name, 0))


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

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        launcher=_LauncherByTaskName({"up": 1}),
        fail_fast=True,
    )

    asyncio.run(asyncio.wait_for(orch.run(), timeout=2))

    assert tg.get_task("up").status == TaskStatus.FAILED
    # Without fail-fast, 'down' would remain INITIATED forever and the workflow would hang.
    assert tg.get_task("down").status == TaskStatus.CANCELLED
