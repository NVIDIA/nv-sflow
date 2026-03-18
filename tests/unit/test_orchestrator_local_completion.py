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


def test_orchestrator_marks_completed_on_exit_code_zero():
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)

    logger = logging.getLogger("sflow.tests.orchestrator")
    logger.handlers = []
    logger.propagate = False

    op = BashOperator(BashOperatorConfig(name="bash"))
    task = Task(name="t1", logger=logger, operator=op, script=["echo hi"])
    tg.dag.add_node("t1", task)

    orch = Orchestrator(workflow=wf, poll_interval=0)
    orch._subprocess_launcher = _FakeLauncher()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=1.0))

    assert task.status == TaskStatus.COMPLETED
