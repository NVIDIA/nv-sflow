# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator must reap an operator's external resources (e.g. Docker
containers) around every task launch.

Killing the foreground ``docker run`` client never stops the daemon-managed
container, so a long-running server held at READY until teardown would otherwise
keep running on the host after the workflow finishes. The orchestrator runs the
operator's ``teardown_commands`` before launch (reap stale/retry containers) and
in a ``finally`` (reap orphans left when the launch process was SIGKILLed).
"""

import asyncio
import logging
import types

import pytest

from sflow.core import orchestrator as orch_mod
from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.orchestrator import Orchestrator
from sflow.core.task import Task
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow

_TEARDOWN = ["docker", "rm", "-f", "sflow-server-localhost"]


class _ContainerOperator(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="fake_container"))

    def build_command(self, *, task_name, script, envs) -> Command:
        return Command(exec="true")

    def teardown_commands(self, *, task_name: str) -> list[Command]:
        return [
            Command(exec="docker")
            .add_arg("rm")
            .add_arg("-f")
            .add_arg(f"sflow-{task_name}-localhost")
        ]


class _NoopLauncher:
    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
        return 0


class _HangingLauncher:
    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
        await asyncio.sleep(3600)
        return 0


def _orchestrator(launcher) -> Orchestrator:
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    return Orchestrator(workflow=wf, poll_interval=0.01, launcher=launcher)


def _patch_subprocess_run(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(orch_mod.subprocess, "run", fake_run)
    return calls


def test_teardown_runs_before_and_after_a_successful_launch(monkeypatch):
    calls = _patch_subprocess_run(monkeypatch)
    task = Task(name="server", operator=_ContainerOperator(), logger=logging.getLogger("t"))

    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))

    assert rc == 0
    # Once before launch (stale/retry reap) and once after (orphan reap).
    assert calls == [_TEARDOWN, _TEARDOWN]


def test_teardown_runs_when_the_task_is_cancelled(monkeypatch):
    calls = _patch_subprocess_run(monkeypatch)
    task = Task(name="server", operator=_ContainerOperator(), logger=logging.getLogger("t"))
    orch = _orchestrator(_HangingLauncher())

    async def _run() -> None:
        launch = asyncio.create_task(orch._launch_task_with_timeout(task))
        await asyncio.sleep(0.05)  # let the pre-launch reap + run_async start
        launch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch

    asyncio.run(_run())

    # Pre-launch reap + the finally reap during cancellation unwinding.
    assert calls.count(_TEARDOWN) == 2


def test_non_container_operator_has_no_teardown(monkeypatch):
    calls = _patch_subprocess_run(monkeypatch)

    class _PlainOperator(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

    task = Task(name="plain", operator=_PlainOperator(), logger=logging.getLogger("t"))

    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))

    assert rc == 0
    assert calls == []  # default teardown_commands() is empty -> no docker calls
