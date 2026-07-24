# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A readiness-probed service that exits before READY is FAILED, not COMPLETED.

A task with readiness probe(s) is a service: for dependents it is "satisfied" by
reaching READY (up and serving), not merely by its process exiting. If the process
exits while the task never became READY -- even with exit 0 (e.g. a startup failure
masked into a 0 by a ``mpirun | tee`` pipeline) -- the service failed to start.
Marking it COMPLETED would wrongly satisfy dependents (they would launch against a
dead server), so the orchestrator marks it FAILED instead.
"""

import asyncio
import logging

from sflow.core.orchestrator import Orchestrator
from sflow.core.probe import ProbeStatus, ProbeType
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig


class _ZeroExitLauncher:
    """A process that exits 0 immediately (a server whose failure was masked)."""

    async def run_async(
        self, command, shell: bool = False, output_logger=None, env=None, **kwargs
    ) -> int:
        await asyncio.sleep(0)
        return 0


class _NeverReadyProbe:
    """Minimal readiness probe that never triggers (service never becomes READY)."""

    kind = "fake"
    type = ProbeType.READINESS

    def __init__(self):
        self.status = ProbeStatus.INITIATED
        self.last_attempt = None

    def reset(self):
        self.status = ProbeStatus.INITIATED
        self.last_attempt = None

    def force_due(self):
        pass

    async def probe(self, task):
        return False


class _ReadyOnForcedScanProbe:
    """Readiness probe that the live (interval-gated) loop never observes, but the
    forced final scan after process-exit does -- i.e. the service logged 'ready' and
    exited in the scan gap. ``probe()`` only returns True after ``force_due()``."""

    kind = "fake"
    type = ProbeType.READINESS

    def __init__(self):
        self.status = ProbeStatus.INITIATED
        self.last_attempt = None
        self._forced = False

    def reset(self):
        self.status = ProbeStatus.INITIATED
        self.last_attempt = None
        self._forced = False

    def force_due(self):
        self._forced = True

    async def probe(self, task):
        return self._forced


def _task(name, *, probes=None, status=TaskStatus.INITIATED):
    logger = logging.getLogger(f"sflow.tests.sebr.{name}")
    logger.handlers = []
    logger.propagate = False
    return Task(
        name=name,
        logger=logger,
        operator=BashOperator(BashOperatorConfig(name="bash")),
        script=["echo hi"],
        probes=list(probes or []),
        status=status,
    )


def _orch(wf=None):
    return Orchestrator(workflow=wf or Workflow(name="wf", task_graph=TaskGraph()))


# --- predicate: _exited_before_ready -----------------------------------------


def test_predicate_true_for_running_service_with_readiness_probe():
    t = _task("svc", probes=[_NeverReadyProbe()], status=TaskStatus.RUNNING)
    assert _orch()._exited_before_ready(t) is True


def test_predicate_false_for_batch_task_without_probes():
    t = _task("batch", status=TaskStatus.RUNNING)
    assert _orch()._exited_before_ready(t) is False


def test_predicate_false_when_service_already_reached_ready():
    # A service that DID reach READY then exits is fine (dependents already unblocked).
    t = _task("svc", probes=[_NeverReadyProbe()], status=TaskStatus.READY)
    assert _orch()._exited_before_ready(t) is False


# --- end-to-end through the orchestrator loop --------------------------------


def test_service_exiting_zero_before_ready_is_failed_not_completed():
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    task = _task("server", probes=[_NeverReadyProbe()])
    tg.dag.add_node("server", task)

    orch = Orchestrator(workflow=wf, poll_interval=0, fail_fast=True)
    orch._subprocess_launcher = _ZeroExitLauncher()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=1.0))

    assert task.status == TaskStatus.FAILED
    assert task.exit_code == 0  # the process really did exit 0 (masked failure)


def test_service_ready_then_exit_in_scan_gap_is_completed():
    # Race guard: a service whose readiness the interval-gated live loop never saw,
    # but which IS ready (logged 'ready' then exited 0 in the gap), must be
    # double-confirmed by a forced final scan -> READY -> COMPLETED, not FAILED.
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    task = _task("server", probes=[_ReadyOnForcedScanProbe()])
    tg.dag.add_node("server", task)

    orch = Orchestrator(workflow=wf, poll_interval=0, fail_fast=True)
    orch._subprocess_launcher = _ZeroExitLauncher()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=1.0))

    assert task.status == TaskStatus.COMPLETED


def test_batch_task_exiting_zero_still_completes():
    # Guard: a task WITHOUT readiness probes is unaffected (exit 0 -> COMPLETED).
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    task = _task("batch")
    tg.dag.add_node("batch", task)

    orch = Orchestrator(workflow=wf, poll_interval=0, fail_fast=True)
    orch._subprocess_launcher = _ZeroExitLauncher()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=1.0))

    assert task.status == TaskStatus.COMPLETED
