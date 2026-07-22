# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for the orchestrator flushing per-task log writers before
evaluating file-watching probes.

In stream mode (``--tui`` / ``--no-offload-task-logs``) each task writes its
``<task>.log`` through a :class:`CoalescingFileHandler`, which batches flushes
and only flushes on the *next* emit after its interval. A long-running service
that logs its readiness line ("ready to roll") and then goes idle leaves that
final line in the write buffer indefinitely. :class:`LogWatchProbe` reads the
file from disk, so without an external flush it never observes the readiness
line: the service is stuck RUNNING and its dependents are never submitted
(a deadlock, since the service only writes more lines once a client connects).

The orchestrator flushes all task log handlers once per poll iteration to bound
this staleness; these tests lock that behavior in.
"""

import asyncio
import logging
import time
from pathlib import Path

import pytest

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.orchestrator import Orchestrator
from sflow.core.probe import ProbeStatus, ProbeType
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.logging import CoalescingFileHandler
from sflow.plugins.probes import LogWatchProbe


class _FakeOperator(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="fake"))

    def build_command(self, *, task_name: str, script, envs) -> Command:
        return Command(exec="echo").add_arg("fake")


class _HangingLauncher:
    """Blocks until cancelled — simulates a long-running server process."""

    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            return -1
        return 0


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


class _RecordingSummary:
    """Captures submitted/ready task names for assertions."""

    def __init__(self):
        self.submitted: list[str] = []
        self.ready: list[str] = []

    def task_unblocked(self, task, **kwargs):
        pass

    def task_submitted(self, task, **kwargs):
        self.submitted.append(task.name)

    def task_ready(self, task, **kwargs):
        self.ready.append(task.name)

    def task_failed(self, task, **kwargs):
        pass

    def task_cancelled(self, task, **kwargs):
        pass

    def workflow_finished(self, **kwargs):
        pass


def _coalescing_logger(name: str, log_path: Path) -> tuple[logging.Logger, CoalescingFileHandler]:
    handler = CoalescingFileHandler(str(log_path), flush_interval=1000.0)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]
    logger.propagate = False
    return logger, handler


def test_flush_task_log_handlers_surfaces_buffered_tail(tmp_path: Path):
    """``_flush_task_log_handlers`` pushes a coalesced (buffered) line to disk."""
    log_path = tmp_path / "svc.log"
    logger, handler = _coalescing_logger("sflow.task._flushtest_svc", log_path)
    try:
        # Pretend a flush just happened so the next emit is coalesced (buffered,
        # not flushed) — exactly the state a service is in right after it logs
        # its readiness line and goes idle.
        handler._last_flush = time.monotonic()
        logger.info("The server is fired up and ready to roll!")

        # Precondition: the buffered line is not yet visible on disk.
        assert "ready to roll" not in _read(log_path)

        tg = TaskGraph()
        wf = Workflow(name="wf", task_graph=tg)
        task = Task(
            name="svc",
            operator=_FakeOperator(),
            logger=logger,
            status=TaskStatus.RUNNING,
        )
        tg.dag.add_node("svc", task)
        orch = Orchestrator(workflow=wf, poll_interval=0.01, launcher=_HangingLauncher())

        orch._flush_task_log_handlers()

        assert "ready to roll" in _read(log_path)
    finally:
        handler.close()


def test_readiness_log_watch_fires_for_idle_service(tmp_path: Path):
    """A service that logs 'ready' (buffered) then idles still reaches READY,
    and its dependent task is submitted."""
    wf_out = tmp_path / "wf"
    (wf_out / "server").mkdir(parents=True)
    log_path = wf_out / "server" / "server.log"

    logger, handler = _coalescing_logger("sflow.task._idlesvc_server", log_path)

    probe = LogWatchProbe(
        regex_pattern="ready to roll",
        type=ProbeType.READINESS,
        interval=0,
        timeout=30,
    )
    server = Task(
        name="server",
        operator=_FakeOperator(),
        logger=logger,
        probes=[probe],
    )
    server.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf_out)
    bench = Task(
        name="bench",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task._idlesvc_bench"),
    )

    tg = TaskGraph()
    tg.dag.add_node("server", server)
    tg.dag.add_node("bench", bench)
    tg.dag.add_edge("server", "bench")
    wf = Workflow(name="wf", task_graph=tg)

    class _LogReadyThenHang:
        async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
            # Flush prior output, then write the readiness line so it stays
            # buffered (coalesced) — mimicking a server that prints "ready" and
            # then blocks waiting for its first request.
            handler.flush()
            handler._last_flush = time.monotonic()
            logger.info("The server is fired up and ready to roll!")
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                return -1
            return 0

    summary = _RecordingSummary()
    orch = Orchestrator(
        workflow=wf,
        poll_interval=0.01,
        launcher=_LogReadyThenHang(),
        fail_fast=True,
        execution_summary=summary,
    )

    # Once `server` is READY, `bench` is submitted and hangs, so the run never
    # finishes — time out and inspect state. Without the orchestrator flush the
    # probe would never observe "ready", `server` would stay RUNNING, and `bench`
    # would never be submitted.
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        asyncio.run(asyncio.wait_for(orch.run(), timeout=3))

    assert server.status == TaskStatus.READY
    assert probe.status == ProbeStatus.TRIGGERED
    # The dependent was submitted only because `server` reached READY.
    assert "server" in summary.ready
    assert "bench" in summary.submitted

    handler.close()
