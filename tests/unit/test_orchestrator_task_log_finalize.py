# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator <task>.log finalize: release the driver's file handler, then let
the operator swap in the complete log it captured out-of-band.

The kubernetes operator stops the live ``kubectl logs -f`` stream the moment a
pod is terminal (the K8s log backlog lags pod exit) and dumps the COMPLETE
container log to a temp ``<pod>.pod.log``. Once the task is done the orchestrator
flushes + closes + detaches the per-task ``CoalescingFileHandler`` (single
writer) and the operator swaps the temp file in as ``<task>.log`` and deletes it
-- a single complete log, no duplicate, no loss.
"""

import logging
from pathlib import Path

from sflow.core.orchestrator import Orchestrator
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.logging import CoalescingFileHandler
from sflow.plugins.operators.k8s import K8sOperator, K8sOperatorConfig


class _NoopLauncher:
    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
        return 0


def _orch(wf: Workflow) -> Orchestrator:
    return Orchestrator(workflow=wf, poll_interval=0.01, launcher=_NoopLauncher())


def _task_with_log(name: str, log_path: Path) -> tuple[Task, CoalescingFileHandler]:
    handler = CoalescingFileHandler(str(log_path), flush_interval=1000.0)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(f"sflow.task._finalize_{name}")
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]
    logger.propagate = False
    task = Task(
        name=name,
        operator=K8sOperator(K8sOperatorConfig(name="op", image="img:1")),
        logger=logger,
        status=TaskStatus.FAILED,
    )
    return task, handler


def test_finalize_releases_handler_and_swaps_complete_log(tmp_path: Path):
    task_dir = tmp_path / "decode_server_0"
    task_dir.mkdir()
    log_path = task_dir / "decode_server_0.log"
    task, handler = _task_with_log("decode_server_0", log_path)
    # The live stream wrote a truncated tail (buffered by the coalescing handler);
    # the wrapper captured the full container log to the temp <pod>.pod.log.
    task.logger.info("live-truncated-tail")
    (task_dir / "decode-server-0.pod.log").write_text("complete-A\ncomplete-B\n")

    tg = TaskGraph()
    tg.dag.add_node("decode_server_0", task)
    wf = Workflow(name="wf", task_graph=tg)

    _orch(wf)._finalize_task_log(task)

    # <task>.log is now the complete container log; the temp is gone (no dup).
    assert log_path.read_text() == "complete-A\ncomplete-B\n"
    assert not (task_dir / "decode-server-0.pod.log").exists()
    # The driver's CoalescingFileHandler was flushed, closed, and detached.
    assert not any(
        isinstance(h, CoalescingFileHandler) for h in task.logger.handlers
    )


def test_finalize_without_pod_log_keeps_live_log_and_handler(tmp_path: Path):
    # No <pod>.pod.log (the live stream finished on its own): <task>.log is left
    # as-is and the handler is NOT released (no rewrite needed).
    task_dir = tmp_path / "frontend_server"
    task_dir.mkdir()
    log_path = task_dir / "frontend_server.log"
    task, handler = _task_with_log("frontend_server", log_path)
    try:
        task.logger.info("live-complete")

        tg = TaskGraph()
        tg.dag.add_node("frontend_server", task)
        wf = Workflow(name="wf", task_graph=tg)

        _orch(wf)._finalize_task_log(task)

        # Handler still attached (untouched) and still the single writer.
        assert any(
            isinstance(h, CoalescingFileHandler) for h in task.logger.handlers
        )
        handler.flush()
        assert "live-complete" in log_path.read_text()
    finally:
        handler.close()
