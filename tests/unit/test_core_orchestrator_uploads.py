# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from pathlib import Path

import sflow.core.orchestrator as orchestrator_mod
from sflow.core.orchestrator import Orchestrator
from sflow.core.storage import StorageTarget
from sflow.core.task import ResolvedUpload, Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig


class _FakeLauncher:
    """Always-succeeds launcher; the script content is irrelevant for these tests."""

    async def run_async(
        self, command, shell: bool = False, output_logger=None, env=None, **kwargs
    ) -> int:
        await asyncio.sleep(0)
        return 0


class _RecordingTarget(StorageTarget):
    def __init__(self, name: str, *, raise_on_upload: bool = False):
        super().__init__(name)
        self.prefix = ""
        self.calls: list[tuple[Path, str]] = []
        self.raise_on_upload = raise_on_upload

    async def upload(self, local_path: Path, remote_key: str) -> None:
        if self.raise_on_upload:
            raise RuntimeError("simulated upload failure")
        self.calls.append((local_path, remote_key))

    def plan(self, local_path: Path, remote_key: str) -> str:
        return f"recording://{self.name}/{remote_key}"


def _make_workflow_with_task(
    tmp_path: Path,
    *,
    upload_on_error: str,
) -> tuple[Workflow, Task]:
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    logger = logging.getLogger(f"sflow.tests.orchestrator.uploads.{upload_on_error}")
    logger.handlers = []
    logger.propagate = False

    out_dir = tmp_path / "t1"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("ok")

    op = BashOperator(BashOperatorConfig(name="bash"))
    task = Task(name="t1", logger=logger, operator=op, script=["echo hi"])
    task.envs["SFLOW_TASK_OUTPUT_DIR"] = str(out_dir)
    task.uploads = [
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr=None,
            on_error=upload_on_error,
        )
    ]
    tg.dag.add_node("t1", task)
    return wf, task


def test_orchestrator_uploads_on_completion(tmp_path: Path):
    wf, task = _make_workflow_with_task(tmp_path, upload_on_error="warn")
    target = _RecordingTarget("bucket")

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        storage_targets={"bucket": target},
    )
    orch._subprocess_launcher = _FakeLauncher()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=2.0))

    assert task.status == TaskStatus.COMPLETED
    assert len(target.calls) == 1
    assert target.calls[0][0].name == "results.csv"


def test_orchestrator_marks_failed_when_on_error_fail(tmp_path: Path):
    wf, task = _make_workflow_with_task(tmp_path, upload_on_error="fail")
    target = _RecordingTarget("bucket", raise_on_upload=True)

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        storage_targets={"bucket": target},
    )
    orch._subprocess_launcher = _FakeLauncher()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=2.0))

    assert task.status == TaskStatus.FAILED


def test_orchestrator_warn_keeps_task_completed_on_upload_failure(tmp_path: Path):
    wf, task = _make_workflow_with_task(tmp_path, upload_on_error="warn")
    target = _RecordingTarget("bucket", raise_on_upload=True)

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        storage_targets={"bucket": target},
    )
    orch._subprocess_launcher = _FakeLauncher()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=2.0))

    # Upload failed but on_error=warn => task stays COMPLETED.
    assert task.status == TaskStatus.COMPLETED


def test_orchestrator_no_uploads_skips_upload_path(tmp_path: Path):
    """Backward-compat: workflows without storage_targets / uploads behave exactly as before."""
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    logger = logging.getLogger("sflow.tests.orchestrator.uploads.none")
    logger.handlers = []
    logger.propagate = False
    op = BashOperator(BashOperatorConfig(name="bash"))
    task = Task(name="t1", logger=logger, operator=op, script=["echo hi"])
    tg.dag.add_node("t1", task)

    orch = Orchestrator(workflow=wf, poll_interval=0)
    orch._subprocess_launcher = _FakeLauncher()
    asyncio.run(asyncio.wait_for(orch.run(), timeout=1.0))
    assert task.status == TaskStatus.COMPLETED


def test_orchestrator_keeps_task_finalizing_during_uploads(tmp_path: Path, monkeypatch):
    wf, task = _make_workflow_with_task(tmp_path, upload_on_error="warn")
    seen_statuses: list[TaskStatus] = []

    async def _run_uploads(task_arg, storage_targets, *, results=None):  # noqa: ARG001
        seen_statuses.append(task_arg.status)
        await asyncio.sleep(0)
        return True

    monkeypatch.setattr(orchestrator_mod, "run_task_uploads", _run_uploads)

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0,
        storage_targets={"bucket": _RecordingTarget("bucket")},
    )
    orch._subprocess_launcher = _FakeLauncher()

    asyncio.run(asyncio.wait_for(orch.run(), timeout=1.0))

    assert seen_statuses == [TaskStatus.FINALIZING]
    assert task.status == TaskStatus.COMPLETED
