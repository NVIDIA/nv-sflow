# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import sflow.app.sflow as sflow_app_mod
from sflow.config.schema import SflowConfig, TaskConfig, WorkflowConfig
from sflow.core.backend import Allocation, Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.core.state import SflowState
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.plugins.operators.srun import SrunOperator, SrunOperatorConfig
from collections.abc import Sequence


class _BackendWithAllocation(Backend):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.released = False

    async def allocate(self) -> Allocation:
        raise RuntimeError("not used")

    async def release(self, allocation: Allocation) -> None:
        self.released = True

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        # Minimal implementation to satisfy Backend ABC for unit tests.
        return BashOperator(BashOperatorConfig(name=name))


def test_sflow_app_releases_backend_allocation_on_success(tmp_path, monkeypatch):
    # Minimal config file (ConfigLoader requires a real file).
    f = tmp_path / "sflow.yaml"
    f.write_text(
        'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t1\n      script:\n        - echo hi\n'
    )

    backend = _BackendWithAllocation("b1")
    backend.allocation = Allocation(
        allocation_id="1",
        nodes=[ComputeNode(name="n1", ip_address="127.0.0.1", index=0)],
    )

    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {"b1": backend}
    state.default_backend = backend

    async def _fake_build_state(
        config: SflowConfig,
        *,
        allocate: bool = True,
        workspace_dir=None,
        output_dir=None,
    ) -> SflowState:
        return state

    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)

    # Run should finish immediately (empty graph) and still release allocation.
    sflow_app_mod.SflowApp().run(file=Path(f), dry_run=False)

    assert backend.released is True
    assert backend.allocation is None


def test_sflow_app_raises_on_failed_tasks_and_still_releases_backends(
    tmp_path, monkeypatch
):
    f = tmp_path / "sflow.yaml"
    f.write_text(
        'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t1\n      script:\n        - echo hi\n'
    )

    backend = _BackendWithAllocation("b1")
    backend.allocation = Allocation(
        allocation_id="1",
        nodes=[ComputeNode(name="n1", ip_address="127.0.0.1", index=0)],
    )

    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    state = SflowState(workflow=wf)
    state.backends = {"b1": backend}
    state.default_backend = backend

    # Seed a FAILED task as terminal so Orchestrator.run() returns immediately,
    # then SflowApp should detect failure and raise.
    t = Task(
        name="t1",
        logger=sflow_app_mod.get_logger("sflow.task.t1"),
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    t.status = TaskStatus.FAILED
    tg.dag.add_node("t1", t)

    async def _fake_build_state(
        config: SflowConfig,
        *,
        allocate: bool = True,
        workspace_dir=None,
        output_dir=None,
    ) -> SflowState:
        return state

    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)

    try:
        sflow_app_mod.SflowApp().run(file=Path(f), dry_run=False)
        raise AssertionError("Expected SflowApp.run() to raise on failed tasks")
    except RuntimeError as e:
        assert "failed" in str(e).lower()

    assert backend.released is True
    assert backend.allocation is None


def test_sflow_app_does_not_release_when_dry_run(tmp_path, monkeypatch):
    f = tmp_path / "sflow.yaml"
    f.write_text(
        'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t1\n      script:\n        - echo hi\n'
    )

    backend = _BackendWithAllocation("b1")
    backend.allocation = Allocation(
        allocation_id="1",
        nodes=[ComputeNode(name="n1", ip_address="127.0.0.1", index=0)],
    )

    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {"b1": backend}
    state.default_backend = backend

    async def _fake_build_state(
        config: SflowConfig,
        *,
        allocate: bool = True,
        workspace_dir=None,
        output_dir=None,
    ) -> SflowState:
        return state

    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)

    out_dir = tmp_path / "out"
    sflow_app_mod.SflowApp().run(file=Path(f), dry_run=True, output_dir=out_dir)

    # dry-run path should not attempt cleanup of (fake) allocation.
    assert backend.released is False
    assert backend.allocation is not None

    # dry-run should not create any output directories/files.
    assert out_dir.exists() is False


def test_sflow_app_mounts_sflow_dirs_for_srun_container_tasks(tmp_path, monkeypatch):
    # Minimal config file (ConfigLoader requires a real file).
    f = tmp_path / "sflow.yaml"
    f.write_text(
        'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t1\n      script:\n        - echo hi\n'
    )

    # Build a state with an srun operator that uses a container image.
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    state = SflowState(workflow=wf)
    t = Task(
        name="t1",
        logger=sflow_app_mod.get_logger("sflow.task.t1"),
        operator=SrunOperator(
            SrunOperatorConfig(name="srun", container_image="docker://alpine:latest")
        ),
    )
    tg.dag.add_node("t1", t)

    async def _fake_build_state(
        config: SflowConfig,
        *,
        allocate: bool = True,
        workspace_dir=None,
        output_dir=None,
    ) -> SflowState:
        return state

    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)

    out_dir = tmp_path / "out"
    sflow_app_mod.SflowApp().run(
        file=Path(f),
        dry_run=True,
        workspace_dir=tmp_path,
        output_dir=out_dir,
    )

    # Mounts should include the computed SFLOW dirs (host path == container path).
    mounts = list(t.operator.config.container_mounts or [])
    wf_out = out_dir / "_dry_run" / "wf"
    task_out = wf_out / "t1"
    assert f"{tmp_path}:{tmp_path}:rw" in mounts
    assert f"{out_dir}:{out_dir}:rw" in mounts
    assert f"{wf_out}:{wf_out}:rw" in mounts
    assert f"{task_out}:{task_out}:rw" in mounts
