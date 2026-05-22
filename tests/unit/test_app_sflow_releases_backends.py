# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
from collections.abc import Sequence
from pathlib import Path

import pytest
import sflow.app.sflow as sflow_app_mod
from sflow.config.schema import SflowConfig, TaskConfig, WorkflowConfig
from sflow.core.backend import Allocation, Backend
from sflow.core.command_log import get_active_command_log_router
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.core.state import SflowState
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.plugins.operators.srun import SrunOperator, SrunOperatorConfig


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
        source_files=None,
    ) -> SflowState:
        return state

    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)

    # Run should finish immediately (empty graph) and still release allocation.
    workflow_out_dir = sflow_app_mod.SflowApp().run(
        file=Path(f), dry_run=False, output_dir=tmp_path / "out"
    )

    assert backend.released is True
    assert backend.allocation is None
    assert workflow_out_dir is not None
    summary = workflow_out_dir / "sflow_summary.log"
    assert summary.is_file()
    summary_text = summary.read_text()
    assert "Sflow Summary" in summary_text
    assert "Runtime" in summary_text
    assert "sflow executable" in summary_text


def test_sflow_app_releases_backend_when_summary_start_fails(tmp_path, monkeypatch):
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
        source_files=None,
    ) -> SflowState:
        return state

    class _FailingSummaryWriter:
        def __init__(self, path: Path):
            self.path = path

        def start(self, **kwargs):
            raise RuntimeError("summary start failed")

    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)
    monkeypatch.setattr(sflow_app_mod, "SflowSummaryWriter", _FailingSummaryWriter)

    try:
        sflow_app_mod.SflowApp().run(
            file=Path(f),
            dry_run=False,
            output_dir=tmp_path / "out",
        )
        raise AssertionError("Expected SflowApp.run() to raise")
    except RuntimeError as e:
        assert "summary start failed" in str(e)

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
        source_files=None,
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


@pytest.mark.parametrize(
    ("sig", "expected_exception", "expected_code"),
    [
        (signal.SIGINT, KeyboardInterrupt, None),
        (signal.SIGTERM, SystemExit, 128 + int(signal.SIGTERM)),
    ],
)
def test_sflow_app_preserves_signal_exit_when_signal_cancels_tasks(
    tmp_path, monkeypatch, sig, expected_exception, expected_code
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
    task = Task(
        name="t1",
        logger=sflow_app_mod.get_logger("sflow.task.t1"),
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    tg.dag.add_node("t1", task)

    async def _fake_build_state(
        config: SflowConfig,
        *,
        allocate: bool = True,
        workspace_dir=None,
        output_dir=None,
        source_files=None,
    ) -> SflowState:
        return state

    callbacks = {}

    class _FakeLoop:
        def add_signal_handler(self, sig, callback, *args):
            callbacks[sig] = (callback, args)

        def remove_signal_handler(self, sig):
            return True

    class _SignalOrchestrator:
        def __init__(self, workflow, **kwargs):
            self.workflow = workflow
            self.stop_reason = None
            self.execution_summary = kwargs.get("execution_summary")

        def request_stop(self, reason=None):
            self.stop_reason = reason

        async def run(self):
            callback, args = callbacks[sig]
            callback(*args)
            task.status = TaskStatus.CANCELLED
            if self.execution_summary is not None:
                self.execution_summary.workflow_finished(
                    status="CANCELLED",
                    detail="Workflow 'wf' cancelled: 1 task(s) cancelled (t1)",
                )

    class _RecordingSummaryWriter:
        instances = []

        def __init__(self, path):
            self.workflow_finished_calls = []
            self.__class__.instances.append(self)

        def start(self, **kwargs):
            pass

        def workflow_finished(self, **kwargs):
            self.workflow_finished_calls.append(kwargs)

    import sflow.core.orchestrator as orchestrator_mod

    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: _FakeLoop())
    monkeypatch.setattr(orchestrator_mod, "Orchestrator", _SignalOrchestrator)
    monkeypatch.setattr(sflow_app_mod, "SflowSummaryWriter", _RecordingSummaryWriter)

    with pytest.raises(expected_exception) as exc_info:
        sflow_app_mod.SflowApp().run(
            file=Path(f),
            dry_run=False,
            output_dir=tmp_path / "out",
        )

    if expected_code is not None:
        assert exc_info.value.code == expected_code
    assert backend.released is True
    assert backend.allocation is None
    summary_writer = _RecordingSummaryWriter.instances[0]
    assert summary_writer.workflow_finished_calls
    assert summary_writer.workflow_finished_calls[-1]["status"] == "CANCELLED"


def test_sflow_app_propagates_sigint_from_tui_interrupt_handler(
    tmp_path, monkeypatch
):
    f = tmp_path / "sflow.yaml"
    f.write_text(
        'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t1\n      script:\n        - sleep 30\n'
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
    task = Task(
        name="t1",
        logger=sflow_app_mod.get_logger("sflow.task.t1"),
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    tg.dag.add_node("t1", task)

    async def _fake_build_state(
        config: SflowConfig,
        *,
        allocate: bool = True,
        workspace_dir=None,
        output_dir=None,
        source_files=None,
    ) -> SflowState:
        return state

    active_ui = None

    class _FakeRichTui:
        def __init__(self, *args, **kwargs):
            nonlocal active_ui
            active_ui = self
            self.workflow = None
            self.interrupt_handler = None

        async def start_async(self):
            pass

        async def stop_async(self):
            pass

        def refresh(self):
            pass

        def set_workflow(self, workflow):
            self.workflow = workflow

        def set_interrupt_handler(self, handler):
            self.interrupt_handler = handler

    class _SignalDuringRunOrchestrator:
        def __init__(self, workflow, **kwargs):
            self.workflow = workflow
            self.stop_reason = None

        def request_stop(self, reason=None):
            self.stop_reason = reason

        async def run(self):
            assert active_ui is not None
            assert active_ui.interrupt_handler is not None
            active_ui.interrupt_handler()
            for workflow_task in self.workflow.get_tasks():
                workflow_task.status = TaskStatus.CANCELLED

    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)
    import sflow.core.orchestrator as orchestrator_mod
    import sflow.ui.rich_tui as rich_tui_mod

    monkeypatch.setattr(orchestrator_mod, "Orchestrator", _SignalDuringRunOrchestrator)
    monkeypatch.setattr(rich_tui_mod, "RichTui", _FakeRichTui)

    with pytest.raises(KeyboardInterrupt):
        sflow_app_mod.SflowApp().run(
            file=Path(f),
            dry_run=False,
            output_dir=tmp_path / "out",
            tui=True,
        )

    assert backend.released is True
    assert backend.allocation is None


def test_sflow_app_resets_command_log_router_when_build_state_fails(
    tmp_path, monkeypatch
):
    f = tmp_path / "sflow.yaml"
    f.write_text(
        'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t1\n      script:\n        - echo hi\n'
    )

    async def _fake_build_state(
        config: SflowConfig,
        *,
        allocate: bool = True,
        workspace_dir=None,
        output_dir=None,
        source_files=None,
    ) -> SflowState:
        raise RuntimeError("build_state failed")

    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)

    try:
        sflow_app_mod.SflowApp().run(
            file=Path(f),
            dry_run=False,
            output_dir=tmp_path / "out",
        )
        raise AssertionError("Expected SflowApp.run() to raise")
    except RuntimeError as e:
        assert "build_state failed" in str(e)

    assert get_active_command_log_router() is None


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
        source_files=None,
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
        source_files=None,
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
