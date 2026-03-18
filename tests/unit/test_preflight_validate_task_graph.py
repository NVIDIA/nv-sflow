# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

import sflow.app.assembly as assembly_mod
from sflow.config.schema import (
    GpuResourceConfig,
    ReplicaConfig,
    ResourcesConfig,
    SflowConfig,
    TaskConfig,
    WorkflowConfig,
)
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


def _slurm_gpu_config(*, gpus: int, replicas: int = 2) -> SflowConfig:
    return SflowConfig(
        version="0.1",
        backends=[
            {
                "name": "slurm_cluster",
                "type": "slurm",
                "default": True,
                "account": "acct",
                "partition": "batch",
                "time": "00:10:00",
                "nodes": 1,
                "gpus_per_node": 4,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="prefill_server",
                    script=["echo hi"],
                    replicas=ReplicaConfig(count=replicas, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=gpus)),
                )
            ],
        ),
    )


def test_preflight_validate_task_graph_restores_backend_allocations():
    config = _slurm_gpu_config(gpus=2)
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state = assembly_mod.resolve_global_variables(config, state)
    state = assembly_mod.resolve_backends(config, state)

    assert state.backends["slurm_cluster"].allocation is None

    assembly_mod.preflight_validate_task_graph(config, state)

    assert state.backends["slurm_cluster"].allocation is None


def test_build_state_raises_gpu_capacity_error_before_allocate_backends(monkeypatch):
    called = False

    def _noop_preflight_backends(_state):
        return None

    async def _fake_allocate_backends(state):
        nonlocal called
        called = True
        return state

    monkeypatch.setattr(assembly_mod, "preflight_validate_backends", _noop_preflight_backends)
    monkeypatch.setattr(assembly_mod, "allocate_backends", _fake_allocate_backends)

    config = _slurm_gpu_config(gpus=3)

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(assembly_mod.build_state(config, allocate=True))

    msg = str(exc_info.value)
    assert "remain available" in msg or "has only 4 GPUs" in msg
    assert "backend_gpu_state=" in msg
    assert "nodes=1" in msg
    assert "gpus_per_node=4" in msg
    assert "total_capacity=4" in msg
    assert "already_allocated=3" in msg
    assert "remaining=1" in msg

    assert called is False
