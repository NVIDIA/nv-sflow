# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

import sflow.app.assembly as assembly_mod
from sflow.config.schema import SflowConfig, TaskConfig, WorkflowConfig
from sflow.core.backend import Allocation, Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from collections.abc import Sequence


class _BackendWithRelease(Backend):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.released = False

    async def allocate(self) -> Allocation:  # pragma: no cover
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


def test_build_state_releases_backends_when_build_task_graph_raises(monkeypatch):
    backend = _BackendWithRelease("b1")
    backend.allocation = Allocation(
        allocation_id="1",
        nodes=[ComputeNode(name="n1", ip_address="10.0.0.1", index=0)],
    )

    async def _fake_allocate_backends(state: SflowState) -> SflowState:
        state.backends = {"b1": backend}
        state.default_backend = backend
        return state

    def _fake_resolve_backends(config: SflowConfig, state: SflowState) -> SflowState:
        # Keep as no-op; allocate_backends will populate backends.
        return state

    def _fake_build_task_graph(
        config: SflowConfig, state: SflowState, *, workspace_dir=None
    ) -> TaskGraph:
        raise ValueError("boom")

    monkeypatch.setattr(assembly_mod, "resolve_backends", _fake_resolve_backends)
    monkeypatch.setattr(
        assembly_mod, "preflight_validate_task_graph", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(assembly_mod, "allocate_backends", _fake_allocate_backends)
    monkeypatch.setattr(assembly_mod, "build_task_graph", _fake_build_task_graph)

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(assembly_mod.build_state(config, allocate=True))

    assert backend.released is True
    assert backend.allocation is None
