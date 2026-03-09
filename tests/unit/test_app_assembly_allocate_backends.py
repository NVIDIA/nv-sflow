# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from sflow.app.assembly import allocate_backends
from sflow.core.backend import Allocation, Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from collections.abc import Sequence


class _FakeBackend(Backend):
    def __init__(self, name: str, *, should_fail: bool = False):
        super().__init__(name=name)
        self.should_fail = should_fail
        self.allocate_calls = 0
        self.release_calls = 0

    async def allocate(self) -> Allocation:
        self.allocate_calls += 1
        if self.should_fail:
            raise RuntimeError(f"boom:{self.name}")
        return Allocation(
            allocation_id=f"job-{self.name}",
            nodes=[ComputeNode(name=f"n-{self.name}", ip_address="127.0.0.1", index=0)],
        )

    async def release(self, allocation: Allocation) -> None:
        self.release_calls += 1

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        # Minimal implementation to satisfy Backend ABC for unit tests.
        return BashOperator(BashOperatorConfig(name=name))


def _state_with_backends(backends: dict[str, Backend]) -> SflowState:
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    state = SflowState(workflow=wf)
    state.backends = backends
    return state


def test_allocate_backends_allocates_all_unallocated_and_sets_default_backend():
    b1 = _FakeBackend("b1")
    b2 = _FakeBackend("b2")
    state = _state_with_backends({"b1": b1, "b2": b2})

    out = asyncio.run(allocate_backends(state))

    assert out is state
    assert state.default_backend is b1  # first insertion order
    assert b1.allocate_calls == 1
    assert b2.allocate_calls == 1
    assert b1.allocation is not None
    assert b2.allocation is not None

    # Second call should be a no-op (idempotent)
    asyncio.run(allocate_backends(state))
    assert b1.allocate_calls == 1
    assert b2.allocate_calls == 1


def test_allocate_backends_failure_releases_allocations_from_batch():
    ok = _FakeBackend("ok")
    bad = _FakeBackend("bad", should_fail=True)
    state = _state_with_backends({"ok": ok, "bad": bad})

    with pytest.raises(RuntimeError, match=r"boom:bad"):
        asyncio.run(allocate_backends(state))

    # The successful one should have been released/cleared by cleanup
    assert ok.release_calls == 1
    assert ok.allocation is None
