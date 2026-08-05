# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run vs real-run parity for node/GPU placement with pinned GPU indices.

``--dry-run`` plans against ``Backend.placeholder_allocation()``; a real run plans
against the allocation ``Backend.allocate()`` returns. Placement must be identical
across the two, otherwise the map a user reads before submitting is a lie.

The invariant that makes this hold: placement is a pure function of
(node count, per-node ``num_gpus``, node order, task order) -- never of node
*names*, which are the one thing that legitimately differs between a placeholder
and a real Slurm allocation.
"""

import asyncio
from collections.abc import Sequence

from sflow.app.assembly import build_state
from sflow.app.resource_planner import plan_resource_placements
from sflow.config.resolver import ExpressionResolver
from sflow.config.schema import (
    GpuResourceConfig,
    ReplicaConfig,
    ResourcesConfig,
    SflowConfig,
    TaskConfig,
    WorkflowConfig,
)
from sflow.core.backend import Allocation, Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow

HOLD = "workflow_completion"


def _tasks() -> list[TaskConfig]:
    """A workload covering all three placement rules at once."""
    return [
        TaskConfig(
            name="pinned_low",
            script=["echo a"],
            resources=ResourcesConfig(
                gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD)
            ),
        ),
        TaskConfig(
            name="pinned_low_again",
            script=["echo b"],
            depends_on=["pinned_low"],
            resources=ResourcesConfig(
                gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD)
            ),
        ),
        TaskConfig(
            name="backfill",
            script=["echo c"],
            depends_on=["pinned_low_again"],
            resources=ResourcesConfig(
                gpus=GpuResourceConfig(indices=[2, 3], release_after=HOLD)
            ),
        ),
        TaskConfig(
            name="fanout",
            script=["echo d"],
            depends_on=["backfill"],
            resources=ResourcesConfig(
                gpus=GpuResourceConfig(count=4, indices=[0, 1], release_after=HOLD)
            ),
        ),
        TaskConfig(
            name="flexible",
            script=["echo e"],
            depends_on=["fanout"],
            resources=ResourcesConfig(
                gpus=GpuResourceConfig(count=2, release_after=HOLD)
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Path A: the real build_state() entry point, both allocate=False and True
# ---------------------------------------------------------------------------


def _local_config() -> SflowConfig:
    return SflowConfig(
        version="0.1",
        backends=[
            {
                "name": "box",
                "type": "local",
                "default": True,
                "nodes": 4,
                "gpus_per_node": 4,
            }
        ],
        workflow=WorkflowConfig(name="parity", tasks=_tasks()),
    )


def _placement_fingerprint(state: SflowState) -> list[tuple[str, tuple[int, ...], str]]:
    """(task, assigned node *positions*, CUDA_VISIBLE_DEVICES) for every task.

    Node positions rather than names: a real Slurm allocation returns cluster
    hostnames where the placeholder returns ``<backend>-node<i>``, and that
    difference is expected. What must not change is *which* node in the
    allocation order a task lands on, and which GPU indices it gets.
    """
    backend = state.default_backend
    order = {node.name: i for i, node in enumerate(backend.allocation.nodes)}
    out = []
    for task in state.workflow.task_graph.get_tasks():
        assigned = tuple(
            order[n] for n in (getattr(task, "assigned_nodes", None) or []) if n in order
        )
        out.append((task.name, assigned, task.envs.get("CUDA_VISIBLE_DEVICES", "")))
    return sorted(out)


def test_local_backend_dry_run_and_real_run_place_gpus_identically():
    # The local backend's allocate() returns placeholder_allocation(), so this
    # drives both real code paths (allocate=False and allocate=True) end to end
    # through build_state -- including the real run's extra
    # preflight_validate_task_graph pass over placeholder allocations.
    dry = asyncio.run(build_state(_local_config(), allocate=False))
    real = asyncio.run(build_state(_local_config(), allocate=True))

    assert _placement_fingerprint(dry) == _placement_fingerprint(real)
    # Guard against the comparison passing vacuously.
    assert any(cvd for _n, _pos, cvd in _placement_fingerprint(dry))


# ---------------------------------------------------------------------------
# Path B: placeholder vs real-hostname allocation (the Slurm shape)
# ---------------------------------------------------------------------------


class _FakeBackend(Backend):
    def __init__(self, name: str, allocation: Allocation):
        super().__init__(name=name)
        self.allocation = allocation

    async def allocate(self) -> Allocation:
        raise RuntimeError("not used in this unit test")

    async def release(self, allocation: Allocation) -> None:
        raise RuntimeError("not used in this unit test")

    def default_operator(
        self, *, name: str, assigned_nodes: Sequence[str] | None = None
    ) -> Operator:
        raise RuntimeError("not used in this unit test")


def _plan_with_nodes(node_names: list[str], *, gpus_per_node: int):
    backend = _FakeBackend(
        "slurm_cluster",
        Allocation(
            allocation_id="a",
            nodes=[
                ComputeNode(
                    name=name,
                    ip_address=f"10.0.0.{i}",
                    index=i,
                    num_gpus=gpus_per_node,
                )
                for i, name in enumerate(node_names)
            ],
        ),
    )
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {"slurm_cluster": backend}
    state.default_backend = backend
    tasks = _tasks()
    placements = plan_resource_placements(
        SflowConfig(
            version="0.1", workflow=WorkflowConfig(name="wf", tasks=tasks)
        ),
        state,
        resolver=ExpressionResolver(),
        ctx={"variables": {}, "workflow": {"name": "wf"}},
        replica_names_by_base={t.name: [t.name] for t in tasks},
        replica_policy_by_base={t.name: "parallel" for t in tasks},
    )
    order = {name: i for i, name in enumerate(node_names)}
    return {
        name: (tuple(order[n] for n in p.assigned_nodes), p.cuda_visible_devices)
        for name, p in placements.items()
    }


def test_slurm_placeholder_and_real_hostnames_place_gpus_identically():
    # Slurm is the one backend whose real allocate() does not just return
    # placeholder_allocation(): it parses cluster hostnames. num_gpus still comes
    # from the same backend.gpus_per_node config value in both paths, so only the
    # names differ -- placement must not.
    placeholder = _plan_with_nodes(
        [f"slurm_cluster-node{i}" for i in range(4)], gpus_per_node=4
    )
    real = _plan_with_nodes(
        ["dgx-042", "dgx-117", "dgx-003", "dgx-198"], gpus_per_node=4
    )

    assert placeholder == real
    # The workload actually exercises pinning, backfill and fanout.
    assert placeholder["pinned_low"] == ((0,), "0,1")
    assert placeholder["pinned_low_again"] == ((1,), "0,1")
    assert placeholder["backfill"] == ((0,), "2,3")
    assert placeholder["fanout"] == ((2, 3), "0,1")
    assert placeholder["flexible"] == ((1,), "2,3")


def test_replica_placement_is_name_independent():
    tasks = [
        TaskConfig(
            name="worker",
            script=["echo w"],
            replicas=ReplicaConfig(count=3, policy="parallel"),
            resources=ResourcesConfig(
                gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD)
            ),
        )
    ]

    def _plan(node_names: list[str]):
        backend = _FakeBackend(
            "b",
            Allocation(
                allocation_id="a",
                nodes=[
                    ComputeNode(
                        name=name, ip_address="", index=i, num_gpus=4
                    )
                    for i, name in enumerate(node_names)
                ],
            ),
        )
        state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
        state.backends = {"b": backend}
        state.default_backend = backend
        placements = plan_resource_placements(
            SflowConfig(
                version="0.1", workflow=WorkflowConfig(name="wf", tasks=tasks)
            ),
            state,
            resolver=ExpressionResolver(),
            ctx={"variables": {}, "workflow": {"name": "wf"}},
            replica_names_by_base={"worker": ["worker_0", "worker_1", "worker_2"]},
            replica_policy_by_base={"worker": "parallel"},
        )
        order = {name: i for i, name in enumerate(node_names)}
        return {
            name: (tuple(order[n] for n in p.assigned_nodes), p.cuda_visible_devices)
            for name, p in placements.items()
        }

    assert _plan([f"b-node{i}" for i in range(3)]) == _plan(["zz", "aa", "mm"])
