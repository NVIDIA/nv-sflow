# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import pytest

from sflow.app.resource_planner import GpuReservationPlanner, plan_resource_placements
from sflow.config.resolver import ExpressionResolver
from sflow.config.schema import (
    GpuResourceConfig,
    NodeResourceConfig,
    ReplicaConfig,
    ResourcesConfig,
    SflowConfig,
    TaskConfig,
    WorkflowConfig,
)
from sflow.core.backend import Allocation, Backend, BackendCapabilities
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


class _FakeBackend(Backend):
    def __init__(self, name: str, allocation: Allocation | None):
        super().__init__(name=name)
        self.allocation = allocation

    async def allocate(self) -> Allocation:
        raise RuntimeError("not used in this unit test")

    async def release(self, allocation: Allocation) -> None:
        raise RuntimeError("not used in this unit test")

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        raise RuntimeError("not used in this unit test")


def test_format_blockers_preserves_gpu_reservation_chain_order():
    planner = GpuReservationPlanner()
    planner.set_ancestors({
        "batch_2_worker_2": {
            "check_entire_env",
            "persistent_worker_0",
        }
    })
    planner.has_readiness = {
        "check_entire_env": False,
        "persistent_worker_0": True,
    }
    planner.gpu_release_explicit = {
        "check_entire_env": False,
        "persistent_worker_0": False,
    }
    planner.gpu_release_after = {
        "check_entire_env": "task_completion",
        "persistent_worker_0": "workflow_completion",
    }
    planner.reserve(
        backend_name="slurm_cluster",
        node_name="slurm_cluster-node0",
        task_name="check_entire_env",
        start=0,
        count=4,
    )
    planner.reserve(
        backend_name="slurm_cluster",
        node_name="slurm_cluster-node0",
        task_name="persistent_worker_0",
        start=0,
        count=1,
    )

    output = planner.format_blockers(
        backend_name="slurm_cluster",
        node_name="slurm_cluster-node0",
        task_name="batch_2_worker_2",
    )

    assert "GPU 0: check_entire_env -> persistent_worker_0" in output


def test_format_blockers_includes_stage_timeline_summary():
    planner = GpuReservationPlanner()
    planner.set_ancestors({
        "batch_2_worker_2": {
            "check_entire_env",
            "worker_release_after_completion_0",
        }
    })
    planner.set_task_stages({
        "check_entire_env": 0,
        "worker_release_after_completion_0": 1,
        "batch_2_worker_0": 2,
        "batch_2_worker_2": 2,
    })
    planner.gpu_release_after = {
        "check_entire_env": "task_completion",
        "worker_release_after_completion_0": "task_completion",
        "batch_2_worker_0": "task_completion",
    }
    planner.reserve(
        backend_name="slurm_cluster",
        node_name="slurm_cluster-node0",
        task_name="check_entire_env",
        start=0,
        count=1,
    )
    planner.reserve(
        backend_name="slurm_cluster",
        node_name="slurm_cluster-node0",
        task_name="worker_release_after_completion_0",
        start=0,
        count=1,
    )
    planner.reserve(
        backend_name="slurm_cluster",
        node_name="slurm_cluster-node0",
        task_name="batch_2_worker_0",
        start=0,
        count=1,
    )

    output = planner.format_blockers(
        backend_name="slurm_cluster",
        node_name="slurm_cluster-node0",
        task_name="batch_2_worker_2",
    )

    assert "Timeline:" in output
    assert "Stage 0: check_entire_env uses GPU 0" in output
    assert "Stage 1: worker_release_after_completion_0 reuses GPU 0" in output
    assert "Stage 2: batch_2_worker_0 reuses GPU 0" in output
    assert "Failed placement: batch_2_worker_2 needs GPU 0, but it is blocked by batch_2_worker_0" in output


def test_plan_resource_placements_reuses_inferred_task_completion_gpus():
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="rel0",
                nodes=[
                    ComputeNode(
                        name="n1",
                        ip_address="10.0.0.1",
                        index=0,
                        num_gpus=8,
                    )
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="check_entire_env",
                    script=["echo check"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=8)),
                ),
                TaskConfig(
                    name="worker",
                    script=["echo worker"],
                    replicas=ReplicaConfig(count=4, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                    depends_on=["check_entire_env"],
                ),
            ],
        ),
    )

    placements = plan_resource_placements(
        config,
        state,
        resolver=ExpressionResolver(),
        ctx={"variables": {}, "workflow": {"name": "wf"}},
        replica_names_by_base={
            "check_entire_env": ["check_entire_env"],
            "worker": ["worker_0", "worker_1", "worker_2", "worker_3"],
        },
        replica_policy_by_base={
            "check_entire_env": "parallel",
            "worker": "parallel",
        },
    )

    assert placements["check_entire_env"].assigned_nodes == ["n1"]
    assert placements["check_entire_env"].cuda_visible_devices == "0,1,2,3,4,5,6,7"
    assert placements["check_entire_env"].resource_release_after == {
        "gpus": "task_completion"
    }
    assert placements["worker_0"].cuda_visible_devices == "0,1"
    assert placements["worker_1"].cuda_visible_devices == "2,3"
    assert placements["worker_2"].cuda_visible_devices == "4,5"
    assert placements["worker_3"].cuda_visible_devices == "6,7"


def test_plan_resource_placements_computes_gpu_slice_uniformly_for_non_gpu_env_backend():
    # supports_gpu_env=False (e.g. Kubernetes): the planner computes the GPU slice
    # uniformly (so node packing / oversubscription checks stay consistent across
    # backends); it is simply not injected into the env (Backend.resource_env
    # returns {}). The count is also carried on the placement.
    backend = _FakeBackend(
        "k8s",
        allocation=Allocation(
            allocation_id="kubernetes",
            nodes=[
                ComputeNode(name="k8s-node0", ip_address="", index=0, num_gpus=8)
            ],
            owned=False,
        ),
    )
    backend.capabilities = BackendCapabilities(
        supports_node_placement=False,
        supports_gpu_env=False,
        supports_host_path_mounts=False,
        has_runtime_node_addresses=False,
    )
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {"k8s": backend}
    state.default_backend = backend
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="serve",
                    script=["python serve.py"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=4)),
                ),
            ],
        ),
    )

    placements = plan_resource_placements(
        config,
        state,
        resolver=ExpressionResolver(),
        ctx={"variables": {}, "workflow": {"name": "wf"}},
        replica_names_by_base={"serve": ["serve"]},
        replica_policy_by_base={"serve": "parallel"},
    )

    placement = placements["serve"]
    assert placement.gpu_count == 4
    # The slice is computed uniformly (not None); it just won't be injected as env.
    assert placement.cuda_visible_devices == "0,1,2,3"
    assert placement.assigned_nodes == []


def test_plan_resource_placements_packs_disjoint_slices_on_one_node():
    # Replicas sharing a single node must get disjoint, in-order device slices --
    # the planner packs numeric indices, which intra-node backends then map onto
    # physical devices (docker reserves the real GPUs at launch).
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {
        "docker": _FakeBackend(
            "docker",
            allocation=Allocation(
                allocation_id="docker",
                nodes=[
                    ComputeNode(
                        name="localhost",
                        ip_address="127.0.0.1",
                        index=0,
                        num_gpus=8,
                    )
                ],
                owned=False,
            ),
        )
    }
    state.default_backend = state.backends["docker"]
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="worker",
                    script=["echo worker"],
                    replicas=ReplicaConfig(count=4, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                ),
            ],
        ),
    )

    placements = plan_resource_placements(
        config,
        state,
        resolver=ExpressionResolver(),
        ctx={"variables": {}, "workflow": {"name": "wf"}},
        replica_names_by_base={
            "worker": ["worker_0", "worker_1", "worker_2", "worker_3"]
        },
        replica_policy_by_base={"worker": "parallel"},
    )

    # Each replica's two GPUs are disjoint from every other replica's, in order.
    assert placements["worker_0"].cuda_visible_devices == "0,1"
    assert placements["worker_1"].cuda_visible_devices == "2,3"
    assert placements["worker_2"].cuda_visible_devices == "4,5"
    assert placements["worker_3"].cuda_visible_devices == "6,7"


def _plan_nodes_and_gpus(*, gpus: int, node_count: int = 2, caps=(8, 8)):
    """Plan one task pinned to ``node_count`` nodes asking for ``gpus`` GPUs total."""
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {
        "b": _FakeBackend(
            "b",
            allocation=Allocation(
                allocation_id="a",
                nodes=[
                    ComputeNode(name=f"n{i}", ip_address="", index=i, num_gpus=cap)
                    for i, cap in enumerate(caps)
                ],
                owned=False,
            ),
        )
    }
    state.default_backend = state.backends["b"]
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t",
                    script=["echo t"],
                    resources=ResourcesConfig(
                        nodes=NodeResourceConfig(count=node_count),
                        gpus=GpuResourceConfig(count=gpus),
                    ),
                )
            ],
        ),
    )
    return plan_resource_placements(
        config,
        state,
        resolver=ExpressionResolver(),
        ctx={"variables": {}, "workflow": {"name": "wf"}},
        replica_names_by_base={"t": ["t"]},
        replica_policy_by_base={"t": "parallel"},
    )


@pytest.mark.parametrize("gpus", [1, 3, 5], ids=["fewer-than-nodes", "odd", "odd-larger"])
def test_gpu_count_not_divisible_by_node_count_is_rejected(gpus):
    """`nodes.count: 2` + `gpus.count: 1` is unsatisfiable, not a rounding hint.

    `gpus.count` is a per-task TOTAL and a GPU run never straddles a node, so
    every assigned node takes the same slice. This used to `ceil()` and reserve
    that many on EVERY node -- silently consuming 2 GPUs for a 1-GPU request and
    4 for a 3-GPU request. The k8s operator has always rejected it; the planner
    now does too, so every backend agrees.
    """
    with pytest.raises(ValueError, match="must be a positive multiple of the node count"):
        _plan_nodes_and_gpus(gpus=gpus, node_count=2)


@pytest.mark.parametrize(
    "gpus,expected", [(2, "0"), (4, "0,1"), (8, "0,1,2,3")],
    ids=["1-per-node", "2-per-node", "4-per-node"],
)
def test_gpu_count_divisible_by_node_count_splits_evenly(gpus, expected):
    placements = _plan_nodes_and_gpus(gpus=gpus, node_count=2)
    assert placements["t"].cuda_visible_devices == expected
    assert placements["t"].assigned_nodes == ["n0", "n1"]


def test_error_message_names_a_usable_count():
    with pytest.raises(ValueError) as ei:
        _plan_nodes_and_gpus(gpus=3, node_count=2)
    # Tells the user exactly what to write instead of just what is wrong.
    assert "Use 4 for 2 GPU(s) per node" in str(ei.value)


def test_per_node_share_exceeding_the_smallest_node_is_rejected():
    """Divisible, and within total capacity, but not placeable.

    caps (8, 2) hold 10 GPUs between them, so the total-capacity check passes --
    yet 5 per node does not fit the 2-GPU node. This used to clamp per_node to 2
    and silently reserve 4 of the 10 requested GPUs.
    """
    with pytest.raises(ValueError, match="smallest assigned node has only 2"):
        _plan_nodes_and_gpus(gpus=10, node_count=2, caps=(8, 2))


def test_single_node_task_is_unaffected_by_the_divisibility_rule():
    # The rule is about splitting across nodes; a 1-node task keeps taking any
    # count that fits, including counts that are not multiples of anything.
    placements = _plan_nodes_and_gpus(gpus=3, node_count=1, caps=(8,))
    assert placements["t"].cuda_visible_devices == "0,1,2"


# ---------------------------------------------------------------------------
# Plan-time GPU transition info: who is the LAST user of a device
# ---------------------------------------------------------------------------


def _gpu_planner(*, ancestors, release_after, has_readiness=None):
    from sflow.app.resource_planner import GpuReservationPlanner

    planner = GpuReservationPlanner()
    planner.set_ancestors(ancestors)
    planner.gpu_release_after = dict(release_after)
    planner.has_readiness = dict(has_readiness or {})
    return planner


def _reserve(planner, task, start, count, node="n0"):
    planner.reserve(
        backend_name="b", node_name=node, task_name=task, start=start, count=count
    )


def test_a_task_whose_gpus_a_successor_reuses_is_flagged():
    """The runtime needs this to know that completing `taskx` must NOT publish its
    device: `tasky` is planned onto it but has not been submitted yet."""
    planner = _gpu_planner(
        ancestors={"tasky": {"taskx"}},
        release_after={"taskx": "task_completion", "tasky": "task_completion"},
    )
    _reserve(planner, "taskx", 0, 1)
    _reserve(planner, "tasky", 0, 1)

    assert planner.tasks_whose_gpus_are_reused() == {"taskx"}


def test_the_last_user_of_a_device_is_not_flagged():
    """It is the one whose completion genuinely frees the GPU for the whole host."""
    planner = _gpu_planner(
        ancestors={"tasky": {"taskx"}},
        release_after={"taskx": "task_completion", "tasky": "task_completion"},
    )
    _reserve(planner, "taskx", 0, 1)
    _reserve(planner, "tasky", 0, 1)

    assert "tasky" not in planner.tasks_whose_gpus_are_reused()


def test_a_task_on_a_device_nobody_else_wants_is_not_flagged():
    planner = _gpu_planner(
        ancestors={"tasky": {"taskx"}},
        release_after={"taskx": "task_completion", "tasky": "task_completion"},
    )
    _reserve(planner, "taskx", 0, 1)
    _reserve(planner, "tasky", 1, 1)  # a different device

    assert planner.tasks_whose_gpus_are_reused() == set()


def test_an_unrelated_task_on_the_same_device_does_not_count_as_reuse():
    """Reuse only ever happens along a dependency edge; two unrelated tasks are
    not a hand-over, so flagging one would hold its GPU for nothing."""
    planner = _gpu_planner(
        ancestors={},  # no ancestry between them
        release_after={"a": "task_completion", "b": "task_completion"},
    )
    _reserve(planner, "a", 0, 1)
    _reserve(planner, "b", 0, 1)

    assert planner.tasks_whose_gpus_are_reused() == set()


def test_a_workflow_completion_owner_is_never_flagged():
    """Nothing in this run may reuse it, so completing it frees the device."""
    planner = _gpu_planner(
        ancestors={"later": {"pinned"}},
        release_after={"pinned": "workflow_completion", "later": "task_completion"},
    )
    _reserve(planner, "pinned", 0, 1)
    _reserve(planner, "later", 0, 1)

    assert "pinned" not in planner.tasks_whose_gpus_are_reused()


def test_a_partial_device_overlap_still_flags_the_owner():
    """The successor only needs SOME of the owner's devices for the hand-over to
    matter -- publishing the rest early is the same bug."""
    planner = _gpu_planner(
        ancestors={"small": {"big"}},
        release_after={"big": "task_completion", "small": "task_completion"},
    )
    _reserve(planner, "big", 0, 4)
    _reserve(planner, "small", 2, 1)

    assert planner.tasks_whose_gpus_are_reused() == {"big"}
