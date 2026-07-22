# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from sflow.app.resource_planner import GpuReservationPlanner, plan_resource_placements
from sflow.config.resolver import ExpressionResolver
from sflow.config.schema import (
    GpuResourceConfig,
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
