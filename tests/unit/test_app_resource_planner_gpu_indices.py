# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placement tests for ``task.resources.gpus.indices``.

Three rules are covered here (see the design spec):

* Rule 1 -- ``indices`` only: exactly those GPU indices on the first node where
  they are all free. The node scan restarts at node 0 every time, so a later
  task can backfill low-numbered nodes.
* Rule 2 -- ``count`` + ``indices``: ``count`` is the total across nodes,
  ``indices`` is the per-node slice, so ``nodes_needed = count / len(indices)``.
* Rule 3 -- ``count`` only: index-agnostic, takes any contiguous idle run but
  never straddles a node boundary.
"""

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from sflow.app.resource_planner import plan_resource_placements
from sflow.config.resolver import ExpressionResolver
from sflow.config.schema import (
    GpuResourceConfig,
    NodeResourceConfig,
    ReplicaConfig,
    ResourceReleaseAfter,
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

HOLD = ResourceReleaseAfter.WORKFLOW_COMPLETION


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


def _backend(*, nodes: int, gpus_per_node: int | None, name: str = "b1") -> _FakeBackend:
    return _FakeBackend(
        name,
        allocation=Allocation(
            allocation_id="alloc0",
            nodes=[
                ComputeNode(
                    name=f"n{i}",
                    ip_address=f"10.0.0.{i}",
                    index=i,
                    num_gpus=gpus_per_node,
                )
                for i in range(nodes)
            ],
        ),
    )


def _plan(
    tasks: list[TaskConfig],
    *,
    backend: _FakeBackend,
    replicas: dict[str, list[str]] | None = None,
    variables: dict | None = None,
):
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {backend.name: backend}
    state.default_backend = backend
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(name="wf", tasks=tasks),
    )
    replica_names = replicas or {t.name: [t.name] for t in tasks}
    # Mirror the task's own replica policy -- hardcoding "parallel" here would
    # silently skip the sequential-replica DAG chaining under test.
    replica_policy = {
        t.name: getattr(t.replicas, "policy", "parallel") if t.replicas else "parallel"
        for t in tasks
    }
    return plan_resource_placements(
        config,
        state,
        resolver=ExpressionResolver(),
        ctx={
            "variables": variables or {},
            "workflow": {"name": "wf"},
        },
        replica_names_by_base=replica_names,
        replica_policy_by_base=replica_policy,
    )


def _task(
    name: str,
    *,
    gpus: GpuResourceConfig | None = None,
    nodes: NodeResourceConfig | None = None,
    depends_on: list[str] | None = None,
    replicas: ReplicaConfig | None = None,
) -> TaskConfig:
    return TaskConfig(
        name=name,
        script=[f"echo {name}"],
        depends_on=depends_on or [],
        replicas=replicas,
        resources=ResourcesConfig(gpus=gpus, nodes=nodes),
    )


# ---------------------------------------------------------------------------
# Rule 1 -- indices only
# ---------------------------------------------------------------------------


def test_indices_only_pins_the_requested_devices_on_the_first_node():
    placements = _plan(
        [_task("serve", gpus=GpuResourceConfig(indices=[0, 1]))],
        backend=_backend(nodes=2, gpus_per_node=4),
    )

    assert placements["serve"].assigned_nodes == ["n0"]
    assert placements["serve"].cuda_visible_devices == "0,1"
    # gpu_count falls back to len(indices) so operators still size the request.
    assert placements["serve"].gpu_count == 2


def test_indices_only_skips_a_busy_node_then_backfills_the_idle_slot():
    # a takes n0's GPUs 0,1 -> b wants 0,1 too and must skip to n1 -> c wants
    # 2,3, which are still idle back on n0.
    placements = _plan(
        [
            _task("a", gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD)),
            _task(
                "b",
                gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD),
                depends_on=["a"],
            ),
            _task(
                "c",
                gpus=GpuResourceConfig(indices=[2, 3], release_after=HOLD),
                depends_on=["b"],
            ),
        ],
        backend=_backend(nodes=2, gpus_per_node=4),
    )

    assert placements["a"].assigned_nodes == ["n0"]
    assert placements["a"].cuda_visible_devices == "0,1"
    assert placements["b"].assigned_nodes == ["n1"]
    assert placements["b"].cuda_visible_devices == "0,1"
    assert placements["c"].assigned_nodes == ["n0"]
    assert placements["c"].cuda_visible_devices == "2,3"


def test_indices_only_preserves_user_order_in_cuda_visible_devices():
    # CUDA_VISIBLE_DEVICES order defines the device remapping, so [1, 0] must
    # not be normalised to "0,1".
    placements = _plan(
        [_task("serve", gpus=GpuResourceConfig(indices=[1, 0]))],
        backend=_backend(nodes=1, gpus_per_node=4),
    )

    assert placements["serve"].cuda_visible_devices == "1,0"


def test_indices_only_supports_non_contiguous_devices():
    placements = _plan(
        [
            _task("a", gpus=GpuResourceConfig(indices=[0, 2], release_after=HOLD)),
            _task(
                "b",
                gpus=GpuResourceConfig(indices=[1, 3], release_after=HOLD),
                depends_on=["a"],
            ),
        ],
        backend=_backend(nodes=1, gpus_per_node=4),
    )

    assert placements["a"].cuda_visible_devices == "0,2"
    # The gap left by a is reusable: b fits on the same node.
    assert placements["b"].assigned_nodes == ["n0"]
    assert placements["b"].cuda_visible_devices == "1,3"


def test_indices_only_replicas_spread_across_nodes():
    placements = _plan(
        [
            _task(
                "worker",
                gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD),
                replicas=ReplicaConfig(count=3, policy="parallel"),
            )
        ],
        backend=_backend(nodes=3, gpus_per_node=4),
        replicas={"worker": ["worker_0", "worker_1", "worker_2"]},
    )

    assert placements["worker_0"].assigned_nodes == ["n0"]
    assert placements["worker_1"].assigned_nodes == ["n1"]
    assert placements["worker_2"].assigned_nodes == ["n2"]
    for name in ("worker_0", "worker_1", "worker_2"):
        assert placements[name].cuda_visible_devices == "0,1"


def test_indices_only_sequential_replicas_reuse_the_same_slots():
    # Sequential replicas are chained in the DAG, so each is an upstream
    # dependency of the next. With the inferred task_completion policy the
    # pinned slots are handed down instead of forcing every replica onto a
    # fresh node (which would exhaust the pool).
    placements = _plan(
        [
            _task(
                "worker",
                gpus=GpuResourceConfig(indices=[0, 1]),
                replicas=ReplicaConfig(count=3, policy="sequential"),
            )
        ],
        backend=_backend(nodes=2, gpus_per_node=4),
        replicas={"worker": ["worker_0", "worker_1", "worker_2"]},
    )

    for name in ("worker_0", "worker_1", "worker_2"):
        assert placements[name].assigned_nodes == ["n0"]
        assert placements[name].cuda_visible_devices == "0,1"


def test_indices_accepts_an_expression_resolving_to_a_list():
    placements = _plan(
        [_task("serve", gpus=GpuResourceConfig(indices="${{ variables.GPU_IDS }}"))],
        backend=_backend(nodes=1, gpus_per_node=4),
        variables={"GPU_IDS": [2, 3]},
    )

    assert placements["serve"].cuda_visible_devices == "2,3"


# ---------------------------------------------------------------------------
# Rule 2 -- count + indices
# ---------------------------------------------------------------------------


def test_count_with_indices_fans_out_over_nodes_using_the_same_slots():
    # 8 GPUs total, 2 indices per node -> 4 nodes, GPUs 0,1 on each.
    placements = _plan(
        [_task("train", gpus=GpuResourceConfig(count=8, indices=[0, 1]))],
        backend=_backend(nodes=4, gpus_per_node=4),
    )

    placement = placements["train"]
    assert placement.assigned_nodes == ["n0", "n1", "n2", "n3"]
    assert placement.cuda_visible_devices == "0,1"
    assert placement.gpu_count == 8
    assert placement.nodes_inferred_from_gpus is True


def test_count_with_indices_matching_one_node_stays_on_one_node():
    placements = _plan(
        [_task("train", gpus=GpuResourceConfig(count=2, indices=[0, 1]))],
        backend=_backend(nodes=4, gpus_per_node=4),
    )

    assert placements["train"].assigned_nodes == ["n0"]
    assert placements["train"].cuda_visible_devices == "0,1"


def test_count_with_indices_skips_nodes_whose_slots_are_taken():
    placements = _plan(
        [
            _task("a", gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD)),
            _task(
                "b",
                gpus=GpuResourceConfig(count=4, indices=[0, 1], release_after=HOLD),
                depends_on=["a"],
            ),
        ],
        backend=_backend(nodes=3, gpus_per_node=4),
    )

    assert placements["a"].assigned_nodes == ["n0"]
    assert placements["b"].assigned_nodes == ["n1", "n2"]
    assert placements["b"].cuda_visible_devices == "0,1"


def test_count_with_indices_honours_explicitly_pinned_nodes():
    placements = _plan(
        [
            _task(
                "train",
                gpus=GpuResourceConfig(count=4, indices=[2, 3]),
                nodes=NodeResourceConfig(indices=[1, 2]),
            )
        ],
        backend=_backend(nodes=4, gpus_per_node=4),
    )

    assert placements["train"].assigned_nodes == ["n1", "n2"]
    assert placements["train"].cuda_visible_devices == "2,3"


def test_indices_only_with_pinned_nodes_uses_every_pinned_node():
    placements = _plan(
        [
            _task(
                "train",
                gpus=GpuResourceConfig(indices=[0, 1]),
                nodes=NodeResourceConfig(count=2),
            )
        ],
        backend=_backend(nodes=4, gpus_per_node=4),
    )

    placement = placements["train"]
    assert placement.assigned_nodes == ["n0", "n1"]
    assert placement.cuda_visible_devices == "0,1"
    # 2 pinned nodes x 2 indices
    assert placement.gpu_count == 4


# ---------------------------------------------------------------------------
# Rule 3 -- count only never straddles a node boundary
# ---------------------------------------------------------------------------


def test_count_only_skips_a_partially_free_node_rather_than_straddling():
    # n0 has GPUs 2,3 free; a 4-GPU request must skip it and take all of n1.
    placements = _plan(
        [
            _task("a", gpus=GpuResourceConfig(count=2, release_after=HOLD)),
            _task(
                "b",
                gpus=GpuResourceConfig(count=4, release_after=HOLD),
                depends_on=["a"],
            ),
        ],
        backend=_backend(nodes=2, gpus_per_node=4),
    )

    assert placements["a"].assigned_nodes == ["n0"]
    assert placements["a"].cuda_visible_devices == "0,1"
    assert placements["b"].assigned_nodes == ["n1"]
    assert placements["b"].cuda_visible_devices == "0,1,2,3"


def test_count_only_takes_the_remaining_contiguous_run_when_it_fits():
    placements = _plan(
        [
            _task("a", gpus=GpuResourceConfig(count=2, release_after=HOLD)),
            _task(
                "b",
                gpus=GpuResourceConfig(count=2, release_after=HOLD),
                depends_on=["a"],
            ),
        ],
        backend=_backend(nodes=2, gpus_per_node=4),
    )

    assert placements["b"].assigned_nodes == ["n0"]
    assert placements["b"].cuda_visible_devices == "2,3"


def test_count_only_packs_around_an_indices_reservation():
    # An indices task holding 0,1 must not stop a count-only task from using the
    # rest of the same node.
    placements = _plan(
        [
            _task("pinned", gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD)),
            _task(
                "flexible",
                gpus=GpuResourceConfig(count=2, release_after=HOLD),
                depends_on=["pinned"],
            ),
        ],
        backend=_backend(nodes=2, gpus_per_node=4),
    )

    assert placements["flexible"].assigned_nodes == ["n0"]
    assert placements["flexible"].cuda_visible_devices == "2,3"


def test_indices_task_packs_around_a_count_only_reservation():
    placements = _plan(
        [
            _task("flexible", gpus=GpuResourceConfig(count=2, release_after=HOLD)),
            _task(
                "pinned",
                gpus=GpuResourceConfig(indices=[2, 3], release_after=HOLD),
                depends_on=["flexible"],
            ),
        ],
        backend=_backend(nodes=2, gpus_per_node=4),
    )

    assert placements["flexible"].cuda_visible_devices == "0,1"
    assert placements["pinned"].assigned_nodes == ["n0"]
    assert placements["pinned"].cuda_visible_devices == "2,3"


# ---------------------------------------------------------------------------
# Backends that report no GPU capacity (local/docker without gpus_per_node)
# ---------------------------------------------------------------------------


def test_count_with_indices_without_known_capacity_still_sizes_the_node_span():
    # No node reports num_gpus, so there are no real slots to pack against. The
    # request shape must still be honoured positionally -- previously this fell
    # through to "all allocated nodes" and then failed its own count-vs-placement
    # check with a misleading error.
    placements = _plan(
        [_task("train", gpus=GpuResourceConfig(count=4, indices=[0, 1]))],
        backend=_backend(nodes=4, gpus_per_node=None),
    )

    placement = placements["train"]
    assert placement.assigned_nodes == ["n0", "n1"]
    assert placement.cuda_visible_devices == "0,1"
    assert placement.gpu_count == 4


def test_indices_only_without_known_capacity_uses_a_single_node():
    placements = _plan(
        [_task("serve", gpus=GpuResourceConfig(indices=[2, 3]))],
        backend=_backend(nodes=4, gpus_per_node=None),
    )

    assert placements["serve"].assigned_nodes == ["n0"]
    assert placements["serve"].cuda_visible_devices == "2,3"


def test_indices_without_known_capacity_raises_when_pool_too_small():
    with pytest.raises(ValueError, match="needs 4 node"):
        _plan(
            [_task("train", gpus=GpuResourceConfig(count=8, indices=[0, 1]))],
            backend=_backend(nodes=1, gpus_per_node=None),
        )


def test_indices_without_known_capacity_still_skips_reserved_slots():
    # Reservations are tracked per node regardless of whether the node reports a
    # GPU capacity. Without consulting them, an unknown-capacity backend
    # double-booked the same indices on the first node instead of moving on.
    placements = _plan(
        [
            _task("a", gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD)),
            _task(
                "b",
                gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD),
                depends_on=["a"],
            ),
            _task(
                "c",
                gpus=GpuResourceConfig(indices=[2, 3], release_after=HOLD),
                depends_on=["b"],
            ),
        ],
        backend=_backend(nodes=3, gpus_per_node=None),
    )

    assert placements["a"].assigned_nodes == ["n0"]
    assert placements["b"].assigned_nodes == ["n1"]
    # ...and the idle high slots on n0 are still reachable, same as with a
    # known capacity.
    assert placements["c"].assigned_nodes == ["n0"]
    assert placements["c"].cuda_visible_devices == "2,3"


def test_pinned_nodes_without_known_capacity_still_reject_reserved_slots():
    # Explicit resources.nodes bypasses the index scan, so the conflict check in
    # _pinned_cuda_visible_devices is the only guard on this route.
    with pytest.raises(ValueError, match="already reserved"):
        _plan(
            [
                _task(
                    "holder",
                    gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD),
                    nodes=NodeResourceConfig(indices=[0]),
                ),
                _task(
                    "intruder",
                    gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD),
                    nodes=NodeResourceConfig(indices=[0]),
                    depends_on=["holder"],
                ),
            ],
            backend=_backend(nodes=2, gpus_per_node=None),
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_indices_out_of_range_for_every_node_raises():
    with pytest.raises(ValueError, match="resources.gpus.indices"):
        _plan(
            [_task("serve", gpus=GpuResourceConfig(indices=[6, 7]))],
            backend=_backend(nodes=2, gpus_per_node=4),
        )


def test_indices_busy_on_every_node_raises_with_blockers():
    with pytest.raises(ValueError) as excinfo:
        _plan(
            [
                _task("a", gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD)),
                _task(
                    "b",
                    gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD),
                    depends_on=["a"],
                ),
            ],
            backend=_backend(nodes=1, gpus_per_node=4),
        )

    message = str(excinfo.value)
    assert "b" in message
    # Reuses the existing GPU reservation diagnostics.
    assert "GPU 0" in message


def test_count_not_a_multiple_of_indices_raises():
    with pytest.raises(ValueError, match="multiple"):
        _plan(
            [_task("train", gpus=GpuResourceConfig(count=7, indices=[0, 1]))],
            backend=_backend(nodes=4, gpus_per_node=4),
        )


def test_count_smaller_than_indices_raises():
    with pytest.raises(ValueError, match="multiple"):
        _plan(
            [_task("train", gpus=GpuResourceConfig(count=1, indices=[0, 1]))],
            backend=_backend(nodes=4, gpus_per_node=4),
        )


def test_count_conflicting_with_pinned_node_count_raises():
    with pytest.raises(ValueError, match="resources.gpus.count"):
        _plan(
            [
                _task(
                    "train",
                    gpus=GpuResourceConfig(count=8, indices=[0, 1]),
                    nodes=NodeResourceConfig(indices=[0, 1]),
                )
            ],
            backend=_backend(nodes=4, gpus_per_node=4),
        )


def test_pinned_nodes_reject_indices_beyond_that_node_capacity():
    # With resources.nodes pinned, node selection returns before the GPU-index
    # scan runs, so this raise is the ONLY capacity guard on that route.
    with pytest.raises(ValueError, match="exceed node .* capacity"):
        _plan(
            [
                _task(
                    "train",
                    gpus=GpuResourceConfig(indices=[6, 7]),
                    nodes=NodeResourceConfig(indices=[0]),
                )
            ],
            backend=_backend(nodes=2, gpus_per_node=4),
        )


def test_pinned_nodes_reject_indices_already_reserved_on_that_node():
    # Same route: the scan can't skip to a free node because the user pinned
    # this one, so the conflict must surface as an error rather than silently
    # double-booking the devices.
    with pytest.raises(ValueError, match="already reserved"):
        _plan(
            [
                _task(
                    "holder",
                    gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD),
                    nodes=NodeResourceConfig(indices=[0]),
                ),
                _task(
                    "intruder",
                    gpus=GpuResourceConfig(indices=[0, 1], release_after=HOLD),
                    nodes=NodeResourceConfig(indices=[0]),
                    depends_on=["holder"],
                ),
            ],
            backend=_backend(nodes=2, gpus_per_node=4),
        )


def test_zero_count_with_indices_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        _plan(
            [_task("train", gpus=GpuResourceConfig(count=0, indices=[0, 1]))],
            backend=_backend(nodes=2, gpus_per_node=4),
        )


def test_indices_rejected_on_backend_without_gpu_env_support():
    backend = _backend(nodes=1, gpus_per_node=8, name="k8s")
    backend.capabilities = BackendCapabilities(
        supports_node_placement=True,
        supports_gpu_env=False,
        supports_host_path_mounts=False,
        has_runtime_node_addresses=True,
        supports_gpu_sharing=False,
    )

    with pytest.raises(ValueError, match="does not support"):
        _plan(
            [_task("serve", gpus=GpuResourceConfig(indices=[0, 1]))],
            backend=backend,
        )


def test_not_enough_nodes_for_count_with_indices_raises():
    with pytest.raises(ValueError, match="node"):
        _plan(
            [_task("train", gpus=GpuResourceConfig(count=8, indices=[0, 1]))],
            backend=_backend(nodes=2, gpus_per_node=4),
        )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_gpu_resource_config_requires_count_or_indices():
    with pytest.raises(ValidationError, match="count.*indices|indices.*count"):
        GpuResourceConfig()


def test_gpu_resource_config_rejects_negative_indices():
    with pytest.raises(ValidationError, match="non-negative"):
        GpuResourceConfig(indices=[-1])


def test_gpu_resource_config_rejects_duplicate_indices():
    with pytest.raises(ValidationError, match="duplicate"):
        GpuResourceConfig(indices=[0, 1, 0])


def test_gpu_resource_config_rejects_boolean_indices():
    # YAML reads `yes`/`no`/`on`/`off` as bools; without this guard Pydantic
    # coerces them to 0/1 and the task silently runs on the wrong device.
    with pytest.raises(ValidationError, match="bool"):
        GpuResourceConfig(indices=[True, False])


def test_gpu_resource_config_rejects_empty_indices():
    with pytest.raises(ValidationError, match="empty"):
        GpuResourceConfig(indices=[])


def test_gpu_resource_config_rejects_non_list_indices():
    with pytest.raises(ValidationError, match="list or an expression"):
        GpuResourceConfig(indices="0,1")


def test_gpu_resource_config_accepts_count_only():
    assert GpuResourceConfig(count=4).indices is None


def test_gpu_resource_config_accepts_indices_only():
    assert GpuResourceConfig(indices=[0, 1]).count is None


# ---------------------------------------------------------------------------
# Node reservation is opt-in, and stays opt-in across a serialization round trip
# ---------------------------------------------------------------------------


def test_node_pin_alone_does_not_claim_the_node():
    """Placement is not exclusion: CPU-only tasks pinned to a node share it.

    nats, etcd, a frontend and a benchmark client all sit on node 0 of an inference
    recipe. None of them names a release policy, so none of them claims the node.
    """
    placements = _plan(
        [
            _task("nats", nodes=NodeResourceConfig(indices=[0])),
            _task("etcd", nodes=NodeResourceConfig(indices=[0])),
            _task("bench", nodes=NodeResourceConfig(indices=[0])),
        ],
        backend=_backend(nodes=2, gpus_per_node=4),
    )
    shared = [placements[n].assigned_nodes for n in ("nats", "etcd", "bench")]
    assert shared == [shared[0]] * 3, shared  # all three on the same node
    assert len(shared[0]) == 1


def test_naming_a_policy_is_what_claims_the_node():
    with pytest.raises(ValueError, match="do not remain available"):
        _plan(
            [
                _task("holder", nodes=NodeResourceConfig(indices=[0], release_after=HOLD)),
                _task("intruder", nodes=NodeResourceConfig(indices=[0], release_after=HOLD)),
            ],
            backend=_backend(nodes=2, gpus_per_node=4),
        )


def test_a_dumped_and_reloaded_config_plans_the_same_way():
    """The plan must not change because the config went through YAML.

    `sflow compose` writes a config out, and `sflow batch` writes one for its dry run
    whenever --nodes / --gpus-per-node re-plan the backends. A dump materializes every
    default, so anything that reads "was this field set?" flips -- which turned every
    node-pinned task into a reservation holder and failed plans that were fine from
    source.
    """
    tasks = [
        _task("nats", nodes=NodeResourceConfig(indices=[0])),
        _task("etcd", nodes=NodeResourceConfig(indices=[0])),
        _task("bench", nodes=NodeResourceConfig(indices=[0])),
    ]
    from_source = _plan(tasks, backend=_backend(nodes=2, gpus_per_node=4))

    config = SflowConfig(version="0.1", workflow=WorkflowConfig(name="wf", tasks=tasks))
    reloaded = SflowConfig.model_validate(config.model_dump(mode="json", exclude_none=True))
    after_round_trip = _plan(
        list(reloaded.workflow.tasks), backend=_backend(nodes=2, gpus_per_node=4)
    )

    assert {n: p.assigned_nodes for n, p in after_round_trip.items()} == {
        n: p.assigned_nodes for n, p in from_source.items()
    }


def test_an_explicit_null_policy_reports_the_same_as_omitting_it():
    """`release_after: null` names no policy, and the REPORT has to agree with the plan.

    A placement's `resource_release_after` feeds the dry-run log and the execution
    summary. Deciding it from "was the field assigned?" rather than from its value made
    it announce `releases nodes after task completion` for a task the planner had
    correctly given no reservation to -- so these two configs, which mean exactly the
    same thing, described themselves differently.
    """
    omitted = _plan(
        [_task("pinned", nodes=NodeResourceConfig(indices=[0]))],
        backend=_backend(nodes=2, gpus_per_node=4),
    )
    explicit_null = _plan(
        [_task("pinned", nodes=NodeResourceConfig(indices=[0], release_after=None))],
        backend=_backend(nodes=2, gpus_per_node=4),
    )

    assert "nodes" not in omitted["pinned"].resource_release_after
    assert (
        explicit_null["pinned"].resource_release_after
        == omitted["pinned"].resource_release_after
    )


def test_a_named_policy_survives_the_round_trip_too():
    """The opposite direction: an explicit claim must not be lost by a dump."""
    tasks = [
        _task("holder", nodes=NodeResourceConfig(indices=[0], release_after=HOLD)),
        _task("intruder", nodes=NodeResourceConfig(indices=[0], release_after=HOLD)),
    ]
    config = SflowConfig(version="0.1", workflow=WorkflowConfig(name="wf", tasks=tasks))
    reloaded = SflowConfig.model_validate(config.model_dump(mode="json", exclude_none=True))

    with pytest.raises(ValueError, match="do not remain available"):
        _plan(list(reloaded.workflow.tasks), backend=_backend(nodes=2, gpus_per_node=4))
