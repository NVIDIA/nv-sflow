# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the k8s merge-pod grouping planner (``_plan_merge_groups``).

The grouping runs at assembly time after resource placement: it bundles the
single-node GPU tasks the planner co-located on one physical node (when the
backend opts in via ``merge_colocated_gpu_pods``) into one merged pod owned by a
deterministic leader, packs each member's CUDA_VISIBLE_DEVICES over the union GPU
range, and makes the leader wait on the union of the members' external deps.
"""

import logging

import pytest

from sflow.app import assembly as assembly_mod
from sflow.app.assembly import (
    _plan_merge_groups,
    _warn_channel_contention,
    _warn_interconnect_hints,
)
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph


@pytest.fixture(autouse=True)
def _sflow_logs_propagate(monkeypatch):
    # Other tests call the global ``configure_logging()``, which sets the parent
    # ``sflow`` logger to ``propagate=False`` (it owns a Rich console handler).
    # That stops assembly log records from reaching pytest's caplog handler
    # (installed on the root logger), leaving ``caplog.records`` empty regardless
    # of what the code logs. Force propagation for this module so caplog captures
    # the assembly info/warning lines independent of test-execution order.
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)


class _FakeBackend:
    def __init__(
        self,
        name: str,
        merge: bool = True,
        compute_domain_channel=None,
        nvlink_domain_scope=None,
        rdma_enabled: bool = False,
    ):
        self.name = name
        self.merge_colocated_gpu_pods = merge
        self.compute_domain_channel = compute_domain_channel
        self.nvlink_domain_scope = nvlink_domain_scope
        self.rdma_enabled = rdma_enabled


class _FakePlacement:
    def __init__(self, backend, assigned_nodes, gpu_count):
        self.backend = backend
        self.assigned_nodes = assigned_nodes
        self.gpu_count = gpu_count


class _RecordingOperator:
    def __init__(self, image=None):
        self.merge_call = None
        self._image = image

    def container_images(self):
        return [self._image] if self._image else []

    def apply_merge_group(self, *, members, union_gpus):
        self.merge_call = (list(members), union_gpus)


def _task(name: str, image=None) -> Task:
    return Task(
        name=name,
        logger=logging.getLogger(f"test.{name}"),
        operator=_RecordingOperator(image),
        status=TaskStatus.INITIATED,
    )


def _graph(names):
    tg = TaskGraph()
    for n in names:
        tg.dag.add_node(n, _task(n))
    return tg


def _graph_with_images(name_to_image):
    tg = TaskGraph()
    for n, img in name_to_image.items():
        tg.dag.add_node(n, _task(n, image=img))
    return tg


def test_groups_colocated_gpu_tasks_and_sets_leader_follower():
    tg = _graph(["decode", "prefill"])
    be = _FakeBackend("k8s")
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),
        "prefill": _FakePlacement(be, ["node-a"], 2),
    }
    _plan_merge_groups(tg, placements)

    leader = tg.get_task("decode")  # sorted order: decode < prefill
    follower = tg.get_task("prefill")
    assert leader.is_merge_leader is True
    assert leader.merge_leader is None
    assert leader.merge_members == ["decode", "prefill"]
    assert follower.is_merge_follower is True
    assert follower.merge_leader == "decode"
    assert follower.merge_members == []
    assert leader.merge_group_id == follower.merge_group_id
    assert leader.merge_group_id


def test_packs_cuda_visible_devices_over_union_and_calls_operator():
    tg = _graph(["decode", "prefill"])
    be = _FakeBackend("k8s")
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),
        "prefill": _FakePlacement(be, ["node-a"], 2),
    }
    _plan_merge_groups(tg, placements)

    leader = tg.get_task("decode")
    follower = tg.get_task("prefill")
    # Every member sees ALL 6 union GPUs, with its OWN slice first (so cross-member
    # cuda_ipc/NVLink P2P works); a single-GPU visibility would force IB.
    assert leader.merge_cuda_visible_devices == "0,1,2,3,4,5"
    assert follower.merge_cuda_visible_devices == "4,5,0,1,2,3"
    # The leader operator is handed the ordered member Task objects + union GPUs.
    members, union = leader.operator.merge_call
    assert union == 6
    assert [m.name for m in members] == ["decode", "prefill"]


def test_every_member_sees_all_union_gpus_own_first():
    # Regression: a single-GPU-per-member CUDA_VISIBLE_DEVICES hides peer GPUs and
    # blocks cross-member cuda_ipc/NVLink (UCX falls back to IB). Each member must
    # see ALL union GPUs, with its own listed first.
    names = ["decode_0", "decode_1", "decode_2", "prefill_0"]
    tg = _graph(names)
    be = _FakeBackend("k8s")
    placements = {n: _FakePlacement(be, ["node-a"], 1) for n in names}
    _plan_merge_groups(tg, placements)
    assert tg.get_task("decode_0").merge_cuda_visible_devices == "0,1,2,3"
    assert tg.get_task("decode_1").merge_cuda_visible_devices == "1,0,2,3"
    assert tg.get_task("decode_2").merge_cuda_visible_devices == "2,0,1,3"
    assert tg.get_task("prefill_0").merge_cuda_visible_devices == "3,0,1,2"
    for n in names:
        cvd = tg.get_task(n).merge_cuda_visible_devices
        assert sorted(int(x) for x in cvd.split(",")) == [0, 1, 2, 3]


def test_merge_logs_transparent_line_when_merging(caplog):
    tg = _graph(["decode", "prefill"])
    be = _FakeBackend("k8s")
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),
        "prefill": _FakePlacement(be, ["node-a"], 2),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.INFO, logger="sflow.app.assembly"):
        _plan_merge_groups(tg, placements)
    msgs = [r.getMessage() for r in caplog.records]
    # One transparent line naming the merged tasks + node.
    assert any(
        "merged" in m.lower() and "node-a" in m and "decode" in m and "prefill" in m
        for m in msgs
    )


def test_no_merge_when_backend_disabled():
    tg = _graph(["decode", "prefill"])
    be = _FakeBackend("k8s", merge=False)
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),
        "prefill": _FakePlacement(be, ["node-a"], 2),
    }
    _plan_merge_groups(tg, placements)
    assert tg.get_task("decode").is_merge_leader is False
    assert tg.get_task("prefill").is_merge_follower is False


def test_no_merge_for_single_gpu_task_on_node():
    tg = _graph(["solo"])
    be = _FakeBackend("k8s")
    _plan_merge_groups(tg, {"solo": _FakePlacement(be, ["node-a"], 8)})
    assert tg.get_task("solo").is_merge_leader is False


def test_cpu_only_task_is_not_merged():
    tg = _graph(["decode", "etcd"])
    be = _FakeBackend("k8s")
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),
        "etcd": _FakePlacement(be, ["node-a"], 0),
    }
    _plan_merge_groups(tg, placements)
    # Only one GPU task on the node -> nothing to merge; etcd stays independent.
    assert tg.get_task("decode").is_merge_leader is False
    assert tg.get_task("etcd").is_merge_follower is False


def test_multi_node_gpu_task_is_not_merged():
    tg = _graph(["big", "decode"])
    be = _FakeBackend("k8s")
    placements = {
        "big": _FakePlacement(be, ["node-a", "node-b"], 16),
        "decode": _FakePlacement(be, ["node-a"], 4),
    }
    _plan_merge_groups(tg, placements)
    assert tg.get_task("big").is_merge_leader is False
    assert tg.get_task("big").is_merge_follower is False


def test_different_nodes_are_separate_groups():
    tg = _graph(["a", "b"])
    be = _FakeBackend("k8s")
    placements = {
        "a": _FakePlacement(be, ["node-a"], 4),
        "b": _FakePlacement(be, ["node-b"], 4),
    }
    _plan_merge_groups(tg, placements)
    assert tg.get_task("a").is_merge_leader is False
    assert tg.get_task("b").is_merge_leader is False


def test_no_merge_when_container_images_differ():
    # A merged pod is a single container, so co-located tasks that would launch
    # DIFFERENT images must NOT be merged (a follower would otherwise silently run
    # in the leader's image).
    tg = _graph_with_images({"decode": "img-a:1", "prefill": "img-b:1"})
    be = _FakeBackend("k8s")
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),
        "prefill": _FakePlacement(be, ["node-a"], 2),
    }
    _plan_merge_groups(tg, placements)
    assert tg.get_task("decode").is_merge_leader is False
    assert tg.get_task("prefill").is_merge_follower is False


def test_merge_when_container_images_match():
    # Same node AND same image -> merges into one pod as before.
    tg = _graph_with_images({"decode": "img-a:1", "prefill": "img-a:1"})
    be = _FakeBackend("k8s")
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),
        "prefill": _FakePlacement(be, ["node-a"], 2),
    }
    _plan_merge_groups(tg, placements)
    assert tg.get_task("decode").is_merge_leader is True
    assert tg.get_task("prefill").merge_leader == "decode"


def test_direct_intra_group_dependency_gates_instead_of_raising():
    tg = _graph(["a", "b"])
    tg.dag.add_edge("a", "b")  # b depends on a; both merged on one node
    be = _FakeBackend("k8s")
    placements = {
        "a": _FakePlacement(be, ["node-a"], 2),
        "b": _FakePlacement(be, ["node-a"], 2),
    }
    _plan_merge_groups(tg, placements)  # no raise
    # a is the leader (sorted first); both are still merged.
    assert tg.get_task("a").is_merge_leader is True
    assert tg.get_task("b").merge_leader == "a"
    # b gates on a; a gates on nobody.
    assert tg.get_task("b").merge_gate_after == ["a"]
    assert tg.get_task("a").merge_gate_after == []


def test_non_dependent_members_have_empty_gate():
    tg = _graph(["decode", "prefill"])  # no edge between them
    be = _FakeBackend("k8s")
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),
        "prefill": _FakePlacement(be, ["node-a"], 2),
    }
    _plan_merge_groups(tg, placements)
    assert tg.get_task("decode").merge_gate_after == []
    assert tg.get_task("prefill").merge_gate_after == []


def test_multi_and_chained_intra_group_deps_gate_directly():
    # c depends on a AND b; b depends on a. All three merged on one node.
    tg = _graph(["a", "b", "c"])
    tg.dag.add_edge("a", "b")  # b depends on a
    tg.dag.add_edge("a", "c")  # c depends on a
    tg.dag.add_edge("b", "c")  # c depends on b
    be = _FakeBackend("k8s")
    placements = {n: _FakePlacement(be, ["node-a"], 2) for n in ("a", "b", "c")}
    _plan_merge_groups(tg, placements)
    assert tg.get_task("a").merge_gate_after == []
    assert tg.get_task("b").merge_gate_after == ["a"]
    assert tg.get_task("c").merge_gate_after == ["a", "b"]


def test_gated_member_still_inherits_external_deps_on_leader():
    # b depends on a (member) and a depends on e (external CPU-only). The leader (a)
    # must still wait for e; the intra-group edge a<-b becomes a gate, not a leader dep.
    tg = _graph(["a", "b", "e"])
    tg.dag.add_edge("a", "b")  # b depends on a
    tg.dag.add_edge("e", "a")  # a depends on e
    be = _FakeBackend("k8s")
    placements = {
        "a": _FakePlacement(be, ["node-a"], 2),
        "b": _FakePlacement(be, ["node-a"], 2),
        "e": _FakePlacement(be, ["node-a"], 0),  # CPU-only -> external
    }
    _plan_merge_groups(tg, placements)
    assert "e" in tg.dag.get_dependencies("a")   # external dep hoisted onto leader
    assert tg.get_task("b").merge_gate_after == ["a"]


def test_intra_group_transitive_dependency_raises():
    # a depends on x, x depends on b, with a and b merged and x a non-member.
    # There is no direct a<->b edge, so only a transitive walk catches the
    # ordering the concurrent pod can't honor (else it deadlocks at run time).
    tg = _graph(["a", "b", "x"])
    tg.dag.add_edge("x", "a")  # a depends on x
    tg.dag.add_edge("b", "x")  # x depends on b
    be = _FakeBackend("k8s")
    placements = {
        "a": _FakePlacement(be, ["node-a"], 2),
        "b": _FakePlacement(be, ["node-a"], 2),
    }
    with pytest.raises(ValueError, match="transitively"):
        _plan_merge_groups(tg, placements)


def test_merge_raises_when_leader_operator_lacks_apply_merge_group():
    # A backend that opts into merge but whose co-located tasks use an operator
    # without apply_merge_group would promote followers that never actually launch
    # in a merged pod -> fail loudly at plan time instead of hanging the group.
    class _NoMergeOperator:
        def container_images(self):
            return []

        # intentionally NO apply_merge_group

    tg = TaskGraph()
    for n in ("decode", "prefill"):
        tg.dag.add_node(
            n,
            Task(
                name=n,
                logger=logging.getLogger(f"test.{n}"),
                operator=_NoMergeOperator(),
                status=TaskStatus.INITIATED,
            ),
        )
    be = _FakeBackend("k8s")
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),
        "prefill": _FakePlacement(be, ["node-a"], 2),
    }
    with pytest.raises(ValueError, match="apply_merge_group"):
        _plan_merge_groups(tg, placements)


def test_leader_inherits_external_deps_of_all_members():
    tg = _graph(["download", "decode", "prefill"])
    tg.dag.add_edge("download", "prefill")  # only prefill depends on download
    be = _FakeBackend("k8s")
    placements = {
        "download": _FakePlacement(be, ["node-a"], 0),  # CPU-only -> external
        "decode": _FakePlacement(be, ["node-a"], 2),
        "prefill": _FakePlacement(be, ["node-a"], 2),
    }
    _plan_merge_groups(tg, placements)
    # Leader (decode) must wait for the union of members' external deps.
    assert "download" in tg.dag.get_dependencies("decode")


# ---------------------------------------------------------------------------
# channel-contention guard (component 3): a claimed IMEX channel + merge off +
# >1 GPU task on a node -> hard-warn (the driver publishes ONE channel per node).
# ---------------------------------------------------------------------------


def _warn_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_channel_contention_warns_when_merge_off_and_multiple_gpu_tasks_per_node(caplog):
    be = _FakeBackend("k8s", merge=False, compute_domain_channel="cd-chan")
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 2),
        "decode": _FakePlacement(be, ["node-a"], 2),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_channel_contention(placements)
    msgs = _warn_messages(caplog)
    assert any("channel" in m.lower() and "node-a" in m for m in msgs)


def test_no_channel_contention_when_merge_on(caplog):
    be = _FakeBackend("k8s", merge=True, compute_domain_channel="cd-chan")
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 2),
        "decode": _FakePlacement(be, ["node-a"], 2),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_channel_contention(placements)
    assert not _warn_messages(caplog)


def test_no_channel_contention_when_no_channel(caplog):
    be = _FakeBackend("k8s", merge=False, compute_domain_channel=None)
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 2),
        "decode": _FakePlacement(be, ["node-a"], 2),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_channel_contention(placements)
    assert not _warn_messages(caplog)


def test_no_channel_contention_with_one_gpu_task_per_node(caplog):
    be = _FakeBackend("k8s", merge=False, compute_domain_channel="cd-chan")
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 4),
        "decode": _FakePlacement(be, ["node-b"], 4),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_channel_contention(placements)
    assert not _warn_messages(caplog)


def test_channel_contention_ignores_cpu_only_tasks(caplog):
    be = _FakeBackend("k8s", merge=False, compute_domain_channel="cd-chan")
    placements = {
        "decode": _FakePlacement(be, ["node-a"], 4),  # one GPU task
        "etcd": _FakePlacement(be, ["node-a"], 0),  # CPU-only -> does not claim
        "nats": _FakePlacement(be, ["node-a"], 0),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_channel_contention(placements)
    assert not _warn_messages(caplog)


# ---------------------------------------------------------------------------
# app-agnostic interconnect hints (component 7): cross-node placement + scope +
# IB + channel -> hint the missing (framework/admin-owned) pieces.
# ---------------------------------------------------------------------------


def test_hint_cross_node_node_scope_ib_down(caplog):
    # Cross-node GPU placement, scope 'node', IB down, NO channel -> no fast
    # cross-node interconnect; recommend co-locating or enabling IB, AND flag that
    # an NVL72 rack may be mis-detected (point to compute_domain.channel).
    be = _FakeBackend("k8s", nvlink_domain_scope="node", rdma_enabled=False)
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 4),
        "decode": _FakePlacement(be, ["node-b"], 4),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_interconnect_hints(placements)
    msgs = _warn_messages(caplog)
    assert any(("co-locate" in m.lower() or "IB" in m) for m in msgs)
    # Clarity: acknowledge a possible NVL72 scope mis-detection + the fix.
    assert any("compute_domain.channel" in m for m in msgs)


def test_no_hint_node_scope_with_channel(caplog):
    # Scope under-detected as 'node' (e.g. the ComputeDomain CRD is not readable
    # under the caller's RBAC) but an IMEX channel IS configured -> the channel is
    # the cross-node NVLink path, so emit NO misleading "slow TCP" hint.
    be = _FakeBackend(
        "k8s",
        nvlink_domain_scope="node",
        rdma_enabled=False,
        compute_domain_channel="cd-k8s-demo",
    )
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 4),
        "decode": _FakePlacement(be, ["node-b"], 4),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_interconnect_hints(placements)
    assert not _warn_messages(caplog)


def test_hint_cross_node_rack_scope_no_channel(caplog):
    be = _FakeBackend("k8s", nvlink_domain_scope="rack", compute_domain_channel=None)
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 4),
        "decode": _FakePlacement(be, ["node-b"], 4),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_interconnect_hints(placements)
    msgs = _warn_messages(caplog)
    assert any("compute_domain.channel" in m for m in msgs)


def test_hint_rack_scope_explains_no_auto_pick(caplog):
    # sflow does NOT auto-select a ComputeDomain channel (a cluster may expose
    # several); the cross-node-GPU warning must tell the user to pick one explicitly.
    be = _FakeBackend("k8s", nvlink_domain_scope="rack", compute_domain_channel=None)
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 4),
        "decode": _FakePlacement(be, ["node-b"], 4),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_interconnect_hints(placements)
    msgs = _warn_messages(caplog)
    assert any("does not auto-pick" in m for m in msgs)
    assert any("kubectl get computedomains" in m for m in msgs)


def test_no_hint_rack_scope_with_channel(caplog):
    be = _FakeBackend(
        "k8s", nvlink_domain_scope="rack", compute_domain_channel="cd-chan"
    )
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 4),
        "decode": _FakePlacement(be, ["node-b"], 4),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_interconnect_hints(placements)
    assert not _warn_messages(caplog)


def test_no_hint_when_single_node_placement(caplog):
    be = _FakeBackend("k8s", nvlink_domain_scope="node", rdma_enabled=False)
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 4),
        "decode": _FakePlacement(be, ["node-a"], 4),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_interconnect_hints(placements)
    assert not _warn_messages(caplog)


def test_no_hint_node_scope_with_ib_up(caplog):
    be = _FakeBackend("k8s", nvlink_domain_scope="node", rdma_enabled=True)
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 4),
        "decode": _FakePlacement(be, ["node-b"], 4),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_interconnect_hints(placements)
    assert not _warn_messages(caplog)


def test_no_hint_when_scope_unknown(caplog):
    be = _FakeBackend("k8s", nvlink_domain_scope=None)
    placements = {
        "prefill": _FakePlacement(be, ["node-a"], 4),
        "decode": _FakePlacement(be, ["node-b"], 4),
    }
    assembly_mod._logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="sflow.app.assembly"):
        _warn_interconnect_hints(placements)
    assert not _warn_messages(caplog)


# --- member order follows the DAG (upstream first), not the alphabet ---------
# The leader OWNS the shared pod: its execute() holds the container and deletes it
# on return. A downstream one-shot leader could therefore never resolve -- its own
# siblings keep alive the pod it is waiting on. Ordering members upstream-first
# makes the pod's owner the member the others wait on.


def test_member_order_follows_dag_not_alphabet_so_upstream_leads():
    # 'benchmark' sorts first but depends on 'server'; 'server' must lead.
    tg = _graph(["benchmark", "server"])
    tg.dag.add_edge("server", "benchmark")  # server -> benchmark
    be = _FakeBackend("k8s")
    placements = {
        "benchmark": _FakePlacement(be, ["node-a"], 1),
        "server": _FakePlacement(be, ["node-a"], 4),
    }
    _plan_merge_groups(tg, placements)

    leader = tg.get_task("server")
    assert leader.is_merge_leader is True
    assert leader.merge_members == ["server", "benchmark"]  # upstream first
    assert tg.get_task("benchmark").merge_leader == "server"
    # The downstream member is still gated behind its upstream in-pod.
    assert tg.get_task("benchmark").merge_gate_after == ["server"]


def test_dag_order_puts_terminal_task_last_in_a_realistic_group():
    # frontends serve; harness consumes them and exits -> harness must not lead.
    names = ["frontend_a", "frontend_b", "mlperf_harness"]
    tg = _graph(names)
    tg.dag.add_edge("frontend_a", "mlperf_harness")
    tg.dag.add_edge("frontend_b", "mlperf_harness")
    be = _FakeBackend("k8s")
    placements = {n: _FakePlacement(be, ["node-a"], 1) for n in names}
    _plan_merge_groups(tg, placements)

    leader = tg.get_task("frontend_a")
    assert leader.is_merge_leader is True
    assert leader.merge_members == ["frontend_a", "frontend_b", "mlperf_harness"]


def test_independent_members_keep_deterministic_name_order():
    # No intra-group edge -> the DAG says nothing; stay on stable name order.
    tg = _graph(["beta", "alpha"])
    be = _FakeBackend("k8s")
    placements = {
        "alpha": _FakePlacement(be, ["node-a"], 2),
        "beta": _FakePlacement(be, ["node-a"], 2),
    }
    _plan_merge_groups(tg, placements)

    assert tg.get_task("alpha").is_merge_leader is True
    assert tg.get_task("alpha").merge_members == ["alpha", "beta"]


def test_dag_order_drives_gpu_slice_packing():
    # Slices are packed in member order, so upstream-first also owns the first GPUs.
    tg = _graph(["benchmark", "server"])
    tg.dag.add_edge("server", "benchmark")
    be = _FakeBackend("k8s")
    placements = {
        "benchmark": _FakePlacement(be, ["node-a"], 2),
        "server": _FakePlacement(be, ["node-a"], 4),
    }
    _plan_merge_groups(tg, placements)

    assert tg.get_task("server").merge_cuda_visible_devices == "0,1,2,3,4,5"
    assert tg.get_task("benchmark").merge_cuda_visible_devices == "4,5,0,1,2,3"
