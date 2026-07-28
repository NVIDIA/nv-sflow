# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K8s merge-pod DAG regression suite: the DAG must progress and terminate for ANY
mix of members in a merged pod.

Several co-located single-node GPU tasks are folded into ONE pod (``merge_colocated
_gpu_pods``), where each member runs as a background process in the shared
container. That container only exits once EVERY member's script returns, so with a
long-running member in the group the pod NEVER reaches a terminal phase. Any design
that reads a member's outcome from the pod therefore deadlocks the workflow -- the
observed bug: a merged terminal task stayed non-terminal forever and the run hung
until it was killed by hand.

The contract these tests pin down:

* every member's status comes from ITS OWN exit (the ``[[sflow-member-done:<rc>]]``
  marker it writes when its own script returns) -- never from the pod, never from a
  sibling, never from the leader;
* the shared pod's lifetime is decoupled from all of that: the leader's
  ``execute()`` lingers to hold the pod and the driver reclaims it at teardown once
  the DAG is done;
* so the DAG reaches a terminal state for EVERY group composition -- all one-shot,
  all services, or any mix, whichever member happens to lead.

Composition is the axis that used to break, so it is parametrized explicitly:
whether a member is a service (readiness-probed, never exits on its own) or a
one-shot, and whether the member that ends up leading is one or the other.

Assembly-time grouping/ordering lives in ``test_app_assembly_merge_groups.py`` and
the marker plumbing in ``test_plugin_operators_k8s.py``; this module is about the
DAG outcome that the two together must guarantee.
"""

import asyncio
import logging

import pytest

from sflow.app.assembly import _plan_merge_groups
from sflow.core.orchestrator import Orchestrator
from sflow.core.probe import ProbeType
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


class _FakeBackend:
    def __init__(self, name="k8s", merge=True):
        self.name = name
        self.merge_colocated_gpu_pods = merge
        self.compute_domain_channel = None
        self.nvlink_domain_scope = None
        self.rdma_enabled = False


class _FakePlacement:
    def __init__(self, backend, assigned_nodes, gpu_count):
        self.backend = backend
        self.assigned_nodes = assigned_nodes
        self.gpu_count = gpu_count


class _MergeOperator:
    """Stands in for the k8s operator: groups merge, and each member reports its
    own exit code exactly as ``merged_member_exit_code`` does off the member's log."""

    def __init__(self, image="img:1"):
        self._image = image
        self.merge_call = None
        self.rc_by_task: dict[str, int] = {}

    def container_images(self):
        return [self._image]

    def apply_merge_group(self, *, members, union_gpus):
        self.merge_call = ([m.name for m in members], union_gpus)

    def merged_member_exit_code(self, task):
        return self.rc_by_task.get(task.name)


class _ReadinessProbe:
    """Marks a member as a service: it reaches READY and never exits on its own."""

    type = ProbeType.READINESS


def _task(name, operator, *, service=False):
    t = Task(
        name=name,
        logger=logging.getLogger(f"test.{name}"),
        operator=operator,
        status=TaskStatus.INITIATED,
    )
    if service:
        t.probes = [_ReadinessProbe()]
    return t


def _build(members, *, edges=(), gpus=1, extra_tasks=()):
    """Plan a merged group from ``members`` = [(name, is_service), ...].

    Returns (orchestrator, workflow, operator). ``edges`` are (upstream, downstream)
    DAG edges; ``extra_tasks`` are non-GPU tasks that stay OUTSIDE the merged pod.
    """
    op = _MergeOperator()
    tg = TaskGraph()
    for name, service in members:
        tg.dag.add_node(name, _task(name, op, service=service))
    for name in extra_tasks:
        tg.dag.add_node(name, _task(name, op))
    for up, down in edges:
        tg.dag.add_edge(up, down)
    be = _FakeBackend()
    placements = {
        name: _FakePlacement(be, ["node-a"], gpus) for name, _ in members
    }
    _plan_merge_groups(tg, placements)
    wf = Workflow(name="wf", task_graph=tg)
    orch = Orchestrator(workflow=wf, poll_interval=0.01, fail_fast=True)
    return orch, wf, op


def _start(wf):
    """Put every merged member in the state the leader's submit would leave it."""
    for t in wf.get_tasks():
        if t.is_merge_leader or t.is_merge_follower:
            t.status = TaskStatus.RUNNING


def _reach_ready(wf):
    """Services reach READY on their own probes while their scripts keep running."""
    for t in wf.get_tasks():
        if getattr(t, "probes", None) and t.status == TaskStatus.RUNNING:
            t.status = TaskStatus.READY


def _finish(orch, wf, op, rc_by_task):
    """A member's script returns -> it writes its own marker -> driver resolves it."""
    op.rc_by_task.update(rc_by_task)
    asyncio.run(orch._resolve_finished_merge_members())


# --- composition matrix: the DAG must terminate for ANY mix ------------------


@pytest.mark.parametrize(
    "members,label",
    [
        ([("alpha", False), ("beta", False)], "all one-shot"),
        ([("server", True), ("worker", True)], "all services"),
        ([("server", True), ("bench", False)], "service + one-shot"),
        # 'bench' sorts first and is one-shot: without DAG ordering it would lead.
        ([("bench", False), ("server", True)], "one-shot first by name"),
        (
            [("api", True), ("cache", True), ("probe", False), ("report", False)],
            "2 services + 2 one-shots",
        ),
    ],
)
def test_dag_terminates_for_any_merged_pod_composition(members, label):
    orch, wf, op = _build(members)
    _start(wf)
    _reach_ready(wf)
    # Every one-shot member's script returns cleanly; services keep serving.
    _finish(orch, wf, op, {n: 0 for n, service in members if not service})

    assert wf.is_finished() is True, f"DAG did not terminate for: {label}"
    for name, service in members:
        t = wf.get_task(name)
        expected = TaskStatus.READY if service else TaskStatus.COMPLETED
        assert t.status == expected, f"{name} in '{label}'"


@pytest.mark.parametrize("leader_is_service", [True, False])
def test_dag_terminates_whichever_member_leads(leader_is_service):
    # The leader OWNS the pod (its execute() holds the container), so it used to be
    # the one shape that could not resolve. Both leaders must now terminate.
    members = [("aaa", leader_is_service), ("zzz", not leader_is_service)]
    orch, wf, op = _build(members)
    leader = next(t for t in wf.get_tasks() if t.is_merge_leader)
    assert leader.name == "aaa"  # no intra-group edge -> name order
    _start(wf)
    _reach_ready(wf)
    _finish(orch, wf, op, {n: 0 for n, service in members if not service})

    assert wf.is_finished() is True
    assert leader.status.is_terminal()


def test_one_shot_leader_completes_while_its_service_sibling_serves():
    # The exact deadlock shape: leader is one-shot, its sibling keeps the pod alive.
    orch, wf, op = _build([("bench", False), ("server", True)])
    leader = wf.get_task("bench")
    assert leader.is_merge_leader is True  # no edge -> 'bench' leads by name
    _start(wf)
    _reach_ready(wf)
    _finish(orch, wf, op, {"bench": 0})

    assert leader.status == TaskStatus.COMPLETED
    assert wf.get_task("server").status == TaskStatus.READY  # pod still needed
    assert wf.is_finished() is True


def test_service_leader_with_one_shot_terminal_member():
    # The reported production shape: routers lead, the harness is the terminal task.
    members = [("frontend_a", True), ("frontend_b", True), ("harness", False)]
    orch, wf, op = _build(
        members, edges=[("frontend_a", "harness"), ("frontend_b", "harness")]
    )
    assert wf.get_task("frontend_a").is_merge_leader is True  # upstream leads
    _start(wf)
    _reach_ready(wf)
    _finish(orch, wf, op, {"harness": 0})

    assert wf.get_task("harness").status == TaskStatus.COMPLETED
    assert wf.is_finished() is True


# --- failures and edges still travel the DAG --------------------------------


def test_failed_merged_member_is_terminal_and_stops_the_dag():
    orch, wf, op = _build([("server", True), ("bench", False)])
    _start(wf)
    _reach_ready(wf)
    _finish(orch, wf, op, {"bench": 9})

    bench = wf.get_task("bench")
    assert bench.status == TaskStatus.FAILED
    assert bench.exit_code == 9
    assert wf.is_finished() is True  # terminal, so the run ends instead of hanging


def test_unfinished_member_keeps_the_dag_open():
    # Guard against the opposite regression: nothing may resolve without a marker.
    orch, wf, op = _build([("server", True), ("bench", False)])
    _start(wf)
    _reach_ready(wf)
    _finish(orch, wf, op, {})  # bench still running -> no marker

    assert wf.get_task("bench").status == TaskStatus.RUNNING
    assert wf.is_finished() is False


def test_external_downstream_task_unblocks_when_a_merged_member_completes():
    # The DAG edge out of the pod must be honored: a task OUTSIDE the merged pod
    # that depends on a merged member becomes submittable once that member is done.
    orch, wf, op = _build(
        [("server", True), ("producer", False)],
        edges=[("producer", "consumer")],
        extra_tasks=["consumer"],
    )
    consumer = wf.get_task("consumer")
    assert consumer.is_merge_follower is False  # not merged (no GPU placement)
    _start(wf)
    _reach_ready(wf)
    assert consumer.name not in [t.name for t in wf.get_tasks_to_submit()]

    _finish(orch, wf, op, {"producer": 0})
    assert consumer.name in [t.name for t in wf.get_tasks_to_submit()]


def test_resolution_is_idempotent_across_poll_ticks():
    # The marker stays in the log and the poll loop re-runs every tick; a resolved
    # member must not be finalized twice (double result parsing / uploads).
    orch, wf, op = _build([("server", True), ("bench", False)])
    _start(wf)
    _reach_ready(wf)
    _finish(orch, wf, op, {"bench": 0})
    first = wf.get_task("bench").status
    asyncio.run(orch._resolve_finished_merge_members())
    asyncio.run(orch._resolve_finished_merge_members())

    assert wf.get_task("bench").status == first == TaskStatus.COMPLETED


def test_service_member_is_never_resolved_by_a_sibling_finishing():
    # A service has no marker until it stops, so a finished sibling must not drag
    # it to COMPLETED (that would wrongly satisfy dependents expecting it to serve).
    orch, wf, op = _build([("server", True), ("bench", False)])
    _start(wf)
    _reach_ready(wf)
    _finish(orch, wf, op, {"bench": 0})

    assert wf.get_task("server").status == TaskStatus.READY
    assert wf.get_task("server").exit_code is None
