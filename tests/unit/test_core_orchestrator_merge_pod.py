# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge-pod follower lifecycle in the orchestrator.

Merge-pod followers run as background processes inside their leader's single pod,
so the orchestrator never launches them on their own: the leader promotes them to
RUNNING when it starts (so their own probes/logs are evaluated) and mirrors its
terminal outcome onto them when it finishes.
"""

import asyncio
import logging

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.orchestrator import Orchestrator
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


class _Op(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="fake"))

    def build_command(self, *, task_name, script, envs) -> Command:
        return Command(exec="echo").add_arg("x")


class _RecordingSummary:
    def __init__(self):
        self.submitted: list[str] = []
        self.failed: list[str] = []
        self.cancelled: list[str] = []

    def task_submitted(self, task, **_):
        self.submitted.append(task.name)

    def task_failed(self, task, **_):
        self.failed.append(task.name)

    def task_cancelled(self, task, **_):
        self.cancelled.append(task.name)


def _mk(leader_status=TaskStatus.RUNNING, follower_status=TaskStatus.INITIATED):
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    leader = Task(
        name="decode",
        logger=logging.getLogger("decode"),
        operator=_Op(),
        status=leader_status,
    )
    follower = Task(
        name="prefill",
        logger=logging.getLogger("prefill"),
        operator=_Op(),
        status=follower_status,
    )
    leader.merge_members = ["decode", "prefill"]
    leader.merge_group_id = "k8s:node-0"
    follower.merge_leader = "decode"
    follower.merge_group_id = "k8s:node-0"
    tg.dag.add_node("decode", leader)
    tg.dag.add_node("prefill", follower)
    summary = _RecordingSummary()
    orch = Orchestrator(
        workflow=wf, poll_interval=0.01, fail_fast=True, execution_summary=summary
    )
    return orch, leader, follower, summary


class _GateOp(_Op):
    def __init__(self):
        super().__init__()
        self.opened: list[str] = []
        self.rc = True  # exec success

    async def open_merge_gate(self, dep_name: str) -> bool:
        self.opened.append(dep_name)
        return self.rc


def _mk_gated(dep_status=TaskStatus.READY, member_status=TaskStatus.RUNNING):
    """Leader 'lead' hosts members 'dep' and 'gated'; 'gated' waits on 'dep'."""
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    op = _GateOp()
    lead = Task(name="lead", logger=logging.getLogger("lead"), operator=op,
                status=TaskStatus.RUNNING)
    dep = Task(name="dep", logger=logging.getLogger("dep"), operator=_Op(),
               status=dep_status)
    gated = Task(name="gated", logger=logging.getLogger("gated"), operator=_Op(),
                 status=member_status)
    lead.merge_members = ["lead", "dep", "gated"]
    dep.merge_leader = "lead"
    gated.merge_leader = "lead"
    gated.merge_gate_after = ["dep"]
    for name, t in (("lead", lead), ("dep", dep), ("gated", gated)):
        tg.dag.add_node(name, t)
    orch = Orchestrator(workflow=wf, poll_interval=0.01, fail_fast=True)
    return orch, op, dep, gated


def test_signal_opens_gate_when_dependency_ready():
    orch, op, _dep, _gated = _mk_gated(dep_status=TaskStatus.READY)
    asyncio.run(orch._signal_merge_gates())
    assert op.opened == ["dep"]


def test_signal_opens_gate_when_dependency_completed():
    orch, op, _dep, _gated = _mk_gated(dep_status=TaskStatus.COMPLETED)
    asyncio.run(orch._signal_merge_gates())
    assert op.opened == ["dep"]


def test_signal_skips_when_dependency_not_met():
    orch, op, _dep, _gated = _mk_gated(dep_status=TaskStatus.RUNNING)
    asyncio.run(orch._signal_merge_gates())
    assert op.opened == []


def test_signal_is_idempotent_after_success():
    orch, op, _dep, _gated = _mk_gated(dep_status=TaskStatus.READY)
    asyncio.run(orch._signal_merge_gates())
    asyncio.run(orch._signal_merge_gates())
    assert op.opened == ["dep"]  # touched once, not re-touched


def test_signal_retries_after_exec_failure():
    orch, op, _dep, _gated = _mk_gated(dep_status=TaskStatus.READY)
    op.rc = False  # exec fails first time
    asyncio.run(orch._signal_merge_gates())
    op.rc = True   # succeeds on retry
    asyncio.run(orch._signal_merge_gates())
    assert op.opened == ["dep", "dep"]  # retried, then marked done


def test_signal_noop_when_operator_lacks_open_merge_gate():
    # Leader operator without open_merge_gate (plain _Op) -> skipped, no crash.
    orch, _leader, _follower, _summary = _mk()
    asyncio.run(orch._signal_merge_gates())  # must not raise


def test_signal_gate_reopens_after_leader_retry_recreates_pod():
    # A merged pod that fails and retries is recreated fresh (empty
    # /tmp/sflow-merge-gate), so a gate opened for the old instance must be forgotten
    # and re-touched on the new one -- else a READY-gated member blocks forever.
    orch, op, dep, gated = _mk_gated(dep_status=TaskStatus.READY)
    lead = orch.workflow.get_task("lead")

    # Pod instance 1: dependency READY -> gate opened once, recorded.
    asyncio.run(orch._signal_merge_gates())
    assert op.opened == ["dep"]
    assert ("gated", "dep") in orch._merge_gates_opened

    # Leader fails with retries left -> INITIATED + propagate = pod recreation.
    lead.status = TaskStatus.INITIATED
    asyncio.run(orch._propagate_merge_leader_status(lead))
    assert ("gated", "dep") not in orch._merge_gates_opened  # forgotten for new pod

    # Pod instance 2: leader RUNNING again, member re-promoted, dep READY again ->
    # the gate re-touches the fresh pod's marker.
    lead.status = TaskStatus.RUNNING
    gated.status = TaskStatus.RUNNING
    dep.status = TaskStatus.READY
    asyncio.run(orch._signal_merge_gates())
    assert op.opened == ["dep", "dep"]  # touched again on the recreated pod


def test_propagate_initiated_preserves_other_groups_opened_gates():
    # Retrying one group must not forget gates opened for a DIFFERENT merge group.
    orch, _op, _dep, _gated = _mk_gated(dep_status=TaskStatus.READY)
    lead = orch.workflow.get_task("lead")
    orch._merge_gates_opened.add(("gated", "dep"))       # this group
    orch._merge_gates_opened.add(("other_member", "other_dep"))  # a different group

    lead.status = TaskStatus.INITIATED
    asyncio.run(orch._propagate_merge_leader_status(lead))

    assert ("gated", "dep") not in orch._merge_gates_opened          # cleared
    assert ("other_member", "other_dep") in orch._merge_gates_opened  # preserved


def test_promote_merge_followers_sets_running_and_records_submitted():
    orch, leader, follower, summary = _mk()
    asyncio.run(orch._promote_merge_followers(leader))
    assert follower.status == TaskStatus.RUNNING
    assert "prefill" in summary.submitted


def test_promote_skips_follower_not_initiated():
    orch, leader, follower, _ = _mk(follower_status=TaskStatus.READY)
    asyncio.run(orch._promote_merge_followers(leader))
    assert follower.status == TaskStatus.READY  # already started; untouched


def test_propagate_completed_finalizes_followers():
    orch, leader, follower, _ = _mk(follower_status=TaskStatus.RUNNING)
    leader.status = TaskStatus.COMPLETED
    asyncio.run(orch._propagate_merge_leader_status(leader))
    assert follower.status == TaskStatus.COMPLETED
    assert follower.exit_code == 0


def test_propagate_failed_fails_followers_with_leader_exit_code():
    orch, leader, follower, summary = _mk(follower_status=TaskStatus.RUNNING)
    leader.status = TaskStatus.FAILED
    leader.exit_code = 3
    asyncio.run(orch._propagate_merge_leader_status(leader))
    assert follower.status == TaskStatus.FAILED
    assert follower.exit_code == 3
    assert "prefill" in summary.failed


def test_propagate_cancelled_cancels_followers():
    orch, leader, follower, summary = _mk(follower_status=TaskStatus.RUNNING)
    leader.status = TaskStatus.CANCELLED
    asyncio.run(orch._propagate_merge_leader_status(leader))
    assert follower.status == TaskStatus.CANCELLED
    assert "prefill" in summary.cancelled


def test_propagate_leaves_ready_follower_untouched():
    # A long-lived service that already signalled READY is terminal; leader
    # completion must not clobber it.
    orch, leader, follower, _ = _mk(follower_status=TaskStatus.READY)
    leader.status = TaskStatus.COMPLETED
    asyncio.run(orch._propagate_merge_leader_status(leader))
    assert follower.status == TaskStatus.READY


def test_propagate_retry_resets_followers_to_initiated():
    orch, leader, follower, _ = _mk(follower_status=TaskStatus.RUNNING)
    leader.status = TaskStatus.INITIATED  # leader scheduled for retry
    asyncio.run(orch._propagate_merge_leader_status(leader))
    assert follower.status == TaskStatus.INITIATED


def test_propagate_failed_overrides_ready_follower():
    # In a merged pod the follower's processes share the leader's single container;
    # if the leader FAILS, a follower that had reached READY is dead too and must
    # be failed (not left looking healthy).
    orch, leader, follower, summary = _mk(follower_status=TaskStatus.READY)
    leader.status = TaskStatus.FAILED
    leader.exit_code = 7
    asyncio.run(orch._propagate_merge_leader_status(leader))
    assert follower.status == TaskStatus.FAILED
    assert follower.exit_code == 7
    assert "prefill" in summary.failed


def test_propagate_cancelled_overrides_ready_follower():
    orch, leader, follower, summary = _mk(follower_status=TaskStatus.READY)
    leader.status = TaskStatus.CANCELLED
    asyncio.run(orch._propagate_merge_leader_status(leader))
    assert follower.status == TaskStatus.CANCELLED
    assert "prefill" in summary.cancelled


def test_propagate_retry_resets_ready_follower():
    # A retry recreates the shared pod, so even a READY follower must reset so it
    # re-promotes on the next attempt (otherwise it is orphaned as READY).
    orch, leader, follower, _ = _mk(follower_status=TaskStatus.READY)
    leader.status = TaskStatus.INITIATED
    asyncio.run(orch._propagate_merge_leader_status(leader))
    assert follower.status == TaskStatus.INITIATED


def test_propagate_failed_leaves_already_failed_follower_unchanged():
    # A follower that already failed on its own is not re-failed (no double-record).
    orch, leader, follower, summary = _mk(follower_status=TaskStatus.FAILED)
    leader.status = TaskStatus.FAILED
    asyncio.run(orch._propagate_merge_leader_status(leader))
    assert follower.status == TaskStatus.FAILED
    assert "prefill" not in summary.failed


# --- one-shot member finishing inside a still-running merged pod -------------
# A merge pod's container blocks in `wait` while any member is a long-lived
# service, so a finished one-shot member (e.g. the workflow's TERMINAL task) is
# invisible in the pod phase and used to hang the run forever. Each member now
# echoes [[sflow-member-done:<rc>]] into its own <task>.log; the orchestrator
# resolves the member from it while the leader keeps owning the pod.


class _DoneOp(_Op):
    """Operator exposing the k8s duck-typed member-done reader."""

    def __init__(self, rc_by_task=None):
        super().__init__()
        self._rc = rc_by_task or {}

    def merged_member_exit_code(self, task):
        return self._rc.get(task.name)


def _mk_done(rc, follower_status=TaskStatus.RUNNING):
    orch, leader, follower, summary = _mk(follower_status=follower_status)
    leader.status = TaskStatus.READY  # service leader: pod stays alive
    follower.operator = _DoneOp({follower.name: rc})
    return orch, leader, follower, summary


def test_finished_merge_follower_completes_while_leader_still_runs():
    orch, leader, follower, _ = _mk_done(0)
    asyncio.run(orch._resolve_finished_merge_members())
    assert follower.status == TaskStatus.COMPLETED
    assert follower.exit_code == 0
    assert leader.status == TaskStatus.READY  # leader untouched


def test_finished_merge_follower_fails_on_nonzero_rc():
    orch, leader, follower, summary = _mk_done(7)
    asyncio.run(orch._resolve_finished_merge_members())
    assert follower.status == TaskStatus.FAILED
    assert follower.exit_code == 7
    assert "prefill" in summary.failed


def test_merge_follower_untouched_until_marker_appears():
    orch, _leader, follower, _ = _mk_done(None)  # no marker yet
    asyncio.run(orch._resolve_finished_merge_members())
    assert follower.status == TaskStatus.RUNNING


def test_ready_service_follower_is_not_resolved_without_marker():
    orch, _leader, follower, _ = _mk_done(None, follower_status=TaskStatus.READY)
    asyncio.run(orch._resolve_finished_merge_members())
    assert follower.status == TaskStatus.READY


def test_resolve_noop_when_operator_lacks_reader():
    orch, _leader, follower, _ = _mk(follower_status=TaskStatus.RUNNING)
    asyncio.run(orch._resolve_finished_merge_members())  # plain _Op
    assert follower.status == TaskStatus.RUNNING


def test_resolve_survives_reader_error():
    orch, _leader, follower, _ = _mk_done(0)

    def _boom(_task):
        raise OSError("log vanished")

    follower.operator.merged_member_exit_code = _boom
    asyncio.run(orch._resolve_finished_merge_members())
    assert follower.status == TaskStatus.RUNNING


# --- pod lifetime is decoupled from every member's status -------------------
# The leader's execute() owns the shared pod (it deletes it on return), so tying
# the LEADER's status to pod-terminal deadlocked it too: it waited on a pod its
# own siblings kept alive. Every member -- leader included -- now resolves from
# its own done-marker, and the pod is reclaimed later by the driver's teardown.


def test_merge_leader_resolves_from_its_own_marker_while_pod_lives():
    orch, leader, follower, _ = _mk(follower_status=TaskStatus.READY)
    leader.operator = _DoneOp({leader.name: 0})
    asyncio.run(orch._resolve_finished_merge_members())
    assert leader.status == TaskStatus.COMPLETED
    assert leader.exit_code == 0
    # The sibling service is untouched: the pod must stay up for it.
    assert follower.status == TaskStatus.READY


def test_one_shot_leader_and_service_follower_lets_dag_finish():
    # The 'benchmark leads, server follows' shape: the leader is one-shot, its
    # sibling serves. Leader completing is what lets the DAG reach terminal.
    orch, leader, follower, _ = _mk(follower_status=TaskStatus.READY)
    leader.operator = _DoneOp({leader.name: 0})
    asyncio.run(orch._resolve_finished_merge_members())
    assert leader.status.is_terminal()


def test_resolving_a_member_is_idempotent_across_ticks():
    # The poll loop re-runs every tick and the marker stays in the log, so a
    # resolved member must not be re-finalized (double uploads/result parsing).
    orch, _leader, follower, summary = _mk(follower_status=TaskStatus.RUNNING)
    follower.operator = _DoneOp({follower.name: 3})
    asyncio.run(orch._resolve_finished_merge_members())
    assert follower.status == TaskStatus.FAILED
    asyncio.run(orch._resolve_finished_merge_members())  # second tick
    assert summary.failed == ["prefill"]  # recorded exactly once
