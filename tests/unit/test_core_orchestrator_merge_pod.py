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
