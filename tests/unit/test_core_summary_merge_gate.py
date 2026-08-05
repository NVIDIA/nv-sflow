# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge-pod gated members must not read as running from submission.

Every member of a merged pod is launched when the shared pod starts, but a member
with in-group dependencies blocks in its in-pod gate until they are met. Reporting
only the submission made a real run's `client` look like it ran for 39.8s when it
actually worked for ~5s -- and made it the longest bar in the duration chart.
"""

import logging
import time

from sflow.core.execution_summary import SflowSummaryWriter
from sflow.core.task import Task, TaskStatus
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig


def _task(name, *, gate_after=None):
    t = Task(
        name=name,
        logger=logging.getLogger(f"sflow.task.{name}"),
        operator=BashOperator(BashOperatorConfig(name="bash")),
        script=[f"echo {name}"],
    )
    t.backend_name = "local"
    t.operator_name = "bash"
    t.status = TaskStatus.RUNNING
    if gate_after:
        t.merge_gate_after = list(gate_after)
    return t


def _writer(tmp_path):
    return SflowSummaryWriter(tmp_path / "summary.log")


def _events(w, name):
    return [(e.event, e.details) for e in w._timeline if e.task_name == name]


def test_submitted_row_says_the_member_is_gated(tmp_path):
    w = _writer(tmp_path)
    w.task_submitted(_task("client", gate_after=["service_a", "service_b"]))
    (event, details), = _events(w, "client")
    assert event == "SUBMITTED"
    assert "gated_on" in " ".join(f"{k}={v}" for k, v in details.items()), details


def test_ungated_event_marks_the_real_start(tmp_path):
    w = _writer(tmp_path)
    client = _task("client", gate_after=["service_a"])
    w.task_submitted(client)
    w.task_gate_opened(client)
    assert [e for e, _ in _events(w, "client")] == ["SUBMITTED", "UNGATED"]
    assert "client" in w._task_ungated


def test_duration_measures_work_not_wait(tmp_path):
    """The headline fix: a gated member's bar starts when its gate opened."""
    w = _writer(tmp_path)
    client = _task("client", gate_after=["service_a"])
    w.task_submitted(client)
    w._task_started["client"] = time.monotonic() - 40.0   # submitted 40s ago
    w.task_gate_opened(client)
    w._task_ungated["client"] = time.monotonic() - 5.0    # gate opened 5s ago
    client.status = TaskStatus.COMPLETED
    w.task_completed(client)

    line = [x for x in w._duration_chart_lines([client]) if "client" in x][0]
    secs = float(line.split("|")[-1].strip().split("s")[0])
    assert 4.0 < secs < 6.0, f"expected ~5s of work, chart said {secs}s: {line}"


def test_ungated_member_is_unaffected(tmp_path):
    """A normal (non-merge) task keeps submission-to-finish duration."""
    w = _writer(tmp_path)
    t = _task("solo")
    w.task_submitted(t)
    assert "gated_on" not in str(_events(w, "solo"))
    w._task_started["solo"] = time.monotonic() - 10.0
    t.status = TaskStatus.COMPLETED
    w.task_completed(t)
    line = [x for x in w._duration_chart_lines([t]) if "solo" in x][0]
    secs = float(line.split("|")[-1].strip().split("s")[0])
    assert 9.0 < secs < 11.0, line


def test_rendered_rows_read_clearly(tmp_path):
    """The point of the fix is what a human reads, so assert the rendered text."""
    w = _writer(tmp_path)
    c = _task("client", gate_after=["service_a", "service_b"])
    w.task_submitted(c)
    w.task_gate_opened(c)
    rows = "\n".join(w._timeline_lines())
    assert "gated in shared pod, waiting on service_a,service_b" in rows
    assert "in-pod gate opened; work starts now" in rows


def test_plain_task_submitted_row_is_unchanged(tmp_path):
    """No new noise for the overwhelmingly common non-merge case."""
    w = _writer(tmp_path)
    w.task_submitted(_task("solo"))
    row = [x for x in w._timeline_lines() if "solo" in x][0]
    assert "gated" not in row and "attempt=" in row
