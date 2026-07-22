# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MonitorRegistry refcount/dedup + orchestrator task-monitor lifecycle."""

import asyncio
import logging

from sflow.core.command import Command
from sflow.core.monitor import MonitorConsumer, MonitorRegistry, NodeCollector
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.orchestrator import Orchestrator
from sflow.core.probe import Probe, ProbeType
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


class _RecordingLauncher:
    """Records launched / cancelled collector + task names; hangs unless told to complete."""

    def __init__(self, complete_names=None):
        self.complete_names = set(complete_names or [])
        self.launched: list[str] = []
        self.cancelled: list[str] = []

    async def run_async(self, command, output_logger=None, env=None, task_name=None, **kwargs):
        self.launched.append(task_name)
        if task_name in self.complete_names:
            await asyncio.sleep(0)
            return 0
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.append(task_name)
            return -1


class _FakeOperator(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="fake"))

    def build_command(self, *, task_name, script, envs) -> Command:
        return Command(exec="echo").add_arg("fake")


class _AlwaysReadyProbe(Probe):
    def __init__(self):
        super().__init__(type=ProbeType.READINESS, interval=0, timeout=10, success_threshold=1)

    async def check(self, task) -> bool:
        return True


def _collectors(*nodes, backend="b"):
    return {
        (backend, n): NodeCollector(
            key=(backend, n),
            name=f"sflow_monitor_{n}",
            command=Command(exec="echo").add_arg("x"),
        )
        for n in nodes
    }


def _registry(collectors, launcher, tmp_path):
    return MonitorRegistry(
        collectors,
        raw_dir=tmp_path / "sflow_monitor" / "raw",
        out_dir=tmp_path / "sflow_monitor",
        overview_path=tmp_path / "sflow_monitor.log",
        launcher=launcher,
    )


def test_registry_dedup_and_refcount(tmp_path):
    """Overlapping consumers on the same node share ONE collector (singleton)."""
    rec = _RecordingLauncher()
    reg = _registry(_collectors("n0", "n1"), rec, tmp_path)
    wf = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "n0"), ("b", "n1")])
    task = MonitorConsumer(owner="task:work", name="work", keys=[("b", "n0")])

    async def scenario():
        await reg.acquire(wf)
        await asyncio.sleep(0.02)  # let the scheduled collector tasks start
        assert sorted(rec.launched) == ["sflow_monitor_n0", "sflow_monitor_n1"]
        # Task reuses node0's collector -> no new launch.
        await reg.acquire(task)
        await asyncio.sleep(0.02)
        assert sorted(rec.launched) == ["sflow_monitor_n0", "sflow_monitor_n1"]
        # Releasing the task does NOT stop node0 (still held by workflow).
        await reg.release(task)
        assert rec.cancelled == []
        # Releasing the workflow stops both collectors.
        await reg.release(wf)
        assert sorted(rec.cancelled) == ["sflow_monitor_n0", "sflow_monitor_n1"]

    asyncio.run(scenario())


def test_registry_shutdown_cancels_all(tmp_path):
    rec = _RecordingLauncher()
    reg = _registry(_collectors("n0", "n1"), rec, tmp_path)
    wf = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "n0"), ("b", "n1")])

    async def scenario():
        await reg.acquire(wf)
        await asyncio.sleep(0.02)  # let the scheduled collector tasks start
        await reg.shutdown()
        assert sorted(rec.cancelled) == ["sflow_monitor_n0", "sflow_monitor_n1"]

    asyncio.run(scenario())


def test_orchestrator_releases_task_monitor_on_completion(tmp_path):
    """A benchmark task's monitor is torn down when its process exits."""
    rec = _RecordingLauncher(complete_names={"work"})
    reg = _registry(_collectors("n0"), rec, tmp_path)

    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    work = Task(
        name="work",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.work"),
        monitor=MonitorConsumer(owner="task:work", name="work", keys=[("b", "n0")]),
    )
    tg.dag.add_node("work", work)

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0.01,
        launcher=rec,
        monitor_registry=reg,
    )
    asyncio.run(asyncio.wait_for(orch.run(), timeout=5))

    assert work.status == TaskStatus.COMPLETED
    assert rec.launched.count("sflow_monitor_n0") == 1
    assert "sflow_monitor_n0" in rec.cancelled


def test_orchestrator_keeps_ready_service_monitor_until_teardown(tmp_path):
    """A READY service keeps its monitor running until workflow teardown.

    The monitor is NOT released when the task transitions to READY (its process
    is still alive); it is only torn down in run()'s finally.
    """
    rec = _RecordingLauncher(complete_names={"bench"})
    reg = _registry(_collectors("n0"), rec, tmp_path)

    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    server = Task(
        name="server",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.server"),
        probes=[_AlwaysReadyProbe()],
        monitor=MonitorConsumer(owner="task:server", name="server", keys=[("b", "n0")]),
    )
    bench = Task(
        name="bench",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.bench"),
    )
    tg.dag.add_node("server", server)
    tg.dag.add_node("bench", bench)
    tg.dag.add_edge("server", "bench")

    orch = Orchestrator(
        workflow=wf,
        poll_interval=0.01,
        launcher=rec,
        monitor_registry=reg,
    )

    async def scenario():
        await asyncio.wait_for(orch.run(), timeout=5)
        # Clean up the still-hanging READY server workload process task.
        for proc_task in list(orch._subprocess_tasks.values()):
            proc_task.cancel()
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert server.status == TaskStatus.READY
    assert bench.status == TaskStatus.COMPLETED
    # Collector launched exactly once (no duplicate) and only torn down at teardown.
    assert rec.launched.count("sflow_monitor_n0") == 1
    assert rec.cancelled.count("sflow_monitor_n0") == 1


def test_orchestrator_records_task_events_for_timeline(tmp_path):
    """Task status changes are stamped on the registry as timeline markers and
    flow through report_spec() to the post-processor."""
    rec = _RecordingLauncher(complete_names={"work"})
    reg = _registry(_collectors("n0"), rec, tmp_path)

    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    work = Task(
        name="work",
        operator=_FakeOperator(),
        logger=logging.getLogger("sflow.task.work"),
        monitor=MonitorConsumer(owner="task:work", name="work", keys=[("b", "n0")]),
    )
    tg.dag.add_node("work", work)

    orch = Orchestrator(
        workflow=wf, poll_interval=0.01, launcher=rec, monitor_registry=reg
    )
    asyncio.run(asyncio.wait_for(orch.run(), timeout=5))

    events = reg.task_events
    pairs = {(e["task"], e["event"]) for e in events}
    assert ("work", "submit") in pairs
    assert ("work", "done") in pairs
    assert all(isinstance(e["ts"], float) for e in events)
    # report_spec carries the events through to the post-processor.
    assert reg.report_spec()["task_events"] == events


def test_record_task_event_noop_path_without_registry(tmp_path):
    """_record_summary must not blow up (or record) when monitoring is disabled."""
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    orch = Orchestrator(workflow=wf, poll_interval=0.01, launcher=_RecordingLauncher())
    # No monitor_registry -> the monitor-event hook is a silent no-op.
    orch._record_summary(
        "task_submitted",
        Task(name="x", operator=_FakeOperator(), logger=logging.getLogger("t")),
    )  # must not raise
