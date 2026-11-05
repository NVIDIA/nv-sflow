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


class _CommandRecordingLauncher(_RecordingLauncher):
    """Also records the actual commands, so teardown can be asserted."""

    def __init__(self, complete_names=None):
        super().__init__(complete_names)
        self.commands: list[str] = []

    async def run_async(self, command, output_logger=None, env=None, task_name=None, **kwargs):
        self.commands.append(command.as_str() if hasattr(command, "as_str") else str(command))
        # Teardown commands are short-lived by nature; only the collector itself
        # is long-running, so don't apply the base class's hang here.
        if task_name and task_name.endswith("_stop"):
            await asyncio.sleep(0)
            return 0
        return await super().run_async(
            command, output_logger=output_logger, env=env, task_name=task_name, **kwargs
        )


def test_stopping_a_collector_runs_its_stop_command(tmp_path):
    """Cancelling the launch is not enough for every transport.

    A container collector's `docker run` is only a CLIENT: killing it leaves the
    container running on the daemon, sampling forever. Transports that need an
    explicit teardown carry one, and the registry runs it on the last release.
    """
    collectors = _collectors("n0")
    collectors[("b", "n0")].stop_command = Command(exec="docker").add_arg(
        "rm"
    ).add_arg("-f").add_arg("sflow-monitor-n0")
    launcher = _CommandRecordingLauncher()
    registry = _registry(collectors, launcher, tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "n0")])

    async def _go():
        await registry.acquire(consumer)
        # Let the collector task actually start before tearing it down.
        await asyncio.sleep(0)
        await registry.release(consumer)

    asyncio.run(_go())

    assert "sflow_monitor_n0" in launcher.cancelled
    assert any("docker rm -f sflow-monitor-n0" in c for c in launcher.commands), (
        launcher.commands
    )


def test_a_collector_without_a_stop_command_runs_nothing_extra(tmp_path):
    """An in-place collector dies with its process; a spurious teardown command
    would just be noise."""
    launcher = _CommandRecordingLauncher()
    registry = _registry(_collectors("n0"), launcher, tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "n0")])

    async def _go():
        await registry.acquire(consumer)
        await asyncio.sleep(0)
        await registry.release(consumer)

    asyncio.run(_go())

    assert len(launcher.commands) == 1


def test_a_hanging_stop_command_cannot_wedge_shutdown(tmp_path, monkeypatch):
    """Teardown can cross the network (`docker rm -f` against an unreachable
    host). Awaiting it unbounded hung the whole workflow's shutdown with no
    output -- so it is capped, and a stuck teardown merely delays the exit.
    """
    monkeypatch.setattr("sflow.core.monitor._COLLECTOR_STOP_TIMEOUT_S", 0.05)

    class _HangingStopLauncher(_RecordingLauncher):
        async def run_async(self, command, output_logger=None, env=None, task_name=None, **kwargs):
            if task_name and task_name.endswith("_stop"):
                await asyncio.sleep(3600)
                return 0
            return await super().run_async(
                command, output_logger=output_logger, env=env,
                task_name=task_name, **kwargs
            )

    collectors = _collectors("n0")
    collectors[("b", "n0")].stop_command = Command(exec="docker").add_arg("rm")
    registry = _registry(collectors, _HangingStopLauncher(), tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "n0")])

    async def _go():
        await registry.acquire(consumer)
        await asyncio.sleep(0)
        await registry.release(consumer)

    # The point is that this returns at all.
    asyncio.run(asyncio.wait_for(_go(), timeout=10))


def test_a_collector_that_cannot_start_is_reported(tmp_path, caplog):
    """A container collector has several likely ways to fail (no python3 in the
    image, no NVIDIA runtime, unreachable daemon). The collector's own output
    goes to a per-task logger at INFO, so without this the run finished
    'successfully' with an empty monitor report and nothing above INFO anywhere.
    """

    class _FailingLauncher:
        async def run_async(self, command, output_logger=None, env=None, task_name=None, **kwargs):
            return 127  # e.g. docker: exec: "python3": not found

    registry = _registry(_collectors("n0"), _FailingLauncher(), tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "n0")])

    async def _go():
        await registry.acquire(consumer)
        await asyncio.sleep(0)
        await registry.release(consumer)

    with caplog.at_level(logging.WARNING):
        asyncio.run(_go())

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("exited early" in m and "n0" in m for m in warnings), warnings


def test_a_normally_released_collector_does_not_warn(tmp_path, caplog):
    """Cancellation at the last release is the ordinary path, not a failure."""
    registry = _registry(_collectors("n0"), _RecordingLauncher(), tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "n0")])

    async def _go():
        await registry.acquire(consumer)
        await asyncio.sleep(0)
        await registry.release(consumer)

    with caplog.at_level(logging.WARNING):
        asyncio.run(_go())

    assert not [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING and "exited early" in r.getMessage()
    ]


def test_collectors_are_stopped_concurrently(tmp_path):
    """Teardown can cross the network per node (`docker rm -f` over ssh).

    Serially that cost the SUM of the pool -- measured at ~12s per node of dead
    time after the workflow had already finished, so ~90s on an eight-host pool.
    """
    stop_overlap = {"now": 0, "peak": 0}

    class _SlowStopLauncher(_RecordingLauncher):
        async def run_async(self, command, output_logger=None, env=None, task_name=None, **kwargs):
            if task_name and task_name.endswith("_stop"):
                stop_overlap["now"] += 1
                stop_overlap["peak"] = max(stop_overlap["peak"], stop_overlap["now"])
                await asyncio.sleep(0.15)
                stop_overlap["now"] -= 1
                return 0
            return await super().run_async(
                command, output_logger=output_logger, env=env,
                task_name=task_name, **kwargs
            )

    collectors = _collectors("n0", "n1", "n2")
    for key in collectors:
        collectors[key].stop_command = Command(exec="docker").add_arg("rm")
    registry = _registry(collectors, _SlowStopLauncher(), tmp_path)
    consumer = MonitorConsumer(
        owner="workflow", name="workflow",
        keys=[("b", "n0"), ("b", "n1"), ("b", "n2")],
    )

    async def _go():
        await registry.acquire(consumer)
        await asyncio.sleep(0)
        await registry.release(consumer)

    asyncio.run(_go())

    assert stop_overlap["peak"] == 3, (
        f"expected all 3 teardowns in flight at once, peak was {stop_overlap['peak']}"
    )


class _StreamingLauncher:
    """A launcher that feeds rows to the sink, the way a real collector would."""

    def __init__(self, rows):
        self._rows = rows
        self.sinks: list[object] = []
        self.commands: list[object] = []
        self.envs: list[dict] = []

    async def run_async(
        self, command, *, output_logger=None, env=None, task_name=None, stdout_sink=None
    ):
        self.commands.append(command)
        self.envs.append(env or {})
        # A teardown command is a short-lived client that exits on its own; only
        # the collector itself runs until cancelled. Modelling both as "sleep
        # forever" would make every test here wait out the real 30s teardown
        # bound in _stop_collector.
        if task_name and task_name.endswith("_stop"):
            return 0
        self.sinks.append(stdout_sink)
        if stdout_sink is not None:
            for row in self._rows:
                stdout_sink(row)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            return -1


def test_a_streaming_collector_is_demultiplexed_and_closed_by_the_registry(tmp_path):
    """The seam between _start_collector and StreamedSampleWriter.

    Both halves were covered on their own -- the writer directly, the plan-time
    ``stream_node_id`` as a value -- while nothing drove one through the other.
    A collector that streamed but whose sink was never wired would have passed
    both and produced an empty report.
    """
    launcher = _StreamingLauncher(
        ["cpu\t1.0,5.0", "gpu\t#fields=index,power.draw", "cpu\t2.0,6.0"]
    )
    collectors = {
        ("b", "dgx-a"): NodeCollector(
            key=("b", "dgx-a"),
            name="sflow_monitor_dgx-a",
            command=Command(exec="docker").add_arg("run"),
            envs={"SFLOW_TASK_OUTPUT_DIR": "/x"},
            stream_node_id="3",
        )
    }
    reg = _registry(collectors, launcher, tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "dgx-a")])

    async def scenario():
        await reg.acquire(consumer)
        await asyncio.sleep(0.05)
        await reg.release(consumer)

    asyncio.run(scenario())

    assert launcher.sinks[0] is not None, "streaming collector got no sink"
    # Filed under the PLAN-TIME node name and the collector's node id.
    raw = tmp_path / "sflow_monitor" / "raw"
    assert (raw / "cpu_monitor_node_3_dgx-a.log").read_text().splitlines() == [
        "1.0,5.0",
        "2.0,6.0",
    ]
    assert (raw / "gpu_monitor_node_3_dgx-a.log").read_text().splitlines() == [
        "#fields=index,power.draw"
    ]


def test_a_non_streaming_collector_gets_no_sink(tmp_path):
    """Slurm/local collectors write their own files; a sink would double-write."""
    launcher = _StreamingLauncher([])
    reg = _registry(_collectors("n0"), launcher, tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "n0")])

    async def scenario():
        await reg.acquire(consumer)
        await asyncio.sleep(0.02)
        await reg.release(consumer)

    asyncio.run(scenario())

    assert launcher.sinks == [None]


def test_a_stop_command_runs_after_the_collector_is_cancelled(tmp_path):
    """Killing a `docker run` client leaves the container sampling on the daemon."""
    launcher = _StreamingLauncher([])
    collectors = {
        ("b", "dgx-a"): NodeCollector(
            key=("b", "dgx-a"),
            name="sflow_monitor_dgx-a",
            command=Command(exec="docker").add_arg("run"),
            stream_node_id="0",
            stop_command=Command(exec="docker").add_arg("rm").add_arg("-f"),
        )
    }
    reg = _registry(collectors, launcher, tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "dgx-a")])

    async def scenario():
        await reg.acquire(consumer)
        await asyncio.sleep(0.02)
        await reg.release(consumer)

    asyncio.run(scenario())

    execs = [c.as_list() for c in launcher.commands]
    assert ["docker", "rm", "-f"] in execs, execs


class _DyingLauncher:
    """A collector that fails on its own instead of being released."""

    def __init__(self, *, exc=None, code=0):
        self._exc = exc
        self._code = code

    async def run_async(
        self, command, *, output_logger=None, env=None, task_name=None, stdout_sink=None
    ):
        await asyncio.sleep(0)
        if self._exc is not None:
            raise self._exc
        return self._code


def _run_until_collector_finishes(reg, consumer):
    async def scenario():
        await reg.acquire(consumer)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())


def test_a_collector_that_raises_is_reported_not_swallowed(tmp_path, caplog):
    """Monitoring is passive and must not fail the run -- but not silently.

    A container that cannot start (no python3 in the image, no NVIDIA runtime,
    unreachable daemon) otherwise leaves nothing above INFO anywhere, and the
    workflow finishes "successfully" with an empty monitor report.
    """
    reg = _registry(
        _collectors("dgx-a"), _DyingLauncher(exc=RuntimeError("no such image")), tmp_path
    )
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "dgx-a")])

    with caplog.at_level("WARNING"):
        _run_until_collector_finishes(reg, consumer)

    assert "dgx-a" in caplog.text
    assert "no such image" in caplog.text


def test_a_collector_exiting_nonzero_is_reported(tmp_path, caplog):
    reg = _registry(_collectors("dgx-a"), _DyingLauncher(code=2), tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "dgx-a")])

    with caplog.at_level("WARNING"):
        _run_until_collector_finishes(reg, consumer)

    assert "exited early" in caplog.text
    assert "dgx-a" in caplog.text


def test_normal_teardown_is_not_reported_as_a_failure(tmp_path, caplog):
    """Cancellation is the ordinary path, and a signal-killed collector reports a
    NEGATIVE code -- neither is a fault worth warning about."""
    reg = _registry(_collectors("n0"), _RecordingLauncher(), tmp_path)
    consumer = MonitorConsumer(owner="workflow", name="workflow", keys=[("b", "n0")])

    async def scenario():
        await reg.acquire(consumer)
        await asyncio.sleep(0.02)
        await reg.release(consumer)
        await asyncio.sleep(0.02)

    with caplog.at_level("WARNING"):
        asyncio.run(scenario())

    assert "exited early" not in caplog.text
    assert "failed" not in caplog.text
