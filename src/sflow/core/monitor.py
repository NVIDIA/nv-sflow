# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime objects for the ``monitor:`` feature.

The monitor SCHEDULE is computed at plan time (see ``sflow.app.monitor_planner``):
the set of per-node collectors, their scope unions and commands, and one
``MonitorConsumer`` per logical monitor (the workflow monitor and each task
monitor). At RUN time the orchestrator / app merely fire the pre-computed
triggers by calling ``MonitorRegistry.acquire`` / ``release``.

Singleton semantics: there is at most one collector per ``(backend, node)``.
A collector starts when its first consumer acquires it and is torn down when its
last consumer releases it. Logical monitors are views over the shared per-node
data; GPU/scope subsets and time windows are applied at report time only.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sflow.logging import get_logger

from .launcher import SubprocessLauncher

if TYPE_CHECKING:
    from .command import Command

_logger = get_logger(__name__)

CollectorKey = tuple[str, str]  # (backend_name, node_name)

# Scope tags a streaming collector may prefix its rows with. Mirrors
# hardware_monitor.ALL_SCOPES, kept as a literal here so this module does not
# import the standalone collector script.
MONITOR_STREAM_SCOPES = frozenset({"cpu", "gpu", "memory", "disk", "network"})

# Ceiling on a collector's teardown command (see MonitorDelivery.stop_argv).
# Generous enough for a `docker rm -f` over ssh, short enough that an
# unreachable host delays shutdown rather than hanging it.
_COLLECTOR_STOP_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class MonitorDelivery:
    """HOW a backend puts the collector on a node and gets its samples back.

    Defaults describe a node that shares the driver's filesystem (Slurm, local,
    single-machine Docker): reference the materialized script by path and let it
    write its own files.
    """

    # Tokens before ``python3`` -- e.g. a ``docker run`` that lands the collector
    # on another machine. Empty means run it right here.
    launch_prefix: tuple[str, ...] = ()
    # The node shares no filesystem with the driver. One flag, because the two
    # consequences are inseparable: the source must go in on stdin (redirected
    # by the DRIVER's shell, so nothing is copied to the node and a `context:`
    # host with no ssh endpoint still works) and the samples must come back on
    # stdout. A delivery that did one without the other could not work.
    offhost: bool = False
    # Whether the collector will be able to see GPUs where it lands. False when
    # a container is deliberately started without them, so the gpu scope is
    # dropped rather than filling a log with "nvidia-smi not found" rows.
    # Host deliveries leave this True: nvidia-smi is simply there, or it is not.
    provides_gpus: bool = True
    # Teardown for transports where killing the launched process is not enough.
    # A `docker run` is only a client: kill it and the container keeps sampling
    # on the daemon. Empty when the collector dies with its own process.
    stop_argv: tuple[str, ...] = ()


class StreamedSampleWriter:
    """Reassembles a streaming collector's stdout into raw sample files.

    Collectors on nodes the driver shares no filesystem with stream
    ``<scope>\\t<row>`` instead of writing files themselves. This writes the very
    layout ``hardware_monitor.py`` would have written locally, so the
    post-processor is unaware of which transport delivered the samples.

    ``node_name`` is the PLAN-TIME node name, not the machine's own hostname:
    the post-processor matches that field against a consumer's node list, and a
    mismatch makes it silently widen a per-task report to every node.
    """

    def __init__(self, raw_dir: Path, *, node_id: str, node_name: str) -> None:
        self._raw_dir = Path(raw_dir)
        self._name = f"{{}}_monitor_node_{node_id}_{node_name}.log"
        self._handles: dict[str, Any] = {}

    def __call__(self, line: str) -> None:
        scope, separator, payload = line.partition("\t")
        # Startup banners and any other untagged chatter are not samples.
        if not separator or scope not in MONITOR_STREAM_SCOPES:
            return
        handle = self._handles.get(scope)
        if handle is None:
            self._raw_dir.mkdir(parents=True, exist_ok=True)
            handle = (self._raw_dir / self._name.format(scope)).open("a", buffering=1)
            self._handles[scope] = handle
        handle.write(payload + "\n")
        handle.flush()

    def close(self) -> None:
        for handle in self._handles.values():
            with contextlib.suppress(Exception):
                handle.close()
        self._handles.clear()


@dataclass
class MonitorConsumer:
    """A resolved logical monitor (the workflow monitor or one task's monitor).

    A consumer references the per-node collectors (``keys``) it needs and carries
    the reporting filters (``nodes`` / ``gpus`` / ``scopes``) applied post-run.
    ``start_ts`` / ``end_ts`` are stamped at runtime by the registry and define
    the consumer's reporting time window.
    """

    owner: str  # "workflow" or "task:<name>"
    name: str  # display name: "workflow" or the task name
    keys: list[CollectorKey] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    gpus: list[int] | None = None
    scopes: list[str] = field(default_factory=list)
    report: bool = False
    report_formats: list[str] = field(default_factory=lambda: ["csv", "svg"])
    start_ts: float | None = None
    end_ts: float | None = None

    def to_spec(self) -> dict[str, Any]:
        """Return the dict consumed by the post-processor report spec."""
        return {
            "name": self.name,
            "owner": self.owner,
            "nodes": list(self.nodes),
            "gpus": list(self.gpus) if self.gpus is not None else None,
            "scopes": list(self.scopes),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "report": bool(self.report),
            "formats": list(self.report_formats),
        }


@dataclass
class TaskResourceView:
    """A per-task (or per-replica) hardware report derived from a monitor.

    The resources (``nodes`` / ``gpus``) are a task's assignment, resolved at plan
    time. The reporting time window is resolved at post-process time from the
    lifecycle events of ``window_tasks`` -- the monitored task itself for a natural
    view, or the triggering owner for a ``used_by_tasks`` cross view (so a cross
    view shows the target's resources over the *owner's* run).
    """

    label: str  # output folder name (e.g. "server_0" or "kv__monitored_by__bench")
    task: str  # config task the sampled resources belong to
    triggered_by: str  # "workflow" or "task:<owner>"
    title: str  # human-readable chart title
    nodes: list[str] = field(default_factory=list)
    gpus: list[int] | None = None
    scopes: list[str] = field(default_factory=list)
    report_formats: list[str] = field(default_factory=lambda: ["csv", "svg"])
    window_tasks: list[str] = field(default_factory=list)
    log_window: dict[str, Any] | None = None
    # Every monitor owner that asked for this folder. Views are deduped by label,
    # so a task covered by both the workflow monitor and its own monitor keeps the
    # FIRST `triggered_by` -- counting folders by that alone credits them all to
    # the workflow and reports the task monitor as writing nothing. Plan-time only
    # (not in `to_spec`).
    contributors: list[str] = field(default_factory=list)
    cross: bool = False  # True for a used_by_tasks (B monitored-by A) view

    def to_spec(self) -> dict[str, Any]:
        """Return the dict consumed by the post-processor (``name`` = folder)."""
        return {
            "name": self.label,
            "label": self.label,
            "task": self.task,
            "triggered_by": self.triggered_by,
            "title": self.title,
            "nodes": list(self.nodes),
            "gpus": list(self.gpus) if self.gpus is not None else None,
            "scopes": list(self.scopes),
            "formats": list(self.report_formats),
            "window_tasks": list(self.window_tasks),
            "log_window": self.log_window,
            "cross": bool(self.cross),
        }


@dataclass
class NodeCollector:
    """One collector process for a single ``(backend, node)``."""

    key: CollectorKey
    name: str
    command: "Command"
    envs: dict[str, str] = field(default_factory=dict)
    # Set when this collector streams tagged samples on stdout because its node
    # shares no filesystem with the driver. Carries the node id the driver files
    # the demultiplexed rows under; None means the collector writes its own files.
    stream_node_id: str | None = None
    # Teardown run after the launched process is cancelled, for transports the
    # cancel alone does not stop (see MonitorDelivery.stop_argv).
    stop_command: "Command | None" = None


class MonitorRegistry:
    """Refcounted, singleton-per-node collector manager.

    Created at plan time with the full set of possible collectors. At runtime it
    launches a collector on a node's first ``acquire`` and cancels it on the
    node's last ``release``. All methods are coroutine-safe under a single asyncio
    loop (the orchestrator's).
    """

    def __init__(
        self,
        collectors: dict[CollectorKey, NodeCollector],
        *,
        raw_dir: Path,
        out_dir: Path,
        overview_path: Path,
        workflow_name: str = "",
        interval_ms: int | None = None,
        launcher: SubprocessLauncher | None = None,
    ):
        self._collectors = collectors
        self._launcher = launcher or SubprocessLauncher()
        self._refcount: dict[CollectorKey, int] = {}
        self._running: dict[CollectorKey, asyncio.Task[Any]] = {}
        # Open demultiplexers for streaming collectors, by node.
        self._writers: dict[CollectorKey, StreamedSampleWriter] = {}
        self._lock = asyncio.Lock()
        self._consumers: list[MonitorConsumer] = []
        # Per-task / per-replica report views (resource-scoped), built at plan time.
        self._task_views: list[TaskResourceView] = []
        # Task lifecycle markers (submit/ready/done/fail/cancel) for timeline charts,
        # stamped by the orchestrator as transitions happen.
        self._task_events: list[dict[str, Any]] = []

        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.overview_path = Path(overview_path)
        self.workflow_name = workflow_name
        self.interval_ms = interval_ms

    @property
    def has_collectors(self) -> bool:
        return bool(self._collectors)

    @property
    def collector_count(self) -> int:
        return len(self._collectors)

    def register_consumer(self, consumer: MonitorConsumer) -> None:
        """Record a consumer so it appears in the post-run report spec."""
        self._consumers.append(consumer)

    @property
    def consumers(self) -> list[MonitorConsumer]:
        return list(self._consumers)

    def register_task_view(self, view: TaskResourceView) -> None:
        """Record a per-task report view for the post-run report spec."""
        self._task_views.append(view)

    @property
    def task_views(self) -> list[TaskResourceView]:
        return list(self._task_views)

    def record_task_event(self, ts: float, task: str, event: str) -> None:
        """Record a task status change (timeline marker). Cheap, sync, append-only."""
        self._task_events.append(
            {"ts": float(ts), "task": str(task), "event": str(event)}
        )

    @property
    def task_events(self) -> list[dict[str, Any]]:
        return list(self._task_events)

    async def acquire(self, consumer: MonitorConsumer | None) -> None:
        """Start (or reuse) the collectors a consumer needs; bump refcounts."""
        if consumer is None or not consumer.keys:
            return
        if consumer.start_ts is None:
            consumer.start_ts = time.time()
        async with self._lock:
            for key in consumer.keys:
                self._refcount[key] = self._refcount.get(key, 0) + 1
                if self._refcount[key] == 1:
                    self._start_collector(key)

    async def release(self, consumer: MonitorConsumer | None) -> None:
        """Drop a consumer's refcounts; cancel collectors that reach zero."""
        if consumer is None or not consumer.keys:
            return
        consumer.end_ts = time.time()
        async with self._lock:
            finished: list[CollectorKey] = []
            for key in consumer.keys:
                if key not in self._refcount:
                    continue
                self._refcount[key] -= 1
                if self._refcount[key] <= 0:
                    self._refcount.pop(key, None)
                    finished.append(key)
            # Together, not one after another: stopping a node can mean a
            # `docker rm -f` across the network, so serially this cost the SUM
            # of the pool -- ~12s per node of dead time after the workflow had
            # already finished.
            await self._stop_collectors(finished)

    async def shutdown(self) -> None:
        """Force-cancel any still-running collectors (defensive teardown)."""
        async with self._lock:
            self._refcount.clear()
            await self._stop_collectors(list(self._running))

    async def _stop_collectors(self, keys: "Sequence[CollectorKey]") -> None:
        """Stop every given collector concurrently."""
        await asyncio.gather(*(self._stop_collector(key) for key in keys))

    def _start_collector(self, key: CollectorKey) -> None:
        collector = self._collectors.get(key)
        if collector is None or key in self._running:
            return
        logger = get_logger(f"sflow.monitor.{collector.name}")
        _logger.info(f"Starting hardware monitor collector on node '{key[1]}'")
        sink = None
        if collector.stream_node_id is not None:
            writer = StreamedSampleWriter(
                self.raw_dir,
                node_id=collector.stream_node_id,
                node_name=key[1],
            )
            self._writers[key] = writer
            sink = writer
        task = asyncio.create_task(
            self._launcher.run_async(
                collector.command,
                output_logger=logger,
                env=collector.envs,
                task_name=collector.name,
                stdout_sink=sink,
            )
        )
        task.add_done_callback(
            lambda finished, node=key[1], name=collector.name: (
                self._warn_if_collector_died(finished, node=node, name=name)
            )
        )
        self._running[key] = task

    @staticmethod
    def _warn_if_collector_died(task: "asyncio.Task[Any]", *, node: str, name: str) -> None:
        """Say so when a collector stops on its own rather than being released.

        Monitoring is passive and must never fail a good run, so this only warns
        -- but it must not be SILENT. The collector's own output goes to a
        per-task logger at INFO, so a container that cannot start (no `python3`
        in the image, no NVIDIA runtime, unreachable daemon) used to leave
        nothing above INFO anywhere, and the run finished "successfully" with an
        empty monitor report.
        """
        if task.cancelled():
            return  # The ordinary path: released by its last consumer.
        exception = task.exception()
        if exception is not None:
            _logger.warning(
                f"Hardware monitor collector on node '{node}' failed: {exception}. "
                f"No samples will be collected for this node."
            )
            return
        code = task.result()
        # Positive only. A negative code is "killed by signal", which is what
        # stopping a collector looks like to a launcher that reports the code
        # instead of propagating the cancellation -- normal teardown, not a fault.
        if isinstance(code, int) and code > 0:
            _logger.warning(
                f"Hardware monitor collector on node '{node}' exited early "
                f"(code {code}); no samples will be collected for this node. "
                f"Its output is logged under '{name}'."
            )

    async def _stop_collector(self, key: CollectorKey) -> None:
        task = self._running.pop(key, None)
        if task is None:
            return
        _logger.info(f"Stopping hardware monitor collector on node '{key[1]}'")
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        collector = self._collectors.get(key)
        if collector is not None and collector.stop_command is not None:
            # Bounded on purpose. This teardown can reach across the network (a
            # `docker rm -f` against an unreachable host's daemon), and an
            # unbounded await here would wedge the whole workflow's shutdown
            # with no output. Best effort otherwise: a teardown that fails
            # because the container is already gone is a success, not an error.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self._launcher.run_async(
                        collector.stop_command,
                        task_name=f"{collector.name}_stop",
                    ),
                    timeout=_COLLECTOR_STOP_TIMEOUT_S,
                )
        # After the task is awaited, so any row the collector emitted on its way
        # out has already been written.
        writer = self._writers.pop(key, None)
        if writer is not None:
            writer.close()

    def report_spec(self) -> dict[str, Any]:
        """Build the JSON-able spec for ``postprocess_monitor_timeline``."""
        return {
            "workflow_name": self.workflow_name,
            "raw_dir": str(self.raw_dir),
            "out_dir": str(self.out_dir),
            "overview_path": str(self.overview_path),
            "interval_ms": self.interval_ms,
            "consumers": [c.to_spec() for c in self._consumers],
            "task_reports": [v.to_spec() for v in self._task_views],
            "task_events": list(self._task_events),
        }
