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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sflow.logging import get_logger

from .launcher import SubprocessLauncher

if TYPE_CHECKING:
    from .command import Command

_logger = get_logger(__name__)

CollectorKey = tuple[str, str]  # (backend_name, node_name)


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
            for key in consumer.keys:
                if key not in self._refcount:
                    continue
                self._refcount[key] -= 1
                if self._refcount[key] <= 0:
                    self._refcount.pop(key, None)
                    await self._stop_collector(key)

    async def shutdown(self) -> None:
        """Force-cancel any still-running collectors (defensive teardown)."""
        async with self._lock:
            self._refcount.clear()
            for key in list(self._running):
                await self._stop_collector(key)

    def _start_collector(self, key: CollectorKey) -> None:
        collector = self._collectors.get(key)
        if collector is None or key in self._running:
            return
        logger = get_logger(f"sflow.monitor.{collector.name}")
        _logger.info(f"Starting hardware monitor collector on node '{key[1]}'")
        self._running[key] = asyncio.create_task(
            self._launcher.run_async(
                collector.command,
                output_logger=logger,
                env=collector.envs,
                task_name=collector.name,
            )
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
