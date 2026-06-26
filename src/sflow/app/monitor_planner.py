# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-time builder for the ``monitor:`` feature.

Runs after the DAG is built (see ``assembly.build_state``). It resolves every
logical monitor (``workflow.monitor`` + each ``task.monitor``) into a
``MonitorConsumer``, dedups their targets into one collector per ``(backend,
node)`` (with the union of requested scopes), builds a bare-node collector
command per node, and returns a ready ``MonitorRegistry``.

Design invariants (see the plan):
* SINGLETON per node -- at most one collector per physical node.
* BARE node -- collectors run on the host via the backend's default operator,
  no container, ``ntasks_per_node=1``, ``overlap`` on, no GPU reservation.
* PASSIVE -- monitors never reserve nodes/GPUs; GPU/scope subsets are reporting
  filters applied post-run.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sflow.config.schema import (
    DEFAULT_MONITOR_INTERVAL_MS,
    DEFAULT_MONITOR_REPORT_FORMATS,
    MONITOR_BUILTIN_SCOPES,
    MonitorConfig,
    SflowConfig,
)
from sflow.core.backend import Backend
from sflow.core.monitor import (
    CollectorKey,
    MonitorConsumer,
    MonitorRegistry,
    NodeCollector,
    TaskResourceView,
)
from sflow.core.state import SflowState
from sflow.core.task import Task
from sflow.logging import get_logger
from sflow.monitoring import HARDWARE_MONITOR_FILENAME, hardware_monitor_source
from sflow.utils.gpu import parse_cuda_visible_devices

_logger = get_logger(__name__)

MONITOR_DIRNAME = "sflow_monitor"
MONITOR_RAW_DIRNAME = "raw"
MONITOR_OVERVIEW_FILENAME = "sflow_monitor.log"

# CLI-driven monitor injection (``--enable-*-monitor``) lives in
# ``sflow.app.monitor_cli``; this module is the plan-time DAG -> registry builder.


@dataclass
class _CollectorUnit:
    """Accumulated per-node collection requirements across all consumers."""

    key: CollectorKey
    backend: Backend
    scopes: set[str] = field(default_factory=set)
    custom: list[str] = field(default_factory=list)
    gpu_fields: str | None = None
    interval_ms: int = DEFAULT_MONITOR_INTERVAL_MS

    def merge_interval(self, interval_ms: int) -> None:
        self.interval_ms = min(self.interval_ms, interval_ms)


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"monitor {field_name} must be an int, got bool {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"monitor {field_name} must resolve to an int, got {value!r}; "
                "use a concrete value (monitor resources do not support "
                "${{ }} expressions yet)"
            ) from exc
    raise ValueError(f"monitor {field_name} must be an int, got {type(value).__name__}")


def _effective_builtin_scopes(monitor: MonitorConfig) -> list[str]:
    """Built-in scopes active for a monitor (all when ``scopes`` omitted)."""
    if monitor.scopes is None:
        return list(MONITOR_BUILTIN_SCOPES)
    return monitor.scopes.active_builtin_scopes()


def _custom_commands(monitor: MonitorConfig) -> list[str]:
    if monitor.scopes is None or monitor.scopes.custom is None:
        return []
    return list(monitor.scopes.custom.script)


def _gpu_fields(monitor: MonitorConfig) -> str | None:
    if monitor.scopes is None or monitor.scopes.gpu is None:
        return None
    return monitor.scopes.gpu.fields


def _collector_interval_ms(monitor: MonitorConfig) -> int:
    """Finest sampling interval across the monitor + its scopes (ms)."""
    intervals = [_coerce_int(monitor.interval, field_name="interval")]
    if monitor.scopes is not None:
        for name in MONITOR_BUILTIN_SCOPES:
            scope = getattr(monitor.scopes, name)
            if scope is not None and scope.enabled and scope.interval is not None:
                intervals.append(_coerce_int(scope.interval, field_name=f"scopes.{name}.interval"))
    return max(1, min(intervals))


def _backend_node_names(backend: Backend | None) -> list[str]:
    if backend is None or backend.allocation is None:
        return []
    return [n.name for n in backend.allocation.nodes]


class _MonitorPlanner:
    def __init__(self, config: SflowConfig, state: SflowState, *, output_dir: Path):
        self.config = config
        self.state = state
        self.output_dir = Path(output_dir)
        self.out_dir = self.output_dir / MONITOR_DIRNAME
        self.raw_dir = self.out_dir / MONITOR_RAW_DIRNAME
        self.overview_path = self.output_dir / MONITOR_OVERVIEW_FILENAME

        self._tasks_by_name: dict[str, Task] = {
            t.name: t for t in state.workflow.get_tasks()
        }
        self._units: dict[CollectorKey, _CollectorUnit] = {}
        # Map each config task name -> its runtime tasks (replicas included).
        self._replicas_by_base: dict[str, list[Task]] = self._build_replicas_by_base()

    # --- task / backend lookup -------------------------------------------------

    def _build_replicas_by_base(self) -> dict[str, list[Task]]:
        """Group runtime tasks under their config task name via ``Task.base_name``.

        ``base_name`` is stamped at assembly time (it is exactly the config task
        name a replica derives from), so this is an exact grouping with no
        name-prefix heuristics: a task named ``server`` can never swallow a
        distinct ``server_warmup`` task's replicas.
        """
        config_names = [t.name for t in self.config.workflow.tasks]
        by_base: dict[str, list[Task]] = {name: [] for name in config_names}
        for rt in self._tasks_by_name.values():
            base = rt.base_name or rt.name
            if base in by_base:
                by_base[base].append(rt)
        return by_base

    def _runtime_tasks_for(self, config_name: str) -> list[Task]:
        """Return all runtime tasks (replicas included) for a config task name."""
        return self._replicas_by_base.get(config_name, [])

    def _backend_for_task(self, task: Task) -> Backend | None:
        if task.backend_name and task.backend_name in self.state.backends:
            return self.state.backends[task.backend_name]
        return self._default_backend()

    def _default_backend(self) -> Backend | None:
        if self.state.default_backend is not None:
            return self.state.default_backend
        return next(iter(self.state.backends.values()), None)

    # --- node / gpu resolution -------------------------------------------------

    def _resolve_keys_and_gpus(
        self, monitor: MonitorConfig, *, owner_task: Task | None
    ) -> tuple[list[CollectorKey], list[int] | None]:
        resources = monitor.resources

        # used_by_tasks: union of referenced tasks' assigned nodes + GPUs.
        if resources is not None and resources.used_by_tasks:
            keys: list[CollectorKey] = []
            gpus: set[int] = set()
            for ref in resources.used_by_tasks:
                for rt in self._runtime_tasks_for(ref):
                    backend = self._backend_for_task(rt)
                    if backend is None:
                        continue
                    for node in rt.assigned_nodes:
                        keys.append((backend.name, node))
                    gpus.update(parse_cuda_visible_devices(rt.envs.get("CUDA_VISIBLE_DEVICES")))
            return _dedup_keys(keys), (sorted(gpus) if gpus else None)

        # Otherwise resolve against a single backend (task's or default).
        if owner_task is not None:
            backend = self._backend_for_task(owner_task)
            default_nodes = list(owner_task.assigned_nodes)
        else:
            backend = self._default_backend()
            default_nodes = _backend_node_names(backend)

        if backend is None:
            return [], None

        node_names = self._resolve_nodes(resources, backend, default_nodes)
        gpus = self._resolve_gpus(resources)
        return [(backend.name, node) for node in node_names], gpus

    def _resolve_nodes(
        self, resources: Any, backend: Backend, default_nodes: list[str]
    ) -> list[str]:
        nodes_conf = getattr(resources, "nodes", None) if resources else None
        if nodes_conf is None:
            return list(default_nodes)

        alloc_nodes = _backend_node_names(backend) or list(default_nodes)
        n = len(alloc_nodes)

        # exclude
        if nodes_conf.exclude is not None:
            raw = (
                nodes_conf.exclude
                if isinstance(nodes_conf.exclude, list)
                else [nodes_conf.exclude]
            )
            drop = set()
            for idx_val in raw:
                idx = _coerce_int(idx_val, field_name="resources.nodes.exclude")
                drop.add(idx if idx >= 0 else idx + n)
            alloc_nodes = [node for i, node in enumerate(alloc_nodes) if i not in drop]
            n = len(alloc_nodes)

        # indices
        if nodes_conf.indices is not None:
            chosen: list[str] = []
            for idx_val in nodes_conf.indices:
                idx = _coerce_int(idx_val, field_name="resources.nodes.indices")
                resolved = idx if idx >= 0 else idx + n
                if resolved < 0 or resolved >= n:
                    raise ValueError(
                        f"monitor resources.nodes.indices index {idx} out of range "
                        f"for {n} node(s)"
                    )
                chosen.append(alloc_nodes[resolved])
            return chosen

        # count
        if nodes_conf.count is not None:
            count = _coerce_int(nodes_conf.count, field_name="resources.nodes.count")
            if count <= 0:
                raise ValueError("monitor resources.nodes.count must be > 0")
            return alloc_nodes[:count]

        return alloc_nodes

    def _resolve_gpus(self, resources: Any) -> list[int] | None:
        gpus_conf = getattr(resources, "gpus", None) if resources else None
        if gpus_conf is None:
            return None
        count = _coerce_int(gpus_conf.count, field_name="resources.gpus.count")
        if count <= 0:
            raise ValueError("monitor resources.gpus.count must be > 0")
        return list(range(count))

    # --- consumer + unit assembly ---------------------------------------------

    def _add_consumer(
        self, monitor: MonitorConfig, *, owner: str, name: str, owner_task: Task | None
    ) -> MonitorConsumer | None:
        keys, gpus = self._resolve_keys_and_gpus(monitor, owner_task=owner_task)
        keys = _dedup_keys(keys)
        if not keys:
            _logger.warning(
                f"Monitor '{name}' resolved to no nodes; skipping. "
                "Check the target backend allocation / used_by_tasks."
            )
            return None

        scopes = _effective_builtin_scopes(monitor)
        custom = _custom_commands(monitor)
        gpu_fields = _gpu_fields(monitor)
        interval_ms = _collector_interval_ms(monitor)
        # Only the workflow (whole-pool) monitor writes a standalone aggregate
        # folder. Per-task monitors are rendered as resource-scoped task views
        # (see `_build_task_views`), so their consumer skips the duplicate folder.
        report = (
            owner == "workflow"
            and monitor.report is not None
            and monitor.report.enabled
        )
        formats = (
            monitor.report.format
            if monitor.report is not None
            else list(DEFAULT_MONITOR_REPORT_FORMATS)
        )

        # Accumulate per-node collection requirements (scope union, finest interval).
        for key in keys:
            unit = self._units.get(key)
            if unit is None:
                backend = self.state.backends.get(key[0])
                if backend is None:
                    continue
                unit = _CollectorUnit(key=key, backend=backend, interval_ms=interval_ms)
                self._units[key] = unit
            unit.scopes.update(scopes)
            unit.merge_interval(interval_ms)
            if gpu_fields and unit.gpu_fields is None:
                unit.gpu_fields = gpu_fields
            for cmd in custom:
                if cmd not in unit.custom:
                    unit.custom.append(cmd)

        consumer = MonitorConsumer(
            owner=owner,
            name=name,
            keys=keys,
            nodes=sorted({node for _b, node in keys}),
            gpus=gpus,
            scopes=scopes,
            report=report,
            report_formats=list(formats),
        )
        return consumer

    # --- per-task resource views ----------------------------------------------

    def _build_task_views(self) -> list[TaskResourceView]:
        """Per-task / per-replica report views derived from every monitor.

        * the workflow monitor -> one natural view per covered task (and replica);
        * a task monitor -> views for its own task;
        * a ``used_by_tasks`` monitor (owner A watching B) -> cross views over B's
          resources, windowed to A's run, labelled ``B__monitored_by__A``.

        Deduplicated by output label (scopes/formats unioned on collision), so a
        task covered by both the workflow monitor and its own monitor yields one
        folder. Only monitors with ``report.enabled`` contribute views.
        """
        views: dict[str, TaskResourceView] = {}

        def _add(view: TaskResourceView) -> None:
            existing = views.get(view.label)
            if existing is None:
                views[view.label] = view
                return
            existing.scopes = _union_list(existing.scopes, view.scopes)
            existing.report_formats = _union_list(
                existing.report_formats, view.report_formats
            )

        def _emit(monitor, *, triggered_by: str, owner_short: str, owner_window):
            if monitor.report is None or not monitor.report.enabled:
                return
            scopes = _effective_builtin_scopes(monitor)
            formats = list(monitor.report.format)
            used = monitor.resources.used_by_tasks if monitor.resources else None
            if used:
                for target in used:
                    for view in self._views_for(
                        target, triggered_by=triggered_by, owner_short=owner_short,
                        cross=True, scopes=scopes, formats=formats,
                        window_names=owner_window,
                    ):
                        _add(view)
            elif triggered_by == "workflow":
                for task_conf in self.config.workflow.tasks:
                    for view in self._views_for(
                        task_conf.name, triggered_by=triggered_by,
                        owner_short=owner_short, cross=False, scopes=scopes,
                        formats=formats, window_names=None,
                    ):
                        _add(view)
            else:
                for view in self._views_for(
                    owner_short, triggered_by=triggered_by, owner_short=owner_short,
                    cross=False, scopes=scopes, formats=formats, window_names=None,
                ):
                    _add(view)

        if self.config.workflow.monitor is not None:
            _emit(
                self.config.workflow.monitor,
                triggered_by="workflow", owner_short="workflow", owner_window=[],
            )
        for task_conf in self.config.workflow.tasks:
            if task_conf.monitor is None:
                continue
            owner_window = [rt.name for rt in self._runtime_tasks_for(task_conf.name)]
            _emit(
                task_conf.monitor,
                triggered_by=f"task:{task_conf.name}", owner_short=task_conf.name,
                owner_window=owner_window,
            )
        return list(views.values())

    def _views_for(
        self, resource_task: str, *, triggered_by: str, owner_short: str,
        cross: bool, scopes: list[str], formats: list[str], window_names,
    ) -> list[TaskResourceView]:
        """Combined + per-replica views for one task's resources.

        ``window_names`` is the owner's runtime task names for a cross view (or
        ``[]`` for the workflow owner -> full run); ``None`` for a natural view,
        which windows each view by its own task/replica.
        """
        rts = [
            rt
            for rt in self._runtime_tasks_for(resource_task)
            if rt.assigned_nodes
            or parse_cuda_visible_devices(rt.envs.get("CUDA_VISIBLE_DEVICES"))
        ]
        if not rts:
            return []
        suffix = f"__monitored_by__{owner_short}" if cross else ""
        out: list[TaskResourceView] = []
        combined_nodes: list[str] = []
        combined_gpus: set[int] = set()
        own_names: list[str] = []
        for rt in rts:
            nodes = list(rt.assigned_nodes)
            gpus = parse_cuda_visible_devices(rt.envs.get("CUDA_VISIBLE_DEVICES"))
            combined_nodes.extend(nodes)
            combined_gpus.update(gpus)
            own_names.append(rt.name)
            if len(rts) > 1:
                out.append(
                    _make_task_view(
                        label=f"{rt.name}{suffix}", task=resource_task, who=rt.name,
                        triggered_by=triggered_by, owner_short=owner_short,
                        cross=cross, nodes=nodes, gpus=gpus, scopes=scopes,
                        formats=formats,
                        window_tasks=(window_names if cross else [rt.name]),
                    )
                )
        out.append(
            _make_task_view(
                label=f"{resource_task}{suffix}", task=resource_task,
                who=resource_task, triggered_by=triggered_by,
                owner_short=owner_short, cross=cross, nodes=combined_nodes,
                gpus=sorted(combined_gpus), scopes=scopes, formats=formats,
                window_tasks=(window_names if cross else own_names),
            )
        )
        return out

    # --- command building ------------------------------------------------------

    def _build_collectors(self, hw_script_path: Path) -> dict[CollectorKey, NodeCollector]:
        collectors: dict[CollectorKey, NodeCollector] = {}
        for key, unit in self._units.items():
            backend = unit.backend
            node_name = key[1]
            collector_name = f"sflow_monitor_{node_name}"
            operator = self._build_bare_operator(backend, collector_name, node_name)
            script = _build_collector_script(
                hw_script_path,
                interval_ms=unit.interval_ms,
                scopes=[s for s in MONITOR_BUILTIN_SCOPES if s in unit.scopes],
                gpu_fields=unit.gpu_fields,
                custom=unit.custom,
            )
            envs = {"SFLOW_TASK_OUTPUT_DIR": str(self.raw_dir)}
            command = operator.build_command(
                task_name=collector_name, script=script, envs=envs
            )
            collectors[key] = NodeCollector(
                key=key, name=collector_name, command=command, envs=envs
            )
        return collectors

    def _build_bare_operator(self, backend: Backend, collector_name: str, node_name: str):
        # The backend owns bare-node monitor operator construction + configuration
        # (host-level execution, no container, single overlapping task, no GPU
        # reservation). Slurm/local run on the host already; Docker returns a host
        # bash operator so the collector sees the physical node and can read the
        # materialized hardware_monitor.py.
        return backend.monitor_operator(
            name=collector_name, assigned_nodes=[node_name]
        )

    # --- entry point -----------------------------------------------------------

    def build(self, *, materialize: bool) -> MonitorRegistry | None:
        consumers: list[MonitorConsumer] = []

        workflow_consumer: MonitorConsumer | None = None
        if self.config.workflow.monitor is not None:
            workflow_consumer = self._add_consumer(
                self.config.workflow.monitor,
                owner="workflow",
                name="workflow",
                owner_task=None,
            )
            if workflow_consumer is not None:
                consumers.append(workflow_consumer)

        # One consumer per RUNTIME task (replicas included) so the orchestrator can
        # acquire/release at task granularity. The registry's per-node refcount
        # still dedups collectors across all consumers + the workflow monitor.
        task_consumers: list[tuple[Task, MonitorConsumer]] = []
        for task_conf in self.config.workflow.tasks:
            if task_conf.monitor is None:
                continue
            for rt in self._runtime_tasks_for(task_conf.name):
                consumer = self._add_consumer(
                    task_conf.monitor,
                    owner=f"task:{rt.name}",
                    name=rt.name,
                    owner_task=rt,
                )
                if consumer is not None:
                    task_consumers.append((rt, consumer))
                    consumers.append(consumer)

        if not self._units:
            return None

        # Materialize the bundled collector into the run dir (shared FS) so srun
        # can launch it on compute nodes. Reference it by absolute path.
        hw_script_path = self.out_dir / HARDWARE_MONITOR_FILENAME
        if materialize:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            hw_script_path.write_text(hardware_monitor_source(), encoding="utf-8")

        collectors = self._build_collectors(hw_script_path)

        intervals = [u.interval_ms for u in self._units.values()]
        registry = MonitorRegistry(
            collectors,
            raw_dir=self.raw_dir,
            out_dir=self.out_dir,
            overview_path=self.overview_path,
            workflow_name=self.config.workflow.name,
            interval_ms=min(intervals) if intervals else None,
        )
        for consumer in consumers:
            registry.register_consumer(consumer)

        # Per-task / per-replica resource views (resolved windows are stamped from
        # task events at post-process time).
        for view in self._build_task_views():
            registry.register_task_view(view)

        # Attach consumers to runtime objects: workflow -> state, task -> Task.
        self.state.workflow_monitor = workflow_consumer
        for rt, consumer in task_consumers:
            rt.monitor = consumer

        return registry


def _dedup_keys(keys: list[CollectorKey]) -> list[CollectorKey]:
    seen: set[CollectorKey] = set()
    out: list[CollectorKey] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _union_list(a: list[str], b: list[str]) -> list[str]:
    """Order-preserving union (``a`` first, then new items from ``b``)."""
    out = list(a)
    for item in b:
        if item not in out:
            out.append(item)
    return out


def _dedup_str(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _make_task_view(
    *, label: str, task: str, who: str, triggered_by: str, owner_short: str,
    cross: bool, nodes: list[str], gpus: list[int], scopes: list[str],
    formats: list[str], window_tasks,
) -> TaskResourceView:
    title = (
        f"{who} hardware timeline (monitored by {owner_short})"
        if cross
        else f"{who} hardware timeline"
    )
    return TaskResourceView(
        label=label,
        task=task,
        triggered_by=triggered_by,
        title=title,
        nodes=_dedup_str(nodes),
        gpus=(sorted(set(gpus)) if gpus else None),
        scopes=list(scopes),
        report_formats=list(formats),
        window_tasks=list(window_tasks or []),
        cross=cross,
    )


def _build_collector_script(
    hw_script_path: Path,
    *,
    interval_ms: int,
    scopes: list[str],
    gpu_fields: str | None,
    custom: list[str],
) -> list[str]:
    lines = ['echo "Starting hardware monitor"']
    # Custom commands run in the background alongside the built-in collector.
    for cmd in custom:
        lines.append(f"{cmd} &")
    if scopes:
        main = (
            f"python3 {shlex.quote(str(hw_script_path))} "
            f"--interval-ms {int(interval_ms)} --scopes {','.join(scopes)}"
        )
        if gpu_fields:
            main += f" --gpu-fields {shlex.quote(gpu_fields)}"
        lines.append(main)
    elif custom:
        # Only custom commands -> keep the step alive until they finish.
        lines.append("wait")
    return lines


def run_monitor_postprocess(registry: MonitorRegistry) -> None:
    """Run the single deferred post-process pass for a workflow's monitors.

    Produces ``sflow_monitor.log`` plus opt-in per-consumer reports under
    ``sflow_monitor/``. Best-effort: never raises (monitor reporting must not
    fail an otherwise-successful workflow).
    """
    if not registry.has_collectors:
        return
    # Surface that the run is done and sflow is now aggregating samples: this pass
    # can take a few seconds on long runs, so the hint keeps it from looking hung.
    _logger.info(
        "Workflow finished; post-processing hardware monitor samples into "
        f"{registry.overview_path} (this may take a moment on long runs)..."
    )
    try:
        from sflow.monitoring import postprocess_monitor_timeline

        result = postprocess_monitor_timeline.process(registry.report_spec())
        _logger.info(
            f"Hardware monitor report written: {result.get('sample_count', 0)} samples "
            f"-> {registry.overview_path}"
        )
    except Exception:
        _logger.warning("Hardware monitor post-processing failed", exc_info=True)


def build_monitor_registry(
    config: SflowConfig,
    state: SflowState,
    *,
    output_dir: Path | str | None,
    materialize: bool,
) -> MonitorRegistry | None:
    """Build the monitor registry/schedule for a workflow (plan time).

    Returns ``None`` when no monitors are configured. Also attaches the workflow
    consumer to ``state.workflow_monitor`` and each task consumer to its runtime
    ``Task.monitor``.
    """
    if config.workflow.monitor is None and not any(
        t.monitor is not None for t in config.workflow.tasks
    ):
        return None
    if output_dir is None:
        _logger.warning("Monitor configured but no output_dir available; skipping.")
        return None
    planner = _MonitorPlanner(config, state, output_dir=Path(output_dir))
    return planner.build(materialize=materialize)
