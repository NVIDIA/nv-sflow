# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sflow.utils.gpu import parse_cuda_visible_devices

from .task import Task, TaskStatus
from .uploads import UploadResult
from .workflow import Workflow

_RESOURCE_CHART_MARKS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# Fixed render order + short labels for upload rows in the Uploads section.
_UPLOAD_STATUS_ORDER = ("uploaded", "failed", "skipped", "dry-run")
_UPLOAD_STATUS_LABEL = {
    "uploaded": "[OK]  ",
    "failed": "[FAIL]",
    "skipped": "[SKIP]",
    "dry-run": "[DRY] ",
}


def format_upload_row(result: UploadResult) -> str:
    """Render one upload row: ``[STATUS] <source-basename> -> <destination>``,
    with ``(error)  [on_error=...]`` appended for failures.

    Shared by the summary file's Uploads section and the console end-of-run
    block so both render identically.
    """
    label = _UPLOAD_STATUS_LABEL.get(result.status, f"[{result.status}]")
    source = Path(result.source).name or result.source
    row = f"{label} {source} -> {result.destination}"
    if result.status == "failed":
        if result.error:
            row += f"  ({result.error})"
        row += f"  [on_error={result.on_error}]"
    return row


def format_upload_counts(results: list[UploadResult]) -> str:
    """Return compact upload counts in stable display order, e.g. ``uploaded=2``."""
    counts = Counter(result.status for result in results)
    parts = [
        f"{status}={counts[status]}"
        for status in _UPLOAD_STATUS_ORDER
        if counts.get(status)
    ]
    return ", ".join(parts) if parts else "none"

@dataclass
class _TimelineEvent:
    event: str
    task_name: str
    timestamp: datetime
    elapsed: float
    attempt: int | None
    status: str | None
    details: dict[str, str]


class SflowSummaryWriter:
    """Render a live, terminal-friendly summary of workflow execution."""

    def __init__(self, path: Path | str, *, debounce_interval: float = 0.25):
        self.path = Path(path)
        self._debounce_interval = max(0.0, float(debounce_interval))
        self._workflow: Workflow | None = None
        self._output_dir: Path | None = None
        self._runtime_info_text = ""
        self._command_log_paths: dict[str, Path] = {}
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None
        self._started_monotonic = time.monotonic()
        self._ended_monotonic: float | None = None
        self._status = "RUNNING"
        self._workflow_detail: str | None = None
        self._timeline: list[_TimelineEvent] = []
        self._failure_hints: list[str] = []
        self._network_warnings: list[str] = []
        self._uploads: list[UploadResult] = []
        self._task_started: dict[str, float] = {}
        self._task_ready: dict[str, float] = {}
        self._task_finished: dict[str, float] = {}
        self._render_task: asyncio.Task[None] | None = None
        self._render_dirty = False
        self._render_generation = 0
        self._last_written_generation = -1
        self._write_lock = threading.Lock()

    def start(
        self,
        *,
        workflow: Workflow,
        output_dir: Path | str,
        runtime_info_text: str,
        command_log_paths: dict[str, Path | str],
    ) -> None:
        self._workflow = workflow
        self._output_dir = Path(output_dir)
        self._runtime_info_text = runtime_info_text
        self._command_log_paths = {
            name: Path(path) for name, path in command_log_paths.items()
        }
        self._started_at = datetime.now().astimezone()
        self._started_monotonic = time.monotonic()
        self._status = "RUNNING"
        self._schedule_render()

    def task_unblocked(self, task: Task, **_: Any) -> None:
        self._append_event("UNBLOCKED", task)

    def task_submitted(self, task: Task, **_: Any) -> None:
        self._task_started[task.name] = time.monotonic()
        self._append_event("SUBMITTED", task)

    def task_completed(self, task: Task, **_: Any) -> None:
        self._task_finished[task.name] = time.monotonic()
        self._append_event("COMPLETED", task, exit_code=task.exit_code)

    def task_retry(
        self,
        task: Task,
        *,
        exit_code: int | None = None,
        delay: float | None = None,
        **_: Any,
    ) -> None:
        details = []
        if exit_code is not None:
            details.append(f"exit={exit_code}")
        if delay is not None:
            details.append(f"retry_in={delay:.3f}s")
        self._append_event("RETRY", task, extra_details=details)

    def task_failed(
        self,
        task: Task,
        *,
        reason: str | None = None,
        exit_code: int | None = None,
        **_: Any,
    ) -> None:
        self._task_finished[task.name] = time.monotonic()
        exit_code = task.exit_code if exit_code is None else exit_code
        details = [f"reason={reason}"] if reason else []
        self._append_event("FAILED", task, exit_code=exit_code, extra_details=details)
        self._failure_hints.append(self._format_failure_hint(task, reason, exit_code))

    def task_cancelled(
        self,
        task: Task,
        *,
        reason: str | None = None,
        **_: Any,
    ) -> None:
        self._task_finished[task.name] = time.monotonic()
        details = [f"reason={reason}"] if reason else []
        self._append_event("CANCELLED", task, extra_details=details)
        self._failure_hints.append(self._format_failure_hint(task, reason, None))

    def task_ready(self, task: Task, **_: Any) -> None:
        ready_at = time.monotonic()
        self._task_ready[task.name] = ready_at
        self._task_finished[task.name] = ready_at
        self._append_event("READY", task)

    def record_network_warning(self, task: Task, message: str, **_: Any) -> None:
        """Record a non-fatal network warning (e.g. RDMA -> TCP fallback).

        Surfaced in the dedicated 'Network Warnings' section, prefixed with the
        task name. Duck-typed like the other summary hooks, so consumers that
        don't implement it are unaffected.
        """
        if not message:
            return
        self._network_warnings.append(f"{task.name}: {message}")
        self._schedule_render()

    def record_uploads(self, results: list[UploadResult], **_: Any) -> None:
        """Record upload outcomes for the dedicated end-of-run Uploads section.

        Called by the orchestrator (per-task uploads) and the app (workflow
        ``upload_all``). Duck-typed like the other summary hooks, so consumers
        that don't implement it are unaffected.
        """
        if not results:
            return
        self._uploads.extend(results)
        self._schedule_render()

    @property
    def upload_results(self) -> list[UploadResult]:
        """All recorded upload outcomes, in the order they were reported."""
        return list(self._uploads)

    def workflow_finished(
        self,
        *,
        status: str | None = None,
        detail: str | None = None,
        **_: Any,
    ) -> None:
        self._ended_at = datetime.now().astimezone()
        self._ended_monotonic = time.monotonic()
        if status is None and self._workflow is not None:
            status = self._infer_status()
        self._status = status or self._status
        if detail:
            self._workflow_detail = detail
        self._schedule_render()
        self.flush()

    def _append_event(
        self,
        event: str,
        task: Task,
        *,
        exit_code: int | None = None,
        extra_details: list[str] | None = None,
    ) -> None:
        now = datetime.now().astimezone()
        elapsed = time.monotonic() - self._started_monotonic
        details = self._task_details(task)
        if exit_code is not None:
            details["exit"] = str(exit_code)
        for detail in extra_details or []:
            key, sep, value = detail.partition("=")
            if sep:
                details[key] = value
            else:
                details[detail] = ""
        self._timeline.append(
            _TimelineEvent(
                event=event,
                task_name=task.name,
                timestamp=now,
                elapsed=elapsed,
                attempt=int(getattr(task, "attempts", 0)) or None,
                status=str(getattr(task, "status", "")) or None,
                details=details,
            )
        )
        self._schedule_render()

    def _render(self) -> None:
        summary = self._build_summary()
        if summary is None:
            return
        self._write_summary(summary, self._next_render_generation())

    def _build_summary(self) -> str | None:
        if self._workflow is None:
            return None
        return "\n".join(self._lines()) + "\n"

    def _write_summary(self, text: str, generation: int) -> None:
        with self._write_lock:
            if generation < self._last_written_generation:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    delete=False,
                ) as tmp:
                    tmp_path = Path(tmp.name)
                    tmp.write(text)
                tmp_path.replace(self.path)
                self._last_written_generation = generation
            finally:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()

    def _schedule_render(self) -> None:
        self._render_dirty = True
        self._next_render_generation()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._render()
            self._render_dirty = False
            return
        if self._render_task is None or self._render_task.done():
            self._render_task = loop.create_task(self._debounced_render())

    async def _debounced_render(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._debounce_interval)
                self._render_dirty = False
                generation = self._render_generation
                summary = self._build_summary()
                if summary is not None:
                    await asyncio.to_thread(
                        self._write_summary,
                        summary,
                        generation,
                    )
                if not self._render_dirty:
                    return
        except asyncio.CancelledError:
            return

    def flush(self) -> None:
        if self._render_task is not None and not self._render_task.done():
            self._render_task.cancel()
        self._render_task = None
        self._render_dirty = False
        self._render()

    def _next_render_generation(self) -> int:
        self._render_generation += 1
        return self._render_generation

    def _lines(self) -> list[str]:
        assert self._workflow is not None
        workflow = self._workflow
        tasks = workflow.get_tasks()
        started = self._format_datetime(self._started_at)
        ended = self._format_datetime(self._ended_at)
        duration = self._workflow_duration()

        lines = [
            "Sflow Summary",
            "=============",
            f"Workflow : {workflow.name}",
            f"Status   : {self._status}",
            f"Started  : {started}",
            f"Ended    : {ended}",
            f"Duration : {duration}",
            f"Output   : {self._output_dir}",
            f"Tasks    : {len(tasks)}",
            f"Summary  : {self.path}",
        ]
        lines.extend(self._end_summary_lines(tasks))
        lines.extend(["", "Runtime", "-------"])
        lines.extend(self._runtime_info_text.splitlines() or ["(runtime unavailable)"])

        lines.extend(self._network_warning_lines())

        if self._status == "FAILED":
            lines.extend(["", "Failure Hints", "-------------"])
            lines.extend(self._failure_hints or ["(none)"])

        lines.extend(["", "Task Duration Chart", "-------------------"])
        lines.extend(self._duration_chart_lines(tasks))
        lines.extend(["", "Timeline", "--------"])
        lines.extend(self._timeline_lines())
        lines.extend(["", "GPU Usage Chart", "---------------"])
        lines.extend(self._gpu_usage_chart_lines(tasks))
        lines.extend(["", "Node Usage Chart", "----------------"])
        lines.extend(self._node_usage_chart_lines(tasks))

        lines.extend(["", "Command Logs", "------------"])
        existing_command_logs = {
            name: path
            for name, path in self._command_log_paths.items()
            if path.exists()
        }
        if existing_command_logs:
            for name, path in sorted(existing_command_logs.items()):
                lines.append(f"{name}: {path}")
        else:
            lines.append("(none)")

        lines.extend(self._upload_section_lines())

        lines.extend(["", "Workflow DAG", "------------"])
        lines.extend(f"  {line}" for line in workflow.task_graph.dag.render_ascii())
        lines.extend(["", "Dependencies", "------------"])
        lines.extend(self._dependency_lines(workflow))
        if self._status != "FAILED":
            lines.extend(["", "Failure Hints", "-------------"])
            lines.extend(self._failure_hints or ["(none)"])
        return lines

    def _task_details(self, task: Task) -> dict[str, str]:
        details: dict[str, str] = {}
        if self._workflow is not None:
            dag = self._workflow.task_graph.dag
            deps = [
                f"{name}:{dag[name].status}"
                for name in dag.get_dependencies(task.name)
                if name in dag.nodes
            ]
            dependents = dag.get_dependents(task.name)
            details["deps"] = ", ".join(deps) if deps else "none"
            details["next"] = ", ".join(dependents) if dependents else "none"
        backend = getattr(task, "backend_name", None)
        if backend:
            details["backend"] = str(backend)
        operator_name = self._operator_name(task)
        if operator_name:
            details["operator"] = str(operator_name)
        log_path = self._task_log_path(task)
        if log_path is not None:
            details["log"] = str(log_path)
        return details

    def _dependency_lines(self, workflow: Workflow) -> list[str]:
        lines = []
        dag = workflow.task_graph.dag
        for name in dag.topological_sort():
            deps = dag.get_dependencies(name)
            if deps:
                lines.append(f"{', '.join(deps)} -> {name}")
            else:
                lines.append(f"START -> {name}")
        return lines or ["(none)"]

    def _timeline_lines(self) -> list[str]:
        if not self._timeline:
            return ["(no task events yet)"]
        task_width = self._chart_label_width(event.task_name for event in self._timeline)
        lines = [
            "  ".join(
                [
                    "Time",
                    "Elapsed",
                    f"{'Task':<{task_width}}",
                    "Event",
                    "Summary",
                ]
            ),
            "  ".join(
                [
                    "--------",
                    "--------",
                    "-" * task_width,
                    "-----------",
                    "-------",
                ]
            ),
        ]
        for event in self._timeline:
            lines.append(
                "  ".join(
                    [
                        event.timestamp.strftime("%H:%M:%S"),
                        f"+{event.elapsed:06.3f}s",
                        f"{event.task_name:<{task_width}}",
                        f"{event.event:<11}",
                        self._event_summary(event),
                    ]
                )
            )
        return lines

    def _event_summary(self, event: _TimelineEvent) -> str:
        if event.event == "UNBLOCKED":
            return "dependencies satisfied; ready to submit"
        if event.event == "SUBMITTED":
            return f"attempt={event.attempt or '-'}"
        if event.event == "COMPLETED":
            return "exit=" + event.details.get("exit", "0")
        if event.event == "FAILED":
            reason = event.details.get("reason")
            exit_code = event.details.get("exit")
            if reason and exit_code:
                return f"exit={exit_code}; {reason}"
            return reason or (f"exit={exit_code}" if exit_code else "failed")
        if event.event == "RETRY":
            retry_in = event.details.get("retry_in")
            exit_code = event.details.get("exit")
            parts = []
            if exit_code:
                parts.append(f"exit={exit_code}")
            if retry_in:
                parts.append(f"retry in {retry_in}")
            return "; ".join(parts) if parts else "retry scheduled"
        if event.event == "CANCELLED":
            return event.details.get("reason", "cancelled")
        if event.event == "READY":
            return "readiness satisfied"
        return event.status or ""

    def _duration_chart_lines(self, tasks: list[Task]) -> list[str]:
        if not tasks:
            return ["(no tasks)"]
        width = 30
        window = self._chart_window(tasks)
        if window is None:
            return ["(no completed task timings yet)"]
        origin, span, now = window
        label_width = self._chart_label_width(task.name for task in tasks)
        lines = []
        for task in tasks:
            start = self._task_started.get(task.name)
            if start is None:
                bar = "." * width
                duration = 0.0
            else:
                end = self._task_finished.get(task.name, now)
                bar = self._chart_bar(start, end, origin=origin, span=span, width=width)
                duration = max(end - start, 0.0)
            lines.append(
                f"{task.name:<{label_width}} |{bar}| {duration:.3f}s {task.status}"
            )
        return lines

    def _gpu_usage_chart_lines(self, tasks: list[Task]) -> list[str]:
        rows: list[tuple[str, Task]] = []
        for task in tasks:
            gpu_ids = self._task_gpu_ids(task)
            if not gpu_ids:
                continue
            node_names = task.assigned_nodes or [""]
            for node_name in node_names:
                for gpu_id in gpu_ids:
                    label = f"{node_name} GPU {gpu_id}" if node_name else f"GPU {gpu_id}"
                    rows.append((label, task))
        return self._resource_chart_lines(
            rows,
            resource_type="gpus",
            empty_message="(no GPU timings)",
        )

    def _node_usage_chart_lines(self, tasks: list[Task]) -> list[str]:
        rows = [
            (node_name, task)
            for task in tasks
            for node_name in task.assigned_nodes
        ]
        return self._resource_chart_lines(
            rows,
            resource_type="nodes",
            empty_message="(no node timings)",
        )

    def _resource_chart_lines(
        self,
        rows: list[tuple[str, Task]],
        *,
        resource_type: str,
        empty_message: str,
    ) -> list[str]:
        if not rows:
            return [empty_message]
        width = 30
        window = self._resource_chart_window(rows, resource_type=resource_type)
        if window is None:
            return ["(no completed task timings yet)"]
        origin, span, now = window
        grouped_rows: dict[str, list[Task]] = {}
        for label, task in rows:
            start = self._task_started.get(task.name)
            if start is None:
                continue
            grouped_rows.setdefault(label, []).append(task)

        if not grouped_rows:
            return ["(no completed task timings yet)"]

        label_width = self._chart_label_width(grouped_rows.keys())
        global_marks: dict[str, str] = {}
        lines: list[str] = []
        if resource_type == "nodes":
            lines.append(
                "Hint: '*' marks multiple tasks active on this resource at the same time."
            )
        lines.append("Legend:")
        for _label, task in rows:
            start = self._task_started.get(task.name)
            if start is None or task.name in global_marks:
                continue
            end = self._resource_end(task, now, resource_type=resource_type)
            global_marks[task.name] = self._chart_mark(len(global_marks))
            lines.append(
                f"  {global_marks[task.name]}={task.name} "
                f"{max(end - start, 0.0):.3f}s {task.status}"
            )
        lines.append("")

        for label, row_tasks in grouped_rows.items():
            track = ["."] * width
            for task in row_tasks:
                start = self._task_started.get(task.name)
                if start is None:
                    continue
                end = self._resource_end(task, now, resource_type=resource_type)
                mark = global_marks[task.name]
                left, right = self._chart_bounds(
                    start,
                    end,
                    origin=origin,
                    span=span,
                    width=width,
                )
                for index in range(left, right):
                    track[index] = mark if track[index] in {".", mark} else "*"
            lines.append(f"{label:<{label_width}} |{''.join(track)}|")
        return lines

    def _resource_chart_window(
        self,
        rows: list[tuple[str, Task]],
        *,
        resource_type: str,
    ) -> tuple[float, float, float] | None:
        now = self._ended_monotonic or time.monotonic()
        starts = [
            self._task_started.get(task.name)
            for _label, task in rows
            if task.name in self._task_started
        ]
        if not starts:
            return None
        origin = min(starts)
        latest = max(
            self._resource_end(task, now, resource_type=resource_type)
            for _label, task in rows
            if task.name in self._task_started
        )
        span = max(latest - origin, 0.001)
        return origin, span, now

    def _resource_end(self, task: Task, now: float, *, resource_type: str) -> float:
        policy = task.resource_release_after.get(resource_type)
        if policy is None and resource_type == "nodes":
            policy = task.resource_release_after.get("gpus")
        if policy == "task_ready":
            return self._task_ready.get(
                task.name,
                self._task_finished.get(task.name, now),
            )
        if policy == "workflow_completion":
            return now
        if policy == "task_completion" and task.status == TaskStatus.READY:
            return now
        return self._task_finished.get(task.name, now)

    def _chart_window(self, tasks: list[Task]) -> tuple[float, float, float] | None:
        now = self._ended_monotonic or time.monotonic()
        starts = [
            self._task_started.get(t.name)
            for t in tasks
            if t.name in self._task_started
        ]
        if not starts:
            return None
        origin = min(starts)
        latest = max(self._task_finished.get(t.name, now) for t in tasks)
        span = max(latest - origin, 0.001)
        return origin, span, now

    @staticmethod
    def _chart_bar(
        start: float,
        end: float,
        *,
        origin: float,
        span: float,
        width: int,
    ) -> str:
        left, right = SflowSummaryWriter._chart_bounds(
            start,
            end,
            origin=origin,
            span=span,
            width=width,
        )
        return "." * left + "#" * (right - left) + "." * (width - right)

    @staticmethod
    def _chart_bounds(
        start: float,
        end: float,
        *,
        origin: float,
        span: float,
        width: int,
    ) -> tuple[int, int]:
        left = int(((start - origin) / span) * (width - 1))
        right = max(left + 1, int(((end - origin) / span) * width))
        left = max(0, min(left, width - 1))
        right = max(left + 1, min(right, width))
        return left, right

    @staticmethod
    def _chart_label_width(labels: Any) -> int:
        return max(20, *(len(str(label)) for label in labels))

    @staticmethod
    def _chart_mark(index: int) -> str:
        if index < len(_RESOURCE_CHART_MARKS):
            return _RESOURCE_CHART_MARKS[index]
        return "?"

    @staticmethod
    def _task_gpu_ids(task: Task) -> list[str]:
        return [
            str(gpu_id)
            for gpu_id in parse_cuda_visible_devices(
                task.envs.get("CUDA_VISIBLE_DEVICES")
            )
        ]

    def _end_summary_lines(self, tasks: list[Task]) -> list[str]:
        counts = Counter(str(task.status) for task in tasks)
        lines = []
        if self._workflow_detail:
            lines.append(f"Workflow Detail : {self._workflow_detail}")
        if counts:
            lines.append(
                "Counts       : "
                + ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
            )
        else:
            lines.append("Counts       : none")
        failed_or_cancelled = [
            task.name
            for task in tasks
            if task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}
        ]
        lines.append(
            "FAILED/CANCELLED Tasks : "
            + (", ".join(failed_or_cancelled) if failed_or_cancelled else "none")
        )
        if self._uploads:
            lines.append(f"Uploads     : {format_upload_counts(self._uploads)}")
        return lines

    def _network_warning_lines(self) -> list[str]:
        """Render the 'Network Warnings' section, or [] when none were recorded."""
        if not self._network_warnings:
            return []
        return ["", "Network Warnings", "----------------", *self._network_warnings]

    def _upload_section_lines(self) -> list[str]:
        """Render the dedicated 'Uploads' section, or [] when no uploads ran."""
        if not self._uploads:
            return []
        lines = ["", "Uploads", "-------", f"Counts : {format_upload_counts(self._uploads)}"]
        grouped: dict[str, list[UploadResult]] = {}
        for result in self._uploads:
            grouped.setdefault(result.task, []).append(result)
        for task_name, rows in grouped.items():
            lines.append(f"{task_name}:")
            lines.extend(f"  {format_upload_row(row)}" for row in rows)
        return lines

    def _format_failure_hint(
        self,
        task: Task,
        reason: str | None,
        exit_code: int | None,
    ) -> str:
        parts = [task.name]
        if exit_code is not None:
            parts.append(f"exit={exit_code}")
        parts.append(f"attempts={int(getattr(task, 'attempts', 0))}")
        if reason:
            parts.append(f"reason={reason}")
        log_path = self._task_log_path(task)
        if log_path is not None:
            parts.append(f"log={log_path}")
        return "  ".join(parts)

    def _task_log_path(self, task: Task) -> Path | None:
        task_dir = task.envs.get("SFLOW_TASK_OUTPUT_DIR")
        if not task_dir:
            return None
        return Path(task_dir) / f"{task.name}.log"

    @staticmethod
    def _join_names(names: list[str]) -> str:
        return ", ".join(names)

    @staticmethod
    def _operator_name(task: Task) -> str | None:
        operator = getattr(task, "operator_name", None) or getattr(
            getattr(task, "operator", None), "config", None
        )
        if isinstance(operator, str):
            return operator
        return getattr(operator, "name", None) or getattr(operator, "type", None)

    def _workflow_duration(self) -> str:
        if self._ended_monotonic is not None:
            elapsed = self._ended_monotonic - self._started_monotonic
        else:
            elapsed = time.monotonic() - self._started_monotonic
        return f"{max(elapsed, 0.0):.3f}s"

    def _infer_status(self) -> str:
        assert self._workflow is not None
        statuses = [task.status for task in self._workflow.get_tasks()]
        if any(status in {TaskStatus.FAILED, TaskStatus.TIMEOUT} for status in statuses):
            return "FAILED"
        if any(status == TaskStatus.CANCELLED for status in statuses):
            return "CANCELLED"
        if all(status in {TaskStatus.COMPLETED, TaskStatus.READY} for status in statuses):
            return "COMPLETED"
        return "RUNNING"

    @staticmethod
    def _format_datetime(value: datetime | None) -> str:
        if value is None:
            return "(pending)"
        return value.isoformat(timespec="seconds")
