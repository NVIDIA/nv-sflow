# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Deque, Iterable

from rich.console import Console
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, RichLog, Static

from sflow.core.task import Task, TaskStatus
from sflow.core.workflow import Workflow


class _TuiLogHandler(logging.Handler):
    """A logging handler that appends LogRecords to a deque."""

    def __init__(
        self,
        sink: Deque[logging.LogRecord],
        *,
        level: int = logging.INFO,
        log_lock: threading.Lock | None = None,
    ):
        super().__init__(level=level)
        self._sink = sink
        self._lock = log_lock or threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Copy to avoid surprises if other handlers mutate the record.
            record = logging.makeLogRecord(record.__dict__.copy())
        except Exception:
            # Best-effort: never break the workflow due to UI.
            pass
        with self._lock:
            self._sink.append(record)


@dataclass
class RichTuiConfig:
    left_width: int = 56
    header_height: int = 5
    log_panel_title: str = "Logs"
    task_panel_title: str = "Tasks"
    backend_panel_title: str = "Backends"
    backend_panel_height: int = 6
    max_log_lines: int = 4000
    # If <= 0, auto-compute based on the right panel height.
    tail_log_lines: int = 0
    refresh_per_second: int = 2


class RichTui(AbstractContextManager["RichTui"]):
    """
    Compatibility facade for the Textual terminal UI.

    The public name is kept so callers do not need to know whether the TUI is
    implemented by Rich Live or Textual.
    """

    def __init__(
        self,
        workflow: Workflow | None,
        *,
        workflow_name: str | None = None,
        console: Console | None = None,
        config: RichTuiConfig | None = None,
        logger_name: str = "sflow",
        log_buffer: Deque[logging.LogRecord] | None = None,
        log_lock: threading.Lock | None = None,
        attach_log_handler: bool = True,
    ):
        self._workflow = workflow
        self._workflow_name = workflow_name or (
            workflow.name if workflow is not None else "workflow"
        )
        self._console = console or Console()
        self._config = config or RichTuiConfig()
        self._logger_name = logger_name
        self._attach_log_handler = bool(attach_log_handler)

        self._logs: Deque[logging.LogRecord] = (
            log_buffer
            if log_buffer is not None
            else deque(maxlen=self._config.max_log_lines)
        )
        self._logs_lock = log_lock or threading.Lock()
        self._handler: _TuiLogHandler | None = None
        if self._attach_log_handler:
            self._handler = _TuiLogHandler(
                self._logs,
                level=logging.DEBUG,
                log_lock=self._logs_lock,
            )

        self._start_time = time.time()
        self._app = _SflowTextualApp(self)
        self._app_task: asyncio.Task | None = None
        self._app_thread: threading.Thread | None = None
        self._interrupt_handler: Callable[[], None] | None = None

    @property
    def workflow(self) -> Workflow | None:
        return self._workflow

    def set_workflow(self, workflow: Workflow) -> None:
        self._workflow = workflow
        if not self._workflow_name:
            self._workflow_name = workflow.name

    def set_interrupt_handler(self, handler: Callable[[], None] | None) -> None:
        self._interrupt_handler = handler

    def _request_interrupt(self) -> None:
        if self._interrupt_handler is not None:
            self._interrupt_handler()

    def __enter__(self) -> "RichTui":
        # Synchronous context use is retained for non-interactive tests and callers.
        # Interactive Textual runs must use start_async() so Textual can install
        # signal handlers on the main thread.
        # Attach a log capture handler to the sflow logger (optional; caller may attach earlier).
        self._attach_handler()
        self._app_thread = threading.Thread(
            target=self._run_headless_app,
            name="sflow-textual-tui",
            daemon=True,
        )
        self._app_thread.start()
        self.refresh()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._request_app_exit_from_thread()
        if self._app_thread is not None:
            self._app_thread.join(timeout=2.0)
            self._app_thread = None

        self._detach_handler()

    async def start_async(self) -> None:
        self._attach_handler()
        if self._app_task is None:
            self._app_task = asyncio.create_task(
                self._app.run_async(
                    headless=not bool(getattr(self._console, "is_terminal", False)),
                    mouse=True,
                )
            )
            await asyncio.sleep(0)
        self.refresh()

    async def stop_async(self) -> None:
        try:
            if self._app.is_running:
                self._app.exit()
        except Exception:
            pass
        if self._app_task is not None:
            try:
                await asyncio.wait_for(self._app_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._app_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._app_task
            except Exception:
                pass
            finally:
                self._app_task = None
        self._detach_handler()

    def _run_headless_app(self) -> None:
        try:
            self._app.run(headless=True, mouse=False)
        except Exception:
            # Best effort: UI failures must not break workflow execution.
            logging.getLogger(self._logger_name).debug("Textual TUI exited", exc_info=True)

    def _request_app_exit_from_thread(self) -> None:
        try:
            if self._app.is_running:
                self._app.call_from_thread(self._app.exit)
        except Exception:
            pass

    def _attach_handler(self) -> None:
        if self._handler is not None:
            logger = logging.getLogger(self._logger_name)
            if self._handler not in logger.handlers:
                logger.addHandler(self._handler)

    def _detach_handler(self) -> None:
        if self._handler is not None:
            logger = logging.getLogger(self._logger_name)
            if self._handler in logger.handlers:
                logger.removeHandler(self._handler)

    @staticmethod
    def _status_style(status: TaskStatus) -> str:
        return {
            TaskStatus.INITIATED: "dim",
            TaskStatus.RUNNING: "yellow",
            TaskStatus.READY: "cyan",
            TaskStatus.FINALIZING: "cyan",
            TaskStatus.COMPLETED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.TIMEOUT: "red",
            TaskStatus.CANCELLED: "magenta",
        }.get(status, "white")

    @staticmethod
    def _status_display(task: Task) -> str:
        """Status text with the live sub-status appended while RUNNING.

        e.g. ``RUNNING (Pending: Unschedulable)`` for a k8s task whose pod is still
        scheduling. Only shown while RUNNING so a stale note never lingers on a
        terminal status.
        """
        detail = getattr(task, "status_detail", None)
        if detail and task.status == TaskStatus.RUNNING:
            return f"{task.status} ({detail})"
        return str(task.status)

    def _build_task_table(self, tasks: Iterable[Task]) -> Table:
        t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        t.add_column("Task", overflow="fold", no_wrap=False)
        t.add_column("Status", justify="left")
        t.add_column("Exit", justify="right", width=4)
        t.add_column("Nodes", overflow="fold")

        for task in tasks:
            status = task.status
            status_text = Text(
                self._status_display(task), style=self._status_style(status)
            )

            exit_code = getattr(task, "exit_code", None)
            exit_str = "" if exit_code is None else str(int(exit_code))

            nodes = getattr(task, "assigned_nodes", None) or []
            if isinstance(nodes, str):
                nodes_str = nodes
            else:
                nodes_str = ",".join(str(node) for node in nodes)

            t.add_row(task.name, status_text, exit_str, nodes_str)
        return t

    def _ordered_tasks(self) -> list[Task]:
        if self._workflow is None:
            return []
        try:
            return [
                self._workflow.get_task(name)
                for name in self._workflow.task_graph.dag.topological_sort()
            ]
        except Exception:
            return self._workflow.get_tasks()

    def _backend_rows(self, tasks: list[Task]) -> list[tuple[str, str, str, str, str]]:
        # Best-effort backend allocation summary derived from tasks / operator configs.
        # - backend name: task.backend_name (assembly populates it)
        # - allocation id: for srun operator, config.job_id
        # - nodes: prefer task.assigned_nodes, fall back to srun config.nodelist
        by_backend: dict[str, dict[str, object]] = {}

        for t in tasks:
            b = getattr(t, "backend_name", None) or "default"
            entry = by_backend.setdefault(
                b, {"tasks": 0, "alloc_ids": set(), "nodes": set()}
            )
            entry["tasks"] = int(entry["tasks"]) + 1

            # allocation id (slurm): from operator config if available
            try:
                cfg = getattr(getattr(t, "operator", None), "config", None)
                job_id = getattr(cfg, "job_id", None)
                if job_id not in (None, "", "0"):
                    entry["alloc_ids"].add(str(job_id))  # type: ignore[union-attr]
            except Exception:
                pass

            # nodes: prefer assigned_nodes; otherwise try operator config nodelist
            nodes = getattr(t, "assigned_nodes", None) or []
            if nodes:
                for n in nodes:
                    entry["nodes"].add(str(n))  # type: ignore[union-attr]
            else:
                try:
                    cfg = getattr(getattr(t, "operator", None), "config", None)
                    nodelist = getattr(cfg, "nodelist", None) or []
                    for n in nodelist:
                        entry["nodes"].add(str(n))  # type: ignore[union-attr]
                except Exception:
                    pass

        rows = []
        for b, info in sorted(by_backend.items(), key=lambda x: x[0]):
            alloc_ids = sorted(list(info["alloc_ids"]))  # type: ignore[arg-type]
            nodes = sorted(list(info["nodes"]))  # type: ignore[arg-type]
            alloc = ",".join(alloc_ids) if alloc_ids else ""
            nodes_str = ",".join(nodes)
            rows.append((b, alloc, str(len(nodes)), str(info["tasks"]), nodes_str))
        return rows

    @staticmethod
    def _level_style(levelno: int) -> str:
        if levelno >= logging.CRITICAL:
            return "bold red"
        if levelno >= logging.ERROR:
            return "red"
        if levelno >= logging.WARNING:
            return "yellow"
        if levelno >= logging.INFO:
            return "green"
        return "dim"

    def _record_to_text(self, rec: logging.LogRecord) -> Text:
        ts = datetime.fromtimestamp(getattr(rec, "created", time.time())).strftime(
            "%H:%M:%S"
        )
        level = getattr(rec, "levelname", "INFO")
        name = getattr(rec, "name", "")
        try:
            msg = rec.getMessage()
        except Exception:
            msg = str(getattr(rec, "msg", ""))

        line = Text.assemble(
            (ts, "dim"),
            " ",
            (f"{level:<8}", self._level_style(getattr(rec, "levelno", logging.INFO))),
            " ",
            (f"{name}:", "cyan"),
            " ",
            (msg, ""),
        )

        if getattr(rec, "exc_info", None):
            try:
                exc_text = logging.Formatter().formatException(rec.exc_info)  # type: ignore[arg-type]
                line.append("\n")
                line.append(exc_text, style="red")
            except Exception:
                pass
        return line

    def _done_count(self, tasks: list[Task]) -> tuple[int, int, dict[str, int]]:
        total = len(tasks)
        counts: dict[str, int] = {}
        for t in tasks:
            k = str(t.status)
            counts[k] = counts.get(k, 0) + 1
        done = sum(
            counts.get(k, 0)
            for k in ("READY", "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED")
        )
        return done, total, counts

    def _header_text(self) -> str:
        elapsed = time.time() - self._start_time
        tasks = list(self._workflow.get_tasks()) if self._workflow is not None else []
        done, total, counts = self._done_count(tasks)

        def _bar(done_n: int, total_n: int, width: int = 22) -> str:
            if total_n <= 0:
                return "░" * width
            ratio = max(0.0, min(1.0, done_n / total_n))
            filled = int(round(ratio * width))
            return ("█" * filled) + ("░" * (width - filled))

        # Best-effort: show workflow output dir if present in task envs.
        out_dir = ""
        for t in tasks:
            out_dir = t.envs.get("SFLOW_WORKFLOW_OUTPUT_DIR", "") or ""
            if out_dir:
                break
        run_id = ""
        if out_dir:
            try:
                run_id = out_dir.rstrip("/").split("/")[-1]
            except Exception:
                run_id = ""
        workflow_name = self._workflow.name if self._workflow is not None else self._workflow_name
        counts_text = "  ".join(
            [
                f"RUNNING {counts.get('RUNNING', 0)}",
                f"FINALIZING {counts.get('FINALIZING', 0)}",
                f"READY {counts.get('READY', 0)}",
                f"FAILED {counts.get('FAILED', 0)}",
                f"CANCELLED {counts.get('CANCELLED', 0)}",
            ]
        )
        return "\n".join(
            [
                f"sflow | workflow: {workflow_name}" + (f" | run: {run_id}" if run_id else ""),
                f"{_bar(done, total)}  {done}/{total} done",
                counts_text,
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | elapsed {elapsed:.1f}s",
                f"output: {out_dir}" if out_dir else "",
            ]
        )

    def refresh(self) -> None:
        """Re-render current workflow state + log tail."""
        if self._app.is_running:
            try:
                self._app.refresh_from_owner()
            except Exception:
                try:
                    self._app.call_from_thread(self._app.refresh_from_owner)
                except Exception:
                    pass

class _SflowTextualApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    #header {
        /* _header_text() renders 5 lines and the round border costs 2 rows, so
           anything less than 7 silently clips the last lines -- which is how the
           elapsed clock and the output dir went missing. */
        height: 7;
        border: round gray;
        padding: 0 1;
    }
    #body {
        layout: horizontal;
        height: 1fr;
    }
    #left {
        width: 56;
        layout: vertical;
    }
    #tasks {
        height: 1fr;
        border: round green;
    }
    #backends {
        height: 6;
        border: round magenta;
    }
    #logs {
        width: 1fr;
        border: round blue;
    }
    """

    BINDINGS = [
        ("ctrl+c", "interrupt", "cancel run"),
        ("q", "quit", "quit"),
        ("tab", "focus_next", "focus next"),
    ]

    def __init__(self, owner: RichTui):
        super().__init__()
        self._owner = owner
        self._last_rendered_log_record: logging.LogRecord | None = None
        self._task_rows_snapshot: (
            tuple[tuple[str, str, str, str, str], ...] | None
        ) = None
        self._backend_rows_snapshot: (
            tuple[tuple[str, str, str, str, str], ...] | None
        ) = None

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield DataTable(id="tasks")
                yield DataTable(id="backends")
            yield RichLog(
                id="logs",
                max_lines=self._owner._config.max_log_lines,
                wrap=True,
                highlight=False,
                markup=False,
            )

    def on_mount(self) -> None:
        self.query_one("#logs", RichLog).focus()
        self.refresh_from_owner(force=True)
        # The header carries a wall clock and an elapsed counter, but refreshes
        # are event-driven -- with no task transitions or log lines the clock
        # would sit frozen. Tick the header (only) once a second.
        self.set_interval(1.0, self._update_header)

    def action_interrupt(self) -> None:
        self._owner._request_interrupt()
        self.exit(return_code=130)

    def refresh_from_owner(self, *, force: bool = False) -> None:
        self._update_header()
        tasks = self._owner._ordered_tasks()
        self._update_tasks(tasks, force=force)
        self._update_backends(tasks, force=force)
        self._update_logs(force=force)

    def _update_header(self) -> None:
        # Guarded here rather than at each call site: the 1s timer can land after
        # the widget is gone (teardown after exit()), and an exception inside a
        # Textual callback surfaces in the UI. Same best-effort stance as
        # RichTui.refresh().
        try:
            self.query_one("#header", Static).update(self._owner._header_text())
        except Exception:
            pass

    def _task_rows(
        self, tasks: list[Task]
    ) -> tuple[tuple[str, str, str, str, str], ...]:
        rows = []
        for task in tasks:
            exit_code = getattr(task, "exit_code", None)
            exit_str = "" if exit_code is None else str(int(exit_code))
            nodes = getattr(task, "assigned_nodes", None) or []
            if isinstance(nodes, str):
                nodes_str = nodes
            else:
                nodes_str = ",".join(str(node) for node in nodes)
            # (name, status value [for styling], status display [with sub-status],
            # exit, nodes). The display is snapshotted too, so a sub-status change
            # triggers a re-render.
            rows.append(
                (
                    task.name,
                    str(task.status),
                    self._owner._status_display(task),
                    exit_str,
                    nodes_str,
                )
            )
        return tuple(rows)

    def _update_tasks(self, tasks: list[Task], *, force: bool = False) -> None:
        rows = self._task_rows(tasks)
        if not force and rows == self._task_rows_snapshot:
            return
        self._task_rows_snapshot = rows

        table = self.query_one("#tasks", DataTable)
        table.clear(columns=True)
        table.add_columns("Task", "Status", "Exit", "Nodes")

        for name, status_value, status_display, exit_str, nodes_str in rows:
            table.add_row(
                name,
                Text(
                    status_display,
                    style=self._owner._status_style(TaskStatus(status_value)),
                ),
                exit_str,
                nodes_str,
            )

    def _update_backends(self, tasks: list[Task], *, force: bool = False) -> None:
        rows = tuple(self._owner._backend_rows(tasks))
        if not force and rows == self._backend_rows_snapshot:
            return
        self._backend_rows_snapshot = rows

        table = self.query_one("#backends", DataTable)
        table.clear(columns=True)
        table.add_columns("Backend", "Alloc", "Node Count", "Tasks", "Nodes")
        for row in rows:
            table.add_row(*row)

    def _update_logs(self, *, force: bool = False) -> None:
        log = self.query_one("#logs", RichLog)
        records = list(self._owner._logs)
        start_index = 0

        if force:
            log.clear()
        elif self._last_rendered_log_record is not None:
            for idx, record in enumerate(records):
                if record is self._last_rendered_log_record:
                    start_index = idx + 1
                    break
            else:
                # The bounded deque rolled over since the last refresh.
                log.clear()
                start_index = 0

        # Auto-follow the tail only when the view is already pinned to the bottom
        # (or on a full refresh). If the user scrolled up to read, keep their
        # position as new log lines arrive instead of yanking back to the end.
        try:
            at_bottom = log.scroll_offset.y >= log.max_scroll_y
        except Exception:
            at_bottom = True
        follow = force or at_bottom

        for record in records[start_index:]:
            log.write(self._owner._record_to_text(record), scroll_end=follow)
        self._last_rendered_log_record = records[-1] if records else None


def maybe_rich_tui(
    enabled: bool,
    workflow: Workflow,
    *,
    tail_log_lines: int | None = None,
    log_buffer: Deque[logging.LogRecord] | None = None,
    log_lock: threading.Lock | None = None,
    attach_log_handler: bool = True,
) -> AbstractContextManager[RichTui] | nullcontext:
    if not enabled:
        return nullcontext()
    cfg = RichTuiConfig()
    if tail_log_lines is not None:
        cfg.tail_log_lines = int(tail_log_lines)
    return RichTui(
        workflow,
        config=cfg,
        log_buffer=log_buffer,
        log_lock=log_lock,
        attach_log_handler=attach_log_handler,
    )


def attach_tui_log_buffer(
    log_buffer: Deque[logging.LogRecord],
    *,
    logger_name: str = "sflow",
    level: int = logging.DEBUG,
    log_lock: threading.Lock | None = None,
) -> logging.Handler:
    """Attach a handler that appends LogRecords to `log_buffer`."""
    h = _TuiLogHandler(log_buffer, level=level, log_lock=log_lock)
    logging.getLogger(logger_name).addHandler(h)
    return h


def detach_tui_log_buffer(
    handler: logging.Handler,
    *,
    logger_name: str = "sflow",
) -> None:
    logger = logging.getLogger(logger_name)
    if handler in logger.handlers:
        logger.removeHandler(handler)
