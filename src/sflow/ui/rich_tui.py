# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Iterable

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sflow.core.task import Task, TaskStatus
from sflow.core.workflow import Workflow


class _TuiLogHandler(logging.Handler):
    """A logging handler that appends LogRecords to a deque."""

    def __init__(self, sink: Deque[logging.LogRecord], *, level: int = logging.INFO):
        super().__init__(level=level)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Copy to avoid surprises if other handlers mutate the record.
            self._sink.append(logging.makeLogRecord(record.__dict__.copy()))
        except Exception:
            # Best-effort: never break the workflow due to UI.
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
    refresh_per_second: int = 10


class RichTui(AbstractContextManager["RichTui"]):
    """
    Rich terminal UI:
    - Left: task table (name/status/attempt/backend/next retry)
    - Right: scrolling log tail (last N lines)
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
        self._handler: _TuiLogHandler | None = None
        if self._attach_log_handler:
            self._handler = _TuiLogHandler(self._logs, level=logging.DEBUG)
            # We render LogRecords ourselves in the panel with Rich styles.

        self._layout = self._build_layout()
        self._live: Live | None = None

        self._start_time = time.time()

    @property
    def workflow(self) -> Workflow | None:
        return self._workflow

    def set_workflow(self, workflow: Workflow) -> None:
        self._workflow = workflow
        if not self._workflow_name:
            self._workflow_name = workflow.name

    def __enter__(self) -> "RichTui":
        # Attach a log capture handler to the sflow logger (optional; caller may attach earlier).
        if self._handler is not None:
            logger = logging.getLogger(self._logger_name)
            logger.addHandler(self._handler)

        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=self._config.refresh_per_second,
            transient=False,
        )
        self._live.__enter__()
        # Render an initial frame immediately.
        self.refresh()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handler is not None:
            logger = logging.getLogger(self._logger_name)
            try:
                if self._handler in logger.handlers:
                    logger.removeHandler(self._handler)
            finally:
                if self._live is not None:
                    self._live.__exit__(exc_type, exc, tb)
                    self._live = None
        else:
            if self._live is not None:
                self._live.__exit__(exc_type, exc, tb)
                self._live = None

    @staticmethod
    def _status_style(status: TaskStatus) -> str:
        return {
            TaskStatus.INITIATED: "dim",
            TaskStatus.RUNNING: "yellow",
            TaskStatus.READY: "cyan",
            TaskStatus.COMPLETED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.TIMEOUT: "red",
            TaskStatus.CANCELLED: "magenta",
        }.get(status, "white")

    def _build_task_table(self, tasks: Iterable[Task]) -> Table:
        t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        t.add_column("Task", overflow="fold", no_wrap=False)
        t.add_column("Status", justify="left")
        t.add_column("Exit", justify="right", width=4)
        t.add_column("Nodes", overflow="fold")

        for task in tasks:
            status = task.status
            status_text = Text(str(status), style=self._status_style(status))

            exit_code = getattr(task, "exit_code", None)
            exit_str = "" if exit_code is None else str(int(exit_code))

            nodes = getattr(task, "assigned_nodes", None) or []
            if isinstance(nodes, str):
                nodes_str = nodes
            else:
                nodes_list = list(nodes)
                if len(nodes_list) <= 2:
                    nodes_str = ",".join(nodes_list)
                else:
                    nodes_str = (
                        f"{nodes_list[0]},{nodes_list[1]},…(+{len(nodes_list) - 2})"
                    )

            t.add_row(task.name, status_text, exit_str, nodes_str)
        return t

    def _build_backend_panel(self, tasks: list[Task]) -> Panel:
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

        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column("Backend")
        table.add_column("Alloc", overflow="fold")
        table.add_column("Nodes", overflow="fold")
        table.add_column("Tasks", justify="right", width=5)

        for b, info in sorted(by_backend.items(), key=lambda x: x[0]):
            alloc_ids = sorted(list(info["alloc_ids"]))  # type: ignore[arg-type]
            nodes = sorted(list(info["nodes"]))  # type: ignore[arg-type]
            alloc = ",".join(alloc_ids) if alloc_ids else ""
            if len(nodes) <= 2:
                nodes_str = ",".join(nodes)
            elif nodes:
                nodes_str = f"{nodes[0]},{nodes[1]},…(+{len(nodes) - 2})"
            else:
                nodes_str = ""
            table.add_row(b, alloc, nodes_str, str(info["tasks"]))

        return Panel(
            table, title=self._config.backend_panel_title, border_style="magenta"
        )

    def _build_log_panel(self) -> Panel:
        # Show only the last N *visual* lines (after wrapping to pane width) to simulate scrolling.
        # If configured tail <= 0, auto-fit to the right panel height.

        # Approximate available content width/height in the right pane.
        # - console width includes everything
        # - left pane is fixed-width
        # - panel borders/padding take a few extra columns
        try:
            total_w = int(getattr(self._console.size, "width", 0) or 0)
        except Exception:
            total_w = 0

        try:
            total_h = int(getattr(self._console.size, "height", 0) or 0)
        except Exception:
            total_h = 0

        # Layout uses a fixed header size.
        body_h = max(1, total_h - int(self._config.header_height)) if total_h > 0 else 0
        # Panel border consumes 2 rows.
        inner_h = max(1, body_h - 2) if body_h > 0 else 0

        cfg_tail = int(self._config.tail_log_lines)
        if cfg_tail <= 0 and inner_h > 0:
            tail_n = max(10, inner_h)
        else:
            tail_n = max(10, cfg_tail)

        if total_w <= 0:
            # Fallback width; will still behave well.
            content_w = 80
        else:
            content_w = max(20, total_w - int(self._config.left_width) - 6)

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

        def _record_to_text(rec: logging.LogRecord) -> Text:
            ts = datetime.fromtimestamp(getattr(rec, "created", time.time())).strftime(
                "%H:%M:%S"
            )
            level = getattr(rec, "levelname", "INFO")
            name = getattr(rec, "name", "")
            msg = ""
            try:
                msg = rec.getMessage()
            except Exception:
                msg = str(getattr(rec, "msg", ""))

            line = Text.assemble(
                (ts, "dim"),
                " ",
                (f"{level:<8}", _level_style(getattr(rec, "levelno", logging.INFO))),
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

        # Build tail *visual* lines from the end, bounded by tail_n.
        tail_lines: Deque[Text] = deque()
        for rec in reversed(self._logs):
            t = _record_to_text(rec)
            # Split explicit newlines, then wrap each line.
            for t_line in reversed(t.split("\n")):
                wrapped = t_line.wrap(self._console, width=content_w, overflow="fold")
                for w in reversed(wrapped):
                    tail_lines.appendleft(w)
                    if len(tail_lines) >= tail_n:
                        break
                if len(tail_lines) >= tail_n:
                    break
            if len(tail_lines) >= tail_n:
                break

        body_renderable = Group(*tail_lines) if tail_lines else Text("")
        # Important: Align.bottom makes the view "auto scroll" to the newest lines.
        return Panel(
            Align(body_renderable, vertical="bottom"),
            title=self._config.log_panel_title,
            border_style="blue",
        )

    def _build_header(self) -> Panel:
        elapsed = time.time() - self._start_time

        tasks = list(self._workflow.get_tasks()) if self._workflow is not None else []
        total = len(tasks)
        counts: dict[str, int] = {}
        for t in tasks:
            k = str(t.status)
            counts[k] = counts.get(k, 0) + 1

        done = sum(
            counts.get(k, 0) for k in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED")
        )

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

        # Header layout: 2-column grid.
        grid = Table.grid(expand=True)
        grid.add_column(ratio=2)
        grid.add_column(justify="right", ratio=3)

        left = Group(
            Text("sflow", style="bold cyan"),
            Text(
                f"workflow: {(self._workflow.name if self._workflow is not None else self._workflow_name)}"
                + (f"  |  run: {run_id}" if run_id else ""),
                style="bold",
            ),
        )

        right = Group(
            Text(
                f"{_bar(done, total)}  {done}/{total} done",
                style="green" if counts.get("FAILED", 0) == 0 else "yellow",
            ),
            Text(
                "  ".join(
                    [
                        f"RUNNING {counts.get('RUNNING', 0)}",
                        f"READY {counts.get('READY', 0)}",
                        f"FAILED {counts.get('FAILED', 0)}",
                        f"CANCELLED {counts.get('CANCELLED', 0)}",
                    ]
                ),
                style="dim",
            ),
            Text(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  elapsed {elapsed:.1f}s",
                style="dim",
            ),
        )

        grid.add_row(left, right)

        if out_dir:
            extra = Text(f"output: {out_dir}", style="dim")
            return Panel(Group(grid, extra), border_style="dim")

        return Panel(grid, border_style="dim")

    def _build_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=int(self._config.header_height)),
            Layout(name="body", ratio=1),
        )
        layout["body"].split_row(
            Layout(name="left", size=self._config.left_width),
            Layout(name="right", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="left_tasks", ratio=1),
            Layout(name="left_backends", size=int(self._config.backend_panel_height)),
        )
        return layout

    def refresh(self) -> None:
        """Re-render current workflow state + log tail."""
        # Header
        self._layout["header"].update(self._build_header())

        # Tasks (sorted by name for stable display)
        tasks = (
            sorted(self._workflow.get_tasks(), key=lambda x: x.name)
            if self._workflow is not None
            else []
        )
        table = self._build_task_table(tasks)
        self._layout["left_tasks"].update(
            Panel(table, title=self._config.task_panel_title, border_style="green")
        )
        self._layout["left_backends"].update(self._build_backend_panel(tasks))

        # Logs
        self._layout["right"].update(self._build_log_panel())

        # Force a refresh if Live is active.
        if self._live is not None:
            self._live.refresh()


def maybe_rich_tui(
    enabled: bool,
    workflow: Workflow,
    *,
    tail_log_lines: int | None = None,
    log_buffer: Deque[logging.LogRecord] | None = None,
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
        attach_log_handler=attach_log_handler,
    )


def attach_tui_log_buffer(
    log_buffer: Deque[logging.LogRecord],
    *,
    logger_name: str = "sflow",
    level: int = logging.DEBUG,
) -> logging.Handler:
    """Attach a handler that appends LogRecords to `log_buffer`."""
    h = _TuiLogHandler(log_buffer, level=level)
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
