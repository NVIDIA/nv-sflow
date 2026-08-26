import logging
import asyncio
import threading
import time
from collections import deque
from io import StringIO

from rich.console import Console
from rich.text import Text
from textual.coordinate import Coordinate
from textual.widgets import RichLog

from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from pathlib import Path

from sflow.ui import rich_tui as rich_tui_module
from sflow.ui.rich_tui import RichTui, RichTuiConfig, _SflowTextualApp


def _task(name: str) -> Task:
    return Task(
        name=name,
        logger=logging.getLogger(f"sflow.task.{name}"),
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )


def test_rich_tui_task_panel_uses_dag_order():
    tg = TaskGraph()
    prepare = _task("z_prepare")
    train = _task("a_train")
    evaluate = _task("m_evaluate")
    tg.dag.add_node(prepare.name, prepare)
    tg.dag.add_node(train.name, train)
    tg.dag.add_node(evaluate.name, evaluate)
    tg.dag.add_edge(prepare.name, train.name)
    tg.dag.add_edge(train.name, evaluate.name)

    tui = RichTui(
        Workflow(name="wf", task_graph=tg),
        attach_log_handler=False,
    )

    assert [task.name for task in tui._ordered_tasks()] == [
        "z_prepare",
        "a_train",
        "m_evaluate",
    ]


def test_rich_tui_header_counts_ready_tasks_as_done():
    tg = TaskGraph()
    service = _task("service")
    worker = _task("worker")
    service.status = TaskStatus.READY
    worker.status = TaskStatus.RUNNING
    tg.dag.add_node(service.name, service)
    tg.dag.add_node(worker.name, worker)
    tg.dag.add_edge(service.name, worker.name)

    tui = RichTui(
        Workflow(name="wf", task_graph=tg),
        attach_log_handler=False,
    )

    plain = tui._header_text()

    assert "1/2 done" in plain
    assert "READY 1" in plain


def test_rich_tui_header_counts_finalizing_tasks_as_active():
    tg = TaskGraph()
    finalizing = _task("finalizing")
    waiting = _task("waiting")
    finalizing.status = TaskStatus.FINALIZING
    waiting.status = TaskStatus.INITIATED
    tg.dag.add_node(finalizing.name, finalizing)
    tg.dag.add_node(waiting.name, waiting)
    tg.dag.add_edge(finalizing.name, waiting.name)

    tui = RichTui(
        Workflow(name="wf", task_graph=tg),
        attach_log_handler=False,
    )

    plain = tui._header_text()

    assert "0/2 done" in plain
    assert "FINALIZING 1" in plain


def test_rich_tui_backend_rows_include_unique_node_count():
    task_a = _task("task_a")
    task_a.backend_name = "slurm"
    task_a.assigned_nodes = ["node-1", "node-2"]
    task_b = _task("task_b")
    task_b.backend_name = "slurm"
    task_b.assigned_nodes = ["node-2", "node-3"]
    tui = RichTui(None, attach_log_handler=False)

    assert tui._backend_rows([task_a, task_b]) == [
        ("slurm", "", "3", "2", "node-1,node-2,node-3")
    ]


def test_rich_tui_task_rows_display_full_node_list():
    task = _task("task_a")
    task.assigned_nodes = ["node-1", "node-2", "node-3"]
    tui = RichTui(None, attach_log_handler=False)

    # Rows carry (name, status_value, status_display, exit, nodes); with no
    # sub-status the display equals the raw status value.
    assert tui._app._task_rows([task]) == (
        ("task_a", "INITIATED", "INITIATED", "", "node-1,node-2,node-3"),
    )


def test_rich_tui_task_rows_append_substatus_for_running_task():
    task = _task("decode")
    task.status = TaskStatus.RUNNING
    task.status_detail = "Pending: Unschedulable"
    tui = RichTui(None, attach_log_handler=False)

    assert tui._app._task_rows([task]) == (
        ("decode", "RUNNING", "RUNNING (Pending: Unschedulable)", "", ""),
    )


def test_rich_tui_task_rows_ignore_substatus_when_not_running():
    task = _task("decode")
    task.status = TaskStatus.COMPLETED
    task.status_detail = "Pending: Unschedulable"  # stale; only shown while RUNNING
    tui = RichTui(None, attach_log_handler=False)

    rows = tui._app._task_rows([task])
    assert rows[0][2] == "COMPLETED"


def test_rich_tui_default_refresh_matches_cli_default():
    assert RichTuiConfig().refresh_per_second == 2


def _log_record(message: str) -> logging.LogRecord:
    record = logging.LogRecord(
        name="sflow.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.created = time.mktime((2026, 5, 21, 14, 0, 0, 0, 0, -1))
    return record

def test_rich_tui_uses_textual_richlog_for_native_scrolling():
    logs = deque(_log_record(f"line {i}") for i in range(8))
    tui = RichTui(
        None,
        console=Console(width=100, height=20),
        config=RichTuiConfig(tail_log_lines=3),
        log_buffer=logs,
        attach_log_handler=False,
    )

    async def _run_app():
        async with tui._app.run_test() as _pilot:
            log = tui._app.query_one("#logs", RichLog)
            assert log.allow_vertical_scroll is True
            tui._app.refresh_from_owner(force=True)
            assert len(log.lines) >= 8

    asyncio.run(_run_app())


def test_rich_tui_bounds_textual_richlog_memory():
    tui = RichTui(
        None,
        console=Console(width=100, height=20),
        config=RichTuiConfig(max_log_lines=7),
        attach_log_handler=False,
    )

    async def _run_app():
        async with tui._app.run_test() as _pilot:
            log = tui._app.query_one("#logs", RichLog)
            assert log.max_lines == 7

    asyncio.run(_run_app())


def test_rich_tui_log_rollover_continues_rendering_new_records():
    logs = deque((_log_record(f"line {i}") for i in range(3)), maxlen=3)
    tui = RichTui(
        None,
        console=Console(width=100, height=20),
        log_buffer=logs,
        attach_log_handler=False,
    )

    async def _run_app():
        async with tui._app.run_test() as _pilot:
            log = tui._app.query_one("#logs", RichLog)
            tui._app.refresh_from_owner(force=True)
            logs.append(_log_record("line 3"))
            tui._app.refresh_from_owner()
            rendered = "\n".join(str(line) for line in log.lines)
            assert "line 3" in rendered

    asyncio.run(_run_app())


def test_rich_tui_handler_and_renderer_share_log_lock():
    lock = threading.Lock()
    tui = RichTui(None, log_lock=lock, attach_log_handler=True)

    assert tui._logs_lock is lock
    assert tui._handler is not None
    assert tui._handler._lock is lock


def test_rich_tui_skips_unchanged_task_and_backend_table_rebuilds(monkeypatch):
    task = _task("task_a")
    task.backend_name = "local"
    task.status = TaskStatus.RUNNING
    tg = TaskGraph()
    tg.dag.add_node(task.name, task)
    tui = RichTui(
        Workflow(name="wf", task_graph=tg),
        console=Console(width=100, height=20),
        attach_log_handler=False,
    )

    async def _run_app():
        async with tui._app.run_test() as _pilot:
            tui._app.refresh_from_owner(force=True)
            task_table = tui._app.query_one("#tasks")
            backend_table = tui._app.query_one("#backends")
            calls = {"tasks": 0, "backends": 0}
            original_task_clear = task_table.clear
            original_backend_clear = backend_table.clear

            def _task_clear(*args, **kwargs):
                calls["tasks"] += 1
                return original_task_clear(*args, **kwargs)

            def _backend_clear(*args, **kwargs):
                calls["backends"] += 1
                return original_backend_clear(*args, **kwargs)

            monkeypatch.setattr(task_table, "clear", _task_clear)
            monkeypatch.setattr(backend_table, "clear", _backend_clear)

            tui._app.refresh_from_owner()
            tui._app.refresh_from_owner()

            assert calls == {"tasks": 0, "backends": 0}

    asyncio.run(_run_app())


def test_rich_tui_task_status_cells_use_status_colors():
    task = _task("task_a")
    task.status = TaskStatus.FAILED
    tg = TaskGraph()
    tg.dag.add_node(task.name, task)
    tui = RichTui(
        Workflow(name="wf", task_graph=tg),
        console=Console(width=100, height=20),
        attach_log_handler=False,
    )

    async def _run_app():
        async with tui._app.run_test() as _pilot:
            tui._app.refresh_from_owner(force=True)
            table = tui._app.query_one("#tasks")
            status_cell = table.get_cell_at(Coordinate(0, 1))
            assert isinstance(status_cell, Text)
            assert status_cell.plain == "FAILED"
            assert status_cell.style == "red"

    asyncio.run(_run_app())


def test_rich_tui_task_status_cell_shows_pod_substatus_while_running():
    task = _task("decode")
    task.status = TaskStatus.RUNNING
    task.status_detail = "Pending: Unschedulable"
    tg = TaskGraph()
    tg.dag.add_node(task.name, task)
    tui = RichTui(
        Workflow(name="wf", task_graph=tg),
        console=Console(width=120, height=20),
        attach_log_handler=False,
    )

    async def _run_app():
        async with tui._app.run_test() as _pilot:
            tui._app.refresh_from_owner(force=True)
            table = tui._app.query_one("#tasks")
            status_cell = table.get_cell_at(Coordinate(0, 1))
            assert isinstance(status_cell, Text)
            assert status_cell.plain == "RUNNING (Pending: Unschedulable)"
            # Still styled by the underlying RUNNING status.
            assert status_cell.style == "yellow"

    asyncio.run(_run_app())


def test_rich_tui_task_status_cells_style_finalizing():
    task = _task("task_a")
    task.status = TaskStatus.FINALIZING
    tg = TaskGraph()
    tg.dag.add_node(task.name, task)
    tui = RichTui(
        Workflow(name="wf", task_graph=tg),
        console=Console(width=100, height=20),
        attach_log_handler=False,
    )

    async def _run_app():
        async with tui._app.run_test() as _pilot:
            tui._app.refresh_from_owner(force=True)
            table = tui._app.query_one("#tasks")
            status_cell = table.get_cell_at(Coordinate(0, 1))
            assert isinstance(status_cell, Text)
            assert status_cell.plain == "FINALIZING"
            assert status_cell.style == "cyan"

    asyncio.run(_run_app())


def test_rich_tui_backend_table_orders_counts_before_full_nodes():
    task_a = _task("task_a")
    task_a.backend_name = "slurm"
    task_a.assigned_nodes = ["node-1", "node-2"]
    task_b = _task("task_b")
    task_b.backend_name = "slurm"
    task_b.assigned_nodes = ["node-3"]
    tg = TaskGraph()
    tg.dag.add_node(task_a.name, task_a)
    tg.dag.add_node(task_b.name, task_b)
    tui = RichTui(
        Workflow(name="wf", task_graph=tg),
        console=Console(width=100, height=20),
        attach_log_handler=False,
    )

    async def _run_app():
        async with tui._app.run_test() as _pilot:
            tui._app.refresh_from_owner(force=True)
            table = tui._app.query_one("#backends")
            assert table.get_cell_at(Coordinate(0, 0)) == "slurm"
            assert table.get_cell_at(Coordinate(0, 1)) == ""
            assert table.get_cell_at(Coordinate(0, 2)) == "3"
            assert table.get_cell_at(Coordinate(0, 3)) == "2"
            assert table.get_cell_at(Coordinate(0, 4)) == "node-1,node-2,node-3"

    asyncio.run(_run_app())


def test_rich_tui_context_manager_starts_and_stops_headless_textual_app():
    tui = RichTui(
        None,
        console=Console(file=StringIO(), force_terminal=False),
        attach_log_handler=False,
    )

    with tui as active:
        assert active is tui
        assert tui._app_thread is not None

    assert tui._app_thread is None


def test_rich_tui_async_start_runs_textual_without_worker_thread():
    tui = RichTui(
        None,
        console=Console(file=StringIO(), force_terminal=False),
        attach_log_handler=False,
    )

    async def _start_stop():
        await tui.start_async()
        assert tui._app_task is not None
        assert tui._app_thread is None
        await tui.stop_async()

    asyncio.run(_start_stop())
    assert tui._app_task is None


def test_rich_tui_refresh_updates_directly_when_called_on_textual_loop():
    class _FakeApp:
        is_running = True

        def __init__(self):
            self.refreshed = 0
            self.thread_calls = 0

        def refresh_from_owner(self):
            self.refreshed += 1

        def call_from_thread(self, callback):
            self.thread_calls += 1
            raise RuntimeError("The `call_from_thread` method must run in a different thread from the app")

    tui = RichTui(None, attach_log_handler=False)
    fake_app = _FakeApp()
    tui._app = fake_app

    tui.refresh()

    assert fake_app.refreshed == 1
    assert fake_app.thread_calls == 0


def test_rich_tui_binds_ctrl_c_to_interrupt_action():
    actions = {binding[0]: binding[1] for binding in _SflowTextualApp.BINDINGS}

    assert actions.get("ctrl+c") == "interrupt"


def test_rich_tui_interrupt_action_notifies_owner():
    events = []
    tui = RichTui(None, attach_log_handler=False)
    tui.set_interrupt_handler(lambda: events.append("interrupt"))

    tui._app.action_interrupt()

    assert events == ["interrupt"]


def test_rich_tui_ctrl_c_key_requests_interrupt():
    events = []
    tui = RichTui(
        None,
        console=Console(file=StringIO(), force_terminal=False),
        attach_log_handler=False,
    )
    tui.set_interrupt_handler(lambda: events.append("interrupt"))

    async def _run_app():
        async with tui._app.run_test() as pilot:
            await pilot.press("ctrl+c")

    asyncio.run(_run_app())

    assert events == ["interrupt"]


def test_header_box_is_tall_enough_for_every_rendered_line():
    """A short #header silently CLIPS its last lines rather than scrolling.

    The elapsed clock is the 4th of 5 rendered lines, so a height that forgot to
    budget for the round border (2 rows) hid it completely -- the box looked fine
    and the text was simply gone.
    """
    import re as _re

    tui = RichTui(None, attach_log_handler=False)
    rendered = tui._header_text().split("\n")
    assert any("elapsed" in line for line in rendered), rendered

    height = int(
        _re.search(r"#header \{.*?height: (\d+);", _SflowTextualApp.CSS, _re.S).group(1)
    )
    # +2 for the round border's top and bottom rows.
    assert height >= len(rendered) + 2, (height, len(rendered))


def test_header_is_ticked_on_a_timer_not_only_on_events():
    """Elapsed must keep advancing while no task transitions and no logs arrive."""
    source = Path(rich_tui_module.__file__).read_text(encoding="utf-8")
    on_mount = source.split("def on_mount")[1].split("    def ")[0]
    assert "set_interval" in on_mount and "_update_header" in on_mount, on_mount
