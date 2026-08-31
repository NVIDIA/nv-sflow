# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import re
from pathlib import Path

import pytest
import sflow.core.execution_summary as execution_summary_mod
from sflow.core.execution_summary import (
    SflowSummaryWriter,
    _resource_row_key,
)
from sflow.core.task import Task, TaskStatus
from sflow.core.uploads import UploadResult
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
import gc


def _task(name: str, tmp_path: Path) -> Task:
    task = Task(
        name=name,
        logger=logging.getLogger(f"sflow.task.{name}"),
        operator=BashOperator(BashOperatorConfig(name="bash")),
        script=[f"echo {name}"],
    )
    task.backend_name = "local"
    task.operator_name = "bash"
    task.envs["SFLOW_TASK_OUTPUT_DIR"] = str(tmp_path / name)
    return task


def _section_lines(text: str, start: str, end: str) -> list[str]:
    lines = text.splitlines()
    start_idx = lines.index(start)
    end_idx = lines.index(end)
    return lines[start_idx + 2 : end_idx - 1]


def _top_summary_text(text: str) -> str:
    return text.split("\n\nRuntime", 1)[0]


def _section_index(text: str, title: str) -> int:
    return text.splitlines().index(title)


def test_summary_writer_debounces_event_writes_inside_running_loop(tmp_path, monkeypatch):
    tg = TaskGraph()
    task = _task("server", tmp_path)
    tg.dag.add_node("server", task)
    workflow = Workflow(name="wf", task_graph=tg)

    summary_path = tmp_path / "sflow_summary.log"
    writer = SflowSummaryWriter(summary_path, debounce_interval=0.05)
    writes: list[str] = []
    original_write = writer._write_summary

    def _record_write(text, generation):
        writes.append(text)
        original_write(text, generation)

    monkeypatch.setattr(writer, "_write_summary", _record_write)

    async def _exercise() -> None:
        writer.start(
            workflow=workflow,
            output_dir=tmp_path,
            runtime_info_text="runtime",
            command_log_paths={},
        )
        writer.task_unblocked(task)
        task.attempts = 1
        writer.task_submitted(task)
        await asyncio.sleep(0.15)

    asyncio.run(_exercise())

    assert len(writes) == 1
    assert "UNBLOCKED" in summary_path.read_text()
    assert "SUBMITTED" in summary_path.read_text()


def test_summary_writer_workflow_finished_flushes_pending_debounced_write(tmp_path):
    tg = TaskGraph()
    task = _task("server", tmp_path)
    tg.dag.add_node("server", task)
    workflow = Workflow(name="wf", task_graph=tg)

    summary_path = tmp_path / "sflow_summary.log"
    writer = SflowSummaryWriter(summary_path, debounce_interval=30)

    async def _exercise() -> None:
        writer.start(
            workflow=workflow,
            output_dir=tmp_path,
            runtime_info_text="runtime",
            command_log_paths={},
        )
        task.status = TaskStatus.COMPLETED
        writer.workflow_finished(status="COMPLETED")

    asyncio.run(_exercise())

    text = summary_path.read_text()
    assert "Status   : COMPLETED" in text


def test_summary_writer_infers_running_while_task_is_finalizing(tmp_path):
    tg = TaskGraph()
    task = _task("server", tmp_path)
    task.status = TaskStatus.FINALIZING
    tg.dag.add_node("server", task)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )

    assert writer._infer_status() == "RUNNING"


def test_summary_writer_write_failure_removes_temp_file(tmp_path, monkeypatch):
    summary_path = tmp_path / "sflow_summary.log"
    temp_path = tmp_path / ".sflow_summary.log.failed"

    class _FailingTempFile:
        name = str(temp_path)

        def __enter__(self):
            temp_path.write_text("")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, text):
            raise OSError("disk full")

    monkeypatch.setattr(
        execution_summary_mod.tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: _FailingTempFile(),
    )

    writer = SflowSummaryWriter(summary_path)
    with pytest.raises(OSError, match="disk full"):
        writer._write_summary("summary text", 1)

    assert not temp_path.exists()
    assert not summary_path.exists()


def test_summary_renders_network_warnings_section(tmp_path):
    # A recorded network warning (e.g. RDMA -> TCP fallback) surfaces in a
    # dedicated section, prefixed with the task name for triage.
    tg = TaskGraph()
    server = _task("decode_server_0", tmp_path)
    tg.dag.add_node("decode_server_0", server)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="rt",
        command_log_paths={},
    )
    writer.record_network_warning(
        server, "RDMA NIC unusable in 1/1 pod(s) (all ports DOWN)"
    )
    writer.workflow_finished(status="READY")

    text = (tmp_path / "sflow_summary.log").read_text()
    assert "Network Warnings" in text
    assert (
        "decode_server_0: RDMA NIC unusable in 1/1 pod(s) (all ports DOWN)" in text
    )


def test_summary_omits_network_warnings_section_when_none(tmp_path):
    tg = TaskGraph()
    server = _task("s", tmp_path)
    tg.dag.add_node("s", server)
    workflow = Workflow(name="wf", task_graph=tg)
    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="rt",
        command_log_paths={},
    )
    writer.workflow_finished(status="READY")
    assert "Network Warnings" not in (tmp_path / "sflow_summary.log").read_text()


def test_summary_writer_renders_header_dag_timeline_chart_and_final_summary(tmp_path):
    tg = TaskGraph()
    load = _task("load", tmp_path)
    bench = _task("bench", tmp_path)
    tg.dag.add_node("load", load)
    tg.dag.add_node("bench", bench)
    tg.dag.add_edge("load", "bench")
    workflow = Workflow(name="wf", task_graph=tg)

    summary_path = tmp_path / "sflow_summary.log"
    bash_log = tmp_path / "bash_cmds.log"
    bash_log.write_text("bash command\n")
    writer = SflowSummaryWriter(summary_path)
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime info text",
        command_log_paths={
            "bash": bash_log,
            "slurm": tmp_path / "slurm_cmds.log",
        },
        command_text="sflow run -f cfg.yaml --set X=1",
    )

    writer.task_unblocked(load)
    load.attempts = 1
    writer.task_submitted(load)
    load.status = TaskStatus.COMPLETED
    load.exit_code = 0
    writer.task_completed(load)
    bench.attempts = 1
    writer.task_submitted(bench)
    bench.status = TaskStatus.FAILED
    bench.exit_code = 2
    writer.task_failed(bench, reason="process exit", exit_code=2)
    writer.workflow_finished(status="FAILED")

    text = summary_path.read_text()

    assert "Sflow Summary" in text
    assert "Workflow : wf" in text
    assert "Status   : FAILED" in text
    assert f"Summary  : {summary_path}" in text
    # The actual invocation is recorded so a later reader knows what was run.
    assert "Command  : sflow run -f cfg.yaml --set X=1" in text
    assert "Runtime" in text
    assert "runtime info text" in text
    assert "Command Logs" in text
    assert "bash_cmds.log" in text
    assert "slurm_cmds.log" not in text
    assert "Workflow DAG" in text
    assert "load" in text
    assert "bench" in text
    assert "Dependencies" in text
    assert "load -> bench" in text
    assert "Task Context" not in text
    assert "Backend/Operator" not in text
    assert "bench" in text
    assert "  log:" not in text
    assert "Timeline" in text
    assert re.search(r"Time\s+Elapsed\s+Task\s+Event\s+Summary", text)
    assert "UNBLOCKED" in text
    assert "SUBMITTED" in text
    assert "COMPLETED" in text
    assert "FAILED" in text
    assert "deps=[" not in text
    assert "deps:" not in text
    assert "next:" not in text
    assert "backend/operator:" not in text
    assert "Task Duration Chart" in text
    assert "End Summary" not in text
    assert "Final Status" not in text
    top_summary = _top_summary_text(text)
    assert "Counts       : COMPLETED=1, FAILED=1" in top_summary
    assert "FAILED/CANCELLED Tasks : bench" in top_summary
    assert "FAILED/CANCELLED Tasks" in text
    assert (
        _section_index(text, "Runtime")
        < _section_index(text, "Failure Hints")
        < _section_index(text, "Task Duration Chart")
        < _section_index(text, "Timeline")
        < _section_index(text, "GPU Usage Chart")
        < _section_index(text, "Node Usage Chart")
        < _section_index(text, "Command Logs")
        < _section_index(text, "Workflow DAG")
        < _section_index(text, "Dependencies")
    )
    assert "bench" in text


def test_summary_writer_aligns_timeline_and_duration_chart_for_long_task_names(tmp_path):
    tg = TaskGraph()
    short = _task("short", tmp_path)
    long = _task("worker_release_after_completion_0", tmp_path)
    tg.dag.add_node("short", short)
    tg.dag.add_node("worker_release_after_completion_0", long)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )

    writer.task_submitted(short)
    short.status = TaskStatus.COMPLETED
    writer.task_completed(short)
    writer.task_submitted(long)
    long.status = TaskStatus.COMPLETED
    writer.task_completed(long)
    writer.workflow_finished(status="COMPLETED")

    text = (tmp_path / "sflow_summary.log").read_text()
    duration_lines = _section_lines(text, "Task Duration Chart", "Timeline")
    pipe_columns = {line.index("|") for line in duration_lines if "|" in line}
    assert len(pipe_columns) == 1

    timeline_lines = _section_lines(text, "Timeline", "GPU Usage Chart")
    submitted_columns = [
        line.index("SUBMITTED")
        for line in timeline_lines
        if "SUBMITTED" in line
    ]
    completed_columns = [
        line.index("COMPLETED")
        for line in timeline_lines
        if "COMPLETED" in line
    ]
    assert len(set(submitted_columns)) == 1
    assert len(set(completed_columns)) == 1

    assert (
        _section_index(text, "Runtime")
        < _section_index(text, "Task Duration Chart")
        < _section_index(text, "Timeline")
        < _section_index(text, "GPU Usage Chart")
        < _section_index(text, "Node Usage Chart")
        < _section_index(text, "Command Logs")
        < _section_index(text, "Workflow DAG")
        < _section_index(text, "Dependencies")
        < _section_index(text, "Failure Hints")
    )


def test_summary_writer_records_failure_hints_with_task_log_path(tmp_path):
    tg = TaskGraph()
    task = _task("server", tmp_path)
    tg.dag.add_node("server", task)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )

    task.attempts = 2
    task.exit_code = 7
    writer.task_failed(task, reason="readiness probe timed out", exit_code=7)
    writer.workflow_finished(status="FAILED")

    text = (tmp_path / "sflow_summary.log").read_text()
    assert "Failure Hints" in text
    assert "server" in text
    assert "exit=7" in text
    assert "attempts=2" in text
    assert "readiness probe timed out" in text
    assert str(tmp_path / "server" / "server.log") in text


def test_summary_writer_records_workflow_failure_detail(tmp_path):
    tg = TaskGraph()
    task = _task("server", tmp_path)
    tg.dag.add_node("server", task)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )

    detail = "Workflow 'wf' failed: 1 task(s) failed (server)"
    writer.workflow_finished(status="FAILED", detail=detail)

    text = (tmp_path / "sflow_summary.log").read_text()
    top_summary = _top_summary_text(text)
    assert "Workflow Detail : Workflow 'wf' failed: 1 task(s) failed (server)" in top_summary
    assert "End Summary" not in text


def test_summary_writer_renders_gpu_and_node_usage_charts(tmp_path):
    tg = TaskGraph()
    gpu_task = _task("gpu_task", tmp_path)
    gpu_task.assigned_nodes = ["node-a"]
    gpu_task.envs["CUDA_VISIBLE_DEVICES"] = "0-2,bad"
    gpu_task_2 = _task("gpu_task_2", tmp_path)
    gpu_task_2.assigned_nodes = ["node-a"]
    gpu_task_2.envs["CUDA_VISIBLE_DEVICES"] = "0"
    node_task = _task("node_task", tmp_path)
    node_task.assigned_nodes = ["node-a", "node-b"]
    tg.dag.add_node("gpu_task", gpu_task)
    tg.dag.add_node("gpu_task_2", gpu_task_2)
    tg.dag.add_node("node_task", node_task)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )

    gpu_task.attempts = 1
    writer.task_submitted(gpu_task)
    gpu_task.status = TaskStatus.READY
    writer.task_ready(gpu_task)
    gpu_task_2.attempts = 1
    writer.task_submitted(gpu_task_2)
    gpu_task_2.status = TaskStatus.COMPLETED
    writer.task_completed(gpu_task_2)
    node_task.attempts = 1
    writer.task_submitted(node_task)
    node_task.status = TaskStatus.COMPLETED
    writer.task_completed(node_task)
    writer.workflow_finished(status="COMPLETED")

    text = (tmp_path / "sflow_summary.log").read_text()
    gpu_lines = _section_lines(text, "GPU Usage Chart", "Node Usage Chart")
    node_lines = _section_lines(text, "Node Usage Chart", "Command Logs")
    assert "GPU Usage Chart" in text
    assert gpu_lines[0] == "Legend:"
    assert re.search(
        r"Legend:\n"
        r"  A=gpu_task \d+\.\d{3}s READY\n"
        r"  B=gpu_task_2 \d+\.\d{3}s COMPLETED",
        "\n".join(gpu_lines),
    )
    assert sum(line.startswith("node-a GPU 0 ") for line in gpu_lines) == 1
    assert sum(line.startswith("node-a GPU 1 ") for line in gpu_lines) == 1
    assert sum(line.startswith("node-a GPU 2 ") for line in gpu_lines) == 1
    assert re.search(
        r"node-a GPU 0\s+\|[AB.*]{30}\|$",
        "\n".join(gpu_lines),
        re.MULTILINE,
    )
    assert re.search(
        r"node-a GPU 1\s+\|[A.*]{30}\|$",
        "\n".join(gpu_lines),
        re.MULTILINE,
    )
    assert re.search(
        r"node-a GPU 2\s+\|[A.*]{30}\|$",
        "\n".join(gpu_lines),
        re.MULTILINE,
    )
    assert "node-a GPU 0-2" not in text
    assert "bad" not in text
    assert "Node Usage Chart" in text
    assert node_lines[0] == "Hint: '*' marks multiple tasks active on this resource at the same time."
    assert node_lines[1] == "Legend:"
    assert re.search(
        r"Legend:\n"
        r"  A=gpu_task \d+\.\d{3}s READY\n"
        r"  B=gpu_task_2 \d+\.\d{3}s COMPLETED\n"
        r"  C=node_task \d+\.\d{3}s COMPLETED",
        "\n".join(node_lines),
    )
    assert sum(line.startswith("node-a ") for line in node_lines) == 1
    assert sum(line.startswith("node-b ") for line in node_lines) == 1
    assert re.search(
        r"node-a\s+\|[ABC.*]{30}\|$",
        "\n".join(node_lines),
        re.MULTILINE,
    )
    assert re.search(
        r"node-b\s+\|[C.*]{30}\|$",
        "\n".join(node_lines),
        re.MULTILINE,
    )


def test_summary_writer_extends_resource_chart_for_held_ready_task(tmp_path):
    tg = TaskGraph()
    service = _task("service", tmp_path)
    service.assigned_nodes = ["node-a"]
    service.envs["CUDA_VISIBLE_DEVICES"] = "0"
    service.resource_release_after["gpus"] = "workflow_completion"
    tg.dag.add_node("service", service)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )
    writer.task_submitted(service)
    service.status = TaskStatus.READY
    writer.task_ready(service)
    writer.workflow_finished(status="COMPLETED")

    writer._task_started[service.name] = 0.0
    writer._task_finished[service.name] = 2.0
    writer._ended_monotonic = 10.0
    writer._render()

    text = (tmp_path / "sflow_summary.log").read_text()
    assert re.search(
        r"Legend:\n"
        r"  A=service 10\.000s READY",
        text,
    )
    assert re.search(
        r"node-a GPU 0\s+\|A{30}\|$",
        text,
        re.MULTILINE,
    )


def test_summary_writer_releases_resource_at_task_ready_timestamp(tmp_path):
    tg = TaskGraph()
    service = _task("service", tmp_path)
    service.assigned_nodes = ["node-a"]
    service.envs["CUDA_VISIBLE_DEVICES"] = "0"
    service.resource_release_after["gpus"] = "task_ready"
    tg.dag.add_node("service", service)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )
    writer.task_submitted(service)
    service.status = TaskStatus.READY
    writer.task_ready(service)
    service.status = TaskStatus.COMPLETED
    writer.task_completed(service)
    writer.workflow_finished(status="COMPLETED")

    writer._task_started[service.name] = 0.0
    writer._task_ready[service.name] = 2.0
    writer._task_finished[service.name] = 10.0
    writer._ended_monotonic = 10.0
    writer._render()

    text = (tmp_path / "sflow_summary.log").read_text()
    assert re.search(
        r"Legend:\n"
        r"  A=service 2\.000s COMPLETED",
        text,
    )
    assert re.search(
        r"node-a GPU 0\s+\|A{30}\|$",
        text,
        re.MULTILINE,
    )


def test_summary_writer_renders_gpu_usage_without_assigned_nodes(tmp_path):
    tg = TaskGraph()
    task = _task("local_gpu_task", tmp_path)
    task.envs["CUDA_VISIBLE_DEVICES"] = "0"
    tg.dag.add_node("local_gpu_task", task)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )
    writer.task_submitted(task)
    task.status = TaskStatus.COMPLETED
    writer.task_completed(task)
    writer.workflow_finished(status="COMPLETED")

    text = (tmp_path / "sflow_summary.log").read_text()
    assert re.search(
        r"Legend:\n"
        r"  A=local_gpu_task \d+\.\d{3}s COMPLETED",
        text,
    )
    assert re.search(
        r"GPU 0\s+\|[A.]{30}\|$",
        text,
        re.MULTILINE,
    )


def test_summary_includes_uploads_section(tmp_path):
    tg = TaskGraph()
    task = _task("produce_results", tmp_path)
    tg.dag.add_node("produce_results", task)
    workflow = Workflow(name="wf", task_graph=tg)

    summary_path = tmp_path / "sflow_summary.log"
    writer = SflowSummaryWriter(summary_path, debounce_interval=30)

    async def _exercise() -> None:
        writer.start(
            workflow=workflow,
            output_dir=tmp_path,
            runtime_info_text="runtime",
            command_log_paths={},
        )
        writer.record_uploads(
            [
                UploadResult(
                    task="produce_results",
                    target="results_bucket",
                    source="/out/produce_results/results.csv",
                    destination="s3://bucket/main/results.csv",
                    status="uploaded",
                    on_error="warn",
                ),
                UploadResult(
                    task="produce_results",
                    target="results_bucket",
                    source="/out/produce_results/summary.json",
                    destination="s3://bucket/summary.json",
                    status="failed",
                    on_error="warn",
                    error="Unable to parse config file",
                ),
            ]
        )
        task.status = TaskStatus.COMPLETED
        writer.workflow_finished(status="COMPLETED")

    asyncio.run(_exercise())

    text = summary_path.read_text()
    # Dedicated section with a header + counts.
    assert "Uploads     : uploaded=1, failed=1" in _top_summary_text(text)
    assert "\nUploads\n-------\n" in text
    assert "uploaded=1" in text
    assert "failed=1" in text
    # Grouped by task, with per-file destination and the failure reason.
    assert "produce_results:" in text
    assert "s3://bucket/main/results.csv" in text
    assert "Unable to parse config file" in text


def test_summary_omits_uploads_section_when_no_uploads(tmp_path):
    tg = TaskGraph()
    task = _task("t", tmp_path)
    tg.dag.add_node("t", task)
    workflow = Workflow(name="wf", task_graph=tg)

    summary_path = tmp_path / "sflow_summary.log"
    writer = SflowSummaryWriter(summary_path, debounce_interval=30)

    async def _exercise() -> None:
        writer.start(
            workflow=workflow,
            output_dir=tmp_path,
            runtime_info_text="runtime",
            command_log_paths={},
        )
        task.status = TaskStatus.COMPLETED
        writer.workflow_finished(status="COMPLETED")

    asyncio.run(_exercise())

    text = summary_path.read_text()
    assert "Uploads     :" not in _top_summary_text(text)
    assert "\nUploads\n-------\n" not in text


def test_summary_renders_probe_traces_section(tmp_path):
    from sflow.core.probe import ProbeAttempt, ProbeType
    from sflow.plugins.probes.log_watch import LogWatchProbe

    tg = TaskGraph()
    task = _task("server", tmp_path)
    probe = LogWatchProbe(
        regex_pattern="Application startup complete",
        type=ProbeType.READINESS,
        interval=0,
        timeout=10,
    )
    probe.last_attempt = ProbeAttempt(
        ok=False, runtime=1.5, detail="no match (0/1) | last line: 'loading weights 42%'"
    )
    task.probes = [probe]
    tg.dag.add_node("server", task)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )
    writer.flush()

    text = (tmp_path / "sflow_summary.log").read_text(encoding="utf-8")
    assert "Probe Traces (last attempt)" in text
    row = next(
        line
        for line in text.splitlines()
        if line.startswith("server") and "log_watch" in line
    )
    assert "readiness" in row and "[FAIL]" in row
    assert "loading weights 42%" in row


def test_probe_trace_refreshes_while_task_stays_running(tmp_path):
    """A stuck DAG (task RUNNING, readiness never satisfied) must still update the
    probe trace on each check: record_probe_attempt schedules the render, since no
    task-status event fires to trigger one while the task sits RUNNING."""
    from sflow.core.probe import ProbeAttempt, ProbeType
    from sflow.plugins.probes.log_watch import LogWatchProbe

    tg = TaskGraph()
    task = _task("server", tmp_path)
    probe = LogWatchProbe(
        regex_pattern="ready", type=ProbeType.READINESS, interval=0, timeout=10
    )
    task.probes = [probe]
    tg.dag.add_node("server", task)
    workflow = Workflow(name="wf", task_graph=tg)
    summary_path = tmp_path / "sflow_summary.log"
    writer = SflowSummaryWriter(summary_path)
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )

    # First check (miss). Only record_probe_attempt fires -- no status event.
    probe.last_attempt = ProbeAttempt(
        ok=False, runtime=0.1, detail="no match | last line: 'step 1'"
    )
    writer.record_probe_attempt(task)
    writer.flush()
    assert "step 1" in summary_path.read_text(encoding="utf-8")

    # A later check while STILL RUNNING refreshes the trace to the newest line.
    probe.last_attempt = ProbeAttempt(
        ok=False, runtime=0.1, detail="no match | last line: 'step 2'"
    )
    writer.record_probe_attempt(task)
    writer.flush()
    text = summary_path.read_text(encoding="utf-8")
    assert "step 2" in text and "step 1" not in text


def test_gpu_chart_uses_physical_devices_not_container_visible_env(tmp_path):
    """Two docker tasks on disjoint GPUs must not be drawn on the same rows.

    A containerized backend re-indexes every container to CUDA_VISIBLE_DEVICES=
    0..N-1, so reading that env would show both tasks on GPU 0/1. The chart must
    use the devices actually reserved (or, before launch, the planner's slice).
    """
    tg = TaskGraph()
    a = _task("a", tmp_path)
    a.assigned_nodes = ["node-a"]
    a.cuda_visible_devices = "0,1"
    a.envs["CUDA_VISIBLE_DEVICES"] = "0,1"  # what the container sees
    a.reserved_gpu_indices = [4, 5]  # what it actually got at launch

    b = _task("b", tmp_path)
    b.assigned_nodes = ["node-a"]
    b.cuda_visible_devices = "2,3"
    b.envs["CUDA_VISIBLE_DEVICES"] = "0,1"  # identical env, different GPUs
    b.reserved_gpu_indices = [6, 7]

    tg.dag.add_node("a", a)
    tg.dag.add_node("b", b)
    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=Workflow(name="wf", task_graph=tg),
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )
    for task in (a, b):
        task.attempts = 1
        writer.task_submitted(task)
        task.status = TaskStatus.COMPLETED
        writer.task_completed(task)
    writer.workflow_finished(status="COMPLETED")

    gpu_lines = _section_lines(
        (tmp_path / "sflow_summary.log").read_text(),
        "GPU Usage Chart",
        "Node Usage Chart",
    )
    joined = "\n".join(gpu_lines)
    for index in (4, 5, 6, 7):
        assert f"node-a GPU {index} " in joined
    # The virtual container indices must not appear as if they were physical.
    assert "node-a GPU 0 " not in joined
    assert "node-a GPU 1 " not in joined


def test_gpu_chart_falls_back_to_the_planner_slice_before_launch(tmp_path):
    # No reservation recorded (every non-docker backend, or a task that has not
    # launched): the planner's slice is still better than the container env.
    tg = TaskGraph()
    task = _task("t", tmp_path)
    task.assigned_nodes = ["node-a"]
    task.cuda_visible_devices = "2,3"
    task.envs["CUDA_VISIBLE_DEVICES"] = "0,1"
    tg.dag.add_node("t", task)
    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=Workflow(name="wf", task_graph=tg),
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )
    task.attempts = 1
    writer.task_submitted(task)
    task.status = TaskStatus.COMPLETED
    writer.task_completed(task)
    writer.workflow_finished(status="COMPLETED")

    joined = "\n".join(
        _section_lines(
            (tmp_path / "sflow_summary.log").read_text(),
            "GPU Usage Chart",
            "Node Usage Chart",
        )
    )
    assert "node-a GPU 2 " in joined and "node-a GPU 3 " in joined
    assert "node-a GPU 0 " not in joined


def test_gpu_chart_omits_backends_that_do_not_assign_visible_devices(tmp_path):
    """A k8s task must not appear on invented GPU rows.

    Kubernetes never injects CUDA_VISIBLE_DEVICES -- the device plugin / DRA picks
    the physical GPUs -- so the planner's slice is a capacity-planning artifact,
    not a record of which devices the pod used.
    """
    tg = TaskGraph()
    task = _task("k8s_task", tmp_path)
    task.assigned_nodes = ["node-a"]
    task.cuda_visible_devices = "2,3"  # planner slice, no env injected
    tg.dag.add_node("k8s_task", task)
    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=Workflow(name="wf", task_graph=tg),
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )
    task.attempts = 1
    writer.task_submitted(task)
    task.status = TaskStatus.COMPLETED
    writer.task_completed(task)
    writer.workflow_finished(status="COMPLETED")

    text = (tmp_path / "sflow_summary.log").read_text()
    gpu_lines = _section_lines(text, "GPU Usage Chart", "Node Usage Chart")
    joined = "\n".join(gpu_lines)
    assert "GPU 2" not in joined and "GPU 3" not in joined
    assert "(no GPU timings)" in joined
    # The node chart still covers it -- only device attribution is withheld.
    assert "node-a" in "\n".join(_section_lines(text, "Node Usage Chart", "Command Logs"))


def test_summary_renders_node_topology_section(tmp_path):
    # A backend's reservation-stage CPU/NUMA/GPU probe surfaces in a dedicated
    # 'Node Topology' section so it's visible when reviewing results later. Read from
    # the backend at render time (report is populated during allocation).
    tg = TaskGraph()
    server = _task("decode_server_0", tmp_path)
    tg.dag.add_node("decode_server_0", server)
    workflow = Workflow(name="wf", task_graph=tg)

    class _FakeBackend:
        node_topology_report = "gb300-node-a:\nnproc=144\ncpuset=0-143"

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow, output_dir=tmp_path, runtime_info_text="rt",
        command_log_paths={}, backends={"cluster": _FakeBackend()},
    )
    writer.workflow_finished(status="READY")

    text = (tmp_path / "sflow_summary.log").read_text()
    assert "Node Topology" in text
    assert "[backend cluster]" in text
    assert "nproc=144" in text and "cpuset=0-143" in text


def test_summary_omits_node_topology_when_no_backend_report(tmp_path):
    # A backend with no topology report (non-K8s, or probe skipped) -> no section.
    tg = TaskGraph()
    server = _task("s", tmp_path)
    tg.dag.add_node("s", server)
    workflow = Workflow(name="wf", task_graph=tg)

    class _NoTopoBackend:
        node_topology_report = None

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow, output_dir=tmp_path, runtime_info_text="rt",
        command_log_paths={}, backends={"local": _NoTopoBackend()},
    )
    writer.workflow_finished(status="READY")
    assert "Node Topology" not in (tmp_path / "sflow_summary.log").read_text()


def _writer_and_workflow(tmp_path):
    tg = TaskGraph()
    tg.dag.add_node("t", _task("t", tmp_path))
    workflow = Workflow(name="wf", task_graph=tg)
    return SflowSummaryWriter(tmp_path / "sflow_summary.log"), workflow


def test_summary_writer_arms_and_disarms_the_loop_watchdog(tmp_path):
    """The stall detector is only useful if the run actually arms and releases it.

    Armed too late (or not at all) and a freeze leaves no evidence -- the failure mode
    is self-concealing, since sflow's own log records come from the thread that is
    blocked. Left armed after the run and a quiet shutdown gets reported as a stall,
    and the daemon thread keeps pinging through interpreter exit.
    """
    writer, workflow = _writer_and_workflow(tmp_path)
    states: dict = {}

    async def _exercise() -> None:
        writer.start(
            workflow=workflow,
            output_dir=tmp_path,
            runtime_info_text="runtime",
            command_log_paths={},
        )
        states["armed"] = writer._loop_watchdog is not None
        writer.workflow_finished(status="COMPLETED")
        states["released"] = writer._loop_watchdog is None

    asyncio.run(_exercise())

    assert states["armed"], "the watchdog must be armed for the life of the run"
    assert states["released"], "the watchdog must be released when the run ends"


def test_summary_writer_skips_the_watchdog_outside_a_running_loop(tmp_path, recwarn):
    """A synchronous caller must degrade cleanly, not arm a watchdog on a dead loop.

    ``EventLoopWatchdog.start`` raises when there is no running loop; the writer has to
    catch that and carry on. If it ever stopped raising, a sync caller would get a
    watcher thread that reports a false stall 30 seconds later and writes a stack dump
    -- and `start()` must not leave an un-awaited coroutine behind either.
    """
    writer, workflow = _writer_and_workflow(tmp_path)

    writer.start(                      # no running loop: this must not raise
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )

    assert writer._loop_watchdog is None, "no loop to watch means no watchdog"
    assert not (tmp_path / "loop_stalls.txt").exists()
    gc.collect()
    assert not [
        w for w in recwarn.list if issubclass(w.category, RuntimeWarning)
    ], "arming must not leak an un-awaited beat coroutine"
    writer.workflow_finished(status="COMPLETED")   # must be a no-op, not an AttributeError


# ---------------------------------------------------------------------------
# GPU Assignment: physical device ids vs what the task itself could see
# ---------------------------------------------------------------------------


def _gpu_task(name, tmp_path, *, reserved=None, visible=None, planned=None):
    task = _task(name, tmp_path)
    if reserved is not None:
        task.reserved_gpu_indices = list(reserved)
    if planned is not None:
        task.cuda_visible_devices = planned
    if visible is not None:
        task.envs["CUDA_VISIBLE_DEVICES"] = visible
    return task


def _render(tmp_path, tasks, name="wf"):
    tg = TaskGraph()
    for task in tasks:
        tg.dag.add_node(task.name, task)
    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=Workflow(name=name, task_graph=tg),
        output_dir=tmp_path,
        runtime_info_text="rt",
        command_log_paths={},
    )
    writer.workflow_finished(status="COMPLETED")
    return (tmp_path / "sflow_summary.log").read_text()


def test_gpu_assignment_reports_physical_ids_not_the_container_view(tmp_path):
    """A docker task handed physical GPUs 2,3 sees CUDA_VISIBLE_DEVICES=0,1. The
    summary must answer "which card did this run on", which the task's own env
    cannot -- so both columns are shown."""
    task = _gpu_task("trainer", tmp_path, reserved=[2, 3], visible="0,1")
    text = _render(tmp_path, [task])

    assert "GPU Assignment" in text
    section = text.split("GPU Assignment", 1)[1]
    row = [ln for ln in section.splitlines() if ln.startswith("trainer")][0]
    assert "2,3" in row, row
    assert row.index("2,3") < row.index("0,1"), "physical column must come first"


def test_gpu_assignment_warns_only_when_the_two_views_differ(tmp_path):
    """The hint is the whole reason the section exists on docker; on a backend
    that does not re-index it would be a lie, so it must not appear."""
    remapped = _gpu_task("docker_task", tmp_path, reserved=[6, 7], visible="0,1")
    assert "re-indexes devices" in _render(tmp_path, [remapped])

    # Slurm-style: the env IS the physical slice, so no remap and no hint.
    plain = _gpu_task("slurm_task", tmp_path, planned="2,3", visible="2,3")
    text = _render(tmp_path, [plain])
    assert "GPU Assignment" in text
    assert "re-indexes devices" not in text


def test_gpu_assignment_is_omitted_when_no_task_used_a_gpu(tmp_path):
    text = _render(tmp_path, [_task("cpu_only", tmp_path)])
    assert "GPU Assignment" not in text


def test_gpu_assignment_skips_backends_that_never_expose_device_ids(tmp_path):
    """Kubernetes: the device plugin picks the GPUs and injects no env, so sflow
    never learns their physical ids. Printing the planner's provisional slice
    would invent numbers the pod never used."""
    k8s_task = _task("k8s_task", tmp_path)
    k8s_task.cuda_visible_devices = "0,1"  # plan-time only; no env injected
    assert "GPU Assignment" not in _render(tmp_path, [k8s_task])


def test_gpu_assignment_lists_every_gpu_task(tmp_path):
    tasks = [
        _gpu_task("server", tmp_path, reserved=[1, 2], visible="0,1"),
        _gpu_task("client", tmp_path, reserved=[3], visible="0"),
        _task("report", tmp_path),
    ]
    section = _render(tmp_path, tasks).split("GPU Assignment", 1)[1]
    assert any(ln.startswith("server") and "1,2" in ln for ln in section.splitlines())
    assert any(ln.startswith("client") and "3" in ln for ln in section.splitlines())
    assert not any(ln.startswith("report") for ln in section.splitlines())


def test_gpu_usage_chart_rows_sorted_by_node_then_gpu_index(tmp_path):
    """Rows read node-by-node, GPU 0..N.

    They are collected in task order, so without an explicit sort the chart
    interleaves nodes by "which task touched this GPU first" -- e.g. a warmup
    task holding GPUs 0,2 on two nodes pushes every GPU 1,3 row to the bottom.
    """
    tg = TaskGraph()
    tasks = []
    # warmup grabs GPUs 0 and 2 on both nodes; workers take 1 and 3 afterwards.
    for name, gpus in (("warmup", [0, 2]), ("worker", [1, 3])):
        task = _task(name, tmp_path)
        task.assigned_nodes = ["node2", "node10"]
        task.reserved_gpu_indices = list(gpus)
        tg.dag.add_node(name, task)
        tasks.append(task)
    workflow = Workflow(name="wf", task_graph=tg)

    writer = SflowSummaryWriter(tmp_path / "sflow_summary.log")
    writer.start(
        workflow=workflow,
        output_dir=tmp_path,
        runtime_info_text="runtime",
        command_log_paths={},
    )
    for task in tasks:
        writer.task_submitted(task)
        task.status = TaskStatus.COMPLETED
        writer.task_completed(task)
    writer.workflow_finished(status="COMPLETED")

    rows = [
        line.split("|")[0].strip()
        for line in _section_lines(
            (tmp_path / "sflow_summary.log").read_text(),
            "GPU Usage Chart",
            "Node Usage Chart",
        )
        if "|" in line
    ]
    # node2 before node10 (numeric, not lexicographic), GPUs ascending within.
    assert rows == [
        "node2 GPU 0",
        "node2 GPU 1",
        "node2 GPU 2",
        "node2 GPU 3",
        "node10 GPU 0",
        "node10 GPU 1",
        "node10 GPU 2",
        "node10 GPU 3",
    ], rows
    assert rows == sorted(rows, key=_resource_row_key)
