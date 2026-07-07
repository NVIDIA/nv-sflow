# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import re
from pathlib import Path

import pytest
import sflow.core.execution_summary as execution_summary_mod
from sflow.core.execution_summary import SflowSummaryWriter
from sflow.core.task import Task, TaskStatus
from sflow.core.uploads import UploadResult
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig


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
        server, "RDMA requested but pods fell back to slow TCP (all ports DOWN)"
    )
    writer.workflow_finished(status="READY")

    text = (tmp_path / "sflow_summary.log").read_text()
    assert "Network Warnings" in text
    assert (
        "decode_server_0: RDMA requested but pods fell back to slow TCP "
        "(all ports DOWN)" in text
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
