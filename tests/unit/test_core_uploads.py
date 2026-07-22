# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from pathlib import Path

import zipfile

from sflow.core.storage import StorageTarget
from sflow.core.task import ResolvedUpload, Task
from sflow.core.uploads import (
    ResolvedWorkflowUpload,
    UploadResult,
    run_task_uploads,
    run_workflow_upload,
)


class _RecordingTarget(StorageTarget):
    def __init__(self, name: str, *, raise_on_upload: bool = False, prefix: str = ""):
        super().__init__(name)
        self.prefix = prefix
        self.calls: list[tuple[Path, str]] = []
        self.raise_on_upload = raise_on_upload

    async def upload(self, local_path: Path, remote_key: str) -> None:
        if self.raise_on_upload:
            raise RuntimeError("boom")
        self.calls.append((local_path, remote_key))

    def plan(self, local_path: Path, remote_key: str) -> str:
        return f"recording://{self.name}/{remote_key}"


class _RecordingLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture_upload_logs():
    """Attach a direct handler to the upload logger.

    This avoids test-order coupling with global ``configure_logging()`` state,
    which may set the parent ``sflow`` logger to ``propagate=False``.
    """
    logger = logging.getLogger("sflow.core.uploads")
    handler = _RecordingLogHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger, handler, old_level


def _release_upload_logs(logger, handler, old_level) -> str:
    logger.removeHandler(handler)
    logger.setLevel(old_level)
    return "\n".join(record.getMessage() for record in handler.records)


def _make_task(name: str, output_dir: Path, *uploads: ResolvedUpload) -> Task:
    t = Task(name=name, logger=logging.getLogger("t"), operator=None)  # type: ignore[arg-type]
    t.envs["SFLOW_TASK_OUTPUT_DIR"] = str(output_dir)
    t.uploads = list(uploads)
    return t


def test_upload_resolves_task_output_dir_expression(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")

    target = _RecordingTarget("bucket")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr=None,
            on_error="warn",
        ),
    )

    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))

    assert ok is True
    assert len(target.calls) == 1
    local, key = target.calls[0]
    assert local == out_dir / "results.csv"
    assert key == "results.csv"


def test_upload_glob_expands_and_uses_basename(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "a.json").write_text("{}")
    (out_dir / "b.json").write_text("{}")
    (out_dir / "ignore.txt").write_text("nope")

    target = _RecordingTarget("bucket", prefix="runs/")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/*.json",
            to_expr=None,
            on_error="warn",
        ),
    )

    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))

    assert ok is True
    keys = sorted(k for _, k in target.calls)
    assert keys == ["runs/a.json", "runs/b.json"]


def test_upload_empty_glob_warn_keeps_ok(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    target = _RecordingTarget("bucket")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/missing-*.csv",
            on_error="warn",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is True
    # Empty glob with on_error=warn => no calls made, but upload returns OK.
    assert target.calls == []


def test_upload_empty_glob_fail_returns_false(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    target = _RecordingTarget("bucket")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/missing-*.csv",
            on_error="fail",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is False


def test_upload_target_exception_warn_swallows(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "x.txt").write_text("x")
    target = _RecordingTarget("bucket", raise_on_upload=True)
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/x.txt",
            on_error="warn",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is True


def test_upload_target_exception_fail_returns_false(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "x.txt").write_text("x")
    target = _RecordingTarget("bucket", raise_on_upload=True)
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/x.txt",
            on_error="fail",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is False


def test_unknown_target_warn(tmp_path: Path):
    task = _make_task(
        "task1",
        tmp_path,
        ResolvedUpload(
            target="ghost",
            from_expr="x.txt",
            on_error="warn",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {}))
    # Unknown target with on_error=warn => returns True (no fatal).
    assert ok is True


def test_unknown_target_fail_returns_false(tmp_path: Path):
    task = _make_task(
        "task1",
        tmp_path,
        ResolvedUpload(
            target="ghost",
            from_expr="x.txt",
            on_error="fail",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {}))
    assert ok is False


def test_dry_run_does_not_invoke_upload(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "x.txt").write_text("x")
    target = _RecordingTarget("bucket")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/x.txt",
            on_error="warn",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}, dry_run=True))
    assert ok is True
    # The whole point of dry-run: target.upload() is never called.
    assert target.calls == []


def test_to_template_with_trailing_slash_treated_as_dir(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "report.json").write_text("{}")

    target = _RecordingTarget("bucket", prefix="p/")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/report.json",
            to_expr="results/",
            on_error="warn",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is True
    assert target.calls == [(out_dir / "report.json", "p/results/report.json")]


# ---------------------------------------------------------------------------
# Replica auto-rename (disambiguate_with)
# ---------------------------------------------------------------------------


def test_insert_replica_suffix_helper():
    from sflow.core.uploads import _insert_replica_suffix

    assert _insert_replica_suffix("main/results.csv", "r0") == "main/results_r0.csv"
    assert _insert_replica_suffix("results.csv", "r0") == "results_r0.csv"
    # No extension: suffix is appended to the bare name.
    assert _insert_replica_suffix("results", "r0") == "results_r0"
    # Only the final extension is split (posixpath.splitext semantics).
    assert _insert_replica_suffix("a/b/c.tar.gz", "r0") == "a/b/c.tar_r0.gz"


def test_disambiguate_literal_to_inserts_replica_suffix(tmp_path: Path):
    out_dir = tmp_path / "benchmark_0"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")

    target = _RecordingTarget("bucket")
    task = _make_task(
        "benchmark_0",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr="main/results.csv",
            on_error="warn",
            disambiguate_with="benchmark_0",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is True
    assert target.calls == [
        (out_dir / "results.csv", "main/results_benchmark_0.csv")
    ]


def test_disambiguate_dir_to_inserts_replica_suffix(tmp_path: Path):
    out_dir = tmp_path / "benchmark_1"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")

    target = _RecordingTarget("bucket", prefix="runs/")
    task = _make_task(
        "benchmark_1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr="main/",
            on_error="warn",
            disambiguate_with="benchmark_1",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is True
    assert target.calls == [
        (out_dir / "results.csv", "runs/main/results_benchmark_1.csv")
    ]


def test_disambiguate_omitted_to_inserts_replica_suffix(tmp_path: Path):
    out_dir = tmp_path / "benchmark_2"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")

    target = _RecordingTarget("bucket")
    task = _make_task(
        "benchmark_2",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr=None,
            on_error="warn",
            disambiguate_with="benchmark_2",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is True
    assert target.calls == [(out_dir / "results.csv", "results_benchmark_2.csv")]


def test_disambiguate_glob_suffixes_each_file(tmp_path: Path):
    out_dir = tmp_path / "benchmark_0"
    out_dir.mkdir()
    (out_dir / "a.json").write_text("{}")
    (out_dir / "b.json").write_text("{}")

    target = _RecordingTarget("bucket", prefix="runs/")
    task = _make_task(
        "benchmark_0",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/*.json",
            to_expr=None,
            on_error="warn",
            disambiguate_with="benchmark_0",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is True
    keys = sorted(k for _, k in target.calls)
    assert keys == ["runs/a_benchmark_0.json", "runs/b_benchmark_0.json"]


def test_no_disambiguate_when_label_none(tmp_path: Path):
    # disambiguate_with=None (the default) leaves the key untouched.
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")

    target = _RecordingTarget("bucket")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr="main/results.csv",
        ),
    )
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    assert ok is True
    assert target.calls == [(out_dir / "results.csv", "main/results.csv")]


# ---------------------------------------------------------------------------
# Workflow-level upload (zip-the-whole-output-dir)
# ---------------------------------------------------------------------------


def _make_workflow_dir(tmp_path: Path, run_id: str = "wf-20260101-abc") -> Path:
    wf_dir = tmp_path / run_id
    wf_dir.mkdir()
    (wf_dir / "sflow.log").write_text("hello\n")
    (wf_dir / "task1").mkdir()
    (wf_dir / "task1" / "task1.log").write_text("task1 log\n")
    (wf_dir / "task1" / "outputs.json").write_text("{}")
    return wf_dir


def test_workflow_upload_zips_full_output_dir(tmp_path: Path):
    wf_dir = _make_workflow_dir(tmp_path)
    target = _RecordingTarget("bucket", prefix="runs/")
    spec = ResolvedWorkflowUpload(target="bucket", to_expr=None, on_error="warn")

    # The temp zip is cleaned up after upload, so capture its contents during the
    # upload call (when the file still exists).
    captured: dict[str, list[str]] = {}

    async def _capture(local_path: Path, remote_key: str) -> None:
        with zipfile.ZipFile(local_path) as zf:
            captured["names"] = sorted(zf.namelist())
        target.calls.append((local_path, remote_key))

    target.upload = _capture  # type: ignore[assignment]

    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="wf",
            workflow_out_dir=wf_dir,
            storage_targets={"bucket": target},
        )
    )

    assert ok is True
    assert len(target.calls) == 1
    _local_zip, remote_key = target.calls[0]
    # Default key is `<run_id>.zip`, joined under the target prefix.
    assert remote_key == f"runs/{wf_dir.name}.zip"
    # Zip must contain every regular file under the workflow output dir.
    assert captured["names"] == sorted(
        ["sflow.log", "task1/outputs.json", "task1/task1.log"]
    )


def test_workflow_upload_resolves_to_expression(tmp_path: Path):
    wf_dir = _make_workflow_dir(tmp_path, run_id="myrun")
    target = _RecordingTarget("bucket")
    spec = ResolvedWorkflowUpload(
        target="bucket",
        to_expr="archive/${{ workflow.name }}/${{ workflow.run_id }}.zip",
        on_error="warn",
    )

    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="my_workflow",
            workflow_out_dir=wf_dir,
            storage_targets={"bucket": target},
        )
    )

    assert ok is True
    _local, remote_key = target.calls[0]
    assert remote_key == "archive/my_workflow/myrun.zip"


def test_workflow_upload_rejects_path_traversal(tmp_path: Path):
    wf_dir = _make_workflow_dir(tmp_path)
    target = _RecordingTarget("bucket")
    spec = ResolvedWorkflowUpload(
        target="bucket",
        to_expr="../escape.zip",
        on_error="fail",
    )

    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="wf",
            workflow_out_dir=wf_dir,
            storage_targets={"bucket": target},
        )
    )
    assert ok is False
    assert target.calls == []


def test_workflow_upload_unknown_target_warn(tmp_path: Path):
    wf_dir = _make_workflow_dir(tmp_path)
    spec = ResolvedWorkflowUpload(target="ghost", on_error="warn")
    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="wf",
            workflow_out_dir=wf_dir,
            storage_targets={},
        )
    )
    assert ok is True


def test_workflow_upload_unknown_target_fail(tmp_path: Path):
    wf_dir = _make_workflow_dir(tmp_path)
    spec = ResolvedWorkflowUpload(target="ghost", on_error="fail")
    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="wf",
            workflow_out_dir=wf_dir,
            storage_targets={},
        )
    )
    assert ok is False


def test_workflow_upload_target_exception_fail_returns_false(tmp_path: Path):
    wf_dir = _make_workflow_dir(tmp_path)
    target = _RecordingTarget("bucket", raise_on_upload=True)
    spec = ResolvedWorkflowUpload(target="bucket", on_error="fail")
    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="wf",
            workflow_out_dir=wf_dir,
            storage_targets={"bucket": target},
        )
    )
    assert ok is False


def test_workflow_upload_zip_failure_warn_swallows(tmp_path: Path, monkeypatch):
    wf_dir = _make_workflow_dir(tmp_path)
    target = _RecordingTarget("bucket")

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("sflow.core.uploads._zip_directory", _boom)

    spec = ResolvedWorkflowUpload(target="bucket", on_error="warn")
    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="wf",
            workflow_out_dir=wf_dir,
            storage_targets={"bucket": target},
        )
    )
    assert ok is True
    # Zip failure means upload is never attempted.
    assert target.calls == []


def test_workflow_upload_zip_failure_fail_returns_false(tmp_path: Path, monkeypatch):
    wf_dir = _make_workflow_dir(tmp_path)
    target = _RecordingTarget("bucket")

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("sflow.core.uploads._zip_directory", _boom)

    spec = ResolvedWorkflowUpload(target="bucket", on_error="fail")
    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="wf",
            workflow_out_dir=wf_dir,
            storage_targets={"bucket": target},
        )
    )
    assert ok is False
    assert target.calls == []


def test_workflow_upload_dry_run_skips_upload(tmp_path: Path):
    wf_dir = _make_workflow_dir(tmp_path)
    target = _RecordingTarget("bucket")
    spec = ResolvedWorkflowUpload(target="bucket", on_error="warn")
    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="wf",
            workflow_out_dir=wf_dir,
            storage_targets={"bucket": target},
            dry_run=True,
        )
    )
    assert ok is True
    assert target.calls == []


# ---------------------------------------------------------------------------
# Structured result collection (for the end-of-run upload summary).
# `run_task_uploads` / `run_workflow_upload` keep their bool return; when a
# `results=` list is provided they append one UploadResult per file/outcome.
# ---------------------------------------------------------------------------


def test_task_upload_collects_uploaded_result(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")
    target = _RecordingTarget("bucket", prefix="p/")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr="main/results.csv",
            on_error="warn",
        ),
    )

    results: list[UploadResult] = []
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}, results=results))

    assert ok is True
    assert len(results) == 1
    r = results[0]
    assert r.task == "task1"
    assert r.target == "bucket"
    assert r.status == "uploaded"
    assert r.source == str(out_dir / "results.csv")
    assert r.destination == "recording://bucket/p/main/results.csv"
    assert r.on_error == "warn"
    assert r.error is None


def test_task_upload_logs_one_line_uploading_hint(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")
    target = _RecordingTarget("bucket")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr=None,
            on_error="warn",
        ),
    )
    logger, handler, old_level = _capture_upload_logs()
    try:
        ok = asyncio.run(run_task_uploads(task, {"bucket": target}))
    finally:
        logs = _release_upload_logs(logger, handler, old_level)

    assert ok is True
    assert (
        f"Task 'task1' upload[0]: uploading {out_dir / 'results.csv'} "
        "-> recording://bucket/results.csv"
    ) in logs


def test_task_upload_collects_failed_result_warn(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")
    target = _RecordingTarget("bucket", raise_on_upload=True)
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr=None,
            on_error="warn",
        ),
    )

    results: list[UploadResult] = []
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}, results=results))

    assert ok is True  # on_error=warn is not fatal
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].on_error == "warn"
    assert results[0].error is not None and "boom" in results[0].error


def test_task_upload_collects_failed_result_fail(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")
    target = _RecordingTarget("bucket", raise_on_upload=True)
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr=None,
            on_error="fail",
        ),
    )

    results: list[UploadResult] = []
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}, results=results))

    assert ok is False
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].on_error == "fail"


def test_task_upload_collects_skipped_when_no_match(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    target = _RecordingTarget("bucket")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/*.json",
            to_expr=None,
            on_error="warn",
        ),
    )

    results: list[UploadResult] = []
    ok = asyncio.run(run_task_uploads(task, {"bucket": target}, results=results))

    assert ok is True
    assert len(results) == 1
    assert results[0].status == "skipped"


def test_task_upload_collects_failed_when_target_missing(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "r.csv").write_text("x\n")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="ghost",
            from_expr="${{ task.output_dir }}/r.csv",
            to_expr=None,
            on_error="warn",
        ),
    )

    results: list[UploadResult] = []
    ok = asyncio.run(run_task_uploads(task, {}, results=results))

    assert ok is True
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].target == "ghost"
    assert results[0].error is not None and "not found" in results[0].error


def test_task_upload_collects_dry_run(tmp_path: Path):
    out_dir = tmp_path / "task1"
    out_dir.mkdir()
    (out_dir / "results.csv").write_text("k,v\n")
    target = _RecordingTarget("bucket")
    task = _make_task(
        "task1",
        out_dir,
        ResolvedUpload(
            target="bucket",
            from_expr="${{ task.output_dir }}/results.csv",
            to_expr=None,
            on_error="warn",
        ),
    )

    results: list[UploadResult] = []
    ok = asyncio.run(
        run_task_uploads(task, {"bucket": target}, dry_run=True, results=results)
    )

    assert ok is True
    assert len(results) == 1
    assert results[0].status == "dry-run"
    assert target.calls == []


def test_workflow_upload_collects_result(tmp_path: Path):
    wf_dir = _make_workflow_dir(tmp_path)
    target = _RecordingTarget("bucket", prefix="runs/")
    spec = ResolvedWorkflowUpload(target="bucket", to_expr=None, on_error="warn")

    results: list[UploadResult] = []
    ok = asyncio.run(
        run_workflow_upload(
            spec,
            workflow_name="wf",
            workflow_out_dir=wf_dir,
            storage_targets={"bucket": target},
            results=results,
        )
    )

    assert ok is True
    assert len(results) == 1
    r = results[0]
    assert r.task == "workflow.upload_all"
    assert r.target == "bucket"
    assert r.status == "uploaded"
    assert r.destination.endswith(".zip")


def test_workflow_upload_logs_one_line_uploading_hint(tmp_path: Path):
    wf_dir = _make_workflow_dir(tmp_path)
    target = _RecordingTarget("bucket", prefix="runs/")
    spec = ResolvedWorkflowUpload(target="bucket", to_expr=None, on_error="warn")
    logger, handler, old_level = _capture_upload_logs()
    try:
        ok = asyncio.run(
            run_workflow_upload(
                spec,
                workflow_name="wf",
                workflow_out_dir=wf_dir,
                storage_targets={"bucket": target},
            )
        )
    finally:
        logs = _release_upload_logs(logger, handler, old_level)

    assert ok is True
    assert (
        f"workflow.upload_all: uploading {wf_dir} "
        f"-> recording://bucket/runs/{wf_dir.name}.zip"
    ) in logs
