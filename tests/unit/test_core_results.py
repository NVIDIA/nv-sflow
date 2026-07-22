# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ``sflow.core.results``.

These tests encode the behavior described in
``docs/developer/dev-notes/result-parsing.md`` (section "Implementation plan").
They are intentionally written against the public surface of the module so they
exercise the same code paths the orchestrator will use at runtime.

The tests are designed to drive the implementation: they fail until
``parse_results_from_text`` / ``collect_task_result`` / ``normalize_result_file``
/ ``update_workflow_results`` / path helpers are implemented.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

import pytest

import sflow.core.results as results_module
from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.results import (
    SCHEMA_VERSION_TASK,
    SCHEMA_VERSION_WORKFLOW,
    collect_task_result,
    normalize_result_file,
    parse_results_from_text,
    task_result_path,
    update_workflow_results,
    workflow_results_path,
)
from sflow.core.task import ResultConfigRuntime, ResultSpec, Task, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NoopOperator(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="noop"))

    def build_command(self, *, task_name, script, envs) -> Command:  # pragma: no cover
        return Command(exec="true")


def _make_task(
    name: str,
    *,
    task_out: Path,
    workflow_out: Path | None = None,
) -> Task:
    """Build a minimal Task with the env vars sflow normally injects."""
    t = Task(
        name=name,
        logger=logging.getLogger(f"sflow.tests.results.{name}"),
        operator=_NoopOperator(),
        script=["true"],
    )
    t.envs["SFLOW_TASK_OUTPUT_DIR"] = str(task_out)
    if workflow_out is not None:
        t.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(workflow_out)
    return t


def _write_log(task_out: Path, task_name: str, lines: list[str]) -> Path:
    """
    Write a task log file in the same format sflow's launcher does.

    The merged log goes through a Python logging Formatter that prepends
    ``ts - logger - LEVEL - `` to each user message, so the parser must
    tolerate that prefix. We use a representative one here.
    """
    p = task_out / f"{task_name}.log"
    prefix = "2026-05-13 13:00:00,000 - sflow.task.x - INFO - "
    p.write_text("\n".join(prefix + line if line else "" for line in lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPathHelpers:
    def test_task_result_path_uses_task_out_dir(self, tmp_path):
        task_out = tmp_path / "t1"
        task_out.mkdir()
        t = _make_task("t1", task_out=task_out)

        assert task_result_path(t) == task_out / "result.json"

    def test_task_result_path_returns_none_when_not_set(self):
        t = Task(
            name="t1",
            logger=logging.getLogger("sflow.tests.results.none"),
            operator=_NoopOperator(),
            script=["true"],
        )
        assert task_result_path(t) is None

    def test_workflow_results_path_uses_workflow_out_dir(self, tmp_path):
        wf_out = tmp_path / "wf"
        task_out = wf_out / "t1"
        task_out.mkdir(parents=True)
        t = _make_task("t1", task_out=task_out, workflow_out=wf_out)

        assert workflow_results_path(t) == wf_out / "results.json"

    def test_workflow_results_path_returns_none_when_not_set(self, tmp_path):
        task_out = tmp_path / "t1"
        task_out.mkdir()
        t = _make_task("t1", task_out=task_out)
        assert workflow_results_path(t) is None


# ---------------------------------------------------------------------------
# parse_results_from_text – pure parser
# ---------------------------------------------------------------------------


class TestParseResultsFromText:
    def _text(self, *lines: str) -> str:
        prefix = "2026-05-13 13:00:00,000 - sflow.task.x - INFO - "
        return "\n".join(prefix + line for line in lines) + "\n"

    def test_simple_regex_map_uses_last_aggregate_by_default(self):
        text = self._text("TTFT: 40.0 ms", "TTFT: 42.5 ms", "tok/s: 123.0")
        specs = [
            ResultSpec(name="ttft", regex=r"TTFT:\s*([0-9.]+)\s*ms"),
            ResultSpec(name="tps", regex=r"tok/s:\s*([0-9.]+)"),
        ]
        out = parse_results_from_text(text, specs)
        assert out["ok"] is True
        assert out["values"]["ttft"] == 42.5
        assert out["values"]["tps"] == 123.0

    def test_named_value_group_is_preferred(self):
        text = self._text("TTFT: 11.0 ms")
        specs = [
            ResultSpec(name="ttft", regex=r"TTFT:\s*(?P<value>[0-9.]+)\s*ms"),
        ]
        out = parse_results_from_text(text, specs)
        assert out["values"]["ttft"] == 11.0

    def test_full_match_used_when_no_capture_group(self):
        text = self._text("STATUS_OK")
        specs = [ResultSpec(name="status", regex=r"STATUS_\w+")]
        out = parse_results_from_text(text, specs)
        assert out["values"]["status"] == "STATUS_OK"

    def test_group_index_extracts_specific_group(self):
        text = self._text("dim=4096x32")
        specs = [
            ResultSpec(name="cols", regex=r"dim=(\d+)x(\d+)", group=2, type="int"),
        ]
        out = parse_results_from_text(text, specs)
        assert out["values"]["cols"] == 32

    @pytest.mark.parametrize(
        ("aggregate", "expected"),
        [
            ("first", 1.0),
            ("last", 5.0),
            ("list", [1.0, 3.0, 5.0]),
            ("count", 3),
            ("min", 1.0),
            ("max", 5.0),
            ("sum", 9.0),
            ("avg", 3.0),
        ],
    )
    def test_aggregations(self, aggregate, expected):
        text = self._text("v=1.0", "v=3.0", "v=5.0")
        specs = [
            ResultSpec(
                name="v",
                regex=r"v=([0-9.]+)",
                type="float",
                aggregate=aggregate,
            )
        ]
        out = parse_results_from_text(text, specs)
        assert out["values"]["v"] == expected

    def test_type_auto_infers_int_float_bool_json(self):
        text = self._text("i=42", "f=1.5", "b=true", 'j={"a": 1}')
        specs = [
            ResultSpec(name="i", regex=r"i=(\S+)"),
            ResultSpec(name="f", regex=r"f=(\S+)"),
            ResultSpec(name="b", regex=r"b=(\S+)"),
            # `\S+` would stop at the space inside the JSON object; use a JSON-aware capture.
            ResultSpec(name="j", regex=r"j=(\{.*\})"),
        ]
        out = parse_results_from_text(text, specs)
        assert out["values"]["i"] == 42
        assert out["values"]["f"] == 1.5
        assert out["values"]["b"] is True
        assert out["values"]["j"] == {"a": 1}

    def test_type_explicit_cast_failure_records_error(self):
        text = self._text("v=not_a_number")
        specs = [
            ResultSpec(
                name="v", regex=r"v=(\S+)", type="float", required=False
            )
        ]
        out = parse_results_from_text(text, specs)
        assert any("v" in str(e) for e in out.get("errors", []))

    def test_required_missing_marks_payload_not_ok(self):
        text = self._text("nothing here")
        specs = [
            ResultSpec(name="ttft", regex=r"TTFT:\s*([0-9.]+)", required=True),
        ]
        out = parse_results_from_text(text, specs)
        assert out["ok"] is False
        assert "ttft" not in out["values"]

    def test_optional_missing_keeps_payload_ok(self):
        text = self._text("nothing")
        specs = [ResultSpec(name="ttft", regex=r"TTFT:\s*([0-9.]+)")]
        out = parse_results_from_text(text, specs)
        assert out["ok"] is True
        assert "ttft" not in out["values"]

    def test_metadata_includes_unit_type_aggregate(self):
        text = self._text("TTFT: 1.0 ms")
        specs = [
            ResultSpec(
                name="ttft",
                regex=r"TTFT:\s*([0-9.]+)\s*ms",
                type="float",
                unit="ms",
                aggregate="last",
            )
        ]
        out = parse_results_from_text(text, specs)
        meta = out["metadata"]["ttft"]
        assert meta["unit"] == "ms"
        assert meta["type"] == "float"
        assert meta["aggregate"] == "last"

    def test_matches_capture_line_number_and_text(self):
        text = self._text("hello", "TTFT: 1.0 ms")
        specs = [ResultSpec(name="ttft", regex=r"TTFT:\s*([0-9.]+)\s*ms")]
        out = parse_results_from_text(text, specs)
        match_entries = out["matches"]["ttft"]
        assert match_entries
        first = match_entries[0]
        assert first["text"].startswith("TTFT")
        assert first["line"] >= 1
        assert first["value"] == 1.0

    def test_strips_sflow_logging_prefix(self):
        text = self._text("TTFT: 9.5 ms")
        specs = [ResultSpec(name="ttft", regex=r"^TTFT:\s*([0-9.]+)\s*ms$")]
        out = parse_results_from_text(text, specs)
        assert out["values"]["ttft"] == 9.5


# ---------------------------------------------------------------------------
# normalize_result_file – file-source normalization
# ---------------------------------------------------------------------------


class TestNormalizeResultFile:
    def test_plain_object_treated_as_values(self, tmp_path):
        task_out = tmp_path / "t1"
        task_out.mkdir()
        src = task_out / "tmp.json"
        src.write_text(json.dumps({"ttft": 42.5, "tps": 123.0}))

        t = _make_task("t1", task_out=task_out, workflow_out=tmp_path)
        payload = normalize_result_file(src, t)

        assert payload["schema_version"] == SCHEMA_VERSION_TASK
        assert payload["task"] == "t1"
        assert payload["values"] == {"ttft": 42.5, "tps": 123.0}
        assert payload["source"]["type"] == "file"

    def test_already_v1_payload_is_validated_and_kept(self, tmp_path):
        task_out = tmp_path / "t1"
        task_out.mkdir()
        src = task_out / "result.json"
        existing = {
            "schema_version": SCHEMA_VERSION_TASK,
            "task": "t1",
            "status": "completed",
            "ok": True,
            "values": {"ttft": 7.0},
        }
        src.write_text(json.dumps(existing))

        t = _make_task("t1", task_out=task_out, workflow_out=tmp_path)
        payload = normalize_result_file(src, t)

        assert payload["schema_version"] == SCHEMA_VERSION_TASK
        assert payload["values"]["ttft"] == 7.0


# ---------------------------------------------------------------------------
# collect_task_result – integration of parser + writer + index update
# ---------------------------------------------------------------------------


class TestCollectTaskResult:
    def test_writes_canonical_result_json_for_regex_specs(self, tmp_path):
        wf_out = tmp_path / "wf"
        task_out = wf_out / "benchmark"
        task_out.mkdir(parents=True)
        _write_log(task_out, "benchmark", ["TTFT: 42.5 ms", "tok/s: 123.0"])

        t = _make_task("benchmark", task_out=task_out, workflow_out=wf_out)
        t.result_config = ResultConfigRuntime(
            specs=[
                ResultSpec(name="ttft", regex=r"TTFT:\s*([0-9.]+)\s*ms", type="float", unit="ms"),
                ResultSpec(name="tps", regex=r"tok/s:\s*([0-9.]+)", type="float", unit="tok/s"),
            ],
        )

        payload = asyncio.run(collect_task_result(t))

        out_path = task_out / "result.json"
        assert out_path.exists()
        on_disk = json.loads(out_path.read_text())
        assert on_disk["schema_version"] == SCHEMA_VERSION_TASK
        assert on_disk["task"] == "benchmark"
        assert on_disk["values"]["ttft"] == 42.5
        assert on_disk["values"]["tps"] == 123.0
        assert payload == on_disk
        assert t.result == payload

    def test_writes_canonical_result_json_for_file_source(self, tmp_path):
        wf_out = tmp_path / "wf"
        task_out = wf_out / "benchmark"
        task_out.mkdir(parents=True)

        # Task wrote tmp.json directly.
        (task_out / "tmp.json").write_text(json.dumps({"ttft": 1.0}))

        t = _make_task("benchmark", task_out=task_out, workflow_out=wf_out)
        t.result_config = ResultConfigRuntime(file="tmp.json")

        asyncio.run(collect_task_result(t))

        out_path = task_out / "result.json"
        assert out_path.exists()
        # The source file must NOT be deleted or renamed.
        assert (task_out / "tmp.json").exists()
        on_disk = json.loads(out_path.read_text())
        assert on_disk["values"]["ttft"] == 1.0

    def test_direct_write_to_result_file_is_idempotent(self, tmp_path):
        """When the task wrote directly to ${SFLOW_TASK_RESULT_FILE}, sflow rewrites the same path
        atomically (same source and canonical path)."""
        wf_out = tmp_path / "wf"
        task_out = wf_out / "benchmark"
        task_out.mkdir(parents=True)

        (task_out / "result.json").write_text(json.dumps({"ttft": 5.0}))

        t = _make_task("benchmark", task_out=task_out, workflow_out=wf_out)
        t.result_config = ResultConfigRuntime(file="result.json")

        asyncio.run(collect_task_result(t))

        on_disk = json.loads((task_out / "result.json").read_text())
        assert on_disk["values"]["ttft"] == 5.0
        assert on_disk["schema_version"] == SCHEMA_VERSION_TASK

    def test_missing_log_is_best_effort(self, tmp_path):
        wf_out = tmp_path / "wf"
        task_out = wf_out / "benchmark"
        task_out.mkdir(parents=True)
        # No log file written.

        t = _make_task("benchmark", task_out=task_out, workflow_out=wf_out)
        t.result_config = ResultConfigRuntime(
            specs=[ResultSpec(name="ttft", regex=r"TTFT:\s*([0-9.]+)\s*ms")],
        )

        payload = asyncio.run(collect_task_result(t))
        assert payload is not None
        # Best-effort: we still produce a result.json with empty values
        assert (task_out / "result.json").exists()
        assert payload["values"] == {}

    def test_required_missing_marks_payload_ok_false(self, tmp_path):
        wf_out = tmp_path / "wf"
        task_out = wf_out / "benchmark"
        task_out.mkdir(parents=True)
        _write_log(task_out, "benchmark", ["nothing relevant"])

        t = _make_task("benchmark", task_out=task_out, workflow_out=wf_out)
        t.result_config = ResultConfigRuntime(
            specs=[
                ResultSpec(
                    name="ttft",
                    regex=r"TTFT:\s*([0-9.]+)\s*ms",
                    required=True,
                )
            ],
        )

        payload = asyncio.run(collect_task_result(t))
        assert payload["ok"] is False

    def test_updates_workflow_results_index(self, tmp_path):
        wf_out = tmp_path / "wf"
        task_out = wf_out / "benchmark"
        task_out.mkdir(parents=True)
        _write_log(task_out, "benchmark", ["TTFT: 42.5 ms"])

        t = _make_task("benchmark", task_out=task_out, workflow_out=wf_out)
        t.status = type(t.status).COMPLETED  # type: ignore[attr-defined]
        t.result_config = ResultConfigRuntime(
            specs=[ResultSpec(name="ttft", regex=r"TTFT:\s*([0-9.]+)\s*ms", type="float")],
        )

        asyncio.run(collect_task_result(t))

        idx_path = wf_out / "results.json"
        assert idx_path.exists()
        idx = json.loads(idx_path.read_text())
        assert idx["schema_version"] == SCHEMA_VERSION_WORKFLOW
        task_entry = idx["tasks"]["benchmark"]
        assert task_entry["values"]["ttft"] == 42.5
        # The index should point back to the per-task result.json (relative to workflow output dir).
        assert task_entry["result_file"].endswith("result.json")

    def test_finalizing_task_records_completed_status_in_result_files(self, tmp_path):
        wf_out = tmp_path / "wf"
        task_out = wf_out / "benchmark"
        task_out.mkdir(parents=True)
        _write_log(task_out, "benchmark", ["TTFT: 42.5 ms"])

        t = _make_task("benchmark", task_out=task_out, workflow_out=wf_out)
        t.status = TaskStatus.FINALIZING
        t.result_config = ResultConfigRuntime(
            specs=[ResultSpec(name="ttft", regex=r"TTFT:\s*([0-9.]+)\s*ms", type="float")],
        )

        payload = asyncio.run(collect_task_result(t))

        on_disk = json.loads((task_out / "result.json").read_text())
        idx = json.loads((wf_out / "results.json").read_text())

        assert payload["status"] == "COMPLETED"
        assert on_disk["status"] == "COMPLETED"
        assert idx["tasks"]["benchmark"]["status"] == "COMPLETED"

    def test_log_read_and_parse_run_off_event_loop(self, tmp_path, monkeypatch):
        """Regression guard: the CPU-bound regex scan must run on a worker
        thread, not the orchestrator event loop, so a multi-GB log can't stall
        probes/heartbeats of other running tasks."""
        wf_out = tmp_path / "wf"
        task_out = wf_out / "benchmark"
        task_out.mkdir(parents=True)
        _write_log(task_out, "benchmark", ["TTFT: 42.5 ms"])

        t = _make_task("benchmark", task_out=task_out, workflow_out=wf_out)
        t.result_config = ResultConfigRuntime(
            specs=[ResultSpec(name="ttft", regex=r"TTFT:\s*([0-9.]+)\s*ms", type="float")],
        )

        main_thread = threading.current_thread()
        parse_threads: list[threading.Thread] = []
        real_parse = results_module.parse_results_from_text

        def _tracking_parse(text, specs):
            parse_threads.append(threading.current_thread())
            return real_parse(text, specs)

        monkeypatch.setattr(results_module, "parse_results_from_text", _tracking_parse)

        payload = asyncio.run(collect_task_result(t))

        assert payload["values"]["ttft"] == 42.5
        assert parse_threads, "parse_results_from_text was never invoked"
        assert all(th is not main_thread for th in parse_threads), (
            "regex parsing must run on a worker thread, not the event loop"
        )


# ---------------------------------------------------------------------------
# update_workflow_results – atomic index update
# ---------------------------------------------------------------------------


class TestUpdateWorkflowResults:
    def test_index_lock_returns_existing_lock_inserted_during_lazy_creation(
        self, monkeypatch, tmp_path
    ):
        class RacingLockDict(dict):
            def __init__(self, competing_lock):
                super().__init__()
                self.competing_lock = competing_lock

            def get(self, key, default=None):
                if key not in self:
                    super().__setitem__(key, self.competing_lock)
                return None

            def setdefault(self, key, default=None):
                if key not in self:
                    super().__setitem__(key, self.competing_lock)
                return super().setdefault(key, default)

        path = tmp_path / "wf" / "results.json"
        competing_lock = asyncio.Lock()
        lock_table = RacingLockDict(competing_lock)
        monkeypatch.setattr(results_module, "_workflow_index_locks", lock_table)

        lock = results_module._index_lock(path)

        assert lock is competing_lock
        assert lock_table[path] is competing_lock

    def test_creates_index_when_missing(self, tmp_path):
        wf_out = tmp_path / "wf"
        task_out = wf_out / "t1"
        task_out.mkdir(parents=True)

        t = _make_task("t1", task_out=task_out, workflow_out=wf_out)
        payload = {
            "schema_version": SCHEMA_VERSION_TASK,
            "task": "t1",
            "status": "completed",
            "ok": True,
            "values": {"ttft": 1.0},
        }
        update_workflow_results(t, payload)

        idx_path = wf_out / "results.json"
        assert idx_path.exists()
        idx = json.loads(idx_path.read_text())
        assert idx["schema_version"] == SCHEMA_VERSION_WORKFLOW
        assert idx["tasks"]["t1"]["values"] == {"ttft": 1.0}

    def test_merges_with_existing_tasks(self, tmp_path):
        wf_out = tmp_path / "wf"
        wf_out.mkdir(parents=True)
        # Pre-existing index from an earlier task in the workflow.
        (wf_out / "results.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION_WORKFLOW,
                    "workflow": "wf",
                    "tasks": {
                        "earlier": {
                            "status": "completed",
                            "ok": True,
                            "result_file": "earlier/result.json",
                            "values": {"x": 1},
                        }
                    },
                }
            )
        )

        task_out = wf_out / "t2"
        task_out.mkdir()
        t = _make_task("t2", task_out=task_out, workflow_out=wf_out)
        payload = {
            "schema_version": SCHEMA_VERSION_TASK,
            "task": "t2",
            "status": "completed",
            "ok": True,
            "values": {"y": 2},
        }
        update_workflow_results(t, payload)

        idx = json.loads((wf_out / "results.json").read_text())
        assert "earlier" in idx["tasks"]
        assert "t2" in idx["tasks"]
        assert idx["tasks"]["t2"]["values"] == {"y": 2}
