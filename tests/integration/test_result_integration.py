# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test for the consolidated ``result`` task entry.

Runs a real local-backend workflow that exercises:
- regex-map ``result`` parsing from the per-task merged log
- file-source ``result`` via direct write to ``$SFLOW_TASK_RESULT_FILE``
- a downstream task asserting on both per-task ``result.json`` files and the
  workflow-level ``results.json`` index

See ``docs/developer/dev-notes/result-parsing.md``.
"""

import json
from pathlib import Path

from sflow.app.sflow import SflowApp
from sflow.core.results import SCHEMA_VERSION_TASK, SCHEMA_VERSION_WORKFLOW


def test_e2e_result_regex_map_and_file_source(tmp_path: Path):
    guide = Path(__file__).parent / "guide" / "sflow_local_result.yaml"
    cfg_path = tmp_path / "sflow.yaml"
    cfg_path.write_text(guide.read_text())

    out_dir = tmp_path / "out"
    SflowApp().run(
        file=cfg_path,
        dry_run=False,
        workspace_dir=tmp_path,
        output_dir=out_dir,
    )

    # Locate the single workflow run dir.
    runs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1, f"Expected a single run dir, got {runs}"
    run_dir = runs[0]

    # ------------------------------------------------------------------
    # 1. Regex-map task: per-task result.json with parsed values
    # ------------------------------------------------------------------
    log_result_path = run_dir / "benchmark_log" / "result.json"
    assert log_result_path.exists(), f"Missing {log_result_path}"
    log_payload = json.loads(log_result_path.read_text())

    assert log_payload["schema_version"] == SCHEMA_VERSION_TASK
    assert log_payload["task"] == "benchmark_log"
    assert log_payload["status"] == "COMPLETED"
    assert log_payload["ok"] is True

    # `last` aggregate (default) selects the second TTFT match.
    assert log_payload["values"]["ttft"] == 42.5
    assert log_payload["values"]["tps"] == 123.0
    # `auto` type infers int when the captured value has no decimal point.
    assert log_payload["values"]["latency_p99"] == 88

    # Metadata is recorded for each spec.
    meta = log_payload["metadata"]
    assert set(meta) == {"ttft", "tps", "latency_p99"}
    for entry in meta.values():
        assert entry["aggregate"] == "last"
        assert entry["type"] == "auto"

    # Source info points at the merged log file the parser used.
    assert log_payload["source"]["type"] == "log"
    assert log_payload["source"]["path"] == "benchmark_log.log"

    # Matches include line numbers and the matched text.
    ttft_matches = log_payload["matches"]["ttft"]
    assert len(ttft_matches) == 2
    assert ttft_matches[0]["value"] == 40.0
    assert ttft_matches[1]["value"] == 42.5
    assert all("TTFT" in m["text"] for m in ttft_matches)
    assert all(isinstance(m["line"], int) and m["line"] >= 1 for m in ttft_matches)

    # ------------------------------------------------------------------
    # 2. File-source task: source file is normalized into canonical schema
    # ------------------------------------------------------------------
    file_result_path = run_dir / "benchmark_file" / "result.json"
    assert file_result_path.exists()
    file_payload = json.loads(file_result_path.read_text())

    assert file_payload["schema_version"] == SCHEMA_VERSION_TASK
    assert file_payload["task"] == "benchmark_file"
    assert file_payload["status"] == "COMPLETED"
    assert file_payload["values"] == {"throughput": 999.5, "errors": 0}
    assert file_payload["source"]["type"] == "file"
    assert file_payload["source"]["path"] == "result.json"

    # ------------------------------------------------------------------
    # 3. Workflow-level results.json index
    # ------------------------------------------------------------------
    index_path = run_dir / "results.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text())

    assert index["schema_version"] == SCHEMA_VERSION_WORKFLOW
    assert set(index["tasks"]) >= {"benchmark_log", "benchmark_file"}

    log_entry = index["tasks"]["benchmark_log"]
    assert log_entry["ok"] is True
    assert log_entry["status"] == "COMPLETED"
    assert log_entry["result_file"] == "benchmark_log/result.json"
    assert log_entry["values"]["ttft"] == 42.5
    assert log_entry["values"]["tps"] == 123.0
    assert log_entry["values"]["latency_p99"] == 88

    file_entry = index["tasks"]["benchmark_file"]
    assert file_entry["status"] == "COMPLETED"
    assert file_entry["result_file"] == "benchmark_file/result.json"
    assert file_entry["values"] == {"throughput": 999.5, "errors": 0}

    # ------------------------------------------------------------------
    # 4. Downstream verify task ran successfully (its own assertions passed)
    # ------------------------------------------------------------------
    verify_summary = run_dir / "verify" / "verify.txt"
    assert verify_summary.exists()
    assert "ALL_CHECKS_PASSED" in verify_summary.read_text()
