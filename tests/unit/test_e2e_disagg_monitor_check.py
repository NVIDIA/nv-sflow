# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the disagg used_by_tasks monitor-targeting e2e checker."""

import csv
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_CSV_FIELDS = [
    "timestamp", "timestamp_iso", "node_id", "hostname", "resource_type",
    "resource_id", "metric_name", "metric_value", "metric_unit", "metric_text",
    "source_log",
]


def _load_checker():
    path = REPO_ROOT / "tests" / "e2e_tests" / "check_disagg_monitor.py"
    spec = importlib.util.spec_from_file_location("check_disagg_monitor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_timeline(folder: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows: (hostname, resource_type, resource_id)."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "timeline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for host, rtype, rid in rows:
            writer.writerow(
                {
                    "timestamp": "100.000", "timestamp_iso": "t", "node_id": "0",
                    "hostname": host, "resource_type": rtype, "resource_id": rid,
                    "metric_name": "m", "metric_value": "1.0", "metric_unit": "",
                    "metric_text": "", "source_log": "x",
                }
            )


def _mon(tmp_path: Path) -> Path:
    # Reports live under a group naming what sets their time range; the checker
    # must find them there, not at the pre-grouping flat root.
    return tmp_path / "run" / "sflow_monitor" / "lifecycle"


def test_targeting_pass_when_cross_matches_servers(tmp_path):
    mon = _mon(tmp_path)
    # Servers on GPU nodes; cross views match their servers exactly.
    _write_timeline(mon / "prefill_server", [("nodeA", "gpu", "0"), ("nodeA", "gpu", "1")])
    _write_timeline(mon / "prefill_server__monitored_by__benchmark", [("nodeA", "gpu", "0"), ("nodeA", "gpu", "1")])
    _write_timeline(mon / "decode_server", [("nodeB", "gpu", "0"), ("nodeB", "gpu", "1")])
    _write_timeline(mon / "decode_server__monitored_by__benchmark", [("nodeB", "gpu", "0"), ("nodeB", "gpu", "1")])
    # Benchmark client: different node, no GPU reservation.
    _write_timeline(mon / "benchmark", [("node0", "cpu", "system")])

    checker = _load_checker()
    rc, msgs = checker.check(tmp_path / "run")
    assert rc == 0, "\n".join(msgs)
    assert all("OK" in m or "tracks the server" in m for m in msgs)


def test_targeting_pass_for_single_agg_server(tmp_path):
    """Server set is discovered from the cross folders, so a single-server
    (aggregated / server-client) recipe is checked too."""
    mon = _mon(tmp_path)
    _write_timeline(mon / "agg_server", [("nodeA", "gpu", "0"), ("nodeA", "gpu", "1")])
    _write_timeline(mon / "agg_server__monitored_by__benchmark", [("nodeA", "gpu", "0"), ("nodeA", "gpu", "1")])
    _write_timeline(mon / "benchmark", [("node0", "cpu", "system")])

    checker = _load_checker()
    rc, msgs = checker.check(tmp_path / "run")
    assert rc == 0, "\n".join(msgs)
    assert any("agg_server" in m for m in msgs)


def test_targeting_fail_when_cross_samples_benchmark_resource(tmp_path):
    mon = _mon(tmp_path)
    # prefill cross WRONGLY samples the benchmark's node, not the server's.
    _write_timeline(mon / "prefill_server", [("nodeA", "gpu", "0")])
    _write_timeline(mon / "prefill_server__monitored_by__benchmark", [("node0", "gpu", "0")])
    _write_timeline(mon / "decode_server", [("nodeB", "gpu", "0")])
    _write_timeline(mon / "decode_server__monitored_by__benchmark", [("nodeB", "gpu", "0")])
    _write_timeline(mon / "benchmark", [("node0", "gpu", "0")])

    checker = _load_checker()
    rc, msgs = checker.check(tmp_path / "run")
    assert rc == 1
    assert any("prefill_server" in m and "FAIL" in m for m in msgs)


def test_targeting_fail_when_cross_report_has_no_timeline(tmp_path):
    mon = _mon(tmp_path)
    _write_timeline(mon / "prefill_server", [("nodeA", "gpu", "0")])
    # The cross folder exists (so it is discovered) but has no timeline.csv.
    (mon / "prefill_server__monitored_by__benchmark").mkdir(parents=True, exist_ok=True)
    _write_timeline(mon / "decode_server", [("nodeB", "gpu", "0")])
    _write_timeline(mon / "decode_server__monitored_by__benchmark", [("nodeB", "gpu", "0")])
    _write_timeline(mon / "benchmark", [("node0", "cpu", "system")])

    checker = _load_checker()
    rc, msgs = checker.check(tmp_path / "run")
    assert rc == 1
    assert any("missing cross report" in m for m in msgs)


def test_targeting_skip_when_no_cross_reports(tmp_path):
    mon = _mon(tmp_path)
    # A non-disagg workflow: natural views only, no *__monitored_by__benchmark.
    _write_timeline(mon / "workflow", [("nodeA", "gpu", "0")])
    _write_timeline(mon / "some_task", [("nodeA", "cpu", "system")])

    checker = _load_checker()
    rc, _msgs = checker.check(tmp_path / "run")
    assert rc == 2  # not applicable
