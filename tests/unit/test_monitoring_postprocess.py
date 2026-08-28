# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-processor: raw parsing, consumer filtering, and terminal overview."""

import csv
import json
import re
from datetime import datetime

import pytest

from sflow.monitoring import postprocess_monitor_timeline as pp


def _write_raw(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "cpu_monitor_node_0_host.log").write_text(
        "100.000,50.00,1.0,1.0,1.0\n"
        "101.000,60.00,1.0,1.0,1.0\n"
        "102.000,70.00,1.0,1.0,1.0\n"
    )
    (raw_dir / "gpu_monitor_node_0_host.log").write_text(
        "100.000,0,10,5,40,40,50.0,1000,500,12000,2000\n"
        "101.000,0,20,5,41,40,60.0,1000,500,12000,2100\n"
        "102.000,0,30,5,42,40,70.0,1000,500,12000,2200\n"
    )


def _read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _task_log_line(timestamp, message):
    text = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    return f"{text} - sflow.task.bench - INFO - {message}\n"


def _write_cpu_samples(raw_dir, timestamps=range(100, 105)):
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "cpu_monitor_node_0_host.log").write_text(
        "".join(f"{ts:.3f},50.00,1.0,1.0,1.0\n" for ts in timestamps)
    )


def test_overview_and_full_report(tmp_path):
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_raw(raw_dir)
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"

    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "interval_ms": 1000,
        "png": False,
        "consumers": [
            {
                "name": "all",
                "owner": "all",
                "nodes": None,
                "gpus": None,
                "scopes": None,
                "start_ts": None,
                "end_ts": None,
                "report": True,
            }
        ],
    }
    result = pp.process(spec)
    assert result["sample_count"] > 0

    text = overview.read_text()
    assert "Sflow Monitor" in text
    assert "Metric Summary" in text
    assert "cpu_utilization_pct" in text
    assert "gpu_utilization_pct" in text
    assert "Timelines (cluster avg)" in text

    summary = _read_csv(out_dir / "lifecycle" / "all" / "summary.csv")
    resource_types = {row["resource_type"] for row in summary}
    assert "cpu" in resource_types and "gpu" in resource_types


def test_consumer_filtered_by_scope_and_time_window(tmp_path):
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_raw(raw_dir)
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"

    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "interval_ms": 1000,
        "png": False,
        "consumers": [
            {
                "name": "work",
                "owner": "task:work",
                "nodes": ["host"],
                "gpus": None,
                "scopes": ["cpu"],  # cpu only
                "start_ts": 101.0,  # window excludes the 100.0 sample
                "end_ts": 102.0,
                "report": True,
            }
        ],
    }
    pp.process(spec)

    timeline = _read_csv(out_dir / "lifecycle" / "work" / "timeline.csv")
    # Scope filter: only cpu rows.
    assert {row["resource_type"] for row in timeline} == {"cpu"}
    # Time window filter: only timestamps 101.0 and 102.0.
    timestamps = {float(row["timestamp"]) for row in timeline}
    assert timestamps == {101.0, 102.0}


def _spec(raw_dir, out_dir, overview, *, formats):
    return {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "interval_ms": 1000,
        "consumers": [
            {
                "name": "all",
                "owner": "all",
                "nodes": None,
                "gpus": None,
                "scopes": None,
                "start_ts": None,
                "end_ts": None,
                "report": True,
                "formats": formats,
            }
        ],
    }


def test_svg_report_rendered_pure_stdlib(tmp_path):
    """SVG (the default visual format) is produced with no third-party deps."""
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_raw(raw_dir)
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    pp.process(_spec(raw_dir, out_dir, overview, formats=["csv", "svg"]))
    svg_path = out_dir / "lifecycle" / "all" / "timeline.svg"
    assert svg_path.is_file()
    text = svg_path.read_text()
    assert text.startswith("<svg")
    assert "</svg>" in text
    assert "polyline" in text


def test_png_report_rendered(tmp_path, fp):
    """With png requested, a timeline.png is produced (matplotlib is in dev env)."""
    pytest.importorskip("matplotlib")
    fp.allow_unregistered(True)
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_raw(raw_dir)
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    pp.process(_spec(raw_dir, out_dir, overview, formats=["csv", "png"]))
    png_path = out_dir / "lifecycle" / "all" / "timeline.png"
    assert png_path.is_file()
    assert png_path.stat().st_size > 0


def test_png_report_rendered_with_event_markers(tmp_path, fp):
    """The PNG marker path (merged axvlines + direct annotations) renders cleanly."""
    pytest.importorskip("matplotlib")
    fp.allow_unregistered(True)
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_raw(raw_dir)
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    spec = _spec(raw_dir, out_dir, overview, formats=["png"])
    spec["task_events"] = [
        {"ts": 100.5, "task": "server", "event": "submit"},
        {"ts": 102.0, "task": "server", "event": "ready"},
    ]
    pp.process(spec)
    png_path = out_dir / "lifecycle" / "all" / "timeline.png"
    assert png_path.is_file()
    assert png_path.stat().st_size > 0


def test_png_skipped_when_matplotlib_unavailable(tmp_path, monkeypatch, capsys):
    """Without matplotlib, PNG is skipped but CSV + SVG are still written."""
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_raw(raw_dir)
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"

    monkeypatch.setattr(pp, "_matplotlib_available", lambda: False)
    pp.process(_spec(raw_dir, out_dir, overview, formats=["csv", "svg", "png"]))

    # CSV + SVG still produced; no PNG; single install hint emitted.
    assert (out_dir / "lifecycle" / "all" / "summary.csv").is_file()
    assert (out_dir / "lifecycle" / "all" / "timeline.svg").is_file()
    assert not (out_dir / "lifecycle" / "all" / "timeline.png").exists()
    err = capsys.readouterr().err
    assert "sflow[monitor]" in err
    assert err.count("PNG monitor timelines require matplotlib") == 1


def test_gpu_gpu_filter(tmp_path):
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Two GPUs per timestamp.
    (raw_dir / "gpu_monitor_node_0_host.log").write_text(
        "100.000,0,10,5,40,40,50.0,1000,500,12000,2000\n"
        "100.000,1,90,5,40,40,50.0,1000,500,12000,2000\n"
    )
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "interval_ms": 1000,
        "png": False,
        "consumers": [
            {
                "name": "g0",
                "owner": "task:g0",
                "nodes": ["host"],
                "gpus": [0],  # only GPU 0
                "scopes": ["gpu"],
                "start_ts": None,
                "end_ts": None,
                "report": True,
            }
        ],
    }
    pp.process(spec)
    timeline = _read_csv(out_dir / "lifecycle" / "g0" / "timeline.csv")
    resource_ids = {row["resource_id"] for row in timeline}
    assert resource_ids == {"0"}


def test_overview_humanizes_gpu_memory_to_gib(tmp_path):
    """Large GPU memory (MiB) renders rounded in GiB in the overview (no
    scientific notation); the CSV keeps the raw MiB source of truth."""
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # gpu cols: ts,index,util,mem_util,temp,mem_temp,power,clk_sm,clk_mem,mem_total,mem_used
    # mem_used 2580 -> 10100 MiB (~2.5 -> 9.9 GiB); mem_total 81920 MiB (80 GiB).
    (raw_dir / "gpu_monitor_node_0_host.log").write_text(
        "100.000,0,10,5,40,40,50.0,1000,500,81920,2580\n"
        "101.000,0,50,5,45,40,90.0,1000,500,81920,7300\n"
        "102.000,0,90,5,50,40,99.0,1000,500,81920,10100\n"
    )
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    pp.process(_spec(raw_dir, out_dir, overview, formats=["csv", "svg"]))

    text = overview.read_text()
    spark_line = next(
        ln for ln in text.splitlines() if "GPU mem used" in ln and "|" in ln
    )
    # GiB units, rounded, and never scientific notation.
    assert "GiB" in spark_line
    assert "e+" not in spark_line and "e-" not in spark_line
    assert "max=9.9" in spark_line  # 10100 MiB -> 9.86 GiB -> 9.9
    # Metric Summary row is GiB too (81920 MiB -> 80 GiB).
    assert any(
        "gpu_memory_total_mib" in ln and "GiB" in ln for ln in text.splitlines()
    )

    # The SVG timeline panel auto-adapts its label/units to match (no raw MiB).
    svg = (out_dir / "lifecycle" / "all" / "timeline.svg").read_text()
    assert "GPU mem used GiB" in svg
    assert "GPU mem used MiB" not in svg

    # CSV stays raw MiB (machine-readable source of truth, unscaled).
    summary = _read_csv(out_dir / "lifecycle" / "all" / "summary.csv")
    used = next(r for r in summary if r["metric_name"] == "gpu_memory_used_mib")
    assert used["metric_unit"] == "MiB"
    assert float(used["max_value"]) == 10100.0


def test_gpu_custom_fields_header_maps_columns(tmp_path):
    """A ``#fields=`` header lets custom --gpu-fields map to the right metrics
    (known fields keep curated names; unknown fields get a generic name)."""
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Non-default layout: index + power.draw + a field the curated map doesn't know.
    (raw_dir / "gpu_monitor_node_0_host.log").write_text(
        "#fields=index,power.draw,fan.speed\n"
        "100.000,0,50.0,30\n"
        "101.000,0,60.0,35\n"
    )
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    pp.process(_spec(raw_dir, out_dir, overview, formats=["csv"]))

    timeline = _read_csv(out_dir / "lifecycle" / "all" / "timeline.csv")
    names = {row["metric_name"] for row in timeline}
    # Known field keeps its curated name/unit; unknown field gets a generic name.
    assert "gpu_power_w" in names
    assert "gpu_fan_speed" in names
    # Default-layout metric names must NOT appear (we are not the default layout).
    assert "gpu_utilization_pct" not in names


def test_gpu_filter_keeps_all_when_subset_matches_nothing(tmp_path):
    """A requested GPU subset that matches no sampled GPU id keeps all GPUs
    (best-effort) instead of producing an empty report."""
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_raw(raw_dir)  # samples GPU id 0 only
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "interval_ms": 1000,
        "consumers": [
            {
                "name": "g7",
                "owner": "task:g7",
                "nodes": None,
                "gpus": [7],  # no GPU 7 was sampled
                "scopes": ["gpu"],
                "start_ts": None,
                "end_ts": None,
                "report": True,
            }
        ],
    }
    pp.process(spec)
    timeline = _read_csv(out_dir / "lifecycle" / "g7" / "timeline.csv")
    # Fallback kept the sampled GPU 0 rather than emitting nothing.
    assert {row["resource_id"] for row in timeline} == {"0"}


def test_task_reports_write_per_task_folders(tmp_path):
    """task_reports produce one resource-scoped folder each, listed in overview."""
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "cpu_monitor_node_0_hostA.log").write_text(
        "100.000,50.00,1.0,1.0,1.0\n101.000,60.00,1.0,1.0,1.0\n"
    )
    (raw_dir / "cpu_monitor_node_1_hostB.log").write_text(
        "100.000,10.00,1.0,1.0,1.0\n101.000,20.00,1.0,1.0,1.0\n"
    )
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "interval_ms": 1000,
        "consumers": [],
        "task_reports": [
            {
                "name": "server", "label": "server", "title": "server hw",
                "nodes": ["hostA"], "gpus": None, "scopes": ["cpu"],
                "formats": ["csv", "svg"], "window_tasks": [], "cross": False,
            }
        ],
        "task_events": [],
    }
    res = pp.process(spec)
    assert res["task_report_count"] == 1
    timeline = _read_csv(out_dir / "lifecycle" / "server" / "timeline.csv")
    # Node filter keeps only hostA's samples.
    assert {row["hostname"] for row in timeline} == {"hostA"}
    assert (out_dir / "lifecycle" / "server" / "timeline.svg").is_file()
    text = overview.read_text()
    assert "Task Reports" in text and "server" in text


def test_task_report_gpu_separation_on_shared_node(tmp_path):
    """Two tasks on the same node separate by their CUDA device subset."""
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{ts}.000,{g},{g * 10},5,40,40,50.0,1000,500,12000,2000"
        for ts in (100, 101)
        for g in range(4)
    ]
    (raw_dir / "gpu_monitor_node_0_host.log").write_text("\n".join(rows) + "\n")
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "interval_ms": 1000,
        "consumers": [],
        "task_reports": [
            {
                "name": "prefill", "label": "prefill", "title": "prefill hw",
                "nodes": ["host"], "gpus": [0, 1], "scopes": ["gpu"],
                "formats": ["csv"], "window_tasks": [], "cross": False,
            },
            {
                "name": "decode", "label": "decode", "title": "decode hw",
                "nodes": ["host"], "gpus": [2, 3], "scopes": ["gpu"],
                "formats": ["csv"], "window_tasks": [], "cross": False,
            },
        ],
        "task_events": [],
    }
    pp.process(spec)
    prefill = _read_csv(out_dir / "lifecycle" / "prefill" / "timeline.csv")
    decode = _read_csv(out_dir / "lifecycle" / "decode" / "timeline.csv")
    assert {r["resource_id"] for r in prefill} == {"0", "1"}
    assert {r["resource_id"] for r in decode} == {"2", "3"}


def test_cross_view_windowed_by_owner_with_title_note(tmp_path):
    """A B__monitored_by__A view clips to A's run window and labels the chart."""
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "cpu_monitor_node_0_host.log").write_text(
        "100.000,50.00,1.0,1.0,1.0\n"
        "101.000,60.00,1.0,1.0,1.0\n"
        "102.000,70.00,1.0,1.0,1.0\n"
        "103.000,80.00,1.0,1.0,1.0\n"
    )
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "interval_ms": 1000,
        "consumers": [],
        "task_reports": [
            {
                "name": "server__monitored_by__bench",
                "label": "server__monitored_by__bench",
                "title": "server hardware timeline (monitored by bench)",
                "nodes": ["host"], "gpus": None, "scopes": ["cpu"],
                "formats": ["csv", "svg"], "window_tasks": ["bench"], "cross": True,
            }
        ],
        # bench ran [101, 102]; the cross view must clip to that window.
        "task_events": [
            {"ts": 101.0, "task": "bench", "event": "submit"},
            {"ts": 102.0, "task": "bench", "event": "done"},
        ],
    }
    pp.process(spec)
    folder = out_dir / "lifecycle" / "server__monitored_by__bench"
    timeline = _read_csv(folder / "timeline.csv")
    assert {float(r["timestamp"]) for r in timeline} == {101.0, 102.0}
    assert "monitored by bench" in (folder / "timeline.svg").read_text()


def test_timeline_svg_and_overview_include_task_event_markers(tmp_path):
    """Task status changes render as vertical markers (+ legend) on the SVG and
    are listed in the overview; events outside the plotted window are dropped."""
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_raw(raw_dir)  # samples at ts 100, 101, 102
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "interval_ms": 1000,
        "consumers": [
            {
                "name": "all",
                "owner": "all",
                "nodes": None,
                "gpus": None,
                "scopes": None,
                "start_ts": None,
                "end_ts": None,
                "report": True,
                "formats": ["csv", "svg"],
            }
        ],
        "task_events": [
            {"ts": 100.5, "task": "server", "event": "submit"},
            {"ts": 101.5, "task": "server", "event": "ready"},
            {"ts": 102.0, "task": "aiperf", "event": "done"},
            {"ts": 9999.0, "task": "late_task", "event": "fail"},  # outside window
        ],
    }
    pp.process(spec)

    svg = (out_dir / "lifecycle" / "all" / "timeline.svg").read_text()
    # Transitions are labelled IN PLACE -- no legend to cross-reference. The old
    # encoding (colour = event, dash = task) needed two lookups per rule and
    # reused the device series colours, so both legends are gone on purpose.
    assert "Events (color):" not in svg and "Tasks (line style):" not in svg
    # Each transition names itself beside its own rule...
    assert ">server submit<" in svg and ">server ready<" in svg
    assert ">aiperf done<" in svg
    # ...and the two kinds live in SEPARATE bands: `submit` above the panels,
    # `ready`/`done` below them. That split is what stops neighbouring labels
    # colliding, so assert the geometry, not just the text.
    ys = {
        text: float(y)
        for y, text in re.findall(
            r'<text x="[\d.]+" y="([\d.]+)" text-anchor="middle" '
            r'font-size="9" fill="#5f6772">([^<]+)<', svg
        )
    }
    assert ys["server submit"] < ys["server ready"], ys
    assert ys["server submit"] < ys["aiperf done"], ys
    # Rules are drawn in neutral ink only -- never in a series colour, or a marker
    # would be indistinguishable from a GPU line on the same canvas.
    marker_strokes = set(re.findall(r'stroke="(#[0-9a-f]{6})" stroke-width="[\d.]+"'
                                    r'(?: stroke-dasharray="4 4")? opacity="0.85"', svg))
    assert marker_strokes <= {pp._MARKER_INK}, marker_strokes
    assert not marker_strokes & set(pp._SERIES_COLORS), "marker reused a series colour"
    # The event outside the plotted [origin, origin+window] is not drawn.
    assert "late_task" not in svg

    # The overview lists every recorded event (full log, not windowed).
    text = overview.read_text()
    assert "Task Events" in text
    assert "submit" in text and "server" in text


def test_log_marker_window_normalizes_prefixes_and_filters_samples(tmp_path):
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_cpu_samples(raw_dir)
    task_dir = tmp_path / "bench"
    task_dir.mkdir()
    (task_dir / "bench.log").write_text(
        _task_log_line(100.5, "0: WARMUP [done].*")
        + _task_log_line(102.5, "0: BENCHMARK_DONE")
    )
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "consumers": [],
        "task_reports": [
            {
                "name": "bench",
                "label": "bench",
                "nodes": ["host"],
                "gpus": None,
                "scopes": ["cpu"],
                "formats": ["csv"],
                "window_tasks": ["bench"],
                "log_window": {
                    "start": {
                        "pattern": [
                            "WARMUP [done].*",
                            r"re:^WARMUP \[done\]\.\*$",
                        ],
                        "select": "first",
                    },
                    "end": {
                        "pattern": ["re:^BENCHMARK_DONE$"],
                        "select": "last",
                    },
                },
            }
        ],
        "task_events": [
            {"ts": 99.0, "task": "bench", "event": "submit"},
            {"ts": 104.0, "task": "bench", "event": "done"},
        ],
    }

    pp.process(spec)

    timeline = _read_csv(out_dir / "windowed" / "bench" / "timeline.csv")
    assert {float(row["timestamp"]) for row in timeline} == {101.0, 102.0}
    artifact = json.loads((out_dir / "windowed" / "bench" / "window.json").read_text())
    assert artifact["status"] == "matched"
    assert artifact["start"]["line"] == "WARMUP [done].*"
    assert artifact["start"]["matched_patterns"] == [
        "WARMUP [done].*",
        r"re:^WARMUP \[done\]\.\*$",
    ]
    assert "window=matched" in overview.read_text()


def test_log_marker_window_uses_final_attempt_and_byte_offset_tiebreak(tmp_path):
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        _task_log_line(100.0, "WARMUP_DONE old")
        + _task_log_line(120.0, "BENCHMARK_DONE old")
        + _task_log_line(200.0, "WARMUP_DONE first")
        + _task_log_line(202.0, "WARMUP_DONE tied-first")
        + _task_log_line(202.0, "WARMUP_DONE tied-last")
        + _task_log_line(203.0, "BENCHMARK_DONE first")
        + _task_log_line(204.0, "BENCHMARK_DONE last")
    )
    events = [
        {"ts": 99.0, "task": "bench", "event": "submit"},
        {"ts": 150.0, "task": "bench", "event": "fail"},
        # The log prefix truncates 200.0005 to 200.000; its 1ms bucket still
        # overlaps this submit and must remain part of the final attempt.
        {"ts": 200.0005, "task": "bench", "event": "submit"},
        {"ts": 205.0, "task": "bench", "event": "done"},
    ]
    window = {
        "start": {"pattern": "WARMUP_DONE", "select": "last"},
        "end": {"pattern": "BENCHMARK_DONE", "select": "first"},
    }

    source = pp._resolve_log_source("bench", log_path, window, events)

    assert source["status"] == "matched"
    assert source["start"]["match_count"] == 3
    assert source["start"]["matched_patterns"] == ["WARMUP_DONE"]
    assert source["start"]["line"] == "WARMUP_DONE tied-last"
    assert source["end"]["line"] == "BENCHMARK_DONE first"


@pytest.mark.parametrize("end_timestamp", [201.0, 200.0])
def test_log_marker_window_rejects_zero_or_reversed_range(tmp_path, end_timestamp):
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        _task_log_line(201.0, "START") + _task_log_line(end_timestamp, "END")
    )
    source = pp._resolve_log_source(
        "bench",
        log_path,
        {
            "start": {"pattern": "START", "select": "first"},
            "end": {"pattern": "END", "select": "last"},
        },
        [
            {"ts": 199.0, "task": "bench", "event": "submit"},
            {"ts": 202.0, "task": "bench", "event": "done"},
        ],
    )
    assert source["status"] == "unresolved"
    assert source["error"] == "end marker not found after the selected start"


def test_cross_log_window_envelopes_all_owner_replicas(tmp_path):
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_cpu_samples(raw_dir)
    events = []
    for task, start, end in (("bench_0", 100.5, 102.5), ("bench_1", 101.5, 103.5)):
        task_dir = tmp_path / task
        task_dir.mkdir()
        (task_dir / f"{task}.log").write_text(
            _task_log_line(start, "START") + _task_log_line(end, "END")
        )
        events.extend(
            [
                {"ts": 99.0, "task": task, "event": "submit"},
                {"ts": 104.0, "task": task, "event": "done"},
            ]
        )
    out_dir = tmp_path / "sflow_monitor"
    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(tmp_path / "sflow_monitor.log"),
        "consumers": [],
        "task_reports": [
            {
                "name": "server__monitored_by__bench",
                "label": "server__monitored_by__bench",
                "nodes": ["host"],
                "gpus": None,
                "scopes": ["cpu"],
                "formats": ["csv"],
                "window_tasks": ["bench_0", "bench_1"],
                "cross": True,
                "log_window": {
                    "start": {"pattern": ["START"], "select": "first"},
                    "end": {"pattern": ["END"], "select": "last"},
                },
            }
        ],
        "task_events": events,
    }

    pp.process(spec)

    folder = out_dir / "windowed" / "server__monitored_by__bench"
    assert {float(row["timestamp"]) for row in _read_csv(folder / "timeline.csv")} == {
        101.0,
        102.0,
        103.0,
    }
    artifact = json.loads((folder / "window.json").read_text())
    assert artifact["status"] == "matched"
    assert len(artifact["sources"]) == 2


def test_unresolved_log_window_skips_detail_and_keeps_diagnostics(tmp_path, capsys):
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_cpu_samples(raw_dir)
    task_dir = tmp_path / "bench"
    task_dir.mkdir()
    (task_dir / "bench.log").write_text(_task_log_line(101.0, "START"))
    out_dir = tmp_path / "sflow_monitor"
    overview = tmp_path / "sflow_monitor.log"
    report = {
        "name": "bench",
        "label": "bench",
        "nodes": ["host"],
        "gpus": None,
        "scopes": ["cpu"],
        "formats": ["csv"],
        "window_tasks": ["bench"],
        "log_window": {
            "start": {"pattern": ["START"], "select": "first"},
            "end": {"pattern": ["END"], "select": "last"},
        },
    }
    spec = {
        "workflow_name": "wf",
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "overview_path": str(overview),
        "consumers": [],
        "task_reports": [report],
        "task_events": [
            {"ts": 99.0, "task": "bench", "event": "submit"},
            {"ts": 104.0, "task": "bench", "event": "done"},
        ],
    }

    pp.process(spec)

    folder = out_dir / "windowed" / "bench"
    assert not (folder / "timeline.csv").exists()
    # Unresolved windows are named apart so an empty report folder is self-explaining.
    assert not (folder / "window.json").exists()
    artifact = json.loads((folder / "window_not_found.json").read_text())
    assert artifact["status"] == "unresolved"
    assert "window=unresolved" in overview.read_text()
    assert "end marker not found" in capsys.readouterr().err


def test_log_window_resolution_is_cached_across_reports(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sflow_monitor" / "raw"
    _write_cpu_samples(raw_dir)
    task_dir = tmp_path / "bench"
    task_dir.mkdir()
    (task_dir / "bench.log").write_text(
        _task_log_line(100.5, "START") + _task_log_line(102.5, "END")
    )
    calls = 0
    original = pp._resolve_log_source

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pp, "_resolve_log_source", counted)
    window = {
        "start": {"pattern": ["START"], "select": "first"},
        "end": {"pattern": ["END"], "select": "last"},
    }
    reports = [
        {
            "name": name,
            "label": name,
            "nodes": ["host"],
            "gpus": None,
            "scopes": ["cpu"],
            "formats": ["csv"],
            "window_tasks": ["bench"],
            "log_window": window,
        }
        for name in ("natural", "cross")
    ]
    pp.process(
        {
            "workflow_name": "wf",
            "raw_dir": str(raw_dir),
            "out_dir": str(tmp_path / "sflow_monitor"),
            "overview_path": str(tmp_path / "sflow_monitor.log"),
            "consumers": [],
            "task_reports": reports,
            "task_events": [
                {"ts": 99.0, "task": "bench", "event": "submit"},
                {"ts": 104.0, "task": "bench", "event": "done"},
            ],
        }
    )
    assert calls == 1


# --- task-log marker window corner cases -----------------------------------


def _marker_events(task: str = "bench", start: float = 99.0, end: float = 205.0):
    return [
        {"ts": start, "task": task, "event": "submit"},
        {"ts": end, "task": task, "event": "done"},
    ]


def _marker_window():
    return {
        "start": {"pattern": "START", "select": "first"},
        "end": {"pattern": "END", "select": "last"},
    }


@pytest.mark.parametrize(
    ("raw_message", "normalized"),
    [
        ("0: START", "START"),
        ("[pod/bench-0/bench] START", "START"),
        ("[rank 0] START", "[rank 0] START"),
        ("worker: START", "worker: START"),
    ],
)
def test_only_known_transport_prefixes_are_removed(raw_message, normalized):
    parsed = pp._parse_timestamped_log_line(_task_log_line(100.0, raw_message).rstrip())
    assert parsed is not None
    _timestamp, _timestamp_text, message = parsed
    assert message == normalized


def test_unprefixed_line_is_not_a_timestamped_candidate():
    assert pp._parse_timestamped_log_line("0: START") is None
    assert pp._parse_timestamped_log_line("START") is None


def test_selected_line_is_bounded_and_invalid_utf8_does_not_abort(tmp_path):
    log_path = tmp_path / "bench.log"
    start_prefix = _task_log_line(100.0, "").encode()
    log_path.write_bytes(
        start_prefix.rstrip(b"\n")
        + b"START "
        + b"x" * 5000
        + b"\xff\n"
        + _task_log_line(102.0, "END").encode()
    )
    source = pp._resolve_log_source("bench", log_path, _marker_window(), _marker_events())
    assert source["status"] == "matched"
    assert len(source["start"]["line"]) == 4096
    assert source["start"]["line_truncated"] is True


@pytest.mark.parametrize(
    ("events", "write_log", "error"),
    [
        ([], True, "final attempt has no submit event"),
        (_marker_events(), False, "source log is missing"),
    ],
)
def test_missing_lifecycle_or_log_is_unresolved(tmp_path, events, write_log, error):
    log_path = tmp_path / "bench.log"
    if write_log:
        log_path.write_text(_task_log_line(100.0, "START") + _task_log_line(102.0, "END"))
    source = pp._resolve_log_source("bench", log_path, _marker_window(), events)
    assert source["status"] == "unresolved"
    assert error in source["error"]


def test_owner_without_terminal_event_still_resolves(tmp_path):
    """A service torn down at workflow end never goes terminal; the markers
    themselves close the window, so the search bound stays open."""
    log_path = tmp_path / "bench.log"
    log_path.write_text(_task_log_line(100.0, "START") + _task_log_line(102.0, "END"))
    source = pp._resolve_log_source(
        "bench",
        log_path,
        _marker_window(),
        [
            {"ts": 99.0, "task": "bench", "event": "submit"},
            {"ts": 99.5, "task": "bench", "event": "ready"},
        ],
    )
    assert source["status"] == "matched"
    assert source["end"]["line"] == "END"


def test_end_is_selected_after_the_chosen_start(tmp_path):
    """`start:last` + `end:first` means the run that follows the last warmup,
    not the first end anywhere in the attempt."""
    log_path = tmp_path / "bench.log"
    log_path.write_text(
        _task_log_line(100.0, "WARMUP_DONE")
        + _task_log_line(101.0, "RUN_DONE")
        + _task_log_line(102.0, "WARMUP_DONE")
        + _task_log_line(103.0, "RUN_DONE")
    )
    source = pp._resolve_log_source(
        "bench",
        log_path,
        {
            "start": {"pattern": "WARMUP_DONE", "select": "last"},
            "end": {"pattern": "RUN_DONE", "select": "first"},
        },
        _marker_events(start=99.0, end=110.0),
    )
    assert source["status"] == "matched"
    assert source["start"]["timestamp_iso"].endswith(",000")
    assert source["end"]["timestamp"] - source["start"]["timestamp"] == 1.0


def test_one_failed_owner_makes_aggregate_unresolved(tmp_path):
    out_dir = tmp_path / "sflow_monitor"
    for task, content in (
        ("bench_0", _task_log_line(100.0, "START") + _task_log_line(102.0, "END")),
        ("bench_1", _task_log_line(101.0, "START")),
    ):
        task_dir = tmp_path / task
        task_dir.mkdir()
        (task_dir / f"{task}.log").write_text(content)
    events = _marker_events("bench_0") + _marker_events("bench_1")
    start, end, artifact = pp._resolve_report_log_window(
        {"log_window": _marker_window(), "window_tasks": ["bench_0", "bench_1"]},
        out_dir,
        events,
        {},
    )
    assert start is None and end is None
    assert artifact["status"] == "unresolved"
    assert artifact["sources"][0]["status"] == "matched"
    assert (
        artifact["sources"][1]["error"]
        == "end marker not found after the selected start"
    )


def test_cluster_average_buckets_samples_from_nodes_that_never_share_a_timestamp():
    """One plotted point must average every node, not whichever one sampled first.

    Each collector samples on its own clock, so cross-node timestamps essentially
    never coincide. Grouping by exact timestamp puts ONE host in each point and
    the line sawtooths between a busy node and its idle peers -- which is what a
    real 3-node run looked like before bucketing.
    """
    rows = []
    for step in range(4):
        # Same sampling period (2s), staggered start per host: never equal.
        for offset, host, value in ((0.00, "a", 90.0), (0.37, "b", 0.0), (0.71, "c", 0.0)):
            rows.append(
                {
                    "timestamp": 100.0 + step * 2.0 + offset,
                    "hostname": host,
                    "resource_type": "gpu",
                    "metric_name": "gpu_utilization_pct",
                    "metric_value": value,
                }
            )

    series = pp._metric_timeseries(rows, "gpu", "gpu_utilization_pct")

    # One point per sampling period, each averaging all three hosts (90+0+0)/3.
    assert len(series) == 4, series
    assert [round(v, 3) for _ts, v in series] == [30.0] * 4
    # Without bucketing this is 12 points alternating 90, 0, 0 -- the sawtooth.
    assert pp._sample_period([(r["timestamp"], r["hostname"]) for r in rows]) == 2.0


def _gpu_rows(hosts_gpus, timestamps=(100.0, 102.0, 104.0)):
    rows = []
    for host, gpu, value in hosts_gpus:
        for ts in timestamps:
            rows.append(
                {
                    "timestamp": ts, "timestamp_iso": "t", "node_id": "0",
                    "hostname": host, "resource_type": "gpu", "resource_id": str(gpu),
                    "metric_name": "gpu_utilization_pct", "metric_value": float(value),
                    "metric_unit": "%", "metric_text": "", "source_log": "x",
                }
            )
    return rows


def test_gpu_panel_draws_one_line_per_device_with_a_legend(tmp_path):
    """A single averaged line hides how many GPUs are in scope and which is hot."""
    rows = _gpu_rows([("n1", 0, 90.0), ("n1", 1, 0.0), ("n1", 2, 0.0)])

    series = pp._metric_series_by_resource(rows, "gpu", "gpu_utilization_pct")
    assert [label for label, _ in series] == ["GPU 0", "GPU 1", "GPU 2"]

    svg_path = tmp_path / "timeline.svg"
    assert pp._render_svg(rows, svg_path, title="t")
    svg = svg_path.read_text()
    # One polyline per GPU, each a different colour.
    assert svg.count("<polyline") == 3, svg.count("<polyline")
    colours = {c for c in pp._SERIES_COLORS[:3] if f'stroke="{c}"' in svg}
    assert len(colours) == 3, colours
    # Named once in a shared legend, NOT repeated per panel with its own numbers.
    for device in ("GPU 0", "GPU 1", "GPU 2"):
        assert svg.count(f">{device}<") == 1, (device, svg.count(f">{device}<"))
    # One shared vertical scale per panel, spanning every device on it.
    assert "max 90.00" in svg and "min 0.00" in svg


def test_coincident_device_lines_stay_distinguishable(tmp_path):
    """Two GPUs with IDENTICAL values must not render as one line.

    A tensor-parallel task allocates the same footprint on every rank, so
    `gpu_memory_used_mib` for its GPUs is often identical to the byte -- the
    second polyline lands exactly on the first and the panel looks like it only
    ever had one device, which reads as a collection bug. Colour cannot fix that
    (nothing of the lower line is visible to be coloured); the dash can.
    """
    rows = _gpu_rows([("n1", 0, 50.0), ("n1", 1, 50.0)])

    svg_path = tmp_path / "timeline.svg"
    assert pp._render_svg(rows, svg_path, title="t")
    svg = svg_path.read_text()

    lines = re.findall(r"<polyline [^>]*>", svg)
    assert len(lines) == 2, lines
    dashes = [
        re.search(r'stroke-dasharray="([^"]+)"', ln).group(1)
        if "stroke-dasharray" in ln
        else ""
        for ln in lines
    ]
    assert dashes[0] != dashes[1], dashes
    # The legend swatch must carry its line's dash, or the legend stops matching.
    legend, _h = pp._build_device_legend_svg(["GPU 0", "GPU 1"], x0=8, max_x=800)
    swatches = [frag for frag in legend if frag.startswith("<line")]
    assert ("stroke-dasharray" in swatches[0]) is False
    assert "stroke-dasharray" in swatches[1], swatches[1]


def test_single_series_panels_stay_solid(tmp_path):
    """cpu/mem/disk/net have no per-device split -- dashing them is noise."""
    rows = _gpu_rows([("n1", 0, 10.0)])
    for ts in (100.0, 102.0, 104.0):
        rows.append(
            {
                "timestamp": ts, "timestamp_iso": "t", "node_id": "0",
                "hostname": "n1", "resource_type": "cpu", "resource_id": "cpu",
                "metric_name": "cpu_utilization_pct", "metric_value": 12.0,
                "metric_unit": "%", "metric_text": "", "source_log": "x",
            }
        )
    svg_path = tmp_path / "timeline.svg"
    assert pp._render_svg(rows, svg_path, title="t")
    assert "stroke-dasharray" not in svg_path.read_text()


def test_report_splits_into_one_image_per_node_named_by_hostname(tmp_path):
    """Charts for several nodes on one canvas are unreadable, so split them."""
    rows = _gpu_rows([("nodeA", 0, 10.0), ("nodeB", 0, 20.0)])
    consumer = {
        "name": "srv", "nodes": ["nodeA", "nodeB"], "gpus": [0], "scopes": None,
        "start_ts": None, "end_ts": None, "formats": ["csv", "svg"],
    }
    pp._write_consumer_report(rows, consumer, tmp_path, matplotlib_ok=False, events=[])

    folder = tmp_path / "lifecycle" / "srv"
    images = sorted(f.name for f in folder.glob("*.svg"))
    assert images == ["timeline.nodeA.svg", "timeline.nodeB.svg"], images
    # Not a combined chart, and each image covers only its own node.
    assert not (folder / "timeline.svg").exists()
    assert "nodeA" in (folder / "timeline.nodeA.svg").read_text()
    # CSVs stay combined -- they are the machine-readable source of truth.
    assert {r["hostname"] for r in _read_csv(folder / "timeline.csv")} == {"nodeA", "nodeB"}


def test_single_node_report_keeps_the_plain_timeline_name(tmp_path):
    rows = _gpu_rows([("only", 0, 10.0), ("only", 1, 20.0)])
    consumer = {
        "name": "srv", "nodes": ["only"], "gpus": [0, 1], "scopes": None,
        "start_ts": None, "end_ts": None, "formats": ["csv", "svg"],
    }
    pp._write_consumer_report(rows, consumer, tmp_path, matplotlib_ok=False, events=[])
    folder = tmp_path / "lifecycle" / "srv"
    assert [f.name for f in folder.glob("*.svg")] == ["timeline.svg"]



def test_legend_colour_matches_the_line_when_a_device_misses_one_metric(tmp_path):
    """A device absent from one family must not shift the colours after it.

    nvidia-smi reports N/A for some fields on some devices (MIG mode, a per-field
    collector error), so a GPU can appear in one panel and not the next. Keying
    the colour on each panel's own index then draws the survivor in the colour the
    shared legend has already given to the device that dropped out.
    """
    rows = _gpu_rows([("n1", 0, 10.0), ("n1", 1, 90.0)])
    # Only GPU 1 reports power -- GPU 0 is N/A for it.
    for ts in (100.0, 102.0, 104.0):
        rows.append(
            {
                "timestamp": ts, "timestamp_iso": "t", "node_id": "0",
                "hostname": "n1", "resource_type": "gpu", "resource_id": "1",
                "metric_name": "gpu_power_w", "metric_value": 250.0,
                "metric_unit": "W", "metric_text": "", "source_log": "x",
            }
        )

    svg_path = tmp_path / "timeline.svg"
    assert pp._render_svg(rows, svg_path, title="t")
    svg = svg_path.read_text()

    gpu0, gpu1 = pp._SERIES_COLORS[0], pp._SERIES_COLORS[1]
    strokes = re.findall(r'<polyline points="[^"]*" fill="none" stroke="(#[0-9a-f]{6})"', svg)
    # util panel: GPU 0 then GPU 1; power panel: GPU 1 ONLY -- still GPU 1's colour.
    assert strokes == [gpu0, gpu1, gpu1], strokes
    assert svg.count(">GPU 0<") == 1 and svg.count(">GPU 1<") == 1


def test_empty_gpu_list_reports_no_gpus_rather_than_all(tmp_path, capsys):
    """`[]` (task reserved no GPU) must not read as `None` (no subset -> all).

    `filter_rows` already drops every GPU row for `[]`; the overview has to agree,
    or it advertises `gpus=all` for a folder holding no GPU data at all.
    """
    rows = _gpu_rows([("n1", 0, 90.0)]) + [
        {
            "timestamp": 100.0, "timestamp_iso": "t", "node_id": "0",
            "hostname": "n1", "resource_type": "cpu", "resource_id": "cpu",
            "metric_name": "cpu_utilization_pct", "metric_value": 5.0,
            "metric_unit": "%", "metric_text": "", "source_log": "x",
        }
    ]
    kept = pp.filter_rows(rows, nodes=["n1"], gpus=[], scopes=None,
                          start_ts=None, end_ts=None)
    assert {r["resource_type"] for r in kept} == {"cpu"}
    # ... and `None` still keeps them (the "no subset -> all" half of the pair).
    assert any(
        r["resource_type"] == "gpu"
        for r in pp.filter_rows(rows, nodes=["n1"], gpus=None, scopes=None,
                                start_ts=None, end_ts=None)
    )

    assert pp.gpu_label(None) == "all"
    assert pp.gpu_label([]) == "none"
    assert pp.gpu_label([0, 1]) == "0,1"

    overview = pp.render_overview(
        rows, consumers=[], events=[], workflow_name="w",
        task_reports=[{"name": "cpu_task", "label": "cpu_task", "nodes": ["n1"],
                       "gpus": [], "scopes": ["cpu", "gpu"]}],
        out_dir=tmp_path, overview_path=tmp_path / "o.txt", interval_ms=1000,
    )
    line = next(ln for ln in overview.splitlines() if "cpu_task" in ln)
    assert "gpus=none" in line, line


def test_matched_window_covering_no_samples_warns(tmp_path, capsys):
    """A resolved window with no rows leaves empty CSVs and no chart.

    Nothing upstream warns (the markers DID match), so without this the folder is
    silently empty -- the shape a clock/timezone offset between the task's host
    and this one produces.
    """
    rows = _gpu_rows([("n1", 0, 90.0)])  # samples at t=100..104
    consumer = {
        "name": "bench", "nodes": ["n1"], "gpus": [0], "scopes": None,
        "start_ts": 900.0, "end_ts": 910.0,  # resolved window, far from the samples
        "formats": ["csv", "svg"],
    }
    pp._write_consumer_report(
        rows, consumer, tmp_path, matplotlib_ok=False, events=[], windowed=True
    )
    err = capsys.readouterr().err
    assert "matched but covers no samples" in err, err

    # A lifecycle report legitimately has no samples sometimes -- stay quiet there.
    pp._write_consumer_report(
        rows, consumer, tmp_path, matplotlib_ok=False, events=[], windowed=False
    )
    assert "covers no samples" not in capsys.readouterr().err


def test_near_simultaneous_events_merge_into_one_labelled_transition():
    """7 tasks changing state at once is ONE transition, not 7 rules.

    A workflow emits an event per task per change, so the raw markers read as a
    barcode. Merging is span-relative (not pixel-relative) so the SVG and PNG
    renderers, which have different geometry, always agree on the grouping.
    """
    events = [
        {"ts": 100.0, "task": "a", "event": "submit"},
        {"ts": 100.4, "task": "b", "event": "submit"},
        {"ts": 100.8, "task": "c", "event": "submit"},
        {"ts": 160.0, "task": "a", "event": "done"},
        {"ts": 190.0, "task": "b", "event": "fail"},
    ]
    selected = pp._select_marker_events(events, origin=100.0, max_elapsed=100.0)
    groups = pp._merge_marker_events(selected, origin=100.0, max_elapsed=100.0)

    # Bands split by MEANING: only `submit` starts something, so it goes on top;
    # `done` and `fail` both end a task, so both go underneath.
    assert [(g["label_top"], g["label_bottom"]) for g in groups] == [
        ("3 tasks submit", ""),
        ("", "a done"),
        ("", "b fail"),
    ]
    # A failure must never be silently dropped from the chart.
    assert any(g["label_bottom"] == "b fail" for g in groups)
    assert not any("fail" in str(g["label_top"]) for g in groups)

    # A tighter span must NOT glue unrelated transitions together: the same
    # events over a 10x longer plot still resolve to separate edges.
    wide = pp._merge_marker_events(
        pp._select_marker_events(events, origin=100.0, max_elapsed=1000.0),
        origin=100.0, max_elapsed=1000.0,
    )
    assert len(wide) == 3, [g["label_top"] for g in wide]


def test_marker_label_drops_whole_items_rather_than_cutting_mid_word():
    """`3 tasks submit +2 more` still says what happened; a mid-word cut does not."""
    pairs = [("submit", f"task_{i}") for i in range(3)]
    pairs += [("ready", "a_very_long_decode_server_name"), ("done", "env_check")]
    label = pp._marker_group_label(pairs)
    assert label == "3 tasks submit +2 more", label
    assert "\u2026" not in label
    # A single change keeps its real name -- that is the useful case.
    assert pp._marker_group_label([("ready", "prefill_0")]) == "prefill_0 ready"


def _skewed_spec(tmp_path, *, delta, latency=0.0, interval=5.0, span=24.0):
    """A run where the node clock is `delta` ahead, built from first principles."""
    t_start, t_stop = 1000.0, 1000.0 + span
    ticks = []
    t = t_start + latency
    while t <= t_stop:
        ticks.append(t + delta)          # stamped on the NODE clock
        t += interval
    raw = tmp_path / "sflow_monitor" / "raw"
    _write_cpu_samples(raw, timestamps=ticks)
    return raw, [{
        "name": "workflow", "owner": "workflow", "nodes": ["host"], "gpus": None,
        "scopes": None, "start_ts": t_start, "end_ts": t_stop, "report": True,
        "formats": ["csv"],
    }]


def test_clock_bracket_really_bounds_the_true_offset():
    """The bound must contain the truth for any latency / kill phase."""
    interval, t_start, t_stop = 5.0, 1000.0, 1024.0
    for delta in (-30.0, 0.0, 36.0):
        for latency in (0.0, 1.0, 2.0):
            ticks, t = [], t_start + latency
            while t <= t_stop:
                ticks.append(t + delta)
                t += interval
            lo, hi = pp._clock_bracket(
                ticks[0], ticks[-1], t_start, t_stop, interval
            )
            assert lo <= delta <= hi, (delta, latency, lo, hi)


def test_skewed_node_is_calibrated_from_the_files_alone(tmp_path, capsys):
    """No beacon, no extra handshake -- just the collection bracket and the samples."""
    raw, consumers = _skewed_spec(tmp_path, delta=36.0, latency=2.0)
    before = (raw / "cpu_monitor_node_0_host.log").read_bytes()

    rows = pp.parse_monitor_logs(raw)
    offsets = pp._estimate_clock_offsets(rows, consumers, 5000)
    estimate, lo, hi = offsets["host"]
    assert lo <= 36.0 <= hi, (lo, hi)
    assert abs(estimate - 36.0) <= 5.0, estimate  # within one sampling interval

    pp._align_sample_clocks(rows, offsets)
    # Samples now sit inside the driver-clock collection bracket, which is the
    # whole point -- before the shift they were ~36s past its end.
    assert min(float(r["timestamp"]) for r in rows) >= 1000.0 - 5.0
    assert max(float(r["timestamp"]) for r in rows) <= 1024.0 + 5.0
    assert "is ahead of this host" in capsys.readouterr().err
    assert (raw / "cpu_monitor_node_0_host.log").read_bytes() == before


def test_healthy_clock_is_never_corrected(tmp_path, capsys):
    """A correct clock always yields a bracket straddling zero -> no correction."""
    raw, consumers = _skewed_spec(tmp_path, delta=0.0, latency=1.0)
    rows = pp.parse_monitor_logs(raw)
    assert pp._estimate_clock_offsets(rows, consumers, 5000) == {}
    original = [float(r["timestamp"]) for r in rows]
    pp._align_sample_clocks(rows, {})
    assert [float(r["timestamp"]) for r in rows] == original
    assert capsys.readouterr().err == ""


def test_node_behind_the_driver_is_also_corrected(tmp_path):
    """Skew has a sign; a node running slow breaks reports exactly the same way."""
    raw, consumers = _skewed_spec(tmp_path, delta=-30.0, latency=1.0)
    offsets = pp._estimate_clock_offsets(pp.parse_monitor_logs(raw), consumers, 5000)
    estimate, lo, hi = offsets["host"]
    assert lo <= -30.0 <= hi and estimate < 0, (estimate, lo, hi)


def test_no_calibration_without_a_driver_side_bracket(tmp_path):
    """Consumers with no acquire/release stamps give nothing to measure against."""
    raw, consumers = _skewed_spec(tmp_path, delta=36.0)
    consumers[0]["start_ts"] = consumers[0]["end_ts"] = None
    assert pp._estimate_clock_offsets(pp.parse_monitor_logs(raw), consumers, 5000) == {}


def test_only_fail_is_red_cancel_stays_neutral(tmp_path):
    """`fail` earns a reserved status colour; `cancel` does not.

    sflow cancels still-running services on purpose at the end of every healthy
    workflow, so painting cancel red would cry wolf on runs where nothing is
    wrong. The colour also ships WITH the event's own text label, never alone.
    """
    rows = _gpu_rows([("n1", 0, 50.0)])
    svg_path = tmp_path / "timeline.svg"
    assert pp._render_svg(rows, svg_path, title="t", events=[
        {"ts": 100.0, "task": "srv", "event": "submit"},
        {"ts": 101.0, "task": "srv", "event": "done"},
        {"ts": 102.5, "task": "svc", "event": "cancel"},
        {"ts": 104.0, "task": "boom", "event": "fail"},
    ])
    svg = svg_path.read_text()

    # Anchor on stroke-width: marker rules are 1/1.4/1.6, label leaders are 0.8.
    # Opacity and dash both encode start-vs-end now, so matching on either makes
    # the regex miss half the rules -- which is exactly how this first broke.
    rules = re.findall(
        r'<line x1="[\d.]+"[^>]*stroke="(#[0-9a-f]{6})" stroke-width="1(?:\.\d)?"'
        r'(?: stroke-dasharray="[^"]+")? opacity=', svg
    )
    assert rules.count(pp._MARKER_FAIL_INK) == 1, rules   # exactly the fail rule
    assert set(rules) == {pp._MARKER_INK, pp._MARKER_FAIL_INK}, rules

    inks = dict(
        (t, c) for c, t in re.findall(r'font-size="9" fill="(#[0-9a-f]{6})">([^<]+)<', svg)
    )
    assert inks["boom fail"] == pp._MARKER_FAIL_INK, inks
    assert inks["svc cancel"] == pp._MARKER_LABEL_INK, inks
    assert inks["srv done"] == pp._MARKER_LABEL_INK, inks

    # The status red must stay distinguishable from the categorical series reds,
    # or a rule reads as a GPU line on a panel that plots one.
    assert pp._MARKER_FAIL_INK not in pp._SERIES_COLORS


def test_start_rules_are_dotted_and_end_rules_solid(tmp_path):
    """Line style separates "something started" from "something ended".

    Colour is already spoken for (neutral vs the reserved fail red), so the
    start/end distinction rides on stroke style instead -- readable in greyscale
    and without consulting the label.
    """
    rows = _gpu_rows([("n1", 0, 50.0)])
    svg_path = tmp_path / "timeline.svg"
    assert pp._render_svg(rows, svg_path, title="t", events=[
        {"ts": 100.0, "task": "srv", "event": "submit"},
        {"ts": 102.0, "task": "srv", "event": "done"},
    ])
    svg = svg_path.read_text()
    rules = re.findall(
        r'<line x1="([\d.]+)"[^>]*stroke-width="1(?:\.\d)?"'
        r'( stroke-dasharray="[^"]+")? opacity=', svg
    )
    assert len(rules) == 2, rules
    dotted = [bool(d) for _x, d in rules]
    assert dotted == [True, False], rules  # submit dotted, done solid

    # A merged group holding BOTH counts as an end: the stronger event wins, so
    # the rule is solid rather than advertising only the start.
    both = tmp_path / "both.svg"
    assert pp._render_svg(rows, both, title="t", events=[
        {"ts": 100.0, "task": "a", "event": "submit"},
        {"ts": 100.02, "task": "b", "event": "done"},  # inside the merge tolerance
    ])
    merged = re.findall(
        r'<line x1="[\d.]+"[^>]*stroke-width="1(?:\.\d)?"'
        r'( stroke-dasharray="[^"]+")? opacity=', both.read_text()
    )
    assert len(merged) == 1 and not merged[0], merged


def test_sample_emitted_during_teardown_does_not_fake_a_skew(tmp_path, capsys):
    """`release()` stamps end_ts BEFORE awaiting teardown, so a healthy collector
    can emit one more sample after it. That makes the two bounds contradict, and
    a contradiction must decline rather than clamp to a confident wrong answer."""
    raw = tmp_path / "sflow_monitor" / "raw"
    _write_cpu_samples(raw, timestamps=[1001, 1006, 1011, 1016, 1021, 1026])
    consumers = [{"nodes": ["host"], "start_ts": 1000.0, "end_ts": 1024.0}]

    assert pp._clock_bracket(1001, 1026, 1000.0, 1024.0, 5.0) is None
    rows = pp.parse_monitor_logs(raw)
    assert pp._estimate_clock_offsets(rows, consumers, 5000) == {}
    pp._align_sample_clocks(rows, {})
    assert capsys.readouterr().err == ""


def test_postprocessor_stays_standard_library_only():
    """It is designed to run as a materialized standalone script (see `main()` and
    `monitoring.postprocess_source`), which only holds if it imports no sflow."""
    from pathlib import Path as _P
    source = _P(pp.__file__).read_text(encoding="utf-8")
    offenders = re.findall(r"^\s*(?:from|import)\s+sflow.*$", source, re.M)
    assert not offenders, offenders


def test_start_and_end_rules_differ_on_more_than_dash_pattern(tmp_path):
    """Pattern alone is too subtle at 1px -- weight and opacity must differ too."""
    rows = _gpu_rows([("n1", 0, 50.0)])
    svg_path = tmp_path / "t.svg"
    assert pp._render_svg(rows, svg_path, title="t", events=[
        {"ts": 100.0, "task": "srv", "event": "submit"},
        {"ts": 102.0, "task": "srv", "event": "done"},
    ])
    found = re.findall(
        r'<line x1="[\d.]+"[^>]*stroke-width="(1(?:\.\d)?)"'
        r'( stroke-dasharray="[^"]+")? opacity="([\d.]+)"', svg_path.read_text()
    )
    (sw, dash, op), (esw, edash, eop) = found
    assert dash and not edash                      # start dotted, end solid
    assert float(esw) > float(sw)                  # end is heavier
    assert float(eop) > float(op)                  # end is more opaque


def test_values_never_render_in_scientific_notation():
    """`%g` flips to an exponent above 1e3 / below 1e-2 -- exactly where GPU power,
    clocks and network rates live, so charts and the summary table read
    `2.24e+03` instead of `2245`."""
    for value in (2244.63, 3186.0, 21591.0, 1.23e-5, 1e-9, -2244.63, 0.0, 0.5):
        assert "e" not in pp._fmt(value).lower(), (value, pp._fmt(value))
    assert pp._fmt(2244.63) == "2245"
    assert pp._fmt(90.0) == "90.00"          # unchanged for ordinary magnitudes
    assert pp._fmt(1e-9) == "0"              # underflow, not a bare "0."


def test_only_the_skewed_node_is_shifted_in_a_two_node_run(tmp_path, capsys):
    """Offsets are per host: a healthy node must not ride along with a bad one.

    Also covers the node-name fallback -- `nodeB` is named by no consumer (the
    planned-name vs reported-hostname mismatch `filter_rows` already tolerates),
    so its bracket falls back to the whole monitor's live period.
    """
    t_start, t_stop, interval, skew = 1000.0, 1024.0, 5.0, 36.0
    raw = tmp_path / "sflow_monitor" / "raw"
    raw.mkdir(parents=True)
    ticks = [t_start + i * interval for i in range(5)]
    raw.joinpath("cpu_monitor_node_0_nodeA.log").write_text(
        "".join(f"{t + skew:.3f},50.00,1.0,1.0,1.0\n" for t in ticks)
    )
    raw.joinpath("cpu_monitor_node_1_nodeB.log").write_text(
        "".join(f"{t + 1.0:.3f},50.00,1.0,1.0,1.0\n" for t in ticks)
    )
    consumers = [{"nodes": ["nodeA"], "start_ts": t_start, "end_ts": t_stop}]

    rows = pp.parse_monitor_logs(raw)
    offsets = pp._estimate_clock_offsets(rows, consumers, int(interval * 1000))
    assert set(offsets) == {"nodeA"}, offsets
    assert offsets["nodeA"][1] <= skew <= offsets["nodeA"][2]

    before = {r["hostname"]: float(r["timestamp"]) for r in rows}
    pp._align_sample_clocks(rows, offsets)
    after = {r["hostname"]: float(r["timestamp"]) for r in rows}
    assert after["nodeB"] == before["nodeB"], "healthy node must not move"
    assert after["nodeA"] < before["nodeA"], "skewed node must move back"

    err = capsys.readouterr().err
    assert "nodeA" in err and "nodeB" not in err, err
