# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-processor: raw parsing, consumer filtering, and terminal overview."""

import csv

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

    summary = _read_csv(out_dir / "all" / "summary.csv")
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

    timeline = _read_csv(out_dir / "work" / "timeline.csv")
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
    svg_path = out_dir / "all" / "timeline.svg"
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
    png_path = out_dir / "all" / "timeline.png"
    assert png_path.is_file()
    assert png_path.stat().st_size > 0


def test_png_report_rendered_with_event_markers(tmp_path, fp):
    """The PNG marker path (axvline + rotated labels + legend) renders cleanly."""
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
    png_path = out_dir / "all" / "timeline.png"
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
    assert (out_dir / "all" / "summary.csv").is_file()
    assert (out_dir / "all" / "timeline.svg").is_file()
    assert not (out_dir / "all" / "timeline.png").exists()
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
    timeline = _read_csv(out_dir / "g0" / "timeline.csv")
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
    svg = (out_dir / "all" / "timeline.svg").read_text()
    assert "GPU mem used GiB" in svg
    assert "GPU mem used MiB" not in svg

    # CSV stays raw MiB (machine-readable source of truth, unscaled).
    summary = _read_csv(out_dir / "all" / "summary.csv")
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

    timeline = _read_csv(out_dir / "all" / "timeline.csv")
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
    timeline = _read_csv(out_dir / "g7" / "timeline.csv")
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
    timeline = _read_csv(out_dir / "server" / "timeline.csv")
    # Node filter keeps only hostA's samples.
    assert {row["hostname"] for row in timeline} == {"hostA"}
    assert (out_dir / "server" / "timeline.svg").is_file()
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
    prefill = _read_csv(out_dir / "prefill" / "timeline.csv")
    decode = _read_csv(out_dir / "decode" / "timeline.csv")
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
    folder = out_dir / "server__monitored_by__bench"
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

    svg = (out_dir / "all" / "timeline.svg").read_text()
    # Vertical markers (color = event, dash = task) + two explanatory legends.
    assert "stroke-dasharray" in svg
    assert "Events (color):" in svg and "Tasks (line style):" in svg
    assert "server" in svg and "aiperf" in svg
    assert ">submit<" in svg and ">ready<" in svg and ">done<" in svg
    # The event outside the plotted [origin, origin+window] is not drawn.
    assert "late_task" not in svg

    # The overview lists every recorded event (full log, not windowed).
    text = overview.read_text()
    assert "Task Events" in text
    assert "submit" in text and "server" in text
