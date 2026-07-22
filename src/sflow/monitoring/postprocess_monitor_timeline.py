# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-process raw hardware-monitor logs into readable reports.

Runs ONCE after a workflow finishes (never during the run), over the shared raw
sample directory produced by ``hardware_monitor.py``. It emits:

* ``sflow_monitor.log`` -- a terminal-friendly overview (per-metric min/avg/max
  tables + ASCII sparkline timelines + per-consumer windows), styled after
  ``sflow_summary.log``.
* ``<out-dir>/<consumer>/timeline.csv`` + ``summary.csv`` -- one detailed view per
  reporting consumer (workflow / task), filtered to that consumer's nodes, GPU
  subset, scopes, and time window.
* Optional ``*.png`` timelines when ``--png`` is given and matplotlib is available.

Driven by a JSON report spec (``--spec``) written by sflow; without a spec it
produces a single ``all`` report over every sample. Standard library only,
except the optional PNG path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

MONITOR_FILE_RE = re.compile(
    r"(?P<resource_type>gpu|cpu|memory|disk|network)_monitor_node_"
    r"(?P<node_id>[^_]+)_(?P<hostname>.+)\.log$"
)

# Metric column layouts emitted by hardware_monitor.py (after the leading ts).
_GPU_METRICS = (
    ("gpu_utilization_pct", "%"),
    ("gpu_memory_utilization_pct", "%"),
    ("gpu_temperature_c", "C"),
    ("gpu_memory_temperature_c", "C"),
    ("gpu_power_w", "W"),
    ("gpu_clocks_sm_mhz", "MHz"),
    ("gpu_clocks_mem_mhz", "MHz"),
    ("gpu_memory_total_mib", "MiB"),
    ("gpu_memory_used_mib", "MiB"),
)
_CPU_METRICS = (
    ("cpu_utilization_pct", "%"),
    ("load_1m", "load"),
    ("load_5m", "load"),
    ("load_15m", "load"),
)
_MEMORY_METRICS = (
    ("memory_total_kib", "KiB"),
    ("memory_available_kib", "KiB"),
    ("memory_used_kib", "KiB"),
    ("memory_used_pct", "%"),
    ("swap_total_kib", "KiB"),
    ("swap_free_kib", "KiB"),
)
_DISK_METRICS = (
    ("disk_total_bytes", "bytes"),
    ("disk_used_bytes", "bytes"),
    ("disk_free_bytes", "bytes"),
    ("disk_used_pct", "%"),
)
_NETWORK_METRICS = (
    ("rx_bytes", "bytes"),
    ("tx_bytes", "bytes"),
    ("rx_bytes_per_s", "bytes/s"),
    ("tx_bytes_per_s", "bytes/s"),
    ("rx_packets_per_s", "packets/s"),
    ("tx_packets_per_s", "packets/s"),
)

# Key metrics rendered as sparklines in the terminal overview.
_OVERVIEW_SPARKLINES = (
    ("gpu", "gpu_utilization_pct", "GPU util %"),
    ("gpu", "gpu_memory_used_mib", "GPU mem used MiB"),
    ("gpu", "gpu_power_w", "GPU power W"),
    ("gpu", "gpu_temperature_c", "GPU temp C"),
    ("cpu", "cpu_utilization_pct", "CPU util %"),
    ("memory", "memory_used_pct", "Mem used %"),
    ("disk", "disk_used_pct", "Disk used %"),
    ("network", "rx_bytes_per_s", "Net RX MiB/s"),
    ("network", "tx_bytes_per_s", "Net TX MiB/s"),
)

_SPARK_TICKS = "▁▂▃▄▅▆▇█"

# Task lifecycle markers drawn on the timeline (color per status change). Order
# controls legend layout; unknown event types are ignored.
_EVENT_ORDER = ("submit", "ready", "done", "fail", "cancel")
_EVENT_STYLE = {
    "submit": "#1f77b4",  # blue
    "ready": "#2ca02c",  # green
    "done": "#7f7f7f",  # gray
    "fail": "#d62728",  # red
    "cancel": "#ff7f0e",  # orange
}


def _select_marker_events(
    events: "Iterable[dict[str, object]] | None",
    *,
    origin: float,
    max_elapsed: float,
) -> list[dict[str, object]]:
    """Known-style events inside the plotted ``[origin, origin+max_elapsed]`` window,
    sorted by timestamp (so markers and labels render left-to-right)."""
    selected: list[dict[str, object]] = []
    for ev in events or []:
        if str(ev.get("event")) not in _EVENT_STYLE:
            continue
        try:
            ts = float(ev.get("ts"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if origin <= ts <= origin + max_elapsed:
            selected.append({"ts": ts, "task": str(ev.get("task", "")), "event": str(ev["event"])})
    selected.sort(key=lambda e: float(e["ts"]))
    return selected


def _events_present(marker_events: list[dict[str, object]]) -> list[str]:
    """Event types present among the markers, in canonical legend order."""
    seen = {str(e["event"]) for e in marker_events}
    return [e for e in _EVENT_ORDER if e in seen]


# Distinguishable line styles cycled per task so each task's markers share one
# style (color still encodes the event type). Each entry is
# (name, svg_stroke_dasharray, matplotlib_linestyle); "" dasharray = solid.
_TASK_LINE_STYLES: tuple[tuple[str, str, object], ...] = (
    ("solid", "", "-"),
    ("dashed", "6 3", (0, (6, 3))),
    ("dotted", "1 3", (0, (1, 3))),
    ("dash-dot", "7 3 1 3", (0, (7, 3, 1, 3))),
    ("long-dash", "12 4", (0, (12, 4))),
    ("dash-dot-dot", "7 3 1 3 1 3", (0, (7, 3, 1, 3, 1, 3))),
)


def _assign_task_styles(
    marker_events: list[dict[str, object]],
) -> dict[str, tuple[str, object]]:
    """Map each task -> (svg_dasharray, matplotlib_linestyle), cycled stably.

    Tasks are sorted so the mapping is deterministic across SVG/PNG/consumers.
    With more tasks than styles the styles repeat (best-effort distinctness).
    """
    tasks = sorted({str(e["task"]) for e in marker_events})
    return {
        task: _TASK_LINE_STYLES[i % len(_TASK_LINE_STYLES)][1:]
        for i, task in enumerate(tasks)
    }


def _build_legend_svg(
    present_events: list[str],
    task_styles: dict[str, tuple[str, object]],
    *,
    x0: float,
    max_x: float,
) -> tuple[list[str], float]:
    """Build the bottom legend (event colors + task line styles) in local coords.

    Returns the SVG fragments (y measured from 0) and the total legend height, so
    the caller can size the canvas and translate the group into place.
    """
    parts: list[str] = []
    row_h = 16.0
    y = 8.0

    def _emit_row(heading: str, heading_w: float, items: list[tuple[str, str, str]]) -> None:
        # items: (label, swatch_color, swatch_dasharray)
        nonlocal y
        parts.append(
            f'<text x="{x0:.0f}" y="{y + 3:.0f}" font-size="9" '
            f'fill="#222222">{heading}</text>'
        )
        x = x0 + heading_w
        for label, color, dash in items:
            label = label if len(label) <= 16 else label[:15] + "\u2026"
            item_w = 22 + 4 + 6.0 * len(label) + 16
            if x + item_w > max_x and x > x0 + heading_w:
                x = x0 + heading_w
                y += row_h
            da = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x + 20:.0f}" y2="{y:.0f}" '
                f'stroke="{color}" stroke-width="2"{da}/>'
            )
            parts.append(
                f'<text x="{x + 24:.0f}" y="{y + 3:.0f}" font-size="9" '
                f'fill="#444444">{_xml_escape(label)}</text>'
            )
            x += item_w
        y += row_h

    if present_events:
        _emit_row(
            "Events (color):",
            100,
            [(e, _EVENT_STYLE[e], "") for e in present_events],
        )
    if task_styles:
        _emit_row(
            "Tasks (line style):",
            118,
            [(task, "#555555", dash) for task, (dash, _ls) in task_styles.items()],
        )
    return parts, y
_CSV_FIELDS = [
    "timestamp",
    "timestamp_iso",
    "node_id",
    "hostname",
    "resource_type",
    "resource_id",
    "metric_name",
    "metric_value",
    "metric_unit",
    "metric_text",
    "source_log",
]
_SUMMARY_FIELDS = [
    "resource_type",
    "metric_name",
    "metric_unit",
    "datapoints",
    "min_value",
    "avg_value",
    "max_value",
]


# nvidia-smi --query-gpu field -> (metric_name, unit). Used to map the GPU log's
# columns to metrics; reproduces _GPU_METRICS for the default field set and lets
# custom `monitor.scopes.gpu.fields` still resolve to meaningful names.
_GPU_FIELD_MAP = {
    "utilization.gpu": ("gpu_utilization_pct", "%"),
    "utilization.memory": ("gpu_memory_utilization_pct", "%"),
    "temperature.gpu": ("gpu_temperature_c", "C"),
    "temperature.memory": ("gpu_memory_temperature_c", "C"),
    "power.draw": ("gpu_power_w", "W"),
    "clocks.sm": ("gpu_clocks_sm_mhz", "MHz"),
    "clocks.mem": ("gpu_clocks_mem_mhz", "MHz"),
    "memory.total": ("gpu_memory_total_mib", "MiB"),
    "memory.used": ("gpu_memory_used_mib", "MiB"),
}

_GPU_FIELDS_HEADER = "#fields="


def _gpu_metrics_from_fields(fields_csv: str) -> list[tuple[str, str]]:
    """Build the GPU ``(metric_name, unit)`` list from a ``#fields=`` header.

    The first field is the GPU id (the row's ``resource_id``), so it is dropped.
    Known nvidia-smi fields map to curated names/units; unknown fields get a
    generic ``gpu_<sanitized>`` name with an empty unit.
    """
    tokens = [t.strip() for t in fields_csv.split(",") if t.strip()]
    metrics: list[tuple[str, str]] = []
    for token in tokens[1:]:
        name, unit = _GPU_FIELD_MAP.get(token.lower(), ("", ""))
        if not name:
            name = "gpu_" + re.sub(r"[^0-9a-z]+", "_", token.lower()).strip("_")
            unit = ""
        metrics.append((name, unit))
    return metrics


def parse_monitor_log(monitor_log_path: Path) -> list[dict[str, object]]:
    """Parse one ``<scope>_monitor_node_<id>_<host>.log`` file into metric rows."""
    match = MONITOR_FILE_RE.match(monitor_log_path.name)
    if not match:
        return []
    resource_type = match.group("resource_type")
    node_id = match.group("node_id")
    hostname = match.group("hostname")
    rows: list[dict[str, object]] = []

    raw_lines = monitor_log_path.read_text().splitlines()

    # A GPU log may lead with a `#fields=` header describing its column layout;
    # honor it so a custom --gpu-fields set still maps to the correct metrics.
    # Without a header (older logs / standalone runs) fall back to the default.
    gpu_metrics = _GPU_METRICS
    if resource_type == "gpu":
        for raw_line in raw_lines:
            if raw_line.startswith(_GPU_FIELDS_HEADER):
                parsed = _gpu_metrics_from_fields(raw_line[len(_GPU_FIELDS_HEADER):])
                if parsed:
                    gpu_metrics = tuple(parsed)
                break

    for raw_line in raw_lines:
        if not raw_line.strip() or raw_line.startswith("#"):
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        try:
            timestamp = float(parts[0])
        except (IndexError, ValueError):
            continue
        timestamp_iso = datetime.fromtimestamp(timestamp).isoformat(
            sep=" ", timespec="milliseconds"
        )

        def _base_row(
            resource_id: str, metric_name: str, metric_value: object, metric_unit: str
        ) -> dict[str, object]:
            return {
                "timestamp": timestamp,
                "timestamp_iso": timestamp_iso,
                "node_id": node_id,
                "hostname": hostname,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_unit": metric_unit,
                "metric_text": "",
                "source_log": str(monitor_log_path),
            }

        if resource_type == "gpu":
            if len(parts) >= 3 and parts[1] == "ERROR":
                err = _base_row("all", "gpu_monitor_error", "", "")
                err["metric_text"] = ",".join(parts[2:])
                rows.append(err)
                continue
            # parts: ts, gpu_id, <one value per metric>.
            if len(parts) < 2 + len(gpu_metrics):
                continue
            resource_id = parts[1]
            metrics = [
                (name, parts[2 + i], unit)
                for i, (name, unit) in enumerate(gpu_metrics)
            ]
        elif resource_type == "cpu":
            if len(parts) < 5:
                continue
            resource_id = "system"
            metrics = [
                (name, parts[1 + i], unit)
                for i, (name, unit) in enumerate(_CPU_METRICS)
            ]
        elif resource_type == "memory":
            if len(parts) < 7:
                continue
            resource_id = "system"
            metrics = [
                (name, parts[1 + i], unit)
                for i, (name, unit) in enumerate(_MEMORY_METRICS)
            ]
        elif resource_type == "disk":
            if len(parts) < 6:
                continue
            resource_id = parts[1]
            metrics = [
                (name, parts[2 + i], unit)
                for i, (name, unit) in enumerate(_DISK_METRICS)
            ]
        elif resource_type == "network":
            if len(parts) < 7:
                continue
            resource_id = "aggregate"
            metrics = [
                (name, parts[1 + i], unit)
                for i, (name, unit) in enumerate(_NETWORK_METRICS)
            ]
        else:
            continue

        for metric_name, raw_value, metric_unit in metrics:
            try:
                metric_value: object = float(raw_value)
            except ValueError:
                metric_value = raw_value
            rows.append(_base_row(resource_id, metric_name, metric_value, metric_unit))
    return rows


def parse_monitor_logs(raw_dir: Path) -> list[dict[str, object]]:
    """Parse every monitor log under ``raw_dir`` into a sorted list of rows."""
    rows: list[dict[str, object]] = []
    if not raw_dir.is_dir():
        return rows
    for monitor_log_path in sorted(raw_dir.iterdir()):
        if monitor_log_path.is_file():
            rows.extend(parse_monitor_log(monitor_log_path))
    rows.sort(
        key=lambda row: (
            float(row["timestamp"]),
            str(row["resource_type"]),
            str(row["node_id"]),
            str(row["resource_id"]),
            str(row["metric_name"]),
        )
    )
    return rows


def filter_rows(
    rows: list[dict[str, object]],
    *,
    nodes: list[str] | None,
    gpus: list[int] | None,
    scopes: list[str] | None,
    start_ts: float | None,
    end_ts: float | None,
) -> list[dict[str, object]]:
    """Filter rows to a consumer's nodes / GPU subset / scopes / time window."""
    node_set = set(nodes) if nodes else None
    # Best effort: if a node set is given but matches nothing (e.g. local backend
    # where assigned node names differ from the sampled hostname), keep all nodes.
    if node_set is not None and not any(
        str(row["hostname"]) in node_set for row in rows
    ):
        node_set = None
    gpu_set = {str(g) for g in gpus} if gpus else None
    # Best effort (mirrors the node fallback above): if a GPU subset is requested
    # but matches no sampled GPU id -- e.g. the consumer's logical CUDA indices
    # differ from nvidia-smi's physical `index`, or no gpu scope was collected --
    # keep all GPUs rather than emit an empty report.
    if gpu_set is not None and not any(
        str(row["resource_type"]) == "gpu" and str(row["resource_id"]) in gpu_set
        for row in rows
    ):
        gpu_set = None
    scope_set = set(scopes) if scopes else None

    out: list[dict[str, object]] = []
    for row in rows:
        if scope_set is not None and str(row["resource_type"]) not in scope_set:
            continue
        if node_set is not None and str(row["hostname"]) not in node_set:
            continue
        if (
            gpu_set is not None
            and row["resource_type"] == "gpu"
            and str(row["resource_id"]) not in gpu_set
        ):
            continue
        ts = float(row["timestamp"])
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        out.append(row)
    return out


def summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("metric_value")
        if isinstance(value, (int, float)) and not math.isnan(value):
            key = (
                str(row["resource_type"]),
                str(row["metric_name"]),
                str(row["metric_unit"]),
            )
            grouped[key].append(float(value))
    summary: list[dict[str, object]] = []
    for key in sorted(grouped):
        values = grouped[key]
        summary.append(
            {
                "resource_type": key[0],
                "metric_name": key[1],
                "metric_unit": key[2],
                "datapoints": len(values),
                "min_value": min(values),
                "avg_value": sum(values) / len(values),
                "max_value": max(values),
            }
        )
    return summary


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return ""
    # Downsample to `width` buckets by averaging.
    if len(values) > width:
        bucket = len(values) / width
        sampled = []
        for i in range(width):
            lo = int(i * bucket)
            hi = max(lo + 1, int((i + 1) * bucket))
            chunk = values[lo:hi]
            sampled.append(sum(chunk) / len(chunk))
        values = sampled
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        return _SPARK_TICKS[0] * len(values)
    out = []
    for value in values:
        idx = int((value - lo) / span * (len(_SPARK_TICKS) - 1))
        out.append(_SPARK_TICKS[max(0, min(idx, len(_SPARK_TICKS) - 1))])
    return "".join(out)


def _metric_timeseries(
    rows: list[dict[str, object]], resource_type: str, metric_name: str
) -> list[tuple[float, float]]:
    """Cluster-averaged ``(timestamp, value)`` series for a metric."""
    by_ts: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["resource_type"] != resource_type or row["metric_name"] != metric_name:
            continue
        value = row.get("metric_value")
        if isinstance(value, (int, float)) and not math.isnan(value):
            by_ts[float(row["timestamp"])].append(float(value))
    return [(ts, sum(v) / len(v)) for ts, v in sorted(by_ts.items())]


def _series_for_metric(
    rows: list[dict[str, object]], resource_type: str, metric_name: str
) -> list[float]:
    """Cluster-averaged values (one per timestamp) for a metric."""
    return [v for _ts, v in _metric_timeseries(rows, resource_type, metric_name)]


def _plottable_families(
    rows: list[dict[str, object]],
) -> list[tuple[str, str, str]]:
    """``(label, resource_type, metric_name)`` for overview metrics that have data.

    The shared panel/family selection for both the SVG and PNG timelines.
    """
    return [
        (label, resource_type, metric_name)
        for resource_type, metric_name, label in _OVERVIEW_SPARKLINES
        if _metric_timeseries(rows, resource_type, metric_name)
    ]


def _fmt(value: float) -> str:
    if abs(value) >= 1000 or (value and abs(value) < 0.01):
        return f"{value:.3g}"
    return f"{value:.2f}"


# Absolute byte-size units (NOT rates like "bytes/s"), smallest -> largest.
_SIZE_UNITS = ("bytes", "KiB", "MiB", "GiB", "TiB", "PiB")


def _scale_size(values: list[float], base_unit: str) -> tuple[list[float], str]:
    """Scale same-unit byte-size values to one readable unit (chosen by the max
    magnitude), e.g. MiB -> GiB once >= 1024 MiB. Returns ``(scaled, display_unit)``.
    Values in a non-size unit (``%``, ``W``, ``bytes/s`` rates, ...) pass through.
    """
    if base_unit not in _SIZE_UNITS or not values:
        return values, base_unit
    start = _SIZE_UNITS.index(base_unit)
    idx = start
    ref = max(abs(v) for v in values)
    while ref >= 1024.0 and idx < len(_SIZE_UNITS) - 1:
        ref /= 1024.0
        idx += 1
    factor = 1024.0 ** (idx - start)
    return [v / factor for v in values], _SIZE_UNITS[idx]


def _fmt_size(value: float, unit: str) -> str:
    """Render an (already-scaled) byte-size value as a clean, non-scientific number:
    whole numbers for bytes/KiB/MiB, one trimmed decimal for GiB and up (so GPU
    memory reads as ``9.9`` / ``10`` GiB, never ``1.01e+04`` MiB)."""
    if unit in ("bytes", "KiB", "MiB"):
        return f"{round(value)}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def render_overview(
    rows: list[dict[str, object]],
    *,
    workflow_name: str,
    out_dir: Path,
    overview_path: Path,
    interval_ms: int | None,
    consumers: list[dict[str, object]],
    events: "Iterable[dict[str, object]] | None" = None,
    task_reports: "list[dict[str, object]] | None" = None,
) -> str:
    """Render the terminal-friendly ``sflow_monitor.log`` overview text."""
    lines: list[str] = ["Sflow Monitor", "============="]
    lines.append(f"Workflow : {workflow_name}")

    hostnames = sorted({str(r["hostname"]) for r in rows})
    scopes_present = sorted({str(r["resource_type"]) for r in rows})
    timestamps = [float(r["timestamp"]) for r in rows]
    if timestamps:
        start_iso = datetime.fromtimestamp(min(timestamps)).isoformat(
            sep=" ", timespec="seconds"
        )
        end_iso = datetime.fromtimestamp(max(timestamps)).isoformat(
            sep=" ", timespec="seconds"
        )
        duration = max(timestamps) - min(timestamps)
    else:
        start_iso = end_iso = "(none)"
        duration = 0.0

    lines.append(f"Nodes    : {', '.join(hostnames) or '(none)'}")
    lines.append(f"Scopes   : {', '.join(scopes_present) or '(none)'}")
    if interval_ms is not None:
        lines.append(f"Interval : {interval_ms} ms")
    lines.append(f"Window   : {start_iso} -> {end_iso} ({duration:.1f}s)")
    lines.append(f"Samples  : {len(rows)}")
    lines.append(f"Reports  : {out_dir}")
    lines.append(f"Overview : {overview_path}")

    # Per-metric min/avg/max table.
    lines.extend(["", "Metric Summary", "--------------"])
    summary = summary_rows(rows)
    if summary:
        name_w = max(11, *(len(str(s["metric_name"])) for s in summary))
        type_w = max(8, *(len(str(s["resource_type"])) for s in summary))
        header = (
            f"{'resource':<{type_w}}  {'metric':<{name_w}}  "
            f"{'unit':<8}  {'datapoints':>10}  {'min':>10}  {'avg':>10}  {'max':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for s in summary:
            unit = str(s["metric_unit"])
            vals = [
                float(s["min_value"]),
                float(s["avg_value"]),
                float(s["max_value"]),
            ]
            # Humanize byte-size metrics (MiB/KiB/bytes -> GiB/... rounded); other
            # units (%, W, C, load, bytes/s rates) keep the generic formatter.
            scaled, disp_unit = _scale_size(vals, unit)
            if unit in _SIZE_UNITS:
                cells = [_fmt_size(v, disp_unit) for v in scaled]
            else:
                cells = [_fmt(v) for v in vals]
            lines.append(
                f"{str(s['resource_type']):<{type_w}}  "
                f"{str(s['metric_name']):<{name_w}}  "
                f"{disp_unit:<8}  "
                f"{int(s['datapoints']):>10}  "
                f"{cells[0]:>10}  {cells[1]:>10}  {cells[2]:>10}"
            )
    else:
        lines.append("(no numeric samples collected)")

    # Sparkline timelines for key metrics (cluster average).
    lines.extend(["", "Timelines (cluster avg)", "-----------------------"])
    any_spark = False
    label_w = max(len(label) for _t, _m, label in _OVERVIEW_SPARKLINES)
    for resource_type, metric_name, label in _OVERVIEW_SPARKLINES:
        series = _series_for_metric(rows, resource_type, metric_name)
        if not series:
            continue
        any_spark = True
        # Same scaling/units as the Metric Summary table and the SVG/PNG timelines.
        scaled, disp_label, stat = _scaled_metric_series(metric_name, label, series)
        spark = _sparkline(scaled)
        lines.append(
            f"{disp_label:<{label_w}} |{spark}| "
            f"min={stat(min(scaled))} avg={stat(sum(scaled) / len(scaled))} max={stat(max(scaled))}"
        )
    if not any_spark:
        lines.append("(no timeline metrics collected)")

    # Error rows (e.g. nvidia-smi failures).
    errors = [r for r in rows if r["metric_name"] == "gpu_monitor_error"]
    if errors:
        lines.extend(["", "GPU Collection Errors", "---------------------"])
        seen: set[str] = set()
        for row in errors:
            text = str(row.get("metric_text") or "")
            key = f"{row['hostname']}:{text}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{row['hostname']}: {text}")

    # Per-consumer windows.
    lines.extend(["", "Monitors", "--------"])
    if consumers:
        for consumer in consumers:
            name = str(consumer.get("name", "?"))
            owner = str(consumer.get("owner", name))
            nodes = consumer.get("nodes") or []
            gpus = consumer.get("gpus")
            scopes = consumer.get("scopes") or []
            start_ts = consumer.get("start_ts")
            end_ts = consumer.get("end_ts")
            window = "(full run)"
            if isinstance(start_ts, (int, float)) and isinstance(end_ts, (int, float)):
                window = f"{end_ts - start_ts:.1f}s"
            gpu_str = "all" if not gpus else ",".join(str(g) for g in gpus)
            report = " [report]" if consumer.get("report") else ""
            lines.append(
                f"- {name} ({owner}){report}: nodes={', '.join(nodes) or 'all'} "
                f"gpus={gpu_str} scopes={','.join(scopes) or 'all'} window={window}"
            )
    else:
        lines.append("(none)")

    # Per-task resource reports (one folder per entry under the reports dir).
    if task_reports:
        lines.extend(["", "Task Reports", "------------"])
        for report in task_reports:
            label = str(report.get("label") or report.get("name") or "?")
            nodes = report.get("nodes") or []
            gpus = report.get("gpus")
            gpu_str = "all" if not gpus else ",".join(str(g) for g in gpus)
            scopes = report.get("scopes") or []
            tag = " [cross]" if report.get("cross") else ""
            node_str = ", ".join(str(n) for n in nodes) or "all"
            scope_str = ",".join(str(s) for s in scopes) or "all"
            lines.append(
                f"- {label}{tag}: nodes={node_str} gpus={gpu_str} scopes={scope_str}"
            )

    # Task lifecycle events (the markers overlaid on the SVG/PNG timelines).
    marker_events: list[tuple[float, str, str]] = []
    for e in events or []:
        event = str(e.get("event"))
        if event not in _EVENT_STYLE:
            continue
        try:
            marker_events.append((float(e["ts"]), event, str(e.get("task", ""))))  # type: ignore[arg-type]
        except (TypeError, ValueError, KeyError):
            continue
    if marker_events:
        lines.extend(["", "Task Events", "-----------"])
        ev_origin = min(timestamps) if timestamps else min(t for t, _e, _n in marker_events)
        for ts, event, task in sorted(marker_events):
            iso = datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")
            lines.append(f"  +{ts - ev_origin:7.1f}s  {iso}  {event:<6}  {task}")

    lines.extend(["", f"Detailed reports: {out_dir}"])
    return "\n".join(lines) + "\n"


def _scaled_metric_series(
    metric_name: str, label: str, values: list[float]
) -> tuple[list[float], str, Callable[[float], str]]:
    """Scale a metric's plotted values to display units.

    Returns ``(scaled_values, display_label, value_formatter)`` and is the single
    place that keeps the overview sparklines, the SVG timeline and the PNG timeline
    on the SAME units as the Metric Summary table:

    * ``*_bytes_per_s`` rates -> MiB/s (the label already says ".../s");
    * ``*_mib`` absolute sizes -> MiB/GiB/... auto-scaled by the series max, with the
      unit token swapped into the label and values rounded via :func:`_fmt_size`
      (so GPU memory reads ``9.9 GiB``, never ``1.01e+04 MiB``);
    * everything else is unchanged (generic :func:`_fmt`).
    """
    if metric_name.endswith("_bytes_per_s"):
        scale = 1.0 / (1024.0 * 1024.0)
        return [v * scale for v in values], label, _fmt
    if metric_name.endswith("_mib"):
        scaled, unit = _scale_size(list(values), "MiB")
        disp_label = label.replace("MiB", unit) if unit != "MiB" else label
        return scaled, disp_label, (lambda v, _u=unit: _fmt_size(v, _u))
    return list(values), label, _fmt


def _xml_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _render_svg(
    rows: list[dict[str, object]],
    svg_path: Path,
    *,
    title: str,
    events: "Iterable[dict[str, object]] | None" = None,
) -> bool:
    """Render a multi-panel line timeline as an SVG file (pure stdlib, no deps).

    When ``events`` (task status changes) fall inside the plotted window, each is
    drawn as a vertical dashed marker spanning every panel, colored by event type,
    with a small rotated task label and a color legend.
    """
    families = _plottable_families(rows)
    if not families:
        return False

    all_ts = [float(r["timestamp"]) for r in rows]
    origin = min(all_ts) if all_ts else 0.0
    max_elapsed = max((max(all_ts) - origin) if all_ts else 0.0, 1.0)

    marker_events = _select_marker_events(
        events, origin=origin, max_elapsed=max_elapsed
    )
    task_styles = _assign_task_styles(marker_events)
    present_events = _events_present(marker_events)

    width = 900
    left, right, gap, ph = 96, 118, 26, 84
    top = 50
    n = len(families)
    plot_w = width - left - right
    panels_bottom = top + (n - 1) * (ph + gap) + ph
    caption_y = panels_bottom + 22

    # A bottom legend band (event colors + task line styles) is laid out first so
    # the canvas can be sized to fit it.
    legend_parts: list[str] = []
    legend_h = 0.0
    if marker_events:
        legend_parts, legend_h = _build_legend_svg(
            present_events, task_styles, x0=8, max_x=width - 8
        )
    legend_y0 = caption_y + 6
    height = int(legend_y0 + legend_h + 8) if marker_events else int(caption_y + 8)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="monospace" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="24" text-anchor="middle" '
        f'font-size="14">{_xml_escape(title)}</text>',
    ]

    for i, (label, resource_type, metric_name) in enumerate(families):
        panel_top = top + i * (ph + gap)
        timeseries = _metric_timeseries(rows, resource_type, metric_name)
        # Auto-scale to display units (e.g. GPU mem MiB -> GiB), matching the
        # summary table / sparklines, and adapt the panel label + min/max readouts.
        scaled_vals, disp_label, stat = _scaled_metric_series(
            metric_name, label, [v for _ts, v in timeseries]
        )
        series = [(ts - origin, sv) for (ts, _v), sv in zip(timeseries, scaled_vals)]
        ys = [v for _t, v in series]
        ymin, ymax = min(ys), max(ys)
        if ymax <= ymin:
            ymax = ymin + 1.0

        def _x(t: float) -> float:
            return left + (t / max_elapsed) * plot_w

        def _y(v: float, _lo: float = ymin, _hi: float = ymax) -> float:
            return panel_top + ph - ((v - _lo) / (_hi - _lo)) * ph

        parts.append(
            f'<rect x="{left}" y="{panel_top}" width="{plot_w}" height="{ph}" '
            f'fill="#fafafa" stroke="#cccccc"/>'
        )
        for grid in (1, 2):
            gy = panel_top + ph * grid / 3
            parts.append(
                f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" '
                f'y2="{gy:.1f}" stroke="#eeeeee"/>'
            )
        points = " ".join(f"{_x(t):.1f},{_y(v):.1f}" for t, v in series)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#1f77b4" '
            f'stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{panel_top + ph / 2:.0f}" text-anchor="end" '
            f'dominant-baseline="middle">{_xml_escape(disp_label)}</text>'
        )
        parts.append(
            f'<text x="{left + plot_w + 6}" y="{panel_top + 12}" font-size="10" '
            f'fill="#666666">max {stat(ymax)}</text>'
        )
        parts.append(
            f'<text x="{left + plot_w + 6}" y="{panel_top + ph}" font-size="10" '
            f'fill="#666666">min {stat(ymin)}</text>'
        )

    # Task lifecycle markers: one vertical line per event spanning all panels.
    # Color encodes the event type; the dash pattern encodes the task (legend
    # below maps both), so a task's markers are recognizable at a glance.
    if marker_events:
        panels_top = top
        for ev in marker_events:
            color = _EVENT_STYLE[str(ev["event"])]
            dash, _ls = task_styles[str(ev["task"])]
            ex = left + ((float(ev["ts"]) - origin) / max_elapsed) * plot_w
            da = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<line x1="{ex:.1f}" y1="{panels_top}" x2="{ex:.1f}" '
                f'y2="{panels_bottom}" stroke="{color}" stroke-width="1.2" '
                f'opacity="0.9"{da}/>'
            )

    parts.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{caption_y:.0f}" text-anchor="middle" '
        f'font-size="10" fill="#666666">Elapsed time (s): 0 .. '
        f'{max_elapsed:.0f}</text>'
    )
    if legend_parts:
        parts.append(f'<g transform="translate(0,{legend_y0:.0f})">')
        parts.extend(legend_parts)
        parts.append("</g>")
    parts.append("</svg>")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return True


class _MatplotlibUnavailable(RuntimeError):
    """Raised when matplotlib cannot be imported for PNG rendering."""


# Install hint shown once when PNG is requested but matplotlib is not installed.
# matplotlib is an OPTIONAL extra (it is heavy); CSV + the stdlib SVG timeline are
# still produced, so PNG simply degrades away.
_PNG_INSTALL_HINT = (
    "PNG monitor timelines require matplotlib (an optional extra). "
    "CSV and SVG reports were still written. Install it with "
    "`pip install sflow[monitor]` (or `uv pip install sflow[monitor]`) to enable "
    "PNG, or drop 'png' from monitor.report.format to silence this message."
)


def _matplotlib_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("matplotlib") is not None


def _render_png(
    rows: list[dict[str, object]],
    png_path: Path,
    *,
    title: str,
    events: "Iterable[dict[str, object]] | None" = None,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # matplotlib is an optional extra; treat as soft-skip
        raise _MatplotlibUnavailable(str(exc)) from exc

    families = _plottable_families(rows)
    if not families:
        return False

    timestamps = [float(r["timestamp"]) for r in rows]
    origin = min(timestamps) if timestamps else 0.0
    max_elapsed = max((max(timestamps) - origin) if timestamps else 0.0, 1.0)
    marker_events = _select_marker_events(
        events, origin=origin, max_elapsed=max_elapsed
    )
    figure, axes = plt.subplots(
        len(families), 1, figsize=(16, max(8, 2.2 * len(families))), sharex=True
    )
    if len(families) == 1:
        axes = [axes]
    for (label, resource_type, metric_name), axis in zip(families, axes):
        # Match the overview/SVG scaling + auto units so the axis label (e.g.
        # "MiB/s", or GPU mem in "GiB") reflects the plotted values.
        timeseries = _metric_timeseries(rows, resource_type, metric_name)
        scaled_vals, disp_label, _stat = _scaled_metric_series(
            metric_name, label, [v for _ts, v in timeseries]
        )
        xs = [ts - origin for ts, _ in timeseries]
        axis.plot(xs, scaled_vals, linewidth=1.4)
        axis.set_ylabel(disp_label)
        axis.grid(alpha=0.25)

    # Task lifecycle markers: a vertical line on every panel, colored by event
    # type and dashed by task. Two legends explain both encodings.
    if marker_events:
        task_styles = _assign_task_styles(marker_events)
        for axis in axes:
            for ev in marker_events:
                axis.axvline(
                    float(ev["ts"]) - origin,
                    color=_EVENT_STYLE[str(ev["event"])],
                    linestyle=task_styles[str(ev["task"])][1],
                    linewidth=1.2,
                    alpha=0.8,
                )
        top_axis = axes[0]
        event_handles = [
            Line2D([0], [0], color=_EVENT_STYLE[e], linewidth=2, label=e)
            for e in _events_present(marker_events)
        ]
        task_handles = [
            Line2D([0], [0], color="#555555", linestyle=ls, linewidth=1.5, label=task)
            for task, (_dash, ls) in task_styles.items()
        ]
        event_legend = top_axis.legend(
            handles=event_handles,
            title="events (color)",
            loc="upper right",
            fontsize=8,
            title_fontsize=8,
            ncol=len(event_handles),
        )
        top_axis.add_artist(event_legend)
        top_axis.legend(
            handles=task_handles,
            title="tasks (line style)",
            loc="upper left",
            fontsize=8,
            title_fontsize=8,
        )

    axes[-1].set_xlabel("Elapsed time (s)")
    figure.suptitle(title, fontsize=14)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return True


# Lifecycle events that open / close a task's reporting window (see
# `_resolve_view_window`). A start with no terminal event leaves the window open.
_WINDOW_START_EVENTS = {"submit", "ready"}
_WINDOW_END_EVENTS = {"done", "fail", "cancel"}


def _resolve_view_window(
    window_tasks: "Iterable[object]",
    task_events: "Iterable[dict[str, object]]",
) -> tuple[float | None, float | None]:
    """Wall-clock ``[start, end]`` for a task view from lifecycle events.

    ``window_tasks`` are runtime task names whose events bound the window; empty
    means no clipping (full run). A task that started but has no terminal event
    (e.g. a long-running service torn down at the end) leaves ``end`` open.
    """
    names = {str(t) for t in window_tasks}
    if not names:
        return None, None
    starts: list[float] = []
    ends: list[float] = []
    for event in task_events or []:
        if str(event.get("task")) not in names:
            continue
        try:
            ts = float(event.get("ts"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        kind = str(event.get("event"))
        if kind in _WINDOW_START_EVENTS:
            starts.append(ts)
        elif kind in _WINDOW_END_EVENTS:
            ends.append(ts)
    return (min(starts) if starts else None), (max(ends) if ends else None)


def _write_consumer_report(
    rows: list[dict[str, object]],
    consumer: dict[str, object],
    out_dir: Path,
    *,
    matplotlib_ok: bool,
    events: "Iterable[dict[str, object]] | None" = None,
    title: str | None = None,
) -> None:
    name = str(consumer.get("name", "consumer"))
    chart_title = str(title or f"{name} hardware timeline")
    formats = [str(f).lower() for f in (consumer.get("formats") or ["csv", "svg"])]
    filtered = filter_rows(
        rows,
        nodes=consumer.get("nodes"),  # type: ignore[arg-type]
        gpus=consumer.get("gpus"),  # type: ignore[arg-type]
        scopes=consumer.get("scopes"),  # type: ignore[arg-type]
        start_ts=consumer.get("start_ts"),  # type: ignore[arg-type]
        end_ts=consumer.get("end_ts"),  # type: ignore[arg-type]
    )
    events = list(events or [])
    consumer_dir = out_dir / name
    # CSV is always written (it is the machine-readable source of truth).
    _write_csv(consumer_dir / "timeline.csv", filtered, _CSV_FIELDS)
    _write_csv(consumer_dir / "summary.csv", summary_rows(filtered), _SUMMARY_FIELDS)

    # SVG: lightweight, pure-stdlib vector timeline (the default visual format).
    if "svg" in formats and filtered:
        try:
            if not _render_svg(
                filtered,
                consumer_dir / "timeline.svg",
                title=chart_title,
                events=events,
            ):
                print(
                    f"WARN: no plottable metrics for '{name}'; SVG skipped",
                    file=sys.stderr,
                )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"WARN: SVG rendering failed for '{name}': {exc}", file=sys.stderr)

    # PNG: optional raster output via matplotlib. `process` only enables this when
    # matplotlib is importable, so a failure here is a genuine plotting error.
    if "png" in formats and matplotlib_ok and filtered:
        try:
            if not _render_png(
                filtered,
                consumer_dir / "timeline.png",
                title=chart_title,
                events=events,
            ):
                print(
                    f"WARN: no plottable metrics for '{name}'; PNG skipped",
                    file=sys.stderr,
                )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"WARN: PNG rendering failed for '{name}': {exc}", file=sys.stderr)


def process(spec: dict[str, object]) -> dict[str, object]:
    """Run the full post-process pass from a report spec dict."""
    raw_dir = Path(str(spec["raw_dir"]))
    out_dir = Path(str(spec["out_dir"]))
    overview_path = Path(str(spec["overview_path"]))
    workflow_name = str(spec.get("workflow_name", ""))
    interval_ms = spec.get("interval_ms")
    consumers = list(spec.get("consumers") or [])
    task_reports = list(spec.get("task_reports") or [])
    task_events = list(spec.get("task_events") or [])

    rows = parse_monitor_logs(raw_dir)

    # PNG is the only format that needs a third-party lib (matplotlib, optional).
    # CSV + SVG are pure stdlib. If anything requested PNG but matplotlib is
    # missing, emit a single hint and skip PNG (CSV/SVG still produced).
    def _wants_png(entry: dict[str, object]) -> bool:
        return "png" in [str(f).lower() for f in (entry.get("formats") or [])]

    png_requested = any(
        _wants_png(c) for c in consumers if c.get("report")
    ) or any(_wants_png(r) for r in task_reports)
    matplotlib_ok = True
    if png_requested and not _matplotlib_available():
        print(f"WARN: {_PNG_INSTALL_HINT}", file=sys.stderr)
        matplotlib_ok = False

    # Detailed per-consumer reports (only those that opted in -- in practice just
    # the whole-pool workflow aggregate; per-task monitors render as task views).
    for consumer in consumers:
        if consumer.get("report"):
            _write_consumer_report(
                rows, consumer, out_dir, matplotlib_ok=matplotlib_ok, events=task_events
            )

    # Per-task / per-replica resource reports. Each is filtered to the task's
    # nodes/GPUs and clipped to its (or, for a cross view, the owner's) run window.
    for report in task_reports:
        start_ts, end_ts = _resolve_view_window(
            report.get("window_tasks") or [], task_events
        )
        entry = dict(report)
        entry["start_ts"] = start_ts
        entry["end_ts"] = end_ts
        _write_consumer_report(
            rows, entry, out_dir, matplotlib_ok=matplotlib_ok,
            events=task_events, title=str(report.get("title") or ""),
        )

    # Always write the terminal overview when any samples exist.
    overview = render_overview(
        rows,
        workflow_name=workflow_name,
        out_dir=out_dir,
        overview_path=overview_path,
        interval_ms=int(interval_ms) if isinstance(interval_ms, (int, float)) else None,
        consumers=consumers,
        events=task_events,
        task_reports=task_reports,
    )
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    overview_path.write_text(overview, encoding="utf-8")

    return {
        "sample_count": len(rows),
        "consumer_count": len(consumers),
        "task_report_count": len(task_reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="", help="Path to a JSON report spec.")
    parser.add_argument("--raw-dir", default="", help="Raw monitor sample directory.")
    parser.add_argument("--out-dir", default="", help="Output (reports) directory.")
    parser.add_argument("--overview-path", default="", help="Path for sflow_monitor.log.")
    parser.add_argument("--workflow-name", default="")
    parser.add_argument("--interval-ms", type=int, default=None)
    parser.add_argument(
        "--formats",
        default="csv,svg",
        help="Comma-separated report formats: csv, svg, png (default: csv,svg).",
    )
    args = parser.parse_args()

    if args.spec:
        spec = json.loads(Path(args.spec).read_text())
    else:
        if not (args.raw_dir and args.out_dir and args.overview_path):
            parser.error("either --spec or all of --raw-dir/--out-dir/--overview-path are required")
        formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
        spec = {
            "workflow_name": args.workflow_name,
            "raw_dir": args.raw_dir,
            "out_dir": args.out_dir,
            "overview_path": args.overview_path,
            "interval_ms": args.interval_ms,
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

    result = process(spec)
    print(
        f"Monitor post-process complete: {result['sample_count']} samples, "
        f"{result['consumer_count']} consumer(s). Overview: {spec['overview_path']}"
    )


if __name__ == "__main__":
    main()
