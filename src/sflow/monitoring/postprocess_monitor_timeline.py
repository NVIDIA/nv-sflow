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

# Task lifecycle changes drawn on the timeline. Order controls label ordering
# within a merged transition; unknown event types are ignored. There is no colour
# per event any more -- see `_ALERT_EVENTS` below for why.
_EVENT_ORDER = ("submit", "ready", "done", "fail", "cancel")
_KNOWN_EVENTS = frozenset(_EVENT_ORDER)
# Which band a label goes in, split by what the event MEANS rather than by name:
# `submit` is the only one that starts something, so it sits above the plot;
# everything else -- `ready`, `done`, and the `fail`/`cancel` that also end a task
# -- sits below it. Splitting the two halves how many labels compete for any one
# strip of x, which is what made neighbouring ones collide.
_TOP_EVENTS = frozenset({"submit"})


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
        if str(ev.get("event")) not in _KNOWN_EVENTS:
            continue
        try:
            ts = float(ev.get("ts"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if origin <= ts <= origin + max_elapsed:
            selected.append({"ts": ts, "task": str(ev.get("task", "")), "event": str(ev["event"])})
    selected.sort(key=lambda e: float(e["ts"]))
    return selected


# Transitions are ANNOTATIONS, not a series, so they do not get a categorical
# colour: routine ones recede to neutral ink and the exceptional ones keep a
# reserved status colour. Previously colour encoded the event and a dash pattern
# encoded the task, which cost two legend lookups per line -- and every event
# colour was also a device series colour (`#1f77b4` meant both "GPU 0" and
# "submit" on the same canvas).
_MARKER_INK = "#9aa0a6"
_MARKER_LABEL_INK = "#5f6772"
# `fail` -- and ONLY fail -- gets a reserved status colour: a cancel is usually
# sflow tearing a service down on purpose, so flagging it red would cry wolf on
# every healthy run. Deliberately darker than the `#d62728` in `_SERIES_COLORS`,
# so a rule can never be mistaken for a GPU line on the same canvas; it always
# ships alongside its own text label, never as colour alone.
_MARKER_FAIL_INK = "#a61b1b"
_FAIL_EVENTS = frozenset({"fail"})
# Events closer together than this FRACTION of the plotted span are one
# transition as far as a reader is concerned; drawing each separately makes a
# barcode. A fraction, not pixels, so the SVG and the PNG merge identically
# without either having to know the other's geometry.
_MARKER_MERGE_FRACTION = 0.02
# Roughly the width one transition owns before it collides with its neighbour.
_MARKER_LABEL_CHARS = 30


def _merge_marker_events(
    marker_events: list[dict[str, object]],
    *,
    origin: float,
    max_elapsed: float,
) -> list[dict[str, object]]:
    """Collapse near-simultaneous events into one labelled transition.

    A workflow has a handful of phase edges but emits an event per task per
    change, so 7 tasks produce ~15 rules that read as noise. Grouping by x
    position recovers the edges the reader actually cares about.
    """
    tolerance = max_elapsed * _MARKER_MERGE_FRACTION
    groups: list[dict[str, object]] = []
    for ev in marker_events:
        rel = float(ev["ts"]) - origin
        if groups and rel - float(groups[-1]["rel"]) <= tolerance:
            groups[-1]["pairs"].append((str(ev["event"]), str(ev["task"])))  # type: ignore[union-attr]
        else:
            groups.append({"rel": rel, "pairs": [(str(ev["event"]), str(ev["task"]))]})
    for group in groups:
        pairs = group["pairs"]  # type: ignore[assignment]
        group["failed"] = any(e in _FAIL_EVENTS for e, _t in pairs)  # type: ignore[union-attr]
        # Two labels per transition: what STARTED (top) and what finished or came
        # up (bottom). Either may be empty, in which case that band skips it.
        group["label_top"] = _marker_group_label(
            [p for p in pairs if p[0] in _TOP_EVENTS]  # type: ignore[union-attr]
        )
        group["label_bottom"] = _marker_group_label(
            [p for p in pairs if p[0] not in _TOP_EVENTS]  # type: ignore[union-attr]
        )
    return groups


def _marker_group_label(pairs: list[tuple[str, str]]) -> str:
    """One short line naming what changed, e.g. `bench ready` / `3 tasks done`.

    Names the task when exactly one changed (the useful case) and counts them
    otherwise, so the label stays narrow enough to sit over its own rule.
    """
    by_event: dict[str, set[str]] = {}
    for event, task in pairs:
        by_event.setdefault(event, set()).add(task)
    bits = []
    for event in _EVENT_ORDER:
        tasks = by_event.get(event)
        if not tasks:
            continue
        who = next(iter(tasks)) if len(tasks) == 1 else f"{len(tasks)} tasks"
        bits.append(f"{who} {event}")
    # Drop WHOLE items rather than cutting mid-word: "3 tasks submit +2 more"
    # still says what happened, "3 tasks submit / decode_serve\u2026" does not.
    label = ""
    for i, bit in enumerate(bits):
        candidate = f"{label} / {bit}" if label else bit
        if len(candidate) > _MARKER_LABEL_CHARS:
            return f"{label} +{len(bits) - i} more" if label else bit
        label = candidate
    return label


def _build_device_legend_svg(
    labels: list[str], *, x0: float, max_x: float
) -> tuple[list[str], float]:
    """Bottom band mapping each series colour to its device.

    Drawn once per image: the colour for a device is the same in every panel, so
    repeating it in each panel's gutter is noise. Returns fragments in local
    coords (y from 0) plus the band height so the caller can size the canvas.
    """
    parts: list[str] = []
    x, y, row_h = x0, 10.0, 14.0
    for idx, label in enumerate(labels):
        entry_w = 26.0 + 6.5 * len(label)
        if x + entry_w > max_x and x > x0:
            x, y = x0, y + row_h
        colour = _SERIES_COLORS[idx % len(_SERIES_COLORS)]
        parts.append(
            f'<line x1="{x:.1f}" y1="{y - 3:.1f}" x2="{x + 16:.1f}" '
            f'y2="{y - 3:.1f}" stroke="{colour}" stroke-width="2"'
            f'{_series_dash(idx)}/>'
        )
        parts.append(
            f'<text x="{x + 20:.1f}" y="{y:.1f}" font-size="9" fill="#444444">'
            f'{_xml_escape(label)}</text>'
        )
        x += entry_w
    return parts, (y + 6.0 if parts else 0.0)


# Height of one label band (two staggered rows plus breathing room).
_MARKER_BAND_H = 28.0


def _build_marker_labels_svg(
    groups: list[dict[str, object]],
    *,
    key: str,
    above: bool,
    left: float,
    plot_w: float,
    max_elapsed: float,
    anchor_y: float,
    width: float,
) -> list[str]:
    """Name each transition directly beside its own rule, in one band.

    Direct labels replace the old pair of legends (event colour + task dash):
    identifying a rule used to mean two lookups, and neither encoding survived
    being printed in greyscale. Called once per band -- ``submit`` above the plot,
    ``ready``/``done`` below it -- so the two never compete for the same strip of
    canvas. Within a band labels still alternate between two rows, and a short
    leader ties each back to its rule.
    """
    parts: list[str] = []
    drawn = 0
    for group in groups:
        text = str(group.get(key) or "")
        if not text:
            continue
        x = left + (float(group["rel"]) / max_elapsed) * plot_w
        # Stagger by DRAWN count, not group index: a band that skips groups would
        # otherwise leave a row empty and put two neighbours back on one line.
        row = drawn % 2
        drawn += 1
        if above:
            y = anchor_y - _MARKER_BAND_H + 12.0 + row * 11.0
            leader_from, leader_to = y + 3.0, anchor_y
        else:
            y = anchor_y + 13.0 + row * 11.0
            leader_from, leader_to = anchor_y, y - 8.0
        # A failure only ever lands in the lower band, so the upper one stays
        # neutral even for a group that also carries one.
        ink = (
            _MARKER_FAIL_INK
            if group.get("failed") and not above
            else _MARKER_LABEL_INK
        )
        # Keep the text on the canvas even when a transition sits at either edge.
        tx = min(max(x, 46.0), width - 46.0)
        parts.append(
            f'<text x="{tx:.1f}" y="{y:.1f}" text-anchor="middle" font-size="9" '
            f'fill="{ink}">{_xml_escape(text)}</text>'
        )
        parts.append(
            f'<line x1="{x:.1f}" y1="{leader_from:.1f}" x2="{x:.1f}" '
            f'y2="{leader_to:.1f}" stroke="{ink}" stroke-width="0.8" '
            f'opacity="0.5"/>'
        )
    return parts


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
    # `None` means "no GPU subset requested -> all"; an EMPTY list means "this task
    # reserved no GPUs", which must report none rather than every GPU on its nodes.
    gpu_set = {str(g) for g in gpus} if gpus is not None else None
    # Best effort (mirrors the node fallback above): if a NON-EMPTY GPU subset is
    # requested but matches no sampled GPU id -- e.g. the consumer's logical CUDA
    # indices differ from nvidia-smi's physical `index`, or no gpu scope was
    # collected -- keep all GPUs rather than emit an empty report.
    if gpu_set and not any(
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


def _sample_period(samples: list[tuple[float, str]]) -> float:
    """Collector sampling period, as the median gap between one host's samples.

    Needed because every node samples on its own clock, so timestamps from
    different nodes essentially never coincide. Grouping them by exact timestamp
    puts ONE host in each point, and a "cluster average" then sawtooths between a
    busy node and its idle peers instead of averaging them. Returns 0.0 when
    there is nothing to infer from (single sample per host).
    """
    per_host: dict[str, set[float]] = defaultdict(set)
    for ts, host in samples:
        per_host[host].add(ts)
    gaps: list[float] = []
    for timestamps in per_host.values():
        ordered = sorted(timestamps)
        gaps.extend(b - a for a, b in zip(ordered, ordered[1:]) if b > a)
    if not gaps:
        return 0.0
    gaps.sort()
    return gaps[len(gaps) // 2]


def _metric_timeseries(
    rows: list[dict[str, object]], resource_type: str, metric_name: str
) -> list[tuple[float, float]]:
    """Cluster-averaged ``(timestamp, value)`` series for a metric.

    Samples are bucketed to the collector period first so that one point averages
    every node that sampled in that window -- see :func:`_sample_period`.
    """
    picked: list[tuple[float, str, float]] = []
    for row in rows:
        if row["resource_type"] != resource_type or row["metric_name"] != metric_name:
            continue
        value = row.get("metric_value")
        if isinstance(value, (int, float)) and not math.isnan(value):
            picked.append(
                (float(row["timestamp"]), str(row.get("hostname", "")), float(value))
            )
    period = _sample_period([(ts, host) for ts, host, _v in picked])
    by_ts: dict[float, list[float]] = defaultdict(list)
    for ts, _host, value in picked:
        by_ts[round(ts / period) * period if period else ts].append(value)
    return [(ts, sum(v) / len(v)) for ts, v in sorted(by_ts.items())]


def _series_for_metric(
    rows: list[dict[str, object]], resource_type: str, metric_name: str
) -> list[float]:
    """Cluster-averaged values (one per timestamp) for a metric."""
    return [v for _ts, v in _metric_timeseries(rows, resource_type, metric_name)]


# Line colors for per-resource series (one per GPU). Cycled; chosen to stay
# distinguishable in both the SVG and a greyscale print.
_SERIES_COLORS = (
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22",
)

# Dash pattern per device, cycled alongside _SERIES_COLORS and keyed on the same
# index. Colour alone cannot separate two lines that COINCIDE, and coincidence is
# the normal case for a tensor-parallel task: every rank allocates the same
# weights + KV-cache footprint, so `gpu_memory_used_mib` for its GPUs is often
# identical to the byte. The later line then hides exactly under the earlier one
# and the panel is indistinguishable from one where the second device was never
# drawn at all -- which reads as a collection bug rather than as real data.
_SERIES_DASHES = ("", "5,3", "1.5,2.5", "7,2,1.5,2")


def _series_dash(idx: int) -> str:
    """SVG dash attribute for series *idx*, or "" for the solid first series."""
    pattern = _SERIES_DASHES[idx % len(_SERIES_DASHES)]
    return f' stroke-dasharray="{pattern}"' if pattern else ""


def gpu_label(gpus: object) -> str:
    """Render a report's GPU subset for humans.

    `None` means "no subset requested -> all"; an EMPTY list means "this task
    reserved no GPUs", which :func:`filter_rows` honours by dropping every GPU
    row. Truthiness collapses the two and advertises `all` for a report that
    contains no GPU data at all.
    """
    if gpus is None:
        return "all"
    return ",".join(str(g) for g in gpus) or "none"  # type: ignore[union-attr]


def _metric_series_by_resource(
    rows: list[dict[str, object]], resource_type: str, metric_name: str
) -> list[tuple[str, list[tuple[float, float]]]]:
    """One ``(label, [(ts, value)])`` series per physical resource, time-ordered.

    Used for GPU panels so a report shows every device it tracks instead of one
    averaged line -- an average hides both how many GPUs are in scope and a
    single hot or idle device among them. Timestamps are bucketed exactly as in
    :func:`_metric_timeseries` so series from different nodes line up.
    """
    picked: list[tuple[float, str, str, float]] = []
    for row in rows:
        if row["resource_type"] != resource_type or row["metric_name"] != metric_name:
            continue
        value = row.get("metric_value")
        if isinstance(value, (int, float)) and not math.isnan(value):
            picked.append(
                (
                    float(row["timestamp"]),
                    str(row.get("hostname", "")),
                    str(row.get("resource_id", "")),
                    float(value),
                )
            )
    if not picked:
        return []
    period = _sample_period([(ts, host) for ts, host, _rid, _v in picked])
    by_key: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ts, host, rid, value in picked:
        bucket = round(ts / period) * period if period else ts
        by_key[(host, rid)][bucket].append(value)

    out: list[tuple[str, list[tuple[float, float]]]] = []
    for (_host, rid), buckets in sorted(
        by_key.items(), key=lambda kv: _resource_sort_key(kv[0])
    ):
        # Callers always pass single-host rows (`_write_consumer_report` splits a
        # multi-node report into one image per host), so the id alone is unambiguous.
        label = f"{resource_type.upper()} {rid}"
        out.append(
            (label, [(ts, sum(v) / len(v)) for ts, v in sorted(buckets.items())])
        )
    return out


def _resource_sort_key(key: tuple[str, str]) -> tuple:
    """Sort series by host, then resource id numerically (GPU 2 before GPU 10)."""
    host, rid = key
    return (host, int(rid)) if rid.isdigit() else (host, float("inf"), rid)


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
    """Plain decimal, never scientific: ``2245``, not ``2.24e+03``.

    `%g` switches to an exponent above 1e3 and below 1e-2, which is unreadable on
    a chart axis and in the summary table -- GPU power and clocks live in exactly
    that range. Precision follows magnitude instead, mirroring :func:`_fmt_size`.
    """
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:.0f}"
    if magnitude >= 1 or value == 0:
        return f"{value:.2f}"
    if magnitude >= 0.01:
        return f"{value:.3f}"
    # Sub-0.01 still gets real digits rather than an exponent; an underflow to all
    # zeros collapses to "0" instead of a bare "0." .
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


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
            gpu_str = gpu_label(gpus)
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
            gpu_str = gpu_label(gpus)
            scopes = report.get("scopes") or []
            tag = " [cross]" if report.get("cross") else ""
            window_status = report.get("window_status")
            window_note = f" window={window_status}" if window_status else ""
            node_str = ", ".join(str(n) for n in nodes) or "all"
            scope_str = ",".join(str(s) for s in scopes) or "all"
            # Prefix with the group so the line doubles as the path to open.
            group = (
                WINDOWED_DIRNAME
                if report.get("log_window") is not None
                else LIFECYCLE_DIRNAME
            )
            lines.append(
                f"- {group}/{label}{tag}: nodes={node_str} gpus={gpu_str} "
                f"scopes={scope_str}{window_note}"
            )

    # Task lifecycle events (the markers overlaid on the SVG/PNG timelines).
    marker_events: list[tuple[float, str, str]] = []
    for e in events or []:
        event = str(e.get("event"))
        if event not in _KNOWN_EVENTS:
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
    width = 900
    left, right, gap, ph = 96, 118, 26, 84
    n = len(families)
    plot_w = width - left - right
    marker_groups = _merge_marker_events(
        marker_events, origin=origin, max_elapsed=max_elapsed
    )
    # Headroom above the panels for the transition labels (two staggered rows).
    has_top = any(g.get("label_top") for g in marker_groups)
    has_bottom = any(g.get("label_bottom") for g in marker_groups)
    marker_band_top = _MARKER_BAND_H if has_top else 0.0
    marker_band_bottom = _MARKER_BAND_H if has_bottom else 0.0
    top = 50 + marker_band_top
    panels_bottom = top + (n - 1) * (ph + gap) + ph
    caption_y = panels_bottom + 22 + marker_band_bottom

    # Device colours are shared by every panel, so their legend is built once.
    # Union across ALL gpu families, not just the first: a device that reports
    # N/A for one field (MIG mode, a per-field collector error) drops out of that
    # family only, and keying the colour on each panel's own index would then
    # shift every line after it against this legend.
    device_labels: list[str] = []
    for _lbl, _rtype, _mname in families:
        if _rtype == "gpu":
            for lab, _ser in _metric_series_by_resource(rows, _rtype, _mname):
                if lab not in device_labels:
                    device_labels.append(lab)
    device_parts, device_h = (
        _build_device_legend_svg(device_labels, x0=8, max_x=width - 8)
        if device_labels
        else ([], 0.0)
    )

    # Only ONE legend remains -- the device colours. Transitions are named in
    # place above their rules, so they need no legend at all.
    device_y0 = caption_y + 6
    height = int(device_y0 + device_h + 8)

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
        # GPU panels draw ONE LINE PER DEVICE: an average hides both how many
        # GPUs the report covers and a single hot (or idle) device among them.
        # Node-level scopes have one value per node, so they stay averaged.
        per_resource = (
            _metric_series_by_resource(rows, resource_type, metric_name)
            if resource_type == "gpu"
            else []
        )
        if per_resource:
            # One scale across every device so the lines are comparable.
            flat = [v for _lbl, ser in per_resource for _ts, v in ser]
            scaled_flat, disp_label, stat = _scaled_metric_series(
                metric_name, label, flat
            )
            multi: list[tuple[str, list[tuple[float, float]]]] = []
            cursor = 0
            for lbl, ser in per_resource:
                chunk = scaled_flat[cursor : cursor + len(ser)]
                multi.append(
                    (lbl, [(ts - origin, sv) for (ts, _v), sv in zip(ser, chunk)])
                )
                cursor += len(ser)
            ys = scaled_flat
        else:
            timeseries = _metric_timeseries(rows, resource_type, metric_name)
            # Auto-scale to display units (e.g. GPU mem MiB -> GiB), matching the
            # summary table / sparklines, and adapt the panel label + readouts.
            scaled_vals, disp_label, stat = _scaled_metric_series(
                metric_name, label, [v for _ts, v in timeseries]
            )
            multi = [
                ("", [(ts - origin, sv) for (ts, _v), sv in zip(timeseries, scaled_vals)])
            ]
            ys = scaled_vals
        if not ys:
            continue
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
        for series_label, series in multi:
            # Look the style up BY LABEL so it always matches the shared legend.
            # Index 0 for a panel with no per-device series (cpu/mem/disk/net):
            # first colour, no dash, i.e. unchanged from a single solid line.
            idx = (
                device_labels.index(series_label)
                if series_label in device_labels
                else 0
            )
            colour = _SERIES_COLORS[idx % len(_SERIES_COLORS)]
            points = " ".join(f"{_x(t):.1f},{_y(v):.1f}" for t, v in series)
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{colour}" '
                f'stroke-width="1.5"{_series_dash(idx)}/>'
            )
        parts.append(
            f'<text x="{left - 8}" y="{panel_top + ph / 2:.0f}" text-anchor="end" '
            f'dominant-baseline="middle">{_xml_escape(disp_label)}</text>'
        )
        # ONE shared vertical scale per panel, spanning every device drawn on it
        # (i.e. this node's GPUs). Per-line min/max readouts would suggest each
        # line has its own axis, which is exactly the confusing part. Device
        # colours are identical across panels, so that legend is drawn once at the
        # bottom instead of repeated in every gutter.
        parts.append(
            f'<text x="{left + plot_w + 6}" y="{panel_top + 12}" font-size="10" '
            f'fill="#666666">max {stat(ymax)}</text>'
        )
        parts.append(
            f'<text x="{left + plot_w + 6}" y="{panel_top + ph}" font-size="10" '
            f'fill="#666666">min {stat(ymin)}</text>'
        )

    # Task lifecycle transitions: one rule per MERGED group, spanning all panels.
    # Routine changes recede to neutral ink so they never compete with the data
    # lines; only fail/cancel keep a reserved status colour, and they draw solid
    # and slightly heavier so the one marker worth reacting to stands out.
    for group in marker_groups:
        ex = left + (float(group["rel"]) / max_elapsed) * plot_w
        failed = bool(group.get("failed"))
        # Started-here vs ended-here, separated on THREE channels at once --
        # pattern, weight and opacity. Pattern alone (a 1px dotted line beside a
        # 1px solid one at the same opacity) is too subtle to read at a glance,
        # which is the whole point of the distinction. A merged group holding both
        # counts as an end, the stronger event: `label_bottom` is non-empty
        # exactly when an end event is present, so it doubles as the test.
        if group.get("label_bottom"):
            style = f'stroke-width="{1.6 if failed else 1.4}"'
            opacity = 0.95
        else:
            style = 'stroke-width="1" stroke-dasharray="2 4"'
            opacity = 0.6
        parts.append(
            f'<line x1="{ex:.1f}" y1="{top}" x2="{ex:.1f}" y2="{panels_bottom}" '
            f'stroke="{_MARKER_FAIL_INK if failed else _MARKER_INK}" '
            f'{style} opacity="{opacity}"/>'
        )
    parts.extend(
        _build_marker_labels_svg(
            marker_groups, key="label_top", above=True, left=left, plot_w=plot_w,
            max_elapsed=max_elapsed, anchor_y=top, width=width,
        )
    )
    parts.extend(
        _build_marker_labels_svg(
            marker_groups, key="label_bottom", above=False, left=left,
            plot_w=plot_w, max_elapsed=max_elapsed, anchor_y=panels_bottom,
            width=width,
        )
    )

    parts.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{caption_y:.0f}" text-anchor="middle" '
        f'font-size="10" fill="#666666">Elapsed time (s): 0 .. '
        f'{max_elapsed:.0f}</text>'
    )
    if device_parts:
        parts.append(f'<g transform="translate(0,{device_y0:.0f})">')
        parts.extend(device_parts)
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
    device_legend_drawn = False
    for (label, resource_type, metric_name), axis in zip(families, axes):
        # Match the overview/SVG scaling + auto units so the axis label (e.g.
        # "MiB/s", or GPU mem in "GiB") reflects the plotted values.
        # One line per GPU (see _render_svg); node-level scopes stay averaged.
        per_resource = (
            _metric_series_by_resource(rows, resource_type, metric_name)
            if resource_type == "gpu"
            else []
        )
        if per_resource:
            flat = [v for _lbl, ser in per_resource for _ts, v in ser]
            scaled_flat, disp_label, _stat = _scaled_metric_series(
                metric_name, label, flat
            )
            cursor = 0
            for series_label, ser in per_resource:
                chunk = scaled_flat[cursor : cursor + len(ser)]
                axis.plot(
                    [ts - origin for ts, _v in ser],
                    chunk,
                    linewidth=1.4,
                    label=series_label,
                )
                cursor += len(ser)
            # Once per image, not once per panel: the colour for a device is the
            # same on every panel, so repeating it three more times is noise.
            if not device_legend_drawn:
                axis.legend(
                    fontsize=6, ncol=max(1, len(per_resource) // 4), loc="upper right"
                )
                device_legend_drawn = True
        else:
            timeseries = _metric_timeseries(rows, resource_type, metric_name)
            scaled_vals, disp_label, _stat = _scaled_metric_series(
                metric_name, label, [v for _ts, v in timeseries]
            )
            axis.plot([ts - origin for ts, _ in timeseries], scaled_vals, linewidth=1.4)
        axis.set_ylabel(disp_label)
        axis.grid(alpha=0.25)

    # Task lifecycle transitions: mirrors the SVG exactly -- near-simultaneous
    # events merged into one dashed neutral rule, `submit` named above the first
    # panel and `ready`/`done` below the last, instead of decoded through a pair
    # of legends.
    if marker_events:
        marker_groups = _merge_marker_events(
            marker_events, origin=origin, max_elapsed=max_elapsed
        )
        for axis in axes:
            for group in marker_groups:
                failed = bool(group.get("failed"))
                # Mirrors the SVG: faint sparse dots for a start, a solid
                # heavier line for an end.
                ended = bool(group.get("label_bottom"))
                axis.axvline(
                    float(group["rel"]),
                    color=_MARKER_FAIL_INK if failed else _MARKER_INK,
                    linestyle="-" if ended else (0, (2, 4)),
                    linewidth=(1.6 if failed else 1.4) if ended else 1.0,
                    alpha=0.95 if ended else 0.6,
                )

        def _annotate(axis, key: str, *, above: bool) -> None:
            drawn = 0
            for group in marker_groups:
                text = str(group.get(key) or "")
                if not text:
                    continue
                # Stagger by DRAWN count so a skipped group never collapses two
                # neighbours onto the same row.
                step = 4 + (drawn % 2) * 9
                drawn += 1
                axis.annotate(
                    text,
                    xy=(float(group["rel"]), 1.0 if above else 0.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, step if above else -(step + 22)),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if above else "top",
                    fontsize=6,
                    color=(
                        _MARKER_FAIL_INK
                        if group.get("failed") and not above
                        else _MARKER_LABEL_INK
                    ),
                )

        _annotate(axes[0], "label_top", above=True)
        _annotate(axes[-1], "label_bottom", above=False)

    # Extra pad so the axis title clears the `ready`/`done` band drawn beneath it.
    axes[-1].set_xlabel("Elapsed time (s)", labelpad=28)
    figure.suptitle(title, fontsize=14)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return True


# Lifecycle events that open / close a task's reporting window (see
# `_resolve_view_window`). A start with no terminal event leaves the window open.
_WINDOW_START_EVENTS = {"submit", "ready"}
_WINDOW_END_EVENTS = {"done", "fail", "cancel"}

# Same prefix as utils.parser._SFLOW_LOG_PREFIX_RE, restated here because this
# module is standard-library-only (it also runs as a materialized standalone
# script). Milliseconds are required, not optional: every writer emits them
# (logging.Formatter and all three core.log_offload prefixers).
_TIMESTAMPED_LOG_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r" - \S+ - [A-Z]+ - (?P<message>.*)$"
)
_TRANSPORT_PREFIX_RE = re.compile(r"^(?:\d+:\s+|\[pod/[^\]]+\]\s+)")
_WINDOW_LINE_MAX_CHARS = 4096

# Report folders are grouped by what decides their time range, so a directory
# listing answers "is this the whole run or the measured phase?" without opening
# window.json. `lifecycle/` covers the whole-pool aggregate and every task view
# bounded by submit/ready/done events; `windowed/` holds the marker-clipped ones.
MONITOR_RAW_HINT = "sflow_monitor/raw/"

LIFECYCLE_DIRNAME = "lifecycle"
WINDOWED_DIRNAME = "windowed"


def _report_dir(out_dir: Path, name: str, *, windowed: bool) -> Path:
    return out_dir / (WINDOWED_DIRNAME if windowed else LIFECYCLE_DIRNAME) / name


def _final_attempt_window(
    task: str, task_events: "Iterable[dict[str, object]]"
) -> tuple[float | None, float]:
    submits: list[float] = []
    terminals: list[float] = []
    for event in task_events or []:
        if str(event.get("task")) != task:
            continue
        try:
            ts = float(event.get("ts"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        kind = str(event.get("event"))
        if kind == "submit":
            submits.append(ts)
        elif kind in _WINDOW_END_EVENTS:
            terminals.append(ts)
    if not submits:
        return None, float("inf")
    start = max(submits)
    ends = [ts for ts in terminals if ts >= start]
    # A long-running service torn down at the end may never emit a terminal
    # event (same caveat as _resolve_view_window). The marker window's end comes
    # from the log, so an absent terminal just leaves the search bound open.
    return start, (max(ends) if ends else float("inf"))


def _parse_timestamped_log_line(line: str) -> tuple[float, str, str] | None:
    match = _TIMESTAMPED_LOG_RE.match(line)
    if match is None:
        return None
    timestamp_text = match.group("timestamp")
    try:
        timestamp = datetime.strptime(
            timestamp_text, "%Y-%m-%d %H:%M:%S,%f"
        ).timestamp()
    except ValueError:
        return None
    message = match.group("message")
    transport = _TRANSPORT_PREFIX_RE.match(message)
    if transport is not None:
        message = message[transport.end() :]
    return timestamp, timestamp_text, message


def _compile_marker_patterns(patterns: object) -> list[tuple[str, re.Pattern[str]]]:
    values = [patterns] if isinstance(patterns, str) else list(patterns)  # type: ignore[arg-type]
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for value in [str(v) for v in values]:
        source = None
        for prefix in ("regex:", "re:"):
            if value.startswith(prefix):
                source = value[len(prefix) :]
                break
        compiled.append(
            (value, re.compile(source if source is not None else re.escape(value)))
        )
    return compiled


def _select_log_boundary(
    candidates: list[dict[str, object]],
    boundary_spec: dict[str, object],
) -> dict[str, object] | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["timestamp"]),
            int(candidate["byte_offset"]),
        ),
    )
    select = str(boundary_spec["select"])
    selected = dict(ordered[0] if select == "first" else ordered[-1])
    # `matched_patterns` already records which configured patterns hit this line;
    # echoing the whole configured list back adds nothing the YAML doesn't have.
    selected.update({"select": select, "match_count": len(candidates)})
    return selected


def _resolve_log_source(
    task: str,
    log_path: Path,
    log_window: dict[str, object],
    task_events: "Iterable[dict[str, object]]",
) -> dict[str, object]:
    source: dict[str, object] = {
        "runtime_task": task,
        "source_log": str(log_path),
        "status": "unresolved",
    }
    attempt_start, attempt_end = _final_attempt_window(task, task_events)
    if attempt_start is None:
        source["error"] = "final attempt has no submit event"
        return source
    if not log_path.is_file():
        source["error"] = "source log is missing or unreadable"
        return source

    # Shape is guaranteed upstream: schema.py validates the patterns (non-empty,
    # regexes compile) and `select` is a Literal, and monitor_planner always emits
    # both boundaries with `select` filled in.
    boundary_specs = {
        name: dict(log_window[name])  # type: ignore[arg-type]
        for name in ("start", "end")
    }
    matchers = {
        name: _compile_marker_patterns(spec["pattern"])
        for name, spec in boundary_specs.items()
    }

    candidates: dict[str, list[dict[str, object]]] = {"start": [], "end": []}
    try:
        with log_path.open("rb") as handle:
            while True:
                byte_offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                parsed = _parse_timestamped_log_line(
                    raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                )
                if parsed is None:
                    continue
                timestamp, timestamp_text, message = parsed
                # The prefix truncates to milliseconds. Accept the line when that
                # displayed-time bucket can overlap the final attempt.
                if timestamp + 0.001 <= attempt_start or timestamp > attempt_end:
                    continue
                for name in ("start", "end"):
                    matched = [
                        pattern
                        for pattern, matcher in matchers[name]
                        if matcher.search(message)
                    ]
                    if matched:
                        candidates[name].append(
                            {
                                "timestamp": timestamp,
                                "timestamp_iso": timestamp_text,
                                "byte_offset": byte_offset,
                                "line": message[:_WINDOW_LINE_MAX_CHARS],
                                "line_truncated": len(message) > _WINDOW_LINE_MAX_CHARS,
                                "matched_patterns": matched,
                            }
                        )
    except OSError as exc:
        source["error"] = f"source log is unreadable: {exc}"
        return source

    start = _select_log_boundary(candidates["start"], boundary_specs["start"])
    if start is None:
        source["error"] = "start marker not found in final attempt"
        return source
    source["start"] = start
    # `end` closes *this* window, so only ends strictly after the selected start
    # qualify -- otherwise `start:last` + `end:first` (the natural "last warmup,
    # then the run that follows" spelling) picks an earlier cycle's end and the
    # range comes out reversed.
    end = _select_log_boundary(
        [
            candidate
            for candidate in candidates["end"]
            if float(candidate["timestamp"]) > float(start["timestamp"])
        ],
        boundary_specs["end"],
    )
    if end is None:
        source["error"] = "end marker not found after the selected start"
        return source
    source["end"] = end
    source["status"] = "matched"
    return source


def _resolve_report_log_window(
    report: dict[str, object],
    out_dir: Path,
    task_events: "Iterable[dict[str, object]]",
    cache: dict[tuple[str, str], dict[str, object]],
) -> tuple[float | None, float | None, dict[str, object]]:
    log_window = dict(report["log_window"])  # type: ignore[arg-type]
    spec_key = json.dumps(log_window, sort_keys=True)
    sources: list[dict[str, object]] = []
    for task in [str(name) for name in (report.get("window_tasks") or [])]:
        key = (task, spec_key)
        if key not in cache:
            # ponytail: mirrors core.outputs.task_log_path's default layout. A
            # task that overrides SFLOW_TASK_OUTPUT_DIR lands elsewhere and
            # resolves unresolved -- window.json names the path that was probed.
            # Carry the real path in the report spec if that override shows up.
            log_path = out_dir.parent / task / f"{task}.log"
            cache[key] = _resolve_log_source(task, log_path, log_window, task_events)
        sources.append(cache[key])

    artifact: dict[str, object] = {
        "schema_version": "sflow.monitor-window.v1",
        "status": "unresolved",
        "sources": sources,
    }
    failed = [source for source in sources if source.get("status") != "matched"]
    if not sources:
        artifact["error"] = "marker report has no owner runtime task"
        return None, None, artifact
    if failed:
        names = ", ".join(str(source["runtime_task"]) for source in failed)
        artifact["error"] = f"unresolved owner runtime task(s): {names}"
        return None, None, artifact

    start = min(
        (dict(source["start"]) for source in sources),  # type: ignore[arg-type]
        key=lambda boundary: float(boundary["timestamp"]),
    )
    end = max(
        (dict(source["end"]) for source in sources),  # type: ignore[arg-type]
        key=lambda boundary: float(boundary["timestamp"]),
    )
    # Every source has end > start, so min(start) < max(end) holds by construction.
    start_ts = float(start["timestamp"])
    end_ts = float(end["timestamp"])
    artifact.update(
        {
            "status": "matched",
            "start": start,
            "end": end,
            "duration_seconds": end_ts - start_ts,
        }
    )
    return start_ts, end_ts, artifact


def _write_window_artifact(
    out_dir: Path, name: str, artifact: dict[str, object]
) -> None:
    report_dir = _report_dir(out_dir, name, windowed=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    # An unresolved window leaves no timeline.csv, so the filename is the fastest
    # signal for "this folder is empty because the markers did not match".
    matched = artifact.get("status") == "matched"
    filename = "window.json" if matched else "window_not_found.json"
    (report_dir / filename).write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    windowed: bool = False,
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
    if windowed and not filtered:
        # The markers matched, so nothing upstream warns -- but the resolved range
        # holds no samples, so this folder gets empty CSVs and no chart. Most often
        # a clock/timezone offset between the task's host and this one: marker times
        # are parsed from the task log as LOCAL time, samples are epoch.
        print(
            f"WARN: marker window for '{name}' matched but covers no samples; "
            "check for a clock/timezone offset between the task host and this host",
            file=sys.stderr,
        )
    consumer_dir = _report_dir(out_dir, name, windowed=windowed)
    # CSV is always written (it is the machine-readable source of truth).
    _write_csv(consumer_dir / "timeline.csv", filtered, _CSV_FIELDS)
    _write_csv(consumer_dir / "summary.csv", summary_rows(filtered), _SUMMARY_FIELDS)

    # One image per node once a report spans more than one: a single chart with
    # every node's devices on it is unreadable, and the per-device legend has
    # nowhere to put them. Named by the real hostname so a panel maps to a
    # machine you can go and look at. The CSVs above stay combined.
    hosts = sorted({str(row["hostname"]) for row in filtered})
    if len(hosts) > 1:
        renders = [
            (
                f"timeline.{host}",
                [r for r in filtered if str(r["hostname"]) == host],
                f"{chart_title} - {host}",
            )
            for host in hosts
        ]
    else:
        renders = [("timeline", filtered, chart_title)]

    for stem, subset, subtitle in renders:
        if not subset:
            continue
        # SVG: lightweight, pure-stdlib vector timeline (the default format).
        if "svg" in formats:
            try:
                if not _render_svg(
                    subset, consumer_dir / f"{stem}.svg", title=subtitle, events=events
                ):
                    print(
                        f"WARN: no plottable metrics for '{name}'; SVG skipped",
                        file=sys.stderr,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"WARN: SVG rendering failed for '{name}': {exc}", file=sys.stderr)

        # PNG: optional raster output via matplotlib. `process` only enables this
        # when matplotlib is importable, so a failure here is a real plot error.
        if "png" in formats and matplotlib_ok:
            try:
                if not _render_png(
                    subset, consumer_dir / f"{stem}.png", title=subtitle, events=events
                ):
                    print(
                        f"WARN: no plottable metrics for '{name}'; PNG skipped",
                        file=sys.stderr,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"WARN: PNG rendering failed for '{name}': {exc}", file=sys.stderr)


# Samples are stamped by the COMPUTE NODE; task events and consumer windows are
# stamped by the DRIVER. Every report intersects the two, so a node whose clock is
# off by more than a run's length produces reports whose windows miss their own
# samples -- empty CSVs and no charts, while the raw logs look perfectly healthy.
# The offset is measurable from files we already have, with no extra handshake:
# the driver knows when it started and stopped collecting, and the samples carry
# the node's own view of that same period.
#
# Correcting a gap smaller than this is not worth it -- the estimate below is only
# accurate to about one sampling interval, so a small "skew" is measurement noise.
_CLOCK_ALIGN_MIN_SECONDS = 1.0


def _clock_bracket(
    n_first: float, n_last: float, t_start: float, t_stop: float, interval: float
) -> tuple[float, float] | None:
    """Bound ``delta = node_clock - driver_clock`` from the collection bracket.

    Let the collector run over driver-clock ``[t_start, t_stop]`` and report
    samples over node-clock ``[n_first, n_last]``. Three facts bound ``delta``:

    * the last sample cannot be written after the collector is killed
      -> ``delta >= n_last - t_stop``;
    * it samples every ``interval``, so the last one is at most one interval
      before the kill -> ``delta <= n_last - t_stop + interval``;
    * the first sample cannot precede the launch (startup latency only makes it
      later) -> ``delta <= n_first - t_start``.

    Every bound is conservative, so the true offset really does lie in the
    returned range -- it is a bound, not a guess. Returns None when the bounds
    CONTRADICT each other (``hi < lo``): that proves one of the assumptions above
    is violated -- most often `release()` stamps `end_ts` before it awaits the
    collector's teardown, so a healthy node can emit one more sample after
    `t_stop`. Declining is the only safe answer; clamping the range would turn
    "my model is wrong" into a zero-width bracket, i.e. maximum confidence.
    """
    lo = n_last - t_stop
    hi = lo + interval if interval > 0 else n_first - t_start
    if interval > 0:
        hi = min(hi, n_first - t_start)
    return (lo, hi) if hi >= lo else None


def _estimate_clock_offsets(
    rows: list[dict[str, object]],
    consumers: list[dict[str, object]],
    interval_ms: object,
) -> dict[str, tuple[float, float, float]]:
    """Per-host ``(estimate, lo, hi)`` clock offset, or ``{}`` when clocks agree.

    The driver-clock collection bracket comes from the consumers' acquire/release
    stamps. A bracket that still contains zero means "no measurable skew" -- the
    healthy case, and the reason a good clock is never "corrected".
    """
    interval = max(float(interval_ms or 0) / 1000.0, 0.0)
    spans: list[tuple[set[str], float, float]] = []
    for consumer in consumers:
        start, stop = consumer.get("start_ts"), consumer.get("end_ts")
        if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
            continue
        nodes = {str(n) for n in (consumer.get("nodes") or [])}
        spans.append((nodes, float(start), float(stop)))
    if not spans:
        return {}

    per_host: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_host[str(row["hostname"])].append(float(row["timestamp"]))

    out: dict[str, tuple[float, float, float]] = {}
    for host, timestamps in per_host.items():
        # Consumers naming this host bound its collector; if none do (the planned
        # node name need not equal the reported hostname -- the same mismatch
        # `filter_rows` tolerates) fall back to the whole monitor's live period.
        matched = [(s, e) for nodes, s, e in spans if host in nodes]
        if not matched:
            matched = [(s, e) for _nodes, s, e in spans]
        bracket = _clock_bracket(
            min(timestamps), max(timestamps),
            min(s for s, _e in matched), max(e for _s, e in matched),
            interval,
        )
        if bracket is None:
            continue
        lo, hi = bracket
        # Only act when the bracket EXCLUDES zero: a correct clock always yields a
        # bracket straddling it, so this cannot fire on a healthy node.
        if lo > _CLOCK_ALIGN_MIN_SECONDS or hi < -_CLOCK_ALIGN_MIN_SECONDS:
            out[host] = ((lo + hi) / 2.0, lo, hi)
    return out


def _align_sample_clocks(
    rows: list[dict[str, object]], offsets: dict[str, tuple[float, float, float]]
) -> None:
    """Shift samples onto the driver's clock IN MEMORY -- never the raw logs.

    Applied after parsing and before any filtering, so every downstream window,
    CSV and chart agrees. ``sflow_monitor/raw/`` keeps the node's own timestamps:
    it is the record of what the node reported, and it is also the evidence the
    clock is wrong.
    """
    if not offsets:
        return
    for row in rows:
        entry = offsets.get(str(row["hostname"]))
        if entry is None:
            continue
        shifted = float(row["timestamp"]) - entry[0]
        row["timestamp"] = shifted
        # Keep the human-readable column consistent with the value beside it.
        row["timestamp_iso"] = datetime.fromtimestamp(shifted).isoformat(
            sep=" ", timespec="milliseconds"
        )
    # Hosts shift by different amounts, so the rows are no longer in time order.
    rows.sort(key=lambda row: float(row["timestamp"]))
    for host, (estimate, lo, hi) in sorted(offsets.items()):
        print(
            f"WARN: clock on '{host}' is ahead of this host by {lo:.1f}..{hi:.1f}s; "
            f"monitor samples were shifted by {estimate:+.1f}s for REPORTING only "
            f"(raw logs under {MONITOR_RAW_HINT} keep the node's own timestamps). "
            f"Windows are accurate to about the sampling interval until the clocks "
            f"are synced (NTP/chrony).",
            file=sys.stderr,
        )


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
    # Calibrate the node clock against the driver's BEFORE any window is applied,
    # otherwise a skewed node filters every report down to nothing.
    _align_sample_clocks(rows, _estimate_clock_offsets(rows, consumers, interval_ms))

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
    window_cache: dict[tuple[str, str], dict[str, object]] = {}
    for report in task_reports:
        if report.get("log_window") is not None:
            start_ts, end_ts, artifact = _resolve_report_log_window(
                report, out_dir, task_events, window_cache
            )
            name = str(report.get("name", "consumer"))
            _write_window_artifact(out_dir, name, artifact)
            report["window_status"] = artifact["status"]
            if artifact["status"] != "matched":
                details = "; ".join(
                    f"{source.get('runtime_task')} ({source.get('source_log')}): "
                    f"{source.get('error')}"
                    for source in artifact.get("sources", [])  # type: ignore[union-attr]
                    if source.get("status") != "matched"
                )
                reason = str(artifact.get("error") or "marker window unresolved")
                if details:
                    reason = f"{reason}; {details}"
                print(
                    f"WARN: marker window unresolved for report '{name}': {reason}",
                    file=sys.stderr,
                )
                continue
        else:
            start_ts, end_ts = _resolve_view_window(
                report.get("window_tasks") or [], task_events
            )
        entry = dict(report)
        entry["start_ts"] = start_ts
        entry["end_ts"] = end_ts
        _write_consumer_report(
            rows, entry, out_dir, matplotlib_ok=matplotlib_ok,
            events=task_events, title=str(report.get("title") or ""),
            windowed=report.get("log_window") is not None,
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
