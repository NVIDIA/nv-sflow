#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a workflow's ``used_by_tasks`` monitor targeted the SERVERS.

The self-contained slurm inference recipes attach a monitor to the ``benchmark``
task with ``resources.used_by_tasks: [<server tasks>]`` -- the prefill/decode
servers for disagg, ``agg_server`` for aggregated, ``sglang_server`` for the
server/client sample, etc. With the workflow monitor also enabled (CI injects
``--enable-workflow-monitor``) the post-processor writes, under
``<run>/.../sflow_monitor/``:

* natural views: ``<server>/``, ``benchmark/``
* cross views:   ``<server>__monitored_by__benchmark/``

The set of server(s) is **discovered** from the ``*__monitored_by__benchmark``
folders, so this works for any recipe shape. For each cross view it asserts the
sampled resource footprint (hostnames + GPU ids, read from ``timeline.csv``)
matches the corresponding SERVER's natural view and differs from the BENCHMARK's
-- proving the monitor sampled the servers' resources, not the benchmark client's
own node. Pure standard library so it runs on any CI node without installing sflow.

Exit codes: 0 = targeting correct, 1 = wrong/missing, 2 = not applicable.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BENCH = "benchmark"
CROSS_SUFFIX = f"__monitored_by__{BENCH}"

Footprint = tuple[int, frozenset, frozenset]


def _find_monitor_dir(out_dir: Path) -> Path | None:
    """Locate the ``sflow_monitor/`` reports dir under a run output dir."""
    direct = out_dir / "sflow_monitor"
    if direct.is_dir():
        return direct
    for cand in sorted(out_dir.glob("*/sflow_monitor")):
        if cand.is_dir():
            return cand
    matches = sorted(out_dir.rglob("sflow_monitor"))
    return matches[0] if matches else None


def _footprint(timeline_csv: Path) -> Footprint:
    """``(row_count, hosts, gpu_pairs)`` sampled in a report's ``timeline.csv``."""
    hosts: set[str] = set()
    gpus: set[tuple[str, str]] = set()
    rows = 0
    with timeline_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            host = (row.get("hostname") or "").strip()
            if host:
                hosts.add(host)
            if (row.get("resource_type") or "") == "gpu":
                gpus.add((host, (row.get("resource_id") or "").strip()))
    return rows, frozenset(hosts), frozenset(gpus)


def check(out_dir: Path) -> tuple[int, list[str]]:
    """Return ``(exit_code, messages)`` for one workflow output dir."""
    mon = _find_monitor_dir(out_dir)
    if mon is None:
        return 2, [f"no sflow_monitor/ under {out_dir} (nothing to check)"]
    cross_dirs = sorted(d for d in mon.glob(f"*{CROSS_SUFFIX}") if d.is_dir())
    if not cross_dirs:
        return 2, [f"no *{CROSS_SUFFIX} reports under {mon} (not a used_by_tasks run)"]

    bench_tl = mon / BENCH / "timeline.csv"
    bench_fp = _footprint(bench_tl) if bench_tl.is_file() else None

    # Discover the monitored server(s) from the cross-view folder names, so this
    # works for any recipe shape (prefill/decode, agg_server, sglang_server, ...).
    servers = [d.name[: -len(CROSS_SUFFIX)] for d in cross_dirs]

    msgs: list[str] = []
    errors = 0
    for server in servers:
        cross_tl = mon / f"{server}{CROSS_SUFFIX}" / "timeline.csv"
        if not cross_tl.is_file():
            msgs.append(f"FAIL {server}: missing cross report {cross_tl}")
            errors += 1
            continue
        c_rows, c_hosts, c_gpus = _footprint(cross_tl)
        if c_rows == 0:
            msgs.append(f"FAIL {server}: cross report has no samples ({cross_tl})")
            errors += 1
            continue

        server_tl = mon / server / "timeline.csv"
        if server_tl.is_file():
            _, s_hosts, s_gpus = _footprint(server_tl)
            if (c_hosts, c_gpus) != (s_hosts, s_gpus):
                msgs.append(
                    f"FAIL {server}: cross footprint != server footprint "
                    f"(cross hosts={sorted(c_hosts)} gpus={sorted(c_gpus)} vs "
                    f"server hosts={sorted(s_hosts)} gpus={sorted(s_gpus)})"
                )
                errors += 1
                continue
            # "Not the benchmark's resource", asserted only when the server and
            # benchmark genuinely resolve to different resources (otherwise the
            # topology is indistinguishable and we cannot tell them apart).
            if bench_fp is not None:
                _, b_hosts, b_gpus = bench_fp
                distinguishable = (s_hosts, s_gpus) != (b_hosts, b_gpus)
                if distinguishable and (c_hosts, c_gpus) == (b_hosts, b_gpus):
                    msgs.append(
                        f"FAIL {server}: cross footprint matches the BENCHMARK's, "
                        "not the server's"
                    )
                    errors += 1
                    continue
            msgs.append(
                f"OK   {server}: cross report tracks the server's resources "
                f"(hosts={sorted(c_hosts)} gpus={sorted(c_gpus)})"
            )
        else:
            # No workflow-monitor natural view to compare against -> weaker check:
            # the cross report must carry GPU samples (servers use GPUs) and, when
            # available, differ from the benchmark's footprint.
            if not c_gpus:
                msgs.append(
                    f"FAIL {server}: cross report has no GPU samples and no server "
                    "natural view to compare against"
                )
                errors += 1
                continue
            if bench_fp is not None and (c_hosts, c_gpus) == (bench_fp[1], bench_fp[2]):
                msgs.append(f"FAIL {server}: cross footprint matches the BENCHMARK's")
                errors += 1
                continue
            msgs.append(
                f"OK   {server}: cross report has server GPU samples "
                "(no natural view to cross-check)"
            )

    return (1 if errors else 0), msgs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="workflow output (run) directory")
    args = parser.parse_args()
    rc, msgs = check(Path(args.output_dir))
    for message in msgs:
        print(f"  {message}")
    label = {0: "PASS", 1: "FAIL", 2: "SKIP"}[rc]
    print(f"DISAGG MONITOR TARGETING: {label} ({Path(args.output_dir).name})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
