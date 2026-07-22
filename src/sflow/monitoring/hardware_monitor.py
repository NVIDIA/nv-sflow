# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bare-node hardware monitor collector.

Runs as a lightweight sidecar on a single compute node, sampling hardware
metrics at a fixed interval and appending CSV rows to per-node log files under
``$SFLOW_TASK_OUTPUT_DIR``. It uses only the Python standard library plus
``nvidia-smi`` (for the ``gpu`` scope), so it can run directly on the host
without a container.

One process is expected per node (sflow launches it with ``ntasks_per_node=1``).
Output filenames embed the node id + hostname so multiple nodes coexist in the
same shared directory:

    <scope>_monitor_node_<node_id>_<hostname>.log

Column layouts (the post-processor depends on these):
    gpu     : ts,index,util_gpu,util_mem,temp_gpu,temp_mem,power,clk_sm,clk_mem,mem_total,mem_used
    cpu     : ts,cpu_pct,load1,load5,load15
    memory  : ts,mem_total,mem_available,mem_used,mem_pct,swap_total,swap_free
    disk    : ts,mount,total_bytes,used_bytes,free_bytes,used_pct
    network : ts,rx_bytes,tx_bytes,rx_bytes_per_s,tx_bytes_per_s,rx_packets_per_s,tx_packets_per_s
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ALL_SCOPES = ("gpu", "cpu", "memory", "disk", "network")

DEFAULT_GPU_QUERY_FIELDS = (
    "index,utilization.gpu,utilization.memory,temperature.gpu,"
    "temperature.memory,power.draw,clocks.sm,clocks.mem,memory.total,memory.used"
)


def _timestamp() -> str:
    current = time.time()
    seconds = int(current)
    millis = int((current - seconds) * 1000)
    return f"{seconds}.{millis:03d}"


def _read_cpu_totals() -> tuple[int, int]:
    cpu_line = Path("/proc/stat").read_text().splitlines()[0]
    parts = [int(part) for part in cpu_line.split()[1:]]
    idle = parts[3]
    iowait = parts[4] if len(parts) > 4 else 0
    idle_total = idle + iowait
    total = sum(parts)
    return total, idle_total


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw_value = line.split(":", 1)
        values[key] = int(raw_value.strip().split()[0])
    return values


def _read_network_totals() -> dict[str, int]:
    totals = {"rx_bytes": 0, "rx_packets": 0, "tx_bytes": 0, "tx_packets": 0}
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        iface, raw_values = [part.strip() for part in line.split(":", 1)]
        if iface == "lo":
            continue
        fields = raw_values.split()
        totals["rx_bytes"] += int(fields[0])
        totals["rx_packets"] += int(fields[1])
        totals["tx_bytes"] += int(fields[8])
        totals["tx_packets"] += int(fields[9])
    return totals


def _append_lines(handle, lines: list[str]) -> None:
    for line in lines:
        handle.write(line + "\n")
    handle.flush()


def _collect_gpu_lines(ts: str, query_fields: str) -> list[str]:
    command = [
        "nvidia-smi",
        f"--query-gpu={query_fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return [f"{ts},ERROR,nvidia-smi not found"]
    if result.returncode != 0:
        stderr = result.stderr.strip().replace("\n", " ") or "nvidia-smi failed"
        return [f"{ts},ERROR,{stderr}"]
    return [f"{ts},{line}" for line in result.stdout.splitlines() if line.strip()]


def _parse_scopes(raw: str) -> list[str]:
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not requested or "all" in requested:
        return list(ALL_SCOPES)
    invalid = [item for item in requested if item not in ALL_SCOPES]
    if invalid:
        raise SystemExit(
            f"hardware_monitor: unknown scope(s) {invalid}; valid: {list(ALL_SCOPES)}"
        )
    # Preserve canonical order and drop duplicates.
    return [scope for scope in ALL_SCOPES if scope in requested]


def _build_log_paths(
    output_dir: Path, node_id: str, hostname: str, scopes: list[str]
) -> dict[str, Path]:
    suffix = f"node_{node_id}_{hostname}.log"
    return {scope: output_dir / f"{scope}_monitor_{suffix}" for scope in scopes}


def _format_log_destinations(log_paths: dict[str, Path]) -> str:
    return ", ".join(f"{name}={path}" for name, path in log_paths.items())


def _log_startup_success(
    output_dir: Path | str, interval_ms: int, log_paths: dict[str, Path]
) -> None:
    print(
        "Hardware monitor started successfully: "
        f"interval_ms={interval_ms} output_dir={output_dir}. "
        f"Logs are saved to: {_format_log_destinations(log_paths)}",
        flush=True,
    )


def _log_startup_failure(
    output_dir: Path | str, log_paths: dict[str, Path], error: Exception
) -> None:
    print(
        "Hardware monitor failed to start: "
        f"output_dir={output_dir} error={error}. "
        f"Logs are saved to: {_format_log_destinations(log_paths)}",
        file=sys.stderr,
        flush=True,
    )


def _sample_once(
    handles: dict,
    ts: str,
    gpu_fields: str,
    previous_cpu: tuple[int, int],
    previous_network: dict[str, int],
    previous_sample_time: float,
) -> tuple[tuple[int, int], dict[str, int], float]:
    """Collect one sample for every active scope; return updated rolling state.

    Raising here is caught by the caller so a single bad read never kills the
    monitor; the rolling ``cpu`` / ``network`` deltas simply resume next tick.
    """
    if "gpu" in handles:
        _append_lines(handles["gpu"], _collect_gpu_lines(ts, gpu_fields))

    if "cpu" in handles:
        current_cpu = _read_cpu_totals()
        delta_total = current_cpu[0] - previous_cpu[0]
        delta_idle = current_cpu[1] - previous_cpu[1]
        cpu_pct = 0.0
        if delta_total > 0:
            cpu_pct = 100.0 * (1.0 - (delta_idle / delta_total))
        load1, load5, load15 = os.getloadavg()
        handles["cpu"].write(
            f"{ts},{cpu_pct:.2f},{load1:.2f},{load5:.2f},{load15:.2f}\n"
        )
        handles["cpu"].flush()
        previous_cpu = current_cpu

    if "memory" in handles:
        meminfo = _read_meminfo()
        mem_total = meminfo.get("MemTotal", 0)
        mem_available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        mem_used = max(mem_total - mem_available, 0)
        mem_pct = (mem_used / mem_total * 100.0) if mem_total else 0.0
        swap_total = meminfo.get("SwapTotal", 0)
        swap_free = meminfo.get("SwapFree", 0)
        handles["memory"].write(
            f"{ts},{mem_total},{mem_available},{mem_used},{mem_pct:.2f},{swap_total},{swap_free}\n"
        )
        handles["memory"].flush()

    if "disk" in handles:
        statvfs = os.statvfs("/")
        total_bytes = statvfs.f_blocks * statvfs.f_frsize
        free_bytes = statvfs.f_bavail * statvfs.f_frsize
        used_bytes = max(total_bytes - free_bytes, 0)
        used_pct = (used_bytes / total_bytes * 100.0) if total_bytes else 0.0
        handles["disk"].write(
            f"{ts},/,{total_bytes},{used_bytes},{free_bytes},{used_pct:.2f}\n"
        )
        handles["disk"].flush()

    if "network" in handles:
        now = time.monotonic()
        elapsed_s = max(now - previous_sample_time, 1e-6)
        current_network = _read_network_totals()
        rx_bytes_per_s = (
            current_network["rx_bytes"] - previous_network.get("rx_bytes", current_network["rx_bytes"])
        ) / elapsed_s
        tx_bytes_per_s = (
            current_network["tx_bytes"] - previous_network.get("tx_bytes", current_network["tx_bytes"])
        ) / elapsed_s
        rx_packets_per_s = (
            current_network["rx_packets"] - previous_network.get("rx_packets", current_network["rx_packets"])
        ) / elapsed_s
        tx_packets_per_s = (
            current_network["tx_packets"] - previous_network.get("tx_packets", current_network["tx_packets"])
        ) / elapsed_s
        handles["network"].write(
            f"{ts},{current_network['rx_bytes']},{current_network['tx_bytes']},"
            f"{rx_bytes_per_s:.2f},{tx_bytes_per_s:.2f},"
            f"{rx_packets_per_s:.2f},{tx_packets_per_s:.2f}\n"
        )
        handles["network"].flush()
        previous_network = current_network
        previous_sample_time = now

    return previous_cpu, previous_network, previous_sample_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default mirrors sflow's DEFAULT_MONITOR_INTERVAL_MS (kept as a literal so this
    # collector stays standalone / importable-free on compute nodes).
    parser.add_argument("--interval-ms", type=int, default=5000)
    parser.add_argument(
        "--scopes",
        default="all",
        help=(
            "Comma-separated hardware scopes to collect "
            f"({', '.join(ALL_SCOPES)}, or 'all'). Default: all."
        ),
    )
    parser.add_argument(
        "--gpu-fields",
        default=DEFAULT_GPU_QUERY_FIELDS,
        help="nvidia-smi --query-gpu fields for the gpu scope.",
    )
    args = parser.parse_args()

    scopes = _parse_scopes(args.scopes)
    interval_ms = max(args.interval_ms, 1)
    interval_s = interval_ms / 1000.0
    node_id = os.environ.get("SLURM_NODEID", "0")
    hostname = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    output_dir_env = os.environ.get("SFLOW_TASK_OUTPUT_DIR")
    # Resolve the output dir + log paths up front so a startup failure can still
    # report the intended destinations (placeholder when the env var is missing).
    output_dir = (
        Path(output_dir_env)
        if output_dir_env
        else Path("<unset SFLOW_TASK_OUTPUT_DIR>")
    )
    log_paths = _build_log_paths(output_dir, node_id, hostname, scopes)

    handles: dict = {}
    try:
        if not output_dir_env:
            raise KeyError("SFLOW_TASK_OUTPUT_DIR")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, path in log_paths.items():
            handles[name] = path.open("a", buffering=1)
        # Record the GPU field layout as a header so the post-processor maps each
        # column to a metric even when --gpu-fields overrides the default set.
        if "gpu" in handles:
            handles["gpu"].write(f"#fields={args.gpu_fields}\n")
            handles["gpu"].flush()
    except Exception as error:
        # Close any handles opened before the failure so we don't leak fds.
        for handle in handles.values():
            handle.close()
        _log_startup_failure(output_dir, log_paths, error)
        raise

    _log_startup_success(output_dir, interval_ms, log_paths)

    previous_cpu = _read_cpu_totals() if "cpu" in handles else (0, 0)
    previous_network = _read_network_totals() if "network" in handles else {}
    previous_sample_time = time.monotonic()

    try:
        while True:
            loop_start = time.monotonic()
            ts = _timestamp()
            try:
                previous_cpu, previous_network, previous_sample_time = _sample_once(
                    handles,
                    ts,
                    args.gpu_fields,
                    previous_cpu,
                    previous_network,
                    previous_sample_time,
                )
            except Exception as error:
                # Best-effort: a single bad read (e.g. a transient /proc parse
                # error) must not kill monitoring for the rest of the run.
                print(
                    f"hardware_monitor: sample error: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            sleep_s = interval_s - (time.monotonic() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
