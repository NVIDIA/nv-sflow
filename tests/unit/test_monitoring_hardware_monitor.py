# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bare-node hardware monitor collector: sampling + loop resilience."""

import io
import sys

import pytest

from sflow.monitoring import hardware_monitor

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="reads /proc/meminfo and os.statvfs (Linux)",
)


def test_sample_once_writes_memory_and_disk():
    handles = {"memory": io.StringIO(), "disk": io.StringIO()}
    prev_cpu, prev_net, prev_ts = hardware_monitor._sample_once(
        handles, "123.000", "", (0, 0), {}, 0.0
    )
    mem = handles["memory"].getvalue().strip().split(",")
    disk = handles["disk"].getvalue().strip().split(",")
    # memory: ts,mem_total,mem_available,mem_used,mem_pct,swap_total,swap_free
    assert mem[0] == "123.000" and len(mem) == 7
    # disk: ts,mount,total,used,free,used_pct
    assert disk[0] == "123.000" and disk[1] == "/" and len(disk) == 6
    # No cpu/network scopes -> rolling state passes through unchanged.
    assert prev_cpu == (0, 0) and prev_net == {} and prev_ts == 0.0


def test_main_loop_survives_transient_sample_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SFLOW_TASK_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys, "argv", ["hw", "--interval-ms", "100", "--scopes", "memory"]
    )

    real_meminfo = hardware_monitor._read_meminfo
    calls = {"n": 0}

    def flaky_meminfo():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient /proc read")
        return real_meminfo()

    monkeypatch.setattr(hardware_monitor, "_read_meminfo", flaky_meminfo)

    sleeps = {"n": 0}

    def fake_sleep(_seconds):
        sleeps["n"] += 1
        if sleeps["n"] >= 3:  # let a couple of good samples through, then stop
            raise KeyboardInterrupt

    monkeypatch.setattr(hardware_monitor.time, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        hardware_monitor.main()

    # The first sample raised, but monitoring kept going and recorded later rows.
    logs = list(tmp_path.glob("memory_monitor_*.log"))
    assert logs, "expected a memory monitor log file"
    assert logs[0].read_text().strip(), "expected sampled rows after the error"
    assert "sample error" in capsys.readouterr().err
