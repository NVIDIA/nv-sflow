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


def test_disk_root_can_be_pointed_at_a_bind_mounted_host_root(monkeypatch, tmp_path):
    """A collector in a container measures its image overlay otherwise.

    Whoever puts it there binds the node's real root somewhere and names it, so
    the numbers describe the machine rather than the image -- while the row is
    still filed as `/`, which is what that path IS on the node.
    """
    seen: list[str] = []
    real_statvfs = hardware_monitor.os.statvfs
    monkeypatch.setattr(
        hardware_monitor.os,
        "statvfs",
        lambda p: (seen.append(p), real_statvfs("/"))[1],
    )
    monkeypatch.setenv("SFLOW_MONITOR_DISK_ROOT", str(tmp_path))

    handles = {"disk": io.StringIO()}
    hardware_monitor._sample_once(handles, "123.000", "", (0, 0), {}, 0.0)

    assert seen == [str(tmp_path)]
    assert handles["disk"].getvalue().strip().split(",")[1] == "/"


def test_disk_root_defaults_to_this_machines_root():
    """Unset on a bare host -- slurm/local must keep sampling `/` as before."""
    seen: list[str] = []
    real_statvfs = hardware_monitor.os.statvfs
    original = hardware_monitor.os.statvfs
    hardware_monitor.os.statvfs = lambda p: (seen.append(p), real_statvfs("/"))[1]
    try:
        hardware_monitor._sample_once(
            {"disk": io.StringIO()}, "123.000", "", (0, 0), {}, 0.0
        )
    finally:
        hardware_monitor.os.statvfs = original

    assert seen == ["/"]


def test_tagged_stream_prefixes_each_line_with_its_scope():
    out = io.StringIO()
    stream = hardware_monitor._TaggedStream("memory", out)
    stream.write("123.000,1,2\n")
    # _append_lines hands over one line per write; the gpu header arrives the
    # same way. Multi-line writes must still tag every line, not just the first.
    stream.write("124.000,3,4\n125.000,5,6\n")
    assert out.getvalue().splitlines() == [
        "memory\t123.000,1,2",
        "memory\t124.000,3,4",
        "memory\t125.000,5,6",
    ]


def test_stdout_mode_streams_samples_and_keeps_stdout_data_only(
    tmp_path, monkeypatch, capsys
):
    """--stdout carries samples over the pipe for a driver that cannot see this
    machine's filesystem. Status text must go to stderr, or it would be parsed
    as a sample row."""
    monkeypatch.setenv("SFLOW_TASK_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys, "argv", ["hw", "--interval-ms", "100", "--scopes", "memory", "--stdout"]
    )

    sleeps = {"n": 0}

    def fake_sleep(_seconds):
        sleeps["n"] += 1
        if sleeps["n"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(hardware_monitor.time, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        hardware_monitor.main()

    captured = capsys.readouterr()
    rows = [line for line in captured.out.splitlines() if line.strip()]
    assert rows, "expected sample rows on stdout"
    assert all(line.startswith("memory\t") for line in rows), captured.out
    # Streaming must not also write files: the directory is the driver's, and on
    # a remote host it is a path that happens to exist and belongs to no one.
    assert not list(tmp_path.glob("*.log"))
    assert "Hardware monitor started successfully" in captured.err


def test_stdout_mode_emits_the_gpu_fields_header(monkeypatch, capsys):
    """The post-processor maps gpu columns from this header, so it has to
    survive the stream like any other row."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["hw", "--interval-ms", "100", "--scopes", "gpu", "--stdout",
         "--gpu-fields", "index,power.draw"],
    )
    monkeypatch.setattr(
        hardware_monitor, "_collect_gpu_lines", lambda ts, fields: [f"{ts},0,42"]
    )

    def fake_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(hardware_monitor.time, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        hardware_monitor.main()

    lines = capsys.readouterr().out.splitlines()
    assert "gpu\t#fields=index,power.draw" in lines


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


class TestLoopbackIsReportedNotDropped:
    """`lo` is excluded from rx/tx but counted on its own.

    Two processes on one host talk over loopback, so a transfer that never
    reaches a NIC is invisible in rx/tx. A co-located disaggregated inference
    run moved ~112 GiB of KV that way while every network metric read zero --
    the run looked idle when it was entirely transfer-bound.
    """

    @staticmethod
    def _proc_net_dev(tmp_path, lo_rx, eth_rx, eth_tx):
        p = tmp_path / "dev"
        p.write_text(
            "Inter-|   Receive                    |  Transmit\n"
            " face |bytes    packets errs drop fifo frame compressed multicast|"
            "bytes    packets errs drop fifo colls carrier compressed\n"
            f"    lo: {lo_rx} 10 0 0 0 0 0 0 {lo_rx} 10 0 0 0 0 0 0\n"
            f"  eth0: {eth_rx} 20 0 0 0 0 0 0 {eth_tx} 30 0 0 0 0 0 0\n"
        )
        return p

    def _totals(self, monkeypatch, tmp_path, **kw):
        from sflow.monitoring import hardware_monitor as hm

        path = self._proc_net_dev(tmp_path, **kw)
        monkeypatch.setattr(hm, "Path", lambda _p: path)
        return hm._read_network_totals()

    def test_loopback_is_counted_separately(self, monkeypatch, tmp_path):
        totals = self._totals(monkeypatch, tmp_path, lo_rx=5000, eth_rx=100, eth_tx=200)

        assert totals["loopback_bytes"] == 5000
        # And kept OUT of rx/tx, which mean "left this machine".
        assert totals["rx_bytes"] == 100
        assert totals["tx_bytes"] == 200

    def test_a_purely_local_transfer_is_no_longer_invisible(self, monkeypatch, tmp_path):
        """The exact case that hid the KV handoff: busy loopback, idle NIC."""
        totals = self._totals(monkeypatch, tmp_path, lo_rx=10**11, eth_rx=0, eth_tx=0)

        assert totals["rx_bytes"] == 0 and totals["tx_bytes"] == 0
        assert totals["loopback_bytes"] == 10**11, "the traffic must show somewhere"


def test_postprocessor_still_reads_rows_written_before_loopback_existed():
    """The column was appended, so older raw logs are one field short."""
    from sflow.monitoring.postprocess_monitor_timeline import _NETWORK_METRICS

    old_row = "1790000000.0,111,222,1.0,2.0,3.0,4.0".split(",")
    parsed = [
        (name, old_row[1 + i])
        for i, (name, _unit) in enumerate(_NETWORK_METRICS)
        if 1 + i < len(old_row)
    ]

    names = [n for n, _ in parsed]
    assert "rx_bytes_per_s" in names and "tx_packets_per_s" in names
    assert "loopback_bytes_per_s" not in names, "absent, not a crash"
