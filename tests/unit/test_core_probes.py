# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sflow.core.probe import Probe, ProbeStatus, ProbeTimeoutError, ProbeType
from sflow.core.probe_transport import LocalProbeTransport, ProbeTransport
from sflow.plugins.probes import (
    HttpGetProbe,
    HttpPostProbe,
    LogWatchProbe,
    TcpPortProbe,
)
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.core.task import Task


class _DummyLogger:
    handlers = []
    propagate = False

    def info(self, *args, **kwargs):  # pragma: no cover
        return


def test_log_watch_probe_triggers_when_pattern_appears(tmp_path: Path):
    # Create a fake workflow output dir structure like SflowApp does:
    # <wf_out>/<task>/<task>.log
    wf_out = tmp_path / "wf"
    (wf_out / "svc").mkdir(parents=True)
    log_path = wf_out / "svc" / "svc.log"
    log_path.write_text("booting...\n")

    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    t.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf_out)

    p = LogWatchProbe(
        regex_pattern=r"READY", type=ProbeType.READINESS, interval=0, timeout=1
    )

    # First tick: not ready yet
    triggered = asyncio.run(p.probe(t))
    assert triggered is False
    assert p.status == ProbeStatus.INITIATED

    # Append readiness line
    log_path.write_text(log_path.read_text() + "READY\n")

    # Next tick: should trigger
    triggered = asyncio.run(p.probe(t))
    assert triggered is True


def test_log_watch_probe_treats_pattern_as_literal_string_by_default(tmp_path: Path):
    wf_out = tmp_path / "wf"
    (wf_out / "svc").mkdir(parents=True)
    log_path = wf_out / "svc" / "svc.log"
    log_path.write_text("Traceback (most recent call last):\nboom\n")

    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    t.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf_out)

    # Parentheses should be matched literally (not treated as regex grouping).
    p = LogWatchProbe(
        regex_pattern="Traceback (most recent call last)",
        type=ProbeType.READINESS,
        interval=0,
        timeout=1,
    )
    assert asyncio.run(p.probe(t)) is True


def test_log_watch_probe_supports_regex_prefix(tmp_path: Path):
    wf_out = tmp_path / "wf"
    (wf_out / "svc").mkdir(parents=True)
    log_path = wf_out / "svc" / "svc.log"
    log_path.write_text("READY 123\n")

    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    t.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf_out)

    p = LogWatchProbe(
        regex_pattern=r"re:READY\s+\d+",
        type=ProbeType.READINESS,
        interval=0,
        timeout=1,
    )
    assert asyncio.run(p.probe(t)) is True


def test_log_watch_probe_match_count(tmp_path: Path):
    """Probe triggers only after pattern is matched match_count times."""
    wf_out = tmp_path / "wf"
    (wf_out / "svc").mkdir(parents=True)
    log_path = wf_out / "svc" / "svc.log"
    log_path.write_text("READY\n")  # 1 match

    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    t.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf_out)

    p = LogWatchProbe(
        regex_pattern="READY",
        type=ProbeType.READINESS,
        interval=0,
        timeout=1,
        match_count=2,
    )

    # Only 1 match so far: not ready
    triggered = asyncio.run(p.probe(t))
    assert triggered is False

    log_path.write_text(log_path.read_text() + "READY\n")  # 2 matches

    triggered = asyncio.run(p.probe(t))
    assert triggered is True


def _make_log_task(tmp_path: Path, content: str = ""):
    wf_out = tmp_path / "wf"
    (wf_out / "svc").mkdir(parents=True)
    log_path = wf_out / "svc" / "svc.log"
    log_path.write_text(content)
    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    t.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf_out)
    return t, log_path


def test_log_watch_probe_scans_incrementally_without_recounting(tmp_path: Path):
    """check() reads only the appended tail and accumulates matches across calls.

    Regression guard for the old behavior that re-read the whole file and
    re-counted every match on each tick (the previously-unused ``_offset``).
    """
    t, log_path = _make_log_task(tmp_path, "booting...\n")
    p = LogWatchProbe(
        regex_pattern="READY",
        type=ProbeType.READINESS,
        interval=0,
        timeout=10,
        match_count=2,
    )

    # First check consumes the existing complete line; no match yet.
    assert asyncio.run(p.check(t)) is False
    assert p._offset == len(b"booting...\n")
    assert p._match_total == 0

    # A half-written line (no trailing newline) is NOT counted or consumed yet.
    log_path.write_text(log_path.read_text() + "READY")
    assert asyncio.run(p.check(t)) is False
    assert p._offset == len(b"booting...\n")
    assert p._match_total == 0

    # Completing the line makes it count exactly once.
    log_path.write_text(log_path.read_text() + "\n")
    assert asyncio.run(p.check(t)) is False  # 1 of 2
    assert p._match_total == 1

    # A second match reaches the threshold.
    log_path.write_text(log_path.read_text() + "READY\n")
    assert asyncio.run(p.check(t)) is True
    assert p._match_total == 2

    # Repeated checks with no new data must not re-count.
    assert asyncio.run(p.check(t)) is True
    assert p._match_total == 2


def test_log_watch_probe_rescans_after_truncation(tmp_path: Path):
    """If the watched log is truncated/rotated (shrinks), the scan restarts."""
    t, log_path = _make_log_task(tmp_path, "first line READY\n")
    p = LogWatchProbe(
        regex_pattern="READY", type=ProbeType.READINESS, interval=0, timeout=10
    )
    assert asyncio.run(p.check(t)) is True
    assert p._offset == len(b"first line READY\n")
    assert p._match_total == 1

    # Rotate: replace with shorter content (size < offset) holding a fresh match.
    log_path.write_text("READY\n")
    assert asyncio.run(p.check(t)) is True
    # Re-scanned from the start rather than seeking past the new (shorter) EOF.
    assert p._offset == len(b"READY\n")
    assert p._match_total == 1


def test_log_watch_probe_reset_restarts_incremental_scan(tmp_path: Path):
    """reset() (called on retry) clears the offset and accumulated match count."""
    t, log_path = _make_log_task(tmp_path, "READY\n")
    p = LogWatchProbe(
        regex_pattern="READY", type=ProbeType.READINESS, interval=0, timeout=10
    )
    assert asyncio.run(p.check(t)) is True
    assert p._offset == len(b"READY\n")
    assert p._match_total == 1

    p.reset()
    assert p._offset == 0
    assert p._match_total == 0
    assert p.status == ProbeStatus.INITIATED

    # After reset the same content is scanned again from the beginning.
    assert asyncio.run(p.check(t)) is True
    assert p._match_total == 1


def test_two_log_watch_probes_on_one_task_read_disk_independently(tmp_path: Path):
    """Regression for the cross-probe 'stealing' bug: two log_watch probes on ONE
    task read the SAME <task>.log, each with its OWN byte offset, so both see every
    line. (The bug lived only in the removed shared-cursor K8s fresh source; the
    per-probe disk offset makes it structurally impossible -- this guards it.)"""
    t, log_path = _make_log_task(
        tmp_path, "starting\nINFO: Application startup complete.\nserving\n"
    )
    ready = LogWatchProbe(
        regex_pattern="Application startup complete",
        type=ProbeType.READINESS,
        interval=0,
        timeout=10,
    )
    failure = LogWatchProbe(
        regex_pattern="Traceback (most recent call last)",
        type=ProbeType.FAILURE,
        interval=0,
        failure_threshold=1,
    )
    # Orchestrator checks each probe in order every tick; the failure probe reading
    # first must NOT consume the readiness marker -- per-probe offset => independent.
    assert asyncio.run(failure.check(t)) is False  # whole file, no Traceback
    assert asyncio.run(ready.check(t)) is True  # still sees the marker
    # Each advanced its OWN offset over the whole file (they don't share state).
    assert ready._offset == failure._offset == log_path.stat().st_size


def test_merged_member_probe_reads_demuxed_log_including_quiet_tail(tmp_path: Path):
    """Regression for the merged-pod "3 of 4 update, 1 stale" bug, now that the probe
    reads each member's DEMUXED ``<task>.log`` (the separate fresh source is gone).

    A quiet member that prints its readiness marker then falls silent while a peer
    keeps logging must still have that marker ON DISK -- the demuxer's periodic
    ``_Router.flush`` lands it (the mawk block-buffering fix) -- and the member's own
    probe, reading ``<SFLOW_WORKFLOW_OUTPUT_DIR>/<name>/<name>.log``, must see it.
    Also pins the invariant that the probe's read path == the demuxer's output path
    (the operator builds ``merge_tag_paths[name]`` from ``SFLOW_TASK_OUTPUT_DIR``,
    which ``configure_task_runtime`` sets to ``SFLOW_WORKFLOW_OUTPUT_DIR/<name>``)."""
    from sflow.plugins.k8s.log_demux import _Router

    wf = tmp_path / "wf"
    # configure_task_runtime: each member's task dir = <workflow_out_dir>/<name>.
    decode_dir = wf / "decode_server"
    prefill_dir = wf / "prefill_server"
    decode_dir.mkdir(parents=True)
    prefill_dir.mkdir(parents=True)
    # The operator routes each member's tagged lines to <TASK_OUTPUT_DIR>/<name>.log.
    routes = {
        b"decode_server": str(decode_dir / "decode_server.log"),
        b"prefill_server": str(prefill_dir / "prefill_server.log"),
    }
    router = _Router(str(wf / "leader.log"), routes)

    # decode_server is the QUIET member: it prints its marker then goes silent, while
    # prefill_server keeps chattering. The demuxer's interval flush lands decode's
    # tail on disk (the mawk failure mode was: it froze, buffered, never written).
    router.route(b"[[sflow-mux:decode_server]] INFO: Application startup complete.")
    router.route(b"[[sflow-mux:prefill_server]] loading weights 10%")
    router.flush()  # periodic flush -> quiet member's line is on disk NOW
    router.route(b"[[sflow-mux:prefill_server]] loading weights 20%")
    router.flush()

    decode = Task(
        name="decode_server",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    decode.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] = str(wf)
    decode.envs["SFLOW_TASK_OUTPUT_DIR"] = str(decode_dir)
    probe = LogWatchProbe(
        regex_pattern="Application startup complete",
        type=ProbeType.READINESS,
        interval=0,
        timeout=10,
    )
    # The probe reads EXACTLY the file the demuxer wrote for this member.
    assert probe._log_path(decode) == Path(routes[b"decode_server"])
    # ...and sees the quiet member's marker (would be missing under the mawk bug).
    assert asyncio.run(probe.check(decode)) is True


# --- TcpPortProbe on_node tests ---


def _mock_connection():
    """Return (reader, writer) where writer.close() is sync and writer.wait_closed() is async."""
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock(return_value=None)
    return (MagicMock(), writer)


def test_tcp_port_probe_on_node_first_passes_when_port_open():
    """on_node=first: probe passes when port is open on the configured host."""
    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    p = TcpPortProbe(
        host="10.0.0.1",
        port=8000,
        on_node="first",
        type=ProbeType.READINESS,
        interval=0,
        timeout=1,
    )
    mock_open = AsyncMock(return_value=_mock_connection())
    with patch("sflow.core.probe_transport.asyncio.open_connection", mock_open):
        result = asyncio.run(p.check(t))
    assert result is True
    mock_open.assert_called_once_with("10.0.0.1", 8000)


def test_tcp_port_probe_on_node_first_fails_when_port_closed():
    """on_node=first: probe fails when connection fails."""
    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    p = TcpPortProbe(
        host="10.0.0.1",
        port=8000,
        on_node="first",
        type=ProbeType.READINESS,
        interval=0,
        timeout=1,
    )
    with patch(
        "sflow.core.probe_transport.asyncio.open_connection",
        AsyncMock(side_effect=ConnectionRefusedError()),
    ):
        result = asyncio.run(p.check(t))
    assert result is False


def test_tcp_port_probe_on_node_each_passes_when_all_ports_open():
    """on_node=each: probe passes when port is open on every assigned node."""
    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    t.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] = "10.0.0.1,10.0.0.2,10.0.0.3"
    p = TcpPortProbe(
        host="10.0.0.1",
        port=8000,
        on_node="each",
        type=ProbeType.READINESS,
        interval=0,
        timeout=1,
    )
    mock_open = AsyncMock(return_value=_mock_connection())
    with patch("sflow.core.probe_transport.asyncio.open_connection", mock_open):
        result = asyncio.run(p.check(t))
    assert result is True
    assert mock_open.call_count == 3
    mock_open.assert_any_call("10.0.0.1", 8000)
    mock_open.assert_any_call("10.0.0.2", 8000)
    mock_open.assert_any_call("10.0.0.3", 8000)


def test_tcp_port_probe_on_node_each_fails_when_one_port_closed():
    """on_node=each: probe fails when port is closed on any assigned node."""
    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    t.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] = "10.0.0.1,10.0.0.2"
    p = TcpPortProbe(
        host="10.0.0.1",
        port=8000,
        on_node="each",
        type=ProbeType.READINESS,
        interval=0,
        timeout=1,
    )

    call_count = 0

    async def open_connection_second_fails(host, port):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ConnectionRefusedError()
        return _mock_connection()

    with patch(
        "sflow.core.probe_transport.asyncio.open_connection",
        side_effect=open_connection_second_fails,
    ):
        result = asyncio.run(p.check(t))
    assert result is False


def test_tcp_port_probe_on_node_each_fallback_when_no_assigned_ips():
    """on_node=each with no SFLOW_TASK_ASSIGNED_NODE_IPS falls back to probe host."""
    t = Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )
    # No SFLOW_TASK_ASSIGNED_NODE_IPS (e.g. local backend)
    p = TcpPortProbe(
        host="127.0.0.1",
        port=8000,
        on_node="each",
        type=ProbeType.READINESS,
        interval=0,
        timeout=1,
    )
    mock_open = AsyncMock(return_value=_mock_connection())
    with patch("sflow.core.probe_transport.asyncio.open_connection", mock_open):
        result = asyncio.run(p.check(t))
    assert result is True
    mock_open.assert_called_once_with("127.0.0.1", 8000)


# --- Probe timeout semantics tests ---


class _AlwaysFailProbe(Probe):
    """Concrete probe that always returns False (never ready)."""

    async def check(self, task: Task) -> bool:
        return False


class _AlwaysPassProbe(Probe):
    """Concrete probe that always returns True."""

    async def check(self, task: Task) -> bool:
        return True


class _SlowCheckProbe(Probe):
    """Probe whose check takes a configurable amount of time."""

    def __init__(self, check_duration: float = 0, **kwargs):
        super().__init__(**kwargs)
        self._check_duration = check_duration

    async def check(self, task: Task) -> bool:
        await asyncio.sleep(self._check_duration)
        return True


def _make_task() -> Task:
    return Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )


def test_readiness_probe_raises_timeout_error_after_deadline():
    """Readiness probe raises ProbeTimeoutError when overall timeout is exceeded."""
    t = _make_task()
    p = _AlwaysFailProbe(type=ProbeType.READINESS, timeout=1, interval=0)

    # First tick: within deadline, just returns False
    result = asyncio.run(p.probe(t))
    assert result is False
    assert p.timed_out is False

    # Simulate time passing beyond the deadline
    p._started_at = time.time() - 2

    with pytest.raises(ProbeTimeoutError, match="timed out after"):
        asyncio.run(p.probe(t))
    assert p.timed_out is True


def test_readiness_probe_succeeds_before_deadline():
    """Readiness probe triggers normally when check passes within the deadline."""
    t = _make_task()
    p = _AlwaysPassProbe(type=ProbeType.READINESS, timeout=600, interval=0)

    result = asyncio.run(p.probe(t))
    assert result is True
    assert p.timed_out is False
    assert p.status == ProbeStatus.INITIATED  # status set by orchestrator


def test_failure_probe_does_not_raise_timeout():
    """Failure probes should never raise ProbeTimeoutError (timeout only for readiness)."""
    t = _make_task()
    p = _AlwaysFailProbe(
        type=ProbeType.FAILURE, timeout=1, interval=0, failure_threshold=1,
    )

    # Simulate time passing beyond the timeout
    p._started_at = time.time() - 2

    # Should NOT raise — failure probes have no overall deadline
    result = asyncio.run(p.probe(t))
    assert result is False
    assert p.timed_out is False


def test_check_timeout_caps_individual_attempt():
    """check_timeout limits how long each individual check can take."""
    t = _make_task()
    p = _SlowCheckProbe(
        check_duration=5,
        type=ProbeType.READINESS,
        timeout=1200,
        each_check_timeout=1,
        interval=0,
    )

    start = time.time()
    result = asyncio.run(p.probe(t))
    elapsed = time.time() - start

    assert result is False
    assert elapsed < 3


def test_probe_reset_clears_timed_out():
    """reset() clears the timed_out flag and resets the deadline."""
    t = _make_task()
    p = _AlwaysFailProbe(type=ProbeType.READINESS, timeout=1, interval=0)

    # Trigger a timeout
    p._started_at = time.time() - 2
    with pytest.raises(ProbeTimeoutError):
        asyncio.run(p.probe(t))
    assert p.timed_out is True

    # Reset should clear everything
    p.reset()
    assert p.timed_out is False
    assert p.status == ProbeStatus.INITIATED
    assert p._success_streak == 0

    # Should work again after reset (no timeout)
    result = asyncio.run(p.probe(t))
    assert result is False
    assert p.timed_out is False


def test_probe_default_values():
    """Verify default parameter values match the new semantics."""
    p = _AlwaysPassProbe(type=ProbeType.READINESS)
    assert p.timeout == 1200
    assert p.each_check_timeout == 30
    assert p.interval == 5
    assert p.success_threshold == 1
    assert p.failure_threshold == 3


def test_effective_check_timeout_honors_each_check_timeout():
    """The per-attempt timeout honors each_check_timeout, independent of interval.

    interval is the gap between checks, not a bound on a single attempt, so it must
    never shrink an explicitly configured each_check_timeout (regression: the old
    interval cap silently reduced a 30s attempt to the default 5s interval).
    """
    # Larger than interval -> still honored (NOT capped to interval).
    p = _AlwaysPassProbe(
        type=ProbeType.READINESS, each_check_timeout=30, interval=20
    )
    assert p.each_check_timeout == 30
    assert p.effective_check_timeout == 30

    # Within interval -> used as-is.
    p2 = _AlwaysPassProbe(
        type=ProbeType.READINESS, each_check_timeout=10, interval=20
    )
    assert p2.effective_check_timeout == 10

    # The default-interval trap: interval=5 must not cap an explicit 30s timeout.
    p3 = _AlwaysPassProbe(
        type=ProbeType.READINESS, each_check_timeout=30, interval=5
    )
    assert p3.effective_check_timeout == 30

    # No interval gating (interval=0) -> honored.
    p4 = _AlwaysPassProbe(
        type=ProbeType.READINESS, each_check_timeout=30, interval=0
    )
    assert p4.effective_check_timeout == 30


# ---------------------------------------------------------------------------
# Probe transport delegation (TCP/HTTP probes route I/O through a transport)
# ---------------------------------------------------------------------------


class _FakeTransport(ProbeTransport):
    def __init__(self, *, tcp_result=True, http_status=200):
        self.tcp_calls: list[tuple] = []
        self.http_calls: list[dict] = []
        self._tcp_result = tcp_result
        self._http_status = http_status

    async def tcp_connect(self, host, port, timeout):
        self.tcp_calls.append((host, port, timeout))
        if callable(self._tcp_result):
            return self._tcp_result(host, port)
        return self._tcp_result

    async def http_request(self, *, method, url, headers, body, timeout):
        self.http_calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "body": body,
                "timeout": timeout,
            }
        )
        return self._http_status


def _probe_task():
    return Task(
        name="svc",
        logger=_DummyLogger(),  # type: ignore[arg-type]
        operator=BashOperator(BashOperatorConfig(name="bash")),
    )


def test_tcp_probe_delegates_to_transport_first():
    ft = _FakeTransport(tcp_result=True)
    p = TcpPortProbe(
        host="10.0.0.1", port=8000, type=ProbeType.READINESS, transport=ft, interval=0
    )
    assert asyncio.run(p.check(_probe_task())) is True
    assert ft.tcp_calls == [("10.0.0.1", 8000, p.effective_check_timeout)]


def test_tcp_probe_each_iterates_assigned_ips_and_stops_on_failure():
    seen: list[str] = []

    def result(host, port):
        seen.append(host)
        return host != "10.0.0.3"  # fail on the second host

    ft = _FakeTransport(tcp_result=result)
    p = TcpPortProbe(
        host="ignored",
        port=9,
        on_node="each",
        type=ProbeType.READINESS,
        transport=ft,
        interval=0,
    )
    t = _probe_task()
    t.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] = "10.0.0.1,10.0.0.3,10.0.0.9"
    assert asyncio.run(p.check(t)) is False
    assert seen == ["10.0.0.1", "10.0.0.3"]


def test_http_get_probe_delegates_and_maps_status():
    ft = _FakeTransport(http_status=200)
    p = HttpGetProbe(
        url="http://svc/health",
        headers={"A": "b"},
        type=ProbeType.READINESS,
        transport=ft,
        interval=0,
    )
    assert asyncio.run(p.check(_probe_task())) is True
    call = ft.http_calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://svc/health"
    assert call["headers"] == {"A": "b"}
    assert call["body"] is None


def test_http_get_probe_non_success_status_is_false():
    ft = _FakeTransport(http_status=500)
    p = HttpGetProbe(
        url="http://svc", type=ProbeType.READINESS, transport=ft, interval=0
    )
    assert asyncio.run(p.check(_probe_task())) is False


def test_http_get_probe_none_status_is_false():
    ft = _FakeTransport(http_status=None)
    p = HttpGetProbe(
        url="http://svc", type=ProbeType.READINESS, transport=ft, interval=0
    )
    assert asyncio.run(p.check(_probe_task())) is False


def test_http_post_probe_passes_body():
    ft = _FakeTransport(http_status=204)
    p = HttpPostProbe(
        url="http://svc/v1",
        body="{}",
        type=ProbeType.READINESS,
        transport=ft,
        interval=0,
    )
    assert asyncio.run(p.check(_probe_task())) is True
    call = ft.http_calls[0]
    assert call["method"] == "POST"
    assert call["body"] == "{}"


# --- Probe last-attempt trace (surfaced in sflow_summary.log) ---


def test_log_watch_probe_last_attempt_trace_on_match(tmp_path: Path):
    t, _ = _make_log_task(tmp_path, "booting\nINFO: Application startup complete.\n")
    p = LogWatchProbe(
        regex_pattern="Application startup complete",
        type=ProbeType.READINESS,
        interval=0,
        timeout=10,
    )
    assert asyncio.run(p.probe(t)) is True
    a = p.last_attempt
    assert a is not None and a.ok is True
    assert "matched" in a.detail
    assert "INFO: Application startup complete." in a.detail  # the matched line
    assert a.runtime >= 0.0


def test_log_watch_probe_last_attempt_trace_on_miss(tmp_path: Path):
    t, _ = _make_log_task(tmp_path, "loading weights 1%\nloading weights 2%\n")
    p = LogWatchProbe(
        regex_pattern="Application startup complete",
        type=ProbeType.READINESS,
        interval=0,
        timeout=10,
    )
    assert asyncio.run(p.probe(t)) is False
    a = p.last_attempt
    assert a is not None and a.ok is False
    assert "no match" in a.detail
    assert "loading weights 2%" in a.detail  # the LAST line the probe saw


def test_tcp_port_probe_last_attempt_trace():
    ft = _FakeTransport(tcp_result=False)
    p = TcpPortProbe(
        host="10.0.0.1", port=8000, type=ProbeType.READINESS, transport=ft, interval=0
    )
    assert asyncio.run(p.probe(_probe_task())) is False
    a = p.last_attempt
    assert a is not None and a.ok is False
    assert a.detail == "tcp 10.0.0.1:8000 closed/unreachable"
    assert p.kind == "tcp_port"


def test_http_probe_last_attempt_trace():
    ft = _FakeTransport(http_status=503)
    p = HttpGetProbe(
        url="http://svc:8000/health", type=ProbeType.READINESS, transport=ft, interval=0
    )
    assert asyncio.run(p.probe(_probe_task())) is False
    a = p.last_attempt
    assert a is not None and a.ok is False
    assert a.detail == "GET http://svc:8000/health -> 503"


def test_probe_last_attempt_trace_on_check_timeout():
    class _SlowTransport(_FakeTransport):
        async def tcp_connect(self, host, port, timeout):
            await asyncio.sleep(5)
            return True

    p = TcpPortProbe(
        host="h", port=1, type=ProbeType.READINESS, transport=_SlowTransport(),
        interval=0, each_check_timeout=1,
    )
    assert asyncio.run(p.probe(_probe_task())) is False
    a = p.last_attempt
    assert a is not None and a.ok is False
    assert "timed out" in a.detail


def test_probes_default_to_local_transport():
    assert isinstance(
        TcpPortProbe(host="127.0.0.1", port=1, type=ProbeType.READINESS)._transport,
        LocalProbeTransport,
    )
    assert isinstance(
        HttpGetProbe(url="http://x", type=ProbeType.READINESS)._transport,
        LocalProbeTransport,
    )
    assert isinstance(
        HttpPostProbe(url="http://x", type=ProbeType.READINESS)._transport,
        LocalProbeTransport,
    )
