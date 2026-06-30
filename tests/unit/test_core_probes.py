# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sflow.core.probe import Probe, ProbeStatus, ProbeTimeoutError, ProbeType
from sflow.plugins.probes import LogWatchProbe, TcpPortProbe
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
    with patch("sflow.plugins.probes.tcp_port.asyncio.open_connection", mock_open):
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
        "sflow.plugins.probes.tcp_port.asyncio.open_connection",
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
    with patch("sflow.plugins.probes.tcp_port.asyncio.open_connection", mock_open):
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
        "sflow.plugins.probes.tcp_port.asyncio.open_connection",
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
    with patch("sflow.plugins.probes.tcp_port.asyncio.open_connection", mock_open):
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


def test_effective_check_timeout_falls_back_to_interval():
    """each_check_timeout is capped to the interval when it would exceed it."""
    # Larger than interval -> falls back to interval (configured value preserved).
    p = _AlwaysPassProbe(
        type=ProbeType.READINESS, each_check_timeout=30, interval=20
    )
    assert p.each_check_timeout == 30
    assert p.effective_check_timeout == 20

    # Within interval -> used as-is.
    p2 = _AlwaysPassProbe(
        type=ProbeType.READINESS, each_check_timeout=10, interval=20
    )
    assert p2.effective_check_timeout == 10

    # No interval gating (interval=0) -> no cap.
    p3 = _AlwaysPassProbe(
        type=ProbeType.READINESS, each_check_timeout=30, interval=0
    )
    assert p3.effective_check_timeout == 30
