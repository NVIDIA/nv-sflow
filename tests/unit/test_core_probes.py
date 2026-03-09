# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from sflow.core.probe import ProbeStatus, ProbeType
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
