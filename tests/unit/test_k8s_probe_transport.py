# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from sflow.plugins.k8s.probe import (
    K8sExecProbeTransport,
    build_http_probe_argv,
    build_tcp_probe_argv,
    parse_http_status,
    parse_tcp_connected,
)


# ---------------------------------------------------------------------------
# curl argv builders (pure)
# ---------------------------------------------------------------------------


def test_build_tcp_probe_argv():
    argv = build_tcp_probe_argv("10.0.0.1", 8000, 5)
    assert argv[0] == "curl"
    assert "http://10.0.0.1:8000" in argv
    assert "%{time_connect}" in argv
    assert "--connect-timeout" in argv
    assert "-m" in argv
    assert "5" in argv


def test_build_tcp_probe_argv_ipv6_is_bracketed():
    argv = build_tcp_probe_argv("::1", 80, 3)
    assert "http://[::1]:80" in argv


def test_build_tcp_probe_argv_timeout_floor():
    # timeout <= 0 is floored to 1
    argv = build_tcp_probe_argv("h", 1, 0)
    assert "1" in argv


def test_build_http_probe_argv_get():
    argv = build_http_probe_argv(
        method="GET", url="http://x/h", headers={"A": "b"}, timeout=4, has_body=False
    )
    assert argv[:2] == ["curl", "-sS"]
    assert "-X" in argv and "GET" in argv
    assert "%{http_code}" in argv
    hidx = argv.index("-H")
    assert argv[hidx + 1] == "A: b"
    assert "--data-binary" not in argv
    assert argv[-1] == "http://x/h"


def test_build_http_probe_argv_post_uses_stdin_data():
    argv = build_http_probe_argv(
        method="POST", url="http://x", headers=None, timeout=2, has_body=True
    )
    assert "--data-binary" in argv
    assert argv[argv.index("--data-binary") + 1] == "@-"
    assert "POST" in argv


# ---------------------------------------------------------------------------
# result parsers
# ---------------------------------------------------------------------------


def test_parse_tcp_connected():
    assert parse_tcp_connected("0.001234") is True
    assert parse_tcp_connected("0.000000") is False
    assert parse_tcp_connected("") is False
    assert parse_tcp_connected("garbage") is False


def test_parse_http_status():
    assert parse_http_status("200") == 200
    assert parse_http_status("301") == 301
    assert parse_http_status("500") == 500
    assert parse_http_status("000") is None
    assert parse_http_status("") is None
    assert parse_http_status("weird") is None


# ---------------------------------------------------------------------------
# K8sExecProbeTransport
# ---------------------------------------------------------------------------


def test_transport_tcp_connect_uses_exec_no_stdin():
    calls = []

    async def fake_exec(argv, stdin):
        calls.append((argv, stdin))
        return (0, "0.010", "")

    tr = K8sExecProbeTransport(exec_fn=fake_exec)
    assert asyncio.run(tr.tcp_connect("h", 1, 2)) is True
    assert calls[0][1] is None
    assert calls[0][0][0] == "curl"


def test_transport_tcp_connect_failure():
    async def fake_exec(argv, stdin):
        return (7, "0.000000", "connection refused")

    tr = K8sExecProbeTransport(exec_fn=fake_exec)
    assert asyncio.run(tr.tcp_connect("h", 1, 2)) is False


def test_transport_http_get_returns_status():
    async def fake_exec(argv, stdin):
        return (0, "200", "")

    tr = K8sExecProbeTransport(exec_fn=fake_exec)
    status = asyncio.run(
        tr.http_request(method="GET", url="http://x", headers=None, body=None, timeout=1)
    )
    assert status == 200


def test_transport_http_post_pipes_body_and_defaults_content_type():
    seen = {}

    async def fake_exec(argv, stdin):
        seen["argv"] = argv
        seen["stdin"] = stdin
        return (0, "204", "")

    tr = K8sExecProbeTransport(exec_fn=fake_exec)
    status = asyncio.run(
        tr.http_request(
            method="POST", url="http://x/v1", headers=None, body="hello", timeout=1
        )
    )
    assert status == 204
    assert seen["stdin"] == b"hello"
    argv = seen["argv"]
    headers = [argv[i + 1] for i, a in enumerate(argv) if a == "-H"]
    assert any(h.lower().startswith("content-type:") for h in headers)
    assert "--data-binary" in argv


def test_transport_http_post_respects_explicit_content_type():
    seen = {}

    async def fake_exec(argv, stdin):
        seen["argv"] = argv
        return (0, "200", "")

    tr = K8sExecProbeTransport(exec_fn=fake_exec)
    asyncio.run(
        tr.http_request(
            method="POST",
            url="http://x",
            headers={"Content-Type": "application/json"},
            body="{}",
            timeout=1,
        )
    )
    argv = seen["argv"]
    headers = [argv[i + 1] for i, a in enumerate(argv) if a == "-H"]
    content_types = [h for h in headers if h.lower().startswith("content-type:")]
    assert content_types == ["Content-Type: application/json"]


def test_transport_exec_exception_is_safe():
    async def boom(argv, stdin):
        raise RuntimeError("kaboom")

    tr = K8sExecProbeTransport(exec_fn=boom)
    assert asyncio.run(tr.tcp_connect("h", 1, 1)) is False
    assert (
        asyncio.run(
            tr.http_request(
                method="GET", url="http://x", headers=None, body=None, timeout=1
            )
        )
        is None
    )
