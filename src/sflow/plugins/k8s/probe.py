# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-cluster probe transport for the Kubernetes backend.

Runs a task's TCP/HTTP readiness/failure checks from *inside* the cluster by
``kubectl exec``-ing ``curl`` in a per-allocation probe pod, instead of dialing
the target from the sflow driver host. This lets probes work when the driver has
no route to the pod network / node IPs (``kubectl exec`` tunnels through the API
server, which the driver already reaches).

The curl argv builders are pure functions (no kubectl/pod state) so they can be
unit-tested directly; the transport only wires them to an injected ``exec_fn``.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from sflow.core.probe_transport import ProbeTransport

# exec_fn(argv, stdin) -> (returncode, stdout, stderr). ``argv`` is the command
# to run *inside* the probe pod (e.g. ["curl", ...]); the implementation prepends
# ``kubectl exec [-i] <pod> -- ``. ``stdin`` (when not None) is piped to the
# process (used for HTTP POST bodies via ``curl --data-binary @-``).
ExecFn = Callable[[list[str], Optional[bytes]], Awaitable[tuple[int, str, str]]]

_CONTENT_TYPE_DEFAULT = "text/plain; charset=utf-8"


def _host_port(host: str, port: int) -> str:
    # Bracket IPv6 literals so curl parses ``[::1]:8000`` correctly.
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def build_tcp_probe_argv(host: str, port: int, timeout: int) -> list[str]:
    """curl argv that reports the TCP connect time to ``host:port``.

    We only care whether the TCP handshake completed, so ``-w %{time_connect}``
    is printed regardless of what the (possibly non-HTTP) service does next; a
    value > 0 means the port accepted the connection.
    """
    t = max(int(timeout), 1)
    return [
        "curl",
        "-sS",
        "-o",
        "/dev/null",
        "--connect-timeout",
        str(t),
        "-m",
        str(t),
        "-w",
        "%{time_connect}",
        f"http://{_host_port(host, port)}",
    ]


def build_http_probe_argv(
    *,
    method: str,
    url: str,
    headers: Optional[dict[str, str]],
    timeout: int,
    has_body: bool,
) -> list[str]:
    """curl argv that prints the HTTP status code (``-w %{http_code}``)."""
    t = max(int(timeout), 1)
    argv = [
        "curl",
        "-sS",
        "-o",
        "/dev/null",
        "-m",
        str(t),
        "-X",
        method.upper(),
        "-w",
        "%{http_code}",
    ]
    for key, value in (headers or {}).items():
        argv += ["-H", f"{key}: {value}"]
    if has_body:
        # Read the request body from stdin (piped by the exec_fn), avoiding any
        # shell-escaping of arbitrary body content.
        argv += ["--data-binary", "@-"]
    argv.append(url)
    return argv


def parse_tcp_connected(stdout: str) -> bool:
    """True iff curl's ``%{time_connect}`` output indicates a completed connect."""
    try:
        return float((stdout or "").strip() or "0") > 0.0
    except ValueError:
        return False


def parse_http_status(stdout: str) -> Optional[int]:
    """Parse curl's ``%{http_code}`` output; ``000``/empty -> None (no response)."""
    try:
        code = int((stdout or "").strip())
    except ValueError:
        return None
    return code if code > 0 else None


class K8sExecProbeTransport(ProbeTransport):
    """Runs probe checks via ``curl`` inside a Kubernetes probe pod."""

    def __init__(self, *, exec_fn: ExecFn):
        self._exec = exec_fn

    async def tcp_connect(self, host: str, port: int, timeout: int) -> bool:
        argv = build_tcp_probe_argv(host, port, timeout)
        try:
            _rc, out, _err = await self._exec(argv, None)
        except Exception:
            return False
        return parse_tcp_connected(out)

    async def http_request(
        self,
        *,
        method: str,
        url: str,
        headers: Optional[dict[str, str]],
        body: Optional[str],
        timeout: int,
    ) -> Optional[int]:
        has_body = body is not None
        hdrs = dict(headers or {})
        stdin: bytes | None = None
        if has_body:
            # Match LocalProbeTransport: default a text/plain Content-Type when
            # the caller didn't set one, and pipe the body via stdin.
            if not any(k.lower() == "content-type" for k in hdrs):
                hdrs["Content-Type"] = _CONTENT_TYPE_DEFAULT
            stdin = (body or "").encode("utf-8")
        argv = build_http_probe_argv(
            method=method,
            url=url,
            headers=hdrs,
            timeout=timeout,
            has_body=has_body,
        )
        try:
            _rc, out, _err = await self._exec(argv, stdin)
        except Exception:
            return None
        return parse_http_status(out)
