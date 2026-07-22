# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional
from urllib import request


class ProbeTransport(ABC):
    """Abstraction for *where* a network probe check runs.

    The probe classes (TCP/HTTP) own the scheduling/threshold logic; the
    transport owns the actual network I/O. The default
    :class:`LocalProbeTransport` runs checks from the sflow driver host (the
    original behavior). Remote backends whose driver host cannot reach the
    workload network (e.g. Kubernetes) can supply a transport that runs the same
    check from *inside* the cluster instead.
    """

    @abstractmethod
    async def tcp_connect(self, host: str, port: int, timeout: int) -> bool:
        """Return True iff a TCP connection to ``host:port`` succeeds."""
        raise NotImplementedError

    @abstractmethod
    async def http_request(
        self,
        *,
        method: str,
        url: str,
        headers: Optional[dict[str, str]],
        body: Optional[str],
        timeout: int,
    ) -> Optional[int]:
        """Perform an HTTP(S) request and return the status code.

        Returns the integer HTTP status code, or ``None`` if the request could
        not be completed (connection error, timeout, etc.).
        """
        raise NotImplementedError


class LocalProbeTransport(ProbeTransport):
    """Runs probe checks directly from the sflow driver host (default)."""

    async def tcp_connect(self, host: str, port: int, timeout: int) -> bool:
        # The per-attempt deadline is enforced by the caller (Probe.probe wraps
        # check() in asyncio.wait_for), matching the original TcpPortProbe.
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except Exception:
            return False

    async def http_request(
        self,
        *,
        method: str,
        url: str,
        headers: Optional[dict[str, str]],
        body: Optional[str],
        timeout: int,
    ) -> Optional[int]:
        method_upper = method.upper()

        def _do() -> int:
            data: bytes | None = None
            hdrs = dict(headers or {})
            if body is not None:
                data = body.encode("utf-8")
                hdrs.setdefault("Content-Type", "text/plain; charset=utf-8")
            req = request.Request(url, data=data, headers=hdrs, method=method_upper)
            with request.urlopen(req, timeout=max(timeout, 1)) as resp:  # nosec B310
                return int(resp.status)

        try:
            return await asyncio.to_thread(_do)
        except Exception:
            return None


_DEFAULT_TRANSPORT = LocalProbeTransport()


def default_probe_transport() -> ProbeTransport:
    """Return the shared driver-local transport used when none is injected."""
    return _DEFAULT_TRANSPORT
