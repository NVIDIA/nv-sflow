# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from sflow.core.probe import Probe, ProbeType
from sflow.core.probe_transport import ProbeTransport, default_probe_transport


class TcpPortProbe(Probe):
    """
    TCP port probe. on_node: "first" = condition met when port is open on the
    first assigned node; "each" = condition met when port is open on every
    assigned node.

    The actual connection is delegated to a :class:`ProbeTransport` so the check
    can run either from the sflow driver host (default) or from inside a remote
    backend's network (e.g. a Kubernetes probe pod).
    """

    kind = "tcp_port"

    def __init__(
        self,
        *,
        host: str,
        port: int,
        on_node: Literal["first", "each"] = "first",
        type: ProbeType,
        transport: ProbeTransport | None = None,
        **kwargs,
    ):
        super().__init__(type=type, **kwargs)
        self._host = str(host)
        self._port = int(port)
        self._on_node = on_node
        self._transport = transport or default_probe_transport()

    async def check(self, task) -> bool:  # type: ignore[override]
        timeout = self.effective_check_timeout
        if self._on_node == "first":
            ok = await self._transport.tcp_connect(self._host, self._port, timeout)
            self._attempt_detail = self._one_detail(self._host, ok)
            return ok

        # each: require port open on every node assigned to this task
        ips_raw = task.envs.get("SFLOW_TASK_ASSIGNED_NODE_IPS", "").strip()
        hosts = [h.strip() for h in ips_raw.split(",") if h.strip()]
        if not hosts:
            ok = await self._transport.tcp_connect(self._host, self._port, timeout)
            self._attempt_detail = self._one_detail(self._host, ok)
            return ok
        for host in hosts:
            if not await self._transport.tcp_connect(host, self._port, timeout):
                self._attempt_detail = (
                    f"tcp {host}:{self._port} closed/unreachable "
                    f"(on_node=each, {len(hosts)} node(s))"
                )
                return False
        self._attempt_detail = f"tcp :{self._port} open on all {len(hosts)} node(s)"
        return True

    def _one_detail(self, host: str, ok: bool) -> str:
        return f"tcp {host}:{self._port} {'open' if ok else 'closed/unreachable'}"
