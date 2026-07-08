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
            return await self._transport.tcp_connect(self._host, self._port, timeout)

        # each: require port open on every node assigned to this task
        ips_raw = task.envs.get("SFLOW_TASK_ASSIGNED_NODE_IPS", "").strip()
        hosts = [h.strip() for h in ips_raw.split(",") if h.strip()]
        if not hosts:
            return await self._transport.tcp_connect(self._host, self._port, timeout)
        for host in hosts:
            if not await self._transport.tcp_connect(host, self._port, timeout):
                return False
        return True
