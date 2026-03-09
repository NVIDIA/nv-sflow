# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Literal

from sflow.core.probe import Probe, ProbeType


async def _check_one(host: str, port: int) -> bool:
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


class TcpPortProbe(Probe):
    """
    TCP port probe. on_node: "first" = condition met when port is open on the
    first assigned node; "each" = condition met when port is open on every
    assigned node.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        on_node: Literal["first", "each"] = "first",
        type: ProbeType,
        **kwargs,
    ):
        super().__init__(type=type, **kwargs)
        self._host = str(host)
        self._port = int(port)
        self._on_node = on_node

    async def check(self, task) -> bool:  # type: ignore[override]
        if self._on_node == "first":
            return await _check_one(self._host, self._port)

        # each: require port open on every node assigned to this task
        ips_raw = task.envs.get("SFLOW_TASK_ASSIGNED_NODE_IPS", "").strip()
        hosts = [h.strip() for h in ips_raw.split(",") if h.strip()]
        if not hosts:
            return await _check_one(self._host, self._port)
        for host in hosts:
            if not await _check_one(host, self._port):
                return False
        return True
