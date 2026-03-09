# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from sflow.plugins.backends.local import LocalBackend, LocalBackendConfig


def test_local_backend_allocate_returns_localhost_nodes():
    backend = LocalBackend(LocalBackendConfig(name="local", type="local", nodes=2))
    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "local"
    assert [n.name for n in allocation.nodes] == ["localhost", "localhost-1"]
    assert [n.ip_address for n in allocation.nodes] == ["127.0.0.1", "127.0.0.1"]
    assert [n.index for n in allocation.nodes] == [0, 1]


def test_local_backend_allocate_populates_num_gpus_when_configured():
    backend = LocalBackend(
        LocalBackendConfig(name="local", type="local", nodes=2, gpus_per_node=8)
    )
    allocation = asyncio.run(backend.allocate())

    assert [n.num_gpus for n in allocation.nodes] == [8, 8]
