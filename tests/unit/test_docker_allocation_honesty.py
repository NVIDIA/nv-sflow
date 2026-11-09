# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Docker allocation must describe real hardware.

The resource planner is deliberately backend-agnostic: it reads node count and
per-node GPU capacity off the allocation and applies one rule everywhere. That
only works while every backend's allocation is true. A declared `hosts:` pool is
true by construction -- each entry is a real machine with its own devices, the
same shape a Slurm allocation hands back after discovery.

Synthetic local nodes are where it breaks. `nodes: 2` with no `hosts:` invents
two nodes that are secretly the same machine, and each was reported with the
machine's FULL gpu count -- so a 4-GPU box advertised 8 GPUs, and the planner,
correctly applying the ordinary rule, handed out device indices twice over.
Fixing that in the planner would mean a docker-shaped exception to a rule that
is supposed to be uniform, so the allocation is corrected instead.
"""

import asyncio

import pytest

from sflow.plugins.backends.docker import DockerBackend, DockerBackendConfig


def _backend(**kwargs) -> DockerBackend:
    return DockerBackend(
        DockerBackendConfig(
            name="docker", type="docker", image="ubuntu:22.04", **kwargs
        )
    )


class TestMultiNodeRequiresAPool:
    """`nodes: >1` without `hosts:` is refused outright.

    It used to invent `localhost-1` from this machine and report the box's full
    GPU count once per fake node, so the planner -- correctly applying its one
    backend-agnostic rule -- handed the same physical devices out twice. Three
    guards grew to make that fiction safe. Nothing shipped ever used it, so the
    fiction went instead.
    """

    def test_multi_node_without_hosts_is_rejected(self):
        with pytest.raises(ValueError, match="declares no 'hosts:'"):
            _backend(nodes=2)

    def test_the_error_says_how_to_fix_it(self):
        with pytest.raises(ValueError) as excinfo:
            _backend(nodes=3, gpus_per_node=4)

        message = str(excinfo.value)
        assert "3 nodes" in message
        assert "hosts:" in message, "name the fix"
        assert "nodes: 1" in message, "and the other fix"

    def test_a_single_local_node_with_gpus_is_fine(self):
        backend = _backend(nodes=1, gpus_per_node=4)

        allocation = asyncio.run(backend.allocate())
        assert [n.name for n in allocation.nodes] == ["localhost"]
        assert [n.num_gpus for n in allocation.nodes] == [4]

    def test_the_default_is_still_one_local_node(self):
        allocation = asyncio.run(_backend().allocate())

        assert [n.name for n in allocation.nodes] == ["localhost"]


class TestDeclaredHostsAreTrustedAsRealMachines:
    """A declared pool is the Slurm-cluster analogue: real, distinct machines."""

    def test_each_host_contributes_its_own_gpus(self):
        backend = _backend(
            hosts=[
                {"name": "dgx-a", "address": "10.0.0.11", "gpus_per_node": 8},
                {"name": "dgx-b", "address": "10.0.0.12", "gpus_per_node": 8},
            ]
        )
        allocation = asyncio.run(backend.allocate())

        assert [n.num_gpus for n in allocation.nodes] == [8, 8]

    def test_hosts_may_differ_in_gpu_count(self):
        backend = _backend(
            hosts=[
                {"name": "big", "address": "10.0.0.11", "gpus_per_node": 8},
                {"name": "small", "address": "10.0.0.12", "gpus_per_node": 2},
            ]
        )
        allocation = asyncio.run(backend.allocate())

        # The planner reads capacity per node; nothing here flattens it.
        assert [n.num_gpus for n in allocation.nodes] == [8, 2]

    def test_a_host_pool_falls_back_to_the_backend_wide_gpu_count(self):
        backend = _backend(
            gpus_per_node=4,
            hosts=[
                {"name": "a", "address": "10.0.0.11"},
                {"name": "b", "address": "10.0.0.12"},
            ],
        )
        allocation = asyncio.run(backend.allocate())

        assert [n.num_gpus for n in allocation.nodes] == [4, 4]
