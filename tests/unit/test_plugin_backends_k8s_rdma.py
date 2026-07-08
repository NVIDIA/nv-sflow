# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Kubernetes RDMA provider strategy (``_k8s_rdma``)."""

import inspect

from sflow.plugins.backends._k8s_rdma import (
    GkeRdmaProvider,
    HostDeviceRdmaProvider,
    RdmaDetectContext,
    RdmaPlan,
    SharedDevicePluginRdmaProvider,
    build_network_env,
    detect_rdma,
)


def _ctx(
    *,
    hcas=("mlx5_0", "mlx5_1"),
    iface="eth0",
    host_network=True,
    allocatable=None,
    gib_installed=False,
):
    return RdmaDetectContext(
        node_name="node-a",
        node_allocatable=allocatable or {},
        hcas=list(hcas),
        primary_iface=iface,
        host_network=host_network,
        gib_installed=gib_installed,
    )


class TestBuildNetworkEnv:
    def test_has_no_ucx_net_devices_argument(self):
        assert "ucx_net_devices" not in inspect.signature(build_network_env).parameters

    def test_pins_socket_iface_and_exposes_hcas(self):
        env = build_network_env(socket_iface="eth0", rdma_hcas="mlx5_0,mlx5_1")
        assert env["NCCL_SOCKET_IFNAME"] == "eth0"
        assert env["GLOO_SOCKET_IFNAME"] == "eth0"
        assert env["SFLOW_PRIMARY_IFACE"] == "eth0"
        assert env["SFLOW_RDMA_HCAS"] == "mlx5_0,mlx5_1"
        assert "UCX_NET_DEVICES" not in env

    def test_nccl_ib_hca(self):
        env = build_network_env(nccl_ib_hca="mlx5_0")
        assert "UCX_NET_DEVICES" not in env
        assert "SFLOW_UCX_NET_DEVICES" not in env
        assert env["NCCL_IB_HCA"] == "mlx5_0"


class TestRdmaPlanFactories:
    def test_disabled(self):
        p = RdmaPlan.disabled()
        assert not p.enabled and p.net_env == {} and p.nic_specs == ()


class TestGkeProvider:
    _alloc = {
        "networking.gke.io.networks/rdma-1": "1",
        "networking.gke.io.networks/rdma-0": "1",
        "nvidia.com/gpu": "8",
    }

    def test_applies_only_with_gke_resources_and_hcas(self):
        p = GkeRdmaProvider()
        assert p.applies(_ctx(allocatable=self._alloc)) is True
        assert p.applies(_ctx(allocatable={})) is False
        assert p.applies(_ctx(hcas=(), allocatable=self._alloc)) is False

    def test_build_plan_sorts_nics_and_maps_mlx5(self):
        plan = GkeRdmaProvider().build_plan(
            _ctx(allocatable=self._alloc, gib_installed=True)
        )
        assert plan.provider == "gke" and plan.enabled
        assert plan.nic_specs == (
            ("networking.gke.io.networks/rdma-0", "mlx5_0"),
            ("networking.gke.io.networks/rdma-1", "mlx5_1"),
        )
        assert plan.ipc_lock and plan.lib_mounts and plan.nccl_env_script
        assert plan.host_device_paths == ()

    def test_build_plan_omits_lib_mounts_when_gib_absent(self):
        # No installer -> no lib mounts + no NCCL tuning: mounting the driver path
        # /usr/local/nvidia from a missing host dir would mask libcuda.so.1. NICs
        # are still granted (NCCL uses its built-in IB transport over RoCE).
        plan = GkeRdmaProvider().build_plan(
            _ctx(allocatable=self._alloc, gib_installed=False)
        )
        assert plan.provider == "gke" and plan.enabled
        assert plan.nic_specs  # NICs still granted
        assert plan.ipc_lock
        assert plan.lib_mounts == ()
        assert plan.nccl_env_script == ""


class TestSharedDevicePluginProvider:
    def test_applies_on_rdma_resource(self):
        p = SharedDevicePluginRdmaProvider()
        assert p.applies(_ctx(allocatable={"rdma/hca": "1"})) is True
        assert p.applies(_ctx(allocatable={"nvidia.com/gpu": "8"})) is False

    def test_auto_picks_highest_count(self):
        alloc = {"rdma/hca_a": "1", "rdma/hca_b": "4"}
        plan = SharedDevicePluginRdmaProvider().build_plan(_ctx(allocatable=alloc))
        assert plan.nic_specs == (("rdma/hca_b", "mlx5_0"), ("rdma/hca_b", "mlx5_1"))
        assert plan.ipc_lock and plan.host_device_paths == ()
        assert plan.allow_runtime_affinity  # pod sees all HCAs -> runtime selection


class TestHostDeviceProvider:
    def test_applies_only_with_hcas_and_host_network(self):
        p = HostDeviceRdmaProvider()
        assert p.applies(_ctx(host_network=True)) is True
        assert p.applies(_ctx(host_network=False)) is False
        assert p.applies(_ctx(hcas=(), host_network=True)) is False

    def test_build_plan_mounts_device_and_no_resource(self):
        plan = HostDeviceRdmaProvider().build_plan(_ctx())
        assert plan.provider == "host_device" and plan.enabled
        assert plan.host_device_paths == ("/dev/infiniband",)
        assert [r for r, _h in plan.nic_specs] == ["", ""]
        assert plan.ipc_lock


class TestDetectRdmaChain:
    def test_gke_wins_over_shared_and_host(self):
        alloc = {"networking.gke.io.networks/rdma-0": "1", "rdma/hca": "1"}
        assert detect_rdma(_ctx(allocatable=alloc)).provider == "gke"

    def test_shared_wins_over_host(self):
        assert detect_rdma(_ctx(allocatable={"rdma/hca": "1"})).provider == (
            "shared_device_plugin"
        )

    def test_host_device_fallback(self):
        assert detect_rdma(_ctx(allocatable={"nvidia.com/gpu": "8"})).provider == (
            "host_device"
        )

    def test_tcp_fallback_when_no_hcas(self):
        plan = detect_rdma(_ctx(hcas=(), allocatable={}))
        assert plan.provider == "none" and not plan.enabled
        assert "UCX_NET_DEVICES" not in plan.net_env
        assert plan.net_env["NCCL_SOCKET_IFNAME"] == "eth0"

    def test_disabled_when_no_iface_and_no_rdma(self):
        assert detect_rdma(_ctx(hcas=(), iface="", allocatable={})).net_env == {}

    def test_forced_provider_skips_chain(self):
        # rdma/hca present, but forcing host_device uses the device mount instead.
        plan = detect_rdma(_ctx(allocatable={"rdma/hca": "1"}), forced="host_device")
        assert plan.provider == "host_device"

    def test_forced_shared_auto_picks_highest_count(self):
        alloc = {"rdma/a": "1", "rdma/b": "9"}
        plan = detect_rdma(_ctx(allocatable=alloc), forced="shared_device_plugin")
        assert {r for r, _h in plan.nic_specs} == {"rdma/b"}  # highest count wins
