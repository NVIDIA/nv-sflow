# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import unittest.mock as mock

import pytest

from sflow.core.backend import Allocation
from sflow.plugins.backends import kubernetes as k8s_mod
from sflow.plugins.backends.kubernetes import (
    KubernetesBackend,
    KubernetesBackendConfig,
    KubernetesDraConfig,
    KubernetesReservationConfig,
)
from sflow.plugins.operators.bash import BashOperator


def _cfg(**kwargs) -> KubernetesBackendConfig:
    base = {"name": "k8s", "type": "kubernetes"}
    base.update(kwargs)
    return KubernetesBackendConfig(**base)


# ---------------------------------------------------------------------------
# placeholder_allocation
# ---------------------------------------------------------------------------


def test_placeholder_allocation_returns_synthetic_nodes():
    backend = KubernetesBackend(_cfg(nodes=2, gpus_per_node=8))
    allocation = backend.placeholder_allocation()
    assert allocation.allocation_id == "kubernetes"
    assert allocation.owned is False
    assert [n.name for n in allocation.nodes] == ["k8s-node0", "k8s-node1"]
    assert [n.num_gpus for n in allocation.nodes] == [8, 8]
    assert all(n.ip_address for n in allocation.nodes)


# ---------------------------------------------------------------------------
# PVC volume config
# ---------------------------------------------------------------------------


def test_volumes_property_normalizes_pvc_config():
    backend = KubernetesBackend(_cfg(volumes=[
        {"name": "model-store", "claim": "model-pvc", "mount_path": "/models"},
        {"name": "scratch", "claim": "scratch-pvc", "mount_path": "/scratch",
         "sub_path": "out", "read_only": False},
    ]))
    vols = backend.volumes
    assert vols[0] == {"name": "model-store", "claim": "model-pvc",
                       "mount_path": "/models", "sub_path": None, "read_only": True,
                       "ensure_writable": False}
    assert vols[1]["sub_path"] == "out" and vols[1]["read_only"] is False


def test_volumes_property_empty_when_unset():
    assert KubernetesBackend(_cfg()).volumes == []


def test_volume_mount_path_must_be_absolute():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _cfg(volumes=[{"name": "v", "claim": "c", "mount_path": "relative/path"}])


def test_volume_pvc_defaults_read_only_true():
    be = KubernetesBackend(
        _cfg(volumes=[{"name": "m", "claim": "model-pvc", "mount_path": "/models"}])
    )
    vol = be.volumes[0]
    assert vol["claim"] == "model-pvc"
    assert vol["read_only"] is True  # PVCs default read-only
    assert "empty_dir" not in vol


def test_volume_emptydir_defaults_writable():
    be = KubernetesBackend(
        _cfg(volumes=[{"name": "kernel-cache", "empty_dir": {}, "mount_path": "/cache"}])
    )
    vol = be.volumes[0]
    assert vol["name"] == "kernel-cache"
    assert vol["mount_path"] == "/cache"
    assert vol["empty_dir"] == {"medium": "", "size_limit": None}
    assert "claim" not in vol
    assert vol["read_only"] is False  # emptyDir scratch defaults writable


def test_volume_emptydir_medium_and_size_limit():
    be = KubernetesBackend(
        _cfg(
            volumes=[
                {
                    "name": "c",
                    "empty_dir": {"medium": "Memory", "size_limit": "10Gi"},
                    "mount_path": "/cache",
                }
            ]
        )
    )
    assert be.volumes[0]["empty_dir"] == {"medium": "Memory", "size_limit": "10Gi"}


def test_volume_requires_exactly_one_source():
    import pydantic

    # neither claim nor empty_dir
    with pytest.raises(pydantic.ValidationError):
        _cfg(volumes=[{"name": "x", "mount_path": "/x"}])
    # both claim and empty_dir
    with pytest.raises(pydantic.ValidationError):
        _cfg(
            volumes=[
                {"name": "x", "claim": "c", "empty_dir": {}, "mount_path": "/x"}
            ]
        )


def test_volume_ensure_writable_valid_on_writable_pvc():
    be = KubernetesBackend(
        _cfg(
            volumes=[
                {
                    "name": "kernel-cache",
                    "claim": "c",
                    "mount_path": "/cache",
                    "sub_path": "sflow-kernel-cache",
                    "read_only": False,
                    "ensure_writable": True,
                }
            ]
        )
    )
    vol = be.volumes[0]
    assert vol["ensure_writable"] is True
    assert vol["read_only"] is False


def test_volume_ensure_writable_requires_read_only_false():
    import pydantic

    # PVC read_only defaults to True -> ensure_writable is contradictory
    with pytest.raises(pydantic.ValidationError):
        _cfg(
            volumes=[
                {"name": "c", "claim": "x", "mount_path": "/cache",
                 "ensure_writable": True}
            ]
        )


def test_volume_ensure_writable_rejected_on_emptydir():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _cfg(
            volumes=[
                {"name": "c", "empty_dir": {}, "mount_path": "/cache",
                 "read_only": False, "ensure_writable": True}
            ]
        )


def test_rdma_config_modes():
    # auto (default): nothing injected until detection runs at allocation.
    assert KubernetesBackend(_cfg()).network_env == {}
    # off: never inject.
    assert KubernetesBackend(_cfg(rdma="off")).network_env == {}
    # a named provider pins that mechanism (chain still runs at allocation).
    be = KubernetesBackend(_cfg(rdma="shared_device_plugin"))
    assert be._rdma_mode == "auto" and be._rdma_forced == "shared_device_plugin"
    assert be.network_env == {}


def test_rdma_invalid_string_rejected():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _cfg(rdma="bogus")


def _fake_kubectl(probe_out, *, node_json=None, node_name="node-a", installer_pods=""):
    """Fake ``_kubectl`` answering the RDMA exec probe + node-allocatable fetch.

    ``installer_pods`` is the space-separated pod name list returned for the
    per-node gIB-installer presence check (``get pods -n kube-system
    --field-selector status.phase=Running,spec.nodeName=<node>``).
    """

    async def _run(args):
        a = list(args)
        if a and a[0] == "exec":
            return 0, probe_out, ""
        if a[:2] == ["get", "pod"]:
            return 0, node_name, ""
        if a[:2] == ["get", "node"]:
            return (0, node_json, "") if node_json is not None else (1, "", "")
        if a[:2] == ["get", "pods"]:
            return 0, installer_pods, ""
        return 0, "", ""

    return _run


def test_detect_network_env_host_device_when_hcas_and_host_network(monkeypatch):
    # HCAs present + host_network + no device plugin -> the host-device provider
    # grants verbs access via /dev/infiniband + IPC_LOCK (RDMA, not the TCP fallback).
    be = KubernetesBackend(_cfg())  # host_network defaults True
    monkeypatch.setattr(
        be, "_kubectl",
        _fake_kubectl("HCA mlx5_0\nHCA mlx5_1\nHCA mlx5_2\nIFACE eth0\n"),
    )
    asyncio.run(be._detect_network_env(["res-0"]))
    assert be.rdma_enabled is True
    assert be.rdma_host_device_paths == ["/dev/infiniband"]
    assert be.rdma_ipc_lock is True
    # Every NIC spec is resource-less (access via the device mount, not a resource).
    assert [r for r, _h in be.rdma_nic_specs] == ["", "", ""]
    env = be.network_env
    assert env["NCCL_SOCKET_IFNAME"] == "eth0"  # control NIC pinned
    assert env["SFLOW_RDMA_HCAS"] == "mlx5_0,mlx5_1,mlx5_2"
    # UCX_NET_DEVICES is set per-pod by the operator, not backend-wide.
    assert "UCX_NET_DEVICES" not in env


def test_detect_network_env_pins_routable_iface_when_no_host_network(monkeypatch):
    # Without host_network the host-device path does not apply; fall back to pinning
    # the routable TCP NIC (and only expose the HCAs informationally).
    be = KubernetesBackend(_cfg(host_network=False))
    monkeypatch.setattr(
        be, "_kubectl", _fake_kubectl("HCA mlx5_0\nHCA mlx5_1\nIFACE eth0\n")
    )
    asyncio.run(be._detect_network_env(["res-0"]))
    assert be.rdma_enabled is False
    env = be.network_env
    assert "UCX_NET_DEVICES" not in env
    assert env["NCCL_SOCKET_IFNAME"] == "eth0"
    assert env["SFLOW_RDMA_HCAS"] == "mlx5_0,mlx5_1"  # informational only
    assert "NCCL_IB_HCA" not in env


def test_build_time_tcp_fallback_warns_when_hcas_present(monkeypatch, caplog):
    # HCAs exist on the node but no usable provider (host_network off) -> the
    # build-time plan degrades to TCP; sflow must WARN so slow KV transport is not
    # silent.
    be = KubernetesBackend(_cfg(host_network=False))
    monkeypatch.setattr(
        be, "_kubectl", _fake_kubectl("HCA mlx5_0\nHCA mlx5_1\nIFACE eth0\n")
    )
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="sflow.plugins.backends.kubernetes"):
        asyncio.run(be._detect_network_env(["res-0"]))
    assert be.rdma_enabled is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING when RDMA HCAs degrade to TCP"
    msg = warnings[0].getMessage()
    assert "TCP" in msg and "mlx5_0" in msg


def test_build_time_no_hcas_does_not_warn(monkeypatch, caplog):
    # No RDMA hardware at all -> TCP is expected, not a degradation: no WARNING.
    be = KubernetesBackend(_cfg())

    async def fake_kubectl(args):
        return 0, "IFACE eth0\n", ""

    monkeypatch.setattr(be, "_kubectl", fake_kubectl)
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="sflow.plugins.backends.kubernetes"):
        asyncio.run(be._detect_network_env(["res-0"]))
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_detect_network_env_gke_provider(monkeypatch):
    be = KubernetesBackend(_cfg())
    node_json = json.dumps(
        {
            "status": {
                "allocatable": {
                    "networking.gke.io.networks/rdma-0": "1",
                    "networking.gke.io.networks/rdma-1": "1",
                }
            }
        }
    )
    monkeypatch.setattr(
        be, "_kubectl",
        _fake_kubectl(
            "HCA mlx5_0\nHCA mlx5_1\nIFACE eth0\n", node_json=node_json,
            installer_pods="nccl-rdma-installer-abcde",  # gIB installer on node
        ),
    )
    asyncio.run(be._detect_network_env(["res-0"]))
    assert be.rdma_enabled is True
    assert be.rdma_nic_specs == [
        ("networking.gke.io.networks/rdma-0", "mlx5_0"),
        ("networking.gke.io.networks/rdma-1", "mlx5_1"),
    ]
    assert be.rdma_host_device_paths == []  # device plugin, no host mount
    assert be.rdma_lib_mounts  # GKE gIB libs mounted (installer present)


def _gke_node_json():
    return json.dumps(
        {"status": {"allocatable": {"networking.gke.io.networks/rdma-0": "1"}}}
    )


def test_gke_rdma_warns_when_gib_installer_absent(monkeypatch, caplog):
    # GKE RDMA active but no `nccl-rdma-installer` DaemonSet -> warn that multi-node
    # NCCL will use the untuned built-in IB transport (gIB tuning is absent).
    be = KubernetesBackend(_cfg())
    monkeypatch.setattr(
        be, "_kubectl",
        _fake_kubectl(
            "HCA mlx5_0\nIFACE eth0\n", node_json=_gke_node_json(),
            installer_pods="anetd-xxxxx fluentbit-gke-yyyyy",  # no installer on node
        ),
    )
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="sflow.plugins.backends.kubernetes"):
        asyncio.run(be._detect_network_env(["res-0"]))
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("nccl-rdma-installer" in m and "gIB" in m for m in msgs)
    # Installer absent -> the gIB lib mounts are NOT emitted (never mask the driver
    # path /usr/local/nvidia with an empty dir).
    assert be.rdma_lib_mounts == []
    assert be.rdma_nccl_env_script == ""
    assert be.rdma_enabled is True  # RDMA NICs still granted (NCCL built-in IB)


def test_gke_rdma_no_warn_when_gib_installer_present(monkeypatch, caplog):
    # Installer present -> no gIB warning.
    be = KubernetesBackend(_cfg())
    monkeypatch.setattr(
        be, "_kubectl",
        _fake_kubectl(
            "HCA mlx5_0\nIFACE eth0\n", node_json=_gke_node_json(),
            installer_pods="nvidia-gpu-device-plugin-zzzzz nccl-rdma-installer-abcde",
        ),
    )
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="sflow.plugins.backends.kubernetes"):
        asyncio.run(be._detect_network_env(["res-0"]))
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("nccl-rdma-installer" in m for m in msgs)
    # Installer present -> gIB libs mounted + NCCL tuning wired.
    assert be.rdma_lib_mounts
    assert be.rdma_nccl_env_script


def test_gib_installer_probe_returns_false_on_kubectl_error(monkeypatch):
    # The gIB probe is best-effort: any exec failure must return False so sflow
    # omits the lib mounts (never risk masking the driver path) rather than raise.
    be = KubernetesBackend(_cfg())

    async def _boom(_args):
        raise RuntimeError("kubectl exploded")

    monkeypatch.setattr(be, "_kubectl", _boom)
    assert asyncio.run(be._gib_installer_present()) is False


def test_gib_installer_probe_returns_false_on_nonzero_rc(monkeypatch):
    # A non-zero kubectl exit (e.g. RBAC denies listing DaemonSets) also yields
    # False -- absence of proof is treated as "not installed".
    be = KubernetesBackend(_cfg())

    async def _denied(_args):
        return 1, "", "forbidden"

    monkeypatch.setattr(be, "_kubectl", _denied)
    assert asyncio.run(be._gib_installer_present()) is False


def test_gib_installer_probe_scopes_to_scheduling_node(monkeypatch):
    # The probe must ask only about the specific scheduling node (per-node check),
    # so a partial rollout that has not reached this node reads as "not installed"
    # instead of a cluster-wide false positive.
    seen: dict[str, list[str]] = {}

    async def _capture(args):
        a = list(args)
        if a[:2] == ["get", "pods"]:
            seen["args"] = a
        return 0, "nccl-rdma-installer-abcde", ""

    be = KubernetesBackend(_cfg())
    monkeypatch.setattr(be, "_kubectl", _capture)
    assert asyncio.run(be._gib_installer_present("node-7")) is True
    joined = " ".join(seen["args"])
    assert "--field-selector" in seen["args"]
    assert "spec.nodeName=node-7" in joined
    assert "status.phase=Running" in joined


def test_detect_network_env_shared_device_plugin_provider(monkeypatch):
    be = KubernetesBackend(_cfg())
    node_json = json.dumps({"status": {"allocatable": {"rdma/hca": "2"}}})
    monkeypatch.setattr(
        be, "_kubectl",
        _fake_kubectl("HCA mlx5_0\nHCA mlx5_1\nIFACE eth0\n", node_json=node_json),
    )
    asyncio.run(be._detect_network_env(["res-0"]))
    assert be.rdma_enabled is True
    # Shared resource -> all HCAs behind the one rdma/* resource.
    assert be.rdma_nic_specs == [("rdma/hca", "mlx5_0"), ("rdma/hca", "mlx5_1")]
    assert be.rdma_ipc_lock is True
    assert be.rdma_host_device_paths == []


def test_detect_network_env_pins_iface_even_without_rdma(monkeypatch):
    be = KubernetesBackend(_cfg())

    async def fake_kubectl(args):
        return 0, "IFACE eth0\n", ""  # routable NIC, no RDMA HCAs

    monkeypatch.setattr(be, "_kubectl", fake_kubectl)
    asyncio.run(be._detect_network_env(["res-0"]))
    env = be.network_env
    assert "UCX_NET_DEVICES" not in env
    assert env["NCCL_SOCKET_IFNAME"] == "eth0"
    assert "SFLOW_RDMA_HCAS" not in env


def test_detect_network_env_nothing_when_no_iface(monkeypatch):
    be = KubernetesBackend(_cfg())

    async def fake_kubectl(args):
        return 0, "", ""  # no default route, no HCAs

    monkeypatch.setattr(be, "_kubectl", fake_kubectl)
    asyncio.run(be._detect_network_env(["res-0"]))
    assert be.network_env == {}


def test_resolve_config_preserves_rdma():
    class _Id:
        def resolve(self, v, ctx):
            return v

    conf = _cfg(rdma="auto")
    resolved = KubernetesBackend.resolve_config(
        conf, resolver=_Id(), ctx={}, workflow_name="wf"
    )
    assert resolved.rdma == "auto"


def test_rdma_forces_named_provider():
    # A specific `rdma` value pins that provider (skips the auto chain order).
    be = KubernetesBackend(_cfg(rdma="host_device"))
    assert be._rdma_mode == "auto" and be._rdma_forced == "host_device"


def test_rdma_off_disables_detection():
    be = KubernetesBackend(_cfg(rdma="off"))
    assert be._rdma_mode == "off" and be._rdma_forced is None
    assert be.network_env == {} and be.rdma_enabled is False


def test_resolve_config_preserves_and_resolves_pvc_volumes():
    # Regression: resolve_config() rebuilds the backend config field-by-field, so it
    # must carry `volumes` through (and resolve each entry's Resolvable fields),
    # otherwise the PVC is silently dropped and fs:// falls back to a hostPath.
    class _Resolver:
        def resolve(self, value, ctx):
            if isinstance(value, str):
                return value.replace("PVCNAME", "real-pvc")
            return value

    conf = _cfg(volumes=[
        {"name": "model-store", "claim": "PVCNAME", "mount_path": "/models",
         "read_only": True},
    ])
    resolved = KubernetesBackend.resolve_config(
        conf, resolver=_Resolver(), ctx={}, workflow_name="wf"
    )
    assert resolved.volumes is not None and len(resolved.volumes) == 1
    v = resolved.volumes[0]
    assert v.claim == "real-pvc"  # Resolvable field was resolved
    assert v.mount_path == "/models" and v.read_only is True
    # And the live backend object exposes it for the operator.
    assert KubernetesBackend(resolved).volumes[0]["claim"] == "real-pvc"


# ---------------------------------------------------------------------------
# NVLink-domain scope detection (component 2)
# ---------------------------------------------------------------------------


def _fake_kubectl_sync(*, product="", crd_present=False):
    """Fake ``_kubectl_sync`` answering the CRD presence + GPU-product queries."""

    def _run(args, *, timeout="10s"):
        a = list(args)
        if a[:2] == ["get", "crd"]:
            return (0, "customresourcedefinition.apiextensions.k8s.io/computedomains.resource.nvidia.com", "") if crd_present else (1, "", "NotFound")
        if a[:2] == ["get", "nodes"]:
            return (0, product, "") if product else (0, "", "")
        return 0, "", ""

    return _run


def test_nvlink_scope_from_pure_helper():
    be = KubernetesBackend(_cfg())
    # GB200/GB300-class board + ComputeDomain CRD -> rack-scoped MNNVL.
    assert be._nvlink_scope_from(product="NVIDIA-GB200", compute_domain_crd=True) == "rack"
    assert be._nvlink_scope_from(product="NVIDIA GB300", compute_domain_crd=True) == "rack"
    # GB200 board but no IMEX driver/CRD -> only intra-node NVLink.
    assert be._nvlink_scope_from(product="NVIDIA-GB200", compute_domain_crd=False) == "node"
    # NVSwitch/NVLink node GPUs (B200/H100) -> node scope.
    assert be._nvlink_scope_from(product="NVIDIA-B200", compute_domain_crd=False) == "node"
    assert be._nvlink_scope_from(product="NVIDIA-H100-80GB-HBM3", compute_domain_crd=True) == "node"
    # Unknown/absent product -> off (no NVLink assumption).
    assert be._nvlink_scope_from(product="", compute_domain_crd=True) == "off"
    assert be._nvlink_scope_from(product="Tesla-T4", compute_domain_crd=False) == "off"


def test_detect_nvlink_scope_explicit_override_wins_without_kubectl():
    for override in ("node", "rack", "off"):
        be = KubernetesBackend(_cfg(nvlink_domain=override))

        def _boom(*a, **k):
            raise AssertionError("detection must not run when overridden")

        be._kubectl_sync = _boom  # type: ignore[assignment]
        assert be._detect_nvlink_scope() == override
        assert be.nvlink_domain_scope == override


def test_detect_nvlink_scope_rack_from_product_and_crd(monkeypatch):
    be = KubernetesBackend(_cfg())  # nvlink_domain defaults to auto
    monkeypatch.setattr(
        be, "_kubectl_sync", _fake_kubectl_sync(product="NVIDIA-GB200", crd_present=True)
    )
    assert be._detect_nvlink_scope() == "rack"
    assert be.nvlink_domain_scope == "rack"


def test_detect_nvlink_scope_node_when_crd_absent(monkeypatch):
    be = KubernetesBackend(_cfg())
    monkeypatch.setattr(
        be, "_kubectl_sync", _fake_kubectl_sync(product="NVIDIA-GB200", crd_present=False)
    )
    assert be._detect_nvlink_scope() == "node"


def test_detect_nvlink_scope_off_when_undetectable(monkeypatch):
    be = KubernetesBackend(_cfg())
    monkeypatch.setattr(be, "_kubectl_sync", _fake_kubectl_sync(product="", crd_present=False))
    assert be._detect_nvlink_scope() == "off"


def test_detect_nvlink_scope_warn_only_on_kubectl_error(monkeypatch):
    be = KubernetesBackend(_cfg())

    def _err(args, *, timeout="10s"):
        raise RuntimeError("kubectl blew up")

    monkeypatch.setattr(be, "_kubectl_sync", _err)
    # Never raises; degrades to off.
    assert be._detect_nvlink_scope() == "off"


def test_detect_nvlink_scope_logs_resolved_scope(monkeypatch):
    be = KubernetesBackend(_cfg())
    monkeypatch.setattr(
        be, "_kubectl_sync", _fake_kubectl_sync(product="NVIDIA-GB200", crd_present=True)
    )
    messages, handler = _capture_info()
    k8s_mod._logger.addHandler(handler)
    k8s_mod._logger.setLevel(logging.INFO)
    try:
        be._detect_nvlink_scope()
    finally:
        k8s_mod._logger.removeHandler(handler)
    assert any("NVLink" in m and "rack" in m for m in messages)


def test_nvlink_domain_scope_none_before_detection():
    # auto + no detection yet (e.g. dry-run) -> unresolved (None).
    assert KubernetesBackend(_cfg()).nvlink_domain_scope is None


# ---------------------------------------------------------------------------
# ComputeDomain detection + `auto` channel resolution (component 4)
# ---------------------------------------------------------------------------


def _cds_json(*pairs):
    """Build a `kubectl get computedomains -o json` payload from (name, channel)."""
    items = [
        {
            "metadata": {"name": name},
            "spec": {"channel": {"resourceClaimTemplate": {"name": channel}}},
        }
        for name, channel in pairs
    ]
    return json.dumps({"items": items})


def test_detect_compute_domains_parses_json(monkeypatch):
    be = KubernetesBackend(_cfg(namespace="ml"))

    def fake_sync(args, *, timeout="10s"):
        if args[:2] == ["get", "computedomains"]:
            return 0, _cds_json(("cd-a", "cd-a-channel"), ("cd-b", "cd-b-channel")), ""
        return 1, "", ""

    monkeypatch.setattr(be, "_kubectl_sync", fake_sync)
    assert be._detect_compute_domains() == [
        ("cd-a", "cd-a-channel"),
        ("cd-b", "cd-b-channel"),
    ]


def test_detect_compute_domains_empty_on_error(monkeypatch):
    be = KubernetesBackend(_cfg())
    monkeypatch.setattr(be, "_kubectl_sync", lambda args, *, timeout="10s": (1, "", "boom"))
    assert be._detect_compute_domains() == []


def test_resolve_auto_channel_uses_sole_domain(monkeypatch):
    be = KubernetesBackend(
        _cfg(dra=KubernetesDraConfig(use_compute_domain_channel="auto"))
    )
    monkeypatch.setattr(
        be, "_detect_compute_domains", lambda: [("cd-only", "cd-only-channel")]
    )
    be._resolve_use_compute_domain_channel()
    assert be.compute_domain_channel == "cd-only-channel"


def test_resolve_auto_channel_zero_domains_hints(monkeypatch, caplog):
    be = KubernetesBackend(
        _cfg(dra=KubernetesDraConfig(use_compute_domain_channel="auto"))
    )
    monkeypatch.setattr(be, "_detect_compute_domains", lambda: [])
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="sflow.plugins.backends.kubernetes"):
        be._resolve_use_compute_domain_channel()
    assert be.compute_domain_channel is None
    assert any("no existing ComputeDomain" in r.getMessage() for r in caplog.records)


def test_resolve_auto_channel_many_domains_hints(monkeypatch, caplog):
    be = KubernetesBackend(
        _cfg(dra=KubernetesDraConfig(use_compute_domain_channel="auto"))
    )
    monkeypatch.setattr(
        be,
        "_detect_compute_domains",
        lambda: [("cd-a", "cd-a-channel"), ("cd-b", "cd-b-channel")],
    )
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="sflow.plugins.backends.kubernetes"):
        be._resolve_use_compute_domain_channel()
    # Ambiguous -> never guess a domain.
    assert be.compute_domain_channel is None
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("cd-a" in m and "cd-b" in m for m in msgs)


def test_resolve_named_channel_is_noop(monkeypatch):
    be = KubernetesBackend(
        _cfg(dra=KubernetesDraConfig(use_compute_domain_channel="cd-named"))
    )

    def _boom():
        raise AssertionError("must not detect for a named channel")

    monkeypatch.setattr(be, "_detect_compute_domains", _boom)
    be._resolve_use_compute_domain_channel()
    assert be.compute_domain_channel == "cd-named"


def test_resolve_off_channel_is_noop(monkeypatch):
    be = KubernetesBackend(_cfg())

    def _boom():
        raise AssertionError("must not detect when channel is off")

    monkeypatch.setattr(be, "_detect_compute_domains", _boom)
    be._resolve_use_compute_domain_channel()
    assert be.compute_domain_channel is None


def test_resolve_auto_channel_best_effort_on_error(monkeypatch):
    be = KubernetesBackend(
        _cfg(dra=KubernetesDraConfig(use_compute_domain_channel="auto"))
    )

    def _err():
        raise RuntimeError("kubectl blew up")

    monkeypatch.setattr(be, "_detect_compute_domains", _err)
    be._resolve_use_compute_domain_channel()  # must not raise
    assert be.compute_domain_channel is None


# ---------------------------------------------------------------------------
# NVLink-domain placement (component 6): reservation podAffinity + validation
# ---------------------------------------------------------------------------


def _capture_reservation_render(monkeypatch, backend):
    captured: dict = {}

    def fake_render(**kwargs):
        captured.update(kwargs)
        return {"kind": "Pod", "metadata": {"name": kwargs["pod_name"]}, "spec": {}}

    monkeypatch.setattr(k8s_mod, "render_reservation_pod_manifest", fake_render)

    async def fake_alloc(alloc_id, pod_names, manifests):
        return Allocation(allocation_id=alloc_id, nodes=[], owned=True)

    async def fake_detect(pod_names):
        return None

    monkeypatch.setattr(backend, "_allocate_reserved", fake_alloc)
    monkeypatch.setattr(backend, "_detect_network_env", fake_detect)
    return captured


def test_reservation_gets_nvlink_podaffinity_when_rack_multinode(monkeypatch):
    backend = KubernetesBackend(
        _cfg(
            nodes=2,
            gpus_per_node=0,
            nvlink_domain="rack",  # explicit override -> no detection kubectl
            dra=KubernetesDraConfig(nvlink_domain_label_key="nvidia.com/gpu.clique"),
        )
    )
    captured = _capture_reservation_render(monkeypatch, backend)
    asyncio.run(backend.allocate())
    assert captured.get("nvlink_domain_topology_key") == "nvidia.com/gpu.clique"


def test_reservation_no_nvlink_podaffinity_when_node_scope(monkeypatch):
    backend = KubernetesBackend(
        _cfg(
            nodes=2,
            gpus_per_node=0,
            nvlink_domain="node",
            dra=KubernetesDraConfig(nvlink_domain_label_key="nvidia.com/gpu.clique"),
        )
    )
    captured = _capture_reservation_render(monkeypatch, backend)
    asyncio.run(backend.allocate())
    assert captured.get("nvlink_domain_topology_key") is None


def test_reservation_no_nvlink_podaffinity_single_node(monkeypatch):
    backend = KubernetesBackend(
        _cfg(
            nodes=1,
            gpus_per_node=0,
            nvlink_domain="rack",
            dra=KubernetesDraConfig(nvlink_domain_label_key="nvidia.com/gpu.clique"),
        )
    )
    captured = _capture_reservation_render(monkeypatch, backend)
    asyncio.run(backend.allocate())
    assert captured.get("nvlink_domain_topology_key") is None


def test_reservation_no_nvlink_podaffinity_without_label_key(monkeypatch):
    backend = KubernetesBackend(_cfg(nodes=2, gpus_per_node=0, nvlink_domain="rack"))
    captured = _capture_reservation_render(monkeypatch, backend)
    asyncio.run(backend.allocate())
    assert captured.get("nvlink_domain_topology_key") is None


def test_node_label_reads_escaped_jsonpath(monkeypatch):
    be = KubernetesBackend(_cfg())
    seen: dict = {}

    async def fake_kubectl(args):
        seen["args"] = list(args)
        return 0, "clique-7", ""

    monkeypatch.setattr(be, "_kubectl", fake_kubectl)
    val = asyncio.run(be._node_label("node-a", "nvidia.com/gpu.clique"))
    assert val == "clique-7"
    assert any("nvidia\\.com/gpu\\.clique" in a for a in seen["args"])


def _rack_backend_with_label():
    return KubernetesBackend(
        _cfg(
            nvlink_domain="rack",
            dra=KubernetesDraConfig(nvlink_domain_label_key="nvidia.com/gpu.clique"),
        )
    )


def test_validate_nvlink_placement_warns_on_span(monkeypatch, caplog):
    be = _rack_backend_with_label()
    labels = {"node-a": "clique-1", "node-b": "clique-2"}

    async def fake_label(n, key, *, fallback=""):
        return labels[n]

    monkeypatch.setattr(be, "_node_label", fake_label)
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="sflow.plugins.backends.kubernetes"):
        asyncio.run(be._validate_nvlink_domain_placement(["node-a", "node-b"]))
    assert any(
        "domain" in r.getMessage().lower()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


def test_validate_nvlink_placement_warns_on_missing_label(monkeypatch, caplog):
    be = _rack_backend_with_label()

    async def fake_label(n, key, *, fallback=""):
        return {"node-a": "", "node-b": "clique-1"}[n]

    monkeypatch.setattr(be, "_node_label", fake_label)
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="sflow.plugins.backends.kubernetes"):
        asyncio.run(be._validate_nvlink_domain_placement(["node-a", "node-b"]))
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("label" in m.lower() for m in msgs)


def test_validate_nvlink_placement_ok_same_domain(monkeypatch, caplog):
    be = _rack_backend_with_label()

    async def fake_label(n, key, *, fallback=""):
        return "clique-1"

    monkeypatch.setattr(be, "_node_label", fake_label)
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="sflow.plugins.backends.kubernetes"):
        asyncio.run(be._validate_nvlink_domain_placement(["node-a", "node-b"]))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_validate_nvlink_placement_noop_when_node_scope(monkeypatch):
    be = KubernetesBackend(
        _cfg(
            nvlink_domain="node",
            dra=KubernetesDraConfig(nvlink_domain_label_key="nvidia.com/gpu.clique"),
        )
    )

    def _boom(*a, **k):
        raise AssertionError("must not read node labels on node scope")

    monkeypatch.setattr(be, "_node_label", _boom)
    asyncio.run(be._validate_nvlink_domain_placement(["node-a", "node-b"]))  # no-op


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_capabilities_node_placement_and_no_gpu_sharing():
    backend = KubernetesBackend(_cfg())
    caps = backend.capabilities
    assert caps.supports_node_placement is True
    assert caps.has_runtime_node_addresses is True
    assert caps.supports_gpu_env is False
    # GPUs are hard-exclusive under DRA/device-plugin.
    assert caps.supports_gpu_sharing is False
    # Node-level hardware monitoring is not implemented for k8s yet (would need a
    # DCGM/DaemonSet collector), so the planner skips it.
    assert caps.supports_host_monitoring is False


def test_scheduling_and_dra_defaults():
    backend = KubernetesBackend(_cfg())
    assert backend.scheduling == "dra"
    assert backend.gpu_device_class == "gpu.nvidia.com"
    # Default tolerations let pods land on tainted GPU nodes.
    assert backend.tolerations == [
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    ]


def test_dra_config_surfaced():
    backend = KubernetesBackend(
        _cfg(dra=KubernetesDraConfig(gpu_device_class="mig.nvidia.com"))
    )
    assert backend.gpu_device_class == "mig.nvidia.com"


# ---------------------------------------------------------------------------
# default_operator / monitor_operator
# ---------------------------------------------------------------------------


def test_default_operator_raises_for_missing_operator():
    backend = KubernetesBackend(_cfg())
    with pytest.raises(ValueError, match="explicit 'k8s' operator"):
        backend.default_operator(name="task", assigned_nodes=["gpu-node-a"])


def test_monitor_operator_runs_on_driver_via_bash():
    backend = KubernetesBackend(_cfg())
    operator = backend.monitor_operator(name="mon", assigned_nodes=["gpu-node-a"])
    assert isinstance(operator, BashOperator)


# ---------------------------------------------------------------------------
# preflight_validate
# ---------------------------------------------------------------------------


def _capture_warnings():
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord):
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    return messages, _Capture()


def test_preflight_raises_when_kubectl_missing():
    backend = KubernetesBackend(_cfg())
    with mock.patch("shutil.which", return_value=None):
        with pytest.raises(ValueError, match="kubectl"):
            backend.preflight_validate()


def test_preflight_warns_when_multi_node_without_host_network():
    backend = KubernetesBackend(_cfg(nodes=2, host_network=False))
    messages, handler = _capture_warnings()
    k8s_mod._logger.addHandler(handler)
    try:
        with (
            mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
            mock.patch.object(backend, "_preflight_check_connectivity"),
            mock.patch.object(backend, "_preflight_check_dra"),
        ):
            backend.preflight_validate()
    finally:
        k8s_mod._logger.removeHandler(handler)
    assert any("host_network" in m for m in messages)


def test_preflight_dra_warns_when_deviceclass_missing():
    backend = KubernetesBackend(_cfg(scheduling="dra"))
    messages, handler = _capture_warnings()
    k8s_mod._logger.addHandler(handler)
    fake = mock.Mock(returncode=1, stdout="", stderr="")
    try:
        with (
            mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
            mock.patch.object(backend, "_preflight_check_connectivity"),
            mock.patch("subprocess.run", return_value=fake),
        ):
            backend.preflight_validate()
    finally:
        k8s_mod._logger.removeHandler(handler)
    assert any("DeviceClass" in m for m in messages)


def test_preflight_device_plugin_skips_deviceclass_check():
    backend = KubernetesBackend(_cfg(scheduling="device_plugin"))
    with (
        mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
        mock.patch.object(backend, "_preflight_check_connectivity"),
        mock.patch.object(backend, "_detect_nvlink_scope"),
        mock.patch("subprocess.run") as run,
    ):
        backend.preflight_validate()
    run.assert_not_called()


# ---------------------------------------------------------------------------
# preflight: connectivity + RBAC
# ---------------------------------------------------------------------------


def _kubectl_run(answer):
    """Return a subprocess.run side_effect from answer(argv) -> (rc, out, err)."""

    def _run(argv, *a, **k):
        rc, out, err = answer(argv)
        return mock.Mock(returncode=rc, stdout=out, stderr=err)

    return _run


def _can_i_verb_resource(argv: list[str]) -> tuple[str, str]:
    i = argv.index("can-i")
    return argv[i + 1], argv[i + 2]


def test_preflight_connectivity_passes_when_authorized():
    backend = KubernetesBackend(_cfg(namespace="ml", scheduling="device_plugin"))

    def answer(argv):
        if "namespace" in argv and "get" in argv:
            return 0, "namespace/ml", ""
        if "can-i" in argv:
            return 0, "yes", ""
        return 0, "", ""

    with (
        mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
        mock.patch("subprocess.run", side_effect=_kubectl_run(answer)),
    ):
        backend.preflight_validate()  # must not raise


def test_preflight_connectivity_fails_when_unreachable():
    backend = KubernetesBackend(_cfg(namespace="ml"))

    def answer(argv):
        return 1, "", "Unable to connect to the server: dial tcp ... i/o timeout"

    with (
        mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
        mock.patch("subprocess.run", side_effect=_kubectl_run(answer)),
    ):
        with pytest.raises(ValueError, match="Cannot reach or authenticate"):
            backend.preflight_validate()


def test_preflight_connectivity_fails_when_namespace_missing():
    backend = KubernetesBackend(_cfg(namespace="ml"))

    def answer(argv):
        if "namespace" in argv:
            return 1, "", 'Error from server (NotFound): namespaces "ml" not found'
        return 0, "yes", ""

    with (
        mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
        mock.patch("subprocess.run", side_effect=_kubectl_run(answer)),
    ):
        with pytest.raises(ValueError, match="was not found"):
            backend.preflight_validate()


def test_preflight_connectivity_fails_on_missing_rbac():
    backend = KubernetesBackend(_cfg(namespace="ml", scheduling="device_plugin"))

    def answer(argv):
        if "namespace" in argv and "get" in argv:
            return 0, "namespace/ml", ""
        if "can-i" in argv:
            verb, resource = _can_i_verb_resource(argv)
            if verb == "create" and resource == "secrets":
                return 1, "no", ""
            return 0, "yes", ""
        return 0, "", ""

    with (
        mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
        mock.patch("subprocess.run", side_effect=_kubectl_run(answer)),
    ):
        with pytest.raises(ValueError, match="create secrets"):
            backend.preflight_validate()


def test_preflight_connectivity_checks_dra_resource_permissions():
    backend = KubernetesBackend(_cfg(namespace="ml", scheduling="dra"))

    def answer(argv):
        if "namespace" in argv and "get" in argv:
            return 0, "namespace/ml", ""
        if "can-i" in argv:
            verb, resource = _can_i_verb_resource(argv)
            if resource == "resourceclaimtemplates.resource.k8s.io":
                return 1, "no", ""
            return 0, "yes", ""
        return 0, "", ""

    with (
        mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
        mock.patch("subprocess.run", side_effect=_kubectl_run(answer)),
    ):
        with pytest.raises(ValueError, match="resourceclaimtemplates"):
            backend.preflight_validate()


def test_preflight_connectivity_skipped_via_env(monkeypatch):
    monkeypatch.setenv("SFLOW_SKIP_K8S_PREFLIGHT", "1")
    backend = KubernetesBackend(_cfg(namespace="ml", scheduling="device_plugin"))
    with (
        mock.patch("shutil.which", return_value="/usr/bin/kubectl"),
        mock.patch.object(backend, "_detect_nvlink_scope"),
        mock.patch("subprocess.run") as run,
    ):
        backend.preflight_validate()
    run.assert_not_called()


# ---------------------------------------------------------------------------
# emergency_release
# ---------------------------------------------------------------------------


def test_emergency_release_uses_pending_alloc_id():
    backend = KubernetesBackend(_cfg())
    backend._pending_alloc_id = "pending-abc"
    with mock.patch("subprocess.run") as run:
        backend.emergency_release(Allocation(allocation_id=None, nodes=[], owned=False))
    calls = [str(c) for c in run.call_args_list]
    assert any("pending-abc" in c for c in calls)
    assert backend._pending_alloc_id is None


def test_emergency_release_is_noop_when_no_id():
    backend = KubernetesBackend(_cfg())
    with mock.patch("subprocess.run") as run:
        backend.emergency_release(Allocation(allocation_id=None, nodes=[], owned=False))
    run.assert_not_called()


# ---------------------------------------------------------------------------
# cleanup sweep: configmaps + secrets
# ---------------------------------------------------------------------------


def test_release_alloc_deletes_configmaps_and_secrets():
    backend = KubernetesBackend(_cfg(namespace="ml"))
    kinds: list[str] = []

    async def fake_kubectl(args):
        if "delete" in args:
            kinds.append(args[args.index("delete") + 1])
        return 0, "", ""

    with mock.patch.object(backend, "_kubectl", side_effect=fake_kubectl):
        asyncio.run(backend.release(Allocation(allocation_id="abc", nodes=[], owned=True)))

    assert "pod" in kinds
    assert "configmap" in kinds
    assert "secret" in kinds


def test_emergency_release_deletes_configmaps_and_secrets_sync():
    backend = KubernetesBackend(_cfg(namespace="ml"))
    backend._pending_alloc_id = "abc"
    with mock.patch("subprocess.run") as run:
        backend.emergency_release(Allocation(allocation_id=None, nodes=[], owned=False))
    argvs = [c.args[0] for c in run.call_args_list]
    kinds = [argv[argv.index("delete") + 1] for argv in argvs if "delete" in argv]
    assert "configmap" in kinds and "secret" in kinds


# ---------------------------------------------------------------------------
# dry_run_details / config
# ---------------------------------------------------------------------------


def test_dry_run_details_dra():
    backend = KubernetesBackend(
        _cfg(namespace="bench", image_pull_policy="IfNotPresent", nodes=3, gpus_per_node=4)
    )
    details = dict(backend.dry_run_details())
    assert "image" not in details
    assert details["namespace"] == "bench"
    assert details["nodes"] == "3"
    assert details["gpus_per_node"] == "4"
    assert details["scheduling"] == "dra"
    assert details["gpu_device_class"] == "gpu.nvidia.com"


def test_dry_run_details_device_plugin_omits_device_class():
    backend = KubernetesBackend(_cfg(scheduling="device_plugin", gpus_per_node=4))
    details = dict(backend.dry_run_details())
    assert details["scheduling"] == "device_plugin"
    assert "gpu_device_class" not in details


# ---------------------------------------------------------------------------
# CLI kube access (KubectlConfig)
# ---------------------------------------------------------------------------


def test_apply_kubectl_config_global_args_and_namespace_override():
    from sflow.core.kubectl_config import KubectlConfig

    backend = KubernetesBackend(_cfg(namespace="default"))
    backend.apply_kubectl_config(
        KubectlConfig(
            kubeconfig="/k/cfg",
            context="ctx",
            namespace="override-ns",
            extra_args=["--insecure-skip-tls-verify"],
        )
    )
    assert backend.namespace == "override-ns"
    assert backend.kubectl_global_args == [
        "--kubeconfig",
        "/k/cfg",
        "--context",
        "ctx",
        "--insecure-skip-tls-verify",
    ]
    details = dict(backend.dry_run_details())
    assert details["kubeconfig"] == "/k/cfg"
    assert details["context"] == "ctx"
    assert details["kubectl_args"] == "['--insecure-skip-tls-verify']"


def test_backend_reads_node_filters_from_config():
    backend = KubernetesBackend(
        _cfg(include_nodes=["want-1"], exclude_nodes=["bad-1", "bad-2"])
    )
    assert backend.include_nodes == ["want-1"]
    assert backend.exclude_nodes == ["bad-1", "bad-2"]


def test_backend_normalizes_comma_joined_node_filters_from_config():
    backend = KubernetesBackend(_cfg(exclude_nodes=["bad-1,bad-2", "bad-3"]))
    assert backend.exclude_nodes == ["bad-1", "bad-2", "bad-3"]


def test_apply_kubectl_config_warns_on_generic_args_applied_as_kubectl_globals():
    # A generic --extra-args value (Slurm-ism) fanned into kubectl globals must warn,
    # since kubectl rejects unknown global flags and every call would fail.
    from sflow.core.kubectl_config import KubectlConfig

    backend = KubernetesBackend(_cfg(namespace="default"))
    messages, handler = _capture_warnings()
    k8s_mod._logger.addHandler(handler)
    try:
        backend.apply_kubectl_config(
            KubectlConfig(
                extra_args=["--gpus-per-node=4"],
                generic_extra_args=["--gpus-per-node=4"],
            )
        )
    finally:
        k8s_mod._logger.removeHandler(handler)
    assert any("--gpus-per-node=4" in m and "kubectl" in m for m in messages)


def test_apply_kubectl_config_no_warning_for_explicit_kubectl_args():
    # An explicit --extra-kubectl-args value (no generic origin) must NOT warn.
    from sflow.core.kubectl_config import KubectlConfig

    backend = KubernetesBackend(_cfg(namespace="default"))
    messages, handler = _capture_warnings()
    k8s_mod._logger.addHandler(handler)
    try:
        backend.apply_kubectl_config(
            KubectlConfig(extra_args=["--insecure-skip-tls-verify"])
        )
    finally:
        k8s_mod._logger.removeHandler(handler)
    assert not any("global flags" in m for m in messages)


def test_node_filters_threaded_into_reservation_manifests(monkeypatch):
    # Regression guard: include/exclude node filters must reach the reservation pod
    # render (the place that decides node placement), not just sit on the backend.
    backend = KubernetesBackend(
        _cfg(nodes=1, gpus_per_node=0, include_nodes=["want-1"], exclude_nodes=["bad-1"])
    )

    captured: dict = {}

    def fake_render(**kwargs):
        captured.update(kwargs)
        return {"kind": "Pod", "metadata": {"name": kwargs["pod_name"]}, "spec": {}}

    monkeypatch.setattr(k8s_mod, "render_reservation_pod_manifest", fake_render)

    async def fake_alloc(alloc_id, pod_names, manifests):
        return Allocation(allocation_id=alloc_id, nodes=[], owned=True)

    async def fake_detect(pod_names):
        return None

    monkeypatch.setattr(backend, "_allocate_reserved", fake_alloc)
    monkeypatch.setattr(backend, "_detect_network_env", fake_detect)
    monkeypatch.setattr(backend, "_detect_nvlink_scope", lambda: "off")
    asyncio.run(backend.allocate())
    assert captured.get("include_nodes") == ["want-1"]
    assert captured.get("exclude_nodes") == ["bad-1"]


def test_kubectl_global_args_prefixed_on_sync_kubectl_calls():
    from sflow.core.kubectl_config import KubectlConfig

    backend = KubernetesBackend(_cfg())
    backend.apply_kubectl_config(KubectlConfig(kubeconfig="/k/cfg", context="ctx"))
    backend._pending_alloc_id = "abc"
    with mock.patch("subprocess.run") as run:
        backend.emergency_release(Allocation(allocation_id=None, nodes=[], owned=False))
    argvs = [c.args[0] for c in run.call_args_list]
    assert argvs and all(
        argv[:5] == ["kubectl", "--kubeconfig", "/k/cfg", "--context", "ctx"]
        for argv in argvs
    )


def test_config_has_no_image_field():
    assert "image" not in KubernetesBackendConfig.model_fields
    conf = _cfg(image="nvcr.io/example/app:1.0")
    assert not hasattr(conf, "image")


# ---------------------------------------------------------------------------
# resolve_config
# ---------------------------------------------------------------------------


class _IdentityResolver:
    def resolve(self, value, ctx):
        return value


def test_resolve_config_preserves_namespace_scheduling_and_dra():
    conf = _cfg(
        namespace="bench",
        scheduling="dra",
        dra=KubernetesDraConfig(
            gpu_device_class="gpu.nvidia.com", create_compute_domain=True
        ),
        reservation=KubernetesReservationConfig(timeout=120),
    )
    resolved = KubernetesBackend.resolve_config(
        conf, resolver=_IdentityResolver(), ctx={}, workflow_name="wf"
    )
    assert resolved.namespace == "bench"
    assert resolved.scheduling == "dra"
    assert resolved.dra.gpu_device_class == "gpu.nvidia.com"
    assert resolved.dra.create_compute_domain is True
    assert resolved.reservation.timeout == 120
    assert not hasattr(resolved, "image")


def test_resolve_config_preserves_dra_rdma_coalloc():
    conf = _cfg(
        scheduling="dra",
        dra=KubernetesDraConfig(
            rdma_device_class="rdma.nvidia.com",
            rdma_match_attribute="resource.kubernetes.io/pcieRoot",
        ),
    )
    resolved = KubernetesBackend.resolve_config(
        conf, resolver=_IdentityResolver(), ctx={}, workflow_name="wf"
    )
    assert resolved.dra.rdma_device_class == "rdma.nvidia.com"
    assert resolved.dra.rdma_match_attribute == "resource.kubernetes.io/pcieRoot"


def test_backend_exposes_dra_rdma_properties_with_defaults():
    # Default: co-allocation off, and runtime affinity off until a provider enables it.
    plain = KubernetesBackend(_cfg(scheduling="dra"))
    assert plain.dra_rdma_device_class is None
    assert plain.dra_rdma_match_attribute == "resource.kubernetes.io/pcieRoot"
    assert plain.rdma_runtime_affinity is False

    configured = KubernetesBackend(
        _cfg(
            scheduling="dra",
            dra=KubernetesDraConfig(
                rdma_device_class="dra.net", rdma_match_attribute="dra.net/numaNode"
            ),
        )
    )
    assert configured.dra_rdma_device_class == "dra.net"
    assert configured.dra_rdma_match_attribute == "dra.net/numaNode"


def test_host_ipc_config_default_and_override():
    # Off by default (privileged); opt-in enables cross-pod CUDA IPC / NVLink.
    assert KubernetesBackend(_cfg()).host_ipc is False
    assert KubernetesBackend(_cfg(host_ipc=True)).host_ipc is True


def test_resolve_config_preserves_host_ipc():
    resolved = KubernetesBackend.resolve_config(
        _cfg(host_ipc=True), resolver=_IdentityResolver(), ctx={}, workflow_name="wf"
    )
    assert resolved.host_ipc is True
    assert KubernetesBackend(resolved).host_ipc is True


def test_merge_colocated_gpu_pods_tristate_resolves_to_bool():
    # Tri-state: default `auto` (and `on`/True) enable merging; `off`/False disable.
    # The backend property returns the RESOLVED bool so _plan_merge_groups is unchanged.
    assert KubernetesBackend(_cfg()).merge_colocated_gpu_pods is True  # default auto
    assert KubernetesBackend(_cfg(merge_colocated_gpu_pods="auto")).merge_colocated_gpu_pods is True
    assert KubernetesBackend(_cfg(merge_colocated_gpu_pods="on")).merge_colocated_gpu_pods is True
    assert KubernetesBackend(_cfg(merge_colocated_gpu_pods=True)).merge_colocated_gpu_pods is True
    assert KubernetesBackend(_cfg(merge_colocated_gpu_pods="off")).merge_colocated_gpu_pods is False
    assert KubernetesBackend(_cfg(merge_colocated_gpu_pods=False)).merge_colocated_gpu_pods is False


def test_merge_colocated_gpu_pods_invalid_rejected():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _cfg(merge_colocated_gpu_pods="sometimes")


def test_resolve_config_preserves_merge_colocated_gpu_pods():
    # `off` must survive resolve_config as `off` (NOT coerced to a truthy bool).
    resolved = KubernetesBackend.resolve_config(
        _cfg(merge_colocated_gpu_pods="off"),
        resolver=_IdentityResolver(),
        ctx={},
        workflow_name="wf",
    )
    assert KubernetesBackend(resolved).merge_colocated_gpu_pods is False
    resolved_on = KubernetesBackend.resolve_config(
        _cfg(merge_colocated_gpu_pods=True),
        resolver=_IdentityResolver(),
        ctx={},
        workflow_name="wf",
    )
    assert KubernetesBackend(resolved_on).merge_colocated_gpu_pods is True


def test_merge_colocated_gpu_pods_surfaced_in_dry_run_details():
    # Enabled (default auto) -> surfaced; explicitly off -> not surfaced.
    details = dict(KubernetesBackend(_cfg()).dry_run_details())
    assert details.get("merge_colocated_gpu_pods") == "True"
    assert "merge_colocated_gpu_pods" not in dict(
        KubernetesBackend(_cfg(merge_colocated_gpu_pods="off")).dry_run_details()
    )


# ---------------------------------------------------------------------------
# reservation: discovery + GPU-holding placeholders + handoff mapping
# ---------------------------------------------------------------------------


def _reserve(backend: KubernetesBackend) -> tuple[Allocation, list[dict]]:
    applied: list[dict] = []

    async def fake_apply(manifest):
        applied.append(manifest)

    scheduled = {0: ("gpu-node-a", "10.0.0.1"), 1: ("gpu-node-b", "10.0.0.2")}

    async def fake_wait(pod_name):
        return scheduled[int(pod_name.rsplit("-", 1)[1])]

    async def fake_internal(node_name, *, fallback=""):
        return {"gpu-node-a": "10.1.0.1", "gpu-node-b": "10.1.0.2"}[node_name]

    async def fake_detect(pod_names):
        return None

    async def fake_probe_pod(alloc_id):
        # The in-cluster probe pod has dedicated tests; keep reservation tests
        # focused on the placeholder/task pods they assert on.
        return None

    with (
        mock.patch.object(backend, "_apply_manifest", side_effect=fake_apply),
        mock.patch.object(backend, "_wait_for_pod_scheduled", side_effect=fake_wait),
        mock.patch.object(backend, "_node_internal_ip", side_effect=fake_internal),
        mock.patch.object(backend, "_detect_network_env", side_effect=fake_detect),
        mock.patch.object(backend, "_detect_nvlink_scope", return_value="off"),
        mock.patch.object(backend, "_create_probe_pod", side_effect=fake_probe_pod),
    ):
        allocation = asyncio.run(backend.allocate())
    return allocation, applied


def test_reserved_allocation_discovers_real_nodes_and_handoff_map():
    backend = KubernetesBackend(_cfg(nodes=2, gpus_per_node=8))
    allocation, _ = _reserve(backend)

    assert allocation.owned is True
    assert [n.name for n in allocation.nodes] == ["gpu-node-a", "gpu-node-b"]
    assert [n.ip_address for n in allocation.nodes] == ["10.1.0.1", "10.1.0.2"]
    assert [n.num_gpus for n in allocation.nodes] == [8, 8]
    # node -> placeholder pod mapping is recorded for the create-before-destroy handoff.
    assert backend.reservation_pod_for_node("gpu-node-a").endswith("-0")
    assert backend.reservation_pod_for_node("gpu-node-b").endswith("-1")


def test_dra_placeholders_hold_gpus_via_resource_claim_template():
    backend = KubernetesBackend(_cfg(nodes=2, gpus_per_node=8, scheduling="dra"))
    _, applied = _reserve(backend)

    rcts = [m for m in applied if m.get("kind") == "ResourceClaimTemplate"]
    assert len(rcts) == 1
    req = rcts[0]["spec"]["spec"]["devices"]["requests"][0]
    assert req["exactly"]["deviceClassName"] == "gpu.nvidia.com"
    assert req["exactly"]["count"] == 8
    # Each placeholder pod claims the GPUs (hard reservation) via the template.
    pods = [m for m in applied if m.get("kind") == "Pod"]
    assert pods and all(
        p["spec"]["resourceClaims"][0]["resourceClaimTemplateName"].endswith("-gpu")
        for p in pods
    )


def test_device_plugin_placeholders_hold_gpus_via_limit():
    backend = KubernetesBackend(
        _cfg(nodes=2, gpus_per_node=8, scheduling="device_plugin")
    )
    _, applied = _reserve(backend)

    assert not [m for m in applied if m.get("kind") == "ResourceClaimTemplate"]
    pods = [m for m in applied if m.get("kind") == "Pod"]
    assert pods and all(
        p["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "8"
        for p in pods
    )


def test_compute_domain_allocation_creates_compute_domain():
    backend = KubernetesBackend(
        _cfg(
            nodes=2,
            gpus_per_node=8,
            dra=KubernetesDraConfig(create_compute_domain=True),
        )
    )
    _, applied = _reserve(backend)

    cds = [m for m in applied if m.get("kind") == "ComputeDomain"]
    assert len(cds) == 1
    assert cds[0]["spec"]["numNodes"] == 2
    assert backend.compute_domain_channel is not None


def test_compute_domain_created_with_device_plugin_scheduling():
    # ComputeDomain (IMEX / NVLink) is decoupled from GPU scheduling: it stands up
    # even on device_plugin GPUs (ComputeDomain-only DRA-driver clusters), so
    # co-located pods can use cuda_ipc over the NVLink fabric.
    backend = KubernetesBackend(
        _cfg(
            nodes=2,
            gpus_per_node=8,
            scheduling="device_plugin",
            dra=KubernetesDraConfig(create_compute_domain=True),
        )
    )
    _, applied = _reserve(backend)
    cds = [m for m in applied if m.get("kind") == "ComputeDomain"]
    assert len(cds) == 1 and backend.compute_domain_channel is not None
    # RBAC must request ComputeDomain perms even without DRA GPU scheduling.
    perms = backend._required_permissions()
    assert ("create", "computedomains.resource.nvidia.com", True) in perms
    # ...but NOT the DRA GPU-allocation perms (GPUs stay on the device plugin).
    assert ("create", "resourceclaimtemplates.resource.k8s.io", True) not in perms


def test_use_compute_domain_channel_reuses_existing_without_creating():
    # Referencing an existing channel: NO ComputeDomain is created, no computedomains
    # RBAC required, and the channel is exposed for task pods to claim.
    backend = KubernetesBackend(
        _cfg(
            nodes=2,
            gpus_per_node=8,
            scheduling="device_plugin",
            dra=KubernetesDraConfig(use_compute_domain_channel="cd-existing"),
        )
    )
    assert backend.compute_domain_channel == "cd-existing"
    perms = backend._required_permissions()
    assert ("create", "computedomains.resource.nvidia.com", True) not in perms
    assert ("delete", "computedomains.resource.nvidia.com", True) not in perms
    _, applied = _reserve(backend)
    assert not [m for m in applied if m.get("kind") == "ComputeDomain"]  # none created


def test_use_compute_domain_channel_empty_is_off():
    backend = KubernetesBackend(
        _cfg(dra=KubernetesDraConfig(use_compute_domain_channel="  "))
    )
    assert backend.compute_domain_channel is None


def test_use_compute_domain_channel_auto_not_claimed_until_resolved():
    # `auto` is resolved to the single existing ComputeDomain at preflight/allocate
    # (component 4); before that resolution there is no channel to claim.
    backend = KubernetesBackend(
        _cfg(dra=KubernetesDraConfig(use_compute_domain_channel="auto"))
    )
    assert backend.compute_domain_channel is None
    # ...and `auto` never creates a domain (detection-only path).
    perms = backend._required_permissions()
    assert ("create", "computedomains.resource.nvidia.com", True) not in perms


def test_resolve_config_preserves_use_compute_domain_channel():
    conf = _cfg(dra=KubernetesDraConfig(use_compute_domain_channel="cd-x"))
    resolved = KubernetesBackend.resolve_config(
        conf, resolver=_IdentityResolver(), ctx={}, workflow_name="wf"
    )
    assert resolved.dra.use_compute_domain_channel == "cd-x"


# ---------------------------------------------------------------------------
# deprecated dra aliases: compute_domain -> create_compute_domain,
# compute_domain_channel -> use_compute_domain_channel
# ---------------------------------------------------------------------------


def test_deprecated_compute_domain_alias_still_accepted():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dra = KubernetesDraConfig(compute_domain=True)
    assert dra.create_compute_domain is True
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    # The old attribute name is gone (renamed field).
    assert not hasattr(dra, "compute_domain")


def test_deprecated_compute_domain_channel_alias_still_accepted():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dra = KubernetesDraConfig(compute_domain_channel="cd-legacy")
    assert dra.use_compute_domain_channel == "cd-legacy"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert KubernetesBackend(_cfg(dra=dra)).compute_domain_channel == "cd-legacy"


# ---------------------------------------------------------------------------
# nvlink_domain + nvlink_domain_label_key config
# ---------------------------------------------------------------------------


def test_nvlink_domain_default_is_auto():
    assert KubernetesBackend(_cfg()).nvlink_domain == "auto"


def test_nvlink_domain_override_accepted():
    for v in ("node", "rack", "off", "auto"):
        assert KubernetesBackend(_cfg(nvlink_domain=v)).nvlink_domain == v


def test_nvlink_domain_invalid_rejected():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _cfg(nvlink_domain="bogus")


def test_nvlink_domain_label_key_default_and_set():
    assert KubernetesBackend(_cfg()).nvlink_domain_label_key is None
    be = KubernetesBackend(
        _cfg(dra=KubernetesDraConfig(nvlink_domain_label_key="nvidia.com/gpu.clique"))
    )
    assert be.nvlink_domain_label_key == "nvidia.com/gpu.clique"


def test_resolve_config_preserves_nvlink_domain_and_label_key():
    conf = _cfg(
        nvlink_domain="rack",
        dra=KubernetesDraConfig(nvlink_domain_label_key="nvidia.com/gpu.clique"),
    )
    resolved = KubernetesBackend.resolve_config(
        conf, resolver=_IdentityResolver(), ctx={}, workflow_name="wf"
    )
    assert resolved.nvlink_domain == "rack"
    assert resolved.dra.nvlink_domain_label_key == "nvidia.com/gpu.clique"
    assert KubernetesBackend(resolved).nvlink_domain == "rack"


# ---------------------------------------------------------------------------
# reservation wait: diagnostic logging
# ---------------------------------------------------------------------------


def _capture_info():
    messages: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord):
            messages.append(record.getMessage())

    return messages, _Cap()


def test_wait_for_pod_scheduled_logs_reason_then_returns(monkeypatch):
    monkeypatch.setattr(k8s_mod, "_RESERVE_POLL_INTERVAL", 0)
    backend = KubernetesBackend(_cfg(namespace="ml"))

    phase_calls = {"n": 0}

    async def fake_kubectl(args):
        if "events" in args:
            return 0, "", ""
        if any("nodeName" in a for a in args):
            phase_calls["n"] += 1
            if phase_calls["n"] == 1:
                return 0, ",,Pending", ""  # not scheduled yet
            return 0, "gpu-node-a,10.0.0.5,Pending", ""  # nodeName set -> scheduled
        if any("waiting.reason" in a for a in args):
            return 0, "", ""  # no container waiting reason
        if any("PodScheduled" in a for a in args):
            return 0, "0/18 nodes available: device class gpu.nvidia.com does not exist.", ""
        return 0, "", ""

    messages, handler = _capture_info()
    k8s_mod._logger.addHandler(handler)
    k8s_mod._logger.setLevel(logging.INFO)
    try:
        with mock.patch.object(backend, "_kubectl", side_effect=fake_kubectl):
            node, ip = asyncio.run(backend._wait_for_pod_scheduled("sflow-res-k8s-abc-0"))
    finally:
        k8s_mod._logger.removeHandler(handler)

    assert (node, ip) == ("gpu-node-a", "10.0.0.5")
    assert any("waiting for reservation pod" in m for m in messages)
    assert any("does not exist" in m for m in messages)


def test_wait_for_pod_scheduled_timeout_includes_events(monkeypatch):
    monkeypatch.setattr(k8s_mod, "_RESERVE_POLL_INTERVAL", 0)
    backend = KubernetesBackend(
        _cfg(namespace="ml", reservation=KubernetesReservationConfig(timeout=2))
    )

    async def fake_kubectl(args):
        if "events" in args:
            return 0, "5m Warning FailedScheduling pod/x device class gpu.nvidia.com does not exist", ""
        if any("nodeName" in a for a in args):
            return 0, ",,Pending", ""
        return 0, "", ""

    with mock.patch.object(backend, "_kubectl", side_effect=fake_kubectl):
        with pytest.raises(RuntimeError) as ei:
            asyncio.run(backend._wait_for_pod_scheduled("podx"))

    assert "was not scheduled" in str(ei.value)
    assert "Recent events" in str(ei.value)
    assert "does not exist" in str(ei.value)


# ---------------------------------------------------------------------------
# allocate: release reservation pods when cancelled (Ctrl+C mid-allocation)
# ---------------------------------------------------------------------------


def test_allocate_releases_reservation_on_cancel():
    backend = KubernetesBackend(_cfg(nodes=1, gpus_per_node=0))
    released: list[str] = []

    async def fake_release(alloc_id):
        released.append(alloc_id)

    async def boom(*a, **k):
        raise asyncio.CancelledError()

    with (
        mock.patch.object(backend, "_allocate_reserved", side_effect=boom),
        mock.patch.object(backend, "_release_alloc", side_effect=fake_release),
        mock.patch.object(backend, "_detect_nvlink_scope", return_value="off"),
    ):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(backend.allocate())

    assert len(released) == 1
    assert backend._pending_alloc_id is None


# ---------------------------------------------------------------------------
# In-cluster probe pod
# ---------------------------------------------------------------------------


def test_probe_transport_enabled_returns_transport(monkeypatch):
    # The transport exists whenever in-cluster probing is enabled -- the pod is
    # created lazily on first use, not required up front.
    monkeypatch.delenv("SFLOW_K8S_PROBE_VIA_POD", raising=False)
    be = KubernetesBackend(_cfg())
    assert be.probe_transport() is not None


def test_probe_transport_disabled_by_env(monkeypatch):
    be = KubernetesBackend(_cfg())
    be._probe_pod_name = "sflow-probe-k8s-abc"
    monkeypatch.setenv("SFLOW_K8S_PROBE_VIA_POD", "0")
    assert be.probe_transport() is None


def test_create_probe_pod_applies_with_alloc_label(monkeypatch):
    monkeypatch.delenv("SFLOW_K8S_PROBE_VIA_POD", raising=False)
    be = KubernetesBackend(_cfg(namespace="ns"))
    be._pending_alloc_id = "abc"
    applied = []

    async def fake_apply(manifest):
        applied.append(manifest)

    monkeypatch.setattr(be, "_apply_manifest", fake_apply)
    asyncio.run(be._create_probe_pod())

    assert be._probe_pod_name == "sflow-probe-k8s-abc"
    assert applied
    labels = applied[0]["metadata"]["labels"]
    assert labels[k8s_mod.SFLOW_ALLOC_LABEL] == "abc"


def test_create_probe_pod_noop_without_alloc_id(monkeypatch):
    be = KubernetesBackend(_cfg())

    async def boom(manifest):
        raise AssertionError("should not apply without an allocation id")

    monkeypatch.setattr(be, "_apply_manifest", boom)
    asyncio.run(be._create_probe_pod())
    assert be._probe_pod_name is None


def test_create_probe_pod_apply_failure_is_nonfatal(monkeypatch):
    be = KubernetesBackend(_cfg())
    be._pending_alloc_id = "abc"

    async def boom(manifest):
        raise RuntimeError("apply failed")

    monkeypatch.setattr(be, "_apply_manifest", boom)
    asyncio.run(be._create_probe_pod())  # must not raise
    assert be._probe_pod_name is None


def test_kickoff_probe_pod_disabled_is_noop(monkeypatch):
    be = KubernetesBackend(_cfg())
    monkeypatch.setenv("SFLOW_K8S_PROBE_VIA_POD", "0")
    be._kickoff_probe_pod()
    assert be._probe_pod_task is None


def test_kickoff_probe_pod_is_idempotent(monkeypatch):
    monkeypatch.delenv("SFLOW_K8S_PROBE_VIA_POD", raising=False)
    be = KubernetesBackend(_cfg())
    created = []

    async def fake_create():
        created.append(1)

    monkeypatch.setattr(be, "_create_probe_pod", fake_create)

    async def run():
        be._kickoff_probe_pod()
        first = be._probe_pod_task
        be._kickoff_probe_pod()  # already in flight -> same task
        assert be._probe_pod_task is first
        await first

    asyncio.run(run())
    assert created == [1]


def test_exec_in_probe_pod_builds_kubectl_exec_args(monkeypatch):
    be = KubernetesBackend(_cfg(namespace="ns"))
    be._probe_pod_name = "pp"
    seen = {}

    async def fake_kubectl(args):
        seen["args"] = list(args)
        return 0, "ok", ""

    monkeypatch.setattr(be, "_kubectl", fake_kubectl)
    rc, out, _err = asyncio.run(be._exec_in_probe_pod(["curl", "http://x"]))
    assert rc == 0 and out == "ok"
    args = seen["args"]
    assert args[0] == "exec"
    assert "pp" in args
    assert args[-2:] == ["curl", "http://x"]
    assert "--namespace" in args and "ns" in args
    assert "-i" not in args  # no stdin -> no -i


def test_exec_in_probe_pod_without_pod_kicks_off_creation_and_reports_not_ready(
    monkeypatch,
):
    be = KubernetesBackend(_cfg())
    calls = []
    monkeypatch.setattr(be, "_kickoff_probe_pod", lambda: calls.append(1))
    rc, _out, _err = asyncio.run(be._exec_in_probe_pod(["curl"]))
    assert rc == 1
    assert calls == [1]  # first use triggers lazy creation
