# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cluster-portability overrides + gpus_per_node reconciliation on the k8s backend.

Covers the fixes that replace hardcoded NVIDIA/GKE conventions with overridable,
detected values: the device-plugin GPU resource name, the GPU-product label key, the
gIB installer namespace, the reservation placeholder image, the output-collection
grace window, the DRA API-version preflight, and the live gpus_per_node reconciliation
against the reserved node's real ``status.allocatable``.

Every new field defaults to the historical value, so these tests also guard against a
silent revert through ``resolve_config`` (which rebuilds the config field-by-field).
"""

import asyncio
import logging

import pytest

from sflow.plugins.backends.kubernetes import (
    KubernetesBackend,
    KubernetesBackendConfig,
    KubernetesReservationConfig,
)
from sflow.plugins.k8s.render import (
    RESERVATION_POD_IMAGE,
    render_reservation_pod_manifest,
    render_task_pod,
)


def _cfg(**kwargs) -> KubernetesBackendConfig:
    base = {"name": "k8s", "type": "kubernetes"}
    base.update(kwargs)
    return KubernetesBackendConfig(**base)


class _Id:
    """Identity resolver (mirrors the existing resolve_config regression tests)."""

    def resolve(self, value, ctx):
        return value


def _capture_sflow_warnings(monkeypatch, caplog):
    """Enable propagation of the (non-propagating) sflow logger so caplog sees it."""
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    return caplog.at_level(logging.WARNING, logger="sflow.plugins.backends.kubernetes")


# ---------------------------------------------------------------------------
# Defaults reproduce the historical conventions (no behavior change unless set)
# ---------------------------------------------------------------------------


def test_portability_defaults_match_conventions():
    be = KubernetesBackend(_cfg())
    assert be._gpu_resource_name == "nvidia.com/gpu"
    assert be._gpu_product_label_key == "nvidia.com/gpu.product"
    assert be._gib_installer_namespace == "kube-system"
    assert be._collect_grace_seconds == 120
    assert be._placeholder_image is None  # -> RESERVATION_POD_IMAGE at render
    # Properties the operator reads via apply_backend_context.
    assert be.gpu_resource_name == "nvidia.com/gpu"
    assert be.collect_grace_seconds == 120


def test_resolve_config_roundtrips_portability_overrides():
    # Regression: resolve_config() rebuilds the config field-by-field; a dropped field
    # silently reverts the override to its default.
    conf = _cfg(
        namespace="ns",
        gpu_resource_name="amd.com/gpu",
        gpu_product_label_key="custom.io/product",
        gib_installer_namespace="gpu-operator",
        collect_grace_seconds=300,
        reservation=KubernetesReservationConfig(placeholder_image="mirror/bash:5"),
    )
    resolved = KubernetesBackend.resolve_config(
        conf, resolver=_Id(), ctx={}, workflow_name="wf"
    )
    assert resolved.gpu_resource_name == "amd.com/gpu"
    assert resolved.gpu_product_label_key == "custom.io/product"
    assert resolved.gib_installer_namespace == "gpu-operator"
    assert resolved.collect_grace_seconds == 300
    assert resolved.reservation.placeholder_image == "mirror/bash:5"
    # And the live backend object exposes them.
    be = KubernetesBackend(resolved)
    assert be._gpu_resource_name == "amd.com/gpu"
    assert be._collect_grace_seconds == 300
    assert be._placeholder_image == "mirror/bash:5"


# ---------------------------------------------------------------------------
# Device-plugin GPU resource name (default nvidia.com/gpu) is overridable
# ---------------------------------------------------------------------------


def test_render_task_pod_uses_gpu_resource_name():
    pod = render_task_pod(
        pod_name="p",
        image="img",
        configmap_name="cm",
        scheduling="device_plugin",
        per_pod_gpus=2,
        gpu_resource_name="amd.com/gpu",
    )
    assert pod["spec"]["containers"][0]["resources"]["limits"] == {"amd.com/gpu": "2"}


def test_render_task_pod_defaults_to_nvidia_gpu():
    pod = render_task_pod(
        pod_name="p",
        image="img",
        configmap_name="cm",
        scheduling="device_plugin",
        per_pod_gpus=1,
    )
    assert pod["spec"]["containers"][0]["resources"]["limits"] == {"nvidia.com/gpu": "1"}


def test_gpu_quota_probe_uses_configured_resource_name(monkeypatch):
    be = KubernetesBackend(_cfg(namespace="ns", gpu_resource_name="amd.com/gpu"))

    async def _kubectl(args, **_kw):
        # The device is advertised under requests.<resource-name> in the quota.
        return 0, '{"requests.amd.com/gpu":"8"}', ""

    monkeypatch.setattr(be, "_kubectl", _kubectl)
    assert asyncio.run(be._gpu_quota_present()) is True


# ---------------------------------------------------------------------------
# Reservation placeholder image + GPU resource name
# ---------------------------------------------------------------------------


def test_render_reservation_pod_honors_placeholder_image_and_gpu_name():
    pod = render_reservation_pod_manifest(
        pod_name="res",
        allocation_id="a",
        scheduling="device_plugin",
        gpu_count=4,
        placeholder_image="mirror/bash:5",
        gpu_resource_name="amd.com/gpu",
    )
    c = pod["spec"]["containers"][0]
    assert c["image"] == "mirror/bash:5"
    assert c["resources"]["limits"] == {"amd.com/gpu": "4"}


def test_render_reservation_pod_defaults_image_and_gpu_name():
    pod = render_reservation_pod_manifest(
        pod_name="res", allocation_id="a", scheduling="device_plugin", gpu_count=8
    )
    c = pod["spec"]["containers"][0]
    assert c["image"] == RESERVATION_POD_IMAGE
    assert c["resources"]["limits"] == {"nvidia.com/gpu": "8"}


# ---------------------------------------------------------------------------
# gIB installer namespace override
# ---------------------------------------------------------------------------


def test_gib_installer_namespace_override(monkeypatch):
    be = KubernetesBackend(_cfg(gib_installer_namespace="gpu-operator"))
    seen: dict = {}

    async def _kubectl(args, **_kw):
        seen["args"] = list(args)
        return 0, "nccl-rdma-installer-abc", ""

    monkeypatch.setattr(be, "_kubectl", _kubectl)
    assert asyncio.run(be._gib_installer_present("node-a")) is True
    assert "gpu-operator" in seen["args"]
    assert "kube-system" not in seen["args"]


def test_gib_installer_namespace_defaults_to_kube_system(monkeypatch):
    be = KubernetesBackend(_cfg())
    seen: dict = {}

    async def _kubectl(args, **_kw):
        seen["args"] = list(args)
        return 0, "nccl-rdma-installer-abc", ""

    monkeypatch.setattr(be, "_kubectl", _kubectl)
    asyncio.run(be._gib_installer_present("node-a"))
    assert "kube-system" in seen["args"]


# ---------------------------------------------------------------------------
# GPU-product label key override (NVLink-scope detection)
# ---------------------------------------------------------------------------


def test_gpu_product_label_key_override(monkeypatch):
    be = KubernetesBackend(_cfg(gpu_product_label_key="custom.io/gpu-product"))
    seen: dict = {}

    def _kubectl_sync(args, timeout=None, **_kw):
        seen["args"] = list(args)
        return 0, "H100", ""

    monkeypatch.setattr(be, "_kubectl_sync", _kubectl_sync)
    assert be._detect_gpu_product() == "H100"
    jsonpath = next(a for a in seen["args"] if a.startswith("jsonpath="))
    # Dots escaped so jsonpath selects the label, not nested fields.
    assert r"custom\.io/gpu-product" in jsonpath


def test_gpu_product_label_key_defaults_to_nvidia(monkeypatch):
    be = KubernetesBackend(_cfg())
    seen: dict = {}

    def _kubectl_sync(args, timeout=None, **_kw):
        seen["args"] = list(args)
        return 0, "GB200", ""

    monkeypatch.setattr(be, "_kubectl_sync", _kubectl_sync)
    be._detect_gpu_product()
    jsonpath = next(a for a in seen["args"] if a.startswith("jsonpath="))
    assert r"nvidia\.com/gpu\.product" in jsonpath


# ---------------------------------------------------------------------------
# gpus_per_node preflight: derive/validate against candidate nodes' real capacity
# (runs before planning + allocation, so a derived value feeds the planner)
# ---------------------------------------------------------------------------


def test_preflight_derives_gpus_per_node_when_unset(monkeypatch):
    be = KubernetesBackend(_cfg(scheduling="device_plugin"))  # gpus_per_node unset
    assert be._gpu_per_node is None
    monkeypatch.setattr(be, "_detect_node_gpu_capacity", lambda: 4)
    be._preflight_validate_gpus_per_node()
    assert be._gpu_per_node == 4  # feeds planning via placeholder_allocation


def test_preflight_warns_when_configured_exceeds_capacity(monkeypatch, caplog):
    be = KubernetesBackend(_cfg(scheduling="device_plugin", gpus_per_node=16))
    monkeypatch.setattr(be, "_detect_node_gpu_capacity", lambda: 8)
    with _capture_sflow_warnings(monkeypatch, caplog):
        be._preflight_validate_gpus_per_node()
    assert be._gpu_per_node == 16  # configured value is left untouched
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("exceeds" in m and "Pending" in m for m in warns)


def test_preflight_leaves_partial_node_config_alone(monkeypatch, caplog):
    # A configured value BELOW capacity is a legitimate partial-node reservation --
    # no warning (that would be crying wolf on intended behavior).
    be = KubernetesBackend(_cfg(scheduling="device_plugin", gpus_per_node=2))
    monkeypatch.setattr(be, "_detect_node_gpu_capacity", lambda: 8)
    with _capture_sflow_warnings(monkeypatch, caplog):
        be._preflight_validate_gpus_per_node()
    assert be._gpu_per_node == 2
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_preflight_skips_cpu_only(monkeypatch):
    # gpus_per_node=0 is intentional CPU-only: no derive, no warn, no capacity lookup.
    be = KubernetesBackend(_cfg(scheduling="device_plugin", gpus_per_node=0))
    called = {"n": 0}
    monkeypatch.setattr(
        be, "_detect_node_gpu_capacity", lambda: called.__setitem__("n", 1) or 8
    )
    be._preflight_validate_gpus_per_node()
    assert be._gpu_per_node == 0 and called["n"] == 0


def test_preflight_no_warning_when_configured_matches(monkeypatch, caplog):
    be = KubernetesBackend(_cfg(scheduling="device_plugin", gpus_per_node=8))
    monkeypatch.setattr(be, "_detect_node_gpu_capacity", lambda: 8)
    with _capture_sflow_warnings(monkeypatch, caplog):
        be._preflight_validate_gpus_per_node()
    assert be._gpu_per_node == 8
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_preflight_skips_dra(monkeypatch):
    # DRA does not expose GPUs as an extended resource; skipped, no capacity lookup.
    be = KubernetesBackend(_cfg(scheduling="dra", gpus_per_node=8))
    called = {"n": 0}
    monkeypatch.setattr(
        be, "_detect_node_gpu_capacity", lambda: called.__setitem__("n", 1) or 4
    )
    be._preflight_validate_gpus_per_node()
    assert be._gpu_per_node == 8 and called["n"] == 0


def test_preflight_best_effort_when_capacity_unknown(monkeypatch):
    # Capacity lookup returns None (kubectl failed / no GPU node) -> leave config as-is.
    be = KubernetesBackend(_cfg(scheduling="device_plugin", gpus_per_node=8))
    monkeypatch.setattr(be, "_detect_node_gpu_capacity", lambda: None)
    be._preflight_validate_gpus_per_node()  # must not raise
    assert be._gpu_per_node == 8


def test_detect_node_gpu_capacity_queries_allocatable(monkeypatch):
    be = KubernetesBackend(_cfg(scheduling="device_plugin"))
    seen: dict = {}

    def _kubectl_sync(args, timeout=None, **_kw):
        seen["args"] = list(args)
        return 0, "8 4 8", ""  # heterogeneous pool -> representative max

    monkeypatch.setattr(be, "_kubectl_sync", _kubectl_sync)
    assert be._detect_node_gpu_capacity() == 8
    jsonpath = next(a for a in seen["args"] if a.startswith("jsonpath="))
    assert r"allocatable.nvidia\.com/gpu" in jsonpath


def test_detect_node_gpu_capacity_honors_custom_resource_name(monkeypatch):
    be = KubernetesBackend(_cfg(scheduling="device_plugin", gpu_resource_name="amd.com/gpu"))
    seen: dict = {}

    def _kubectl_sync(args, timeout=None, **_kw):
        seen["args"] = list(args)
        return 0, "6", ""

    monkeypatch.setattr(be, "_kubectl_sync", _kubectl_sync)
    assert be._detect_node_gpu_capacity() == 6
    jsonpath = next(a for a in seen["args"] if a.startswith("jsonpath="))
    assert r"allocatable.amd\.com/gpu" in jsonpath


def test_detect_node_gpu_capacity_none_when_query_fails(monkeypatch):
    be = KubernetesBackend(_cfg(scheduling="device_plugin"))
    monkeypatch.setattr(be, "_kubectl_sync", lambda args, timeout=None, **_kw: (1, "", "err"))
    assert be._detect_node_gpu_capacity() is None


# ---------------------------------------------------------------------------
# DRA API-version preflight (resource.k8s.io/v1 served?)
# ---------------------------------------------------------------------------


def test_preflight_dra_warns_when_ga_api_not_served(monkeypatch, caplog):
    be = KubernetesBackend(_cfg(scheduling="dra", namespace="ns"))

    def _sync(args, timeout=None, **_kw):
        if args[:1] == ["api-versions"]:
            return 0, "v1\napps/v1\nresource.k8s.io/v1beta1\n", ""
        if args[:2] == ["get", "deviceclass"]:
            return 0, "deviceclass.resource.k8s.io/gpu.nvidia.com", ""
        return 0, "", ""

    monkeypatch.setattr(be, "_kubectl_sync", _sync)
    with _capture_sflow_warnings(monkeypatch, caplog):
        be._preflight_check_dra()
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "not served" in m and "resource.k8s.io/v1" in m and "v1beta1" in m
        for m in warns
    )


def test_preflight_dra_quiet_when_ga_api_served(monkeypatch, caplog):
    be = KubernetesBackend(_cfg(scheduling="dra", namespace="ns"))

    def _sync(args, timeout=None, **_kw):
        if args[:1] == ["api-versions"]:
            return 0, "v1\nresource.k8s.io/v1\nresource.k8s.io/v1beta1\n", ""
        if args[:2] == ["get", "deviceclass"]:
            return 0, "deviceclass.resource.k8s.io/gpu.nvidia.com", ""
        return 0, "", ""

    monkeypatch.setattr(be, "_kubectl_sync", _sync)
    with _capture_sflow_warnings(monkeypatch, caplog):
        be._preflight_check_dra()
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("not served" in m for m in warns)
