# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
        dra=KubernetesDraConfig(gpu_device_class="gpu.nvidia.com", compute_domain=True),
        reservation=KubernetesReservationConfig(timeout=120),
    )
    resolved = KubernetesBackend.resolve_config(
        conf, resolver=_IdentityResolver(), ctx={}, workflow_name="wf"
    )
    assert resolved.namespace == "bench"
    assert resolved.scheduling == "dra"
    assert resolved.dra.gpu_device_class == "gpu.nvidia.com"
    assert resolved.dra.compute_domain is True
    assert resolved.reservation.timeout == 120
    assert not hasattr(resolved, "image")


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

    with (
        mock.patch.object(backend, "_apply_manifest", side_effect=fake_apply),
        mock.patch.object(backend, "_wait_for_pod_scheduled", side_effect=fake_wait),
        mock.patch.object(backend, "_node_internal_ip", side_effect=fake_internal),
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
            dra=KubernetesDraConfig(compute_domain=True),
        )
    )
    _, applied = _reserve(backend)

    cds = [m for m in applied if m.get("kind") == "ComputeDomain"]
    assert len(cds) == 1
    assert cds[0]["spec"]["numNodes"] == 2
    assert backend.compute_domain_channel is not None


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
    ):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(backend.allocate())

    assert len(released) == 1
    assert backend._pending_alloc_id is None
