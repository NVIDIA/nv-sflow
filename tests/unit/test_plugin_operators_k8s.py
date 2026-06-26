# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single ``k8s`` operator: DRA vs device-plugin GPU requests,
multi-node pod-set + env wiring, ConfigMap script mount, hostname pinning, and
the create-before-destroy handoff gating (GPU tasks only)."""

import json
import logging

import pytest

from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig
from sflow.plugins.operators.k8s import K8sOperator, K8sOperatorConfig

_MARK = "SFLOW_K8S_MANIFEST"


def _backend(scheduling="dra", gpus_per_node=8, nodes=2, namespace="default"):
    backend = KubernetesBackend(
        KubernetesBackendConfig(
            name="k8s",
            type="kubernetes",
            namespace=namespace,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            scheduling=scheduling,
        )
    )
    backend.allocation = Allocation(
        allocation_id="abc",
        nodes=[
            ComputeNode(name=f"node-{i}", ip_address=f"10.0.0.{i + 1}", index=i,
                        num_gpus=gpus_per_node)
            for i in range(nodes)
        ],
        owned=True,
    )
    backend._node_to_resv_pod = {f"node-{i}": f"res-{i}" for i in range(nodes)}
    return backend


def _build(backend, assigned_nodes, *, gpu_count, task="t", script=("run",), envs=None):
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=list(assigned_nodes), artifacts=[],
        gpu_count=gpu_count,
    )
    cmd = op.build_command(task_name=task, script=list(script), envs=dict(envs or {}))
    shell = cmd.as_list()[-1]
    after = shell.split(_MARK, 1)[1]
    body = after.split("\n" + _MARK, 1)[0]
    manifest = json.loads(body.split("\n", 1)[1])
    return manifest, shell


def _pods(manifest):
    return [i for i in manifest["items"] if i["kind"] == "Pod"]


# ---------------------------------------------------------------------------
# DRA path
# ---------------------------------------------------------------------------


def test_dra_single_node_renders_claim_template_and_pod_claim():
    manifest, shell = _build(_backend("dra", 8), ["node-0"], gpu_count=4)
    kinds = [i["kind"] for i in manifest["items"]]
    assert "ConfigMap" in kinds and "ResourceClaimTemplate" in kinds
    rct = [i for i in manifest["items"] if i["kind"] == "ResourceClaimTemplate"][0]
    req = rct["spec"]["spec"]["devices"]["requests"][0]["exactly"]
    assert req["deviceClassName"] == "gpu.nvidia.com"
    assert req["count"] == 4
    pod = _pods(manifest)[0]
    assert pod["spec"]["resourceClaims"][0]["resourceClaimTemplateName"].endswith("-gpu")
    assert pod["spec"]["containers"][0]["resources"]["claims"] == [{"name": "gpu"}]
    # GPU task -> create-before-destroy handoff deletes the node's placeholder.
    assert "res-0" in shell


def test_dra_device_selectors_inherited_from_backend():
    backend = _backend("dra", 8)
    backend._device_selectors = ['device.attributes["x"].product == "H100"']
    manifest, _ = _build(backend, ["node-0"], gpu_count=2)
    rct = [i for i in manifest["items"] if i["kind"] == "ResourceClaimTemplate"][0]
    sel = rct["spec"]["spec"]["devices"]["requests"][0]["exactly"]["selectors"]
    assert sel == [{"cel": {"expression": 'device.attributes["x"].product == "H100"'}}]


# ---------------------------------------------------------------------------
# device_plugin path
# ---------------------------------------------------------------------------


def test_device_plugin_renders_limit_no_claim():
    manifest, shell = _build(_backend("device_plugin", 8), ["node-0"], gpu_count=4)
    assert not [i for i in manifest["items"] if i["kind"] == "ResourceClaimTemplate"]
    pod = _pods(manifest)[0]
    assert pod["spec"]["containers"][0]["resources"]["limits"] == {"nvidia.com/gpu": "4"}
    assert "resourceClaims" not in pod["spec"]
    assert "res-0" in shell  # GPU task still hands off


# ---------------------------------------------------------------------------
# multi-node pod set + env wiring
# ---------------------------------------------------------------------------


def test_multinode_splits_pods_and_wires_env():
    manifest, shell = _build(_backend("dra", 8, nodes=2), ["node-0", "node-1"], gpu_count=16)
    pods = _pods(manifest)
    assert [p["metadata"]["name"] for p in pods] == ["t-0", "t-1"]
    # 16 GPUs across 2 nodes -> 8 per pod.
    rct = [i for i in manifest["items"] if i["kind"] == "ResourceClaimTemplate"][0]
    assert rct["spec"]["spec"]["devices"]["requests"][0]["exactly"]["count"] == 8
    # Per-pod env: node index + leader address (= node-0 IP).
    env0 = {e["name"]: e["value"] for e in pods[0]["spec"]["containers"][0]["env"]}
    env1 = {e["name"]: e["value"] for e in pods[1]["spec"]["containers"][0]["env"]}
    assert env0["SFLOW_TASK_NODE_INDEX"] == "0"
    assert env1["SFLOW_TASK_NODE_INDEX"] == "1"
    assert env0["SFLOW_LEADER_ADDRESS"] == "10.0.0.1"
    assert env1["SFLOW_LEADER_ADDRESS"] == "10.0.0.1"
    # Each pod pinned to its node; both placeholders handed off; parallel logs.
    assert pods[0]["spec"]["nodeSelector"]["kubernetes.io/hostname"] == "node-0"
    assert pods[1]["spec"]["nodeSelector"]["kubernetes.io/hostname"] == "node-1"
    assert "res-0" in shell and "res-1" in shell
    assert shell.count("kubectl logs -f") == 2


# ---------------------------------------------------------------------------
# ConfigMap + hostname pin + tolerations
# ---------------------------------------------------------------------------


def test_script_mounted_via_configmap_and_tolerations_present():
    manifest, _ = _build(_backend("dra", 8), ["node-0"], gpu_count=2,
                         script=["echo hi", "python run.py"])
    cm = [i for i in manifest["items"] if i["kind"] == "ConfigMap"][0]
    assert cm["data"]["entrypoint.sh"] == "echo hi\npython run.py"
    pod = _pods(manifest)[0]
    assert pod["spec"]["containers"][0]["command"] == ["bash", "-l", "/sflow/entrypoint.sh"]
    vols = pod["spec"]["volumes"][0]
    assert vols["configMap"]["name"] == cm["metadata"]["name"]
    assert {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"} in (
        pod["spec"]["tolerations"]
    )


# ---------------------------------------------------------------------------
# CPU-only overlap: no handoff, no GPU resources
# ---------------------------------------------------------------------------


def test_cpu_only_task_does_not_delete_placeholder():
    manifest, shell = _build(_backend("dra", 8), ["node-0"], gpu_count=None)
    pod = _pods(manifest)[0]
    assert "resources" not in pod["spec"]["containers"][0]
    assert "resourceClaims" not in pod["spec"]
    # No placeholder deletion -> the node's GPUs stay reserved for GPU tasks.
    assert "res-0" not in shell


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_non_divisible_gpu_count_raises():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    with pytest.raises(ValueError, match="not divisible"):
        op.apply_backend_context(
            backend=_backend("dra", 8, nodes=2),
            assigned_nodes=["node-0", "node-1"],
            artifacts=[],
            gpu_count=3,
        )


def test_secret_envfrom_used_when_envs_present():
    manifest, shell = _build(_backend("dra", 8), ["node-0"], gpu_count=2,
                             envs={"TOKEN": "x"})
    pod = _pods(manifest)[0]
    assert pod["spec"]["containers"][0]["envFrom"][0]["secretRef"]["name"].endswith("-env")
    assert "kubectl create secret generic" in shell


# ---------------------------------------------------------------------------
# release_after coercion (planner) for non-gpu-sharing backends
# ---------------------------------------------------------------------------


def test_gpu_release_after_task_ready_coerced_on_k8s():
    from sflow.app import resource_planner as rp
    from sflow.app.assembly import build_task_graph
    from sflow.config.schema import (
        GpuResourceConfig,
        ResourcesConfig,
        SflowConfig,
        TaskConfig,
        WorkflowConfig,
    )
    from sflow.core.state import SflowState
    from sflow.core.task_graph import TaskGraph
    from sflow.core.workflow import Workflow

    backend = _backend("dra", 8, nodes=1)
    backend.allocation = backend.placeholder_allocation()
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {"k8s": backend}
    state.default_backend = backend
    config = SflowConfig(
        version="0.1",
        backends=[{"name": "k8s", "type": "kubernetes", "default": True,
                   "namespace": "default", "nodes": 1, "gpus_per_node": 8}],
        operators=[{"name": "svc", "type": "k8s", "image": "img:1"}],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="a",
                    operator="svc",
                    resources=ResourcesConfig(
                        gpus=GpuResourceConfig(count=8, release_after="task_ready")
                    ),
                    script=["run"],
                ),
            ],
        ),
    )

    messages: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    handler = _Cap()
    rp._logger.addHandler(handler)
    try:
        build_task_graph(config, state)
    finally:
        rp._logger.removeHandler(handler)

    assert any("task_ready is not supported" in m for m in messages)


def test_operator_wrapper_carries_kubectl_global_args():
    from sflow.core.kubectl_config import KubectlConfig

    backend = _backend("dra", 8, nodes=1)
    backend.apply_kubectl_config(KubectlConfig(kubeconfig="/k/cfg", context="ctx"))
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=backend, assigned_nodes=["node-0"], artifacts=[],
                             gpu_count=2)
    shell = op.build_command(task_name="t", script=["run"], envs={}).as_list()[-1]
    assert 'kubectl() { command kubectl --kubeconfig /k/cfg --context ctx "$@"; }' in shell


def test_operator_wrapper_omits_kubectl_function_without_global_args():
    backend = _backend("dra", 8, nodes=1)
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=backend, assigned_nodes=["node-0"], artifacts=[],
                             gpu_count=2)
    shell = op.build_command(task_name="t", script=["run"], envs={}).as_list()[-1]
    assert "command kubectl" not in shell


# ---------------------------------------------------------------------------
# task-startup wait: diagnostic echo
# ---------------------------------------------------------------------------


def test_wrapper_echoes_phase_and_reason_while_waiting():
    _, shell = _build(_backend("dra", 8), ["node-0"], gpu_count=2)
    # Heartbeat/echo of the pod's phase + reason while it starts up.
    assert "[sflow]" in shell
    assert "phase=" in shell
    assert "reason=" in shell
    # On the timeout/Failed path, dump describe + events for the pod.
    assert "kubectl describe pod" in shell


def test_gpu_oversubscription_detected_at_plan_time():
    # Planning is uniform across backends: the planner reserves GPU intervals for
    # k8s too, so over-requesting GPUs on a node is caught at plan time (rather
    # than leaving a pod Pending at runtime).
    from sflow.app.assembly import build_task_graph
    from sflow.config.schema import (
        GpuResourceConfig,
        ResourcesConfig,
        SflowConfig,
        TaskConfig,
        WorkflowConfig,
    )
    from sflow.core.state import SflowState
    from sflow.core.task_graph import TaskGraph
    from sflow.core.workflow import Workflow

    backend = _backend("dra", gpus_per_node=4, nodes=1)
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {"k8s": backend}
    state.default_backend = backend
    config = SflowConfig(
        version="0.1",
        backends=[{"name": "k8s", "type": "kubernetes", "default": True,
                   "namespace": "default", "nodes": 1, "gpus_per_node": 4}],
        operators=[{"name": "svc", "type": "k8s", "image": "img:1"}],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(name="a", operator="svc",
                           resources=ResourcesConfig(gpus=GpuResourceConfig(count=4)),
                           script=["run"]),
                TaskConfig(name="b", operator="svc",
                           resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                           script=["run"]),
            ],
        ),
    )

    with pytest.raises(ValueError, match="remain available"):
        build_task_graph(config, state)


# ---------------------------------------------------------------------------
# cleanup: allocation label on every task object
# ---------------------------------------------------------------------------


def test_task_objects_carry_allocation_label():
    manifest, shell = _build(_backend("dra", 8), ["node-0"], gpu_count=2,
                             envs={"TOKEN": "x"})
    cm = [i for i in manifest["items"] if i["kind"] == "ConfigMap"][0]
    assert cm["metadata"]["labels"]["sflow.ai/allocation"] == "abc"
    pod = _pods(manifest)[0]
    assert pod["metadata"]["labels"]["sflow.ai/allocation"] == "abc"
    rct = [i for i in manifest["items"] if i["kind"] == "ResourceClaimTemplate"][0]
    assert rct["metadata"]["labels"]["sflow.ai/allocation"] == "abc"
    # The env Secret is created imperatively, so it is labeled in the wrapper.
    assert "kubectl label secret" in shell
    assert "sflow.ai/allocation=abc" in shell


# ---------------------------------------------------------------------------
# cleanup: wrapper traps signals so Ctrl+C deletes the task objects
# ---------------------------------------------------------------------------


def test_wrapper_traps_int_term_exit():
    _, shell = _build(_backend("dra", 8), ["node-0"], gpu_count=2, envs={"T": "x"})
    assert "trap cleanup INT TERM EXIT" in shell
    # cleanup disarms itself to avoid re-entry when it exits.
    assert "trap - INT TERM EXIT" in shell
