# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single ``k8s`` operator: DRA vs device-plugin GPU requests,
multi-node pod-set + env wiring, ConfigMap script mount, hostname pinning, and
the create-before-destroy handoff gating (GPU tasks only)."""

import json
import logging
import os
import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

from sflow.core.artifact import Artifact
from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.plugins.backends._k8s_rdma import RdmaPlan
from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig
from sflow.plugins.operators import _k8s_shell as k8s_shell
from sflow.plugins.operators.k8s import K8sOperator, K8sOperatorConfig

_MARK = "SFLOW_K8S_MANIFEST"


def _backend(scheduling="dra", gpus_per_node=8, nodes=2, namespace="default",
             volumes=None):
    backend = KubernetesBackend(
        KubernetesBackendConfig(
            name="k8s",
            type="kubernetes",
            namespace=namespace,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            scheduling=scheduling,
            volumes=volumes,
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


def _build(
    backend, assigned_nodes, *, gpu_count, task="t", script=("run",), envs=None,
    artifacts=(), cuda_visible=None,
):
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=list(assigned_nodes),
        artifacts=list(artifacts), gpu_count=gpu_count,
        cuda_visible_devices=cuda_visible,
    )
    cmd = op.build_command(task_name=task, script=list(script), envs=dict(envs or {}))
    shell = cmd.as_list()[-1]
    after = shell.split(_MARK, 1)[1]
    body = after.split("\n" + _MARK, 1)[0]
    manifest = json.loads(body.split("\n", 1)[1])
    return manifest, shell


def _pods(manifest):
    return [i for i in manifest["items"] if i["kind"] == "Pod"]


def _entrypoint(manifest):
    """The task entrypoint script text from the rendered ConfigMap."""
    for item in manifest["items"]:
        if item["kind"] == "ConfigMap" and "entrypoint.sh" in item.get("data", {}):
            return item["data"]["entrypoint.sh"]
    return ""


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
    # Each pod pinned to its node; both placeholders handed off in the apply step.
    assert pods[0]["spec"]["nodeSelector"]["kubernetes.io/hostname"] == "node-0"
    assert pods[1]["spec"]["nodeSelector"]["kubernetes.io/hostname"] == "node-1"
    assert "res-0" in shell and "res-1" in shell
    # Log streaming is now a separate driver-managed step, not in the apply command.
    assert "kubectl logs -f" not in shell


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


# ---------------------------------------------------------------------------
# decoupled execution: build_command is the apply step only (start the pod);
# the driver (execute) streams + watches + stops. The apply command must NOT
# stream logs, poll the exit code, or register a cleanup trap.
# ---------------------------------------------------------------------------


def test_build_command_is_apply_only():
    _, shell = _build(_backend("dra", 8), ["node-0"], gpu_count=2, envs={"T": "x"})
    # Apply step: create the env secret, apply the manifest, hand off the
    # placeholder, and wait for the pod to start.
    assert "kubectl create secret generic" in shell
    assert "kubectl apply -f -" in shell
    assert "res-0" in shell  # handoff (GPU task)
    assert "[sflow]" in shell and "phase=" in shell  # wait-for-ready diagnostics
    # It is NOT the old all-in-one wrapper: no live stream, no exit-code poll,
    # and no cleanup trap (the driver owns the stream/status/teardown now).
    assert "kubectl logs -f" not in shell
    assert "terminated.exitCode" not in shell
    assert "trap cleanup" not in shell


def test_k8s_operator_manages_own_execution():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    assert op.manages_own_execution() is True


def test_log_stream_command_is_bare_follow_with_global_args():
    from sflow.core.kubectl_config import KubectlConfig
    from sflow.plugins.operators._k8s_shell import build_log_stream_command

    cmd = build_log_stream_command(
        "pod/t-0",
        ns_args=["--namespace", "ns"],
        kubectl_global_args=["--kubeconfig", "/k/cfg"],
    )
    argv = cmd.as_list()
    assert argv == [
        "kubectl", "--kubeconfig", "/k/cfg", "logs", "-f", "pod/t-0",
        "--namespace", "ns", "--all-containers", "--prefix",
    ]
    # The operator builds one stream command per pod with the backend's kube flags.
    backend = _backend("dra", 8, nodes=1, namespace="ns")
    backend.apply_kubectl_config(KubectlConfig(kubeconfig="/k/cfg"))
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=backend, assigned_nodes=["node-0"], artifacts=[],
                             gpu_count=2)
    plan = op._build_execution_plan(task_name="t", script=["run"], envs={})
    assert [c.as_list() for c in plan.log_stream_commands] == [
        ["kubectl", "--kubeconfig", "/k/cfg", "logs", "-f", "pod/t",
         "--namespace", "ns", "--all-containers", "--prefix"]
    ]
    assert plan.task_log_path is None  # no SFLOW_TASK_OUTPUT_DIR -> no offload file


def test_execution_plan_task_log_path_and_cleanup_refs(tmp_path):
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend("dra", 8, nodes=1, namespace="ns"),
                             assigned_nodes=["node-0"], artifacts=[], gpu_count=2)
    plan = op._build_execution_plan(
        task_name="decode_server_0", script=["run"],
        envs={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path), "TOKEN": "x"},
    )
    assert plan.pod_refs == ["pod/decode-server-0"]
    # The pod log is offloaded straight to <task>.log (raw task name), which the
    # decoupled console tailer reads.
    assert plan.task_log_path == str(tmp_path / "decode_server_0.log")
    # Cleanup deletes this task's own objects by name (backstop: backend sweep).
    assert "pod/decode-server-0" in plan.cleanup_refs
    assert "configmap/decode-server-0-cfg" in plan.cleanup_refs
    assert "secret/decode-server-0-env" in plan.cleanup_refs
    assert any(r.startswith("resourceclaimtemplate") for r in plan.cleanup_refs)


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
# cleanup: the driver (execute) deletes the task objects; the apply step does
# not register a trap (the backend's allocation-label sweep is the backstop).
# ---------------------------------------------------------------------------


def test_apply_command_has_no_cleanup_trap():
    _, shell = _build(_backend("dra", 8), ["node-0"], gpu_count=2, envs={"T": "x"})
    assert "trap cleanup" not in shell
    assert "trap - INT TERM EXIT" not in shell


# ---------------------------------------------------------------------------
# K8s-native artifact injection (file:// -> ConfigMap, fs:// -> hostPath)
# ---------------------------------------------------------------------------


def test_inline_file_artifact_injected_as_configmap_and_mounted():
    # file:// + inline content -> a ConfigMap carries the content (in-cluster, not on
    # the controller disk) and is mounted at the resolved path inside the pod.
    art = Artifact(
        name="PREFILL_CONFIG",
        uri="file://prefill_config.yaml",
        path=Path("/out/run/prefill_config.yaml"),
        content="max_batch_size: 128\n",
    )
    manifest, shell = _build(
        _backend("device_plugin", 8), ["node-0"], gpu_count=0,
        script=["run --cfg /out/run/prefill_config.yaml"],
        artifacts=[art],
    )
    art_cms = [
        i for i in manifest["items"]
        if i["kind"] == "ConfigMap" and i["metadata"]["name"].endswith("-artifacts")
    ]
    assert len(art_cms) == 1
    assert art_cms[0]["data"] == {"PREFILL_CONFIG": "max_batch_size: 128\n"}
    assert art_cms[0]["metadata"]["labels"]["sflow.ai/allocation"] == "abc"

    mounts = _pods(manifest)[0]["spec"]["containers"][0]["volumeMounts"]
    assert any(
        m.get("subPath") == "PREFILL_CONFIG"
        and m["mountPath"] == "/out/run/prefill_config.yaml"
        and m.get("readOnly")
        for m in mounts
    )
    # The artifacts ConfigMap is torn down with the rest of the task objects.
    assert "-artifacts" in shell


def test_fs_artifact_injected_as_hostpath_when_referenced():
    art = Artifact(
        name="MODEL", uri="fs:///shared/models/qwen", path=Path("/shared/models/qwen")
    )
    manifest, _ = _build(
        _backend("device_plugin", 8), ["node-0"], gpu_count=0,
        script=["run --model /shared/models/qwen"],
        artifacts=[art],
    )
    vols = _pods(manifest)[0]["spec"]["volumes"]
    assert any(v.get("hostPath", {}).get("path") == "/shared/models/qwen" for v in vols)


def test_artifact_not_mounted_when_not_referenced_by_script():
    # CPU-only infra pods (e.g. etcd) shouldn't get the model hostPath mounted.
    art = Artifact(
        name="MODEL", uri="fs:///shared/models/qwen", path=Path("/shared/models/qwen")
    )
    manifest, _ = _build(
        _backend("device_plugin", 8), ["node-0"], gpu_count=0,
        script=["etcd --data-dir /tmp/etcd"],
        artifacts=[art],
    )
    vols = _pods(manifest)[0]["spec"]["volumes"]
    assert all("hostPath" not in v for v in vols)
    assert not [
        i for i in manifest["items"]
        if i["kind"] == "ConfigMap" and i["metadata"]["name"].endswith("-artifacts")
    ]


def test_artifact_referenced_by_env_name_is_injected():
    # Scripts may use the ${NAME} env convenience instead of the resolved path.
    art = Artifact(
        name="MODEL", uri="fs:///shared/models/qwen", path=Path("/shared/models/qwen")
    )
    manifest, _ = _build(
        _backend("device_plugin", 8), ["node-0"], gpu_count=0,
        script=["run --model ${MODEL}"],
        artifacts=[art],
    )
    vols = _pods(manifest)[0]["spec"]["volumes"]
    assert any(v.get("hostPath", {}).get("path") == "/shared/models/qwen" for v in vols)


def test_fs_artifact_hostpath_type_pins_directory_when_path_exists(tmp_path):
    # When the controller can see the fs:// path (e.g. a shared filesystem mounted
    # on both controller and nodes), pin hostPath type=Directory so the kubelet
    # fails loudly on nodes that lack it instead of mounting an empty dir.
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    art = Artifact(name="MODEL", uri=f"fs://{model_dir}", path=model_dir)
    manifest, _ = _build(
        _backend("device_plugin", 8), ["node-0"], gpu_count=0,
        script=[f"run --model {model_dir}"],
        artifacts=[art],
    )
    hp = [v for v in _pods(manifest)[0]["spec"]["volumes"] if v.get("hostPath")][0]
    assert hp["hostPath"] == {"path": str(model_dir), "type": "Directory"}


# ---------------------------------------------------------------------------
# PVC mounts (pre-existing PersistentVolumeClaims backing fs:// paths)
# ---------------------------------------------------------------------------


def test_fs_artifact_under_pvc_mounts_pvc_not_hostpath():
    # fs:// path living under a declared PVC mount -> mount the PVC, skip hostPath.
    be = _backend(
        "device_plugin", 8,
        volumes=[{"name": "model-store", "claim": "model-pvc", "mount_path": "/models"}],
    )
    art = Artifact(
        name="MODEL", uri="fs:///models/Qwen3-8B-NVFP4",
        path=Path("/models/Qwen3-8B-NVFP4"),
    )
    manifest, _ = _build(
        be, ["node-0"], gpu_count=0,
        script=["run --model /models/Qwen3-8B-NVFP4"], artifacts=[art],
    )
    pod = _pods(manifest)[0]
    vols = pod["spec"]["volumes"]
    pvc = [v for v in vols if v.get("persistentVolumeClaim")]
    assert len(pvc) == 1
    assert pvc[0]["persistentVolumeClaim"] == {"claimName": "model-pvc", "readOnly": True}
    # The covered fs:// path is served by the PVC, not a hostPath.
    assert all("hostPath" not in v for v in vols)
    mt = [
        x for x in pod["spec"]["containers"][0]["volumeMounts"]
        if x["name"] == pvc[0]["name"]
    ][0]
    assert mt == {"name": pvc[0]["name"], "mountPath": "/models", "readOnly": True}


def test_declared_pvc_mounted_into_pod_even_without_script_reference():
    # Backend volumes are workflow-wide storage: mounted into every task pod even
    # when the script never names the path -- e.g. the dynamo frontend loads the
    # model card via discovery, so it needs /models mounted despite its script
    # being just `python3 -m dynamo.frontend`.
    be = _backend(
        "device_plugin", 8,
        volumes=[{"name": "model-store", "claim": "model-pvc", "mount_path": "/models"}],
    )
    art = Artifact(
        name="MODEL", uri="fs:///models/Qwen3-8B-NVFP4",
        path=Path("/models/Qwen3-8B-NVFP4"),
    )
    manifest, _ = _build(
        be, ["node-0"], gpu_count=0,
        script=["python3 -m dynamo.frontend --http-port 8000"], artifacts=[art],
    )
    pod = _pods(manifest)[0]
    pvc = [v for v in pod["spec"]["volumes"] if v.get("persistentVolumeClaim")]
    assert len(pvc) == 1
    assert pvc[0]["persistentVolumeClaim"]["claimName"] == "model-pvc"
    mt = [
        x for x in pod["spec"]["containers"][0]["volumeMounts"]
        if x["name"] == pvc[0]["name"]
    ][0]
    assert mt["mountPath"] == "/models"


def test_no_pvc_mounted_when_backend_has_no_volumes():
    # Without declared backend volumes, an fs:// path falls back to a hostPath
    # (no PVC), even if referenced.
    be = _backend("device_plugin", 8)
    art = Artifact(
        name="MODEL", uri="fs:///models/Qwen3-8B-NVFP4",
        path=Path("/models/Qwen3-8B-NVFP4"),
    )
    manifest, _ = _build(
        be, ["node-0"], gpu_count=0, script=["run --model /models/Qwen3-8B-NVFP4"],
        artifacts=[art],
    )
    vols = _pods(manifest)[0]["spec"]["volumes"]
    assert all("persistentVolumeClaim" not in v for v in vols)


def test_network_env_injected_into_pod_env():
    # When the backend detected RDMA, every task pod gets NCCL/gloo socket env
    # plus optional NCCL_IB_HCA. UCX is left unset.
    be = _backend("device_plugin", 8)
    be._rdma_plan = RdmaPlan(  # simulate reservation-time detection
        provider="explicit",
        enabled=False,
        net_env={
            "NCCL_IB_HCA": "mlx5_0,mlx5_1",
            "NCCL_SOCKET_IFNAME": "eth0",
            "GLOO_SOCKET_IFNAME": "eth0",
        },
    )
    manifest, _ = _build(be, ["node-0"], gpu_count=2)
    env = {
        e["name"]: e["value"]
        for e in _pods(manifest)[0]["spec"]["containers"][0].get("env", [])
    }
    assert "UCX_NET_DEVICES" not in env
    assert env.get("NCCL_IB_HCA") == "mlx5_0,mlx5_1"
    assert env.get("NCCL_SOCKET_IFNAME") == "eth0"
    assert env.get("GLOO_SOCKET_IFNAME") == "eth0"


def test_no_network_env_when_no_rdma_detected():
    manifest, _ = _build(_backend("device_plugin", 8), ["node-0"], gpu_count=2)
    env = {
        e["name"]: e["value"]
        for e in _pods(manifest)[0]["spec"]["containers"][0].get("env", [])
    }
    assert "UCX_NET_DEVICES" not in env and "NCCL_IB_HCA" not in env


def test_shared_device_plugin_plan_requests_single_deduped_resource():
    # A shared rdma/* resource maps to many HCAs but must be requested ONCE as a
    # limit. The shared resource grants access to every node HCA, so NIC selection
    # is deferred to the in-pod runtime affinity preamble (allow_runtime_affinity).
    be = _backend("device_plugin", 8, nodes=1)
    be._rdma_plan = RdmaPlan(
        provider="shared_device_plugin",
        enabled=True,
        nic_specs=tuple(("rdma/hca", f"mlx5_{i}") for i in range(4)),
        ipc_lock=True,
        allow_runtime_affinity=True,
    )
    manifest, _ = _build(be, ["node-0"], gpu_count=4, cuda_visible="0,1,2,3")
    pod = _pods(manifest)[0]
    limits = pod["spec"]["containers"][0]["resources"]["limits"]
    assert limits["rdma/hca"] == "1"  # one shared resource, not four
    assert limits["nvidia.com/gpu"] == "4"
    sc = pod["spec"]["containers"][0]["securityContext"]
    assert sc["capabilities"]["add"] == ["IPC_LOCK"]
    # NIC pinned in-pod (verified) -> no static build-time UCX/NCCL_IB env.
    env = {e["name"]: e["value"] for e in pod["spec"]["containers"][0].get("env", [])}
    assert "NCCL_IB_HCA" not in env and "UCX_NET_DEVICES" not in env
    assert "_sflow_rdma_setup" in _entrypoint(manifest)


def test_host_device_plan_mounts_dev_infiniband_no_extended_resource():
    # host-device provider: no extended RDMA resource, verbs access via the
    # /dev/infiniband hostPath mount + IPC_LOCK. The mount exposes every node HCA,
    # so NIC selection is deferred to the in-pod runtime affinity preamble.
    be = _backend("device_plugin", 8, nodes=1)
    be._rdma_plan = RdmaPlan(
        provider="host_device",
        enabled=True,
        nic_specs=tuple(("", f"mlx5_{i}") for i in range(4)),
        ipc_lock=True,
        host_device_paths=("/dev/infiniband",),
        allow_runtime_affinity=True,
    )
    manifest, _ = _build(be, ["node-0"], gpu_count=4, cuda_visible="0,1,2,3")
    pod = _pods(manifest)[0]
    vols = pod["spec"]["volumes"]
    assert any(v.get("hostPath", {}).get("path") == "/dev/infiniband" for v in vols)
    limits = pod["spec"]["containers"][0]["resources"].get("limits", {})
    assert not any(k.startswith("rdma/") or "gke.io" in k for k in limits)
    sc = pod["spec"]["containers"][0]["securityContext"]
    assert sc["capabilities"]["add"] == ["IPC_LOCK"]
    # NIC pinned in-pod (verified) -> no static build-time UCX/NCCL_IB env; the
    # preamble runs before the workload and falls back to TCP if RDMA is unusable.
    env = {e["name"]: e["value"] for e in pod["spec"]["containers"][0].get("env", [])}
    assert "UCX_NET_DEVICES" not in env and "NCCL_IB_HCA" not in env
    assert "_sflow_rdma_setup" in _entrypoint(manifest)


def test_dra_rdma_coalloc_adds_nic_request_and_constraint():
    # scheduling: dra + dra.rdma_device_class -> the per-task claim co-requests a
    # NIC on the same PCIe root as the GPU, the pod gets IPC_LOCK, and the runtime
    # affinity preamble sets the env to the co-allocated NIC (with TCP fallback).
    be = _backend("dra", 8, nodes=1)
    be._dra_rdma_device_class = "rdma.nvidia.com"
    manifest, _ = _build(be, ["node-0"], gpu_count=2, cuda_visible="0,1")
    rct = [i for i in manifest["items"] if i["kind"] == "ResourceClaimTemplate"][0]
    devices = rct["spec"]["spec"]["devices"]
    assert [r["name"] for r in devices["requests"]] == ["gpu", "rdma"]
    nic = [r for r in devices["requests"] if r["name"] == "rdma"][0]["exactly"]
    assert nic["deviceClassName"] == "rdma.nvidia.com"
    assert nic["count"] == 2
    assert devices["constraints"] == [
        {
            "requests": ["gpu", "rdma"],
            "matchAttribute": "resource.kubernetes.io/pcieRoot",
        }
    ]
    # The container references the single claim -> receives both GPU and NIC.
    pod = _pods(manifest)[0]
    assert pod["spec"]["containers"][0]["resources"]["claims"] == [{"name": "gpu"}]
    sc = pod["spec"]["containers"][0]["securityContext"]
    assert sc["capabilities"]["add"] == ["IPC_LOCK"]
    assert "_sflow_rdma_setup" in _entrypoint(manifest)


def test_dra_rdma_coalloc_respects_custom_match_attribute():
    be = _backend("dra", 8, nodes=1)
    be._dra_rdma_device_class = "dra.net"
    be._dra_rdma_match_attribute = "dra.net/numaNode"
    manifest, _ = _build(be, ["node-0"], gpu_count=1, cuda_visible="0")
    devices = [
        i for i in manifest["items"] if i["kind"] == "ResourceClaimTemplate"
    ][0]["spec"]["spec"]["devices"]
    assert devices["constraints"][0]["matchAttribute"] == "dra.net/numaNode"
    nic = [r for r in devices["requests"] if r["name"] == "rdma"][0]["exactly"]
    assert nic["deviceClassName"] == "dra.net"


def test_host_ipc_threads_to_pod_spec_and_shares_dev_shm():
    # Backend host_ipc -> pod spec.hostIPC + shared hostPath /dev/shm, so co-located
    # prefill/decode pods can do cross-pod CUDA IPC (NVLink) for KV transfer.
    be = _backend("device_plugin", 8, nodes=1)
    be._host_ipc = True
    manifest, _ = _build(be, ["node-0"], gpu_count=4, cuda_visible="0,1,2,3")
    pod = _pods(manifest)[0]
    assert pod["spec"]["hostIPC"] is True
    dshm = [v for v in pod["spec"]["volumes"] if v["name"] == "dshm"][0]
    assert dshm["hostPath"] == {"path": "/dev/shm", "type": "Directory"}


def test_host_ipc_off_by_default_keeps_private_dev_shm():
    manifest, _ = _build(_backend("device_plugin", 8, nodes=1), ["node-0"], gpu_count=4)
    pod = _pods(manifest)[0]
    assert "hostIPC" not in pod["spec"]
    dshm = [v for v in pod["spec"]["volumes"] if v["name"] == "dshm"][0]
    assert "emptyDir" in dshm


def test_compute_domain_channel_claimed_with_device_plugin_gpus():
    # ComputeDomain (IMEX) is decoupled from scheduling: a device_plugin GPU pod can
    # ALSO claim the compute-domain channel, so the nvidia.com/gpu limit and the DRA
    # channel claim coexist on the same pod (needed for NVLink KV transfer).
    be = _backend("device_plugin", 8, nodes=1)
    be._compute_domain_channel = "cd-chan"
    manifest, _ = _build(be, ["node-0"], gpu_count=4)
    pod = _pods(manifest)[0]
    claims = pod["spec"].get("resourceClaims", [])
    assert any(c.get("resourceClaimTemplateName") == "cd-chan" for c in claims)
    res = pod["spec"]["containers"][0]["resources"]
    assert res["limits"]["nvidia.com/gpu"] == "4"
    assert {"name": "compute-domain-channel"} in res["claims"]


def test_compute_domain_channel_not_claimed_by_cpu_only_pods():
    # Infra pods (no GPUs) must NOT claim the compute-domain channel. The DRA driver
    # publishes ONE single-allocation IMEX channel per node, so a CPU-only pod (e.g.
    # nats/etcd/frontend) claiming it starves the node's single channel and leaves
    # co-located pods Pending ("cannot allocate all claims"). Only GPU pods join the
    # IMEX domain.
    be = _backend("device_plugin", 8, nodes=1)
    be._compute_domain_channel = "cd-chan"
    manifest, _ = _build(be, ["node-0"], gpu_count=0)
    pod = _pods(manifest)[0]
    claims = pod["spec"].get("resourceClaims", [])
    assert not any(c.get("resourceClaimTemplateName") == "cd-chan" for c in claims)
    res = pod["spec"]["containers"][0].get("resources", {})
    assert all(
        c.get("name") != "compute-domain-channel" for c in res.get("claims", [])
    )


def test_dra_without_rdma_device_class_stays_gpu_only():
    # Default DRA path is unchanged: one GPU request, no NIC, no constraint, and
    # no affinity preamble (no RDMA provider configured).
    manifest, _ = _build(_backend("dra", 8, nodes=1), ["node-0"], gpu_count=2)
    devices = [
        i for i in manifest["items"] if i["kind"] == "ResourceClaimTemplate"
    ][0]["spec"]["spec"]["devices"]
    assert [r["name"] for r in devices["requests"]] == ["gpu"]
    assert "constraints" not in devices
    assert "_sflow_rdma_setup" not in _entrypoint(manifest)


# ---------------------------------------------------------------------------
# Scoped per-pod RDMA NIC slice: node-aware, aligned to the node-local GPU slot
# ---------------------------------------------------------------------------


def _rdma_backend(gpus_per_node=8, nics=8):
    """A single-node backend with the RDMA fast path enabled and ``nics`` NICs."""
    be = _backend("device_plugin", gpus_per_node, nodes=1)
    be._rdma_plan = RdmaPlan(
        provider="gke",
        enabled=True,
        nic_specs=tuple(
            (f"networking.gke.io.networks/rdma-{i}", f"mlx5_{i}") for i in range(nics)
        ),
        ipc_lock=True,
    )
    return be


def _rdma_nic_indices(pod):
    """The rdma-N NIC indices this pod requests, from its container limits."""
    limits = pod["spec"]["containers"][0].get("resources", {}).get("limits", {})
    out = []
    for key in limits:
        m = re.fullmatch(r"networking\.gke\.io\.networks/rdma-(\d+)", key)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def test_rdma_nic_window_aligns_to_node_local_gpu_slot():
    # A decode pod holding node-local GPU slot 4-7 must claim NICs rdma-4..7
    # (aligned to its GPUs), NOT rdma-0..3 -- otherwise it collides with prefill
    # pods packed onto GPUs 0-3 of the same node and never schedules.
    manifest, _ = _build(
        _rdma_backend(), ["node-0"], gpu_count=4, cuda_visible="4,5,6,7"
    )
    pod = _pods(manifest)[0]
    assert _rdma_nic_indices(pod) == [4, 5, 6, 7]
    env = {e["name"]: e["value"] for e in pod["spec"]["containers"][0]["env"]}
    assert env["NCCL_IB_HCA"] == "mlx5_4,mlx5_5,mlx5_6,mlx5_7"
    assert "UCX_NET_DEVICES" not in env


def test_colocated_prefill_decode_get_disjoint_nic_windows():
    # 2x TP2 prefill + 1x TP4 decode packed onto one 8-GPU/8-NIC node: each pod's
    # NIC window is aligned to its node-local GPU slot, so all three are disjoint
    # and fit (the bug packed every task onto rdma-0.. -> decode stayed Pending).
    be = _rdma_backend()
    p0 = _pods(_build(be, ["node-0"], gpu_count=2, cuda_visible="0,1")[0])[0]
    p1 = _pods(_build(be, ["node-0"], gpu_count=2, cuda_visible="2,3")[0])[0]
    d0 = _pods(_build(be, ["node-0"], gpu_count=4, cuda_visible="4,5,6,7")[0])[0]
    nics_p0 = set(_rdma_nic_indices(p0))
    nics_p1 = set(_rdma_nic_indices(p1))
    nics_d0 = set(_rdma_nic_indices(d0))
    assert nics_p0 == {0, 1}
    assert nics_p1 == {2, 3}
    assert nics_d0 == {4, 5, 6, 7}
    assert nics_p0.isdisjoint(nics_d0)
    assert nics_p1.isdisjoint(nics_d0)


def test_rdma_nic_window_falls_back_to_replica_index_without_gpu_slot():
    # Without a planner-reserved GPU slot (cuda_visible_devices=None, e.g. dry-run
    # with no allocation), same-task replicas still get disjoint NICs by offsetting
    # on the replica index.
    manifest, _ = _build(
        _rdma_backend(), ["node-0"], gpu_count=2,
        envs={"SFLOW_REPLICA_INDEX": "1"},
    )
    assert _rdma_nic_indices(_pods(manifest)[0]) == [2, 3]


def test_gke_keeps_static_pin_and_no_runtime_affinity_preamble():
    # GKE grants a fixed per-pod NIC subset (allow_runtime_affinity=False), so it
    # must keep the build-time per-pod pin and must NOT get the expose-all runtime
    # affinity preamble (which assumes the pod can see every node HCA).
    manifest, _ = _build(
        _rdma_backend(), ["node-0"], gpu_count=4, cuda_visible="4,5,6,7"
    )
    pod = _pods(manifest)[0]
    env = {e["name"]: e["value"] for e in pod["spec"]["containers"][0]["env"]}
    assert env["NCCL_IB_HCA"] == "mlx5_4,mlx5_5,mlx5_6,mlx5_7"  # static pin kept
    assert "UCX_NET_DEVICES" not in env
    assert _rdma_nic_indices(pod) == [4, 5, 6, 7]
    assert "_sflow_rdma_setup" not in _entrypoint(manifest)  # no affinity preamble


# ---------------------------------------------------------------------------
# Fail-fast on unrecoverable pod-start states (Unschedulable / ImagePullBackOff)
# ---------------------------------------------------------------------------


def _write_kubectl_stub(bin_dir: Path, body: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "kubectl"
    stub.write_text("#!/usr/bin/env bash\n" + body)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _run_wait_lines(
    tmp_path, monkeypatch, fake_process, *, stub_body, env=None, grace=2, sleep_secs=0
):
    """Run the generated pod-start wait bash with a stubbed ``kubectl`` on PATH."""
    # The autouse fake_process fixture blocks real subprocesses; let bash (and its
    # child kubectl stub, an OS-level process it doesn't intercept) run for real.
    fake_process.allow_unregistered(True)
    monkeypatch.setattr(k8s_shell, "UNRECOVERABLE_GRACE_POLLS", grace)
    monkeypatch.setattr(k8s_shell, "POLL_SLEEP_SECS", sleep_secs)
    bin_dir = tmp_path / "bin"
    _write_kubectl_stub(bin_dir, stub_body)
    lines = k8s_shell.wait_for_pod_ready_lines("pod/x", "", label="x")
    script = "set -euo pipefail\n" + "\n".join(lines)
    run_env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=run_env, timeout=30
    )


def test_wait_fails_fast_on_persistent_unschedulable(tmp_path, monkeypatch, fake_process):
    # A pod that stays Pending/Unschedulable past the grace window aborts the apply
    # step (exit 1) instead of waiting out the full budget -> the task fails fast
    # with the reason, rather than hanging until the readiness-probe timeout.
    stub = textwrap.dedent("""\
        if [ "$1" = "describe" ]; then echo DESCRIBE; exit 0; fi
        if [ "$1" = "get" ] && [ "$2" = "events" ]; then echo ""; exit 0; fi
        case "$*" in
          *"status.phase"*) echo "Pending";;
          *'PodScheduled")].reason'*) echo "Unschedulable";;
          *) echo "";;
        esac
    """)
    r = _run_wait_lines(tmp_path, monkeypatch, fake_process, stub_body=stub)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "Unschedulable" in r.stdout
    assert "aborting" in r.stdout.lower()


def test_wait_fails_fast_on_image_pull_backoff(tmp_path, monkeypatch, fake_process):
    stub = textwrap.dedent("""\
        if [ "$1" = "describe" ]; then echo DESCRIBE; exit 0; fi
        if [ "$1" = "get" ] && [ "$2" = "events" ]; then echo ""; exit 0; fi
        case "$*" in
          *"status.phase"*) echo "Pending";;
          *"waiting.reason"*) echo "ImagePullBackOff";;
          *) echo "";;
        esac
    """)
    r = _run_wait_lines(tmp_path, monkeypatch, fake_process, stub_body=stub)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "ImagePullBackOff" in r.stdout


def test_wait_tolerates_transient_state_then_running(tmp_path, monkeypatch, fake_process):
    # A brief Unschedulable that clears (the pod goes Running within the grace
    # window) must NOT fail the task -- the bad-state streak resets on recovery.
    counter = tmp_path / "n"
    stub = textwrap.dedent("""\
        if [ "$1" = "describe" ]; then echo DESCRIBE; exit 0; fi
        if [ "$1" = "get" ] && [ "$2" = "events" ]; then echo ""; exit 0; fi
        case "$*" in
          *"status.phase"*)
            n=$(cat "$COUNTER" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$COUNTER";
            if [ "$n" -le 1 ]; then echo "Pending"; else echo "Running"; fi;;
          *'PodScheduled")].reason'*) echo "Unschedulable";;
          *) echo "";;
        esac
    """)
    r = _run_wait_lines(
        tmp_path, monkeypatch, fake_process, stub_body=stub,
        env={"COUNTER": str(counter)}, grace=2,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "final phase=Running" in r.stdout


def test_apply_shell_contains_fail_fast_tokens():
    _, shell = _build(_backend("dra", 8), ["node-0"], gpu_count=2)
    assert "Unschedulable" in shell
    assert "ImagePullBackOff" in shell
    assert "CrashLoopBackOff" in shell
    assert "exit 1" in shell


def test_pods_get_ram_backed_dev_shm_by_default():
    # K8s' 64Mi /dev/shm segfaults MPI/NCCL; task pods get a RAM-backed tmpfs.
    manifest, _ = _build(_backend("device_plugin", 8), ["node-0"], gpu_count=2)
    pod = _pods(manifest)[0]
    dshm = [v for v in pod["spec"]["volumes"] if v["name"] == "dshm"][0]
    assert dshm["emptyDir"] == {"medium": "Memory"}
    assert any(
        mt["mountPath"] == "/dev/shm"
        for mt in pod["spec"]["containers"][0]["volumeMounts"]
    )


def test_shm_size_config_caps_dev_shm():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1", shm_size="16Gi"))
    op.apply_backend_context(
        backend=_backend("device_plugin", 8), assigned_nodes=["node-0"],
        artifacts=[], gpu_count=0,
    )
    shell = op.build_command(task_name="t", script=["run"], envs={}).as_list()[-1]
    body = shell.split(_MARK, 1)[1].split("\n" + _MARK, 1)[0]
    pod = _pods(json.loads(body.split("\n", 1)[1]))[0]
    dshm = [v for v in pod["spec"]["volumes"] if v["name"] == "dshm"][0]
    assert dshm["emptyDir"] == {"medium": "Memory", "sizeLimit": "16Gi"}


def test_multiple_fs_artifacts_under_same_pvc_dedup_to_one_volume():
    be = _backend(
        "device_plugin", 8,
        volumes=[{"name": "store", "claim": "pvc", "mount_path": "/data"}],
    )
    a1 = Artifact(name="A", uri="fs:///data/a", path=Path("/data/a"))
    a2 = Artifact(name="B", uri="fs:///data/b", path=Path("/data/b"))
    manifest, _ = _build(
        be, ["node-0"], gpu_count=0, script=["run /data/a /data/b"],
        artifacts=[a1, a2],
    )
    pvc = [v for v in _pods(manifest)[0]["spec"]["volumes"] if v.get("persistentVolumeClaim")]
    assert len(pvc) == 1


# ---------------------------------------------------------------------------
# Manifest persistence: the rendered List manifest is saved to the task output
# dir as readable YAML (auditability), actual-run-only and best-effort.
# ---------------------------------------------------------------------------


def test_build_command_persists_manifest_yaml_when_output_dir_exists(tmp_path):
    # On an actual run SFLOW_TASK_OUTPUT_DIR exists; the rendered List manifest is
    # also written next to <task>.log as readable YAML for auditability.
    manifest, _ = _build(
        _backend("dra", 8), ["node-0"], gpu_count=2,
        envs={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
    )
    saved = tmp_path / "t.k8s.yaml"
    assert saved.exists()
    text = saved.read_text()
    assert text.startswith("# Auto-generated by sflow")
    # The file round-trips to exactly the manifest applied via `kubectl apply -f -`.
    loaded = yaml.safe_load(text)
    assert loaded == manifest
    kinds = [i["kind"] for i in loaded["items"]]
    assert "ConfigMap" in kinds and "Pod" in kinds


def test_build_command_does_not_persist_manifest_without_output_dir(tmp_path):
    # Dry-run-like: SFLOW_TASK_OUTPUT_DIR points at a dir that does not exist, so
    # nothing is written (and the command is still built without error).
    missing = tmp_path / "does-not-exist"
    manifest, shell = _build(
        _backend("dra", 8), ["node-0"], gpu_count=2,
        envs={"SFLOW_TASK_OUTPUT_DIR": str(missing)},
    )
    assert not missing.exists()
    assert not (missing / "t.k8s.yaml").exists()
    # The command was still rendered normally.
    assert manifest["kind"] == "List"
    assert _MARK in shell


# ---------------------------------------------------------------------------
# network_fallback_status: surface the in-pod RDMA->TCP fallback from <task>.log
# ---------------------------------------------------------------------------


def _task_with_out(name, out_dir):
    from sflow.core.task import Task

    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    task = Task(
        name=name,
        logger=logging.getLogger(f"sflow.task.{name}"),
        operator=op,
        script=[f"echo {name}"],
    )
    if out_dir is not None:
        task.envs["SFLOW_TASK_OUTPUT_DIR"] = str(out_dir)
    return task, op


def test_network_fallback_status_detects_tcp_from_task_log(tmp_path):
    task_dir = tmp_path / "decode_server_0"
    task_dir.mkdir()
    (task_dir / "decode_server_0.log").write_text(
        "[pod/x/x] [sflow-rdma] WARNING RDMA requested but unusable in pod "
        "(no InfiniBand/RoCE port is ACTIVE (all ports DOWN)): using TCP "
        "for NCCL (NCCL_IB_DISABLE=1); UCX device selection left to the library\n"
    )
    task, op = _task_with_out("decode_server_0", task_dir)
    status = op.network_fallback_status(task)
    assert status is not None and status.degraded_to_tcp
    assert "all ports DOWN" in status.reason


def test_network_fallback_status_none_without_marker(tmp_path):
    task_dir = tmp_path / "s"
    task_dir.mkdir()
    (task_dir / "s.log").write_text("regular vllm output\n")
    task, op = _task_with_out("s", task_dir)
    assert op.network_fallback_status(task) is None


def test_network_fallback_status_none_without_output_dir(tmp_path):
    task, op = _task_with_out("s", None)
    assert op.network_fallback_status(task) is None


def test_network_fallback_status_none_when_log_missing(tmp_path):
    # Output dir set but no <task>.log yet -> best-effort None (never raises).
    task, op = _task_with_out("s", tmp_path / "nolog")
    assert op.network_fallback_status(task) is None


# ---------------------------------------------------------------------------
# Merge-pod mode: several co-located GPU tasks share one pod (union of GPUs)
# ---------------------------------------------------------------------------


def _member_task(name, script, envs, cvd):
    from sflow.core.task import Task, TaskStatus

    t = Task(
        name=name,
        logger=logging.getLogger(f"test.{name}"),
        operator=object(),  # unused: the leader operator reads member attrs only
        status=TaskStatus.INITIATED,
        script=list(script),
        envs=dict(envs),
    )
    t.merge_cuda_visible_devices = cvd
    return t


def _merged_build(backend, *, scheduling_gpu=True):
    """Build a merge-pod plan for decode(4 GPU)+prefill(2 GPU) -> union 6 on node-0."""
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=["node-0"], artifacts=[],
        gpu_count=4, cuda_visible_devices="0,1,2,3",
    )
    decode = _member_task(
        "decode", ["run-decode"],
        {"MODEL": "m", "SFLOW_TASK_OUTPUT_DIR": "/out/decode"}, "0,1,2,3",
    )
    prefill = _member_task(
        "prefill", ["run-prefill"],
        {"MODEL": "m", "ROLE": "prefill", "SFLOW_TASK_OUTPUT_DIR": "/out/prefill"}, "4,5",
    )
    op.apply_merge_group(members=[decode, prefill], union_gpus=6)
    cmd = op.build_command(task_name="decode", script=["run-decode"], envs=decode.envs)
    shell = cmd.as_list()[-1]
    after = shell.split(_MARK, 1)[1]
    body = after.split("\n" + _MARK, 1)[0]
    manifest = json.loads(body.split("\n", 1)[1])
    return manifest, shell, op


def test_merge_pod_single_pod_with_union_gpus():
    manifest, shell, _ = _merged_build(_backend("dra", 8))
    pods = _pods(manifest)
    # Pod is named after the merged members (not the leader alone) so it's clear
    # in `kubectl get pods` that it runs several tasks.
    assert len(pods) == 1
    assert pods[0]["metadata"]["name"].startswith("merged-decode-prefill-")
    # Union GPU request: the one container's RCT asks for 6 (4 decode + 2 prefill).
    rct = [i for i in manifest["items"] if i["kind"] == "ResourceClaimTemplate"][0]
    assert rct["spec"]["spec"]["devices"]["requests"][0]["exactly"]["count"] == 6
    pod = pods[0]
    assert pod["spec"]["containers"][0]["resources"]["claims"] == [{"name": "gpu"}]
    # Pinned to the shared node; placeholder handed off once.
    assert pod["spec"]["nodeSelector"]["kubernetes.io/hostname"] == "node-0"
    assert "res-0" in shell


def test_merge_pod_device_plugin_union_limit():
    manifest, _, _ = _merged_build(_backend("device_plugin", 8))
    pod = _pods(manifest)[0]
    assert pod["spec"]["containers"][0]["resources"]["limits"] == {"nvidia.com/gpu": "6"}


def test_merge_pod_configmap_has_launcher_and_member_scripts():
    manifest, _, _ = _merged_build(_backend("dra", 8))
    cm = [
        i for i in manifest["items"]
        if i["kind"] == "ConfigMap" and "entrypoint.sh" in i.get("data", {})
    ][0]
    data = cm["data"]
    assert data["merge_decode.sh"] == "run-decode"
    assert data["merge_prefill.sh"] == "run-prefill"
    launcher = data["entrypoint.sh"]
    # Each member launched with its packed CUDA_VISIBLE_DEVICES + tagged output.
    assert "_sflow_run decode 0,1,2,3 /sflow/merge_decode.sh" in launcher
    assert "_sflow_run prefill 4,5 /sflow/merge_prefill.sh" in launcher
    assert "[[sflow-mux:" in launcher
    assert 'export CUDA_VISIBLE_DEVICES="$_cvd"' in launcher


def test_merge_pod_per_member_env_secrets_not_envfrom():
    manifest, shell, _ = _merged_build(_backend("dra", 8))
    container = _pods(manifest)[0]["spec"]["containers"][0]
    # No container-wide envFrom (would collide across members); env is per-member.
    assert "envFrom" not in container
    mounts = {m["mountPath"] for m in container["volumeMounts"]}
    assert "/sflow/menv/decode" in mounts and "/sflow/menv/prefill" in mounts
    # The apply step creates one env Secret per member (named off the merged pod)
    # from prefix-namespaced vars.
    pod_base = _pods(manifest)[0]["metadata"]["name"]
    assert f"{pod_base}-menv-0" in shell and f"{pod_base}-menv-1" in shell
    assert "${SFMERGE0__MODEL-}" in shell
    assert "${SFMERGE1__ROLE-}" in shell


def test_merge_pod_plan_demux_paths_and_launcher_env():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend("dra", 8), assigned_nodes=["node-0"], artifacts=[],
        gpu_count=4, cuda_visible_devices="0,1,2,3",
    )
    decode = _member_task(
        "decode", ["run-decode"],
        {"MODEL": "m", "SFLOW_TASK_OUTPUT_DIR": "/out/decode"}, "0,1",
    )
    prefill = _member_task(
        "prefill", ["run-prefill"],
        {"ROLE": "prefill", "SFLOW_TASK_OUTPUT_DIR": "/out/prefill"}, "2,3",
    )
    op.apply_merge_group(members=[decode, prefill], union_gpus=4)
    plan = op._build_execution_plan(
        task_name="decode", script=["run-decode"], envs=decode.envs
    )
    # One pod named after the merged members.
    assert len(plan.pod_refs) == 1
    assert plan.pod_refs[0].startswith("pod/merged-decode-prefill-")
    pod_base = plan.pod_refs[0].split("/", 1)[1]
    # Each member's stream is demuxed to its own <task>.log; leader is the default.
    assert plan.merge_tag_paths == {
        "decode": "/out/decode/decode.log",
        "prefill": "/out/prefill/prefill.log",
    }
    assert plan.task_log_path == "/out/decode/decode.log"
    # Launcher env is prefix-namespaced so members' identical keys don't collide.
    assert plan.merge_launcher_env["SFMERGE0__MODEL"] == "m"
    assert plan.merge_launcher_env["SFMERGE1__ROLE"] == "prefill"
    # Per-member env Secrets (named off the merged pod) are cleaned up with the task.
    assert f"secret/{pod_base}-menv-0" in plan.cleanup_refs
    assert f"secret/{pod_base}-menv-1" in plan.cleanup_refs


def test_merge_pod_host_ipc_propagated():
    backend = _backend("dra", 8)
    backend._host_ipc = True
    manifest, _, _ = _merged_build(backend)
    pod = _pods(manifest)[0]
    assert pod["spec"]["hostIPC"] is True


@pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
def test_merge_pod_launcher_and_apply_are_valid_bash(fake_process):
    # The merged launcher (entrypoint.sh) and apply command are generated shell;
    # `bash -n` parses them without executing to catch quoting/syntax breakage.
    fake_process.allow_unregistered(True)  # let the real bash run for the check
    from sflow.plugins.operators._k8s_shell import (
        build_merged_apply_command,
        merged_launcher_lines,
    )

    launcher = "\n".join(
        merged_launcher_lines(
            [
                ("decode", "0,1,2,3", "/sflow/merge_decode.sh", "/sflow/menv/decode/envsh"),
                ("prefill", "4,5", "/sflow/merge_prefill.sh", "/sflow/menv/prefill/envsh"),
            ],
            preamble_lines=["echo preamble"],
        )
    )
    apply = build_merged_apply_command(
        manifest_json='{"kind":"List"}',
        ns_seg=" --namespace ns",
        pod_name="decode",
        member_env_secrets=[
            ("decode-menv-0", [("MODEL", "SFMERGE0__MODEL")]),
            ("decode-menv-1", [("ROLE", "SFMERGE1__ROLE")]),
        ],
        handoff_delete_pods=["res-0"],
        allocation_id="abc",
    ).as_list()[-1]
    for text in (launcher, apply):
        proc = subprocess.run(
            [shutil.which("bash"), "-n"], input=text, capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
