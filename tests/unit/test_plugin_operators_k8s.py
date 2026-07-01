# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single ``k8s`` operator: DRA vs device-plugin GPU requests,
multi-node pod-set + env wiring, ConfigMap script mount, hostname pinning, and
the create-before-destroy handoff gating (GPU tasks only)."""

import json
import logging
from pathlib import Path

import pytest
import yaml

from sflow.core.artifact import Artifact
from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig
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
    artifacts=(),
):
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=list(assigned_nodes),
        artifacts=list(artifacts), gpu_count=gpu_count,
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
    # When the backend detected RDMA, every task pod gets the IB/NCCL/UCX/gloo env
    # so all libs use the fast NICs + the routable control interface.
    be = _backend("device_plugin", 8)
    be._rdma_env = {  # simulate reservation-time detection
        "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1",
        "NCCL_IB_HCA": "mlx5_0,mlx5_1",
        "NCCL_SOCKET_IFNAME": "eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
    }
    manifest, _ = _build(be, ["node-0"], gpu_count=2)
    env = {
        e["name"]: e["value"]
        for e in _pods(manifest)[0]["spec"]["containers"][0].get("env", [])
    }
    assert env.get("UCX_NET_DEVICES") == "mlx5_0:1,mlx5_1:1"
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
