# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single ``k8s`` operator: DRA vs device-plugin GPU requests,
multi-node pod-set + env wiring, ConfigMap script mount, hostname pinning, and
the create-before-destroy handoff gating (GPU tasks only)."""

import asyncio
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
from sflow.plugins.k8s.rdma import RdmaPlan
from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig
from sflow.plugins.k8s import shell as k8s_shell
from sflow.plugins.operators.k8s import K8sOperator, K8sOperatorConfig

_MARK = "SFLOW_K8S_MANIFEST"


# CPU-request policy is opt-in and UNSET by default in the product, so the fixture
# mirrors that (cpu_per_gpu/cpu_request=None -> BestEffort). Tests that exercise the
# CPU-request path pass cpu_per_gpu / cpu_request explicitly.
def _backend(scheduling="dra", gpus_per_node=8, nodes=2, namespace="default",
             volumes=None, cpu_per_gpu=None, cpu_request=None,
             collect_max_file_size=None):
    backend = KubernetesBackend(
        KubernetesBackendConfig(
            name="k8s",
            type="kubernetes",
            namespace=namespace,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            scheduling=scheduling,
            volumes=volumes,
            cpu_per_gpu=cpu_per_gpu,
            cpu_request=cpu_request,
            collect_max_file_size=collect_max_file_size,
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
    artifacts=(), cuda_visible=None, op_resources=None,
):
    op = K8sOperator(
        K8sOperatorConfig(name="op", image="img:1", **(op_resources or {}))
    )
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
# NVIDIA driver discoverability (libcuda.so.1) on every GPU pod
# ---------------------------------------------------------------------------


def test_gpu_pod_prepends_nvidia_driver_to_ld_library_path():
    # GKE bind-mounts the host driver into /usr/local/nvidia but does not add it to
    # the loader path. Without this, a single-node GPU pod can't load libcuda.so.1
    # and vLLM/torch fail with "Failed to infer device type". Applies to every GPU
    # pod, not just multi-node RDMA ones.
    manifest, _ = _build(_backend("device_plugin", 8), ["node-0"], gpu_count=4)
    entry = _entrypoint(manifest)
    assert "export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}" in entry
    assert "export PATH=/usr/local/nvidia/bin:${PATH:-}" in entry
    # The preamble runs before the task's own script.
    assert entry.index("/usr/local/nvidia/lib64") < entry.index("run")


def test_cpu_only_pod_has_no_nvidia_driver_preamble():
    # GPU-less pods (frontend/etcd/nats) must not get the driver env.
    manifest, _ = _build(_backend("device_plugin", 8), ["node-0"], gpu_count=None)
    assert "/usr/local/nvidia" not in _entrypoint(manifest)


# ---------------------------------------------------------------------------
# multi-node pod set + env wiring
# ---------------------------------------------------------------------------


def test_multinode_splits_pods_and_wires_env():
    manifest, shell = _build(_backend("dra", 8, nodes=2), ["node-0", "node-1"], gpu_count=16)
    pods = _pods(manifest)
    # Pod names are allocation-scoped ("abc" in tests) so parallel runs don't collide.
    assert [p["metadata"]["name"] for p in pods] == ["t-abc-0", "t-abc-1"]
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
    # The task's script is mounted as the entrypoint (a GPU pod also gets the
    # NVIDIA driver preamble prepended, so match the tail rather than the whole).
    assert cm["data"]["entrypoint.sh"].endswith("echo hi\npython run.py")
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
    # Default (opt-in) CPU policy is unset -> no cpu request, no GPU claim/limit, and
    # no pod-level resourceClaims for a CPU-only task.
    assert "resources" not in pod["spec"]["containers"][0]
    assert "resourceClaims" not in pod["spec"]
    # No placeholder deletion -> the node's GPUs stay reserved for GPU tasks.
    assert "res-0" not in shell


# ---------------------------------------------------------------------------
# CPU/memory requests (requests-only policy: no default CPU/memory limits)
# ---------------------------------------------------------------------------


def test_gpu_pod_cpu_request_scales_with_gpus():
    # GPU pod: requests.cpu = cpu_per_gpu (8) x per-pod GPUs; CPU goes to requests
    # so the nvidia.com/gpu limit is untouched and no cpu limit is added.
    manifest, _ = _build(
        _backend("device_plugin", 8, nodes=1, cpu_per_gpu=8), ["node-0"], gpu_count=4
    )
    res = _pods(manifest)[0]["spec"]["containers"][0]["resources"]
    assert res["requests"]["cpu"] == "32"  # 8 * 4
    assert res["limits"] == {"nvidia.com/gpu": "4"}  # no cpu limit added


def test_cpu_only_pod_gets_backend_cpu_request():
    manifest, _ = _build(_backend("dra", 8, cpu_request=4), ["node-0"], gpu_count=None)
    res = _pods(manifest)[0]["spec"]["containers"][0]["resources"]
    assert res["requests"]["cpu"] == "4"
    assert "limits" not in res and "claims" not in res


def test_backend_cpu_policy_is_configurable():
    be = _backend("device_plugin", 8, nodes=1, cpu_per_gpu=6, cpu_request=2)
    gpu_manifest, _ = _build(be, ["node-0"], gpu_count=2)
    gpu_res = _pods(gpu_manifest)[0]["spec"]["containers"][0]["resources"]
    assert gpu_res["requests"]["cpu"] == "12"  # 6 * 2
    cpu_manifest, _ = _build(be, ["node-0"], gpu_count=None)
    cpu_res = _pods(cpu_manifest)[0]["spec"]["containers"][0]["resources"]
    assert cpu_res["requests"]["cpu"] == "2"


def test_operator_resources_override_wins():
    # Explicit operator cpu/memory fields beat the computed 8xGPU default and can add
    # the (otherwise-absent) CPU/memory limits.
    manifest, _ = _build(
        _backend("device_plugin", 8, nodes=1), ["node-0"], gpu_count=4,
        op_resources={
            "cpu": 16, "memory": "32Gi", "cpu_limit": 24, "memory_limit": "48Gi",
        },
    )
    res = _pods(manifest)[0]["spec"]["containers"][0]["resources"]
    assert res["requests"] == {"cpu": "16", "memory": "32Gi"}
    assert res["limits"]["cpu"] == "24"
    assert res["limits"]["memory"] == "48Gi"
    assert res["limits"]["nvidia.com/gpu"] == "4"  # GPU limit still present


def test_cpu_request_disabled_when_backend_zero():
    # cpu_per_gpu/cpu_request = 0 opts out (no CPU request injected).
    manifest, _ = _build(
        _backend("dra", 8, cpu_per_gpu=0, cpu_request=0), ["node-0"], gpu_count=None
    )
    res = _pods(manifest)[0]["spec"]["containers"][0].get("resources", {})
    assert "requests" not in res


def test_no_cpu_request_when_backend_policy_unset():
    # Opt-in policy: backend leaves cpu_per_gpu/cpu_request unset -> NO cpu request on
    # either pod class (BestEffort), matching a stock pod manifest.
    be = _backend("device_plugin", 8, nodes=1, cpu_per_gpu=None, cpu_request=None)
    gpu_manifest, _ = _build(be, ["node-0"], gpu_count=4)
    gpu_res = _pods(gpu_manifest)[0]["spec"]["containers"][0].get("resources", {})
    assert "requests" not in gpu_res  # GPU pod: no cpu request
    assert gpu_res.get("limits") == {"nvidia.com/gpu": "4"}  # GPU limit still emitted
    cpu_manifest, _ = _build(be, ["node-0"], gpu_count=None)
    cpu_res = _pods(cpu_manifest)[0]["spec"]["containers"][0].get("resources", {})
    assert "requests" not in cpu_res  # CPU-only pod: no cpu request


def test_operator_cpu_overrides_when_backend_policy_unset():
    # Even with the backend policy unset, an explicit operator ``cpu`` still emits.
    be = _backend("device_plugin", 8, nodes=1, cpu_per_gpu=None, cpu_request=None)
    manifest, _ = _build(be, ["node-0"], gpu_count=4, op_resources={"cpu": 10})
    res = _pods(manifest)[0]["spec"]["containers"][0]["resources"]
    assert res["requests"]["cpu"] == "10"


def test_operator_cpu_zero_opts_out_over_backend_policy():
    # Operator ``cpu: 0`` is an explicit opt-out (no cpu request), consistent with the
    # backend knobs -- it wins over an opted-in backend cpu_per_gpu policy.
    be = _backend("device_plugin", 8, nodes=1, cpu_per_gpu=8)
    manifest, _ = _build(be, ["node-0"], gpu_count=4, op_resources={"cpu": 0})
    res = _pods(manifest)[0]["spec"]["containers"][0].get("resources", {})
    assert "requests" not in res
    assert res.get("limits") == {"nvidia.com/gpu": "4"}  # GPU limit still emitted


# ---------------------------------------------------------------------------
# SFLOW_* output dir emptyDir mount for K8s (the driver host dir isn't reachable
# in a pod) + node-local output collection back to the driver
# ---------------------------------------------------------------------------


def test_k8s_mounts_emptydir_at_output_dir_and_stages_output():
    from sflow.plugins.operators.k8s_operator import (
        _SFLOW_COLLECT_DONE,
        _SFLOW_COLLECT_READY_MARKER,
    )

    envs = {
        "SFLOW_WORKSPACE_DIR": "/host/proj",
        "SFLOW_OUTPUT_DIR": "/host/proj/sflow_output",
        "SFLOW_WORKFLOW_OUTPUT_DIR": "/host/proj/sflow_output/run-9",
        "SFLOW_TASK_OUTPUT_DIR": "/host/proj/sflow_output/run-9/bench",
        "SFLOW_TASK_RESULT_FILE": "/host/proj/sflow_output/run-9/bench/result.json",
    }
    manifest, _ = _build(
        _backend("device_plugin", 8, nodes=1), ["node-0"], gpu_count=4,
        task="bench", envs=envs,
    )
    pod = _pods(manifest)[0]["spec"]
    c = pod["containers"][0]
    # Writable emptyDir mounted at the resolved SFLOW_OUTPUT_DIR (env is NOT remapped).
    assert any(
        m["mountPath"] == "/host/proj/sflow_output" and m["name"] == "sflow-scratch"
        for m in c["volumeMounts"]
    )
    assert any(
        v["name"] == "sflow-scratch" and "emptyDir" in v for v in pod["volumes"]
    )
    assert any(ic["name"] == "sflow-ensure-writable" for ic in pod["initContainers"])
    ep = _entrypoint(manifest)
    # No env remap/export; just mkdir of the per-task subdir (driver path via envFrom).
    assert 'mkdir -p "$SFLOW_TASK_OUTPUT_DIR"' in ep
    assert "export SFLOW_TASK_OUTPUT_DIR" not in ep
    # kubectl-cp handoff: stage <=cap files into ONE tar.gz (size-guarded, skip-empty),
    # signal readiness (NOT the file bytes), then wait for the driver's done-sentinel.
    assert "tar czf" in ep and "-size" in ep
    assert "wc -l" in ep and "-gt 0" in ep
    # Collection scans the WHOLE workflow output dir (task dir is a subtree) so
    # workflow-level files (e.g. aiperf_concurrency_*) are captured too, not just the
    # per-task dir.
    assert 'cd "$SFLOW_WORKFLOW_OUTPUT_DIR"' in ep
    assert 'find "$SFLOW_WORKFLOW_OUTPUT_DIR" -type f -size' in ep  # large-file warn
    assert _SFLOW_COLLECT_READY_MARKER in ep
    assert _SFLOW_COLLECT_DONE in ep and "while [ ! -f" in ep
    assert "nothing to collect" in ep
    # Collection runs from an EXIT trap (armed BEFORE the user script) so it fires even
    # when the recipe calls `exit`; xtrace+errexit are silenced so a recipe's set -x/-e
    # can't flood the log with the wait loop or abort the block.
    assert "trap _sflow_collect EXIT" in ep
    assert "{ __sflow_rc=$?; set +ex; } 2>/dev/null" in ep
    # No base64 blob is ever emitted to the log anymore.
    assert "base64" not in ep


def _make_tgz(files: dict) -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_k8s_unpack_collected_tar_extracts_files(tmp_path):
    from sflow.plugins.operators.k8s_operator import _unpack_collected_tar

    dest = tmp_path / "bench"
    dest.mkdir()
    blob = _make_tgz(
        {"result.json": b'{"tps": 1.5}', "sub/metrics.csv": b"a,b\n1,2\n"}
    )
    extracted, skipped = _unpack_collected_tar(blob, str(dest))
    assert sorted(extracted) == ["result.json", "sub/metrics.csv"]
    assert skipped == []
    assert (dest / "result.json").read_bytes() == b'{"tps": 1.5}'
    assert (dest / "sub" / "metrics.csv").read_bytes() == b"a,b\n1,2\n"


def test_k8s_unpack_collected_tar_does_not_overwrite(tmp_path):
    from sflow.plugins.operators.k8s_operator import (
        _collect_summary_line,
        _unpack_collected_tar,
    )

    dest = tmp_path / "bench"
    dest.mkdir()
    # A file already on the driver (same basename) must be kept -- the pod's node-local
    # dir shares the path/basenames but is a different host.
    (dest / "result.json").write_text("DRIVER-ORIGINAL")
    blob = _make_tgz({"result.json": b"POD-COPY", "metrics.csv": b"a,b\n1,2\n"})
    extracted, skipped = _unpack_collected_tar(blob, str(dest))
    assert extracted == ["metrics.csv"]
    assert skipped == ["result.json"]
    assert (dest / "result.json").read_text() == "DRIVER-ORIGINAL"  # not clobbered
    assert (dest / "metrics.csv").read_bytes() == b"a,b\n1,2\n"
    summary = _collect_summary_line(str(dest), extracted, skipped)
    assert "collected 1 file(s)" in summary
    assert "kept 1 existing driver file(s) (not overwritten): result.json" in summary


def test_k8s_wait_for_marker_found(tmp_path):
    import asyncio

    from sflow.plugins.operators.k8s_operator import (
        _SFLOW_COLLECT_READY_MARKER,
        _wait_for_marker,
    )

    log = tmp_path / "bench.log"
    log.write_text(f"[pod/bench] hello\n[pod/bench] staged {_SFLOW_COLLECT_READY_MARKER}\n")
    found = asyncio.run(
        asyncio.wait_for(
            _wait_for_marker(str(log), _SFLOW_COLLECT_READY_MARKER.encode(), interval=0.01),
            timeout=2,
        )
    )
    assert found is True


def test_k8s_collect_exclude_rel_lists_injected_file_artifacts():
    from sflow.plugins.operators.k8s_operator import _collect_exclude_rel

    class _Art:
        def __init__(self, uri, path, content):
            self.uri, self.path, self.content = uri, path, content

    wf = "/host/out/run-9"
    arts = [
        _Art("file://prefill_config.yaml", f"{wf}/prefill_config.yaml", "x"),  # under wf
        _Art("file://sub/decode.yaml", f"{wf}/sub/decode.yaml", "y"),          # nested
        _Art("fs:///models/Qwen", "/models/Qwen", None),                       # fs:// -> skip
        _Art("file://abs.yaml", "/etc/abs.yaml", "z"),                         # outside wf
        _Art("file://noc.yaml", f"{wf}/noc.yaml", None),                       # no content
    ]
    # Only injected file:// artifacts that live UNDER the workflow dir are excluded.
    assert _collect_exclude_rel(arts, wf) == ["./prefill_config.yaml", "./sub/decode.yaml"]
    assert _collect_exclude_rel(arts, None) == []


def test_k8s_collect_trap_excludes_injected_artifacts():
    from sflow.plugins.operators.k8s_operator import _sflow_output_collect_trap

    ep = _sflow_output_collect_trap(10 * 1024 * 1024, 120, ["./prefill_config.yaml"])
    # The shared artifact is dropped from BOTH the count and stage finds.
    assert "! -path ./prefill_config.yaml" in ep
    # No exclusions -> no ! -path clause at all.
    assert "! -path" not in _sflow_output_collect_trap(10 * 1024 * 1024, 120, [])


def test_k8s_collection_disabled_when_size_zero():
    from sflow.plugins.operators.k8s_operator import _SFLOW_COLLECT_READY_MARKER

    manifest, _ = _build(
        _backend("device_plugin", 8, nodes=1), ["node-0"], gpu_count=4,
        op_resources={"collect_max_file_size": 0},
        envs={
            "SFLOW_OUTPUT_DIR": "/host/out",
            "SFLOW_TASK_OUTPUT_DIR": "/host/out/run/t",
            "SFLOW_WORKFLOW_OUTPUT_DIR": "/host/out/run",
        },
    )
    ep = _entrypoint(manifest)
    assert 'mkdir -p "$SFLOW_TASK_OUTPUT_DIR"' in ep  # dir setup still happens
    assert _SFLOW_COLLECT_READY_MARKER not in ep  # but no collection epilogue


def test_backend_collect_max_file_size_inherited_when_operator_unset():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend("device_plugin", 8, nodes=1, collect_max_file_size="2Mi"),
        assigned_nodes=["node-0"], artifacts=[], gpu_count=0,
    )
    assert op._collect_max_file_bytes == 2 * 2**20


def test_operator_collect_max_file_size_overrides_backend():
    op = K8sOperator(
        K8sOperatorConfig(name="op", image="img:1", collect_max_file_size="1Mi")
    )
    op.apply_backend_context(
        backend=_backend("device_plugin", 8, nodes=1, collect_max_file_size="9Mi"),
        assigned_nodes=["node-0"], artifacts=[], gpu_count=0,
    )
    assert op._collect_max_file_bytes == 1 * 2**20  # operator override wins


def test_collect_max_file_size_default_when_neither_set():
    from sflow.plugins.operators.k8s_operator import _SFLOW_COLLECT_MAX_FILE_BYTES

    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend("device_plugin", 8, nodes=1),  # no cap set
        assigned_nodes=["node-0"], artifacts=[], gpu_count=0,
    )
    assert op._collect_max_file_bytes == _SFLOW_COLLECT_MAX_FILE_BYTES


def test_merged_keeps_member_sflow_dirs_unchanged():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend("dra", 8), assigned_nodes=["node-0"], artifacts=[],
        gpu_count=4, cuda_visible_devices="0,1,2,3",
    )
    decode = _member_task(
        "decode", ["run-decode"],
        {"SFLOW_OUTPUT_DIR": "/out",
         "SFLOW_TASK_OUTPUT_DIR": "/out/run/decode",
         "SFLOW_WORKFLOW_OUTPUT_DIR": "/out/run"},
        "0,1,2,3",
    )
    op.apply_merge_group(members=[decode], union_gpus=4)
    plan = op._build_merged_execution_plan(task_name="decode", envs=decode.envs)
    env = plan.merge_launcher_env or {}
    tod = [v for k, v in env.items() if k.endswith("__SFLOW_TASK_OUTPUT_DIR")]
    # Env is NOT remapped: the member keeps its driver-host path (a writable emptyDir is
    # mounted at SFLOW_OUTPUT_DIR so it's valid + writable in the pod).
    assert tod == ["/out/run/decode"]
    assert plan.merge_tag_paths["decode"] == "/out/run/decode/decode.log"


def test_merged_pod_has_scratch_mount_at_output_dir():
    manifest, _, _ = _merged_build(_backend("dra", 8))
    pod = _pods(manifest)[0]["spec"]
    # One shared emptyDir at the resolved SFLOW_OUTPUT_DIR covers every member's dir.
    assert any(
        m["mountPath"] == "/out" and m["name"] == "sflow-scratch"
        for m in pod["containers"][0]["volumeMounts"]
    )
    assert any(v["name"] == "sflow-scratch" for v in pod["volumes"])


def test_parse_size_bytes():
    from sflow.plugins.operators.k8s_operator import _parse_size_bytes

    assert _parse_size_bytes(None, 999) == 999
    assert _parse_size_bytes(1234, 0) == 1234
    assert _parse_size_bytes("10Mi", 0) == 10 * 2**20
    assert _parse_size_bytes("500K", 0) == 500 * 10**3
    assert _parse_size_bytes("2G", 0) == 2 * 10**9
    assert _parse_size_bytes("garbage", 42) == 42
    assert _parse_size_bytes(-5, 0) == 0


def test_k8s_no_scratch_or_mkdir_when_pass_envs_false():
    manifest, _ = _build(
        _backend("device_plugin", 8, nodes=1), ["node-0"], gpu_count=4,
        op_resources={"pass_envs": False},
        envs={"SFLOW_OUTPUT_DIR": "/host/out",
              "SFLOW_TASK_OUTPUT_DIR": "/host/out/t"},
    )
    pod = _pods(manifest)[0]["spec"]
    # No env Secret -> no output emptyDir mount and no mkdir preamble.
    assert not any(
        m["name"] == "sflow-scratch"
        for m in pod["containers"][0]["volumeMounts"]
    )
    assert 'mkdir -p "$SFLOW_TASK_OUTPUT_DIR"' not in _entrypoint(manifest)


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


def test_apply_then_handoff_ordering():
    """create-before-destroy applies then deletes; destroy-before-create reverses it."""
    from sflow.plugins.k8s.shell import apply_then_handoff_lines

    apply = "cat <<X | kubectl apply -f -"
    cbd = "\n".join(apply_then_handoff_lines(apply, ["res-0"], " --namespace ns"))
    assert cbd.index("kubectl apply -f -") < cbd.index("kubectl delete pod")
    dbc = "\n".join(
        apply_then_handoff_lines(
            apply, ["res-0"], " --namespace ns", handoff_before_apply=True
        )
    )
    assert dbc.index("kubectl delete pod") < dbc.index("kubectl apply -f -")
    # No placeholders -> only the apply line (no delete) in either mode.
    assert apply_then_handoff_lines(apply, [], " --namespace ns") == [apply]
    assert (
        apply_then_handoff_lines(apply, [], " --namespace ns", handoff_before_apply=True)
        == [apply]
    )


def test_k8s_operator_manages_own_execution():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    assert op.manages_own_execution() is True


def test_log_stream_command_is_bare_follow_with_global_args():
    from sflow.core.kubectl_config import KubectlConfig
    from sflow.plugins.k8s.shell import build_log_stream_command

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
        ["kubectl", "--kubeconfig", "/k/cfg", "logs", "-f", "pod/t-abc",
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
    # Object names are allocation-scoped ("abc" in tests) so parallel runs never
    # collide; the offload log still uses the raw task name (decoupled from names).
    assert plan.pod_refs == ["pod/decode-server-0-abc"]
    # The pod log is offloaded straight to <task>.log (raw task name), which the
    # decoupled console tailer reads.
    assert plan.task_log_path == str(tmp_path / "decode_server_0.log")
    # Cleanup deletes this task's own objects by name (backstop: backend sweep).
    assert "pod/decode-server-0-abc" in plan.cleanup_refs
    assert "configmap/decode-server-0-abc-cfg" in plan.cleanup_refs
    assert "secret/decode-server-0-abc-env" in plan.cleanup_refs
    assert any(r.startswith("resourceclaimtemplate") for r in plan.cleanup_refs)


def test_object_names_scoped_by_allocation_id_so_parallel_runs_dont_collide():
    # Two runs of the same recipe share a namespace; their pod/ConfigMap/Secret
    # names must differ so one run's `kubectl apply` never clobbers another's.
    def _names(alloc_id):
        be = _backend("device_plugin", 8, nodes=1)
        be.allocation = Allocation(
            allocation_id=alloc_id, nodes=be.allocation.nodes, owned=True
        )
        manifest, _ = _build(be, ["node-0"], gpu_count=2, task="frontend_server")
        return {i["kind"]: i["metadata"]["name"] for i in manifest["items"]}

    run_a = _names("aaa11111")
    run_b = _names("bbb22222")
    assert run_a["Pod"] == "frontend-server-aaa11111"
    assert run_b["Pod"] == "frontend-server-bbb22222"
    assert run_a["ConfigMap"] == "frontend-server-aaa11111-cfg"
    # Every object name is unique per run -> parallel runs don't interfere.
    assert set(run_a.values()).isdisjoint(run_b.values())


def test_object_names_unscoped_without_allocation_id():
    # Dry-run / no allocation: names stay the bare task name (nothing is applied to
    # a live cluster, so there is nothing to collide with).
    be = _backend("device_plugin", 8, nodes=1)
    be.allocation = None
    manifest, _ = _build(be, ["node-0"], gpu_count=2, task="frontend_server")
    assert _pods(manifest)[0]["metadata"]["name"] == "frontend-server"


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


def test_fs_artifact_mounted_even_when_not_referenced_by_script():
    # fs:// artifacts are workflow-wide storage (like declared PVCs): mounted into
    # EVERY task pod even when the script never names the path -- e.g. the dynamo
    # frontend loads the model card from the path it discovers via etcd, so it needs
    # the model dir present despite its script being just `python3 -m dynamo.frontend`.
    art = Artifact(
        name="MODEL", uri="fs:///shared/models/qwen", path=Path("/shared/models/qwen")
    )
    manifest, _ = _build(
        _backend("device_plugin", 8), ["node-0"], gpu_count=0,
        script=["python3 -m dynamo.frontend --http-port 8000"],
        artifacts=[art],
    )
    vols = _pods(manifest)[0]["spec"]["volumes"]
    assert any(v.get("hostPath", {}).get("path") == "/shared/models/qwen" for v in vols)


def test_inline_file_artifact_injected_into_every_pod_even_when_not_referenced():
    # Workflow-level artifacts are global (aligned with the Slurm path): an inline
    # file:// helper is injected (ConfigMap + subPath mount) into EVERY task pod,
    # even one whose script never names it.
    art = Artifact(
        name="PREFILL_CONFIG",
        uri="file://prefill_config.yaml",
        path=Path("/out/run/prefill_config.yaml"),
        content="max_batch_size: 128\n",
    )
    manifest, _ = _build(
        _backend("device_plugin", 8), ["node-0"], gpu_count=0,
        script=["etcd --data-dir /tmp/etcd"],
        artifacts=[art],
    )
    art_cms = [
        i for i in manifest["items"]
        if i["kind"] == "ConfigMap" and i["metadata"]["name"].endswith("-artifacts")
    ]
    assert len(art_cms) == 1
    assert art_cms[0]["data"] == {"PREFILL_CONFIG": "max_batch_size: 128\n"}
    mounts = _pods(manifest)[0]["spec"]["containers"][0]["volumeMounts"]
    assert any(
        m.get("subPath") == "PREFILL_CONFIG"
        and m["mountPath"] == "/out/run/prefill_config.yaml"
        for m in mounts
    )


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


def test_rdma_disable_injects_socket_disable_env_in_pods():
    # `rdma: disable` is the explicit clean kill switch: the backend's net_env carries
    # the NCCL socket-forcing envs, which the operator injects into every task pod
    # so NCCL never probes dead HCAs (no external-IB-plugin abort). UCX untouched.
    be = _backend("device_plugin", 8, nodes=1)
    be._rdma_plan = RdmaPlan.off()
    manifest, _ = _build(be, ["node-0"], gpu_count=4)
    env = {
        e["name"]: e["value"]
        for e in _pods(manifest)[0]["spec"]["containers"][0].get("env", [])
    }
    assert env.get("NCCL_IB_DISABLE") == "1"
    assert env.get("NCCL_IBEXT_DISABLE") == "1"
    assert env.get("NCCL_NET_PLUGIN") == "none"
    assert "UCX_NET_DEVICES" not in env


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


def test_mnnvl_env_defaults_helper():
    # Pure logic: no channel -> nothing; a channel -> both transport enables; and a
    # key the workflow already set (recipe variable / `-s`) is skipped so it stays in
    # the env Secret and wins (sflow never re-adds it to the overriding container env).
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    assert op._mnnvl_env_defaults(None, {}) == {}
    assert op._mnnvl_env_defaults("cd", {}) == {
        "NCCL_MNNVL_ENABLE": "1",
        "UCX_CUDA_IPC_ENABLE_MNNVL": "y",
    }
    assert op._mnnvl_env_defaults("cd", {"UCX_CUDA_IPC_ENABLE_MNNVL": "n"}) == {
        "NCCL_MNNVL_ENABLE": "1",
    }


def test_compute_domain_gpu_pod_defaults_mnnvl_transport_env():
    # A GPU pod joining the IMEX ComputeDomain gets BOTH MNNVL transport enables so
    # NCCL collectives and UCX cuda_ipc (NIXL KV / MPI GPU transfers) ride the rack
    # NVLink fabric instead of intra-node NVLink + (possibly slow-TCP) network.
    be = _backend("device_plugin", 8, nodes=1)
    be._compute_domain_channel = "cd-chan"
    manifest, _ = _build(be, ["node-0"], gpu_count=4)
    env = {
        e["name"]: e["value"]
        for e in _pods(manifest)[0]["spec"]["containers"][0].get("env", [])
    }
    assert env.get("NCCL_MNNVL_ENABLE") == "1"
    assert env.get("UCX_CUDA_IPC_ENABLE_MNNVL") == "y"


def test_compute_domain_cpu_pod_omits_mnnvl_transport_env():
    # CPU-only pods do not join the channel (see channel-claim gating), so they must
    # not get the MNNVL enables either.
    be = _backend("device_plugin", 8, nodes=1)
    be._compute_domain_channel = "cd-chan"
    manifest, _ = _build(be, ["node-0"], gpu_count=0)
    env = {
        e["name"]: e["value"]
        for e in _pods(manifest)[0]["spec"]["containers"][0].get("env", [])
    }
    assert "NCCL_MNNVL_ENABLE" not in env
    assert "UCX_CUDA_IPC_ENABLE_MNNVL" not in env


def test_compute_domain_mnnvl_defaults_yield_to_recipe_env():
    # A recipe/`-s` value lands in the task env (-> env Secret / envFrom). sflow must
    # NOT re-add it to container env (which outranks envFrom) or it would silently
    # override the explicit choice. So those keys stay out of container env entirely.
    be = _backend("device_plugin", 8, nodes=1)
    be._compute_domain_channel = "cd-chan"
    manifest, _ = _build(
        be, ["node-0"], gpu_count=4,
        envs={"NCCL_MNNVL_ENABLE": "0", "UCX_CUDA_IPC_ENABLE_MNNVL": "n"},
    )
    env = {
        e["name"]: e["value"]
        for e in _pods(manifest)[0]["spec"]["containers"][0].get("env", [])
    }
    assert "NCCL_MNNVL_ENABLE" not in env
    assert "UCX_CUDA_IPC_ENABLE_MNNVL" not in env


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
    env = {e["name"]: e["value"] for e in pod["spec"]["containers"][0].get("env", [])}
    # NIC *resources* are aligned to the GPU slot, but NIC *selection* is left to
    # NCCL/gIB -- sflow never pins NCCL_IB_HCA.
    assert "NCCL_IB_HCA" not in env
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


def test_gke_no_ib_pin_and_no_runtime_affinity_preamble():
    # GKE leaves NIC selection to NCCL/gIB (allow_runtime_affinity=False): NO
    # build-time NCCL_IB_HCA pin and NO expose-all runtime affinity preamble. It still
    # requests the GPU-aligned per-pod NIC window (rdma-4..7) for scheduling.
    manifest, _ = _build(
        _rdma_backend(), ["node-0"], gpu_count=4, cuda_visible="4,5,6,7"
    )
    pod = _pods(manifest)[0]
    env = {e["name"]: e["value"] for e in pod["spec"]["containers"][0].get("env", [])}
    assert "NCCL_IB_HCA" not in env  # not pinned -> library auto-selects
    assert "UCX_NET_DEVICES" not in env
    assert _rdma_nic_indices(pod) == [4, 5, 6, 7]
    assert "_sflow_rdma_setup" not in _entrypoint(manifest)  # no affinity preamble


def test_rdma_full_node_pod_is_not_ib_pinned():
    # A pod that owns ALL the node's NICs is left UNPINNED (NCCL auto-selects).
    # Force-pinning every HCA without the GKE gIB NCCL tuning drove an unstable
    # all-NIC RDMA config that reset the node; only partial-node pods are pinned.
    manifest, _ = _build(
        _rdma_backend(), ["node-0"], gpu_count=8,
        cuda_visible="0,1,2,3,4,5,6,7",
    )
    pod = _pods(manifest)[0]
    assert _rdma_nic_indices(pod) == [0, 1, 2, 3, 4, 5, 6, 7]  # all node NICs granted
    env = {e["name"]: e["value"] for e in pod["spec"]["containers"][0].get("env", [])}
    assert "NCCL_IB_HCA" not in env
    assert "UCX_NET_DEVICES" not in env


def _rdma_multinode_backend(gpus_per_node=8, nics=8, nodes=2, gib=True):
    """A multi-node GKE RDMA backend (triggers the gIB preamble on n>1 pods).

    ``gib=False`` models a GKE cluster WITHOUT the ``nccl-rdma-installer`` DaemonSet:
    the provider grants the NICs but emits no lib mounts / NCCL tuning script.
    """
    from sflow.plugins.k8s.rdma import (
        GKE_NCCL_ENV_SCRIPT,
        GKE_RDMA_LIB_MOUNTS,
    )

    be = _backend("device_plugin", gpus_per_node, nodes=nodes)
    be._rdma_plan = RdmaPlan(
        provider="gke",
        enabled=True,
        nic_specs=tuple(
            (f"networking.gke.io.networks/rdma-{i}", f"mlx5_{i}") for i in range(nics)
        ),
        ipc_lock=True,
        lib_mounts=GKE_RDMA_LIB_MOUNTS if gib else (),
        nccl_env_script=GKE_NCCL_ENV_SCRIPT if gib else "",
    )
    return be


def test_multinode_gib_full_node_no_untuned_ib_pin():
    # TP16 across 2 nodes (8 GPUs/node = full node per node): the gIB preamble
    # sources set_nccl_env.sh ONLY if present and must NOT emit an untuned
    # `NCCL_IB_HCA=<all HCAs>` pin (that all-NIC config reset the node). A full-node
    # pod is never pinned; NCCL/gIB auto-select.
    manifest, _ = _build(
        _rdma_multinode_backend(), ["node-0", "node-1"], gpu_count=16,
    )
    entry = _entrypoint(manifest)
    assert "if [ -f /usr/local/gib/scripts/set_nccl_env.sh ]; then" in entry
    assert "source /usr/local/gib/scripts/set_nccl_env.sh" in entry
    # No unguarded (untuned) source form, and no NCCL_IB_HCA pin at all.
    assert "] && source /usr/local/gib" not in entry
    assert "NCCL_IB_HCA" not in entry


def test_multinode_gib_partial_node_sources_config_no_pin():
    # A partial-node multi-node pod sources the gIB config but is NEVER NIC-pinned:
    # NIC selection is left to NCCL/gIB (topology-aware). sflow never pins NCCL_IB_HCA.
    manifest, _ = _build(
        _rdma_multinode_backend(), ["node-0", "node-1"], gpu_count=8,
        cuda_visible="0,1,2,3",
    )
    entry = _entrypoint(manifest)
    assert "source /usr/local/gib/scripts/set_nccl_env.sh" in entry
    assert "NCCL_IB_HCA" not in entry


def test_singlenode_gib_sources_config_workload_agnostic():
    # gIB is workload-agnostic infra: a SINGLE-node GKE pod still sources
    # set_nccl_env.sh (-> NCCL_CONF_FILE) so the cluster-wide auto-loaded gIB tuner is
    # configured and NCCL init doesn't abort. No NCCL_IB_HCA pin (single-node uses
    # NVLink, not cross-node NCCL), and the gIB config dir is mounted.
    manifest, _ = _build(
        _rdma_multinode_backend(), ["node-0"], gpu_count=8,
        cuda_visible="0,1,2,3,4,5,6,7",
    )
    entry = _entrypoint(manifest)
    assert "source /usr/local/gib/scripts/set_nccl_env.sh" in entry
    assert "NCCL_IB_HCA" not in entry  # single-node -> no pin
    vols = {
        v.get("hostPath", {}).get("path")
        for v in _pods(manifest)[0]["spec"]["volumes"]
    }
    assert "/home/kubernetes/bin/gib" in vols  # gIB config mounted


def test_singlenode_partial_gib_sources_config_but_no_pin():
    # A single-node partial-node pod (4 of 8 GPUs, e.g. two workers packed per node)
    # still sources the gIB config but is NOT NIC-pinned: the pin is only for
    # cross-node NCCL; single-node collectives use NVLink.
    manifest, _ = _build(
        _rdma_multinode_backend(), ["node-0"], gpu_count=4, cuda_visible="0,1,2,3",
    )
    entry = _entrypoint(manifest)
    assert "source /usr/local/gib/scripts/set_nccl_env.sh" in entry
    assert "NCCL_IB_HCA" not in entry


def test_singlenode_without_gib_no_gib_preamble():
    # No gIB installer -> nothing auto-loads -> no gIB source line and no gIB mounts.
    manifest, _ = _build(
        _rdma_multinode_backend(gib=False), ["node-0"], gpu_count=8,
        cuda_visible="0,1,2,3,4,5,6,7",
    )
    entry = _entrypoint(manifest)
    assert "set_nccl_env.sh" not in entry
    vols = {
        v.get("hostPath", {}).get("path")
        for v in _pods(manifest)[0]["spec"]["volumes"]
    }
    assert "/home/kubernetes/bin/gib" not in vols


def test_multinode_no_gib_full_node_emits_no_nccl_env_at_all():
    # Multi-node GKE WITHOUT the gIB installer (no lib mounts / no NCCL tuning): the
    # gIB preamble must be omitted entirely and a full-node pod must NOT be pinned --
    # neither in the entrypoint nor the pod env. NCCL falls back to its built-in IB
    # transport and auto-selects across every granted NIC.
    manifest, _ = _build(
        _rdma_multinode_backend(gib=False), ["node-0", "node-1"], gpu_count=16,
    )
    entry = _entrypoint(manifest)
    assert "set_nccl_env.sh" not in entry  # no gIB source line
    assert "NCCL_IB_HCA" not in entry
    env = {
        e["name"]: e["value"]
        for e in _pods(manifest)[0]["spec"]["containers"][0].get("env", [])
    }
    assert "NCCL_IB_HCA" not in env  # full-node pod is never pinned
    assert "UCX_NET_DEVICES" not in env


def test_multinode_no_gib_partial_node_no_pin():
    # Multi-node GKE WITHOUT gIB, partial-node pod (4 of 8 GPUs/node): no gIB preamble
    # and -- like every path -- NO NCCL_IB_HCA pin. NIC selection is left to NCCL's
    # built-in IB transport; UCX is never pinned either.
    manifest, _ = _build(
        _rdma_multinode_backend(gib=False), ["node-0", "node-1"], gpu_count=8,
        cuda_visible="0,1,2,3",
    )
    entry = _entrypoint(manifest)
    assert "set_nccl_env.sh" not in entry  # no gIB preamble
    env = {
        e["name"]: e["value"]
        for e in _pods(manifest)[0]["spec"]["containers"][0].get("env", [])
    }
    assert "NCCL_IB_HCA" not in env
    assert "UCX_NET_DEVICES" not in env


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


def test_tty_config_allocates_container_stdin_and_tty():
    # tty=True on the operator config threads to the rendered pod container so a
    # progress bar (aiperf/pip) streams live to <task>.log via `kubectl logs`.
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1", tty=True))
    op.apply_backend_context(
        backend=_backend("device_plugin", 8), assigned_nodes=["node-0"],
        artifacts=[], gpu_count=0,
    )
    shell = op.build_command(task_name="t", script=["run"], envs={}).as_list()[-1]
    body = shell.split(_MARK, 1)[1].split("\n" + _MARK, 1)[0]
    pod = _pods(json.loads(body.split("\n", 1)[1]))[0]
    c = pod["spec"]["containers"][0]
    assert c["tty"] is True and c["stdin"] is True


def test_tty_defaults_off_no_container_tty():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend("device_plugin", 8), assigned_nodes=["node-0"],
        artifacts=[], gpu_count=0,
    )
    shell = op.build_command(task_name="t", script=["run"], envs={}).as_list()[-1]
    body = shell.split(_MARK, 1)[1].split("\n" + _MARK, 1)[0]
    pod = _pods(json.loads(body.split("\n", 1)[1]))[0]
    c = pod["spec"]["containers"][0]
    assert "tty" not in c and "stdin" not in c


def test_additional_k8s_fields_thread_to_manifest():
    # Curated pod/container passthroughs + the two raw override escape hatches all
    # thread from the operator config into the rendered pod manifest.
    op = K8sOperator(K8sOperatorConfig(
        name="op", image="img:1",
        image_pull_secrets=["regcred"],
        service_account="sa-1",
        labels={"team": "x"},
        env=[{"name": "FOO", "value": "bar"}],
        working_dir="/work",
        pod_overrides={"hostPID": True},
        container_overrides={"tty": True},
    ))
    op.apply_backend_context(
        backend=_backend("device_plugin", 8), assigned_nodes=["node-0"],
        artifacts=[], gpu_count=0,
    )
    shell = op.build_command(task_name="t", script=["run"], envs={}).as_list()[-1]
    body = shell.split(_MARK, 1)[1].split("\n" + _MARK, 1)[0]
    pod = _pods(json.loads(body.split("\n", 1)[1]))[0]
    spec = pod["spec"]
    assert spec["imagePullSecrets"] == [{"name": "regcred"}]
    assert spec["serviceAccountName"] == "sa-1"
    assert spec["hostPID"] is True
    c = spec["containers"][0]
    assert c["workingDir"] == "/work"
    assert c["tty"] is True
    env = {e["name"]: e.get("value") for e in c["env"]}
    assert env["FOO"] == "bar"
    assert pod["metadata"]["labels"]["team"] == "x"


def test_manifest_override_conflict_emits_warning(monkeypatch, caplog):
    # pod_overrides.restartPolicy clobbers sflow's managed "Never" -> the operator
    # must warn that an sflow-intended manifest value was overridden.
    op = K8sOperator(K8sOperatorConfig(
        name="op", image="img:1",
        pod_overrides={"restartPolicy": "OnFailure"},
    ))
    op.apply_backend_context(
        backend=_backend("device_plugin", 8), assigned_nodes=["node-0"],
        artifacts=[], gpu_count=0,
    )
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(
        logging.WARNING, logger="sflow.plugins.operators.k8s_operator"
    ):
        op.build_command(task_name="t", script=["run"], envs={})
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "sflow-managed manifest values" in m and "pod.spec.restartPolicy" in m
        for m in msgs
    )


def test_no_override_conflict_no_warning(monkeypatch, caplog):
    # Adding only brand-new keys (no collision with sflow-managed values) must not
    # emit an override warning.
    op = K8sOperator(K8sOperatorConfig(
        name="op", image="img:1",
        service_account="sa-1", pod_overrides={"hostPID": True},
    ))
    op.apply_backend_context(
        backend=_backend("device_plugin", 8), assigned_nodes=["node-0"],
        artifacts=[], gpu_count=0,
    )
    monkeypatch.setattr(logging.getLogger("sflow"), "propagate", True)
    with caplog.at_level(
        logging.WARNING, logger="sflow.plugins.operators.k8s_operator"
    ):
        op.build_command(task_name="t", script=["run"], envs={})
    assert not any(
        "sflow-managed manifest values" in r.getMessage()
        for r in caplog.records
    )


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


def test_network_fallback_status_detects_unusable_nic_from_task_log(tmp_path):
    task_dir = tmp_path / "decode_server_0"
    task_dir.mkdir()
    (task_dir / "decode_server_0.log").write_text(
        "[pod/x/x] [sflow-rdma] no InfiniBand/RoCE port is ACTIVE (all ports "
        "DOWN): no usable RDMA NIC -- leaving transport to NCCL/UCX auto-detect; "
        "to force sockets set NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none "
        "NCCL_IBEXT_DISABLE=1 for your cluster\n"
    )
    task, op = _task_with_out("decode_server_0", task_dir)
    status = op.network_fallback_status(task)
    assert status is not None and status.rdma_nic_unusable
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


def _member_task(name, script, envs, cvd, gate_after=None):
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
    t.merge_gate_after = list(gate_after or [])
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
        {"MODEL": "m", "SFLOW_OUTPUT_DIR": "/out",
         "SFLOW_TASK_OUTPUT_DIR": "/out/decode"}, "0,1,2,3",
    )
    prefill = _member_task(
        "prefill", ["run-prefill"],
        {"MODEL": "m", "ROLE": "prefill", "SFLOW_OUTPUT_DIR": "/out",
         "SFLOW_TASK_OUTPUT_DIR": "/out/prefill"}, "4,5",
    )
    op.apply_merge_group(members=[decode, prefill], union_gpus=6)
    cmd = op.build_command(task_name="decode", script=["run-decode"], envs=decode.envs)
    shell = cmd.as_list()[-1]
    after = shell.split(_MARK, 1)[1]
    body = after.split("\n" + _MARK, 1)[0]
    manifest = json.loads(body.split("\n", 1)[1])
    return manifest, shell, op


def test_merged_gib_pod_sources_config():
    # A merged pod is single-node but still on a gIB cluster, so its launcher must
    # source the gIB config once (env inherited by every member) or the members'
    # auto-loaded gIB plugins abort NCCL init. No NCCL_IB_HCA pin (single-node).
    manifest, shell, _ = _merged_build(_rdma_multinode_backend())
    assert "source /usr/local/gib/scripts/set_nccl_env.sh" in shell
    assert "NCCL_IB_HCA" not in shell
    vols = {
        v.get("hostPath", {}).get("path")
        for v in _pods(manifest)[0]["spec"]["volumes"]
    }
    assert "/home/kubernetes/bin/gib" in vols


def test_merge_pod_single_pod_with_union_gpus():
    manifest, shell, _ = _merged_build(_backend("dra", 8, cpu_per_gpu=8))
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
    # Merged pod CPU request scales with the union GPU count (cpu_per_gpu 8 x 6).
    assert pod["spec"]["containers"][0]["resources"]["requests"]["cpu"] == "48"
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
    # Members have SFLOW_TASK_OUTPUT_DIR, so each script is prefixed with an mkdir of
    # the per-task dir (under the emptyDir at SFLOW_OUTPUT_DIR), then the user command.
    assert data["merge_decode.sh"].endswith("run-decode")
    assert 'mkdir -p "$SFLOW_TASK_OUTPUT_DIR"' in data["merge_decode.sh"]
    assert data["merge_prefill.sh"].endswith("run-prefill")
    launcher = data["entrypoint.sh"]
    # Each member launched with its packed CUDA_VISIBLE_DEVICES + tagged output.
    assert "_sflow_run decode 0,1,2,3 /sflow/merge_decode.sh" in launcher
    assert "_sflow_run prefill 4,5 /sflow/merge_prefill.sh" in launcher
    assert "[[sflow-mux:" in launcher
    # A member's final line without a trailing newline is still emitted (not dropped).
    assert '|| [ -n "$_sflow_line" ]' in launcher
    assert 'export CUDA_VISIBLE_DEVICES="$_cvd"' in launcher
    # NVIDIA_VISIBLE_DEVICES mirrors the same per-member slice.
    assert 'export NVIDIA_VISIBLE_DEVICES="$_cvd"' in launcher


def test_merged_launcher_renders_gate_only_for_dependent_member():
    from sflow.plugins.k8s.shell import (
        MERGE_GATE_DIR,
        merge_gate_marker,
        merged_launcher_lines,
    )

    lines = merged_launcher_lines(
        [
            ("prefill", "4,5", "/sflow/merge_prefill.sh", "/sflow/p/envsh", ""),
            ("decode", "0,1,2,3", "/sflow/merge_decode.sh", "/sflow/d/envsh", "prefill"),
        ]
    )
    text = "\n".join(lines)
    # Gate scaffolding present.
    assert "_sflow_gate() {" in text
    assert 'mkdir -p "$_SFLOW_GATE_DIR"' in text
    assert f"_SFLOW_GATE_DIR={MERGE_GATE_DIR}" in text  # /tmp/... needs no shell quoting
    assert "/tmp/sflow-merge-gate" in text
    # decode waits for prefill; prefill has an empty gate arg.
    assert (
        "_sflow_run decode 0,1,2,3 /sflow/merge_decode.sh /sflow/d/envsh prefill" in text
    )
    assert "_sflow_run prefill 4,5 /sflow/merge_prefill.sh /sflow/p/envsh ''" in text
    # Marker path helper matches what the loop checks.
    assert merge_gate_marker("prefill") == "/tmp/sflow-merge-gate/prefill.open"


def test_merged_gate_does_not_fail_dependency_on_empty_rc_read():
    # Race guard: a dependency's rc file is written non-atomically (`echo "$?" > f`
    # truncates then writes), so a poll can `cat` it EMPTY mid-write. An empty read must
    # be treated as "still writing" (keep waiting), NOT reach `return "$_drc"` (an empty
    # `_drc` is a bash "numeric argument required" error -> the gated member would be
    # wrongly skipped as if its dependency had failed).
    from sflow.plugins.k8s.shell import merged_launcher_lines

    text = "\n".join(
        merged_launcher_lines(
            [
                ("dep", "0", "/s/dep.sh", "/s/env", ""),
                ("gated", "0", "/s/gated.sh", "/s/env", "dep"),
            ]
        )
    )
    assert '[ -z "$_drc" ]' in text  # empty read -> keep waiting, don't fail the dep


def test_merged_launcher_backward_compatible_with_4_tuples():
    # Existing callers pass 4-tuples (no gate); must still render, empty gate arg.
    from sflow.plugins.k8s.shell import merged_launcher_lines

    lines = merged_launcher_lines([("m", "0", "/s/m.sh", "/s/envsh")])
    text = "\n".join(lines)
    assert "_sflow_run m 0 /s/m.sh /s/envsh ''" in text


def test_merged_plan_threads_gate_into_launcher_for_dependent_member():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend("dra", 8), assigned_nodes=["node-0"], artifacts=[],
        gpu_count=4, cuda_visible_devices="0,1,2,3",
    )
    prefill = _member_task(
        "prefill", ["run-prefill"],
        {"SFLOW_OUTPUT_DIR": "/out", "SFLOW_TASK_OUTPUT_DIR": "/out/prefill"}, "4,5",
    )
    decode = _member_task(
        "decode", ["run-decode"],
        {"SFLOW_OUTPUT_DIR": "/out", "SFLOW_TASK_OUTPUT_DIR": "/out/decode"}, "0,1,2,3",
        gate_after=["prefill"],
    )
    op.apply_merge_group(members=[decode, prefill], union_gpus=6)
    cmd = op.build_command(task_name="decode", script=["run-decode"], envs=decode.envs)
    launcher = cmd.as_list()[-1]
    # decode gates on prefill; prefill gates on nobody (empty last arg).
    assert (
        "_sflow_run decode 0,1,2,3 /sflow/merge_decode.sh" in launcher
        and "/envsh prefill" in launcher
    )
    assert "_sflow_run prefill 4,5 /sflow/merge_prefill.sh" in launcher
    assert "/envsh ''" in launcher  # prefill's empty gate arg


def test_open_merge_gate_execs_touch_marker(monkeypatch):
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend("dra", 8), assigned_nodes=["node-0"], artifacts=[],
        gpu_count=4, cuda_visible_devices="0,1,2,3",
    )
    decode = _member_task("decode", ["run-decode"], {}, "0,1,2,3")
    prefill = _member_task("prefill", ["run-prefill"], {}, "4,5")
    op.apply_merge_group(members=[decode, prefill], union_gpus=6)

    calls = []

    async def _fake_run_kubectl(args, *, global_args=None):
        calls.append(list(args))
        return 0, "", ""

    from sflow.plugins.operators import k8s_operator as mod
    monkeypatch.setattr(mod.k8s_lifecycle, "run_kubectl", _fake_run_kubectl)

    ok = asyncio.run(op.open_merge_gate("prefill"))
    assert ok is True
    assert len(calls) == 1
    args = calls[0]
    assert args[0] == "exec"
    assert "--" in args and "sh" in args and "-c" in args
    joined = " ".join(args)
    assert "/tmp/sflow-merge-gate/prefill.open" in joined
    assert "mkdir -p /tmp/sflow-merge-gate" in joined


def test_open_merge_gate_noop_when_not_merge_leader():
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    assert asyncio.run(op.open_merge_gate("prefill")) is False


def test_end_to_end_assembly_gate_renders_in_launcher():
    # Real planner sets merge_gate_after; real operator renders the gate. decode
    # depends on prefill; both co-located GPU tasks on one node with merge on.
    from sflow.app.assembly import _plan_merge_groups
    from sflow.core.task import Task, TaskStatus
    from sflow.core.task_graph import TaskGraph

    class _BE:
        name = "k8s"
        merge_colocated_gpu_pods = True
        compute_domain_channel = None
        nvlink_domain_scope = None
        rdma_enabled = False

    class _PL:
        def __init__(self, be, nodes, gpu):
            self.backend, self.assigned_nodes, self.gpu_count = be, nodes, gpu

    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend("dra", 8), assigned_nodes=["node-0"], artifacts=[],
        gpu_count=4, cuda_visible_devices="0,1,2,3",
    )
    # Both members share the real operator so grouping sees one image and the leader
    # renders the merged pod. The planner packs CVD + records the gate itself.
    decode = Task(
        name="decode", logger=logging.getLogger("test.decode"), operator=op,
        status=TaskStatus.INITIATED, script=["run-decode"],
        envs={"SFLOW_OUTPUT_DIR": "/out", "SFLOW_TASK_OUTPUT_DIR": "/out/decode"},
    )
    prefill = Task(
        name="prefill", logger=logging.getLogger("test.prefill"), operator=op,
        status=TaskStatus.INITIATED, script=["run-prefill"],
        envs={"SFLOW_OUTPUT_DIR": "/out", "SFLOW_TASK_OUTPUT_DIR": "/out/prefill"},
    )
    tg = TaskGraph()
    tg.dag.add_node("decode", decode)
    tg.dag.add_node("prefill", prefill)
    tg.dag.add_edge("prefill", "decode")  # decode depends on prefill
    be = _BE()
    _plan_merge_groups(
        tg, {"decode": _PL(be, ["node-0"], 4), "prefill": _PL(be, ["node-0"], 2)}
    )
    # Planner recorded the gate; leader is 'decode' (sorted first).
    assert decode.merge_gate_after == ["prefill"]
    assert decode.is_merge_leader is True

    cmd = op.build_command(task_name="decode", script=["run-decode"], envs=decode.envs)
    shell = cmd.as_list()[-1]
    after = shell.split(_MARK, 1)[1]
    body = after.split("\n" + _MARK, 1)[0]
    manifest = json.loads(body.split("\n", 1)[1])
    launcher = _entrypoint(manifest)  # real entrypoint.sh (unescaped newlines)
    assert "_sflow_gate() {" in launcher
    decode_line = next(
        ln for ln in launcher.splitlines() if ln.startswith("_sflow_run decode")
    )
    prefill_line = next(
        ln for ln in launcher.splitlines() if ln.startswith("_sflow_run prefill")
    )
    assert decode_line.endswith(" prefill")  # decode's gate arg = prefill
    assert prefill_line.endswith(" ''")       # prefill has no gate


@pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
def test_merged_gate_blocks_until_dependency_completes(tmp_path, fake_process):
    # A gated member must not run its script until the dependency's rc file shows 0.
    # We simulate a dependency that "completes" (exit 0) and assert the gated member
    # then runs.
    fake_process.allow_unregistered(True)  # real bash
    from sflow.plugins.k8s.shell import bash_lc_command, merged_launcher_lines

    dep_ok = tmp_path / "dep_ok.sh"
    dep_ok.write_text("exit 0\n")
    gated = tmp_path / "gated.sh"
    gated.write_text("printf 'GATED-RAN\\n'\n")
    noenv = str(tmp_path / "noenv")
    # Dependency 'dep' runs and exits 0; 'gated' waits for 'dep' then runs.
    cmd = bash_lc_command(
        merged_launcher_lines(
            [
                ("dep", "0", str(dep_ok), noenv, ""),
                ("gated", "0", str(gated), noenv, "dep"),
            ]
        )
    )
    result = subprocess.run(
        [str(a) for a in cmd.as_list()], text=True, capture_output=True, timeout=30
    )
    assert "[[sflow-mux:gated]] GATED-RAN" in result.stdout, result.stderr


@pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
def test_merged_launcher_emits_final_member_line_without_newline(tmp_path, fake_process):
    # Behavioral check of the `|| [ -n "$_sflow_line" ]` read-loop fix: a member whose
    # LAST line has no trailing newline must still be tagged + emitted. Plain `read`
    # returns non-zero at EOF and would silently drop it (e.g. a no-newline readiness
    # marker). Runs the REAL generated launcher, not just a string assertion.
    fake_process.allow_unregistered(True)  # real bash
    from sflow.plugins.k8s.shell import bash_lc_command, merged_launcher_lines

    member = tmp_path / "m.sh"
    # Two lines; the SECOND is written WITHOUT a trailing newline.
    member.write_text("printf 'ready line\\n'; printf 'FINAL-NO-NEWLINE'\n")
    noenv = str(tmp_path / "noenv")  # missing env file -> launcher skips sourcing it
    cmd = bash_lc_command(merged_launcher_lines([("m", "0", str(member), noenv)]))
    result = subprocess.run(
        [str(a) for a in cmd.as_list()], text=True, capture_output=True
    )
    # Both the normal line and the final no-newline line are emitted with the mux tag.
    assert "[[sflow-mux:m]] ready line" in result.stdout, result.stderr
    assert "[[sflow-mux:m]] FINAL-NO-NEWLINE" in result.stdout, result.stderr


def test_merge_pod_launcher_prepends_nvidia_driver_to_ld_library_path():
    # The union-GPU merge pod also needs the driver on LD_LIBRARY_PATH. The launcher
    # exports it once in the parent shell (inherited by every member subshell),
    # before any member starts.
    manifest, _, _ = _merged_build(_backend("device_plugin", 8))
    launcher = _entrypoint(manifest)
    assert "export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}" in launcher
    assert "export PATH=/usr/local/nvidia/bin:${PATH:-}" in launcher
    assert launcher.index("/usr/local/nvidia/lib64") < launcher.index("_sflow_run decode")


def test_merge_pod_rdma_exposes_all_node_nics_and_no_ib_pin():
    # A merged pod is the only GPU pod on its node, so it requests EVERY node RDMA
    # NIC (not a union-sized window) and does NOT build-time pin NCCL_IB_HCA -- NCCL
    # and UCX auto-select across all exposed NICs. IPC_LOCK is still granted so verbs
    # memory can be pinned. (_rdma_backend advertises 8 NICs; the merge union is 6.)
    manifest, _, _ = _merged_build(_rdma_backend())
    container = _pods(manifest)[0]["spec"]["containers"][0]
    assert _rdma_nic_indices(_pods(manifest)[0]) == [0, 1, 2, 3, 4, 5, 6, 7]
    env = {e["name"]: e["value"] for e in container.get("env", [])}
    assert "NCCL_IB_HCA" not in env
    assert "UCX_NET_DEVICES" not in env
    caps = container["securityContext"]["capabilities"]["add"]
    assert "IPC_LOCK" in caps


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


def test_merge_pod_same_env_key_isolated_across_members():
    # Regression: two merged members that set the SAME env key with DIFFERENT values
    # must NOT collapse/overwrite each other. Each member gets its own prefix in the
    # apply env, its own Secret, and its own mounted env file sourced in its own
    # subshell -- so each member sub-shell sees ITS OWN task's value.
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend("dra", 8), assigned_nodes=["node-0"], artifacts=[],
        gpu_count=4, cuda_visible_devices="0,1,2,3",
    )
    decode = _member_task(
        "decode", ["run-decode"],
        {"PORT": "8000", "SFLOW_TASK_OUTPUT_DIR": "/out/decode"}, "0,1",
    )
    prefill = _member_task(
        "prefill", ["run-prefill"],
        {"PORT": "8001", "SFLOW_TASK_OUTPUT_DIR": "/out/prefill"}, "2,3",
    )
    op.apply_merge_group(members=[decode, prefill], union_gpus=4)

    plan = op._build_execution_plan(
        task_name="decode", script=["run-decode"], envs=decode.envs
    )
    # Same key name PORT -> distinct prefixed vars, each holding its OWN value
    # (no last-writer-wins collapse), and no shared bare key.
    assert plan.merge_launcher_env["SFMERGE0__PORT"] == "8000"
    assert plan.merge_launcher_env["SFMERGE1__PORT"] == "8001"
    assert "PORT" not in plan.merge_launcher_env

    # Structurally: no container-wide envFrom, distinct per-member mount paths, and
    # each member's PORT written from its OWN prefixed var into its OWN Secret.
    cmd = op.build_command(
        task_name="decode", script=["run-decode"], envs=decode.envs
    )
    shell = cmd.as_list()[-1]
    manifest = json.loads(
        shell.split(_MARK, 1)[1].split("\n" + _MARK, 1)[0].split("\n", 1)[1]
    )
    container = _pods(manifest)[0]["spec"]["containers"][0]
    assert "envFrom" not in container
    mounts = {m["mountPath"] for m in container["volumeMounts"]}
    assert {"/sflow/menv/decode", "/sflow/menv/prefill"} <= mounts
    assert "${SFMERGE0__PORT-}" in shell and "${SFMERGE1__PORT-}" in shell


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
    from sflow.plugins.k8s.shell import (
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
