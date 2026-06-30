# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sflow.plugins.operators._k8s_render import (
    COMPUTE_DOMAIN_CLAIM_NAME,
    RESERVATION_POD_IMAGE,
    SFLOW_ALLOC_LABEL,
    render_compute_domain_manifest,
    render_configmap,
    render_reservation_pod_manifest,
    render_resource_claim_template,
    render_task_pod,
)


# ---------------------------------------------------------------------------
# render_resource_claim_template (DRA)
# ---------------------------------------------------------------------------


def test_resource_claim_template_shape():
    m = render_resource_claim_template(name="gpu-rct", device_class="gpu.nvidia.com", count=4)
    assert m["apiVersion"] == "resource.k8s.io/v1"
    assert m["kind"] == "ResourceClaimTemplate"
    req = m["spec"]["spec"]["devices"]["requests"][0]
    assert req["name"] == "gpu"
    assert req["exactly"] == {
        "deviceClassName": "gpu.nvidia.com",
        "allocationMode": "ExactCount",
        "count": 4,
    }


def test_resource_claim_template_selectors_and_alloc_label():
    m = render_resource_claim_template(
        name="gpu-rct",
        device_class="gpu.nvidia.com",
        count=1,
        selectors=["device.attributes['x'].product == 'H100'"],
        allocation_id="abc",
    )
    req = m["spec"]["spec"]["devices"]["requests"][0]["exactly"]
    assert req["selectors"] == [
        {"cel": {"expression": "device.attributes['x'].product == 'H100'"}}
    ]
    assert m["metadata"]["labels"][SFLOW_ALLOC_LABEL] == "abc"


# ---------------------------------------------------------------------------
# render_configmap
# ---------------------------------------------------------------------------


def test_configmap_shape():
    m = render_configmap(name="t-cfg", data={"entrypoint.sh": "echo hi"}, namespace="ns")
    assert m["kind"] == "ConfigMap"
    assert m["metadata"]["namespace"] == "ns"
    assert m["data"] == {"entrypoint.sh": "echo hi"}


# ---------------------------------------------------------------------------
# render_compute_domain_manifest (Multi-Node NVLink)
# ---------------------------------------------------------------------------


def test_compute_domain_manifest_shape():
    m = render_compute_domain_manifest(
        name="cd", num_nodes=4, channel_template_name="cd-channel",
        allocation_id="abc", namespace="ns",
    )
    assert m["apiVersion"] == "resource.nvidia.com/v1beta1"
    assert m["kind"] == "ComputeDomain"
    assert m["spec"]["numNodes"] == 4
    assert m["spec"]["channel"]["resourceClaimTemplate"]["name"] == "cd-channel"
    assert m["metadata"]["labels"][SFLOW_ALLOC_LABEL] == "abc"


def test_compute_domain_manifest_omits_namespace_when_not_given():
    m = render_compute_domain_manifest(
        name="cd", num_nodes=2, channel_template_name="ch", allocation_id="abc"
    )
    assert "namespace" not in m["metadata"]


# ---------------------------------------------------------------------------
# render_reservation_pod_manifest (GPU-holding placeholder)
# ---------------------------------------------------------------------------


def test_reservation_pod_is_a_sleeper_with_short_grace_period():
    m = render_reservation_pod_manifest(pod_name="pod-0", allocation_id="abc")
    spec = m["spec"]
    assert spec["containers"][0]["image"] == RESERVATION_POD_IMAGE
    assert spec["containers"][0]["command"][0] == "sh"
    assert spec["terminationGracePeriodSeconds"] == 5


def test_reservation_pod_dra_holds_gpus_via_claim():
    m = render_reservation_pod_manifest(
        pod_name="pod-0", allocation_id="abc", scheduling="dra",
        gpu_count=8, resource_claim_name="resv-rct",
    )
    spec = m["spec"]
    assert spec["resourceClaims"] == [
        {"name": "gpu", "resourceClaimTemplateName": "resv-rct"}
    ]
    assert spec["containers"][0]["resources"]["claims"] == [{"name": "gpu"}]


def test_reservation_pod_device_plugin_holds_gpus_via_limit():
    m = render_reservation_pod_manifest(
        pod_name="pod-0", allocation_id="abc", scheduling="device_plugin", gpu_count=8,
    )
    spec = m["spec"]
    assert spec["containers"][0]["resources"]["limits"] == {"nvidia.com/gpu": "8"}
    assert "resourceClaims" not in spec


def test_reservation_pod_no_resources_when_gpu_count_none():
    m = render_reservation_pod_manifest(pod_name="pod-0", allocation_id="abc")
    assert "resources" not in m["spec"]["containers"][0]
    assert "resourceClaims" not in m["spec"]


def test_reservation_pod_host_network_and_node_selector_and_tolerations():
    m = render_reservation_pod_manifest(
        pod_name="pod-0", allocation_id="abc", host_network=True,
        node_selector={"pool": "gpu"},
        tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists"}],
    )
    spec = m["spec"]
    assert spec["hostNetwork"] is True
    assert spec["nodeSelector"] == {"pool": "gpu"}
    assert spec["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists"}]


def test_reservation_pod_labels_and_anti_affinity_scoped_to_reservations():
    m = render_reservation_pod_manifest(pod_name="pod-0", allocation_id="abc")
    assert m["metadata"]["labels"][SFLOW_ALLOC_LABEL] == "abc"
    assert m["metadata"]["labels"]["sflow.ai/role"] == "reservation"
    anti = m["spec"]["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ][0]
    # Anti-affinity targets only other reservation pods (role label), not the
    # broad allocation label -- otherwise it would also repel the allocation's
    # task pods (anti-affinity is symmetric) and block CPU-only tasks pinned to a
    # still-reserved node.
    assert anti["labelSelector"]["matchLabels"] == {
        SFLOW_ALLOC_LABEL: "abc",
        "sflow.ai/role": "reservation",
    }
    assert anti["topologyKey"] == "kubernetes.io/hostname"
    # No exclude_nodes -> no nodeAffinity.
    assert "nodeAffinity" not in m["spec"]["affinity"]


def test_reservation_pod_exclude_nodes_adds_hostname_not_in_node_affinity():
    m = render_reservation_pod_manifest(
        pod_name="pod-0", allocation_id="abc", exclude_nodes=["bad-1", "bad-2"]
    )
    term = m["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"][0]
    assert term == {
        "key": "kubernetes.io/hostname",
        "operator": "NotIn",
        "values": ["bad-1", "bad-2"],
    }
    # The one-per-node spread anti-affinity is preserved alongside it.
    assert "podAntiAffinity" in m["spec"]["affinity"]


# ---------------------------------------------------------------------------
# render_task_pod
# ---------------------------------------------------------------------------


def test_task_pod_dra_claim_and_configmap_mount_and_hostname_pin():
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg", scheduling="dra",
        per_pod_gpus=2, resource_claim_name="t-gpu", assigned_node="node-a",
        compute_domain_channel="cd-channel",
    )
    spec = m["spec"]
    assert spec["containers"][0]["command"] == ["bash", "-l", "/sflow/entrypoint.sh"]
    assert spec["volumes"][0]["configMap"]["name"] == "t-cfg"
    assert spec["nodeSelector"]["kubernetes.io/hostname"] == "node-a"
    # DRA GPU claim + ComputeDomain channel claim.
    claim_names = {c["name"] for c in spec["resourceClaims"]}
    assert claim_names == {"gpu", COMPUTE_DOMAIN_CLAIM_NAME}


def test_task_pod_device_plugin_limit():
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        scheduling="device_plugin", per_pod_gpus=4,
    )
    assert m["spec"]["containers"][0]["resources"]["limits"] == {"nvidia.com/gpu": "4"}
    assert "resourceClaims" not in m["spec"]


def test_task_pod_inline_file_artifact_mounted_from_configmap():
    # file:// inline artifacts: one ConfigMap volume, each mounted read-only at its
    # resolved in-pod path via subPath (so remote pods see the content).
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        artifacts_configmap_name="t-artifacts",
        file_artifact_mounts=[("/out/run/prefill_config.yaml", "PREFILL_CONFIG")],
    )
    spec = m["spec"]
    art_vol = [v for v in spec["volumes"] if v["name"] == "sflow-artifacts"]
    assert art_vol and art_vol[0]["configMap"]["name"] == "t-artifacts"
    mounts = spec["containers"][0]["volumeMounts"]
    art_mount = [m for m in mounts if m["name"] == "sflow-artifacts"][0]
    assert art_mount == {
        "name": "sflow-artifacts",
        "mountPath": "/out/run/prefill_config.yaml",
        "subPath": "PREFILL_CONFIG",
        "readOnly": True,
    }
    # Script ConfigMap mount is preserved.
    assert any(mt["mountPath"] == "/sflow" for mt in mounts)


def test_task_pod_fs_artifact_mounted_as_hostpath():
    # fs:// artifacts: hostPath-mounted at the same node path so the pod sees it.
    # A known type makes the kubelet reject a missing node path loudly.
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        host_path_mounts=[("/shared/models/qwen", "Directory")],
    )
    spec = m["spec"]
    hp = [v for v in spec["volumes"] if v.get("hostPath")]
    assert hp and hp[0]["hostPath"] == {"path": "/shared/models/qwen", "type": "Directory"}
    mount = [
        mt for mt in spec["containers"][0]["volumeMounts"]
        if mt["mountPath"] == "/shared/models/qwen"
    ][0]
    assert mount["name"] == hp[0]["name"]


def test_task_pod_fs_artifact_hostpath_omits_type_when_unknown():
    # When the type is unknown (controller can't stat it), no type is set so the
    # kubelet stays lenient (e.g. node-only paths / runtime-created output dirs).
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        host_path_mounts=[("/mnt/out", "")],
    )
    hp = [v for v in m["spec"]["volumes"] if v.get("hostPath")][0]
    assert hp["hostPath"] == {"path": "/mnt/out"}


def test_task_pod_without_artifacts_has_script_and_shm_volumes():
    # Even with no artifacts, task pods get the script ConfigMap and a RAM-backed
    # /dev/shm (the 64Mi K8s default segfaults MPI/NCCL).
    m = render_task_pod(pod_name="t", image="img:1", configmap_name="t-cfg")
    spec = m["spec"]
    assert [v["name"] for v in spec["volumes"]] == ["sflow-scripts", "dshm"]
    assert [mt["mountPath"] for mt in spec["containers"][0]["volumeMounts"]] == [
        "/sflow", "/dev/shm",
    ]
    dshm = [v for v in spec["volumes"] if v["name"] == "dshm"][0]
    assert dshm["emptyDir"] == {"medium": "Memory"}


def test_task_pod_shm_size_caps_tmpfs():
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg", shm_size="16Gi"
    )
    dshm = [v for v in m["spec"]["volumes"] if v["name"] == "dshm"][0]
    assert dshm["emptyDir"] == {"medium": "Memory", "sizeLimit": "16Gi"}


def test_task_pod_pvc_mounted():
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        pvc_mounts=[{"name": "model-store", "claim": "model-pvc",
                     "mount_path": "/models", "sub_path": None, "read_only": True}],
    )
    spec = m["spec"]
    pv = [v for v in spec["volumes"] if v.get("persistentVolumeClaim")][0]
    assert pv["name"] == "model-store"
    assert pv["persistentVolumeClaim"] == {"claimName": "model-pvc", "readOnly": True}
    mt = [x for x in spec["containers"][0]["volumeMounts"] if x["name"] == "model-store"][0]
    assert mt == {"name": "model-store", "mountPath": "/models", "readOnly": True}


def test_task_pod_pvc_subpath_and_read_write():
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        pvc_mounts=[{"name": "v", "claim": "c", "mount_path": "/data",
                     "sub_path": "models", "read_only": False}],
    )
    pv = [v for v in m["spec"]["volumes"] if v["name"] == "v"][0]
    assert pv["persistentVolumeClaim"] == {"claimName": "c", "readOnly": False}
    mt = [x for x in m["spec"]["containers"][0]["volumeMounts"] if x["name"] == "v"][0]
    assert mt == {"name": "v", "mountPath": "/data", "readOnly": False, "subPath": "models"}
