# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sflow.plugins.operators._k8s_render import (
    COMPUTE_DOMAIN_CLAIM_NAME,
    PROBE_POD_IMAGE_DEFAULT,
    PROBE_ROLE,
    RESERVATION_POD_IMAGE,
    SFLOW_ALLOC_LABEL,
    SFLOW_ROLE_LABEL,
    render_compute_domain_manifest,
    render_configmap,
    render_probe_pod_manifest,
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


def test_resource_claim_template_gpu_only_has_no_constraints():
    # Without a NIC device class the claim is GPU-only (no NIC request/constraint).
    m = render_resource_claim_template(
        name="gpu-rct", device_class="gpu.nvidia.com", count=2
    )
    devices = m["spec"]["spec"]["devices"]
    assert [r["name"] for r in devices["requests"]] == ["gpu"]
    assert "constraints" not in devices


def test_resource_claim_template_coallocates_nic_with_pcie_root_constraint():
    # NIC co-allocation: a second request + a matchAttribute constraint aligning
    # the GPU and NIC on the same PCIe root complex (default pcieRoot).
    m = render_resource_claim_template(
        name="gpu-rct",
        device_class="gpu.nvidia.com",
        count=2,
        nic_device_class="rdma.nvidia.com",
    )
    devices = m["spec"]["spec"]["devices"]
    assert [r["name"] for r in devices["requests"]] == ["gpu", "rdma"]
    nic = [r for r in devices["requests"] if r["name"] == "rdma"][0]["exactly"]
    assert nic == {
        "deviceClassName": "rdma.nvidia.com",
        "allocationMode": "ExactCount",
        "count": 2,
    }
    assert devices["constraints"] == [
        {
            "requests": ["gpu", "rdma"],
            "matchAttribute": "resource.kubernetes.io/pcieRoot",
        }
    ]


def test_resource_claim_template_nic_count_and_attribute_overrides():
    m = render_resource_claim_template(
        name="gpu-rct",
        device_class="gpu.nvidia.com",
        count=4,
        nic_device_class="dra.net",
        nic_count=1,
        nic_selectors=["device.attributes['dra.net'].rdma == true"],
        match_attribute="dra.net/numaNode",
    )
    devices = m["spec"]["spec"]["devices"]
    nic = [r for r in devices["requests"] if r["name"] == "rdma"][0]["exactly"]
    assert nic["count"] == 1  # NIC count independent of GPU count
    assert nic["selectors"] == [
        {"cel": {"expression": "device.attributes['dra.net'].rdma == true"}}
    ]
    assert devices["constraints"][0]["matchAttribute"] == "dra.net/numaNode"


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


def test_reservation_pod_include_nodes_adds_hostname_in_node_affinity():
    m = render_reservation_pod_manifest(
        pod_name="pod-0", allocation_id="abc", include_nodes=["want-1", "want-2"]
    )
    exprs = m["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    assert exprs == [
        {"key": "kubernetes.io/hostname", "operator": "In", "values": ["want-1", "want-2"]}
    ]


def test_reservation_pod_include_and_exclude_combine_in_one_term():
    m = render_reservation_pod_manifest(
        pod_name="pod-0",
        allocation_id="abc",
        include_nodes=["want-1"],
        exclude_nodes=["bad-1"],
    )
    exprs = m["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    assert exprs == [
        {"key": "kubernetes.io/hostname", "operator": "In", "values": ["want-1"]},
        {"key": "kubernetes.io/hostname", "operator": "NotIn", "values": ["bad-1"]},
    ]


def test_reservation_pod_no_node_filters_has_no_node_affinity():
    m = render_reservation_pod_manifest(pod_name="pod-0", allocation_id="abc")
    assert "nodeAffinity" not in m["spec"]["affinity"]


def test_reservation_pod_nvlink_domain_podaffinity_when_key_set():
    # When a NVLink-domain topology key is given, all reservation pods must land in
    # ONE domain (podAffinity, topologyKey = the label), keeping the per-hostname
    # one-per-node spread (podAntiAffinity).
    m = render_reservation_pod_manifest(
        pod_name="pod-0",
        allocation_id="abc",
        nvlink_domain_topology_key="nvidia.com/gpu.clique",
    )
    affinity = m["spec"]["affinity"]
    aff = affinity["podAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
    assert aff["topologyKey"] == "nvidia.com/gpu.clique"
    assert aff["labelSelector"]["matchLabels"] == {
        SFLOW_ALLOC_LABEL: "abc",
        "sflow.ai/role": "reservation",
    }
    # The one-per-node spread anti-affinity is preserved.
    assert "podAntiAffinity" in affinity


def test_reservation_pod_no_podaffinity_without_key():
    m = render_reservation_pod_manifest(pod_name="pod-0", allocation_id="abc")
    assert "podAffinity" not in m["spec"]["affinity"]


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


def test_task_pod_rdma_host_device_mount_and_ipc_lock():
    # Host-device RDMA (no device plugin): /dev/infiniband is hostPath-mounted
    # (type Directory) and the container gets CAP_IPC_LOCK for verbs access.
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        rdma_host_device_paths=["/dev/infiniband"],
        rdma_ipc_lock=True,
    )
    spec = m["spec"]
    dev = [
        v for v in spec["volumes"]
        if v.get("hostPath", {}).get("path") == "/dev/infiniband"
    ]
    assert dev and dev[0]["hostPath"] == {
        "path": "/dev/infiniband", "type": "Directory"
    }
    mount = [
        mt for mt in spec["containers"][0]["volumeMounts"]
        if mt["mountPath"] == "/dev/infiniband"
    ][0]
    assert mount["name"] == dev[0]["name"]
    sc = spec["containers"][0]["securityContext"]
    assert sc["capabilities"]["add"] == ["IPC_LOCK"]


def test_task_pod_rdma_lib_mounts_require_existing_host_path():
    # Lib mounts are emitted ONLY when the gIB installer is detected (so the host
    # paths exist); the hostPath type must be `Directory` (require existence), never
    # DirectoryOrCreate -- creating an empty dir at /usr/local/nvidia would mask the
    # driver (libcuda.so.1). A genuinely-missing path should fail the pod loudly, not
    # silently mask a critical mount.
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        rdma_lib_mounts=[("/home/kubernetes/bin/gib", "/usr/local/gib")],
    )
    spec = m["spec"]
    gib = [
        v for v in spec["volumes"]
        if v.get("hostPath", {}).get("path") == "/home/kubernetes/bin/gib"
    ]
    assert gib and gib[0]["hostPath"] == {
        "path": "/home/kubernetes/bin/gib", "type": "Directory"
    }
    mount = [
        mt for mt in spec["containers"][0]["volumeMounts"]
        if mt["mountPath"] == "/usr/local/gib"
    ][0]
    assert mount["readOnly"] is True


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


def test_task_pod_defaults_no_host_ipc():
    m = render_task_pod(pod_name="t", image="img:1", configmap_name="t-cfg")
    assert "hostIPC" not in m["spec"]
    dshm = [v for v in m["spec"]["volumes"] if v["name"] == "dshm"][0]
    assert "emptyDir" in dshm and "hostPath" not in dshm


def test_task_pod_host_ipc_shares_ipc_namespace_and_dev_shm():
    # host_ipc -> spec.hostIPC + /dev/shm from the node's shared hostPath (so
    # co-located pods can do cross-pod CUDA IPC / NVLink KV transfer).
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg", host_ipc=True
    )
    assert m["spec"]["hostIPC"] is True
    dshm = [v for v in m["spec"]["volumes"] if v["name"] == "dshm"][0]
    assert dshm["hostPath"] == {"path": "/dev/shm", "type": "Directory"}
    assert "emptyDir" not in dshm
    # still mounted at /dev/shm in the container
    assert any(
        mt["mountPath"] == "/dev/shm"
        for mt in m["spec"]["containers"][0]["volumeMounts"]
    )


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


def test_task_pod_emptydir_volume():
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        pvc_mounts=[{"name": "kernel-cache",
                     "empty_dir": {"medium": "", "size_limit": None},
                     "mount_path": "/cache", "read_only": False}],
    )
    vol = [v for v in m["spec"]["volumes"] if v["name"] == "kernel-cache"][0]
    assert vol == {"name": "kernel-cache", "emptyDir": {}}
    mt = [x for x in m["spec"]["containers"][0]["volumeMounts"]
          if x["name"] == "kernel-cache"][0]
    assert mt == {"name": "kernel-cache", "mountPath": "/cache", "readOnly": False}


def test_task_pod_emptydir_medium_and_size_limit():
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        pvc_mounts=[{"name": "c",
                     "empty_dir": {"medium": "Memory", "size_limit": "10Gi"},
                     "mount_path": "/cache", "read_only": False}],
    )
    vol = [v for v in m["spec"]["volumes"] if v["name"] == "c"][0]
    assert vol == {"name": "c", "emptyDir": {"medium": "Memory", "sizeLimit": "10Gi"}}


def test_task_pod_ensure_writable_injects_root_init_container():
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        pvc_mounts=[{"name": "kernel-cache", "claim": "c", "mount_path": "/cache",
                     "sub_path": "sflow-kernel-cache", "read_only": False,
                     "ensure_writable": True}],
    )
    inits = m["spec"].get("initContainers")
    assert inits and len(inits) == 1
    ic = inits[0]
    assert ic["image"] == "img:1"  # reuses the task image (already pulled)
    assert ic["securityContext"] == {"runAsUser": 0, "runAsGroup": 0}
    # Mounts the volume ROOT (no subPath) so it can create+chmod the subPath dir.
    im = [x for x in ic["volumeMounts"] if x["name"] == "kernel-cache"][0]
    assert im["mountPath"] == "/sflow-ensure-writable/kernel-cache"
    assert "subPath" not in im
    cmd = ic["command"][-1]
    assert "chmod 0777" in cmd
    assert "/sflow-ensure-writable/kernel-cache/sflow-kernel-cache" in cmd
    # The workload container still mounts the PVC writable via subPath.
    mt = [x for x in m["spec"]["containers"][0]["volumeMounts"]
          if x["name"] == "kernel-cache"][0]
    assert mt == {"name": "kernel-cache", "mountPath": "/cache",
                  "readOnly": False, "subPath": "sflow-kernel-cache"}


def test_task_pod_no_init_container_without_ensure_writable():
    m = render_task_pod(
        pod_name="t", image="img:1", configmap_name="t-cfg",
        pvc_mounts=[{"name": "v", "claim": "c", "mount_path": "/data",
                     "read_only": False}],
    )
    assert "initContainers" not in m["spec"]


# ---------------------------------------------------------------------------
# render_probe_pod_manifest (in-cluster probe pod)
# ---------------------------------------------------------------------------


def test_probe_pod_manifest_defaults():
    m = render_probe_pod_manifest(pod_name="sflow-probe-x", allocation_id="abc")
    assert m["apiVersion"] == "v1"
    assert m["kind"] == "Pod"
    assert m["metadata"]["name"] == "sflow-probe-x"
    assert m["metadata"]["labels"][SFLOW_ALLOC_LABEL] == "abc"
    assert m["metadata"]["labels"][SFLOW_ROLE_LABEL] == PROBE_ROLE
    container = m["spec"]["containers"][0]
    assert container["image"] == PROBE_POD_IMAGE_DEFAULT
    assert container["command"][0] == "sh"
    # A probe pod holds no GPUs and needs no placement affinity.
    assert "resources" not in container
    assert "affinity" not in m["spec"]
    assert "nodeSelector" not in m["spec"]


def test_probe_pod_manifest_overrides():
    m = render_probe_pod_manifest(
        pod_name="p",
        allocation_id="a",
        image="myrepo/curl:1",
        namespace="ns",
        image_pull_policy="IfNotPresent",
        node_selector={"disktype": "ssd"},
        tolerations=[
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ],
    )
    assert m["metadata"]["namespace"] == "ns"
    container = m["spec"]["containers"][0]
    assert container["image"] == "myrepo/curl:1"
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert m["spec"]["nodeSelector"] == {"disktype": "ssd"}
    assert m["spec"]["tolerations"][0]["key"] == "nvidia.com/gpu"
