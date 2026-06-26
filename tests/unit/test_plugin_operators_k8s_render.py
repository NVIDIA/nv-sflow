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


def test_reservation_pod_labels_and_anti_affinity_scoped_to_allocation():
    m = render_reservation_pod_manifest(pod_name="pod-0", allocation_id="abc")
    assert m["metadata"]["labels"][SFLOW_ALLOC_LABEL] == "abc"
    assert m["metadata"]["labels"]["sflow.ai/role"] == "reservation"
    anti = m["spec"]["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ][0]
    assert anti["labelSelector"]["matchLabels"] == {SFLOW_ALLOC_LABEL: "abc"}
    assert anti["topologyKey"] == "kubernetes.io/hostname"


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
