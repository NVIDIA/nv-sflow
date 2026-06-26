# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative manifest builders for the kubernetes operator.

This module owns the *declarative* half of the kubernetes operator: it returns
plain manifest dicts for ``kubectl apply``. The *imperative* half (name
sanitization, env secrets, the kubectl wrapper script) lives in ``_k8s_shell``.

GPUs are requested two ways, selected by the backend ``scheduling`` field:

* ``dra``           -> a ``resource.k8s.io/v1`` ``ResourceClaimTemplate`` from a
                       DeviceClass (default ``gpu.nvidia.com``), referenced by the
                       pod's ``spec.resourceClaims`` + ``container.resources.claims``.
* ``device_plugin`` -> the legacy ``nvidia.com/gpu`` container limit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Label applied to reservation/placeholder pods (and the ComputeDomain / GPU
# ResourceClaimTemplate) so the backend can select/clean up everything belonging
# to a single allocation.
SFLOW_ALLOC_LABEL = "sflow.ai/allocation"

# Label applied to all objects a single task renders (pods, ConfigMap, per-task
# ResourceClaimTemplate) so the operator can clean them up as a group.
SFLOW_TASK_LABEL = "sflow.ai/task"

# Standard node label carrying the node's hostname; the placement primitive for
# pinning pods onto reserved nodes (nodeSelector / nodeAffinity / anti-affinity).
HOSTNAME_LABEL = "kubernetes.io/hostname"

# Core DRA (Dynamic Resource Allocation) API. GA in Kubernetes 1.34+.
RESOURCE_API_VERSION = "resource.k8s.io/v1"
# In-pod name for the GPU ResourceClaim (referenced by container.resources.claims).
GPU_CLAIM_NAME = "gpu"

# NVIDIA DRA ComputeDomain (Multi-Node NVLink / IMEX) constants.
COMPUTE_DOMAIN_API_VERSION = "resource.nvidia.com/v1beta1"
COMPUTE_DOMAIN_KIND = "ComputeDomain"
# In-pod name referencing the ComputeDomain channel ResourceClaimTemplate.
COMPUTE_DOMAIN_CLAIM_NAME = "compute-domain-channel"

# Fixed, non-user-configurable image for reservation/placeholder pods. The
# Kubernetes backend has no `image` field (workload images come from each task's
# operator); reservation pods are pure placeholders that only hold/discover
# nodes (+ their GPUs), so a small sleeper image is all they need. `bash:5` is
# alpine-based (~13MB) and ships the shell + sleep the placeholder command uses.
RESERVATION_POD_IMAGE = "bash:5"

# Where the task script ConfigMap is mounted, and the entrypoint file name.
SFLOW_SCRIPT_DIR = "/sflow"
SFLOW_ENTRYPOINT_FILE = "entrypoint.sh"
SFLOW_ENTRYPOINT_PATH = f"{SFLOW_SCRIPT_DIR}/{SFLOW_ENTRYPOINT_FILE}"
_SCRIPT_VOLUME_NAME = "sflow-scripts"

# gpu-operator typically taints GPU nodes; tolerate it by default so both the
# GPU workloads and the CPU-only infra pods (etcd/nats/frontend) can land there.
DEFAULT_GPU_TOLERATION: dict[str, Any] = {
    "key": "nvidia.com/gpu",
    "operator": "Exists",
    "effect": "NoSchedule",
}

# Placeholder command for reservation pods: a TERM-responsive sleeper. Reservation
# pods only hold/discover nodes (+ GPUs); the workload runs in a separate pinned
# task pod, so the placeholder just idles until deleted on the create-before-destroy
# handoff (or at release()).
_SLEEPER_CMD = ["sh", "-c", "trap 'exit 0' TERM INT; sleep infinity & wait"]

# Short grace period so the create-before-destroy handoff (delete placeholder ->
# the already-Pending task pod binds) frees the node's GPUs quickly instead of
# waiting out the 30s default.
_RESERVATION_TERMINATION_GRACE_SECONDS = 5


def _manifest_metadata(
    name: str, namespace: str | None, *, labels: Mapping[str, str] | None = None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": name}
    if namespace:
        metadata["namespace"] = namespace
    if labels:
        metadata["labels"] = dict(labels)
    return metadata


def _merged_node_selector(
    node_selector: Mapping[str, str] | None,
    assigned_node: str | None,
) -> dict[str, str] | None:
    """Combine a base node_selector with a single-node hostname pin.

    Pin via the standard ``kubernetes.io/hostname`` label so the scheduler still
    runs its GPU/taint checks (unlike ``spec.nodeName``, which bypasses them).
    """
    merged = dict(node_selector or {})
    if assigned_node:
        merged[HOSTNAME_LABEL] = assigned_node
    return merged or None


def _gpu_request(
    *,
    scheduling: str,
    gpu_count: int | None,
    resource_claim_name: str | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Return (container_resources, pod_resource_claims) for a GPU request.

    ``dra``           -> a container claim ref + a pod resourceClaims entry.
    ``device_plugin`` -> an ``nvidia.com/gpu`` limit, no pod-level claim.
    """
    resources: dict[str, Any] = {}
    pod_claims: list[dict[str, str]] = []
    if not gpu_count or gpu_count <= 0:
        return resources, pod_claims
    if scheduling == "dra" and resource_claim_name:
        resources["claims"] = [{"name": GPU_CLAIM_NAME}]
        pod_claims.append(
            {
                "name": GPU_CLAIM_NAME,
                "resourceClaimTemplateName": resource_claim_name,
            }
        )
    elif scheduling == "device_plugin":
        resources["limits"] = {"nvidia.com/gpu": str(gpu_count)}
    return resources, pod_claims


def render_resource_claim_template(
    *,
    name: str,
    device_class: str,
    count: int,
    selectors: Sequence[str] | None = None,
    namespace: str | None = None,
    allocation_id: str | None = None,
    task_label: str | None = None,
) -> dict[str, Any]:
    """Render a ``resource.k8s.io/v1`` ``ResourceClaimTemplate`` for GPUs.

    Kubernetes generates one ResourceClaim per consuming pod from this template,
    so each pod gets its own (exclusive) set of ``count`` devices from
    ``device_class``. ``selectors`` are optional CEL expressions narrowing the
    eligible devices.
    """
    exactly: dict[str, Any] = {
        "deviceClassName": device_class,
        "allocationMode": "ExactCount",
        "count": int(count),
    }
    if selectors:
        exactly["selectors"] = [{"cel": {"expression": str(s)}} for s in selectors]
    labels: dict[str, str] = {}
    if allocation_id:
        labels[SFLOW_ALLOC_LABEL] = allocation_id
    if task_label:
        labels[SFLOW_TASK_LABEL] = task_label
    return {
        "apiVersion": RESOURCE_API_VERSION,
        "kind": "ResourceClaimTemplate",
        "metadata": _manifest_metadata(name, namespace, labels=labels or None),
        "spec": {
            "spec": {
                "devices": {
                    "requests": [{"name": GPU_CLAIM_NAME, "exactly": exactly}]
                }
            }
        },
    }


def render_configmap(
    *,
    name: str,
    data: Mapping[str, str],
    namespace: str | None = None,
    task_label: str | None = None,
    allocation_id: str | None = None,
) -> dict[str, Any]:
    """Render a ConfigMap carrying the (non-secret) task entrypoint script."""
    labels: dict[str, str] = {}
    if task_label:
        labels[SFLOW_TASK_LABEL] = task_label
    if allocation_id:
        labels[SFLOW_ALLOC_LABEL] = allocation_id
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": _manifest_metadata(name, namespace, labels=labels or None),
        "data": dict(data),
    }


def render_compute_domain_manifest(
    *,
    name: str,
    num_nodes: int,
    channel_template_name: str,
    allocation_id: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Render an NVIDIA DRA ``ComputeDomain`` (Multi-Node NVLink / IMEX fabric).

    The DRA controller watches this CR and generates the channel
    ``ResourceClaimTemplate`` named ``channel_template_name``; multi-node task
    pods then claim a channel from it to get the IMEX fabric wired across the
    nodes the pods land on.
    """
    return {
        "apiVersion": COMPUTE_DOMAIN_API_VERSION,
        "kind": COMPUTE_DOMAIN_KIND,
        "metadata": _manifest_metadata(
            name, namespace, labels={SFLOW_ALLOC_LABEL: allocation_id}
        ),
        "spec": {
            "numNodes": num_nodes,
            "channel": {"resourceClaimTemplate": {"name": channel_template_name}},
        },
    }


def render_reservation_pod_manifest(
    *,
    pod_name: str,
    allocation_id: str,
    namespace: str | None = None,
    image_pull_policy: str | None = None,
    scheduling: str = "dra",
    gpu_count: int | None = None,
    resource_claim_name: str | None = None,
    host_network: bool = False,
    node_selector: Mapping[str, str] | None = None,
    tolerations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render a placeholder Pod that reserves a node (one per node via anti-affinity).

    The pod is a pure sleeper that **holds the node's GPUs** (hard reservation)
    via the same GPU path as tasks (a DRA claim or an ``nvidia.com/gpu`` limit),
    until a GPU task is pinned to the node and the operator deletes it on the
    create-before-destroy handoff (or release() tears the allocation down).
    """
    container: dict[str, Any] = {
        "name": "reserve",
        "image": RESERVATION_POD_IMAGE,
        "command": list(_SLEEPER_CMD),
    }
    if image_pull_policy:
        container["imagePullPolicy"] = image_pull_policy
    resources, pod_claims = _gpu_request(
        scheduling=scheduling,
        gpu_count=gpu_count,
        resource_claim_name=resource_claim_name,
    )
    if resources:
        container["resources"] = resources

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": _RESERVATION_TERMINATION_GRACE_SECONDS,
        "containers": [container],
        "affinity": {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {
                            "matchLabels": {SFLOW_ALLOC_LABEL: allocation_id}
                        },
                        "topologyKey": HOSTNAME_LABEL,
                    }
                ]
            }
        },
    }
    if pod_claims:
        pod_spec["resourceClaims"] = pod_claims
    if host_network:
        pod_spec["hostNetwork"] = True
    if node_selector:
        pod_spec["nodeSelector"] = dict(node_selector)
    if tolerations:
        pod_spec["tolerations"] = [dict(t) for t in tolerations]

    metadata = _manifest_metadata(
        pod_name,
        namespace,
        labels={SFLOW_ALLOC_LABEL: allocation_id, "sflow.ai/role": "reservation"},
    )
    return {"apiVersion": "v1", "kind": "Pod", "metadata": metadata, "spec": pod_spec}


def render_task_pod(
    *,
    pod_name: str,
    image: str,
    configmap_name: str,
    namespace: str | None = None,
    image_pull_policy: str | None = None,
    restart_policy: str = "Never",
    env_secret_name: str | None = None,
    scheduling: str = "dra",
    per_pod_gpus: int | None = None,
    resource_claim_name: str | None = None,
    host_network: bool = False,
    node_selector: Mapping[str, str] | None = None,
    assigned_node: str | None = None,
    tolerations: Sequence[Mapping[str, Any]] | None = None,
    extra_env: Mapping[str, str] | None = None,
    compute_domain_channel: str | None = None,
    task_label: str | None = None,
    allocation_id: str | None = None,
) -> dict[str, Any]:
    """Render one task Pod manifest (dict) for ``kubectl apply``.

    The user script is mounted from ``configmap_name`` and run as the container
    entrypoint. GPUs come from a DRA ResourceClaimTemplate or an ``nvidia.com/gpu``
    limit (see ``scheduling``). ``assigned_node`` pins the pod via a
    ``kubernetes.io/hostname`` nodeSelector; ``extra_env`` carries per-pod literals
    (e.g. ``SFLOW_TASK_NODE_INDEX`` / ``SFLOW_LEADER_ADDRESS`` for multi-node).
    """
    container: dict[str, Any] = {
        "name": pod_name,
        "image": image,
        "command": ["bash", "-l", SFLOW_ENTRYPOINT_PATH],
        "volumeMounts": [
            {"name": _SCRIPT_VOLUME_NAME, "mountPath": SFLOW_SCRIPT_DIR}
        ],
    }
    if image_pull_policy:
        container["imagePullPolicy"] = image_pull_policy
    if env_secret_name:
        container["envFrom"] = [{"secretRef": {"name": env_secret_name}}]
    if extra_env:
        container["env"] = [
            {"name": str(k), "value": str(v)} for k, v in extra_env.items()
        ]

    resources, pod_claims = _gpu_request(
        scheduling=scheduling,
        gpu_count=per_pod_gpus,
        resource_claim_name=resource_claim_name,
    )
    if compute_domain_channel:
        resources.setdefault("claims", []).append({"name": COMPUTE_DOMAIN_CLAIM_NAME})
        pod_claims.append(
            {
                "name": COMPUTE_DOMAIN_CLAIM_NAME,
                "resourceClaimTemplateName": compute_domain_channel,
            }
        )
    if resources:
        container["resources"] = resources

    pod_spec: dict[str, Any] = {
        "restartPolicy": restart_policy,
        "containers": [container],
        "volumes": [
            {"name": _SCRIPT_VOLUME_NAME, "configMap": {"name": configmap_name}}
        ],
    }
    if pod_claims:
        pod_spec["resourceClaims"] = pod_claims
    if host_network:
        pod_spec["hostNetwork"] = True
    effective_selector = _merged_node_selector(node_selector, assigned_node)
    if effective_selector:
        pod_spec["nodeSelector"] = effective_selector
    if tolerations:
        pod_spec["tolerations"] = [dict(t) for t in tolerations]

    labels: dict[str, str] = {}
    if task_label:
        labels[SFLOW_TASK_LABEL] = task_label
    if allocation_id:
        labels[SFLOW_ALLOC_LABEL] = allocation_id
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": _manifest_metadata(pod_name, namespace, labels=labels or None),
        "spec": pod_spec,
    }
