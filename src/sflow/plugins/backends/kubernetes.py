# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
import warnings
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from sflow.config.schema import BackendConfig, Resolvable, is_expression
from sflow.core.backend import (
    Allocation,
    Backend,
    BackendCapabilities,
    configure_bare_monitor_operator,
)
from sflow.core.backend_registry import register_backend
from sflow.core.compute_node import ComputeNode
from sflow.core.kubectl_config import kubectl_global_args
from sflow.core.operator import Operator
from sflow.logging import get_logger
from sflow.plugins.k8s.render import (
    DEFAULT_GPU_TOLERATION,
    PROBE_POD_IMAGE_DEFAULT,
    RESOURCE_API_VERSION,
    SFLOW_ALLOC_LABEL,
    render_compute_domain_manifest,
    render_probe_pod_manifest,
    render_reservation_pod_manifest,
    render_resource_claim_template,
)
from sflow.plugins.k8s.capabilities import (
    COMPUTE_DOMAIN,
    MPI_OPERATOR,
    CapabilityState,
    ClusterCapability,
    detect_capability_state,
)
from sflow.plugins.k8s.shell import sanitize_name
from sflow.plugins.k8s.probe import K8sExecProbeTransport
from sflow.plugins.k8s.rdma import (
    RDMA_PROVIDER_KEYS,
    RdmaDetectContext,
    RdmaPlan,
    detect_rdma,
)
from sflow.utils.node_filters import normalize_node_list, resolve_node_filters

_logger = get_logger(__name__)

_RESERVE_POLL_INTERVAL = 2

# How often (seconds) to emit a "still waiting" heartbeat while a reservation pod
# is unscheduled, in addition to logging immediately whenever the phase changes.
# Overridable at runtime via $SFLOW_K8S_WAIT_HEARTBEAT_SECS.
_RESERVE_HEARTBEAT_INTERVAL = 30

# Shell for the reservation-stage node-topology probe (log-only): the pod's cpuset
# size + the CPU-bearing NUMA nodes (Grace/GB boxes also expose CPU-less GPU-HBM NUMA
# nodes, which the grep drops) + each GPU's CPU/NUMA affinity (nvidia-smi topo). The
# trailing ``; true`` forces exit 0 so a node missing numactl/nvidia-smi still logs
# what it has (only a failed ``kubectl exec`` -- pod not ready -- yields non-zero).
_NODE_TOPOLOGY_PROBE_SH = (
    'echo "nproc=$(nproc)"; '
    'echo "cpuset=$(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null)"; '
    "command -v numactl >/dev/null 2>&1 && "
    "numactl -H 2>/dev/null | grep -E 'available:|node [0-9]+ cpus: [0-9]'; "
    "command -v nvidia-smi >/dev/null 2>&1 && "
    "{ echo '--- nvidia-smi topo -m ---'; nvidia-smi topo -m 2>/dev/null; } ; true"
)

# How GPUs are requested for both placeholder and task pods. Default: device_plugin.
#   device_plugin -> nvidia.com/gpu device-plugin limit (default; widely available)
#   dra           -> resource.k8s.io ResourceClaimTemplate (nvidia-dra-driver-gpu,
#                    Kubernetes 1.34+). NOTE: DRA GPU allocation is a work in progress.
SchedulingMode = Literal["device_plugin", "dra"]

# RDMA/IB provisioning for task GPU pods (see k8s.rdma.py). "auto" runs the
# provider chain (GKE -> shared-device-plugin -> host-device) and then exposes all
# node NICs so NCCL/UCX auto-select each GPU's closest device; "disable" turns RDMA
# off (named "disable" not "off" so unquoted YAML does not coerce it to the bool
# False -- the "Norway problem"); a specific provider key forces that mechanism.
RdmaMode = Literal["auto", "disable", "gke", "shared_device_plugin", "host_device"]

# Merge-pod tri-state (see ``merge_colocated_gpu_pods``). ``disable``/``False``
# disable merging; ``auto`` (default) and ``on``/``True`` enable it. Resolved to a
# bool (via a ``@property``) so ``_plan_merge_groups`` -- which reads that bool and
# already self-guards to >=2 co-located GPU tasks -- is unchanged. The string
# turn-off token is ``disable`` (not ``off``, which unquoted YAML coerces to the bool
# False -- the "Norway problem"); the field also accepts a real ``bool``.
MergeMode = Literal["auto", "on", "disable"]

# NVLink-domain scope. ``auto`` detects ``node`` vs ``rack`` from the cluster's GPU
# product + ComputeDomain CRD presence (component 2); ``node``/``rack``/``disable``
# override that detection (``disable`` -> no NVLink-domain co-location). The turn-off
# token is ``disable`` (not ``off``) so an unquoted YAML value is not coerced to the
# bool False -- the "Norway problem". The scope is advisory (it drives
# interconnect-priority warnings + straddle validation) and, together with
# ``dra.nvlink_domain_label_key``, gates the reservation-pod NVLink-domain co-location
# affinity -- see the ``nvlink_domain`` field for exactly what it changes.
NvlinkDomain = Literal["auto", "node", "rack", "disable"]


def _merge_enabled(value: bool | str) -> bool:
    """Resolve the ``merge_colocated_gpu_pods`` tri-state to a bool.

    ``disable``/``False`` (and empty) disable merging; everything else (``auto``
    default, ``on``, ``True``) enables it. Legacy ``off``/``no``/``0`` strings are
    still honored as disable, and an unquoted YAML ``off`` arrives here as ``False``.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("disable", "off", "false", "no", "0", "")


# GPU product substrings (matched against the ``nvidia.com/gpu.product`` node
# label, upper-cased) used to detect the NVLink-domain scope. RACK-class boards
# form a rack-wide MNNVL fabric (NVL72) when an IMEX ComputeDomain driver is
# present; NODE-class boards have NVSwitch/NVLink within a node only.
_RACK_SCOPE_PRODUCTS = ("GB200", "GB300")
_NODE_SCOPE_PRODUCTS = (
    "B200",
    "B300",
    "H100",
    "H200",
    "H800",
    "A100",
    "A800",
    "GH200",
    "H20",
)
# kubelet >= v1.29 fixed the `kubectl logs -f` follow-stall after a container log
# rotation (kubernetes/kubernetes#115701 -> #115702). On older kubelets the
# offloaded follow that writes <task>.log can silently stop after a rotation, so a
# log_watch probe reading <task>.log may miss its readiness/failure marker until the
# terminal finalize re-fetch -- preflight warns (see _preflight_check_kubelet_version).
_KUBELET_LOGS_FOLLOW_FIXED_MINOR = (1, 29)


def _parse_k8s_minor(version: str) -> tuple[int, int] | None:
    """Extract ``(major, minor)`` from a k8s version string, or None.

    Handles the usual forms -- ``v1.29.0``, ``1.28.4``, ``v1.30.2+k3s1``,
    ``v1.28.4-gke.1234`` -> ``(1, 29)`` / ``(1, 28)`` / ``(1, 30)`` / ``(1, 28)``.
    The minor field may carry a distro suffix on rare builds, so only its leading
    digits are read.
    """
    v = version.strip().lstrip("vV")
    parts = v.split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
    except ValueError:
        return None
    minor_digits = ""
    for ch in parts[1]:
        if ch.isdigit():
            minor_digits += ch
        else:
            break
    if not minor_digits:
        return None
    return major, int(minor_digits)


# Optional cluster infra sflow can opportunistically use (ComputeDomain/IMEX for
# rack-scope NVLink, the Kubeflow MPI Operator for the k8s_mpi operator route, ...)
# is modeled uniformly as ClusterCapability + CapabilityState -- see
# k8s.capabilities and the backend's detect_capability/capability_state.


class KubernetesDraConfig(BaseModel):
    """DRA (Dynamic Resource Allocation) options for ``scheduling: dra``.

    DRA GPU allocation is a **work in progress**; the default GPU request mode is
    ``device_plugin`` (``nvidia.com/gpu``). Opt into DRA with ``scheduling: dra`` on a
    cluster running ``nvidia-dra-driver-gpu`` (Kubernetes 1.34+).
    """

    # DeviceClass GPUs are requested from (nvidia-dra-driver-gpu default).
    gpu_device_class: Resolvable[str] = "gpu.nvidia.com"
    # Optional CEL expressions narrowing eligible devices (e.g. by product/memory).
    device_selectors: list[Resolvable[str]] | None = None
    # Node label KEY identifying the physical NVLink domain (e.g.
    # `nvidia.com/gpu.clique`). Used ONLY for placement + straddle validation
    # (component 6) on clusters with MULTIPLE NVLink domains -- NOT for the IMEX
    # channel claim. Redundant on a single-domain (one NVL72 rack) cluster.
    nvlink_domain_label_key: str | None = None
    # GPU<->NIC topology co-allocation (opt-in). When set, each task GPU
    # ResourceClaim additionally requests a NIC from this DeviceClass with a
    # ``constraints.matchAttribute`` (see ``rdma_match_attribute``) so the
    # scheduler places the GPU and NIC on the same PCIe root complex. Requires a
    # NIC DRA driver that publishes the match attribute (e.g. NVIDIA
    # ``rdma.nvidia.com``, DRANET ``dra.net``, GKE ``mrdma.google.com``). Left
    # unset -> GPU-only claim (today's behavior); the NIC comes from the `rdma`
    # device-plugin/host-device path instead.
    rdma_device_class: Resolvable[str] | None = None
    # Attribute the GPU and NIC requests must share. Default is the standardized
    # PCIe root complex key (KEP-4381), published by both the NVIDIA GPU DRA
    # driver and DRANET. NOTE: on GB300/Vera Rubin/Fractal the NIC's matching root
    # is its Data-Direct sub-interface root, which some NIC DRA drivers do not yet
    # expose -- override per cluster if co-allocation finds no candidates.
    rdma_match_attribute: Resolvable[str] = "resource.kubernetes.io/pcieRoot"


class KubernetesComputeDomainConfig(BaseModel):
    """Multi-Node NVLink (IMEX ComputeDomain) options -- a top-level backend block.

    Independent of GPU scheduling: the NVIDIA DRA driver's ComputeDomain capability
    can run on its own ("ComputeDomain-only mode"), so this works with BOTH
    ``scheduling: device_plugin`` and ``scheduling: dra``. NVIDIA's docs confirm
    ComputeDomain and DRA GPU allocation are independent features of the driver,
    which is why this is NOT nested under ``dra:``.

    (Migrated from ``dra.use_compute_domain_channel`` / ``dra.create_compute_domain``,
    both still accepted as deprecated aliases -- see KubernetesBackendConfig.)
    """

    # JOIN an existing ComputeDomain: the channel ResourceClaimTemplate name every
    # GPU pod claims to get an IMEX channel (the runtime wiring that lets the MNNVL
    # fabric span nodes). Value is the CR's spec.channel.resourceClaimTemplate.name,
    # or ``auto`` (claim the sole existing ComputeDomain when there is exactly one),
    # or ``off``/empty/None (no claim). If set to a ComputeDomain CR name by mistake,
    # sflow best-effort auto-corrects it to that domain's channel template name (they
    # usually differ, e.g. CR ``<domain>`` -> template ``cd-<domain>``). Joining needs
    # no ``computedomains`` RBAC (only claims a pre-existing template).
    channel: Resolvable[str] | None = None
    # CREATE an NVIDIA ComputeDomain CR (so pods get a fresh IMEX channel) instead of
    # joining an existing one. Admin/privileged: needs RBAC to create
    # ``computedomains.resource.nvidia.com``. Default off -- the default path is to
    # join an existing domain (see ``channel``, incl. ``channel: auto``).
    create: bool = False


class KubernetesEmptyDirConfig(BaseModel):
    """An ephemeral, node-local scratch volume (Kubernetes ``emptyDir``).

    Unlike a PVC, an ``emptyDir`` is created empty for each pod, is writable by
    any container user (no PVC/NFS ownership or root-squash problems), and is
    deleted when the pod is removed. It is the right choice for scratch that does
    not need to persist across runs -- e.g. the JIT/kernel cache -- and, being a
    real volume, does not count against the container image layer's storage.
    """

    # emptyDir backing: "" (default) uses the node's disk (ephemeral-storage);
    # "Memory" uses a tmpfs (RAM-backed -- avoid for large kernel caches).
    medium: Literal["", "Memory"] = ""
    # Optional cap on the volume size (e.g. "50Gi"). None -> no explicit limit.
    size_limit: Resolvable[str] | None = None


class KubernetesVolumeConfig(BaseModel):
    """A volume to mount into task pods: a pre-existing PVC or an ``emptyDir``.

    A backend ``volumes:`` entry is workflow-wide storage: it is mounted into
    EVERY task pod of this backend (as a pod volume + container volumeMount at
    ``mount_path``), regardless of whether a task's script references that path.
    This is intentional -- it is backend-level info and must be available to all
    sflow tasks on the backend (e.g. a task that loads a model via discovery,
    without ever naming the path, still needs it mounted).

    Set exactly one source:

    * ``claim`` -- an existing PersistentVolumeClaim (the PVC + its data must
      already exist in the backend namespace; sflow only references it). Common
      for cluster-resident data (e.g. a model on shared storage): declare the PVC
      + where it mounts, then point an ``fs://`` artifact at a path under
      ``mount_path``. When an ``fs://`` artifact path is covered by a declared PVC
      ``mount_path`` the PVC serves it, so the per-artifact hostPath fallback is
      skipped -- that path matching governs ONLY the hostPath fallback, never
      whether the PVC itself is mounted (it always is).
    * ``empty_dir`` -- a per-pod ephemeral scratch volume (see
      :class:`KubernetesEmptyDirConfig`). Writable by any container user; ideal
      for a JIT/kernel cache that need not persist across runs.
    """

    # Pod volume name (DNS-1123); also the key linking volume <-> volumeMount.
    name: str
    # Name of an existing PersistentVolumeClaim (spec.volumes[].pvc.claimName).
    # Mutually exclusive with ``empty_dir``.
    claim: Resolvable[str] | None = None
    # An ephemeral scratch volume instead of a PVC. Mutually exclusive with ``claim``.
    empty_dir: KubernetesEmptyDirConfig | None = None
    # Absolute path the volume is mounted at inside each task pod.
    mount_path: Resolvable[str]
    # Optional path within the volume to mount (volumeMount.subPath).
    sub_path: Resolvable[str] | None = None
    # Mount read-only. Default (None) resolves per source: PVCs default read-only
    # (model/data are typically read-only, and this is required to share one PVC
    # across pods on multiple nodes), while an emptyDir defaults writable (that is
    # the point of scratch). Set explicitly to override.
    read_only: bool | None = None
    # Fix the classic "subPath created root-owned -> non-root pod can't write it"
    # gotcha: inject a root initContainer that ``mkdir -p`` + ``chmod 0777`` the
    # mounted path (the subPath dir, or the mount root if no sub_path) before the
    # workload runs. Best-effort -- on a root-squashed / read-only backing volume
    # it cannot help (nothing can, from in-cluster). Requires ``read_only: false``
    # and a PVC ``claim`` (an emptyDir is already writable).
    ensure_writable: bool = False

    @field_validator("mount_path")
    @classmethod
    def _mount_path_absolute(cls, v: Resolvable[str]) -> Resolvable[str]:
        # Skip template expressions; they are validated after resolution.
        if isinstance(v, str) and "${{" not in v and not v.startswith("/"):
            raise ValueError(f"volume mount_path must be absolute, got: {v!r}")
        return v

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "KubernetesVolumeConfig":
        if (self.claim is None) == (self.empty_dir is None):
            raise ValueError(
                f"kubernetes volume '{self.name}' must set exactly one of "
                "'claim' or 'empty_dir'"
            )
        if self.ensure_writable:
            if self.empty_dir is not None:
                raise ValueError(
                    f"kubernetes volume '{self.name}': ensure_writable applies to "
                    "PVC volumes only (an emptyDir is already writable)"
                )
            if self.effective_read_only():
                raise ValueError(
                    f"kubernetes volume '{self.name}': ensure_writable requires "
                    "read_only: false"
                )
        return self

    def effective_read_only(self) -> bool:
        """Resolve the read-only default per source (PVC=True, emptyDir=False)."""
        if self.read_only is not None:
            return self.read_only
        return self.empty_dir is None


class KubernetesReservationConfig(BaseModel):
    """Tuning for the Kubernetes backend's node reservation."""

    # Seconds to wait for every placeholder pod to be scheduled onto a node.
    timeout: Resolvable[int] = 600
    # Container image for the reservation PLACEHOLDER pods -- the lightweight sleeper
    # that holds a node + its GPUs until the real task pod swaps in. It never runs the
    # workload (set the workload image on the operator). None -> the built-in
    # ``bash:5`` (RESERVATION_POD_IMAGE). Override for air-gapped clusters /
    # private-registry mirrors that can't pull from Docker Hub.
    placeholder_image: Resolvable[str] | None = None
    # GPU node handoff order. sflow reserves nodes with placeholder pods (which hold the
    # GPUs), then swaps in the task pod. Default create-before-destroy applies the task pod
    # first (Pending) then deletes the placeholder -> no node-loss gap; BUT under a namespace
    # ResourceQuota on GPUs it double-counts placeholder + task requests, so the task pod is
    # rejected at admission ("exceeded quota"). destroy_before_create deletes the placeholder
    # first (frees its quota) then applies the task -> quota-safe, small node-loss window.
    #   "auto" (default): destroy_before_create IFF a GPU ResourceQuota is detected in the
    #   namespace at allocation time, else create_before_destroy.
    handoff: Resolvable[str] = "auto"

    @field_validator("handoff")
    @classmethod
    def _handoff_must_be_known(cls, v: Any) -> Any:
        # Allow unresolved expressions (resolved in resolve_config); validate concrete
        # values against the known set -- so a typo cannot silently fall through to "auto".
        allowed = ("auto", "create_before_destroy", "destroy_before_create")
        if is_expression(v) or v in allowed:
            return v
        raise ValueError(
            f"reservation.handoff must be one of {list(allowed)} or an expression, got {v!r}"
        )


class KubernetesBackendConfig(BackendConfig):
    type: Literal["kubernetes"] = "kubernetes"
    # NOTE: there is intentionally no `image` field. Workload images are an
    # operator concern (the `k8s` operator's `image:`); reservation/placeholder
    # pods use the fixed internal sleeper image (RESERVATION_POD_IMAGE).
    namespace: Resolvable[str] | None = None
    image_pull_policy: Resolvable[str] | None = None
    # Image for the per-allocation probe pod that runs in-cluster TCP/HTTP
    # readiness/failure checks on the driver's behalf (see k8s.probe). It only
    # needs `curl`. Override for air-gapped/mirror registries. In-cluster probing
    # is on by default for the k8s backend; set env SFLOW_K8S_PROBE_VIA_POD=0 to
    # disable it and probe directly from the sflow driver host instead.
    probe_pod_image: Resolvable[str] = PROBE_POD_IMAGE_DEFAULT
    nodes: Resolvable[int] = 1
    extra_args: list[Resolvable[str]] | None = None
    # Node selector applied to all pods created under this backend.
    node_selector: dict[str, str] | None = None
    # Use host networking for all pods (pod IP == node IP, needed when other
    # tasks address this backend's nodes by the IPs returned from allocation).
    host_network: bool = True
    # Share the node's IPC namespace across task pods (pod spec hostIPC) and mount a
    # shared hostPath /dev/shm. This lets co-located pods do cross-pod CUDA IPC, so
    # NIXL/UCX KV transfer between same-node prefill/decode workers can use NVLink
    # (cuda_ipc) instead of falling back to TCP. Off by default (privileged; a pod
    # can see the node's SysV IPC + shared memory). NOTE: on GB200/GB300 the NVLink
    # KV path is Multi-Node NVLink (MNNVL) fabric handles, which additionally needs
    # UCX_CUDA_IPC_ENABLE_MNNVL=y + vLLM VMM allocation (--enable-cumem-allocator)
    # and an IMEX domain (nvidia-imex, or DRA `compute_domain.create`).
    host_ipc: bool = False
    # Merge co-located GPU tasks into ONE pod (single container requesting the
    # union of their GPUs) so intra-node NVLink/cuda_ipc works between them. When
    # the planner assigns >=2 single-node GPU tasks to the same physical node,
    # they run as concurrent background processes in one pod; each keeps its own
    # <task>.log, probes, readiness, and dependents. Tri-state (bool |
    # "auto"/"on"/"disable"), default "auto" (ENABLED): merging is sflow-owned pod
    # topology -- it enables intra-node NVLink/cuda_ipc between co-located workers
    # AND guarantees one channel-claiming GPU pod per node (the NVIDIA driver
    # publishes one IMEX channel per node). Only concurrent GPU tasks merge;
    # CPU-only infra (etcd/nats/frontend) stays in its own pod. Set "disable"/False to
    # opt out (a bool toggle, so an unquoted YAML `off` -> False also disables).
    merge_colocated_gpu_pods: MergeMode | bool = "auto"
    # NVLink-domain scope -- ADVISORY ONLY; almost every recipe leaves this at "auto"
    # (the default), which detects node|rack from the GPU product + ComputeDomain CRD.
    #
    # This is NOT the Multi-Node NVLink switch: cross-node NVLink (MNNVL) is enabled by
    # `compute_domain.channel`, NOT by this field. Setting `nvlink_domain` does NOT pin
    # NCCL/UCX transport, inject any env, or change GPU scheduling.
    #
    # Set "node"/"rack"/"disable" ONLY to correct a wrong auto-detection (e.g. the
    # ComputeDomain-CRD probe lacks RBAC and degrades to "disable"). The resolved scope
    # only affects, best-effort and warn-only (never hard-fails):
    #   * the cross-node interconnect-priority hint warning + straddle validation; and
    #   * WITH dra.nvlink_domain_label_key, on a cluster that has MULTIPLE NVLink
    #     domains, a reservation-pod podAffinity (topologyKey = that label) that
    #     co-locates a task's pods in ONE physical domain -- and even that has NO
    #     manifest effect without the label key (single-domain clusters need none).
    nvlink_domain: NvlinkDomain = "auto"
    # GPU request mode (see SchedulingMode). Default: device_plugin (nvidia.com/gpu).
    scheduling: SchedulingMode = "device_plugin"
    # Extended-resource name GPUs are requested under in ``device_plugin`` mode
    # (default ``nvidia.com/gpu``, the NVIDIA device plugin). Override for clusters
    # that expose GPUs under a different name (a MIG profile like
    # ``nvidia.com/mig-1g.5gb``, or another vendor/plugin). Ignored under
    # ``scheduling: dra`` -- use ``dra.gpu_device_class`` there.
    gpu_resource_name: Resolvable[str] = "nvidia.com/gpu"
    # Node label KEY carrying the GPU product string, read for NVLink-domain scope
    # auto-detection (default ``nvidia.com/gpu.product``, the GPU-Operator / GFD
    # convention). Override for clusters that publish the product under a different
    # label key. The resolved scope is advisory/warn-only, so a wrong key only
    # degrades detection (correctable via ``nvlink_domain``).
    gpu_product_label_key: Resolvable[str] = "nvidia.com/gpu.product"
    # DRA GPU-allocation options (used when scheduling == "dra"; DRA is a WIP).
    dra: KubernetesDraConfig | None = None
    # Multi-Node NVLink (IMEX ComputeDomain) options. Independent of `scheduling`
    # (works with device_plugin or dra) -- see KubernetesComputeDomainConfig. The
    # pre-refactor `dra.use_compute_domain_channel` / `dra.create_compute_domain`
    # (and older `dra.compute_domain_channel` / `dra.compute_domain`) still work,
    # migrated here with a DeprecationWarning by `_migrate_legacy_compute_domain`.
    compute_domain: KubernetesComputeDomainConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_compute_domain(cls, data: Any) -> Any:
        """Migrate the pre-refactor ComputeDomain keys out of ``dra:`` into the
        top-level ``compute_domain:`` block.

        ComputeDomain (Multi-Node NVLink / IMEX) is independent of DRA GPU
        scheduling, so it moved from ``dra.{use_compute_domain_channel,
        create_compute_domain}`` (and the older ``dra.{compute_domain_channel,
        compute_domain}`` aliases) to ``compute_domain.{channel,create}``. The old
        keys are still accepted with a one-time ``DeprecationWarning``.
        """
        if not isinstance(data, dict):
            return data
        dra = data.get("dra")
        if not isinstance(dra, dict):
            return data
        migrated: dict[str, Any] = {}
        # newest alias first so it wins if both are (wrongly) present
        for old in ("use_compute_domain_channel", "compute_domain_channel"):
            if old in dra:
                value = dra.pop(old)
                if "channel" not in migrated:
                    migrated["channel"] = value
                warnings.warn(
                    f"KubernetesBackendConfig 'dra.{old}' is deprecated; use the "
                    "top-level 'compute_domain.channel' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        for old in ("create_compute_domain", "compute_domain"):
            if old in dra:
                value = dra.pop(old)
                if "create" not in migrated:
                    migrated["create"] = value
                warnings.warn(
                    f"KubernetesBackendConfig 'dra.{old}' is deprecated; use the "
                    "top-level 'compute_domain.create' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        if migrated:
            existing = data.get("compute_domain")
            if isinstance(existing, dict):
                for k, v in migrated.items():
                    existing.setdefault(k, v)
            elif existing is None:
                data["compute_domain"] = migrated
        return data
    # Pod tolerations applied to placeholder + task pods. None -> tolerate
    # nvidia.com/gpu so pods can land on gpu-operator-tainted GPU nodes.
    tolerations: list[dict[str, Any]] | None = None
    # Pre-existing PersistentVolumeClaims to mount into task pods (e.g. shared model
    # storage that fs:// artifacts point into). See KubernetesVolumeConfig.
    volumes: list[KubernetesVolumeConfig] | None = None
    # RDMA/IB provisioning for task GPU pods (disaggregated KV transfer + NCCL):
    #   "auto" (default) -> at reservation, detect RDMA via the provider chain
    #                       (GKE multi-NIC, k8s-rdma-shared-dev-plugin rdma/*, or a
    #                       generic host-device /dev/infiniband fallback), grant the
    #                       pods verbs access + CAP_IPC_LOCK, then expose all node
    #                       NICs so NCCL/UCX auto-select each GPU's closest device
    #                       (verified in-pod; TCP fallback if RDMA is unusable).
    #   "disable"        -> cleanly turn RDMA off (force NCCL onto sockets:
    #                       NCCL_IB_DISABLE=1 / NCCL_IBEXT_DISABLE=1 /
    #                       NCCL_NET_PLUGIN=none). Named "disable" (not "off") so
    #                       unquoted YAML does not coerce it to the bool False.
    #   "gke" | "shared_device_plugin" | "host_device"
    #                    -> force that provider (skip auto-detection order; e.g. opt
    #                       out of the privileged-ish host-device path).
    # Best-effort: if nothing is found, nothing is injected. Fine-tune NIC selection
    # per pod at runtime with the SFLOW_RDMA_AFFINITY env (auto|explicit|off).
    rdma: RdmaMode = "auto"
    # Namespace to look in for the GKE gIB installer (``nccl-rdma-installer``
    # DaemonSet) when detecting GPUDirect-RDMA host paths. Default ``kube-system``
    # (the installer's standard namespace); override if your cluster installs it
    # elsewhere. A wrong value only makes sflow skip the gIB lib mounts (warn-only;
    # multi-node NCCL falls back to the untuned built-in IB transport).
    gib_installer_namespace: Resolvable[str] = "kube-system"
    # CPU-request policy for task pods (requests-only: sflow never sets a CPU limit).
    # OPT-IN and UNSET by default: when neither is set, NO cpu request is injected and
    # pods run unconstrained (BestEffort CPU). Set ``cpu_per_gpu`` to give GPU task
    # pods ``cpu_per_gpu * per-pod GPUs`` cores and/or ``cpu_request`` to give pods
    # without GPUs that many cores -- a cgroup-weight floor so a CPU-hungry pod (e.g.
    # an aiperf client) isn't starved BestEffort. Overridable per task via the operator
    # ``cpu`` field. A value of 0 also injects no request for that class.
    cpu_per_gpu: Resolvable[int] | None = None
    cpu_request: Resolvable[int] | None = None
    # Default per-file cap for auto-collecting a k8s task's node-local output dir back
    # to the driver (see the ``k8s`` operator ``collect_max_file_size``). Applies to
    # every k8s / k8s_mpi task on this backend unless the operator sets its own. Bytes
    # (int) or a size string ("10Mi"); 0 disables. None -> the built-in 10 MiB default.
    collect_max_file_size: Resolvable[int] | str | None = None
    # Grace period (seconds) the k8s/k8s_mpi operator allows for copying a task's
    # node-local output dir back to the driver before the pod is torn down. Applies
    # to every task on this backend. Default 120; increase for large output trees
    # over a slow API server.
    collect_grace_seconds: Resolvable[int] = 120
    # Optional tuning for the (always-on) reserve+discover+pin behavior.
    reservation: KubernetesReservationConfig | None = None

    def planning_node_count(self) -> Resolvable[int] | None:
        return self.nodes


@register_backend("kubernetes", KubernetesBackendConfig)
class KubernetesBackend(Backend):
    """Kubernetes backend: reserve+discover nodes, then run each task as pinned pod(s)."""

    # A failed command in a pod should fail the task by default; an explicit
    # ``fail_fast`` in the task YAML still overrides this (see Backend.default_fail_fast).
    default_fail_fast: bool = True

    def __init__(self, config: KubernetesBackendConfig):
        super().__init__(name=config.name)
        self.config = config
        resv = config.reservation
        self._reservation_timeout = (
            int(resv.timeout) if resv is not None and resv.timeout is not None else 600
        )
        # GPU node handoff order (see KubernetesReservationConfig.handoff). "auto" is
        # resolved against a detected GPU ResourceQuota in allocate(); until then it stays
        # create-before-destroy (dry-run / no-allocate paths render the classic order).
        self._reservation_handoff = (
            str(resv.handoff) if resv is not None and resv.handoff is not None else "auto"
        )
        self._placeholder_image: str | None = (
            str(resv.placeholder_image)
            if resv is not None and resv.placeholder_image is not None
            else None
        )
        self._handoff_destroy_first = False
        # The Kubernetes backend always reserves real nodes (placeholder pods),
        # discovers their names + InternalIPs, and pins task pods onto them.
        # GPUs are hard-exclusive (DRA/device-plugin), so supports_gpu_sharing is
        # False: the planner coerces GPU release_after=task_ready -> task_completion.
        self.capabilities = BackendCapabilities(
            supports_node_placement=True,
            supports_gpu_env=False,
            supports_host_path_mounts=False,
            has_runtime_node_addresses=True,
            supports_gpu_sharing=False,
            # Node-level hardware monitoring is not implemented on k8s yet (needs a
            # DCGM/DaemonSet collector); the monitor planner skips it rather than
            # sampling the sflow driver host and reporting misleading metrics.
            supports_host_monitoring=False,
        )
        # Populated in allocate(): maps a real k8s node name -> the placeholder
        # pod holding it (used for the create-before-destroy GPU handoff).
        self._node_to_resv_pod: dict[str, str] = {}
        # Formatted per-node CPU/NUMA/GPU topology captured at reservation (log-only +
        # surfaced in sflow_summary.log). None until the reservation-stage probe runs.
        self._node_topology_report: str | None = None
        # The channel ResourceClaimTemplate name GPU task pods claim (Multi-Node
        # NVLink / IMEX). Set from a named compute_domain.channel (__init__),
        # a created domain (compute_domain.create, allocate()), or `auto`
        # resolution to the sole existing domain (component 4).
        self._compute_domain_channel: str | None = None
        self._namespace = (
            str(config.namespace) if config.namespace is not None else None
        )
        self._image_pull_policy = (
            str(config.image_pull_policy)
            if config.image_pull_policy is not None
            else None
        )
        self._probe_pod_image = (
            str(config.probe_pod_image)
            if config.probe_pod_image is not None
            else PROBE_POD_IMAGE_DEFAULT
        )
        # Per-allocation probe pod. Created lazily on the first TCP/HTTP probe
        # (see _kickoff_probe_pod), so probe-less / log_watch-only workflows never
        # create one. ``_probe_pod_name`` stays None until it exists (or when
        # in-cluster probing is disabled); ``_probe_pod_task`` is the in-flight
        # creation task.
        self._probe_pod_name: str | None = None
        self._probe_pod_task: "asyncio.Task[None] | None" = None
        self._nodes = int(config.nodes) if config.nodes is not None else 1
        self._gpu_per_node = (
            int(config.gpus_per_node) if config.gpus_per_node is not None else None
        )
        # CPU-request policy for task pods (resolved to ints by resolve_config).
        # None => unset (inject no cpu request; pods run BestEffort by default).
        self._cpu_per_gpu = (
            None if config.cpu_per_gpu is None else int(config.cpu_per_gpu)
        )
        self._cpu_request = (
            None if config.cpu_request is None else int(config.cpu_request)
        )
        # Default node-local output-dir collection cap for this backend's k8s tasks
        # (raw int/str; the operator parses + applies it). None -> operator default.
        self._collect_max_file_size = config.collect_max_file_size
        self._collect_grace_seconds: int = int(config.collect_grace_seconds)
        self._extra_args = [str(a) for a in (config.extra_args or [])]
        self._node_selector: dict[str, str] | None = config.node_selector
        self._host_network: bool = bool(config.host_network)
        self._host_ipc: bool = bool(config.host_ipc)
        self._merge_colocated_gpu_pods: bool = _merge_enabled(
            config.merge_colocated_gpu_pods
        )
        # NVLink-domain scope override ("auto"/"node"/"rack"/"disable"). An explicit
        # override wins; "auto" is resolved to node|rack|disable by detection at
        # preflight/allocate (component 2), cached in _nvlink_domain_scope_detected.
        self._nvlink_domain_cfg: str = str(config.nvlink_domain or "auto")
        self._nvlink_domain_override: str | None = (
            None if self._nvlink_domain_cfg == "auto" else self._nvlink_domain_cfg
        )
        self._nvlink_domain_scope_detected: str | None = None
        self._scheduling: str = str(config.scheduling)
        # Device-plugin GPU resource name + GPU-product label key (overridable per
        # cluster; default to the NVIDIA conventions).
        self._gpu_resource_name: str = str(config.gpu_resource_name)
        self._gpu_product_label_key: str = str(config.gpu_product_label_key)
        self._gib_installer_namespace: str = str(config.gib_installer_namespace)
        dra = config.dra
        self._gpu_device_class = (
            str(dra.gpu_device_class)
            if dra is not None and dra.gpu_device_class is not None
            else "gpu.nvidia.com"
        )
        self._device_selectors: list[str] | None = (
            [str(s) for s in dra.device_selectors]
            if dra is not None and dra.device_selectors
            else None
        )
        # Multi-Node NVLink (IMEX ComputeDomain): top-level `compute_domain:` block,
        # independent of GPU scheduling. (Legacy `dra.{use_compute_domain_channel,
        # create_compute_domain}` are migrated into it by the config validator.) The
        # channel/create classification is shared with the `--kube-compute-domain-*`
        # CLI override (apply_kubectl_config) via `_configure_compute_domain`.
        cd = config.compute_domain
        self._configure_compute_domain(
            channel=cd.channel if cd is not None else None,
            create=bool(cd.create) if cd is not None else False,
        )
        # Node label key for NVLink-domain placement/validation (component 6).
        self._nvlink_domain_label_key: str | None = (
            str(dra.nvlink_domain_label_key)
            if dra is not None and dra.nvlink_domain_label_key
            else None
        )
        # DRA GPU<->NIC topology co-allocation (opt-in; see KubernetesDraConfig).
        self._dra_rdma_device_class: str | None = (
            str(dra.rdma_device_class)
            if dra is not None and dra.rdma_device_class is not None
            else None
        )
        self._dra_rdma_match_attribute: str = (
            str(dra.rdma_match_attribute)
            if dra is not None and dra.rdma_match_attribute is not None
            else "resource.kubernetes.io/pcieRoot"
        )
        self._tolerations: list[dict[str, Any]] | None = config.tolerations
        self._volumes = list(config.volumes or [])
        # RDMA provisioning (see `rdma` config + k8s.rdma.py). "disable" turns it
        # off; otherwise the provider chain runs in allocate() -- `_rdma_forced` pins
        # one provider when `rdma` names a specific key (else the whole chain is
        # tried). The resolved `_rdma_plan` (net env + per-pod NIC specs + device
        # grants) is filled in by _detect_network_env().
        # RDMA provisioning (see `rdma` config + k8s.rdma.py): "disable" turns it off,
        # else the provider chain runs in allocate(). Extracted into _configure_rdma so
        # the CLI `--kube-rdma` override (apply_kubectl_config) re-derives it identically
        # -- letting a run on a broken/IB-less cluster force `disable` without editing the
        # cluster-agnostic recipe (which should ship `rdma: auto`).
        self._rdma_forced: str | None = None
        self._configure_rdma(str(config.rdma or "auto"))
        # CLI-level kube access (set via apply_kubectl_config from `sflow run`
        # flags); prefixed onto every kubectl call sflow makes for this backend.
        self._kubeconfig: str | None = None
        self._kube_context: str | None = None
        self._kubectl_extra_args: list[str] = []
        # Node include/exclude host lists (from `--include-nodes` / `--exclude-nodes`
        # or the backend config fields), applied as hostname In/NotIn nodeAffinity on
        # the reservation pods so the whole allocation is restricted / steered.
        self._include_nodes: list[str] = normalize_node_list(config.include_nodes)
        self._exclude_nodes: list[str] = normalize_node_list(config.exclude_nodes)
        self._pending_alloc_id: str | None = None
        # k8s_mpi operator support: the mpi.route values of any k8s_mpi operators
        # (set by the app before preflight via note_mpi_operator_routes), and
        # whether the Kubeflow mpi-operator CRD exists (detected at preflight; None
        # until then -> route:auto falls back to the pods route, e.g. in dry-run).
        self._mpi_operator_routes: list[str] = []
        # Detected states of optional cluster capabilities (mpi-operator,
        # ComputeDomain, ...), keyed by ClusterCapability.key. Empty until preflight
        # detection runs -> capability_state() returns UNKNOWN (dry-run -> unavailable).
        self._capability_states: dict[str, CapabilityState] = {}

    def _configure_compute_domain(self, *, channel: Any, create: bool) -> None:
        """Derive the ComputeDomain (Multi-Node NVLink / IMEX) internal state.

        Shared by ``__init__`` (recipe ``compute_domain:`` values) and
        ``apply_kubectl_config`` (the ``--kube-compute-domain-*`` CLI override), so both
        paths classify a channel value identically:

        - ``None`` / empty / ``disable`` (legacy ``off`` / ``false`` / ``no``) -> off
          (no channel claimed). ``disable`` is the YAML-safe turn-off token.
        - ``auto`` -> resolve the sole existing ComputeDomain at preflight/allocate.
        - anything else -> a named channel template, claimed immediately.

        A ComputeDomain is only *created* (and later deleted) by sflow when ``create`` is
        set AND no existing channel is joined -- creating and joining are mutually
        exclusive, and joining an existing channel needs no ``computedomains`` RBAC.
        """
        # CREATE a ComputeDomain CR (admin/privileged, explicit opt-in).
        self._create_compute_domain: bool = bool(create)
        # JOIN an existing ComputeDomain by its channel template name.
        raw_channel = str(channel).strip() if channel is not None else ""
        low_channel = raw_channel.lower()
        if low_channel in ("", "disable", "off", "false", "no"):
            self._use_compute_domain_channel_cfg: str | None = None
        elif low_channel == "auto":
            self._use_compute_domain_channel_cfg = "auto"
        else:
            self._use_compute_domain_channel_cfg = raw_channel
        # Only create (and later delete) a ComputeDomain when asked to AND not joining an
        # existing channel (named or auto). Joining needs no `computedomains` RBAC.
        self._creates_own_compute_domain: bool = (
            self._create_compute_domain
            and self._use_compute_domain_channel_cfg is None
        )
        # A named channel is exposed immediately (no allocate-time creation); "auto" is
        # filled in by ComputeDomain detection (component 4); a created domain is filled
        # in by allocate().
        self._compute_domain_channel = (
            self._use_compute_domain_channel_cfg
            if self._use_compute_domain_channel_cfg not in (None, "auto")
            else None
        )

    def _configure_rdma(self, rdma_mode: str) -> None:
        """Set the RDMA mode / initial plan from a mode string.

        Shared by ``__init__`` (the recipe ``rdma:`` field) and
        ``apply_kubectl_config`` (the ``--kube-rdma`` CLI override) so both derive the
        same state. ``"disable"`` is the explicit kill switch (RDMA off, NCCL onto
        sockets); any other value runs the provider chain in ``allocate()`` and pins one
        provider when it names a specific key. The real plan (net env + per-pod NIC
        specs + device grants) is filled in later by ``_detect_network_env()``.
        """
        rdma_mode = str(rdma_mode or "auto")
        if rdma_mode == "disable":
            self._rdma_mode = "disable"
            self._rdma_forced = None
            # Explicit kill switch: RDMA cleanly off for the whole workflow. The provider
            # chain never runs, but reservation-time detection still probes the routable
            # control NIC and replaces this with RdmaPlan.off(socket_iface=...) so
            # SFLOW_PRIMARY_IFACE is set (cross-node NCCL/gloo ride sockets). Placeholder
            # until then (disable flags only, no iface).
            self._rdma_plan = RdmaPlan.off()
        else:
            self._rdma_mode = "auto"
            self._rdma_forced = rdma_mode if rdma_mode in RDMA_PROVIDER_KEYS else None
            # Neutral placeholder until reservation-time detection fills it in; injects
            # nothing so `auto` never force-downgrades before detection.
            self._rdma_plan = RdmaPlan.disabled()

    def apply_kubectl_config(self, cfg: Any) -> None:
        """Apply CLI-level kube access (``KubectlConfig``) to this backend.

        Sets the global kubectl flags (``--kubeconfig`` / ``--context`` + any
        passthroughs) used by every kubectl call, overrides the namespace when
        ``--kube-namespace`` was given, merges any ``--kube-node-selector`` labels into
        this backend's node_selector (CLI wins on key conflicts), and overrides the
        Multi-Node NVLink (ComputeDomain) settings when ``--kube-compute-domain-channel``
        / ``--kube-compute-domain-create`` were given. The recipe stays cluster-agnostic.
        """
        self._kubeconfig = cfg.kubeconfig or None
        self._kube_context = cfg.context or None
        self._kubectl_extra_args = [str(a) for a in (cfg.extra_args or [])]
        if cfg.namespace:
            self._namespace = str(cfg.namespace)
        cli_node_selector = getattr(cfg, "node_selector", None)
        if cli_node_selector:
            merged = dict(self._node_selector or {})
            merged.update({str(k): str(v) for k, v in cli_node_selector.items()})
            self._node_selector = merged
        # --kube-skip-pvc: debug aid for clusters lacking the recipe's PVCs. Drop every
        # PVC-backed volume (a `volumes:` entry with a `claim`) so pods schedule without
        # editing the recipe volume-by-volume; emptyDir volumes are kept. The PVC data
        # (e.g. a model cache) is then NOT mounted, so workloads that need it will fail --
        # this is for quick scheduling/plumbing checks only.
        if getattr(cfg, "skip_pvc", False):
            pvc_volumes = [v for v in self._volumes if v.claim is not None]
            if pvc_volumes:
                self._volumes = [v for v in self._volumes if v.claim is None]
                _logger.warning(
                    "Kubernetes backend '%s': --kube-skip-pvc dropped %d PVC-backed "
                    "volume(s): %s. Their data will NOT be mounted (debug override).",
                    self.name,
                    len(pvc_volumes),
                    ", ".join(
                        f"{v.name} (claim={v.claim}) at {v.mount_path}"
                        for v in pvc_volumes
                    ),
                )
        # --kube-compute-domain-channel / --kube-compute-domain-create: override the
        # recipe's ComputeDomain (Multi-Node NVLink / IMEX) settings for this run. Only
        # re-derive when at least one was given (tri-state: None => keep the recipe
        # value); fall back to the current classified value for the one not overridden.
        cli_cd_channel = getattr(cfg, "compute_domain_channel", None)
        cli_cd_create = getattr(cfg, "compute_domain_create", None)
        if cli_cd_channel is not None or cli_cd_create is not None:
            self._configure_compute_domain(
                channel=(
                    cli_cd_channel
                    if cli_cd_channel is not None
                    else self._use_compute_domain_channel_cfg
                ),
                create=(
                    cli_cd_create
                    if cli_cd_create is not None
                    else self._create_compute_domain
                ),
            )
        # --kube-rdma: override the recipe's `rdma:` mode for this run without editing the
        # recipe. Intended for a run cluster whose IB/RoCE fabric is down or absent -- pass
        # `--kube-rdma disable` so the shipped recipes can keep `rdma: auto`. None => keep
        # the recipe value.
        cli_rdma = getattr(cfg, "rdma", None)
        if cli_rdma is not None and str(cli_rdma) != self._rdma_mode:
            _logger.warning(
                "Kubernetes backend '%s': --kube-rdma overrides the recipe rdma mode "
                "(%s -> %s) for this run.",
                self.name,
                self._rdma_mode,
                cli_rdma,
            )
            self._configure_rdma(str(cli_rdma))
        # --kube-handoff: override the recipe's `reservation.handoff` (GPU node handoff order)
        # for this run without editing the recipe. Intended for a quota-constrained cluster:
        # pass `--kube-handoff destroy_before_create` to force the quota-safe delete-before-
        # create handoff so shipped recipes can keep `handoff: auto`. Resolved (against a live
        # quota probe for "auto") later in _resolve_handoff_mode(). None => keep the recipe
        # value.
        cli_handoff = getattr(cfg, "handoff", None)
        if cli_handoff is not None:
            allowed_handoff = ("auto", "create_before_destroy", "destroy_before_create")
            if str(cli_handoff) not in allowed_handoff:
                raise ValueError(
                    f"--kube-handoff must be one of {list(allowed_handoff)}, "
                    f"got {cli_handoff!r}"
                )
            if str(cli_handoff) != str(self._reservation_handoff):
                _logger.warning(
                    "Kubernetes backend '%s': --kube-handoff overrides the recipe "
                    "reservation.handoff mode (%s -> %s) for this run.",
                    self.name,
                    self._reservation_handoff,
                    cli_handoff,
                )
                self._reservation_handoff = str(cli_handoff)
        # Generic `--extra-args` are fanned into every backend channel, so they also
        # land here as kubectl global flags. kubectl (unlike salloc/`docker run`)
        # rejects unknown globals, so a Slurm/docker-ism passed generically would
        # break every call. Warn once, pointing at the type-specific flags.
        generic = [str(a) for a in (getattr(cfg, "generic_extra_args", None) or [])]
        if generic:
            _logger.warning(
                "Kubernetes backend '%s': applying generic --extra-args as kubectl "
                "global flags: %s. If these were meant for a Slurm/docker backend, "
                "pass them via --extra-salloc-args / --extra-docker-args instead; as "
                "kubectl global flags they must be valid kubectl options or every "
                "kubectl call will fail.",
                self.name,
                generic,
            )

    def _global_args(self) -> list[str]:
        """kubectl global flags (``--kubeconfig`` / ``--context`` + passthroughs).

        Delegates to :func:`kubectl_global_args` so the CLI (``KubectlConfig``) and
        the backend build the exact same flags from one implementation.
        """
        return kubectl_global_args(
            self._kubeconfig, self._kube_context, self._kubectl_extra_args
        )

    @property
    def kubectl_global_args(self) -> list[str]:
        """Global kubectl flags, injected into the per-task operator wrapper."""
        return self._global_args()

    @property
    def namespace(self) -> str | None:
        """Backend namespace, injected into operators (one namespace per backend)."""
        return self._namespace

    @property
    def host_network(self) -> bool:
        return self._host_network

    @property
    def host_ipc(self) -> bool:
        """Whether task pods share the node IPC namespace + /dev/shm (cross-pod CUDA IPC)."""
        return self._host_ipc

    @property
    def cpu_per_gpu(self) -> int | None:
        """CPU cores requested per GPU on GPU task pods (``None`` => inject none)."""
        return self._cpu_per_gpu

    @property
    def cpu_request(self) -> int | None:
        """CPU cores requested on task pods without GPUs (``None`` => inject none)."""
        return self._cpu_request

    @property
    def collect_max_file_size(self) -> int | str | None:
        """Backend default per-file cap for node-local output collection (or ``None``)."""
        return self._collect_max_file_size

    @property
    def collect_grace_seconds(self) -> int:
        """Grace period (s) for the operator's node-local output collection sidecar."""
        return self._collect_grace_seconds

    @property
    def merge_colocated_gpu_pods(self) -> bool:
        """Whether co-located GPU tasks are merged into one pod (shared NVLink).

        The tri-state (bool | "auto"/"on"/"disable") config is resolved to this bool so
        ``_plan_merge_groups`` (which reads it) stays unchanged.
        """
        return self._merge_colocated_gpu_pods

    @property
    def nvlink_domain(self) -> str:
        """Configured NVLink-domain scope ("auto"/"node"/"rack"/"disable").

        This is the user-facing override; the DETECTED scope (once resolved at
        preflight/allocate) is exposed via ``nvlink_domain_scope`` (component 2).
        """
        return self._nvlink_domain_cfg

    @property
    def nvlink_domain_label_key(self) -> str | None:
        """Node label key identifying the physical NVLink domain (placement only)."""
        return self._nvlink_domain_label_key

    @property
    def nvlink_domain_scope(self) -> str | None:
        """Resolved NVLink-domain scope: ``node`` / ``rack`` / ``disable``, or ``None``.

        An explicit ``nvlink_domain`` override wins; otherwise the value detected
        at preflight/allocate (component 2). ``None`` means auto with detection not
        yet run (e.g. dry-run) -- consumers treat it as "unknown".
        """
        return self._nvlink_domain_override or self._nvlink_domain_scope_detected

    @property
    def node_selector(self) -> dict[str, str] | None:
        return self._node_selector

    @property
    def include_nodes(self) -> list[str]:
        """Node hostnames the allocation is restricted to (hostname In nodeAffinity)."""
        return list(self._include_nodes)

    @property
    def exclude_nodes(self) -> list[str]:
        """Node hostnames all pods are steered away from (hostname NotIn nodeAffinity)."""
        return list(self._exclude_nodes)

    @property
    def network_env(self) -> dict[str, str]:
        """Network env vars (NCCL/gloo device+interface) to inject into task pods.

        Empty unless RDMA was detected (auto), an explicit device list was given,
        or the routable interface was pinned. With ``rdma: disable`` it carries the
        NCCL socket-forcing envs (see :meth:`RdmaPlan.off`) so RDMA is cleanly
        disabled workflow-wide, plus -- once allocation-time detection has probed it
        -- the routable control NIC (``NCCL_SOCKET_IFNAME`` / ``SFLOW_PRIMARY_IFACE``)
        that cross-node NCCL/gloo sockets fall back to when IB is off. UCX device
        selection is intentionally left to the workload/library.
        """
        return dict(self._rdma_plan.net_env)

    @property
    def rdma_enabled(self) -> bool:
        """True when task GPU pods should get scoped RDMA device access."""
        return self._rdma_plan.enabled

    @property
    def rdma_nic_specs(self) -> list[tuple[str, str]]:
        """Per-node RDMA NICs as ``(resource_name, hca_name)``.

        ``resource_name`` is ``""`` for the host-device provider (no extended
        resource; access via the hostPath device mount). The operator requests a
        per-pod slice (sized to the pod's GPU count) and de-dups the non-empty
        resources. NIC *selection* (``NCCL_IB_HCA`` / UCX) is left to the libraries;
        the backend never pins it.
        """
        return list(self._rdma_plan.nic_specs)

    @property
    def rdma_lib_mounts(self) -> list[tuple[str, str]]:
        """gIB hostPath lib mounts ``(host_path, mount_path)`` for multi-node NCCL (GKE).

        Empty unless the active provider needs them (only GKE gIB today). The
        operator mounts these only on multi-node GPU pods.
        """
        return list(self._rdma_plan.lib_mounts)

    @property
    def rdma_nccl_env_script(self) -> str:
        """NCCL tuning script sourced for multi-node pods (GKE gIB), or ""."""
        return self._rdma_plan.nccl_env_script

    @property
    def rdma_host_device_paths(self) -> list[str]:
        """Host device dirs (e.g. ``/dev/infiniband``) to hostPath-mount into RDMA pods.

        Non-empty only for the host-device provider, where verbs access comes from
        the device mount + ``CAP_IPC_LOCK`` rather than an extended resource.
        """
        return list(self._rdma_plan.host_device_paths)

    @property
    def rdma_ipc_lock(self) -> bool:
        """Whether RDMA task pods need ``CAP_IPC_LOCK`` (to pin memory for verbs)."""
        return self._rdma_plan.ipc_lock

    @property
    def rdma_runtime_affinity(self) -> bool:
        """Whether task pods should select their GPU-local RDMA NIC at runtime.

        True for providers where the pod can see every node HCA (host-device,
        shared device plugin): the device plugin/DRA picks the physical GPU, so
        the co-located NIC is only known in-pod. The operator then injects the
        runtime affinity preamble (see ``k8s.rdma_preamble``) instead of a static
        per-pod NIC pin. False for GKE (fixed per-pod NIC subset) and when RDMA is
        off/unavailable.
        """
        return self._rdma_plan.allow_runtime_affinity

    @property
    def scheduling(self) -> str:
        return self._scheduling

    @property
    def gpu_resource_name(self) -> str:
        """Device-plugin extended-resource name for GPU requests (default nvidia.com/gpu)."""
        return self._gpu_resource_name

    @property
    def gpu_device_class(self) -> str:
        return self._gpu_device_class

    @property
    def device_selectors(self) -> list[str] | None:
        return self._device_selectors

    @property
    def dra_rdma_device_class(self) -> str | None:
        """NIC DeviceClass for DRA GPU<->NIC co-allocation, or None (opt-in)."""
        return self._dra_rdma_device_class

    @property
    def dra_rdma_match_attribute(self) -> str:
        """Attribute the co-allocated GPU + NIC must share (default pcieRoot)."""
        return self._dra_rdma_match_attribute

    @property
    def tolerations(self) -> list[dict[str, Any]]:
        """Effective pod tolerations (default: tolerate nvidia.com/gpu)."""
        if self._tolerations is not None:
            return [dict(t) for t in self._tolerations]
        return [dict(DEFAULT_GPU_TOLERATION)]

    @property
    def volumes(self) -> list[dict[str, Any]]:
        """Volume mounts (resolved) injected into task pods by the operator."""
        out: list[dict[str, Any]] = []
        for v in self._volumes:
            entry: dict[str, Any] = {
                "name": str(v.name),
                "mount_path": str(v.mount_path),
                "sub_path": str(v.sub_path) if v.sub_path is not None else None,
                "read_only": v.effective_read_only(),
            }
            if v.empty_dir is not None:
                entry["empty_dir"] = {
                    "medium": str(v.empty_dir.medium or ""),
                    "size_limit": (
                        str(v.empty_dir.size_limit)
                        if v.empty_dir.size_limit is not None
                        else None
                    ),
                }
            else:
                entry["claim"] = str(v.claim)
                entry["ensure_writable"] = bool(v.ensure_writable)
            out.append(entry)
        return out

    @property
    def compute_domain_channel(self) -> str | None:
        """ComputeDomain channel ResourceClaimTemplate name, or None."""
        return self._compute_domain_channel

    @property
    def handoff_destroy_first(self) -> bool:
        """True -> destroy-before-create GPU handoff (quota-safe). Resolved in allocate()."""
        return self._handoff_destroy_first

    def reservation_pod_for_node(self, node_name: str) -> str | None:
        """Return the placeholder pod holding ``node_name`` (create-before-destroy handoff)."""
        return self._node_to_resv_pod.get(node_name)

    def note_mpi_operator_routes(self, routes: list[str]) -> None:
        """Record the ``mpi.route`` of every ``k8s_mpi`` operator in the workflow.

        Called by the app before preflight so the RBAC check requires
        ``mpijobs.kubeflow.org`` verbs only when the operator route may be used,
        and so ``route: auto`` can resolve against the detected CRD.
        """
        self._mpi_operator_routes = [str(r) for r in routes]

    def capability_state(self, cap: ClusterCapability) -> CapabilityState:
        """Detected :class:`CapabilityState` of an optional cluster capability.

        ``UNKNOWN`` until preflight detection has run (e.g. dry-run) -- consumers
        treat that as unavailable for opportunistic paths, but not as a hard
        "absent" signal. Injected into operators via ``apply_backend_context`` so
        e.g. the ``k8s_mpi`` operator resolves ``route: auto`` from
        ``capability_state(MPI_OPERATOR).usable``.
        """
        return self._capability_states.get(cap.key, CapabilityState.UNKNOWN)

    # ------------------------------------------------------------------
    # kubectl helpers
    # ------------------------------------------------------------------

    def _ns_args(self) -> list[str]:
        return ["--namespace", self._namespace] if self._namespace else []

    async def _kubectl(self, args: list[str]) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            *self._global_args(),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (
            proc.returncode,
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
        )

    async def _resolve_handoff_mode(self) -> None:
        """Resolve the GPU node handoff order (create- vs destroy-before-create).

        Explicit config wins; ``auto`` -> destroy-before-create iff a GPU ResourceQuota is
        present in the namespace (where create-before-destroy double-counts the placeholder
        + task GPU requests and the task pod is rejected at admission). Idempotent.
        """
        mode = (self._reservation_handoff or "auto").lower()
        if mode == "destroy_before_create":
            self._handoff_destroy_first = True
            return
        if mode == "create_before_destroy":
            self._handoff_destroy_first = False
            return
        self._handoff_destroy_first = await self._gpu_quota_present()
        if self._handoff_destroy_first:
            _logger.info(
                "Kubernetes backend '%s': GPU ResourceQuota detected in namespace '%s' -> "
                "using destroy-before-create node handoff (quota-safe; brief node-loss gap).",
                self.name,
                self._namespace or "<default>",
            )

    async def _gpu_quota_present(self) -> bool:
        """True if a ResourceQuota in the namespace limits ``requests.nvidia.com/gpu``."""
        if not self._namespace:
            return False
        rc, out, _ = await self._kubectl(
            [
                "get",
                "resourcequota",
                *self._ns_args(),
                "-o",
                "jsonpath={.items[*].status.hard}",
            ]
        )
        return rc == 0 and f"requests.{self._gpu_resource_name}" in out

    async def _apply_manifest(self, manifest: dict[str, Any]) -> None:
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            *self._global_args(),
            "apply",
            "-f",
            "-",
            *self._ns_args(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        data = json.dumps(manifest, separators=(",", ":")).encode()
        _, stderr = await proc.communicate(data)
        if proc.returncode != 0:
            name = manifest.get("metadata", {}).get("name", "?")
            kind = manifest.get("kind", "object")
            raise RuntimeError(
                f"Failed to create reservation {kind} '{name}': "
                f"{stderr.decode(errors='replace').strip()}"
            )

    # ------------------------------------------------------------------
    # In-cluster probe pod (see k8s.probe.K8sExecProbeTransport)
    # ------------------------------------------------------------------

    def _probe_via_pod_enabled(self) -> bool:
        """Whether TCP/HTTP probes route through the in-cluster probe pod.

        On by default; ``SFLOW_K8S_PROBE_VIA_POD=0`` (or false/no/off) disables
        it so probes run directly from the sflow driver host instead.
        """
        val = os.environ.get("SFLOW_K8S_PROBE_VIA_POD")
        if val is None:
            return True
        return val.strip().lower() not in ("0", "false", "no", "off", "")

    def _kickoff_probe_pod(self) -> None:
        """Start background creation of the probe pod on first network-probe use.

        Lazy on purpose: a workflow with no TCP/HTTP probes (e.g. log_watch-only
        or probe-less batch jobs) never creates the pod. Idempotent -- the pod is
        created at most once per allocation. Once created it lives until
        ``release()`` (like the task pods it probes), which is the window during
        which any of the DAG's network probes may run; an individual readiness
        probe still "ends" as usual once it triggers (it just stops running curl).
        """
        if not self._probe_via_pod_enabled():
            return
        if self._probe_pod_name is not None:
            return
        if self._probe_pod_task is not None and not self._probe_pod_task.done():
            return
        self._probe_pod_task = asyncio.ensure_future(self._create_probe_pod())

    async def _create_probe_pod(self) -> None:
        """Apply the probe pod for this allocation (best-effort; never fatal)."""
        alloc_id = (
            getattr(self.allocation, "allocation_id", None) or self._pending_alloc_id
        )
        if not alloc_id:
            return
        pod_name = f"sflow-probe-{sanitize_name(self.name)[:12]}-{alloc_id}"
        try:
            await self._apply_manifest(
                render_probe_pod_manifest(
                    pod_name=pod_name,
                    allocation_id=alloc_id,
                    image=self._probe_pod_image,
                    namespace=self._namespace,
                    image_pull_policy=self._image_pull_policy,
                    node_selector=self._node_selector,
                    tolerations=self._effective_tolerations(),
                )
            )
        except Exception as e:
            _logger.warning(
                "Kubernetes backend '%s': could not create probe pod (%s); "
                "TCP/HTTP probes will keep retrying until it exists.",
                self.name,
                e,
            )
            return
        self._probe_pod_name = pod_name
        _logger.info(
            "Kubernetes backend '%s': created in-cluster probe pod '%s' for "
            "TCP/HTTP probes.",
            self.name,
            pod_name,
        )

    async def _await_probe_pod_task(self) -> None:
        """Let any in-flight probe-pod creation finish before release() reaps it.

        Guarantees create-then-delete ordering: if a probe pod is still being
        applied when the allocation is released, wait for the apply to land so the
        label-based delete in ``release()`` reliably removes it (no orphan).
        """
        task = self._probe_pod_task
        self._probe_pod_task = None
        if task is not None and not task.done():
            with suppress(Exception):
                await task

    async def _exec_in_probe_pod(
        self, argv: list[str], stdin: bytes | None = None
    ) -> tuple[int, str, str]:
        """Run ``argv`` inside the probe pod via ``kubectl exec`` (stdin optional).

        Creates the probe pod on first use; until it is up, returns a non-zero
        result so the probe simply retries (readiness stays not-ready and a
        failure probe stays not-failed rather than spuriously firing).
        """
        pod = self._probe_pod_name
        if not pod:
            self._kickoff_probe_pod()
            return (1, "", "probe pod not ready")
        exec_args = ["exec"]
        if stdin is not None:
            exec_args.append("-i")
        exec_args += [pod, *self._ns_args(), "--", *argv]
        if stdin is None:
            return await self._kubectl(exec_args)
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            *self._global_args(),
            *exec_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(stdin)
        return (
            proc.returncode,
            out.decode(errors="replace").strip(),
            err.decode(errors="replace").strip(),
        )

    def probe_transport(self) -> "K8sExecProbeTransport | None":
        """In-cluster probe transport, or None to probe from the driver host.

        The transport exists whenever in-cluster probing is enabled; the probe
        pod itself is created lazily on the first TCP/HTTP check (see
        ``_exec_in_probe_pod``), so probe-less / log_watch-only workflows never
        create one.
        """
        if not self._probe_via_pod_enabled():
            return None
        return K8sExecProbeTransport(exec_fn=self._exec_in_probe_pod)

    def _reservation_retries(self) -> int:
        """Poll attempts for reservation readiness, derived from the timeout."""
        # Guard against a zero poll interval (e.g. tests set it to 0 to skip the
        # sleep): fall back to 1s so the retry count stays well-defined.
        interval = _RESERVE_POLL_INTERVAL or 1
        return max(1, int(self._reservation_timeout) // interval)

    def _effective_tolerations(self) -> list[dict[str, Any]]:
        return self.tolerations

    async def _pod_wait_reason(self, pod_name: str) -> str:
        """Best-effort short reason a pod is not yet Running (or ""):

        prefers the container waiting reason+message, falling back to the
        ``PodScheduled`` condition message (where ``FailedScheduling`` lives).
        """
        rc, out, _ = await self._kubectl(
            [
                "get",
                "pod",
                pod_name,
                *self._ns_args(),
                "-o",
                "jsonpath={.status.containerStatuses[*].state.waiting.reason} "
                "{.status.containerStatuses[*].state.waiting.message}",
            ]
        )
        reason = out.strip() if rc == 0 else ""
        if reason:
            return reason
        rc, out, _ = await self._kubectl(
            [
                "get",
                "pod",
                pod_name,
                *self._ns_args(),
                "-o",
                'jsonpath={.status.conditions[?(@.type=="PodScheduled")].message}',
            ]
        )
        return out.strip() if rc == 0 else ""

    async def _pod_diagnostics(self, pod_name: str) -> str:
        """Best-effort recent events for ``pod_name`` (or "") to append to errors."""
        rc, out, _ = await self._kubectl(
            [
                "get",
                "events",
                *self._ns_args(),
                "--field-selector",
                f"involvedObject.name={pod_name}",
                "--sort-by=.lastTimestamp",
            ]
        )
        if rc != 0 or not out:
            return ""
        recent = "\n    ".join(out.splitlines()[-10:])
        return f"\n  Recent events:\n    {recent}"

    async def _wait_for_pod_scheduled(self, pod_name: str) -> tuple[str, str]:
        """Block until ``pod_name`` is scheduled, returning (node_name, host_ip).

        ``spec.nodeName`` appears once the scheduler binds the placeholder pod;
        ``status.hostIP`` follows shortly after. While unscheduled, emit a log
        line on every phase/reason change plus a periodic heartbeat, so a stuck
        reservation is never silent.
        """
        ns_args = self._ns_args()
        start = time.monotonic()
        last_phase: str | None = None
        last_heartbeat = start
        try:
            heartbeat = int(
                os.environ.get("SFLOW_K8S_WAIT_HEARTBEAT_SECS")
                or _RESERVE_HEARTBEAT_INTERVAL
            )
        except ValueError:
            heartbeat = _RESERVE_HEARTBEAT_INTERVAL
        for _ in range(self._reservation_retries()):
            rc, out, _ = await self._kubectl(
                [
                    "get",
                    "pod",
                    pod_name,
                    *ns_args,
                    "-o",
                    "jsonpath={.spec.nodeName},{.status.hostIP},{.status.phase}",
                ]
            )
            node_name = host_ip = phase = ""
            if rc == 0 and out and out.count(",") >= 2:
                node_name, host_ip, phase = out.split(",", 2)

            now = time.monotonic()
            if phase and (
                phase != last_phase
                or (now - last_heartbeat) >= heartbeat
            ):
                reason = await self._pod_wait_reason(pod_name)
                _logger.info(
                    "Kubernetes backend '%s' waiting for reservation pod '%s': "
                    "phase=%s%s (%ds elapsed)",
                    self.name,
                    pod_name,
                    phase,
                    f" reason={reason}" if reason else "",
                    int(now - start),
                )
                last_phase = phase
                last_heartbeat = now

            if phase in ("Failed", "Unknown"):
                raise RuntimeError(
                    f"Reservation pod '{pod_name}' entered terminal phase '{phase}' "
                    "before being scheduled. Check node resources, GPU "
                    "availability, and node_selector."
                    + await self._pod_diagnostics(pod_name)
                )
            if node_name:
                return node_name, host_ip
            await asyncio.sleep(_RESERVE_POLL_INTERVAL)
        raise RuntimeError(
            f"Reservation pod '{pod_name}' was not scheduled onto a node within "
            f"{self._reservation_timeout}s. Check cluster capacity, GPU "
            "availability, and node_selector settings."
            + await self._pod_diagnostics(pod_name)
        )

    async def _node_internal_ip(self, node_name: str, *, fallback: str = "") -> str:
        """Resolve a node's InternalIP via kubectl, falling back to ``fallback``."""
        rc, out, _ = await self._kubectl(
            [
                "get",
                "node",
                node_name,
                "-o",
                'jsonpath={.status.addresses[?(@.type=="InternalIP")].address}',
            ]
        )
        if rc == 0 and out:
            return out.split()[0]
        return fallback

    async def _node_label(
        self, node_name: str, label_key: str, *, fallback: str = ""
    ) -> str:
        """Resolve a node's label value via kubectl, falling back to ``fallback``.

        Mirrors ``_node_internal_ip``; dots in ``label_key`` are escaped for the
        jsonpath selector (e.g. ``nvidia.com/gpu.clique``).
        """
        escaped = label_key.replace(".", r"\.")
        rc, out, _err = await self._kubectl(
            [
                "get",
                "node",
                node_name,
                "-o",
                f"jsonpath={{.metadata.labels.{escaped}}}",
            ]
        )
        if rc == 0 and out:
            return out.strip()
        return fallback

    async def _validate_nvlink_domain_placement(
        self, node_names: Sequence[str]
    ) -> None:
        """Post-schedule check that reserved nodes share ONE NVLink domain.

        Warn-only (no hard fail): reads each node's ``nvlink_domain_label_key``
        label and warns if reserved nodes straddle multiple domain values or lack
        the label -- cross-node NVLink KV won't work in that case. No-op unless a
        label key is configured, scope is ``rack``, and there is more than one node.
        """
        key = self._nvlink_domain_label_key
        if not key or self.nvlink_domain_scope != "rack" or len(node_names) < 2:
            return
        labels = await asyncio.gather(
            *[self._node_label(n, key) for n in node_names]
        )
        missing = [n for n, lbl in zip(node_names, labels, strict=True) if not lbl]
        if missing:
            _logger.warning(
                "Kubernetes backend '%s': reserved node(s) %s lack the NVLink-domain "
                "label '%s'; cross-node NVLink may not work. Verify the label key or "
                "set nvlink_domain: node.",
                self.name,
                ", ".join(missing),
                key,
            )
            return
        values = {lbl for lbl in labels if lbl}
        if len(values) > 1:
            pairs = ", ".join(
                f"{n}={lbl}"
                for n, lbl in zip(node_names, labels, strict=True)
            )
            _logger.warning(
                "Kubernetes backend '%s': reserved nodes span multiple NVLink "
                "domains ('%s' -> %s); cross-node NVLink KV transfer will not work. "
                "Ensure the cluster can place all reserved nodes in one domain, or "
                "co-locate prefill+decode per node.",
                self.name,
                key,
                pairs,
            )

    async def _delete_by_alloc_label(self, kind: str, alloc_id: str) -> None:
        await self._kubectl(
            [
                "delete",
                kind,
                "-l",
                f"{SFLOW_ALLOC_LABEL}={alloc_id}",
                *self._ns_args(),
                "--ignore-not-found",
            ]
        )

    def _delete_by_alloc_label_sync(self, kind: str, alloc_id: str) -> None:
        subprocess.run(
            [
                "kubectl",
                *self._global_args(),
                "delete",
                kind,
                "-l",
                f"{SFLOW_ALLOC_LABEL}={alloc_id}",
                *self._ns_args(),
                "--ignore-not-found",
            ],
            capture_output=True,
        )

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def _kubectl_sync(
        self, args: list[str], *, timeout: str = "10s"
    ) -> tuple[int, str, str]:
        """Run a kubectl command synchronously (preflight), returning (rc, out, err)."""
        argv = [
            "kubectl",
            *self._global_args(),
            *args,
            f"--request-timeout={timeout}",
        ]
        try:
            result = subprocess.run(argv, capture_output=True, text=True)
        except Exception as e:  # kubectl missing / spawn failure
            return 1, "", str(e)
        return (
            result.returncode,
            (result.stdout or "").strip(),
            (result.stderr or "").strip(),
        )

    def _access_context(self) -> str:
        """Human-readable kube access descriptor for error messages."""
        return (
            f" (context={self._kube_context or 'current-context'}, "
            f"kubeconfig={self._kubeconfig or '$KUBECONFIG/~/.kube/config'}"
            + (f", namespace={self._namespace}" if self._namespace else "")
            + ")"
        )

    def _api_resource_served(self, crd_name: str) -> bool:
        """Whether ``crd_name`` (e.g. ``mpijobs.kubeflow.org``) is served by the API.

        The least-privilege way to detect cluster infra installed as a CRD /
        aggregated API (mpi-operator, ComputeDomain/IMEX, ...): it uses API
        discovery (``kubectl api-resources``), whose endpoints are open to any
        authenticated user, so it needs NO cluster-scoped ``get
        customresourcedefinitions`` RBAC (which namespaced users often lack -- a
        denied CRD get would otherwise make sflow assume the infra is absent and
        silently disable it). Discovery is version-agnostic (any ``<group>/vN``
        counts) and namespace-agnostic (API availability is cluster-wide; only the
        resource's *instances* are namespaced).

        Order: discovery first; a direct ``get crd`` fallback ONLY when discovery
        itself errors (e.g. a broken *unrelated* aggregated APIService fails the
        whole discovery walk) and did not surface the resource -- that fallback
        needs CRD RBAC and may be denied.

        Note: this answers "is the API served" (type installed). Detecting what a
        node advertises (RDMA NICs, GPU product) or what CR *instances* exist
        (ComputeDomain channels) is a data read of node/CR contents -- not
        answerable by discovery, so those keep their targeted ``get`` reads.
        """
        resource, _, group = crd_name.partition(".")
        rc, out, _err = self._kubectl_sync(
            ["api-resources", f"--api-group={group}", "-o", "name"], timeout="10s"
        )
        served = {ln.strip() for ln in out.splitlines() if ln.strip()}
        # Trust a positive hit even on a non-zero rc: kubectl still prints the
        # resources it COULD discover when an unrelated group's discovery fails.
        if any(n == crd_name or n.startswith(f"{resource}.") for n in served):
            return True
        if rc == 0:
            return False  # clean discovery, resource absent -> definitively not served
        # Discovery itself failed -> fall back to a direct CRD get (needs CRD RBAC).
        rc, _out, _err = self._kubectl_sync(
            ["get", "crd", crd_name, "-o", "name"], timeout="5s"
        )
        return rc == 0

    def _can_i(self, verb: str, resource: str, *, namespaced: bool) -> bool:
        """``kubectl auth can-i <verb> <resource> [-n ns]`` -> True iff ``yes``.

        Low privilege (``auth can-i`` answers for any authed API). Used to gate
        ``route: auto`` on whether the current creds can actually USE an optional
        API, so "installed but not permitted" degrades gracefully. A non-``yes``
        answer (incl. an unreachable API) is treated as ``no``; API reachability is
        validated separately in ``preflight_validate``.
        """
        args = ["auth", "can-i", verb, resource]
        if namespaced and self._namespace:
            args += ["-n", self._namespace]
        _rc, out, _err = self._kubectl_sync(args)
        return bool(out) and out.splitlines()[0].strip() == "yes"

    def detect_capability(
        self, cap: ClusterCapability, *, check_usable: bool
    ) -> CapabilityState:
        """Detect + cache the :class:`CapabilityState` of a cluster capability.

        Wires the backend's low-privilege probes -- ``_api_resource_served`` (API
        discovery; no ``get customresourcedefinitions`` RBAC) and ``_can_i``
        (``auth can-i``) -- into the pure ``detect_capability_state``. ``check_usable``
        adds the (per-verb) RBAC probe so callers that must *drive* the resource
        (e.g. ``route: auto``) distinguish INSTALLED-but-not-permitted from USABLE;
        presence-only callers pass ``False`` and skip the extra round-trips.
        """
        state = detect_capability_state(
            cap,
            is_served=self._api_resource_served,
            can_i=self._can_i,
            check_usable=check_usable,
        )
        self._capability_states[cap.key] = state
        _logger.debug(
            "Kubernetes backend '%s': capability '%s' (%s) -> %s.",
            self.name,
            cap.key,
            cap.api_resource,
            state.value,
        )
        return state

    def _detect_mpi_operator_crd(self) -> None:
        """Best-effort (preflight): detect the mpi-operator capability.

        ``route: auto`` must key off USABILITY (installed AND the creds can drive
        MPIJobs in this namespace), not mere presence -- a namespaced user can see
        the cluster-wide CRD yet lack the RBAC, so probing usability lets auto fall
        back to pods instead of hard-failing the RBAC gate. Only probe usability for
        ``route: auto`` (``operator`` hard-requires the RBAC in preflight; ``pods``
        never detects).
        """
        self.detect_capability(
            MPI_OPERATOR,
            check_usable=any(r == "auto" for r in self._mpi_operator_routes),
        )

    def _wants_mpi_operator_rbac(self) -> bool:
        """Whether preflight must HARD-require MPIJob RBAC.

        Only ``route: operator`` (an explicit choice) does -> a clear RBAC error if
        the verbs are denied. ``route: auto`` never hard-requires them: missing RBAC
        means "installed but not usable", and the operator falls back to the pods
        route (see the usability probe in ``_detect_mpi_operator_crd``), so adding
        them to the required set would turn a graceful fallback into a hard failure.
        """
        return any(r == "operator" for r in self._mpi_operator_routes)

    def _required_permissions(self) -> list[tuple[str, str, bool]]:
        """(verb, resource, namespaced) tuples for the kubectl ops sflow performs."""
        perms: list[tuple[str, str, bool]] = [
            ("create", "pods", True),
            ("delete", "pods", True),
            ("get", "pods", True),
            ("get", "pods/log", True),
            ("create", "configmaps", True),
            ("delete", "configmaps", True),
            ("create", "secrets", True),
            ("delete", "secrets", True),
            ("get", "nodes", False),
        ]
        if self._scheduling == "dra":
            perms += [
                ("create", "resourceclaimtemplates.resource.k8s.io", True),
                ("delete", "resourceclaimtemplates.resource.k8s.io", True),
                ("get", "deviceclasses.resource.k8s.io", False),
            ]
        # ComputeDomain (Multi-Node NVLink / IMEX) is independent of GPU scheduling:
        # it can be layered on `device_plugin` GPUs (e.g. clusters that run the NVIDIA
        # DRA driver in ComputeDomain-only mode). Only require create/delete RBAC when
        # sflow actually creates one -- reusing an existing channel needs no CD perms.
        if self._creates_own_compute_domain:
            perms += [
                ("create", "computedomains.resource.nvidia.com", True),
                ("delete", "computedomains.resource.nvidia.com", True),
            ]
        # k8s_mpi operator route: sflow applies + watches + deletes an MPIJob CR and
        # reads the mpi-operator's launcher pod. Gated so recipes that only use the
        # pods route (or no MPI at all) don't require the Kubeflow CRD's RBAC.
        if self._wants_mpi_operator_rbac():
            perms += [
                (v, MPI_OPERATOR.api_resource, MPI_OPERATOR.namespaced)
                for v in MPI_OPERATOR.use_verbs
            ]
        return perms

    def preflight_validate(self) -> None:
        if shutil.which("kubectl") is None:
            raise ValueError(
                f"Pre-flight validation failed for Kubernetes backend '{self.name}'. "
                "Missing required command: kubectl. Ensure kubectl is installed and available on PATH."
            )
        if self._nodes > 1 and not self._host_network:
            _logger.warning(
                "Kubernetes backend '%s' has %d nodes but 'host_network' is False. "
                "Tasks that reference ${{ backends.%s.nodes[i].ip_address }} will "
                "receive node IPs, but workload pods will listen on pod IPs "
                "(CNI-assigned). Set 'host_network: true' so pod IPs match node IPs.",
                self.name,
                self._nodes,
                self.name,
            )
        self._preflight_check_connectivity()
        # Warn (never fail) if a node's kubelet predates the `kubectl logs -f`
        # rotation-follow fix (< v1.29): <task>.log can go stale and a log_watch
        # probe could miss its marker. Runs after connectivity (needs cluster access).
        self._preflight_check_kubelet_version()
        if self._scheduling == "dra":
            self._preflight_check_dra()
        # Derive/validate gpus_per_node against candidate GPU nodes' real capacity.
        # Runs BEFORE planning + allocation, so a derived count feeds the planner and
        # an over-capacity value is caught here (clear message) instead of later as an
        # unschedulable placeholder stuck Pending. device_plugin only.
        self._preflight_validate_gpus_per_node()
        # Resolve the NVLink-domain scope now (best-effort) so downstream planning
        # (interconnect selection, placement, hints) has it. Warn-only.
        self._detect_nvlink_scope()
        # Resolve `compute_domain.channel: auto` to the sole existing domain
        # (best-effort; zero/many -> hint). No-op unless configured `auto`.
        self._resolve_use_compute_domain_channel()
        # Hard-fail a NAMED compute_domain.channel when the cluster has no IMEX
        # ComputeDomain support at all, so the user unsets it (real run only).
        self._preflight_check_compute_domain()

    @staticmethod
    def _preflight_skipped() -> bool:
        """True when SFLOW_SKIP_K8S_PREFLIGHT bypasses cluster-touching preflight."""
        skip = os.environ.get("SFLOW_SKIP_K8S_PREFLIGHT", "")
        return bool(skip and skip.lower() not in ("0", "false", "no"))

    def _detect_node_gpu_capacity(self) -> int | None:
        """Best-effort: GPUs advertised by candidate nodes' ``status.allocatable``.

        Queries the nodes (filtered by ``node_selector`` when set) for the
        ``gpu_resource_name`` extended resource and returns the max advertised count
        (a representative full-node capacity), or None on any failure / no GPU node.
        """
        jsonpath_key = self._gpu_resource_name.replace(".", r"\.")
        args = [
            "get",
            "nodes",
            "-o",
            "jsonpath={.items[*].status.allocatable." + jsonpath_key + "}",
        ]
        if self._node_selector:
            args += [
                "-l",
                ",".join(f"{k}={v}" for k, v in self._node_selector.items()),
            ]
        rc, out, _err = self._kubectl_sync(args, timeout="10s")
        if rc != 0 or not out:
            return None
        counts: list[int] = []
        for tok in out.split():
            try:
                counts.append(int(tok))
            except (TypeError, ValueError):
                continue
        counts = [c for c in counts if c > 0]
        return max(counts) if counts else None

    def _preflight_validate_gpus_per_node(self) -> None:
        """Derive/validate ``gpus_per_node`` against candidate nodes' real GPU capacity.

        Runs at preflight (before planning + allocation), device_plugin only:

        * when ``gpus_per_node`` is unset -> adopt the detected count, so the planner
          (which reads it via ``placeholder_allocation``) and the reservation use the
          node's real capacity instead of leaving it unknown; and
        * when it is set ABOVE the largest candidate node's capacity -> warn: the GPU
          placeholder/task pods would be unschedulable (stuck ``Pending``).

        A configured value BELOW capacity is a legitimate partial-node reservation (and
        ``0`` is intentional CPU-only), so those are left alone. Best-effort and
        warn-only: any lookup failure leaves the configured value untouched. Skipped
        under ``dra`` (GPUs are not an extended resource there; the DeviceClass governs
        the count).
        """
        if self._scheduling != "device_plugin":
            return
        # No cluster access requested -- connectivity preflight already warned.
        if self._preflight_skipped():
            return
        # A configured 0 is intentional CPU-only -- nothing to derive or validate.
        if self._gpu_per_node == 0:
            return
        detected = self._detect_node_gpu_capacity()
        if detected is None:
            return
        if self._gpu_per_node is None:
            _logger.info(
                "Kubernetes backend '%s': detected %d GPU(s)/node from candidate "
                "node allocatable '%s'; using it as gpus_per_node.",
                self.name,
                detected,
                self._gpu_resource_name,
            )
            self._gpu_per_node = detected
        elif self._gpu_per_node > detected:
            _logger.warning(
                "Kubernetes backend '%s': configured gpus_per_node=%d exceeds the "
                "largest candidate node's capacity (%d %s). GPU pods requesting that "
                "many will stay Pending (unschedulable) -- lower gpus_per_node or "
                "target larger nodes.",
                self.name,
                self._gpu_per_node,
                detected,
                self._gpu_resource_name,
            )

    def _preflight_check_connectivity(self) -> None:
        """Fail fast if the kube access can't perform the operations sflow needs.

        Runs only on real ``sflow run`` (preflight is skipped in --dry-run). Uses
        non-mutating calls: ``kubectl get namespace`` (reachability + auth +
        namespace existence) and ``kubectl auth can-i`` for each required verb/
        resource (RBAC). Hard-fails with an actionable message; set
        ``SFLOW_SKIP_K8S_PREFLIGHT=1`` to bypass.
        """
        if self._preflight_skipped():
            _logger.warning(
                "Kubernetes backend '%s': skipping connectivity/RBAC preflight "
                "(SFLOW_SKIP_K8S_PREFLIGHT set).",
                self.name,
            )
            return

        ctx = self._access_context()
        prefix = f"Pre-flight validation failed for Kubernetes backend '{self.name}'."

        # Reachability + auth + namespace existence (an authenticated GET).
        if self._namespace:
            rc, _out, err = self._kubectl_sync(
                ["get", "namespace", self._namespace, "-o", "name"]
            )
            if rc != 0:
                low = err.lower()
                if "not found" in low or "notfound" in low:
                    raise ValueError(
                        f"{prefix} Namespace '{self._namespace}' was not found{ctx}. "
                        "Create it, or select another with --kube-namespace."
                    )
                raise ValueError(
                    f"{prefix} Cannot reach or authenticate to the Kubernetes "
                    f"cluster{ctx}: {err or 'kubectl could not contact the API server'}. "
                    "Check --kubeconfig / --kube-context."
                )

        # Detect the Kubeflow MPI Operator CRD so `k8s_mpi` route:auto can resolve,
        # and the RBAC check below only requires mpijobs verbs when the operator
        # route may actually be used.
        if any(r in ("auto", "operator") for r in self._mpi_operator_routes):
            self._detect_mpi_operator_crd()

        # RBAC: verify each operation sflow performs (also re-checks reachability/
        # auth, since `kubectl auth can-i` only answers yes/no for a live, authed API).
        denied: list[str] = []
        for verb, resource, namespaced in self._required_permissions():
            args = ["auth", "can-i", verb, resource]
            if namespaced and self._namespace:
                args += ["-n", self._namespace]
            rc, out, err = self._kubectl_sync(args)
            answer = out.splitlines()[0].strip() if out else ""
            if answer == "yes":
                continue
            if answer == "no":
                scope = (
                    f" in namespace '{self._namespace}'"
                    if namespaced and self._namespace
                    else " (cluster-scoped)"
                )
                denied.append(f"{verb} {resource}{scope}")
                continue
            # Neither yes nor no -> the API is unreachable or auth failed.
            raise ValueError(
                f"{prefix} Cannot reach or authenticate to the Kubernetes "
                f"cluster{ctx}: {err or 'kubectl auth can-i did not return a result'}. "
                "Check --kubeconfig / --kube-context."
            )

        if denied:
            raise ValueError(
                f"{prefix} The current credentials lack Kubernetes permissions "
                f"required to run this workflow{ctx}: "
                + "; ".join(denied)
                + ". Grant these (RBAC), pick a different --kube-context, or set "
                "SFLOW_SKIP_K8S_PREFLIGHT=1 to bypass this check."
            )

    def _detect_kubelet_versions(self) -> list[str] | None:
        """Kubelet versions of candidate nodes (filtered by ``node_selector``).

        Uses the kubelet version (the component that serves ``kubectl logs`` and
        carries the follow-stall bug), not just the API-server version. Falls back to
        the API-server ``gitVersion`` only when nodes can't be listed (e.g. namespaced
        RBAC) -- an approximation, since a node's kubelet may lag the API server.
        Returns None on any failure (best-effort).
        """
        args = [
            "get",
            "nodes",
            "-o",
            "jsonpath={.items[*].status.nodeInfo.kubeletVersion}",
        ]
        if self._node_selector:
            args += [
                "-l",
                ",".join(f"{k}={v}" for k, v in self._node_selector.items()),
            ]
        rc, out, _err = self._kubectl_sync(args, timeout="10s")
        if rc == 0 and out.split():
            return out.split()
        # Fallback: API server version (can't list nodes -> approximate).
        rc, out, _err = self._kubectl_sync(
            ["version", "-o", "jsonpath={.serverVersion.gitVersion}"], timeout="10s"
        )
        return [out.strip()] if rc == 0 and out.strip() else None

    def _preflight_check_kubelet_version(self) -> None:
        """Warn (never fail) if any node's kubelet predates the ``kubectl logs -f``
        rotation-follow fix (< v1.29).

        The K8s ``log_watch`` probe reads ``<task>.log``, written by the offloaded
        ``kubectl logs -f``. On kubelet < v1.29 that follow can silently stop after a
        container log rotation (kubernetes/kubernetes#115701, fixed by #115702 in
        v1.29), so a chatty task's ``<task>.log`` may go stale and a readiness/failure
        ``log_watch`` marker be missed until the terminal ``finalize_*`` re-fetch.
        Best-effort + warn-only; skipped with ``SFLOW_SKIP_K8S_PREFLIGHT``.
        """
        if self._preflight_skipped():
            return
        versions = self._detect_kubelet_versions()
        if not versions:
            return
        old = sorted(
            {
                v
                for v in versions
                if (parsed := _parse_k8s_minor(v)) is not None
                and parsed < _KUBELET_LOGS_FOLLOW_FIXED_MINOR
            }
        )
        if not old:
            return
        _logger.warning(
            "Kubernetes backend '%s': %d node(s) run kubelet older than v1.29 (%s). "
            "`kubectl logs -f` can silently stall after a container log rotation on "
            "these (kubernetes/kubernetes#115701, fixed in v1.29), so a chatty task's "
            "<task>.log may fall behind and a `log_watch` readiness/failure probe "
            "could miss its marker until the task ends. Upgrade the node kubelet to "
            ">= 1.29, use tcp_port/http readiness for high-volume tasks on these "
            "nodes, or set SFLOW_SKIP_K8S_PREFLIGHT=1 to silence this check.",
            self.name,
            len(old),
            ", ".join(old),
        )

    def _preflight_check_dra(self) -> None:
        """Best-effort: warn (never fail) if the DRA API version or DeviceClass is
        not served/visible.

        sflow renders the GA DRA API (``RESOURCE_API_VERSION`` = ``resource.k8s.io/v1``,
        Kubernetes 1.34+). A cluster serving only ``v1beta1``/``v1alpha3`` would reject
        the ResourceClaimTemplates, so -- unlike the MPI-operator/ComputeDomain
        capabilities -- verify the exact served version here before ``scheduling: dra``
        is used.
        """
        if self._preflight_skipped():
            return
        rc_api, out_api, _ = self._kubectl_sync(["api-versions"], timeout="10s")
        if rc_api == 0 and RESOURCE_API_VERSION not in out_api.split():
            served = [
                line
                for line in out_api.split()
                if line.startswith("resource.k8s.io/")
            ]
            _logger.warning(
                "Kubernetes backend '%s': the DRA API '%s' is not served by this "
                "cluster%s (scheduling: dra). sflow renders that apiVersion, so "
                "ResourceClaimTemplates would be rejected. Upgrade to Kubernetes 1.34+ "
                "or use 'scheduling: device_plugin'.",
                self.name,
                RESOURCE_API_VERSION,
                f" (found {', '.join(served)})" if served else "",
            )
        rc, _out, _err = self._kubectl_sync(
            ["get", "deviceclass", self._gpu_device_class, "-o", "name"], timeout="5s"
        )
        if rc != 0:
            _logger.warning(
                "Kubernetes backend '%s': could not verify DRA DeviceClass '%s' "
                "(scheduling: dra). Ensure nvidia-dra-driver-gpu is installed and "
                "the cluster is on Kubernetes 1.34+ (resource.k8s.io/v1), or use "
                "'scheduling: device_plugin'.",
                self.name,
                self._gpu_device_class,
            )

    # ------------------------------------------------------------------
    # NVLink-domain scope detection (component 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _nvlink_scope_from(*, product: str, compute_domain_crd: bool) -> str:
        """Classify the NVLink-domain scope from a GPU product + CRD presence.

        * RACK-class board (GB200/GB300) + ComputeDomain CRD -> ``rack`` (MNNVL).
        * RACK-class board WITHOUT the CRD -> ``node`` (only intra-node NVLink; the
          cross-node IMEX fabric is unavailable).
        * NODE-class NVSwitch/NVLink board (B200/H100/...) -> ``node``.
        * Otherwise -> ``off`` (no NVLink assumption).
        """
        p = (product or "").upper()
        is_rack = any(tok in p for tok in _RACK_SCOPE_PRODUCTS)
        is_node = any(tok in p for tok in _NODE_SCOPE_PRODUCTS)
        if is_rack:
            return "rack" if compute_domain_crd else "node"
        if is_node:
            return "node"
        return "disable"

    def _compute_domain_crd_present(self) -> bool:
        """Best-effort: whether the ComputeDomain CRD is installed (IMEX driver).

        NVLink-scope detection only needs PRESENCE, so this probes the capability
        with ``check_usable=False`` (discovery-first -> no cluster-scoped ``get
        customresourcedefinitions`` RBAC; the create/delete RBAC for ComputeDomains
        sflow makes is gated separately in ``_required_permissions``).
        """
        return self.detect_capability(COMPUTE_DOMAIN, check_usable=False).installed

    def _detect_gpu_product(self) -> str:
        """Best-effort: the configured GPU-product label of a (selected) node.

        Reads ``self._gpu_product_label_key`` (default ``nvidia.com/gpu.product``).
        """
        # Escape dots in the label key so jsonpath selects the label with that exact
        # name rather than descending into nested fields.
        jsonpath_key = self._gpu_product_label_key.replace(".", r"\.")
        args = [
            "get",
            "nodes",
            "-o",
            "jsonpath={.items[*].metadata.labels." + jsonpath_key + "}",
        ]
        if self._node_selector:
            args += [
                "-l",
                ",".join(f"{k}={v}" for k, v in self._node_selector.items()),
            ]
        rc, out, _err = self._kubectl_sync(args, timeout="5s")
        if rc == 0 and out:
            # jsonpath joins matches with spaces; the products repeat per node, so
            # take the first non-empty token.
            for tok in out.split():
                if tok:
                    return tok
        return ""

    def _detect_nvlink_scope(self) -> str:
        """Resolve the NVLink-domain scope (node|rack|disable). Best-effort, warn-only.

        Returns the explicit ``nvlink_domain`` override when set; otherwise detects
        from the GPU product label + ComputeDomain CRD presence, caches the result,
        and logs it. Any detection failure degrades to ``disable`` (never raises).
        """
        if self._nvlink_domain_override is not None:
            return self._nvlink_domain_override
        if self._nvlink_domain_scope_detected is not None:
            return self._nvlink_domain_scope_detected
        try:
            product = self._detect_gpu_product()
            crd = self._compute_domain_crd_present()
            scope = self._nvlink_scope_from(product=product, compute_domain_crd=crd)
        except Exception as e:  # detection is best-effort; never break allocation
            _logger.info(
                "Kubernetes backend '%s': NVLink-domain scope detection failed "
                "(%s); assuming 'disable'.",
                self.name,
                e,
            )
            self._nvlink_domain_scope_detected = "disable"
            return "disable"
        self._nvlink_domain_scope_detected = scope
        _logger.info(
            "Kubernetes backend '%s': detected NVLink-domain scope '%s' "
            "(GPU product '%s', ComputeDomain CRD %s).",
            self.name,
            scope,
            product or "(unknown)",
            "present" if crd else "absent",
        )
        return scope

    # ------------------------------------------------------------------
    # ComputeDomain detection + `auto` channel resolution (component 4)
    # ------------------------------------------------------------------

    def _detect_compute_domains(self) -> list[tuple[str, str]]:
        """Best-effort: existing ComputeDomains as ``(name, channel_template)``.

        Read-only (``kubectl get computedomains -o json``, namespaced). Returns
        ``[]`` on any failure (CRD absent, no RBAC, parse error) so callers degrade
        to a hint rather than erroring. Never creates anything.
        """
        rc, out, _err = self._kubectl_sync(
            ["get", "computedomains", *self._ns_args(), "-o", "json"]
        )
        if rc != 0 or not out:
            return []
        try:
            items = json.loads(out).get("items", [])
        except (ValueError, AttributeError):
            return []
        result: list[tuple[str, str]] = []
        for it in items:
            meta = it.get("metadata", {}) if isinstance(it, dict) else {}
            spec = it.get("spec", {}) if isinstance(it, dict) else {}
            channel = (
                spec.get("channel", {})
                .get("resourceClaimTemplate", {})
                .get("name", "")
            )
            result.append((str(meta.get("name", "")), str(channel)))
        return result

    def _resolve_use_compute_domain_channel(self) -> None:
        """Resolve/verify ``use_compute_domain_channel`` against existing domains.

        - ``off``/empty/None -> nothing to claim (no detection).
        - a named value -> verify it is a channel template name; if it is actually a
          ComputeDomain *name*, auto-correct it to that domain's channel template
          (see ``_verify_or_autocorrect_named_channel``).
        - ``auto`` -> resolve to the sole usable ComputeDomain channel: exactly one
          -> claim it; zero or an incomplete domain -> hint (admin must provision or
          finish one, or run intra-node); many -> hint (ambiguous; name one). Never
          claims a guessed/ambiguous domain.

        Best-effort throughout (never raises).
        """
        cfg = self._use_compute_domain_channel_cfg
        if cfg is None:
            return  # off/empty: nothing to claim, nothing to detect
        if cfg != "auto":
            # A named channel: verify it, and auto-correct a ComputeDomain name to
            # its channel template name (the common, confusing mistake).
            self._verify_or_autocorrect_named_channel(cfg)
            return
        if self._compute_domain_channel is not None:
            return  # already resolved
        try:
            domains = self._detect_compute_domains()
        except Exception as e:  # best-effort detection
            _logger.info(
                "Kubernetes backend '%s': ComputeDomain detection failed (%s); "
                "'use_compute_domain_channel: auto' left unresolved.",
                self.name,
                e,
            )
            return
        channels = [(name, ch) for name, ch in domains if ch]
        if len(channels) == 1:
            name, channel = channels[0]
            self._compute_domain_channel = channel
            _logger.info(
                "Kubernetes backend '%s': 'use_compute_domain_channel: auto' "
                "resolved to the sole existing ComputeDomain '%s' (channel '%s').",
                self.name,
                name,
                channel,
            )
        elif not domains:
            _logger.warning(
                "Kubernetes backend '%s': 'compute_domain.channel: auto' found "
                "no existing ComputeDomain to join. Cross-node NVLink KV needs an "
                "admin-provisioned IMEX ComputeDomain -- either provision one and "
                "name its channel via compute_domain.channel, set "
                "compute_domain.create: true to create one, or run intra-node "
                "(co-located prefill+decode over NVLink).",
                self.name,
            )
        elif not channels:
            names = ", ".join(sorted(name for name, _channel in domains))
            _logger.warning(
                "Kubernetes backend '%s': 'compute_domain.channel: auto' found "
                "ComputeDomain(s) with no usable channel template (%s). Wait for "
                "spec.channel.resourceClaimTemplate.name to be populated, set an "
                "explicit usable template name, or run intra-node; sflow will not "
                "guess.",
                self.name,
                names,
            )
        else:
            # Pair each ComputeDomain with its channel template so the user sets the
            # RIGHT value: compute_domain.channel wants the channel TEMPLATE name
            # (spec.channel.resourceClaimTemplate.name), NOT the ComputeDomain's own
            # CR name. Naming the CR points the pod at a non-existent
            # ResourceClaimTemplate -> it stays Pending with FailedResourceClaimCreation.
            pairs = "; ".join(
                f"ComputeDomain '{name}' -> template '{ch}'"
                for name, ch in sorted(channels)
            )
            example = min(ch for _name, ch in channels)
            _logger.warning(
                "Kubernetes backend '%s': 'compute_domain.channel: auto' is "
                "ambiguous -- multiple ComputeDomains exist (%s). Set "
                "compute_domain.channel to the channel TEMPLATE name "
                "(spec.channel.resourceClaimTemplate.name), e.g. '%s' -- NOT the "
                "ComputeDomain name; sflow will not guess.",
                self.name,
                pairs,
                example,
            )

    def _preflight_check_compute_domain(self) -> None:
        """Hard-fail (real run) when a NAMED ``compute_domain.channel`` is configured
        but the cluster has no IMEX ComputeDomain support (CRD definitively ABSENT).

        A named channel is an EXPLICIT request for a specific ComputeDomain, so a
        cluster that cannot provide one is a config mistake the user should fix. Only
        a definitive ABSENT raises (API discovery ran and ``compute-domain.nvidia.com``
        is not served); UNKNOWN / undetectable stays best-effort (never blocks). Never
        fires for ``off``/empty (asks for nothing) or ``auto`` (degrades gracefully).
        Runs only on real ``sflow run`` -- preflight is skipped in --dry-run.
        """
        cfg = self._use_compute_domain_channel_cfg
        if cfg is None or cfg == "auto":
            return
        try:
            state = self.detect_capability(COMPUTE_DOMAIN, check_usable=False)
        except Exception:
            return  # cannot determine -> stay best-effort, do not block the run
        if state is CapabilityState.ABSENT:
            raise ValueError(
                f"Kubernetes backend '{self.name}': compute_domain.channel is set to "
                f"'{cfg}', but this cluster has no IMEX ComputeDomain support -- the "
                "'compute-domain.nvidia.com' CRD (NVIDIA DRA driver ComputeDomain) is "
                "not installed/served. Remove compute_domain.channel from the recipe "
                "(or set it to 'disable') to run without Multi-Node NVLink (MNNVL), or "
                "install the NVIDIA DRA driver's ComputeDomain support on the cluster."
            )

    def _verify_or_autocorrect_named_channel(self, value: str) -> None:
        """Auto-correct a ComputeDomain *name* to its channel *template* name.

        ``compute_domain.channel`` must be the channel template name
        (``spec.channel.resourceClaimTemplate.name``). Users frequently set it to the
        ComputeDomain's own CR name by mistake -> the pod then claims a non-existent
        ResourceClaimTemplate and stays Pending (FailedResourceClaimCreation). When
        the value matches a detected ComputeDomain name (and is not already a valid
        template), rewrite it to that domain's channel template.

        No-op when the value is already a channel template name. Best-effort: on
        detection failure / no domains / no match, keep the value verbatim so
        offline and no-RBAC clusters behave exactly as before.
        """
        try:
            domains = self._detect_compute_domains()
        except Exception:  # best-effort; keep the user's value verbatim
            return
        templates = {ch for _name, ch in domains if ch}
        if not templates or value in templates:
            return  # nothing detected, or already a valid template -> no-op
        template_by_cd_name = {name: ch for name, ch in domains if ch}
        template = template_by_cd_name.get(value)
        if template and template != value:
            self._compute_domain_channel = template
            _logger.warning(
                "Kubernetes backend '%s': compute_domain.channel '%s' is a "
                "ComputeDomain name, not a channel template -- auto-correcting to its "
                "channel template '%s' (spec.channel.resourceClaimTemplate.name). Set "
                "it to '%s' to silence this.",
                self.name,
                value,
                template,
                template,
            )

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def placeholder_allocation(self) -> Allocation:
        count = max(int(self._nodes), 1)
        nodes = [
            ComputeNode(
                name=f"{self.name}-node{i}",
                ip_address=f"0.0.0.{i + 1}",
                index=i,
                num_gpus=self._gpu_per_node,
            )
            for i in range(count)
        ]
        return Allocation(allocation_id="kubernetes", nodes=nodes, owned=False)

    async def allocate(self) -> Allocation:
        alloc_id = uuid.uuid4().hex[:8]
        self._pending_alloc_id = alloc_id
        # Ensure the NVLink-domain scope + `auto` channel are resolved (idempotent;
        # preflight usually ran them, but a skipped/partial preflight must not leave
        # them unset).
        self._detect_nvlink_scope()
        self._resolve_use_compute_domain_channel()
        await self._resolve_handoff_mode()
        count = max(self._nodes, 1)
        name_prefix = sanitize_name(self.name)[:12]
        pod_names = [f"sflow-res-{name_prefix}-{alloc_id}-{i}" for i in range(count)]

        gph = self._gpu_per_node
        holds_gpus = bool(gph and gph > 0)

        # DRA placeholders hold the node's GPUs via a shared ResourceClaimTemplate
        # (one ResourceClaim is generated per placeholder pod -> distinct devices
        # per node). Created before the pods so the claim exists at scheduling.
        resv_rct_name: str | None = None
        if self._scheduling == "dra" and holds_gpus:
            resv_rct_name = f"sflow-res-{name_prefix}-{alloc_id}-gpu"

        # NVLink-domain placement: on a multi-domain, rack-scoped cluster, pin all
        # reservation pods into one physical NVLink domain (topologyKey = the label)
        # so cross-node NVLink is possible. Only when a label key is configured.
        nvlink_topo_key = (
            self._nvlink_domain_label_key
            if (
                self._nvlink_domain_label_key
                and self.nvlink_domain_scope == "rack"
                and count > 1
            )
            else None
        )

        manifests = [
            render_reservation_pod_manifest(
                pod_name=pod_names[i],
                allocation_id=alloc_id,
                namespace=self._namespace,
                image_pull_policy=self._image_pull_policy,
                scheduling=self._scheduling,
                gpu_count=gph if holds_gpus else None,
                resource_claim_name=resv_rct_name,
                host_network=self._host_network,
                node_selector=self._node_selector,
                tolerations=self._effective_tolerations(),
                include_nodes=self._include_nodes,
                exclude_nodes=self._exclude_nodes,
                nvlink_domain_topology_key=nvlink_topo_key,
                placeholder_image=self._placeholder_image,
                gpu_resource_name=self._gpu_resource_name,
            )
            for i in range(count)
        ]

        try:
            if resv_rct_name is not None:
                await self._apply_manifest(
                    render_resource_claim_template(
                        name=resv_rct_name,
                        namespace=self._namespace,
                        device_class=self._gpu_device_class,
                        count=int(gph),
                        selectors=self._device_selectors,
                        allocation_id=alloc_id,
                    )
                )
            allocation = await self._allocate_reserved(alloc_id, pod_names, manifests)
            # `disable` runs detection too, but only to pin the routable control NIC
            # (SFLOW_PRIMARY_IFACE / NCCL socket iface); it never runs the RDMA
            # provider chain -- cross-node NCCL/gloo still ride sockets when IB is off.
            if self._rdma_mode in ("auto", "disable"):
                await self._detect_network_env(pod_names)
            # Log-only CPU/NUMA/GPU topology of the reserved GPU nodes (diagnostics for
            # rank binding); best-effort, never affects the allocation.
            if holds_gpus:
                await self._probe_node_topology(pod_names)
            return allocation
        except BaseException:
            # BaseException (not just Exception) so a cancel (Ctrl+C) mid-allocation
            # still releases the reservation pods instead of leaking them.
            await self._release_alloc(alloc_id)
            self._pending_alloc_id = None
            raise

    async def _probe_node_topology(self, pod_names: list[str]) -> None:
        """Best-effort, log-only: dump each reserved node's CPU / NUMA / GPU topology.

        Purely informational -- surfaces the pod's cpuset size, which NUMA nodes carry
        CPUs, and each GPU's CPU/NUMA affinity, so it's visible whether CPU<->NUMA<->GPU
        binding is sane (e.g. an 8-CPU cpuset confined to one NUMA node while the GPUs
        span two). NEVER affects scheduling, reservation, or rank binding; any failure
        (no numactl/nvidia-smi, exec denied, pod still starting) is skipped. Probes all
        reserved nodes concurrently since they can differ.
        """
        if not pod_names:
            return
        pod_to_node = {p: n for n, p in self._node_to_resv_pod.items()}
        reports: dict[str, str] = {}

        async def _probe(pod: str) -> None:
            try:
                rc, out, _ = await self._kubectl(
                    ["exec", pod, *self._ns_args(), "--", "sh", "-c",
                     _NODE_TOPOLOGY_PROBE_SH]
                )
            except Exception:
                return  # best-effort: never let a topology dump break allocation
            if rc == 0 and out.strip():
                node = pod_to_node.get(pod, pod)
                reports[node] = out.strip()
                _logger.info(
                    "Kubernetes backend '%s': reserved node '%s' CPU/NUMA/GPU "
                    "topology:\n%s",
                    self.name, node, out.strip(),
                )

        await asyncio.gather(*(_probe(p) for p in pod_names), return_exceptions=True)
        if reports:
            # Stored for the summary file's Node Topology section (read at render time).
            self._node_topology_report = "\n\n".join(
                f"{node}:\n{reports[node]}" for node in sorted(reports)
            )

    @property
    def node_topology_report(self) -> str | None:
        """Per-node CPU/NUMA/GPU topology from the reservation probe (or None)."""
        return self._node_topology_report

    async def _detect_network_env(self, pod_names: list[str]) -> None:
        """Best-effort: detect RDMA + build the IB/NCCL/UCX/NIXL plan.

        The reservation pods run with ``host_network`` (when configured), so they
        see the host NICs. We probe each RDMA netdev's InfiniBand device under
        ``/sys/class/net/<dev>/device/infiniband`` (e.g. ``mlx5_0``) and the routable
        control interface from the default route (``/proc/net/route``), fetch the
        scheduling node's ``allocatable``, then run the RDMA provider chain (see
        ``k8s.rdma.detect_rdma``) to build the ``RdmaPlan`` injected into task pods.
        Any failure (no RDMA, exec denied, pod not ready) leaves the disabled/TCP
        plan -- never fatal.
        """
        if not pod_names:
            return
        detect = (
            'for d in /sys/class/net/*/device/infiniband; do '
            'for ib in $(ls "$d" 2>/dev/null); do echo "HCA $ib"; done; done; '
            "awk '$2==\"00000000\"{print \"IFACE \"$1; exit}' /proc/net/route 2>/dev/null"
        )
        out = ""
        try:
            # The reservation sleeper may still be ContainerCreating right after
            # scheduling; retry a few times before giving up (best-effort).
            for _ in range(5):
                rc, out, _err = await self._kubectl(
                    ["exec", pod_names[0], *self._ns_args(), "--", "sh", "-c", detect]
                )
                if rc == 0:
                    break
                await asyncio.sleep(3)
            else:
                _logger.info(
                    "Kubernetes backend '%s': RDMA autodetect skipped (could not exec "
                    "reservation pod); device/interface selection left to the libs.",
                    self.name,
                )
                return
        except Exception as e:  # never let best-effort detection break allocation
            _logger.info(
                "Kubernetes backend '%s': RDMA autodetect errored (%s); skipping.",
                self.name,
                e,
            )
            return
        hcas = sorted(
            {ln.split(None, 1)[1].strip() for ln in out.splitlines() if ln.startswith("HCA ")}
        )
        ifaces = [
            ln.split(None, 1)[1].strip() for ln in out.splitlines() if ln.startswith("IFACE ")
        ]
        primary_iface = ifaces[0] if ifaces else ""

        # `rdma: disable` -- explicit kill switch. Keep the RDMA/IB transport OFF
        # (never run the provider chain, so no HCA is provisioned), but still pin the
        # probed routable NIC: NCCL/gloo fall back to sockets when IB is disabled, so
        # SFLOW_PRIMARY_IFACE (read by recipes) and NCCL_SOCKET_IFNAME must name a
        # real cross-node device even though we have no IB.
        if self._rdma_mode == "disable":
            self._rdma_plan = RdmaPlan.off(socket_iface=primary_iface)
            _logger.info(
                "Kubernetes backend '%s': RDMA disabled (rdma: disable); NCCL forced "
                "onto sockets, control interface pinned to '%s' for cross-node "
                "NCCL/gloo.",
                self.name,
                primary_iface or "(none)",
            )
            return

        # Fetch the scheduling node's allocatable once, then let the provider chain
        # (GKE -> shared-device-plugin -> host-device -> TCP) pick how task pods
        # request RDMA. The per-pod NIC slice + device grants are applied by the
        # operator from the resulting plan.
        node_name, allocatable = await self._node_allocatable(pod_names[0])
        # gIB installer presence gates the GKE lib mounts (see RdmaDetectContext):
        # its host paths include the driver dir (/usr/local/nvidia), so mounting
        # them when absent would mask libcuda.so.1. Only probe when the node
        # advertises GKE RDMA NICs and the provider chain may pick GKE.
        gib_installed = False
        gke_rdma = any(
            "networking.gke.io.networks/rdma-" in str(k) for k in allocatable
        )
        if gke_rdma and self._rdma_forced in (None, "gke"):
            gib_installed = await self._gib_installer_present(node_name)
        ctx = RdmaDetectContext(
            node_name=node_name,
            node_allocatable=allocatable,
            hcas=hcas,
            primary_iface=primary_iface,
            host_network=self._host_network,
            gib_installed=gib_installed,
        )
        self._rdma_plan = detect_rdma(ctx, forced=self._rdma_forced)
        if self._rdma_plan.enabled and self._rdma_plan.provider == "gke" and not gib_installed:
            self._warn_gib_installer_absent()
        self._log_rdma_plan(hcas, primary_iface)

    async def _gib_installer_present(self, node_name: str = "") -> bool:
        """True if the GKE gIB installer (``nccl-rdma-installer`` DaemonSet) has a
        Running pod ON ``node_name`` -- a per-node proxy for its host paths
        (``/home/kubernetes/bin/gib`` + ``/home/kubernetes/bin/nvidia``) existing on
        the scheduling node.

        Checking the specific node (not merely that the DaemonSet exists somewhere in
        the cluster) avoids a false positive during a partial/rolling install: a node
        that has not yet received the installer would otherwise get ``type: Directory``
        gIB mounts that fail to schedule. Best-effort: any probe failure (permissions,
        exec error, no node name) returns False, so sflow omits the lib mounts rather
        than risk masking the driver path. Falls back to a cluster-wide check when the
        scheduling node is unknown."""
        fields = "status.phase=Running"
        if node_name:
            fields += f",spec.nodeName={node_name}"
        try:
            rc, out, _err = await self._kubectl(
                [
                    "get",
                    "pods",
                    "-n",
                    self._gib_installer_namespace,
                    "--field-selector",
                    fields,
                    "-o",
                    "jsonpath={.items[*].metadata.name}",
                ]
            )
        except Exception:  # never let a best-effort probe break allocation
            return False
        return rc == 0 and "nccl-rdma-installer" in out

    def _warn_gib_installer_absent(self) -> None:
        """Hint that GKE RDMA is active but the gIB installer is not on the node."""
        _logger.warning(
            "Kubernetes backend '%s': GKE GPUDirect-RDMA is active but no running "
            "'nccl-rdma-installer' pod was found on the scheduling node, so "
            "'/home/kubernetes/bin/gib' + the gIB NCCL tuning (NCCL_NET=gIB / "
            "set_nccl_env.sh) are absent. sflow will NOT mount the gIB libs (mounting "
            "the missing driver path /usr/local/nvidia would mask libcuda.so.1), and "
            "multi-node NCCL runs over the untuned built-in IB transport (single-node "
            "RDMA is unaffected). For tuned GPUDirect-RDMA, deploy the installer: "
            "kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/"
            "container-engine-accelerators/master/gpudirect-rdma/nccl-rdma-installer.yaml",
            self.name,
        )

    def _log_rdma_plan(self, hcas: list[str], primary_iface: str) -> None:
        """Emit one informative line about the resolved RDMA plan."""
        plan = self._rdma_plan
        if plan.enabled:
            _logger.info(
                "Kubernetes backend '%s': RDMA enabled via '%s' (%d NIC(s): %s; "
                "control iface '%s').",
                self.name,
                plan.provider,
                len(plan.nic_specs),
                ",".join(h for _r, h in plan.nic_specs),
                primary_iface or "(none)",
            )
        elif plan.net_env and hcas:
            # RDMA hardware is on the node but no provider matched (e.g. no device
            # plugin and host_network is off). Pods will run KV/NCCL over slow TCP,
            # so warn -- this is a degradation the user probably did not intend.
            _logger.warning(
                "Kubernetes backend '%s': RDMA HCAs detected (%s) but no usable RDMA "
                "provider matched; pods will fall back to slow TCP on interface '%s' "
                "for UCX/NCCL/NIXL. Enable an RDMA device plugin or host_network to "
                "use RDMA.",
                self.name,
                ",".join(hcas),
                primary_iface,
            )
        elif plan.net_env:
            _logger.info(
                "Kubernetes backend '%s': no scoped RDMA available; pinned NCCL/gloo "
                "socket interface to '%s'.",
                self.name,
                primary_iface,
            )
        else:
            _logger.info(
                "Kubernetes backend '%s': no routable interface detected; leaving "
                "UCX/NCCL device selection to the libs.",
                self.name,
            )

    async def _node_allocatable(self, pod_name: str) -> tuple[str, dict[str, str]]:
        """Return the pod's scheduling node name + its ``status.allocatable`` map.

        Best-effort: returns ``("", {})`` (or ``(node, {})``) on any failure so RDMA
        detection degrades to the TCP fallback rather than erroring.
        """
        rc, node, _err = await self._kubectl(
            ["get", "pod", pod_name, *self._ns_args(), "-o", "jsonpath={.spec.nodeName}"]
        )
        node = (node or "").strip()
        if rc != 0 or not node:
            return "", {}
        rc, out, _err = await self._kubectl(["get", "node", node, "-o", "json"])
        if rc != 0 or not out:
            return node, {}
        try:
            allocatable = json.loads(out).get("status", {}).get("allocatable", {})
        except (ValueError, AttributeError):
            return node, {}
        return node, {str(k): str(v) for k, v in allocatable.items()}

    async def _allocate_reserved(
        self,
        alloc_id: str,
        pod_names: list[str],
        manifests: list[dict[str, Any]],
    ) -> Allocation:
        """Reserve real nodes, then discover their identities (names + InternalIPs)."""
        await asyncio.gather(*[self._apply_manifest(m) for m in manifests])
        scheduled: tuple[tuple[str, str], ...] = await asyncio.gather(
            *[self._wait_for_pod_scheduled(name) for name in pod_names]
        )
        internal_ips: tuple[str, ...] = await asyncio.gather(
            *[
                self._node_internal_ip(node_name, fallback=host_ip)
                for node_name, host_ip in scheduled
            ]
        )

        node_names = [s[0] for s in scheduled]
        # Map each real node -> the placeholder pod holding it, for the handoff.
        self._node_to_resv_pod = dict(zip(node_names, pod_names, strict=True))

        # Post-schedule NVLink-domain validation: warn if reserved nodes straddle
        # NVLink domains (rack scope + label key configured). Warn-only.
        await self._validate_nvlink_domain_placement(node_names)

        # NVLink (MNNVL / IMEX): stand up a ComputeDomain so task pods can claim an
        # IMEX channel keyed off this allocation. Independent of the GPU scheduling
        # mode -- works with device_plugin GPUs too (ComputeDomain-only DRA driver).
        # Skipped when reusing an existing channel (self._compute_domain_channel set).
        if self._creates_own_compute_domain:
            await self._create_compute_domain_cr(alloc_id, num_nodes=len(node_names))

        self._pending_alloc_id = None
        nodes = [
            ComputeNode(
                name=node_names[i],
                ip_address=internal_ips[i],
                index=i,
                num_gpus=self._gpu_per_node,
            )
            for i in range(len(pod_names))
        ]
        _logger.info(
            "Kubernetes backend '%s' reserved %d node(s): %s",
            self.name,
            len(nodes),
            ", ".join(f"{n.name}={n.ip_address}" for n in nodes),
        )
        return Allocation(allocation_id=alloc_id, nodes=nodes, owned=True)

    async def _create_compute_domain_cr(self, alloc_id: str, *, num_nodes: int) -> None:
        """Apply a ComputeDomain CR for this allocation (Multi-Node NVLink / IMEX)."""
        name_prefix = sanitize_name(self.name)[:12]
        cd_name = f"sflow-cd-{name_prefix}-{alloc_id}"
        channel_template = f"{cd_name}-channel"
        manifest = render_compute_domain_manifest(
            name=cd_name,
            num_nodes=num_nodes,
            channel_template_name=channel_template,
            allocation_id=alloc_id,
            namespace=self._namespace,
        )
        await self._apply_manifest(manifest)
        self._compute_domain_channel = channel_template
        _logger.info(
            "Kubernetes backend '%s' created ComputeDomain '%s' (numNodes=%d); "
            "multi-node task pods will claim channel template '%s'.",
            self.name,
            cd_name,
            num_nodes,
            channel_template,
        )

    async def _release_alloc(self, alloc_id: str) -> None:
        """Delete every object belonging to ``alloc_id`` (best-effort, concurrent)."""
        deletions = [
            self._delete_by_alloc_label("pod", alloc_id),
            self._delete_by_alloc_label("configmap", alloc_id),
            self._delete_by_alloc_label("secret", alloc_id),
            self._delete_by_alloc_label("resourceclaimtemplate.resource.k8s.io", alloc_id),
        ]
        # Only delete a ComputeDomain we created; a reused (existing) domain is not
        # ours to remove -- and we may lack computedomains RBAC entirely.
        if self._creates_own_compute_domain:
            deletions.append(
                self._delete_by_alloc_label("computedomain.resource.nvidia.com", alloc_id)
            )
        await asyncio.gather(*deletions, return_exceptions=True)

    async def release(self, allocation: Allocation) -> None:
        # Let any in-flight probe-pod creation finish so the label-based delete
        # below reliably reaps it (avoids a create-after-delete orphan).
        await self._await_probe_pod_task()
        if not allocation.allocation_id:
            return
        await self._release_alloc(allocation.allocation_id)

    def emergency_release(self, allocation: Allocation) -> None:
        # Best-effort: stop any in-flight probe-pod creation; the label-based
        # pod delete below reaps a probe pod that was already applied.
        if self._probe_pod_task is not None and not self._probe_pod_task.done():
            self._probe_pod_task.cancel()
        alloc_id = self._pending_alloc_id or getattr(allocation, "allocation_id", None)
        if not alloc_id:
            return
        self._delete_by_alloc_label_sync("pod", alloc_id)
        self._delete_by_alloc_label_sync("configmap", alloc_id)
        self._delete_by_alloc_label_sync("secret", alloc_id)
        self._delete_by_alloc_label_sync(
            "resourceclaimtemplate.resource.k8s.io", alloc_id
        )
        if self._creates_own_compute_domain:
            self._delete_by_alloc_label_sync(
                "computedomain.resource.nvidia.com", alloc_id
            )
        self._pending_alloc_id = None

    def dry_run_details(self) -> list[tuple[str, str]]:
        details: list[tuple[str, str]] = [("nodes", str(self._nodes))]
        if self._namespace is not None:
            details.append(("namespace", self._namespace))
        if self._gpu_per_node is not None:
            details.append(("gpus_per_node", str(self._gpu_per_node)))
        if self._image_pull_policy is not None:
            details.append(("image_pull_policy", self._image_pull_policy))
        details.append(("scheduling", self._scheduling))
        if self._scheduling == "dra":
            details.append(("gpu_device_class", self._gpu_device_class))
        if self._node_selector:
            details.append(
                ("node_selector", str(dict(sorted(self._node_selector.items()))))
            )
        if self._host_network:
            details.append(("host_network", str(self._host_network)))
        if self._host_ipc:
            details.append(("host_ipc", str(self._host_ipc)))
        if self._merge_colocated_gpu_pods:
            details.append(
                ("merge_colocated_gpu_pods", str(self._merge_colocated_gpu_pods))
            )
        if self._nvlink_domain_cfg != "auto":
            details.append(("nvlink_domain", self._nvlink_domain_cfg))
        if self._nvlink_domain_label_key:
            details.append(
                ("nvlink_domain_label_key", self._nvlink_domain_label_key)
            )
        if self._use_compute_domain_channel_cfg:
            details.append(
                ("use_compute_domain_channel", self._use_compute_domain_channel_cfg)
            )
        if self._create_compute_domain:
            details.append(("create_compute_domain", str(self._create_compute_domain)))
        details.append(("reservation_timeout", str(self._reservation_timeout)))
        if self._kubeconfig:
            details.append(("kubeconfig", self._kubeconfig))
        if self._kube_context:
            details.append(("context", self._kube_context))
        if self._kubectl_extra_args:
            details.append(("kubectl_args", str(list(self._kubectl_extra_args))))
        if self._include_nodes:
            details.append(("include_nodes", str(list(self._include_nodes))))
        if self._exclude_nodes:
            details.append(("exclude_nodes", str(list(self._exclude_nodes))))
        if self._rdma_mode == "disable":
            details.append(("rdma", "disable"))
        elif self._rdma_forced:
            details.append(("rdma", self._rdma_forced))
        if self._rdma_plan.net_env:
            details.append(
                ("rdma_env", str(dict(sorted(self._rdma_plan.net_env.items()))))
            )
        if self._extra_args:
            details.append(("extra_args", str(list(self._extra_args))))
        return details

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        # Every workload needs an image, which lives on an operator, so a task
        # with no operator cannot run on Kubernetes.
        raise ValueError(
            f"Kubernetes backend '{self.name}': task has no operator. Kubernetes "
            "tasks require an explicit 'k8s' operator that carries a workload "
            "'image:'. The backend has no image of its own."
        )

    def monitor_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        # Node-level hardware monitoring is NOT implemented for Kubernetes yet (a
        # privileged DCGM / nvidia-smi DaemonSet would be the proper path). The
        # backend advertises this via ``capabilities.supports_host_monitoring =
        # False``, and the monitor planner (``_MonitorPlanner._monitorable``) skips
        # k8s monitor targets entirely -- so this method is normally never reached.
        # It remains only as a defensive fallback: run the collector on the sflow
        # driver host and warn loudly that the metrics reflect the driver, not the
        # reserved k8s node(s).
        from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig

        _logger.warning(
            "Monitor '%s': backend '%s' is Kubernetes; node-level hardware "
            "monitoring is not implemented yet, so it is normally skipped. This "
            "fallback collects on the sflow driver host, NOT the reserved "
            "node(s) -- the metrics do not reflect the GPU nodes.",
            name,
            self.name,
        )
        operator = BashOperator(BashOperatorConfig(name=name, log_to_file=False))
        return configure_bare_monitor_operator(
            operator, backend=self, assigned_nodes=assigned_nodes
        )

    @classmethod
    def resolve_config(
        cls,
        conf: KubernetesBackendConfig,
        *,
        resolver: Any,
        ctx: dict[str, Any],
        workflow_name: str,
    ) -> KubernetesBackendConfig:
        nodes = resolver.resolve(conf.nodes, ctx) if conf.nodes is not None else 1
        try:
            nodes_i = max(int(nodes), 1)
        except Exception as e:
            raise ValueError(
                f"Backend '{conf.name}' nodes must resolve to int, got {nodes!r}"
            ) from e

        gpus_per_node = None
        if conf.gpus_per_node is not None:
            resolved = resolver.resolve(conf.gpus_per_node, ctx)
            try:
                gpus_per_node = int(resolved)
            except Exception as e:
                raise ValueError(
                    f"Backend '{conf.name}' gpus_per_node must resolve to int, got {resolved!r}"
                ) from e
            if gpus_per_node < 0:
                raise ValueError(
                    f"Backend '{conf.name}' gpus_per_node must be >= 0, got {gpus_per_node}"
                )

        def _resolved_nonneg_int(field: str, raw: Any) -> int:
            value = resolver.resolve(raw, ctx)
            try:
                iv = int(value)
            except Exception as e:
                raise ValueError(
                    f"Backend '{conf.name}' {field} must resolve to int, got {value!r}"
                ) from e
            if iv < 0:
                raise ValueError(f"Backend '{conf.name}' {field} must be >= 0, got {iv}")
            return iv

        cpu_per_gpu = (
            None
            if conf.cpu_per_gpu is None
            else _resolved_nonneg_int("cpu_per_gpu", conf.cpu_per_gpu)
        )
        collect_max_file_size = (
            None
            if conf.collect_max_file_size is None
            else resolver.resolve(conf.collect_max_file_size, ctx)
        )
        cpu_request = (
            None
            if conf.cpu_request is None
            else _resolved_nonneg_int("cpu_request", conf.cpu_request)
        )

        namespace = (
            str(resolver.resolve(conf.namespace, ctx))
            if conf.namespace is not None
            else None
        )
        image_pull_policy = (
            str(resolver.resolve(conf.image_pull_policy, ctx))
            if conf.image_pull_policy is not None
            else None
        )
        # Resolve the probe-pod image like the other Resolvable fields (a recipe/CLI
        # override or a `${{ }}` expression); fall back to the built-in default when
        # unset. Must be re-emitted below -- omitting it from the rebuilt config
        # silently reverts any override to PROBE_POD_IMAGE_DEFAULT.
        probe_pod_image = (
            str(resolver.resolve(conf.probe_pod_image, ctx))
            if conf.probe_pod_image is not None
            else PROBE_POD_IMAGE_DEFAULT
        )
        extra_args = [str(resolver.resolve(a, ctx)) for a in (conf.extra_args or [])]
        # Cluster-portability overrides (default to the NVIDIA/GKE conventions).
        # Re-emitted below -- omitting any of these silently reverts an override.
        gpu_resource_name = str(resolver.resolve(conf.gpu_resource_name, ctx))
        gpu_product_label_key = str(resolver.resolve(conf.gpu_product_label_key, ctx))
        gib_installer_namespace = str(
            resolver.resolve(conf.gib_installer_namespace, ctx)
        )
        collect_grace_seconds = _resolved_nonneg_int(
            "collect_grace_seconds", conf.collect_grace_seconds
        )

        dra = None
        if conf.dra is not None:
            d = conf.dra
            dra = KubernetesDraConfig(
                gpu_device_class=str(resolver.resolve(d.gpu_device_class, ctx)),
                device_selectors=(
                    [str(resolver.resolve(s, ctx)) for s in d.device_selectors]
                    if d.device_selectors
                    else None
                ),
                nvlink_domain_label_key=d.nvlink_domain_label_key,
                rdma_device_class=(
                    str(resolver.resolve(d.rdma_device_class, ctx))
                    if d.rdma_device_class is not None
                    else None
                ),
                rdma_match_attribute=str(
                    resolver.resolve(d.rdma_match_attribute, ctx)
                ),
            )

        # Multi-Node NVLink (IMEX ComputeDomain): a top-level block, independent of
        # `dra` / GPU scheduling.
        compute_domain = None
        if conf.compute_domain is not None:
            c = conf.compute_domain
            compute_domain = KubernetesComputeDomainConfig(
                channel=(
                    str(resolver.resolve(c.channel, ctx))
                    if c.channel is not None
                    else None
                ),
                create=bool(c.create),
            )

        reservation = None
        if conf.reservation is not None:
            r = conf.reservation
            timeout_raw = resolver.resolve(r.timeout, ctx)
            try:
                timeout_i = int(timeout_raw)
            except Exception as e:
                raise ValueError(
                    f"Backend '{conf.name}' reservation.timeout must resolve to int, "
                    f"got {timeout_raw!r}"
                ) from e
            reservation = KubernetesReservationConfig(
                timeout=timeout_i,
                # Resolve the placeholder image override (or keep None -> the built-in
                # bash:5); dropping it here would silently revert an override.
                placeholder_image=(
                    str(resolver.resolve(r.placeholder_image, ctx))
                    if r.placeholder_image is not None
                    else None
                ),
                # Carry + resolve the handoff mode (a ${{ ... }} expression or literal);
                # dropping it silently reverts an explicit mode to "auto".
                handoff=str(resolver.resolve(r.handoff, ctx)),
            )

        volumes = None
        if conf.volumes:
            volumes = [
                KubernetesVolumeConfig(
                    name=v.name,
                    claim=(
                        str(resolver.resolve(v.claim, ctx))
                        if v.claim is not None
                        else None
                    ),
                    empty_dir=(
                        KubernetesEmptyDirConfig(
                            medium=v.empty_dir.medium,
                            size_limit=(
                                str(resolver.resolve(v.empty_dir.size_limit, ctx))
                                if v.empty_dir.size_limit is not None
                                else None
                            ),
                        )
                        if v.empty_dir is not None
                        else None
                    ),
                    mount_path=str(resolver.resolve(v.mount_path, ctx)),
                    sub_path=(
                        str(resolver.resolve(v.sub_path, ctx))
                        if v.sub_path is not None
                        else None
                    ),
                    read_only=v.read_only,
                    ensure_writable=v.ensure_writable,
                )
                for v in conf.volumes
            ]

        include_nodes, exclude_nodes = resolve_node_filters(resolver, conf, ctx)

        return KubernetesBackendConfig(
            name=conf.name,
            type="kubernetes",
            default=bool(getattr(conf, "default", False)),
            namespace=namespace,
            image_pull_policy=image_pull_policy,
            probe_pod_image=probe_pod_image,
            nodes=nodes_i,
            gpus_per_node=gpus_per_node,
            cpu_per_gpu=cpu_per_gpu,
            cpu_request=cpu_request,
            collect_max_file_size=collect_max_file_size,
            collect_grace_seconds=collect_grace_seconds,
            extra_args=extra_args,
            include_nodes=include_nodes,
            exclude_nodes=exclude_nodes,
            node_selector=(
                {
                    str(resolver.resolve(k, ctx)): str(resolver.resolve(v, ctx))
                    for k, v in conf.node_selector.items()
                }
                if conf.node_selector
                else None
            ),
            host_network=bool(conf.host_network),
            host_ipc=bool(conf.host_ipc),
            # Pass the tri-state through unchanged -- coercing to bool() here would
            # turn "disable" (a non-empty, truthy string) into True.
            merge_colocated_gpu_pods=conf.merge_colocated_gpu_pods,
            nvlink_domain=conf.nvlink_domain,
            scheduling=conf.scheduling,
            gpu_resource_name=gpu_resource_name,
            gpu_product_label_key=gpu_product_label_key,
            dra=dra,
            compute_domain=compute_domain,
            tolerations=conf.tolerations,
            volumes=volumes,
            rdma=conf.rdma,
            gib_installer_namespace=gib_installer_namespace,
            reservation=reservation,
        )
