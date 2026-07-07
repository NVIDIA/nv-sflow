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
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from sflow.config.schema import BackendConfig, Resolvable
from sflow.core.backend import (
    Allocation,
    Backend,
    BackendCapabilities,
    configure_bare_monitor_operator,
)
from sflow.core.backend_registry import register_backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.logging import get_logger
from sflow.plugins.operators._k8s_render import (
    DEFAULT_GPU_TOLERATION,
    SFLOW_ALLOC_LABEL,
    render_compute_domain_manifest,
    render_reservation_pod_manifest,
    render_resource_claim_template,
)
from sflow.plugins.operators._k8s_shell import sanitize_name
from sflow.plugins.backends._k8s_rdma import (
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

# How GPUs are requested for both placeholder and task pods.
#   dra           -> resource.k8s.io ResourceClaimTemplate (nvidia-dra-driver-gpu)
#   device_plugin -> legacy nvidia.com/gpu device-plugin limit
SchedulingMode = Literal["dra", "device_plugin"]

# RDMA/IB provisioning for task GPU pods (see _k8s_rdma.py). "auto" runs the
# provider chain (GKE -> shared-device-plugin -> host-device) and then exposes all
# node NICs so NCCL/UCX auto-select each GPU's closest device; "off" disables RDMA;
# a specific provider key forces that mechanism (skip auto-detection order).
RdmaMode = Literal["auto", "off", "gke", "shared_device_plugin", "host_device"]

# Merge-pod tri-state (see ``merge_colocated_gpu_pods``). ``off``/``False`` disable
# merging; ``auto`` (default) and ``on``/``True`` enable it. The backend resolves
# this to a bool (via a ``@property``) so ``_plan_merge_groups`` -- which reads that
# bool and already self-guards to >=2 co-located GPU tasks -- is unchanged.
MergeMode = Literal["auto", "on", "off"]

# NVLink-domain scope, which drives the interconnect priority order (NVLink -> IB ->
# TCP). ``auto`` detects ``node`` vs ``rack`` from the cluster's GPU product +
# ComputeDomain CRD presence (component 2); ``node``/``rack``/``off`` force it.
NvlinkDomain = Literal["auto", "node", "rack", "off"]


def _merge_enabled(value: bool | str) -> bool:
    """Resolve the ``merge_colocated_gpu_pods`` tri-state to a bool.

    ``off``/``False`` (and empty) disable merging; everything else (``auto``
    default, ``on``, ``True``) enables it.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("off", "false", "no", "0", "")


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
# ComputeDomain CRD (installed by the NVIDIA DRA driver in IMEX/ComputeDomain
# mode); its presence is the cross-node-fabric signal for rack scope.
_COMPUTE_DOMAIN_CRD = "computedomains.resource.nvidia.com"


class KubernetesDraConfig(BaseModel):
    """DRA (Dynamic Resource Allocation) options for ``scheduling: dra``."""

    # DeviceClass GPUs are requested from (nvidia-dra-driver-gpu default).
    gpu_device_class: Resolvable[str] = "gpu.nvidia.com"
    # Optional CEL expressions narrowing eligible devices (e.g. by product/memory).
    device_selectors: list[Resolvable[str]] | None = None
    # CREATE an NVIDIA ComputeDomain CR so pods get a Multi-Node NVLink (IMEX)
    # channel. This is an admin/privileged op: it requires RBAC to create
    # `computedomains.resource.nvidia.com`. Default off -- the default path is to
    # DETECT an existing domain and hint (see `use_compute_domain_channel: auto`).
    # (Renamed from `compute_domain`, still accepted as a deprecated alias.)
    create_compute_domain: bool = False
    # JOIN an existing ComputeDomain: the DRA ResourceClaimTemplate every GPU pod
    # claims to get an IMEX channel (the runtime wiring that lets MNNVL fabric
    # handles cross nodes). Value is the CR's
    # spec.channel.resourceClaimTemplate.name, or `auto` (claim the sole existing
    # ComputeDomain when there is exactly one -- component 4), or `off`/empty/None
    # (no claim). No ComputeDomain is created, so this needs no `computedomains`
    # RBAC (only claiming an existing template). Works with any GPU scheduling
    # (device_plugin or dra). (Renamed from `compute_domain_channel`, still accepted
    # as a deprecated alias -- the recipe sets it via COMPUTE_DOMAIN_CHANNEL.)
    use_compute_domain_channel: Resolvable[str] | None = None
    # Node label KEY identifying the physical NVLink domain (e.g.
    # `nvidia.com/gpu.clique`). Used ONLY for placement + straddle validation
    # (component 6) on clusters with MULTIPLE NVLink domains -- NOT for the IMEX
    # channel claim. Redundant on a single-domain (one NVL72 rack) cluster.
    nvlink_domain_label_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_deprecated_compute_domain_keys(cls, data: Any) -> Any:
        """Accept the pre-rename `compute_domain` / `compute_domain_channel` keys.

        Maps each to its new name (`create_compute_domain` /
        `use_compute_domain_channel`) when the new key is absent, emitting a
        one-time ``DeprecationWarning`` so existing recipes/tests keep working.
        """
        if not isinstance(data, dict):
            return data
        renames = {
            "compute_domain": "create_compute_domain",
            "compute_domain_channel": "use_compute_domain_channel",
        }
        for old, new in renames.items():
            if old in data:
                value = data.pop(old)
                if new not in data:
                    data[new] = value
                warnings.warn(
                    f"KubernetesDraConfig '{old}' is deprecated; use '{new}' "
                    "instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        return data
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


class KubernetesVolumeConfig(BaseModel):
    """A pre-existing PersistentVolumeClaim (PVC) to mount into task pods.

    A backend ``volumes:`` entry is workflow-wide storage: it is mounted into
    EVERY task pod of this backend (as a pod volume + container volumeMount at
    ``mount_path``), regardless of whether a task's script references that path.
    This is intentional -- a PVC is backend-level info and must be available to
    all sflow tasks on the backend (e.g. a task that loads a model via discovery,
    without ever naming the path, still needs it mounted).

    A common use is cluster-resident data (e.g. a model on shared storage) reaching
    pods without a node-local hostPath: declare the PVC + where it mounts, then point
    an ``fs://`` artifact at a path under ``mount_path``. When an ``fs://`` artifact
    path is covered by a declared PVC ``mount_path`` the PVC serves it, so the
    per-artifact hostPath fallback is skipped -- that path matching governs ONLY the
    hostPath fallback, never whether the PVC itself is mounted (it always is).

    The PVC itself (and its data) must already exist in the backend namespace --
    sflow does not create or populate it.
    """

    # Pod volume name (DNS-1123); also the key linking volume <-> volumeMount.
    name: str
    # Name of the existing PersistentVolumeClaim (spec.volumes[].pvc.claimName).
    claim: Resolvable[str]
    # Absolute path the PVC is mounted at inside each task pod.
    mount_path: Resolvable[str]
    # Optional path within the PVC to mount (volumeMount.subPath).
    sub_path: Resolvable[str] | None = None
    # Mount read-only (default: True -- model/data PVCs are typically read-only and
    # this is required to share a single PVC across pods on multiple nodes).
    read_only: bool = True

    @field_validator("mount_path")
    @classmethod
    def _mount_path_absolute(cls, v: Resolvable[str]) -> Resolvable[str]:
        # Skip template expressions; they are validated after resolution.
        if isinstance(v, str) and "${{" not in v and not v.startswith("/"):
            raise ValueError(f"volume mount_path must be absolute, got: {v!r}")
        return v


class KubernetesReservationConfig(BaseModel):
    """Tuning for the Kubernetes backend's node reservation."""

    # Seconds to wait for every placeholder pod to be scheduled onto a node.
    timeout: Resolvable[int] = 600


class KubernetesBackendConfig(BackendConfig):
    type: Literal["kubernetes"] = "kubernetes"
    # NOTE: there is intentionally no `image` field. Workload images are an
    # operator concern (the `k8s` operator's `image:`); reservation/placeholder
    # pods use the fixed internal sleeper image (RESERVATION_POD_IMAGE).
    namespace: Resolvable[str] | None = None
    image_pull_policy: Resolvable[str] | None = None
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
    # and an IMEX domain (nvidia-imex, or DRA `dra.create_compute_domain`).
    host_ipc: bool = False
    # Merge co-located GPU tasks into ONE pod (single container requesting the
    # union of their GPUs) so intra-node NVLink/cuda_ipc works between them. When
    # the planner assigns >=2 single-node GPU tasks to the same physical node,
    # they run as concurrent background processes in one pod; each keeps its own
    # <task>.log, probes, readiness, and dependents. Tri-state (bool |
    # "auto"/"on"/"off"), default "auto" (ENABLED): merging is sflow-owned pod
    # topology -- it enables intra-node NVLink/cuda_ipc between co-located workers
    # AND guarantees one channel-claiming GPU pod per node (the NVIDIA driver
    # publishes one IMEX channel per node). Only concurrent GPU tasks merge;
    # CPU-only infra (etcd/nats/frontend) stays in its own pod. Set "off"/False to
    # opt out. See docs/user/backends.md.
    merge_colocated_gpu_pods: MergeMode | bool = "auto"
    # NVLink-domain scope driving the interconnect priority order (see NvlinkDomain).
    # "auto" detects node|rack from the cluster (GPU product + ComputeDomain CRD);
    # "node"/"rack"/"off" override. Best-effort/warn-only; never hard-fails.
    nvlink_domain: NvlinkDomain = "auto"
    # GPU request mode (see SchedulingMode).
    scheduling: SchedulingMode = "dra"
    # DRA options (used when scheduling == "dra").
    dra: KubernetesDraConfig | None = None
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
    #   "off"            -> inject nothing (recipe/cluster handles it).
    #   "gke" | "shared_device_plugin" | "host_device"
    #                    -> force that provider (skip auto-detection order; e.g. opt
    #                       out of the privileged-ish host-device path).
    # Best-effort: if nothing is found, nothing is injected. Fine-tune NIC selection
    # per pod at runtime with the SFLOW_RDMA_AFFINITY env (auto|explicit|off).
    rdma: RdmaMode = "auto"
    # Optional tuning for the (always-on) reserve+discover+pin behavior.
    reservation: KubernetesReservationConfig | None = None

    def planning_node_count(self) -> Resolvable[int] | None:
        return self.nodes


@register_backend("kubernetes", KubernetesBackendConfig)
class KubernetesBackend(Backend):
    """Kubernetes backend: reserve+discover nodes, then run each task as pinned pod(s)."""

    def __init__(self, config: KubernetesBackendConfig):
        super().__init__(name=config.name)
        self.config = config
        resv = config.reservation
        self._reservation_timeout = (
            int(resv.timeout) if resv is not None and resv.timeout is not None else 600
        )
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
        # The channel ResourceClaimTemplate name GPU task pods claim (Multi-Node
        # NVLink / IMEX). Set from a named dra.use_compute_domain_channel (__init__),
        # a created domain (dra.create_compute_domain, allocate()), or `auto`
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
        self._nodes = int(config.nodes) if config.nodes is not None else 1
        self._gpu_per_node = (
            int(config.gpus_per_node) if config.gpus_per_node is not None else None
        )
        self._extra_args = [str(a) for a in (config.extra_args or [])]
        self._node_selector: dict[str, str] | None = config.node_selector
        self._host_network: bool = bool(config.host_network)
        self._host_ipc: bool = bool(config.host_ipc)
        self._merge_colocated_gpu_pods: bool = _merge_enabled(
            config.merge_colocated_gpu_pods
        )
        # NVLink-domain scope override ("auto"/"node"/"rack"/"off"). An explicit
        # override wins; "auto" is resolved to node|rack|off by detection at
        # preflight/allocate (component 2), cached in _nvlink_domain_scope_detected.
        self._nvlink_domain_cfg: str = str(config.nvlink_domain or "auto")
        self._nvlink_domain_override: str | None = (
            None if self._nvlink_domain_cfg == "auto" else self._nvlink_domain_cfg
        )
        self._nvlink_domain_scope_detected: str | None = None
        self._scheduling: str = str(config.scheduling)
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
        # CREATE a ComputeDomain CR (admin/privileged, explicit opt-in).
        self._create_compute_domain: bool = (
            bool(dra.create_compute_domain) if dra is not None else False
        )
        # JOIN an existing ComputeDomain by its channel template name. Classify the
        # configured value: None/empty/"off" -> off; "auto" -> resolve to the sole
        # existing domain at preflight/allocate (component 4); anything else -> a
        # named channel, claimed immediately.
        raw_channel = (
            str(dra.use_compute_domain_channel).strip()
            if dra is not None and dra.use_compute_domain_channel is not None
            else ""
        )
        low_channel = raw_channel.lower()
        if low_channel in ("", "off", "false", "no"):
            self._use_compute_domain_channel_cfg: str | None = None
        elif low_channel == "auto":
            self._use_compute_domain_channel_cfg = "auto"
        else:
            self._use_compute_domain_channel_cfg = raw_channel
        # Node label key for NVLink-domain placement/validation (component 6).
        self._nvlink_domain_label_key: str | None = (
            str(dra.nvlink_domain_label_key)
            if dra is not None and dra.nvlink_domain_label_key
            else None
        )
        # We only create (and later delete) a ComputeDomain when asked to AND not
        # joining an existing channel (named or auto). Joining an existing channel
        # needs no `computedomains` RBAC -- pods just claim the pre-existing template.
        self._creates_own_compute_domain: bool = (
            self._create_compute_domain
            and self._use_compute_domain_channel_cfg is None
        )
        # A named channel is exposed immediately (no allocate-time creation); "auto"
        # is filled in by ComputeDomain detection (component 4); a created domain is
        # filled in by allocate().
        if self._use_compute_domain_channel_cfg not in (None, "auto"):
            self._compute_domain_channel = self._use_compute_domain_channel_cfg
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
        # RDMA provisioning (see `rdma` config + _k8s_rdma.py). "off" disables it;
        # otherwise the provider chain runs in allocate() -- `_rdma_forced` pins one
        # provider when `rdma` names a specific key (else the whole chain is tried).
        # The resolved `_rdma_plan` (net env + per-pod NIC specs + device grants) is
        # filled in by _detect_network_env().
        rdma_mode = str(config.rdma or "auto")
        if rdma_mode == "off":
            self._rdma_mode = "off"
            self._rdma_forced: str | None = None
        else:
            self._rdma_mode = "auto"
            self._rdma_forced = rdma_mode if rdma_mode in RDMA_PROVIDER_KEYS else None
        self._rdma_plan = RdmaPlan.disabled()
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

    def apply_kubectl_config(self, cfg: Any) -> None:
        """Apply CLI-level kube access (``KubectlConfig``) to this backend.

        Sets the global kubectl flags (``--kubeconfig`` / ``--context`` + any
        passthroughs) used by every kubectl call, and overrides the namespace when
        ``--kube-namespace`` was given. The recipe itself stays cluster-agnostic.
        """
        self._kubeconfig = cfg.kubeconfig or None
        self._kube_context = cfg.context or None
        self._kubectl_extra_args = [str(a) for a in (cfg.extra_args or [])]
        if cfg.namespace:
            self._namespace = str(cfg.namespace)
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
        """kubectl global flags (``--kubeconfig`` / ``--context`` + passthroughs)."""
        args: list[str] = []
        if self._kubeconfig:
            args += ["--kubeconfig", self._kubeconfig]
        if self._kube_context:
            args += ["--context", self._kube_context]
        args += list(self._kubectl_extra_args)
        return args

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
    def merge_colocated_gpu_pods(self) -> bool:
        """Whether co-located GPU tasks are merged into one pod (shared NVLink).

        The tri-state (bool | "auto"/"on"/"off") config is resolved to this bool so
        ``_plan_merge_groups`` (which reads it) stays unchanged.
        """
        return self._merge_colocated_gpu_pods

    @property
    def nvlink_domain(self) -> str:
        """Configured NVLink-domain scope ("auto"/"node"/"rack"/"off").

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
        """Resolved NVLink-domain scope: ``node`` / ``rack`` / ``off``, or ``None``.

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
        or the TCP fallback pinned the routable interface. UCX device selection is
        intentionally left to the workload/library.
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
        resources, setting a matching ``NCCL_IB_HCA`` when build-time pinning is
        used. UCX is never pinned by the backend.
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
        runtime affinity preamble (see ``_k8s_rdma_preamble``) instead of a static
        per-pod NIC pin. False for GKE (fixed per-pod NIC subset) and when RDMA is
        off/unavailable.
        """
        return self._rdma_plan.allow_runtime_affinity

    @property
    def scheduling(self) -> str:
        return self._scheduling

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
        """PVC mounts (resolved) injected into task pods by the operator."""
        out: list[dict[str, Any]] = []
        for v in self._volumes:
            out.append(
                {
                    "name": str(v.name),
                    "claim": str(v.claim),
                    "mount_path": str(v.mount_path),
                    "sub_path": str(v.sub_path) if v.sub_path is not None else None,
                    "read_only": bool(v.read_only),
                }
            )
        return out

    @property
    def compute_domain_channel(self) -> str | None:
        """ComputeDomain channel ResourceClaimTemplate name, or None."""
        return self._compute_domain_channel

    def reservation_pod_for_node(self, node_name: str) -> str | None:
        """Return the placeholder pod holding ``node_name`` (create-before-destroy handoff)."""
        return self._node_to_resv_pod.get(node_name)

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
        if self._scheduling == "dra":
            self._preflight_check_dra()
        # Resolve the NVLink-domain scope now (best-effort) so downstream planning
        # (interconnect selection, placement, hints) has it. Warn-only.
        self._detect_nvlink_scope()
        # Resolve `use_compute_domain_channel: auto` to the sole existing domain
        # (best-effort; zero/many -> hint). No-op unless configured `auto`.
        self._resolve_use_compute_domain_channel()

    def _preflight_check_connectivity(self) -> None:
        """Fail fast if the kube access can't perform the operations sflow needs.

        Runs only on real ``sflow run`` (preflight is skipped in --dry-run). Uses
        non-mutating calls: ``kubectl get namespace`` (reachability + auth +
        namespace existence) and ``kubectl auth can-i`` for each required verb/
        resource (RBAC). Hard-fails with an actionable message; set
        ``SFLOW_SKIP_K8S_PREFLIGHT=1`` to bypass.
        """
        skip = os.environ.get("SFLOW_SKIP_K8S_PREFLIGHT", "")
        if skip and skip.lower() not in ("0", "false", "no"):
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

    def _preflight_check_dra(self) -> None:
        """Best-effort: warn (never fail) if the DRA DeviceClass is not visible."""
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
        return "off"

    def _compute_domain_crd_present(self) -> bool:
        """Best-effort: whether the ComputeDomain CRD is installed (IMEX driver)."""
        rc, _out, _err = self._kubectl_sync(
            ["get", "crd", _COMPUTE_DOMAIN_CRD, "-o", "name"], timeout="5s"
        )
        return rc == 0

    def _detect_gpu_product(self) -> str:
        """Best-effort: the ``nvidia.com/gpu.product`` label of a (selected) node."""
        args = [
            "get",
            "nodes",
            "-o",
            r"jsonpath={.items[*].metadata.labels.nvidia\.com/gpu\.product}",
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
        """Resolve the NVLink-domain scope (node|rack|off). Best-effort, warn-only.

        Returns the explicit ``nvlink_domain`` override when set; otherwise detects
        from the GPU product label + ComputeDomain CRD presence, caches the result,
        and logs it. Any detection failure degrades to ``off`` (never raises).
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
                "(%s); assuming 'off'.",
                self.name,
                e,
            )
            self._nvlink_domain_scope_detected = "off"
            return "off"
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
        """Resolve ``use_compute_domain_channel: auto`` to the sole existing domain.

        No-op unless the config is ``auto`` and not already resolved. Exactly one
        ComputeDomain -> claim its channel; zero -> skip + hint (admin must
        provision one, or run intra-node); many -> skip + hint (ambiguous; name
        one). Never claims a guessed/ambiguous domain. Best-effort (never raises).
        """
        if self._use_compute_domain_channel_cfg != "auto":
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
                "Kubernetes backend '%s': 'use_compute_domain_channel: auto' found "
                "no existing ComputeDomain to join. Cross-node NVLink KV needs an "
                "admin-provisioned IMEX ComputeDomain -- either provision one and "
                "name its channel via dra.use_compute_domain_channel, set "
                "dra.create_compute_domain: true to create one, or run intra-node "
                "(co-located prefill+decode over NVLink).",
                self.name,
            )
        else:
            names = ", ".join(sorted(name for name, _ch in channels))
            _logger.warning(
                "Kubernetes backend '%s': 'use_compute_domain_channel: auto' is "
                "ambiguous -- multiple ComputeDomains exist (%s). Name the one to "
                "join via dra.use_compute_domain_channel; sflow will not guess.",
                self.name,
                names,
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
            if self._rdma_mode == "auto":
                await self._detect_network_env(pod_names)
            return allocation
        except BaseException:
            # BaseException (not just Exception) so a cancel (Ctrl+C) mid-allocation
            # still releases the reservation pods instead of leaking them.
            await self._release_alloc(alloc_id)
            self._pending_alloc_id = None
            raise

    async def _detect_network_env(self, pod_names: list[str]) -> None:
        """Best-effort: detect RDMA + build the IB/NCCL/UCX/NIXL plan.

        The reservation pods run with ``host_network`` (when configured), so they
        see the host NICs. We probe each RDMA netdev's InfiniBand device under
        ``/sys/class/net/<dev>/device/infiniband`` (e.g. ``mlx5_0``) and the routable
        control interface from the default route (``/proc/net/route``), fetch the
        scheduling node's ``allocatable``, then run the RDMA provider chain (see
        ``_k8s_rdma.detect_rdma``) to build the ``RdmaPlan`` injected into task pods.
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

        # Fetch the scheduling node's allocatable once, then let the provider chain
        # (GKE -> shared-device-plugin -> host-device -> TCP) pick how task pods
        # request RDMA. The per-pod NIC slice + device grants are applied by the
        # operator from the resulting plan.
        node_name, allocatable = await self._node_allocatable(pod_names[0])
        ctx = RdmaDetectContext(
            node_name=node_name,
            node_allocatable=allocatable,
            hcas=hcas,
            primary_iface=primary_iface,
            host_network=self._host_network,
        )
        self._rdma_plan = detect_rdma(ctx, forced=self._rdma_forced)
        self._log_rdma_plan(hcas, primary_iface)

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
        if not allocation.allocation_id:
            return
        await self._release_alloc(allocation.allocation_id)

    def emergency_release(self, allocation: Allocation) -> None:
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
        if self._rdma_mode == "off":
            details.append(("rdma", "off"))
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
        extra_args = [str(resolver.resolve(a, ctx)) for a in (conf.extra_args or [])]

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
                create_compute_domain=bool(d.create_compute_domain),
                use_compute_domain_channel=(
                    str(resolver.resolve(d.use_compute_domain_channel, ctx))
                    if d.use_compute_domain_channel is not None
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
            reservation = KubernetesReservationConfig(timeout=timeout_i)

        volumes = None
        if conf.volumes:
            volumes = [
                KubernetesVolumeConfig(
                    name=v.name,
                    claim=str(resolver.resolve(v.claim, ctx)),
                    mount_path=str(resolver.resolve(v.mount_path, ctx)),
                    sub_path=(
                        str(resolver.resolve(v.sub_path, ctx))
                        if v.sub_path is not None
                        else None
                    ),
                    read_only=bool(v.read_only),
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
            nodes=nodes_i,
            gpus_per_node=gpus_per_node,
            extra_args=extra_args,
            include_nodes=include_nodes,
            exclude_nodes=exclude_nodes,
            node_selector=conf.node_selector,
            host_network=bool(conf.host_network),
            host_ipc=bool(conf.host_ipc),
            # Pass the tri-state through unchanged -- coercing to bool() here would
            # turn "off" (a non-empty, truthy string) into True.
            merge_colocated_gpu_pods=conf.merge_colocated_gpu_pods,
            nvlink_domain=conf.nvlink_domain,
            scheduling=conf.scheduling,
            dra=dra,
            tolerations=conf.tolerations,
            volumes=volumes,
            rdma=conf.rdma,
            reservation=reservation,
        )
