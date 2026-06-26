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
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from sflow.config.schema import BackendConfig, Resolvable
from sflow.core.backend import (
    Allocation,
    Backend,
    BackendCapabilities,
    configure_bare_monitor_operator,
)
from sflow.logging import get_logger

_logger = get_logger(__name__)
from sflow.core.backend_registry import register_backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.plugins.operators._k8s_render import (
    DEFAULT_GPU_TOLERATION,
    SFLOW_ALLOC_LABEL,
    render_compute_domain_manifest,
    render_reservation_pod_manifest,
    render_resource_claim_template,
)
from sflow.plugins.operators._k8s_shell import sanitize_name

_RESERVE_POLL_INTERVAL = 2
# How often (seconds) to emit a "still waiting" heartbeat while a reservation pod
# is unscheduled, in addition to logging immediately whenever the phase changes.
# Overridable at runtime via $SFLOW_K8S_WAIT_HEARTBEAT_SECS.
_RESERVE_HEARTBEAT_INTERVAL = 30

# How GPUs are requested for both placeholder and task pods.
#   dra           -> resource.k8s.io ResourceClaimTemplate (nvidia-dra-driver-gpu)
#   device_plugin -> legacy nvidia.com/gpu device-plugin limit
SchedulingMode = Literal["dra", "device_plugin"]


class KubernetesDraConfig(BaseModel):
    """DRA (Dynamic Resource Allocation) options for ``scheduling: dra``."""

    # DeviceClass GPUs are requested from (nvidia-dra-driver-gpu default).
    gpu_device_class: Resolvable[str] = "gpu.nvidia.com"
    # Optional CEL expressions narrowing eligible devices (e.g. by product/memory).
    device_selectors: list[Resolvable[str]] | None = None
    # Stand up an NVIDIA ComputeDomain so multi-node pods get a Multi-Node NVLink
    # (IMEX) channel. Only meaningful for multi-node GPU tasks on NVLink hardware.
    compute_domain: bool = False


class KubernetesVolumeConfig(BaseModel):
    """A pre-existing PersistentVolumeClaim (PVC) to mount into task pods.

    This is how cluster-resident data (e.g. a model staged on shared storage)
    reaches pods without a node-local hostPath: declare the PVC + where it mounts,
    then point an ``fs://`` artifact at a path under ``mount_path``. The operator
    mounts the PVC into the pod(s) of any task whose script references a path under
    ``mount_path`` (and skips the hostPath fallback for those paths).

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
        )
        # Populated in allocate(): maps a real k8s node name -> the placeholder
        # pod holding it (used for the create-before-destroy GPU handoff).
        self._node_to_resv_pod: dict[str, str] = {}
        # Populated in allocate() when dra.compute_domain: the channel
        # ResourceClaimTemplate name multi-node task pods claim (Multi-Node NVLink).
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
        self._compute_domain: bool = bool(dra.compute_domain) if dra is not None else False
        self._tolerations: list[dict[str, Any]] | None = config.tolerations
        self._volumes = list(config.volumes or [])
        # CLI-level kube access (set via apply_kubectl_config from `sflow run`
        # flags); prefixed onto every kubectl call sflow makes for this backend.
        self._kubeconfig: str | None = None
        self._kube_context: str | None = None
        self._kubectl_extra_args: list[str] = []
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
    def node_selector(self) -> dict[str, str] | None:
        return self._node_selector

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
            if self._compute_domain:
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
            return await self._allocate_reserved(alloc_id, pod_names, manifests)
        except BaseException:
            # BaseException (not just Exception) so a cancel (Ctrl+C) mid-allocation
            # still releases the reservation pods instead of leaking them.
            await self._release_alloc(alloc_id)
            self._pending_alloc_id = None
            raise

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

        # dra multi-node NVLink: stand up a ComputeDomain so task pods can claim
        # an IMEX channel keyed off this allocation.
        if self._scheduling == "dra" and self._compute_domain:
            await self._create_compute_domain(alloc_id, num_nodes=len(node_names))

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

    async def _create_compute_domain(self, alloc_id: str, *, num_nodes: int) -> None:
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
        await asyncio.gather(
            self._delete_by_alloc_label("pod", alloc_id),
            self._delete_by_alloc_label("configmap", alloc_id),
            self._delete_by_alloc_label("secret", alloc_id),
            self._delete_by_alloc_label("resourceclaimtemplate.resource.k8s.io", alloc_id),
            self._delete_by_alloc_label("computedomain.resource.nvidia.com", alloc_id),
            return_exceptions=True,
        )

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
        if self._compute_domain:
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
            details.append(("compute_domain", str(self._compute_domain)))
        details.append(("reservation_timeout", str(self._reservation_timeout)))
        if self._kubeconfig:
            details.append(("kubeconfig", self._kubeconfig))
        if self._kube_context:
            details.append(("context", self._kube_context))
        if self._kubectl_extra_args:
            details.append(("kubectl_args", str(list(self._kubectl_extra_args))))
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
        # Hardware monitoring must observe the physical node, but sflow has no
        # node-level collector mechanism on Kubernetes yet (a privileged DCGM /
        # nvidia-smi DaemonSet would be the proper path). As an interim, run the
        # collector on the sflow driver host via bash and warn that the metrics
        # reflect the driver, not the reserved k8s node(s).
        from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig

        _logger.warning(
            "Monitor '%s': backend '%s' is Kubernetes; hardware metrics are "
            "collected on the sflow driver host, not the reserved node(s). "
            "Node-level GPU monitoring on k8s (DCGM/DaemonSet) is not yet "
            "implemented.",
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
                compute_domain=bool(d.compute_domain),
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

        return KubernetesBackendConfig(
            name=conf.name,
            type="kubernetes",
            default=bool(getattr(conf, "default", False)),
            namespace=namespace,
            image_pull_policy=image_pull_policy,
            nodes=nodes_i,
            gpus_per_node=gpus_per_node,
            extra_args=extra_args,
            node_selector=conf.node_selector,
            host_network=bool(conf.host_network),
            scheduling=conf.scheduling,
            dra=dra,
            tolerations=conf.tolerations,
            reservation=reservation,
        )
