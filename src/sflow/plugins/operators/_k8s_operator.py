# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared base for the kubernetes container operator.

The single ``k8s`` operator (see ``k8s.py``) renders each task into its own
scheduler-placed pod(s): one pod for a single-node task, or N pods (one per
assigned node, leader = index 0) for a multi-node task. GPUs are requested via
DRA (``resource.k8s.io`` ResourceClaimTemplate) or the legacy ``nvidia.com/gpu``
device-plugin limit, selected by the backend ``scheduling`` field. The backend's
reserve+discover+pin context (namespace, assigned nodes, scheduling/DRA config,
node IPs, placeholder pods to hand off) is injected via ``apply_backend_context``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ConfigDict, field_validator

from sflow.core.command import Command
from sflow.core.log_offload import unsupported_offload_warning
from sflow.core.operator import Operator, OperatorConfig
from sflow.logging import get_logger
from sflow.plugins.operators._k8s_render import (
    DEFAULT_GPU_TOLERATION,
    SFLOW_ENTRYPOINT_FILE,
    render_configmap,
    render_resource_claim_template,
    render_task_pod,
)
from sflow.plugins.operators._k8s_shell import (
    build_k8s_task_command,
    configmap_data_key,
    namespace_segment,
    sanitize_name,
)
from sflow.utils.container import validate_container_image_reference

_logger = get_logger(__name__)


class K8sContainerOperatorConfig(OperatorConfig):
    """Base config for the kubernetes container operator.

    The concrete subclass sets the ``type`` literal. ``namespace`` is
    intentionally absent: it is backend-owned and injected at runtime (one
    namespace per backend), so ``extra="forbid"`` makes setting it on an operator
    an error.
    """

    model_config = ConfigDict(extra="forbid")

    name: str

    image: str
    image_pull_policy: str | None = None
    restart: str = "Never"
    pass_envs: bool = True
    # Use host networking for the pod (pod IP == node IP). None means inherit
    # from the kubernetes backend at runtime; True/False overrides explicitly.
    host_network: bool | None = None
    # Constrain the pod to nodes matching these labels. None means inherit the
    # backend's node_selector at runtime.
    node_selector: dict[str, str] | None = None
    # DRA overrides (None -> inherit the backend's dra config). The DeviceClass
    # GPUs are requested from, and optional CEL selectors narrowing eligible
    # devices. Ignored under ``scheduling: device_plugin``.
    device_class: str | None = None
    device_selectors: list[str] | None = None
    # Pod tolerations. None -> inherit the backend's tolerations (default:
    # tolerate ``nvidia.com/gpu`` so pods can land on tainted GPU nodes).
    tolerations: list[dict[str, Any]] | None = None

    def container_images(self) -> list[str]:
        return [self.image] if self.image else []

    @field_validator("image")
    @classmethod
    def image_must_be_valid(cls, value: str) -> str:
        type_field = cls.model_fields.get("type")
        type_name = (
            type_field.default
            if type_field is not None and isinstance(type_field.default, str)
            else "kubernetes"
        )
        validate_container_image_reference(
            value,
            source=f"{type_name} operator config: 'image'",
        )
        return value

    def runtime_warnings(self) -> list[str]:
        # Per-task log offload is not implemented for the kubernetes operator yet
        # (logs stream through the sflow driver via `kubectl logs`); surface that
        # at dry-run instead of silently ignoring an offload request.
        return unsupported_offload_warning(self.type)


class K8sContainerOperator(Operator):
    """Render a task into pinned, scheduler-placed pod(s) and ``kubectl apply`` them."""

    def __init__(self, config: K8sContainerOperatorConfig):
        super().__init__(config)
        self.config: K8sContainerOperatorConfig = config
        self._image: str = config.image
        # Backend-injected context (see apply_backend_context).
        self._namespace: str | None = None
        self._node_count: int = 1
        self._assigned_node_names: list[str] = []
        self._assigned_node_ips: list[str] = []
        self._node_placement: bool = False
        self._scheduling: str = "dra"
        self._gpu_device_class: str = "gpu.nvidia.com"
        self._device_selectors: list[str] | None = config.device_selectors
        # Per-pod GPU count (resources.gpus.count // node_count); 0 == no GPUs.
        self._per_pod_gpus: int = 0
        self._host_network: bool = (
            bool(config.host_network) if config.host_network is not None else False
        )
        self._node_selector: dict[str, str] | None = config.node_selector
        self._tolerations: list[dict[str, Any]] | None = config.tolerations
        # Placeholder pods to delete on the create-before-destroy handoff. Only
        # populated for GPU tasks so CPU-only tasks coexist with the placeholder.
        self._handoff_pods: list[str] = []
        # ComputeDomain channel ResourceClaimTemplate (dra multi-node NVLink).
        self._compute_domain_channel: str | None = None
        # CLI-level kube access flags (from `sflow run`), prefixed onto every
        # kubectl call in the per-task wrapper; read from the backend below.
        self._kubectl_global_args: list[str] = []
        # Backend allocation id, stamped onto every task object as the allocation
        # label so the backend's label-selector sweep can delete them all.
        self._allocation_id: str | None = None
        # Resolved workflow artifacts (injected via apply_backend_context). Used to
        # mount file:// inline content (ConfigMap) and fs:// paths (hostPath) into
        # each task's pod(s) the K8s-native way.
        self._artifacts: list[Any] = []
        # Pre-existing PVC mounts declared on the backend (shared storage that fs://
        # artifacts can live on); injected via apply_backend_context.
        self._pvc_volumes: list[dict[str, Any]] = []

    def apply_backend_context(
        self,
        *,
        backend: Any,
        assigned_nodes: Sequence[str],
        artifacts: Sequence[Any],
        cuda_visible_devices: str | None = None,
        gpu_count: int | None = None,
    ) -> None:
        self._namespace = getattr(backend, "namespace", None)
        self._node_count = max(len(assigned_nodes), 1)
        self._assigned_node_names = list(assigned_nodes or [])
        self._artifacts = list(artifacts or [])
        self._pvc_volumes = list(getattr(backend, "volumes", None) or [])
        self._scheduling = str(getattr(backend, "scheduling", "dra"))
        self._gpu_device_class = self.config.device_class or str(
            getattr(backend, "gpu_device_class", "gpu.nvidia.com")
        )
        if self.config.device_selectors is not None:
            self._device_selectors = self.config.device_selectors
        else:
            self._device_selectors = getattr(backend, "device_selectors", None)

        # Explicit operator value wins; None inherits from the backend.
        if self.config.host_network is not None:
            self._host_network = self.config.host_network
        else:
            self._host_network = bool(getattr(backend, "host_network", False))
        if self.config.node_selector is not None:
            self._node_selector = self.config.node_selector
        else:
            self._node_selector = getattr(backend, "node_selector", None)
        if self.config.tolerations is not None:
            self._tolerations = self.config.tolerations
        else:
            self._tolerations = getattr(backend, "tolerations", None)

        self._node_placement = bool(
            getattr(
                getattr(backend, "capabilities", None),
                "supports_node_placement",
                False,
            )
        )

        # Per-pod GPU count: the planner's resources.gpus.count is a per-task
        # total; split it evenly across the assigned nodes (one pod per node).
        total_gpus = int(gpu_count) if gpu_count else 0
        if total_gpus and total_gpus % self._node_count != 0:
            raise ValueError(
                f"k8s operator '{self.config.name}': resources.gpus.count="
                f"{total_gpus} is not divisible by the {self._node_count} assigned "
                "node(s); request a multiple of the node count (each node's pod "
                "gets count/nodes GPUs, bounded by the backend's gpus_per_node)."
            )
        self._per_pod_gpus = (total_gpus // self._node_count) if total_gpus else 0

        # Real node IPs (for multi-node leader/peer env wiring), discovered at
        # allocation time and carried on the backend allocation.
        self._assigned_node_ips = []
        alloc = getattr(backend, "allocation", None)
        self._allocation_id = getattr(alloc, "allocation_id", None) if alloc else None
        if alloc is not None:
            by_name = {n.name: n.ip_address for n in alloc.nodes}
            self._assigned_node_ips = [
                by_name.get(name, "") for name in self._assigned_node_names
            ]

        # Create-before-destroy handoff: the assigned node(s)' placeholder pods to
        # delete AFTER applying the (Pending) task pod. GPU tasks only -- CPU-only
        # tasks keep the placeholder (and the node's GPUs) reserved for the GPU
        # workloads that overlap on the same node.
        self._handoff_pods = []
        if self._node_placement and self._per_pod_gpus > 0:
            resolver = getattr(backend, "reservation_pod_for_node", None)
            if callable(resolver):
                for node_name in self._assigned_node_names:
                    pod = resolver(node_name)
                    if pod:
                        self._handoff_pods.append(pod)

        # dra multi-node NVLink/IMEX: pods claim a ComputeDomain channel.
        self._compute_domain_channel = (
            getattr(backend, "compute_domain_channel", None)
            if self._scheduling == "dra"
            else None
        )

        # CLI-level kube access flags, applied to every kubectl call in the wrapper.
        self._kubectl_global_args = list(
            getattr(backend, "kubectl_global_args", []) or []
        )

    def _effective_tolerations(self) -> list[dict[str, Any]]:
        if self._tolerations is not None:
            return [dict(t) for t in self._tolerations]
        return [dict(DEFAULT_GPU_TOLERATION)]

    def _pin_node(self, index: int) -> str | None:
        """Hostname to pin pod ``index`` onto, or None when placement is off."""
        if self._node_placement and index < len(self._assigned_node_names):
            return self._assigned_node_names[index]
        return None

    def _covering_pvc(self, path_str: str) -> dict[str, Any] | None:
        """The configured PVC whose mount_path is ``path_str`` or a parent of it."""
        p = Path(path_str)
        for vol in self._pvc_volumes:
            mount_path = vol.get("mount_path")
            if not mount_path:
                continue
            mp = Path(str(mount_path))
            if p == mp or mp in p.parents:
                return vol
        return None

    @staticmethod
    def _host_path_type(path_str: str) -> str:
        """Best-effort hostPath ``type`` for an fs:// path, via a controller stat.

        If the controller can see the path (typical when it lives on a shared
        filesystem mounted on both controller and nodes), pin the type so the
        kubelet rejects the pod with a clear error when a node lacks it -- instead
        of silently creating an empty dir. When the controller can't see it (a
        node-only path, or an output dir created at runtime), return "" so the
        kubelet stays lenient.
        """
        try:
            p = Path(path_str)
            if p.is_dir():
                return "Directory"
            if p.is_file():
                return "File"
        except OSError:
            pass
        return ""

    def _artifact_injection(
        self, script: Sequence[str]
    ) -> tuple[
        dict[str, str],
        list[tuple[str, str]],
        list[tuple[str, str]],
        list[dict[str, Any]],
    ]:
        """K8s-native artifact wiring for a task's pod(s), from resolved artifacts.

        Returns ``(configmap_data, file_mounts, host_path_mounts, pvc_mounts)``:

        * ``configmap_data`` -- ``{key: inline_content}`` for ``file://`` artifacts
          declared with inline content; these become one ConfigMap mounted into the
          pod (so the content lives in the cluster, not on the controller's disk).
        * ``file_mounts`` -- ``[(in_pod_path, key)]`` subPath mounts for the above,
          placing each file at its resolved ``${{ artifacts.NAME.path }}`` location.
        * ``host_path_mounts`` -- ``[(node_path, hostpath_type)]`` for ``fs://``
          (and non-inline ``file://``) artifacts NOT served by a PVC, hostPath-
          mounted at the same path so a shared / node-local location is visible.
        * ``pvc_mounts`` -- pre-existing PVC volumes (deduped) to mount because a
          referenced ``fs://`` artifact path lives under their ``mount_path``.

        Only artifacts the task's (already-resolved) script references are injected,
        so e.g. the model ``fs://`` path is not mounted into CPU-only infra pods.
        """
        joined = "\n".join(script)
        cm_data: dict[str, str] = {}
        file_mounts: list[tuple[str, str]] = []
        host_path_mounts: list[tuple[str, str]] = []
        pvc_by_name: dict[str, dict[str, Any]] = {}
        seen_paths: set[str] = set()
        for art in self._artifacts:
            uri = str(getattr(art, "uri", "") or "")
            scheme = urlparse(uri).scheme.lower()
            if scheme not in ("file", "fs"):
                continue
            path = getattr(art, "path", None)
            if path is None:
                continue
            path_str = str(path)
            name = str(getattr(art, "name", "") or "")
            # Inject only what the task uses: its resolved path or ${NAME} env ref.
            if path_str not in joined and not (name and name in joined):
                continue
            content = getattr(art, "content", None)
            if scheme == "file" and content is not None:
                key = configmap_data_key(name)
                cm_data[key] = content
                file_mounts.append((path_str, key))
                continue
            # A path served by a declared PVC: mount the PVC (once), not a hostPath.
            pvc = self._covering_pvc(path_str)
            if pvc is not None:
                vol_name = sanitize_name(str(pvc["name"]))
                pvc_by_name.setdefault(vol_name, {**pvc, "name": vol_name})
            elif path_str not in seen_paths:
                seen_paths.add(path_str)
                host_path_mounts.append((path_str, self._host_path_type(path_str)))
        return cm_data, file_mounts, host_path_mounts, list(pvc_by_name.values())

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        if not self._image:
            raise ValueError(
                f"k8s operator '{self.config.name}' has no image configured; set "
                "'image' on the operator (the kubernetes backend has no image)."
            )
        c = self.config
        base = sanitize_name(task_name)
        n = self._node_count
        pod_names = [base] if n == 1 else [f"{base}-{i}" for i in range(n)]
        configmap_name = sanitize_name(f"{base}-cfg")
        secret_name = sanitize_name(f"{base}-env")
        use_secret = bool(c.pass_envs and envs)
        rct_name = (
            sanitize_name(f"{base}-gpu")
            if self._scheduling == "dra" and self._per_pod_gpus > 0
            else None
        )
        tolerations = self._effective_tolerations()

        # K8s-native artifact injection (file:// -> ConfigMap, fs:// -> PVC/hostPath).
        cm_data, file_mounts, host_path_mounts, pvc_mounts = self._artifact_injection(
            script
        )
        artifacts_cm_name = sanitize_name(f"{base}-artifacts") if cm_data else None

        items: list[dict[str, Any]] = [
            render_configmap(
                name=configmap_name,
                namespace=self._namespace,
                data={SFLOW_ENTRYPOINT_FILE: "\n".join(script)},
                task_label=base,
                allocation_id=self._allocation_id,
            )
        ]
        if artifacts_cm_name is not None:
            items.append(
                render_configmap(
                    name=artifacts_cm_name,
                    namespace=self._namespace,
                    data=cm_data,
                    task_label=base,
                    allocation_id=self._allocation_id,
                )
            )
        if rct_name is not None:
            items.append(
                render_resource_claim_template(
                    name=rct_name,
                    namespace=self._namespace,
                    device_class=self._gpu_device_class,
                    count=self._per_pod_gpus,
                    selectors=self._device_selectors,
                    task_label=base,
                    allocation_id=self._allocation_id,
                )
            )
        for i, pod_name in enumerate(pod_names):
            extra_env: dict[str, str] = {}
            if n > 1:
                extra_env["SFLOW_TASK_NODE_INDEX"] = str(i)
                if self._assigned_node_ips:
                    extra_env["SFLOW_LEADER_ADDRESS"] = self._assigned_node_ips[0]
            items.append(
                render_task_pod(
                    pod_name=pod_name,
                    image=self._image,
                    configmap_name=configmap_name,
                    namespace=self._namespace,
                    image_pull_policy=c.image_pull_policy,
                    restart_policy=c.restart,
                    env_secret_name=secret_name if use_secret else None,
                    scheduling=self._scheduling,
                    per_pod_gpus=self._per_pod_gpus,
                    resource_claim_name=rct_name,
                    host_network=self._host_network,
                    node_selector=self._node_selector,
                    assigned_node=self._pin_node(i),
                    tolerations=tolerations,
                    extra_env=extra_env or None,
                    compute_domain_channel=self._compute_domain_channel,
                    task_label=base,
                    allocation_id=self._allocation_id,
                    artifacts_configmap_name=artifacts_cm_name,
                    file_artifact_mounts=file_mounts,
                    host_path_mounts=host_path_mounts,
                    pvc_mounts=pvc_mounts,
                )
            )

        manifest = {"apiVersion": "v1", "kind": "List", "items": items}
        return build_k8s_task_command(
            manifest_json=json.dumps(manifest, separators=(",", ":")),
            ns_seg=namespace_segment(self._namespace),
            pod_names=pod_names,
            configmap_name=configmap_name,
            rct_name=rct_name,
            secret_name=secret_name if use_secret else None,
            envs=envs,
            handoff_delete_pods=self._handoff_pods,
            parallel_logs=n > 1,
            kubectl_global_args=self._kubectl_global_args,
            allocation_id=self._allocation_id,
            artifacts_configmap_name=artifacts_cm_name,
        )
