# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified multi-node MPI operator for Kubernetes (``type: k8s_mpi``).

One operator, two interchangeable execution routes -- so a single recipe (a plain
``mpirun -np N ... <workload>`` with no SSH glue) runs on any cluster:

* **operator** -- emit a Kubeflow ``MPIJob`` CR; the mpi-operator owns the SSH
  keypair / hostfile / ``sshd`` / wait-for-workers. Requires the
  ``mpijobs.kubeflow.org`` CRD + controller.
* **pods** -- N plain pods (the base ``k8s`` render) with sflow's MPI bootstrap
  preamble + transparent ``mpirun`` wrapper injected (see ``k8s.mpi_bootstrap``).
* **auto** (default) -- use the operator when its CRD is present, else fall back
  to pods.

All of the SSH/hostfile/``sshd``/keypair glue is driver-owned; the recipe keeps
only its app-specific ``mpirun`` launch line. See
``docs/superpowers/plans/k8s_mpi.md``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sflow.core.operator_registry import register_operator
from sflow.logging import get_logger
from sflow.plugins.k8s import lifecycle as k8s_lifecycle
from sflow.plugins.k8s import mpi_lifecycle as mpi_lifecycle
from sflow.plugins.k8s.mpi_bootstrap import (
    DEFAULT_SSH_AUTH_MOUNT_PATH,
    SSH_PRIVATE_KEY_ENV,
    SSH_PUBLIC_KEY_ENV,
    build_launcher_preamble,
    build_mpi_bootstrap_preamble,
    build_mpijob_manifest,
    build_worker_command,
    final_launch_index,
    inject_mpi_scheduling,
    strip_final_launch,
)
from sflow.plugins.k8s.capabilities import MPI_OPERATOR, CapabilityState
from sflow.plugins.operators.k8s_operator import (
    K8sContainerOperator,
    K8sContainerOperatorConfig,
    _K8sExecPlan,
    _sflow_pod_mkdir_preamble,
    _sflow_pod_output_dir,
)
from sflow.plugins.k8s.rdma_preamble import build_rdma_affinity_preamble
from sflow.plugins.k8s.render import (
    SFLOW_ENTRYPOINT_FILE,
    render_configmap,
    render_resource_claim_template,
    render_task_pod,
)
from sflow.plugins.k8s.shell import (
    build_log_stream_command,
    build_manifest_apply_command,
    namespace_segment,
    sanitize_name,
)

_logger = get_logger(__name__)


class MpiConfig(BaseModel):
    """The ``mpi:`` block on a ``k8s_mpi`` operator (route + MPIJob knobs)."""

    model_config = ConfigDict(extra="forbid")

    # Execution route: auto-detect the mpi-operator CRD (default), or force one.
    route: Literal["auto", "operator", "pods"] = "auto"
    # Hostfile slots per node / MPIJob slotsPerWorker. None -> the per-node GPU
    # count (resources.gpus.count / nodes). Accepts a ``${{ }}`` expression.
    slots_per_worker: int | str | None = None
    # runLauncherAsWorker: the launcher node also runs ranks (the HTTP server).
    run_launcher_as_worker: bool = True
    # MPIJob launcherCreationPolicy: wait for worker sshd before the launcher.
    launcher_creation_policy: Literal["AtStartup", "WaitForWorkersReady"] = (
        "WaitForWorkersReady"
    )
    # sshd port (avoids the node's :22 under host_network). Matches the
    # mpi-operator's own default so the operator route needs no extra ssh config.
    ssh_port: int | str = 2222
    # MPIJob mpiImplementation.
    mpi_implementation: Literal["OpenMPI", "Intel", "MPICH"] = "OpenMPI"
    # Install openssh-server at startup if the image lacks sshd (both routes).
    ensure_sshd: bool = True
    # ADDITIVE, app-specific env-namespace prefixes to forward to remote ranks
    # (e.g. TRTLLM_, TLLM_, FLASHINFER_). sflow ALWAYS forwards its built-in
    # transport/system set (NCCL_/UCX_/GLOO_/NVSHMEM_/OMPI_MCA_/SFLOW_ + PATH/
    # LD_LIBRARY_PATH), AND auto-forwards every var the recipe itself ``export``s
    # (diffed against a pre-recipe snapshot), so a plain ``export FOO=bar`` in the
    # script reaches workers without listing it here. Use this only for vars that
    # come from the POD env (declared task env / extra_env) -- present before the
    # recipe, so not caught by the export diff -- but still needed on remote ranks.
    # Default: none.
    forward_env_prefixes: list[str] = Field(default_factory=list)
    # OMP_NUM_THREADS to set for every rank. Injected as pod env AND always forwarded to
    # remote ranks (see _BUILTIN_FORWARD_VARS). When several MPI ranks share a node, each
    # rank's libgomp otherwise defaults to the node's full CPU count and their combined
    # thread pools exhaust the process/pthread limit ("libgomp: Thread creation failed:
    # Resource temporarily unavailable") during model weight load. Default 8 is a safe cap
    # for GPU-bound serving; raise it for CPU-bound phases (roughly CPUs-per-node divided by
    # ranks-per-node). Set to null/0 to inject nothing and leave the image default.
    omp_num_threads: int | None = 8
    # CPU binding for each MPI rank, injected into the mpirun wrapper (both routes) ONLY
    # when several ranks share a pod (slots_per_worker > 1); a single-rank pod is untouched.
    # OMP_NUM_THREADS caps libgomp but NOT the LLVM/MLIR ThreadPool that CuteDSL/TRT-LLM
    # autotuning JIT spins up, which sizes to the rank's CPU affinity mask -- so an unbound
    # rank reads the whole node's cores and, with several ranks per pod, blows past the
    # cgroup pid limit ("pthread_create failed: Resource temporarily unavailable"). Binding
    # shrinks that mask. "core" (default): PE = nproc/ranks cores per rank (nproc read from
    # the pod's live cpuset at launch), giving each rank an isolated core slice and the
    # tightest thread cap -- the reliable choice when a GPU pod's cpuset lands within a
    # single NUMA node (typical on K8s GB200/GB300). "numa": one rank per NUMA domain
    # (adds NUMA-local memory) -- only partitions ranks when the cpuset spans >1 NUMA node.
    # "none": inject nothing (pre-feature behaviour). A recipe that already passes
    # --bind-to/--map-by keeps its own binding (sflow injects nothing).
    cpu_bind: Literal["core", "numa", "none"] = "core"
    # How long a worker's per-node setup (image apt-install + weight staging) may take
    # before its readiness probe reaps it (operator route). The worker's readiness probe
    # (tcpSocket on the ssh port) is what launcherCreationPolicy=WaitForWorkersReady gates
    # on, so this is effectively the setup budget. Rendered as a probe with a fixed 5s
    # poll and failureThreshold = ceil(timeout / 5). Default ~15 min -- generous for
    # typical setup (weights on a PVC, short warmup) while still detecting a stuck worker
    # reasonably fast; RAISE it for a large first-time weight download over slow storage.
    # Accepts a ``${{ }}`` expression.
    worker_setup_timeout_seconds: int | str = 900
    # Seconds to wait for the mpi-operator controller to create the launcher pod after
    # the MPIJob CR is applied (operator route). Worker sshd startup + WaitForWorkersReady
    # can add latency, and a slow per-node setup (see worker_readiness_* above) pushes
    # launcher creation out further, so raise this in lockstep when the setup budget grows.
    # Default matches mpi_lifecycle._LAUNCHER_DISCOVERY_TIMEOUT. Accepts a ``${{ }}`` expression.
    launcher_discovery_timeout: int | str = 600


class K8sMpiOperatorConfig(K8sContainerOperatorConfig):
    type: Literal["k8s_mpi"] = "k8s_mpi"
    mpi: MpiConfig = Field(default_factory=MpiConfig)


def _generate_ssh_keypair_b64() -> tuple[str, str]:
    """Generate an ephemeral RSA keypair; return ``(private_b64, public_b64)``.

    Driver-side (pods route): the keypair is injected into every server pod via
    the task env Secret so all pods trust the same per-run key with no shared PVC
    and no in-pod keygen/exchange. Uses ``ssh-keygen`` (near-universal where
    kubectl runs); raises a clear error if it is unavailable.
    """
    with tempfile.TemporaryDirectory() as tmp:
        key_path = os.path.join(tmp, "id_rsa")
        try:
            subprocess.run(
                ["ssh-keygen", "-t", "rsa", "-b", "2048", "-N", "", "-f", key_path, "-q"],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "k8s_mpi pods route needs 'ssh-keygen' on the sflow driver host to "
                "generate the per-run MPI SSH keypair. Install openssh-client, or "
                "use the operator route (mpi.route: operator) where the mpi-operator "
                "owns the keys."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"k8s_mpi: ssh-keygen failed to generate the MPI keypair: "
                f"{exc.stderr.decode(errors='replace') if exc.stderr else exc}"
            ) from exc
        with open(key_path, "rb") as fh:
            priv = base64.b64encode(fh.read()).decode("ascii")
        with open(key_path + ".pub", "rb") as fh:
            pub = base64.b64encode(fh.read()).decode("ascii")
    return priv, pub


@register_operator("k8s_mpi", K8sMpiOperatorConfig)
class K8sMpiOperator(K8sContainerOperator):
    """Multi-node MPI on Kubernetes via the mpi-operator (MPIJob) or plain pods.

    Reuses the base ``k8s`` operator's backend context, pod render, reservation
    handoff, and (pods route) lifecycle; adds route resolution, the MPI bootstrap
    preamble + ``mpirun`` wrapper (pods route), and the MPIJob CR render + watch
    (operator route). ``manages_own_execution`` stays True (inherited).

    Worker parity: the recipe's setup (everything except the final ``mpirun``) runs
    on EVERY node so filesystem side-effects match the launcher, while env reaches
    remote ranks via the wrapper's ``mpirun -x`` auto-forward. An ordering barrier
    (a worker's ``sshd`` comes up only after its setup; the leader / mpi-operator
    waits for it) ensures no rank launches before that worker's setup finished.
    """

    config: K8sMpiOperatorConfig

    def __init__(self, config: K8sMpiOperatorConfig):
        super().__init__(config)
        self.config = config
        # mpi-operator capability state, injected from the backend at preflight
        # (UNKNOWN in dry-run -> route:auto uses pods). route:operator keys off
        # INSTALLED (its "not installed" error fires only on ABSENT); route:auto
        # keys off USABLE (installed AND the creds can drive MPIJobs here).
        self._mpi_state: CapabilityState = CapabilityState.UNKNOWN
        # Per-run ephemeral SSH keypair (pods route), generated once and shared by
        # every server pod via the env Secret. Cached so build_command + execute
        # (which both build the plan) agree on the same key.
        self._mpi_ssh_keypair: tuple[str, str] | None = None

    def apply_backend_context(self, *, backend: Any, **kwargs: Any) -> None:
        super().apply_backend_context(backend=backend, **kwargs)
        cap_state = getattr(backend, "capability_state", None)
        if callable(cap_state):
            self._mpi_state = cap_state(MPI_OPERATOR)

    # ------------------------------------------------------------------
    # Route + config helpers
    # ------------------------------------------------------------------
    def _resolved_route(self) -> str:
        """Resolve ``mpi.route`` (auto/operator/pods) against cluster detection.

        ``operator`` requires the operator INSTALLED (error if detected absent), and
        preflight hard-checks its MPIJob RBAC. ``auto`` uses the operator only when
        it is USABLE (installed AND the creds can drive MPIJobs in this namespace),
        else the pods route -- so an installed-but-not-permitted operator falls back
        gracefully instead of a hard RBAC failure. Both default to pods when
        detection has not run (e.g. dry-run).
        """
        route = self.config.mpi.route
        if route == "pods":
            return "pods"
        if route == "operator":
            if self._mpi_state == CapabilityState.ABSENT:
                raise ValueError(
                    f"k8s_mpi operator '{self.config.name}': mpi.route=operator but "
                    "the Kubeflow MPI Operator CRD (mpijobs.kubeflow.org) was not "
                    "found on the cluster. Install the mpi-operator, or use "
                    "mpi.route: pods (or auto)."
                )
            return "operator"
        # auto: use the operator only if it is USABLE (installed + creds can drive
        # MPIJobs here); an installed-but-not-permitted operator -> pods fallback.
        return "operator" if self._mpi_state.usable else "pods"

    def _mpi_slots_per_worker(self) -> int:
        """Hostfile slots / MPIJob slotsPerWorker: config value, else per-node GPUs."""
        raw = self.config.mpi.slots_per_worker
        if raw is not None and str(raw).strip() != "":
            try:
                return max(int(raw), 1)
            except (TypeError, ValueError):
                pass
        return max(self._per_pod_gpus, 1)

    def _ssh_port(self) -> int:
        try:
            return int(self.config.mpi.ssh_port)
        except (TypeError, ValueError):
            return 2222

    def _worker_readiness_period_seconds(self) -> int:
        # Fixed poll granularity; the tunable setup budget is worker_setup_timeout_seconds.
        return 5

    def _worker_readiness_failure_threshold(self) -> int:
        # Kill window = failureThreshold x periodSeconds; derive threshold from the single
        # worker_setup_timeout_seconds knob (ceil division) so the two never drift.
        period = self._worker_readiness_period_seconds()
        try:
            timeout = int(self.config.mpi.worker_setup_timeout_seconds)
        except (TypeError, ValueError):
            timeout = 900
        return max(-(-timeout // period), 1)  # ceil(timeout / period), at least 1

    def _launcher_discovery_timeout(self) -> float:
        try:
            return max(float(self.config.mpi.launcher_discovery_timeout), 0.0)
        except (TypeError, ValueError):
            return mpi_lifecycle._LAUNCHER_DISCOVERY_TIMEOUT

    def _forward_prefixes(self) -> list[str]:
        return [str(p) for p in (self.config.mpi.forward_env_prefixes or [])]

    def _extra_pod_env_defaults(self) -> dict[str, str]:
        """Operator-set inline pod env (static; and, for MPI, forwarded to remote ranks via
        ``_BUILTIN_FORWARD_VARS``). Overrides the base to cap per-rank OpenMP: with several
        MPI ranks/node, each rank's libgomp otherwise defaults to the node's full CPU count
        and their combined pools exhaust the pthread/process limit ("Thread creation failed")
        at model load. A recipe ``export OMP_NUM_THREADS=...`` still overrides it at runtime.
        Set ``mpi.omp_num_threads: null`` (or 0) to leave the image default."""
        base = dict(super()._extra_pod_env_defaults())
        n = self.config.mpi.omp_num_threads
        if n:
            base.setdefault("OMP_NUM_THREADS", str(n))
        return base

    def _ssh_keypair(self) -> tuple[str, str]:
        if self._mpi_ssh_keypair is None:
            self._mpi_ssh_keypair = _generate_ssh_keypair_b64()
        return self._mpi_ssh_keypair

    def _inject_pods_keypair_env(self, env: Mapping[str, str]) -> dict[str, str]:
        """Add the shared ephemeral SSH keypair to the task env (pods route only).

        Delivered via the task ENV -- not via ``_build_execution_plan`` -- because
        the env-Secret's values are read from the apply command's OWN process env
        at runtime (``secret_printf_lines`` emits ``printf ... "${KEY-}"``). The
        apply command runs with this env, so the real key reaches the Secret AND
        the Secret's key list (built from the same env) includes it. Multi-node
        pods route only: single-node needs no SSH, and the operator route lets the
        mpi-operator own the keys. Without this the Secret value is empty and the
        bootstrap preamble aborts with "no injected SSH keypair" (exit 1).
        """
        out = dict(env)
        if self._resolved_route() == "pods" and self._node_count > 1:
            priv, pub = self._ssh_keypair()
            out[SSH_PRIVATE_KEY_ENV] = priv
            out[SSH_PUBLIC_KEY_ENV] = pub
        return out

    # ------------------------------------------------------------------
    # Pods route: reuse the base N-pod build + inject the bootstrap
    # ------------------------------------------------------------------
    def _extra_entrypoint_preamble(self) -> list[str]:
        """Inject the MPI bootstrap into the pods-route entrypoint (base hook)."""
        if self._resolved_route() != "pods":
            return []
        return build_mpi_bootstrap_preamble(
            ssh_port=self._ssh_port(),
            gpus_per_node=max(self._per_pod_gpus, 1),
            ensure_sshd=self.config.mpi.ensure_sshd,
            forward_prefixes=self._forward_prefixes(),
            cpu_bind=self.config.mpi.cpu_bind,
        )

    @staticmethod
    def _auto_exec_launch(script: Sequence[str]) -> list[str]:
        """Prepend ``exec`` to the recipe's final ``mpirun`` launch line.

        ``exec`` makes ``mpirun`` replace the entrypoint shell as the container's
        main process, so Kubernetes' SIGTERM on pod deletion reaches it directly
        (graceful shutdown; no lingering parent bash that swallows signals). That
        is sflow's concern, not the user's -- the recipe writes a plain
        ``mpirun -np N ... <workload>`` and sflow adds ``exec`` here.

        Idempotent and conservative: the launch is located by
        :func:`final_launch_index` (only the last real command, and only when it is an
        ``mpirun`` launch). A line already starting with ``exec`` is left as-is, so
        existing ``exec mpirun`` recipes keep working, and a non-``mpirun`` final
        command (e.g. a user wrapper or post-launch step) is never touched.
        """
        lines = list(script)
        idx = final_launch_index(lines)
        if idx is None:
            return lines
        stripped = lines[idx].lstrip()
        if not stripped.startswith("exec "):
            indent = lines[idx][: len(lines[idx]) - len(stripped)]
            lines[idx] = f"{indent}exec {stripped}"
        return lines

    def _build_execution_plan(
        self, *, task_name: str, script: Sequence[str], envs: Mapping[str, str]
    ) -> _K8sExecPlan:
        # Auto-`exec` the mpirun launch so the recipe writes a plain `mpirun ...`;
        # applied once here so BOTH routes (operator + pods) inherit it.
        script = self._auto_exec_launch(script)
        if self._resolved_route() == "operator":
            return self._build_mpijob_execution_plan(
                task_name=task_name, script=script, envs=envs
            )
        # Pods route: the per-role split (worker brings up sshd + idles; leader waits
        # for workers then launches) lives INSIDE the mpirun shim (see
        # build_mpi_bootstrap_preamble -> _role_barrier_block), so it engages at the real
        # mpirun call regardless of how the recipe writes its launch line -- no source
        # gate to insert here. The recipe setup still runs on every pod (parity).
        # Pods route: reuse the base N-pod build. The ephemeral SSH keypair is added
        # to the task env by execute() (via _inject_pods_keypair_env), NOT here: the
        # env-Secret's values are read from the apply command's OWN process env at
        # runtime, which build_command()/dry-run has no way to supply. When execute()
        # provides it, the keypair is already in ``envs`` here and flows into the
        # Secret automatically.
        plan = super()._build_execution_plan(
            task_name=task_name, script=script, envs=envs
        )
        # Multi-node: the pods are one MPI world group, so resolve the moment ANY rank
        # pod goes terminal (a finished/dead rank breaks the group) instead of hanging
        # on the survivors -- an idle worker, or a leader still up after a worker died.
        if self._node_count > 1:
            plan.mpi_world_group = True
        return plan

    # ------------------------------------------------------------------
    # Operator route: render an MPIJob CR
    # ------------------------------------------------------------------
    def _build_mpijob_execution_plan(
        self, *, task_name: str, script: Sequence[str], envs: Mapping[str, str]
    ) -> _K8sExecPlan:
        if not self._image:
            raise ValueError(
                f"k8s_mpi operator '{self.config.name}' has no image configured; "
                "set 'image' on the operator."
            )
        c = self.config
        # Allocation-scoped so parallel runs in one namespace never collide (the
        # MPIJob/pod names and cfg/env/gpu/artifacts objects derive from `base`).
        base = self._scoped_base(task_name)
        job_name = base
        n = self._node_count
        slots = self._mpi_slots_per_worker()
        server_hostnames = list(self._assigned_node_names)
        run_launcher_as_worker = bool(c.mpi.run_launcher_as_worker)
        worker_replicas = max(n - 1, 0) if run_launcher_as_worker else n

        configmap_name = sanitize_name(f"{base}-cfg")
        secret_name = sanitize_name(f"{base}-env")
        use_secret = bool(c.pass_envs and envs)
        rct_name = (
            sanitize_name(f"{base}-gpu")
            if self._scheduling == "dra" and self._per_pod_gpus > 0
            else None
        )
        tolerations = self._effective_tolerations()

        # A launcher/worker pod owns its whole node (one per node), so expose every
        # node NIC and let NCCL/UCX pick -- mirrors the merged / full-node path.
        rdma_nic_resources, rdma_hcas = self._rdma_all_nics()
        dra_coalloc = bool(
            self._scheduling == "dra"
            and self._dra_rdma_device_class
            and self._per_pod_gpus > 0
        )
        runtime_affinity = (bool(rdma_hcas) and self._rdma_runtime_affinity) or dra_coalloc

        # Launcher entrypoint: gpu-driver + [gib|rdma] + launcher preamble (mpirun
        # wrapper for -x forwarding) + the user's mpirun script. The mpi-operator
        # injects the hostfile + SSH transport, so the wrapper adds only -x here.
        launcher_script: list[str] = build_launcher_preamble(
            forward_prefixes=self._forward_prefixes(),
            cpu_bind=c.mpi.cpu_bind,
            slots=slots,
        ) + list(script)
        rdma_lib_mounts: list[tuple[str, str]] = []
        if rdma_hcas and self._rdma_lib_mounts:
            rdma_lib_mounts = self._rdma_lib_mounts
            launcher_script = self._gib_preamble() + launcher_script
        elif runtime_affinity:
            launcher_script = (
                build_rdma_affinity_preamble(
                    self._network_env.get("SFLOW_PRIMARY_IFACE", "")
                )
                + launcher_script
            )
        launcher_script = self._gpu_driver_preamble(self._per_pod_gpus) + launcher_script
        # K8s: a writable emptyDir is mounted at SFLOW_OUTPUT_DIR (below) so the
        # driver-host SFLOW_* paths are valid + writable in the pod (env unchanged, so
        # mpirun -x forwards them to workers, which get the same mount). Just create the
        # per-task dir before the launch.
        if use_secret:
            launcher_script = _sflow_pod_mkdir_preamble(envs) + launcher_script

        cm_data, file_mounts, host_path_mounts, pvc_mounts = self._artifact_injection()
        artifacts_cm_name = sanitize_name(f"{base}-artifacts") if cm_data else None

        items: list[dict[str, Any]] = [
            render_configmap(
                name=configmap_name,
                namespace=self._namespace,
                data={SFLOW_ENTRYPOINT_FILE: "\n".join(launcher_script)},
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
                    nic_device_class=(
                        self._dra_rdma_device_class if dra_coalloc else None
                    ),
                    nic_count=self._per_pod_gpus if dra_coalloc else None,
                    match_attribute=self._dra_rdma_match_attribute,
                )
            )

        extra_env: dict[str, str] = dict(self._network_env)
        cd_channel = self._compute_domain_channel if self._per_pod_gpus > 0 else None
        extra_env.update(self._mnnvl_env_defaults(cd_channel, envs))
        extra_env.update(self._extra_pod_env_defaults())
        common_pod_kwargs: dict[str, Any] = dict(
            image=self._image,
            configmap_name=configmap_name,
            namespace=self._namespace,
            image_pull_policy=c.image_pull_policy,
            restart_policy=c.restart,
            env_secret_name=secret_name if use_secret else None,
            scheduling=self._scheduling,
            gpu_resource_name=self._gpu_resource_name,
            per_pod_gpus=self._per_pod_gpus,
            resource_claim_name=rct_name,
            host_network=self._host_network,
            host_ipc=self._host_ipc,
            node_selector=self._node_selector,
            tolerations=tolerations,
            extra_env=extra_env or None,
            compute_domain_channel=cd_channel,
            task_label=base,
            allocation_id=self._allocation_id,
            artifacts_configmap_name=artifacts_cm_name,
            file_artifact_mounts=file_mounts,
            host_path_mounts=host_path_mounts,
            pvc_mounts=pvc_mounts,
            shm_size=self._shm_size,
            run_as_root=self._run_as_root,
            sflow_scratch_dir=_sflow_pod_output_dir(envs) if use_secret else None,
            cpu_request=self._pod_cpu_request(self._per_pod_gpus),
            cpu_limit=self._cpu_limit,
            memory_request=self._memory_request,
            memory_limit=self._memory_limit,
            rdma_nic_resources=rdma_nic_resources,
            rdma_ipc_lock=(bool(rdma_hcas) and self._rdma_ipc_lock) or dra_coalloc,
            rdma_host_device_paths=(
                self._rdma_host_device_paths if rdma_hcas else []
            ),
            rdma_lib_mounts=rdma_lib_mounts,
        )

        # Launcher: pinned to reserved node 0 (deterministic HTTP server address);
        # keep the pin, add one-per-node antiAffinity.
        launcher_pod = render_task_pod(
            pod_name=f"{base}-launcher",
            assigned_node=(server_hostnames[0] if server_hostnames else None),
            **common_pod_kwargs,
        )
        launcher_spec = inject_mpi_scheduling(
            launcher_pod["spec"], job_name=job_name, one_pod_per_node=True
        )

        # Worker: runs the recipe's non-mpirun setup (same lines as the launcher, for
        # filesystem/side-effect parity) then execs sshd (the operator ssh-es in to
        # start orted); restricted to the reserved server nodes via nodeAffinity +
        # one-per-node antiAffinity.
        worker_pod = render_task_pod(
            pod_name=f"{base}-worker",
            assigned_node=None,
            **common_pod_kwargs,
        )
        worker_spec = worker_pod["spec"]
        worker_spec["containers"][0]["command"] = build_worker_command(
            ssh_port=self._ssh_port(),
            ensure_sshd=c.mpi.ensure_sshd,
            setup_lines=strip_final_launch(launcher_script),
        )
        # sshd comes up only after the setup, so a readiness probe on the ssh port
        # makes the operator's launcherCreationPolicy=WaitForWorkersReady gate on
        # setup completion (the ordering barrier). The failureThreshold x periodSeconds
        # budget (default 180 x 5s = ~15 min) bounds how long a slow per-node setup may
        # take before the worker is killed unready; both are configurable via
        # mpi.worker_readiness_{failure_threshold,period_seconds}.
        worker_spec["containers"][0]["readinessProbe"] = {
            "tcpSocket": {"port": self._ssh_port()},
            "initialDelaySeconds": 5,
            "periodSeconds": self._worker_readiness_period_seconds(),
            "timeoutSeconds": 5,
            "failureThreshold": self._worker_readiness_failure_threshold(),
        }
        worker_spec = inject_mpi_scheduling(
            worker_spec,
            job_name=job_name,
            include_hostnames=server_hostnames,
            strip_hostname_pin=True,
            one_pod_per_node=True,
        )

        mpijob = build_mpijob_manifest(
            name=job_name,
            namespace=self._namespace,
            launcher_pod_spec=launcher_spec,
            worker_pod_spec=worker_spec,
            worker_replicas=worker_replicas,
            slots_per_worker=slots,
            run_launcher_as_worker=run_launcher_as_worker,
            launcher_creation_policy=c.mpi.launcher_creation_policy,
            mpi_implementation=c.mpi.mpi_implementation,
            ssh_auth_mount_path=DEFAULT_SSH_AUTH_MOUNT_PATH,
            task_label=base,
            allocation_id=self._allocation_id,
        )
        # Single-node MPIJob (launcher-only): drop the empty Worker spec.
        if worker_replicas <= 0:
            mpijob["spec"]["mpiReplicaSpecs"].pop("Worker", None)
        items.append(mpijob)

        manifest = {"apiVersion": "v1", "kind": "List", "items": items}
        self._persist_rendered_manifest(manifest, task_name=task_name, envs=envs)

        ns_seg = namespace_segment(self._namespace)
        ns_args = ["--namespace", self._namespace] if self._namespace else []
        global_args = list(self._kubectl_global_args)
        mpijob_ref = f"mpijob.kubeflow.org/{job_name}"

        task_out = envs.get("SFLOW_TASK_OUTPUT_DIR")
        task_log_path = (
            os.path.join(task_out, f"{task_name}.log") if task_out else None
        )

        cleanup_refs = [mpijob_ref, f"configmap/{configmap_name}"]
        if artifacts_cm_name is not None:
            cleanup_refs.append(f"configmap/{artifacts_cm_name}")
        if use_secret:
            cleanup_refs.append(f"secret/{secret_name}")
        if rct_name is not None:
            cleanup_refs.append(f"resourceclaimtemplate.resource.k8s.io/{rct_name}")

        apply_command = build_manifest_apply_command(
            manifest_json=json.dumps(manifest, separators=(",", ":")),
            ns_seg=ns_seg,
            secret_name=secret_name if use_secret else None,
            envs=envs,
            handoff_delete_pods=self._handoff_pods,
            handoff_before_apply=self._handoff_destroy_first,
            kubectl_global_args=global_args,
            allocation_id=self._allocation_id,
        )
        return _K8sExecPlan(
            apply_command=apply_command,
            pod_refs=[],
            log_stream_commands=[],
            task_log_path=task_log_path,
            cleanup_refs=cleanup_refs,
            global_args=global_args,
            ns_args=ns_args,
            mpijob_name=job_name,
        )

    async def execute(
        self,
        *,
        launcher: Any,
        output_logger: Any,
        env: Mapping[str, str],
        task_name: str,
        script: Sequence[str],
        status_note: Callable[[str | None], None] | None = None,
    ) -> int:
        if self._resolved_route() != "operator":
            # Deliver the ephemeral SSH keypair through the task env so the apply
            # command's runtime Secret creation carries the real key (empty
            # otherwise -> the bootstrap preamble aborts). No-op single-node.
            return await super().execute(
                launcher=launcher,
                output_logger=output_logger,
                env=self._inject_pods_keypair_env(env),
                task_name=task_name,
                script=script,
                status_note=status_note,
            )
        return await self._execute_mpijob(
            launcher=launcher,
            output_logger=output_logger,
            env=env,
            task_name=task_name,
            script=script,
            status_note=status_note,
        )

    async def _execute_mpijob(
        self,
        *,
        launcher: Any,
        output_logger: Any,
        env: Mapping[str, str],
        task_name: str,
        script: Sequence[str],
        status_note: Callable[[str | None], None] | None = None,
    ) -> int:
        """Apply the MPIJob CR, then offload the launcher log + watch CR status.

        Mirrors the base pod ``execute`` but for one MPIJob: apply the CR (+ GPU
        handoff), discover the launcher pod (rank 0 / HTTP server), offload its log
        to ``<task>.log`` for ``log_watch`` probes, and complete on the MPIJob
        ``status.conditions`` (or the launcher pod's terminal status). A long-lived
        READY server never returns until teardown cancels this coroutine; the
        ``finally`` deletes the CR (the controller GCs the child pods via ownerRefs).
        """
        plan = self._build_mpijob_execution_plan(
            task_name=task_name, script=list(script), envs=dict(env)
        )
        job_name = plan.mpijob_name or sanitize_name(task_name)
        mpijob_ref = f"mpijob.kubeflow.org/{job_name}"
        stream_proc: asyncio.subprocess.Process | None = None
        tailer: asyncio.Future | None = None
        if status_note is not None:
            status_note("applying MPIJob")
        try:
            rc = await launcher.run_async(
                plan.apply_command,
                output_logger=output_logger,
                env=env,
                task_name=task_name,
            )
            if rc != 0:
                return rc
            apply_prefix_size = 0
            if plan.task_log_path:
                try:
                    apply_prefix_size = os.path.getsize(plan.task_log_path)
                except OSError:
                    apply_prefix_size = 0
            if status_note is not None:
                status_note("waiting for MPIJob launcher")
            launcher_ref = await mpi_lifecycle.discover_launcher_pod(
                job_name,
                global_args=plan.global_args,
                ns_args=plan.ns_args,
                timeout=self._launcher_discovery_timeout(),
            )
            if status_note is not None:
                status_note(None)
            if launcher_ref and plan.task_log_path:
                stream_proc = await k8s_lifecycle.start_pod_log_file_stream(
                    build_log_stream_command(
                        launcher_ref,
                        ns_args=plan.ns_args,
                        kubectl_global_args=plan.global_args,
                    ),
                    plan.task_log_path,
                )
                tailer = asyncio.ensure_future(
                    k8s_lifecycle.tail_file_to_console(
                        plan.task_log_path, task_name=task_name
                    )
                )
            phase = await mpi_lifecycle.watch_mpijob_until_terminal(
                mpijob_ref,
                launcher_ref,
                global_args=plan.global_args,
                ns_args=plan.ns_args,
            )
            if stream_proc is not None:
                await k8s_lifecycle.terminate_process(stream_proc)
                stream_proc = None
            if tailer is not None:
                tailer.cancel()
                await asyncio.gather(tailer, return_exceptions=True)
                tailer = None
            if launcher_ref and plan.task_log_path:
                await k8s_lifecycle.finalize_complete_log(
                    [launcher_ref],
                    plan.task_log_path,
                    prefix_size=apply_prefix_size,
                    phases=[phase],
                    global_args=plan.global_args,
                    ns_args=plan.ns_args,
                )
            if launcher_ref:
                return await k8s_lifecycle.pod_exit_code(
                    launcher_ref,
                    global_args=plan.global_args,
                    ns_args=plan.ns_args,
                    phase=phase,
                )
            return 0 if phase == "Succeeded" else 1
        finally:
            if tailer is not None and not tailer.done():
                tailer.cancel()
                await asyncio.gather(tailer, return_exceptions=True)
            if stream_proc is not None:
                await k8s_lifecycle.terminate_process(stream_proc)
            await k8s_lifecycle.delete_objects(
                plan.cleanup_refs,
                global_args=plan.global_args,
                ns_args=plan.ns_args,
            )
