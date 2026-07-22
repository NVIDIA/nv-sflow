# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-level Kubernetes access configuration.

The recipe stays cluster-agnostic; the machine running ``sflow run`` carries the
kube access. These values come from ``sflow run`` CLI flags (``--kubeconfig`` /
``--kube-context`` / ``--kube-namespace`` / ``--extra-kubectl-args``) and are applied to
every ``kubectl`` invocation sflow makes -- both the kubernetes backend's own
calls (allocate / discover / release / preflight) and the per-task ``kubectl``
wrapper -- by mapping them onto kubectl's standard global flags.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sflow.utils.extra_args import normalize_extra_args


def kubectl_global_args(
    kubeconfig: str | None,
    context: str | None,
    extra_args: Sequence[str] | None,
) -> list[str]:
    """Build kubectl global flags: ``--kubeconfig`` / ``--context`` + passthroughs.

    Single source of the flag-building logic, shared by
    :meth:`KubectlConfig.global_args` (CLI-derived) and the kubernetes backend
    (instance-derived), so both agree. ``namespace`` is intentionally excluded --
    it overrides the backend namespace and is threaded per-call via ``--namespace``.
    """
    args: list[str] = []
    if kubeconfig:
        args += ["--kubeconfig", str(kubeconfig)]
    if context:
        args += ["--context", str(context)]
    # Shell-split passthroughs so a bundled/whitespace-laden entry can't produce
    # an unparsable kubectl global flag (see normalize_extra_args).
    args += normalize_extra_args(extra_args)
    return args


@dataclass(frozen=True)
class KubectlConfig:
    """Kube access selection passed from the CLI to kubernetes backends."""

    # --kubeconfig: path to the kubeconfig file (else $KUBECONFIG / ~/.kube/config).
    kubeconfig: str | None = None
    # --kube-context: context name within the kubeconfig (else current-context).
    context: str | None = None
    # --kube-namespace: overrides the backend's namespace default for all k8s backends.
    namespace: str | None = None
    # --kube-node-selector (repeatable KEY=VALUE): node-selector labels merged into
    # (and overriding) every k8s backend's node_selector -- applied as the pod
    # nodeSelector and the node-discovery selector (kubectl -l). Keeps cluster/node
    # pool identity (e.g. a `tenant` label) out of the recipe.
    node_selector: dict[str, str] = field(default_factory=dict)
    # --extra-kubectl-args (repeatable): verbatim global kubectl flags, e.g.
    # "--insecure-skip-tls-verify", "--as=admin", "--request-timeout=30s".
    extra_args: list[str] = field(default_factory=list)
    # --kube-compute-domain-channel: overrides `compute_domain.channel` for every k8s
    # backend (a channel template name, "auto", or "disable"/legacy "off"). Lets
    # MNNVL/IMEX be tuned per
    # run without editing the recipe. None => leave the recipe value untouched.
    compute_domain_channel: str | None = None
    # --kube-compute-domain-create / --no-...: overrides `compute_domain.create` for every
    # k8s backend (stand up an sflow-owned ComputeDomain CR). Tri-state: None => leave the
    # recipe value untouched; True/False => force on/off.
    compute_domain_create: bool | None = None
    # --kube-skip-pvc: drop every PVC-backed backend volume (a `volumes:` entry with a
    # `claim`) from all k8s backends for this run, keeping `empty_dir` volumes. A debug
    # aid for clusters that lack the recipe's PVCs -- pods then schedule without editing
    # the recipe volume-by-volume. The PVC data (e.g. a model cache) is NOT mounted, so
    # real workloads that need it will fail; intended for quick scheduling/plumbing checks.
    skip_pvc: bool = False
    # --kube-rdma: overrides the `rdma:` mode for every k8s backend this run (e.g. force
    # "disable" on a cluster whose IB/RoCE fabric is down/absent, so shipped recipes can
    # keep "rdma: auto"). None => leave the recipe value untouched.
    rdma: str | None = None
    # --kube-handoff: overrides `reservation.handoff` (GPU node handoff order) for every k8s
    # backend this run: "auto" (destroy-before-create iff a GPU ResourceQuota is detected),
    # "destroy_before_create" (delete the placeholder pod before applying the task pod --
    # quota-safe, never double-counts placeholder + task GPU requests), or
    # "create_before_destroy" (apply task first, no node-loss gap, but double-counts under a
    # GPU quota). Force "destroy_before_create" on a quota-constrained cluster so shipped
    # recipes keep "auto". None => leave the recipe value untouched.
    handoff: str | None = None
    # The subset of ``extra_args`` that originated from the generic, backend-agnostic
    # ``--extra-args`` (fanned into every backend channel) rather than the
    # kubectl-specific ``--extra-kubectl-args``. Diagnostic only (NOT part of
    # ``global_args()`` -- they are already in ``extra_args``): lets a kubernetes
    # backend warn that a generic arg (e.g. a Slurm-ism like ``--gpus-per-node``) is
    # being applied as a kubectl global flag, where it would break every call.
    generic_extra_args: list[str] = field(default_factory=list)

    def global_args(self) -> list[str]:
        """kubectl global flags for ``--kubeconfig`` / ``--context`` + passthroughs.

        ``namespace`` is intentionally excluded: it overrides the backend
        namespace and is already threaded onto every call via ``--namespace``.
        """
        return kubectl_global_args(self.kubeconfig, self.context, self.extra_args)

    def is_empty(self) -> bool:
        return not (
            self.kubeconfig
            or self.context
            or self.namespace
            or self.extra_args
            or self.node_selector
            or self.compute_domain_channel is not None
            or self.compute_domain_create is not None
            or self.skip_pvc
            or self.rdma is not None
            or self.handoff is not None
        )
