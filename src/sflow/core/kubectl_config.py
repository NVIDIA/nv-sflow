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

from dataclasses import dataclass, field


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
        args: list[str] = []
        if self.kubeconfig:
            args += ["--kubeconfig", str(self.kubeconfig)]
        if self.context:
            args += ["--context", str(self.context)]
        args += [str(a) for a in self.extra_args]
        return args

    def is_empty(self) -> bool:
        return not (
            self.kubeconfig
            or self.context
            or self.namespace
            or self.extra_args
            or self.node_selector
        )
