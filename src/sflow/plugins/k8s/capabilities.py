# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uniform detection of optional Kubernetes cluster infra (CRDs / aggregated APIs).

sflow can opportunistically use cluster infra installed as a CRD or aggregated API
-- the Kubeflow MPI Operator (``MPIJob``), the NVIDIA ComputeDomain / IMEX driver,
and, in future, others (e.g. LeaderWorkerSet). Whether sflow *should* use one is not
a boolean but a tri-state, because a namespaced user can see a cluster-wide CRD yet
lack the RBAC to create/manage its instances:

    ABSENT  ->  INSTALLED  ->  USABLE

(plus ``UNKNOWN`` for "not detected yet", e.g. dry-run). :class:`CapabilityState`
captures this so every consumer decides in the same, standard way:

* presence check ("does the fabric exist?")        -> ``state.installed``
* opportunistic / auto path ("can I drive it?")     -> ``state.usable`` (else fall back)
* explicit / forced path                            -> proceed unless ``ABSENT``; a
  separate hard RBAC check surfaces a clear "grant these" error when not usable.

Add a new capability by declaring a :class:`ClusterCapability` in the registry below;
the backend's ``detect_capability`` + ``capability_state`` then handle it uniformly.

(Lives in the shared ``plugins/k8s`` package so both the Kubernetes backend and the
``k8s_mpi`` operator can import it without a backend<->operators dependency.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class CapabilityState(Enum):
    """Availability of an optional cluster capability (``ABSENT``->``USABLE``)."""

    # Not detected yet (e.g. dry-run, or the capability was never probed). Treated
    # as unavailable by opportunistic paths, but NOT as a hard "absent" signal (so a
    # forced/explicit path does not error before detection has run).
    UNKNOWN = "unknown"
    # The CRD / aggregated API is not served by the cluster.
    ABSENT = "absent"
    # Served, but the current credentials may lack the RBAC to create/manage its
    # instances (a namespaced user can see a cluster-wide CRD without namespace RBAC).
    INSTALLED = "installed"
    # Served AND the current credentials can create/manage its instances here.
    USABLE = "usable"

    @property
    def installed(self) -> bool:
        """Whether the API is served (``INSTALLED`` or ``USABLE``)."""
        return self in (CapabilityState.INSTALLED, CapabilityState.USABLE)

    @property
    def usable(self) -> bool:
        """Whether the API is served AND drivable by the current credentials."""
        return self is CapabilityState.USABLE


@dataclass(frozen=True)
class ClusterCapability:
    """An optional Kubernetes infra resource sflow can opportunistically use.

    ``api_resource`` is the fully-qualified ``<resource>.<group>`` (as ``kubectl get
    <api_resource>`` / discovery report it, e.g. ``mpijobs.kubeflow.org``).
    ``use_verbs`` are the (namespaced, unless ``namespaced=False``) verbs sflow needs
    to *drive* the resource; empty means presence-only (no usability distinction).
    """

    key: str
    api_resource: str
    use_verbs: tuple[str, ...] = ()
    namespaced: bool = True


def detect_capability_state(
    cap: ClusterCapability,
    *,
    is_served: Callable[[str], bool],
    can_i: Callable[..., bool],
    check_usable: bool,
) -> CapabilityState:
    """Resolve ``cap``'s state from two probes (pure; performs no kubectl itself).

    * ``is_served(api_resource) -> bool`` -- is the CRD/API served (API discovery)?
    * ``can_i(verb, resource, *, namespaced) -> bool`` -- may the creds do ``verb``?

    ``check_usable`` gates the (per-verb) RBAC probe: pass ``True`` only when the
    caller needs to *drive* the resource (e.g. ``route: auto``), so presence-only
    callers skip the extra ``auth can-i`` round-trips and get ``INSTALLED``.
    """
    if not is_served(cap.api_resource):
        return CapabilityState.ABSENT
    if not (check_usable and cap.use_verbs):
        return CapabilityState.INSTALLED
    permitted = all(
        can_i(verb, cap.api_resource, namespaced=cap.namespaced)
        for verb in cap.use_verbs
    )
    return CapabilityState.USABLE if permitted else CapabilityState.INSTALLED


# --- Registry of the capabilities sflow knows about -------------------------------
# Extend here for new infra (e.g. LeaderWorkerSet); downstream detection is uniform.
MPI_OPERATOR = ClusterCapability(
    key="mpi_operator",
    api_resource="mpijobs.kubeflow.org",
    use_verbs=("create", "get", "watch", "delete"),
)
COMPUTE_DOMAIN = ClusterCapability(
    key="compute_domain",
    api_resource="computedomains.resource.nvidia.com",
    use_verbs=("create", "delete"),
)
