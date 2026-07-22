# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RDMA/InfiniBand detection strategy for the Kubernetes backend.

Different clusters expose RDMA NICs differently. Rather than special-casing one
provider inline, the backend runs a small **provider chain**: each provider knows
how to recognize its mechanism from the node's advertised resources (plus the
HCAs/interface the backend probed on the reservation pod) and returns an
:class:`RdmaPlan` describing how task pods should request RDMA.

Providers (auto priority order):

* :class:`GkeRdmaProvider`           -- GKE multi-NIC RDMA: one
  ``networking.gke.io.networks/rdma-N`` extended resource per NIC, plus the GKE
  gIB lib mounts + NCCL tuning script.
* :class:`SharedDevicePluginRdmaProvider` -- k8s-rdma-shared-dev-plugin / NVIDIA
  Network Operator: a single shared ``rdma/*`` extended resource grants verbs
  access to the node's HCAs.
* :class:`HostDeviceRdmaProvider`    -- generic bare-metal fallback: no device
  plugin, so grant verbs access by hostPath-mounting ``/dev/infiniband`` +
  ``CAP_IPC_LOCK`` (requires ``host_network``).

When none applies, :meth:`RdmaPlan.tcp_fallback` pins NCCL/gloo socket
interfaces to the routable NIC. UCX device selection is left to the library.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

# GKE GPUDirect-RDMA (gIB) layout installed by the nccl-rdma-installer DaemonSet:
# the NVIDIA driver libs + NCCL gIB plugin live on the node and are hostPath-mounted
# into task pods so multi-node NCCL collectives run over RoCE. Sourcing
# set_nccl_env.sh applies GKE's NCCL tuning. Used only for multi-node GPU pods
# (single-node TP stays on NVLink and needs neither).
GKE_RDMA_LIB_MOUNTS: tuple[tuple[str, str], ...] = (
    ("/home/kubernetes/bin/nvidia", "/usr/local/nvidia"),
    ("/home/kubernetes/bin/gib", "/usr/local/gib"),
)
GKE_NCCL_ENV_SCRIPT = "/usr/local/gib/scripts/set_nccl_env.sh"

# Provider keys usable as an explicit ``rdma`` backend value (force one provider).
RDMA_PROVIDER_KEYS = ("gke", "shared_device_plugin", "host_device")

# Where the host's InfiniBand device nodes live (bare-metal host-device path).
_HOST_IB_DEVICE_PATH = "/dev/infiniband"


def build_network_env(
    *,
    nccl_ib_hca: str = "",
    socket_iface: str = "",
    rdma_hcas: str = "",
) -> dict[str, str]:
    """Assemble device/interface env for IB/NCCL/gloo.

    * ``nccl_ib_hca`` -> ``NCCL_IB_HCA``.
    * ``socket_iface`` -> ``NCCL_SOCKET_IFNAME`` + ``GLOO_SOCKET_IFNAME`` (the
      control/bootstrap NIC; pins it so the libs don't pick a link-local one).
    * ``rdma_hcas`` -> ``SFLOW_RDMA_HCAS`` (informational: RDMA devices present on
      the node).
    """
    env: dict[str, str] = {}
    if nccl_ib_hca:
        env["NCCL_IB_HCA"] = nccl_ib_hca
    if socket_iface:
        env["NCCL_SOCKET_IFNAME"] = socket_iface
        env["GLOO_SOCKET_IFNAME"] = socket_iface
        env["SFLOW_PRIMARY_IFACE"] = socket_iface
    if rdma_hcas:
        env["SFLOW_RDMA_HCAS"] = rdma_hcas
    return env


@dataclass
class RdmaDetectContext:
    """Everything a provider needs, gathered once by the backend.

    The backend performs the (backend-specific) kubectl work -- probing the
    reservation pod for HCAs + the routable interface, and fetching the scheduling
    node's ``allocatable`` -- so the providers themselves stay pure and easy to
    unit-test.
    """

    node_name: str
    node_allocatable: Mapping[str, str]
    hcas: list[str]
    primary_iface: str
    host_network: bool
    # Whether the GKE gIB stack (``nccl-rdma-installer`` DaemonSet) is deployed, so
    # ``/home/kubernetes/bin/gib`` + the tuned NCCL env actually exist on the node.
    # Gates the GKE provider's lib mounts: mounting those host paths when absent
    # would (with DirectoryOrCreate) create empty dirs that MASK the driver at
    # ``/usr/local/nvidia``. When False, no lib mounts + no NCCL tuning are emitted.
    gib_installed: bool = False


@dataclass(frozen=True)
class RdmaPlan:
    """How task pods should request RDMA on this cluster (backend -> operator).

    ``nic_specs`` is a list of ``(resource_name, hca_name)``; ``resource_name`` is
    ``""`` for the host-device provider (no extended resource, access via the
    hostPath device mount). The operator slices this per pod and de-dups the
    (non-empty) resource requests.
    """

    provider: str
    enabled: bool
    nic_specs: tuple[tuple[str, str], ...] = ()
    net_env: Mapping[str, str] = field(default_factory=dict)
    ipc_lock: bool = False
    host_device_paths: tuple[str, ...] = ()
    lib_mounts: tuple[tuple[str, str], ...] = ()
    nccl_env_script: str = ""
    # Whether task pods should choose the GPU-local RDMA NIC at *runtime* (see
    # ``k8s.rdma_preamble``) instead of a build-time slot. Only valid for
    # providers where the pod can see every node HCA (host-device, shared device
    # plugin): the pod's GPU is picked by the device plugin/DRA, so the co-located
    # NIC is only known in-pod. GKE grants a fixed per-pod NIC subset, so it keeps
    # its build-time mapping (this stays False).
    allow_runtime_affinity: bool = False

    @classmethod
    def disabled(cls) -> "RdmaPlan":
        """No RDMA wiring and no env -- the neutral placeholder state.

        Used for ``auto`` before reservation-time detection runs (and in dry-run,
        where it never runs) and whenever nothing applies. Distinct from
        :meth:`off`: this injects NOTHING, so it must never force a transport --
        that would wrongly downgrade an as-yet-undetected fast fabric (IB, or the
        rack NVLink/MNNVL the libraries auto-detect).
        """
        return cls(provider="off", enabled=False)

    @classmethod
    def off(cls, *, socket_iface: str = "") -> "RdmaPlan":
        """``rdma: disable`` -- the explicit, user-requested clean RDMA kill switch.

        Force NCCL onto its built-in socket net so it never probes IB/RoCE HCAs,
        and stop the auto-loaded external IB/SHARP net plugin (HPC-X
        ``nccl_rdma_sharp_plugin`` / gIB "IBext", shipped in many runtime images)
        from dlopening and ABORTING on a dead HCA:

        * ``NCCL_IB_DISABLE=1``    -- off NCCL's built-in IB (verbs) transport.
        * ``NCCL_IBEXT_DISABLE=1`` -- off the external-plugin IBext path.
        * ``NCCL_NET_PLUGIN=none`` -- do not load the external net plugin at all.

        Unlike ``auto`` (which only *hints* when a NIC is unusable, so it never
        suppresses the fast path the libraries auto-detect), ``off`` is an explicit
        choice, so forcing these is intended -- a clean one-knob way to disable
        RDMA. UCX is left alone (it still picks cuda_ipc/NVLink intra-node), and
        NCCL cross-node NVLink/MNNVL (P2P, not a NET transport) is unaffected.

        Disabling RDMA does NOT remove the need for a cross-node control NIC: with
        the IB transport off, NCCL/gloo fall back to *sockets*, so the routable
        interface matters MORE, not less. When the backend has probed it,
        ``socket_iface`` is pinned via ``NCCL_SOCKET_IFNAME`` / ``GLOO_SOCKET_IFNAME``
        and surfaced as ``SFLOW_PRIMARY_IFACE`` so cross-node collectives -- and any
        recipe reading ``$SFLOW_PRIMARY_IFACE`` -- get a real device even without IB.
        Left empty (dry-run / pre-detection / no default route) it injects only the
        disable flags, unchanged from the neutral placeholder.
        """
        net_env: dict[str, str] = {
            "NCCL_IB_DISABLE": "1",
            "NCCL_IBEXT_DISABLE": "1",
            "NCCL_NET_PLUGIN": "none",
        }
        if socket_iface:
            net_env.update(build_network_env(socket_iface=socket_iface))
        return cls(provider="off", enabled=False, net_env=net_env)

    @classmethod
    def tcp_fallback(cls, ctx: RdmaDetectContext) -> "RdmaPlan":
        """No scoped RDMA available: pin NCCL/gloo socket iface to the routable NIC.

        The detected HCAs are still surfaced via ``SFLOW_RDMA_HCAS`` for opt-in.
        UCX device selection is left to the library (cuda_ipc/NVLink, RDMA, or TCP).
        When there is no routable interface either, inject nothing and leave device
        selection to the libs.
        """
        if not ctx.primary_iface:
            return cls(provider="none", enabled=False)
        return cls(
            provider="none",
            enabled=False,
            net_env=build_network_env(
                socket_iface=ctx.primary_iface,
                rdma_hcas=",".join(ctx.hcas),
            ),
        )


class RdmaProvider(ABC):
    """A strategy for recognizing and wiring one RDMA mechanism."""

    key: str

    @abstractmethod
    def applies(self, ctx: RdmaDetectContext) -> bool:
        """Whether this provider's mechanism is present on the node."""

    @abstractmethod
    def build_plan(self, ctx: RdmaDetectContext) -> RdmaPlan:
        """Build the RDMA plan for this mechanism (only called when ``applies``)."""


class GkeRdmaProvider(RdmaProvider):
    """GKE multi-NIC RDMA: one ``networking.gke.io.networks/rdma-N`` per NIC."""

    key = "gke"
    _RE = re.compile(r"networking\.gke\.io\.networks/rdma-(\d+)")

    def _indexed(self, ctx: RdmaDetectContext) -> list[tuple[int, str]]:
        indexed: list[tuple[int, str]] = []
        for resource in ctx.node_allocatable:
            m = self._RE.fullmatch(str(resource))
            if m:
                indexed.append((int(m.group(1)), str(resource)))
        indexed.sort()
        return indexed

    def applies(self, ctx: RdmaDetectContext) -> bool:
        return bool(ctx.hcas) and bool(self._indexed(ctx))

    def build_plan(self, ctx: RdmaDetectContext) -> RdmaPlan:
        # Map each GKE ``rdma-N`` resource (sorted by index) to an IB device name.
        # Prefer the HCA names the backend actually discovered from the node's sysfs
        # (``ctx.hcas``, sorted ascending in ``_detect_network_env``) -- both this list
        # and ``self._indexed`` are index-ordered, so the i-th NIC maps to the i-th
        # discovered HCA. This avoids baking in the ``mlx5_N`` Mellanox naming/ordering
        # assumption, which breaks when the node's HCAs are not contiguous ``mlx5_0..``
        # (different vendor prefix, gaps, or a device order that doesn't match the GKE
        # NIC index). Fall back to the synthesized ``mlx5_{idx}`` only when discovery
        # produced fewer names than NICs (or none), preserving today's behavior on
        # clusters where the mapping already works. (``applies`` guarantees at least
        # one HCA, so the fallback is per-index, not all-or-nothing.)
        indexed = self._indexed(ctx)
        nic_specs = tuple(
            (
                resource,
                ctx.hcas[i] if i < len(ctx.hcas) else f"mlx5_{idx}",
            )
            for i, (idx, resource) in enumerate(indexed)
        )
        # Only mount the gIB libs + wire the NCCL tuning when the installer is
        # actually deployed. The lib mounts include ``/home/kubernetes/bin/nvidia``
        # -> ``/usr/local/nvidia`` (the driver path); bind-mounting a non-existent
        # host dir there would mask ``libcuda.so.1``. So when gIB is absent, request
        # NO lib mounts and leave NCCL on its built-in IB transport (still RoCE).
        lib_mounts = GKE_RDMA_LIB_MOUNTS if ctx.gib_installed else ()
        nccl_env_script = GKE_NCCL_ENV_SCRIPT if ctx.gib_installed else ""
        return RdmaPlan(
            provider=self.key,
            enabled=True,
            nic_specs=nic_specs,
            net_env=build_network_env(
                socket_iface=ctx.primary_iface, rdma_hcas=",".join(ctx.hcas)
            ),
            ipc_lock=True,
            lib_mounts=lib_mounts,
            nccl_env_script=nccl_env_script,
        )


class SharedDevicePluginRdmaProvider(RdmaProvider):
    """k8s-rdma-shared-dev-plugin / NVIDIA Network Operator: a shared ``rdma/*``.

    A single shared extended resource grants a pod verbs access to the node's
    HCAs. The pod requests the resource once (the operator de-dups); the detected
    HCAs are informational (``SFLOW_RDMA_HCAS``) and, for this provider, may drive
    the *runtime* affinity preamble's verified in-pod selection -- sflow never
    build-time pins ``NCCL_IB_HCA``.
    """

    key = "shared_device_plugin"
    _RE = re.compile(r"rdma/.+")

    def _candidates(self, ctx: RdmaDetectContext) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, value in ctx.node_allocatable.items():
            if self._RE.fullmatch(str(name)):
                try:
                    out[str(name)] = int(value)
                except (TypeError, ValueError):
                    out[str(name)] = 0
        return out

    def _pick(self, ctx: RdmaDetectContext) -> str | None:
        cands = self._candidates(ctx)
        if not cands:
            return None
        # Highest advertised count wins (ties broken by name for determinism).
        return max(sorted(cands), key=lambda k: cands[k])

    def applies(self, ctx: RdmaDetectContext) -> bool:
        return bool(ctx.hcas) and self._pick(ctx) is not None

    def build_plan(self, ctx: RdmaDetectContext) -> RdmaPlan:
        resource = self._pick(ctx) or ""
        nic_specs = tuple((resource, hca) for hca in ctx.hcas)
        return RdmaPlan(
            provider=self.key,
            enabled=True,
            nic_specs=nic_specs,
            net_env=build_network_env(
                socket_iface=ctx.primary_iface, rdma_hcas=",".join(ctx.hcas)
            ),
            ipc_lock=True,
            # The shared resource grants verbs access to all node HCAs, so the pod
            # can pick its GPU-local NIC at runtime.
            allow_runtime_affinity=True,
        )


class HostDeviceRdmaProvider(RdmaProvider):
    """Generic bare-metal fallback: hostPath ``/dev/infiniband`` + ``IPC_LOCK``.

    Used when HCAs are present but no device plugin advertises them. Requires
    ``host_network`` so the pod shares the host's NICs. No extended resource is
    requested (``resource_name == ""``); verbs access comes from the device mount.
    """

    key = "host_device"

    def applies(self, ctx: RdmaDetectContext) -> bool:
        return bool(ctx.hcas) and ctx.host_network

    def build_plan(self, ctx: RdmaDetectContext) -> RdmaPlan:
        nic_specs = tuple(("", hca) for hca in ctx.hcas)
        return RdmaPlan(
            provider=self.key,
            enabled=True,
            nic_specs=nic_specs,
            net_env=build_network_env(
                socket_iface=ctx.primary_iface, rdma_hcas=",".join(ctx.hcas)
            ),
            ipc_lock=True,
            host_device_paths=(_HOST_IB_DEVICE_PATH,),
            # The hostPath /dev/infiniband mount exposes all node HCAs, so the pod
            # can pick its GPU-local NIC at runtime.
            allow_runtime_affinity=True,
        )


def detect_rdma(
    ctx: RdmaDetectContext,
    *,
    forced: str | None = None,
) -> RdmaPlan:
    """Run the provider chain and return the first applicable plan (else TCP).

    ``forced`` (from ``rdma``) restricts the chain to one provider key;
    ``auto``/``None`` tries all in priority order. The shared-device-plugin
    provider auto-picks the ``rdma/*`` resource with the highest advertised count.
    """
    chain: list[RdmaProvider] = [
        GkeRdmaProvider(),
        SharedDevicePluginRdmaProvider(),
        HostDeviceRdmaProvider(),
    ]
    if forced and forced in RDMA_PROVIDER_KEYS:
        chain = [p for p in chain if p.key == forced]
    for provider in chain:
        if provider.applies(ctx):
            return provider.build_plan(ctx)
    return RdmaPlan.tcp_fallback(ctx)
