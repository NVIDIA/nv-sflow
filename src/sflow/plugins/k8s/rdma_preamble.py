# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime GPU<->RDMA NIC affinity preamble for Kubernetes task pods.

The Kubernetes GPU request is a *count* (``nvidia.com/gpu: N`` or a DRA claim),
so the device plugin / DRA driver -- not sflow -- decides which physical GPU a
pod gets. Selecting the RDMA NIC at manifest-build time (from a logical GPU slot)
therefore risks two failures:

* **Misplacement** -- pinning ``mlx5_3`` when the pod actually got the GPU next to
  ``mlx5_0`` (crosses the PCIe switch/NUMA boundary, hurting GPUDirect-RDMA BW).
* **Hard failure** -- pinning an RDMA device that is enumerated in sysfs but not
  actually usable in the pod (no ``rdma_cm``/verbs node, or no ACTIVE port). UCX
  then can't open the device and NIXL/UCX aborts with ``NIXL_ERR_BACKEND``
  instead of degrading to TCP. This is strictly worse than not pinning at all.

For providers where the pod can see every node HCA (host-device, shared device
plugin) -- or a DRA co-allocated NIC -- the robust fix is to decide *inside the
pod at runtime*. This module returns a small bash preamble (prepended to the task
entrypoint) whose behavior is chosen by ``SFLOW_RDMA_AFFINITY``:

* ``auto`` (default) -- **let the libraries pick.** NCCL (``NCCL_IB_HCA`` left
  unset) and UCX (``UCX_NET_DEVICES`` left unset so UCX can choose cuda_ipc /
  NVLink, RDMA, or TCP from topology) each select the best transport. sflow sets
  ``UCX_MAX_RNDV_RAILS=1`` so each GPU transfer stays on its single closest NIC
  when RDMA is used.
* ``explicit`` -- pin each GPU to the NIC on its PCIe root (``nvidia-smi`` bus id
  -> sysfs ``pcieRoot``). An escape hatch for fabrics where auto-detection
  mispairs because sysfs distance is not representative (e.g. GB300 Data-Direct
  sub-interfaces, SR-IOV VFs, flat PCIe).
* ``off`` -- inject nothing; leave device selection to the libs/recipe.

In every mode, if RDMA is not actually usable in the pod (missing
``rdma_cm``/verbs node or no ACTIVE port) the preamble does NOT force a transport
fallback -- it only prints a hint. Forcing NCCL onto sockets
(``NCCL_IB_DISABLE=1`` / ``NCCL_NET_PLUGIN=none`` / ``NCCL_IBEXT_DISABLE=1``)
would also suppress the rack-scale NVLink (MNNVL) path that NCCL/UCX auto-detect
on GB200/GB300 -- and "no ACTIVE IB port" is the *expected* state on a pure-MNNVL
rack, so that heuristic is a false "slow TCP" downgrade. sflow therefore leaves
transport selection to the libraries (they pick cuda_ipc/NVLink, MNNVL, RDMA, or
TCP from topology) and only *hints* the socket-forcing envs for the user to set
themselves when their cluster genuinely has no NVLink fabric and an external IB
plugin would otherwise abort on dead HCAs. UCX is likewise never forced.

This preamble applies only to providers that expose all node HCAs; GKE (fixed
per-pod NIC subset) keeps its build-time mapping and does not use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sentinel replaced with the backend-detected routable interface (e.g. ``eth0``).
# Baked in at render time so the preamble needs no runtime env to fall back.
_IFACE_TOKEN = "__SFLOW_PRIMARY_IFACE__"

# Decision markers the preamble prints to pod stdout. Kept as constants so the
# emitter (the bash template below) and the parser (``parse_rdma_runtime_status``)
# agree on the exact sentinels sflow greps for. The task log stream is offloaded
# straight to ``<task>.log`` (the driver is never in the per-line path), so this
# post-hoc parse is how sflow surfaces the in-pod TCP fallback to users.
_RDMA_MARKER = "[sflow-rdma]"
# RDMA NIC not usable in-pod. sflow no longer forces a transport here (that would
# also suppress the rack-NVLink/MNNVL path NCCL/UCX auto-detect); it only prints
# this hint, so the sentinel names the *condition*, not an action sflow took.
_NIC_UNUSABLE_SENTINEL = "no usable RDMA NIC"
_ACTIVE_SENTINELS = ("letting NCCL/UCX auto-select", "NCCL_IB_HCA=")
_DISABLED_SENTINEL = "disabled via SFLOW_RDMA_AFFINITY"
_GPUDIRECT_WARNING_SENTINEL = "WARNING GPUDirect RDMA unavailable:"
_UCX_INTRA_NODE_TCP_RE = re.compile(r"\btcp/[A-Za-z0-9_.:-]+")

# The preamble is a set of shell functions plus a single invocation. Kept as one
# template (token-substituted, not ``str.format``, because of the many braces).
_PREAMBLE_TEMPLATE = r"""
# --- sflow: GPU<->RDMA NIC affinity (runtime) ---
# SFLOW_RDMA_AFFINITY: auto (default) exposes all NICs and lets NCCL/UCX pick the
# GPU-closest device; explicit pins each GPU to the NIC on its PCIe root (for
# fabrics where auto-detection mispairs, e.g. GB300 Data-Direct / SR-IOV); off
# leaves selection to the libs. Any RDMA-unusable pod falls back to TCP.
_sflow_pcie_root() {
  # First path segment under /sys/devices identifies the PCIe root complex
  # (pci<domain>:<bus>), matching the resource.kubernetes.io/pcieRoot convention.
  case "$1" in
    /sys/devices/*) local _x="${1#/sys/devices/}"; printf '%s' "${_x%%/*}" ;;
    *) printf '' ;;
  esac
}
_sflow_rdma_hint() {
  # No usable RDMA NIC in this pod. sflow does NOT force a transport fallback:
  # setting NCCL_IB_DISABLE=1 / NCCL_NET_PLUGIN=none / NCCL_IBEXT_DISABLE=1 here
  # would also suppress the rack-scale NVLink (MNNVL) fabric that NCCL/UCX
  # auto-detect on GB200/GB300 -- and "no ACTIVE IB port" is the NORMAL state on
  # such a rack, so this in-pod probe is NOT a reliable "no fast interconnect"
  # signal. Leave selection to the libraries (they pick cuda_ipc/NVLink/MNNVL,
  # RDMA, or TCP from topology) and only hint. If this cluster genuinely has no
  # NVLink fabric AND an auto-loaded external IB net plugin (HPC-X
  # nccl_rdma_sharp_plugin / gIB "IBext", shipped in many runtime images) ABORTS
  # the process (double-free in its device-open failure path) while probing dead
  # HCAs during ncclCommInitRank, set these yourself (recipe env / -s) to force
  # the built-in socket net:
  #   export NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none NCCL_IBEXT_DISABLE=1
  echo "[sflow-rdma] ${1:-fallback}: no usable RDMA NIC -- leaving transport to NCCL/UCX auto-detect (cuda_ipc/NVLink/MNNVL/RDMA/TCP); to force sockets set NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none NCCL_IBEXT_DISABLE=1 (or set 'rdma: disable' on the backend) for your cluster"
}
_sflow_rdma_usable() {
  # RDMA is usable in-pod only if rdmacm + a verbs node exist AND some IB port is
  # ACTIVE. (The reservation-time probe sees sysfs HCAs but not whether the task
  # pod can actually open them -- this is the in-pod check that prevents the
  # observed NIXL_ERR_BACKEND on a dead HCA.) On failure set _SFLOW_RDMA_REASON to
  # the specific cause so the TCP-fallback marker names it for the user.
  local _dev_dir="${SFLOW_RDMA_DEV_DIR:-/dev/infiniband}"
  local _ib_sys="${SFLOW_IB_SYS_DIR:-/sys/class/infiniband}"
  if [ ! -e "$_dev_dir/rdma_cm" ]; then
    _SFLOW_RDMA_REASON="no RDMA connection-manager device (rdma_cm) in pod"
    return 1
  fi
  if ! ls "$_dev_dir"/uverbs* >/dev/null 2>&1; then
    _SFLOW_RDMA_REASON="no RDMA verbs device (uverbs) in pod"
    return 1
  fi
  local _sf
  for _sf in "$_ib_sys"/*/ports/*/state; do
    [ -f "$_sf" ] || continue
    grep -q ACTIVE "$_sf" 2>/dev/null && return 0
  done
  _SFLOW_RDMA_REASON="no InfiniBand/RoCE port is ACTIVE (all ports DOWN)"
  return 1
}
_sflow_rdma_auto() {
  # NCCL (unset NCCL_IB_HCA) and UCX (unset UCX_NET_DEVICES) each auto-select
  # from topology (cuda_ipc/NVLink, RDMA, or TCP). RAILS=1 keeps RDMA transfers
  # on the single closest NIC per GPU when RDMA is used.
  export UCX_MAX_RNDV_RAILS="${UCX_MAX_RNDV_RAILS:-1}"
  unset NCCL_IB_HCA 2>/dev/null || true
  unset NCCL_IB_DISABLE 2>/dev/null || true
  echo "[sflow-rdma] auto: letting NCCL/UCX auto-select devices (UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS})"
}
_sflow_gpudirect_check() {
  # GPUDirect RDMA requires a host peer-memory path (typically nvidia_peermem).
  # sflow cannot load kernel modules from a pod, but it can make missing node
  # support visible before UCX/NCCL silently stage GPU buffers through host memory.
  local _module_dir="${SFLOW_GPU_DIRECT_SYS_MODULE_DIR:-/sys/module}"
  if [ -e "$_module_dir/nvidia_peermem" ] || [ -e "$_module_dir/nv_peer_mem" ]; then
    echo "[sflow-rdma] GPUDirect RDMA peer-memory driver visible"
  else
    echo "[sflow-rdma] WARNING GPUDirect RDMA unavailable: nvidia_peermem/nv_peer_mem kernel module not visible in pod"
  fi
}
_sflow_rdma_explicit() {
  local _ib_sys="${SFLOW_IB_SYS_DIR:-/sys/class/infiniband}"
  # Enumerate RDMA devices with an ACTIVE port -> "name port pcieRoot" per line.
  local _devs="" _d _dev _port _sf _real _root
  for _d in "$_ib_sys"/*; do
    [ -e "$_d/device" ] || continue
    _dev="$(basename "$_d")"; _port=""
    for _sf in "$_d"/ports/*; do
      [ -f "$_sf/state" ] || continue
      if grep -q ACTIVE "$_sf/state" 2>/dev/null; then _port="$(basename "$_sf")"; break; fi
    done
    [ -n "$_port" ] || continue
    _real="$(readlink -f "$_d/device" 2>/dev/null)" || continue
    _root="$(_sflow_pcie_root "$_real")"
    _devs="${_devs}${_dev} ${_port} ${_root}"$'\n'
  done
  local _gpus _g _dom _rest _bdf _gsys _groot _hca=""
  _gpus="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$_gpus" ]; then
    while IFS= read -r _g; do
      _g="$(printf '%s' "$_g" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
      [ -n "$_g" ] || continue
      # nvidia-smi bus id is <domain>:<bus>:<dev>.<fn>; sysfs uses a 4-hex-digit domain.
      _dom="${_g%%:*}"; _rest="${_g#*:}"
      _bdf="$(printf '%04x' "0x${_dom}" 2>/dev/null || printf '%s' "$_dom"):${_rest}"
      _gsys="$(readlink -f "/sys/bus/pci/devices/${_bdf}" 2>/dev/null)"
      [ -n "$_gsys" ] || continue
      _groot="$(_sflow_pcie_root "$_gsys")"
      local _pick_dev="" _pick_port="" _any_dev="" _any_port="" _rd _rp _rr
      while IFS=' ' read -r _rd _rp _rr; do
        [ -n "$_rd" ] || continue
        [ -n "$_any_dev" ] || { _any_dev="$_rd"; _any_port="$_rp"; }
        if [ -n "$_groot" ] && [ "$_rr" = "$_groot" ]; then
          _pick_dev="$_rd"; _pick_port="$_rp"; break
        fi
      done <<SFLOW_RDMA_DEVS
$_devs
SFLOW_RDMA_DEVS
      if [ -z "$_pick_dev" ]; then
        _pick_dev="$_any_dev"; _pick_port="$_any_port"
        [ -n "$_pick_dev" ] && echo "[sflow-rdma] GPU ${_bdf} (pcieRoot ${_groot:-unknown}) has no co-located NIC; using ${_pick_dev}"
      fi
      if [ -n "$_pick_dev" ]; then
        case " $_hca " in
          *" $_pick_dev "*) : ;;
          *) _hca="${_hca:+$_hca }$_pick_dev" ;;
        esac
        echo "[sflow-rdma] GPU ${_bdf} -> ${_pick_dev}:${_pick_port} (pcieRoot ${_groot:-unknown})"
      fi
    done <<SFLOW_RDMA_GPUS
$_gpus
SFLOW_RDMA_GPUS
  fi
  if [ -z "$_hca" ]; then
    # No GPU topology (e.g. nvidia-smi unavailable): use every ACTIVE HCA.
    local _rd _rp _rr
    while IFS=' ' read -r _rd _rp _rr; do
      [ -n "$_rd" ] || continue
      case " $_hca " in
        *" $_rd "*) : ;;
        *) _hca="${_hca:+$_hca }$_rd" ;;
      esac
    done <<SFLOW_RDMA_ALL
$_devs
SFLOW_RDMA_ALL
    [ -n "$_hca" ] && echo "[sflow-rdma] explicit: no GPU topology info; using all ACTIVE HCAs: $_hca"
  fi
  if [ -n "$_hca" ]; then
    export NCCL_IB_HCA="$(printf '%s' "$_hca" | tr ' ' ',')"
    unset NCCL_IB_DISABLE 2>/dev/null || true
    echo "[sflow-rdma] explicit: NCCL_IB_HCA=${NCCL_IB_HCA} (UCX device selection left to the library)"
  else
    _sflow_rdma_hint "explicit: no usable RDMA device after topology match"
  fi
}
_sflow_rdma_setup() {
  local _mode="${SFLOW_RDMA_AFFINITY:-auto}"
  case "$_mode" in
    off|0|false|no) echo "[sflow-rdma] disabled via SFLOW_RDMA_AFFINITY"; return 0 ;;
  esac
  if ! _sflow_rdma_usable; then
    _sflow_rdma_hint "${_SFLOW_RDMA_REASON:-rdmacm/uverbs/ACTIVE port missing}"
    return 0
  fi
  _sflow_gpudirect_check
  case "$_mode" in
    explicit) _sflow_rdma_explicit ;;
    *) _sflow_rdma_auto ;;
  esac
}
_sflow_rdma_setup
# --- end sflow RDMA affinity ---
""".strip("\n")


def build_rdma_affinity_preamble(primary_iface: str = "") -> list[str]:
    """Return bash lines that steer NCCL onto each GPU's usable RDMA NIC.

    Prepended to the task entrypoint (before the workload launches) for providers
    that expose all node HCAs. Default (``SFLOW_RDMA_AFFINITY=auto``) lets
    NCCL/UCX auto-select devices; ``explicit`` pins NCCL per PCIe root; ``off``
    does nothing. When RDMA is not usable in the pod, sflow does NOT force a
    transport (forcing sockets would also suppress the rack-NVLink/MNNVL path the
    libraries auto-detect); it prints a hint listing the socket-forcing envs for
    the user to set if their cluster has no NVLink fabric. ``primary_iface`` is
    retained for API compatibility but is no longer injected into
    ``UCX_NET_DEVICES``.
    """
    return _PREAMBLE_TEMPLATE.replace(_IFACE_TOKEN, primary_iface or "").split("\n")


@dataclass(frozen=True)
class RdmaRuntimeStatus:
    """Outcome of the in-pod RDMA usability check, parsed from a task log.

    ``rdma_nic_unusable`` is True when at least one pod printed the no-usable-NIC
    marker -- RDMA was provisioned by the backend but not usable inside the pod
    (missing ``rdma_cm``/verbs node, or no ACTIVE port). sflow does NOT force a
    fallback in this case: NCCL/UCX auto-select the transport (rack NVLink/MNNVL
    if present, else TCP), so this is a hint the user may act on (set the
    socket-forcing envs), NOT a definitive "slow TCP" verdict. ``reason`` is the
    human-readable cause from the first such marker. ``pods_degraded`` /
    ``pods_total`` count per-pod decision markers so a multi-pod task can report
    how many replicas hit it. ``ucx_intra_node_tcp`` is True only when UCX's debug
    worker config shows TCP for intra-node transport AND ``cuda_ipc`` never appears
    in any intra-node config -- i.e. NVLink is genuinely unavailable, not merely the
    transient tcp-only startup config (UCX still rides cuda_ipc once a peer's lane is
    up, and its init tables look identical whether or not cuda_ipc ends up used).
    """

    rdma_nic_unusable: bool
    reason: str
    pods_degraded: int
    pods_total: int
    ucx_intra_node_tcp: bool = False
    ucx_transport: str = ""
    gpudirect_rdma_unavailable: bool = False
    gpudirect_rdma_reason: str = ""
    # True when the task's pods are in a rack-scale NVLink (MNNVL) domain (an IMEX
    # ComputeDomain): then ``rdma_nic_unusable`` (IB/RoCE NIC down) is NOT a
    # performance problem -- NCCL cross-node rides rack NVLink and the IB/RoCE NET
    # is only a fallback. Set by the operator (which knows the pod's
    # ComputeDomain), not parsed from the log. Lets the orchestrator skip the
    # "slow TCP" warning.
    mnnvl_crossnode: bool = False


def parse_rdma_runtime_status(log_text: str) -> RdmaRuntimeStatus | None:
    """Parse the ``[sflow-rdma]`` decision marker(s) from a task log.

    Returns ``None`` when the log has no marker at all (the preamble was not
    injected -- e.g. ``rdma: disable`` or a provider that pins a build-time NIC).
    Otherwise returns the aggregate outcome across every pod that logged a
    decision, so the caller can warn once per task.
    """
    unusable_reasons: list[str] = []
    active = 0
    disabled = 0
    ucx_intra_node_tcp_transport = ""
    ucx_intra_node_has_cuda_ipc = False
    gpudirect_rdma_reason = ""
    for raw in log_text.splitlines():
        lower = raw.lower()
        if "ucx" in lower and "intra-node" in lower:
            # cuda_ipc appearing in ANY intra-node worker config means UCX CAN use
            # NVLink for intra-node GPU transfers. Capture the (transient) tcp-only
            # config separately -- see the gating below.
            if "cuda_ipc" in lower:
                ucx_intra_node_has_cuda_ipc = True
            elif "tcp/" in lower and not ucx_intra_node_tcp_transport:
                match = _UCX_INTRA_NODE_TCP_RE.search(raw)
                ucx_intra_node_tcp_transport = (
                    match.group(0) if match else raw.strip()
                )
        idx = raw.find(_RDMA_MARKER)
        if idx < 0:
            continue
        body = raw[idx + len(_RDMA_MARKER) :].strip()
        if _NIC_UNUSABLE_SENTINEL in body:
            unusable_reasons.append(
                body.split(f": {_NIC_UNUSABLE_SENTINEL}", 1)[0].strip()
            )
        elif _GPUDIRECT_WARNING_SENTINEL in body:
            gpudirect_rdma_reason = body.split(
                _GPUDIRECT_WARNING_SENTINEL, 1
            )[1].strip()
        elif any(sentinel in body for sentinel in _ACTIVE_SENTINELS):
            active += 1
        elif _DISABLED_SENTINEL in body:
            disabled += 1
    total = len(unusable_reasons) + active + disabled
    # Only flag intra-node TCP when cuda_ipc/NVLink is NOT available anywhere in the
    # UCX config. UCX's init-time worker config routinely shows a transient tcp-only
    # config (cfg#1, before a peer registers) and labels even the summary rma_am
    # lane "tcp" while the actual GPU KV transfer then rides cuda_ipc once the
    # peer's cuda_ipc lane comes up (cfg#2) -- the init tables look identical
    # whether or not cuda_ipc ends up carrying the transfer. So the only reliable
    # "NVLink unavailable" signal is cuda_ipc never appearing at all (e.g. cross-pod
    # GPU isolation, or MNNVL/VMM memory without the required IMEX domain).
    ucx_intra_node_tcp = (
        bool(ucx_intra_node_tcp_transport) and not ucx_intra_node_has_cuda_ipc
    )
    ucx_transport = ucx_intra_node_tcp_transport if ucx_intra_node_tcp else ""
    gpudirect_rdma_unavailable = bool(gpudirect_rdma_reason)
    if total == 0 and not ucx_intra_node_tcp and not gpudirect_rdma_unavailable:
        return None
    return RdmaRuntimeStatus(
        rdma_nic_unusable=bool(unusable_reasons),
        reason=unusable_reasons[0] if unusable_reasons else "",
        pods_degraded=len(unusable_reasons),
        pods_total=total,
        ucx_intra_node_tcp=ucx_intra_node_tcp,
        ucx_transport=ucx_transport,
        gpudirect_rdma_unavailable=gpudirect_rdma_unavailable,
        gpudirect_rdma_reason=gpudirect_rdma_reason,
    )
