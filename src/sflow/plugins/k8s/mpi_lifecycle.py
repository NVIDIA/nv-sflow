# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver-side lifecycle helpers for the Kubeflow ``MPIJob`` (operator route).

The ``k8s_mpi`` operator's *operator route* hands scheduling to the mpi-operator
controller, so sflow does not own the launcher/worker pods. Completion is driven
by the ``MPIJob`` ``status.conditions`` (``Running``/``Succeeded``/``Failed``)
plus the launcher pod's own terminal status; the launcher pod (rank 0, the HTTP
server) is discovered by the operator's labels and its log is offloaded into
``<task>.log`` exactly like the plain-pod route (so ``log_watch`` probes work).

These are thin async helpers over ``kubectl``; they reuse the pod-level primitives
in ``k8s.lifecycle`` (log offload, one-shot re-fetch, exit code) for the launcher.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from sflow.logging import get_logger
from sflow.plugins.k8s import lifecycle as k8s_lifecycle
from sflow.plugins.k8s.mpi_bootstrap import (
    MPI_JOB_NAME_LABEL,
    MPI_JOB_ROLE_LABEL,
    MPI_LAUNCHER_ROLE,
)

_logger = get_logger(__name__)

# Poll cadence for MPIJob status + launcher-pod discovery.
_MPIJOB_POLL_INTERVAL = 2.0
# How long to wait for the controller to create the launcher pod after the CR is
# applied (worker sshd startup + WaitForWorkersReady can add latency).
_LAUNCHER_DISCOVERY_TIMEOUT = 600.0
_TERMINAL_CONDITIONS = ("Succeeded", "Failed")


async def discover_launcher_pod(
    job_name: str,
    *,
    global_args: Sequence[str],
    ns_args: Sequence[str],
    timeout: float = _LAUNCHER_DISCOVERY_TIMEOUT,
    interval: float = _MPIJOB_POLL_INTERVAL,
) -> str | None:
    """Poll for the MPIJob's launcher pod, returning ``pod/<name>`` or ``None``.

    The mpi-operator labels the launcher pod ``training.kubeflow.org/job-name=<job>``
    + ``training.kubeflow.org/job-role=launcher``; this returns as soon as it
    exists (it may be created late under ``WaitForWorkersReady``). ``None`` on
    timeout / if the CR failed before creating it.
    """
    selector = f"{MPI_JOB_NAME_LABEL}={job_name},{MPI_JOB_ROLE_LABEL}={MPI_LAUNCHER_ROLE}"
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        rc, out, _ = await k8s_lifecycle.run_kubectl(
            ["get", "pods", "-l", selector, *ns_args, "-o", "name"],
            global_args=global_args,
            timeout=k8s_lifecycle.POLL_KUBECTL_TIMEOUT,
        )
        if rc == 0 and out.strip():
            # `kubectl get -o name` -> "pod/<name>" (first line if several).
            return out.strip().splitlines()[0].strip()
        if asyncio.get_event_loop().time() >= deadline:
            return None
        await asyncio.sleep(interval)


async def mpijob_condition(
    mpijob_ref: str, *, global_args: Sequence[str], ns_args: Sequence[str]
) -> str:
    """Return the MPIJob's current terminal condition type, or ``""``.

    Reads ``status.conditions`` and returns ``Succeeded``/``Failed`` when that
    condition is present with ``status == "True"``; otherwise ``""`` (Created /
    Running / not yet reconciled). ``""`` also when the CR is gone/unreadable.
    """
    # One jsonpath per terminal type: the `.status=="True"` filter yields the
    # type name only when that condition is active.
    for cond in _TERMINAL_CONDITIONS:
        jsonpath = (
            "jsonpath={.status.conditions[?(@.type==\"%s\")].status}" % cond
        )
        rc, out, _ = await k8s_lifecycle.run_kubectl(
            # Bounded like the plain-pod status poll: this is the MPI task's
            # completion signal, so an unbounded call here wedges the driver for the
            # kernel's TCP retransmission window with no log output. A timeout just
            # retries on the next tick of watch_mpijob_until_terminal.
            ["get", mpijob_ref, *ns_args, "-o", jsonpath],
            global_args=global_args,
            timeout=k8s_lifecycle.POLL_KUBECTL_TIMEOUT,
        )
        if rc == 0 and out.strip() == "True":
            return cond
    return ""


async def watch_mpijob_until_terminal(
    mpijob_ref: str,
    launcher_ref: str | None,
    *,
    global_args: Sequence[str],
    ns_args: Sequence[str],
    interval: float = _MPIJOB_POLL_INTERVAL,
) -> str:
    """Poll until the MPIJob (or its launcher pod) is terminal; return the phase.

    Returns ``Succeeded``/``Failed`` from the MPIJob condition, or the launcher
    pod's derived phase when its container terminates first (the CR condition can
    lag). For a long-lived READY server neither returns until the workflow cancels
    this coroutine on teardown -- exactly the plain-pod semantics.
    """
    while True:
        cond = await mpijob_condition(
            mpijob_ref, global_args=global_args, ns_args=ns_args
        )
        if cond in _TERMINAL_CONDITIONS:
            return cond
        if launcher_ref:
            # The MPIJob condition is authoritative here; the launcher pod's status is
            # only an early terminal signal. A transient API error (not_found=False,
            # phase="") is ignored -- we simply poll again.
            phase, container_done, container_failed, _not_found = (
                await k8s_lifecycle._pod_terminal_status(
                    launcher_ref, global_args=global_args, ns_args=ns_args
                )
            )
            if phase in ("Succeeded", "Failed"):
                return phase
            if container_done:
                return "Failed" if container_failed else "Succeeded"
        await asyncio.sleep(interval)
