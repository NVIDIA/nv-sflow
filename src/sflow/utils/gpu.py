# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

# Written by a job step that picks its own devices (slurm), read back by run
# reporting. Lives here, next to its only reader, so the producer imports it.
GPU_MARKER_FILE = ".sflow_gpus"


def count_device_tokens(cuda_visible: str | None) -> int:
    """How many devices a ``CUDA_VISIBLE_DEVICES``-style list names.

    Counts non-empty comma-separated tokens without interpreting them, so it
    works for numeric indices *and* for physical ids such as nvidia-smi UUIDs
    (which :func:`parse_cuda_visible_devices` deliberately ignores).
    """
    if not cuda_visible:
        return 0
    return len([t for t in str(cuda_visible).split(",") if t.strip()])


def count_visible_devices(cuda_visible: str | None) -> int:
    """How many devices a ``CUDA_VISIBLE_DEVICES``-style value names.

    Parses first, so the range form expands (``"0-3"`` -> 4), then falls back to
    the raw token count for non-numeric device ids such as nvidia-smi UUIDs,
    which :func:`parse_cuda_visible_devices` deliberately drops.

    Sizing a GPU *claim* (the docker operator's reservation) and sizing the
    *container-visible* slice (the docker backend's ``resource_env``) must agree
    on this number, so both go through here instead of counting tokens
    themselves -- counting raw tokens alone reads ``"0-3"`` as one device.
    """
    return len(parse_cuda_visible_devices(cuda_visible)) or count_device_tokens(
        cuda_visible
    )


def task_gpu_indices(task: Any) -> list[int]:
    """The **physical** GPU indices a task ran on, for run reporting.

    Sources, best first:

    1. ``task.reserved_gpu_indices`` -- devices actually claimed at launch by a
       backend that reserves specific GPUs (docker). Only known once the task has
       launched, and the only fully accurate source.
    1b. ``<task output dir>/`` + :data:`GPU_MARKER_FILE` -- written by the step
       itself (slurm), where the devices are chosen inside the job and can differ
       from the plan: on a GRES partition sflow re-derives its slice positionally
       from what slurmstepd handed the step, so a partial allocation like
       ``3,5,6,7`` turns plan slots ``0,1`` into physical ``3,5``.
    2. ``task.cuda_visible_devices`` -- the planner's slice. Trustworthy as device
       *identity* only when the backend injects a GPU env, which is the signal
       that the slice names real host devices.
    3. ``task.envs["CUDA_VISIBLE_DEVICES"]`` -- last resort for callers that only
       populate the execution env.

    Preferring (2) over (3) matters for containerized backends: docker exposes
    *virtual* indices ``0..N-1`` inside every container, so the env alone would
    report every task as sitting on GPU 0.

    Returns ``[]`` when the backend injects no GPU env at all (kubernetes, whose
    device plugin/DRA picks the devices). There the planner's slice exists purely
    for capacity planning and conflict detection -- reporting it as a device list
    would invent GPU numbers the pod never used. The dry-run allocation map still
    shows it, but that is a *plan*, not a claim about what ran.
    """
    reserved = getattr(task, "reserved_gpu_indices", None)
    if reserved:
        return [int(i) for i in reserved]
    envs = getattr(task, "envs", None) or {}
    if not envs.get("CUDA_VISIBLE_DEVICES"):
        return []
    out_dir = envs.get("SFLOW_TASK_OUTPUT_DIR")
    # Single-node only: the marker holds ONE step's devices, and a multi-node task's
    # nodes can hold different ones. This returns a single flat list, so reporting
    # node 0's devices for every node would be a confident wrong answer -- the plan,
    # which is uniform across nodes by construction, is the honest one there.
    if out_dir and len(getattr(task, "assigned_nodes", None) or []) <= 1:
        try:
            reported = (Path(out_dir) / GPU_MARKER_FILE).read_text()
        except OSError:
            reported = ""
        # An unparseable marker (e.g. CUDA UUID form) must fall through, not drop the
        # task out of run reporting entirely.
        indices = parse_cuda_visible_devices(reported.strip())
        if indices:
            return indices
    return planned_gpu_indices(task)


def planned_gpu_indices(task: Any) -> list[int]:
    """The GPU slice the **planner** assigned to a task, for plan-shaped views.

    The planner's ``task.cuda_visible_devices`` first, falling back to the
    execution env for callers that only populate ``task.envs``. Preferring the
    planner's value matters for containerized backends: docker exposes *virtual*
    indices ``0..N-1`` inside every container, so the env alone would report every
    task as sitting on GPU 0.

    Unlike :func:`task_gpu_indices` this still returns a slice for backends that
    never inject a GPU env (kubernetes). That is deliberate and is why the two are
    separate functions: an allocation map states what was *planned*, so the slice
    is exactly the right answer there, whereas run reporting states what actually
    ran and must not invent device ids the pod never used.
    """
    return parse_cuda_visible_devices(
        getattr(task, "cuda_visible_devices", None)
        or (getattr(task, "envs", None) or {}).get("CUDA_VISIBLE_DEVICES")
    )


def parse_cuda_visible_devices(cuda_visible: str | None) -> list[int]:
    """
    Parse CUDA_VISIBLE_DEVICES into a list of GPU indices.

    Supports comma-separated indices and simple ranges like ``0-3``.
    Non-numeric tokens are ignored.
    """
    if not cuda_visible:
        return []

    indices: list[int] = []
    for part in str(cuda_visible).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            try:
                start_i = int(start_s)
                end_i = int(end_s)
            except ValueError:
                continue
            if start_i <= end_i:
                indices.extend(range(start_i, end_i + 1))
            continue
        try:
            indices.append(int(token))
        except ValueError:
            continue
    return indices
