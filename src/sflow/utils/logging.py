# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sflow.utils.gpu import parse_cuda_visible_devices

_logger = logging.getLogger(__name__)


def isolated_logger(name_hint: str = "capture") -> logging.Logger:
    """Return a fresh, non-propagating logger dedicated to a single operation.

    Subprocess output capture/parsing works by attaching handlers to a logger and
    feeding the process output through it (see ``temporary_handler`` and
    ``SubprocessLauncher.run_async``). Using a *shared* logger for this is unsafe
    when several operations run concurrently (e.g. allocating multiple backends at
    once via ``asyncio.gather``): handlers attached by one operation receive the
    log records emitted by the others, cross-contaminating the parsed results.

    Each call returns a unique ``Logger`` instance, isolating concurrent
    operations from one another. The logger is constructed directly (not via
    ``logging.getLogger``) so it is not registered in the global logging manager
    and does not accumulate over time.
    """
    logger = logging.Logger(f"sflow.{name_hint}.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@contextmanager
def temporary_handler(
    logger: logging.Logger,
    handler: logging.Handler,
) -> Iterator[logging.Logger]:
    """Temporarily attach a handler to a logger.

    The handler is removed on exit (even if an exception is raised). If the handler
    is already attached to the logger, it will not be removed on exit.

    Args:
        logger: Logger to attach the handler to.
        handler: Handler to attach.

    Yields:
        The same logger (for convenience).
    """
    already_attached = handler in logger.handlers
    old_level = logger.level
    old_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not already_attached:
        logger.addHandler(handler)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        if not already_attached:
            logger.removeHandler(handler)


# --- Dry-run output formatting -------------------------------------------------
# A single, standardized layout for the dry-run report:
#   * the whole report is wrapped in an `═`×WIDTH envelope (start/end),
#   * every section starts with a `── Title ──…` divider (WIDTH wide),
#   * section bodies are indented 2 spaces.
DRY_RUN_WIDTH = 60


def dry_run_divider(title: str, width: int = DRY_RUN_WIDTH) -> str:
    """Render a standardized section divider, e.g. ``── Tasks ─────…`` (``width`` wide)."""
    label = f"── {title} "
    return label + "─" * max(0, width - len(label))


def log_dry_run_section(title: str) -> None:
    """Log a blank line followed by a standardized ``── Title ──`` section divider."""
    _logger.info("")
    _logger.info(dry_run_divider(title))


def log_dry_run_envelope(title: str, width: int = DRY_RUN_WIDTH) -> None:
    """Log the report envelope (``═``×width box) used to open and close the dry-run report."""
    _logger.info("")
    _logger.info("═" * width)
    _logger.info(title)
    _logger.info("═" * width)


def build_allocation_map_lines(tasks: list[Any], backends: dict[str, Any]) -> list[str]:
    """
    Build a terminal-friendly allocation map for finalized node and GPU assignments.
    """

    def _unique_preserve(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                out.append(value)
        return out

    lines: list[str] = []
    for backend_name, backend in backends.items():
        alloc = getattr(backend, "allocation", None)
        if alloc is None or not getattr(alloc, "nodes", None):
            continue

        backend_tasks = [
            task
            for task in tasks
            if getattr(task, "backend_name", None) == backend_name
        ]
        if not backend_tasks:
            continue

        node_map: dict[str, dict[str, Any]] = {}
        ordered_node_names: list[str] = []
        for node in alloc.nodes:
            num_gpus = getattr(node, "num_gpus", None)
            try:
                num_gpus = int(num_gpus) if num_gpus is not None else None
            except Exception:
                num_gpus = None
            node_map[node.name] = {
                "num_gpus": num_gpus,
                "gpu_owners": {},
                "tasks": [],
            }
            ordered_node_names.append(node.name)

        for task in backend_tasks:
            assigned_nodes = list(getattr(task, "assigned_nodes", None) or [])
            if not assigned_nodes:
                op_conf = getattr(getattr(task, "operator", None), "config", None)
                assigned_nodes = list(getattr(op_conf, "nodelist", None) or [])
            if not assigned_nodes and alloc.nodes:
                assigned_nodes = [node.name for node in alloc.nodes]

            # Use the planner's computed GPU slice (CUDA_VISIBLE_DEVICES), which is
            # calculated uniformly for every backend. It is carried on the task even
            # when not injected into the execution env (e.g. Kubernetes, where the
            # cluster/DRA assigns physical devices); fall back to the env for callers
            # that only populate task.envs.
            gpu_indices = parse_cuda_visible_devices(
                getattr(task, "cuda_visible_devices", None)
                or getattr(task, "envs", {}).get("CUDA_VISIBLE_DEVICES")
            )

            for node_name in assigned_nodes:
                if node_name not in node_map:
                    node_map[node_name] = {
                        "num_gpus": None,
                        "gpu_owners": {},
                        "tasks": [],
                    }
                    ordered_node_names.append(node_name)
                entry = node_map[node_name]
                entry["tasks"].append(task.name)
                for gpu_idx in gpu_indices:
                    owners = entry["gpu_owners"].setdefault(gpu_idx, [])
                    owners.append(task.name)

        lines.append(f"  - backend '{backend_name}':")
        for position, node_name in enumerate(ordered_node_names):
            entry = node_map[node_name]
            num_gpus = entry["num_gpus"]
            task_names = _unique_preserve(entry["tasks"])
            is_last_node = position == len(ordered_node_names) - 1
            # Lead with the node's task summary, then hang the per-GPU breakdown
            # one level deeper. At equal depth the "Tasks:" line reads as just
            # another GPU row, which is what the extra indent avoids.
            node_glyph = "└─" if is_last_node else "├─"
            cont = "   " if is_last_node else "│  "
            lines.append(f"    {node_glyph} node {node_name}")
            lines.append(
                f"    {cont}├─ Tasks: "
                + (", ".join(task_names) if task_names else "(none)")
            )
            if num_gpus is not None and num_gpus > 0:
                lines.append(f"    {cont}└─ GPUs:")
                for gpu_idx in range(num_gpus):
                    owners = _unique_preserve(entry["gpu_owners"].get(gpu_idx, []))
                    label = " -> ".join(owners) if owners else "."
                    lines.append(f"    {cont}     GPU {gpu_idx}: {label}")
            else:
                lines.append(f"    {cont}└─ GPUs: n/a")
    return lines


def build_resource_rehearsal_lines(tasks: list[Any]) -> list[str]:
    """Build dry-run lines describing task resource release boundaries."""

    def _resource_label(resource: str) -> str:
        return "GPUs" if resource == "gpus" else resource

    def _policy_description(resource: str, policy: str) -> str:
        label = _resource_label(resource)
        if policy == "task_ready":
            return f"releases {label} after task readiness"
        if policy == "task_completion":
            return f"releases {label} after task completion"
        if policy == "workflow_completion":
            return f"keeps {label} until workflow completion"
        return f"{label} release policy: {policy}"

    lines: list[str] = []
    for task in tasks:
        release_after = getattr(task, "resource_release_after", None) or {}
        if not release_after:
            continue
        parts = [
            _policy_description(resource, policy)
            for resource, policy in sorted(release_after.items())
        ]
        if parts:
            lines.append(f"  - {task.name}: " + "; ".join(parts))
    return lines
