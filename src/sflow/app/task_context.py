# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Any

from sflow.core.backend import Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.task import Task, TaskPort
from sflow.core.task_graph import TaskGraph


def compute_task_service(
    *,
    backend: Backend | None,
    assigned_nodes: list[str],
    ports: list[TaskPort],
) -> dict[str, Any]:
    """Resolve task.<name>.service.{host, port, url} (host empty without runtime node IPs)."""
    host = ""
    if backend is not None and assigned_nodes:
        capabilities = getattr(backend, "capabilities", None)
        allocation = getattr(backend, "allocation", None)
        if getattr(capabilities, "has_runtime_node_addresses", True) and allocation:
            node = {n.name: n for n in allocation.nodes}.get(assigned_nodes[0])
            if node is not None:
                host = node.ip_address or ""
    port: Any = ports[0].port if ports else ""
    url = f"http://{host}:{port}" if host and port != "" else ""
    return {"host": host, "port": port, "url": url}


def build_task_info(
    task: Task,
    backends: dict[str, Backend],
) -> dict[str, Any]:
    """Build expression-context info for a single task."""

    backend = backends.get(task.backend_name) if task.backend_name else None
    alloc_nodes_by_name: dict[str, ComputeNode] = {}
    if backend and backend.allocation:
        alloc_nodes_by_name = {n.name: n for n in backend.allocation.nodes}

    task_nodes: list[dict[str, Any]] = []
    for i, node_name_assigned in enumerate(task.assigned_nodes):
        node = alloc_nodes_by_name.get(node_name_assigned)
        if node:
            task_nodes.append(
                {
                    "name": node.name,
                    "ip_address": node.ip_address,
                    "index": i,
                    "num_gpus": node.num_gpus,
                }
            )
        else:
            task_nodes.append(
                {
                    "name": node_name_assigned,
                    "ip_address": "",
                    "index": i,
                    "num_gpus": None,
                }
            )

    gpus: list[int] = []
    cuda_visible = task.envs.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        try:
            gpus = [int(g.strip()) for g in cuda_visible.split(",") if g.strip()]
        except ValueError:
            gpus = []

    return {
        "nodes": task_nodes,
        "gpus": gpus,
        "backend": task.backend_name,
        "operator": task.operator_name,
        "service": compute_task_service(
            backend=backend,
            assigned_nodes=task.assigned_nodes,
            ports=task.ports,
        ),
    }


def build_tasks_ctx(
    task_graph: TaskGraph,
    backends: dict[str, Backend],
    replica_names_by_base: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build expression context for tasks from the task graph."""

    tasks_ctx: dict[str, Any] = {}

    for task in task_graph.get_tasks():
        tasks_ctx[task.name] = build_task_info(task, backends)

    if replica_names_by_base:
        for base_name, replica_names in replica_names_by_base.items():
            if len(replica_names) > 1 or (
                len(replica_names) == 1 and replica_names[0] != base_name
            ):
                replica_list: list[dict[str, Any]] = []
                for replica_name in replica_names:
                    if replica_name in tasks_ctx:
                        replica_list.append(tasks_ctx[replica_name])
                    else:
                        replica_list.append(
                            {
                                "nodes": [],
                                "gpus": [],
                                "backend": None,
                                "operator": None,
                                "service": {"host": "", "port": "", "url": ""},
                            }
                        )
                tasks_ctx[base_name] = replica_list

    return tasks_ctx


TASK_EXPR_RE = re.compile(r"\$\{\{\s*(task\.[^}]+?)\s*\}\}")

TASK_AVAILABLE_ATTRS = ("nodes", "gpus", "backend", "operator", "service")
TASK_NODE_ATTRS = ("name", "ip_address", "index", "num_gpus")
TASK_SERVICE_ATTRS = ("host", "port", "url")


def extract_task_expressions(line: str) -> list[str]:
    """Extract ``${{ task.* }}`` expressions from a script line."""

    return ["${{ " + m.strip() + " }}" for m in TASK_EXPR_RE.findall(line)]


def build_task_expression_hint(
    task_exprs: list[str],
    tasks_ctx: dict[str, Any],
    replica_names_by_base: dict[str, list[str]] | None,
) -> str | None:
    """Return a human-readable hint for common task-expression resolution errors."""

    for expr in task_exprs:
        inner = expr.strip().removeprefix("${{").removesuffix("}}").strip()
        parts = inner.split(".")
        if len(parts) < 3 or parts[0] != "task":
            continue

        raw_task_ref = parts[1]
        bracket_match = re.match(r"(\w+)\[", raw_task_ref)
        has_index = bracket_match is not None
        task_ref = bracket_match.group(1) if bracket_match else raw_task_ref

        ctx_val = tasks_ctx.get(task_ref)

        if isinstance(ctx_val, list) and not has_index:
            rest = ".".join(parts[2:])
            replicas = (
                replica_names_by_base.get(task_ref, []) if replica_names_by_base else []
            )
            replica_display = ", ".join(replicas) if replicas else "N/A"
            return (
                f"'{task_ref}' is a replicated task with "
                f"{len(ctx_val)} replica(s). "
                "Use indexed access like "
                "${{ task."
                + task_ref
                + "[0]."
                + rest
                + " }}"
                + (
                    " or a full replica name like "
                    "${{ task." + replicas[0] + "." + rest + " }}"
                    if replicas
                    else ""
                )
                + f" (replicas: {replica_display})"
            )

        if ctx_val is not None:
            accessed_attr = parts[2].split("[")[0] if len(parts) > 2 else None
            if accessed_attr and accessed_attr not in TASK_AVAILABLE_ATTRS:
                hint = (
                    f"'{accessed_attr}' is not an available task attribute to resolve. "
                    f"Available attributes: {', '.join(TASK_AVAILABLE_ATTRS)}"
                )
                if accessed_attr == "nodes" or accessed_attr in TASK_NODE_ATTRS:
                    pass
                else:
                    hint += ". Each node exposes: " + ", ".join(TASK_NODE_ATTRS)
                return hint
            if accessed_attr == "nodes" and len(parts) > 3:
                node_attr = parts[3].split("[")[0]
                if node_attr not in TASK_NODE_ATTRS:
                    return (
                        f"'{node_attr}' is not an available node attribute. "
                        f"Available node attributes: "
                        f"{', '.join(TASK_NODE_ATTRS)}"
                    )
            if accessed_attr == "service" and len(parts) > 3:
                service_attr = parts[3].split("[")[0]
                if service_attr not in TASK_SERVICE_ATTRS:
                    return (
                        f"'{service_attr}' is not an available service attribute. "
                        f"Available service attributes: "
                        f"{', '.join(TASK_SERVICE_ATTRS)}"
                    )

        if ctx_val is None:
            available = [k for k, v in tasks_ctx.items() if not isinstance(v, list)]
            replicated = [k for k, v in tasks_ctx.items() if isinstance(v, list)]
            parts_hint = []
            if available:
                parts_hint.append("available tasks: " + ", ".join(sorted(available)))
            if replicated:
                parts_hint.append(
                    "replicated tasks (use index): " + ", ".join(sorted(replicated))
                )
            return f"Task '{task_ref}' is not defined. " + (
                "; ".join(parts_hint) if parts_hint else "No tasks found in context."
            )
    return None
