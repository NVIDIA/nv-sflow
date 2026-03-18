# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Application composition root.

This module is the single place where we wire together:
- validated config DTOs (sflow.config.schema)
- core runtime objects (sflow.core.*)
- concrete plugin implementations (sflow.plugins.*)
"""

from __future__ import annotations

import asyncio
import itertools
import math
import re
import shutil
from typing import Any, Literal

from sflow.config.resolver import ExpressionResolver
from sflow.config.schema import SflowConfig
from sflow.core.backend import Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.state import SflowState
from sflow.core.task import OutputSpec, RetryPolicy, Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.variable import Variable, VariableType
from sflow.core.workflow import Workflow
from sflow.logging import get_logger

resolver = ExpressionResolver()

_logger = get_logger(__name__)


def _build_task_info(
    task: Task,
    backends: dict[str, Backend],
) -> dict[str, Any]:
    """
    Build context info dict for a single task.
    """
    # Get the backend for this task to access node information
    backend = backends.get(task.backend_name) if task.backend_name else None
    alloc_nodes_by_name: dict[str, ComputeNode] = {}
    if backend and backend.allocation:
        alloc_nodes_by_name = {n.name: n for n in backend.allocation.nodes}

    # Build nodes list with full node info
    task_nodes: list[dict[str, Any]] = []
    for i, node_name_assigned in enumerate(task.assigned_nodes):
        node = alloc_nodes_by_name.get(node_name_assigned)
        if node:
            task_nodes.append(
                {
                    "name": node.name,
                    "ip_address": node.ip_address,
                    "index": i,  # Use index within task's assigned nodes
                    "num_gpus": node.num_gpus,
                }
            )
        else:
            # Fallback if node not found in allocation
            task_nodes.append(
                {
                    "name": node_name_assigned,
                    "ip_address": "",
                    "index": i,
                    "num_gpus": None,
                }
            )

    # Parse GPU indices from CUDA_VISIBLE_DEVICES env var
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
    }


def _build_tasks_ctx(
    task_graph: TaskGraph,
    backends: dict[str, Backend],
    replica_names_by_base: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Build expression context for tasks from the task graph.

    Provides access to task-specific information like:
    - nodes: List of nodes assigned to the task (with name, ip_address, index, num_gpus)
    - gpus: List of GPU indices assigned to the task

    Supports two access patterns:
    1. By full task name: ${{ task.my_task.nodes[0].ip_address }}
    2. By base name with replica index: ${{ task.my_task[0].nodes[0].ip_address }}

    For replicated tasks (e.g., prefill_server with 2 replicas -> prefill_server_0, prefill_server_1):
    - ${{ task.prefill_server_0.nodes[0].ip_address }} - access by full name
    - ${{ task.prefill_server[0].nodes[0].ip_address }} - access by base name + replica index
    """
    tasks_ctx: dict[str, Any] = {}

    # Build context for each task by its full name
    for task in task_graph.get_tasks():
        tasks_ctx[task.name] = _build_task_info(task, backends)

    # If we have replica mapping, also build indexed access by base task name
    if replica_names_by_base:
        for base_name, replica_names in replica_names_by_base.items():
            if len(replica_names) > 1 or (
                len(replica_names) == 1 and replica_names[0] != base_name
            ):
                # This is a replicated task - build a list for indexed access
                replica_list: list[dict[str, Any]] = []
                for replica_name in replica_names:
                    if replica_name in tasks_ctx:
                        replica_list.append(tasks_ctx[replica_name])
                    else:
                        # Fallback: empty info
                        replica_list.append(
                            {
                                "nodes": [],
                                "gpus": [],
                                "backend": None,
                                "operator": None,
                            }
                        )
                # Add the base name as a list for indexed access
                tasks_ctx[base_name] = replica_list

    return tasks_ctx


_TASK_EXPR_RE = re.compile(r"\$\{\{\s*(task\.[^}]+?)\s*\}\}")

_TASK_AVAILABLE_ATTRS = ("nodes", "gpus", "backend", "operator")
_TASK_NODE_ATTRS = ("name", "ip_address", "index", "num_gpus")


def _extract_task_expressions(line: str) -> list[str]:
    """Extract ``${{ task.* }}`` expressions from a script line."""
    return ["${{ " + m.strip() + " }}" for m in _TASK_EXPR_RE.findall(line)]


def _build_task_expression_hint(
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
            if accessed_attr and accessed_attr not in _TASK_AVAILABLE_ATTRS:
                hint = (
                    f"'{accessed_attr}' is not an available task attribute to resolve. "
                    f"Available attributes: {', '.join(_TASK_AVAILABLE_ATTRS)}"
                )
                if accessed_attr == "nodes" or accessed_attr in _TASK_NODE_ATTRS:
                    pass
                else:
                    hint += ". Each node exposes: " + ", ".join(_TASK_NODE_ATTRS)
                return hint
            if accessed_attr == "nodes" and len(parts) > 3:
                node_attr = parts[3].split("[")[0]
                if node_attr not in _TASK_NODE_ATTRS:
                    return (
                        f"'{node_attr}' is not an available node attribute. "
                        f"Available node attributes: "
                        f"{', '.join(_TASK_NODE_ATTRS)}"
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


def preflight_validate_backends(state: SflowState) -> None:
    """
    REQ-5.1 Pre-flight Validation (MVP):

    Validate backend prerequisites before making allocations / submitting work.

    For v0.1 Slurm usage this primarily means checking required Slurm CLI tools are available.
    """
    for b in (state.backends or {}).values():
        b_type = (
            getattr(getattr(b, "config", None), "type", None) or b.__class__.__name__
        )
        if str(b_type).lower() != "slurm":
            continue

        required = ["salloc", "srun", "scontrol", "scancel"]
        missing = [c for c in required if shutil.which(c) is None]
        if missing:
            raise ValueError(
                "Pre-flight validation failed for Slurm backend "
                f"'{getattr(b, 'name', 'unknown')}'. Missing required commands: "
                f"{', '.join(missing)}. "
                "Ensure Slurm client tools are installed and available on PATH (e.g., load the Slurm module)."
            )


def preflight_validate_container_images(config: SflowConfig, state: SflowState) -> None:
    """
    Validate container image references in srun operators before allocating cluster resources.

    Resolves ``${{ }}`` expressions using currently available variables (global variables
    are resolved at this point; workflow variables may not be) and checks that
    ``container_image`` values look like valid registry references or ``.sqsh`` paths.

    This runs before ``allocate_backends`` so that obviously invalid images are caught
    before ``salloc`` consumes cluster resources.
    """
    from sflow.plugins.operators.srun import _is_valid_container_image

    variables_ctx: dict[str, Any] = {
        name: var.value for name, var in (state.variables or {}).items()
    }
    ctx: dict[str, Any] = {"variables": variables_ctx, **variables_ctx}

    def _try_resolve(raw: Any) -> str:
        if raw is None:
            return ""
        try:
            return (
                str(resolver.resolve(raw, ctx))
                if resolver.has_expression(raw)
                else str(raw)
            )
        except Exception:
            return str(raw)

    _invalid_hint = (
        "Expected a remote registry reference (e.g. 'nvcr.io/org/image:tag') "
        "or a local .sqsh file path (e.g. '/path/to/image.sqsh')"
    )

    def _check_image(image_val: str, *, source: str) -> None:
        if not image_val:
            return
        if "${{" in image_val or "${" in image_val:
            return
        if not _is_valid_container_image(image_val):
            raise ValueError(
                f"Pre-flight validation failed: {source} has invalid container image. "
                f"{_invalid_hint}, got: '{image_val}'"
            )

    def _check_extra_args(extra_args: list, *, source: str) -> None:
        for i, arg in enumerate(extra_args):
            arg_str = str(arg)
            raw_val: str | None = None
            if arg_str.startswith("--container-image="):
                raw_val = arg_str.split("=", 1)[1]
            elif arg_str == "--container-image" and i + 1 < len(extra_args):
                raw_val = str(extra_args[i + 1])
            if raw_val is not None:
                _check_image(_try_resolve(raw_val), source=f"{source} extra_args")

    for op_conf in config.operators or []:
        if getattr(op_conf, "type", None) != "srun":
            continue
        raw_image = getattr(op_conf, "container_image", None)
        if raw_image is not None:
            _check_image(_try_resolve(raw_image), source=f"operator '{op_conf.name}'")
        extra_args = list(getattr(op_conf, "extra_args", None) or [])
        _check_extra_args(extra_args, source=f"operator '{op_conf.name}'")

    for t_conf in config.workflow.tasks or []:
        if t_conf.operator is None or isinstance(t_conf.operator, str):
            continue
        overrides = t_conf.operator.model_dump(exclude={"name"}, exclude_none=True)
        raw_image = overrides.get("container_image")
        if raw_image is not None:
            _check_image(
                _try_resolve(raw_image),
                source=f"task '{t_conf.name}' operator override",
            )
        override_extra = overrides.get("extra_args")
        if override_extra:
            _check_extra_args(
                list(override_extra),
                source=f"task '{t_conf.name}' operator override",
            )


def preflight_validate_task_graph(
    config: SflowConfig,
    state: SflowState,
    *,
    workspace_dir: Any | None = None,
    output_dir: Any | None = None,
) -> None:
    """
    Validate task planning against placeholder backend allocations before real allocation.

    This reuses the normal graph-building and GPU-packing logic with deterministic placeholder
    nodes so capacity/configuration errors surface before `salloc` consumes cluster resources.
    """
    planning_state = SflowState(
        workflow=Workflow(name=config.workflow.name, task_graph=TaskGraph()),
        variables=dict(state.variables),
        artifacts=dict(state.artifacts),
        backends=dict(state.backends),
        default_backend=state.default_backend,
    )
    original_allocations = {
        name: backend.allocation for name, backend in planning_state.backends.items()
    }
    try:
        planning_state = _seed_placeholder_backend_allocations(planning_state)
        planning_state = resolve_artifacts(
            config,
            planning_state,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            materialize=False,
        )
        planning_state = resolve_workflow_variables(
            config,
            planning_state,
            workspace_dir=workspace_dir,
        )
        build_task_graph(config, planning_state, workspace_dir=workspace_dir)
    finally:
        for name, backend in planning_state.backends.items():
            backend.allocation = original_allocations.get(name)


def _artifacts_ctx(
    state: SflowState,
) -> dict[str, Any]:
    """
    Build expression context for artifacts from resolved state.
    """
    return {name: a.to_context_dict() for name, a in (state.artifacts or {}).items()}


def resolve_artifacts(
    config: SflowConfig,
    state: SflowState,
    *,
    workspace_dir: Any | None = None,
    output_dir: Any | None = None,
    materialize: bool = False,
) -> SflowState:
    """
    Resolve artifact URIs (and inline content) into `state.artifacts`.

    - Artifact `uri` and `content` may contain `${{ ... }}` expressions.
    - `file://` / `fs://` artifacts resolve to absolute local paths.
    - Inline content (ArtifactConfig.content) is written only when materialize=True.
    - Backends context is included when backends are available in state.
    """
    from pathlib import Path

    from sflow.core.artifact import Artifact
    from sflow.core.artifact_registry import (
        ensure_builtin_artifacts_registered,
        get_artifact_resolver_for_uri,
    )

    ensure_builtin_artifacts_registered()

    ws_dir = Path(workspace_dir) if workspace_dir is not None else Path.cwd()
    out_dir = Path(output_dir) if output_dir is not None else ws_dir / "sflow_output"
    cache_dir = ws_dir / ".sflow_cache" / "artifacts"

    variables_ctx: dict[str, Any] = {
        name: var.value for name, var in (state.variables or {}).items()
    }
    backends_ctx: dict[str, Any] = {
        name: b.to_dict() for name, b in (state.backends or {}).items()
    }
    ctx: dict[str, Any] = {
        "variables": variables_ctx,
        "backends": backends_ctx,
        "workflow": {"name": config.workflow.name},
        **variables_ctx,
    }

    artifacts: dict[str, Artifact] = {}
    for a_conf in config.artifacts or []:
        uri_raw: Any = a_conf.uri
        uri = (
            str(resolver.resolve(uri_raw, ctx))
            if resolver.has_expression(uri_raw)
            else str(uri_raw)
        )

        content_raw: Any = a_conf.content
        content = (
            str(resolver.resolve(content_raw, ctx))
            if content_raw is not None and resolver.has_expression(content_raw)
            else (str(content_raw) if content_raw is not None else None)
        )

        resolver_obj = get_artifact_resolver_for_uri(uri)
        if resolver_obj is None:
            # Unknown scheme: keep parity with previous behavior by exposing URI as the "path".
            artifacts[a_conf.name] = Artifact(
                name=a_conf.name,
                uri=uri,
                description=a_conf.description,
                path=None,
            )
            continue

        art = resolver_obj.resolve(
            name=a_conf.name,
            uri=uri,
            description=a_conf.description,
            content=content,
            workspace_dir=ws_dir,
            cache_dir=cache_dir,
            output_dir=out_dir,
            materialize=materialize,
        )
        artifacts[art.name] = art

    state.artifacts = artifacts
    return state


def _seed_placeholder_backend_allocations(state: SflowState) -> SflowState:
    """
    Populate placeholder allocations for backends that have not been allocated yet.

    This is primarily used for dry-run / visualize flows so that expressions like:
      ${{ backends.<name>.nodes[0].ip_address }}
    can be resolved without making real allocation calls.
    """
    from sflow.core.backend import Allocation

    if not state.backends:
        return state

    for b in state.backends.values():
        if b.allocation is not None:
            continue

        # Best-effort infer node count from backend private attrs.
        nodes_count = getattr(b, "_nodes", None)
        try:
            nodes_count = int(nodes_count) if nodes_count is not None else 1
        except Exception:
            nodes_count = 1
        nodes_count = max(nodes_count, 1)

        num_gpus = getattr(b, "_gpu_per_node", None)
        try:
            num_gpus = int(num_gpus) if num_gpus is not None else None
        except Exception:
            num_gpus = None

        nodes: list[ComputeNode] = []
        for i in range(nodes_count):
            if b.__class__.__name__ == "LocalBackend" or b.name == "local":
                name = "localhost" if i == 0 else f"localhost-{i}"
                ip = "127.0.0.1"
            else:
                name = f"{b.name}-node{i}"
                # Deterministic placeholder IPs, only meant for rendering.
                ip = f"0.0.0.{i + 1}"
            nodes.append(
                ComputeNode(name=name, ip_address=ip, index=i, num_gpus=num_gpus)
            )

        b.allocation = Allocation(allocation_id="0", nodes=nodes, owned=False)

    return state


def _maybe_int(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return v
    return v


def _cast_variable_value(name: str, value: Any, var_type: VariableType) -> Any:
    if value is None:
        return None

    # Already typed (e.g. YAML int/bool) – leave it unless a cast is needed.
    if var_type == VariableType.STRING:
        return value if isinstance(value, str) else str(value)

    if var_type == VariableType.INTEGER:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError as e:
                raise ValueError(
                    f"Variable '{name}' expected integer, got '{value}'"
                ) from e
        raise ValueError(
            f"Variable '{name}' expected integer, got {type(value).__name__}"
        )

    if var_type == VariableType.FLOAT:
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as e:
                raise ValueError(
                    f"Variable '{name}' expected float, got '{value}'"
                ) from e
        raise ValueError(
            f"Variable '{name}' expected float, got {type(value).__name__}"
        )

    if var_type == VariableType.BOOLEAN:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "y", "on"}:
                return True
            if v in {"false", "0", "no", "n", "off"}:
                return False
            raise ValueError(f"Variable '{name}' expected boolean, got '{value}'")
        raise ValueError(
            f"Variable '{name}' expected boolean, got {type(value).__name__}"
        )

    # LIST/DICT: we don't attempt to parse serialized structures here.
    return value


def _resolve_and_update_variables(
    *,
    state: SflowState,
    variable_confs: list[Any],
    collision: Literal["overwrite", "error"] = "overwrite",
    extra_ctx: dict[str, Any] | None = None,
) -> SflowState:
    """
    Shared variable resolution engine used by global/workflow variable resolution.

    Args:
        state: SflowState to mutate
        variable_confs: list of pydantic VariableConfig-like objects (name/value/type/description/domain)
        collision: "overwrite" (global) or "error" (workflow)
        extra_ctx: extra context keys injected into the render context
    """

    if not variable_confs:
        return state

    variables: dict[str, Variable] = dict(state.variables)

    for v_conf in variable_confs:
        if collision == "error" and v_conf.name in variables:
            raise ValueError(
                f"Workflow variable '{v_conf.name}' conflicts with existing variable"
            )
        variables[v_conf.name] = Variable(
            name=v_conf.name,
            value=v_conf.value,
            description=v_conf.description,
            domain=v_conf.domain,
            type=VariableType(v_conf.type),
        )

    resolved_values: dict[str, Any] = {k: v.value for k, v in variables.items()}

    max_passes = len(variables) + 1
    for _ in range(max_passes):
        progress = False

        ctx_values = {
            k: v for k, v in resolved_values.items() if not resolver.has_expression(v)
        }
        ctx: dict[str, Any] = {"variables": ctx_values, **ctx_values}
        if extra_ctx:
            ctx.update(extra_ctx)

        for name, var in variables.items():
            current = var.value
            if not resolver.has_expression(current):
                continue

            try:
                new_value = resolver.resolve(current, ctx)
            except ValueError as e:
                err = str(e)
                if "Undefined variable" in err or "Error evaluating expression" in err:
                    continue
                raise

            if new_value != current:
                if not resolver.has_expression(new_value):
                    new_value = _cast_variable_value(name, new_value, var.type)
                var.value = new_value
                resolved_values[name] = new_value
                progress = True

        if not progress:
            break

    unresolved = [
        name for name, var in variables.items() if resolver.has_expression(var.value)
    ]
    if unresolved:
        raise ValueError(
            "Unresolved variable expressions (missing refs or cycle): "
            + ", ".join(sorted(unresolved))
        )

    for name, var in variables.items():
        var.value = _cast_variable_value(name, var.value, var.type)

    state.variables = variables
    return state


def resolve_global_variables(config: SflowConfig, state: SflowState) -> SflowState:
    """
    Resolve `${{ ... }}` expressions in `config.variables` and update `state.variables`.

    Resolution model:
    - Variables may reference other variables via `${{ variables.NAME }}` (and also `${{ NAME }}`).
    - We resolve iteratively until no progress is made.
    - If any variables still contain expressions after convergence, we fail fast (cycle or missing ref).
    - Finally, we best-effort cast values according to `Variable.type`.
    """

    return _resolve_and_update_variables(
        state=state,
        variable_confs=list(config.variables or []),
        collision="overwrite",
        extra_ctx=None,
    )


def resolve_backends(config: SflowConfig, state: SflowState) -> SflowState:
    """
    Resolve backend configuration using the current state context and populate `state.backends`.

    Today this mainly means:
    - Resolve `${{ ... }}` expressions inside backend configs using `state.variables`
    - Instantiate concrete backend plugins (e.g. `SlurmBackend`) and store them in `state.backends`

    Notes:
    - This function does not allocate resources (no subprocess calls).
    - Call `resolve_global_variables()` before this to ensure variables are concrete.
    """

    from sflow.core.backend_registry import (
        backend_config_type_adapter,
        ensure_builtin_backends_registered,
        get_backend_class,
    )

    ensure_builtin_backends_registered()

    # Build a simple context from resolved variables (values only)
    variables_ctx: dict[str, Any] = {
        name: var.value for name, var in (state.variables or {}).items()
    }
    ctx = {"variables": variables_ctx, **variables_ctx}

    backends: dict[str, Backend] = dict(state.backends or {})

    backend_confs = list(config.backends or [])
    if not backend_confs:
        # Default to a local backend when none are configured.
        backend_confs = [{"name": "local", "type": "local", "default": True}]

    bconf_adapter = backend_config_type_adapter()

    for b_conf in backend_confs:
        # Support dicts for the synthetic default backend above.
        if hasattr(b_conf, "model_dump"):
            b_conf_obj = b_conf
        else:
            b_conf_obj = bconf_adapter.validate_python(b_conf)

        backend_cls = get_backend_class(getattr(b_conf_obj, "type"))

        # Resolve expressions into a concrete backend config (backend-owned), then instantiate
        # the backend using config-only init.
        if hasattr(backend_cls, "resolve_config"):
            resolved_conf = backend_cls.resolve_config(  # type: ignore[attr-defined]
                b_conf_obj,
                resolver=resolver,
                ctx=ctx,
                workflow_name=config.workflow.name,
            )
        else:
            resolved_conf = b_conf_obj

        backend = backend_cls(resolved_conf)  # type: ignore[call-arg]

        backends[getattr(b_conf_obj, "name")] = backend
        if getattr(b_conf_obj, "default", False):
            state.default_backend = backend

    state.backends = backends
    if state.default_backend is None:
        state.default_backend = next(iter(backends.values()))
    return state


async def allocate_backends(state: SflowState) -> SflowState:
    """
    Allocate resources for all configured backends and store allocations on the backend objects.

    - Only allocates backends that are not already allocated (`backend.allocation is None`).
    - If any allocation fails, best-effort releases allocations made in this call.
    """

    if not state.backends:
        return state

    if state.default_backend is None:
        state.default_backend = next(iter(state.backends.values()))

    to_allocate = [b for b in state.backends.values() if b.allocation is None]
    if not to_allocate:
        return state

    tasks = [b.allocate_resources() for b in to_allocate]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    exc = next((r for r in results if isinstance(r, Exception)), None)
    if exc is not None:
        _logger.error(
            f"Backend allocation failed: {exc}. Releasing allocated resources..."
        )
        # Best-effort cleanup of anything allocated in this batch
        await asyncio.gather(
            *[b.release_resources() for b in to_allocate if b.allocation is not None],
            return_exceptions=True,
        )
        raise exc

    return state


async def release_backends(state: SflowState) -> SflowState:
    """
    Release resources for all allocated backends.

    Best-effort: releases all backends that currently have an allocation, and then
    re-raises the first exception (if any).
    """
    if not state.backends:
        return state

    to_release = [b for b in state.backends.values() if b.allocation is not None]
    if not to_release:
        return state

    results = await asyncio.gather(
        *[b.release_resources() for b in to_release],
        return_exceptions=True,
    )
    exc = next((r for r in results if isinstance(r, Exception)), None)
    if exc is not None:
        _logger.error(f"Backend release failed: {exc}")
        raise exc

    return state


def resolve_workflow_variables(
    config: SflowConfig, state: SflowState, *, workspace_dir: Any | None = None
) -> SflowState:
    """
    Resolve `${{ ... }}` expressions in `config.workflow.variables` and update `state.variables`.

    Semantics:
    - Workflow variables are resolved *after* global variables and may reference them.
    - Workflow variable names must not collide with existing `state.variables` entries.
    - Like global variables, we resolve iteratively and fail fast on missing refs / cycles.
    """

    backends_ctx: dict[str, Any] = {
        name: b.to_dict() for name, b in (state.backends or {}).items()
    }
    variables_ctx: dict[str, Any] = {
        name: var.value for name, var in (state.variables or {}).items()
    }
    # If caller constructed `state` manually (e.g. unit tests) without resolving artifacts,
    # populate artifacts from config so expressions like `${{ artifacts.NAME.path }}` work.
    if (not state.artifacts) and (config.artifacts):
        state = resolve_artifacts(
            config, state, workspace_dir=workspace_dir, materialize=False
        )
    artifacts_ctx = _artifacts_ctx(state)
    extra_ctx: dict[str, Any] = {
        "workflow": {"name": config.workflow.name},
        "backends": backends_ctx,
        "artifacts": artifacts_ctx,
        # Also provide variables context explicitly (in addition to the default ctx builder)
        "variables": variables_ctx,
        **variables_ctx,
    }

    return _resolve_and_update_variables(
        state=state,
        variable_confs=list(config.workflow.variables or []),
        collision="overwrite",
        extra_ctx=extra_ctx,
    )


def build_task_graph(
    config: SflowConfig, state: SflowState, *, workspace_dir: Any | None = None
) -> TaskGraph:
    """
    Build a TaskGraph from workflow task configs using the current (resolved) state.

    Requirements / assumptions:
    - `state.backends` must be populated (via `resolve_backends`).
    - Tasks are launched via Operators (operator-only execution model).
    - If backends have been allocated (`backend.allocation != None`), allocation info (job_id/nodelist)
      is injected into `srun` operator configs unless explicitly provided by the user.
    """

    # Import here to avoid plugin imports in core modules.
    from sflow.core.operator_registry import (
        ensure_builtin_operators_registered,
        get_operator_class,
        operator_config_type_adapter,
    )
    from sflow.core.probe import ProbeType
    from sflow.plugins.probes import (
        HttpGetProbe,
        HttpPostProbe,
        LogWatchProbe,
        TcpPortProbe,
    )

    if not state.backends:
        raise ValueError(
            "No backends are available in state; call resolve_backends first"
        )

    operator_confs = {o.name: o for o in (config.operators or [])}

    # Always enable operator mode: runtime abstraction is removed; operators are the only launch mechanism.
    ensure_builtin_operators_registered()
    operator_adapter = operator_config_type_adapter()

    # Context for resolving expressions (scripts/resources/etc.)
    variables_ctx: dict[str, Any] = {
        name: var.value for name, var in (state.variables or {}).items()
    }
    if (not state.artifacts) and (config.artifacts):
        state = resolve_artifacts(
            config, state, workspace_dir=workspace_dir, materialize=False
        )
    artifacts_ctx = _artifacts_ctx(state)
    backends_ctx: dict[str, Any] = {
        name: b.to_dict() for name, b in (state.backends or {}).items()
    }
    ctx: dict[str, Any] = {
        "variables": variables_ctx,
        "artifacts": artifacts_ctx,
        "backends": backends_ctx,
        "workflow": {"name": config.workflow.name},
        **variables_ctx,
    }

    def _resolve_value(v: Any) -> Any:
        """
        Resolve `${{ ... }}` expressions inside an arbitrary python value using the current ctx.

        We use this for operator configs since operator models may contain expression strings
        (e.g., container_image="${{ variables.IMG }}", extra_args=["--foo=${{ BAR }}"]).
        """
        if resolver.has_expression(v):
            return resolver.resolve(v, ctx)
        if isinstance(v, list):
            return [_resolve_value(x) for x in v]
        if isinstance(v, dict):
            return {k: _resolve_value(val) for k, val in v.items()}
        return v

    # Global GPU allocation cursor (planning-time): (backend_name, node_name) -> next free GPU index.
    # This is intentionally independent of execution parallelism; it models a one-time partitioning
    # of the available GPUs across tasks in graph construction order.
    gpu_next: dict[tuple[str, str], int] = {}
    # Note: sequential replicas still get distinct GPU slices; "sequential" only controls DAG ordering.

    def _resolve_str_list(values: list[Any]) -> list[str]:
        return [
            str(resolver.resolve(v, ctx)) if resolver.has_expression(v) else str(v)
            for v in values
        ]

    def _resolve_replica_count(task_name: str, count_raw: Any) -> int:
        if count_raw is None:
            raise ValueError(f"Task '{task_name}' replicas.count is None")
        resolved = (
            resolver.resolve(count_raw, ctx)
            if resolver.has_expression(count_raw)
            else count_raw
        )
        resolved = _maybe_int(resolved)
        if isinstance(resolved, bool):
            raise ValueError(
                f"Task '{task_name}' replicas.count must resolve to int, got boolean {resolved!r}"
            )
        if isinstance(resolved, int):
            if resolved <= 0:
                raise ValueError(
                    f"Task '{task_name}' replicas.count must be > 0, got {resolved}"
                )
            return resolved
        if isinstance(resolved, str):
            try:
                v = int(resolved)
            except ValueError as e:
                raise ValueError(
                    f"Task '{task_name}' replicas.count must resolve to int, got {resolved!r}"
                ) from e
            if v <= 0:
                raise ValueError(
                    f"Task '{task_name}' replicas.count must be > 0, got {v}"
                )
            return v
        raise ValueError(
            f"Task '{task_name}' replicas.count must resolve to int, got {type(resolved).__name__}"
        )

    def _resolve_replica_policy(task_name: str, policy_raw: Any) -> str:
        """
        Resolve replicas.policy which may be a concrete ReplicaPolicy/str or an expression string.
        """
        resolved = (
            resolver.resolve(policy_raw, ctx)
            if resolver.has_expression(policy_raw)
            else policy_raw
        )
        # Normalize enum-ish values to string
        if hasattr(resolved, "value"):
            resolved = getattr(resolved, "value")
        policy = str(resolved).strip().lower()
        if policy not in {"parallel", "sequential"}:
            raise ValueError(
                f"Task '{task_name}' replicas.policy must be 'parallel' or 'sequential', got {policy!r}"
            )
        return policy

    def _replica_sweep_instances(
        task_name: str, var_names: list[str]
    ) -> list[dict[str, Any]]:
        """
        Expand a replica sweep into per-replica variable assignments.

        Today we implement a simple cartesian product across domains:
        - variables: ["A", "B"] with domains [1,2] and ["x","y"] -> 4 replicas
        """
        if not var_names:
            return [{}]

        domains: list[list[Any]] = []
        for vn in var_names:
            v = (state.variables or {}).get(vn)
            if v is None:
                raise ValueError(
                    f"Task '{task_name}' replicas.variables references unknown variable '{vn}'"
                )
            if not v.domain:
                raise ValueError(
                    f"Task '{task_name}' replicas.variables requires variable '{vn}' to define a non-empty domain"
                )
            domains.append(list(v.domain))

        instances: list[dict[str, Any]] = []
        for combo in itertools.product(*domains):
            instances.append(
                {vn: val for vn, val in zip(var_names, combo, strict=True)}
            )
        return instances

    def _resolve_int(task_name: str, *, field: str, value: Any) -> int:
        resolved = (
            resolver.resolve(value, ctx) if resolver.has_expression(value) else value
        )
        resolved = _maybe_int(resolved)
        if isinstance(resolved, bool):
            raise ValueError(
                f"Task '{task_name}' {field} must resolve to int, got boolean {resolved!r}"
            )
        if isinstance(resolved, int):
            return resolved
        if isinstance(resolved, str):
            try:
                return int(resolved)
            except ValueError as e:
                raise ValueError(
                    f"Task '{task_name}' {field} must resolve to int, got {resolved!r}"
                ) from e
        raise ValueError(
            f"Task '{task_name}' {field} must resolve to int, got {type(resolved).__name__}"
        )

    def _build_probe(
        task_name: str,
        *,
        p_conf: Any,
        p_type: ProbeType,
        default_host: str | None = None,
    ):
        """
        Convert a ProbeConfig (from config schema) into a concrete Probe instance.
        """
        delay = int(getattr(p_conf, "delay", 0))
        timeout = _resolve_int(
            task_name, field=f"probes.{p_type}.timeout", value=p_conf.timeout
        )
        interval = _resolve_int(
            task_name, field=f"probes.{p_type}.interval", value=p_conf.interval
        )
        success_threshold = _resolve_int(
            task_name,
            field=f"probes.{p_type}.success_threshold",
            value=p_conf.success_threshold,
        )
        failure_threshold = _resolve_int(
            task_name,
            field=f"probes.{p_type}.failure_threshold",
            value=p_conf.failure_threshold,
        )

        if delay < 0:
            raise ValueError(f"Task '{task_name}' probes.{p_type}.delay must be >= 0")
        if timeout < 0:
            raise ValueError(f"Task '{task_name}' probes.{p_type}.timeout must be >= 0")
        if interval < 0:
            raise ValueError(
                f"Task '{task_name}' probes.{p_type}.interval must be >= 0"
            )
        if success_threshold <= 0:
            raise ValueError(
                f"Task '{task_name}' probes.{p_type}.success_threshold must be > 0"
            )
        if failure_threshold <= 0:
            raise ValueError(
                f"Task '{task_name}' probes.{p_type}.failure_threshold must be > 0"
            )

        common = dict(
            type=p_type,
            delay=delay,
            timeout=timeout,
            interval=interval,
            success_threshold=success_threshold,
            failure_threshold=failure_threshold,
        )

        if getattr(p_conf, "tcp_port", None) is not None:
            tcp = p_conf.tcp_port
            port = _resolve_int(
                task_name, field=f"probes.{p_type}.tcp_port.port", value=tcp.port
            )
            host_raw = getattr(tcp, "host", None)
            host = (
                str(resolver.resolve(host_raw, ctx))
                if host_raw is not None and resolver.has_expression(host_raw)
                else (
                    str(host_raw)
                    if host_raw is not None
                    else (default_host if default_host is not None else "127.0.0.1")
                )
            )
            on_node = getattr(tcp, "on_node", "first")
            return TcpPortProbe(host=host, port=port, on_node=on_node, **common)

        if getattr(p_conf, "http_get", None) is not None:
            http = p_conf.http_get
            url_raw = str(http.url)
            url = (
                str(resolver.resolve(url_raw, ctx))
                if resolver.has_expression(url_raw)
                else url_raw
            )
            return HttpGetProbe(
                url=url, headers=getattr(http, "headers", None), **common
            )

        if getattr(p_conf, "http_post", None) is not None:
            http = p_conf.http_post
            url_raw = str(http.url)
            url = (
                str(resolver.resolve(url_raw, ctx))
                if resolver.has_expression(url_raw)
                else url_raw
            )
            body_raw = getattr(http, "body", None)
            body = (
                str(resolver.resolve(body_raw, ctx))
                if body_raw is not None and resolver.has_expression(body_raw)
                else body_raw
            )
            return HttpPostProbe(
                url=url,
                headers=getattr(http, "headers", None),
                body=body,
                **common,
            )

        if getattr(p_conf, "log_watch", None) is not None:
            lw = p_conf.log_watch
            match_count = (
                _resolve_int(
                    task_name,
                    field=f"probes.{p_type}.log_watch.match_count",
                    value=lw.match_count,
                )
                if getattr(lw, "match_count", None) is not None
                else 1
            )
            return LogWatchProbe(
                regex_pattern=str(lw.regex_pattern),
                logger_task_name=getattr(lw, "logger", None),
                match_count=match_count,
                **common,
            )

        raise ValueError(
            f"Task '{task_name}' probes.{p_type} has no probe type configured"
        )

    def _resolve_int_list(
        task_name: str, *, field: str, values: list[Any]
    ) -> list[int]:
        out: list[int] = []
        for i, v in enumerate(values):
            out.append(_resolve_int(task_name, field=f"{field}[{i}]", value=v))
        return out

    def _assigned_nodelist(
        *,
        task_name: str,
        base_task_name: str,
        runtime_backend: Backend,
        replica_index: int,
        replica_policy: str,
        nodes_indices_raw: list[Any] | None,
        nodes_count_raw: Any | None,
        nodes_exclude_raw: Any | None,
        gpus_count_raw: Any | None,
    ) -> tuple[list[str], bool]:
        """
        Choose a subset of allocation nodes for this task replica.

        Rules:
        - If nodes.exclude is set: filter out those indices from the pool first.
        - If nodes.indices is set: select exactly those (post-exclude) indices.
        - Else if nodes.count is set: compact allocation slice.
          - parallel replicas: disjoint contiguous slices by replica_index
          - sequential replicas: reuse the first slice (replica_index ignored)
        - Else: use full allocation nodelist.
        """
        if runtime_backend.allocation is None:
            return [], False

        alloc_nodes = list(runtime_backend.allocation.nodes)
        if not alloc_nodes:
            return [], False

        if nodes_exclude_raw is not None:
            raw = (
                nodes_exclude_raw
                if isinstance(nodes_exclude_raw, list)
                else [nodes_exclude_raw]
            )
            exclude_indices = set(
                _resolve_int_list(
                    task_name, field="resources.nodes.exclude", values=raw
                )
            )
            out_of_range = {
                i for i in exclude_indices if i < 0 or i >= len(alloc_nodes)
            }
            if out_of_range:
                raise ValueError(
                    f"Task '{task_name}' resources.nodes.exclude contains index(es) "
                    f"{sorted(out_of_range)} out of range for {len(alloc_nodes)} allocated node(s) "
                    f"(valid: 0..{len(alloc_nodes) - 1})"
                )
            alloc_nodes = [
                n for i, n in enumerate(alloc_nodes) if i not in exclude_indices
            ]
            if not alloc_nodes:
                raise ValueError(
                    f"Task '{task_name}' resources.nodes.exclude removed all nodes from the pool"
                )

        if nodes_indices_raw is not None and nodes_count_raw is not None:
            raise ValueError(
                f"Task '{task_name}' resources.nodes cannot set both 'indices' and 'count'"
            )

        if nodes_indices_raw is not None:
            indices = _resolve_int_list(
                task_name,
                field="resources.nodes.indices",
                values=list(nodes_indices_raw),
            )
            chosen: list[str] = []
            for idx in indices:
                if idx < 0 or idx >= len(alloc_nodes):
                    raise ValueError(
                        f"Task '{task_name}' resources.nodes.indices contains out-of-range index {idx}; "
                        f"allocation has {len(alloc_nodes)} nodes"
                    )
                chosen.append(alloc_nodes[idx].name)
            return chosen, False

        if nodes_count_raw is not None:
            count = _resolve_int(
                task_name, field="resources.nodes.count", value=nodes_count_raw
            )
            if count <= 0:
                raise ValueError(
                    f"Task '{task_name}' resources.nodes.count must be > 0, got {count}"
                )
            start = 0 if replica_policy == "sequential" else replica_index * count
            end = start + count
            if end > len(alloc_nodes):
                raise ValueError(
                    f"Task '{task_name}' needs {count} nodes (replica_index={replica_index}, policy={replica_policy}), "
                    f"but allocation has only {len(alloc_nodes)} nodes"
                )
            return [n.name for n in alloc_nodes[start:end]], False

        # If nodes are not explicitly requested but GPUs are, first try to "pack" the task onto
        # a single allocation node that still has enough remaining GPUs.
        #
        # This enables global GPU sharing across workflow tasks. Example:
        # - Node0 has 4 GPUs
        # - Task A requests 2 GPUs -> gets CUDA_VISIBLE_DEVICES=0,1 on node0
        # - Task B requests 2 GPUs -> gets CUDA_VISIBLE_DEVICES=2,3 on node0
        #
        # We use the same planning-time cursor as CUDA_VISIBLE_DEVICES assignment (`gpu_next`)
        # so the chosen node reflects already-reserved GPU slices from earlier tasks/replicas.
        if gpus_count_raw is not None and runtime_backend.allocation is not None:
            gpus_needed = _resolve_int(
                task_name, field="resources.gpus.count", value=gpus_count_raw
            )
            if gpus_needed <= 0:
                raise ValueError(
                    f"Task '{task_name}' resources.gpus.count must be > 0, got {gpus_needed}"
                )

            alloc_nodes_by_name = {n.name: n for n in runtime_backend.allocation.nodes}
            for node in alloc_nodes:
                n = alloc_nodes_by_name.get(node.name)
                if n is None or getattr(n, "num_gpus", None) is None:
                    continue
                try:
                    cap = int(n.num_gpus)
                except Exception:
                    continue
                if cap <= 0:
                    continue

                cursor_key = (runtime_backend.name, node.name)
                start = gpu_next.get(cursor_key, 0)
                if start + gpus_needed <= cap:
                    # Pin to a single node; CUDA_VISIBLE_DEVICES will allocate a slice later.
                    return [node.name], True

        # If nodes are not explicitly requested but GPUs are, we can infer a minimum node count
        # to satisfy the request based on per-node GPU capacity (when known).
        #
        # Example: 2 nodes x 4 GPUs each, request gpus.count=8 -> allocate 2 nodes.
        if gpus_count_raw is not None and runtime_backend.allocation is not None:
            alloc_nodes_by_name = list(runtime_backend.allocation.nodes)
            caps = [
                int(n.num_gpus)
                for n in alloc_nodes_by_name
                if getattr(n, "num_gpus", None) is not None
            ]
            if caps:
                gpus_needed = _resolve_int(
                    task_name, field="resources.gpus.count", value=gpus_count_raw
                )
                per_node_cap = min(caps)
                if per_node_cap > 0 and gpus_needed > per_node_cap:
                    if gpus_needed % per_node_cap != 0:
                        raise ValueError(
                            f"Task '{task_name}' requests {gpus_needed} GPUs; automatic multi-node expansion requires "
                            f"gpus.count to be a multiple of per-node GPU capacity ({per_node_cap}). "
                            f"Set resources.nodes.count/indices explicitly to override."
                        )
                    nodes_needed = math.ceil(gpus_needed / per_node_cap)

                    # Prefer selecting nodes based on remaining GPU capacity (via `gpu_next`) so we
                    # don't place a multi-node task onto nodes whose GPUs were already reserved by
                    # earlier tasks/replicas.
                    #
                    # Additionally, since CUDA_VISIBLE_DEVICES is computed as a uniform slice across
                    # the assigned nodes (per-node env), we require the selected nodes to share the
                    # same planning cursor (gpu_next) value.
                    alloc_nodes_map = {n.name: n for n in alloc_nodes_by_name}
                    candidates: list[tuple[str, int, int]] = []  # (name, cap, cursor)
                    for node in alloc_nodes:
                        n = alloc_nodes_map.get(node.name)
                        if n is None or getattr(n, "num_gpus", None) is None:
                            continue
                        try:
                            cap = int(n.num_gpus)
                        except Exception:
                            continue
                        cursor = gpu_next.get((runtime_backend.name, node.name), 0)
                        if cap <= 0:
                            continue
                        if cursor + per_node_cap <= cap:
                            candidates.append((node.name, cap, cursor))

                    if candidates:
                        # Group by cursor and pick the smallest cursor group with enough nodes.
                        cursors = sorted({c for _, _, c in candidates})
                        for c in cursors:
                            names = [n for (n, _cap, cur) in candidates if cur == c]
                            if len(names) >= nodes_needed:
                                return names[:nodes_needed], True

                    # Fallback: original contiguous slice behavior.
                    start = (
                        0
                        if replica_policy == "sequential"
                        else replica_index * nodes_needed
                    )
                    end = start + nodes_needed
                    if end > len(alloc_nodes):
                        raise ValueError(
                            f"Task '{task_name}' requests {gpus_needed} GPUs which requires {nodes_needed} nodes "
                            f"(replica_index={replica_index}, policy={replica_policy}), "
                            f"but allocation has only {len(alloc_nodes)} nodes"
                        )
                    return [n.name for n in alloc_nodes[start:end]], True

        # Default multi-task behavior:
        # If multiple *base tasks* target the same backend and did not request nodes explicitly,
        # we round-robin assign a single allocation node per base task to reduce unintended
        # cross-task sharing of all nodes (while still allowing per-replica GPU offsets).
        default_nodes = default_task_nodes.get((runtime_backend.name, base_task_name))
        if default_nodes is not None:
            return list(default_nodes), False

        return [n.name for n in alloc_nodes], False

    def _cuda_visible_devices(
        *,
        task_name: str,
        base_task_name: str,
        runtime_backend: Backend,
        assigned_nodes: list[str],
        nodes_are_pinned: bool,
        replica_index: int,
        replica_policy: str,
        gpus_count_raw: Any | None,
    ) -> str | None:
        if gpus_count_raw is None:
            return None
        count = _resolve_int(
            task_name, field="resources.gpus.count", value=gpus_count_raw
        )
        if count <= 0:
            raise ValueError(
                f"Task '{task_name}' resources.gpus.count must be > 0, got {count}"
            )

        def _backend_gpu_state_summary() -> str:
            if not runtime_backend.allocation:
                return ""
            gpu_nodes: list[tuple[str, int]] = []
            for node in runtime_backend.allocation.nodes:
                num_gpus = getattr(node, "num_gpus", None)
                if num_gpus is None:
                    continue
                try:
                    gpu_nodes.append((node.name, int(num_gpus)))
                except Exception:
                    continue
            if not gpu_nodes:
                return ""

            caps = [cap for _name, cap in gpu_nodes]
            total_capacity = sum(caps)
            total_allocated = sum(
                min(gpu_next.get((runtime_backend.name, n_name), 0), cap)
                for n_name, cap in gpu_nodes
            )
            total_remaining = total_capacity - total_allocated
            if len(set(caps)) == 1:
                per_node_str = f"gpus_per_node={caps[0]}"
            else:
                per_node_str = f"per_node_capacities={caps}"
            return (
                f"backend_gpu_state=(nodes={len(gpu_nodes)}, {per_node_str}, "
                f"total_capacity={total_capacity}, already_allocated={total_allocated}, "
                f"remaining={total_remaining})"
            )

        # Planning-time global GPU allocation (single-node case):
        # If we know the node GPU capacity, allocate a non-overlapping slice across tasks.
        if runtime_backend.allocation and assigned_nodes and len(assigned_nodes) == 1:
            n_name = assigned_nodes[0]
            alloc_nodes_by_name = {n.name: n for n in runtime_backend.allocation.nodes}
            n = alloc_nodes_by_name.get(n_name)
            if n is not None and getattr(n, "num_gpus", None) is not None:
                cap = int(n.num_gpus)
                if cap <= 0:
                    raise ValueError(
                        f"Task '{task_name}' cannot allocate GPUs on node '{n_name}' with non-positive capacity {cap}"
                    )

                cursor_key = (runtime_backend.name, n_name)
                start = gpu_next.get(cursor_key, 0)
                if start + count > cap:
                    available = cap - start
                    still_needed = count - available
                    backend_gpu_state = _backend_gpu_state_summary()
                    raise ValueError(
                        f"Task '{task_name}' requests {count} GPUs on node '{n_name}', but only {available} GPUs "
                        f"remain available (total_capacity={cap}, already_allocated={start}, "
                        f"still_needed={still_needed})"
                        f"{', ' + backend_gpu_state if backend_gpu_state else ''}. "
                        f"Consider increasing backend nodes or reducing concurrent GPU requests."
                    )

                # CUDA_VISIBLE_DEVICES is evaluated per-node, so indices must be local to that node.
                slice_str = ",".join(str(i) for i in range(start, start + count))
                gpu_next[cursor_key] = start + count
                return slice_str

        # Multi-node GPU request: interpret `count` as a total GPU request across assigned nodes,
        # and expose an even-ish per-node slice via CUDA_VISIBLE_DEVICES (since env is per-node).
        if runtime_backend.allocation and assigned_nodes and len(assigned_nodes) > 1:
            alloc_nodes_by_name = {n.name: n for n in runtime_backend.allocation.nodes}
            caps: list[int] = []
            for n_name in assigned_nodes:
                n = alloc_nodes_by_name.get(n_name)
                if n is None or getattr(n, "num_gpus", None) is None:
                    caps = []
                    break
                caps.append(int(n.num_gpus))

            if caps:
                total_cap = sum(caps)
                if count > total_cap:
                    backend_gpu_state = _backend_gpu_state_summary()
                    raise ValueError(
                        f"Task '{task_name}' requests {count} GPUs but assigned nodes have only {total_cap} GPUs total"
                        f"{', ' + backend_gpu_state if backend_gpu_state else ''}"
                    )
                per_node = min(min(caps), math.ceil(count / len(assigned_nodes)))
                if per_node <= 0:
                    raise ValueError(
                        f"Task '{task_name}' resources.gpus.count must be > 0, got {count}"
                    )

                # Reserve per-node GPU slices in the global planning cursor so later tasks
                # don't accidentally "pack" onto nodes that are already fully consumed by
                # this multi-node request.
                cursor_keys = [
                    (runtime_backend.name, n_name) for n_name in assigned_nodes
                ]
                starts = [gpu_next.get(k, 0) for k in cursor_keys]
                if len(set(starts)) != 1:
                    raise ValueError(
                        f"Task '{task_name}' requests GPUs across multiple nodes, but the nodes have different "
                        f"already-allocated GPU cursors {starts}. Pin nodes explicitly to avoid ambiguity."
                    )
                start0 = starts[0]
                for n_name, cap in zip(assigned_nodes, caps, strict=True):
                    if start0 + per_node > cap:
                        available = cap - start0
                        still_needed = per_node - available
                        backend_gpu_state = _backend_gpu_state_summary()
                        raise ValueError(
                            f"Task '{task_name}' requests {per_node} GPUs per node starting at {start0} on node "
                            f"'{n_name}', but only {available} GPUs remain available "
                            f"(total_capacity={cap}, already_allocated={start0}, still_needed={still_needed})"
                            f"{', ' + backend_gpu_state if backend_gpu_state else ''}."
                        )
                for k in cursor_keys:
                    gpu_next[k] = start0 + per_node

                return ",".join(str(i) for i in range(start0, start0 + per_node))

        # If the replica is pinned to specific nodes (indices/count), assume it has its own node slice;
        # GPU indices should be local within each node (replica_index-based offset).
        start = replica_index * count

        # Validate against per-node GPU capacity if known.
        if runtime_backend.allocation and assigned_nodes:
            alloc_nodes_by_name = {n.name: n for n in runtime_backend.allocation.nodes}
            for n_name in assigned_nodes:
                n = alloc_nodes_by_name.get(n_name)
                if n is None:
                    continue
                if getattr(n, "num_gpus", None) is None:
                    continue
                if start + count > int(n.num_gpus):
                    backend_gpu_state = _backend_gpu_state_summary()
                    raise ValueError(
                        f"Task '{task_name}' requests GPUs [{start}..{start + count - 1}] "
                        f"but node '{n_name}' has only {n.num_gpus} GPUs"
                        f"{', ' + backend_gpu_state if backend_gpu_state else ''}."
                    )

        return ",".join(str(i) for i in range(start, start + count))

    def _job_and_nodelist(backend: Backend) -> tuple[str, list[str]]:
        if backend.allocation:
            job_id = str(backend.allocation.allocation_id or "0")
            nodelist = [n.name for n in backend.allocation.nodes]
            return job_id, nodelist
        return "0", []

    task_graph = TaskGraph()

    # ---------------------------------------------------------------------
    # Replica planning: expand base tasks into concrete DAG nodes
    # ---------------------------------------------------------------------
    # base task name -> list of concrete node names (replicas)
    replica_names_by_base: dict[str, list[str]] = {}
    # concrete node name -> per-replica env mappings (stringified)
    replica_envs: dict[str, dict[str, str]] = {}
    # base task name -> replica policy ("parallel" / "sequential")
    replica_policy_by_base: dict[str, str] = {}

    for t_conf in config.workflow.tasks:
        if not t_conf.replicas:
            replica_names_by_base[t_conf.name] = [t_conf.name]
            replica_envs[t_conf.name] = {"SFLOW_REPLICA_INDEX": "0"}
            replica_policy_by_base[t_conf.name] = "parallel"
            continue

        r = t_conf.replicas
        policy = _resolve_replica_policy(t_conf.name, r.policy)
        replica_policy_by_base[t_conf.name] = policy

        sweep_vars = list(r.variables or [])
        instances = (
            _replica_sweep_instances(t_conf.name, sweep_vars) if sweep_vars else []
        )

        if r.count is not None:
            count = _resolve_replica_count(t_conf.name, r.count)
            if instances and count != len(instances):
                raise ValueError(
                    f"Task '{t_conf.name}' replicas.count={count} does not match sweep size {len(instances)} "
                    f"derived from replicas.variables={sweep_vars}"
                )
            if not instances:
                instances = [{} for _ in range(count)]
        else:
            # If count is omitted and no sweep vars are specified, default to 1 replica.
            if not instances:
                instances = [{}]

        # Generate replica names based on sweep variable values or numeric index
        def _make_replica_name(
            base_name: str, idx: int, instance: dict[str, Any], sweep_vars: list[str]
        ) -> str:
            """Generate replica name from variable values (if sweep) or numeric index."""
            if sweep_vars and instance:
                # Use variable values in the order they appear in sweep_vars
                value_parts = []
                for var_name in sweep_vars:
                    if var_name in instance:
                        val = instance[var_name]
                        # Sanitize value for use in task name (replace problematic chars)
                        val_str = (
                            str(val)
                            .replace(".", "_")
                            .replace("-", "_")
                            .replace(" ", "_")
                        )
                        value_parts.append(val_str)
                if value_parts:
                    return f"{base_name}_{'_'.join(value_parts)}"
            # Fallback to numeric index when no sweep variables
            return f"{base_name}_{idx}"

        concrete_names = [
            _make_replica_name(t_conf.name, i, instances[i], sweep_vars)
            for i in range(len(instances))
        ]
        replica_names_by_base[t_conf.name] = concrete_names
        for i, node_name in enumerate(concrete_names):
            env: dict[str, str] = {"SFLOW_REPLICA_INDEX": str(i)}
            for k, v in instances[i].items():
                env[k] = str(v)
            replica_envs[node_name] = env

    # Track sweep variable names per replica (for dry-run display)
    # concrete node name -> list of sweep variable names
    replica_sweep_vars: dict[str, list[str]] = {}
    for t_conf in config.workflow.tasks:
        if t_conf.replicas and t_conf.replicas.variables:
            sweep_var_names = list(t_conf.replicas.variables)
            for node_name in replica_names_by_base.get(t_conf.name, []):
                replica_sweep_vars[node_name] = sweep_var_names

    # ---------------------------------------------------------------------
    # Default node assignment: tasks without explicit resource definitions
    # use ALL nodes from the backend allocation.
    # ---------------------------------------------------------------------
    # (backend_name, base_task_name) -> list[str]
    # This dict remains empty so _assigned_nodelist returns full allocation.
    default_task_nodes: dict[tuple[str, str], list[str]] = {}

    # First pass: add task nodes
    for t_conf in config.workflow.tasks:
        base = t_conf.name
        concrete_nodes = replica_names_by_base.get(base, [base])
        if not concrete_nodes:
            raise ValueError(f"Task '{base}' produced zero replicas")

        for idx, node_name in enumerate(concrete_nodes):
            replica_policy = replica_policy_by_base.get(base, "parallel")

            # Resolve backend reference
            if isinstance(t_conf.backend, str):
                backend = state.backends.get(t_conf.backend)
            elif t_conf.backend is None:
                backend = state.default_backend or next(iter(state.backends.values()))
            else:
                raise NotImplementedError(
                    f"Inline backend overrides are not supported yet for task '{t_conf.name}'"
                )

            if backend is None:
                raise ValueError(f"Task '{t_conf.name}' references unknown backend")

            # Apply resources -> runtime node subset + CUDA_VISIBLE_DEVICES env
            nodes_indices_raw = None
            nodes_count_raw = None
            nodes_exclude_raw = None
            gpus_count_raw = None
            if t_conf.resources:
                if t_conf.resources.nodes:
                    nodes_indices_raw = t_conf.resources.nodes.indices
                    nodes_count_raw = t_conf.resources.nodes.count
                    nodes_exclude_raw = t_conf.resources.nodes.exclude
                if t_conf.resources.gpus:
                    gpus_count_raw = t_conf.resources.gpus.count

            assigned_nodes, nodes_inferred_from_gpus = _assigned_nodelist(
                task_name=node_name,
                base_task_name=base,
                runtime_backend=backend,
                replica_index=idx,
                replica_policy=replica_policy,
                nodes_indices_raw=nodes_indices_raw,
                nodes_count_raw=nodes_count_raw,
                nodes_exclude_raw=nodes_exclude_raw,
                gpus_count_raw=gpus_count_raw,
            )

            # Resolve operator (operator-only execution model).
            task_operator = None
            operator_name: str | None = None
            operator_overrides: dict[str, Any] = {}
            if t_conf.operator:
                if isinstance(t_conf.operator, str):
                    operator_name = t_conf.operator
                else:
                    operator_name = t_conf.operator.name
                    operator_overrides = dict(
                        t_conf.operator.model_dump(exclude={"name"}, exclude_none=True)
                    )

                base_op = operator_confs.get(operator_name)
                if base_op is None:
                    raise ValueError(
                        f"Task '{t_conf.name}' references unknown operator '{operator_name}'"
                    )

                merged = base_op.model_dump()
                if "extra_args" in operator_overrides and merged.get("extra_args"):
                    operator_overrides["extra_args"] = list(
                        merged["extra_args"]
                    ) + list(operator_overrides["extra_args"])
                merged.update(operator_overrides)
                merged["name"] = operator_name
                merged = _resolve_value(merged)
                op_conf = operator_adapter.validate_python(merged)
            else:
                # Default operator is backend-owned behavior.
                operator_name = f"default_{backend.name}"
                task_operator = backend.default_operator(
                    name=operator_name,
                    assigned_nodes=assigned_nodes,
                )
                op_conf = task_operator.config

            # Inject allocation info into srun operators unless explicitly configured.
            if getattr(op_conf, "type", None) == "srun":
                job_id, full_nodelist = _job_and_nodelist(backend)
                effective_nodelist = assigned_nodes if assigned_nodes else full_nodelist
                if getattr(op_conf, "job_id", None) in (None, "", "0"):
                    setattr(op_conf, "job_id", job_id)
                if not getattr(op_conf, "nodelist", None):
                    setattr(op_conf, "nodelist", list(effective_nodelist))
                if getattr(op_conf, "nodes", None) in (None, 0):
                    setattr(op_conf, "nodes", len(effective_nodelist))

                # Auto-add container mounts for local (fs:// / file://) artifacts when using Pyxis containers.
                #
                # Motivation: artifact env vars (e.g. ${MODEL_DIR}) resolve to absolute host paths.
                # When running inside a container, those host paths must be mounted so the same
                # path is visible inside the container on compute nodes.
                container_image = getattr(op_conf, "container_image", None)
                container_name = getattr(op_conf, "container_name", None)
                if container_image or container_name:
                    from pathlib import Path
                    from urllib.parse import urlparse

                    def _mount_key(mount: str) -> tuple[str, str] | None:
                        # Format: "/host:/ctr" or "/host:/ctr:rw"
                        parts = str(mount).split(":", 2)
                        if len(parts) < 2:
                            return None
                        return (parts[0], parts[1])

                    existing_mounts = list(
                        getattr(op_conf, "container_mounts", None) or []
                    )
                    existing_keys: set[tuple[str, str]] = set()
                    for m in existing_mounts:
                        k = _mount_key(m)
                        if k is not None:
                            existing_keys.add(k)

                    auto_mounts: list[str] = []
                    for art in (state.artifacts or {}).values():
                        try:
                            scheme = (
                                urlparse(str(getattr(art, "uri", ""))).scheme or ""
                            ).lower()
                        except Exception:
                            scheme = ""
                        if scheme not in {"fs", "file"}:
                            continue
                        apath = getattr(art, "path", None)
                        if apath is None:
                            continue

                        p = Path(apath)
                        # `.sqsh` artifacts are commonly used as container images/layers; do not
                        # auto-mount them (users can explicitly mount if desired).
                        if str(p).lower().endswith(".sqsh"):
                            continue
                        # For file-like artifacts, prefer mounting the directory containing the file
                        # so the file path stays valid inside the container.
                        mount_src = p
                        try:
                            if p.exists() and p.is_file():
                                mount_src = p.parent
                            elif (not p.exists()) and p.suffix:
                                mount_src = p.parent
                        except Exception:
                            # Best-effort only; fall back to mounting p itself.
                            mount_src = p

                        src = str(mount_src)
                        dst = src  # keep paths identical inside container
                        key = (src, dst)
                        if key in existing_keys:
                            continue
                        # Default to rw to avoid surprising write failures for directory artifacts.
                        auto_mounts.append(f"{src}:{dst}:rw")
                        existing_keys.add(key)

                    if auto_mounts:
                        setattr(
                            op_conf, "container_mounts", existing_mounts + auto_mounts
                        )

            if task_operator is None:
                operator_cls = get_operator_class(op_conf.type)
                task_operator = operator_cls(op_conf)  # type: ignore[arg-type]

            nodes_are_pinned = (nodes_indices_raw is not None) or (
                nodes_count_raw is not None
            )
            if nodes_inferred_from_gpus:
                nodes_are_pinned = True
            cuda_visible = _cuda_visible_devices(
                task_name=node_name,
                base_task_name=base,
                runtime_backend=backend,
                assigned_nodes=assigned_nodes,
                nodes_are_pinned=nodes_are_pinned,
                replica_index=idx,
                replica_policy=replica_policy,
                gpus_count_raw=gpus_count_raw,
            )

            task_logger = get_logger(f"sflow.task.{node_name}")
            task_logger.propagate = False

            # Resolve `${{ ... }}` expressions inside task scripts using the current context.
            # Note: we intentionally do NOT expand `$FOO` style shell variables here; those are
            # handled by `task.envs` + the shell at runtime.
            # Note: `${{ task.* }}` expressions are resolved in a second pass after all tasks are
            # built (see below).
            script = [
                str(resolver.resolve(line, ctx))
                if resolver.has_expression(line) and "task." not in line
                else line
                for line in list(t_conf.script)
            ]
            task = Task(
                name=node_name,
                logger=task_logger,
                operator=task_operator,
                status=TaskStatus.INITIATED,
                script=script,
            )
            task.assigned_nodes = list(assigned_nodes or [])
            task.operator_name = operator_name
            task.sweep_variables = replica_sweep_vars.get(node_name, [])

            # Build SFLOW_TASK_ASSIGNED_NODE_NAMES and SFLOW_TASK_ASSIGNED_NODE_IPS env vars
            # These provide easy access to the task's assigned nodes in scripts
            if assigned_nodes and backend and backend.allocation:
                alloc_nodes_by_name = {n.name: n for n in backend.allocation.nodes}
                node_names: list[str] = []
                node_ips: list[str] = []
                for n_name in assigned_nodes:
                    node_names.append(n_name)
                    node_obj = alloc_nodes_by_name.get(n_name)
                    if node_obj:
                        node_ips.append(node_obj.ip_address)
                    else:
                        node_ips.append("")  # Fallback if node not found
                task.envs["SFLOW_TASK_ASSIGNED_NODE_NAMES"] = ",".join(node_names)
                task.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] = ",".join(node_ips)
            # Outputs (MVP): store parse patterns to be evaluated from the task log after completion.
            if getattr(t_conf, "outputs", None):
                for o in list(t_conf.outputs or []):
                    # source is kept for schema parity; MVP parses from merged log file.
                    task.output_specs.append(
                        OutputSpec(
                            pattern=str(o.pattern),
                            source=str(getattr(o, "source", "stdout")),
                        )
                    )
            # Probes (readiness/failure)
            if t_conf.probes:
                # Default probe host: use the task's assigned node IP (not localhost),
                # so probes can reach services running on remote nodes (e.g. Slurm).
                default_probe_host: str | None = None
                try:
                    alloc = getattr(backend, "allocation", None)
                    if alloc and getattr(alloc, "nodes", None):
                        by_name = {n.name: n.ip_address for n in alloc.nodes}
                        if assigned_nodes:
                            default_probe_host = by_name.get(assigned_nodes[0])
                        if default_probe_host is None:
                            default_probe_host = alloc.nodes[0].ip_address
                except Exception:
                    default_probe_host = None

                if t_conf.probes.readiness is not None:
                    task.probes.append(
                        _build_probe(
                            node_name,
                            p_conf=t_conf.probes.readiness,
                            p_type=ProbeType.READINESS,
                            default_host=default_probe_host,
                        )
                    )
                if t_conf.probes.failure is not None:
                    task.probes.append(
                        _build_probe(
                            node_name,
                            p_conf=t_conf.probes.failure,
                            p_type=ProbeType.FAILURE,
                            default_host=default_probe_host,
                        )
                    )
            task.backend_name = backend.name
            # Optional retry policy (REQ-3.6).
            if t_conf.retries:
                retry_count = _resolve_int(
                    node_name, field="retries.count", value=t_conf.retries.count
                )
                retry_interval = _resolve_int(
                    node_name, field="retries.interval", value=t_conf.retries.interval
                )
                retry_backoff = _resolve_int(
                    node_name,
                    field="retries.backoff",
                    value=t_conf.retries.backoff,
                )
                if retry_count < 0:
                    raise ValueError(
                        f"Task '{node_name}' retries.count must be >= 0, got {retry_count}"
                    )
                if retry_interval < 0:
                    raise ValueError(
                        f"Task '{node_name}' retries.interval must be >= 0, got {retry_interval}"
                    )
                if retry_backoff < 1:
                    raise ValueError(
                        f"Task '{node_name}' retries.backoff must be >= 1, got {retry_backoff}"
                    )
                task.retries = RetryPolicy(
                    count=int(retry_count),
                    interval=float(retry_interval),
                    backoff=float(retry_backoff),
                )
            # Inject all resolved variables into task env by default (SRD intent).
            # Replica envs (including sweep vars) override global variables.
            task.envs.update(
                {k: str(v.value) for k, v in (state.variables or {}).items()}
            )
            # Also inject artifact paths as env vars (SRD REQ-1.5: `${NAME}` convenience).
            for aname, ainfo in (artifacts_ctx or {}).items():
                apath = ainfo.get("path")
                if apath is not None:
                    task.envs.setdefault(aname, str(apath))
            task.envs.update(replica_envs.get(node_name, {}))
            if cuda_visible is not None:
                task.envs["CUDA_VISIBLE_DEVICES"] = cuda_visible
            task_graph.dag.add_node(node_name, task)

            # If the task is replicated sequentially, enforce replica order by chaining edges.
            if (
                t_conf.replicas
                and replica_policy_by_base.get(base) == "sequential"
                and idx > 0
            ):
                task_graph.dag.add_edge(concrete_nodes[idx - 1], node_name)

    # Second pass: add edges
    for t_conf in config.workflow.tasks:
        if t_conf.depends_on:
            for dep in t_conf.depends_on:
                dep_replicas = replica_names_by_base.get(dep, [dep])
                # If the dependency is sequentially replicated, depending on the last replica is sufficient.
                if replica_policy_by_base.get(dep) == "sequential" and dep_replicas:
                    dep_nodes = [dep_replicas[-1]]
                else:
                    dep_nodes = dep_replicas

                # If the *target* task is sequentially replicated, only the first replica
                # needs to depend on upstream tasks; later replicas depend on the chain.
                target_nodes = replica_names_by_base.get(t_conf.name, [t_conf.name])
                if (
                    replica_policy_by_base.get(t_conf.name) == "sequential"
                    and target_nodes
                ):
                    target_nodes = [target_nodes[0]]

                for node_name in target_nodes:
                    for dep_node in dep_nodes:
                        task_graph.dag.add_edge(dep_node, node_name)

    # Third pass: resolve `${{ task.* }}` expressions in task scripts now that all tasks are built.
    # This enables referencing other tasks' assigned nodes and GPUs.
    tasks_ctx = _build_tasks_ctx(
        task_graph, state.backends or {}, replica_names_by_base
    )
    task_ctx: dict[str, Any] = {
        "task": tasks_ctx,
        "variables": variables_ctx,
        "artifacts": artifacts_ctx,
        "backends": backends_ctx,
        "workflow": {"name": config.workflow.name},
        **variables_ctx,
    }

    for task in task_graph.get_tasks():
        new_script: list[str] = []
        for line in task.script:
            if resolver.has_expression(line) and "task." in line:
                try:
                    resolved = str(resolver.resolve(line, task_ctx))
                    new_script.append(resolved)
                except Exception as e:
                    task_exprs = _extract_task_expressions(line)
                    hint = _build_task_expression_hint(
                        task_exprs, tasks_ctx, replica_names_by_base
                    )
                    exprs_display = ", ".join(task_exprs) if task_exprs else "(unknown)"
                    location = resolver._find_expression_in_sources(
                        task_exprs[0] if task_exprs else line
                    )
                    msg = (
                        f"Failed to resolve task expression in "
                        f"'{task.name}' script{location}: {exprs_display}"
                    )
                    if hint:
                        msg += f"\n  Hint: {hint}"
                    raise ValueError(msg) from e
            else:
                new_script.append(line)
        task.script = new_script

    return task_graph


async def build_state(
    config: SflowConfig,
    *,
    allocate: bool = True,
    workspace_dir: Any | None = None,
    output_dir: Any | None = None,
    source_files: list[Any] | None = None,
) -> SflowState:
    """
    Build runtime state from configuration (composition root).

    This is intentionally kept out of core to avoid core importing plugins.
    """
    from pathlib import Path

    if source_files:
        resolver.source_files = [Path(f) for f in source_files]

    # Seed an empty workflow/state; we will populate task graph after resolution/allocation.
    wf = Workflow(name=config.workflow.name, task_graph=TaskGraph())
    state = SflowState(workflow=wf)

    # Resolve global variables and backends.
    state = resolve_global_variables(config, state)
    state = resolve_backends(config, state)

    # Allocate resources (unless dry-run).
    if allocate:
        # REQ-5.1: fail fast before consuming cluster resources.
        preflight_validate_backends(state)
        preflight_validate_container_images(config, state)
        preflight_validate_task_graph(
            config,
            state,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
        )
        state = await allocate_backends(state)
    else:
        # Populate placeholder allocations (for any remaining unallocated backends)
        # so workflow variables can reference backends.*
        state = _seed_placeholder_backend_allocations(state)

    try:
        # Resolve artifacts after allocation so they can reference backend info
        # (e.g. ${{ backends.slurm_cluster.nodes[0].ip_address }} in artifact URIs/content).
        state = resolve_artifacts(
            config,
            state,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            materialize=allocate,
        )

        # Workflow variables may reference backend allocations (e.g. backends.<name>.nodes[0].ip_address),
        # so resolve them after allocation (real or placeholder).
        state = resolve_workflow_variables(config, state)

        # Build the task graph (uses allocation info if present).
        tg = build_task_graph(config, state)
        state.workflow = Workflow(name=config.workflow.name, task_graph=tg)
        return state
    except BaseException:
        # If we allocated real resources, make sure we release them even if planning fails
        # (e.g., GPU planning/validation raises during build_task_graph).
        if allocate:
            try:
                await release_backends(state)
            except Exception as e:
                _logger.error(
                    f"Failed to release backends after build_state failure: {e}"
                )
        raise
