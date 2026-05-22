# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import shutil
from typing import Any

from sflow.config.schema import SflowConfig
from sflow.core.backend import Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.state import SflowState
from sflow.core.variable import build_variables_ctx
from sflow.logging import get_logger
from sflow.resolution import ExpressionResolver, validate_no_deferred_variable_references
from sflow.utils.container import is_valid_container_image

_logger = get_logger(__name__)


def preflight_validate_backends(state: SflowState) -> None:
    """Validate backend prerequisites before allocations/submissions."""

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


def preflight_validate_container_images(
    config: SflowConfig,
    state: SflowState,
    *,
    resolver: ExpressionResolver,
) -> None:
    """Validate container image references in srun operators before allocation."""

    variables_ctx = build_variables_ctx(state.variables)
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

    invalid_hint = (
        "Expected a remote registry reference (e.g. 'nvcr.io/org/image:tag') "
        "or a local .sqsh file path (e.g. '/path/to/image.sqsh')"
    )

    def _check_image(image_val: str, *, source: str) -> None:
        if not image_val:
            return
        if "${{" in image_val or "${" in image_val:
            return
        if not is_valid_container_image(image_val):
            raise ValueError(
                f"Pre-flight validation failed: {source} has invalid container image. "
                f"{invalid_hint}, got: '{image_val}'"
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


def seed_placeholder_backend_allocations(state: SflowState) -> SflowState:
    """Populate deterministic placeholder allocations for unallocated backends."""

    from sflow.core.backend import Allocation

    if not state.backends:
        return state

    for b in state.backends.values():
        if b.allocation is not None:
            continue

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
                ip = f"0.0.0.{i + 1}"
            nodes.append(
                ComputeNode(name=name, ip_address=ip, index=i, num_gpus=num_gpus)
            )

        b.allocation = Allocation(allocation_id="0", nodes=nodes, owned=False)

    return state


def resolve_backends(
    config: SflowConfig,
    state: SflowState,
    *,
    resolver: ExpressionResolver,
) -> SflowState:
    """Resolve backend configuration and populate ``state.backends``."""

    from sflow.core.backend_registry import (
        backend_config_type_adapter,
        ensure_builtin_backends_registered,
        get_backend_class,
    )

    ensure_builtin_backends_registered()

    variables_ctx = build_variables_ctx(state.variables)
    ctx: dict[str, Any] = {"variables": variables_ctx, **variables_ctx}

    backends: dict[str, Backend] = dict(state.backends or {})

    backend_confs = list(config.backends or [])
    if not backend_confs:
        backend_confs = [{"name": "local", "type": "local", "default": True}]

    bconf_adapter = backend_config_type_adapter()

    for b_conf in backend_confs:
        if hasattr(b_conf, "model_dump"):
            b_conf_obj = b_conf
        else:
            b_conf_obj = bconf_adapter.validate_python(b_conf)

        validate_no_deferred_variable_references(
            b_conf_obj.model_dump() if hasattr(b_conf_obj, "model_dump") else b_conf_obj,
            state.variables,
            resolver,
            location=f"backends.{getattr(b_conf_obj, 'name')}",
            usage="backends",
        )

        backend_cls = get_backend_class(getattr(b_conf_obj, "type"))

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
    """Allocate resources for all configured unallocated backends."""

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
        await asyncio.gather(
            *[b.release_resources() for b in to_allocate if b.allocation is not None],
            return_exceptions=True,
        )
        raise exc

    return state


async def release_backends(state: SflowState) -> SflowState:
    """Release resources for all allocated backends."""

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
