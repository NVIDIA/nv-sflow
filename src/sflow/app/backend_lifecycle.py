# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from sflow.config.schema import SflowConfig
from sflow.core.backend import Backend
from sflow.core.state import SflowState
from sflow.core.variable import build_variables_ctx
from sflow.logging import get_logger
from sflow.resolution import ExpressionResolver, validate_no_deferred_variable_references
from sflow.utils.container import (
    extract_container_images_from_extra_args,
    validate_container_image_reference,
)

_logger = get_logger(__name__)


def preflight_validate_backends(state: SflowState) -> None:
    """Validate backend prerequisites before allocations/submissions."""

    for b in (state.backends or {}).values():
        b.preflight_validate()


def preflight_validate_container_images(
    config: SflowConfig,
    state: SflowState,
    *,
    resolver: ExpressionResolver,
) -> None:
    """Validate container image references before allocation."""

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

    def _check_image(image_val: str, *, source: str) -> None:
        validate_container_image_reference(
            image_val,
            source=f"{source} has invalid container image",
            error_prefix="Pre-flight validation failed",
        )

    def _check_operator_config(op_conf: Any, *, source: str) -> None:
        container_images = getattr(op_conf, "container_images", None)
        if callable(container_images):
            images = list(container_images())
        else:
            images = []
            for attr in ("container_image", "image"):
                raw_image = getattr(op_conf, attr, None)
                if raw_image is not None and raw_image not in images:
                    images.append(raw_image)
            images.extend(
                extract_container_images_from_extra_args(
                    list(getattr(op_conf, "extra_args", None) or [])
                )
            )
        for image in images:
            _check_image(_try_resolve(image), source=source)

    for op_conf in config.operators or []:
        _check_operator_config(op_conf, source=f"operator '{op_conf.name}'")

    for backend_conf in config.backends or []:
        for raw_image in backend_conf.container_images():
            _check_image(
                _try_resolve(raw_image),
                source=f"backend '{backend_conf.name}'",
            )

    from sflow.core.operator_registry import operator_config_type_adapter

    operator_adapter = operator_config_type_adapter()
    operator_confs = {op.name: op for op in config.operators or []}
    for t_conf in config.workflow.tasks or []:
        if t_conf.operator is None or isinstance(t_conf.operator, str):
            continue
        base_op = operator_confs.get(t_conf.operator.name)
        if base_op is None:
            continue
        merged = base_op.model_dump()
        overrides = t_conf.operator.model_dump(exclude={"name"}, exclude_none=True)
        if "extra_args" in overrides and merged.get("extra_args"):
            overrides["extra_args"] = list(merged["extra_args"]) + list(
                overrides["extra_args"]
            )
        merged.update(overrides)
        merged["name"] = t_conf.operator.name
        try:
            op_conf = operator_adapter.validate_python(merged)
        except Exception:
            class _OperatorConfigView:
                def __init__(self, values: dict[str, Any]):
                    self.__dict__.update(values)

            op_conf = _OperatorConfigView(merged)
        _check_operator_config(
            op_conf,
            source=f"task '{t_conf.name}' operator override",
        )


def seed_placeholder_backend_allocations(state: SflowState) -> SflowState:
    """Populate deterministic placeholder allocations for unallocated backends."""

    if not state.backends:
        return state

    for b in state.backends.values():
        if b.allocation is not None:
            continue
        b.allocation = b.placeholder_allocation()

    return state


def resolve_backends(
    config: SflowConfig,
    state: SflowState,
    *,
    resolver: ExpressionResolver,
    kubectl_config: Any | None = None,
) -> SflowState:
    """Resolve backend configuration and populate ``state.backends``.

    ``kubectl_config`` (CLI-level kube access from ``sflow run``) is applied to any
    backend that accepts it (the kubernetes backend), keeping recipes
    cluster-agnostic.
    """

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
        if kubectl_config is not None and hasattr(backend, "apply_kubectl_config"):
            backend.apply_kubectl_config(kubectl_config)

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
