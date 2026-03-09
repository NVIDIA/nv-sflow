# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
from operator import or_ as _or
from typing import Annotated, Any, Callable, Mapping, TypeVar

from pydantic import Field, TypeAdapter

from sflow.core.operator import Operator, OperatorConfig

_T = TypeVar("_T", bound=type[Operator])


@dataclass(frozen=True)
class OperatorRegistration:
    type: str
    operator_cls: type[Operator]
    config_cls: type[OperatorConfig]


_REGISTRY: dict[str, OperatorRegistration] = {}


def register_operator(
    type_name: str,
    config_cls: type[OperatorConfig],
) -> Callable[[_T], _T]:
    """
    Decorator to register an Operator implementation in the global registry.

    Usage:
        @register_operator("bash", BashOperatorConfig)
        class BashOperator(Operator): ...
    """

    def _decorator(operator_cls: _T) -> _T:
        existing = _REGISTRY.get(type_name)
        if existing is not None and existing.operator_cls is not operator_cls:
            raise RuntimeError(
                f"Operator type '{type_name}' already registered with "
                f"{existing.operator_cls.__module__}.{existing.operator_cls.__name__}"
            )
        _REGISTRY[type_name] = OperatorRegistration(
            type=type_name,
            operator_cls=operator_cls,
            config_cls=config_cls,
        )
        # Invalidate cached TypeAdapter when registry changes.
        operator_config_type_adapter.cache_clear()
        return operator_cls

    return _decorator


def get_operator_registry() -> Mapping[str, OperatorRegistration]:
    return dict(_REGISTRY)


def get_operator_class(type_name: str) -> type[Operator]:
    reg = _REGISTRY.get(type_name)
    if reg is None:
        raise KeyError(f"Unknown operator type: {type_name!r}")
    return reg.operator_cls


def ensure_builtin_operators_registered() -> None:
    """
    Import built-in operator plugins to populate the registry.

    We keep this import lazy so core/config can validate without eagerly importing
    plugins unless operator configs are actually used.
    """

    # Import triggers registration decorators in modules.
    import sflow.plugins.operators  # noqa: F401


@lru_cache(maxsize=1)
def operator_config_type_adapter() -> TypeAdapter[Any]:
    """
    Build a discriminated TypeAdapter for all registered OperatorConfig models.
    Discriminator field is 'type'.
    """

    config_models = [reg.config_cls for reg in _REGISTRY.values()]
    if not config_models:
        # Fallback: validate as base config (very permissive).
        return TypeAdapter(OperatorConfig)

    # Build a PEP604 union dynamically: A | B | C ...
    union_type: Any = reduce(_or, config_models)
    return TypeAdapter(Annotated[union_type, Field(discriminator="type")])
