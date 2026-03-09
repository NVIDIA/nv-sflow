# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
from operator import or_ as _or
from typing import Annotated, Any, Callable, Mapping, TypeVar

from pydantic import BaseModel, Field, TypeAdapter

from sflow.core.backend import Backend

_T = TypeVar("_T", bound=type[Backend])


@dataclass(frozen=True)
class BackendRegistration:
    type: str
    backend_cls: type[Backend]
    config_cls: type[BaseModel]


_REGISTRY: dict[str, BackendRegistration] = {}


def register_backend(
    type_name: str,
    config_cls: type[BaseModel],
) -> Callable[[_T], _T]:
    """
    Decorator to register a Backend implementation.
    """

    def _decorator(backend_cls: _T) -> _T:
        existing = _REGISTRY.get(type_name)
        if existing is not None and existing.backend_cls is not backend_cls:
            raise RuntimeError(
                f"Backend type '{type_name}' already registered with "
                f"{existing.backend_cls.__module__}.{existing.backend_cls.__name__}"
            )
        _REGISTRY[type_name] = BackendRegistration(
            type=type_name,
            backend_cls=backend_cls,
            config_cls=config_cls,
        )
        backend_config_type_adapter.cache_clear()
        return backend_cls

    return _decorator


def ensure_builtin_backends_registered() -> None:
    # Import triggers registration decorators in modules.
    import sflow.plugins.backends  # noqa: F401


def get_backend_registry() -> Mapping[str, BackendRegistration]:
    return dict(_REGISTRY)


def get_backend_class(type_name: str) -> type[Backend]:
    reg = _REGISTRY.get(type_name)
    if reg is None:
        raise KeyError(f"Unknown backend type: {type_name!r}")
    return reg.backend_cls


@lru_cache(maxsize=1)
def backend_config_type_adapter() -> TypeAdapter[Any]:
    config_models = [reg.config_cls for reg in _REGISTRY.values()]
    if not config_models:
        # Fallback: validate as a plain dict-like model (very permissive).
        return TypeAdapter(dict[str, Any])

    union_type: Any = reduce(_or, config_models)
    return TypeAdapter(Annotated[union_type, Field(discriminator="type")])
