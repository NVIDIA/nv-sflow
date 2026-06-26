# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
from operator import or_ as _or
from typing import Annotated, Any, Callable, Mapping, TypeVar

from pydantic import BaseModel, Field, TypeAdapter

from sflow.core.storage import StorageTarget

_T = TypeVar("_T", bound=type[StorageTarget])


@dataclass(frozen=True)
class StorageRegistration:
    type: str
    target_cls: type[StorageTarget]
    config_cls: type[BaseModel]


_REGISTRY: dict[str, StorageRegistration] = {}


def register_storage(
    type_name: str,
    config_cls: type[BaseModel],
) -> Callable[[_T], _T]:
    """
    Decorator to register a StorageTarget implementation.
    """

    def _decorator(target_cls: _T) -> _T:
        existing = _REGISTRY.get(type_name)
        if existing is not None and existing.target_cls is not target_cls:
            raise RuntimeError(
                f"Storage type '{type_name}' already registered with "
                f"{existing.target_cls.__module__}.{existing.target_cls.__name__}"
            )
        _REGISTRY[type_name] = StorageRegistration(
            type=type_name,
            target_cls=target_cls,
            config_cls=config_cls,
        )
        storage_config_type_adapter.cache_clear()
        return target_cls

    return _decorator


def ensure_builtin_storage_registered() -> None:
    # Import triggers registration decorators in modules.
    import sflow.plugins.storage  # noqa: F401


def get_storage_registry() -> Mapping[str, StorageRegistration]:
    return dict(_REGISTRY)


def get_storage_class(type_name: str) -> type[StorageTarget]:
    reg = _REGISTRY.get(type_name)
    if reg is None:
        raise KeyError(f"Unknown storage type: {type_name!r}")
    return reg.target_cls


@lru_cache(maxsize=1)
def storage_config_type_adapter() -> TypeAdapter[Any]:
    config_models = [reg.config_cls for reg in _REGISTRY.values()]
    if not config_models:
        # Fallback: validate as a plain dict-like model (very permissive).
        return TypeAdapter(dict[str, Any])

    union_type: Any = reduce(_or, config_models)
    return TypeAdapter(Annotated[union_type, Field(discriminator="type")])
