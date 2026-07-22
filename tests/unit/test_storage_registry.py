# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from sflow.config.schema import StorageConfig
from sflow.core.storage import StorageTarget
from sflow.core.storage_registry import (
    ensure_builtin_storage_registered,
    get_storage_class,
    get_storage_registry,
    register_storage,
    storage_config_type_adapter,
)


def test_builtin_s3_is_registered():
    ensure_builtin_storage_registered()
    assert "s3" in get_storage_registry()
    cls = get_storage_class("s3")
    assert issubclass(cls, StorageTarget)


def test_unknown_type_raises():
    ensure_builtin_storage_registered()
    with pytest.raises(KeyError):
        get_storage_class("not_a_real_provider")


def test_adapter_discriminates_by_type():
    ensure_builtin_storage_registered()
    adapter = storage_config_type_adapter()
    parsed = adapter.validate_python(
        {"name": "b1", "type": "s3", "bucket": "my-bucket"}
    )
    assert parsed.type == "s3"
    assert parsed.bucket == "my-bucket"

    with pytest.raises(ValidationError):
        adapter.validate_python({"name": "b1", "type": "nonexistent", "bucket": "x"})


def test_register_storage_rejects_conflict():
    class _FakeConfig(StorageConfig):
        type: Literal["__test_fake_kind__"] = "__test_fake_kind__"

    @register_storage("__test_fake_kind__", _FakeConfig)
    class _FakeTarget(StorageTarget):
        def __init__(self, config: _FakeConfig):
            super().__init__(name=config.name)

        async def upload(self, local_path: Path, remote_key: str) -> None:
            return None

        def plan(self, local_path: Path, remote_key: str) -> str:
            return "fake"

    # Re-registering the same class is a no-op.
    register_storage("__test_fake_kind__", _FakeConfig)(_FakeTarget)

    # Registering a different class for the same type raises.
    class _AnotherTarget(StorageTarget):
        def __init__(self, config: _FakeConfig):
            super().__init__(name=config.name)

        async def upload(self, local_path: Path, remote_key: str) -> None:
            return None

        def plan(self, local_path: Path, remote_key: str) -> str:
            return "another"

    with pytest.raises(RuntimeError, match="already registered"):
        register_storage("__test_fake_kind__", _FakeConfig)(_AnotherTarget)
