# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageTarget(ABC):
    """
    Abstract base class for post-execution storage targets (S3, GCS, Azure, ...).

    A `StorageTarget` instance represents a *named* storage destination configured
    in the top-level `storage:` block of an sflow.yaml. Concrete implementations
    (e.g. `S3StorageTarget`) translate `upload(local_path, remote_key)` into the
    provider-specific transfer call.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def upload(self, local_path: Path, remote_key: str) -> None:
        """
        Upload a single local file to `remote_key` on this target.

        `remote_key` is the full key/path under the target's bucket (or
        equivalent), already including any configured `prefix`.
        """
        raise NotImplementedError

    @abstractmethod
    def plan(self, local_path: Path, remote_key: str) -> str:
        """
        Human-readable destination string for dry-run output, e.g.
        ``s3://bucket/prefix/key``.
        """
        raise NotImplementedError

    def dry_run_warnings(self) -> list[str]:
        """Return planning-time warnings for this target (e.g. missing SDK/credentials).

        Surfaced by ``sflow run --dry-run`` so users learn about likely upload
        failures (missing client library or credentials) before launching a real
        run. Should be a cheap, offline check. Default: no warnings.
        """
        return []
