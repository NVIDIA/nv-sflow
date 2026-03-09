# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Artifact:
    """
    Resolved artifact reference.

    Artifacts are primarily used for expression resolution (`${{ artifacts.NAME.path }}`)
    and for injecting convenience env vars into tasks (e.g. `${NAME}` -> local path).

    This is intentionally a small, side-effect-free data model. Any materialization
    (e.g. writing inline content to disk, downloading, etc.) is handled by artifact
    resolvers (see `sflow.core.artifact_registry`).
    """

    name: str
    uri: str
    description: str | None = None
    # Local materialized path, if applicable (e.g. file:// resolved or downloaded into cache)
    path: Path | None = None

    def to_context_dict(self) -> dict[str, str | None]:
        """
        Context view used by the expression resolver and task env injection.
        """
        return {
            "uri": self.uri,
            # Keep parity with previous behavior: if we don't have a local path,
            # expose the URI as a usable string.
            "path": str(self.path) if self.path is not None else self.uri,
        }
