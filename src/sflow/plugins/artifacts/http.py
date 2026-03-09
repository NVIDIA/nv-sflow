# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from sflow.core.artifact import Artifact
from sflow.core.artifact_registry import http_cache_path, register_artifact_scheme


class _HttpArtifactResolver:
    def resolve(
        self,
        *,
        name: str,
        uri: str,
        description: str | None,
        content: str | None,
        workspace_dir: Path,
        cache_dir: Path,
        output_dir: Path,
        materialize: bool,
    ) -> Artifact:
        if content is not None:
            raise ValueError(
                f"Artifact '{name}' uses inline content but has http(s) URI; only file:// supports inline content"
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        path = http_cache_path(uri, cache_dir=cache_dir)

        if materialize and not path.exists():
            try:
                with urlopen(uri) as resp:
                    data = resp.read()
                path.write_bytes(data)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download artifact '{name}' from {uri!r}"
                ) from e

        return Artifact(name=name, uri=uri, description=description, path=path)


HTTP_ARTIFACT_RESOLVER = register_artifact_scheme("http")(
    register_artifact_scheme("https")(_HttpArtifactResolver())
)
