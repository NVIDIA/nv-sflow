# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sflow.core.artifact import Artifact
from sflow.core.artifact_registry import register_artifact_scheme


class _DockerArtifactResolver:
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
                f"Artifact '{name}' uses inline content but has docker URI; only file:// supports inline content"
            )
        if materialize:
            raise NotImplementedError(
                "docker:// artifacts are not materialized yet. "
                "For now, reference a local path via file:// or fs://, "
                "or ensure your task scripts can handle the docker URI directly."
            )
        return Artifact(name=name, uri=uri, description=description, path=None)


DOCKER_ARTIFACT_RESOLVER = register_artifact_scheme("docker")(_DockerArtifactResolver())
