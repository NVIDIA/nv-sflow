# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sflow.logging import get_logger
from sflow.core.artifact import Artifact
from sflow.core.artifact_registry import (
    register_artifact_scheme,
    resolve_file_like_uri_to_path,
)

_logger = get_logger(__name__)


class _FileArtifactResolver:
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
        remote_filesystem: bool = False,
    ) -> Artifact:
        path = resolve_file_like_uri_to_path(uri, workspace_dir=workspace_dir)
        is_fs_scheme = str(uri).lower().startswith("fs://")

        # For fs:// artifacts that don't exist, create an empty directory with a warning.
        # This allows workflows to reference output directories that will be populated at runtime.
        if is_fs_scheme and materialize and content is None and not path.exists():
            if remote_filesystem:
                # The backend executes off the controller host (e.g. Kubernetes),
                # so an fs:// path refers to a location on the cluster/image, not the
                # local machine. Don't validate or create it locally -- pass it through.
                _logger.info(
                    f"Artifact '{name}' fs:// path '{path}' is treated as a remote "
                    "path (backend executes off-host); not created or validated locally."
                )
            else:
                _logger.warning(
                    f"Artifact '{name}' path does not exist: {path}. "
                    f"Creating empty directory."
                )
                path.mkdir(parents=True, exist_ok=True)

        # Inline content support: only for file:// URIs (validated by schema, but keep a guard).
        if content is not None:
            if not str(uri).startswith("file://"):
                raise ValueError(
                    "Inline artifact content is only supported for 'file://' URIs"
                )

            # For relative file:// URIs, write generated files under the workflow output
            # directory to keep the workspace clean.
            from urllib.parse import unquote, urlparse

            parsed = urlparse(str(uri))
            raw = unquote((parsed.netloc or "") + (parsed.path or ""))
            if not Path(raw).is_absolute():
                path = output_dir / raw

            if remote_filesystem:
                # The backend executes off the controller host (e.g. Kubernetes), so a
                # controller-written file is invisible to the remote pods/nodes. Don't
                # write it here; the off-host operator injects the content natively
                # (e.g. a ConfigMap mounted at this path) from Artifact.content below.
                _logger.info(
                    f"Artifact '{name}' (file://) inline content will be injected "
                    "natively by the off-host backend (not written to the controller); "
                    f"in-task path: {path}"
                )
            else:
                _logger.info(
                    f"Artifact '{name}' (file://) with inline content will be written to: {path}"
                )
                if materialize:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")

        return Artifact(
            name=name, uri=uri, description=description, path=path, content=content
        )


# Register file-like resolvers.
FILE_ARTIFACT_RESOLVER = register_artifact_scheme("file")(
    register_artifact_scheme("fs")(_FileArtifactResolver())
)
