# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, TypeVar
from urllib.parse import unquote, urlparse

from sflow.core.artifact import Artifact

_T = TypeVar("_T")


class ArtifactResolver(Protocol):
    """
    Resolve a configured artifact URI into a local reference (optional materialization).
    """

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
    ) -> Artifact: ...


@dataclass(frozen=True)
class ArtifactRegistration:
    scheme: str
    resolver: ArtifactResolver


_REGISTRY: dict[str, ArtifactRegistration] = {}


def register_artifact_scheme(scheme: str) -> Callable[[_T], _T]:
    """
    Decorator to register an ArtifactResolver instance for a URI scheme.

    Usage:
        @register_artifact_scheme("file")
        class FileResolver: ...
    """

    scheme_norm = (scheme or "").strip().lower()
    if not scheme_norm:
        raise ValueError("scheme must be a non-empty string")

    def _decorator(resolver_obj: _T) -> _T:
        existing = _REGISTRY.get(scheme_norm)
        if existing is not None and existing.resolver is not resolver_obj:
            raise RuntimeError(
                f"Artifact scheme '{scheme_norm}' already registered with "
                f"{existing.resolver.__class__.__module__}.{existing.resolver.__class__.__name__}"
            )
        _REGISTRY[scheme_norm] = ArtifactRegistration(
            scheme=scheme_norm,
            resolver=resolver_obj,  # type: ignore[assignment]
        )
        return resolver_obj

    return _decorator


def ensure_builtin_artifacts_registered() -> None:
    # Import triggers registration decorators in modules.
    import sflow.plugins.artifacts  # noqa: F401


def get_artifact_registry() -> Mapping[str, ArtifactRegistration]:
    return dict(_REGISTRY)


def get_artifact_resolver_for_uri(uri: str) -> ArtifactResolver | None:
    try:
        parsed = urlparse(str(uri))
    except Exception:
        return None
    scheme = (parsed.scheme or "").lower()
    if not scheme:
        return None
    reg = _REGISTRY.get(scheme)
    return reg.resolver if reg is not None else None


# -----------------------------------------------------------------------------
# Built-in helpers for common schemes (used by plugins)
# -----------------------------------------------------------------------------


def resolve_file_like_uri_to_path(uri: str, *, workspace_dir: Path) -> Path:
    """
    Resolve fs:// or file:// URIs into an absolute Path.

    - Relative paths are interpreted relative to workspace_dir.
    - file://localhost/... is treated as an absolute path.
    """
    parsed = urlparse(str(uri))
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"file", "fs"}:
        raise ValueError(f"Not a file-like scheme: {scheme!r}")

    # urlparse("file://x.yaml") -> netloc="x.yaml", path=""
    raw = (parsed.netloc or "") + (parsed.path or "")
    raw = unquote(raw)

    # Ensure we don't treat "file://localhost/..." as a relative path
    if scheme == "file" and parsed.netloc in {"localhost"} and parsed.path:
        raw = unquote(parsed.path)

    p = Path(raw)
    if not p.is_absolute():
        p = workspace_dir / p
    return p


def http_cache_path(uri: str, *, cache_dir: Path) -> Path:
    """
    Deterministic cache path for http(s) artifacts.
    """
    h = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"http_{h}"
