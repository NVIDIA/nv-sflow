# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

# Matches remote registry references: [registry[:port]/][org/]name[:tag][@digest]
_REGISTRY_IMAGE_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._/:-]*(@[a-zA-Z][a-zA-Z0-9]*:[a-fA-F0-9]+)?$"
)


def is_valid_container_image(image: str) -> bool:
    """Return True if *image* looks like a registry reference or local .sqsh file."""
    if not image or not image.strip():
        return False
    image = image.strip()
    if "${{" in image or "${" in image:
        return True
    if image.endswith(".sqsh"):
        return True
    return bool(_REGISTRY_IMAGE_RE.match(image))


def mount_key(mount: str) -> tuple[str, str] | None:
    """Return the source/destination identity for a mount spec."""
    parts = str(mount).split(":", 2)
    if len(parts) < 2:
        return None
    return (parts[0], parts[1])


def append_missing_mounts(
    existing_mounts: Iterable[str],
    candidate_mounts: Iterable[str],
) -> list[str]:
    """Append mounts whose source/destination pair is not already present."""
    merged = list(existing_mounts)
    existing_keys = {
        key for mount in merged if (key := mount_key(mount)) is not None
    }
    for mount in candidate_mounts:
        key = mount_key(mount)
        if key is not None and key in existing_keys:
            continue
        merged.append(mount)
        if key is not None:
            existing_keys.add(key)
    return merged


def extract_container_mounts_from_extra_args(extra_args: list[str]) -> list[str]:
    """
    Extract --container-mounts values from extra_args.

    Supports both ``--container-mounts VALUE`` and ``--container-mounts=VALUE``.
    Comma-separated mount lists are split into individual entries.
    """
    mounts: list[str] = []
    i = 0
    while i < len(extra_args):
        arg = str(extra_args[i])
        if arg == "--container-mounts" and i + 1 < len(extra_args):
            mounts.extend(str(extra_args[i + 1]).split(","))
            i += 2
        elif arg.startswith("--container-mounts="):
            mounts.extend(arg.split("=", 1)[1].split(","))
            i += 1
        else:
            i += 1
    return mounts


def merge_container_mounts_from_extra_args(
    container_mounts: Iterable[str],
    extra_args: list[str],
) -> tuple[list[str], list[str]]:
    """Merge mount flags into a single list and return extra args without mount flags."""
    all_mounts = list(container_mounts)
    filtered_extra_args: list[str] = []
    i = 0
    while i < len(extra_args):
        arg = str(extra_args[i])
        if arg == "--container-mounts" and i + 1 < len(extra_args):
            all_mounts.extend(str(extra_args[i + 1]).split(","))
            i += 2
        elif arg.startswith("--container-mounts="):
            all_mounts.extend(arg.split("=", 1)[1].split(","))
            i += 1
        else:
            filtered_extra_args.append(arg)
            i += 1
    return all_mounts, filtered_extra_args


def local_artifact_mounts(artifacts: Iterable[Any]) -> list[str]:
    """Infer same-path rw mounts for local fs:// and file:// artifact paths."""
    mounts: list[str] = []
    for art in artifacts:
        uri = _artifact_value(art, "uri")
        try:
            scheme = (urlparse(str(uri or "")).scheme or "").lower()
        except Exception:
            scheme = ""
        if scheme not in {"fs", "file"}:
            continue

        apath = _artifact_value(art, "path")
        if apath is None:
            continue

        p = Path(apath)
        if str(p).lower().endswith(".sqsh"):
            continue

        mount_src = p
        try:
            if p.exists() and p.is_file():
                mount_src = p.parent
            elif (not p.exists()) and p.suffix:
                mount_src = p.parent
        except Exception:
            mount_src = p

        src = str(mount_src)
        mounts.append(f"{src}:{src}:rw")
    return mounts


def _artifact_value(artifact: Any, key: str) -> Any:
    if isinstance(artifact, dict):
        return artifact.get(key)
    return getattr(artifact, key, None)
