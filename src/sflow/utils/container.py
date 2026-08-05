# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from sflow.logging import get_logger

_logger = get_logger(__name__)

# Matches remote registry references: [registry[:port]/][org/]name[:tag][@digest].
# ``#`` is accepted because pyxis/enroot separate the registry from the image path with
# it -- ``nvcr.io#nvidia/ai-dynamo/sglang-runtime:1.2.0`` is the documented URI form and
# is what ``--container-image`` is usually given on Slurm.
_REGISTRY_IMAGE_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._/:#-]*(@[a-zA-Z][a-zA-Z0-9]*:[a-fA-F0-9]+)?$"
)

CONTAINER_IMAGE_INVALID_HINT = (
    "Expected a remote registry reference (e.g. 'nvcr.io/org/image:tag', "
    "'nvcr.io#org/image:tag') or a local .sqsh file path "
    "(e.g. '/path/to/image.sqsh')"
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


def validate_container_image_reference(
    image: str | None,
    *,
    source: str,
    error_prefix: str | None = None,
) -> None:
    """WARN (never raise) when *image* does not match a shape we recognise.

    This used to abort the run, and it should not have. :func:`is_valid_container_image`
    is a heuristic regex, and the set of references a container runtime actually accepts
    is larger than it models -- pyxis/enroot alone take ``registry#path:tag``,
    ``docker://`` URIs and site-local schemes, and a rejected recipe had no override
    short of editing sflow. A non-match therefore means "sflow did not recognise this",
    not "this is wrong", which is a warning's job.

    The runtime remains the real authority: an image reference that is genuinely bad
    fails at pull/enroot time with a message from the tool that actually resolves it,
    which is more accurate than anything this regex could say.

    ``error_prefix`` is kept (callers pass it) and now prefixes the warning.
    """
    if image is None:
        return
    image_str = str(image)
    if not image_str:
        return
    if not is_valid_container_image(image_str):
        prefix = f"{error_prefix}: " if error_prefix else ""
        _logger.warning(
            f"{prefix}{source} does not look like a valid container image. "
            f"{CONTAINER_IMAGE_INVALID_HINT}, got: '{image_str}'. "
            "Continuing anyway -- the container runtime will report it if it is "
            "genuinely unusable."
        )


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


def append_runtime_mounts(
    existing_mounts: Iterable[str],
    candidate_mounts: Iterable[str],
) -> list[str]:
    """Append runtime mounts using the shared source/destination dedupe rule."""
    return append_missing_mounts(existing_mounts, candidate_mounts)


def extract_container_images_from_extra_args(extra_args: list[str]) -> list[str]:
    """
    Extract --container-image values from extra_args.

    Supports both ``--container-image VALUE`` and ``--container-image=VALUE``.
    """
    images: list[str] = []
    i = 0
    while i < len(extra_args):
        arg = str(extra_args[i])
        if arg == "--container-image" and i + 1 < len(extra_args):
            images.append(str(extra_args[i + 1]))
            i += 2
        elif arg.startswith("--container-image="):
            images.append(arg.split("=", 1)[1])
            i += 1
        else:
            i += 1
    return images


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


def collect_container_mounts(op_conf: Any) -> list[str]:
    """Return mounts exposed by an operator config or lightweight config object."""
    mount_specs = getattr(op_conf, "mount_specs", None)
    if callable(mount_specs):
        return list(mount_specs())

    mounts: list[str] = []
    for field_name in ("container_mounts", "mounts"):
        field_mounts = list(getattr(op_conf, field_name, None) or [])
        mounts = append_runtime_mounts(mounts, field_mounts)

    extra_mounts = extract_container_mounts_from_extra_args(
        list(getattr(op_conf, "extra_args", None) or [])
    )
    return append_runtime_mounts(mounts, extra_mounts)


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
