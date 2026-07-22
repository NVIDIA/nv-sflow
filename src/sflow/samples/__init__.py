# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sample workflow YAML files for sflow.

This package contains example workflow configurations that can be
copied to your project using the `sflow sample` command.
"""

from importlib import resources
from pathlib import Path


def get_samples_dir() -> Path:
    """Get the path to the samples directory."""
    return Path(resources.files(__package__))


def list_samples() -> list[str]:
    """List all self-contained sample workflows as relative subpath names (no
    ``.yaml`` suffix), e.g. ``self_contained/slurm/auto_replica``.

    Samples live under ``self_contained/<backend>/[<app>/]<name>.yaml``. Falls back
    to the legacy flat layout (``<name>.yaml`` at the package root) if present.
    """
    samples_dir = get_samples_dir()
    sc_dir = samples_dir / "self_contained"
    if sc_dir.is_dir():
        return sorted(
            f.relative_to(samples_dir).with_suffix("").as_posix()
            for f in sc_dir.rglob("*.yaml")
        )
    # Legacy flat layout.
    return sorted(f.stem for f in samples_dir.glob("*.yaml"))


def list_modular_samples() -> dict[str, list[str]]:
    """List modular sample bundles under ``modular/``.

    Returns a dict mapping the bundle's relative subpath (e.g.
    ``modular/inference_x_v2``) to the YAML files within it (relative to the
    bundle, no ``.yaml`` suffix).
    """
    samples_dir = get_samples_dir()
    modular_root = samples_dir / "modular"
    result: dict[str, list[str]] = {}
    if not modular_root.is_dir():
        return result
    for bundle in sorted(p for p in modular_root.iterdir() if p.is_dir()):
        if bundle.name.startswith("_"):
            continue
        yamls = sorted(
            f.relative_to(bundle).with_suffix("").as_posix()
            for f in bundle.rglob("*.yaml")
        )
        if yamls:
            result[bundle.relative_to(samples_dir).as_posix()] = yamls
    return result


def get_sample_path(name: str) -> Path | None:
    """Get the path to a specific sample file or folder."""
    samples_dir = get_samples_dir()
    # Try exact match first (file or directory)
    sample_path = samples_dir / name.rstrip("/")
    if sample_path.exists():
        return sample_path
    # Try with .yaml extension
    sample_path = samples_dir / f"{name}.yaml"
    if sample_path.exists():
        return sample_path
    return None


def get_sample_content(name: str) -> str | None:
    """Get the content of a specific sample file."""
    sample_path = get_sample_path(name)
    if sample_path:
        return sample_path.read_text()
    return None
