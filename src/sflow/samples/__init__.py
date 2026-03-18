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
    """List all available sample YAML files (top-level only)."""
    samples_dir = get_samples_dir()
    return sorted([f.name for f in samples_dir.glob("*.yaml")])


def list_modular_samples() -> dict[str, list[str]]:
    """List modular sample sets organized in subdirectories.

    Returns a dict mapping folder name to the list of YAML filenames within it.
    """
    samples_dir = get_samples_dir()
    result: dict[str, list[str]] = {}
    _skip = {"__pycache__", "sflow_output", ".git"}
    for subdir in sorted(samples_dir.iterdir()):
        if (
            subdir.is_dir()
            and not subdir.name.startswith("_")
            and subdir.name not in _skip
        ):
            yamls = sorted([f.name for f in subdir.glob("*.yaml")])
            if yamls:
                result[subdir.name] = yamls
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
