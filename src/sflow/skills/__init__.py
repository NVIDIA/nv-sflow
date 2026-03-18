# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AI agent skills for sflow.

This package contains skill definitions that teach AI coding agents
how to write sflow YAML configs and diagnose sflow errors.
Use the `sflow skill` command to copy skills into your project.
"""

from pathlib import Path


def get_skills_dir() -> Path:
    """Get the path to the packaged skills directory."""
    return Path(__file__).parent


def list_skills() -> list[str]:
    """List all available skill directories."""
    skills_dir = get_skills_dir()
    _skip = {"__pycache__", ".git"}
    return sorted(
        d.name
        for d in skills_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name not in _skip
    )
