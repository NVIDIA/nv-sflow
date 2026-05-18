# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SFLOW CLI - Command Line Interface for the sflow workflow orchestrator
"""

from typing import Optional

import typer

from sflow import __version__

# Documentation link shown in all --help messages
DOCS_URL = "https://nvidia.github.io/nv-sflow/"


def _git_info() -> str:
    """Return branch and short hash from git, or empty string if unavailable."""
    import subprocess

    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            .decode()
            .strip()
        )
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            .decode()
            .strip()
        )
        suffix = ", dirty" if dirty else ""
        return f" ({branch} {sha}{suffix})"
    except Exception:
        return ""


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sflow {__version__}{_git_info()}")
        raise typer.Exit()


# Create the main CLI app
app = typer.Typer(
    name="sflow",
    help="SFLOW - Workflow Orchestrator with Pluggable Backends",
    add_completion=False,
    no_args_is_help=True,
    epilog=f"Documentation: {DOCS_URL}",
)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """SFLOW - Workflow Orchestrator with Pluggable Backends"""


def _register_commands() -> None:
    # Import commands to register them with the app (import side-effects).
    # Keep this in a function to avoid "module level import not at top of file" warnings.
    from . import (
        run,  # noqa: F401
        batch,  # noqa: F401
        visualize,  # noqa: F401
        sample,  # noqa: F401
        compose,  # noqa: F401
        skill,  # noqa: F401
    )


_register_commands()
