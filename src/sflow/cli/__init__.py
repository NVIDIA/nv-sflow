# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SFLOW CLI - Command Line Interface for the sflow workflow orchestrator
"""

import sys
from typing import Optional

import typer

from sflow import runtime_info

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
        typer.echo(runtime_info.format_runtime_info())
        raise typer.Exit()


class _SflowTyper(typer.Typer):
    def __call__(self, *args, **kwargs):
        from sflow.ssh.delegate import predispatch
        from sflow.utils.distribution import reconcile

        # The console-script entry, not import time: library users and tests never
        # have their environment edited by importing sflow.
        reconcile()

        # click's own argv: an explicit list wins, exactly as it does for the
        # parse that follows, so --ssh cannot be honored by one and missed by
        # the other.
        argv = kwargs.get("args", args[0] if args else None)
        try:
            predispatch(list(argv) if argv is not None else sys.argv[1:])
        except typer.Exit as exc:
            raise SystemExit(exc.exit_code) from None
        except (typer.BadParameter, FileNotFoundError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise SystemExit(2) from None
        return super().__call__(*args, **kwargs)


# Create the main CLI app
app = _SflowTyper(
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
        batch,  # noqa: F401
        compose,  # noqa: F401
        remote,  # noqa: F401
        run,  # noqa: F401
        sample,  # noqa: F401
        skill,  # noqa: F401
        upgrade,  # noqa: F401
        visualize,  # noqa: F401
    )


_register_commands()
