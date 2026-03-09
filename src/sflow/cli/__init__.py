# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SFLOW CLI - Command Line Interface for the sflow workflow orchestrator
"""

import typer

# Documentation link shown in all --help messages
DOCS_URL = "https://nvidia.github.io/nv-sflow/"

# Create the main CLI app
app = typer.Typer(
    name="sflow",
    help="SFLOW - Workflow Orchestrator with Pluggable Backends",
    add_completion=False,
    no_args_is_help=True,
    epilog=f"Documentation: {DOCS_URL}",
)


def _register_commands() -> None:
    # Import commands to register them with the app (import side-effects).
    # Keep this in a function to avoid "module level import not at top of file" warnings.
    from . import (
        run,  # noqa: F401
        batch,  # noqa: F401
        visualize,  # noqa: F401
        sample,  # noqa: F401
    )


_register_commands()
