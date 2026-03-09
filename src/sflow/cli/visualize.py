# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for visualizing workflows.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer

from sflow.app.sflow import SflowApp
from sflow.cli import DOCS_URL, app
from sflow.logging import get_logger

_logger = get_logger(__name__)

_sflow_app = SflowApp()


@app.command(epilog=f"Documentation: {DOCS_URL}")
def visualize(
    file: Annotated[
        Path,
        typer.Option(
            "-f",
            "--file",
            help="Path to the sflow.yaml workflow file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("sflow.yaml"),
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Path to the output file. If omitted, writes to <output-dir>/<run_id>/<workflow>.<ext>.",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format (png, svg, pdf, mermaid, dot)",
        ),
    ] = "png",
    workspace_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--workspace-dir",
            help="Workspace root directory. Default: current working directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            help="Global output root directory. Default: <workspace-dir>/sflow_output",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    show_variables: Annotated[
        bool,
        typer.Option(
            "--show-variables",
            help="Include resolved variables in the visualization output (as comments/notes).",
        ),
    ] = False,
):
    """
    Visualize a workflow DAG from a sflow.yaml file.

    The visualization includes explicit Start/End nodes connected to all entry/exit tasks.

    Examples:
        # Generate PNG image
        sflow visualize --file workflow.yaml

        # Generate SVG with custom output path
        sflow visualize --file workflow.yaml --format svg --output dag.svg
    """
    try:
        _logger.info(f"Generating {format.upper()} visualization...")
        result = _sflow_app.visualize(
            file=file,
            output_path=output_path,
            format=format,
            show_variables=show_variables,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
        )

        if result.saved_path:
            _logger.info(f"Workflow visualization saved to: {result.saved_path}")
            typer.echo(f"✓ Saved to {result.saved_path}")
            return

        _logger.warning("Graph visualization not fully implemented")
        typer.echo(f"Graph has {result.task_count} tasks")
        if result.topo_order:
            for task_name in result.topo_order:
                typer.echo(f"  - {task_name}")
        return

    except ValueError as e:
        _logger.error(f"Configuration error: {e}")
        typer.echo(f"✗ Configuration error: {e}", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        _logger.error(f"File not found: {e}")
        typer.echo(f"✗ File not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        _logger.exception(f"Visualization failed: {e}")
        typer.echo(f"✗ Failed: {e}", err=True)
        raise typer.Exit(code=1)
