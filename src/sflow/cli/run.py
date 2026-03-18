# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for running workflows.
"""

from collections import deque
from pathlib import Path
from typing import Annotated, List, Optional

import typer

from sflow.app.sflow import SflowApp
from sflow.cli import DOCS_URL, app
from sflow.config.resolver import enrich_error_with_location
from sflow.logging import configure_logging, get_logger

_logger = get_logger(__name__)

_sflow_app = SflowApp()


@app.command(epilog=f"Documentation: {DOCS_URL}")
def run(
    src_files: Annotated[
        Optional[List[Path]],
        typer.Argument(
            help="Workflow YAML file(s). Multiple files are merged into a single workflow.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    file: Annotated[
        Optional[List[Path]],
        typer.Option(
            "-f",
            "--file",
            help="Path to sflow YAML workflow file(s). Can be specified multiple times to merge configs.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate configuration and show execution plan without running",
        ),
    ] = False,
    task: Annotated[
        Optional[str],
        typer.Option(
            "--task",
            "-t",
            help="Run only a specific task (creates new execution context)",
        ),
    ] = None,
    skip_dependencies: Annotated[
        bool,
        typer.Option(
            "--skip-dependencies",
            help="Skip task dependencies when running a specific task",
        ),
    ] = False,
    resume: Annotated[
        Optional[str],
        typer.Option(
            "--resume",
            help="Resume a previously failed workflow by workflow ID",
        ),
    ] = None,
    set_var: Annotated[
        Optional[List[str]],
        typer.Option(
            "--set",
            "-s",
            help="Override variable value or domain (format: KEY=VALUE or KEY=[1,2,3] for domain). Can be used multiple times.",
        ),
    ] = None,
    artifact: Annotated[
        Optional[List[str]],
        typer.Option(
            "--artifact",
            "-a",
            help="Override artifact URI (format: NAME=URI, can be used multiple times)",
        ),
    ] = None,
    missable_tasks: Annotated[
        Optional[List[str]],
        typer.Option(
            "--missable-tasks",
            "-M",
            help="Task names or glob patterns (e.g. 'prefill_*') that may be absent when composing "
            "modular configs from multiple files. Absent missable tasks are removed from depends_on "
            "and probes with a warning. Only valid with multiple -f files. Repeatable.",
        ),
    ] = None,
    extra_args: Annotated[
        Optional[List[str]],
        typer.Option(
            "--extra-args",
            "-e",
            help="Extra args to pass to slurm backend (e.g. --gpus-per-node=4). Merged with config extra_args and deduplicated.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (debug, info, warning, error, critical). Default: info.",
        ),
    ] = "info",
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
    tui: Annotated[
        bool,
        typer.Option(
            "--tui",
            help="Enable Rich TUI (left: task table, right: log tail).",
        ),
    ] = False,
    tui_refresh: Annotated[
        int,
        typer.Option(
            "--tui-refresh",
            help="TUI refresh rate in frames per second.",
            min=1,
        ),
    ] = 2,
):
    """
    Run a workflow from one or more sflow YAML files.

    When multiple files are given, they are merged into a single workflow
    (variables/artifacts/backends/operators merge by name; tasks concatenate).

    Examples:
        # Basic workflow execution
        sflow run workflow.yaml

        # Merge multiple config files (space-separated)
        sflow run backends.yaml tasks.yaml overrides.yaml

        # Merge multiple config files (repeated -f)
        sflow run -f backends.yaml -f tasks.yaml -f overrides.yaml

        # Dry run - validation only
        sflow run workflow.yaml --dry-run

        # Run with variable overrides
        sflow run workflow.yaml --set SLURM_PARTITION=debug --set NUM_GPUS=4

        # Run with artifact override
        sflow run workflow.yaml --artifact MODEL=fs:///path/to/model
    """
    try:
        files = list(src_files or []) + list(file or [])
        if not files:
            files = [Path("sflow.yaml").resolve()]
        if missable_tasks and len(files) < 2:
            typer.echo(
                "Error: --missable-tasks is only valid with multiple input files (modular configs).",
                err=True,
            )
            raise typer.Exit(code=1)
        tui_enabled = bool(tui) and not bool(dry_run)
        if tui and dry_run:
            typer.echo("⚠ --tui is ignored in --dry-run mode (no live execution).")

        # Configure logging as early as possible.
        # - TUI mode: disable console handler so Live UI isn't interleaved with plain logs.
        configure_logging(level=log_level, console=not tui_enabled)

        # In TUI mode, capture all logs into a shared buffer used by the right pane.
        log_buffer = None
        log_handler = None
        if tui_enabled:
            from sflow.ui.rich_tui import attach_tui_log_buffer, detach_tui_log_buffer

            log_buffer = deque(maxlen=4000)
            log_handler = attach_tui_log_buffer(log_buffer)

        if task:
            _logger.info(f"Running specific task: {task}")
            if skip_dependencies:
                _logger.info("Skipping dependencies")
            raise typer.BadParameter(
                "Selective task execution (--task) is not yet implemented"
            )

        if dry_run:
            _logger.info("Starting dry run (validation only)...")
        else:
            _logger.info("Starting workflow execution...")

        workflow_out_dir = None
        try:
            workflow_out_dir = _sflow_app.run(
                file=files,
                dry_run=dry_run,
                resume=resume,
                variable_overrides=set_var,
                artifact_overrides=artifact,
                missable_tasks=missable_tasks,
                backend_extra_args=extra_args,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                tui=tui_enabled,
                tui_log_buffer=log_buffer,
                tui_refresh_per_second=tui_refresh if tui_enabled else None,
            )
        finally:
            if log_handler is not None:
                detach_tui_log_buffer(log_handler)

        if dry_run:
            _logger.info("Dry run completed successfully")
            typer.echo("✓ Configuration is valid")
        else:
            _logger.info("Workflow completed successfully")
            typer.echo("✓ Workflow completed")
            if workflow_out_dir:
                typer.echo(f"  Output folder: {workflow_out_dir}")

    except ValueError as e:
        msg = enrich_error_with_location(str(e), files)
        _logger.error(f"Configuration error: {msg}")
        typer.echo(f"✗ Configuration error: {msg}", err=True)
        if _sflow_app.last_workflow_output_dir:
            typer.echo(
                f"  Output folder: {_sflow_app.last_workflow_output_dir}", err=True
            )
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        _logger.error(f"File not found: {e}")
        typer.echo(f"✗ File not found: {e}", err=True)
        if _sflow_app.last_workflow_output_dir:
            typer.echo(
                f"  Output folder: {_sflow_app.last_workflow_output_dir}", err=True
            )
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        _logger.info("Workflow cancelled by user")
        typer.echo("\n⚠ Workflow cancelled")
        if _sflow_app.last_workflow_output_dir:
            typer.echo(f"  Output folder: {_sflow_app.last_workflow_output_dir}")
        raise typer.Exit(code=130)
    except Exception as e:
        _logger.exception(f"Workflow execution failed: {e}")
        typer.echo(f"✗ Workflow failed: {e}", err=True)
        if _sflow_app.last_workflow_output_dir:
            typer.echo(
                f"  Output folder: {_sflow_app.last_workflow_output_dir}", err=True
            )
        raise typer.Exit(code=1)
