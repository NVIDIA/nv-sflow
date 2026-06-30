# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for running workflows.
"""

import os
import threading
from collections import deque
from pathlib import Path
from typing import Annotated, List, Optional

import typer

from sflow.app.sflow import SflowApp
from sflow.cli import DOCS_URL, app
from sflow.core.kubectl_config import KubectlConfig
from sflow.cli._args import (
    EnableTaskMonitorOption,
    EnableWorkflowMonitorOption,
    split_list_arg,
)
from sflow.core.log_offload import OFFLOAD_TASK_LOGS_ENV
from sflow.logging import configure_logging, get_logger
from sflow.resolution import enrich_error_with_location
from sflow.runtime_info import log_runtime_info
from sflow.utils.extra_args import dedup_merge_extra_args

_logger = get_logger(__name__)

_sflow_app = SflowApp()


def _read_upload_summary_lines(workflow_out_dir: Path) -> list[str]:
    """Read the detailed Uploads section from ``sflow_summary.log`` if present."""
    summary_path = workflow_out_dir / "sflow_summary.log"
    try:
        lines = summary_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for idx, line in enumerate(lines):
        if line != "Uploads":
            continue
        if idx + 1 >= len(lines) or lines[idx + 1] != "-------":
            continue
        upload_lines: list[str] = []
        for section_line in lines[idx + 2 :]:
            if not section_line:
                break
            if section_line.startswith("Counts :"):
                _, value = section_line.split(":", 1)
                if value := value.strip():
                    upload_lines.append(value)
                continue
            upload_lines.append(section_line.strip())
        return upload_lines
    return []


def _print_run_artifacts(workflow_out_dir: Path, *, err: bool = False) -> None:
    typer.echo(f"  Output folder: {workflow_out_dir}", err=err)
    typer.echo(f"  Summary: {workflow_out_dir / 'sflow_summary.log'}", err=err)
    command_logs = sorted(workflow_out_dir.glob("*_cmds.log"))
    if command_logs:
        paths = ", ".join(str(path) for path in command_logs)
        typer.echo(f"  Command logs: {paths}", err=err)
    upload_lines = _read_upload_summary_lines(workflow_out_dir)
    if upload_lines:
        typer.echo("  Uploads:", err=err)
        for line in upload_lines:
            typer.echo(f"    {line}", err=err)


def _resolve_bulk_input_row(
    *,
    bulk_input: Path,
    row_selectors: list[str],
    cli_files: list[Path],
    cli_set_var: list[str] | None,
    cli_artifact: list[str] | None,
    cli_missable: list[str] | None,
) -> tuple[list[Path], list[str] | None, list[str] | None, list[str] | None]:
    """Resolve a single CSV row into (files, set_var, artifact, missable_tasks).

    Delegates to :func:`sflow.cli.batch.resolve_csv_row` for all CSV parsing,
    column classification, and override merging.
    """
    from sflow.cli.batch import parse_row_selector, resolve_csv_row

    parsed_rows = parse_row_selector(row_selectors)
    if len(parsed_rows) != 1:
        raise typer.BadParameter(
            f"--bulk-input with sflow run requires exactly one row, "
            f"got {len(parsed_rows)}: {parsed_rows}"
        )

    try:
        return resolve_csv_row(
            csv_path=bulk_input,
            row_idx=parsed_rows[0],
            cli_files=cli_files or None,
            cli_set_var=cli_set_var,
            cli_artifact=cli_artifact,
            cli_missable=cli_missable,
        )
    except IndexError as e:
        raise typer.BadParameter(str(e)) from e


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
            help="Generic, backend-agnostic extra args. They are forwarded to "
            "whichever backend the recipe uses: merged into each Slurm backend's "
            "salloc, each docker backend's `docker run`, and every kubectl call's "
            "global flags. Deduplicated by option (CLI wins over the recipe; a more "
            "specific --extra-salloc-args / --extra-docker-args / --extra-kubectl-args "
            "wins over --extra-args on a conflicting option). Repeatable.",
        ),
    ] = None,
    extra_salloc_args: Annotated[
        Optional[List[str]],
        typer.Option(
            "--extra-salloc-args",
            help="Extra args merged into each Slurm backend's salloc (e.g. "
            "--gpus-per-node=4), deduplicated by option against the backend "
            "config's extra_args (CLI wins). Slurm backends only; in a "
            "multi-backend recipe they apply to every Slurm backend's salloc.",
        ),
    ] = None,
    extra_docker_args: Annotated[
        Optional[List[str]],
        typer.Option(
            "--extra-docker-args",
            help="Extra args merged into each docker backend's `docker run` (e.g. "
            "--shm-size=16g), deduplicated by option against the backend config's "
            "extra_args (CLI wins). Docker backends only.",
        ),
    ] = None,
    kubeconfig: Annotated[
        Optional[Path],
        typer.Option(
            "--kubeconfig",
            help="Path to the kubeconfig file for kubernetes backends (also exported "
            "as KUBECONFIG). Default: $KUBECONFIG or ~/.kube/config.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    kube_context: Annotated[
        Optional[str],
        typer.Option(
            "--kube-context",
            help="kubeconfig context to use for kubernetes backends "
            "(default: the kubeconfig's current-context).",
        ),
    ] = None,
    kube_namespace: Annotated[
        Optional[str],
        typer.Option(
            "--kube-namespace",
            help="Override the namespace for all kubernetes backends.",
        ),
    ] = None,
    extra_kubectl_args: Annotated[
        Optional[List[str]],
        typer.Option(
            "--extra-kubectl-args",
            help="Extra global kubectl flag applied to every kubectl call "
            "(e.g. --extra-kubectl-args=--insecure-skip-tls-verify). Repeatable.",
        ),
    ] = None,
    kube_exclude_node: Annotated[
        Optional[List[str]],
        typer.Option(
            "--kube-exclude-node",
            help="Node hostname to keep all kubernetes pods off (e.g. a node with a "
            "broken driver). Applied as a kubernetes.io/hostname NotIn nodeAffinity "
            "on the reservation pods, so the run avoids it without a cluster-wide "
            "cordon. Repeatable; keeps volatile node info out of the recipe.",
        ),
    ] = None,
    enable_workflow_monitor: EnableWorkflowMonitorOption = False,
    enable_task_monitor: EnableTaskMonitorOption = None,
    bulk_input: Annotated[
        Optional[Path],
        typer.Option(
            "--bulk-input",
            "-b",
            help="CSV file to resolve config files and variable overrides from a single row. "
            "Requires --row with a single row index (1-based). "
            "The 'sflow_config_file' column provides YAML paths; other non-reserved columns "
            "are treated as variable or artifact overrides.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    row: Annotated[
        Optional[List[str]],
        typer.Option(
            "--row",
            help="1-based row index in the CSV (requires --bulk-input). Only a single row is supported. "
            "Negative indices select from the end (--row=-1 → last row).",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show full per-task details in the --dry-run plan "
            "(default: one-line summary per task)",
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
    offload_task_logs: Annotated[
        Optional[bool],
        typer.Option(
            "--offload-task-logs/--no-offload-task-logs",
            help="Have each task write its own log via the operator (srun --output, "
            "or a host-side shell redirect for local/docker) through a compute-side "
            "prefixer, instead of streaming it through the sflow driver. ON by default "
            "for local/docker/slurm in non-interactive runs (auto-streams on a "
            "TTY/--tui); pass --no-offload-task-logs to force streaming. Overrides the "
            "backend's offload_task_logs; no-op for k8s/ssh.",
        ),
    ] = None,
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

        # Run a single row from a CSV (bulk-input mode)
        sflow run --bulk-input jobs.csv --row 3

        # Enable default hardware monitoring without editing the recipe
        sflow run workflow.yaml --enable-workflow-monitor
        sflow run workflow.yaml --enable-task-monitor aiperf,decode_server

        # Run a CSV row with additional CLI config files prepended
        sflow run -f common.yaml --bulk-input jobs.csv --row 1
    """
    try:
        if row and bulk_input is None:
            typer.echo("Error: --row requires --bulk-input.", err=True)
            raise typer.Exit(code=1)
        if bulk_input is not None and not row:
            typer.echo("Error: --bulk-input requires --row with a single row index.", err=True)
            raise typer.Exit(code=1)

        if bulk_input is not None:
            files, set_var, artifact, missable_tasks = _resolve_bulk_input_row(
                bulk_input=bulk_input,
                row_selectors=row,
                cli_files=list(src_files or []) + list(file or []),
                cli_set_var=set_var,
                cli_artifact=artifact,
                cli_missable=missable_tasks,
            )
        else:
            files = list(src_files or []) + list(file or [])

        if not files:
            files = [Path("sflow.yaml").resolve()]
        if missable_tasks and len(files) < 2:
            typer.echo(
                "Error: --missable-tasks is only valid with multiple input files (modular configs).",
                err=True,
            )
            raise typer.Exit(code=1)
        # Accept comma- and/or whitespace-separated task names (and repeated flags).
        enable_task_monitor = split_list_arg(enable_task_monitor)

        tui_enabled = bool(tui) and not bool(dry_run)
        if tui and dry_run:
            typer.echo("⚠ --tui is ignored in --dry-run mode (no live execution).")

        # Configure logging as early as possible.
        # - TUI mode: disable console handler so Live UI isn't interleaved with plain logs.
        configure_logging(level=log_level, console=not tui_enabled)
        log_runtime_info()

        # In TUI mode, capture all logs into a shared buffer used by the right pane.
        log_buffer = None
        log_lock = None
        log_handler = None
        if tui_enabled:
            from sflow.ui.rich_tui import attach_tui_log_buffer, detach_tui_log_buffer

            log_buffer = deque(maxlen=4000)
            log_lock = threading.Lock()
            log_handler = attach_tui_log_buffer(log_buffer, log_lock=log_lock)

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

        # Per-invocation override for the per-task log offload. Operators read
        # this env and it wins over backend config, so users can toggle offload
        # without editing recipes (slurm, local, docker).
        if offload_task_logs is not None:
            os.environ[OFFLOAD_TASK_LOGS_ENV] = "1" if offload_task_logs else "0"

        # --extra-args is the generic, backend-agnostic flag: fan its values out to
        # every typed channel (salloc / docker run / kubectl global flags) so whichever
        # backend the recipe uses picks them up. The matching --extra-<type>-args is the
        # `override` arg to dedup_merge_extra_args, so a specific flag wins over the
        # generic one on a conflicting option. Each backend then de-dups again against
        # its own config extra_args (CLI wins) in SflowApp.run.
        generic_extra_args = list(extra_args or [])
        salloc_args = dedup_merge_extra_args(generic_extra_args, list(extra_salloc_args or []))
        docker_args = dedup_merge_extra_args(generic_extra_args, list(extra_docker_args or []))
        kubectl_args = dedup_merge_extra_args(generic_extra_args, list(extra_kubectl_args or []))

        kube_cfg = KubectlConfig(
            kubeconfig=str(kubeconfig) if kubeconfig else None,
            context=kube_context,
            namespace=kube_namespace,
            extra_args=kubectl_args,
            exclude_nodes=list(kube_exclude_node or []),
        )

        # Route the resolved per-type args to their backend type: slurm -> salloc,
        # docker -> `docker run`. Empty buckets are omitted so backends without CLI
        # args are untouched.
        backend_extra_args_by_type: dict[str, list[str]] = {}
        if salloc_args:
            backend_extra_args_by_type["slurm"] = salloc_args
        if docker_args:
            backend_extra_args_by_type["docker"] = docker_args

        workflow_out_dir = None
        try:
            workflow_out_dir = _sflow_app.run(
                file=files,
                dry_run=dry_run,
                verbose=verbose,
                resume=resume,
                variable_overrides=set_var,
                artifact_overrides=artifact,
                missable_tasks=missable_tasks,
                backend_extra_args_by_type=backend_extra_args_by_type or None,
                kubectl_config=kube_cfg,
                enable_workflow_monitor=enable_workflow_monitor,
                enable_task_monitors=enable_task_monitor,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                tui=tui_enabled,
                tui_log_buffer=log_buffer,
                tui_log_lock=log_lock,
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
                _print_run_artifacts(workflow_out_dir)

    except ValueError as e:
        msg = enrich_error_with_location(str(e), files)
        _logger.error(f"Configuration error: {msg}")
        typer.echo(f"✗ Configuration error: {msg}", err=True)
        if _sflow_app.last_workflow_output_dir:
            _print_run_artifacts(_sflow_app.last_workflow_output_dir, err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        _logger.error(f"File not found: {e}")
        typer.echo(f"✗ File not found: {e}", err=True)
        if _sflow_app.last_workflow_output_dir:
            _print_run_artifacts(_sflow_app.last_workflow_output_dir, err=True)
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        _logger.info("Workflow cancelled by user")
        typer.echo("\n⚠ Workflow cancelled")
        if _sflow_app.last_workflow_output_dir:
            _print_run_artifacts(_sflow_app.last_workflow_output_dir)
        raise typer.Exit(code=130)
    except Exception as e:
        _logger.exception(f"Workflow execution failed: {e}")
        typer.echo(f"✗ Workflow failed: {e}", err=True)
        if _sflow_app.last_workflow_output_dir:
            _print_run_artifacts(_sflow_app.last_workflow_output_dir, err=True)
        raise typer.Exit(code=1)
