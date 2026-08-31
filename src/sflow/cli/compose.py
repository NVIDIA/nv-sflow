# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for composing multiple sflow YAML config files into a single valid config
with variables resolved inline.
"""

from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import typer
import yaml

from sflow.app.monitor_cli import inject_cli_monitors_into_dict
from sflow.cli import DOCS_URL, app
from sflow.cli._args import (
    SshFetchOption,
    SshFollowOption,
    SshOption,
    SshRemoteRootOption,
    SshTtyOption,
)
from sflow.config.loader import (
    ConfigLoader,
    _normalize_script_plain_mappings,
    merge_config_dicts,
)
from sflow.logging import configure_logging, get_logger
from sflow.resolution import ExpressionResolver, resolve_variables_inline
from sflow.runtime_info import log_runtime_info

_logger = get_logger(__name__)


def _merged_section_to_list(section: Any) -> list:
    """Convert a name-keyed dict section back to list-of-dicts with 'name' inside each entry."""
    if section is None:
        return []
    if isinstance(section, list):
        return section
    if isinstance(section, dict):
        out: list[dict] = []
        for name, entry in section.items():
            if isinstance(entry, dict):
                out.append({"name": name, **entry})
            else:
                out.append({"name": name, "value": entry})
        return out
    return []


def _normalize_merged_dict(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a merged config dict so all named sections are in list format for clean YAML."""
    for key in ("variables", "artifacts", "backends", "operators"):
        if key in merged and merged[key] is not None:
            merged[key] = _merged_section_to_list(merged[key])

    wf = merged.get("workflow")
    if wf and isinstance(wf, dict):
        if "variables" in wf and wf["variables"] is not None:
            wf["variables"] = _merged_section_to_list(wf["variables"])

    return merged


def _strip_none_values(obj: Any) -> Any:
    """Recursively remove keys with None values from dicts for cleaner YAML output."""
    if isinstance(obj, dict):
        return {k: _strip_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none_values(item) for item in obj]
    return obj


def _compose_files(
    files: List[Path],
    set_var: List[str] | None,
    artifact_overrides: List[str] | None,
    log_level: str,
    resolve: bool = False,
    missable_tasks: List[str] | None = None,
    quiet_missable: bool = False,
    enable_workflow_monitor: bool = False,
    enable_task_monitors: List[str] | None = None,
) -> str:
    """Compose multiple YAML files into a single YAML string.

    When *resolve* is True, resolvable variables are inlined and removed.
    """
    resolver = ExpressionResolver()
    config_dicts: List[Dict[str, Any]] = []
    for path in files:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError(f"Configuration file is empty: {path}")
        _normalize_script_plain_mappings(data)
        syntax = resolver.validate_syntax(data, location=str(path))
        if not syntax.valid:
            details = "\n".join(str(e) for e in syntax.errors)
            raise ValueError(
                f"Expression syntax validation failed in {path}:\n{details}"
            )
        config_dicts.append(data)

    override_warnings: List[str] = []
    merged = merge_config_dicts(
        config_dicts,
        source_labels=[str(p) for p in files],
        override_warnings=override_warnings,
    )

    loader = ConfigLoader()
    if set_var:
        loader._apply_variable_overrides(merged, set_var)
    if artifact_overrides:
        loader._apply_artifact_overrides(merged, artifact_overrides)

    merged = _normalize_merged_dict(merged)

    if missable_tasks:
        from sflow.config.loader import strip_missable_tasks

        missable_stripped = strip_missable_tasks(merged, missable_tasks)
        if missable_stripped and not quiet_missable:
            _logger.warning(
                f"Missable tasks: removed {len(missable_stripped)} reference(s) to absent tasks:"
            )
            for _ms in missable_stripped:
                _logger.warning(f"  ⚠ {_ms}")

    # Inject CLI-enabled monitors so the composed snapshot reflects them.
    inject_cli_monitors_into_dict(
        merged,
        enable_workflow_monitor=enable_workflow_monitor,
        enable_task_monitors=enable_task_monitors,
    )

    from pydantic import ValidationError

    from sflow.config.schema import SflowConfig, validate_node_exclude_indices

    try:
        config = SflowConfig.model_validate(merged)
    except ValidationError as e:
        raise ValueError(f"Composed configuration validation failed:\n{e}")

    validate_node_exclude_indices(config)

    if resolve:
        merged = resolve_variables_inline(merged)

    if override_warnings:
        for w in override_warnings:
            typer.echo(f"  compose override: {w}", err=True)

    cleaned = _strip_none_values(merged)

    def _clean_multiline(text: str) -> str:
        """Strip trailing whitespace per line so PyYAML accepts literal block style."""
        lines = text.split("\n")
        cleaned_lines = [line.rstrip() for line in lines]
        result = "\n".join(cleaned_lines)
        if not result.endswith("\n"):
            result += "\n"
        return result

    class _BlockStringDumper(yaml.Dumper):
        pass

    def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
        if "\n" in data:
            data = _clean_multiline(data)
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    _BlockStringDumper.add_representer(str, _str_representer)

    yaml_output = yaml.dump(
        cleaned,
        Dumper=_BlockStringDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=2000,
    )

    top_level_keys = set(cleaned.keys())
    lines = yaml_output.splitlines()
    result_lines: list[str] = []
    in_tasks = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        key_part = stripped.split(":")[0] if ":" in stripped else ""

        if not line.startswith(" ") and key_part in top_level_keys and result_lines:
            result_lines.append("")

        if stripped == "tasks:":
            in_tasks = True
        elif not line.startswith(" ") and line and not line.startswith("#"):
            in_tasks = False

        is_list_dash = stripped.startswith("- ")
        if is_list_dash and result_lines and result_lines[-1] != "":
            indent = len(line) - len(stripped)
            if indent == 0:
                result_lines.append("")
            elif in_tasks and indent == 2:
                result_lines.append("")

        result_lines.append(line)

    return "\n".join(result_lines) + "\n"


def _run_bulk_compose(
    *,
    csv_path: Path,
    cli_files: list[Path] | None = None,
    cli_set_var: list[str] | None,
    cli_artifact: list[str] | None,
    output_dir: Path,
    log_level: str,
    resolve: bool = False,
    validate: bool = False,
    row_selectors: list[str] | None = None,
    missable_tasks: list[str] | None = None,
) -> None:
    """Compose one YAML file per CSV row.

    When *cli_files* are provided alongside the CSV, they are prepended to
    each row's ``sflow_config_file`` list (common base configs first, then
    row-specific variant configs).  Duplicates are removed by resolved path,
    keeping the first occurrence.
    """
    from datetime import datetime

    from sflow.cli.batch import (
        _RESERVED_CSV_COLUMNS,
        _classify_csv_columns,
        _derive_row_name,
        _parse_kv_list,
        build_all_row_configs,
        build_row_naming_ctx,
        merge_row_overrides,
        parse_row_selector,
        read_bulk_csv,
        resolve_row_files,
        row_missable,
    )

    columns, rows = read_bulk_csv(csv_path)

    csv_dir = csv_path.parent
    resolved_cli_files = [p.resolve() for p in (cli_files or [])]
    cli_var_map = _parse_kv_list(cli_set_var)
    cli_art_map = _parse_kv_list(cli_artifact)

    all_row_configs = build_all_row_configs(rows, csv_dir, resolved_cli_files, missable_tasks)
    var_cols, art_cols = _classify_csv_columns(columns, all_row_configs)

    if resolved_cli_files:
        cli_stems = ", ".join(p.name for p in resolved_cli_files)
        _logger.info(f"CLI config file(s) prepended to each CSV row: {cli_stems}")

    overlap_vars = set(cli_var_map.keys()) & var_cols
    overlap_arts = set(cli_art_map.keys()) & art_cols
    for name in sorted(overlap_vars):
        typer.echo(
            f"  Warning: variable '{name}' specified via --set and also in CSV; "
            f"CLI --set value will take precedence over CSV.",
            err=True,
        )
    for name in sorted(overlap_arts):
        typer.echo(
            f"  Warning: artifact '{name}' specified via --artifact and also in CSV; "
            f"CLI --artifact value will take precedence.",
            err=True,
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bulk_dir = output_dir / f"compose_{stamp}"
    bulk_dir.mkdir(parents=True, exist_ok=True)

    summary: list[str] = []
    warnings: list[str] = []
    failed_count = 0
    row_indices: set[int] | None = None
    if row_selectors:
        row_indices = set(parse_row_selector(row_selectors, n_rows=len(rows)))
    naming_ctx = build_row_naming_ctx(rows)

    for idx, row in enumerate(rows, start=1):
        if row_indices is not None and idx not in row_indices:
            continue
        config_files = resolve_row_files(row, csv_dir, resolved_cli_files)
        set_var, artifacts = merge_row_overrides(row, var_cols, art_cols, cli_var_map, cli_art_map)
        effective_missable = row_missable(row, missable_tasks)

        overrides_desc = ", ".join(
            f"{col}={row[col]}"
            for col in columns
            if col not in _RESERVED_CSV_COLUMNS and row.get(col)
        )

        row_name = _derive_row_name(row, idx, naming_ctx)
        out_path = bulk_dir / f"{row_name}.yaml"
        try:
            yaml_output = _compose_files(
                config_files,
                set_var,
                artifacts,
                log_level,
                resolve=resolve,
                missable_tasks=effective_missable,
            )
            out_path.write_text(yaml_output)
        except (ValueError, FileNotFoundError) as e:
            failed_count += 1
            summary.append(f"  [{idx}] FAILED: ({overrides_desc}) -> {e}")
            continue

        if validate:
            from sflow.app.sflow import SflowApp

            try:
                SflowApp().run(
                    file=config_files,
                    dry_run=True,
                    variable_overrides=list(set_var) if set_var else None,
                    artifact_overrides=list(artifacts) if artifacts else None,
                    missable_tasks=effective_missable,
                )
                summary.append(f"  [{idx}] {out_path.name}: ({overrides_desc})")
            except Exception as e:
                err_short = str(e).split("\n")[0]
                summary.append(f"  [{idx}] {out_path.name}: ({overrides_desc})")
                warnings.append(f"  [{idx}] {out_path.name}: {err_short}")
        else:
            summary.append(f"  [{idx}] {out_path.name}: ({overrides_desc})")

    processed = len(summary)
    generated = processed - failed_count
    row_info = (
        f" (rows: {','.join(str(r) for r in sorted(row_indices))})"
        if row_indices
        else ""
    )
    typer.echo(
        f"\nBulk compose: {generated}/{processed} configs generated from {csv_path.name}{row_info}"
        + (f" ({failed_count} failed validation)" if failed_count else "")
    )
    typer.echo(f"Output directory: {bulk_dir}")
    for line in summary:
        typer.echo(line)
    if warnings:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"WARNINGS: {len(warnings)} config(s) failed dry-run validation:")
        typer.echo(f"{'=' * 60}")
        for w in warnings:
            typer.echo(w)
        typer.echo(f"{'=' * 60}")
    typer.echo(f"\nOutput directory: {bulk_dir}")


@app.command(name="compose", epilog=f"Documentation: {DOCS_URL}")
def compose(
    src_files: Annotated[
        Optional[List[Path]],
        typer.Argument(
            help="Workflow YAML file(s) to merge.",
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
            help="Path to sflow YAML workflow file(s). Can be specified multiple times.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    set_var: Annotated[
        Optional[List[str]],
        typer.Option(
            "--set",
            "-s",
            help="Override variable value (format: KEY=VALUE). Can be used multiple times.",
        ),
    ] = None,
    artifact: Annotated[
        Optional[List[str]],
        typer.Option(
            "--artifact",
            "-a",
            help="Override artifact URI (format: NAME=URI). Can be used multiple times.",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Output file path (single compose) or directory (bulk compose). "
            "If not specified, writes to stdout (single) or ./sflow_output/ (bulk).",
            resolve_path=True,
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
    resolve: Annotated[
        bool,
        typer.Option(
            "-r",
            "--resolve",
            help="Resolve all resolvable variables to literal values inline and remove them "
            "from the variables section. Without this flag, variables are kept as-is.",
        ),
    ] = False,
    validate: Annotated[
        bool,
        typer.Option(
            "-vl",
            "--validate",
            help="Run dry-run validation on each composed config to check for resource "
            "issues (e.g. GPU over-subscription). Configs are still generated but "
            "warnings are shown for rows that would fail at runtime.",
        ),
    ] = False,
    bulk_input: Annotated[
        Optional[Path],
        typer.Option(
            "--bulk-input",
            "-b",
            help="CSV file for bulk compose. "
            "Reserved columns: 'sflow_config_file' (required, space-separated YAML paths), "
            "'job_name' (optional, explicit name for the output YAML file). "
            "All other columns are matched to variable or artifact names as overrides. "
            "When 'job_name' is absent, filenames are auto-derived from unique config file stems.",
        ),
    ] = None,
    row: Annotated[
        Optional[List[str]],
        typer.Option(
            "--row",
            help="Only process specific CSV row(s) by 1-based index. "
            "Supports: single (--row 1), negative (--row=-1 → last row), "
            "multiple (--row 1 --row 3), "
            "comma-separated (--row 1,3,5), Python-style slices with exclusive end "
            "(--row 1:4 → rows 1,2,3; --row 1:6:2 → rows 1,3,5; --row [1:4]), "
            "and open-ended/negative slices (--row=-3: → last 3 rows; --row 3: → row 3 to end). "
            "Negative indices use --row=N syntax to avoid flag ambiguity. "
            "Requires --bulk-input.",
        ),
    ] = None,
    missable_tasks: Annotated[
        Optional[List[str]],
        typer.Option(
            "--missable-tasks",
            "-M",
            help="Task names or glob patterns (e.g. 'prefill_*') that may be absent when composing "
            "modular configs from multiple files. Absent missable tasks are removed from depends_on "
            "and probes with a warning. Only valid with multiple input files or --bulk-input. Repeatable.",
        ),
    ] = None,
    ssh: SshOption = None,
    ssh_follow: SshFollowOption = "none",
    ssh_fetch: SshFetchOption = "logs",
    ssh_remote_root: SshRemoteRootOption = None,
    ssh_tty: SshTtyOption = "auto",
):
    """
    Compose multiple sflow YAML files into a single valid workflow config.

    The compose follows the same strategy as 'sflow run' with multiple files:
    variables/artifacts/backends/operators merge by name (later wins),
    tasks are concatenated in file order.

    The composed config is validated against the sflow schema before output.

    Examples:
        # Compose and print to stdout
        sflow compose backends.yaml workflow.yaml tasks.yaml

        # Compose and write to file
        sflow compose -f backends.yaml -f tasks.yaml -o merged.yaml

        # Compose with variable overrides
        sflow compose backends.yaml tasks.yaml --set SLURM_NODES=4 -o merged.yaml

        # Compose and resolve all variables to literal values
        sflow compose backends.yaml tasks.yaml --resolve -o resolved.yaml

        # Bulk compose: generate one composed YAML per CSV row
        sflow compose --bulk-input jobs.csv -o output_dir

        # Bulk compose with common base config(s) from CLI + variants from CSV
        sflow compose backends.yaml common.yaml --bulk-input variants.csv -o output_dir

        # Bulk compose with validation (warns about resource issues)
        sflow compose --bulk-input jobs.csv --validate -o output_dir
    """
    if ssh is not None:
        from sflow.cli._ssh_delegate import delegate

        delegate(
            "compose",
            connection=ssh,
            follow=ssh_follow,
            fetch=ssh_fetch,
            remote_root=ssh_remote_root,
            tty=ssh_tty,
            workspace_dir=None,
            output_dir=None,
            input_files=[
                *list(src_files or []),
                *list(file or []),
                *([bulk_input] if bulk_input else []),
            ],
            artifact_overrides=artifact,
            bulk_input=bulk_input,
            compose_output=output,
            compose_bulk=bulk_input is not None,
        )

    try:
        configure_logging(
            level=log_level, console=output is not None or bulk_input is not None
        )
        log_runtime_info()

        if row and bulk_input is None:
            typer.echo("Error: --row requires --bulk-input.", err=True)
            raise typer.Exit(code=1)

        # --- Bulk-input mode ---
        if bulk_input is not None:
            cli_files = list(src_files or []) + list(file or [])
            out_dir = output if output else Path.cwd() / "sflow_output"
            _run_bulk_compose(
                csv_path=bulk_input,
                cli_files=cli_files or None,
                cli_set_var=set_var,
                cli_artifact=artifact,
                output_dir=out_dir,
                log_level=log_level,
                resolve=resolve,
                validate=validate,
                row_selectors=row,
                missable_tasks=missable_tasks,
            )
            return

        # --- Single compose mode ---
        files = list(src_files or []) + list(file or [])
        if not files:
            typer.echo("Error: no input files provided.", err=True)
            raise typer.Exit(code=1)

        csv_files = [f for f in files if f.suffix.lower() == ".csv"]
        if csv_files:
            names = ", ".join(str(f) for f in csv_files)
            typer.echo(
                f"Error: CSV file(s) detected in input: {names}\n"
                f"  CSV files cannot be used as workflow YAML inputs directly.\n"
                f"  Did you mean to use --bulk-input (-b)?\n"
                f"  Example: sflow compose --bulk-input {csv_files[0]}",
                err=True,
            )
            raise typer.Exit(code=1)

        if missable_tasks and len(files) < 2:
            typer.echo(
                "Error: --missable-tasks is only valid with multiple input files (modular configs).",
                err=True,
            )
            raise typer.Exit(code=1)

        yaml_output = _compose_files(
            files,
            set_var,
            artifact,
            log_level,
            resolve=resolve,
            missable_tasks=missable_tasks,
        )

        if validate:
            from sflow.app.sflow import SflowApp

            try:
                SflowApp().run(
                    file=files,
                    dry_run=True,
                    variable_overrides=set_var,
                    artifact_overrides=artifact,
                    missable_tasks=missable_tasks,
                )
                typer.echo("Dry-run validation passed.", err=True)
            except Exception as e:
                err_short = str(e).split("\n")[0]
                typer.echo(f"WARNING: dry-run validation failed: {err_short}", err=True)

        if output is not None:
            if output.is_dir():
                typer.echo(
                    f"Error: output path '{output}' is a directory. "
                    f"For single compose, -o must be a file path (e.g. -o merged.yaml). "
                    f"For bulk compose, use --bulk-input.",
                    err=True,
                )
                raise typer.Exit(code=1)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(yaml_output)
            _logger.info(f"Composed config written to {output}")
            typer.echo(f"Composed {len(files)} files -> {output}")
        else:
            typer.echo(yaml_output)

    except ValueError as e:
        _logger.error(f"Compose error: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        _logger.error(f"File not found: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
