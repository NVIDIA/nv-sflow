# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for composing multiple sflow YAML config files into a single valid config
with variables resolved inline.
"""

import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import typer
import yaml
from jinja2 import StrictUndefined, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

from sflow.cli import DOCS_URL, app
from sflow.config.loader import ConfigLoader, merge_config_dicts
from sflow.config.resolver import ExpressionResolver
from sflow.logging import configure_logging, get_logger

_logger = get_logger(__name__)

_EXPR_RE = re.compile(r"\$\{\{(.+?)\}\}")
_SHELL_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Env vars that are NOT sflow variables — never resolve these.
_BUILTIN_ENV_VARS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "SLURM_NODEID",
        "SLURMD_NODENAME",
        "SLURM_JOB_ID",
        "SLURM_JOBID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NNODES",
        "SLURM_STEP_ID",
        "SLURM_STEP_NODELIST",
        "SLURM_JOB_NODELIST",
        "HOME",
        "USER",
        "PATH",
        "PWD",
        "HOSTNAME",
        "SHELL",
        "LANG",
        "TERM",
        "SFLOW_WORKSPACE_DIR",
        "SFLOW_OUTPUT_DIR",
        "SFLOW_WORKFLOW_OUTPUT_DIR",
        "SFLOW_TASK_OUTPUT_DIR",
        "SFLOW_TASK_ASSIGNED_NODE_IPS",
        "COLUMNS",
        "IFS",
    }
)


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


def _extract_variables(merged: Dict[str, Any]) -> dict[str, Any]:
    """Extract a flat {name: value} dict from the variables list in the merged config."""
    out: dict[str, Any] = {}
    for entry in merged.get("variables") or []:
        if isinstance(entry, dict) and "name" in entry:
            out[entry["name"]] = entry.get("value")
    wf = merged.get("workflow")
    if isinstance(wf, dict):
        for entry in wf.get("variables") or []:
            if isinstance(entry, dict) and "name" in entry:
                out[entry["name"]] = entry.get("value")
    return out


def _classify_resolvable(variables: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """Split variables into resolvable (literal or computable) and unresolvable.

    Returns (resolved_values, unresolvable_names).
    Iterates until stable to handle chained dependencies (A references B).
    """
    env = SandboxedEnvironment(
        undefined=StrictUndefined,
        autoescape=False,
        variable_start_string="${{",
        variable_end_string="}}",
    )

    resolved: dict[str, Any] = {}
    pending: dict[str, Any] = dict(variables)

    changed = True
    while changed:
        changed = False
        still_pending: dict[str, Any] = {}
        for name, value in pending.items():
            if not isinstance(value, str) or "${{" not in value:
                resolved[name] = value
                changed = True
                continue
            ctx = {"variables": resolved, **resolved}
            try:
                tpl = env.from_string(str(value))
                result = tpl.render(**ctx)
                resolved[name] = _coerce_type(result)
                changed = True
            except (UndefinedError, Exception):
                still_pending[name] = value
        pending = still_pending

    return resolved, set(pending.keys())


def _coerce_type(value: str) -> Any:
    """Convert a Jinja2 string result back to a native Python type when possible."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.lower() in ("true", "false"):
        return stripped.lower() == "true"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return value


def _resolve_expressions(
    obj: Any, ctx: dict[str, Any], env: SandboxedEnvironment
) -> Any:
    """Walk a data structure, resolving ${{ }} expressions where all refs are available.

    For strings that are a single pure expression, the result is type-coerced
    (e.g. "2" becomes 2).  For mixed strings (literal text + expressions),
    each expression is resolved individually so resolvable parts are inlined
    even when other parts in the same string are not yet resolvable.
    """
    if isinstance(obj, str):
        if "${{" not in obj:
            return obj
        stripped = obj.strip()
        if _EXPR_RE.fullmatch(stripped):
            try:
                result = env.from_string(obj).render(**ctx)
                return _coerce_type(result)
            except (UndefinedError, Exception):
                return obj

        def _replace_match(m: re.Match) -> str:
            expr_text = m.group(0)
            try:
                return env.from_string(expr_text).render(**ctx)
            except (UndefinedError, Exception):
                return expr_text

        return _EXPR_RE.sub(_replace_match, obj)
    if isinstance(obj, list):
        return [_resolve_expressions(item, ctx, env) for item in obj]
    if isinstance(obj, dict):
        return {k: _resolve_expressions(v, ctx, env) for k, v in obj.items()}
    return obj


def _resolve_shell_vars(obj: Any, resolved: dict[str, Any]) -> Any:
    """Replace ${NAME} shell variable references in script strings with literal values."""
    if isinstance(obj, str):

        def _replace(m: re.Match) -> str:
            name = m.group(1)
            if name in _BUILTIN_ENV_VARS or name not in resolved:
                return m.group(0)
            val = resolved[name]
            return str(val)

        return _SHELL_VAR_RE.sub(_replace, obj)
    if isinstance(obj, list):
        return [_resolve_shell_vars(item, resolved) for item in obj]
    if isinstance(obj, dict):
        return {k: _resolve_shell_vars(v, resolved) for k, v in obj.items()}
    return obj


def _remove_resolved_variables(
    section: list | None,
    resolved_names: set[str],
) -> list:
    """Filter a variables list, keeping only entries whose names are NOT in resolved_names."""
    if not section:
        return []
    return [
        entry
        for entry in section
        if not (isinstance(entry, dict) and entry.get("name") in resolved_names)
    ]


def _collect_replica_variable_names(merged: Dict[str, Any]) -> set[str]:
    """Find variable names referenced by any replicas config in any task.

    This includes:
    - ``replicas.variables`` (sweep variables that change per replica)
    - ``replicas.count`` / ``replicas.policy`` (expressions like
      ``${{ variables.NUM_CTX_SERVERS }}``)

    These variables must stay in the variables section so replica
    behaviour remains configurable.
    """
    names: set[str] = set()
    wf = merged.get("workflow")
    if not isinstance(wf, dict):
        return names
    for task in wf.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        replicas = task.get("replicas")
        if not isinstance(replicas, dict):
            continue
        for v in replicas.get("variables") or []:
            if isinstance(v, str):
                names.add(v)
        for field in ("count", "policy"):
            val = replicas.get(field)
            if isinstance(val, str):
                for m in _EXPR_RE.finditer(val):
                    expr = m.group(1).strip()
                    if expr.startswith("variables."):
                        names.add(expr.split(".", 1)[1])
    return names


def _clean_resolved_strings(obj: Any) -> Any:
    """Strip trailing whitespace from resolved script strings.

    YAML block scalars (``>`` / ``|``) append a trailing newline that becomes
    a literal ``\\n`` after Jinja2 rendering.  Cleaning it here keeps the
    composed output readable.
    """
    if isinstance(obj, str):
        return obj.rstrip(" \t\n")
    if isinstance(obj, list):
        return [_clean_resolved_strings(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _clean_resolved_strings(v) for k, v in obj.items()}
    return obj


def _resolve_variables_inline(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve all resolvable variables inline throughout the config.

    - Resolves ${{ variables.X }} expressions to literal values
    - Resolves ${X} shell references in scripts to literal values
    - Removes resolved variables from the variables sections
    - Keeps variables referenced by replicas.variables (needed for sweeps)
    - Keeps expressions that depend on backends/artifacts/task (post-allocation)
    """
    replica_vars = _collect_replica_variable_names(merged)

    variables = _extract_variables(merged)
    resolved, unresolvable = _classify_resolvable(variables)

    # Never resolve replica sweep variables — their value changes per replica.
    for rv in replica_vars:
        resolved.pop(rv, None)

    if not resolved:
        return merged

    env = SandboxedEnvironment(
        undefined=StrictUndefined,
        autoescape=False,
        variable_start_string="${{",
        variable_end_string="}}",
    )
    ctx: dict[str, Any] = {"variables": resolved, **resolved}

    merged = _resolve_expressions(merged, ctx, env)
    merged = _resolve_shell_vars(merged, resolved)
    merged = _clean_resolved_strings(merged)

    removable = set(resolved.keys())

    if "variables" in merged and merged["variables"]:
        merged["variables"] = _remove_resolved_variables(merged["variables"], removable)
        if not merged["variables"]:
            del merged["variables"]

    wf = merged.get("workflow")
    if isinstance(wf, dict) and "variables" in wf and wf["variables"]:
        wf["variables"] = _remove_resolved_variables(wf["variables"], removable)
        if not wf["variables"]:
            del wf["variables"]

    return merged


def _compose_files(
    files: List[Path],
    set_var: List[str] | None,
    artifact_overrides: List[str] | None,
    log_level: str,
    resolve: bool = False,
    missable_tasks: List[str] | None = None,
    quiet_missable: bool = False,
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

    from pydantic import ValidationError

    from sflow.config.schema import SflowConfig, validate_node_exclude_indices

    try:
        config = SflowConfig.model_validate(merged)
    except ValidationError as e:
        raise ValueError(f"Composed configuration validation failed:\n{e}")

    validate_node_exclude_indices(config)

    if resolve:
        merged = _resolve_variables_inline(merged)

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
    row_filter: list[int] | None = None,
    missable_tasks: list[str] | None = None,
) -> None:
    """Compose one YAML file per CSV row.

    When *cli_files* are provided alongside the CSV, they are prepended to
    each row's ``sflow_config_file`` list (common base configs first, then
    row-specific variant configs).  Duplicates are removed by resolved path,
    keeping the first occurrence.
    """
    import csv
    from datetime import datetime

    from sflow.cli.batch import (
        _RESERVED_CSV_COLUMNS,
        _classify_csv_columns,
        _derive_row_name,
        build_row_naming_ctx,
    )

    cli_var_map: dict[str, str] = {}
    for entry in cli_set_var or []:
        if "=" in entry:
            k, v = entry.split("=", 1)
            cli_var_map[k] = v

    cli_art_map: dict[str, str] = {}
    for entry in cli_artifact or []:
        if "=" in entry:
            k, v = entry.split("=", 1)
            cli_art_map[k] = v

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {csv_path}")
        columns = list(reader.fieldnames)
        if "sflow_config_file" not in columns:
            raise ValueError(
                f"CSV file must contain a 'sflow_config_file' column. "
                f"Found columns: {columns}"
            )
        rows: list[dict[str, Any]] = list(reader)

    if not rows:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    csv_dir = csv_path.parent
    resolved_cli_files = [p.resolve() for p in (cli_files or [])]

    def _resolve_config_paths(raw: str) -> list[Path]:
        paths = []
        for p in raw.split():
            fp = Path(p)
            if not fp.is_absolute():
                fp = csv_dir / fp
            paths.append(fp.resolve())
        return paths

    def _merge_and_dedup(base: list[Path], extra: list[Path]) -> list[Path]:
        """Merge two path lists, deduplicating by resolved path (first wins)."""
        seen: set[Path] = set()
        merged: list[Path] = []
        for p in base + extra:
            if p not in seen:
                seen.add(p)
                merged.append(p)
        return merged

    row_configs: list[tuple[list[Path], list[str] | None]] = []
    for r in rows:
        csv_files = _resolve_config_paths(r["sflow_config_file"])
        cfg_files = _merge_and_dedup(resolved_cli_files, csv_files)
        row_m = list(missable_tasks) if missable_tasks else []
        csv_m = (r.get("missable_tasks") or "").strip()
        if csv_m:
            row_m.extend(csv_m.split())
        row_configs.append((cfg_files, row_m or None))
    var_cols, art_cols = _classify_csv_columns(columns, row_configs)

    if resolved_cli_files:
        cli_stems = ", ".join(p.name for p in resolved_cli_files)
        _logger.info(f"CLI config file(s) prepended to each CSV row: {cli_stems}")

    overlap_vars = set(cli_var_map.keys()) & var_cols
    overlap_arts = set(cli_art_map.keys()) & art_cols
    for name in sorted(overlap_vars):
        typer.echo(
            f"  Warning: variable '{name}' specified via --set and also in CSV; "
            f"CSV value will take precedence per row.",
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
    row_indices = set(row_filter) if row_filter else None
    naming_ctx = build_row_naming_ctx(rows)

    for idx, row in enumerate(rows, start=1):
        if row_indices is not None and idx not in row_indices:
            continue
        csv_files = _resolve_config_paths(row["sflow_config_file"])
        config_files = _merge_and_dedup(resolved_cli_files, csv_files)

        merged_vars = dict(cli_var_map)
        for col in var_cols:
            if row.get(col):
                merged_vars[col] = row[col]
        set_var = [f"{k}={v}" for k, v in merged_vars.items()] or None

        merged_arts: dict[str, str] = {}
        for col in art_cols:
            if row.get(col):
                merged_arts[col] = row[col]
        merged_arts.update(cli_art_map)
        artifacts = [f"{k}={v}" for k, v in merged_arts.items()] or None

        overrides_desc = ", ".join(
            f"{col}={row[col]}"
            for col in columns
            if col not in _RESERVED_CSV_COLUMNS and row.get(col)
        )

        row_missable = list(missable_tasks) if missable_tasks else []
        csv_missable = (row.get("missable_tasks") or "").strip()
        if csv_missable:
            row_missable.extend(csv_missable.split())
        effective_missable = row_missable or None

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
                    quiet=True,
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
            "Supports: single (--row 1), multiple (--row 1 --row 3), "
            "comma-separated (--row 1,3,5), and Python-style slices with exclusive end "
            "(--row 1:4 → rows 1,2,3; --row 1:6:2 → rows 1,3,5; --row [1:4]). "
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
    try:
        configure_logging(
            level=log_level, console=output is not None or bulk_input is not None
        )

        if row and bulk_input is None:
            typer.echo("Error: --row requires --bulk-input.", err=True)
            raise typer.Exit(code=1)

        # --- Bulk-input mode ---
        if bulk_input is not None:
            from sflow.cli.batch import parse_row_selector

            cli_files = list(src_files or []) + list(file or [])
            parsed_rows = parse_row_selector(row) if row else None
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
                row_filter=parsed_rows,
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
                    quiet=True,
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
