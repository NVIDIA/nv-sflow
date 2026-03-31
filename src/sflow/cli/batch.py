# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for generating sbatch scripts to run sflow in batch mode.
"""

import csv
import shlex
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, List, Optional

import typer

from sflow.app.sflow import SflowApp
from sflow.cli import DOCS_URL, app
from sflow.config.resolver import enrich_error_with_location
from sflow.logging import configure_logging, get_logger

_logger = get_logger(__name__)

_sflow_app = SflowApp()


def _detect_slurm_account() -> str | None:
    """Try to detect the current user's default Slurm account.

    Queries ``sacctmgr`` for the associations of the current OS user.
    Returns the first account found, or None.
    """
    import os
    import subprocess

    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user:
        try:
            user = subprocess.check_output(["whoami"], text=True).strip()
        except Exception:
            return None
    try:
        out = subprocess.check_output(
            [
                "sacctmgr",
                "show",
                "assoc",
                f"user={user}",
                "format=Account%30",
                "--noheader",
                "--parsable2",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        for line in out.strip().splitlines():
            acct = line.strip()
            if acct:
                return acct
    except Exception:
        pass
    return None


def _detect_slurm_partition() -> str | None:
    """Try to detect the default Slurm partition.

    Looks for a partition marked as default (``*``) in ``sinfo`` output.
    Falls back to the first available partition.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["sinfo", "--noheader", "--format=%P"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        first: str | None = None
        for line in out.strip().splitlines():
            name = line.strip()
            if not name:
                continue
            if first is None:
                first = name.rstrip("*")
            if name.endswith("*"):
                return name.rstrip("*")
        return first
    except Exception:
        pass
    return None


def _resolve_slurm_defaults(
    partition: str | None,
    account: str | None,
) -> tuple[str, str]:
    """Resolve partition and account, auto-detecting from Slurm when not provided.

    Emits warnings for auto-detected values. Raises ``typer.BadParameter``
    if a value cannot be determined.
    """
    if partition is None:
        partition = _detect_slurm_partition()
        if partition:
            typer.echo(
                f"  Warning: --partition not specified, auto-detected: {partition}",
                err=True,
            )
        else:
            raise typer.BadParameter(
                "Could not auto-detect a Slurm partition. Please specify --partition / -p explicitly."
            )

    if account is None:
        account = _detect_slurm_account()
        if account:
            typer.echo(
                f"  Warning: --account not specified, auto-detected: {account}",
                err=True,
            )
        else:
            raise typer.BadParameter(
                "Could not auto-detect a Slurm account. Please specify --account / -A explicitly."
            )

    return partition, account


def _resolve_sbatch_extra_args(
    extra_args: list[str],
    config_files: list[Path],
    set_var: list[str] | None,
) -> list[str]:
    """Resolve ``${{ }}`` expressions in sbatch extra args.

    Supports both ``${{ variables.SLURM_NODES }}`` (full path) and
    ``${{ SLURM_NODES }}`` (shorthand).  Builds a variable context from the
    config YAML files (defaults) with ``set_var`` overrides applied on top,
    then resolves any Jinja2 expressions found in the extra args.
    """
    if not any("${{" in arg for arg in extra_args):
        return list(extra_args)

    from sflow.config.resolver import ExpressionResolver

    var_map: dict[str, Any] = {}
    for cfg_path in config_files:
        try:
            import yaml as _yaml

            with open(cfg_path) as fh:
                data = _yaml.safe_load(fh)
            if data:
                var_map.update(_build_var_map(data))
        except Exception:
            pass

    if set_var:
        for override in set_var:
            if "=" in override:
                k, v = override.split("=", 1)
                var_map[k] = v

    ctx: dict[str, Any] = {"variables": var_map}
    ctx.update(var_map)
    resolver = ExpressionResolver()

    resolved: list[str] = []
    for arg in extra_args:
        if "${{" in arg:
            try:
                resolved.append(str(resolver.resolve(arg, ctx)))
            except Exception:
                resolved.append(arg)
        else:
            resolved.append(arg)
    return resolved


def _generate_sbatch_script(
    *,
    files: list[Path],
    set_var: list[str] | None,
    artifact: list[str] | None,
    missable_tasks: list[str] | None = None,
    log_level: str,
    workspace_dir: Path | None,
    output_dir: Path | None,
    job_name: str,
    sbatch_output: str,
    sbatch_error: str,
    partition: str,
    account: str,
    time: str | None,
    nodes: int | None,
    gpus_per_node: int | None,
    sbatch_extra_args: list[str] | None,
    sflow_venv_path: Path | None,
    sflow_version: str | None,
) -> str:
    """Generate the content of an sbatch script that wraps ``sflow run``."""
    sflow_cmd_parts = ["sflow", "run"]
    for f in files:
        sflow_cmd_parts.extend(["--file", shlex.quote(str(f))])

    if set_var:
        for var in set_var:
            sflow_cmd_parts.extend(["--set", shlex.quote(var)])

    if artifact:
        for art in artifact:
            sflow_cmd_parts.extend(["--artifact", shlex.quote(art)])

    if missable_tasks:
        for mt in missable_tasks:
            sflow_cmd_parts.extend(["--missable-tasks", shlex.quote(mt)])

    if log_level != "info":
        sflow_cmd_parts.extend(["--log-level", log_level])

    if workspace_dir:
        sflow_cmd_parts.extend(["--workspace-dir", shlex.quote(str(workspace_dir))])

    if output_dir:
        sflow_cmd_parts.extend(["--output-dir", shlex.quote(str(output_dir))])

    sbatch_directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={sbatch_output}",
        f"#SBATCH --error={sbatch_error}",
        "#SBATCH --mem=0",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={account}",
    ]

    if nodes is not None:
        sbatch_directives.append(f"#SBATCH --nodes={nodes}")

    if time:
        sbatch_directives.append(f"#SBATCH --time={time}")

    if sbatch_extra_args:
        resolved_extra_args = _resolve_sbatch_extra_args(
            sbatch_extra_args, files, set_var
        )
        for extra_arg in resolved_extra_args:
            sbatch_directives.append(f"#SBATCH {extra_arg}")

    script_lines = [
        "#!/bin/bash",
        "#",
        "# Generated by: sflow batch",
        f"# Workflow file(s): {', '.join(str(f) for f in files)}",
        "#",
        "",
        *sbatch_directives,
        "",
        "set -x",
        "",
    ]

    if sflow_venv_path:
        activate_script = sflow_venv_path / ".sflow_venv" / "bin" / "activate"
    else:
        activate_script = Path.cwd() / ".sflow_venv" / "bin" / "activate"

    activate_path_str = shlex.quote(str(activate_script))
    venv_parent = shlex.quote(str(Path(activate_script).resolve().parent.parent.parent))
    git_ref = sflow_version if sflow_version else "main"

    lock_file = shlex.quote(str(Path(activate_script).resolve().parent.parent.parent / ".sflow_venv.lock"))

    sflow_install_cmd = f"'sflow @ git+https://github.com/NVIDIA/nv-sflow.git@{git_ref}' --prerelease=allow"

    script_lines.extend(
        [
            f"SFLOW_ACTIVATE={activate_path_str}",
            f"SFLOW_LOCK={lock_file}",
            "",
            "# Use flock to prevent concurrent venv creation/install across Slurm jobs",
            f"mkdir -p {venv_parent}",
            '(flock -w 600 9 || { echo "ERROR: timed out waiting for sflow venv lock"; exit 1; }',
            "",
            'if [ -f "$SFLOW_ACTIVATE" ]; then',
            "    # Activate existing Python virtual environment for sflow",
            '    source "$SFLOW_ACTIVATE"',
        ]
    )
    if sflow_version:
        script_lines.append(
            f'    "$VIRTUAL_ENV/bin/uv" pip install {sflow_install_cmd}'
        )
    script_lines.extend(
        [
            "else",
            "    # Venv not found; create from scratch and install sflow",
            f"    cd {venv_parent}",
            "    python3 -m venv .sflow_venv",
            "    source .sflow_venv/bin/activate",
            '    "$VIRTUAL_ENV/bin/pip" install uv',
            f'    "$VIRTUAL_ENV/bin/uv" pip install {sflow_install_cmd}',
            '    "$VIRTUAL_ENV/bin/sflow" --help',
            "fi",
            "",
            ') 9>"$SFLOW_LOCK"',
            "",
            "# Activate venv outside the lock (lock is only for creation/install)",
            'source "$SFLOW_ACTIVATE"',
            "",
        ]
    )

    effective_output_dir = (
        shlex.quote(str(output_dir))
        if output_dir
        else shlex.quote(
            str(workspace_dir / "sflow_output")
            if workspace_dir
            else str(Path.cwd() / "sflow_output")
        )
    )

    script_lines.extend(
        [
            f"cd {shlex.quote(str(workspace_dir))}",
            "",
            "# Run sflow workflow",
            shlex.quote(str(activate_script.resolve().parent / "sflow"))
            + " "
            + " ".join(sflow_cmd_parts[1:]),
            "",
            "# Copy sbatch logs and sflow config(s) to workflow output directory for reference",
            f'SFLOW_WF_DIR=$(find {effective_output_dir} -maxdepth 1 -type d -name "${{SLURM_JOB_ID}}-*" 2>/dev/null | head -1)',
            'if [ -n "$SFLOW_WF_DIR" ] && [ -d "$SFLOW_WF_DIR" ]; then',
            f"    SBATCH_OUT={effective_output_dir}/${{SLURM_JOB_ID}}-sflow-submit.out",
            f"    SBATCH_ERR={effective_output_dir}/${{SLURM_JOB_ID}}-sflow-submit.err",
            '    cp "$SBATCH_OUT" "$SFLOW_WF_DIR/" 2>/dev/null || true',
            '    cp "$SBATCH_ERR" "$SFLOW_WF_DIR/" 2>/dev/null || true',
            *[
                f'    cp {shlex.quote(str(f))} "$SFLOW_WF_DIR/" 2>/dev/null || true'
                for f in files
            ],
            "fi",
            "",
        ]
    )

    return "\n".join(script_lines)


_RESERVED_CSV_COLUMNS = frozenset({"sflow_config_file", "job_name", "missable_tasks"})
_NODE_COLUMN_NAMES = frozenset({"SLURM_NODES", "NUM_SLURM_NODES", "NUM_NODES"})


def parse_row_selector(values: list[str]) -> list[int]:
    """Parse ``--row`` values into a flat sorted list of 1-based row indices.

    Supported formats (all 1-based; slice end is **exclusive** like Python):

    * Single int:        ``--row 1``
    * Comma-separated:   ``--row 1,3,5``   or   ``--row [1,3,5]``
    * Slice:             ``--row 1:4``      →  rows 1, 2, 3
    * Slice with step:   ``--row 1:6:2``    →  rows 1, 3, 5
    * Brackets optional: ``--row [1:4]``    same as ``--row 1:4``

    Multiple ``--row`` flags are combined:  ``--row 1:3 --row 7``  →  [1, 2, 7]
    """
    indices: set[int] = set()
    for raw in values:
        token = raw.strip().strip("[]")
        if not token:
            continue
        if "," in token:
            for part in token.split(","):
                part = part.strip()
                if part:
                    indices.update(_parse_single_or_slice(part))
        else:
            indices.update(_parse_single_or_slice(token))
    return sorted(indices)


def _parse_single_or_slice(token: str) -> list[int]:
    """Parse a single int or a start:stop[:step] slice into 1-based indices."""
    if ":" in token:
        parts = token.split(":")
        if len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            step = 1
        elif len(parts) == 3:
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise typer.BadParameter(
                f"Invalid slice: '{token}' (expected start:stop or start:stop:step)"
            )
        if step == 0:
            raise typer.BadParameter("Slice step cannot be zero")
        return list(range(start, stop, step))
    return [int(token)]


_MAX_NAME_LEN = 30


def _sanitize_name(name: str, max_len: int = _MAX_NAME_LEN) -> str:
    """Sanitize a name for use as a filename / Slurm job name.

    Replaces non-alphanumeric characters (except ``_`` and ``-``) with ``_``,
    collapses consecutive ``_``, strips leading/trailing ``_``, and truncates.
    """
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned[:max_len].rstrip("_") if cleaned else "row"


def _dedup_words(name: str) -> str:
    """Remove duplicate words from an underscore-separated name, preserving order.

    ``trtllm_prefill_trtllm_decode`` → ``trtllm_prefill_decode``
    """
    seen: set[str] = set()
    out: list[str] = []
    for word in name.split("_"):
        if word and word not in seen:
            seen.add(word)
            out.append(word)
    return "_".join(out)


def _normalize_col_value(value: str) -> str | None:
    """Normalize a CSV column value for inclusion in a derived name.

    Returns ``None`` to skip the value entirely, or a shortened string:

    * Path URIs (``fs:///…``, ``s3://…``) → ``None``
    * Absolute paths (``/…``) → ``None``
    * Container images (``registry/image:tag``, e.g. ``nvcr.io/…:0.8.0``) → ``None``
    * Everything else → returned as-is
    """
    if "://" in value or value.startswith("/") or "/" in value:
        return None
    return value


def _path_to_stem(raw: str) -> str:
    """Convert a config file path to a descriptive stem for naming.

    Joins directory components and the file stem with ``_`` so that relative
    paths like ``trtllm/prefill.yaml`` become ``trtllm_prefill`` instead of
    just ``prefill``.  Absolute paths and bare filenames fall back to the
    plain stem.
    """
    p = Path(raw.strip())
    if p.is_absolute():
        return p.stem
    parts = list(p.parent.parts) + [p.stem]
    parts = [part for part in parts if part not in (".", "..")]
    return "_".join(parts) if parts else p.stem


class _RowNamingCtx:
    """Precomputed context for deriving row names from a CSV.

    Built once from all rows, then passed to each ``_derive_row_name`` call
    so that common-stem detection and differing-column detection are O(1) per row
    instead of O(R) per row (eliminating the O(R²) total cost).
    """

    __slots__ = ("common_stems", "differing_cols", "cli_nodes", "fallback_base")

    def __init__(
        self,
        all_rows: list[dict[str, str]],
        fallback_base: str = "sflow",
        cli_nodes: int | None = None,
    ) -> None:
        self.fallback_base = fallback_base
        self.cli_nodes = cli_nodes

        all_stem_sets = [
            {_path_to_stem(p) for p in r["sflow_config_file"].split()} for r in all_rows
        ]
        self.common_stems: set[str] = (
            set.intersection(*all_stem_sets) if all_stem_sets else set()
        )

        skip_cols = _RESERVED_CSV_COLUMNS | _NODE_COLUMN_NAMES
        self.differing_cols: list[str] = []
        if all_rows:
            candidate_cols = [c for c in all_rows[0] if c not in skip_cols]
            for col in candidate_cols:
                all_vals = {(r.get(col) or "").strip() for r in all_rows}
                if len(all_vals) > 1:
                    self.differing_cols.append(col)


def build_row_naming_ctx(
    all_rows: list[dict[str, str]],
    fallback_base: str = "sflow",
    cli_nodes: int | None = None,
) -> _RowNamingCtx:
    """Build the shared naming context once before iterating rows."""
    return _RowNamingCtx(all_rows, fallback_base=fallback_base, cli_nodes=cli_nodes)


def _resolve_node_count(
    row: dict[str, str],
    cli_nodes: int | None,
) -> str | None:
    """Return the node count for a row as ``<N>n``, or None if unknown."""
    if cli_nodes is not None:
        return f"{cli_nodes}n"
    for col in _NODE_COLUMN_NAMES:
        val = (row.get(col) or "").strip()
        if val:
            try:
                return f"{int(val)}n"
            except ValueError:
                pass
    return None


def _derive_row_name(
    row: dict[str, str],
    idx: int,
    ctx: _RowNamingCtx,
) -> str:
    """Derive a descriptive name for a bulk-input CSV row.

    Uses a precomputed ``_RowNamingCtx`` so each call is O(F + C) instead of
    O(R * (F + C)), keeping total batch naming O(R * (F + C)) instead of O(R²).

    Priority:
    1. Explicit ``job_name`` column value (if present and non-empty).
    2. Auto-derived from multiple sources:
       a. Unique config file stems (stems not shared by every row).
          Handles relative paths (``../../dir/file.yaml`` → stem ``file``).
       b. Node count (always included as ``<N>n``, from CLI --nodes or CSV column).
       c. Unique column values (short, non-path values that differ across rows;
          node columns are excluded since they are already handled above).
    3. Falls back to *fallback_base* when nothing distinguishes the row.

    A post-processing step deduplicates repeated words
    (e.g. ``trtllm_prefill_trtllm_decode`` → ``trtllm_prefill_decode``).
    The row index ``_{idx:03d}`` is always appended for guaranteed uniqueness.
    The base name is capped at 30 characters before the suffix.
    """
    explicit = (row.get("job_name") or "").strip()
    if explicit:
        return f"{_sanitize_name(explicit)}_{idx:03d}"

    parts: list[str] = []

    row_stems = [_path_to_stem(p) for p in row["sflow_config_file"].split()]
    parts.extend(s for s in row_stems if s not in ctx.common_stems)

    node_label = _resolve_node_count(row, ctx.cli_nodes)
    if node_label:
        parts.append(node_label)

    for col in ctx.differing_cols:
        val = (row.get(col) or "").strip()
        if not val:
            continue
        normed = _normalize_col_value(val)
        if normed is None:
            continue
        parts.append(normed)

    base = "_".join(parts) if parts else ctx.fallback_base
    base = _dedup_words(base)
    return f"{_sanitize_name(base)}_{idx:03d}"


def _submit_sbatch(script_path: Path) -> str:
    """Submit an sbatch script and return the stdout message (e.g. job id)."""
    import subprocess

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _build_var_map(
    data: dict, cli_overrides: list[str] | None = None
) -> dict[str, Any]:
    """Build a variable name->value map from raw sflow YAML data.

    Handles both variable formats:
    - dict-of-dict: ``{KEY: {description: …, value: …}}``
    - list-of-dict: ``[{name: KEY, value: …}]``
    - simple dict:  ``{KEY: scalar_value}``

    Also includes workflow-level variables.  CLI ``--set`` overrides
    (``KEY=VALUE`` strings) are applied last with highest priority.
    """
    var_map: dict[str, Any] = {}
    raw_vars = data.get("variables") or []
    if isinstance(raw_vars, dict):
        for k, v in raw_vars.items():
            if isinstance(v, dict):
                var_map[k] = v.get("value")
            else:
                var_map[k] = v
    elif isinstance(raw_vars, list):
        for v in raw_vars:
            if isinstance(v, dict) and "name" in v:
                var_map[v["name"]] = v.get("value")

    wf = data.get("workflow")
    if isinstance(wf, dict):
        wf_vars = wf.get("variables") or []
        if isinstance(wf_vars, dict):
            for k, v in wf_vars.items():
                if isinstance(v, dict):
                    var_map[k] = v.get("value")
                else:
                    var_map[k] = v
        elif isinstance(wf_vars, list):
            for v in wf_vars:
                if isinstance(v, dict) and "name" in v:
                    var_map[v["name"]] = v.get("value")

    for entry in cli_overrides or []:
        if "=" in entry:
            k, v_str = entry.split("=", 1)
            var_map[k] = v_str

    return var_map


def _resolve_backend_int_field(
    data: dict, field: str, var_map: dict[str, Any]
) -> int | None:
    """Resolve an integer field from the first slurm backend definition.

    If the value is a ``${{ variables.X }}`` expression, look it up in *var_map*.
    Returns the resolved integer, or ``None`` if the field is absent or unresolvable.
    """
    import re as _re

    for b in data.get("backends") or []:
        if not isinstance(b, dict) or b.get("type") != "slurm":
            continue
        if field not in b:
            continue
        val = b[field]
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            m = _re.search(r"\$\{\{\s*variables\.(\w+)\s*\}\}", val)
            if m:
                ref_val = var_map.get(m.group(1))
                if ref_val is not None:
                    try:
                        return int(ref_val)
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
    return None


def _derive_gpus_per_node(
    config_files: list[Path],
    cli_overrides: list[str] | None = None,
) -> int | None:
    """Derive gpus_per_node from sflow config files' backend definitions.

    Merges variables from all files, then checks each file's backends.
    CLI ``--set`` overrides are applied to the variable map.
    """
    import yaml as _yaml

    merged_var_map: dict[str, Any] = {}
    all_data: list[dict] = []

    for f in config_files:
        try:
            with open(f) as fh:
                raw = _yaml.safe_load(fh)
            if isinstance(raw, dict):
                all_data.append(raw)
                merged_var_map.update(_build_var_map(raw))
        except Exception:
            continue

    for entry in cli_overrides or []:
        if "=" in entry:
            k, v = entry.split("=", 1)
            merged_var_map[k] = v

    for d in all_data:
        result = _resolve_backend_int_field(d, "gpus_per_node", merged_var_map)
        if result is not None:
            return result
    return None


def _derive_nodes(
    config_files: list[Path],
    cli_overrides: list[str] | None = None,
) -> int | None:
    """Derive nodes from sflow config files' backend definitions.

    Merges variables from all files, then checks each file's backends.
    CLI ``--set`` overrides are applied to the variable map.
    """
    import yaml as _yaml

    merged_var_map: dict[str, Any] = {}
    all_data: list[dict] = []

    for f in config_files:
        try:
            with open(f) as fh:
                raw = _yaml.safe_load(fh)
            if isinstance(raw, dict):
                all_data.append(raw)
                merged_var_map.update(_build_var_map(raw))
        except Exception:
            continue

    for entry in cli_overrides or []:
        if "=" in entry:
            k, v = entry.split("=", 1)
            merged_var_map[k] = v

    for d in all_data:
        result = _resolve_backend_int_field(d, "nodes", merged_var_map)
        if result is not None:
            return result
    return None


def _classify_csv_columns(
    columns: list[str],
    row_configs: list[tuple[list[Path], list[str] | None]],
) -> tuple[set[str], set[str]]:
    """Classify CSV column names as variable overrides or artifact overrides.

    Checks columns against ALL config file sets (one per CSV row).
    Each entry is ``(config_files, per_row_missable_tasks)``.
    A column is valid if it matches a variable or artifact in ANY row's config.

    Returns (var_columns, art_columns).
    Raises ValueError if a column matches neither in any config set.
    """
    from sflow.config.loader import ConfigLoader

    var_names: set[str] = set()
    art_names: set[str] = set()
    seen: set[tuple[str, ...]] = set()
    load_errors: list[tuple[tuple[str, ...], Exception]] = []
    loaded_count = 0

    for config_files, row_missable in row_configs:
        key = tuple(str(f) for f in config_files)
        if key in seen:
            continue
        seen.add(key)
        try:
            config = ConfigLoader().load_configs(
                config_files, missable_tasks=row_missable
            )
        except Exception as exc:
            load_errors.append((key, exc))
            continue
        loaded_count += 1
        for v in config.variables or []:
            var_names.add(v.name)
        wf = config.workflow
        if wf and wf.variables:
            for v in wf.variables:
                var_names.add(v.name)
        for a in config.artifacts or []:
            art_names.add(a.name)

    if load_errors:
        _logger.warning(
            f"{len(load_errors)} config file set(s) failed to load "
            f"({loaded_count} succeeded):"
        )
        for files, exc in load_errors:
            file_list = " + ".join(files)
            _logger.warning(f"  ⚠ [{file_list}]: {exc}")
        if loaded_count == 0:
            _logger.warning(
                "  No config sets loaded successfully. If tasks from one file "
                "reference tasks in another, consider adding --missable-tasks / -M "
                "or a 'missable_tasks' CSV column."
            )

    var_cols: set[str] = set()
    art_cols: set[str] = set()
    for col in columns:
        if col in _RESERVED_CSV_COLUMNS:
            continue
        if col in var_names:
            var_cols.add(col)
        elif col in art_names:
            art_cols.add(col)
        else:
            msg = (
                f"CSV column '{col}' is not a variable or artifact "
                f"defined in any of the config file sets"
            )
            if load_errors and loaded_count == 0:
                msg += (
                    f". Note: all {len(load_errors)} config set(s) failed to load"
                    f" — the root cause is likely a config loading error above, "
                    f"not a missing variable. Common fix: add --missable-tasks / -M "
                    f"for tasks referenced in depends_on that don't exist in "
                    f"all files, or add a 'missable_tasks' column to the CSV."
                )
            raise ValueError(msg)
    return var_cols, art_cols


def read_bulk_csv(csv_path: Path) -> tuple[list[str], list[dict]]:
    """Read and validate a bulk-input CSV file.

    Returns (columns, rows).
    Raises ValueError if the file is empty or lacks the ``sflow_config_file`` column.
    """
    import csv

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {csv_path}")
        columns = list(reader.fieldnames)
        if "sflow_config_file" not in columns:
            raise ValueError(
                f"CSV must contain a 'sflow_config_file' column. Found: {columns}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file has no data rows: {csv_path}")
    return columns, rows


def resolve_row_files(
    row: dict, csv_dir: Path, resolved_cli_files: list[Path],
) -> list[Path]:
    """Resolve and dedup config file paths for a single CSV row.

    CLI files are prepended; CSV paths are resolved relative to *csv_dir*.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    for p in resolved_cli_files + [(csv_dir / x).resolve() for x in row["sflow_config_file"].split()]:
        if p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


def row_missable(row: dict, cli_missable: list[str] | None) -> list[str] | None:
    """Merge CLI and CSV ``missable_tasks`` for a single row."""
    m = list(cli_missable) if cli_missable else []
    csv_m = (row.get("missable_tasks") or "").strip()
    if csv_m:
        m.extend(csv_m.split())
    return m or None


def build_all_row_configs(
    rows: list[dict],
    csv_dir: Path,
    resolved_cli_files: list[Path],
    cli_missable: list[str] | None,
) -> list[tuple[list[Path], list[str] | None]]:
    """Build (config_files, missable) tuples for all rows, for column classification."""
    return [
        (resolve_row_files(r, csv_dir, resolved_cli_files), row_missable(r, cli_missable))
        for r in rows
    ]


def _parse_kv_list(entries: list[str] | None) -> dict[str, str]:
    """Parse a list of 'KEY=VALUE' strings into a dict."""
    result: dict[str, str] = {}
    for entry in entries or []:
        if "=" in entry:
            k, v = entry.split("=", 1)
            result[k] = v
    return result


def merge_row_overrides(
    row: dict,
    var_cols: set[str],
    art_cols: set[str],
    cli_var_map: dict[str, str],
    cli_art_map: dict[str, str],
) -> tuple[list[str] | None, list[str] | None]:
    """Merge CLI and CSV overrides for a single row.

    For variables, CSV values take precedence over CLI ``--set``.
    For artifacts, CLI ``--artifact`` takes precedence over CSV values.

    Returns (set_var_list, artifact_list).
    """
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

    return set_var, artifacts


def resolve_csv_row(
    csv_path: Path,
    row_idx: int,
    cli_files: list[Path] | None = None,
    cli_set_var: list[str] | None = None,
    cli_artifact: list[str] | None = None,
    cli_missable: list[str] | None = None,
) -> tuple[list[Path], list[str] | None, list[str] | None, list[str] | None]:
    """Resolve a single CSV row into (config_files, set_var, artifact, missable_tasks).

    High-level convenience that reads the CSV, classifies columns, and merges
    overrides for the selected row (1-based index).
    Used by ``sflow run --bulk-input``.
    """
    columns, rows = read_bulk_csv(csv_path)
    if row_idx < 1 or row_idx > len(rows):
        raise IndexError(f"Row {row_idx} out of range (CSV has {len(rows)} rows)")

    csv_dir = csv_path.parent
    resolved_cli = [fp.resolve() for fp in (cli_files or [])]

    all_row_configs = build_all_row_configs(rows, csv_dir, resolved_cli, cli_missable)
    var_cols, art_cols = _classify_csv_columns(columns, all_row_configs)

    row = rows[row_idx - 1]
    config_files = resolve_row_files(row, csv_dir, resolved_cli)
    missable = row_missable(row, cli_missable)

    cli_var_map = _parse_kv_list(cli_set_var)
    cli_art_map = _parse_kv_list(cli_artifact)
    set_var, artifacts = merge_row_overrides(row, var_cols, art_cols, cli_var_map, cli_art_map)

    return config_files, set_var, artifacts, missable


def _scan_sflow_yamls(paths: list[Path]) -> list[Path]:
    """Scan file paths, directories, and glob patterns for valid sflow YAML configs.

    A valid sflow YAML is a ``*.yaml`` or ``*.yml`` file whose top-level
    mapping contains a ``version`` key.

    Supports:
    - Explicit file paths (``workflow.yaml``)
    - Directories (scanned for ``*.yaml`` / ``*.yml``)
    - Glob patterns (``examples/slurm_*``, ``configs/**/*.yaml``)
    """
    import glob as _glob

    import yaml as _yaml

    candidates: list[Path] = []
    for p in paths:
        if p.is_dir():
            candidates.extend(sorted(p.glob("*.yaml")))
            candidates.extend(sorted(p.glob("*.yml")))
        elif p.is_file():
            if p.suffix in (".yaml", ".yml"):
                candidates.append(p)
        else:
            expanded = sorted(Path(m) for m in _glob.glob(str(p)))
            if expanded:
                for ep in expanded:
                    if ep.is_file() and ep.suffix in (".yaml", ".yml"):
                        candidates.append(ep)
                    elif ep.is_dir():
                        candidates.extend(sorted(ep.glob("*.yaml")))
                        candidates.extend(sorted(ep.glob("*.yml")))

    valid: list[Path] = []
    for f in candidates:
        try:
            with open(f) as fh:
                data = _yaml.safe_load(fh)
            if isinstance(data, dict) and "version" in data:
                valid.append(f.resolve())
        except Exception:
            continue
    return sorted(set(valid))


def _run_bulk_submit(
    *,
    yaml_files: list[Path],
    cli_set_var: list[str] | None,
    cli_artifact: list[str] | None,
    log_level: str,
    workspace_dir: Path | None,
    output_dir: Path | None,
    sbatch_output: str,
    sbatch_error: str,
    partition: str,
    account: str,
    time: str | None,
    nodes: int | None,
    gpus_per_node: int | None,
    sbatch_extra_args: list[str] | None,
    sflow_venv_path: Path | None,
    sflow_version: str | None,
    submit: bool,
    missable_tasks: list[str] | None = None,
    resolve: bool = False,
) -> None:
    """Process multiple self-contained sflow YAML configs as individual batch jobs."""
    import re as _re
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    effective_output = output_dir or Path.cwd() / "sflow_output"
    bulk_dir = effective_output / f"bulk_submit_{stamp}"
    bulk_dir.mkdir(parents=True, exist_ok=True)

    _SBATCH_JOB_ID_RE = _re.compile(r"Submitted batch job (\d+)")

    cli_var_keys: set[str] = set()
    for entry in cli_set_var or []:
        if "=" in entry:
            cli_var_keys.add(entry.split("=", 1)[0])

    summary: list[str] = []
    failures: list[str] = []
    failed_count = 0
    result_rows: list[dict[str, str]] = []

    for idx, yaml_file in enumerate(yaml_files, start=1):
        job_name = _sanitize_name(yaml_file.stem)
        typer.echo(f"\n[{idx}/{len(yaml_files)}] Processing {yaml_file.name} ...")

        # Warn about CLI variable overrides
        if cli_var_keys:
            try:
                import yaml as _yaml

                with open(yaml_file) as fh:
                    data = _yaml.safe_load(fh)
                config_var_names: set[str] = set()
                raw_vars = data.get("variables") or []
                if isinstance(raw_vars, dict):
                    config_var_names.update(raw_vars.keys())
                elif isinstance(raw_vars, list):
                    for v in raw_vars:
                        if isinstance(v, dict) and "name" in v:
                            config_var_names.add(v["name"])
                        elif isinstance(v, str):
                            config_var_names.add(v)
                wf = data.get("workflow")
                if isinstance(wf, dict):
                    wf_vars = wf.get("variables") or []
                    if isinstance(wf_vars, dict):
                        config_var_names.update(wf_vars.keys())
                    elif isinstance(wf_vars, list):
                        for v in wf_vars:
                            if isinstance(v, dict) and "name" in v:
                                config_var_names.add(v["name"])
                overlap = cli_var_keys & config_var_names
                for name in sorted(overlap):
                    typer.echo(
                        f"  Warning: variable '{name}' in {yaml_file.name} overridden by --set",
                        err=True,
                    )
            except Exception:
                pass

        # Derive gpus_per_node: config value wins over CLI
        config_gpus = _derive_gpus_per_node([yaml_file], cli_overrides=cli_set_var)
        row_gpus = config_gpus if config_gpus is not None else gpus_per_node
        if (
            gpus_per_node is not None
            and config_gpus is not None
            and gpus_per_node != config_gpus
        ):
            typer.echo(
                f"  Warning: --gpus-per-node={gpus_per_node} overridden by "
                f"{yaml_file.name} config value ({config_gpus})",
                err=True,
            )

        # Dry-run validation
        try:
            _sflow_app.run(
                file=[yaml_file],
                dry_run=True,
                quiet=True,
                variable_overrides=list(cli_set_var) if cli_set_var else None,
                artifact_overrides=list(cli_artifact) if cli_artifact else None,
                missable_tasks=missable_tasks,
                slurm_nodes=nodes,
                slurm_gpus_per_node=row_gpus,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
            )
        except Exception as e:
            failed_count += 1
            err_short = str(e).split("\n")[0]
            summary.append(f"  [{idx}] {yaml_file.name}: SKIPPED (dry-run failed)")
            failures.append(f"  [{idx}] {yaml_file.name}: {err_short}")
            fail_row: dict[str, str] = {
                "sflow_config_file": str(yaml_file),
                "job_name": job_name,
                "slurm_job_id": "FAILED",
                "sflow_output_dir": "",
                "status": "dry-run failed",
            }
            if resolve:
                fail_row["composed_sflow_config"] = ""
            result_rows.append(fail_row)
            continue

        # Determine node count from config if not given via CLI
        row_nodes = nodes
        if row_nodes is None:
            try:
                import yaml as _yaml

                with open(yaml_file) as fh:
                    data = _yaml.safe_load(fh)

                var_map = _build_var_map(data, cli_overrides=cli_set_var)

                row_nodes = _resolve_backend_int_field(data, "nodes", var_map)

                if row_nodes is None:
                    for name in _NODE_COLUMN_NAMES:
                        if name in var_map:
                            try:
                                row_nodes = int(var_map[name])
                            except (ValueError, TypeError):
                                pass
                            if row_nodes is not None:
                                break
            except Exception:
                pass

        script = _generate_sbatch_script(
            files=[yaml_file],
            set_var=cli_set_var,
            artifact=cli_artifact,
            missable_tasks=missable_tasks,
            log_level=log_level,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            job_name=job_name,
            sbatch_output=sbatch_output,
            sbatch_error=sbatch_error,
            partition=partition,
            account=account,
            time=time,
            nodes=row_nodes,
            gpus_per_node=row_gpus,
            sbatch_extra_args=sbatch_extra_args,
            sflow_venv_path=sflow_venv_path,
            sflow_version=sflow_version,
        )
        script_path = bulk_dir / f"{job_name}.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)

        # Generate composed/resolved YAML alongside the sbatch script
        composed_yaml_path: str = ""
        try:
            from sflow.cli.compose import _compose_files

            yaml_output = _compose_files(
                [yaml_file],
                cli_set_var or None,
                cli_artifact or None,
                log_level,
                resolve=resolve,
                missable_tasks=missable_tasks,
                quiet_missable=True,
            )
            yaml_path = bulk_dir / f"{job_name}.yaml"
            yaml_path.write_text(yaml_output)
            composed_yaml_path = str(yaml_path)
        except Exception as e:
            typer.echo(
                f"  Warning: could not generate composed config for {yaml_file.name}: {e}",
                err=True,
            )

        status = "saved"
        job_id = ""
        if submit:
            try:
                msg = _submit_sbatch(script_path)
                status = msg
                m = _SBATCH_JOB_ID_RE.search(msg)
                if m:
                    job_id = m.group(1)
            except RuntimeError as e:
                status = f"FAILED ({e})"

        sflow_output_dir = f"{effective_output}/{job_id}-*" if job_id else ""
        summary.append(f"  [{idx}] {script_path.name}: {yaml_file.name} -> {status}")
        success_row: dict[str, str] = {
            "sflow_config_file": str(yaml_file),
            "job_name": job_name,
            "slurm_job_id": job_id
            if job_id
            else ("not submitted" if not submit else "FAILED"),
            "sflow_output_dir": sflow_output_dir
            if sflow_output_dir
            else ("not submitted" if not submit else ""),
            "status": status,
        }
        if resolve:
            success_row["composed_sflow_config"] = composed_yaml_path
        result_rows.append(success_row)

    generated = len(yaml_files) - failed_count
    typer.echo(
        f"\nBulk submit: {generated}/{len(yaml_files)} configs processed"
        + (f" ({failed_count} failed validation)" if failed_count else "")
    )
    for line in summary:
        typer.echo(line)
    if failures:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"ERRORS: {len(failures)} config(s) failed dry-run validation:")
        typer.echo(f"{'=' * 60}")
        for f in failures:
            typer.echo(f)
        typer.echo(f"{'=' * 60}")

    if result_rows:
        import csv

        results_csv = bulk_dir / "results.csv"
        fieldnames = [
            "sflow_config_file",
            "job_name",
            "slurm_job_id",
            "sflow_output_dir",
            "status",
        ]
        if resolve:
            fieldnames.append("composed_sflow_config")
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result_rows)
        typer.echo(f"\nResults CSV: {results_csv}")

    typer.echo(f"Scripts directory: {bulk_dir}")

    if not submit and generated > 0:
        typer.echo("\n(Scripts generated but not submitted. To submit, add: --submit)")


def _run_bulk_edit(
    *,
    csv_path: Path,
    cli_set_var: list[str] | None,
    cli_artifact: list[str] | None,
    log_level: str,
    workspace_dir: Path | None,
    output_dir: Path | None,
    job_name: str,
    sbatch_output: str,
    sbatch_error: str,
    partition: str,
    account: str,
    time: str | None,
    nodes: int | None,
    gpus_per_node: int | None,
    sbatch_extra_args: list[str] | None,
    sflow_venv_path: Path | None,
    sflow_version: str | None,
    submit: bool,
    row_filter: list[int] | None = None,
    resolve: bool = False,
    missable_tasks: list[str] | None = None,
) -> None:
    """Generate (and optionally submit) one sbatch job per CSV row.

    CLI ``--set`` and ``--artifact`` flags provide baseline overrides.
    CSV columns override those baselines per row (with a warning).
    """
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
                f"CSV file must contain a 'sflow_config_file' column "
                f"(use spaces to list multiple YAML files per row, e.g. 'backend.yaml workflow.yaml'). "
                f"Found columns: {columns}"
            )
        rows: list[dict[str, Any]] = list(reader)

    if not rows:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    if nodes is None and not (_NODE_COLUMN_NAMES & set(columns)):
        raise ValueError(
            f"--nodes was not provided and the CSV does not contain a node-count column. "
            f"Either pass --nodes or add one of these columns to the CSV: "
            f"{', '.join(sorted(_NODE_COLUMN_NAMES))}"
        )

    csv_dir = csv_path.parent

    def _resolve_config_paths(raw: str) -> list[Path]:
        paths = []
        for p in raw.split():
            fp = Path(p)
            if not fp.is_absolute():
                fp = csv_dir / fp
            paths.append(fp.resolve())
        return paths

    row_configs: list[tuple[list[Path], list[str] | None]] = []
    for r in rows:
        cfg_files = _resolve_config_paths(r["sflow_config_file"])
        row_m = list(missable_tasks) if missable_tasks else []
        csv_m = (r.get("missable_tasks") or "").strip()
        if csv_m:
            row_m.extend(csv_m.split())
        row_configs.append((cfg_files, row_m or None))
    var_cols, art_cols = _classify_csv_columns(columns, row_configs)

    csv_var_names = var_cols
    csv_art_names = art_cols
    overlap_vars = set(cli_var_map.keys()) & csv_var_names
    overlap_arts = set(cli_art_map.keys()) & csv_art_names
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
    bulk_dir = Path(output_dir or Path.cwd() / "sflow_output") / f"bulk_input_{stamp}"
    bulk_dir.mkdir(parents=True, exist_ok=True)

    import re as _re

    _SBATCH_JOB_ID_RE = _re.compile(r"Submitted batch job (\d+)")

    summary: list[str] = []
    dry_run_failures: list[str] = []
    failed_count = 0
    result_rows: list[dict[str, str]] = []
    effective_output_dir = output_dir or Path.cwd() / "sflow_output"

    row_indices = set(row_filter) if row_filter else None
    naming_ctx = build_row_naming_ctx(rows, fallback_base=job_name, cli_nodes=nodes)

    for idx, row in enumerate(rows, start=1):
        if row_indices is not None and idx not in row_indices:
            continue
        config_files = _resolve_config_paths(row["sflow_config_file"])

        merged_vars = dict(cli_var_map)
        for col in csv_var_names:
            if row.get(col):
                merged_vars[col] = row[col]
        set_var = [f"{k}={v}" for k, v in merged_vars.items()]

        merged_arts: dict[str, str] = {}
        for col in csv_art_names:
            if row.get(col):
                merged_arts[col] = row[col]
        merged_arts.update(cli_art_map)
        artifacts = [f"{k}={v}" for k, v in merged_arts.items()]

        all_overrides: dict[str, str] = {}
        for col in columns:
            if col not in _RESERVED_CSV_COLUMNS and row.get(col):
                all_overrides[col] = row[col]
        all_overrides.update(cli_art_map)
        overrides_desc = ", ".join(f"{k}={v}" for k, v in all_overrides.items())

        result_row = dict(row)

        row_missable = list(missable_tasks) if missable_tasks else []
        csv_missable = (row.get("missable_tasks") or "").strip()
        if csv_missable:
            row_missable.extend(csv_missable.split())
        effective_missable = row_missable or None

        # Derive gpus_per_node: config/CSV value wins over CLI
        config_gpus = _derive_gpus_per_node(config_files, cli_overrides=set_var)
        row_gpus = config_gpus if config_gpus is not None else gpus_per_node
        if (
            gpus_per_node is not None
            and config_gpus is not None
            and gpus_per_node != config_gpus
        ):
            typer.echo(
                f"  Warning: --gpus-per-node={gpus_per_node} overridden by "
                f"config value ({config_gpus}) for row {idx}",
                err=True,
            )

        try:
            _sflow_app.run(
                file=config_files,
                dry_run=True,
                quiet=True,
                variable_overrides=set_var or None,
                artifact_overrides=artifacts or None,
                missable_tasks=effective_missable,
                slurm_nodes=nodes,
                slurm_gpus_per_node=row_gpus,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
            )
        except Exception as e:
            failed_count += 1
            err_short = str(e).split("\n")[0]
            summary.append(f"  [{idx}] SKIPPED: ({overrides_desc})")
            dry_run_failures.append(f"  [{idx}] {err_short}")
            result_row["slurm_job_id"] = "FAILED"
            result_row["sflow_output_dir"] = ""
            if resolve:
                result_row["composed_sflow_config"] = ""
            result_rows.append(result_row)
            continue

        # Generate merged sflow config YAML alongside the sbatch script
        try:
            from sflow.cli.compose import _compose_files

            yaml_output = _compose_files(
                config_files,
                set_var or None,
                artifacts or None,
                log_level,
                resolve=resolve,
                missable_tasks=effective_missable,
                quiet_missable=True,
            )
        except Exception as e:
            typer.echo(
                f"  Warning: could not generate merged config for row {idx}: {e}",
                err=True,
            )
            yaml_output = None

        row_nodes = nodes
        if row_nodes is None:
            for col_name in _NODE_COLUMN_NAMES:
                val = row.get(col_name)
                if val:
                    try:
                        row_nodes = int(val)
                    except ValueError:
                        pass
                    break

        row_name = _derive_row_name(row, idx, naming_ctx)

        composed_config_path = ""
        if yaml_output:
            merged_yaml_path = bulk_dir / f"{row_name}.yaml"
            merged_yaml_path.write_text(yaml_output)
            composed_config_path = str(merged_yaml_path)

        script_path = bulk_dir / f"{row_name}.sh"
        script = _generate_sbatch_script(
            files=config_files,
            set_var=set_var or None,
            artifact=artifacts or None,
            missable_tasks=effective_missable,
            log_level=log_level,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            job_name=row_name,
            sbatch_output=sbatch_output,
            sbatch_error=sbatch_error,
            partition=partition,
            account=account,
            time=time,
            nodes=row_nodes,
            gpus_per_node=row_gpus,
            sbatch_extra_args=sbatch_extra_args,
            sflow_venv_path=sflow_venv_path,
            sflow_version=sflow_version,
        )
        script_path.write_text(script)
        script_path.chmod(0o755)

        status = "saved"
        job_id = ""
        if submit:
            try:
                msg = _submit_sbatch(script_path)
                status = msg
                m = _SBATCH_JOB_ID_RE.search(msg)
                if m:
                    job_id = m.group(1)
            except RuntimeError as e:
                status = f"FAILED ({e})"

        result_row["slurm_job_id"] = job_id
        result_row["sflow_output_dir"] = (
            f"{effective_output_dir}/{job_id}-*" if job_id else ""
        )
        if resolve:
            result_row["composed_sflow_config"] = composed_config_path
        result_rows.append(result_row)
        summary.append(f"  [{idx}] {script_path.name}: ({overrides_desc}) -> {status}")

    processed = len(summary)
    generated = processed - failed_count
    row_info = (
        f" (rows: {','.join(str(r) for r in sorted(row_indices))})"
        if row_indices
        else ""
    )
    typer.echo(
        f"\nBulk input: {generated}/{processed} jobs generated from {csv_path.name}{row_info}"
        + (f" ({failed_count} failed dry-run)" if failed_count else "")
    )
    typer.echo(f"Scripts directory: {bulk_dir}")
    for line in summary:
        typer.echo(line)
    if dry_run_failures:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"ERRORS: {len(dry_run_failures)} row(s) failed dry-run validation:")
        typer.echo(f"{'=' * 60}")
        for f in dry_run_failures:
            typer.echo(f)
        typer.echo(f"{'=' * 60}")

    if result_rows:
        results_csv = bulk_dir / "results.csv"
        result_columns = columns + ["slurm_job_id", "sflow_output_dir"]
        if resolve:
            result_columns.append("composed_sflow_config")
        for rr in result_rows:
            if not rr.get("slurm_job_id"):
                rr["slurm_job_id"] = "not submitted" if not submit else ""
            if not rr.get("sflow_output_dir"):
                rr["sflow_output_dir"] = "not submitted" if not submit else ""
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_columns)
            writer.writeheader()
            writer.writerows(result_rows)
        typer.echo(f"\nResults CSV: {results_csv}")
    typer.echo(f"Scripts directory: {bulk_dir}")

    if not submit and generated > 0:
        typer.echo("\n(Scripts generated but not submitted. To submit, add: --submit)")


@app.command(epilog=f"Documentation: {DOCS_URL}")
def batch(
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
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level for sflow run (debug, info, warning, error, critical). Default: info.",
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
    ] = Path.cwd(),
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            help="Global output root directory for sflow. Default: <workspace-dir>/sflow_output",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path.cwd() / "sflow_output",
    # sbatch options
    job_name: Annotated[
        str,
        typer.Option(
            "--job-name",
            "-J",
            help="Slurm job name",
        ),
    ] = "sflow",
    sbatch_output: Annotated[
        str,
        typer.Option(
            "--sbatch-output",
            "-O",
            help="Slurm output file pattern. Default: sflow_output/%j-sflow-submit.out",
        ),
    ] = str(Path.cwd() / "sflow_output" / "%j-sflow-submit.out"),
    sbatch_error: Annotated[
        str,
        typer.Option(
            "--sbatch-error",
            "-E",
            help="Slurm error file pattern. Default: sflow_output/%j-sflow-submit.err",
        ),
    ] = str(Path.cwd() / "sflow_output" / "%j-sflow-submit.err"),
    partition: Annotated[
        Optional[str],
        typer.Option(
            "--partition",
            "-p",
            help="Slurm partition. Auto-detected from the cluster if not specified.",
        ),
    ] = None,
    account: Annotated[
        Optional[str],
        typer.Option(
            "--account",
            "-A",
            help="Slurm account. Auto-detected from the current user's associations if not specified.",
        ),
    ] = None,
    time: Annotated[
        Optional[str],
        typer.Option(
            "--time",
            help="Slurm time limit (e.g., 01:00:00)",
        ),
    ] = None,
    nodes: Annotated[
        Optional[int],
        typer.Option(
            "--nodes",
            "-N",
            help="Number of nodes for sbatch. If omitted in single-job mode, derived from the config's backends[].nodes field. "
            "With --bulk-input, optional if the CSV contains a SLURM_NODES, NUM_SLURM_NODES, or NUM_NODES column.",
        ),
    ] = None,
    gpus_per_node: Annotated[
        Optional[int],
        typer.Option(
            "--gpus-per-node",
            "-G",
            help="Number of GPUs per node for cluster topology. If not set, derived from the sflow config's backend definition. "
            "Applied to slurm backend config for resource planning. "
            "This does NOT add a sbatch directive; use -e '--gpus-per-node=N' if your cluster requires it.",
        ),
    ] = None,
    sbatch_extra_args: Annotated[
        Optional[List[str]],
        typer.Option(
            "--sbatch-extra-args",
            "-e",
            help="Additional sbatch directives to append as '#SBATCH' lines. "
            "Supports ${{ variables.X }} or ${{ X }} expressions resolved from the sflow config "
            "(e.g., -e '--segment=${{ SLURM_NODES }}'). "
            "Variable values from --set overrides and CSV bulk-input columns are applied "
            "before resolution. Use single quotes to prevent shell expansion. "
            "Can be used multiple times.",
        ),
    ] = None,
    # runtime options
    sflow_venv_path: Annotated[
        Optional[Path],
        typer.Option(
            "--sflow-venv-path",
            "-v",
            help="Path to Python virtual environment for sflow (e.g., /path/to/.venv). "
            "The script will activate this venv before running sflow, pay extra attention to the arch of python ( x86 / arm ) when using existing venv.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path.cwd() / "sflow_compute_node_venv",
    sflow_version: Annotated[
        Optional[str],
        typer.Option(
            "--sflow-version",
            help="Git ref (branch or tag) to install from the GitHub repo (e.g., 'main', 'v0.1.0'). If not specified, reuse the installed version in the existing venv, or create a new venv and install the latest main version.",
        ),
    ] = None,
    missable_tasks: Annotated[
        Optional[List[str]],
        typer.Option(
            "--missable-tasks",
            "-M",
            help="Task names or glob patterns (e.g. 'prefill_*') that may be absent when composing "
            "modular configs from multiple files. Absent missable tasks are removed from depends_on "
            "and probes with a warning. Only valid with multiple input files or --bulk-input/--bulk-submit. Repeatable.",
        ),
    ] = None,
    # output options
    sbatch_path: Annotated[
        Optional[Path],
        typer.Option(
            "--sbatch-path",
            "-o",
            help="Write the sbatch script to this file instead of stdout",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            help="Submit the job immediately after generating the script",
        ),
    ] = False,
    bulk_input: Annotated[
        Optional[Path],
        typer.Option(
            "--bulk-input",
            "-b",
            help="CSV file for bulk job generation. "
            "Reserved columns: 'sflow_config_file' (required; to merge multiple YAML files into "
            "one workflow, list them space-separated in a single cell, e.g. 'backend.yaml workflow.yaml'), "
            "'job_name' (optional, explicit name for the generated script and Slurm job), "
            "'missable_tasks' (optional, space-separated task names/globs). "
            "All other columns are matched to variable or artifact names as overrides. "
            "When 'job_name' is absent, names are auto-derived from unique config file stems. "
            "If --nodes is not provided, the CSV must contain one of: SLURM_NODES, NUM_SLURM_NODES, or NUM_NODES.",
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
    resolve: Annotated[
        bool,
        typer.Option(
            "-r",
            "--resolve",
            help="Resolve all resolvable variables to literal values in the generated YAML config "
            "(same as sflow compose --resolve). Works with single-job, --bulk-input, and --bulk-submit modes.",
        ),
    ] = False,
    bulk_submit: Annotated[
        Optional[List[Path]],
        typer.Option(
            "--bulk-submit",
            "-B",
            help="File path(s), folder(s), or glob pattern(s) of self-contained sflow YAML configs. "
            "Each valid YAML is processed as a standalone batch job (no merging). "
            "Folders are scanned for *.yaml/*.yml files. Glob patterns (e.g. 'examples/slurm_*') are expanded. "
            "CLI flags (--set, --artifact, --partition, etc.) are applied to every config.",
        ),
    ] = None,
):
    """
    Generate an sbatch script for running sflow in Slurm batch mode.

    This command creates a bash script with sbatch directives that wraps
    the 'sflow run' command for headless execution on a Slurm cluster.

    Three modes are supported:

    1. Single-job mode (default): generate one sbatch script from one or more
       YAML config files (merged into a single workflow). Requires --nodes.

    2. Bulk-input mode (--bulk-input): read a CSV file where each row defines
       a job with its own config files and variable/artifact overrides.

    3. Bulk-submit mode (--bulk-submit): pass file paths or folder paths of
       self-contained sflow YAML configs. Each valid YAML is processed as a
       standalone batch job (no merging). Folders are scanned for *.yaml/*.yml
       files. CLI flags (--set, --artifact, etc.) are applied to all configs.
       Warns when --set overrides a variable already defined in a config.

    ┌─────────────────────────────────────────────────────────────┐
    │  --bulk-input vs --bulk-submit                              │
    ├──────────────────────┬──────────────────────────────────────┤
    │                      │                                      │
    │  --bulk-input (-b)   │  --bulk-submit (-B)                  │
    │                      │                                      │
    │  CSV-driven          │  File/folder-driven                  │
    │                      │                                      │
    │  jobs.csv            │  ./examples/                         │
    │   ├─ row 1 ──┐      │   ├─ sglang_agg.yaml ──→ job 1      │
    │   ├─ row 2 ──┤      │   ├─ vllm_agg.yaml   ──→ job 2      │
    │   └─ row 3 ──┘      │   └─ trtllm_agg.yaml ──→ job 3      │
    │                      │                                      │
    │  Each row can:       │  Each YAML is:                       │
    │  · merge N files     │  · self-contained (no merging)       │
    │  · override vars     │  · CLI --set applied uniformly       │
    │  · override arts     │  · nodes from config or CLI          │
    │                      │                                      │
    │  Use when configs    │  Use when each YAML is a complete    │
    │  are modular and     │  standalone workflow ready to run     │
    │  need per-row        │  as-is                               │
    │  customization       │                                      │
    └──────────────────────┴──────────────────────────────────────┘

    Sbatch stdout/stderr logs are automatically copied into the sflow workflow
    output directory at the end of each generated script.

    CSV format for --bulk-input:

        sflow_config_file              SLURM_NODES  MODEL_PATH
        backend.yaml sglang/agg.yaml   2            /models/llama
        backend.yaml trtllm/agg.yaml   4            /models/llama

        The 'sflow_config_file' column is required. To merge multiple YAML
        files into one workflow, list them space-separated in a single cell.
        Other columns are matched to variable or artifact names as overrides.

    Examples:
        # Generate sbatch script to stdout
        sflow batch workflow.yaml

        # Merge multiple config files
        sflow batch backends.yaml workflow.yaml tasks.yaml

        # Generate and save to file
        sflow batch workflow.yaml --sbatch-path run_workflow.sh

        # Generate with Slurm options
        sflow batch workflow.yaml --partition gpu --time 02:00:00 --account myaccount

        # Generate with GPU allocation
        sflow batch workflow.yaml --nodes 2 --gpus-per-node 8

        # Generate and submit immediately
        sflow batch workflow.yaml --partition gpu --submit

        # With variable overrides
        sflow batch workflow.yaml --set NUM_GPUS=8 --set MODEL=llama

        # With custom virtual environment
        sflow batch workflow.yaml --sflow-venv-path /path/to/.venv

        # With extra sbatch directives (supports ${{ variables.X }} expressions)
        sflow batch workflow.yaml -e "--exclusive" -e "--segment=${{ variables.SLURM_NODES }}"

        # Bulk input: generate one job per CSV row (--nodes not required)
        sflow batch --bulk-input jobs.csv --partition gpu --account myaccount

        # Bulk submit: process all YAML files in a folder as standalone workflows
        sflow batch --bulk-submit ./examples/ --partition gpu --account myaccount --submit

        # Bulk submit: specific files
        sflow batch -B sglang_agg.yaml -B vllm_agg.yaml --partition gpu --submit

        # Bulk submit: with variable overrides applied to all configs
        sflow batch -B ./examples/ --set SLURM_NODES=2 --partition gpu --submit
    """
    configure_logging(level=log_level, console=True)

    partition, account = _resolve_slurm_defaults(partition, account)

    if row and bulk_input is None:
        typer.echo("Error: --row requires --bulk-input.", err=True)
        raise typer.Exit(code=1)

    # --- Bulk-edit mode ---
    if bulk_input is not None:
        parsed_rows = parse_row_selector(row) if row else None
        try:
            _run_bulk_edit(
                csv_path=bulk_input,
                cli_set_var=set_var,
                cli_artifact=artifact,
                log_level=log_level,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                job_name=job_name,
                sbatch_output=sbatch_output,
                sbatch_error=sbatch_error,
                partition=partition,
                account=account,
                time=time,
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                sbatch_extra_args=sbatch_extra_args,
                sflow_venv_path=sflow_venv_path,
                sflow_version=sflow_version,
                submit=submit,
                row_filter=parsed_rows,
                resolve=resolve,
                missable_tasks=missable_tasks,
            )
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        return

    # --- Bulk-submit mode ---
    if bulk_submit is not None:
        all_paths = list(bulk_submit)
        if src_files:
            all_paths.extend(src_files)
        if file:
            all_paths.extend(file)

        csv_in_bulk_submit = [p for p in all_paths if p.is_file() and p.suffix.lower() == ".csv"]
        if csv_in_bulk_submit:
            names = ", ".join(str(f) for f in csv_in_bulk_submit)
            typer.echo(
                f"Error: CSV file(s) detected in --bulk-submit input: {names}\n"
                f"  --bulk-submit expects sflow YAML files or directories, not CSV.\n"
                f"  Did you mean to use --bulk-input (-b)?\n"
                f"  Example: sflow batch --bulk-input {csv_in_bulk_submit[0]}",
                err=True,
            )
            raise typer.Exit(code=1)

        yaml_files = _scan_sflow_yamls(all_paths)
        if not yaml_files:
            typer.echo(
                "Error: no valid sflow YAML files found in the provided paths.",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"Found {len(yaml_files)} sflow YAML config(s):")
        for yf in yaml_files:
            typer.echo(f"  - {yf.name}")
        try:
            _run_bulk_submit(
                yaml_files=yaml_files,
                cli_set_var=set_var,
                cli_artifact=artifact,
                log_level=log_level,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                sbatch_output=sbatch_output,
                sbatch_error=sbatch_error,
                partition=partition,
                account=account,
                time=time,
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                sbatch_extra_args=sbatch_extra_args,
                sflow_venv_path=sflow_venv_path,
                sflow_version=sflow_version,
                submit=submit,
                missable_tasks=missable_tasks,
                resolve=resolve,
            )
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        return

    # --- Single-job mode ---
    files = list(src_files or []) + list(file or [])
    if not files:
        files = [Path("sflow.yaml").resolve()]

    csv_files = [f for f in files if f.suffix.lower() == ".csv"]
    if csv_files:
        names = ", ".join(str(f) for f in csv_files)
        typer.echo(
            f"Error: CSV file(s) detected in input: {names}\n"
            f"  CSV files cannot be used as workflow YAML inputs directly.\n"
            f"  Did you mean to use --bulk-input (-b)?\n"
            f"  Example: sflow batch --bulk-input {csv_files[0]}",
            err=True,
        )
        raise typer.Exit(code=1)

    if missable_tasks and len(files) < 2:
        typer.echo(
            "Error: --missable-tasks is only valid with multiple input files (modular configs).",
            err=True,
        )
        raise typer.Exit(code=1)

    # Derive nodes from config if not given via CLI
    if nodes is None:
        nodes = _derive_nodes(files, cli_overrides=set_var)
        if nodes is not None:
            typer.echo(
                f"  Info: --nodes not specified, derived from config: {nodes}",
                err=True,
            )
        else:
            typer.echo(
                "Error: --nodes not specified and could not be derived from config backends.",
                err=True,
            )
            raise typer.Exit(code=1)

    # Derive gpus_per_node from config if not given via CLI
    if gpus_per_node is None:
        gpus_per_node = _derive_gpus_per_node(files, cli_overrides=set_var)
        if gpus_per_node is not None:
            typer.echo(
                f"  Info: --gpus-per-node not specified, derived from config: {gpus_per_node}",
                err=True,
            )

    # Run dry-run validation before generating sbatch script
    typer.echo("Running dry-run validation before generating sbatch script...")
    try:
        _sflow_app.run(
            file=files,
            dry_run=True,
            quiet=True,
            variable_overrides=set_var,
            artifact_overrides=artifact,
            missable_tasks=missable_tasks,
            slurm_nodes=nodes,
            slurm_gpus_per_node=gpus_per_node,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
        )
        typer.echo("✓ Dry-run validation passed\n")
    except ValueError as e:
        msg = enrich_error_with_location(str(e), files)
        typer.echo(f"✗ Configuration error: {msg}", err=True)
        typer.echo("Aborting sbatch generation due to configuration errors.", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        typer.echo(f"✗ File not found: {e}", err=True)
        typer.echo("Aborting sbatch generation due to missing files.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"✗ Dry-run validation failed: {e}", err=True)
        typer.echo("Aborting sbatch generation due to validation errors.", err=True)
        raise typer.Exit(code=1)

    script_content = _generate_sbatch_script(
        files=files,
        set_var=set_var,
        artifact=artifact,
        missable_tasks=missable_tasks,
        log_level=log_level,
        workspace_dir=workspace_dir,
        output_dir=output_dir,
        job_name=job_name,
        sbatch_output=sbatch_output,
        sbatch_error=sbatch_error,
        partition=partition,
        account=account,
        time=time,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        sbatch_extra_args=sbatch_extra_args,
        sflow_venv_path=sflow_venv_path,
        sflow_version=sflow_version,
    )

    # Generate composed/resolved YAML alongside the sbatch script
    if sbatch_path:
        try:
            from sflow.cli.compose import _compose_files

            yaml_output = _compose_files(
                files,
                set_var or None,
                artifact or None,
                log_level,
                resolve=resolve,
                missable_tasks=missable_tasks,
                quiet_missable=True,
            )
            yaml_path = sbatch_path.with_suffix(".yaml")
            yaml_path.write_text(yaml_output)
            typer.echo(f"✓ Composed config written to: {yaml_path}")
        except Exception as e:
            typer.echo(f"  Warning: could not generate composed config: {e}", err=True)

    if sbatch_path:
        sbatch_path.write_text(script_content)
        sbatch_path.chmod(0o755)
        typer.echo(script_content)
        typer.echo(f"✓ Sbatch script written to: {sbatch_path}")

        if submit:
            try:
                msg = _submit_sbatch(sbatch_path)
                typer.echo(f"✓ Job submitted: {msg}")
            except RuntimeError as e:
                typer.echo(f"✗ {e}", err=True)
                raise typer.Exit(code=1)
    else:
        typer.echo(script_content)
        typer.echo(
            "\n# (stdout only — to save as a file, add: -o <path>.sh)",
            err=True,
        )

        if submit:
            typer.echo(
                "⚠ Cannot submit without --sbatch-path / -o. "
                "Please specify a file to save the script first.",
                err=True,
            )
            raise typer.Exit(code=1)
