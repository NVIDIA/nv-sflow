# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Support helpers for the SflowApp run lifecycle."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from sflow.core.artifact_registry import resolve_file_like_uri_to_path
from sflow.logging import get_logger
from sflow.utils.container import append_missing_mounts

_logger = get_logger(__name__)


@dataclass(frozen=True)
class RunPaths:
    workspace_dir: Path
    output_dir: Path
    workflow_output_dir: Path
    run_id: str | None


def _default_run_id(workflow_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{workflow_name}-{timestamp}-{secrets.token_hex(3)}"


def build_run_paths(
    *,
    workflow_name: str,
    dry_run: bool,
    workspace_dir: Path | None,
    output_dir: Path | None,
    run_id_factory: Callable[[], str] | None = None,
    slurm_job_id: str | None = None,
) -> RunPaths:
    """Compute workspace/output paths for an sflow run without creating them."""
    ws_dir = Path(workspace_dir) if workspace_dir is not None else Path.cwd()
    out_dir = Path(output_dir) if output_dir is not None else ws_dir / "sflow_output"

    if dry_run:
        return RunPaths(
            workspace_dir=ws_dir,
            output_dir=out_dir,
            workflow_output_dir=out_dir / "_dry_run" / workflow_name,
            run_id=None,
        )

    run_id = (run_id_factory or (lambda: _default_run_id(workflow_name)))()
    if slurm_job_id:
        run_id = f"{slurm_job_id}-{run_id}"
    return RunPaths(
        workspace_dir=ws_dir,
        output_dir=out_dir,
        workflow_output_dir=out_dir / run_id,
        run_id=run_id,
    )


def _raw_variable_values(variable_configs: list[Any] | None) -> dict[str, Any]:
    raw_vars: dict[str, Any] = {}
    for variable_conf in variable_configs or []:
        if not (hasattr(variable_conf, "name") and hasattr(variable_conf, "value")):
            continue
        value = variable_conf.value
        if value is not None and not (isinstance(value, str) and "${{" in value):
            raw_vars[variable_conf.name] = value
    return raw_vars


def _resolve_raw_variable_refs(uri: str, raw_vars: dict[str, Any]) -> str:
    def _resolve_var(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        if ref.startswith("variables."):
            name = ref[len("variables.") :]
            if name in raw_vars:
                return str(raw_vars[name])
        return match.group(0)

    return re.sub(r"\$\{\{(.+?)\}\}", _resolve_var, uri)


def preflight_validate_artifacts(
    artifact_configs: list[Any] | None,
    variable_configs: list[Any] | None,
    workspace_dir: Path,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Validate local artifact paths before a run can allocate backend resources."""
    raw_vars = _raw_variable_values(variable_configs)

    errors: list[str] = []
    warnings: list[str] = []
    for artifact_conf in artifact_configs or []:
        uri = str(artifact_conf.uri)
        if "${{" in uri:
            uri = _resolve_raw_variable_refs(uri, raw_vars)
            if "${{" in uri:
                continue
        try:
            scheme = (urlparse(uri).scheme or "").lower()
        except Exception:
            continue
        if scheme not in {"fs", "file"}:
            continue
        if scheme == "file" and getattr(artifact_conf, "content", None) is not None:
            continue
        try:
            resolved = resolve_file_like_uri_to_path(uri, workspace_dir=workspace_dir)
        except Exception:
            continue
        path_str = str(resolved)
        if "$" in path_str or "{" in path_str:
            continue
        if not resolved.exists():
            if scheme == "fs":
                message = (
                    f"Artifact '{artifact_conf.name}' (fs://) path does not exist: {resolved}"
                )
                if dry_run:
                    warnings.append(message)
                else:
                    errors.append(message)
            else:
                warnings.append(
                    f"Artifact '{artifact_conf.name}' (file://) path does not exist: {resolved}"
                )

    if errors:
        for error in errors:
            _logger.error(f"  ✗ {error}")
    if warnings and not dry_run:
        for warning in warnings:
            _logger.warning(f"  ⚠ {warning}")
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"Artifact path validation failed:\n{details}")
    return warnings


def ensure_sflow_dir_mounts_for_srun_container(
    *,
    task: Any,
    ws_dir: Path,
    out_dir: Path,
    workflow_out_dir: Path,
    task_out_dir: Path,
) -> None:
    """Mount SFLOW_* host directories into Pyxis containers at the same paths."""
    try:
        op = getattr(task, "operator", None)
        op_conf = getattr(op, "config", None)
        if op_conf is None:
            return
        if getattr(op_conf, "type", None) != "srun":
            return
        if not (
            getattr(op_conf, "container_image", None)
            or getattr(op_conf, "container_name", None)
        ):
            return

        existing_mounts = list(getattr(op_conf, "container_mounts", None) or [])
        candidate_mounts = []
        for path in (ws_dir, out_dir, workflow_out_dir, task_out_dir):
            mount_path = str(path)
            candidate_mounts.append(f"{mount_path}:{mount_path}:rw")

        merged_mounts = append_missing_mounts(existing_mounts, candidate_mounts)
        if merged_mounts != existing_mounts:
            setattr(op_conf, "container_mounts", merged_mounts)
    except Exception:
        return


def validate_container_mounts(tasks: list[Any], *, sflow_output_dir: Path) -> list[str]:
    """Check dry-run container mount source paths that should already exist."""
    warnings: list[str] = []
    sflow_out_str = str(sflow_output_dir)
    for task in tasks:
        op = getattr(task, "operator", None)
        op_conf = getattr(op, "config", None)
        if op_conf is None:
            continue

        mounts = getattr(op_conf, "container_mounts", None) or []
        if not mounts:
            mounts = getattr(op_conf, "mounts", None) or []

        for mount_spec in mounts:
            parts = str(mount_spec).split(":", 2)
            if len(parts) < 2:
                continue
            host_path = parts[0]
            if not host_path:
                continue
            if "$" in host_path or "{" in host_path:
                continue
            if host_path.startswith(sflow_out_str):
                continue
            if not Path(host_path).exists():
                warnings.append(
                    f"Task '{task.name}': mount source path does not exist: {host_path}"
                )
    return warnings


def check_enroot_credentials(
    tasks: list[Any],
    *,
    credentials_path: Path | None = None,
) -> str | None:
    """Warn if any srun task uses a container but enroot credentials are missing."""
    uses_container = False
    for task in tasks:
        op_conf = getattr(getattr(task, "operator", None), "config", None)
        if op_conf is None:
            continue
        if getattr(op_conf, "type", None) != "srun":
            continue
        if getattr(op_conf, "container_image", None) or getattr(
            op_conf, "container_name", None
        ):
            uses_container = True
            break
    if not uses_container:
        return None

    creds_path = credentials_path or Path.home() / ".config" / "enroot" / ".credentials"
    if not creds_path.exists():
        return (
            f"srun operator uses container images but enroot credentials "
            f"file not found at {creds_path}. "
            f"Container pulls from authenticated registries (e.g. nvcr.io) "
            f"may fail. See: https://github.com/NVIDIA/enroot/blob/master/doc/cmd/import.md"
        )
    return None


def configure_task_runtime(
    task: Any,
    *,
    ws_dir: Path,
    out_dir: Path,
    workflow_out_dir: Path,
    dry_run: bool,
) -> Path:
    """Set per-task SFLOW_* paths and return the task output directory."""
    task_out_dir = workflow_out_dir / task.name
    if not dry_run:
        task_out_dir.mkdir(parents=True, exist_ok=True)

    task.envs.setdefault("SFLOW_WORKSPACE_DIR", str(ws_dir))
    task.envs.setdefault("SFLOW_OUTPUT_DIR", str(out_dir))
    task.envs.setdefault("SFLOW_WORKFLOW_OUTPUT_DIR", str(workflow_out_dir))
    task.envs.setdefault("SFLOW_TASK_OUTPUT_DIR", str(task_out_dir))
    ensure_sflow_dir_mounts_for_srun_container(
        task=task,
        ws_dir=ws_dir,
        out_dir=out_dir,
        workflow_out_dir=workflow_out_dir,
        task_out_dir=task_out_dir,
    )
    return task_out_dir
