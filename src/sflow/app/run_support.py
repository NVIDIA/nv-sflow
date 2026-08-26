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
from sflow.utils.container import collect_container_mounts

_logger = get_logger(__name__)


# Environment variable names that sflow injects and manages for every task.
# A user-declared variable reusing one of these names collides with sflow's own
# injection at launch (see ``configure_task_runtime`` and the backend/operator
# runtime env), leading to undefined behavior. Keep this in sync with the
# "Injected by sflow" table in docs/user/quick-reference.md.
SFLOW_RESERVED_ENV_VARS = frozenset(
    {
        "SFLOW_WORKSPACE_DIR",
        "SFLOW_OUTPUT_DIR",
        "SFLOW_WORKFLOW_OUTPUT_DIR",
        "SFLOW_TASK_OUTPUT_DIR",
        "SFLOW_TASK_RESULT_FILE",
        "SFLOW_WORKFLOW_RESULT_FILE",
        "SFLOW_REPLICA_INDEX",
        "SFLOW_TASK_ASSIGNED_NODE_NAMES",
        "SFLOW_TASK_ASSIGNED_NODE_IPS",
        "SFLOW_BACKEND_JOB_ID",
        "SFLOW_BACKEND_NODELIST",
        "SFLOW_BACKEND_NUM_NODES",
        "SFLOW_BACKEND_STEP_ID",
        "SFLOW_TASK_NODE_NAME",
        "SFLOW_TASK_NODE_INDEX",
        "SFLOW_TASK_PROCESS_ID",
        "SFLOW_TASK_LOCAL_PROCESS_ID",
        "SFLOW_TASK_NUM_PROCESSES",
        "CUDA_VISIBLE_DEVICES",
    }
)


def find_reserved_env_collisions(variable_names: Any) -> list[str]:
    """Return sorted user variable names that collide with sflow reserved env vars.

    ``variable_names`` is any iterable of names (e.g. the keys of the resolved
    workflow variables). Names not in ``SFLOW_RESERVED_ENV_VARS`` are ignored.
    """
    return sorted(
        {name for name in (variable_names or []) if name in SFLOW_RESERVED_ENV_VARS}
    )


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
    run_id_prefix: str | None = None,
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
    if run_id_prefix:
        run_id = f"{run_id_prefix}-{run_id}"
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


# Off-host backend detection lives here as the single source of truth. There are
# two checks because they run at different stages, but they must agree:
#   * config_uses_offhost_backend  -- by backend *type*, BEFORE resolution.
#   * backends_execute_offhost     -- by *capability*, AFTER resolution.
# When adding a new off-host backend, add its type here AND set
# ``supports_host_path_mounts=False`` on its capabilities so both checks match.
_OFFHOST_BACKEND_TYPES = frozenset({"kubernetes"})


def config_uses_offhost_backend(config: Any) -> bool:
    """True if any configured backend executes off the controller host (e.g. k8s).

    Determined from the backend *type* at config time (before backends are
    resolved). Such backends run tasks remotely, so a local ``fs://`` artifact path
    refers to a location on the cluster/image, not the controller -- it must not be
    hard-validated locally. Pre-resolution twin of :func:`backends_execute_offhost`
    (which checks ``supports_host_path_mounts`` on resolved backends).
    """
    for backend in getattr(config, "backends", None) or []:
        btype = getattr(backend, "type", None)
        if btype is None and isinstance(backend, dict):
            btype = backend.get("type")
        if str(btype) in _OFFHOST_BACKEND_TYPES:
            return True
    return False


def backends_execute_offhost(state: Any) -> bool:
    """True if any resolved backend executes off the controller host (no host mounts).

    Post-resolution twin of :func:`config_uses_offhost_backend`: checks the
    resolved backends' ``supports_host_path_mounts`` capability. Such backends
    (e.g. Kubernetes) run tasks remotely, so local ``fs://`` artifact paths refer
    to the cluster/image rather than the controller and must not be validated or
    created locally.
    """
    return any(
        not getattr(
            getattr(backend, "capabilities", None), "supports_host_path_mounts", True
        )
        for backend in (getattr(state, "backends", None) or {}).values()
    )


def preflight_validate_artifacts(
    artifact_configs: list[Any] | None,
    variable_configs: list[Any] | None,
    workspace_dir: Path,
    *,
    dry_run: bool = False,
    skip_local_fs_validation: bool = False,
) -> list[str]:
    """Validate local artifact paths before a run can allocate backend resources.

    ``skip_local_fs_validation`` demotes a missing ``fs://`` path from an error to a
    warning. It is set when the workflow targets an off-host backend (e.g. Kubernetes),
    where the path lives on the remote cluster/image rather than the controller, and by
    the user via ``--skip-artifact-check`` for the same situation on a backend sflow
    cannot detect (e.g. a Slurm path that only exists on the compute nodes).
    """
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
                if skip_local_fs_validation and not dry_run:
                    warnings.append(
                        f"Artifact '{artifact_conf.name}' (fs://) path '{resolved}' "
                        "does not exist locally; continuing without local validation "
                        "(off-host backend such as Kubernetes, or --skip-artifact-check)."
                    )
                elif dry_run:
                    warnings.append(
                        f"Artifact '{artifact_conf.name}' (fs://) path does not exist: {resolved}"
                    )
                else:
                    errors.append(
                        f"Artifact '{artifact_conf.name}' (fs://) path does not exist: {resolved}"
                    )
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


def ensure_sflow_dir_mounts_for_container_operator(
    *,
    task: Any,
    ws_dir: Path,
    out_dir: Path,
    workflow_out_dir: Path,
    task_out_dir: Path,
) -> None:
    """Mount SFLOW_* host directories for operators that support host mounts."""
    try:
        op = getattr(task, "operator", None)
        op_conf = getattr(op, "config", None)
        if op_conf is None:
            return

        candidate_mounts = []
        for path in (ws_dir, out_dir, workflow_out_dir, task_out_dir):
            mount_path = str(path)
            candidate_mounts.append(f"{mount_path}:{mount_path}:rw")

        append_runtime_mounts = getattr(op_conf, "append_runtime_mounts", None)
        if callable(append_runtime_mounts):
            append_runtime_mounts(candidate_mounts)
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

        mounts = collect_container_mounts(op_conf)

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


def collect_operator_runtime_warnings(tasks: list[Any]) -> list[str]:
    """Collect dry-run warnings from operator config hooks.

    Many operator warnings are task-agnostic (e.g. the same enroot-credentials
    warning emitted by every container task/replica). Identical messages are
    collapsed into a single entry, ordered by first occurrence, with an
    ``(affects N tasks)`` suffix when more than one task is impacted — so the
    output stays readable without losing visibility into the blast radius.
    """
    counts: dict[str, int] = {}
    for task in tasks:
        op_conf = getattr(getattr(task, "operator", None), "config", None)
        if op_conf is None:
            continue
        runtime_warnings = getattr(op_conf, "runtime_warnings", None)
        if not callable(runtime_warnings):
            continue
        try:
            task_warnings = [str(warning) for warning in runtime_warnings()]
        except Exception:
            continue
        # Count each distinct message once per task, so the suffix reflects the
        # number of impacted tasks rather than the raw emission count.
        for message in dict.fromkeys(task_warnings):
            counts[message] = counts.get(message, 0) + 1

    return [
        f"{message} (affects {count} tasks)" if count > 1 else message
        for message, count in counts.items()
    ]


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
    # Consolidated result parsing: direct-write target plus workflow-level index.
    task.envs.setdefault("SFLOW_TASK_RESULT_FILE", str(task_out_dir / "result.json"))
    task.envs.setdefault(
        "SFLOW_WORKFLOW_RESULT_FILE",
        str(workflow_out_dir / "results.json"),
    )
    ensure_sflow_dir_mounts_for_container_operator(
        task=task,
        ws_dir=ws_dir,
        out_dir=out_dir,
        workflow_out_dir=workflow_out_dir,
        task_out_dir=task_out_dir,
    )
    return task_out_dir
