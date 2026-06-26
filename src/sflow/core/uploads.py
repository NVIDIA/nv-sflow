# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-task post-completion upload runner.

Invoked by the orchestrator immediately after a task reaches COMPLETED. Each
`ResolvedUpload` on the task is resolved (`${{ }}` → concrete values), expanded
via glob if applicable, and uploaded to the named `StorageTarget`.

Failures are governed by the spec's `on_error`:
- `warn`: log and continue; the task remains COMPLETED.
- `fail`: accumulate; `run_task_uploads` returns False, and the orchestrator
  flips the task status to FAILED (which the existing fail-fast handles).
"""

from __future__ import annotations

import asyncio
import glob as _glob
import os
import posixpath
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from sflow.config.resolver import ExpressionResolver
from sflow.logging import get_logger

from .task import Task

if TYPE_CHECKING:
    from .storage import StorageTarget


@dataclass(frozen=True)
class ResolvedWorkflowUpload:
    """Runtime representation of `workflow.upload_all:` — zip the whole output dir."""

    target: str
    to_expr: str | None = None
    on_error: Literal["warn", "fail"] = "warn"


@dataclass(frozen=True)
class UploadResult:
    """One file/outcome from an upload spec, collected for the end-of-run summary.

    Populated by ``run_task_uploads`` / ``run_workflow_upload`` when a ``results``
    list is supplied, then rendered as a dedicated "Uploads" section instead of
    being scattered as inline log lines. ``status`` is one of:

    - ``uploaded``: file transferred successfully
    - ``failed``:   transfer or pre-transfer error (severity governed by ``on_error``)
    - ``skipped``:  nothing to do (e.g. a glob that matched no files)
    - ``dry-run``:  planned only; no transfer attempted
    """

    task: str
    target: str
    source: str
    destination: str
    status: Literal["uploaded", "failed", "skipped", "dry-run"]
    on_error: Literal["warn", "fail"]
    error: str | None = None


_logger = get_logger(__name__)

_resolver = ExpressionResolver()


def _build_upload_ctx(task: Task) -> dict[str, Any]:
    """Minimal expression context for resolving upload from/to at task-completion time."""
    return {
        "task": {
            "name": task.name,
            "output_dir": task.envs.get("SFLOW_TASK_OUTPUT_DIR"),
            "backend": task.backend_name,
            "operator": task.operator_name,
            "assigned_nodes": list(task.assigned_nodes or []),
        },
        "env": dict(task.envs),
    }


def _resolve_expr(value: str | None, ctx: dict[str, Any]) -> str | None:
    if value is None:
        return None
    return str(_resolver.resolve(value, ctx))


def _insert_replica_suffix(key: str, label: str) -> str:
    """Insert `_<label>` before the file extension of the key's final path segment.

    Used to disambiguate uploads from replicas of the same task so they don't
    overwrite each other on the storage target. Examples::

        ("main/results.csv", "benchmark_0") -> "main/results_benchmark_0.csv"
        ("results.csv",       "benchmark_0") -> "results_benchmark_0.csv"
        ("results",           "benchmark_0") -> "results_benchmark_0"
    """
    head, tail = posixpath.split(key)
    stem, ext = posixpath.splitext(tail)
    new_tail = f"{stem}_{label}{ext}"
    return posixpath.join(head, new_tail) if head else new_tail


def _compute_remote_key(
    *,
    target_prefix: str,
    to_template: str | None,
    local_path: Path,
    matched_files: list[Path],
    replica_label: str | None = None,
) -> str:
    """Join target prefix + per-file remote key.

    - If `to_template` is None: use basename of the local file under the prefix.
    - If `to_template` ends with `/`: treat as directory; append basename.
    - Otherwise (single match, explicit path): use `to_template` verbatim.

    When `replica_label` is set, `_<replica_label>` is inserted before the
    extension of the key's basename so replicas of the same task write to
    distinct keys instead of silently overwriting each other.

    Raises ValueError when the resolved `to_template` would escape the configured
    prefix (`..` segments) or when a literal `to:` is paired with a glob that
    matched multiple files (silent layout collision).
    """
    if to_template is None:
        key = local_path.name
    elif to_template.endswith("/"):
        key = to_template + local_path.name
    elif len(matched_files) > 1:
        # A literal `to:` against multiple glob matches has ambiguous semantics
        # (layout flips between runs depending on match count). Schema validation
        # catches the obvious cases; this guards against expression-resolved globs.
        raise ValueError(
            f"upload 'to' must end with '/' when matching multiple files; "
            f"got to='{to_template}' with {len(matched_files)} matches"
        )
    else:
        key = to_template

    # Reject path-traversal in user-supplied `to:` so a malicious or careless
    # value can't escape the target's configured prefix.
    if ".." in key.split("/"):
        raise ValueError(
            f"upload 'to' contains '..' which is not allowed (got '{to_template}')"
        )

    # Disambiguate replica uploads (system-generated label; applied after the
    # `..` check since the label itself is trusted).
    if replica_label:
        key = _insert_replica_suffix(key, replica_label)

    if target_prefix:
        # posixpath.join handles leading slashes correctly for object keys.
        if target_prefix.endswith("/"):
            return target_prefix + key.lstrip("/")
        return posixpath.join(target_prefix, key.lstrip("/"))
    return key.lstrip("/")


async def _run_one_upload(
    *,
    task: Task,
    spec_index: int,
    target: "StorageTarget",
    from_expr: str,
    to_expr: str | None,
    on_error: str,
    ctx: dict[str, Any],
    dry_run: bool,
    replica_label: str | None = None,
    results: list[UploadResult] | None = None,
) -> bool:
    """Run a single upload spec. Returns False only on a fatal (on_error=fail) error."""

    def _record(
        status: str,
        source: str,
        destination: str,
        error: str | None = None,
    ) -> None:
        if results is None:
            return
        results.append(
            UploadResult(
                task=task.name,
                target=target.name,
                source=source,
                destination=destination,
                status=status,  # type: ignore[arg-type]
                on_error=on_error,  # type: ignore[arg-type]
                error=error,
            )
        )

    try:
        from_resolved = _resolve_expr(from_expr, ctx)
        to_resolved = _resolve_expr(to_expr, ctx)
    except Exception as e:
        msg = f"Task '{task.name}' upload[{spec_index}]: failed to resolve expression: {e}"
        _record("failed", from_expr, "(unresolved)", error=str(e))
        if on_error == "fail":
            _logger.error(msg)
            return False
        _logger.warning(msg)
        return True

    if not from_resolved:
        msg = (
            f"Task '{task.name}' upload[{spec_index}]: empty 'from' after resolution "
            f"(original: '{from_expr}'). Check for typos in referenced variables."
        )
        _record("failed", from_expr, "(unresolved)", error="empty 'from' after resolution")
        if on_error == "fail":
            _logger.error(msg)
            return False
        _logger.warning(msg)
        return True

    # Anchor relative paths under the task output dir.
    base_dir = task.envs.get("SFLOW_TASK_OUTPUT_DIR")
    if not os.path.isabs(from_resolved) and base_dir:
        from_resolved = os.path.join(base_dir, from_resolved)

    # Expand glob (also handles plain paths — returns a single-element list).
    matches_str = _glob.glob(from_resolved, recursive=True)
    matched_files: list[Path] = [Path(m) for m in matches_str if Path(m).is_file()]

    if not matched_files:
        msg = (
            f"Task '{task.name}' upload[{spec_index}]: pattern '{from_resolved}' "
            f"(from: '{from_expr}') matched no files"
        )
        _record(
            "skipped",
            from_resolved,
            "(no match)",
            error="pattern matched no files",
        )
        if on_error == "fail":
            _logger.error(msg)
            return False
        _logger.warning(msg)
        return True

    target_prefix = getattr(target, "prefix", "") or ""

    any_failed = False
    for local_path in matched_files:
        try:
            remote_key = _compute_remote_key(
                target_prefix=target_prefix,
                to_template=to_resolved,
                local_path=local_path,
                matched_files=matched_files,
                replica_label=replica_label,
            )
        except ValueError as e:
            msg = f"Task '{task.name}' upload[{spec_index}]: {e}"
            _record("failed", str(local_path), "(invalid key)", error=str(e))
            if on_error == "fail":
                _logger.error(msg)
                return False
            _logger.warning(msg)
            return True
        destination = target.plan(local_path, remote_key)
        if dry_run:
            _record("dry-run", str(local_path), destination)
            _logger.info(f"[dry-run] would upload {local_path} -> {destination}")
            continue

        _logger.info(
            f"Task '{task.name}' upload[{spec_index}]: uploading "
            f"{local_path} -> {destination}"
        )
        try:
            await target.upload(local_path, remote_key)
            _record("uploaded", str(local_path), destination)
            # Per-file success is intentionally quiet on the main log; it appears
            # in the end-of-run Uploads summary instead.
            _logger.debug(f"Uploaded {local_path} -> {destination}")
        except Exception as e:
            msg = (
                f"Task '{task.name}' upload[{spec_index}]: failed to upload "
                f"{local_path} -> {destination}: {e}"
            )
            _record("failed", str(local_path), destination, error=str(e))
            if on_error == "fail":
                _logger.error(msg)
                any_failed = True
            else:
                _logger.warning(msg)

    return not any_failed


async def run_task_uploads(
    task: Task,
    storage_targets: dict[str, "StorageTarget"],
    *,
    dry_run: bool = False,
    results: list[UploadResult] | None = None,
) -> bool:
    """
    Run all uploads attached to `task`.

    Returns True if no upload with `on_error="fail"` failed. Uploads with
    `on_error="warn"` log warnings but do not affect the return value.

    When `results` is provided, one `UploadResult` per file/outcome is appended
    so callers can render a consolidated end-of-run upload summary.
    """
    if not task.uploads:
        return True

    ctx = _build_upload_ctx(task)
    all_ok = True

    for i, spec in enumerate(task.uploads):
        target = storage_targets.get(spec.target)
        if target is None:
            msg = (
                f"Task '{task.name}' upload[{i}]: storage target "
                f"'{spec.target}' not found"
            )
            if results is not None:
                results.append(
                    UploadResult(
                        task=task.name,
                        target=spec.target,
                        source=spec.from_expr,
                        destination="(target not found)",
                        status="failed",
                        on_error=spec.on_error,
                        error=f"storage target '{spec.target}' not found",
                    )
                )
            if spec.on_error == "fail":
                _logger.error(msg)
                all_ok = False
            else:
                _logger.warning(msg)
            continue

        ok = await _run_one_upload(
            task=task,
            spec_index=i,
            target=target,
            from_expr=spec.from_expr,
            to_expr=spec.to_expr,
            on_error=spec.on_error,
            ctx=ctx,
            dry_run=dry_run,
            replica_label=spec.disambiguate_with,
            results=results,
        )
        if not ok:
            all_ok = False

    return all_ok


def _zip_directory(src_dir: Path, dest_zip: Path) -> None:
    """Zip the contents of `src_dir` into `dest_zip`, preserving relative paths."""
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path == dest_zip or not path.is_file():
                continue
            zf.write(path, arcname=path.relative_to(src_dir).as_posix())


def _plan_workflow_remote_key(target_prefix: str, key: str) -> str:
    """Mirror `_compute_remote_key`'s prefix join, but for the whole-workflow zip."""
    if target_prefix:
        if target_prefix.endswith("/"):
            return target_prefix + key.lstrip("/")
        return posixpath.join(target_prefix, key.lstrip("/"))
    return key.lstrip("/")


async def run_workflow_upload(
    spec: ResolvedWorkflowUpload,
    workflow_name: str,
    workflow_out_dir: Path,
    storage_targets: dict[str, "StorageTarget"],
    *,
    variables_ctx: dict[str, Any] | None = None,
    dry_run: bool = False,
    results: list[UploadResult] | None = None,
) -> bool:
    """
    Zip `workflow_out_dir` and upload it to the configured storage target.

    Returns True unless the spec is `on_error="fail"` and the upload fails.

    When `results` is provided, an `UploadResult` describing the whole-workflow
    archive upload is appended for the end-of-run upload summary.
    """

    def _record(
        status: str,
        source: str,
        destination: str,
        error: str | None = None,
    ) -> None:
        if results is None:
            return
        results.append(
            UploadResult(
                task="workflow.upload_all",
                target=spec.target,
                source=source,
                destination=destination,
                status=status,  # type: ignore[arg-type]
                on_error=spec.on_error,
                error=error,
            )
        )

    target = storage_targets.get(spec.target)
    if target is None:
        msg = (
            f"workflow.upload_all: storage target '{spec.target}' not found"
        )
        _record(
            "failed",
            str(workflow_out_dir),
            "(target not found)",
            error=f"storage target '{spec.target}' not found",
        )
        if spec.on_error == "fail":
            _logger.error(msg)
            return False
        _logger.warning(msg)
        return True

    run_id = workflow_out_dir.name
    ctx: dict[str, Any] = {
        "workflow": {
            "name": workflow_name,
            "run_id": run_id,
            "output_dir": str(workflow_out_dir),
        },
        "variables": variables_ctx or {},
        **(variables_ctx or {}),
    }

    try:
        to_resolved = _resolve_expr(spec.to_expr, ctx) if spec.to_expr else None
    except Exception as e:
        msg = f"workflow.upload_all: failed to resolve 'to' expression: {e}"
        _record("failed", str(workflow_out_dir), "(unresolved)", error=str(e))
        if spec.on_error == "fail":
            _logger.error(msg)
            return False
        _logger.warning(msg)
        return True

    remote_basename = to_resolved or f"{run_id}.zip"
    if ".." in remote_basename.split("/"):
        msg = (
            f"workflow.upload_all 'to' contains '..' which is not allowed "
            f"(got '{spec.to_expr}')"
        )
        _record(
            "failed",
            str(workflow_out_dir),
            "(invalid key)",
            error="'to' contains '..' which is not allowed",
        )
        if spec.on_error == "fail":
            _logger.error(msg)
            return False
        _logger.warning(msg)
        return True

    target_prefix = getattr(target, "prefix", "") or ""
    remote_key = _plan_workflow_remote_key(target_prefix, remote_basename)
    destination = target.plan(workflow_out_dir, remote_key)

    if dry_run:
        _record("dry-run", str(workflow_out_dir), destination)
        _logger.info(f"[dry-run] would zip {workflow_out_dir} -> {destination}")
        return True

    if not workflow_out_dir.is_dir():
        msg = (
            f"workflow.upload_all: output dir '{workflow_out_dir}' does not exist; "
            f"skipping"
        )
        _record(
            "skipped",
            str(workflow_out_dir),
            destination,
            error="output dir does not exist",
        )
        if spec.on_error == "fail":
            _logger.error(msg)
            return False
        _logger.warning(msg)
        return True

    # Write the zip to a temp file outside `workflow_out_dir` so it isn't
    # included in itself, then clean up after upload.
    tmp_dir = Path(tempfile.mkdtemp(prefix="sflow-upload-"))
    zip_path = tmp_dir / f"{run_id}.zip"
    try:
        try:
            await asyncio.to_thread(_zip_directory, workflow_out_dir, zip_path)
        except Exception as e:
            msg = (
                f"workflow.upload_all: failed to zip {workflow_out_dir} -> "
                f"{zip_path}: {e}"
            )
            _record("failed", str(workflow_out_dir), destination, error=str(e))
            if spec.on_error == "fail":
                _logger.error(msg)
                return False
            _logger.warning(msg)
            return True

        try:
            _logger.info(
                f"workflow.upload_all: uploading {workflow_out_dir} -> {destination}"
            )
            await target.upload(zip_path, remote_key)
            _record("uploaded", str(workflow_out_dir), destination)
            # Quiet on the main log; surfaced in the end-of-run Uploads summary.
            _logger.debug(
                f"Uploaded workflow output {workflow_out_dir} -> {destination}"
            )
        except Exception as e:
            msg = (
                f"workflow.upload_all: failed to upload "
                f"{zip_path} -> {destination}: {e}"
            )
            _record("failed", str(workflow_out_dir), destination, error=str(e))
            if spec.on_error == "fail":
                _logger.error(msg)
                return False
            _logger.warning(msg)
            return True
    finally:
        try:
            if zip_path.exists():
                zip_path.unlink()
            tmp_dir.rmdir()
        except OSError:
            pass

    return True
