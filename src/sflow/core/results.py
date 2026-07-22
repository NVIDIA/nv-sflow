# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Result parsing module.

Implements the consolidated ``result`` task entry described in
``docs/developer/dev-notes/result-parsing.md``.

Public surface:
- ``task_result_path(task)`` -> per-task ``result.json`` path
- ``workflow_results_path(task)`` -> workflow-level ``results.json`` path
- ``collect_task_result(task)`` -> async, parse + write canonical files
- ``parse_results_from_text(text, specs)`` -> pure parser for unit tests
- ``normalize_result_file(path, task)`` -> file-source normalization
- ``update_workflow_results(task, payload)`` -> atomic index update

The module intentionally keeps the new "result" contract separate from the
legacy ``outputs`` MVP (``sflow.core.outputs``) so users can opt into the new
shape without breaking existing pipelines.
"""

from __future__ import annotations

import asyncio
import json
import posixpath
import re
from pathlib import Path
from typing import Any

from sflow.logging import get_logger
from sflow.utils.parser import strip_sflow_log_prefix

from .task import ResultSpec, Task

_logger = get_logger(__name__)

SCHEMA_VERSION_TASK = "sflow.result.v1"
SCHEMA_VERSION_WORKFLOW = "sflow.results.v1"

# Per-index file asyncio.Lock so that concurrent task completions cannot
# corrupt the workflow-level results.json via interleaved read-modify-write.
# Today the orchestrator awaits collect_task_result serially, but this module
# must not depend on that — gating with a lock makes the contract explicit.
_workflow_index_locks: dict[Path, asyncio.Lock] = {}


def _index_lock(path: Path) -> asyncio.Lock:
    _workflow_index_locks.setdefault(path, asyncio.Lock())
    return _workflow_index_locks[path]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def task_result_path(task: Task) -> Path | None:
    """
    Canonical per-task result file:
        ``${SFLOW_TASK_OUTPUT_DIR}/result.json``

    Returns None when the task has no ``SFLOW_TASK_OUTPUT_DIR`` env var.
    """
    task_out = task.envs.get("SFLOW_TASK_OUTPUT_DIR")
    if task_out:
        return Path(task_out) / "result.json"
    return None


def workflow_results_path(task: Task) -> Path | None:
    """
    Canonical workflow-level results index.

    Prefers the explicit ``SFLOW_WORKFLOW_RESULT_FILE`` env var (single source
    of truth set by ``app/sflow.py``) so the writer and any user-side reader
    cannot drift apart. Falls back to ``${SFLOW_WORKFLOW_OUTPUT_DIR}/results.json``
    when the env var is unset (e.g. unit tests that only inject the dir).
    """
    explicit = task.envs.get("SFLOW_WORKFLOW_RESULT_FILE")
    if explicit:
        return Path(explicit)
    wf_out = task.envs.get("SFLOW_WORKFLOW_OUTPUT_DIR")
    if wf_out:
        return Path(wf_out) / "results.json"
    return None


def _task_log_path(task: Task) -> Path | None:
    """Per-task merged log used by regex result specs."""
    task_out = task.envs.get("SFLOW_TASK_OUTPUT_DIR")
    if task_out:
        return Path(task_out) / f"{task.name}.log"
    wf_out = task.envs.get("SFLOW_WORKFLOW_OUTPUT_DIR")
    if wf_out:
        return Path(wf_out) / task.name / f"{task.name}.log"
    return None


# ---------------------------------------------------------------------------
# Pure parser (no I/O)
# ---------------------------------------------------------------------------


def _strip_log_prefix(line: str) -> str:
    """Remove the standard sflow logging prefix from a log line, when present.

    The launcher writes lines via the formatter
    ``%(asctime)s - %(name)s - %(levelname)s - %(message)s`` (and the srun offload
    prefixer reproduces it byte-for-byte). The prefix is stripped only when it
    actually matches, so raw lines or messages containing " - " are preserved.
    """
    return strip_sflow_log_prefix(line)


def _cast_value(raw: str, type_: str) -> Any:
    """Cast a captured string to the requested type. Raises ValueError on failure."""
    if type_ in ("string", None):
        return raw
    if type_ == "int":
        return int(raw)
    if type_ == "float":
        return float(raw)
    if type_ == "bool":
        v = raw.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot cast {raw!r} to bool")
    if type_ == "json":
        return json.loads(raw)
    if type_ == "auto":
        s = raw.strip()
        # int (must succeed before float so "42" stays int, not 42.0)
        try:
            iv = int(s)
            # Reject leading-zero ambiguities? `int("042")` returns 42; that's fine.
            return iv
        except ValueError:
            pass
        # float (covers exponent notation, ".5", "1.5", "1.")
        try:
            return float(s)
        except ValueError:
            pass
        # bool literals
        sl = s.lower()
        if sl == "true":
            return True
        if sl == "false":
            return False
        # JSON arrays/objects
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            try:
                return json.loads(s)
            except (ValueError, json.JSONDecodeError):
                pass
        return raw
    raise ValueError(f"Unknown result type: {type_!r}")


def _aggregate(values: list[Any], aggregate: str) -> Any:
    """Reduce a list of cast values to a single result per ``aggregate``."""
    if aggregate == "count":
        return len(values)
    if aggregate == "list":
        return list(values)
    if not values:
        return None
    if aggregate == "first":
        return values[0]
    if aggregate == "last":
        return values[-1]
    if aggregate == "min":
        return min(values)
    if aggregate == "max":
        return max(values)
    if aggregate == "sum":
        return sum(values)
    if aggregate == "avg":
        return sum(values) / len(values)
    raise ValueError(f"Unknown aggregate: {aggregate!r}")


def _extract_value(match: re.Match[str], spec: ResultSpec) -> str:
    """Extract the captured text from a regex match honoring the spec's group preference.

    Order:
    1. ``spec.group`` (named or positional, when set)
    2. Named ``value`` group, if present
    3. First positional group (group 1), if present
    4. Full match (group 0)
    """
    if spec.group is not None:
        try:
            v = match.group(spec.group)
            if v is not None:
                return v
        except (IndexError, KeyError):
            pass

    groupdict = match.groupdict()
    if "value" in groupdict and groupdict["value"] is not None:
        return groupdict["value"]

    if match.lastindex:
        try:
            v = match.group(1)
            if v is not None:
                return v
        except IndexError:
            pass

    return match.group(0)


def _empty_parsed_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "values": {},
        "metadata": {},
        "matches": {},
        "errors": [],
    }


def parse_results_from_text(
    text: str,
    specs: list[ResultSpec],
) -> dict[str, Any]:
    """
    Pure parser used by ``collect_task_result`` and unit tests.

    Returns a partial payload (without the per-task envelope) shaped like:
        {
            "ok": bool,
            "values": {name: aggregated_value, ...},
            "metadata": {name: {"unit": ..., "type": ..., "aggregate": ...}, ...},
            "matches": {name: [{"line": int, "value": Any, "text": str}, ...], ...},
            "errors": [str, ...],
        }
    """
    payload = _empty_parsed_payload()
    if not specs:
        return payload

    raw_lines = text.splitlines() if text else []
    stripped_lines = [_strip_log_prefix(line) for line in raw_lines]

    # Compile patterns once.
    compiled: dict[str, re.Pattern[str]] = {}
    for spec in specs:
        if not spec.regex:
            payload["errors"].append(f"Result '{spec.name}' has no regex configured")
            if spec.required:
                payload["ok"] = False
            continue
        try:
            compiled[spec.name] = re.compile(spec.regex)
        except re.error as e:
            payload["errors"].append(
                f"Invalid regex for '{spec.name}': {spec.regex!r} ({e})"
            )
            if spec.required:
                payload["ok"] = False

    for spec in specs:
        # Always record metadata so downstream consumers don't have to re-parse YAML.
        payload["metadata"][spec.name] = {
            "unit": spec.unit,
            "type": spec.type,
            "aggregate": spec.aggregate,
        }

        pattern = compiled.get(spec.name)
        if pattern is None:
            continue

        cast_values: list[Any] = []
        spec_matches: list[dict[str, Any]] = []
        had_cast_failure = False

        for line_num, line in enumerate(stripped_lines, start=1):
            for m in pattern.finditer(line):
                raw_value = _extract_value(m, spec)
                try:
                    cast = _cast_value(raw_value, spec.type)
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    had_cast_failure = True
                    payload["errors"].append(
                        f"Failed to cast '{spec.name}' value {raw_value!r} to {spec.type}: {e}"
                    )
                    continue

                cast_values.append(cast)
                spec_matches.append(
                    {
                        "line": line_num,
                        "value": cast,
                        "text": line,
                    }
                )

        if not cast_values:
            if spec.required:
                payload["ok"] = False
                if not had_cast_failure:
                    payload["errors"].append(
                        f"Required result '{spec.name}' did not match any line"
                    )
            continue

        try:
            payload["values"][spec.name] = _aggregate(cast_values, spec.aggregate)
        except (TypeError, ValueError) as e:
            payload["errors"].append(
                f"Aggregation '{spec.aggregate}' failed for '{spec.name}': {e}"
            )
            if spec.required:
                payload["ok"] = False
            continue

        payload["matches"][spec.name] = spec_matches

    return payload


# ---------------------------------------------------------------------------
# File-source normalization
# ---------------------------------------------------------------------------


def _task_status_str(task: Task) -> str:
    status = str(getattr(task.status, "value", task.status))
    # Result collection runs during the orchestrator's FINALIZING phase after
    # the process has already exited successfully. Persist the durable task
    # outcome, not this transient scheduler state.
    if status == "FINALIZING":
        return "COMPLETED"
    return status


def _is_v1_payload(raw: Any) -> bool:
    return (
        isinstance(raw, dict)
        and raw.get("schema_version") == SCHEMA_VERSION_TASK
        and isinstance(raw.get("values"), dict)
    )


def normalize_result_file(path: Path, task: Task) -> dict[str, Any]:
    """
    Load a source JSON file produced by the task and normalize it into the
    canonical ``sflow.result.v1`` payload.
    """
    text = path.read_text()
    raw = json.loads(text)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Result file {path} must contain a JSON object at the top level, "
            f"got {type(raw).__name__}"
        )

    if _is_v1_payload(raw):
        out: dict[str, Any] = dict(raw)
        out.setdefault("task", task.name)
        out.setdefault("status", _task_status_str(task))
        out.setdefault("ok", True)
        out.setdefault("metadata", {})
        out.setdefault("matches", {})
        out.setdefault("errors", [])
        out.setdefault("source", {"type": "file", "path": path.name})
        return out

    return {
        "schema_version": SCHEMA_VERSION_TASK,
        "task": task.name,
        "status": _task_status_str(task),
        "ok": True,
        "source": {"type": "file", "path": path.name},
        "values": raw,
        "metadata": {},
        "matches": {},
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Atomic JSON writer
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via ``tmp -> rename`` to avoid readers seeing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Workflow-level index update
# ---------------------------------------------------------------------------


def _workflow_name_from_task(task: Task) -> str:
    """Best-effort workflow name for the index envelope.

    The Task object does not carry the workflow name directly; we derive it
    from ``SFLOW_WORKFLOW_OUTPUT_DIR`` (basename has the form ``<wf>-<ts>-<id>``
    when sflow generates the run dir; otherwise just the directory name).
    """
    wf_out = task.envs.get("SFLOW_WORKFLOW_OUTPUT_DIR")
    if not wf_out:
        return ""
    return Path(wf_out).name


def _relative_result_file(task: Task) -> str:
    # Always emit forward slashes — results.json is a portable contract for
    # downstream consumers (dashboards, other machines), not an OS-native path.
    task_out = task.envs.get("SFLOW_TASK_OUTPUT_DIR")
    wf_out = task.envs.get("SFLOW_WORKFLOW_OUTPUT_DIR")
    if not task_out:
        return "result.json"
    if not wf_out:
        return (Path(task_out) / "result.json").as_posix()
    try:
        rel = Path(task_out).relative_to(wf_out)
    except ValueError:
        return (Path(task_out) / "result.json").as_posix()
    return (rel / "result.json").as_posix()


def update_workflow_results(task: Task, payload: dict[str, Any]) -> None:
    """
    Update the workflow-level ``results.json`` index with the given task payload.

    Read-modify-write with atomic replace. Concurrent updates from parallel
    tasks are out of scope for the first implementation; the orchestrator
    invokes this on the main asyncio loop after each task completes.
    """
    idx_path = workflow_results_path(task)
    if idx_path is None:
        return

    existing: dict[str, Any]
    if idx_path.exists():
        try:
            loaded = json.loads(idx_path.read_text())
            existing = loaded if isinstance(loaded, dict) else {}
        except (ValueError, OSError) as e:
            _logger.warning(f"Failed to read existing {idx_path}: {e}; rewriting")
            existing = {}
    else:
        existing = {}

    existing_ver = existing.get("schema_version")
    if existing_ver and existing_ver != SCHEMA_VERSION_WORKFLOW:
        # Loud signal so a future v2 reader doesn't lose extra fields silently
        # when an older sflow writes the same run directory.
        _logger.warning(
            f"Workflow results.json at {idx_path} has schema_version "
            f"{existing_ver!r}; rewriting as {SCHEMA_VERSION_WORKFLOW!r} — "
            f"any fields outside the v1 envelope will be dropped."
        )
    if existing_ver != SCHEMA_VERSION_WORKFLOW:
        existing = {
            "schema_version": SCHEMA_VERSION_WORKFLOW,
            "workflow": existing.get("workflow") or _workflow_name_from_task(task),
            "tasks": existing.get("tasks") if isinstance(existing.get("tasks"), dict) else {},
        }
    if not isinstance(existing.get("tasks"), dict):
        existing["tasks"] = {}

    existing["tasks"][task.name] = {
        "status": payload.get("status", _task_status_str(task)),
        "ok": payload.get("ok", True),
        "result_file": _relative_result_file(task),
        "values": payload.get("values", {}),
    }

    _atomic_write_json(idx_path, existing)


async def update_workflow_results_async(task: Task, payload: dict[str, Any]) -> None:
    """Async wrapper that serializes index updates per file path.

    Even though the orchestrator awaits ``collect_task_result`` serially today,
    this module cannot assume the caller is serial. Wrapping the read-modify-
    write under a per-path ``asyncio.Lock`` makes that guarantee explicit so
    a future ``asyncio.gather`` over result collection won't silently drop
    updates to the workflow index.
    """
    idx_path = workflow_results_path(task)
    if idx_path is None:
        return
    lock = _index_lock(idx_path)
    async with lock:
        await asyncio.to_thread(update_workflow_results, task, payload)


# ---------------------------------------------------------------------------
# Top-level entry point used by the orchestrator
# ---------------------------------------------------------------------------


def _failure_envelope(
    task: Task,
    *,
    source: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION_TASK,
        "task": task.name,
        "status": _task_status_str(task),
        "ok": False,
        "source": source or {"type": "log", "path": None},
        "values": {},
        "metadata": {},
        "matches": {},
        "errors": [error] if error else [],
    }


async def collect_task_result(task: Task) -> dict[str, Any]:
    """
    Build the per-task ``result.json`` payload and update the workflow-level index.

    Best-effort by default:
    - Missing log file -> empty values, ``ok: True``.
    - Required match missing -> ``ok: False`` (does not fail the workflow in this release).
    - All file I/O is performed via ``asyncio.to_thread`` and ``tmp -> rename`` for atomicity.
    """
    cfg = task.result_config
    if cfg is None:
        task.result = {}
        return {}

    out_path = task_result_path(task)

    if cfg.file:
        task_out = task.envs.get("SFLOW_TASK_OUTPUT_DIR")
        src_path: Path | None = None
        traversal_error: str | None = None
        if task_out:
            # Reject path traversal in user-supplied `result.file`: a value like
            # "../../etc/passwd" would otherwise let sflow read arbitrary files
            # off the host and echo their contents into result.json.
            normalized = posixpath.normpath(cfg.file)
            if (
                posixpath.isabs(normalized)
                or normalized == ".."
                or normalized.startswith("../")
                or "/../" in normalized
            ):
                traversal_error = (
                    f"result.file '{cfg.file}' must be a relative path within "
                    f"the task output directory; '..' and absolute paths are "
                    f"not allowed"
                )
            else:
                src_path = Path(task_out) / normalized

        if traversal_error is not None:
            payload = _failure_envelope(
                task,
                source={"type": "file", "path": cfg.file},
                error=traversal_error,
            )
        elif src_path is None:
            payload = _failure_envelope(
                task,
                source={"type": "file", "path": cfg.file},
                error="SFLOW_TASK_OUTPUT_DIR is not set",
            )
        else:
            def _read_and_normalize() -> dict[str, Any]:
                return normalize_result_file(src_path, task)  # type: ignore[arg-type]

            try:
                payload = await asyncio.to_thread(_read_and_normalize)
            except FileNotFoundError:
                payload = _failure_envelope(
                    task,
                    source={"type": "file", "path": cfg.file},
                    error=f"Source file not found: {src_path}",
                )
            except Exception as e:
                _logger.warning(
                    f"Failed to load result source file {src_path} for task '{task.name}': {e}"
                )
                payload = _failure_envelope(
                    task,
                    source={"type": "file", "path": cfg.file},
                    error=f"Failed to load source file: {e}",
                )
    else:
        log_path = _task_log_path(task)
        specs = list(cfg.specs)

        # Read AND parse on a worker thread. The regex scan in
        # parse_results_from_text() is CPU-bound and, for multi-GB task logs,
        # would otherwise run on the orchestrator event loop and stall probes /
        # heartbeats of other running tasks. Doing the scan off-loop means only
        # the small parsed payload crosses back to the loop, not the whole log.
        #
        # Note: this still loads the whole log into memory. If multi-GB logs
        # become common, stream line-by-line via re.finditer over a TextIOWrapper.
        def _read_and_parse() -> dict[str, Any]:
            text = ""
            if log_path is not None:
                try:
                    text = log_path.read_text(errors="ignore")
                except FileNotFoundError:
                    text = ""
                except Exception as e:
                    _logger.warning(
                        f"Failed to read log for results parsing: {log_path}: {e}"
                    )
                    text = ""
            return parse_results_from_text(text, specs)

        try:
            parsed = await asyncio.to_thread(_read_and_parse)
        except Exception as e:
            _logger.warning(f"Failed to parse results for task '{task.name}': {e}")
            parsed = _empty_parsed_payload()
            parsed["errors"].append(str(e))
            parsed["ok"] = False

        payload = {
            "schema_version": SCHEMA_VERSION_TASK,
            "task": task.name,
            "status": _task_status_str(task),
            "ok": parsed.get("ok", True),
            "source": {
                "type": cfg.source or "log",
                "path": (log_path.name if log_path is not None else None),
            },
            "values": parsed.get("values", {}),
            "metadata": parsed.get("metadata", {}),
            "matches": parsed.get("matches", {}),
            "errors": parsed.get("errors", []),
        }

    task.result = payload

    if out_path is not None:
        try:
            await asyncio.to_thread(_atomic_write_json, out_path, payload)
        except Exception as e:
            _logger.warning(f"Failed to write result.json for task '{task.name}': {e}")

    try:
        await update_workflow_results_async(task, payload)
    except Exception as e:
        _logger.warning(
            f"Failed to update workflow results.json for task '{task.name}': {e}"
        )

    return payload
