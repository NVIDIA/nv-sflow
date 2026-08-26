# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from fnmatch import fnmatch

from sflow.logging import get_logger
from sflow.resolution import ExpressionResolver

from .schema import SflowConfig

_logger = get_logger(__name__)


def strip_missable_tasks(
    config_data: Dict[str, Any],
    missable_patterns: List[str],
) -> list[str]:
    """Remove references to missable tasks that are not present in the config.

    For modular workflows where some task files may not be included,
    this strips missing tasks from ``depends_on`` lists and probe
    ``logger`` references so that Pydantic validation passes.

    Only tasks that match a missable pattern AND are absent from the
    config are removed.  Tasks that exist are never touched.

    Supports exact names and glob patterns (e.g. ``prefill_*``).

    Returns a list of human-readable descriptions of what was removed.
    """
    wf = config_data.get("workflow")
    if not isinstance(wf, dict):
        return []
    tasks = wf.get("tasks")
    if not tasks or not isinstance(tasks, list):
        return []

    present_names = set()
    for t in tasks:
        if isinstance(t, dict) and "name" in t:
            present_names.add(t["name"])

    def _is_missable_and_absent(name: str) -> bool:
        if name in present_names:
            return False
        return any(fnmatch(name, pat) for pat in missable_patterns)

    stripped: list[str] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue

        if t.get("depends_on"):
            original = list(t["depends_on"])
            t["depends_on"] = [d for d in original if not _is_missable_and_absent(d)]
            removed = set(original) - set(t["depends_on"])
            if removed:
                stripped.extend(f"{t['name']}.depends_on: {r}" for r in removed)
            if not t["depends_on"]:
                del t["depends_on"]

        probes = t.get("probes")
        if isinstance(probes, dict):
            for probe_type in ("readiness", "failure"):
                probe = probes.get(probe_type)
                if isinstance(probe, dict):
                    lw = probe.get("log_watch")
                    if isinstance(lw, dict) and "logger" in lw:
                        if _is_missable_and_absent(lw["logger"]):
                            stripped.append(
                                f"{t['name']}.probes.{probe_type}.log_watch.logger: {lw['logger']}"
                            )
                            del lw["logger"]

    return stripped


def _section_to_merge_dict(section: Any) -> dict:
    """Normalize a named YAML section to ``{name: config_dict}`` for merging.

    Handles both dict format (name-as-key) and list format (name-as-field).
    """
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if isinstance(section, list):
        result: dict = {}
        for item in section:
            if isinstance(item, dict):
                name = item.get("name")
                if name is not None:
                    entry = {k: v for k, v in item.items() if k != "name"}
                    result[name] = entry
        return result
    return {}


def _tasks_to_list(tasks: Any) -> list:
    """Normalize workflow tasks (list or dict) to a plain list of dicts."""
    if tasks is None:
        return []
    if isinstance(tasks, list):
        return list(tasks)
    if isinstance(tasks, dict):
        return [
            {**v, "name": k} if isinstance(v, dict) else {"name": k}
            for k, v in tasks.items()
        ]
    return []


def _render_plain_scalar(value: Any) -> str:
    """Render simple YAML scalar values back to their plain text form."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    return str(value)


def _normalize_script_plain_mappings(config_data: Dict[str, Any]) -> None:
    """Recover script commands that YAML parsed as mappings due to ``: ``.

    YAML treats an unquoted list item like ``- echo "x: y"`` as a mapping.
    ``script`` only accepts command strings, so a one-entry mapping there is
    interpreted as the original shell line rather than a structured object.
    """
    workflow = config_data.get("workflow")
    if not isinstance(workflow, dict):
        return

    tasks = workflow.get("tasks")
    if isinstance(tasks, list):
        task_items = tasks
    elif isinstance(tasks, dict):
        task_items = tasks.values()
    else:
        return

    for task in task_items:
        if not isinstance(task, dict):
            continue
        script = task.get("script")
        if not isinstance(script, list):
            continue

        normalized: list[Any] = []
        changed = False
        for item in script:
            if isinstance(item, dict) and len(item) == 1:
                key, value = next(iter(item.items()))
                if not isinstance(value, (dict, list)):
                    rendered_value = _render_plain_scalar(value)
                    suffix = f" {rendered_value}" if rendered_value else ""
                    normalized.append(f"{key}:{suffix}")
                    changed = True
                    continue
            normalized.append(item)

        if changed:
            task["script"] = normalized


# Top-level and workflow-level sections that are collections of *named* entries.
# These merge by ``name`` (scattered definitions of the same name are deep-merged,
# new names are appended). Everything else is a plain dict/leaf handled generically.
_NAMED_SECTION_KEYS = ("variables", "artifacts", "backends", "operators", "storage")


def _record_origins(
    value: Any, path: str, label: str, origin: Dict[str, str]
) -> None:
    """Record the originating file label for every leaf under ``value``.

    Used so a later override can report where the previous value came from.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            _record_origins(v, f"{path}.{k}" if path else str(k), label, origin)
    else:
        origin[path] = label


def _deep_merge(
    base: Any,
    incoming: Any,
    *,
    path: str,
    label: str,
    origin: Dict[str, str],
    warns: List[str],
) -> Any:
    """Recursively merge ``incoming`` onto ``base`` (non-mutating).

    Dicts are merged key-by-key. Anything else (scalars, lists, or a
    dict-vs-scalar type mismatch) is a leaf: the incoming value wins, and a
    warning is recorded when it differs from the previous value. This is the
    "conflicts only on the root-down leaf value" rule.
    """
    if isinstance(base, dict) and isinstance(incoming, dict):
        result = dict(base)
        for k, v in incoming.items():
            child = f"{path}.{k}" if path else str(k)
            if k in result:
                result[k] = _deep_merge(
                    result[k], v, path=child, label=label, origin=origin, warns=warns
                )
            else:
                result[k] = v
                _record_origins(v, child, label, origin)
        return result

    if base != incoming:
        prev = origin.get(path)
        prev_note = f" (previously from {prev})" if prev else ""
        warns.append(
            f"{path}: overridden by {label}{prev_note}, "
            f"value: {base!r} -> {incoming!r}"
        )
    origin[path] = label
    return incoming


def _merge_named_section(
    base_section: Any,
    incoming_section: Any,
    *,
    section_path: str,
    label: str,
    origin: Dict[str, str],
    warns: List[str],
) -> Dict[str, Any]:
    """Merge a named section (variables/artifacts/backends/operators/storage).

    Returns a name-keyed dict; entries with the same name are deep-merged so a
    definition can be scattered across files (e.g. the operator's ``type`` in
    one file, an override in another).
    """
    merged = _section_to_merge_dict(base_section)
    incoming = _section_to_merge_dict(incoming_section)
    for name, entry in incoming.items():
        if name in merged:
            merged[name] = _deep_merge(
                merged[name],
                entry,
                path=f"{section_path}.{name}",
                label=label,
                origin=origin,
                warns=warns,
            )
        else:
            merged[name] = entry
            _record_origins(entry, f"{section_path}.{name}", label, origin)
    return merged


def _merge_tasks(
    base_tasks: Any,
    incoming_tasks: Any,
    *,
    label: str,
    origin: Dict[str, str],
    warns: List[str],
) -> List[Dict[str, Any]]:
    """Merge ``workflow.tasks`` by name, preserving first-seen order.

    Same-name tasks are deep-merged (scattered task definitions); new names are
    appended. Tasks with no ``name`` cannot be merged by name, so they are kept
    (appended after the named tasks) rather than silently dropped -- schema
    validation then raises a clear "name required" error instead of the task
    vanishing. Returns a list of task dicts.
    """
    merged: Dict[str, Any] = {}
    order: List[str] = []
    unnamed: List[Any] = []
    for t in _tasks_to_list(base_tasks):
        name = t.get("name") if isinstance(t, dict) else None
        if name is None:
            unnamed.append(t)
            continue
        merged[name] = t
        order.append(name)
    for t in _tasks_to_list(incoming_tasks):
        name = t.get("name") if isinstance(t, dict) else None
        if name is None:
            unnamed.append(t)
            continue
        if name in merged:
            merged[name] = _deep_merge(
                merged[name],
                t,
                path=f"workflow.tasks.{name}",
                label=label,
                origin=origin,
                warns=warns,
            )
        else:
            merged[name] = t
            order.append(name)
            _record_origins(t, f"workflow.tasks.{name}", label, origin)
    return [merged[n] for n in order] + unnamed


def _merge_workflow(
    base_wf: Dict[str, Any],
    inc_wf: Dict[str, Any],
    *,
    label: str,
    origin: Dict[str, str],
    warns: List[str],
) -> Dict[str, Any]:
    """Merge a ``workflow`` section: name (last non-null wins), variables and
    tasks (by name), and every other key (timeout/upload_all/monitor/...) via
    generic deep merge."""
    result = dict(base_wf)

    if inc_wf.get("name") is not None:
        prev = result.get("name")
        if prev is not None and prev != inc_wf["name"]:
            warns.append(
                f"workflow.name: overridden by {label} (previously '{prev}'), "
                f"value: {prev!r} -> {inc_wf['name']!r}"
            )
        result["name"] = inc_wf["name"]

    if inc_wf.get("variables") is not None:
        result["variables"] = _merge_named_section(
            result.get("variables"),
            inc_wf["variables"],
            section_path="workflow.variables",
            label=label,
            origin=origin,
            warns=warns,
        )

    if inc_wf.get("tasks") is not None:
        result["tasks"] = _merge_tasks(
            result.get("tasks"),
            inc_wf["tasks"],
            label=label,
            origin=origin,
            warns=warns,
        )

    for k, v in inc_wf.items():
        if k in ("name", "variables", "tasks") or v is None:
            continue
        child = f"workflow.{k}"
        if k in result:
            result[k] = _deep_merge(
                result[k], v, path=child, label=label, origin=origin, warns=warns
            )
        else:
            result[k] = v
            _record_origins(v, child, label, origin)

    return result


def merge_config_dicts(
    config_dicts: List[Dict[str, Any]],
    source_labels: Optional[List[str]] = None,
    override_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Merge multiple raw YAML config dicts into a single config dict.

    Merge strategy (recursive deep merge keyed on the distinguishing ``name``):
      - version: must be consistent across files (hard error on conflict)
      - variables / artifacts / backends / operators / storage: merge by name;
        same-name entries are deep-merged so a definition can be scattered
        across files
      - workflow.name: last non-null wins (no restriction; warning on change)
      - workflow.variables: merge by name
      - workflow.tasks: merge by name (same-name tasks deep-merged), preserving
        first-seen order
      - workflow.timeout / upload_all / monitor and any other key: deep-merged
      - leaf scalars and value-lists (script, depends_on, ...): last file wins,
        with an override warning

    Args:
        config_dicts: Raw YAML dicts to merge.
        source_labels: Human-readable labels for each dict (e.g. file paths).
        override_warnings: If provided, override messages are appended here.

    Raises:
        ValueError: On version conflicts or incomplete merged result.
    """
    if not config_dicts:
        raise ValueError("No configuration data provided for merging")

    if len(config_dicts) == 1:
        return config_dicts[0]

    labels = source_labels or [f"config[{i}]" for i in range(len(config_dicts))]
    merged: Dict[str, Any] = {}

    # Maps a dotted leaf path (e.g. "variables.X.value") to the file label that
    # last set it, so override warnings can report the previous source.
    origin: Dict[str, str] = {}

    warns = override_warnings if override_warnings is not None else []

    for idx, cfg in enumerate(config_dicts):
        if not isinstance(cfg, dict):
            continue
        label = labels[idx]

        if cfg.get("version") is not None:
            if "version" in merged and merged["version"] != cfg["version"]:
                raise ValueError(
                    f"Version conflict: '{merged['version']}' vs "
                    f"'{cfg['version']}' (from {label})"
                )
            merged["version"] = cfg["version"]

        for key in _NAMED_SECTION_KEYS:
            if cfg.get(key) is not None:
                merged[key] = _merge_named_section(
                    merged.get(key),
                    cfg[key],
                    section_path=key,
                    label=label,
                    origin=origin,
                    warns=warns,
                )

        if cfg.get("workflow") is not None:
            merged["workflow"] = _merge_workflow(
                merged.get("workflow") or {},
                cfg["workflow"],
                label=label,
                origin=origin,
                warns=warns,
            )

    errors: list[str] = []
    if "version" not in merged:
        errors.append("No 'version' field found in any input file")
    wf = merged.get("workflow")
    if not wf:
        errors.append("No 'workflow' section found in any input file")
    else:
        if not wf.get("name"):
            errors.append("No workflow name found in any input file")
        if not wf.get("tasks"):
            errors.append("No tasks found in any input file")
    if errors:
        file_list = ", ".join(labels)
        details = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(
            f"Merged configuration from [{file_list}] is incomplete:\n{details}"
        )

    return merged


def _extract_file_contributions(
    config_dicts: List[Dict[str, Any]], paths: List[Path]
) -> List[Dict[str, Any]]:
    """Summarise what each input file contributes to the merged config."""
    result: List[Dict[str, Any]] = []
    for cfg, path in zip(config_dicts, paths):
        contrib: Dict[str, Any] = {"path": path, "sections": []}
        for key in _NAMED_SECTION_KEYS:
            items = cfg.get(key)
            if items:
                names = list(_section_to_merge_dict(items).keys())
                if names:
                    contrib["sections"].append((key, names))
        wf = cfg.get("workflow")
        if wf:
            if wf.get("name"):
                contrib["sections"].append(("workflow.name", [wf["name"]]))
            if wf.get("timeout"):
                contrib["sections"].append(("workflow.timeout", [str(wf["timeout"])]))
            wf_vars = wf.get("variables")
            if wf_vars:
                names = list(_section_to_merge_dict(wf_vars).keys())
                if names:
                    contrib["sections"].append(("workflow.variables", names))
            wf_tasks = wf.get("tasks")
            if wf_tasks:
                task_names = [
                    t["name"] for t in _tasks_to_list(wf_tasks) if "name" in t
                ]
                if task_names:
                    contrib["sections"].append(("workflow.tasks", task_names))
        result.append(contrib)
    return result


class ConfigLoader:
    """
    Parses YAML configuration, validates against the config schema, and applies CLI overrides.

    The loading process:
    1. Parse YAML to dict
    2. Apply CLI overrides (--set, --artifact)
    """

    def __init__(self):
        self._resolver = ExpressionResolver()
        self.config: Optional[SflowConfig] = None
        self.missable_stripped: list[str] = []
        self.source_files: List[Path] = []
        self.file_contributions: List[Dict[str, Any]] = []

    def _warn_unenforced_timeout(self, config: SflowConfig) -> None:
        """Say out loud that `timeout:` does nothing.

        The field is accepted and deep-merged but never read, so a recipe that
        declares one looks bounded and is not -- and a hung workflow then runs
        until something external kills it. Warning is the honest middle: removing
        the field would reject every config that sets it (the models forbid extra
        keys), and enforcing it would start killing runs that have never been
        held to it.
        """
        where = []
        if config.workflow.timeout is not None:
            where.append("workflow")
        where += [t.name for t in config.workflow.tasks if t.timeout is not None]
        if where:
            _logger.warning(
                f"`timeout:` is set on {', '.join(where)} but sflow does not enforce it; "
                "these run unbounded. Use the backend's own limit (e.g. Slurm --time)."
            )

    def load_config(
        self,
        path: Path,
        variable_overrides: Optional[List[str]] = None,
        artifact_overrides: Optional[List[str]] = None,
        missable_tasks: Optional[List[str]] = None,
    ) -> SflowConfig:
        """
        Loads the configuration from a file, applying CLI overrides.

        Args:
            path (Path): Path to the configuration file.
            variable_overrides (Optional[List[str]]): List of variable overrides (key=value).
            artifact_overrides (Optional[List[str]]): List of artifact overrides (key=uri).

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If parsing fails or validation errors occur.

        Returns:
            SflowConfig: The validated configuration.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

        if config_data is None:
            raise ValueError(f"Configuration file is empty: {path}")

        _normalize_script_plain_mappings(config_data)

        syntax = self._resolver.validate_syntax(config_data, location=str(path))
        if not syntax.valid:
            details = "\n".join(str(e) for e in syntax.errors)
            raise ValueError(
                f"Configuration expression syntax validation failed:\n{details}"
            )

        # Apply CLI overrides to the raw dict
        if variable_overrides:
            self._apply_variable_overrides(config_data, variable_overrides)

        if artifact_overrides:
            self._apply_artifact_overrides(config_data, artifact_overrides)

        if missable_tasks:
            self.missable_stripped = strip_missable_tasks(config_data, missable_tasks)

        try:
            config = SflowConfig.model_validate(config_data)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed:\n{e}")

        self._warn_unenforced_timeout(config)
        self.config = config
        return config

    def load_configs(
        self,
        paths: List[Path],
        variable_overrides: Optional[List[str]] = None,
        artifact_overrides: Optional[List[str]] = None,
        missable_tasks: Optional[List[str]] = None,
    ) -> SflowConfig:
        """Load and merge multiple YAML config files into a single ``SflowConfig``.

        For a single path this delegates to :meth:`load_config`.  For multiple
        paths the raw YAML dicts are loaded, validated for expression syntax,
        merged via :func:`merge_config_dicts`, and finally validated by Pydantic.
        """
        if not paths:
            raise ValueError("No configuration file paths provided")

        self.source_files = list(paths)

        if len(paths) == 1:
            return self.load_config(
                paths[0], variable_overrides, artifact_overrides, missable_tasks
            )

        config_dicts: List[Dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration ({path}): {e}")
            if data is None:
                raise ValueError(f"Configuration file is empty: {path}")

            _normalize_script_plain_mappings(data)

            syntax = self._resolver.validate_syntax(data, location=str(path))
            if not syntax.valid:
                details = "\n".join(str(e) for e in syntax.errors)
                raise ValueError(
                    f"Expression syntax validation failed in {path}:\n{details}"
                )
            config_dicts.append(data)

        self.file_contributions = _extract_file_contributions(config_dicts, paths)

        override_warnings: List[str] = []
        merged = merge_config_dicts(
            config_dicts,
            source_labels=[str(p) for p in paths],
            override_warnings=override_warnings,
        )
        if override_warnings:
            _logger.warning("Multi-file merge overrides detected:")
            for w in override_warnings:
                _logger.warning(f"  ⚠ {w}")

        if variable_overrides:
            self._apply_variable_overrides(merged, variable_overrides)
        if artifact_overrides:
            self._apply_artifact_overrides(merged, artifact_overrides)

        if missable_tasks:
            self.missable_stripped = strip_missable_tasks(merged, missable_tasks)

        try:
            config = SflowConfig.model_validate(merged)
        except ValidationError as e:
            raise ValueError(f"Merged configuration validation failed:\n{e}")

        self._warn_unenforced_timeout(config)
        self.config = config
        return config

    def _apply_variable_overrides(
        self, config_data: Dict[str, Any], overrides: List[str]
    ):
        """
        Applies CLI variable overrides to the configuration dictionary.
        Format: KEY=VALUE
        """
        if "variables" not in config_data or config_data["variables"] is None:
            # Initialize as dict to make lookups easier during override application
            # The schema validation will normalize it to list later if needed
            config_data["variables"] = {}

        variables = config_data["variables"]

        # If variables is a list, convert to dict temporarily for easier override handling
        is_list = isinstance(variables, list)
        if is_list:
            var_map = {v.get("name"): v for v in variables}
        else:
            var_map = variables

        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid variable override format: '{override}'. Expected KEY=VALUE."
                )
            key, value = override.split("=", 1)

            # Helper to infer type (int, float, bool, list, or string)
            typed_value = self._infer_type(value)

            # Update or create the variable entry
            if key in var_map:
                entry = var_map[key]
                # If it's a dict (VariableConfig structure), update appropriately
                if isinstance(entry, dict):
                    if isinstance(typed_value, list):
                        # If the value is a list, treat it as a domain for replica sweeps
                        # Set domain to the list and value to the first element
                        entry["domain"] = typed_value
                        if typed_value:
                            entry["value"] = typed_value[0]
                    else:
                        entry["value"] = typed_value
                else:
                    # Fallback or scalar case (if supported in future)
                    # For now, if we converted from list, entry is a dict
                    # If it was a dict, entry might be just the value? No, assumed dict structure.
                    var_map[key] = {"value": typed_value}
            else:
                # Raise error if variable not found in config
                raise ValueError(
                    f"Variable '{key}' specified in overrides is not defined in the configuration."
                )

        # If we converted from list, we need to update the original list
        if is_list:
            # We updated the dict objects in var_map, which are references to items in the list?
            # Yes, if list contains dicts, var_map contains refs to those dicts.
            # But if we assigned new dict to var_map[key], the list is not updated.
            # So we must reconstruct the list or update carefully.
            # Simple approach: Since we only modified mutable dicts in place (entry["value"] = ...),
            # the original list items are updated.
            # EXCEPT if we did var_map[key] = ... (assignment), which we don't do for existing keys above.
            pass

    def _apply_artifact_overrides(
        self, config_data: Dict[str, Any], overrides: List[str]
    ):
        """
        Applies CLI artifact overrides to the configuration dictionary.
        Format: NAME=URI

        When changing from 'file://' to any other URI scheme, the 'content' field is
        automatically removed to prevent validation errors (inline content is only valid for 'file://').
        """
        if "artifacts" not in config_data or config_data["artifacts"] is None:
            config_data["artifacts"] = []

        artifacts = config_data["artifacts"]

        # Handle both list and dict formats
        is_list = isinstance(artifacts, list)
        if is_list:
            artifact_map = {a.get("name"): a for a in artifacts}
        else:
            # Dict format: key is name
            artifact_map = artifacts

        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid artifact override format: '{override}'. Expected NAME=URI."
                )
            name, uri = override.split("=", 1)

            if name in artifact_map:
                artifact_entry = artifact_map[name]
                old_uri = artifact_entry.get("uri", "")

                # Update the URI
                artifact_entry["uri"] = uri

                # Handle type changes: if changing from file:// to any non-file:// scheme, remove content
                # because inline content is only valid for 'file://' URIs
                old_is_file = old_uri.startswith("file://")
                new_is_non_file = not uri.startswith("file://")

                if old_is_file and new_is_non_file:
                    # Remove content field when switching from file:// to fs://
                    if "content" in artifact_entry:
                        del artifact_entry["content"]
            else:
                # Raise error if artifact not found in config
                raise ValueError(
                    f"Artifact '{name}' specified in overrides is not defined in the configuration."
                )

    def _infer_type(self, value: str) -> Any:
        """
        Infers the type of the value string (int, float, bool, list, or str).

        Supports JSON-like list syntax: [1, 2, 3] or ["a", "b", "c"]
        """
        import json

        # Try JSON parsing first (handles lists, nested structures)
        # This catches: [1,2,3], ["a","b"], {"key": "value"}, etc.
        value_stripped = value.strip()
        if value_stripped.startswith("[") or value_stripped.startswith("{"):
            try:
                return json.loads(value_stripped)
            except json.JSONDecodeError:
                pass  # Fall through to other type inference

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try bool
        lower_val = value.lower()
        if lower_val in ("true", "yes", "on"):
            return True
        if lower_val in ("false", "no", "off"):
            return False

        # Default string
        return value
