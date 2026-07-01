# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expression, variable, and artifact resolution services for sflow."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from jinja2 import StrictUndefined, TemplateSyntaxError, UndefinedError
from jinja2 import meta as jinja2_meta
from jinja2 import nodes as jinja2_nodes
from jinja2.sandbox import SandboxedEnvironment

from sflow.config.schema import SflowConfig
from sflow.core.artifact import Artifact
from sflow.core.artifact_registry import (
    ensure_builtin_artifacts_registered,
    get_artifact_resolver_for_uri,
)
from sflow.core.state import SflowState
from sflow.core.variable import (
    Variable,
    VariableType,
    VariableValue,
    build_variables_ctx,
    build_variables_ctx_from_raw,
    extract_domains_from_raw_config,
)

__all__ = [
    "ExpressionResolver",
    "ExpressionValidationError",
    "ValidationResult",
    "artifacts_ctx",
    "cast_variable_value",
    "enrich_error_with_location",
    "find_lines_in_files",
    "maybe_int",
    "resolve_and_update_variables",
    "resolve_artifacts",
    "resolve_deferred_global_variables",
    "resolve_expressions",
    "resolve_global_variables",
    "resolve_variables_inline",
    "resolve_workflow_variables",
    "validate_no_deferred_variable_references",
]


@dataclass
class ExpressionValidationError:
    """Represents a validation error for an expression."""

    expression: str
    error: str
    location: Optional[str] = None

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"Expression error{loc}: {self.error} in '{self.expression}'"


@dataclass
class ValidationResult:
    """Result of expression validation."""

    valid: bool
    errors: List[ExpressionValidationError] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


_LOCATION_HINT_RE = re.compile(r"\(line \d+ in ")
_QUOTED_STRING_RE = re.compile(r"'([^']+)'")


def find_lines_in_files(search_text: str, source_files: List[Path]) -> List[str]:
    """Search *source_files* for *search_text* and return source hints."""
    search_text = search_text.strip()
    if not search_text or not source_files:
        return []
    hits: List[str] = []
    for fpath in source_files:
        try:
            lines = fpath.read_text().splitlines()
            for i, line in enumerate(lines, start=1):
                if search_text in line:
                    hits.append(f"line {i} in {fpath.name}")
        except OSError:
            continue
    return hits


def enrich_error_with_location(error_msg: str, source_files: List[Path]) -> str:
    """Append YAML source location hints to *error_msg* if not already present."""
    if not source_files or _LOCATION_HINT_RE.search(error_msg):
        return error_msg

    candidates: List[str] = []
    for match in re.finditer(r"\$\{\{(.+?)\}\}", error_msg):
        candidates.append("${{" + match.group(1) + "}}")

    quoted = []
    for match in _QUOTED_STRING_RE.finditer(error_msg):
        val = match.group(1)
        if len(val) >= 3 and not val.startswith("${{"):
            quoted.append(val)
    quoted.sort(key=len, reverse=True)
    candidates.extend(quoted)

    best_hits: List[str] | None = None
    for candidate in candidates:
        hits = find_lines_in_files(candidate, source_files)
        if hits and (best_hits is None or len(hits) < len(best_hits)):
            best_hits = hits

    if best_hits:
        return error_msg + "\n  Source: " + ", ".join(best_hits)
    return error_msg


class ExpressionResolver:
    """Resolve and validate sflow ``${{ ... }}`` expressions."""

    VARIABLE_PATTERN = re.compile(r"\$\{\{(.+?)\}\}")

    def __init__(self):
        self._env = SandboxedEnvironment(
            undefined=StrictUndefined,
            autoescape=False,
            variable_start_string="${{",
            variable_end_string="}}",
        )
        self.source_files: List[Path] = []

    def has_expression(self, value: Any) -> bool:
        """Check if a value contains any ``${{ }}`` expressions."""
        if isinstance(value, str):
            return "${{" in value
        if isinstance(value, list):
            return any(self.has_expression(item) for item in value)
        if isinstance(value, dict):
            return any(self.has_expression(v) for v in value.values())
        return False

    def validate_syntax(
        self, value: Any, location: Optional[str] = None
    ) -> ValidationResult:
        """Validate expression syntax without resolving values."""
        errors: List[ExpressionValidationError] = []
        self._validate_recursive(value, location, errors)
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _validate_recursive(
        self,
        value: Any,
        location: Optional[str],
        errors: List[ExpressionValidationError],
    ) -> None:
        if isinstance(value, str):
            if "${{" in value:
                try:
                    self._env.parse(value)
                except TemplateSyntaxError as exc:
                    errors.append(
                        ExpressionValidationError(
                            expression=value, error=str(exc), location=location
                        )
                    )
        elif isinstance(value, list):
            for i, item in enumerate(value):
                item_loc = f"{location}[{i}]" if location else f"[{i}]"
                self._validate_recursive(item, item_loc, errors)
        elif isinstance(value, dict):
            for key, val in value.items():
                item_loc = f"{location}.{key}" if location else key
                self._validate_recursive(val, item_loc, errors)

    def extract_references(self, value: Any) -> Set[str]:
        """Extract undeclared variable names referenced by expressions."""
        references: Set[str] = set()
        self._extract_recursive(value, references)
        return references

    def _extract_recursive(self, value: Any, references: Set[str]) -> None:
        if isinstance(value, str):
            if "${{" in value:
                try:
                    ast = self._env.parse(value)
                    references.update(jinja2_meta.find_undeclared_variables(ast))
                except TemplateSyntaxError:
                    pass
        elif isinstance(value, list):
            for item in value:
                self._extract_recursive(item, references)
        elif isinstance(value, dict):
            for val in value.values():
                self._extract_recursive(val, references)

    def references_attribute(self, value: Any, root: str, attr: str) -> bool:
        """Return True if a ``${{ }}`` expression in ``value`` reads ``root.attr``.

        Detection is AST-based (via the same Jinja2 environment used to resolve
        expressions), so it is robust where naive string matching is not: it
        tolerates any internal whitespace (``${{task.name}}``), finds the
        reference inside a larger sub-expression (``${{ 'p_' + task.name }}``),
        accepts item access (``${{ task['name'] }}``), and — crucially — ignores
        matches inside string literals (``${{ "task.name" }}``). Used e.g. to
        decide whether a user already placed ``task.name`` in an upload ``to:``.
        """
        if not isinstance(value, str) or "${{" not in value:
            return False
        try:
            ast = self._env.parse(value)
        except TemplateSyntaxError:
            return False
        for node in ast.find_all(jinja2_nodes.Getattr):
            if (
                node.attr == attr
                and isinstance(node.node, jinja2_nodes.Name)
                and node.node.name == root
            ):
                return True
        for node in ast.find_all(jinja2_nodes.Getitem):
            if (
                isinstance(node.node, jinja2_nodes.Name)
                and node.node.name == root
                and isinstance(node.arg, jinja2_nodes.Const)
                and node.arg.value == attr
            ):
                return True
        return False

    def resolve(self, value: Any, context: Dict[str, Any]) -> Any:
        """Resolve a value using the given context."""
        if isinstance(value, str):
            if "${{" in value:
                return self._resolve_string(value, context)
            return value
        if isinstance(value, list):
            return [self.resolve(item, context) for item in value]
        if isinstance(value, dict):
            return {key: self.resolve(val, context) for key, val in value.items()}
        return value

    def _find_expression_in_sources(self, expression: str) -> str:
        """Search source YAML files for an expression and return a location hint."""
        hits = find_lines_in_files(expression, self.source_files)
        if hits:
            return " (" + ", ".join(hits) + ")"
        return ""

    def _resolve_string(self, value: str, context: Dict[str, Any]) -> Any:
        try:
            template = self._env.from_string(value)
            return template.render(**context)
        except UndefinedError as exc:
            failing = self._pinpoint_failing_expression(value, context)
            location = self._find_expression_in_sources(failing)
            raise ValueError(
                f"Undefined variable in expression {failing}{location}: {exc}"
            ) from exc
        except TemplateSyntaxError as exc:
            location = self._find_expression_in_sources(value)
            raise ValueError(
                f"Invalid expression syntax in '{value}'{location}: {exc}"
            ) from exc
        except Exception as exc:
            failing = self._pinpoint_failing_expression(value, context)
            location = self._find_expression_in_sources(failing)
            raise ValueError(
                f"Error evaluating expression {failing}{location}: {exc}"
            ) from exc

    def _pinpoint_failing_expression(self, value: str, context: Dict[str, Any]) -> str:
        """Identify which specific ``${{ }}`` expression fails to resolve."""
        matches = self.VARIABLE_PATTERN.findall(value)
        if not matches:
            return repr(value)

        failing: list[str] = []
        for expr_body in matches:
            test_str = "${{ " + expr_body.strip() + " }}"
            try:
                tpl = self._env.from_string(test_str)
                tpl.render(**context)
            except Exception:
                failing.append("${{ " + expr_body.strip() + " }}")

        if failing:
            return ", ".join(failing)
        return ", ".join("${{ " + match.strip() + " }}" for match in matches)

    def resolve_with_partial_context(
        self, value: Any, context: Dict[str, Any], ignore_undefined: bool = False
    ) -> Any:
        """Resolve expressions, optionally ignoring undefined variables."""
        if not ignore_undefined:
            return self.resolve(value, context)

        permissive_env = SandboxedEnvironment(
            autoescape=False,
            variable_start_string="${{",
            variable_end_string="}}",
        )

        def resolve_partial(v: Any) -> Any:
            if isinstance(v, str):
                if "${{" not in v:
                    return v
                try:
                    template = permissive_env.from_string(v)
                    return template.render(**context)
                except UndefinedError:
                    return v
            if isinstance(v, list):
                return [resolve_partial(item) for item in v]
            if isinstance(v, dict):
                return {key: resolve_partial(val) for key, val in v.items()}
            return v

        return resolve_partial(value)


_EXPR_RE = re.compile(r"\$\{\{(.+?)\}\}")
_SHELL_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VARIABLE_ITEM_RE = re.compile(
    r"variables\s*\[\s*(['\"])([A-Za-z_][A-Za-z0-9_]*)\1\s*\]"
)
_VARIABLE_GET_RE = re.compile(
    r"variables\.get\(\s*(['\"])([A-Za-z_][A-Za-z0-9_]*)\1"
)

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
_DEFERRED_GLOBAL_CONTEXTS = frozenset({"artifacts", "backends"})
_WORKFLOW_ONLY_CONTEXTS = frozenset({"task", "workflow"})
_RUNTIME_VARIABLE_CONTEXTS = _DEFERRED_GLOBAL_CONTEXTS | _WORKFLOW_ONLY_CONTEXTS


def _runtime_refs_for_variable(
    variable: Variable,
    resolver: ExpressionResolver,
) -> set[str]:
    if not resolver.has_expression(variable.value):
        return set()
    return resolver.extract_references(variable.value) & _RUNTIME_VARIABLE_CONTEXTS


def _deferred_global_variable_names(
    variables: dict[str, Variable],
    resolver: ExpressionResolver,
) -> set[str]:
    contexts_by_name = _runtime_contexts_by_variable(variables, resolver)
    return {
        name
        for name, contexts in contexts_by_name.items()
        if contexts and contexts <= _DEFERRED_GLOBAL_CONTEXTS
    }


def _runtime_contexts_by_variable(
    variables: dict[str, Variable],
    resolver: ExpressionResolver,
) -> dict[str, set[str]]:
    variable_names = set(variables)
    contexts_by_name = {
        name: _runtime_refs_for_variable(variable, resolver)
        for name, variable in variables.items()
    }

    changed = True
    while changed:
        changed = False
        for name, variable in variables.items():
            deps = _extract_variable_reference_names(variable.value, variable_names)
            merged = set(contexts_by_name[name])
            for dep in deps:
                merged.update(contexts_by_name.get(dep, set()))
            if merged != contexts_by_name[name]:
                contexts_by_name[name] = merged
                changed = True

    return contexts_by_name


def _extract_variable_reference_names_from_expr_body(
    body: str,
    variable_names: set[str],
) -> set[str]:
    refs: set[str] = set()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch in ("'", '"'):
            _quoted, i = _consume_quoted_string(body, i)
            continue

        item_match = _VARIABLE_ITEM_RE.match(body, i)
        if item_match:
            name = item_match.group(2)
            if name in variable_names:
                refs.add(name)
            i = item_match.end()
            continue

        get_match = _VARIABLE_GET_RE.match(body, i)
        if get_match:
            name = get_match.group(2)
            if name in variable_names:
                refs.add(name)
            i = get_match.end()
            continue

        if body.startswith("variables.", i):
            match = _IDENTIFIER_RE.match(body, i + len("variables."))
            if match:
                name = match.group(0)
                if name in variable_names:
                    refs.add(name)
                i = match.end()
                continue

        match = _IDENTIFIER_RE.match(body, i)
        if match:
            name = match.group(0)
            end = match.end()
            prev = body[i - 1] if i > 0 else ""
            if name in variable_names and prev != "." and not (
                prev.isalnum() or prev == "_"
            ):
                refs.add(name)
            i = end
            continue

        i += 1
    return refs


def _extract_variable_reference_names(
    value: Any,
    variable_names: set[str],
) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        for match in _EXPR_RE.finditer(value):
            refs.update(
                _extract_variable_reference_names_from_expr_body(
                    match.group(1),
                    variable_names,
                )
            )
        return refs
    if isinstance(value, list):
        for item in value:
            refs.update(_extract_variable_reference_names(item, variable_names))
        return refs
    if isinstance(value, dict):
        for item in value.values():
            refs.update(_extract_variable_reference_names(item, variable_names))
        return refs
    return refs


def validate_no_deferred_variable_references(
    value: Any,
    variables: dict[str, Variable],
    resolver: ExpressionResolver,
    *,
    location: str,
    usage: str,
    field_kind: str = "pre-allocation field",
) -> None:
    deferred_names = _deferred_global_variable_names(variables, resolver)
    if not deferred_names:
        return

    references = _extract_variable_reference_names(value, set(variables))
    blocked = sorted(deferred_names & references)
    if not blocked:
        return

    if len(blocked) == 1:
        subject = f"Deferred global variable '{blocked[0]}' is"
    else:
        subject = (
            "Deferred global variables "
            + ", ".join(f"'{name}'" for name in blocked)
            + " are"
        )
    raise ValueError(
        f"{subject} referenced in {field_kind} '{location}' and cannot "
        f"be used while resolving {usage}. Deferred globals depend on "
        "backends.* or artifacts.* and are resolved after backend allocation "
        "and artifact resolution; use only static variables in this field."
    )


def _format_runtime_context_hint(
    variables: dict[str, Variable],
    unresolved: list[str],
    resolver: ExpressionResolver,
) -> str:
    runtime_refs_by_name: dict[str, set[str]] = {}
    for name in unresolved:
        runtime_refs = _runtime_refs_for_variable(variables[name], resolver)
        if runtime_refs:
            runtime_refs_by_name[name] = runtime_refs

    if not runtime_refs_by_name:
        return ""

    names = ", ".join(sorted(runtime_refs_by_name))
    contexts = ", ".join(
        f"'{ctx}'"
        for ctx in sorted(
            {ctx for refs in runtime_refs_by_name.values() for ctx in refs}
        )
    )
    return (
        f". Top-level variables are resolved before backend allocation and workflow "
        f"assembly, so contexts like {contexts} are not available there. Move "
        f"{names} under workflow.variables to resolve it after allocation."
    )


def _extract_variables(merged: Dict[str, Any]) -> dict[str, Any]:
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


def _to_jinja_literal(value: Any) -> str:
    return repr(value)


def _consume_quoted_string(text: str, start: int) -> tuple[str, int]:
    quote = text[start]
    end = start + 1
    while end < len(text):
        if text[end] == "\\":
            end += 2
            continue
        if text[end] == quote:
            end += 1
            break
        end += 1
    return text[start:end], end


def _inline_resolved_vars_in_expr_body(
    body: str,
    resolved: dict[str, Any],
    domains: dict[str, list[Any]] | None,
) -> str:
    if not resolved:
        return body

    domain_map = domains or {}
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch in ("'", '"'):
            quoted, i = _consume_quoted_string(body, i)
            out.append(quoted)
            continue

        if body.startswith("variables.", i):
            match = _IDENTIFIER_RE.match(body, i + len("variables."))
            if match:
                name = match.group(0)
                end = match.end()
                if body.startswith(".domain", end) and name in resolved:
                    out.append(_to_jinja_literal(domain_map.get(name, [])))
                    i = end + len(".domain")
                    continue
                if name in resolved:
                    out.append(_to_jinja_literal(resolved[name]))
                    i = end
                    continue

        match = _IDENTIFIER_RE.match(body, i)
        if match:
            name = match.group(0)
            end = match.end()
            prev = body[i - 1] if i > 0 else ""
            if prev != "." and not (prev.isalnum() or prev == "_"):
                if body.startswith(".domain", end) and name in resolved:
                    out.append(_to_jinja_literal(domain_map.get(name, [])))
                    i = end + len(".domain")
                    continue
                if name in resolved:
                    out.append(_to_jinja_literal(resolved[name]))
                    i = end
                    continue

        out.append(ch)
        i += 1
    return "".join(out)


def _inline_resolved_vars_in_jinja(
    text: str,
    resolved: dict[str, Any],
    domains: dict[str, list[Any]] | None = None,
) -> str:
    if "${{" not in text or not resolved:
        return text

    def _rewrite(match: re.Match[str]) -> str:
        expr_text = match.group(0)
        body = expr_text[3:-2]
        rewritten = _inline_resolved_vars_in_expr_body(body, resolved, domains)
        if rewritten == body:
            return expr_text
        return "${{" + rewritten + "}}"

    return _EXPR_RE.sub(_rewrite, text)


def resolve_expressions(
    obj: Any,
    ctx: dict[str, Any],
    env: SandboxedEnvironment,
    resolved: dict[str, Any] | None = None,
    domains: dict[str, list[Any]] | None = None,
) -> Any:
    """Resolve compose-time expressions where all refs are available."""
    if isinstance(obj, str):
        if "${{" not in obj:
            return obj
        stripped = obj.strip()
        if _EXPR_RE.fullmatch(stripped):
            try:
                result = env.from_string(obj).render(**ctx)
                return _coerce_type(result)
            except (UndefinedError, Exception):
                rewritten = _inline_resolved_vars_in_jinja(obj, resolved or {}, domains)
                if rewritten == obj:
                    return obj
                try:
                    result = env.from_string(rewritten).render(**ctx)
                    return _coerce_type(result)
                except (UndefinedError, Exception):
                    return rewritten

        def _replace_match(match: re.Match[str]) -> str:
            expr_text = match.group(0)
            try:
                return env.from_string(expr_text).render(**ctx)
            except (UndefinedError, Exception):
                rewritten = _inline_resolved_vars_in_jinja(
                    expr_text, resolved or {}, domains
                )
                if rewritten == expr_text:
                    return expr_text
                try:
                    return env.from_string(rewritten).render(**ctx)
                except (UndefinedError, Exception):
                    return rewritten

        return _EXPR_RE.sub(_replace_match, obj)
    if isinstance(obj, list):
        return [
            resolve_expressions(item, ctx, env, resolved=resolved, domains=domains)
            for item in obj
        ]
    if isinstance(obj, dict):
        return {
            key: resolve_expressions(val, ctx, env, resolved=resolved, domains=domains)
            for key, val in obj.items()
        }
    return obj


def _resolve_shell_vars(obj: Any, resolved: dict[str, Any]) -> Any:
    if isinstance(obj, str):

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in _BUILTIN_ENV_VARS or name not in resolved:
                return match.group(0)
            return str(resolved[name])

        return _SHELL_VAR_RE.sub(_replace, obj)
    if isinstance(obj, list):
        return [_resolve_shell_vars(item, resolved) for item in obj]
    if isinstance(obj, dict):
        return {key: _resolve_shell_vars(val, resolved) for key, val in obj.items()}
    return obj


def _remove_resolved_variables(
    section: list | None,
    resolved_names: set[str],
) -> list:
    if not section:
        return []
    return [
        entry
        for entry in section
        if not (isinstance(entry, dict) and entry.get("name") in resolved_names)
    ]


def _collect_replica_variable_names(merged: Dict[str, Any]) -> set[str]:
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
        for value in replicas.get("variables") or []:
            if isinstance(value, str):
                names.add(value)
        for field_name in ("count", "policy"):
            val = replicas.get(field_name)
            if isinstance(val, str):
                for match in _EXPR_RE.finditer(val):
                    expr = match.group(1).strip()
                    if expr.startswith("variables."):
                        names.add(expr.split(".", 1)[1])
    return names


def _clean_resolved_strings(obj: Any) -> Any:
    if isinstance(obj, str):
        return obj.rstrip(" \t\n")
    if isinstance(obj, list):
        return [_clean_resolved_strings(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _clean_resolved_strings(val) for key, val in obj.items()}
    return obj


def resolve_variables_inline(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve all statically resolvable variables inline throughout a config."""
    replica_vars = _collect_replica_variable_names(merged)
    variables = _extract_variables(merged)
    domains = extract_domains_from_raw_config(merged)
    resolved, _unresolvable = _classify_resolvable(variables)

    for replica_var in replica_vars:
        resolved.pop(replica_var, None)

    if not resolved and not replica_vars:
        return merged

    env = SandboxedEnvironment(
        undefined=StrictUndefined,
        autoescape=False,
        variable_start_string="${{",
        variable_end_string="}}",
    )
    wrapped = build_variables_ctx_from_raw(resolved, domains)

    for replica_var in replica_vars:
        if replica_var not in wrapped and replica_var in variables:
            wrapped[replica_var] = VariableValue(
                f"${{{{ variables.{replica_var} }}}}",
                domain=domains.get(replica_var),
            )
    ctx: dict[str, Any] = {"variables": wrapped, **wrapped}

    merged = resolve_expressions(merged, ctx, env, resolved=resolved, domains=domains)
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


def artifacts_ctx(state: SflowState) -> dict[str, Any]:
    """Build expression context for artifacts from resolved state."""
    return {name: artifact.to_context_dict() for name, artifact in (state.artifacts or {}).items()}


def resolve_artifacts(
    config: SflowConfig,
    state: SflowState,
    *,
    resolver: ExpressionResolver,
    workspace_dir: Any | None = None,
    output_dir: Any | None = None,
    materialize: bool = False,
    remote_filesystem: bool = False,
) -> SflowState:
    """Resolve artifact URIs and inline content into ``state.artifacts``.

    ``remote_filesystem`` is set when the workflow's backend executes off the
    controller host (e.g. Kubernetes). It is forwarded to artifact resolvers that
    accept it (the file/fs resolver) so local ``fs://`` paths are passed through
    instead of being validated/created on the controller.
    """
    import inspect
    ensure_builtin_artifacts_registered()

    ws_dir = Path(workspace_dir) if workspace_dir is not None else Path.cwd()
    out_dir = Path(output_dir) if output_dir is not None else ws_dir / "sflow_output"
    cache_dir = ws_dir / ".sflow_cache" / "artifacts"

    variables_ctx = build_variables_ctx(state.variables)
    backends_ctx: dict[str, Any] = {
        name: backend.to_dict() for name, backend in (state.backends or {}).items()
    }
    ctx: dict[str, Any] = {
        "variables": variables_ctx,
        "backends": backends_ctx,
        "workflow": {"name": config.workflow.name},
        **variables_ctx,
    }

    artifacts: dict[str, Artifact] = {}
    for artifact_conf in config.artifacts or []:
        validate_no_deferred_variable_references(
            artifact_conf.model_dump()
            if hasattr(artifact_conf, "model_dump")
            else artifact_conf,
            state.variables,
            resolver,
            location=f"artifacts.{artifact_conf.name}",
            usage="artifacts",
            field_kind="artifact definition",
        )
        uri_raw: Any = artifact_conf.uri
        uri = (
            str(resolver.resolve(uri_raw, ctx))
            if resolver.has_expression(uri_raw)
            else str(uri_raw)
        )

        content_raw: Any = artifact_conf.content
        content = (
            str(resolver.resolve(content_raw, ctx))
            if content_raw is not None and resolver.has_expression(content_raw)
            else (str(content_raw) if content_raw is not None else None)
        )

        resolver_obj = get_artifact_resolver_for_uri(uri)
        if resolver_obj is None:
            artifacts[artifact_conf.name] = Artifact(
                name=artifact_conf.name,
                uri=uri,
                description=artifact_conf.description,
                path=None,
            )
            continue

        resolve_kwargs: dict[str, Any] = dict(
            name=artifact_conf.name,
            uri=uri,
            description=artifact_conf.description,
            content=content,
            workspace_dir=ws_dir,
            cache_dir=cache_dir,
            output_dir=out_dir,
            materialize=materialize,
        )
        # Only resolvers that opt in (the file/fs resolver) receive remote_filesystem,
        # so http/hf/docker resolvers keep their existing behavior unchanged.
        if "remote_filesystem" in inspect.signature(resolver_obj.resolve).parameters:
            resolve_kwargs["remote_filesystem"] = remote_filesystem
        artifact = resolver_obj.resolve(**resolve_kwargs)
        artifacts[artifact.name] = artifact

    state.artifacts = artifacts
    return state


def maybe_int(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def cast_variable_value(name: str, value: Any, var_type: VariableType) -> Any:
    if value is None:
        return None

    if var_type == VariableType.STRING:
        return value if isinstance(value, str) else str(value)

    if var_type == VariableType.INTEGER:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(
                    f"Variable '{name}' expected integer, got '{value}'"
                ) from exc
        raise ValueError(
            f"Variable '{name}' expected integer, got {type(value).__name__}"
        )

    if var_type == VariableType.FLOAT:
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(
                    f"Variable '{name}' expected float, got '{value}'"
                ) from exc
        raise ValueError(
            f"Variable '{name}' expected float, got {type(value).__name__}"
        )

    if var_type == VariableType.BOOLEAN:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
            raise ValueError(f"Variable '{name}' expected boolean, got '{value}'")
        raise ValueError(
            f"Variable '{name}' expected boolean, got {type(value).__name__}"
        )

    return value


def variable_context_value(name: str, value: Any, var_type: VariableType) -> Any:
    if var_type == VariableType.STRING:
        return maybe_int(value)
    return cast_variable_value(name, value, var_type)


def resolve_and_update_variables(
    *,
    state: SflowState,
    variable_confs: list[Any],
    resolver: ExpressionResolver,
    collision: Literal["overwrite", "error"] = "overwrite",
    extra_ctx: dict[str, Any] | None = None,
    variable_scope: Literal["global", "deferred_global", "workflow"] = "global",
    available_contexts: frozenset[str] | None = None,
    defer_contexts: frozenset[str] | None = None,
) -> SflowState:
    """Shared variable resolution engine used by global/workflow variables."""
    if not variable_confs:
        return state

    variables: dict[str, Variable] = dict(state.variables)
    for variable_conf in variable_confs:
        if collision == "error" and variable_conf.name in variables:
            raise ValueError(
                f"Workflow variable '{variable_conf.name}' conflicts with existing variable"
            )
        variables[variable_conf.name] = Variable(
            name=variable_conf.name,
            value=variable_conf.value,
            description=variable_conf.description,
            domain=variable_conf.domain,
            type=VariableType(variable_conf.type),
        )

    resolved_values: dict[str, Any] = {
        key: (
            variable.value
            if resolver.has_expression(variable.value)
            else variable_context_value(key, variable.value, variable.type)
        )
        for key, variable in variables.items()
    }
    max_passes = len(variables) + 1
    variable_names = set(variables)
    for _ in range(max_passes):
        progress = False
        ctx_values = {
            key: value
            for key, value in resolved_values.items()
            if not resolver.has_expression(value)
        }
        # Wrap in VariableValue so expressions can read `.domain`/`.value`
        # metadata (e.g. `variables.CONCURRENCY.domain | max`), matching the
        # compose (`build_variables_ctx_from_raw`) and workflow/task-script
        # (`build_variables_ctx`) paths. Arithmetic and string rendering
        # transparently delegate to the underlying value.
        wrapped_values = {
            key: VariableValue(value, domain=variables[key].domain)
            for key, value in ctx_values.items()
        }
        ctx: dict[str, Any] = {"variables": wrapped_values, **wrapped_values}
        if extra_ctx:
            ctx.update(extra_ctx)

        for name, variable in variables.items():
            current = variable.value
            if not resolver.has_expression(current):
                continue

            deps = _extract_variable_reference_names(current, variable_names)
            if any(
                resolver.has_expression(variables[dep].value)
                for dep in deps
            ):
                continue
            if any(dep not in ctx_values for dep in deps):
                continue

            try:
                new_value = resolver.resolve(current, ctx)
            except ValueError as exc:
                err = str(exc)
                if "Undefined variable" in err or "Error evaluating expression" in err:
                    continue
                raise

            if new_value != current:
                if not resolver.has_expression(new_value):
                    new_value = cast_variable_value(name, new_value, variable.type)
                    resolved_value = variable_context_value(
                        name, new_value, variable.type
                    )
                else:
                    resolved_value = new_value
                variable.value = new_value
                resolved_values[name] = resolved_value
                progress = True

        if not progress:
            break

    unresolved = [
        name for name, variable in variables.items() if resolver.has_expression(variable.value)
    ]
    if unresolved:
        deferred = set()
        if defer_contexts:
            allowed_contexts = (available_contexts or frozenset()) | defer_contexts
            contexts_by_name = _runtime_contexts_by_variable(variables, resolver)
            deferred = {
                name
                for name in unresolved
                if (contexts := contexts_by_name.get(name, set()))
                and contexts <= allowed_contexts
                and bool(contexts & defer_contexts)
            }
        unresolved_now = [name for name in unresolved if name not in deferred]
        if unresolved_now:
            message = (
                "Unresolved variable expressions (missing refs or cycle): "
                + ", ".join(sorted(unresolved_now))
            )
            if variable_scope == "global":
                message += _format_runtime_context_hint(
                    variables, unresolved_now, resolver
                )
            raise ValueError(message)

    for name, variable in variables.items():
        if resolver.has_expression(variable.value):
            continue
        variable.value = cast_variable_value(name, variable.value, variable.type)

    state.variables = variables
    return state


def resolve_global_variables(
    config: SflowConfig,
    state: SflowState,
    *,
    resolver: ExpressionResolver,
) -> SflowState:
    return resolve_and_update_variables(
        state=state,
        variable_confs=list(config.variables or []),
        resolver=resolver,
        collision="overwrite",
        extra_ctx=None,
        variable_scope="global",
        defer_contexts=_DEFERRED_GLOBAL_CONTEXTS,
    )


def resolve_deferred_global_variables(
    config: SflowConfig,
    state: SflowState,
    *,
    resolver: ExpressionResolver,
    available_contexts: frozenset[str] | None = None,
    defer_contexts: frozenset[str] | None = None,
) -> SflowState:
    available = available_contexts or _DEFERRED_GLOBAL_CONTEXTS
    extra_ctx: dict[str, Any] = {}
    if "backends" in available:
        extra_ctx["backends"] = {
            name: backend.to_dict() for name, backend in (state.backends or {}).items()
        }
    if "artifacts" in available:
        extra_ctx["artifacts"] = artifacts_ctx(state)
    return resolve_and_update_variables(
        state=state,
        variable_confs=list(config.variables or []),
        resolver=resolver,
        collision="overwrite",
        extra_ctx=extra_ctx,
        variable_scope="deferred_global",
        available_contexts=available,
        defer_contexts=defer_contexts,
    )


def resolve_workflow_variables(
    config: SflowConfig,
    state: SflowState,
    *,
    resolver: ExpressionResolver,
    workspace_dir: Any | None = None,
) -> SflowState:
    backends_ctx: dict[str, Any] = {
        name: backend.to_dict() for name, backend in (state.backends or {}).items()
    }
    variables_ctx = build_variables_ctx(state.variables)
    if (not state.artifacts) and (config.artifacts):
        state = resolve_artifacts(
            config,
            state,
            resolver=resolver,
            workspace_dir=workspace_dir,
            materialize=False,
        )
    artifact_ctx = artifacts_ctx(state)
    extra_ctx: dict[str, Any] = {
        "workflow": {"name": config.workflow.name},
        "backends": backends_ctx,
        "artifacts": artifact_ctx,
        "variables": variables_ctx,
        **variables_ctx,
    }

    return resolve_and_update_variables(
        state=state,
        variable_confs=list(config.workflow.variables or []),
        resolver=resolver,
        collision="overwrite",
        extra_ctx=extra_ctx,
        variable_scope="workflow",
    )
