#!/usr/bin/env python3
"""Validate sflow YAML configuration files for common issues.

Usage:
    python validate_sflow_yaml.py <yaml_file> [<yaml_file> ...]

Checks performed:
    - version field is present and set to "0.1"
    - Top-level keys are from the allowed set
    - Variable references (${{ }}) have valid syntax
    - depends_on references exist as task names
    - Operator references exist as declared operators
    - Backend references exist as declared backends
    - Artifact expression references match declared artifacts
    - GPU resource math consistency (TP * DP * PP vs declared gpus)
    - Common mistake warnings (missing probes, missing depends_on, etc.)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ALLOWED_TOP_LEVEL_KEYS = {"version", "variables", "artifacts", "backends", "operators", "workflow"}
EXPRESSION_PATTERN = re.compile(r"\$\{\{(.+?)\}\}", re.DOTALL)
UNCLOSED_EXPRESSION_PATTERN = re.compile(r"\$\{\{(?!.*\}\})")


class ValidationResult:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def print_report(self) -> None:
        status = "PASS" if self.ok else "FAIL"
        print(f"\n{'=' * 60}")
        print(f"  {status}: {self.filepath}")
        print(f"{'=' * 60}")
        if self.errors:
            print(f"\n  Errors ({len(self.errors)}):")
            for i, e in enumerate(self.errors, 1):
                print(f"    [{i}] {e}")
        if self.warnings:
            print(f"\n  Warnings ({len(self.warnings)}):")
            for i, w in enumerate(self.warnings, 1):
                print(f"    [{i}] {w}")
        if self.ok and not self.warnings:
            print("  No issues found.")
        print()


def _extract_variable_names(config: dict) -> set[str]:
    """Extract declared variable names from both global and workflow variables."""
    names: set[str] = set()
    variables = config.get("variables", {})
    if isinstance(variables, dict):
        names.update(variables.keys())
    elif isinstance(variables, list):
        for v in variables:
            if isinstance(v, dict) and "name" in v:
                names.add(v["name"])

    wf = config.get("workflow", {})
    if isinstance(wf, dict):
        wf_vars = wf.get("variables", {})
        if isinstance(wf_vars, dict):
            names.update(wf_vars.keys())
    return names


def _extract_task_names(config: dict) -> list[str]:
    """Extract task names from workflow."""
    wf = config.get("workflow", {})
    if not isinstance(wf, dict):
        return []
    tasks = wf.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    return [t["name"] for t in tasks if isinstance(t, dict) and "name" in t]


def _extract_operator_names(config: dict) -> set[str]:
    """Extract declared operator names."""
    operators = config.get("operators", [])
    if not isinstance(operators, list):
        return set()
    return {
        o["name"] for o in operators
        if isinstance(o, dict) and "name" in o
    }


def _extract_backend_names(config: dict) -> set[str]:
    """Extract declared backend names."""
    backends = config.get("backends", [])
    if not isinstance(backends, list):
        return set()
    return {
        b["name"] for b in backends
        if isinstance(b, dict) and "name" in b
    }


def _extract_artifact_names(config: dict) -> set[str]:
    """Extract declared artifact names."""
    artifacts = config.get("artifacts", [])
    if not isinstance(artifacts, list):
        return set()
    return {
        a["name"] for a in artifacts
        if isinstance(a, dict) and "name" in a
    }


def _find_expressions(obj, path: str = "") -> list[tuple[str, str]]:
    """Recursively find all ${{ }} expressions and their YAML paths."""
    results: list[tuple[str, str]] = []
    if isinstance(obj, str):
        for m in EXPRESSION_PATTERN.finditer(obj):
            results.append((path, m.group(0)))
        for m in UNCLOSED_EXPRESSION_PATTERN.finditer(obj):
            results.append((path, m.group(0) + " [UNCLOSED]"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            results.extend(_find_expressions(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            results.extend(_find_expressions(v, f"{path}[{i}]"))
    return results


def _resolve_variable_value(variables: dict, name: str) -> int | float | str | None:
    """Try to resolve a simple variable value (non-expression)."""
    var = variables.get(name)
    if var is None:
        return None
    if isinstance(var, dict):
        val = var.get("value")
    else:
        val = var
    if isinstance(val, str) and "${{" in val:
        return None
    return val


def check_version(config: dict, result: ValidationResult) -> None:
    if "version" not in config:
        result.error("Missing required field: 'version'")
        return
    if str(config["version"]) != "0.1":
        result.error(f"Invalid version: '{config['version']}' (must be '0.1')")


def check_top_level_keys(config: dict, result: ValidationResult) -> None:
    unknown = set(config.keys()) - ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        result.error(f"Unknown top-level keys: {', '.join(sorted(unknown))}")


def check_workflow(config: dict, result: ValidationResult) -> None:
    wf = config.get("workflow")
    if wf is None:
        result.warn("No 'workflow' section (OK for modular fragment files)")
        return
    if not isinstance(wf, dict):
        result.error("'workflow' must be a mapping")
        return
    if "name" not in wf:
        result.warn("workflow.name is not set")
    tasks = wf.get("tasks")
    if tasks is None:
        result.warn("No tasks defined in workflow (OK for modular fragment files)")
        return
    if not isinstance(tasks, list) or len(tasks) == 0:
        result.error("workflow.tasks must be a non-empty list")


def check_expressions(config: dict, result: ValidationResult) -> None:
    variable_names = _extract_variable_names(config)
    artifact_names = _extract_artifact_names(config)
    expressions = _find_expressions(config)

    var_ref_pattern = re.compile(r"variables\.(\w+)")
    art_ref_pattern = re.compile(r"artifacts\.(\w+)")

    for path, expr in expressions:
        if "[UNCLOSED]" in expr:
            result.error(f"Unclosed expression at {path}: {expr}")
            continue
        inner = expr[3:-2].strip()
        if not inner:
            result.error(f"Empty expression at {path}: {expr}")
            continue

        for m in var_ref_pattern.finditer(inner):
            var_name = m.group(1)
            if variable_names and var_name not in variable_names:
                result.warn(
                    f"Expression at {path} references variable '{var_name}' "
                    f"which is not declared in this file"
                )

        for m in art_ref_pattern.finditer(inner):
            art_name = m.group(1)
            if artifact_names and art_name not in artifact_names:
                result.warn(
                    f"Expression at {path} references artifact '{art_name}' "
                    f"which is not declared in this file"
                )


def check_depends_on(config: dict, result: ValidationResult) -> None:
    task_names = set(_extract_task_names(config))
    if not task_names:
        return
    wf = config.get("workflow", {})
    tasks = wf.get("tasks", [])
    if not isinstance(tasks, list):
        return
    for task in tasks:
        if not isinstance(task, dict):
            continue
        name = task.get("name", "<unnamed>")
        deps = task.get("depends_on", [])
        if not isinstance(deps, list):
            result.error(f"Task '{name}': depends_on must be a list")
            continue
        for dep in deps:
            if dep not in task_names:
                result.warn(
                    f"Task '{name}': depends_on references '{dep}' which is not "
                    f"defined in this file (OK if using modular composition with --missable-tasks)"
                )


def check_task_names_unique(config: dict, result: ValidationResult) -> None:
    names = _extract_task_names(config)
    seen: set[str] = set()
    for n in names:
        if n in seen:
            result.error(f"Duplicate task name: '{n}'")
        seen.add(n)


def check_operator_references(config: dict, result: ValidationResult) -> None:
    """Check that tasks reference declared operators."""
    operator_names = _extract_operator_names(config)
    if not operator_names:
        return
    wf = config.get("workflow", {})
    tasks = wf.get("tasks", []) if isinstance(wf, dict) else []
    if not isinstance(tasks, list):
        return
    for task in tasks:
        if not isinstance(task, dict):
            continue
        name = task.get("name", "<unnamed>")
        op = task.get("operator")
        if op is None:
            continue
        op_name = op if isinstance(op, str) else op.get("name") if isinstance(op, dict) else None
        if op_name and "${{" not in str(op_name) and op_name not in operator_names:
            result.warn(
                f"Task '{name}': references operator '{op_name}' "
                f"which is not declared in this file (OK if using modular composition)"
            )


def check_artifact_fs_paths(config: dict, result: ValidationResult) -> None:
    """Warn about fs:// artifact paths that don't exist."""
    artifacts = config.get("artifacts", [])
    if not isinstance(artifacts, list):
        return
    for art in artifacts:
        if not isinstance(art, dict):
            continue
        uri = art.get("uri", "")
        name = art.get("name", "<unnamed>")
        if not isinstance(uri, str) or "${{" in uri:
            continue
        if uri.startswith("fs://"):
            fs_path = uri[5:]
            if fs_path and not Path(fs_path).exists():
                result.warn(
                    f"Artifact '{name}': fs:// path does not exist: {fs_path}"
                )


def check_gpu_consistency(config: dict, result: ValidationResult) -> None:
    """Check if GPU resource declarations are consistent with parallelism variables."""
    variables = config.get("variables", {})
    if not isinstance(variables, dict):
        return

    wf = config.get("workflow", {})
    tasks = wf.get("tasks", []) if isinstance(wf, dict) else []
    if not isinstance(tasks, list):
        return

    parallelism_groups = [
        ("CTX", "prefill"),
        ("GEN", "decode"),
        ("AGG", "aggregated"),
    ]

    for prefix, label in parallelism_groups:
        tp = _resolve_variable_value(variables, f"{prefix}_TP_SIZE")
        dp = _resolve_variable_value(variables, f"{prefix}_DP_SIZE")
        pp = _resolve_variable_value(variables, f"{prefix}_PP_SIZE")
        if tp is not None and dp is not None and pp is not None:
            try:
                expected_gpus = int(tp) * int(dp) * int(pp)
                gpus_var = _resolve_variable_value(variables, f"{prefix}_GPUS_PER_WORKER")
                if gpus_var is not None and int(gpus_var) != expected_gpus:
                    result.warn(
                        f"{label}: {prefix}_GPUS_PER_WORKER={gpus_var} but "
                        f"{prefix}_TP*DP*PP = {tp}*{dp}*{pp} = {expected_gpus}"
                    )
            except (ValueError, TypeError):
                pass


def check_common_mistakes(config: dict, result: ValidationResult) -> None:
    """Warn about common mistakes."""
    wf = config.get("workflow", {})
    tasks = wf.get("tasks", []) if isinstance(wf, dict) else []
    if not isinstance(tasks, list):
        return

    for task in tasks:
        if not isinstance(task, dict):
            continue
        name = task.get("name", "<unnamed>")

        script = task.get("script", [])
        if not isinstance(script, list) or len(script) == 0:
            result.error(f"Task '{name}': script must be a non-empty list")

        has_server_keyword = False
        if isinstance(script, list):
            script_text = " ".join(str(s) for s in script)
            has_server_keyword = any(
                kw in script_text.lower()
                for kw in ["server", "serve", "frontend", "nats", "etcd"]
            )

        probes = task.get("probes")
        if has_server_keyword and probes is None:
            result.warn(
                f"Task '{name}': looks like a server but has no readiness probe. "
                f"Consider adding probes.readiness (log_watch or tcp_port)."
            )

        if has_server_keyword and probes and isinstance(probes, dict):
            failure = probes.get("failure")
            if failure is None:
                result.warn(
                    f"Task '{name}': server task has readiness probe but no failure probe. "
                    f"Consider adding probes.failure.log_watch for 'Traceback'."
                )

        if probes and isinstance(probes, dict):
            readiness = probes.get("readiness", {})
            if isinstance(readiness, dict):
                timeout = readiness.get("timeout")
                if timeout is not None and isinstance(timeout, (int, float)) and timeout < 10:
                    result.warn(
                        f"Task '{name}': readiness probe timeout is very short ({timeout}s). "
                        f"Consider at least 60s for container-based tasks."
                    )

        replicas = task.get("replicas")
        if replicas and isinstance(replicas, dict):
            sweep_vars = replicas.get("variables", [])
            if isinstance(sweep_vars, list):
                variables = config.get("variables", {})
                if isinstance(variables, dict):
                    for sv in sweep_vars:
                        var_def = variables.get(sv)
                        if isinstance(var_def, dict) and "domain" not in var_def:
                            result.warn(
                                f"Task '{name}': sweep variable '{sv}' has no 'domain' "
                                f"defined. Add domain to enable sweep."
                            )

    backends = config.get("backends", [])
    if isinstance(backends, list):
        defaults = [b for b in backends if isinstance(b, dict) and b.get("default")]
        if len(defaults) > 1:
            result.error(f"Multiple default backends declared ({len(defaults)}). At most one allowed.")


def validate_file(filepath: str) -> ValidationResult:
    result = ValidationResult(filepath)

    path = Path(filepath)
    if not path.exists():
        result.error(f"File not found: {filepath}")
        return result
    if not path.suffix.lower() in (".yaml", ".yml"):
        result.warn(f"File does not have .yaml/.yml extension: {filepath}")

    try:
        with open(filepath) as f:
            content = f.read()
    except OSError as e:
        result.error(f"Cannot read file: {e}")
        return result

    if not content.strip():
        result.error("File is empty")
        return result

    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError as e:
        result.error(f"YAML syntax error: {e}")
        return result

    if not isinstance(config, dict):
        result.error("Top-level YAML must be a mapping (dict)")
        return result

    check_version(config, result)
    check_top_level_keys(config, result)
    check_workflow(config, result)
    check_expressions(config, result)
    check_depends_on(config, result)
    check_task_names_unique(config, result)
    check_operator_references(config, result)
    check_artifact_fs_paths(config, result)
    check_gpu_consistency(config, result)
    check_common_mistakes(config, result)

    return result


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <yaml_file> [<yaml_file> ...]", file=sys.stderr)
        return 2

    all_ok = True
    for filepath in sys.argv[1:]:
        result = validate_file(filepath)
        result.print_report()
        if not result.ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
