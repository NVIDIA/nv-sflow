#!/usr/bin/env python3
"""Parse sflow log output and categorize errors with suggested fixes.

Usage:
    python parse_sflow_errors.py <log_file>
    python parse_sflow_errors.py -                    # read from stdin
    python parse_sflow_errors.py --json <log_file>    # JSON output
    sflow run -f config.yaml --dry-run 2>&1 | python parse_sflow_errors.py -

Output:
    Structured error summary with categories, suggested fixes, and statistics.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Error pattern definitions
# ---------------------------------------------------------------------------


@dataclass
class ErrorPattern:
    category: str
    pattern: re.Pattern[str]
    description: str
    fix: str


ERROR_PATTERNS: list[ErrorPattern] = [
    # Config loading
    ErrorPattern(
        category="config",
        pattern=re.compile(r"Configuration file not found:\s*(.+)"),
        description="Configuration file not found",
        fix="Check the file path passed to -f / --file. Ensure it exists relative to CWD or use an absolute path.",
    ),
    ErrorPattern(
        category="config",
        pattern=re.compile(r"Configuration file is empty:\s*(.+)"),
        description="Empty configuration file",
        fix='Add valid sflow YAML content. At minimum: version: "0.1"',
    ),
    ErrorPattern(
        category="config",
        pattern=re.compile(r"Error parsing YAML configuration:\s*(.+)", re.DOTALL),
        description="YAML syntax error",
        fix="Fix YAML syntax at the indicated line. Common: bad indentation, missing colons, tabs instead of spaces.",
    ),
    ErrorPattern(
        category="config",
        pattern=re.compile(r"Configuration validation failed:\s*(.+)", re.DOTALL),
        description="Schema validation failed (Pydantic)",
        fix="Check field types and constraints. See schema-reference.md for allowed values. Unknown fields are rejected.",
    ),
    ErrorPattern(
        category="config",
        pattern=re.compile(r"Configuration expression syntax validation failed"),
        description="Expression syntax validation error",
        fix="Check ${{ }} expressions for Jinja2 syntax errors. Ensure brackets are balanced.",
    ),
    # Expression resolution
    ErrorPattern(
        category="expression",
        pattern=re.compile(r"Undefined variable in expression\s*(.+?):\s*(.+)"),
        description="Undefined variable reference",
        fix="Check variable name spelling. Ensure it's declared in 'variables' or 'workflow.variables'.",
    ),
    ErrorPattern(
        category="expression",
        pattern=re.compile(r"Invalid expression syntax in '(.+?)':\s*(.+)"),
        description="Invalid expression syntax",
        fix="Fix the Jinja2 expression. Check filters, operators, and bracket matching.",
    ),
    ErrorPattern(
        category="expression",
        pattern=re.compile(r"Error evaluating expression\s*(.+?):\s*(.+)"),
        description="Expression evaluation error",
        fix="Check expression logic: division by zero, type mismatches, undefined operations.",
    ),
    # Variable/artifact overrides
    ErrorPattern(
        category="override",
        pattern=re.compile(r"Variable '(.+?)' specified in overrides is not defined"),
        description="--set variable not declared in config",
        fix="Declare the variable in the YAML 'variables' section first, then use --set to override.",
    ),
    ErrorPattern(
        category="override",
        pattern=re.compile(r"Invalid variable override format:\s*'(.+?)'"),
        description="Bad --set format",
        fix="Use --set KEY=VALUE format. For lists: --set CONCURRENCY=[16,32,64]",
    ),
    ErrorPattern(
        category="override",
        pattern=re.compile(r"Artifact '(.+?)' specified in overrides is not defined"),
        description="--artifact not declared in config",
        fix="Declare the artifact in the YAML 'artifacts' section first.",
    ),
    # Artifact validation
    ErrorPattern(
        category="artifact",
        pattern=re.compile(r"Artifact path validation failed"),
        description="Artifact fs:// path does not exist",
        fix="Check the fs:// URI path. Create the directory/file or fix the path, or "
            "pass --skip-artifact-check if it only exists where the task runs.",
    ),
    ErrorPattern(
        category="artifact",
        pattern=re.compile(r"Artifact '(.+?)' \(fs://\) path does not exist:\s*(.+)"),
        description="Artifact path missing",
        fix="Create the path or fix the artifact URI, or pass --skip-artifact-check if "
            "it only exists where the task runs.",
    ),
    # Merge errors
    ErrorPattern(
        category="merge",
        pattern=re.compile(r"Version conflict:\s*'(.+?)'\s*vs\s*'(.+?)'"),
        description="Version mismatch between merged files",
        fix='All files must use version: "0.1".',
    ),
    ErrorPattern(
        category="merge",
        pattern=re.compile(r"Workflow name conflict:\s*'(.+?)'\s*vs\s*'(.+?)'"),
        description="Workflow name mismatch between merged files",
        fix="Use the same workflow.name in all files, or omit it from fragment files.",
    ),
    ErrorPattern(
        category="merge",
        pattern=re.compile(r"Merged configuration from \[(.+?)\] is incomplete"),
        description="Incomplete merged configuration",
        fix="Ensure all required sections (version, workflow, tasks) exist across the merged files.",
    ),
    # SLURM
    ErrorPattern(
        category="slurm",
        pattern=re.compile(r"scontrol getaddrs failed with exit code (\d+)"),
        description="scontrol address resolution failed",
        fix="Ensure you're in a SLURM allocation or on a node with scontrol access.",
    ),
    ErrorPattern(
        category="slurm",
        pattern=re.compile(r"salloc failed with exit code (\d+)"),
        description="SLURM allocation failed",
        fix="Check partition/account validity and node availability. Run: sinfo -p <partition>",
    ),
    ErrorPattern(
        category="slurm",
        pattern=re.compile(r"sbatch failed:\s*(.+)"),
        description="sbatch submission failed",
        fix="Check SLURM directives in the generated .sh script. Verify partition, account, time, nodes.",
    ),
    ErrorPattern(
        category="slurm",
        pattern=re.compile(r"srun: error:\s*(.+)"),
        description="srun execution error",
        fix="Check srun arguments, node availability, and that the SLURM allocation is still active.",
    ),
    # Runtime / task
    ErrorPattern(
        category="runtime",
        pattern=re.compile(r"Traceback \(most recent call last\)"),
        description="Python traceback (task crash)",
        fix="Read the full traceback. Common: ModuleNotFoundError, CUDA OOM, ConnectionRefused, FileNotFound.",
    ),
    ErrorPattern(
        category="runtime",
        pattern=re.compile(r"readiness probe timed out after (\d+)s"),
        description="Readiness probe timeout",
        fix="Check task log for startup errors. Verify regex_pattern matches server output. Increase timeout.",
    ),
    ErrorPattern(
        category="runtime",
        pattern=re.compile(r"CUDA out of memory"),
        description="GPU out of memory",
        fix="Reduce batch size, TP/DP size, max_num_tokens, or use a smaller model.",
    ),
    ErrorPattern(
        category="runtime",
        pattern=re.compile(r"Address already in use"),
        description="Port conflict",
        fix="Kill stale processes or use a different port.",
    ),
    ErrorPattern(
        category="runtime",
        pattern=re.compile(r"NCCL error|NCCL communicator"),
        description="NCCL communication failure",
        fix="Set NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME. Check network connectivity between nodes.",
    ),
    ErrorPattern(
        category="runtime",
        pattern=re.compile(r"pyxis:|nvidia-container-cli"),
        description="Container/Pyxis error",
        fix="Verify container image URI and registry access. Try: enroot import docker://<image>",
    ),
    ErrorPattern(
        category="runtime",
        pattern=re.compile(r"ModuleNotFoundError:\s*No module named '(.+?)'"),
        description="Missing Python module",
        fix="Install the missing package in the container or add a pip install step to the task script.",
    ),
    ErrorPattern(
        category="runtime",
        pattern=re.compile(r"ConnectionRefusedError"),
        description="Connection refused",
        fix="Upstream service not ready. Check depends_on ordering and readiness probes.",
    ),
    ErrorPattern(
        category="runtime",
        pattern=re.compile(r"FileNotFoundError:\s*(.+)"),
        description="File not found at runtime",
        fix="Check file paths. Ensure model/data paths are accessible inside the container.",
    ),
    # Schema / rename (config)
    ErrorPattern(
        category="config",
        pattern=re.compile(r"Operator type 'docker' is not valid"),
        description="Operator type 'docker' is not valid (use 'docker_run')",
        fix="Use `type: docker_run` for the operator. The Docker backend is `type: docker`.",
    ),
    ErrorPattern(
        category="config",
        pattern=re.compile(
            r"operator override key\(s\) not valid for operator '(.+?)'"
        ),
        description="Task operator override key not a field of that operator",
        fix="Fix the typo, or move a backend-specific operator setting onto that "
        "operator in the backend fragment (deep-merged per backend) instead of the task.",
    ),
    ErrorPattern(
        category="config",
        pattern=re.compile(
            r"result\.patterns and result\.file are mutually exclusive"
        ),
        description="result.patterns and result.file both set",
        fix="Use either `patterns:` or `file:` in a task `result:`, not both.",
    ),
    ErrorPattern(
        category="config",
        pattern=re.compile(r"result\.file must point to a JSON source path"),
        description="result.file is not a .json path",
        fix="`result.file` must end in '.json'. For a metric named 'file', use `patterns:`.",
    ),
    ErrorPattern(
        category="config",
        pattern=re.compile(r"monitor interval must be >= (\d+)ms"),
        description="monitor interval below floor",
        fix="Set monitor `interval` (ms) at or above the minimum (100ms).",
    ),
    ErrorPattern(
        category="config",
        pattern=re.compile(r"used_by_tasks refers to unknown task"),
        description="monitor used_by_tasks references unknown task",
        fix="Point monitor.resources.used_by_tasks at existing task names.",
    ),
    # Kubernetes
    ErrorPattern(
        category="kubernetes",
        pattern=re.compile(r"ImagePullBackOff|ErrImagePull"),
        description="Kubernetes image pull failed",
        fix="Fix the k8s operator `image`; add `image_pull_secrets` for private registries.",
    ),
    ErrorPattern(
        category="kubernetes",
        pattern=re.compile(r"FailedScheduling|[Ii]nsufficient nvidia\.com/gpu"),
        description="Pod could not be scheduled (no capacity / selector)",
        fix="Check gpus_per_node vs real capacity, scheduling mode, node_selector/tolerations, and that gpus.count is a multiple of the node count.",
    ),
    ErrorPattern(
        category="kubernetes",
        pattern=re.compile(r"\(Forbidden\)|forbidden:|auth can-i|is not allowed"),
        description="Kubernetes RBAC / preflight permission denied",
        fix="Grant the required RBAC or use a namespace where you have it. Bypass with SFLOW_SKIP_K8S_PREFLIGHT=1.",
    ),
    ErrorPattern(
        category="kubernetes",
        pattern=re.compile(
            r"Unable to connect to the server|connection to the server.*refused|namespaces \".+\" not found"
        ),
        description="Kubernetes cluster/namespace unreachable",
        fix="Fix --kubeconfig/--kube-context/--kube-namespace; verify the cluster is reachable.",
    ),
    # Docker
    ErrorPattern(
        category="docker",
        pattern=re.compile(r"Cannot connect to the Docker daemon"),
        description="Docker daemon not reachable",
        fix="Ensure Docker is running (`docker info`). For remote hosts check docker_host/context.",
    ),
    ErrorPattern(
        category="docker",
        pattern=re.compile(
            r"could not select device driver|docker: Error response from daemon"
        ),
        description="docker run error (GPU driver / daemon)",
        fix="For GPUs, install the NVIDIA container toolkit and set gpus_per_node. Check image, mounts, extra_args.",
    ),
    # Storage / uploads
    ErrorPattern(
        category="storage",
        pattern=re.compile(r"requires boto3"),
        description="S3 storage requires boto3",
        fix="Install with: pip install 'sflow[s3]'",
    ),
    ErrorPattern(
        category="storage",
        pattern=re.compile(r"no AWS credentials detected"),
        description="No AWS credentials for S3 upload",
        fix="Provide credentials via the boto3 chain (env vars, ~/.aws/credentials, or IAM role). Never put secrets in YAML.",
    ),
    ErrorPattern(
        category="storage",
        pattern=re.compile(r"upload references unknown storage target"),
        description="Upload target not declared",
        fix="Declare a matching entry under top-level `storage:` or fix the upload `target:`.",
    ),
    ErrorPattern(
        category="storage",
        pattern=re.compile(r"upload 'to' must end with '/'"),
        description="Glob upload 'to' must be a directory",
        fix="End `to:` with '/' (or omit it) when `from:` is a glob.",
    ),
    # Batch/CSV
    ErrorPattern(
        category="batch",
        pattern=re.compile(r"CSV file must contain a 'sflow_config_file' column"),
        description="Missing required CSV column",
        fix="Add a 'sflow_config_file' column to the CSV with comma-separated YAML file paths.",
    ),
    ErrorPattern(
        category="batch",
        pattern=re.compile(r"CSV file is empty:\s*(.+)"),
        description="Empty CSV file",
        fix="Add data rows to the CSV.",
    ),
    ErrorPattern(
        category="batch",
        pattern=re.compile(r"CSV column '(.+?)' is not a variable or artifact"),
        description="Unknown CSV column",
        fix="Rename the column to match a declared variable/artifact, or declare it in the YAML.",
    ),
    ErrorPattern(
        category="batch",
        pattern=re.compile(r"ERRORS:\s*(\d+)\s*config\(s\) failed dry-run validation"),
        description="Bulk dry-run validation failures",
        fix="Fix each listed config. Run sflow run -f <file> --dry-run individually for details.",
    ),
    # CLI
    ErrorPattern(
        category="cli",
        pattern=re.compile(r"NotImplementedError.*--resume is not implemented"),
        description="--resume not implemented",
        fix="Re-run the full workflow instead. --resume is a planned future feature.",
    ),
    ErrorPattern(
        category="cli",
        pattern=re.compile(
            r"Selective task execution \(--task\) is not yet implemented"
        ),
        description="--task not implemented",
        fix="Run the full workflow. Use --missable-tasks to skip optional tasks.",
    ),
    ErrorPattern(
        category="cli",
        pattern=re.compile(r"--missable-tasks is only valid with multiple input files"),
        description="--missable-tasks with single file",
        fix="Use multiple -f flags for modular composition, or remove --missable-tasks.",
    ),
]

CATEGORY_LABELS = {
    "config": "Configuration Loading",
    "expression": "Expression Resolution",
    "override": "Variable/Artifact Overrides",
    "artifact": "Artifact Validation",
    "merge": "File Merge/Composition",
    "slurm": "SLURM Backend",
    "kubernetes": "Kubernetes Backend",
    "docker": "Docker Backend",
    "storage": "Storage / Uploads",
    "runtime": "Runtime / Task Execution",
    "batch": "Batch / CSV Processing",
    "cli": "CLI Arguments",
}

CATEGORY_PRIORITY = list(CATEGORY_LABELS.keys())


@dataclass
class MatchedError:
    line_num: int
    line: str
    pattern: ErrorPattern
    match: re.Match[str]


@dataclass
class ParseResult:
    source: str
    total_lines: int
    matched_errors: list[MatchedError] = field(default_factory=list)
    unmatched_error_lines: list[tuple[int, str]] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.matched_errors) or bool(self.unmatched_error_lines)

    def to_dict(self) -> dict:
        by_category: dict[str, list[dict]] = {}
        for me in self.matched_errors:
            cat = me.pattern.category
            by_category.setdefault(cat, []).append(
                {
                    "line": me.line_num,
                    "description": me.pattern.description,
                    "text": me.line[:200],
                    "fix": me.pattern.fix,
                }
            )
        return {
            "source": self.source,
            "total_lines": self.total_lines,
            "has_errors": self.has_errors,
            "matched_errors": len(self.matched_errors),
            "unmatched_error_lines": len(self.unmatched_error_lines),
            "root_cause": {
                "line": self.matched_errors[0].line_num,
                "description": self.matched_errors[0].pattern.description,
                "fix": self.matched_errors[0].pattern.fix,
            }
            if self.matched_errors
            else None,
            "categories": {
                CATEGORY_LABELS.get(cat, cat): errors
                for cat, errors in by_category.items()
            },
            "summary": {
                CATEGORY_LABELS.get(cat, cat): len(errors)
                for cat, errors in by_category.items()
            },
        }


def parse_log(lines: list[str], source: str = "<stdin>") -> ParseResult:
    result = ParseResult(source=source, total_lines=len(lines))

    error_line_pattern = re.compile(
        r"^.*[\u2717✗]|^.*Error:|^.*ERROR:|^.*error:", re.IGNORECASE
    )

    for line_num, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if not stripped:
            continue

        matched = False
        for ep in ERROR_PATTERNS:
            m = ep.pattern.search(stripped)
            if m:
                result.matched_errors.append(
                    MatchedError(
                        line_num=line_num,
                        line=stripped,
                        pattern=ep,
                        match=m,
                    )
                )
                matched = True
                break

        if not matched and error_line_pattern.search(stripped):
            result.unmatched_error_lines.append((line_num, stripped))

    return result


def print_report(result: ParseResult) -> None:
    print(f"\n{'=' * 70}")
    print(f"  sflow Error Analysis: {result.source}")
    print(f"  Lines scanned: {result.total_lines}")
    print(f"{'=' * 70}")

    if not result.has_errors:
        print("\n  No errors detected.")
        print()
        return

    by_category: dict[str, list[MatchedError]] = {}
    for me in result.matched_errors:
        by_category.setdefault(me.pattern.category, []).append(me)

    root_cause = result.matched_errors[0] if result.matched_errors else None

    if root_cause:
        print("\n  ** Most likely root cause (first error found):")
        print(f"     Line {root_cause.line_num}: {root_cause.pattern.description}")
        print(f"     Fix: {root_cause.pattern.fix}")
        print()

    print(f"  Matched errors: {len(result.matched_errors)}")
    print(f"  Unmatched error-like lines: {len(result.unmatched_error_lines)}")

    if by_category:
        print("\n  Summary by category:")
        for cat in CATEGORY_PRIORITY:
            errors = by_category.get(cat, [])
            if errors:
                label = CATEGORY_LABELS.get(cat, cat)
                print(f"    {label}: {len(errors)}")
    print()

    for cat in CATEGORY_PRIORITY:
        errors = by_category.get(cat, [])
        if not errors:
            continue
        label = CATEGORY_LABELS.get(cat, cat)
        print(f"  [{label}] ({len(errors)} error(s))")
        print(f"  {'-' * 50}")
        for me in errors:
            print(f"    Line {me.line_num}: {me.pattern.description}")
            print(f"      Text: {me.line[:120]}{'...' if len(me.line) > 120 else ''}")
            print(f"      Fix:  {me.pattern.fix}")
            print()

    if result.unmatched_error_lines:
        print(
            f"  [Unmatched Error Lines] ({len(result.unmatched_error_lines)} line(s))"
        )
        print(f"  {'-' * 50}")
        for line_num, line in result.unmatched_error_lines[:10]:
            print(
                f"    Line {line_num}: {line[:120]}{'...' if len(line) > 120 else ''}"
            )
        if len(result.unmatched_error_lines) > 10:
            print(f"    ... and {len(result.unmatched_error_lines) - 10} more")
        print()


def main() -> int:
    json_mode = "--json" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--json"]

    if len(args) < 1:
        print(f"Usage: {sys.argv[0]} [--json] <log_file>", file=sys.stderr)
        print(
            f"       {sys.argv[0]} [--json] -          (read from stdin)",
            file=sys.stderr,
        )
        return 2

    filepath = args[0]

    if filepath == "-":
        lines = sys.stdin.readlines()
        source = "<stdin>"
    else:
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {filepath}", file=sys.stderr)
            return 1
        lines = path.read_text().splitlines()
        source = filepath

    result = parse_log(lines, source)

    if json_mode:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_report(result)

    return 1 if result.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
