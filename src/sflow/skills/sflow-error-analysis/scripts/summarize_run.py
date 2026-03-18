#!/usr/bin/env python3
"""Summarize an sflow run's output directory.

Usage:
    python summarize_run.py <output_dir>/<run_id>/
    python summarize_run.py sflow_output/my_workflow_20260318-120000/

Shows:
    - Task status (pass/fail/running based on log contents)
    - Log file sizes and last lines
    - Tracebacks and error summaries
    - Output JSON contents if present
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


TRACEBACK_PATTERN = re.compile(r"Traceback \(most recent call last\)")
ERROR_PATTERN = re.compile(r"^.*Error:|^.*ERROR:", re.MULTILINE)
READY_PATTERNS = [
    re.compile(r"[Rr]eady|[Ss]erving|[Ll]istening|[Ss]tarted"),
]


def _tail_lines(filepath: Path, n: int = 5) -> list[str]:
    """Read the last N lines of a file efficiently."""
    try:
        text = filepath.read_text(errors="replace")
        lines = text.rstrip().split("\n")
        return lines[-n:]
    except OSError:
        return []


def _count_pattern(filepath: Path, pattern: re.Pattern) -> int:
    """Count matches of a pattern in a file."""
    try:
        text = filepath.read_text(errors="replace")
        return len(pattern.findall(text))
    except OSError:
        return 0


def _extract_last_traceback(filepath: Path) -> str | None:
    """Extract the last traceback from a log file."""
    try:
        text = filepath.read_text(errors="replace")
    except OSError:
        return None

    parts = TRACEBACK_PATTERN.split(text)
    if len(parts) < 2:
        return None

    tb_text = parts[-1]
    lines = tb_text.split("\n")
    tb_lines = ["Traceback (most recent call last):"]
    for line in lines:
        tb_lines.append(line)
        if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            if "Error" in line or "Exception" in line:
                break
    if len(tb_lines) > 20:
        tb_lines = tb_lines[:3] + ["    ..."] + tb_lines[-5:]
    return "\n".join(tb_lines)


def _file_size_human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def summarize_run(run_dir: Path) -> int:
    if not run_dir.is_dir():
        print(f"Error: not a directory: {run_dir}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 70}")
    print(f"  sflow Run Summary: {run_dir.name}")
    print(f"  Path: {run_dir}")
    print(f"{'=' * 70}")

    sflow_log = run_dir / "sflow.log"
    if sflow_log.exists():
        size = _file_size_human(sflow_log.stat().st_size)
        errors = _count_pattern(sflow_log, ERROR_PATTERN)
        print(f"\n  Orchestrator log: {size}, {errors} error line(s)")
    else:
        print("\n  Orchestrator log: not found")

    task_dirs = sorted(
        d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    if not task_dirs:
        print("  No task directories found.")
        print()
        return 0

    tasks_passed = 0
    tasks_failed = 0
    tasks_unknown = 0

    print(
        f"\n  {'Task':<35} {'Status':<10} {'Log Size':<10} {'Errors':<8} {'Last Line'}"
    )
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 40}")

    failed_details: list[tuple[str, str]] = []

    for task_dir in task_dirs:
        task_name = task_dir.name
        log_file = task_dir / f"{task_name}.log"

        if not log_file.exists():
            log_candidates = list(task_dir.glob("*.log"))
            log_file = log_candidates[0] if log_candidates else None

        if log_file is None or not log_file.exists():
            print(f"  {task_name:<35} {'no log':<10} {'-':<10} {'-':<8} -")
            tasks_unknown += 1
            continue

        size = _file_size_human(log_file.stat().st_size)
        tracebacks = _count_pattern(log_file, TRACEBACK_PATTERN)
        error_count = _count_pattern(log_file, ERROR_PATTERN)
        tail = _tail_lines(log_file, 1)
        last_line = tail[0][:40] if tail else "-"

        if tracebacks > 0:
            status = "FAILED"
            tasks_failed += 1
            tb = _extract_last_traceback(log_file)
            if tb:
                failed_details.append((task_name, tb))
        elif error_count > 0:
            status = "WARN"
            tasks_unknown += 1
        else:
            status = "OK"
            tasks_passed += 1

        print(f"  {task_name:<35} {status:<10} {size:<10} {error_count:<8} {last_line}")

    print(
        f"\n  Summary: {tasks_passed} OK, {tasks_failed} FAILED, {tasks_unknown} unknown/warn"
    )

    if failed_details:
        print(f"\n  {'=' * 60}")
        print("  Failed Task Details:")
        print(f"  {'=' * 60}")
        for task_name, tb in failed_details:
            print(f"\n  [{task_name}]")
            for line in tb.split("\n"):
                print(f"    {line}")

    outputs_files = list(run_dir.rglob("outputs.json"))
    if outputs_files:
        print(f"\n  {'=' * 60}")
        print("  Task Outputs:")
        print(f"  {'=' * 60}")
        for of in outputs_files:
            rel = of.relative_to(run_dir)
            try:
                data = json.loads(of.read_text())
                print(f"\n  [{rel}]")
                for k, v in data.items():
                    print(f"    {k}: {v}")
            except (json.JSONDecodeError, OSError):
                print(f"\n  [{rel}] (parse error)")

    print()
    return 1 if tasks_failed > 0 else 0


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_output_dir>", file=sys.stderr)
        print(
            f"  Example: {sys.argv[0]} sflow_output/my_workflow_20260318-120000/",
            file=sys.stderr,
        )
        return 2
    return summarize_run(Path(sys.argv[1]))


if __name__ == "__main__":
    sys.exit(main())
