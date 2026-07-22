#!/usr/bin/env python3
"""One-shot triage of a failed sflow run directory.

Chains the two per-run analyzers and consolidates them into a single diagnosis
so you go from "a task failed" to "here is the root cause, the fix, and the log
to open" in one command:

  1. summarize_run.py       -- per-task status table + failed-task tracebacks
  2. parse_sflow_errors.py  -- categorize the orchestrator log (sflow.log)
  3. parse_sflow_errors.py  -- each failed task's <task>/<task>.log (runtime cause:
                               CUDA OOM, NCCL, ConnectionRefused, ModuleNotFound, ...)

Usage:
    python triage.py <output_dir>/<run_id>/

    # find the latest run dir first, then triage it:
    #   ls -td sflow_output/*/ | head -1

Exit code: 0 when nothing failed, else 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def _task_log(task_dir: Path) -> Path | None:
    """The task's own <task>.log, else the first *.log in the dir."""
    named = task_dir / f"{task_dir.name}.log"
    if named.exists():
        return named
    logs = sorted(task_dir.glob("*.log"))
    return logs[0] if logs else None


def main() -> int:  # noqa: C901 - a linear pipeline reads clearest top to bottom
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Error: not a directory: {run_dir}", file=sys.stderr)
        print("  Pass the run dir: <output_dir>/<run_id>/", file=sys.stderr)
        return 2

    try:
        import parse_sflow_errors as pe  # sibling script
        import summarize_run as sr  # sibling script
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error: could not import sibling scripts: {exc}", file=sys.stderr)
        return 2

    # 1) Human-readable per-task status table (+ failed-task tracebacks).
    _banner("[1/3] Run summary (summarize_run)")
    summary_rc = sr.summarize_run(run_dir)

    # Collected root causes, best (most specific) first.
    suspects: list[tuple[str, object, Path]] = []  # (source_label, MatchedError, log_path)

    # 2) Orchestrator log -- the framework's own error (probe timeout, alloc, RBAC...).
    sflow_log = run_dir / "sflow.log"
    if sflow_log.exists():
        _banner("[2/3] Orchestrator errors (sflow.log)")
        result = pe.parse_log(
            sflow_log.read_text(errors="replace").splitlines(), source=str(sflow_log)
        )
        pe.print_report(result)
        if result.matched_errors:
            suspects.append(("orchestrator", result.matched_errors[0], sflow_log))
    else:
        print("\n  (no sflow.log found in run dir)")

    # 3) Per-task logs -- the *actual* runtime cause usually lives here.
    _banner("[3/3] Failed-task runtime errors (<task>/<task>.log)")
    task_dirs = sorted(
        d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    any_task_error = False
    for task_dir in task_dirs:
        log = _task_log(task_dir)
        if log is None:
            continue
        result = pe.parse_log(
            log.read_text(errors="replace").splitlines(), source=str(log)
        )
        if not result.matched_errors:
            continue
        any_task_error = True
        top = result.matched_errors[0]
        print(f"\n  [{task_dir.name}]  ({log})")
        print(f"    {top.pattern.description}")
        print(f"    Fix: {top.pattern.fix}")
        # Task-log runtime causes are the most specific -> put them first.
        suspects.insert(0, (task_dir.name, top, log))
    if not any_task_error:
        print("\n  (no categorized errors in task logs)")

    # Consolidated verdict.
    _banner("Diagnosis")
    if suspects:
        label, err, log_path = suspects[0]
        where = "task" if label != "orchestrator" else "orchestrator"
        print(f"  Primary suspect ({where}): {label}")
        print(f"  Root cause : {err.pattern.description}")
        print(f"  Fix        : {err.pattern.fix}")
        print(f"  Open next  : {log_path}")
        if len(suspects) > 1:
            print(f"\n  ({len(suspects) - 1} other categorized error(s) above)")
    else:
        print("  No categorized errors found.")
        print("  If a task still failed, open its <task>/<task>.log and read the tail,")
        print("  or re-run with --verbose. Probe timeouts often mean the readiness")
        print("  regex never matched -- check the server's actual log output.")
    print()

    return 1 if (summary_rc != 0 or suspects) else 0


if __name__ == "__main__":
    sys.exit(main())
