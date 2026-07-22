#!/usr/bin/env python3
"""One-shot preflight gate for an sflow recipe -- run it BEFORE a real run.

Chains the static checks and the CLI dry-run into a single PASS/FAIL verdict so
you catch schema mistakes, GPU oversubscription, and expression/plan errors in
one command instead of three:

  1. validate_sflow_yaml.py  -- schema, references, artifact/volume + common-mistake checks
  2. check_gpu_plan.py       -- GPU allocation plan + oversubscription (advisory)
  3. sflow ... --dry-run     -- full plan + expression resolution (the source of truth)

On a dry-run failure the captured output is piped through parse_sflow_errors.py
(when it can be located) to surface the most likely root cause + fix.

Usage:
    python preflight.py <file.yaml> [<file2.yaml> ...] [sflow flags] [-- <extra sflow run args>]

    # examples
    python preflight.py my_workflow.yaml
    python preflight.py slurm.yaml common.yaml sglang/agg.yaml --set TP_SIZE=8
    python preflight.py agg.yaml --missable-tasks prefill_server -- --verbose

Any non-.yaml/.yml token (before an optional ``--``) and everything after ``--``
is forwarded verbatim to ``sflow run`` (e.g. ``--set``, ``--missable-tasks``,
``--kube-namespace``). ``sflow`` is invoked as ``<this-python> -m sflow`` so it
uses the same interpreter/venv that runs this script.

Exit code: 0 when validation has no errors AND the dry-run passes, else 1.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _is_yaml(token: str) -> bool:
    return token.lower().endswith((".yaml", ".yml"))


def _split_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Return (yaml_files, passthrough_sflow_args).

    Everything after a lone ``--`` is passthrough. Before it, ``*.yaml/*.yml``
    are recipe files and everything else is a passthrough sflow flag.
    """
    if "--" in argv:
        sep = argv.index("--")
        left, tail = argv[:sep], argv[sep + 1 :]
    else:
        left, tail = argv, []
    files = [a for a in left if _is_yaml(a)]
    passthrough = [a for a in left if not _is_yaml(a)] + tail
    return files, passthrough


def _banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def _run_validation(files: list[str]) -> bool:
    """Static schema validation. Returns True when no file has errors."""
    _banner("[1/3] Static validation (validate_sflow_yaml)")
    try:
        from validate_sflow_yaml import validate_file  # sibling script
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  (skipped: could not import validate_sflow_yaml: {exc})")
        return True
    all_ok = True
    for f in files:
        result = validate_file(f)
        result.print_report()
        all_ok = all_ok and result.ok
    return all_ok


def _run_gpu_plan(files: list[str]) -> None:
    """GPU allocation plan (advisory -- never blocks preflight)."""
    _banner("[2/3] GPU allocation plan (check_gpu_plan)")
    try:
        from check_gpu_plan import print_plan  # sibling script

        print_plan(files)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  (skipped: {exc})")


def _find_error_parser() -> Path | None:
    """Locate parse_sflow_errors.py (lives in the sibling error-analysis skill)."""
    candidates = [
        _SCRIPT_DIR / "parse_sflow_errors.py",
        _SCRIPT_DIR.parent.parent
        / "sflow-error-analysis"
        / "scripts"
        / "parse_sflow_errors.py",
    ]
    return next((c for c in candidates if c.is_file()), None)


def _run_dry_run(files: list[str], passthrough: list[str]) -> bool:
    """Invoke ``sflow run --dry-run``. Returns True on exit 0."""
    _banner("[3/3] sflow dry-run (plan + expression resolution)")
    cmd = [sys.executable, "-m", "sflow", "run"]
    for f in files:
        cmd += ["-f", f]
    cmd += passthrough + ["--dry-run"]
    print(f"  $ {' '.join(cmd)}\n")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    # Show a trimmed tail; the dry-run is verbose and the verdict is at the end.
    tail = output.strip().splitlines()[-25:]
    print("\n".join(tail))

    ok = proc.returncode == 0
    if not ok:
        parser = _find_error_parser()
        if parser is not None:
            _banner("Root-cause analysis (parse_sflow_errors)")
            subprocess.run(
                [sys.executable, str(parser), "-"],
                input=output,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
    return ok


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2

    files, passthrough = _split_args(argv)
    if not files:
        print("Error: no .yaml/.yml recipe file given.", file=sys.stderr)
        print(f"Usage: {Path(sys.argv[0]).name} <file.yaml> [more.yaml ...] "
              "[sflow flags]", file=sys.stderr)
        return 2

    missing = [f for f in files if not Path(f).exists()]
    if missing:
        print(f"Error: file(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    validation_ok = _run_validation(files)
    _run_gpu_plan(files)  # advisory only
    dry_run_ok = _run_dry_run(files, passthrough)

    _banner("Preflight verdict")
    print(f"  static validation : {'PASS' if validation_ok else 'FAIL (errors above)'}")
    print(f"  sflow dry-run     : {'PASS' if dry_run_ok else 'FAIL (see analysis above)'}")
    overall = validation_ok and dry_run_ok
    print(f"\n  {'READY to run.' if overall else 'NOT ready -- fix the above first.'}\n")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
