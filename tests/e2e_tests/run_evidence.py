# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keeping what a GREEN e2e run produced.

The docker e2e tests drive sflow into pytest's ``tmp_path``, which lives under
/tmp -- outside the CI job workspace, and wiped with the runner. So a passing e2e
used to leave nothing behind but a dot in the pytest output: no sflow summary, no
task logs, no way to confirm the pipeline sample actually did the four-task dance
it claims rather than exiting early on a technicality. A *failing* test at least
dumps its haystack into the assertion message; a passing one told you nothing.

Two halves, and neither replaces the other:

* :func:`archive_run_dir` keeps full fidelity -- every task log, the sflow
  summary, the generated YAML, the GPU reservation registry records -- under a
  directory CI can publish as an artifact, for when you need to dig.
* :func:`echo_run_evidence` puts the summary and each task log's tail in the job
  log itself, so the common question ("did it really run all four tasks?") is
  answered by scrolling, without downloading anything.

Kept in its own module rather than inside the e2e test file so it can be unit
tested: that test module shells out to ``docker info`` and ``nvidia-smi`` at
import time, which a unit test has no business triggering.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Per task log, in the echo only. The archive keeps every line; this is the "did
# it get where it was going" glance, and an unbounded tail would bury the summary
# under container chatter on a job that runs a dozen tasks.
ECHO_TAIL_LINES = 40

# Written by sflow itself; not a task's own log, so they are reported separately.
_DRIVER_LOGS = frozenset({"sflow_summary.log", "sflow.log"})


def _section(title: str, body: str) -> str:
    return f"\n----- {title} -----\n{body.rstrip()}"


def render_run_evidence(root: Path, test_name: str, *, tail: int = ECHO_TAIL_LINES) -> str:
    """The sflow summary plus each task log's tail, as one printable block.

    Returns ``""`` when the tree holds no sflow output at all, so a caller can
    stay quiet for tests that never ran a workflow. Reads whatever sflow actually
    wrote rather than a curated subset: a run that produced *no* summary is itself
    worth seeing in the log.
    """
    root = Path(root)
    if not root.exists():
        return ""
    summaries = sorted(root.rglob("sflow_summary.log"))
    task_logs = [p for p in sorted(root.rglob("*.log")) if p.name not in _DRIVER_LOGS]
    if not summaries and not task_logs:
        return ""

    out = [f"\n=== run evidence: {test_name} ==="]
    for path in summaries:
        try:
            out.append(_section(f"sflow summary ({path.parent.name})", path.read_text(errors="replace")))
        except OSError as e:  # pragma: no cover - unreadable artifact
            out.append(_section(f"sflow summary ({path})", f"<unreadable: {e}>"))
    for path in task_logs:
        title = f"task log {path.parent.name}/{path.name}"
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as e:  # pragma: no cover - unreadable artifact
            out.append(_section(title, f"<unreadable: {e}>"))
            continue
        # Not `lines[-tail:]`: at tail=0 that is `lines[0:]`, i.e. the whole file --
        # the opposite of what "keep 0 lines" asks for. Negative means "no limit".
        shown = lines if tail < 0 else lines[len(lines) - min(tail, len(lines)) :]
        if len(shown) < len(lines):
            title = f"{title} (last {len(shown)} of {len(lines)} lines)"
        out.append(_section(title, "\n".join(shown)))
    return "\n".join(out)


def echo_run_evidence(root: Path, test_name: str) -> None:
    """Print :func:`render_run_evidence`, or nothing when there is none."""
    rendered = render_run_evidence(root, test_name)
    if rendered:
        print(rendered)


def archive_run_dir(root: Path, test_name: str, artifact_root: str | None) -> Path | None:
    """Copy one test's whole run tree under ``artifact_root``; return the destination.

    Whole tree rather than a filtered copy: the generated YAML and the GPU
    reservation registry records are as much a part of the evidence as the logs,
    and a filter would quietly drop whatever a future test starts writing. These
    are text logs from short container runs, so size is not a concern.

    Returns ``None`` when archiving is off (no ``artifact_root``, as on a
    developer box where tmp_path is already on disk) or when there is nothing to
    copy. Never raises: a bookkeeping failure must not turn a green e2e red.
    """
    root = Path(root)
    if not artifact_root or not root.exists():
        return None
    try:
        dest = Path(artifact_root) / test_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, dest, dirs_exist_ok=True)
        print(f"\n[e2e] archived run output for {test_name} -> {dest}")
        return dest
    except Exception as e:  # pragma: no cover - CI bookkeeping only
        print(f"\n[e2e] WARNING: could not archive run output for {test_name}: {e}")
        return None
