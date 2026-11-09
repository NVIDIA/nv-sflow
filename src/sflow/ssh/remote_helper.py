# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The payload script SSH delegation runs on the remote host.

Copied next to the uploaded session as ``remote_helper.py`` and run by the
host's bare ``python3`` (see the payload staging in
:func:`sflow.ssh.delegate.delegate`), on a machine that has neither sflow nor
its dependencies installed. It must therefore import nothing from sflow and
nothing outside the standard library.
"""

import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_TAIL_LINES = 10
_TAIL_MAX_BYTES = 256 * 1024
# sflow's own SIGINT teardown has to unwind allocations, so give it real time
# before escalating; TERM/KILL are the backstop for a wedged controller.
_CANCEL_INT_GRACE = 20.0
_CANCEL_TERM_GRACE = 10.0
_CANCEL_KILL_GRACE = 5.0


def _included_output(path: Path, root: Path, mode: str) -> bool:
    if mode == "all":
        return True
    relative = path.relative_to(root)
    if mode == "batch" and path.suffix in {
        ".out",
        ".err",
        ".sh",
        ".yaml",
        ".yml",
        ".csv",
        ".json",
    }:
        return True
    return (
        path.name
        in {
            "sflow_summary.log",
            "sflow.log",
            "command_trace.jsonl",
            "loop_stalls.txt",
            "sflow_monitor.log",
            # The structured result contract (see core/results.py): dropping
            # these under the "logs" default would report success while the
            # only machine-readable record of it stays on the remote host.
            "result.json",
            "results.json",
            # sflow.utils.gpu.GPU_MARKER_FILE, duplicated: this script runs
            # standalone on the remote host and cannot import sflow.
            # tests/unit/test_cli_ssh_delegate.py asserts the two stay equal.
            "sflow_gpus.log",
        }
        or path.name.endswith("_cmds.log")
        or "sflow_monitor" in relative.parts
        or (path.suffix == ".log" and path.parent.name == path.stem)
    )


def _logs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    found = sorted(
        (
            path
            for path in root.rglob("*.log")
            if path.is_file()
            and not path.is_symlink()
            and _included_output(path, root, "logs")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    summary = next((path for path in found if path.name == "sflow_summary.log"), None)
    others = [path for path in found if path.name != "sflow_summary.log"][:2]
    return ([summary] if summary else []) + others


def _tail_text(path: Path) -> str:
    """Read a bounded tail instead of rescanning an ever-growing task log."""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        start = max(0, end - _TAIL_MAX_BYTES)
        stream.seek(start)
        data = stream.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-_TAIL_LINES:]) or "(empty)"


def _snapshot_text(root: Path) -> str:
    output = []
    for path in _logs(root):
        output.append("==> " + str(path.relative_to(root)) + " <==")
        try:
            output.append(_tail_text(path))
        except OSError as exc:
            output.append(f"(unreadable: {exc})")
    return "\n".join(output) or "(waiting for sflow logs)"


def _follow(root: Path) -> None:
    previous = None
    unchanged = 0
    while True:
        current = _snapshot_text(root)
        unchanged = unchanged + 5 if current == previous else 0
        suffix = f", unchanged for {unchanged}s" if unchanged else ""
        print(f"\n[ssh {time.strftime('%H:%M:%S')}{suffix}]\n{current}", flush=True)
        previous = current
        time.sleep(5)


def _pack(root: Path, mode: str) -> None:
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        if not root.exists():
            return
        for path in sorted(
            path for path in root.rglob("*") if path.is_file() and not path.is_symlink()
        ):
            rel = path.relative_to(root)
            if not _included_output(path, root, mode):
                continue
            archive.add(path, arcname=str(rel), recursive=False)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    """A signalled process still answers ``kill(0)`` until its parent reaps it."""
    try:
        state = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return False
    return state.startswith("Z")


def _signal_and_wait(pid: int, signal_number: int, grace: float) -> bool:
    """Signal the controller's process group, then wait for it to go.

    The group is what reaches the tasks the controller spawned. Signalling our
    own group instead would kill this cancel helper mid-escalation, so that case
    falls back to the controller pid alone.
    """
    try:
        group = os.getpgid(pid)
        if group in (os.getpgid(0), 0, 1):
            os.kill(pid, signal_number)
        else:
            os.killpg(group, signal_number)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal_number)
        except (ProcessLookupError, PermissionError):
            return not _alive(pid)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.25)
    return not _alive(pid)


def _scancel(session: Path) -> None:
    """Cancel the Slurm jobs this session submitted, and only those."""
    jobs_file = session / "submitted-jobs"
    try:
        jobs = [line.strip() for line in jobs_file.read_text().splitlines()]
    except OSError:
        return
    jobs = [job for job in jobs if job.isdigit()]
    if not jobs:
        return
    if shutil.which("scancel") is None:
        print(f"scancel not found; jobs left running: {' '.join(jobs)}", flush=True)
        return
    result = subprocess.run(
        ["scancel", *jobs], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        print(f"cancelled Slurm job(s): {' '.join(jobs)}", flush=True)
    else:
        print(
            f"scancel failed for {' '.join(jobs)}: {result.stderr.strip()}", flush=True
        )


def _cancel(session: Path) -> None:
    """Stop the controller this session started, escalating until it is gone.

    SIGINT first so sflow runs its own teardown (it cancels the allocations it
    owns); TERM and KILL only if that does not land. Jobs already handed to
    Slurm are cancelled separately -- they outlive the controller by design.
    """
    try:
        pid = int((session / "controller.pid").read_text().strip())
    except (OSError, ValueError):
        print("no remote controller recorded", flush=True)
        _scancel(session)
        return
    if not _alive(pid):
        print("remote controller already exited", flush=True)
        _scancel(session)
        return
    for signal_number, grace, label in (
        (signal.SIGINT, _CANCEL_INT_GRACE, "SIGINT"),
        (signal.SIGTERM, _CANCEL_TERM_GRACE, "SIGTERM"),
        (signal.SIGKILL, _CANCEL_KILL_GRACE, "SIGKILL"),
    ):
        print(f"sending {label} to remote controller {pid}", flush=True)
        if _signal_and_wait(pid, signal_number, grace):
            print("remote controller stopped", flush=True)
            _scancel(session)
            return
    print(f"remote controller {pid} is still running", flush=True)
    _scancel(session)
    raise SystemExit(1)


def _bootstrap(runtime: Path, source_project: Path, version: str) -> None:
    """Build one cached runtime under an automatically released file lock."""
    import fcntl

    runtime.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{runtime}.lock")
    with lock_path.open("a+") as lock:
        deadline = time.monotonic() + 300
        waited = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SystemExit("Timed out waiting for sflow runtime bootstrap")
                if not waited:
                    waited = True
                    print(
                        "    another sflow run is building this runtime; waiting",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(1)
        if (runtime / ".ready").is_file():
            print(f"    cached runtime: {runtime}", file=sys.stderr, flush=True)
            return
        temporary = Path(
            tempfile.mkdtemp(prefix=f"{runtime.name}.tmp-", dir=runtime.parent)
        )
        try:
            environment = os.environ.copy()
            environment["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SFLOW"] = version
            uv = shutil.which("uv")
            print(
                f"    building runtime with {'uv' if uv else 'pip'} "
                "(first run for this dependency set)",
                file=sys.stderr,
                flush=True,
            )
            if uv:
                create = [uv, "venv", "--python", sys.executable, str(temporary)]
                install = [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(temporary / "bin" / "python"),
                    str(source_project),
                ]
            else:
                create = [sys.executable, "-m", "venv", str(temporary)]
                install = [
                    str(temporary / "bin" / "pip"),
                    "install",
                    "--disable-pip-version-check",
                    str(source_project),
                ]
            subprocess.run(create, env=environment, check=True)
            subprocess.run(install, env=environment, check=True)
            (temporary / ".ready").touch()
            if runtime.exists():
                shutil.rmtree(runtime)
            temporary.replace(runtime)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def _helper_main() -> None:
    action = sys.argv[1]
    root = Path(sys.argv[2])
    if action == "snapshot":
        print(_snapshot_text(root))
    elif action == "follow":
        _follow(root)
    elif action == "pack":
        _pack(root, sys.argv[3])
    elif action == "cancel":
        _cancel(root)
    elif action == "bootstrap":
        _bootstrap(root, Path(sys.argv[3]), sys.argv[4])
    else:
        raise SystemExit("unknown action")


if __name__ == "__main__":
    _helper_main()
    raise SystemExit
