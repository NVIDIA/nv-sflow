# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote-side helpers and receipt commands for SSH delegation."""

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Annotated

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


# Everything above this line is copied to the remote host and run standalone
# as `remote_helper.py` (see _ssh_delegate.py's payload staging), on a host
# that has neither typer nor sflow installed. These imports only run locally,
# because the __main__ guard above exits first when this file executes
# remotely. A formatter or lint autofix that hoists them to the top of the
# file would import typer/sflow on the remote host and crash it.
import typer

from sflow.cli import DOCS_URL, app
from sflow.cli._ssh_delegate import (
    _fetch,
    _snapshot,
    _ssh_argv,
    _ssh_failure_hint,
)


def _load_receipt(path: Path) -> dict[str, object]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"could not read SSH receipt {path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise typer.BadParameter(f"invalid SSH receipt: {path}")
    for key in ("command", "connection", "remote_session"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise typer.BadParameter(f"SSH receipt is missing {key!r}: {path}")
    if not str(receipt["remote_session"]).startswith("/"):
        raise typer.BadParameter(f"SSH receipt has an invalid remote_session: {path}")
    return receipt


def _receipt_endpoints(data: dict[str, object]) -> tuple[list[str], str, str]:
    session = str(data["remote_session"])
    return (
        _ssh_argv(str(data["connection"])),
        f"{session}/remote_helper.py",
        f"{session}/output",
    )


@app.command(name="watch", epilog=f"Documentation: {DOCS_URL}")
def watch(
    receipt: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    once: Annotated[
        bool,
        typer.Option("--once", help="Print one snapshot and exit."),
    ] = False,
) -> None:
    """Follow logs from an SSH receipt; Ctrl-C does not cancel the remote run."""
    data = _load_receipt(receipt)
    ssh, helper, output = _receipt_endpoints(data)
    if once:
        _snapshot(ssh, helper, output, None, 0, connection_hint=True)
        return
    process = subprocess.Popen(
        [*ssh, f"exec python3 {shlex.quote(helper)} follow {shlex.quote(output)}"],
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    try:
        return_code = process.wait()
        if return_code == 255:
            _ssh_failure_hint(ssh)
        raise typer.Exit(code=return_code)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        typer.echo("Stopped watching; the remote run was not cancelled.", err=True)
        raise typer.Exit(code=130) from None


@app.command(name="fetch", epilog=f"Documentation: {DOCS_URL}")
def fetch(
    receipt: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Override the receipt destination."),
    ] = None,
    all_files: Annotated[
        bool,
        typer.Option("--all", help="Fetch the complete remote output directory."),
    ] = False,
) -> None:
    """Fetch current logs or complete output from an SSH receipt."""
    data = _load_receipt(receipt)
    command = str(data["command"])
    saved_output = data.get("local_output")
    saved_destination = output_dir is None
    if output_dir is None:
        if not isinstance(saved_output, str) or not saved_output:
            raise typer.BadParameter(
                "receipt has no local output; pass --output-dir explicitly"
            )
        output_dir = Path(saved_output)
    destination = output_dir.expanduser().resolve()
    fetch_destination = (
        destination.parent
        if command == "compose" and saved_destination
        else destination
    )
    mode = {"compose": "all", "batch": "batch"}.get(command, "logs")
    if all_files:
        mode = "all"
    ssh, helper, output = _receipt_endpoints(data)
    if not _fetch(
        ssh,
        helper,
        output,
        mode,
        fetch_destination,
    ):
        raise typer.Exit(code=1)
    typer.echo(f"Fetched remote output to {destination}", err=True)
