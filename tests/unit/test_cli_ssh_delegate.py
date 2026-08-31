import io
import json
import os
import shlex
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from sflow.cli import _ssh_delegate as ssh_delegate
from sflow.cli import app
from sflow.cli import remote as ssh_remote
from sflow.cli._ssh_delegate import (
    _copy_input,
    _directory_configs,
    _local_artifacts,
    _receipt_status,
    _requirements,
    _resolve_remote_root,
    _runtime_fingerprint,
    _snapshot,
    _ssh_argv,
    _ssh_failure_hint,
    _tty_argv,
    _without_options,
    predispatch,
)
from sflow.cli.remote import _snapshot_text, fetch, watch


def test_ssh_connection_accepts_options_after_host() -> None:
    assert _ssh_argv("user@login -p 20 -i 'my key'") == [
        "ssh",
        "-p",
        "20",
        "-i",
        "my key",
        "user@login",
    ]


@pytest.mark.parametrize(
    "connection",
    [
        "login -N",
        "login -L 8080:localhost:80",
        "login -o StdinNull=yes",
    ],
)
def test_ssh_connection_rejects_options_that_break_transfer(connection: str) -> None:
    with pytest.raises(typer.BadParameter, match="incompatible"):
        _ssh_argv(connection)


@pytest.mark.parametrize("connection", ["login -t", "login -tt"])
def test_ssh_connection_points_tty_requests_at_the_dedicated_option(
    connection: str,
) -> None:
    # -t on --ssh would also reach the payload upload and the tar fetch, so the
    # PTY has to be requested through --ssh-tty (execution channel only).
    with pytest.raises(typer.BadParameter, match="--ssh-tty"):
        _ssh_argv(connection)


class _Terminal:
    def isatty(self) -> bool:
        return True


class _Pipe:
    def isatty(self) -> bool:
        return False


@pytest.mark.parametrize(
    "mode, command, interactive, expected",
    [
        ("auto", "run", True, ["ssh", "-p", "22", "-tt", "login"]),
        ("auto", "run", False, ["ssh", "-p", "22", "login"]),
        # compose writes YAML on stdout, so auto must not put a PTY in the way.
        ("auto", "compose", True, ["ssh", "-p", "22", "login"]),
        ("always", "compose", False, ["ssh", "-p", "22", "-tt", "login"]),
        ("never", "run", True, ["ssh", "-p", "22", "login"]),
    ],
)
def test_tty_argv_allocates_a_pty_only_for_the_execution_channel(
    monkeypatch, mode: str, command: str, interactive: bool, expected: list[str]
) -> None:
    stream = _Terminal() if interactive else _Pipe()
    monkeypatch.setattr("sflow.cli._ssh_delegate.sys.stdin", stream)
    monkeypatch.setattr("sflow.cli._ssh_delegate.sys.stdout", stream)

    assert _tty_argv(["ssh", "-p", "22", "login"], mode, command) == expected


def test_tty_argv_survives_a_detached_stdio_stream(monkeypatch) -> None:
    class _Closed:
        def isatty(self) -> bool:
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr("sflow.cli._ssh_delegate.sys.stdin", _Closed())
    monkeypatch.setattr("sflow.cli._ssh_delegate.sys.stdout", _Closed())

    assert _tty_argv(["ssh", "login"], "auto", "run") == ["ssh", "login"]


def test_tui_forces_a_pty_over_an_explicit_never(capsys) -> None:
    assert ssh_delegate._tty_mode("never", ["flow.yaml", "--tui"]) == "always"
    assert "--tui needs a remote PTY" in capsys.readouterr().err

    assert ssh_delegate._tty_mode("never", ["flow.yaml"]) == "never"
    assert ssh_delegate._tty_mode("auto", ["flow.yaml", "--tui"]) == "auto"


@pytest.mark.parametrize(
    "follow, live_stream, expected",
    [
        # A PTY means the remote sflow streams task logs itself (and may render
        # a TUI), so nothing else may write to the terminal.
        ("auto", True, "none"),
        # Without a PTY the remote offloads task logs to files; the tail is the
        # only way to see them while attached.
        ("auto", False, "logs"),
        ("logs", True, "logs"),
        ("none", False, "none"),
        ("status", True, "status"),
    ],
)
def test_auto_follow_leaves_one_writer_on_the_terminal(
    follow: str, live_stream: bool, expected: str
) -> None:
    assert ssh_delegate._resolve_follow(follow, live_stream=live_stream) == expected


def test_staged_report_shows_workspace_relative_inputs_and_caps_the_list() -> None:
    staged = {
        Path("/work/flow.yaml"): Path("flow.yaml"),
        Path("/elsewhere/shared.yaml"): Path("_external/shared.yaml"),
        **{Path(f"/work/extra{n}.yaml"): Path(f"extra{n}.yaml") for n in range(5)},
    }

    sampled, remaining = ssh_delegate._staged_report(
        staged, Path("/work"), "/remote/work"
    )

    assert sampled[0] == ("flow.yaml", "/remote/work/flow.yaml")
    # Outside the workspace there is no useful relative name, so stay absolute.
    assert sampled[1] == (
        "/elsewhere/shared.yaml",
        "/remote/work/_external/shared.yaml",
    )
    assert len(sampled) == 5
    assert remaining == 2


def test_delegate_reports_stages_and_remote_workspace_facts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # Stop right before the upload: stages 1-2 and their facts must already be out.
    monkeypatch.setattr(
        ssh_delegate,
        "_resolve_remote_environment",
        lambda *args: ("/remote/root", "cpython-312-Linux-x86_64"),
    )
    monkeypatch.setattr(
        ssh_delegate, "_invocation_id", lambda: "ssh-19700101-000000-abababab"
    )
    monkeypatch.setattr(
        ssh_delegate.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("stop")),
    )
    (tmp_path / "flow.yaml").write_text("version: '0.1'\n")
    monkeypatch.setattr("sys.argv", ["sflow", "run", "flow.yaml", "--ssh", "login"])

    with pytest.raises(typer.BadParameter):
        ssh_delegate.delegate(
            "run",
            connection="login",
            follow="none",
            fetch="none",
            remote_root="/remote/root",
            workspace_dir=tmp_path,
            output_dir=None,
            input_files=[tmp_path / "flow.yaml"],
        )

    stderr = capsys.readouterr().err
    session = "/remote/root/runs/ssh-19700101-000000-abababab"
    assert "[ssh 1/4] connecting to login..." in stderr
    assert "[ssh 2/4] staging 1 input(s)" in stderr
    assert f"remote session: {session}" in stderr
    assert f"remote work dir: {session}/work" in stderr
    assert f"remote output dir: {session}/output" in stderr
    assert f"staged: flow.yaml -> {session}/work/flow.yaml" in stderr
    assert "[ssh 3/4] preparing the remote runtime..." in stderr
    assert "remote runtime: /remote/root/runtimes/" in stderr


@pytest.mark.parametrize("return_code, shows_hint", [(255, True), (1, False)])
def test_remote_probe_suggests_mfa_only_for_ssh_failure(
    monkeypatch, capsys, return_code: int, shows_hint: bool
) -> None:
    monkeypatch.setattr("sflow.cli._ssh_delegate._SSH_FAILURE_HINT_SHOWN", False)
    monkeypatch.setattr(
        "sflow.cli._ssh_delegate.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, return_code, stdout="", stderr="connection failed"
        ),
    )

    with pytest.raises(typer.BadParameter):
        _resolve_remote_root(["ssh", "user@login"], None)

    stderr = capsys.readouterr().err
    assert ("ControlMaster" in stderr) is shows_hint
    assert ("MFA" in stderr) is shows_hint


def test_mfa_hint_prints_copyable_commands_with_connection_options(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr("sflow.cli._ssh_delegate._SSH_FAILURE_HINT_SHOWN", False)

    _ssh_failure_hint(
        [
            "ssh",
            "-p",
            "20",
            "-J",
            "bastion",
            "-i",
            "my key",
            "-M",
            "-o",
            "BatchMode=yes",
            "-S",
            "/old/socket",
            "user@login",
        ]
    )

    stderr = capsys.readouterr().err
    assert "ssh -p 20 -J bastion -i 'my key' -MNf" in stderr
    assert "-o BatchMode=no" in stderr
    assert "Then rerun sflow using:" in stderr
    assert "--ssh" in stderr
    assert "/old/socket" not in stderr


def test_delegate_options_are_removed_from_remote_argv() -> None:
    assert _without_options(
        ["flow.yaml", "--ssh", "host -p 20", "--ssh-fetch=all", "--dry-run"],
        {"--ssh", "--ssh-fetch"},
    ) == ["flow.yaml", "--dry-run"]


def test_unchanged_log_snapshot_is_still_printed(monkeypatch, capsys) -> None:
    result = subprocess.CompletedProcess([], 0, stdout="same tail\n", stderr="")
    monkeypatch.setattr(
        "sflow.cli._ssh_delegate.subprocess.run", lambda *a, **k: result
    )

    previous, unchanged = _snapshot(["ssh", "login"], "/helper", "/out", None, 0)
    _snapshot(["ssh", "login"], "/helper", "/out", previous, unchanged)

    stderr = capsys.readouterr().err
    assert stderr.count("same tail") == 2
    assert "unchanged for 5s" in stderr


def test_snapshot_does_not_read_the_entire_log(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "task" / "task.log"
    log.parent.mkdir()
    log.write_text("".join(f"line-{index}\n" for index in range(100)))
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot must not read the whole log")
        ),
    )

    snapshot = _snapshot_text(tmp_path)

    assert "line-90" in snapshot
    assert "line-99" in snapshot
    assert "line-89" not in snapshot


def test_snapshot_ssh_does_not_consume_local_stdin() -> None:
    result = subprocess.CompletedProcess([], 0, stdout="tail\n", stderr="")
    with patch(
        "sflow.cli._ssh_delegate.subprocess.run", return_value=result
    ) as run_process:
        _snapshot(["ssh", "login"], "/helper", "/out", None, 0)

    assert run_process.call_args.kwargs["stdin"] is subprocess.DEVNULL


def test_only_relative_non_inline_artifacts_are_staged(tmp_path: Path) -> None:
    (tmp_path / "helper.sh").write_text("echo ok\n")
    config = tmp_path / "flow.yaml"
    config.write_text(
        """
artifacts:
  - name: helper
    uri: file://helper.sh
  - name: generated
    uri: file://generated.sh
    content: echo generated
  - name: cluster_data
    uri: fs:///shared/model
"""
    )

    assert _local_artifacts(config, tmp_path) == [tmp_path / "helper.sh"]


def test_relative_artifact_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    config = workspace / "flow.yaml"
    config.write_text("artifacts:\n  secret:\n    uri: file://../secret.txt\n")

    with pytest.raises(typer.BadParameter, match="outside --workspace-dir"):
        _local_artifacts(config, workspace)


def test_input_directory_does_not_follow_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(tmp_path / "outside")

    with pytest.raises(typer.BadParameter, match="symlink"):
        _copy_input(source, tmp_path, tmp_path / "stage")


def test_input_directory_skips_local_metadata_and_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    (source / ".git" / "secret").write_text("no")
    (source / ".venv").mkdir()
    (source / "sflow_output").mkdir()
    (source / "keep.txt").write_text("yes")

    relative = _copy_input(source, tmp_path, tmp_path / "stage")
    copied = tmp_path / "stage" / relative

    assert (copied / "keep.txt").read_text() == "yes"
    assert not (copied / ".git").exists()
    assert not (copied / ".venv").exists()
    assert not (copied / "sflow_output").exists()


def test_ignored_directory_cannot_be_uploaded_as_root(tmp_path: Path) -> None:
    source = tmp_path / ".git"
    source.mkdir()

    with pytest.raises(typer.BadParameter, match="ignored directory"):
        _copy_input(source, tmp_path, tmp_path / "stage")


def test_directory_config_scan_skips_ignored_directories(tmp_path: Path) -> None:
    (tmp_path / "keep.yaml").write_text("version: '0.1'\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "secret.yaml").write_text("secret: true\n")
    (tmp_path / "sflow_output").mkdir()
    (tmp_path / "sflow_output" / "old.yml").write_text("old: true\n")

    assert _directory_configs(tmp_path) == [tmp_path / "keep.yaml"]


def test_receipt_can_be_watched_once(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        '{"command":"batch","connection":"user@login -p 22",'
        '"remote_session":"/remote/run","local_output":"/local/out"}'
    )

    with patch("sflow.cli.remote._snapshot", return_value=("tail", 0)) as snapshot:
        watch(receipt, once=True)

    assert snapshot.call_args.args[:3] == (
        ["ssh", "-p", "22", "user@login"],
        "/remote/run/remote_helper.py",
        "/remote/run/output",
    )


def test_receipt_watch_once_preserves_ssh_failure(tmp_path: Path, monkeypatch) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        '{"command":"run","connection":"user@login","remote_session":"/remote/run"}'
    )
    monkeypatch.setattr(
        "sflow.cli._ssh_delegate.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, returncode=23, stdout="", stderr="remote failure"
        ),
    )

    with pytest.raises(typer.Exit) as exc:
        watch(receipt, once=True)

    assert exc.value.exit_code == 23


def test_receipt_watch_interrupt_returns_130(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        '{"command":"run","connection":"user@login","remote_session":"/remote/run"}'
    )
    with patch("sflow.cli.remote.subprocess.Popen") as popen:
        popen.return_value.wait.side_effect = [KeyboardInterrupt, 0]
        with pytest.raises(typer.Exit) as exc:
            watch(receipt)

    assert exc.value.exit_code == 130
    popen.return_value.terminate.assert_called_once_with()


def test_receipt_fetch_reuses_saved_destination(tmp_path: Path) -> None:
    output = tmp_path / "out"
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "command": "run",
                "connection": "user@login",
                "remote_session": "/remote/run",
                "local_output": str(output),
            }
        )
    )

    with patch("sflow.cli.remote._fetch", return_value=True) as fetch_output:
        fetch(receipt)

    assert fetch_output.call_args.args == (
        ["ssh", "user@login"],
        "/remote/run/remote_helper.py",
        "/remote/run/output",
        "logs",
        output,
    )


@pytest.mark.parametrize("override", [False, True])
def test_compose_receipt_fetch_destination(tmp_path: Path, override: bool) -> None:
    saved_output = tmp_path / "composed.yaml"
    output_dir = tmp_path / "recovered"
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "command": "compose",
                "connection": "user@login",
                "remote_session": "/remote/run",
                "local_output": str(saved_output),
            }
        )
    )

    with patch("sflow.cli.remote._fetch", return_value=True) as fetch_output:
        fetch(receipt, output_dir=output_dir if override else None)

    assert fetch_output.call_args.args[-1] == (output_dir if override else tmp_path)


@pytest.mark.parametrize(
    ("return_code", "status"), [(0, "completed"), (1, "failed"), (255, "unknown")]
)
def test_receipt_status_preserves_unknown_ssh_state(
    return_code: int, status: str
) -> None:
    assert _receipt_status(return_code) == status


def test_interrupted_attached_run_has_unknown_remote_state() -> None:
    assert _receipt_status(130) == "unknown"


def test_successful_batch_submission_is_not_workflow_completion() -> None:
    assert ssh_delegate._receipt_outcome("batch", 0, ["flow.yaml", "--submit"]) == {
        "status": "submitted",
        "execution_status": "completed",
        "backend_status": "submitted",
    }


def test_invocation_ids_are_sortable_and_do_not_depend_on_pid(monkeypatch) -> None:
    # The timestamp orders the folders; uniqueness comes from the random tail,
    # never from the PID or the clock alone (both collide across clients).
    monkeypatch.setattr(ssh_delegate.secrets, "token_hex", lambda size: "ab" * size)
    epoch = time.gmtime(0)
    monkeypatch.setattr(ssh_delegate.time, "gmtime", lambda *args: epoch)

    assert ssh_delegate._invocation_id() == "ssh-19700101-000000-abababab"


def test_runtime_fingerprint_includes_python_platform(monkeypatch) -> None:
    requirements = ["example==1"]
    monkeypatch.setattr(ssh_delegate.platform, "machine", lambda: "arch-one")
    first = _runtime_fingerprint(requirements)
    monkeypatch.setattr(ssh_delegate.platform, "machine", lambda: "arch-two")

    assert _runtime_fingerprint(requirements) != first


def test_receipt_is_written_atomically_with_private_mode(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"

    ssh_delegate._write_receipt(receipt, {"status": "running"})
    ssh_delegate._write_receipt(receipt, {"status": "completed"})

    assert json.loads(receipt.read_text()) == {"status": "completed"}
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600


def _build_archive(archive_path: Path, member: tarfile.TarInfo, data: bytes = b"") -> Path:
    with tarfile.open(archive_path, "w:gz") as archive:
        if member.isfile():
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        else:
            archive.addfile(member)
    return archive_path


def test_safe_extract_rejects_path_traversal_member(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    member = tarfile.TarInfo(name="../escape.txt")
    archive_path = _build_archive(tmp_path / "archive.tar.gz", member, b"pwned")

    with pytest.raises(RuntimeError, match="unsafe path"):
        ssh_delegate._safe_extract(archive_path, destination)

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_rejects_symlink_member(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    member = tarfile.TarInfo(name="link")
    member.type = tarfile.SYMTYPE
    member.linkname = "/etc/passwd"
    archive_path = _build_archive(tmp_path / "archive.tar.gz", member)

    with pytest.raises(RuntimeError, match="unsafe file type"):
        ssh_delegate._safe_extract(archive_path, destination)

    assert not (destination / "link").exists()


def test_safe_extract_rejects_non_regular_member(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    member = tarfile.TarInfo(name="fifo")
    member.type = tarfile.FIFOTYPE
    archive_path = _build_archive(tmp_path / "archive.tar.gz", member)

    with pytest.raises(RuntimeError, match="unsafe file type"):
        ssh_delegate._safe_extract(archive_path, destination)

    assert not (destination / "fifo").exists()


def test_rewrite_staged_paths_rewrites_workspace_relative_and_prefixed_values(
    tmp_path: Path,
) -> None:
    workspace = tmp_path
    input_path = workspace / "data.csv"
    input_path.write_text("a,b\n")
    staged = {input_path.resolve(): Path("data.csv")}

    rewritten = ssh_delegate._rewrite_staged_paths(
        ["run.yaml", "--input=data.csv", "--unrelated", "value"],
        staged,
        workspace,
        "/remote/work",
    )

    assert rewritten == [
        "run.yaml",
        "--input=/remote/work/data.csv",
        "--unrelated",
        "value",
    ]


def test_load_receipt_rejects_missing_required_field(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"command": "run", "connection": "user@host"}))

    with pytest.raises(typer.BadParameter, match="remote_session"):
        ssh_remote._load_receipt(receipt_path)


def test_load_receipt_rejects_non_dict_json(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("[]")

    with pytest.raises(typer.BadParameter, match="invalid SSH receipt"):
        ssh_remote._load_receipt(receipt_path)


def test_fetch_merge_rejects_existing_destination_symlink(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    destination = tmp_path / "destination"
    outside = tmp_path / "outside"
    (staged / "linked").mkdir(parents=True)
    (staged / "linked" / "result.log").write_text("remote")
    destination.mkdir()
    outside.mkdir()
    (destination / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        ssh_delegate._merge_tree(staged, destination)

    assert not (outside / "result.log").exists()


def _record_controller(session: Path, code: str) -> subprocess.Popen:
    session.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    (session / "controller.pid").write_text(f"{process.pid}\n")
    return process


def test_unconfirmed_cancel_keeps_the_state_unknown(monkeypatch, capsys) -> None:
    # An unreachable host during the cancel must not be reported as a cancel:
    # the remote controller may well still be running.
    monkeypatch.setattr("sflow.cli._ssh_delegate._SSH_FAILURE_HINT_SHOWN", False)
    monkeypatch.setattr(
        ssh_delegate.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 255),
    )

    assert not ssh_delegate._cancel_remote(["ssh", "login"], "/helper", "/session")
    assert "ControlMaster" in capsys.readouterr().err


def test_cancel_gives_up_on_a_second_interrupt(monkeypatch, capsys) -> None:
    def _interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(ssh_delegate.subprocess, "run", _interrupted)

    assert not ssh_delegate._cancel_remote(["ssh", "login"], "/helper", "/session")
    assert "Stopped waiting for the remote cancel" in capsys.readouterr().err


def test_confirmed_cancel_is_reported_as_cancelled(monkeypatch) -> None:
    monkeypatch.setattr(
        ssh_delegate.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0),
    )

    assert ssh_delegate._cancel_remote(["ssh", "login"], "/helper", "/session")


def test_cancel_stops_the_remote_controller_with_sigint(
    tmp_path: Path, capsys, fake_process
) -> None:
    fake_process.allow_unregistered(True)
    # SIGINT first so sflow can run its own teardown instead of being killed.
    process = _record_controller(tmp_path, "import time; time.sleep(60)")
    try:
        ssh_remote._cancel(tmp_path)
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    output = capsys.readouterr().out
    assert "sending SIGINT" in output
    assert "remote controller stopped" in output
    assert "SIGKILL" not in output


def test_cancel_escalates_to_kill_when_the_controller_ignores_signals(
    tmp_path: Path, monkeypatch, capsys, fake_process
) -> None:
    fake_process.allow_unregistered(True)
    monkeypatch.setattr(ssh_remote, "_CANCEL_INT_GRACE", 0.5)
    monkeypatch.setattr(ssh_remote, "_CANCEL_TERM_GRACE", 0.5)
    monkeypatch.setattr(ssh_remote, "_CANCEL_KILL_GRACE", 5.0)
    ready = tmp_path / "handlers-installed"
    process = _record_controller(
        tmp_path,
        "import signal, time, pathlib;"
        "signal.signal(signal.SIGINT, signal.SIG_IGN);"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        f"pathlib.Path({str(ready)!r}).touch();"
        "time.sleep(60)",
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready.exists(), "the test controller never installed its handlers"
    try:
        ssh_remote._cancel(tmp_path)
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    output = capsys.readouterr().out
    assert "sending SIGTERM" in output
    assert "sending SIGKILL" in output
    assert "remote controller stopped" in output


def test_cancel_scancels_only_the_jobs_this_session_submitted(
    tmp_path: Path, monkeypatch, capsys, fake_process
) -> None:
    fake_process.allow_unregistered(True)
    session = tmp_path / "session"
    session.mkdir()
    (session / "submitted-jobs").write_text("101\n102\nnot-a-job\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "scancel-calls"
    scancel = fake_bin / "scancel"
    scancel.write_text(f'#!/bin/sh\necho "$@" >> {shlex.quote(str(calls))}\n')
    scancel.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    ssh_remote._cancel(session)

    assert calls.read_text().split() == ["101", "102"]
    assert "no remote controller recorded" in capsys.readouterr().out


def test_cancel_reports_a_controller_it_cannot_stop(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(ssh_remote, "_CANCEL_INT_GRACE", 0.1)
    monkeypatch.setattr(ssh_remote, "_CANCEL_TERM_GRACE", 0.1)
    monkeypatch.setattr(ssh_remote, "_CANCEL_KILL_GRACE", 0.1)
    (tmp_path / "controller.pid").write_text("4242\n")
    monkeypatch.setattr(ssh_remote, "_alive", lambda pid: True)
    monkeypatch.setattr(ssh_remote.os, "killpg", lambda *args: None)
    monkeypatch.setattr(ssh_remote.os, "getpgid", lambda pid: pid)

    # A non-zero exit is what tells the local side to keep the state "unknown".
    with pytest.raises(SystemExit):
        ssh_remote._cancel(tmp_path)


def test_failed_runtime_bootstrap_does_not_leave_a_stale_lock(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    source = tmp_path / "source"
    source.mkdir()
    failure = subprocess.CalledProcessError(1, ["python", "-m", "venv"])

    with (
        patch("sflow.cli.remote.subprocess.run", side_effect=failure),
        pytest.raises(subprocess.CalledProcessError),
    ):
        ssh_remote._bootstrap(runtime, source, "1.0")

    runtime.mkdir()
    (runtime / ".ready").touch()
    with patch("sflow.cli.remote.subprocess.run") as run_process:
        ssh_remote._bootstrap(runtime, source, "1.0")
    run_process.assert_not_called()


def test_default_fetch_excludes_unrelated_log_files(tmp_path: Path) -> None:
    run = tmp_path / "run-id"
    task = run / "task"
    task.mkdir(parents=True)
    summary = run / "sflow_summary.log"
    task_log = task / "task.log"
    unrelated = run / "credentials.log"
    for path in (summary, task_log, unrelated):
        path.write_text("data")

    assert ssh_remote._included_output(summary, tmp_path, "logs")
    assert ssh_remote._included_output(task_log, tmp_path, "logs")
    assert not ssh_remote._included_output(unrelated, tmp_path, "logs")


def test_default_fetch_includes_the_structured_result_contract(tmp_path: Path) -> None:
    run = tmp_path / "run-id"
    task = run / "task"
    task.mkdir(parents=True)
    task_result = task / "result.json"
    workflow_results = run / "results.json"
    unrelated_json = task / "config.json"
    for path in (task_result, workflow_results, unrelated_json):
        path.write_text("{}")

    assert ssh_remote._included_output(task_result, tmp_path, "logs")
    assert ssh_remote._included_output(workflow_results, tmp_path, "logs")
    assert not ssh_remote._included_output(unrelated_json, tmp_path, "logs")
    assert ssh_remote._included_output(unrelated_json, tmp_path, "batch")


def test_fetch_start_failure_is_recoverable(tmp_path: Path) -> None:
    with patch("sflow.cli._ssh_delegate.subprocess.run", side_effect=OSError("boom")):
        assert not ssh_delegate._fetch(
            ["ssh", "login"], "/helper", "/output", "logs", tmp_path / "out"
        )


@pytest.mark.parametrize("command", ["run", "batch", "compose"])
def test_cli_ssh_delegates_before_local_execution(command: str, tmp_path: Path) -> None:
    config = tmp_path / "flow.yaml"
    config.write_text("version: '1'\nworkflow: {name: test, tasks: []}\n")

    with patch(
        "sflow.cli._ssh_delegate.delegate", side_effect=typer.Exit(code=23)
    ) as delegate:
        result = CliRunner().invoke(app, [command, str(config), "--ssh", "login"])

    assert result.exit_code == 23
    assert delegate.call_args.args == (command,)
    assert delegate.call_args.kwargs["connection"] == "login"


def test_cli_ssh_missing_input_is_cli_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(
        "sys.argv", ["sflow", "run", "-f", str(missing), "--ssh", "login"]
    )
    with (
        patch(
            "sflow.cli._ssh_delegate.delegate", side_effect=FileNotFoundError(missing)
        ),
        pytest.raises(SystemExit) as exc,
    ):
        app()

    assert exc.value.code == 2
    assert f"Error: {missing}" in capsys.readouterr().err


def test_raw_predispatch_does_not_validate_remote_paths_locally(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "sflow",
            "run",
            "flow.yaml",
            "--kubeconfig",
            "/remote/only/config",
            "--ssh",
            "login -p 20",
        ],
    )
    with (
        patch("sflow.cli._ssh_delegate.delegate", side_effect=typer.Exit()) as delegate,
        pytest.raises(typer.Exit),
    ):
        predispatch()

    assert delegate.call_args.kwargs["connection"] == "login -p 20"
    assert Path("/remote/only/config") not in delegate.call_args.kwargs["input_files"]


def test_raw_predispatch_is_noop_without_ssh(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["sflow", "run", "flow.yaml", "--dry-run"])
    with patch("sflow.cli._ssh_delegate.delegate") as delegate:
        predispatch()
    delegate.assert_not_called()


def test_compose_round_trip_uses_staged_source(tmp_path: Path, fake_process) -> None:
    fake_process.allow_unregistered(True)
    remote_root = tmp_path / "remote"
    fingerprint = _runtime_fingerprint(_requirements())
    runtime = remote_root / "runtimes" / fingerprint
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "python").write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    (runtime / "bin" / "python").chmod(0o755)
    (runtime / ".ready").touch()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        '#!/bin/sh\nlast=\nfor arg in "$@"; do last=$arg; done\n'
        'exec /bin/sh -c "$last"\n'
    )
    fake_ssh.chmod(0o755)
    fake_python = fake_bin / "python3"
    fake_python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    fake_python.chmod(0o755)

    config = tmp_path / "flow.yaml"
    config.write_text(
        "version: '0.1'\n"
        "workflow:\n  name: smoke\n  tasks:\n"
        "    - name: hello\n      script:\n        - echo hello\n"
    )
    output = tmp_path / "composed.yaml"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sflow",
            "compose",
            str(config),
            "-o",
            str(output),
            "--ssh",
            "login",
            "--ssh-remote-root",
            str(remote_root),
            "--workspace-dir",
            str(tmp_path),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "name: smoke" in output.read_text()

    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sflow",
            "run",
            str(config),
            "--dry-run",
            "--ssh",
            "login",
            "--ssh-remote-root",
            str(remote_root),
            "--workspace-dir",
            str(tmp_path),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert run_result.returncode == 0, run_result.stderr

    slurm_config = tmp_path / "slurm.yaml"
    slurm_config.write_text(
        "version: '0.1'\n"
        "backends:\n  - name: slurm\n    type: slurm\n    default: true\n"
        "    partition: debug\n    account: test\n    nodes: 1\n"
        "    gpus_per_node: 1\n    time: '00:10:00'\n"
        "workflow:\n  name: batch-smoke\n  tasks:\n"
        "    - name: hello\n      script:\n        - echo hello\n"
    )
    batch_script = tmp_path / "job.sh"
    batch_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sflow",
            "batch",
            str(slurm_config),
            "--partition",
            "debug",
            "--account",
            "test",
            "--sbatch-path",
            str(batch_script),
            "--ssh",
            "login",
            "--ssh-remote-root",
            str(remote_root),
            "--workspace-dir",
            str(tmp_path),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert batch_result.returncode == 0, batch_result.stderr
    assert batch_script.is_file()
    assert batch_script.with_suffix(".yaml").is_file()
    assert "source-project" in batch_script.read_text()
    staged_projects = list(
        (remote_root / "runs").glob("*/source-project/pyproject.toml")
    )
    assert staged_projects
    assert "fallback_version" in staged_projects[-1].read_text()

    # Every SSH execution owns one local folder holding its receipt, named
    # exactly like its remote session so the two can be matched by eye.
    sessions = sorted((tmp_path / "sflow_output").glob("ssh-*"))
    assert len(sessions) == 3
    remote_sessions = {path.name for path in (remote_root / "runs").iterdir()}
    for session in sessions:
        receipt = json.loads((session / "receipt.json").read_text())
        assert receipt["session"] == session.name
        assert receipt["session_dir"] == str(session)
        assert session.name in remote_sessions
        assert receipt["remote_session"].endswith(f"/runs/{session.name}")
