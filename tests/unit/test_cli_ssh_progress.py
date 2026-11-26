import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
import yaml
from rich.console import Console

from sflow.logging import configure_logging as configure_workflow_logging
from sflow.ssh import delegate as ssh_delegate
from sflow.ssh import progress, remote_helper
from sflow.ssh.invocation import argv_to_invocation


@pytest.mark.parametrize("level", ["INFO", "WARNING", "ERROR"])
def test_snapshot_preserves_remote_payload_at_every_progress_level(level, capsys):
    progress.configure_logging(level)
    payload = "ERROR: worker failed " + "x" * 240 + "\n  traceback detail"
    transport = Mock()
    transport.capture.return_value = subprocess.CompletedProcess([], 0, payload, "")

    assert ssh_delegate.snapshot(
        ["ssh", "login"], "/helper", "/out", None, 0, transport=transport
    ) == (payload, 0)

    output = capsys.readouterr()
    assert output.out == ""
    assert payload in output.err
    assert ("[ssh] remote log snapshot" in output.err) is (level == "INFO")


def test_recovery_commands_remain_copyable_on_a_narrow_console(monkeypatch, capsys):
    progress._logger.handlers[0].console = Console(stderr=True, width=80)
    monkeypatch.setattr(ssh_delegate, "_SSH_FAILURE_HINT_SHOWN", False)
    ssh_delegate.ssh_failure_hint(
        ["ssh", "-p", "20", "-J", "bastion", "-i", "my key", "user@login"]
    )

    output = capsys.readouterr()
    assert output.out == ""
    lines = output.err.splitlines()
    master = next(line.strip() for line in lines if line.strip().startswith("ssh "))
    assert shlex.split(master) == [
        "ssh",
        "-p",
        "20",
        "-J",
        "bastion",
        "-i",
        "my key",
        "-MNf",
        "-o",
        "BatchMode=no",
        "-o",
        "ControlMaster=yes",
        "-o",
        "ControlPersist=24h",
        "-o",
        "ControlPath=%d/.ssh/sflow-%C",
        "user@login",
    ]
    reuse = next(line.strip() for line in lines if line.strip().startswith("--ssh "))
    assert shlex.split(shlex.split(reuse)[1]) == [
        "-p",
        "20",
        "-J",
        "bastion",
        "-i",
        "my key",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPath=%d/.ssh/sflow-%C",
        "user@login",
    ]


@pytest.mark.parametrize("level", ["INFO", "WARNING"])
def test_follower_formats_and_filters_headers_but_preserves_payload(
    tmp_path, monkeypatch, capsys, level
):
    payload = "worker ERROR: failure details\n" + "x" * 240
    monkeypatch.setattr(remote_helper, "_snapshot_text", lambda root: payload)
    sleep = Mock(side_effect=[None, KeyboardInterrupt])
    monkeypatch.setattr(remote_helper.time, "sleep", sleep)
    monkeypatch.setattr(
        sys, "argv", ["remote_helper.py", "follow", str(tmp_path), level]
    )

    with pytest.raises(KeyboardInterrupt):
        remote_helper._helper_main()

    output = capsys.readouterr()
    assert output.out == f"{payload}\n{payload}\n"
    if level == "INFO":
        assert re.search(
            r"\[\d+/\d+/\d+ \d+:\d+:\d+\]\s+INFO\s+\[ssh\] remote log snapshot",
            output.err,
        )
        assert "unchanged for 5s" in output.err
    else:
        assert output.err == ""


def test_delegate_forwards_log_level_to_follower(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ssh_delegate,
        "_resolve_remote_environment",
        lambda *a, **k: ("/remote/cache", "test"),
    )
    transport = Mock()
    transport.run_to_stderr.return_value = subprocess.CompletedProcess([], 0)
    transport.start_to_stderr.side_effect = OSError("stop after follower command")
    invocation = argv_to_invocation(
        [
            "run",
            "--ssh",
            "login",
            "--log-level",
            "WARNING",
            "--ssh-follow",
            "logs",
            "--ssh-tty",
            "never",
        ]
    )
    recipe = tmp_path / "sflow.yaml"
    recipe.write_text(
        "version: '0.1'\nworkflow:\n  name: follow\n  tasks:\n"
        "    - name: hello\n      script: [echo hello]\n"
    )

    with pytest.raises(typer.Exit):
        ssh_delegate.delegate(invocation, transport=transport)

    assert shlex.split(transport.start_to_stderr.call_args.args[1])[-1] == "WARNING"
    assert "[ssh 1/4]" not in capsys.readouterr().err


@pytest.fixture(autouse=True)
def reset_ssh_progress():
    progress.configure_logging()
    yield
    progress.configure_logging()


def test_ssh_stages_and_paths_use_timestamped_log_columns(capsys):
    ssh_delegate._stage(1, "connecting to login")
    ssh_delegate._fact("remote session", "/cache/runs/current")

    output = capsys.readouterr()
    assert output.out == ""
    assert re.search(r"\[\d+/\d+/\d+ \d+:\d+:\d+\]\s+INFO", output.err)
    assert "[ssh 1/4] connecting to login..." in output.err
    assert "remote session: /cache/runs/current" in output.err


def test_ssh_progress_cannot_leak_into_compose_yaml(capsys):
    # The ordinary workflow logger is on stdout; SSH must not inherit it.
    configure_workflow_logging()
    ssh_delegate._stage(4, "running sflow compose on login")
    ssh_delegate._fact("remote work dir", "/cache/work/[literal]")
    print("workflow:\n  name: composed")

    output = capsys.readouterr()
    assert yaml.safe_load(output.out) == {"workflow": {"name": "composed"}}
    assert "/cache/work/[literal]" in output.err


def test_ssh_warnings_have_severity_instead_of_an_unformatted_prefix(capsys):
    assert ssh_delegate._tty_mode("never", ["--tui"]) == "always"

    output = capsys.readouterr()
    assert output.out == ""
    assert "WARNING" in output.err
    assert "--tui needs a remote PTY" in output.err
    assert "Warning:" not in output.err


def test_ssh_log_level_hides_progress_but_keeps_errors(capsys):
    progress.configure_logging("ERROR")
    ssh_delegate._stage(1, "connecting to login")
    ssh_delegate._fact("remote runtime", "/cache/runtime")
    ssh_delegate._tty_mode("never", ["--tui"])
    transport = Mock()
    transport.capture.return_value = subprocess.CompletedProcess(
        [], 0, '["/missing-model"]', ""
    )

    with pytest.raises(typer.Exit):
        ssh_delegate._check_remote_artifacts(
            ["ssh", "login"], ["/missing-model"], transport=transport
        )

    output = capsys.readouterr()
    assert output.out == ""
    assert "connecting" not in output.err
    assert "remote runtime" not in output.err
    assert "WARNING" not in output.err
    assert "ERROR" in output.err
    assert "/missing-model" in output.err


def test_reconfiguring_ssh_progress_does_not_duplicate_output(capsys):
    progress.configure_logging()
    progress.configure_logging()
    ssh_delegate._stage(1, "connecting to login")

    assert capsys.readouterr().err.count("connecting to login") == 1


def test_cli_log_level_is_used_for_progress_and_remote_bootstrap(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    recipe = tmp_path / "flow.yaml"
    recipe.write_text(
        "workflow:\n  name: log-level\n  tasks:\n    - name: hello\n      script: [echo hello]\n"
    )
    monkeypatch.setattr(
        ssh_delegate,
        "_resolve_remote_environment",
        lambda *a, **k: ("/remote/cache", "test"),
    )
    transport = Mock()
    transport.run_to_stderr.side_effect = OSError("stop before upload")
    invocation = argv_to_invocation(
        [
            "run",
            "-f",
            str(recipe),
            "--ssh",
            "login",
            "--log-level",
            "WARNING",
        ]
    )

    with pytest.raises(typer.BadParameter, match="stop before upload"):
        ssh_delegate.delegate(invocation, transport=transport)

    assert shlex.split(transport.run_to_stderr.call_args.args[1])[-1] == "WARNING"
    assert "[ssh 1/4]" not in capsys.readouterr().err


@pytest.mark.skipif(sys.platform == "win32", reason="SSH bootstrap runs on a Unix host")
@pytest.mark.parametrize("level, expected", [("INFO", True), ("WARNING", False)])
def test_bootstrap_formats_progress_without_sflow_or_rich(
    tmp_path, fake_process, level, expected
):
    fake_process.allow_unregistered(True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / ".ready").touch()
    # Isolated mode removes site/user paths; the helper must remain standalone.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            remote_helper.__file__,
            "bootstrap",
            str(runtime),
            str(tmp_path / "source-project"),
            "1.0",
            level,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    assert result.stdout == ""
    assert ("cached runtime:" in result.stderr) is expected
    assert ("INFO" in result.stderr) is expected
    if expected:
        assert re.search(r"\[\d+/\d+/\d+ \d+:\d+:\d+\]\s+INFO", result.stderr)


@pytest.mark.skipif(sys.platform == "win32", reason="Fake SSH uses a Unix shell")
@pytest.mark.parametrize("level", ["INFO", "WARNING"])
def test_compose_yaml_stdout_stays_clean_through_local_ssh(
    tmp_path, fake_process, level
):
    fake_process.allow_unregistered(True)
    remote_root = tmp_path / "remote"
    runtime = (
        remote_root
        / "runtimes"
        / ssh_delegate._runtime_fingerprint(ssh_delegate._requirements())
    )
    (runtime / "bin").mkdir(parents=True)
    runtime_python = runtime / "bin/python"
    runtime_python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    runtime_python.chmod(0o755)
    (runtime / ".ready").touch()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        '#!/bin/sh\nlast=\nfor arg in "$@"; do last=$arg; done\n'
        'exec /bin/sh -c "$last"\n'
    )
    fake_ssh.chmod(0o755)
    (fake_bin / "python3").symlink_to(sys.executable)
    recipe = tmp_path / "flow.yaml"
    recipe.write_text(
        "version: '0.1'\nworkflow:\n  name: clean_stdout\n  tasks:\n"
        "    - name: hello\n      script: [echo hello]\n"
    )
    environment = os.environ.copy()
    environment["PATH"] = str(fake_bin) + os.pathsep + environment["PATH"]
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sflow",
            "compose",
            str(recipe),
            "--ssh",
            "login",
            "--ssh-remote-root",
            str(remote_root),
            "--log-level",
            level,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(result.stdout)["workflow"]["name"] == "clean_stdout"
    assert ("[ssh 1/4]" in result.stderr) is (level == "INFO")
    assert ("cached runtime:" in result.stderr) is (level == "INFO")
