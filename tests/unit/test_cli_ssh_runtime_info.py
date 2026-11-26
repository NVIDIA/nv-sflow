import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from sflow import runtime_info
from sflow.ssh.delegate import _requirements, _runtime_fingerprint


@pytest.mark.parametrize("source", ["PyPI 0.3.2", "editable checkout 0.3.2"])
def test_ssh_banner_uses_current_invocation_not_cached_install(
    monkeypatch, tmp_path, source
):
    runtime = "/cache/runtimes/shared"
    monkeypatch.setenv("SFLOW_SSH_RUNTIME", runtime)
    monkeypatch.setenv("SFLOW_SSH_SOURCE", "mirrors local " + source)
    monkeypatch.setattr(
        runtime_info,
        "_install_info",
        lambda: ("direct-url", "/obsolete-first-session/source-project"),
    )
    monkeypatch.setattr(runtime_info, "_git_info_for_repo", lambda repo: "")
    monkeypatch.setattr(runtime_info, "_current_console_width", lambda: 160)

    # Both invocations share the cached venv, but execute their own staged code.
    for session in ["first", "second"]:
        package = tmp_path / session / "source-project/src/sflow/__init__.py"
        monkeypatch.setattr(runtime_info.sflow, "__file__", str(package))
        info = runtime_info.format_runtime_info()

        assert "install : ssh-runtime" in info
        assert "source  : mirrors local " + source in info
        assert "runtime : " + runtime in info
        assert "repo    :" not in info
        assert "obsolete-first-session" not in info
        assert "local build" not in info


def test_two_ssh_runs_share_runtime_and_report_ssh_provenance(tmp_path, fake_process):
    fake_process.allow_unregistered(True)
    remote_root = tmp_path / "remote"
    runtime = remote_root / "runtimes" / _runtime_fingerprint(_requirements())
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python").write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    (runtime / "bin/python").chmod(0o755)
    (runtime / ".ready").touch()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "ssh").write_text(
        '#!/bin/sh\nlast=\nfor arg in "$@"; do last=$arg; done\n'
        'exec /bin/sh -c "$last"\n'
    )
    (fake_bin / "ssh").chmod(0o755)
    (fake_bin / "python3").write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    (fake_bin / "python3").chmod(0o755)
    recipe = tmp_path / "flow.yaml"
    recipe.write_text(
        "version: '0.1'\nworkflow:\n  name: banner\n  tasks:\n"
        "    - name: hello\n      script: [echo hello]\n"
    )
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    # Select this worktree rather than the shared venv's editable checkout.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    for _ in range(2):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sflow",
                "run",
                "-f",
                str(recipe),
                "--dry-run",
                "--ssh",
                "login",
                "--ssh-remote-root",
                str(remote_root),
                "--ssh-follow",
                "none",
                "--ssh-fetch",
                "none",
                "--ssh-tty",
                "never",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        output = result.stdout + result.stderr
        assert "cached runtime:" in output
        assert "install : ssh-runtime" in output
        assert "source  : mirrors local" in output
        assert "runtime :" in output
        assert "repo    :" not in output

    assert len(list((remote_root / "runs").iterdir())) == 2
    assert [p for p in (remote_root / "runtimes").iterdir() if p.is_dir()] == [runtime]
