"""Opt-in end-to-end tests for native SSH delegation.

Run with, for example:

    SFLOW_E2E_SSH="user@login -o BatchMode=yes -o ConnectTimeout=15" \
      pytest tests/e2e_tests/test_ssh_delegate_e2e.py -m e2e -v -s

To include a real Slurm submission (the test cancels the submitted job):

    SFLOW_E2E_SLURM_PARTITION=debug SFLOW_E2E_SLURM_ACCOUNT=account ...

Set SFLOW_E2E_REMOTE_ROOT to keep one run's remote sessions and cached runtime
under a directory that can be deleted afterwards (CI scopes it per pipeline).
The `ssh_delegate_e2e` GitLab job runs this file against a real login host.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from sflow.cli._ssh_delegate import _ssh_argv

SSH = os.environ.get("SFLOW_E2E_SSH")
# CI points this at a pipeline-scoped directory so a shared login host can be
# cleaned up per pipeline instead of accumulating runs under the default cache.
REMOTE_ROOT = os.environ.get("SFLOW_E2E_REMOTE_ROOT")
SLURM_PARTITION = os.environ.get("SFLOW_E2E_SLURM_PARTITION")
SLURM_ACCOUNT = os.environ.get("SFLOW_E2E_SLURM_ACCOUNT")
SLURM_GPUS_PER_NODE = os.environ.get("SFLOW_E2E_SLURM_GPUS_PER_NODE", "0")
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not SSH, reason="set SFLOW_E2E_SSH to a configured SSH target"),
]


def _receipts(workspace: Path) -> list[tuple[Path, dict[str, object]]]:
    return [
        (path, json.loads(path.read_text()))
        # One folder per SSH execution, under the default sflow_output/ or
        # whatever --output-dir the invocation used.
        for path in sorted(workspace.glob("*/ssh-*/receipt.json"))
    ]


def _wait_for_running_receipt(
    workspace: Path, process: subprocess.Popen[str], timeout: int = 900
) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for path, receipt in _receipts(workspace):
            if receipt.get("execution_status") == "running":
                return path
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"SSH command exited before execution started ({process.returncode})\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        time.sleep(0.1)
    process.terminate()
    process.wait(timeout=10)
    pytest.fail("timed out waiting for the SSH receipt to enter running state")


def _ssh_args(remote_root: str | None = None) -> list[str]:
    """The delegation flags every case shares, with the CI-scoped remote root."""
    root = remote_root or REMOTE_ROOT
    return ([] if root is None else ["--ssh-remote-root", root]) + [
        "--ssh",
        SSH or "",
    ]


def _run(
    repo: Path,
    *args: str,
    timeout: int = 900,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "sflow", *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(args)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def test_compose_run_heartbeat_and_fetch_over_real_ssh(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    helper = tmp_path / "helper.sh"
    helper.write_text(
        "ssh_e2e_message() {\n"
        "  echo SSH_E2E_STARTED\n"
        "  sleep 11\n"
        "  echo SSH_E2E_FINISHED\n"
        "}\n"
    )
    config = tmp_path / "flow.yaml"
    config.write_text(
        """version: "0.1"

artifacts:
  HELPER:
    uri: file://helper.sh

backends:
  - name: local
    type: local
    default: true

workflow:
  name: ssh_delegate_e2e
  tasks:
    - name: heartbeat
      script:
        - '. "${{ artifacts.HELPER.path }}"'
        - ssh_e2e_message
"""
    )

    composed = tmp_path / "composed.yaml"
    _run(
        repo,
        "compose",
        str(config),
        "-o",
        str(composed),
        "--workspace-dir",
        str(tmp_path),
        *_ssh_args(),
    )
    assert composed.is_file()
    assert "name: ssh_delegate_e2e" in composed.read_text()

    _run(
        repo,
        "run",
        str(config),
        "--dry-run",
        "--workspace-dir",
        str(tmp_path),
        "--ssh-follow",
        "none",
        "--ssh-fetch",
        "none",
        *_ssh_args(),
    )

    output = tmp_path / "remote-output"
    run = _run(
        repo,
        "run",
        str(config),
        "--workspace-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--ssh-follow",
        "logs",
        "--ssh-fetch",
        "all",
        *_ssh_args(),
    )

    assert "unchanged for 5s" in run.stderr
    # Stage labels plus the remote workspace facts they establish.
    for marker in ("[ssh 1/4]", "[ssh 2/4]", "[ssh 3/4]", "[ssh 4/4]"):
        assert marker in run.stderr
    assert "remote work dir: " in run.stderr
    assert "staged: flow.yaml -> " in run.stderr
    assert "remote command: sflow run " in run.stderr
    assert "progress: remote stream plus a 5-second log tail" in run.stderr
    fetched = "\n".join(
        path.read_text(errors="replace") for path in output.rglob("*") if path.is_file()
    )
    assert "SSH_E2E_STARTED" in fetched
    assert "SSH_E2E_FINISHED" in fetched
    # Everything this execution produced lives under one identifiable folder
    # whose name is also the remote session name.
    [session_dir] = sorted(output.glob("ssh-*"))
    assert (session_dir / "receipt.json").is_file()
    session_receipt = json.loads((session_dir / "receipt.json").read_text())
    assert session_receipt["session"] == session_dir.name
    assert str(session_receipt["remote_session"]).endswith(f"/runs/{session_dir.name}")
    receipts = _receipts(tmp_path)
    assert len(receipts) == 3
    assert all(receipt[1]["status"] == "completed" for receipt in receipts)
    assert all(receipt[1]["execution_status"] == "completed" for receipt in receipts)
    assert all(receipt[1]["phase"] == "finished" for receipt in receipts)
    assert sorted(receipt[1]["fetch_status"] for receipt in receipts) == [
        "completed",
        "completed",
        "skipped",
    ]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path, _ in receipts)


def test_compose_stdout_stays_clean_during_first_bootstrap(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    config = tmp_path / "flow.yaml"
    config.write_text(
        'version: "0.1"\nworkflow:\n  name: clean_stdout\n  tasks:\n'
        "    - name: hello\n      script: [echo hello]\n"
    )

    command = [
        sys.executable,
        "-m",
        "sflow",
        "compose",
        str(config),
        "--workspace-dir",
        str(tmp_path),
        "--ssh-remote-root",
        f"~/.cache/sflow/ssh-e2e-concurrent-bootstrap-{os.getpid()}",
        "--ssh",
        SSH or "",
    ]
    processes = [
        subprocess.Popen(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    results: list[tuple[int, str, str]] = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=900)
            results.append((process.returncode, stdout, stderr))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()

    assert all(return_code == 0 for return_code, _, _ in results), results
    assert all(
        yaml.safe_load(stdout)["workflow"]["name"] == "clean_stdout"
        for _, stdout, _ in results
    )
    assert all("Successfully installed" not in stdout for _, stdout, _ in results)
    assert all("[ssh 2/4]" not in stdout for _, stdout, _ in results)
    assert all("[ssh 1/4] connecting" in stderr for _, _, stderr in results)
    # A cold remote root must actually build the runtime, and uv is the fast path.
    assert any("building runtime with" in stderr for _, _, stderr in results)
    receipts = _receipts(tmp_path)
    assert len(receipts) == 2
    assert len({receipt[1]["remote_session"] for receipt in receipts}) == 2
    assert all(receipt[1]["status"] == "completed" for receipt in receipts)


def test_batch_generation_round_trips_over_real_ssh(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    config = tmp_path / "flow.yaml"
    config.write_text(
        """version: "0.1"
backends:
  - name: slurm
    type: slurm
    default: true
    partition: debug
    account: test
    nodes: 1
    gpus_per_node: 1
    time: "00:10:00"
workflow:
  name: ssh_batch_generation_e2e
  tasks:
    - name: hello
      script: [echo hello]
"""
    )
    script = tmp_path / "job.sh"

    _run(
        repo,
        "batch",
        str(config),
        "--partition",
        "debug",
        "--account",
        "test",
        "--sbatch-path",
        str(script),
        "--workspace-dir",
        str(tmp_path),
        *_ssh_args(),
    )

    assert script.is_file()
    assert script.with_suffix(".yaml").is_file()
    assert "source-project" in script.read_text()
    [(_, receipt)] = _receipts(tmp_path)
    assert receipt["status"] == "completed"
    assert receipt["execution_status"] == "completed"
    assert receipt["phase"] == "finished"


def test_local_source_change_is_used_without_rebuilding_runtime(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    overlay = tmp_path / "source-project"
    overlay.mkdir()
    shutil.copy2(repo / "pyproject.toml", overlay / "pyproject.toml")
    shutil.copytree(repo / "src" / "sflow", overlay / "src" / "sflow")
    init = overlay / "src" / "sflow" / "__init__.py"
    init.write_text(init.read_text() + '\nSSH_E2E_SOURCE_MARKER = "SOURCE_ONE"\n')

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "flow.yaml"
    config.write_text(
        """version: "0.1"

artifacts:
  PROBE:
    uri: file://probe.py
    content: |
      import sflow
      print(sflow.SSH_E2E_SOURCE_MARKER)

backends:
  - name: local
    type: local
    default: true

workflow:
  name: ssh_source_refresh_e2e
  tasks:
    - name: probe
      script:
        - python3 "${{ artifacts.PROBE.path }}"
"""
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(overlay / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    remote_root = f"~/.cache/sflow/ssh-e2e-source-refresh-{os.getpid()}"

    def run_source(marker: str) -> tuple[str, subprocess.CompletedProcess[str]]:
        common = (
            str(config),
            "--workspace-dir",
            str(workspace),
            "--ssh-remote-root",
            remote_root,
            "--ssh",
            SSH or "",
        )
        _run(
            repo,
            "run",
            *common,
            "--dry-run",
            "--ssh-follow",
            "none",
            "--ssh-fetch",
            "none",
            env=env,
        )
        output = workspace / marker.lower()
        result = _run(
            repo,
            "run",
            *common,
            "--output-dir",
            str(output),
            "--ssh-follow",
            "none",
            "--ssh-fetch",
            "all",
            env=env,
        )
        text = "\n".join(
            path.read_text(errors="replace")
            for path in output.rglob("*")
            if path.is_file()
        )
        return text, result

    first, _ = run_source("SOURCE_ONE")
    init.write_text(init.read_text().replace("SOURCE_ONE", "SOURCE_TWO"))
    second, second_run = run_source("SOURCE_TWO")

    assert "SOURCE_ONE" in first
    assert "SOURCE_TWO" not in first
    assert "SOURCE_TWO" in second
    assert "SOURCE_ONE" not in second
    assert "Successfully installed" not in second_run.stdout + second_run.stderr


def test_ctrl_c_cancels_the_remote_run_and_keeps_its_logs(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    config = tmp_path / "flow.yaml"
    config.write_text(
        """version: "0.1"
backends:
  - name: local
    type: local
    default: true
workflow:
  name: ssh_interrupt_e2e
  tasks:
    - name: interrupt_me
      script:
        - echo SSH_INTERRUPT_STARTED
        - sleep 120
"""
    )
    output = tmp_path / "cancelled-output"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sflow",
            "run",
            str(config),
            "--workspace-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--ssh-follow",
            "status",
            "--ssh-fetch",
            "all",
            *_ssh_args(),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        receipt_path = _wait_for_running_receipt(tmp_path, process)
        # The receipt flips to "running" as soon as the local delegate() spawns
        # the SSH exec -- before the remote sflow process has imported its own
        # dependencies, printed its banner, or scheduled the task. On a real
        # remote host that startup can easily run past 2s, so a short sleep
        # here races Ctrl-C against the task ever writing SSH_INTERRUPT_STARTED.
        time.sleep(10)
        process.send_signal(signal.SIGINT)
        # The escalation ladder gives the remote controller 20s to unwind before
        # SIGTERM, so allow for a full ladder plus the fetch.
        stdout, stderr = process.communicate(timeout=180)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    context = f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert process.returncode == 130, context
    assert "Cancelling the remote run" in stderr, context
    assert "sending SIGINT to remote controller" in stderr, context
    assert "remote controller stopped" in stderr, context
    assert "Remote run cancelled." in stderr, context

    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "cancelled"
    assert receipt["execution_status"] == "cancelled"
    assert receipt["return_code"] == 130
    assert receipt["phase"] == "finished"
    # A confirmed cancel is not an unknown state, so the partial logs are kept.
    assert receipt["fetch_status"] == "completed"
    fetched = "\n".join(
        path.read_text(errors="replace") for path in output.rglob("*") if path.is_file()
    )
    assert "SSH_INTERRUPT_STARTED" in fetched

    # The remote controller is really gone, not just detached from our client.
    pid_file = f"{receipt['remote_session']}/controller.pid"
    probe = subprocess.run(
        [
            *_ssh_argv(SSH or ""),
            f"pid=$(cat {pid_file}); kill -0 \"$pid\" 2>/dev/null "
            "&& echo ALIVE || echo GONE",
        ],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert "GONE" in probe.stdout, probe


def test_bootstrap_failure_is_reported_without_starting_the_run(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    config = tmp_path / "flow.yaml"
    config.write_text(
        'version: "0.1"\nworkflow:\n  name: bootstrap_fail\n  tasks:\n'
        "    - name: hello\n      script: [echo hello]\n"
    )

    # /dev/null is not a directory, so every mkdir under it fails on the remote
    # side: a bootstrap failure that needs no cooperation from the host.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sflow",
            "run",
            str(config),
            "--workspace-dir",
            str(tmp_path),
            "--ssh-follow",
            "none",
            "--ssh-fetch",
            "none",
            *_ssh_args("/dev/null/sflow-e2e-not-a-directory"),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )

    assert result.returncode != 0, result.stdout
    [(_, receipt)] = _receipts(tmp_path)
    assert receipt["phase"] == "bootstrap"
    assert receipt["execution_status"] == "not_started"
    assert receipt["fetch_status"] == "skipped"
    assert receipt["status"] in {"failed", "unknown"}


def test_unreachable_host_reports_ssh_255_with_a_reconnect_hint(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    config = tmp_path / "flow.yaml"
    config.write_text(
        'version: "0.1"\nworkflow:\n  name: unreachable\n  tasks:\n'
        "    - name: hello\n      script: [echo hello]\n"
    )

    # Port 1 on localhost refuses the connection, which is how OpenSSH produces
    # its transport-level status 255 without depending on a real host.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sflow",
            "run",
            str(config),
            "--workspace-dir",
            str(tmp_path),
            "--ssh-follow",
            "none",
            "--ssh-fetch",
            "none",
            "--ssh",
            "sflow-e2e@127.0.0.1 -p 1 -o BatchMode=yes -o ConnectTimeout=5",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )

    assert result.returncode != 0
    assert "255" in result.stderr
    assert "ControlMaster" in result.stderr
    assert "--ssh " in result.stderr
    assert not _receipts(tmp_path), "no remote session was ever created"


def test_fetch_from_a_missing_remote_session_fails_loudly(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "command": "run",
                "connection": SSH or "",
                "remote_session": "/tmp/sflow-e2e-session-that-does-not-exist",
                "local_output": str(tmp_path / "recovered"),
            }
        )
    )

    result = subprocess.run(
        [sys.executable, "-m", "sflow", "fetch", str(receipt)],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert "failed to fetch remote output" in result.stderr
    assert not (tmp_path / "recovered").exists()


@pytest.mark.parametrize(
    "tty_mode, expected_progress",
    [
        # With a PTY the remote process owns the terminal alone, so auto keeps
        # the local poller out of the way; without one it is the only log view.
        ("always", "progress: remote stream only"),
        ("never", "progress: remote stream plus a 5-second log tail"),
    ],
)
def test_auto_follow_tracks_the_remote_pty(
    tmp_path: Path, tty_mode: str, expected_progress: str
) -> None:
    repo = Path(__file__).resolve().parents[2]
    config = tmp_path / "flow.yaml"
    config.write_text(
        'version: "0.1"\nworkflow:\n  name: tty_mode\n  tasks:\n'
        "    - name: hello\n      script: [echo hello]\n"
    )

    result = _run(
        repo,
        "run",
        str(config),
        "--workspace-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / f"out-{tty_mode}"),
        "--ssh-tty",
        tty_mode,
        "--ssh-fetch",
        "none",
        *_ssh_args(),
    )

    assert expected_progress in result.stderr


@pytest.mark.skipif(
    not (SLURM_PARTITION and SLURM_ACCOUNT),
    reason=(
        "set SFLOW_E2E_SLURM_PARTITION and SFLOW_E2E_SLURM_ACCOUNT to test a "
        "real submission"
    ),
)
def test_real_slurm_submission_receipt_and_cleanup(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    config = tmp_path / "flow.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": "0.1",
                "backends": [
                    {
                        "name": "slurm",
                        "type": "slurm",
                        "default": True,
                        "partition": SLURM_PARTITION,
                        "account": SLURM_ACCOUNT,
                        "nodes": 1,
                        "gpus_per_node": int(SLURM_GPUS_PER_NODE),
                        "time": "00:05:00",
                    }
                ],
                "workflow": {
                    "name": "ssh_submit_e2e",
                    "tasks": [{"name": "hello", "script": ["echo SSH_SLURM_E2E"]}],
                },
            },
            sort_keys=False,
        )
    )
    script = tmp_path / "job.sh"
    job_id: str | None = None
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sflow",
                "batch",
                str(config),
                "--partition",
                SLURM_PARTITION or "",
                "--account",
                SLURM_ACCOUNT or "",
                "--sbatch-path",
                str(script),
                "--submit",
                "--ssh-fetch",
                "none",
                "--workspace-dir",
                str(tmp_path),
                *_ssh_args(),
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
        match = re.search(r"Submitted batch job (\d+)", result.stdout + result.stderr)
        job_id = match.group(1) if match else None
        assert result.returncode == 0, (
            f"submission failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert job_id, f"could not find Slurm job id\n{result.stdout}\n{result.stderr}"
        [(_, receipt)] = _receipts(tmp_path)
        assert receipt["status"] == "submitted"
        assert receipt["execution_status"] == "completed"
        assert receipt["backend_status"] == "submitted"
        assert receipt["fetch_status"] == "skipped"
        assert receipt["phase"] == "finished"

        visible = subprocess.run(
            [*_ssh_argv(SSH or ""), f"scontrol show job {job_id}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert visible.returncode == 0, visible.stderr
        assert f"JobId={job_id}" in visible.stdout
    finally:
        if job_id is not None:
            subprocess.run(
                [*_ssh_argv(SSH or ""), f"scancel {job_id}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
