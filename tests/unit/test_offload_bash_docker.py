# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-task log offload on the bash/local and docker operators."""

import io
import os
import re

from sflow.core.log_offload import (
    LEGACY_OFFLOAD_TASK_LOGS_ENV,
    LOG_PREFIX_HELPER_SRC,
    OFFLOAD_TASK_LOGS_ENV,
)
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.plugins.operators.docker_run import (
    DockerRunOperator,
    DockerRunOperatorConfig,
)

# bash/docker run on the host where SLURM_PROCID is unset, so there is no rank
# prefix (unlike srun, which folds the rank to mirror --label).
_ALIGNED_NORANK_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - sflow\.task\.t - INFO - "
)


def _force_non_tty(monkeypatch):
    monkeypatch.setattr("sflow.core.log_offload.stdout_is_tty", lambda: False)


def _clear_env(monkeypatch):
    monkeypatch.delenv(OFFLOAD_TASK_LOGS_ENV, raising=False)
    monkeypatch.delenv(LEGACY_OFFLOAD_TASK_LOGS_ENV, raising=False)


def _envs(tmp_path):
    task_out = tmp_path / "t"
    task_out.mkdir(parents=True, exist_ok=True)
    return {
        "SFLOW_TASK_OUTPUT_DIR": str(task_out),
        "SFLOW_WORKFLOW_OUTPUT_DIR": str(tmp_path),
    }


# --- bash / local -----------------------------------------------------------


def test_bash_stream_mode_is_plain(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _force_non_tty(monkeypatch)
    # Offload is the default now, so stream mode must be requested explicitly.
    op = BashOperator(BashOperatorConfig(name="t", log_to_file=False))
    parts = op.build_command(
        task_name="t", script=["echo hi"], envs=_envs(tmp_path)
    ).as_list()
    assert parts == ["bash", "-c", "echo hi"]
    assert op.writes_own_task_log() is False


def test_bash_offload_wraps_and_redirects(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _force_non_tty(monkeypatch)
    op = BashOperator(BashOperatorConfig(name="t", log_to_file=True))
    envs = _envs(tmp_path)
    parts = op.build_command(
        task_name="t", script=["echo hi"], envs=envs
    ).as_list()
    assert parts[0] == "bash" and parts[1] == "-c"
    wrapped = parts[-1]
    log_path = os.path.join(envs["SFLOW_TASK_OUTPUT_DIR"], "t.log")
    assert wrapped.startswith("{")
    assert "2>&1 |" in wrapped
    assert f"> {log_path}" in wrapped
    assert wrapped.rstrip().endswith('exit "${PIPESTATUS[0]}"')
    # srun-only flags must never appear on a bash command
    assert "--output" not in wrapped and "--label" not in wrapped
    assert op.writes_own_task_log() is True
    assert (tmp_path / ".sflow" / "log_prefix.py").is_file()


def test_bash_offload_requires_task_output_dir(monkeypatch):
    _clear_env(monkeypatch)
    _force_non_tty(monkeypatch)
    op = BashOperator(BashOperatorConfig(name="t", log_to_file=True))
    # No SFLOW_TASK_OUTPUT_DIR -> nowhere to redirect -> stream mode.
    parts = op.build_command(task_name="t", script=["echo hi"], envs={}).as_list()
    assert parts == ["bash", "-c", "echo hi"]


def test_env_override_enables_offload_over_config(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _force_non_tty(monkeypatch)
    monkeypatch.setenv(OFFLOAD_TASK_LOGS_ENV, "1")
    assert BashOperator(BashOperatorConfig(name="t")).writes_own_task_log() is True
    monkeypatch.setenv(OFFLOAD_TASK_LOGS_ENV, "0")
    assert (
        BashOperator(BashOperatorConfig(name="t", log_to_file=True)).writes_own_task_log()
        is False
    )


def test_legacy_slurm_env_alias_is_recognized(monkeypatch):
    _clear_env(monkeypatch)
    _force_non_tty(monkeypatch)
    monkeypatch.setenv(LEGACY_OFFLOAD_TASK_LOGS_ENV, "1")
    assert BashOperator(BashOperatorConfig(name="t")).writes_own_task_log() is True


def test_tty_session_disables_offload(monkeypatch):
    monkeypatch.setenv(OFFLOAD_TASK_LOGS_ENV, "1")
    monkeypatch.setattr("sflow.core.log_offload.stdout_is_tty", lambda: True)
    assert BashOperator(BashOperatorConfig(name="t")).writes_own_task_log() is False


def test_host_prefixer_omits_rank_when_no_slurm_procid(monkeypatch):
    # On bash/docker (host) SLURM_PROCID is unset, so the prefixer must NOT add a
    # rank label - matching their stream-mode format. Drive the helper directly
    # (the unit suite blocks real subprocesses).
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    namespace: dict = {}
    exec(compile(LOG_PREFIX_HELPER_SRC, "log_prefix.py", "exec"), namespace)
    monkeypatch.setattr("sys.stdin", io.StringIO("hi\n"))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.argv", ["log_prefix.py", "sflow.task.t"])
    namespace["main"]()
    line = out.getvalue().splitlines()[0]
    assert _ALIGNED_NORANK_RE.match(line), line
    assert line.endswith("INFO - hi")  # no "<rank>: " prefix


# --- docker -----------------------------------------------------------------


def test_docker_stream_mode_keeps_docker_exec(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _force_non_tty(monkeypatch)
    # Offload is the default now, so stream mode must be requested explicitly.
    op = DockerRunOperator(
        DockerRunOperatorConfig(name="t", image="busybox", log_to_file=False)
    )
    parts = op.build_command(
        task_name="t", script=["echo hi"], envs=_envs(tmp_path)
    ).as_list()
    assert parts[0] == "docker"
    assert op.writes_own_task_log() is False


def test_docker_offload_wraps_docker_run(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _force_non_tty(monkeypatch)
    op = DockerRunOperator(
        DockerRunOperatorConfig(name="t", image="busybox", log_to_file=True)
    )
    envs = _envs(tmp_path)
    parts = op.build_command(
        task_name="t", script=["echo hi"], envs=envs
    ).as_list()
    assert parts[0] == "bash" and parts[1] == "-c"
    wrapped = parts[-1]
    log_path = os.path.join(envs["SFLOW_TASK_OUTPUT_DIR"], "t.log")
    assert wrapped.startswith("{")
    assert "docker run" in wrapped
    assert "2>&1 |" in wrapped
    assert f"> {log_path}" in wrapped
    assert wrapped.rstrip().endswith('exit "${PIPESTATUS[0]}"')
    assert op.writes_own_task_log() is True


def test_docker_offload_wraps_multi_node_bash_wrapper(tmp_path, monkeypatch):
    """The multi-node launcher itself is a `bash -lc` wrapper; offload must wrap
    that whole wrapper (both containers) and redirect to <task>.log."""
    _clear_env(monkeypatch)
    _force_non_tty(monkeypatch)
    op = DockerRunOperator(
        DockerRunOperatorConfig(name="t", image="busybox", log_to_file=True)
    )
    # Two synthetic local nodes (host=None) -> multi-node bash wrapper.
    op._assigned_nodes = ["localhost", "localhost-1"]
    op._node_hosts = {"localhost": None, "localhost-1": None}
    envs = _envs(tmp_path)
    parts = op.build_command(
        task_name="t", script=["echo hi"], envs=envs
    ).as_list()

    assert parts[0] == "bash" and parts[1] == "-c"
    wrapped = parts[-1]
    log_path = os.path.join(envs["SFLOW_TASK_OUTPUT_DIR"], "t.log")
    assert wrapped.startswith("{")
    # both containers survive inside the offloaded wrapper (names carry the pid)
    assert f"sflow-p{os.getpid()}-t-localhost" in wrapped
    assert f"sflow-p{os.getpid()}-t-localhost-1" in wrapped
    assert f"> {log_path}" in wrapped
    assert wrapped.rstrip().endswith('exit "${PIPESTATUS[0]}"')
