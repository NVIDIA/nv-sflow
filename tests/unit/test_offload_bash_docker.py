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


def test_host_prefixer_strips_ansi_like_stream_mode(monkeypatch):
    """<task>.log must carry no escapes on ANY write path.

    Stream mode strips at write time and k8s at pod-terminal; the offload
    prefixer did neither, so a dynamo worker's coloured `endpoint=generate` --
    stored as `\\e[3mendpoint\\e[0m\\e[2m=\\e[0mgenerate` -- never matched the
    literal log_watch pattern and a healthy worker sat in its readiness probe
    until the timeout. The real 0.8.0 line, driven through the helper exactly
    as the container runs it.
    """
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    coloured = (
        "\x1b[2m2026-10-28T05:26:47Z\x1b[0m \x1b[32m INFO\x1b[0m Registering endpoint "
        "with request plane server \x1b[3mendpoint\x1b[0m\x1b[2m=\x1b[0mgenerate "
        "\x1b[3mtransport\x1b[0m\x1b[2m=\x1b[0m\"tcp\"\n"
    )
    namespace: dict = {}
    exec(compile(LOG_PREFIX_HELPER_SRC, "log_prefix.py", "exec"), namespace)
    monkeypatch.setattr("sys.stdin", io.StringIO(coloured))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.argv", ["log_prefix.py", "sflow.task.t"])
    namespace["main"]()
    line = out.getvalue().splitlines()[0]
    assert "\x1b" not in line, line
    assert line.endswith('request plane server endpoint=generate transport="tcp"'), line


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


def test_docker_offload_is_skipped_for_multi_node(tmp_path, monkeypatch):
    """Multi-node members stream through the launcher, so offload must not apply.

    The launcher already owns member output and labels it per rank. A host-side
    redirect on top would be a second writer on the same <task>.log, and
    `writes_own_task_log` would suppress sflow's own handler for a file nothing
    is writing -- losing the task log entirely.
    """
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

    # A plain docker command, not a bash offload wrapper.
    assert parts[:2] == ["docker", "run"]
    log_path = os.path.join(envs["SFLOW_TASK_OUTPUT_DIR"], "t.log")
    assert not any(str(log_path) in part for part in parts)
    # sflow keeps ownership of <task>.log, since nothing else is writing it.
    assert op.writes_own_task_log() is False
