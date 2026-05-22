# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys

import pytest

import sflow.core.launcher as launcher_mod
from sflow.core.command_log import (
    CommandLogRouter,
)
from sflow.core.launcher import SubprocessLauncher


def test_command_log_router_routes_command_families_and_fallback(tmp_path):
    router = CommandLogRouter(tmp_path)

    router.record(["srun", "--job-name", "wf"], task_name="worker", shell=False)
    router.record(["bash", "-c", "echo hi"], task_name="worker", shell=False)
    router.record([sys.executable, "-c", "print('hi')"], task_name=None, shell=False)
    router.record(["custom-runner", "--flag"], task_name=None, shell=True)

    assert "srun --job-name wf" in (tmp_path / "slurm_cmds.log").read_text()
    assert "task    : worker" in (tmp_path / "slurm_cmds.log").read_text()
    assert "bash -c 'echo hi'" in (tmp_path / "bash_cmds.log").read_text()
    assert "-c 'print('\"'\"'hi'\"'\"')'" in (tmp_path / "python_cmds.log").read_text()
    assert "custom-runner --flag" in (tmp_path / "backend_cmds.log").read_text()
    assert "shell   : True" in (tmp_path / "backend_cmds.log").read_text()


def test_command_log_router_formats_multiline_commands_as_readable_blocks(tmp_path):
    router = CommandLogRouter(tmp_path)

    router.record(
        ["bash", "-c", "echo prepare\npython train.py\n"],
        task_name="prepare_data",
        shell=False,
    )

    text = (tmp_path / "bash_cmds.log").read_text()

    assert "task    : prepare_data" in text
    assert "family  : bash" in text
    assert "shell   : False" in text
    assert "command :" in text
    assert "  bash -c 'echo prepare" in text
    assert "  python train.py" in text
    assert "task=prepare_data" not in text


def test_launcher_does_not_record_command_before_process_starts(monkeypatch):
    recorded = []

    def _record(command, *, task_name, shell):
        recorded.append((command, task_name, shell))

    def _stop_before_subprocess():
        raise RuntimeError("stop before subprocess")

    monkeypatch.setattr(launcher_mod, "record_active_command", _record)
    monkeypatch.setattr(launcher_mod.pty, "openpty", _stop_before_subprocess)

    with pytest.raises(RuntimeError, match="stop before subprocess"):
        asyncio.run(
            SubprocessLauncher().run_async(
                ["bash", "-c", "printf 'TASK_OUTPUT_SECRET\\n'"],
                env={"INJECTED_SECRET": "do-not-log"},
                task_name="hello",
            )
        )

    assert recorded == []
    assert "INJECTED_SECRET" not in repr(recorded)
    assert "do-not-log" not in repr(recorded)


def test_launcher_records_command_after_process_starts_without_env(monkeypatch):
    recorded = []

    class _StartedProcess:
        returncode = 0

        def poll(self):
            return 0

    def _record(command, *, task_name, shell):
        recorded.append((command, task_name, shell))

    def _open_pipe():
        return os.pipe()

    def _popen(*args, **kwargs):
        return _StartedProcess()

    monkeypatch.setattr(launcher_mod, "record_active_command", _record)
    monkeypatch.setattr(launcher_mod.pty, "openpty", _open_pipe)
    monkeypatch.setattr(launcher_mod.subprocess, "Popen", _popen)

    asyncio.run(
        SubprocessLauncher().run_async(
            ["bash", "-c", "printf 'TASK_OUTPUT_SECRET\\n'"],
            env={"INJECTED_SECRET": "do-not-log"},
            task_name="hello",
        )
    )

    assert recorded == [
        (["bash", "-c", "printf 'TASK_OUTPUT_SECRET\\n'"], "hello", False)
    ]
    assert "INJECTED_SECRET" not in repr(recorded)
    assert "do-not-log" not in repr(recorded)
