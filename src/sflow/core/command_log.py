# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .command import Command, format_command

_ACTIVE_COMMAND_LOG_ROUTER: contextvars.ContextVar[CommandLogRouter | None] = (
    contextvars.ContextVar("sflow_command_log_router", default=None)
)

_SLURM_COMMANDS = {"salloc", "srun", "scontrol", "scancel", "sbatch"}
_FAMILY_FILES = {
    "slurm": "slurm_cmds.log",
    "bash": "bash_cmds.log",
    "docker": "docker_cmds.log",
    "ssh": "ssh_cmds.log",
    "python": "python_cmds.log",
    "backend": "backend_cmds.log",
}


class CommandLogRouter:
    """Route command-only launch records to per-backend log files."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)

    def record(
        self,
        command: Command | str | list[str],
        *,
        task_name: str | None,
        shell: bool,
    ) -> None:
        family = self._family(command)
        path = self.path_for_family(family)
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        task = task_name or "-"
        formatted = format_command(command)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{'=' * 80}\n")
            fh.write(f"time    : {timestamp}\n")
            fh.write(f"family  : {family}\n")
            fh.write(f"task    : {task}\n")
            fh.write(f"shell   : {shell}\n")
            fh.write("command :\n")
            for line in formatted.splitlines() or [""]:
                fh.write(f"  {line}\n")

    def path_for_family(self, family: str) -> Path:
        return self.output_dir / _FAMILY_FILES.get(family, _FAMILY_FILES["backend"])

    def planned_paths(self) -> dict[str, Path]:
        return {family: self.output_dir / filename for family, filename in _FAMILY_FILES.items()}

    def existing_paths(self) -> list[Path]:
        return sorted(path for path in self.planned_paths().values() if path.exists())

    def _family(self, command: Command | str | list[str]) -> str:
        executable = _executable_name(command)
        if executable in _SLURM_COMMANDS:
            return "slurm"
        if executable in {"bash", "sh"}:
            return "bash"
        if executable == "docker":
            return "docker"
        if executable == "ssh":
            return "ssh"
        if executable.startswith("python") or executable == Path(sys.executable).name:
            return "python"
        return "backend"


def get_active_command_log_router() -> CommandLogRouter | None:
    return _ACTIVE_COMMAND_LOG_ROUTER.get()


def set_active_command_log_router(
    router: CommandLogRouter | None,
) -> contextvars.Token[CommandLogRouter | None]:
    return _ACTIVE_COMMAND_LOG_ROUTER.set(router)


def reset_active_command_log_router(
    token: contextvars.Token[CommandLogRouter | None],
) -> None:
    _ACTIVE_COMMAND_LOG_ROUTER.reset(token)


def record_active_command(
    command: Command | str | list[str],
    *,
    task_name: str | None,
    shell: bool,
) -> None:
    router = get_active_command_log_router()
    if router is None:
        return
    try:
        router.record(command, task_name=task_name, shell=shell)
    except Exception:
        return


def _executable_name(command: Command | str | list[str]) -> str:
    parts: list[str]
    if isinstance(command, Command):
        parts = command.as_list()
    elif isinstance(command, list):
        parts = [str(part) for part in command]
    else:
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = str(command).split()
    if not parts:
        return ""
    return Path(str(parts[0])).name
