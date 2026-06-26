# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from sflow.core.command import Command
from sflow.core.log_offload import offload_enabled, task_log_path, wrap_with_prefixer
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.operator_registry import register_operator


class BashOperatorConfig(OperatorConfig):
    """
    Local bash operator configuration.

    Step A: keep it small; more fields can be added later (shell, cwd, etc.).
    """

    name: str
    type: Literal["bash"] = "bash"
    # Per-task log offload, ON by default. When enabled (also via the
    # SFLOW_OFFLOAD_TASK_LOGS env / --offload-task-logs flag, which take
    # precedence), the task redirects its own output to <task>.log through a
    # compute-side prefixer instead of streaming it through the sflow driver's
    # pump. Auto-falls back to streaming on an interactive TTY / --tui session.
    log_to_file: bool = True


@register_operator("bash", BashOperatorConfig)
class BashOperator(Operator):
    def __init__(self, config: BashOperatorConfig):
        super().__init__(config)
        self.config: BashOperatorConfig

    def _offload_enabled(self) -> bool:
        return offload_enabled(self.config.log_to_file)

    def writes_own_task_log(self) -> bool:
        # In offload mode the shell redirect owns <task>.log, so sflow must not
        # also attach a FileHandler to that path (single-writer invariant).
        return self._offload_enabled()

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        # Env is injected by SubprocessLauncher(env=...) to avoid leaking env values into logs
        # and to avoid shell quoting issues.
        script_body = "\n".join(list(script))
        log_path = task_log_path(envs, task_name)
        command = Command(exec="bash")
        command.add_arg("-c")
        if self._offload_enabled() and log_path:
            # Same host as the driver: redirect the task's merged output through
            # the prefixer into <task>.log so the driver stops pumping it.
            command.add_arg(
                wrap_with_prefixer(
                    script_body,
                    workflow_out_dir=envs.get("SFLOW_WORKFLOW_OUTPUT_DIR"),
                    task_name=task_name,
                    redirect_to=log_path,
                )
            )
        else:
            command.add_arg(script_body)
        return command
