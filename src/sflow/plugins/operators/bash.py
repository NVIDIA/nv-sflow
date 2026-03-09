# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.operator_registry import register_operator


class BashOperatorConfig(OperatorConfig):
    """
    Local bash operator configuration.

    Step A: keep it small; more fields can be added later (shell, cwd, etc.).
    """

    name: str
    type: Literal["bash"] = "bash"


@register_operator("bash", BashOperatorConfig)
class BashOperator(Operator):
    def __init__(self, config: BashOperatorConfig):
        super().__init__(config)

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        command = Command(exec="bash")
        command.add_arg("-c")
        # Env is injected by SubprocessLauncher(env=...) to avoid leaking env values into logs
        # and to avoid shell quoting issues.
        command.add_arg("\n".join(list(script)))
        return command
