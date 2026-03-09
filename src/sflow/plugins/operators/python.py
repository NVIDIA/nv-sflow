# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.operator_registry import register_operator


class PythonOperatorConfig(OperatorConfig):
    name: str
    type: Literal["python"] = "python"

    python_exec: str = "python"
    extra_args: list[str] = Field(default_factory=list)


@register_operator("python", PythonOperatorConfig)
class PythonOperator(Operator):
    def __init__(self, config: PythonOperatorConfig):
        super().__init__(config)
        self.config: PythonOperatorConfig

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        c = self.config
        code = "\n".join(list(script))
        # Env is injected by SubprocessLauncher(env=...) to avoid leaking env values into logs
        # and to avoid shell quoting issues.
        cmd = Command(exec=c.python_exec)
        for a in list(c.extra_args):
            cmd.add_arg(a)
        cmd.add_arg("-c")
        cmd.add_arg(code)
        return cmd
