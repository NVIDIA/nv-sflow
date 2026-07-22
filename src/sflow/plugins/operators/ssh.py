# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from sflow.core.command import Command
from sflow.core.log_offload import unsupported_offload_warning
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.operator_registry import register_operator
from sflow.utils.script import prepend_envs


class SshOperatorConfig(OperatorConfig):
    name: str
    type: Literal["ssh"] = "ssh"

    host: str
    user: str | None = None
    port: int | None = None
    identity_file: str | None = None
    extra_args: list[str] = Field(default_factory=list)

    def runtime_warnings(self) -> list[str]:
        return unsupported_offload_warning(self.type)


@register_operator("ssh", SshOperatorConfig)
class SshOperator(Operator):
    def __init__(self, config: SshOperatorConfig):
        super().__init__(config)
        self.config: SshOperatorConfig

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        c = self.config
        dest = f"{c.user}@{c.host}" if c.user else c.host

        # TODO(log-offload): per-task log offload is not implemented for this
        # operator yet (logs stream through the sflow driver). To add it, wrap the
        # remote payload with the prefixer (sflow.core.log_offload.wrap_with_prefixer)
        # and redirect to the per-task log ON THE REMOTE HOST; this only helps when
        # that path is on shared storage the driver can read (or is fetched back),
        # like the Slurm case. Also override writes_own_task_log().
        # Build remote payload as a single bash -lc string.
        payload = "\n".join(prepend_envs(list(script), dict(envs)))
        remote_cmd = f"bash -lc {shlex.quote(payload)}"

        cmd = Command(exec="ssh")
        if c.port is not None:
            cmd.add_opt("-p", c.port)
        if c.identity_file is not None:
            cmd.add_opt("-i", c.identity_file)
        for a in c.extra_args:
            cmd.add_opt(a)
        cmd.add_arg(dest)
        cmd.add_arg(remote_cmd)
        return cmd
