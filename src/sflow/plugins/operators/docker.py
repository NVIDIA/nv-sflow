# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.operator_registry import register_operator


class DockerOperatorConfig(OperatorConfig):
    name: str
    type: Literal["docker"] = "docker"

    image: str
    workdir: str | None = None
    mounts: list[str] = Field(default_factory=list)  # e.g. ["/host:/ctr:rw"]
    gpus: str | None = None  # e.g. "all" or "device=0"
    extra_args: list[str] = Field(default_factory=list)
    pass_envs: bool = True


@register_operator("docker", DockerOperatorConfig)
class DockerOperator(Operator):
    def __init__(self, config: DockerOperatorConfig):
        super().__init__(config)
        self.config: DockerOperatorConfig

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        c = self.config

        cmd = Command(exec="docker")
        cmd.add_arg("run")
        cmd.add_opt("--rm")

        if c.gpus is not None:
            cmd.add_opt("--gpus", c.gpus)
        if c.workdir is not None:
            cmd.add_opt("-w", c.workdir)

        for m in c.mounts:
            cmd.add_opt("-v", m, append=True)

        if c.pass_envs:
            # Don't inline env values into the docker command line (it leaks into logs/process lists).
            # Use `-e KEY` to forward the value from the docker client's environment, and rely on
            # SubprocessLauncher(env=...) to inject the actual values.
            for k in dict(envs).keys():
                cmd.add_opt("-e", str(k), append=True)

        for a in c.extra_args:
            cmd.add_opt(a)

        cmd.add_arg(c.image)
        cmd.add_arg("bash")
        cmd.add_arg("-lc")
        # Env is passed via `docker -e KEY` above; no need to export inside the payload.
        cmd.add_arg("\n".join(list(script)))
        return cmd
