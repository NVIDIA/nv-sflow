# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.operator_registry import register_operator


class SrunOperatorConfig(OperatorConfig):
    name: str
    type: Literal["srun"] = "srun"

    # --- Allocation / placement ---
    job_id: str | None = None
    nodes: int | str | None = None
    nodelist: list[str] = Field(default_factory=list)

    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    reservation: str | None = None
    time: str | None = None
    constraint: str | None = None
    exclusive: bool = False

    chdir: str | None = None

    # --- Resources ---
    cpus_per_task: int | str | None = None
    gpus: str | None = None  # e.g. "all", "1", "device=0"
    gpus_per_task: str | None = None
    gres: str | None = None
    mem: str | None = None
    mem_per_cpu: str | None = None

    ntasks: int | str | None = None
    ntasks_per_node: int | str | None = None

    # --- Logging / behavior ---
    export: str = "ALL"
    label: bool = True
    unbuffered: bool = True
    kill_on_bad_exit: bool = False
    overlap: bool = True
    wait: int | str | None = None

    # --- Pyxis / container (srun plugin flags) ---
    container_image: str | None = None
    container_name: str | None = None
    container_mount_home: bool = False
    container_writable: bool = True
    container_mounts: list[str] = Field(default_factory=list)  # "/h:/c:rw"
    container_workdir: str | None = None
    container_remap_root: bool = False

    mpi: str | None = None  # e.g. "pmix", "ucx", "ofi"

    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_and_coerce_types(self) -> "SrunOperatorConfig":
        """
        1. Pyxis: you can either start a container from an image OR attach to a named container,
           but not both at the same time.
        2. Coerce numeric string fields to int where possible.
        """
        if self.container_image and self.container_name:
            raise ValueError(
                "srun operator config: 'container_image' and 'container_name' cannot both be set"
            )

        # Coerce string values to int for numeric fields
        def _to_int(val: int | str | None) -> int | None:
            if val is None:
                return None
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                try:
                    return int(val)
                except ValueError:
                    # Keep as string if it can't be converted (e.g., unresolved expression)
                    return None
            return None

        # Convert numeric fields - only update if successfully converted
        if self.ntasks is not None:
            converted = _to_int(self.ntasks)
            if converted is not None:
                self.ntasks = converted

        if self.ntasks_per_node is not None:
            converted = _to_int(self.ntasks_per_node)
            if converted is not None:
                self.ntasks_per_node = converted

        if self.nodes is not None:
            converted = _to_int(self.nodes)
            if converted is not None:
                self.nodes = converted

        if self.cpus_per_task is not None:
            converted = _to_int(self.cpus_per_task)
            if converted is not None:
                self.cpus_per_task = converted

        if self.wait is not None:
            converted = _to_int(self.wait)
            if converted is not None:
                self.wait = converted

        return self


@register_operator("srun", SrunOperatorConfig)
class SrunOperator(Operator):
    def __init__(self, config: SrunOperatorConfig):
        super().__init__(config)
        self.config: SrunOperatorConfig

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        c = self.config
        command = Command(exec="srun")
        if c.job_id is not None:
            command.add_opt("--jobid", c.job_id)

        # Placement / scheduling
        if c.partition is not None:
            command.add_opt("--partition", c.partition)
        if c.account is not None:
            command.add_opt("--account", c.account)
        if c.qos is not None:
            command.add_opt("--qos", c.qos)
        if c.reservation is not None:
            command.add_opt("--reservation", c.reservation)
        if c.time is not None:
            command.add_opt("--time", c.time)
        if c.constraint is not None:
            command.add_opt("--constraint", c.constraint)
        if c.exclusive:
            command.add_opt("--exclusive")
        if c.chdir is not None:
            command.add_opt("--chdir", c.chdir)

        # Nodes / nodelist
        nodes = (
            c.nodes if c.nodes is not None else (len(c.nodelist) if c.nodelist else 0)
        )
        if nodes:
            command.add_opt("--nodes", nodes)
        if c.nodelist:
            command.add_opt("--nodelist", ",".join(c.nodelist))

        if c.ntasks is not None:
            command.add_opt("--ntasks", c.ntasks)
        if c.ntasks_per_node is not None:
            command.add_opt("--ntasks-per-node", c.ntasks_per_node)

        # Resources
        if c.cpus_per_task is not None:
            command.add_opt("--cpus-per-task", c.cpus_per_task)
        if c.gpus is not None:
            command.add_opt("--gpus", c.gpus)
        if c.gpus_per_task is not None:
            command.add_opt("--gpus-per-task", c.gpus_per_task)
        if c.gres is not None:
            command.add_opt("--gres", c.gres)
        if c.mem is not None:
            command.add_opt("--mem", c.mem)
        if c.mem_per_cpu is not None:
            command.add_opt("--mem-per-cpu", c.mem_per_cpu)

        # Behavior / logging
        command.add_opt("--job-name", task_name)
        if c.unbuffered:
            command.add_opt("--unbuffered")
        if c.export:
            command.add_opt("--export", c.export)
        if c.label:
            command.add_opt("--label")
        if c.kill_on_bad_exit:
            command.add_opt("--kill-on-bad-exit")
        if c.overlap:
            command.add_opt("--overlap")
        if c.wait is not None:
            command.add_opt("--wait", c.wait)
        if c.mpi is not None:
            command.add_opt("--mpi", c.mpi)

        # Pyxis container support
        if c.container_image is not None:
            command.add_opt("--container-image", c.container_image)
        if c.container_mount_home:
            command.add_opt("--container-mount-home")
        if not c.container_mount_home:
            command.add_opt("--no-container-mount-home")
        if c.container_name is not None:
            command.add_opt("--container-name", c.container_name)
        if c.container_writable:
            command.add_opt("--container-writable")

        # Merge container_mounts from config with any --container-mounts in extra_args
        all_mounts: list[str] = list(c.container_mounts) if c.container_mounts else []
        filtered_extra_args: list[str] = []
        i = 0
        while i < len(c.extra_args):
            arg = c.extra_args[i]
            if arg == "--container-mounts" and i + 1 < len(c.extra_args):
                # Next arg is the mount value
                extra_mounts = c.extra_args[i + 1].split(",")
                all_mounts.extend(extra_mounts)
                i += 2
            elif arg.startswith("--container-mounts="):
                # Value is part of the arg itself
                extra_mounts = arg.split("=", 1)[1].split(",")
                all_mounts.extend(extra_mounts)
                i += 1
            else:
                filtered_extra_args.append(arg)
                i += 1

        if all_mounts:
            command.add_opt("--container-mounts", ",".join(all_mounts))

        if c.container_workdir is not None:
            command.add_opt("--container-workdir", c.container_workdir)
        if c.container_remap_root:
            command.add_opt("--container-remap-root")

        for arg in filtered_extra_args:
            command.add_opt(arg)

        command.add_arg("bash")
        command.add_arg("-c")
        # Env is injected by SubprocessLauncher(env=...) and srun --export=ALL will propagate it
        # to remote tasks. Avoid embedding env exports into the script to prevent leaking values.
        command.add_arg("\n".join(list(script)))
        return command
