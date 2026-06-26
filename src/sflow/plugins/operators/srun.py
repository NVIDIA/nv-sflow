# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from sflow.core import log_offload
from sflow.core.command import Command
from sflow.core.command_log import register_command_family
from sflow.core.log_offload import (
    OFFLOAD_TASK_LOGS_ENV,
    offload_enabled,
    offload_env_override,
    wrap_with_prefixer,
)
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.operator_registry import register_operator
from sflow.utils.container import (
    append_runtime_mounts as append_runtime_mount_specs,
    extract_container_images_from_extra_args,
    is_valid_container_image,
    local_artifact_mounts,
    merge_container_mounts_from_extra_args,
    validate_container_image_reference,
)

register_command_family("slurm", {"srun"}, filename="slurm_cmds.log")

# Back-compat re-exports of the (now shared) offload helpers, kept so existing
# imports in tests and scripts/bench_offload_logging.py keep working.
# OFFLOAD_TASK_LOGS_ENV is imported above (now the backend-neutral canonical name).
_offload_env_override = offload_env_override
_stdout_is_tty = log_offload.stdout_is_tty
_LOG_PREFIX_HELPER_SRC = log_offload.LOG_PREFIX_HELPER_SRC


_SLURM_TO_SFLOW_RUNTIME_ENV: tuple[tuple[str, str], ...] = (
    ("SFLOW_BACKEND_JOB_ID", "${SLURM_JOB_ID:-${SLURM_JOBID:-}}"),
    ("SFLOW_BACKEND_NODELIST", "${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"),
    ("SFLOW_BACKEND_NUM_NODES", "${SLURM_NNODES:-}"),
    ("SFLOW_BACKEND_STEP_ID", "${SLURM_STEP_ID:-}"),
    ("SFLOW_TASK_NODE_NAME", "${SLURMD_NODENAME:-}"),
    ("SFLOW_TASK_NODE_INDEX", "${SLURM_NODEID:-}"),
    ("SFLOW_TASK_PROCESS_ID", "${SLURM_PROCID:-}"),
    ("SFLOW_TASK_LOCAL_PROCESS_ID", "${SLURM_LOCALID:-}"),
    ("SFLOW_TASK_NUM_PROCESSES", "${SLURM_NTASKS:-}"),
)


def _slurm_runtime_env_prelude() -> list[str]:
    return [
        f'if [ -n "{source_expr}" ]; then export {target}="{source_expr}"; fi'
        for target, source_expr in _SLURM_TO_SFLOW_RUNTIME_ENV
    ]


def _is_valid_container_image(image: str) -> bool:
    """Backward-compatible alias for the public container image validator."""
    return is_valid_container_image(image)


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
    # Per-task log offload, ON by default. When enabled, srun writes the per-task
    # log itself via --output (offload) and a compute-side prefixer reproduces
    # sflow's log format, so the driver no longer pumps task content line-by-line.
    # The env channel OFFLOAD_TASK_LOGS_ENV takes precedence over this field (see
    # SrunOperator._offload_enabled). Auto-falls back to streaming on an
    # interactive TTY / --tui session.
    log_to_file: bool = True

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

    def container_images(self) -> list[str]:
        images: list[str] = []
        if self.container_image:
            images.append(self.container_image)
        images.extend(extract_container_images_from_extra_args(list(self.extra_args)))
        return images

    def mount_specs(self) -> list[str]:
        mounts, _filtered_extra_args = merge_container_mounts_from_extra_args(
            self.container_mounts or [],
            list(self.extra_args or []),
        )
        return append_runtime_mount_specs([], mounts)

    def uses_container(self) -> bool:
        return bool(self.container_image or self.container_name or self.container_images())

    def append_runtime_mounts(self, mounts: Sequence[str]) -> None:
        if not (self.container_image or self.container_name):
            return
        self.container_mounts = append_runtime_mount_specs(
            list(self.container_mounts or []),
            list(mounts),
        )

    def runtime_warnings(self) -> list[str]:
        warnings: list[str] = []
        if self.uses_container():
            creds_path = Path.home() / ".config" / "enroot" / ".credentials"
            if not creds_path.exists():
                warnings.append(
                    f"srun operator uses container images but enroot credentials "
                    f"file not found at {creds_path}. "
                    f"Container pulls from authenticated registries (e.g. nvcr.io) "
                    f"may fail. See: https://github.com/NVIDIA/enroot/blob/master/doc/cmd/import.md"
                )
        # Mirror offload_enabled() exactly so the warning reflects what actually
        # happens: it applies the env override > config precedence AND the
        # interactive-TTY fallback (offload auto-streams on a TTY/--tui). The
        # earlier inline check omitted the TTY case and falsely warned that
        # offload was enabled while output was actually being streamed.
        if offload_enabled(self.log_to_file):
            warnings.append(
                "srun per-task log offload is enabled: the task container needs "
                "python3 (preferred) or bash >= 5 to reproduce sflow's "
                "millisecond log timestamps; otherwise offloaded logs fall back "
                "to second-resolution timestamps."
            )
        return warnings

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

        if self.container_image:
            validate_container_image_reference(
                self.container_image,
                source="srun operator config: 'container_image'",
            )
        for _img_val in extract_container_images_from_extra_args(list(self.extra_args)):
            validate_container_image_reference(
                _img_val,
                source="srun operator config: '--container-image' in extra_args",
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

    def _offload_enabled(self) -> bool:
        """Whether this task offloads its log to srun --output (aligned mode)."""
        return offload_enabled(self.config.log_to_file)

    def writes_own_task_log(self) -> bool:
        # In offload mode slurmstepd owns <task>.log, so sflow must not also
        # attach a FileHandler to that path (single-writer invariant).
        return self._offload_enabled()

    def apply_backend_context(
        self,
        *,
        backend: Any,
        assigned_nodes: Sequence[str],
        artifacts: Sequence[Any],
        cuda_visible_devices: str | None = None,
        gpu_count: int | None = None,
    ) -> None:
        # gpu_count unused: Slurm GPU comes from --gpus flags + backend CUDA_VISIBLE_DEVICES.
        job_id = "0"
        full_nodelist: list[str] = []
        if getattr(backend, "allocation", None):
            allocation = backend.allocation
            job_id = str(getattr(allocation, "allocation_id", None) or "0")
            full_nodelist = [n.name for n in getattr(allocation, "nodes", [])]

        effective_nodelist = list(assigned_nodes or full_nodelist)
        if self.config.job_id in (None, "", "0"):
            self.config.job_id = job_id
        if not self.config.nodelist:
            self.config.nodelist = effective_nodelist
        if self.config.nodes in (None, 0):
            self.config.nodes = len(effective_nodelist)

        auto_mounts = local_artifact_mounts(artifacts)
        if auto_mounts:
            self.config.append_runtime_mounts(auto_mounts)

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        c = self.config
        command = Command(exec="srun")
        # Offload only engages when we know where slurmstepd should write the log.
        offload = self._offload_enabled() and bool(envs.get("SFLOW_TASK_OUTPUT_DIR"))
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
        if offload:
            # slurmstepd writes the per-task log directly, taking the sflow driver
            # out of the per-line byte path. stderr merges into stdout because
            # --error is omitted, keeping a single output file / single writer.
            command.add_opt(
                "--output",
                os.path.join(str(envs["SFLOW_TASK_OUTPUT_DIR"]), f"{task_name}.log"),
            )
        if c.unbuffered:
            command.add_opt("--unbuffered")
        if c.export:
            command.add_opt("--export", c.export)
        # In offload mode the rank label is folded into the prefixer instead, so
        # the file matches stream mode exactly; --label would otherwise double it.
        if c.label and not offload:
            command.add_opt("--label")
        if c.kill_on_bad_exit:
            command.add_opt("--kill-on-bad-exit")
        if c.overlap:
            command.add_opt("--overlap")
        if c.wait is not None:
            command.add_opt("--wait", c.wait)
        if c.mpi is not None:
            command.add_opt("--mpi", c.mpi)

        # Pyxis container support — only emit container flags when a container is in use
        _has_container = (
            c.container_image is not None
            or c.container_name is not None
            or any(
                a.startswith("--container-image") or a.startswith("--container-name")
                for a in c.extra_args
            )
        )
        if _has_container:
            if c.container_image is not None:
                command.add_opt("--container-image", c.container_image)
            if c.container_name is not None:
                command.add_opt("--container-name", c.container_name)
            if c.container_mount_home:
                command.add_opt("--container-mount-home")
            else:
                command.add_opt("--no-container-mount-home")
            if c.container_writable:
                command.add_opt("--container-writable")
            if c.container_workdir is not None:
                command.add_opt("--container-workdir", c.container_workdir)
            if c.container_remap_root:
                command.add_opt("--container-remap-root")

        # Merge container_mounts from config with any --container-mounts in extra_args.
        all_mounts, filtered_extra_args = merge_container_mounts_from_extra_args(
            c.container_mounts or [],
            list(c.extra_args),
        )

        if _has_container and all_mounts:
            command.add_opt("--container-mounts", ",".join(all_mounts))

        for arg in filtered_extra_args:
            command.add_opt(arg)

        command.add_arg("bash")
        command.add_arg("-c")
        # Env is injected by SubprocessLauncher(env=...) and srun --export=ALL will propagate it
        # to remote tasks. The prelude uses only Slurm-provided variable names, so values are
        # resolved inside the step rather than embedded in the logged command.
        script_body = "\n".join([*_slurm_runtime_env_prelude(), *list(script)])
        if offload:
            command.add_arg(
                self._wrap_script_with_prefixer(script_body, envs, task_name)
            )
        else:
            command.add_arg(script_body)
        return command

    def _wrap_script_with_prefixer(
        self, script_body: str, envs: Mapping[str, str], task_name: str
    ) -> str:
        """Wrap the task script so its merged stdout/stderr flows through the
        prefixer. srun captures the pipeline via ``--output`` (no shell redirect),
        and ``${PIPESTATUS[0]}`` makes the task's exit status (not the prefixer's)
        propagate to srun and the orchestrator.
        """
        return wrap_with_prefixer(
            script_body,
            workflow_out_dir=envs.get("SFLOW_WORKFLOW_OUTPUT_DIR"),
            task_name=task_name,
            redirect_to=None,
        )
