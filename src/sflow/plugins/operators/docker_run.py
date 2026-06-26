# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from typing import Any
from typing import Literal

from pydantic import Field, field_validator

from sflow.core.command import Command
from sflow.core.log_offload import offload_enabled, task_log_path, wrap_with_prefixer
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.operator_registry import register_operator
from sflow.utils.container import (
    append_runtime_mounts as append_runtime_mount_specs,
    validate_container_image_reference,
)


def _safe_container_name(*parts: str) -> str:
    raw = "-".join(part for part in parts if part)
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw).strip("-_.")
    return (sanitized or "sflow-task")[:128]


class DockerRunOperatorConfig(OperatorConfig):
    name: str
    type: Literal["docker_run"] = "docker_run"

    image: str
    workdir: str | None = None
    mounts: list[str] = Field(default_factory=list)  # e.g. ["/host:/ctr:rw"]
    gpus: str | None = None  # e.g. "all" or "device=0"
    extra_args: list[str] = Field(default_factory=list)
    pass_envs: bool = True
    auto_mount_runtime_dirs: bool = True
    # Per-task log offload, ON by default. When enabled (also via the
    # SFLOW_OFFLOAD_TASK_LOGS env / --offload-task-logs flag, which take
    # precedence), the task's output is redirected on the host through a
    # compute-side prefixer into <task>.log instead of streaming through the
    # sflow driver's pump. Auto-falls back to streaming on an interactive
    # TTY / --tui session.
    log_to_file: bool = True

    def container_images(self) -> list[str]:
        return [self.image] if self.image else []

    def mount_specs(self) -> list[str]:
        return list(self.mounts or [])

    def append_runtime_mounts(self, mounts: Sequence[str]) -> None:
        if not self.auto_mount_runtime_dirs:
            return
        self.mounts = append_runtime_mount_specs(list(self.mounts or []), list(mounts))

    @field_validator("image")
    @classmethod
    def image_must_be_valid(cls, value: str) -> str:
        validate_container_image_reference(
            value,
            source="docker_run operator config: 'image'",
        )
        return value


@register_operator("docker_run", DockerRunOperatorConfig)
class DockerRunOperator(Operator):
    def __init__(self, config: DockerRunOperatorConfig):
        super().__init__(config)
        self.config: DockerRunOperatorConfig
        self._assigned_nodes: list[str] = []
        self._node_hosts: dict[str, Any] = {}

    def _offload_enabled(self) -> bool:
        return offload_enabled(self.config.log_to_file)

    def writes_own_task_log(self) -> bool:
        # In offload mode the host-side redirect owns <task>.log, so sflow must
        # not also attach a FileHandler to that path (single-writer invariant).
        return self._offload_enabled()

    def _maybe_offload(
        self, cmd: Command, *, task_name: str, envs: Mapping[str, str]
    ) -> Command:
        """Wrap the docker invocation so its output is prefixed and written to
        <task>.log on the host, taking the driver out of the per-line pump.

        Works uniformly for the single-node ``docker run`` and the multi-node
        ``bash -lc`` forms by piping the whole (shlex-joined) command through the
        prefixer; ``${PIPESTATUS[0]}`` preserves the container/script exit code.
        """
        log_path = task_log_path(envs, task_name)
        if not (self._offload_enabled() and log_path):
            return cmd
        wrapped = wrap_with_prefixer(
            shlex.join(cmd.as_list()),
            workflow_out_dir=envs.get("SFLOW_WORKFLOW_OUTPUT_DIR"),
            task_name=task_name,
            redirect_to=log_path,
        )
        offloaded = Command(exec="bash")
        offloaded.add_arg("-c")
        offloaded.add_arg(wrapped)
        return offloaded

    def apply_backend_context(
        self,
        *,
        backend: Any,
        assigned_nodes: Sequence[str],
        artifacts: Sequence[Any],
        cuda_visible_devices: str | None = None,
        gpu_count: int | None = None,
    ) -> None:
        # gpu_count unused: Docker GPU comes from the cuda_visible_devices slice below.
        self._assigned_nodes = list(assigned_nodes or [])
        host_for_node = getattr(backend, "host_for_node", None)
        self._node_hosts = {
            node_name: host_for_node(node_name)
            for node_name in self._assigned_nodes
            if callable(host_for_node)
        }
        if any(
            host is not None
            and (getattr(host, "docker_host", None) or getattr(host, "context", None))
            for host in self._node_hosts.values()
        ):
            self.config.auto_mount_runtime_dirs = False
        if cuda_visible_devices:
            self.config.gpus = f"device={cuda_visible_devices}"

    def _build_docker_command(
        self,
        *,
        task_name: str,
        node_name: str | None,
        host: Any | None,
        script: Sequence[str],
        envs: Mapping[str, str],
        container_name: str | None = None,
    ) -> Command:
        c = self.config
        cmd = Command(exec="docker")
        if host is not None:
            if getattr(host, "docker_host", None):
                cmd.add_arg("--host")
                cmd.add_arg(str(host.docker_host))
            elif getattr(host, "context", None):
                cmd.add_arg("--context")
                cmd.add_arg(str(host.context))

        cmd.add_arg("run")
        cmd.add_arg("--rm")
        if container_name:
            cmd.add_arg("--name")
            cmd.add_arg(container_name)

        if c.gpus is not None:
            cmd.add_arg("--gpus")
            cmd.add_arg(c.gpus)
        if c.workdir is not None:
            cmd.add_arg("-w")
            cmd.add_arg(c.workdir)

        host_mounts = list(getattr(host, "mounts", None) or []) if host else []
        for m in [*c.mounts, *host_mounts]:
            cmd.add_arg("-v")
            cmd.add_arg(m)

        if c.pass_envs:
            for k in dict(envs).keys():
                cmd.add_arg("-e")
                cmd.add_arg(str(k))

        host_extra_args = list(getattr(host, "extra_args", None) or []) if host else []
        for a in [*c.extra_args, *host_extra_args]:
            cmd.add_arg(a)

        cmd.add_arg(c.image)
        cmd.add_arg("bash")
        cmd.add_arg("-lc")
        cmd.add_arg("\n".join(list(script)))
        return cmd

    def _cleanup_command(self, host: Any | None, container_name: str) -> str:
        parts = ["docker"]
        if host is not None:
            if getattr(host, "docker_host", None):
                parts.extend(["--host", str(host.docker_host)])
            elif getattr(host, "context", None):
                parts.extend(["--context", str(host.context)])
        parts.extend(["rm", "-f", container_name])
        return shlex.join(parts)

    def _launch_specs(self, task_name: str) -> list[tuple[str, Any | None, str]]:
        """``(node_name, host, container_name)`` for every container this task runs.

        ``build_command`` and :meth:`teardown_commands` both derive container names
        from here so the driver can always reap a container by its deterministic
        ``--name`` even after the ``docker run`` client was killed.
        """
        launch_nodes = self._assigned_nodes or [""]
        return [
            (
                node_name,
                self._node_hosts.get(node_name),
                _safe_container_name("sflow", task_name, node_name),
            )
            for node_name in launch_nodes
        ]

    def teardown_commands(self, *, task_name: str) -> list[Command]:
        """Force-remove this task's containers (one per node).

        Killing the foreground ``docker run`` client never stops the
        daemon-managed container, so the orchestrator runs these after the task
        exits (and before relaunch) to guarantee no container outlives the run.
        """
        commands: list[Command] = []
        for _node_name, host, container_name in self._launch_specs(task_name):
            cmd = Command(exec="docker")
            if host is not None:
                if getattr(host, "docker_host", None):
                    cmd.add_arg("--host")
                    cmd.add_arg(str(host.docker_host))
                elif getattr(host, "context", None):
                    cmd.add_arg("--context")
                    cmd.add_arg(str(host.context))
            cmd.add_arg("rm")
            cmd.add_arg("-f")
            cmd.add_arg(container_name)
            commands.append(cmd)
        return commands

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        launch_specs = self._launch_specs(task_name)

        if len(launch_specs) == 1:
            node_name, host, container_name = launch_specs[0]
            # Name the container so the driver can force-remove it on teardown:
            # killing the foreground `docker run` client does not stop the
            # daemon-managed container, so `--rm` alone can leak it.
            cmd = self._build_docker_command(
                task_name=task_name,
                node_name=node_name or None,
                host=host,
                script=script,
                envs=envs,
                container_name=container_name,
            )
            return self._maybe_offload(cmd, task_name=task_name, envs=envs)

        lines = [
            "set -euo pipefail",
            "status=0",
            "pids=\"\"",
            "cleanup() {",
        ]
        run_lines: list[str] = []
        for node_name, host, container_name in launch_specs:
            lines.append(
                f"  {self._cleanup_command(host, container_name)} >/dev/null 2>&1 || true"
            )
            docker_cmd = self._build_docker_command(
                task_name=task_name,
                node_name=node_name,
                host=host,
                script=script,
                envs=envs,
                container_name=container_name,
            )
            run_lines.append(f"{shlex.join(docker_cmd.as_list())} &")
            run_lines.append('pids="$pids $!"')
        lines.extend(
            [
                "}",
                "trap cleanup EXIT",
                "trap 'cleanup; exit 143' HUP INT TERM",
                *run_lines,
                "for pid in $pids; do",
                "  if ! wait \"$pid\"; then status=1; fi",
                "done",
                "exit \"$status\"",
            ]
        )
        cmd = Command(exec="bash")
        cmd.add_arg("-lc")
        cmd.add_arg("\n".join(lines))
        return self._maybe_offload(cmd, task_name=task_name, envs=envs)
