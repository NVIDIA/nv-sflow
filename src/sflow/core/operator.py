# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from sflow.core.command import Command


class OperatorConfig(BaseModel):
    """
    Base configuration for an Operator.

    This is intentionally minimal for Step A: we are introducing the concept without
    wiring it into the main YAML schema yet.
    """

    type: str

    def container_images(self) -> list[str]:
        """Return container image references this operator may launch."""
        return []

    def mount_specs(self) -> list[str]:
        """Return host/container mount specs this operator may launch with."""
        return []

    def uses_container(self) -> bool:
        """Return whether this operator config describes a container launch."""
        return bool(self.container_images())

    def append_runtime_mounts(self, mounts: Sequence[str]) -> None:
        """Append runtime directory mounts when the operator supports host mounts."""
        return None

    def runtime_warnings(self) -> list[str]:
        """Return backend/operator-specific dry-run warnings."""
        return []


class Operator(ABC):
    """
    Abstract base class for task execution operators (Airflow-style).

    Contract:
    - Operator instances are configured ONLY via an OperatorConfig object.
    - Operators must be able to build a launch Command from script/envs.
    """

    def __init__(self, config: OperatorConfig):
        self.config = config

    def apply_backend_context(
        self,
        *,
        backend: Any,
        assigned_nodes: Sequence[str],
        artifacts: Sequence[Any],
        cuda_visible_devices: str | None = None,
        gpu_count: int | None = None,
    ) -> None:
        """Allow operators to consume backend allocation/placement context.

        cuda_visible_devices is the client-planned GPU slice (GPU-env backends only);
        gpu_count is the resolved resources.gpus.count, passed for every backend.
        """
        return None

    def writes_own_task_log(self) -> bool:
        """Whether the operator writes the per-task ``<task>.log`` itself.

        Default False: sflow's launcher pumps the subprocess output into the
        per-task log. Operators that redirect output to the file directly (e.g.
        srun ``--output`` offload) return True so the app skips attaching its own
        FileHandler to that path, preserving a single writer.
        """
        return False

    @abstractmethod
    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        """
        Build the command that launches the task.
        """
        raise NotImplementedError

    def teardown_commands(self, *, task_name: str) -> list["Command"]:
        """Commands to reap external resources a task leaves behind.

        Default: none. Container operators override this to force-remove their
        containers, since killing the launch process does not necessarily stop a
        daemon-managed container. The orchestrator runs these best-effort after a
        task's process ends (and before a relaunch) so nothing outlives the run.
        """
        return []
