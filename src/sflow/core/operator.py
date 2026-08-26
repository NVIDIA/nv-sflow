# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence

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


class ResourcesUnavailable(Exception):
    """Raised by :meth:`Operator.acquire_resources` when a retry may succeed.

    The operator owns the *policy* (may I wait at all, how long, how often); the
    orchestrator owns the *waiting*, so it happens on the event loop where the
    task's ``timeout`` and Ctrl-C still apply. An operator that has exhausted its
    own budget raises an ordinary exception instead, which fails the task.
    """

    #: Floor for ``retry_after``. Zero would make the orchestrator's
    #: ``await asyncio.sleep(...)`` a bare yield, turning the retry loop into a
    #: busy-wait that burns a core while a task waits for resources.
    MIN_RETRY_AFTER = 0.05

    def __init__(self, message: str, *, retry_after: float = 5.0):
        super().__init__(message)
        self.retry_after = max(self.MIN_RETRY_AFTER, float(retry_after))


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

    def runs_shell_script(self) -> bool:
        """Whether the task ``script`` is run as a POSIX shell script.

        Default True: the script lines are joined and run by ``bash`` (bash / srun /
        ssh / docker_run / kubernetes operators), so sflow can prepend
        ``set -e`` to make a failed command fail the task (see
        :func:`sflow.utils.script.prepend_fail_fast`). The ``python`` operator runs
        the script as Python source (``python -c``), so it returns False -- a shell
        ``set -e`` prelude would be a Python SyntaxError.
        """
        return True

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
        task's process ends so nothing outlives the run.
        """
        return []

    def stale_reap_commands(self, *, task_name: str) -> list["Command"]:
        """Commands to reap *orphaned* resources from crashed prior runs.

        Default: none. Unlike :meth:`teardown_commands` (this run's own
        resources, removed by exact identity), these run before a task launches
        and must only remove things whose owning driver process is dead, so a
        concurrently running ``sflow`` on the same host is never disturbed.
        """
        return []

    def acquire_resources(
        self, *, task_name: str, envs: "Mapping[str, str]"
    ) -> list[int] | None:
        """Acquire per-task external resources just before the task launches.

        Default: none. The docker operator overrides this to reserve the task's
        GPUs from the machine-local registry (freed again in
        :meth:`release_resources`), so GPUs are held only while the task runs --
        the k8s per-pod model, rather than reserving for the whole run. Runs on
        the launch thread; may block (file lock / wait-for-gpus).

        Returns the *physical* GPU indices actually acquired, or ``None`` when
        nothing was reserved. The orchestrator records them on the task so run
        reporting (hardware monitor, execution summary) can name the real devices
        rather than the planner's provisional slice.

        MUST make a single non-blocking attempt and return promptly: it runs in a
        worker thread that cannot be cancelled, so anything that sleeps here would
        outlive the task and hang the driver at exit. To wait for a resource,
        raise :class:`ResourcesUnavailable` -- the orchestrator retries on the
        event loop, where the task's ``timeout`` and Ctrl-C still work.
        """
        return None

    def release_resources(
        self, *, task_name: str, reusable: bool = False, handover: bool = False
    ) -> None:
        """Release resources acquired in :meth:`acquire_resources`.

        Default: none. Best-effort and idempotent -- it may be called more than
        once, and without a matching successful acquire. The orchestrator calls
        it in the launch ``finally`` (success, failure, timeout, or cancellation)
        and, for resources the planner marks reusable once the task is READY, at
        that transition too.

        ``reusable=True`` marks the READY hand-back: the task is still running,
        so the resource becomes available to this run's later tasks while
        remaining attributable to it. ``handover=True`` marks the exit of a task
        whose devices a later task of this run is planned onto: the task is gone,
        but the resource must not go back to the whole host until that successor
        has claimed it. An implementation that cannot distinguish them may ignore
        both flags; a plain call (neither set) must fully release.
        """
        return None

    def manages_own_execution(self) -> bool:
        """Whether the orchestrator should run this task via :meth:`execute`.

        Default False: the task is launched as a single subprocess from
        ``build_command()`` by the launcher. Operators that orchestrate their own
        multi-step, driver-managed run (e.g. the kubernetes operator: apply the
        pod, stream logs as a separate process, watch pod status, stop on status
        change) return True so the orchestrator awaits :meth:`execute` instead.
        """
        return False

    async def execute(
        self,
        *,
        launcher: Any,
        output_logger: Any,
        env: Mapping[str, str],
        task_name: str,
        script: Sequence[str],
        status_note: Callable[[str | None], None] | None = None,
    ) -> int:
        """Run the task as a driver-managed flow and return its exit code.

        Only called when :meth:`manages_own_execution` is True. Receives the
        shared ``launcher`` (so sub-steps stream through the same per-task log),
        the task's ``output_logger`` / ``env`` / ``name`` / ``script``. Must return
        an int exit code (0 == success) just like a subprocess, and propagate
        ``asyncio.CancelledError`` on teardown after cleaning up.

        ``status_note`` is an optional callback the operator may call with a short
        live sub-status string (or ``None`` to clear it) -- e.g. a k8s pod's
        ``"Pending: Unschedulable"`` while the task is RUNNING but not yet started;
        it is surfaced next to the task status in the UI. Default: not implemented.
        """
        raise NotImplementedError(
            f"operator '{getattr(self.config, 'type', '?')}' does not implement execute()"
        )
