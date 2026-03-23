# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from datetime import timedelta

from sflow.logging import get_logger

from .launcher import SubprocessLauncher
from .outputs import collect_task_outputs
from .probe import Probe, ProbeStatus, ProbeType
from .task import Task, TaskStatus
from .workflow import Workflow

_logger = get_logger(__name__)


class Orchestrator:
    """
    Orchestrates the execution of a workflow.
    """

    def __init__(
        self,
        workflow: Workflow,
        poll_interval: int = 1,
        launcher: SubprocessLauncher | None = None,
        fail_fast: bool = True,
    ):
        self.workflow = workflow
        self._poll_interval = poll_interval
        self._fail_fast = bool(fail_fast)

        self._subprocess_launcher = launcher or SubprocessLauncher()
        self._subprocess_tasks = dict[str, asyncio.Task]()
        self._stop_event = asyncio.Event()
        self._stop_reason: str | None = None

    def request_stop(self, reason: str | None = None) -> None:
        """
        Request a graceful stop of the workflow execution.
        """
        self._stop_reason = reason or self._stop_reason
        self._stop_event.set()

    async def run(self):
        """
        Starts the orchestration loop.
        """

        _logger.info(f"Starting workflow: {self.workflow.name}")
        start_time = time.time()

        try:
            while not self.workflow.is_finished():
                if self._stop_event.is_set():
                    reason = self._stop_reason or "stop requested"
                    _logger.warning(
                        f"Stopping workflow '{self.workflow.name}' ({reason}). Cancelling running tasks."
                    )

                    # Cancel running subprocess tasks (best-effort).
                    for name, proc_task in list(self._subprocess_tasks.items()):
                        if not proc_task.done():
                            proc_task.cancel()
                        del self._subprocess_tasks[name]

                    # Mark all non-terminal tasks as CANCELLED so workflow can finish.
                    for t in self.workflow.get_tasks():
                        if not t.status.is_terminal():
                            t.status = TaskStatus.CANCELLED
                    break

                await asyncio.sleep(self._poll_interval)

                # Launch tasks simultaneously
                for task in self.workflow.get_tasks_to_submit():
                    if (
                        getattr(task, "next_retry_at", 0.0)
                        and time.time() < task.next_retry_at
                    ):
                        continue
                    _logger.info(f"Submitting task: {task.name}")
                    task.status = TaskStatus.RUNNING
                    task.attempts = int(getattr(task, "attempts", 0)) + 1
                    self._subprocess_tasks[task.name] = asyncio.create_task(
                        self._launch_task_with_timeout(task)
                    )

                # Update task statuses based on completed subprocesses
                finished = []
                for name, proc_task in self._subprocess_tasks.items():
                    if not proc_task.done():
                        # Continue when the task is still running
                        continue

                    # Process the done tasks

                    finished.append(name)

                    t = self.workflow.get_task(name)
                    try:
                        # Note: `result()` may raise CancelledError; treat it as cancellation.
                        exit_code = proc_task.result()
                        t.exit_code = exit_code
                        if exit_code == 0:
                            t.status = TaskStatus.COMPLETED
                            # MVP outputs parsing: parse from task log after completion.
                            if getattr(t, "output_specs", None):
                                await collect_task_outputs(t)
                        else:
                            # Get the exception
                            task_exception = proc_task.exception()
                            retries = getattr(t, "retries", None)
                            attempts = int(getattr(t, "attempts", 0))
                            if retries is not None and (attempts - 1) < int(
                                retries.count
                            ):
                                # Schedule retry with exponential backoff:
                                # failure #1 -> interval * backoff^0
                                # failure #2 -> interval * backoff^1
                                delay = float(retries.interval) * (
                                    float(retries.backoff) ** max(0, attempts - 1)
                                )

                                t.next_retry_at = time.time() + delay
                                # Reset for re-submission.
                                t.status = TaskStatus.INITIATED
                                # Keep the last exit code visible for observability while we retry.
                                for p in getattr(t, "probes", []) or []:
                                    # Reset probe streaks/scheduling too.
                                    p.status = ProbeStatus.INITIATED
                                _logger.warning(
                                    f"Task '{t.name}' failed (exit={exit_code}, exception={task_exception}); "
                                    f"retrying in {delay:.2f}s (attempt {attempts}/{1 + int(retries.count)})"
                                )
                            else:
                                t.status = TaskStatus.FAILED
                                _logger.error(
                                    f"Task '{t.name}' failed (exit={exit_code}, exception={task_exception})"
                                )
                    except asyncio.CancelledError:
                        t.exit_code = None
                        t.status = TaskStatus.CANCELLED
                        _logger.warning(f"Task '{t.name}' cancelled")

                for name in finished:
                    del self._subprocess_tasks[name]

                # Run probes
                for task in self.workflow.get_tasks_to_sync():
                    for probe in task.probes:
                        await self._run_probe(probe, task)

                # Fail-fast: if any task reaches FAILED, cancel remaining work so we don't hang
                # with blocked INITIATED tasks that can never become submittable.
                if self._fail_fast:
                    failed = [
                        t
                        for t in self.workflow.get_tasks()
                        if t.status == TaskStatus.FAILED
                    ]
                    if failed:
                        probe_failed = [
                            t for t in failed if getattr(t, "failed_by_probe", False)
                        ]
                        process_failed = [
                            t
                            for t in failed
                            if not getattr(t, "failed_by_probe", False)
                        ]
                        parts: list[str] = []
                        if probe_failed:
                            parts.append(
                                f"failure probe terminated: {', '.join(t.name for t in probe_failed)}"
                            )
                        if process_failed:
                            parts.append(
                                f"process exited with error: {', '.join(t.name for t in process_failed)}"
                            )
                        _logger.error(
                            f"Fail-fast: {'; '.join(parts)}. Cancelling remaining tasks."
                        )

                        # Cancel all running subprocess tasks (best-effort).
                        for name, proc_task in list(self._subprocess_tasks.items()):
                            if not proc_task.done():
                                proc_task.cancel()
                            del self._subprocess_tasks[name]

                        # Mark all non-terminal tasks as CANCELLED so workflow can finish.
                        for t in self.workflow.get_tasks():
                            if not t.status.is_terminal():
                                t.status = TaskStatus.CANCELLED

                        break

        except asyncio.CancelledError:
            # Cooperative cancellation path (e.g., app shutdown). Do best-effort cleanup
            # of in-flight subprocess tasks and mark remaining tasks as CANCELLED.
            self.request_stop("cancelled")
            for name, proc_task in list(self._subprocess_tasks.items()):
                if not proc_task.done():
                    proc_task.cancel()
                del self._subprocess_tasks[name]
            for t in self.workflow.get_tasks():
                if not t.status.is_terminal():
                    t.status = TaskStatus.CANCELLED
            raise
        except Exception as e:
            _logger.error(f"Workflow execution failed: {e}")
            raise

        finally:
            end_time = time.time()
            duration = timedelta(seconds=end_time - start_time)
            _logger.info(f"Workflow execution finished in {duration}")

    async def _run_probe(self, probe: Probe, task: Task):
        if probe.status == ProbeStatus.INITIATED and await probe.probe(task):
            probe.status = ProbeStatus.TRIGGERED
            if probe.type == ProbeType.READINESS:
                task.status = TaskStatus.READY
                for fname in getattr(task, "readiness_followers", []):
                    try:
                        ftask = self.workflow.get_task(fname)
                    except KeyError:
                        continue
                    if ftask.status == TaskStatus.RUNNING:
                        ftask.status = TaskStatus.READY
                        _logger.info(
                            f"Task '{fname}' set to READY (follows probe from '{task.name}')"
                        )
            elif probe.type == ProbeType.FAILURE:
                task.status = TaskStatus.FAILED
                task.failed_by_probe = True
                probe_detail = (
                    getattr(probe, "_pattern_display", None) or type(probe).__name__
                )
                _logger.error(
                    f"Failure probe triggered for task '{task.name}': "
                    f"pattern matched: '{probe_detail}'. "
                    f"The workflow will be terminated because of this probe — "
                    f"the task process was still running when the failure was detected."
                )
                for fname in getattr(task, "failure_followers", []):
                    try:
                        ftask = self.workflow.get_task(fname)
                    except KeyError:
                        continue
                    if ftask.status == TaskStatus.RUNNING:
                        ftask.status = TaskStatus.FAILED
                        ftask.failed_by_probe = True
                        _logger.error(
                            f"Task '{fname}' set to FAILED (follows probe from '{task.name}')"
                        )

    async def _launch_task_with_timeout(self, task: Task, timeout: int | None = None):
        if timeout:
            async with asyncio.timeout(timeout):
                return await self._subprocess_launcher.run_async(
                    task.launch_command,
                    output_logger=task.logger,
                    env=task.envs,
                    task_name=task.name,
                )
        else:
            return await self._subprocess_launcher.run_async(
                task.launch_command,
                output_logger=task.logger,
                env=task.envs,
                task_name=task.name,
            )
