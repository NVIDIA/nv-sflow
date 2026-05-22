# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from datetime import timedelta
from typing import Any

from sflow.logging import get_logger

from .launcher import SubprocessLauncher
from .outputs import collect_task_outputs
from .probe import Probe, ProbeStatus, ProbeTimeoutError, ProbeType
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
        execution_summary: Any | None = None,
    ):
        self.workflow = workflow
        self._poll_interval = poll_interval
        self._fail_fast = bool(fail_fast)
        self._execution_summary = execution_summary

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
        workflow_error = False
        workflow_detail: str | None = None

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
                            self._record_summary(
                                "task_cancelled", t, reason=reason
                            )
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
                    self._record_summary("task_unblocked", task)
                    task.attempts = int(getattr(task, "attempts", 0)) + 1
                    for p in getattr(task, "probes", []) or []:
                        p.reset()
                    self._record_summary("task_submitted", task)
                    task.status = TaskStatus.RUNNING
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
                            self._record_summary("task_completed", t)
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
                                # Reset for re-submission. Probe reset (deadlines,
                                # streaks) happens in the submit loop when the task
                                # transitions back to RUNNING.
                                t.status = TaskStatus.INITIATED
                                _logger.warning(
                                    f"Task '{t.name}' failed (exit={exit_code}, exception={task_exception}); "
                                    f"retrying in {delay:.2f}s (attempt {attempts}/{1 + int(retries.count)})"
                                )
                                self._record_summary(
                                    "task_retry",
                                    t,
                                    exit_code=exit_code,
                                    delay=delay,
                                )
                            else:
                                t.status = TaskStatus.FAILED
                                _logger.error(
                                    f"Task '{t.name}' failed (exit={exit_code}, exception={task_exception})"
                                )
                                self._record_summary(
                                    "task_failed",
                                    t,
                                    reason="process exit",
                                    exit_code=exit_code,
                                )
                    except asyncio.CancelledError:
                        t.exit_code = None
                        t.status = TaskStatus.CANCELLED
                        _logger.warning(f"Task '{t.name}' cancelled")
                        self._record_summary(
                            "task_cancelled", t, reason="cancelled"
                        )
                    except Exception as exc:
                        t.exit_code = None
                        t.status = TaskStatus.FAILED
                        reason = f"launcher error: {exc}"
                        _logger.error(f"Task '{t.name}' failed ({reason})")
                        self._record_summary("task_failed", t, reason=reason)
                        await self._cancel_sibling_subprocess_tasks(
                            t,
                            reason=f"cancelled after task '{t.name}' failed: {reason}",
                        )
                        raise

                for name in finished:
                    del self._subprocess_tasks[name]

                # Run probes
                for task in self.workflow.get_tasks_to_sync():
                    for probe in task.probes:
                        try:
                            await self._run_probe(probe, task)
                        except Exception as exc:
                            task.status = TaskStatus.FAILED
                            task.failed_by_probe = True
                            reason = f"probe error: {exc}"
                            _logger.error(f"Task '{task.name}' failed ({reason})")
                            self._record_summary("task_failed", task, reason=reason)
                            await self._cancel_sibling_subprocess_tasks(
                                task,
                                reason=(
                                    f"cancelled after task '{task.name}' failed: "
                                    f"{reason}"
                                ),
                            )
                            raise

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
                                self._record_summary(
                                    "task_cancelled", t, reason="fail-fast"
                                )

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
                    self._record_summary("task_cancelled", t, reason="cancelled")
            raise
        except Exception as e:
            workflow_error = True
            workflow_detail = str(e)
            _logger.error(f"Workflow execution failed: {e}")
            raise

        finally:
            end_time = time.time()
            duration = timedelta(seconds=end_time - start_time)
            _logger.info(f"Workflow execution finished in {duration}")
            summary_status = "FAILED" if workflow_error else self._workflow_summary_status()
            summary_detail = self._workflow_summary_detail() or workflow_detail
            self._record_summary(
                "workflow_finished", status=summary_status, detail=summary_detail
            )

    async def _run_probe(self, probe: Probe, task: Task):
        try:
            triggered = probe.status == ProbeStatus.INITIATED and await probe.probe(task)
        except ProbeTimeoutError as exc:
            _logger.error(
                f"Task '{task.name}' readiness probe timed out: {exc}"
            )
            task.status = TaskStatus.FAILED
            task.failed_by_probe = True
            self._record_summary(
                "task_failed",
                task,
                reason=f"readiness probe timed out: {exc}",
            )
            for fname in getattr(task, "readiness_followers", []):
                try:
                    ftask = self.workflow.get_task(fname)
                except KeyError:
                    continue
                if ftask.status == TaskStatus.RUNNING:
                    ftask.status = TaskStatus.FAILED
                    ftask.failed_by_probe = True
                    self._record_summary(
                        "task_failed",
                        ftask,
                        reason=f"readiness probe timed out: {task.name}",
                    )
                    _logger.error(
                        f"Task '{fname}' set to FAILED (follows timed-out probe from '{task.name}')"
                    )
            return

        if not triggered:
            return

        probe.status = ProbeStatus.TRIGGERED
        if probe.type == ProbeType.READINESS:
            readiness_probes = [
                p for p in task.probes if p.type == ProbeType.READINESS
            ]
            if any(p.status != ProbeStatus.TRIGGERED for p in readiness_probes):
                return
            task.status = TaskStatus.READY
            self._record_summary("task_ready", task)
            for fname in getattr(task, "readiness_followers", []):
                try:
                    ftask = self.workflow.get_task(fname)
                except KeyError:
                    continue
                if ftask.status == TaskStatus.RUNNING:
                    ftask.status = TaskStatus.READY
                    self._record_summary("task_ready", ftask)
                    _logger.info(
                        f"Task '{fname}' set to READY (follows probe from '{task.name}')"
                    )
        elif probe.type == ProbeType.FAILURE:
            task.status = TaskStatus.FAILED
            task.failed_by_probe = True
            self._record_summary("task_failed", task, reason="failure probe")
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
                    self._record_summary(
                        "task_failed",
                        ftask,
                        reason=f"failure probe: {task.name}",
                    )
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

    async def _cancel_sibling_subprocess_tasks(
        self,
        failed_task: Task,
        *,
        reason: str,
    ) -> None:
        cleanup_tasks: list[tuple[str, asyncio.Task]] = []
        for name, proc_task in list(self._subprocess_tasks.items()):
            if name == failed_task.name:
                continue
            try:
                sibling = self.workflow.get_task(name)
                if proc_task.done():
                    if sibling.status.is_terminal():
                        self._subprocess_tasks.pop(name, None)
                        continue

                    try:
                        exit_code = proc_task.result()
                    except asyncio.CancelledError:
                        sibling.exit_code = None
                        sibling.status = TaskStatus.CANCELLED
                        self._record_summary(
                            "task_cancelled", sibling, reason=reason
                        )
                    except Exception as exc:
                        sibling.exit_code = None
                        sibling.status = TaskStatus.FAILED
                        sibling_reason = f"launcher error: {exc}"
                        _logger.error(
                            "Sibling task '%s' failed while cleaning up after task '%s' failed (%s)",
                            name,
                            failed_task.name,
                            sibling_reason,
                        )
                        self._record_summary(
                            "task_failed", sibling, reason=sibling_reason
                        )
                    else:
                        sibling.exit_code = exit_code
                        if exit_code == 0:
                            sibling.status = TaskStatus.COMPLETED
                            if getattr(sibling, "output_specs", None):
                                await collect_task_outputs(sibling)
                            self._record_summary("task_completed", sibling)
                        else:
                            sibling.status = TaskStatus.FAILED
                            _logger.error(
                                "Sibling task '%s' failed while cleaning up after task '%s' failed (exit=%s)",
                                name,
                                failed_task.name,
                                exit_code,
                            )
                            self._record_summary(
                                "task_failed",
                                sibling,
                                reason="process exit",
                                exit_code=exit_code,
                            )
                    self._subprocess_tasks.pop(name, None)
                    continue

                proc_task.cancel()
                cleanup_tasks.append((name, proc_task))
                if not sibling.status.is_terminal():
                    sibling.exit_code = None
                    sibling.status = TaskStatus.CANCELLED
                    self._record_summary("task_cancelled", sibling, reason=reason)
            except Exception:
                _logger.error(
                    "Failed to clean up sibling task '%s' after task '%s' failed",
                    name,
                    failed_task.name,
                    exc_info=True,
                )

        if cleanup_tasks:
            results = await asyncio.gather(
                *(proc_task for _, proc_task in cleanup_tasks),
                return_exceptions=True,
            )
            for (name, _proc_task), result in zip(cleanup_tasks, results):
                if isinstance(result, asyncio.CancelledError):
                    continue
                if isinstance(result, BaseException):
                    _logger.error(
                        "Sibling task '%s' cleanup after task '%s' failed: %s",
                        name,
                        failed_task.name,
                        result,
                    )
            for name, _proc_task in cleanup_tasks:
                self._subprocess_tasks.pop(name, None)

    def _record_summary(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        if self._execution_summary is None:
            return
        method = getattr(self._execution_summary, method_name, None)
        if method is None:
            return
        try:
            method(*args, **kwargs)
        except Exception:
            _logger.debug("Execution summary hook failed", exc_info=True)

    def _workflow_summary_status(self) -> str:
        statuses = [t.status for t in self.workflow.get_tasks()]
        if any(status in {TaskStatus.FAILED, TaskStatus.TIMEOUT} for status in statuses):
            return "FAILED"
        if any(status == TaskStatus.CANCELLED for status in statuses):
            return "CANCELLED"
        if all(status in {TaskStatus.COMPLETED, TaskStatus.READY} for status in statuses):
            return "COMPLETED"
        return "RUNNING"

    def _workflow_summary_detail(self) -> str | None:
        tasks = self.workflow.get_tasks()
        failed = [
            t for t in tasks if t.status in {TaskStatus.FAILED, TaskStatus.TIMEOUT}
        ]
        if failed:
            names = ", ".join(t.name for t in failed)
            return (
                f"Workflow '{self.workflow.name}' failed: "
                f"{len(failed)} task(s) failed ({names})"
            )
        cancelled = [t for t in tasks if t.status == TaskStatus.CANCELLED]
        if cancelled:
            names = ", ".join(t.name for t in cancelled)
            return (
                f"Workflow '{self.workflow.name}' cancelled: "
                f"{len(cancelled)} task(s) cancelled ({names})"
            )
        return None
