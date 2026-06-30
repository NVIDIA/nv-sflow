# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import subprocess
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sflow.logging import CoalescingFileHandler, get_logger

from .command import Command
from .launcher import SubprocessLauncher
from .outputs import collect_task_outputs
from .results import collect_task_result
from .probe import Probe, ProbeStatus, ProbeTimeoutError, ProbeType
from .task import Task, TaskStatus
from .uploads import UploadResult, run_task_uploads
from .workflow import Workflow

if TYPE_CHECKING:
    from .monitor import MonitorConsumer, MonitorRegistry
    from .storage import StorageTarget

_logger = get_logger(__name__)

# Task lifecycle transitions surfaced as vertical markers on the monitor timeline.
# Maps the execution-summary hook name to the short event label drawn on charts.
_MONITOR_EVENT_LABELS = {
    "task_submitted": "submit",
    "task_ready": "ready",
    "task_completed": "done",
    "task_failed": "fail",
    "task_cancelled": "cancel",
}


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
        storage_targets: "dict[str, StorageTarget] | None" = None,
        monitor_registry: "MonitorRegistry | None" = None,
    ):
        self.workflow = workflow
        self._poll_interval = poll_interval
        self._fail_fast = bool(fail_fast)
        self._execution_summary = execution_summary

        self._subprocess_launcher = launcher or SubprocessLauncher()
        self._subprocess_tasks = dict[str, asyncio.Task]()
        self._stop_event = asyncio.Event()
        self._stop_reason: str | None = None

        self._storage_targets = storage_targets or {}

        # Hardware monitor registry (plan-time schedule). The orchestrator fires
        # task-monitor triggers; the workflow monitor is owned by the app layer.
        self._monitor_registry: "MonitorRegistry | None" = monitor_registry
        # Owners of task monitors currently held (acquired, not yet released).
        self._held_monitors: dict[str, "MonitorConsumer"] = {}

    async def _acquire_task_monitor(self, task: Task) -> None:
        registry = self._monitor_registry
        monitor = task.monitor
        if registry is None or monitor is None or monitor.owner in self._held_monitors:
            return
        self._held_monitors[monitor.owner] = monitor
        try:
            await registry.acquire(monitor)
        except Exception:
            _logger.debug("Monitor acquire failed", exc_info=True)

    async def _release_task_monitor(self, task: Task) -> None:
        registry = self._monitor_registry
        monitor = task.monitor
        if registry is None or monitor is None:
            return
        if self._held_monitors.pop(monitor.owner, None) is None:
            return
        try:
            await registry.release(monitor)
        except Exception:
            _logger.debug("Monitor release failed", exc_info=True)

    async def _release_all_task_monitors(self) -> None:
        registry = self._monitor_registry
        if registry is None or not self._held_monitors:
            return
        for monitor in list(self._held_monitors.values()):
            try:
                await registry.release(monitor)
            except Exception:
                _logger.debug("Monitor release failed", exc_info=True)
        self._held_monitors.clear()

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
                    await self._release_all_task_monitors()
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
                    # Fire this task's monitor trigger (start collectors on its
                    # nodes, or reuse if already running for another consumer).
                    await self._acquire_task_monitor(task)

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
                            await self._finalize_successful_task(t)
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
                    ft = self.workflow.get_task(name)
                    # Once a task is terminal (not a pending retry), let its
                    # operator finalize <task>.log -- e.g. the k8s operator swaps
                    # in the complete container log it captured on early stop.
                    # Success was already finalized in _finalize_successful_task
                    # (before result parsing), so only handle failed/cancelled here.
                    if ft.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                        self._finalize_task_log(ft)
                    del self._subprocess_tasks[name]
                    # The task's process exited (completed/failed/retry); release
                    # its monitor. A node still referenced by the workflow monitor
                    # or another task stays up (singleton reuse). NOTE: tasks that
                    # reached READY are not in `finished` (their process is still
                    # alive), so their monitor keeps running until teardown.
                    await self._release_task_monitor(ft)

                # Surface buffered per-task log lines to disk before evaluating
                # probes. LogWatchProbe reads <task>.log from disk, but the
                # per-task CoalescingFileHandler defers flushes -- a service that
                # logs its readiness line then goes idle (e.g. an inference server
                # waiting for the first request) leaves that final line buffered.
                # Without this flush the probe never observes "ready", the task
                # never reaches READY, and dependents are never submitted.
                self._flush_task_log_handlers()

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

                        await self._release_all_task_monitors()
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
            await self._release_all_task_monitors()
            raise
        except Exception as e:
            workflow_error = True
            workflow_detail = str(e)
            _logger.error(f"Workflow execution failed: {e}")
            raise

        finally:
            # Release any task monitors still held (e.g. READY services whose
            # process never exited during the loop) so nothing lingers. The
            # workflow monitor is released by the app layer after run() returns.
            await self._release_all_task_monitors()
            end_time = time.time()
            duration = timedelta(seconds=end_time - start_time)
            _logger.info(f"Workflow execution finished in {duration}")
            summary_status = "FAILED" if workflow_error else self._workflow_summary_status()
            summary_detail = self._workflow_summary_detail() or workflow_detail
            self._record_summary(
                "workflow_finished", status=summary_status, detail=summary_detail
            )

    def _task_log_dir(self, t: Task) -> str | None:
        """Directory holding the task's ``<task>.log`` (from its file handler)."""
        logger = getattr(t, "logger", None)
        for handler in getattr(logger, "handlers", []) or []:
            if isinstance(handler, CoalescingFileHandler):
                base = getattr(handler, "baseFilename", None)
                if base:
                    return os.path.dirname(base)
        return None

    def _release_task_log_handler(self, t: Task) -> None:
        """Flush, close, and detach the per-task stream file handler.

        Lets a single writer replace ``<task>.log`` afterwards (e.g. the k8s
        complete-log swap). Only safe once the task is terminal (no retry), since
        a re-submission would otherwise have no handler to write through.
        """
        logger = getattr(t, "logger", None)
        if logger is None:
            return
        for handler in list(getattr(logger, "handlers", []) or []):
            if isinstance(handler, CoalescingFileHandler):
                try:
                    handler.flush()
                    handler.close()
                except Exception:
                    pass
                logger.removeHandler(handler)

    def _finalize_task_log(self, t: Task) -> None:
        """Let a terminal task's operator rewrite ``<task>.log`` in place.

        The k8s operator uses this to swap in the complete container log it
        captured when it stopped the live stream early. Best-effort: a no-op for
        operators that don't override the hook, and never raises. The operator
        calls ``release_handler`` itself, but only when it actually rewrites the
        file, so unaffected tasks keep their handler.
        """
        op = getattr(t, "operator", None)
        finalize = getattr(op, "finalize_task_log", None)
        if not callable(finalize):
            return
        try:
            finalize(
                task_name=t.name,
                task_output_dir=self._task_log_dir(t),
                release_handler=lambda: self._release_task_log_handler(t),
            )
        except Exception as exc:
            _logger.debug(f"finalize_task_log failed for '{t.name}': {exc}")

    async def _finalize_successful_task(self, t: Task) -> None:
        """
        Run post-process work that must finish before DAG dependents can submit.

        ``COMPLETED`` is the dependency-satisfied state, so keep the task in a
        non-terminal status until result parsing and existing upload handling
        have finished.
        """
        t.status = TaskStatus.FINALIZING

        # Swap the complete log into <task>.log BEFORE parsing, so result/output
        # parsing (which reads <task>.log) sees the full log even if the live
        # stream was stopped early (k8s). No-op for other operators / no early stop.
        self._finalize_task_log(t)

        # MVP outputs parsing: parse from task log after process success.
        if getattr(t, "output_specs", None):
            await collect_task_outputs(t)

        # New consolidated result parsing (best-effort by default).
        # See docs/developer/dev-notes/result-parsing.md.
        if getattr(t, "result_config", None) is not None:
            try:
                payload = await collect_task_result(t)
            except Exception as e:
                _logger.warning(
                    f"Result collection failed for task '{t.name}': {e}"
                )
            else:
                # Surface non-fatal result issues (e.g. a required spec that
                # didn't match, a cast failure) at orchestrator level so users
                # don't have to open result.json to find out their key metric is
                # missing.
                if payload and not payload.get("ok", True):
                    errs = payload.get("errors") or []
                    detail = "; ".join(str(e) for e in errs) or "ok=false"
                    _logger.warning(
                        f"Task '{t.name}' result has issues: {detail}"
                    )

        # Preserve existing behavior: uploads still run before the task's final
        # outcome is recorded, and on_error=fail can flip the task to FAILED.
        if self._storage_targets and getattr(t, "uploads", None):
            upload_results: list[UploadResult] = []
            ok = await run_task_uploads(
                t,
                self._storage_targets,
                results=upload_results,
            )
            self._record_summary("record_uploads", upload_results)
            if not ok:
                t.status = TaskStatus.FAILED
                _logger.error(
                    f"Task '{t.name}' marked FAILED due to upload error (on_error=fail)"
                )
        else:
            ok = True

        if ok:
            t.status = TaskStatus.COMPLETED

        if t.status == TaskStatus.COMPLETED:
            self._record_summary("task_completed", t)
        else:
            self._record_summary("task_failed", t, reason="upload error")

    def _flush_task_log_handlers(self) -> None:
        """Flush stream-mode task log writers so file-watching probes read fresh data.

        The per-task ``CoalescingFileHandler`` (stream mode) batches flushes and
        only flushes on the *next* emit after its interval -- so the last line a
        task writes before going idle can sit in the buffer indefinitely. Probes
        such as :class:`LogWatchProbe` read ``<task>.log`` from disk, so an idle
        service's buffered readiness line would otherwise never be observed.
        Flushing here (once per poll) bounds that staleness to one loop tick.

        Only ``CoalescingFileHandler`` is flushed. In offload mode the operator
        writes ``<task>.log`` itself and sflow's ``DeferredTaskLogHandler`` buffers
        driver-side diagnostics that are appended exactly once, post-exit, by the
        launcher; flushing it mid-run would append into the file while the operator
        is still writing it, violating the single-writer invariant.
        """
        for t in self.workflow.get_tasks():
            logger = getattr(t, "logger", None)
            if logger is None:
                continue
            for handler in list(getattr(logger, "handlers", []) or []):
                if not isinstance(handler, CoalescingFileHandler):
                    continue
                try:
                    handler.flush()
                except Exception:
                    # Best-effort: a closed/rotating handler must never break the loop.
                    pass

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

    def _container_teardown_commands(self, task: Task) -> list[Command]:
        """The (best-effort) reap commands an operator declares, or ``[]``.

        Only container operators return any; everything else is a no-op, which
        lets the caller skip the teardown machinery entirely for ordinary tasks.
        """
        operator = getattr(task, "operator", None)
        if operator is None:
            return []
        try:
            return list(operator.teardown_commands(task_name=task.name))
        except Exception:
            return []

    @staticmethod
    def _run_teardown_commands(commands: list[Command]) -> None:
        for cmd in commands:
            try:
                subprocess.run(
                    cmd.as_list(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                )
            except Exception:
                # Teardown is best-effort; never let cleanup break the run.
                pass

    def _teardown_task_containers(self, task: Task) -> None:
        """Synchronously reap external resources (e.g. Docker containers) a task
        leaves behind.

        Kept synchronous so the ``finally`` below can reap during cancellation
        unwinding, where an ``await`` would itself be cancelled but a blocking
        ``subprocess.run`` still completes. Killing the foreground ``docker run``
        client never stops the daemon-managed container, so without this a
        long-running server (e.g. an inference server held at READY until
        teardown) would keep running on the host after the workflow.
        """
        self._run_teardown_commands(self._container_teardown_commands(task))

    async def _launch_task_with_timeout(self, task: Task, timeout: int | None = None):
        # Reap any stale container from a crashed prior run / previous attempt so a
        # deterministic --name cannot collide ("name already in use"). Only container
        # operators declare teardown commands; when there are some, offload the
        # blocking reap to a thread so a slow `docker` daemon can't stall the loop.
        # Ordinary tasks skip this entirely (no await), preserving launch timing.
        stale_reap = self._container_teardown_commands(task)
        if stale_reap:
            await asyncio.to_thread(self._run_teardown_commands, stale_reap)
        try:
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
        finally:
            # Runs on success, failure, timeout, and cancellation -- the only path
            # that guarantees the container is gone if the launch process was
            # SIGKILLed before it could honor --rm.
            self._teardown_task_containers(task)

    async def _cancel_sibling_subprocess_tasks(
        self,
        failed_task: Task,
        *,
        reason: str,
    ) -> None:
        cleanup_tasks: list[tuple[str, asyncio.Task]] = []
        done_tasks: list[tuple[str, asyncio.Task, Task]] = []
        for name, proc_task in list(self._subprocess_tasks.items()):
            if name == failed_task.name:
                continue
            try:
                sibling = self.workflow.get_task(name)
                if proc_task.done():
                    done_tasks.append((name, proc_task, sibling))
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

        for name, proc_task, sibling in done_tasks:
            if sibling.status.is_terminal():
                self._subprocess_tasks.pop(name, None)
                continue

            try:
                exit_code = proc_task.result()
            except asyncio.CancelledError:
                sibling.exit_code = None
                sibling.status = TaskStatus.CANCELLED
                self._record_summary("task_cancelled", sibling, reason=reason)
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
                self._record_summary("task_failed", sibling, reason=sibling_reason)
            else:
                sibling.exit_code = exit_code
                if exit_code == 0:
                    await self._finalize_successful_task(sibling)
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

    def _record_summary(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        # Mirror task lifecycle transitions onto the monitor timeline (best-effort,
        # never affects the execution-summary dispatch below).
        self._record_monitor_event(method_name, args)
        if self._execution_summary is None:
            return
        method = getattr(self._execution_summary, method_name, None)
        if method is None:
            return
        try:
            method(*args, **kwargs)
        except Exception:
            _logger.debug("Execution summary hook failed", exc_info=True)

    def _record_monitor_event(self, method_name: str, args: tuple[Any, ...]) -> None:
        """Stamp a task status change onto the monitor registry as a timeline event.

        No-op unless a monitor is active and the transition is one we surface
        (submit/ready/done/fail/cancel) on a real Task argument.
        """
        registry = self._monitor_registry
        if registry is None:
            return
        label = _MONITOR_EVENT_LABELS.get(method_name)
        if label is None or not args:
            return
        name = getattr(args[0], "name", None)
        if not isinstance(name, str):
            return
        try:
            registry.record_task_event(time.time(), name, label)
        except Exception:
            _logger.debug("Monitor event record failed", exc_info=True)

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
