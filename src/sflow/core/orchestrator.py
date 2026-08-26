# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import subprocess
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sflow.logging import CoalescingFileHandler, get_logger

from .command import Command
from .launcher import SubprocessLauncher
from .operator import Operator, ResourcesUnavailable
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
        # Orphan sweeps (dead-owner leftovers from crashed prior runs) already
        # executed this run, keyed by command. They reap by owner liveness rather
        # than by task, so the same sweep repeated per task would only re-pay the
        # container-daemon round trip on every single launch.
        self._completed_stale_reaps: set[str] = set()

        # Merge-pod gates already opened this run: (member_name, dep_name) pairs.
        # An in-group dependency that reached READY/COMPLETED had its in-pod gate
        # marker touched; recorded so we touch each edge once (idempotent, retried
        # on exec failure). See _signal_merge_gates.
        self._merge_gates_opened: set[tuple[str, str]] = set()

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
                    # Merge-pod follower: it runs as a background process inside its
                    # leader's shared pod, never as its own pod. The leader promotes
                    # it to RUNNING (below); skip launching it here.
                    if getattr(task, "is_merge_follower", False):
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
                    # Merge-pod leader: its followers run as background processes in
                    # the same pod, so promote them to RUNNING alongside it. Each
                    # keeps its own <task>.log + probes but is never launched alone.
                    if getattr(task, "is_merge_leader", False):
                        await self._promote_merge_followers(task)

                # Update task statuses based on completed subprocesses
                finished = []
                for name, proc_task in self._subprocess_tasks.items():
                    if not proc_task.done():
                        # Continue when the task is still running
                        continue

                    # Process the done tasks

                    finished.append(name)

                    t = self.workflow.get_task(name)
                    # A merged member already resolved from its OWN done-marker is
                    # done: its leader's execute() only lingers to hold the shared
                    # pod, so whatever that coroutine reports now (the pod's exit,
                    # or the teardown cancellation that deletes it) must not restate
                    # the task's outcome. Just reap the coroutine.
                    if t.status.is_terminal() and (
                        getattr(t, "is_merge_leader", False)
                        or getattr(t, "is_merge_follower", False)
                    ):
                        continue
                    try:
                        # Note: `result()` may raise CancelledError; treat it as cancellation.
                        exit_code = proc_task.result()
                        t.exit_code = exit_code
                        # A service (readiness-probed) whose process exits before it
                        # ever reached READY failed to start -- even on exit 0 (e.g. a
                        # failure masked by a `mpirun | tee` pipeline). Do NOT mark it
                        # COMPLETED (which would wrongly satisfy dependents that expect
                        # the service to be up); route it through the failure/retry
                        # path with a clear reason instead. But first double-confirm
                        # with a forced final readiness scan, in case the service
                        # became ready and exited in the probe scan gap (see
                        # _recheck_readiness_after_exit) -- only fail if it is STILL
                        # not READY afterwards.
                        before_ready = exit_code == 0 and self._exited_before_ready(t)
                        if before_ready:
                            await self._recheck_readiness_after_exit(t)
                            before_ready = self._exited_before_ready(t)
                        if exit_code == 0 and not before_ready:
                            await self._finalize_successful_task(t)
                        else:
                            # Get the exception
                            task_exception = proc_task.exception()
                            fail_reason = (
                                "service exited before readiness"
                                if before_ready
                                else "process exit"
                            )
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
                                    f"Task '{t.name}' failed ({fail_reason}; exit={exit_code}, "
                                    f"exception={task_exception}); "
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
                                    f"Task '{t.name}' failed ({fail_reason}; exit={exit_code}, "
                                    f"exception={task_exception})"
                                )
                                self._record_summary(
                                    "task_failed",
                                    t,
                                    reason=fail_reason,
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

                    # Merge-pod leader finished: mirror its outcome onto the
                    # followers that ran inside its shared pod (never launched on
                    # their own, so nothing else resolves them).
                    if getattr(t, "is_merge_leader", False):
                        await self._propagate_merge_leader_status(t)

                for name in finished:
                    ft = self.workflow.get_task(name)
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

                # Open in-pod gates for merged members whose in-group dependency just
                # reached READY/COMPLETED, so the gated member's subshell proceeds.
                await self._signal_merge_gates()

                # Resolve merged members from their OWN exit status, so the shared pod's
                # lifetime is decoupled from any member's status: a finished member --
                # e.g. the terminal task -- completes instead of hanging the workflow,
                # and the driver reclaims the pod at teardown once the DAG is done.
                await self._resolve_finished_merge_members()

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

            # Workflow finished: stop any still-running task processes (long-lived
            # READY services whose process never exits on its own) right now,
            # instead of leaving them streaming until interpreter shutdown. For
            # the k8s operator this cancels execute(), which stops the log stream
            # and deletes the pod promptly.
            await self._stop_remaining_subprocess_tasks(reason="workflow finished")

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

    def _exited_before_ready(self, task: Task) -> bool:
        """True when a service task's process exited before it ever became READY.

        A task with readiness probe(s) is a service: for dependents it is "satisfied"
        by reaching READY (up and serving), not merely by its process exiting. If the
        process exits while the task is still RUNNING (never READY) -- even with exit
        0, e.g. a startup failure masked into a 0 exit by a ``mpirun | tee`` pipeline,
        or a crash that returns 0 -- the service failed to start. Marking it COMPLETED
        would wrongly satisfy dependents (they would launch against a dead server), so
        the caller treats it as a failure.

        Returns False for batch tasks (no readiness probes) and for services that DID
        reach READY (``status == READY``): a later clean exit is fine there, since
        dependents were already unblocked when the service became ready.
        """
        if task.status == TaskStatus.READY:
            return False
        return any(
            getattr(p, "type", None) == ProbeType.READINESS
            for p in (getattr(task, "probes", None) or [])
        )

    async def _recheck_readiness_after_exit(self, task: Task) -> None:
        """Give a just-exited service one final readiness scan before failing it.

        There is a gap between a task's process exiting and the (interval-gated)
        readiness probe re-scanning its log: a service can log its readiness line and
        exit 0 in that gap, so ``status`` is still RUNNING at process-exit even though
        it DID become ready. To avoid a false failure, flush the (now complete) task
        log and force each readiness probe to check once more immediately; a match
        flips the task to READY (via :meth:`_run_probe`), and the caller then treats
        the exit as a clean finish rather than a startup failure. If no probe
        triggers, the service genuinely exited before ready.
        """
        self._flush_task_log_handlers()
        for probe in list(getattr(task, "probes", None) or []):
            if task.status == TaskStatus.READY:
                break
            if getattr(probe, "type", None) != ProbeType.READINESS:
                continue
            force_due = getattr(probe, "force_due", None)
            if callable(force_due):
                force_due()
            try:
                await self._run_probe(probe, task)
            except Exception:  # noqa: BLE001 - a final-scan error just means "not ready"
                _logger.debug(
                    "final readiness recheck for '%s' errored", task.name,
                    exc_info=True,
                )

    async def _finalize_successful_task(self, t: Task) -> None:
        """
        Run post-process work that must finish before DAG dependents can submit.

        ``COMPLETED`` is the dependency-satisfied state, so keep the task in a
        non-terminal status until result parsing and existing upload handling
        have finished.
        """
        t.status = TaskStatus.FINALIZING

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
        prev_attempt = probe.last_attempt
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

        # A check actually ran this tick (new last_attempt) -> refresh the summary's
        # Probe Traces section, so even a task stuck RUNNING (readiness never
        # satisfied) keeps showing the probe's latest attempt. No task-status event
        # fires while it sits RUNNING, so nothing else would trigger a render.
        if probe.last_attempt is not prev_attempt:
            self._record_summary("record_probe_attempt", task)

        if not triggered:
            return

        probe.status = ProbeStatus.TRIGGERED
        if probe.type == ProbeType.READINESS:
            readiness_probes = [
                p for p in task.probes if p.type == ProbeType.READINESS
            ]
            if any(p.status != ProbeStatus.TRIGGERED for p in readiness_probes):
                return
            # Hand the reusable resources back BEFORE flipping to READY. READY is
            # what unblocks the dependent the planner packed onto these very GPUs,
            # so the claim must already be gone by the time anything can observe
            # the transition. Releasing after would leave a window -- today a
            # narrow one, since submission runs on this same coroutine, but that
            # is a coincidence of the loop's shape, not a guarantee. Ordering it
            # this way makes the invariant hold no matter who observes READY.
            await self._release_ready_reusable_resources(task)
            task.status = TaskStatus.READY
            self._record_summary("task_ready", task)
            self._surface_network_fallback(task)
            for fname in getattr(task, "readiness_followers", []):
                try:
                    ftask = self.workflow.get_task(fname)
                except KeyError:
                    continue
                if ftask.status == TaskStatus.RUNNING:
                    await self._release_ready_reusable_resources(ftask)
                    ftask.status = TaskStatus.READY
                    self._record_summary("task_ready", ftask)
                    self._surface_network_fallback(ftask)
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

    async def _release_ready_reusable_resources(self, task: Task) -> None:
        """Hand back resources the planner marked reusable once the task is READY.

        ``resources.gpus.release_after: task_ready`` means later tasks may take
        this task's GPUs while it keeps serving -- the planner packs them onto the
        same devices on that basis. An operator that also holds a machine-local
        reservation (docker) must let go at the same moment, or it blocks the very
        reuse that was planned and the dependent task fails to acquire.

        Offloaded to a thread because the release takes a cross-process file lock
        (bounded, but up to ``_LOCK_TIMEOUT_S`` if a holder is wedged) and this
        runs on the event loop that drives every other task's probes. Still
        ``await``ed rather than fired-and-forgotten: the hand-back must be visible
        in the registry *before* READY propagates to the dependent that was packed
        onto these GPUs, or that task's acquire races the release and fails.
        """
        if getattr(task, "resource_release_after", {}).get("gpus") != "task_ready":
            return
        operator = getattr(task, "operator", None)
        if operator is None or type(operator).release_resources is (
            Operator.release_resources
        ):
            return
        try:
            # reusable=True: the task keeps running on the resource, it just
            # stops holding the claim. The operator needs that distinction --
            # dropping the record outright would let its own still-live workload
            # be re-read as a foreign one and block the planned reuse.
            await asyncio.to_thread(
                operator.release_resources, task_name=task.name, reusable=True
            )
        except Exception:
            # Best-effort, like the finally-path release: a cleanup error must not
            # replace the task's real outcome. But it is not harmless -- the
            # dependent the planner packed onto these GPUs will now fail to
            # acquire, and its error ("0 free") points nowhere near the cause. Log
            # it where the user will actually see it, same as that path does.
            _logger.warning(
                "Task '%s': handing back READY-reusable resources failed; tasks "
                "the planner packed onto its GPUs may not be able to acquire them",
                task.name,
                exc_info=True,
            )

    def _container_stale_reap_commands(self, task: Task) -> list[Command]:
        """Reap commands for *orphaned* containers from crashed prior runs, or
        ``[]``. Safe to run before launch: only dead-owner containers are removed,
        never a concurrent run's live ones."""
        operator = getattr(task, "operator", None)
        if operator is None:
            return []
        try:
            return list(operator.stale_reap_commands(task_name=task.name))
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

    async def _stop_remaining_subprocess_tasks(self, *, reason: str) -> None:
        """Cancel and await any still-running task processes, then clear the map.

        Used when the workflow finishes: long-lived READY services keep their
        subprocess (and log stream / pod) alive, so stop them promptly rather than
        leaving them running until the event loop tears down. Cancelling each
        task triggers its launcher/operator cleanup (k8s ``execute`` stops the
        stream and deletes the pod). Safe to call when nothing remains.
        """
        pending = [t for t in self._subprocess_tasks.values() if not t.done()]
        if pending:
            _logger.info(
                f"Stopping {len(pending)} still-running task process(es) ({reason})."
            )
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        self._subprocess_tasks.clear()

    async def _launch_task_with_timeout(self, task: Task, timeout: int | None = None):
        # Reap orphaned containers left by *crashed* prior runs (dead owning PID)
        # before launching, so their leftovers don't accumulate. This must not
        # touch a live concurrent run's containers, so it reaps by dead-owner, not
        # by name. Only container operators declare these; when there are some,
        # offload the blocking reap to a thread so a slow `docker` daemon can't
        # stall the loop. Ordinary tasks skip this entirely (no await).
        # Each distinct sweep runs at most once per run: it reaps by dead owner,
        # not by task, so repeating it before every launch would just re-pay the
        # container-daemon round trip without finding anything new.
        stale_reap = [
            cmd
            for cmd in self._container_stale_reap_commands(task)
            if cmd.as_str() not in self._completed_stale_reaps
        ]
        if stale_reap:
            self._completed_stale_reaps.update(cmd.as_str() for cmd in stale_reap)
            await asyncio.to_thread(self._run_teardown_commands, stale_reap)

        operator = getattr(task, "operator", None)
        acquires = (
            operator is not None
            and type(operator).acquire_resources is not Operator.acquire_resources
        )

        async def _acquire() -> None:
            """Acquire this task's external resources just before launch.

            e.g. the docker operator reserves its GPUs and pins the container to
            them; held only while the task runs, released in the finally. Only
            operators that override the hook take part -- ordinary tasks skip it
            entirely, so their launch timing is unchanged.

            Each attempt is offloaded to a thread (it does blocking file-lock and
            subprocess work) but must not block: an operator that wants to wait
            says so with ResourcesUnavailable and we sleep HERE, on the event
            loop. That keeps a waiting task cancellable and inside its `timeout`
            -- sleeping in the worker thread instead would detach from the task
            and leave the driver unkillable, since the interpreter joins executor
            threads on the way out.
            """
            while True:
                try:
                    acquired_gpus = await asyncio.to_thread(
                        operator.acquire_resources,
                        task_name=task.name,
                        envs=task.envs,
                    )
                except ResourcesUnavailable as e:
                    await asyncio.sleep(e.retry_after)
                    continue
                # Record the physical devices actually claimed so run reporting
                # names the real GPUs instead of the planner's provisional slice.
                if acquired_gpus:
                    task.reserved_gpu_indices = list(acquired_gpus)
                return

        def _run():
            # Operators that orchestrate their own multi-step, driver-managed run
            # (e.g. k8s: apply -> stream -> status-watch -> stop) get awaited
            # directly; everyone else launches one subprocess from build_command.
            # Either way the awaitable resolves to an int exit code.
            operator = getattr(task, "operator", None)
            if operator is not None and getattr(
                operator, "manages_own_execution", lambda: False
            )():
                return operator.execute(
                    launcher=self._subprocess_launcher,
                    output_logger=task.logger,
                    env=task.envs,
                    task_name=task.name,
                    # runnable_script carries the fail-fast prelude (set -e)
                    # for shell operators so a failed command fails the task; script
                    # stays the user's resolved lines.
                    script=task.runnable_script,
                    status_note=lambda note, _t=task: setattr(
                        _t, "status_detail", note
                    ),
                )
            return self._subprocess_launcher.run_async(
                task.launch_command,
                output_logger=task.logger,
                env=task.envs,
                task_name=task.name,
            )

        async def _acquire_and_run():
            # Acquire INSIDE the timeout, so `timeout:` bounds a GPU wait and not
            # just the run.
            if acquires:
                await _acquire()
            return await _run()

        try:
            # Called inside the try so a cancellation still reaches the release below.
            # wait_for, not `async with asyncio.timeout(...)`: that is 3.11+ and we
            # support 3.10. It raises asyncio.TimeoutError -- the builtin only from
            # 3.11 -- so catch that spelling, not TimeoutError.
            if timeout:
                return await asyncio.wait_for(_acquire_and_run(), timeout)
            return await _acquire_and_run()
        finally:
            # Runs on success, failure, timeout, and cancellation -- the only path
            # that guarantees the container is gone if the launch process was
            # SIGKILLed before it could honor --rm.
            self._teardown_task_containers(task)
            # Release the task's acquired resources (e.g. give its reserved GPUs
            # back to the registry). Best-effort; never break the run.
            #
            # Offloaded to a thread for the same reason the READY hand-back is
            # (see _release_ready_reusable_resources): the release takes a
            # cross-process file lock that is bounded but not instant -- up to
            # gpu_reservation._LOCK_TIMEOUT_S if a holder is wedged -- and this
            # runs on the event loop that drives every other task's probes.
            # Calling it inline would stall the whole run at every task teardown.
            #
            # The await is safe in a `finally`: once asyncio.to_thread has handed
            # the call to the executor, the worker runs to completion whatever
            # happens to this coroutine, so a cancellation arriving mid-release
            # can cost us the *wait* but never the release itself.
            if acquires:
                await asyncio.to_thread(
                    self._release_task_resources,
                    operator,
                    task,
                    # Hand over rather than publish when the planner scheduled a
                    # later task of THIS run onto these devices. The successor has
                    # not been submitted yet (that takes another poll tick), and in
                    # that gap a concurrent `sflow run` would happily take a device
                    # this workflow is still counting on -- failing the successor
                    # with "0 free" on a placement that was perfectly valid.
                    # `handover` keeps the claim owned by this run: the successor
                    # can take it, outsiders cannot. The last task on a device is
                    # not flagged, so completing IT frees the device for real.
                    #
                    # Also reached on failure/cancellation, where the successor will
                    # never run. Holding is the safe direction (never hands away a
                    # device the run may still need) and the run-end sweep
                    # (release_all_for_pid) clears anything left over.
                    handover=bool(getattr(task, "gpus_reused_downstream", False)),
                )

    @staticmethod
    def _release_task_resources(
        operator: Operator, task: Task, *, handover: bool
    ) -> None:
        """Release one task's acquired resources. Runs in a worker thread.

        Swallows its own errors rather than letting them reach the caller: this is
        cleanup running in a ``finally``, and raising would replace the task's real
        outcome with a teardown error.
        """
        try:
            if handover:
                operator.release_resources(task_name=task.name, handover=True)
            else:
                # Pass nothing when it would only restate the default. An operator
                # overriding release_resources with the older `(*, task_name)`
                # signature then keeps working for the ordinary release, as it did
                # before hand-over existed -- otherwise it would TypeError on EVERY
                # task, and since this is best-effort that failure would quietly
                # leak its claim for the rest of the run. Such an operator still
                # breaks on the hand-over branch above, exactly as it already does
                # on the READY path -- and an operator written against the older
                # `(*, task_name, reusable=False)` signature now breaks there too,
                # since that branch passes `handover`. Both are the documented
                # limit of what an out-of-tree operator gets for free.
                operator.release_resources(task_name=task.name)
        except Exception:
            # A failure leaks the task's claim (its GPUs stay reserved) for the
            # rest of the run and would otherwise strand every later task with an
            # unexplained "0 free" -- so say so, with the traceback, at a level the
            # user actually sees.
            _logger.warning(
                "Task '%s': releasing acquired resources failed; its "
                "reservation may stay held for the rest of the run",
                task.name,
                exc_info=True,
            )

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

    async def _promote_merge_followers(self, leader: Task) -> None:
        """Start a merge leader's followers alongside it (they share the leader's pod).

        Merge-pod followers are never in ``get_tasks_to_submit`` output we act on
        (the submit loop skips them): they run as background processes inside the
        leader's single container. When the leader is submitted, promote each
        still-unstarted follower to RUNNING -- resetting its probes and acquiring its
        monitor -- so its own readiness/failure probes and <task>.log are evaluated
        exactly like a normally launched task, without launching a second pod.
        """
        for name in getattr(leader, "merge_members", []) or []:
            if name == leader.name:
                continue
            try:
                follower = self.workflow.get_task(name)
            except KeyError:
                continue
            if follower.status != TaskStatus.INITIATED:
                continue
            follower.attempts = int(getattr(follower, "attempts", 0)) + 1
            for p in getattr(follower, "probes", []) or []:
                p.reset()
            self._record_summary("task_submitted", follower)
            follower.status = TaskStatus.RUNNING
            await self._acquire_task_monitor(follower)

    @staticmethod
    def _member_done_rc(task: Task) -> int | None:
        """Return a merged member's exit code once it announced completion, else None.

        Merging is k8s-only, so the marker parsing lives with its emitter in the k8s
        operator (``merged_member_exit_code``) and is reached duck-typed here -- the
        same convention ``_signal_merge_gates`` uses for ``open_merge_gate``, keeping
        core free of any plugin import. Operators without the method (local/slurm/
        docker, which never merge) simply report None.
        """
        operator = getattr(task, "operator", None)
        reader = getattr(operator, "merged_member_exit_code", None)
        if reader is None:
            return None
        try:
            return reader(task)
        except Exception:  # never let a log-read error break the poll loop
            return None

    async def _resolve_finished_merge_members(self) -> None:
        """Give each merged member's OWN exit status to the DAG when its script ends.

        Merged members run as background processes in a shared pod, so a member has
        no pod of its own to end, and the shared container stays up as long as ANY
        member still runs. Tying a member's status to that pod is what hung the run:
        a follower was only ever mirrored from the leader, and the leader itself
        waited on a pod its own siblings kept alive -- so a one-shot member (notably
        the workflow's TERMINAL task) could never reach COMPLETED.

        Every member -- leader included -- instead reports its own rc (the done-marker
        it wrote when its own script returned), and this forwards that to the DAG
        exactly as a normal task's process exit would: rc 0 finalizes it (result
        parsing + uploads), non-zero fails it. No sibling or pod state is consulted.

        This is what decouples the shared pod's lifetime from any member's status.
        The leader's ``execute()`` keeps running (holding the pod) after the leader's
        own task is resolved; once every task's status is terminal the DAG finishes
        and the driver's teardown cancels that ``execute()``, which deletes the pod --
        so the pod outlives every member and is reclaimed by the driver, not by a task.

        Members that have not finished report nothing (no marker yet) and are left
        alone, as is anything already resolved.
        """
        for task in self.workflow.get_tasks():
            if not (
                getattr(task, "is_merge_leader", False)
                or getattr(task, "is_merge_follower", False)
            ):
                continue
            if task.status not in (TaskStatus.RUNNING, TaskStatus.READY):
                continue
            rc = self._member_done_rc(task)
            if rc is None:
                continue  # still running: its script has not returned yet
            task.exit_code = rc
            if rc == 0:
                _logger.info("Merged member '%s' finished (exit=0)", task.name)
                await self._finalize_successful_task(task)
            else:
                task.status = TaskStatus.FAILED
                self._record_summary(
                    "task_failed", task, reason=f"merged member exited with {rc}"
                )

    async def _propagate_merge_leader_status(self, leader: Task) -> None:
        """Mirror a finished merge leader's terminal outcome onto its followers.

        The merged pod is one unit: its single container exit code is the leader's,
        and the followers ran inside it. So when the leader resolves, resolve each
        non-terminal follower the same way -- finalize on success (parsing its own
        <task>.log + running its uploads), fail/cancel on failure/cancel, or reset to
        INITIATED when the leader is retrying so it re-promotes on the next attempt.

        A follower that reached READY is "terminal", but in a merged pod its
        processes live and die with the leader's single container. So when the
        leader ends abnormally (FAILED/TIMEOUT/CANCELLED) or is retrying, a READY
        follower is no longer actually healthy and IS resolved too; only on a clean
        COMPLETED is a READY service left as-is. Followers already in a terminal
        FAILURE (or COMPLETED) state need no further action.
        """
        status = leader.status
        # Leader ended abnormally or is retrying -> the shared container is gone or
        # being recreated, so override a follower that only looks healthy (READY).
        override_ready_followers = status in (
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.CANCELLED,
            TaskStatus.INITIATED,
        )
        # Leader is retrying -> its pod (and the in-pod /tmp/sflow-merge-gate markers)
        # are recreated fresh on the next attempt, so forget which gates we opened for
        # THIS group. Its dependency members re-run and re-reach READY/COMPLETED, and
        # _signal_merge_gates re-touches the new pod's markers. Idempotency WITHIN a
        # pod instance is unchanged: the set repopulates as each gate reopens, and
        # other merge groups' opened edges are preserved.
        if status == TaskStatus.INITIATED and self._merge_gates_opened:
            group_members = set(getattr(leader, "merge_members", []) or [])
            self._merge_gates_opened = {
                edge
                for edge in self._merge_gates_opened
                if edge[0] not in group_members
            }
        for name in getattr(leader, "merge_members", []) or []:
            if name == leader.name:
                continue
            try:
                follower = self.workflow.get_task(name)
            except KeyError:
                continue
            if follower.status == TaskStatus.COMPLETED:
                continue  # already finalized; nothing to mirror
            if follower.status.is_terminal() and not (
                override_ready_followers and follower.status == TaskStatus.READY
            ):
                continue  # already FAILED/etc., or a READY service under a clean leader
            if status == TaskStatus.COMPLETED:
                follower.exit_code = 0
                await self._finalize_successful_task(follower)
            elif status == TaskStatus.INITIATED:
                # Leader is retrying -> reset so the next submit re-promotes it.
                follower.status = TaskStatus.INITIATED
            elif status == TaskStatus.CANCELLED:
                follower.status = TaskStatus.CANCELLED
                self._record_summary(
                    "task_cancelled", follower, reason="merge leader cancelled"
                )
            else:  # FAILED / TIMEOUT
                follower.exit_code = leader.exit_code
                follower.status = TaskStatus.FAILED
                self._record_summary(
                    "task_failed",
                    follower,
                    reason=f"merge leader '{leader.name}' failed",
                )

    async def _signal_merge_gates(self) -> None:
        """Open in-pod gates for merged members whose in-group dependency is met.

        A merged member may depend on another member of the same pod (see
        ``_plan_merge_groups``); its in-pod subshell blocks in ``_sflow_gate`` until
        the dependency is met. COMPLETED is observed in-pod via the dependency's
        exit-code file, but READY (a long-lived service) is known only to the driver,
        so when a dependency reaches READY (or COMPLETED) tell the leader's operator
        to touch the gate marker in the pod. Idempotent: each edge is opened once on
        the first successful exec; a failed exec is retried next tick (self-healing).
        The ``member RUNNING`` guard bounds the work to the window before the
        dependent starts.
        """
        for leader in self.workflow.get_tasks():
            if not getattr(leader, "is_merge_leader", False):
                continue
            if leader.status not in (TaskStatus.RUNNING, TaskStatus.READY):
                continue
            opener = getattr(leader.operator, "open_merge_gate", None)
            if not callable(opener):
                continue
            for member_name in getattr(leader, "merge_members", []) or []:
                try:
                    member = self.workflow.get_task(member_name)
                except KeyError:
                    continue
                if member.status != TaskStatus.RUNNING:
                    continue
                for dep_name in getattr(member, "merge_gate_after", []) or []:
                    key = (member_name, dep_name)
                    if key in self._merge_gates_opened:
                        continue
                    try:
                        dep = self.workflow.get_task(dep_name)
                    except KeyError:
                        continue
                    if dep.status in (TaskStatus.READY, TaskStatus.COMPLETED):
                        if await opener(dep_name):
                            self._merge_gates_opened.add(key)
                            # Once EVERY dependency of this member is open, its in-pod
                            # subshell stops waiting and its work actually begins. Tell
                            # the summary, so the timeline shows a real start instead of
                            # implying the member ran from submission (it was parked in
                            # the shared pod), and so its duration measures work, not wait.
                            if all(
                                (member_name, d) in self._merge_gates_opened
                                for d in (getattr(member, "merge_gate_after", []) or [])
                            ):
                                self._record_summary("task_gate_opened", member)

    def _surface_network_fallback(self, task: Task) -> None:
        """Warn + record when a task's pod(s) degraded RDMA -> TCP at runtime.

        Best-effort and duck-typed: operators that expose
        ``network_fallback_status`` (currently the k8s container operator) report
        whether the in-pod RDMA preamble fell back to slow sockets -- invisible to
        the driver because the pod log stream is offloaded straight to disk. Called
        once, when the task goes READY, so silent slow-TCP KV/NCCL transport is
        surfaced in the log and the run summary. Never raises.
        """
        operator = getattr(task, "operator", None)
        probe_fn = getattr(operator, "network_fallback_status", None)
        if probe_fn is None:
            return
        try:
            status = probe_fn(task)
        except Exception:
            _logger.debug("network_fallback_status hook failed", exc_info=True)
            return
        if status is None:
            return
        messages: list[str] = []
        if getattr(status, "rdma_nic_unusable", False):
            if getattr(status, "mnnvl_crossnode", False):
                # IB/RoCE NIC is down, but the task is in a rack-scale NVLink (MNNVL)
                # ComputeDomain: NCCL cross-node rides NVLink and the IB/RoCE NET is
                # only a fallback -- not a performance concern, so don't warn. Info
                # only.
                _logger.info(
                    "Task '%s': RDMA (IB/RoCE) NIC unusable in %d/%d pod(s) (%s), but "
                    "the task is in an MNNVL NVLink domain (ComputeDomain); NCCL "
                    "cross-node uses rack NVLink (IB/RoCE NET is fallback only).",
                    task.name,
                    status.pods_degraded,
                    status.pods_total,
                    status.reason,
                )
            else:
                # sflow no longer force-sets a socket fallback (it would also suppress
                # the rack-NVLink/MNNVL path NCCL/UCX auto-detect). Surface the
                # condition as an actionable hint: the libraries pick the transport,
                # and the user sets the socket-forcing envs only if their cluster has
                # no NVLink fabric (or an external IB plugin aborts on dead HCAs).
                messages.append(
                    f"RDMA NIC unusable in {status.pods_degraded}/{status.pods_total} "
                    f"pod(s) ({status.reason}); sflow did not force a fallback -- "
                    "NCCL/UCX will auto-select transport (rack NVLink/MNNVL if "
                    "present, else TCP). If this cluster has no NVLink fabric, "
                    "expect slow TCP for cross-node KV/NCCL/NIXL; set "
                    "NCCL_IB_DISABLE=1 / NCCL_NET_PLUGIN=none / NCCL_IBEXT_DISABLE=1 "
                    "yourself to force sockets (also avoids external-IB-plugin aborts "
                    "on dead HCAs)."
                )
        if getattr(status, "ucx_intra_node_tcp", False):
            transport = getattr(status, "ucx_transport", "") or "tcp"
            messages.append(
                f"UCX selected TCP for intra-node transport ({transport}); "
                "cuda_ipc/NVLink is not active. Expect low KV transfer performance. "
                "Intra-node, use regular (non-VMM) KV memory so classic cuda_ipc "
                "rides NVLink. For cross-node MNNVL this needs BOTH an IMEX "
                "ComputeDomain channel (set compute_domain.channel to a name "
                "or 'auto') AND the framework's fabric/VMM KV memory (e.g. vLLM "
                "--enable-sleep-mode + UCX_CUDA_IPC_ENABLE_MNNVL=y, recipe-owned)."
            )
        if getattr(status, "gpudirect_rdma_unavailable", False):
            reason = getattr(status, "gpudirect_rdma_reason", "") or "unknown reason"
            messages.append(
                f"GPUDirect RDMA unavailable ({reason}). RDMA may stage GPU "
                "buffers through host memory; expect lower KV/NCCL performance."
            )
        for message in messages:
            _logger.warning("Task '%s': %s", task.name, message)
            self._record_summary("record_network_warning", task, message)

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
