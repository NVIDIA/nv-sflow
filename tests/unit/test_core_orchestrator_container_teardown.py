# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator must reap an operator's external resources (e.g. Docker
containers) around every task launch.

Killing the foreground ``docker run`` client never stops the daemon-managed
container, so a long-running server held at READY until teardown would otherwise
keep running on the host after the workflow finishes. The orchestrator runs the
operator's ``stale_reap_commands`` before launch (reap orphans from crashed
prior runs, by dead owner) and its ``teardown_commands`` in a ``finally`` (reap
this run's own containers left when the launch process was SIGKILLed).
"""

import asyncio
import logging
import time
import types

import pytest

from sflow.core import orchestrator as orch_mod
from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig, ResourcesUnavailable
from sflow.core.orchestrator import Orchestrator
from sflow.core.probe import Probe, ProbeType
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow

# Pre-launch orphan sweep (by dead owner) vs post-task own-container reap.
_STALE_REAP = ["docker", "rm", "-f", "orphan-sweep"]
_TEARDOWN = ["docker", "rm", "-f", "own-server"]


class _ContainerOperator(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="fake_container"))

    def build_command(self, *, task_name, script, envs) -> Command:
        return Command(exec="true")

    def stale_reap_commands(self, *, task_name: str) -> list[Command]:
        return [
            Command(exec="docker").add_arg("rm").add_arg("-f").add_arg("orphan-sweep")
        ]

    def teardown_commands(self, *, task_name: str) -> list[Command]:
        return [
            Command(exec="docker").add_arg("rm").add_arg("-f").add_arg("own-server")
        ]


class _NoopLauncher:
    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
        return 0


class _HangingLauncher:
    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
        await asyncio.sleep(3600)
        return 0


class _FailingLauncher:
    async def run_async(self, command, output_logger=None, env=None, **kwargs) -> int:
        return 1


def _orchestrator(launcher) -> Orchestrator:
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    return Orchestrator(workflow=wf, poll_interval=0.01, launcher=launcher)


def _patch_subprocess_run(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(orch_mod.subprocess, "run", fake_run)
    return calls


def test_teardown_runs_before_and_after_a_successful_launch(monkeypatch):
    calls = _patch_subprocess_run(monkeypatch)
    task = Task(name="server", operator=_ContainerOperator(), logger=logging.getLogger("t"))

    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))

    assert rc == 0
    # Orphan sweep before launch, own-container reap after.
    assert calls == [_STALE_REAP, _TEARDOWN]


def test_teardown_runs_when_the_task_is_cancelled(monkeypatch):
    calls = _patch_subprocess_run(monkeypatch)
    task = Task(name="server", operator=_ContainerOperator(), logger=logging.getLogger("t"))
    orch = _orchestrator(_HangingLauncher())

    async def _run() -> None:
        launch = asyncio.create_task(orch._launch_task_with_timeout(task))
        await asyncio.sleep(0.05)  # let the pre-launch reap + run_async start
        launch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch

    asyncio.run(_run())

    # Pre-launch orphan sweep, then the finally own-reap during cancellation.
    # Exact sequence, matching the success-path test above: membership alone
    # would also pass if the sweep repeated or the two came out reversed.
    assert calls == [_STALE_REAP, _TEARDOWN]


def test_non_container_operator_has_no_teardown(monkeypatch):
    calls = _patch_subprocess_run(monkeypatch)

    class _PlainOperator(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

    task = Task(name="plain", operator=_PlainOperator(), logger=logging.getLogger("t"))

    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))

    assert rc == 0
    assert calls == []  # default teardown_commands() is empty -> no docker calls


def test_acquire_runs_before_launch_and_release_after(monkeypatch):
    """Operators that override the resource hooks get acquire() before the launch
    command is built and release() in the finally."""
    _patch_subprocess_run(monkeypatch)
    events: list[str] = []

    class _AcquiringOperator(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_acquire"))

        def build_command(self, *, task_name, script, envs) -> Command:
            events.append("build")
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs) -> None:
            events.append("acquire")

        def release_resources(self, *, task_name, reusable: bool = False) -> None:
            events.append("release")

    task = Task(name="t", operator=_AcquiringOperator(), logger=logging.getLogger("t"))
    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))

    assert rc == 0
    assert events == ["acquire", "build", "release"]


def test_plain_operator_is_not_touched_by_resource_hooks(monkeypatch):
    """A plain operator (base no-op hooks) is skipped entirely -- no thread hop,
    no timing change -- and just runs."""
    _patch_subprocess_run(monkeypatch)

    class _Plain(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_plain2"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

    task = Task(name="p", operator=_Plain(), logger=logging.getLogger("t"))
    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))
    assert rc == 0


def test_acquired_gpu_indices_are_recorded_on_the_task(monkeypatch):
    """What acquire_resources claims is stored on the task, so run reporting can
    name the physical devices instead of the planner's provisional slice."""
    _patch_subprocess_run(monkeypatch)

    class _ReservingOperator(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_reserve"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs):
            return [4, 5]

    task = Task(name="t", operator=_ReservingOperator(), logger=logging.getLogger("t"))
    task.cuda_visible_devices = "0,1"  # the plan-time guess
    assert task.reserved_gpu_indices is None

    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))

    assert rc == 0
    assert task.reserved_gpu_indices == [4, 5]


def test_no_reservation_leaves_the_task_field_unset(monkeypatch):
    """An operator that acquires nothing must not fabricate a device list."""
    _patch_subprocess_run(monkeypatch)

    class _NoReservationOperator(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_noreserve"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs):
            return None

    task = Task(
        name="t", operator=_NoReservationOperator(), logger=logging.getLogger("t")
    )
    asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))
    assert task.reserved_gpu_indices is None


class _AlwaysReadyProbe(Probe):
    """Readiness probe that reports ready on its first check."""

    def __init__(self):
        super().__init__(type=ProbeType.READINESS, interval=0, timeout=10)

    async def check(self, task) -> bool:
        return True


def _drive_to_ready(orch, task):
    """Run the task's readiness probe once, taking it through the READY path."""
    asyncio.run(orch._run_probe(task.probes[0], task))


def test_orphan_sweep_runs_once_per_run_not_once_per_task(monkeypatch):
    """The dead-owner sweep is identical for every task, so repeating it before
    each launch would only re-pay the container-daemon round trip."""
    calls = _patch_subprocess_run(monkeypatch)
    orch = _orchestrator(_NoopLauncher())

    for name in ("t1", "t2", "t3"):
        task = Task(
            name=name, operator=_ContainerOperator(), logger=logging.getLogger(name)
        )
        assert asyncio.run(orch._launch_task_with_timeout(task)) == 0

    assert calls.count(_STALE_REAP) == 1, "sweep must not repeat per task"
    # Per-task teardown of that task's OWN containers still runs every time.
    assert calls.count(_TEARDOWN) == 3


def test_each_run_sweeps_again(monkeypatch):
    """The suppression is per-run state, so a fresh Orchestrator sweeps again."""
    calls = _patch_subprocess_run(monkeypatch)
    for _ in range(2):
        task = Task(
            name="t", operator=_ContainerOperator(), logger=logging.getLogger("t")
        )
        asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))
    assert calls.count(_STALE_REAP) == 2


# ---------------------------------------------------------------------------
# waiting for resources: cancellable, bounded, leak-free
# ---------------------------------------------------------------------------


class _WaitsThenSucceeds(Operator):
    """Signals 'not yet' N times, then acquires. Never sleeps (thread-safe)."""

    def __init__(self, refusals: int):
        super().__init__(OperatorConfig(type="waits"))
        self.remaining = refusals
        self.attempts = 0
        self.released = 0
        self.reusable_releases = 0

    def build_command(self, *, task_name, script, envs) -> Command:
        return Command(exec="true")

    def acquire_resources(self, *, task_name, envs):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ResourcesUnavailable("board full", retry_after=0.01)
        return [3, 4]

    def release_resources(self, *, task_name: str, reusable: bool = False) -> None:
        self.released += 1
        self.reusable_releases += int(reusable)


def test_orchestrator_retries_until_resources_free_up(monkeypatch):
    _patch_subprocess_run(monkeypatch)
    op = _WaitsThenSucceeds(refusals=3)
    task = Task(name="t", operator=op, logger=logging.getLogger("t"))

    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))

    assert rc == 0
    assert op.attempts == 4                    # 3 refusals + the successful one
    assert task.reserved_gpu_indices == [3, 4]  # devices recorded for reporting
    assert op.released == 1                     # and handed back in the finally


def test_waiting_for_resources_is_cancellable(monkeypatch):
    """A task waiting on resources must respond to cancellation promptly.

    The wait happens on the event loop, not inside an uncancellable worker
    thread -- otherwise Ctrl-C leaves the driver blocked until the wait expires
    (the interpreter joins executor threads on the way out).
    """
    _patch_subprocess_run(monkeypatch)
    op = _WaitsThenSucceeds(refusals=10_000)  # would wait ~100s
    task = Task(name="t", operator=op, logger=logging.getLogger("t"))

    async def _run():
        orch = _orchestrator(_NoopLauncher())
        fut = asyncio.ensure_future(orch._launch_task_with_timeout(task))
        await asyncio.sleep(0.05)
        fut.cancel()
        started = asyncio.get_running_loop().time()
        with pytest.raises(asyncio.CancelledError):
            await fut
        return asyncio.get_running_loop().time() - started

    elapsed = asyncio.run(asyncio.wait_for(_run(), timeout=10))
    assert elapsed < 1.0, f"cancellation took {elapsed:.2f}s"
    # Cancelled mid-wait still releases: acquire runs inside the try/finally.
    assert op.released == 1


def test_task_timeout_bounds_the_resource_wait(monkeypatch):
    """`timeout:` must cover acquiring resources, not just running the command."""
    _patch_subprocess_run(monkeypatch)
    op = _WaitsThenSucceeds(refusals=10_000)
    task = Task(name="t", operator=op, logger=logging.getLogger("t"))

    async def _run():
        orch = _orchestrator(_NoopLauncher())
        with pytest.raises(asyncio.TimeoutError):
            await orch._launch_task_with_timeout(task, timeout=1)

    asyncio.run(asyncio.wait_for(_run(), timeout=15))
    assert op.released == 1  # timing out still hands the resources back


def test_ready_releases_resources_the_planner_marked_reusable(monkeypatch):
    """`gpus.release_after: task_ready` must free the operator's hold at READY.

    The planner packs a dependent task onto the same devices on that basis, so a
    hold that outlives READY blocks the reuse it planned.
    """
    _patch_subprocess_run(monkeypatch)
    op = _WaitsThenSucceeds(refusals=0)
    task = Task(name="server", operator=op, logger=logging.getLogger("t"))
    task.resource_release_after["gpus"] = "task_ready"
    task.probes = [_AlwaysReadyProbe()]

    _drive_to_ready(_orchestrator(_NoopLauncher()), task)

    assert task.status == TaskStatus.READY
    assert op.released == 1, "READY must hand back task_ready-reusable resources"
    # ...as `reusable`, never as `handover`: the two mean different things to an
    # operator holding a machine-local claim. `handover` says the task is GONE, so
    # a registry would stop accounting for devices this server is still serving on
    # and publish them to other runs the moment its successor finished.
    assert op.reusable_releases == 1, "the READY hand-back must be flagged reusable"


def test_ready_does_not_release_when_gpus_are_held_to_completion(monkeypatch):
    _patch_subprocess_run(monkeypatch)
    op = _WaitsThenSucceeds(refusals=0)
    task = Task(name="server", operator=op, logger=logging.getLogger("t"))
    task.resource_release_after["gpus"] = "task_completion"
    task.probes = [_AlwaysReadyProbe()]

    _drive_to_ready(_orchestrator(_NoopLauncher()), task)

    assert task.status == TaskStatus.READY
    assert op.released == 0


def test_ready_release_is_awaited_not_fired_and_forgotten(monkeypatch):
    """The READY hand-back is offloaded to a thread (it takes a cross-process file
    lock, which must not sit on the event loop) but it is still awaited.

    Ordering is the point: the dependent task the planner packed onto these very
    GPUs acquires as soon as READY propagates. If the release were scheduled
    instead of awaited, that acquire would race a claim that has not been given
    back yet and fail on a full board.
    """
    _patch_subprocess_run(monkeypatch)

    class _SlowRelease(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_slow_release"))
            self.released = 0

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs) -> None:
            return None

        def release_resources(self, *, task_name, reusable: bool = False) -> None:
            # Blocking work, as the real registry release does.
            time.sleep(0.05)
            self.released += 1

    op = _SlowRelease()
    task = Task(name="server", operator=op, logger=logging.getLogger("t"))
    task.resource_release_after["gpus"] = "task_ready"
    task.probes = [_AlwaysReadyProbe()]

    orch = _orchestrator(_NoopLauncher())

    async def _check():
        await orch._release_ready_reusable_resources(task)
        # Complete the moment the coroutine returns -- no lingering task.
        assert op.released == 1, "release must finish before READY propagates"

    asyncio.run(_check())


def test_ready_release_does_not_block_the_event_loop(monkeypatch):
    """While the release runs, other coroutines must keep making progress."""
    _patch_subprocess_run(monkeypatch)

    class _BlockingRelease(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_blocking_release"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs) -> None:
            return None

        def release_resources(self, *, task_name, reusable: bool = False) -> None:
            time.sleep(0.3)

    task = Task(name="server", operator=_BlockingRelease(), logger=logging.getLogger("t"))
    task.resource_release_after["gpus"] = "task_ready"
    orch = _orchestrator(_NoopLauncher())

    ticks = 0

    async def _ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async def _run():
        spinner = asyncio.create_task(_ticker())
        await orch._release_ready_reusable_resources(task)
        spinner.cancel()

    asyncio.run(_run())
    # A release held on the loop would starve the ticker entirely.
    assert ticks > 5, f"event loop was blocked during release (ticks={ticks})"


def test_launch_finally_release_does_not_block_the_event_loop(monkeypatch):
    """The teardown release must be offloaded, exactly like the READY hand-back.

    It takes the same cross-process registry lock (bounded, but up to
    gpu_reservation._LOCK_TIMEOUT_S when a holder is wedged) and runs on the loop
    that drives every other task's probes. Held inline, one task finishing would
    stall the entire run.
    """
    _patch_subprocess_run(monkeypatch)

    class _BlockingRelease(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_blocking_finally_release"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs) -> None:
            return None

        def release_resources(self, *, task_name, reusable: bool = False) -> None:
            time.sleep(0.3)

    task = Task(name="server", operator=_BlockingRelease(), logger=logging.getLogger("t"))
    orch = _orchestrator(_NoopLauncher())

    ticks = 0

    async def _ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async def _run():
        spinner = asyncio.create_task(_ticker())
        await orch._launch_task_with_timeout(task)
        spinner.cancel()

    asyncio.run(_run())
    assert ticks > 5, f"event loop was blocked during release (ticks={ticks})"


def test_launch_finally_release_still_runs_when_the_task_is_cancelled(monkeypatch):
    """Offloading must not cost us the release on the cancellation path.

    The await now sits in a `finally`, so a cancellation can interrupt the *wait*.
    It must not interrupt the release: once handed to the executor the call runs to
    completion, or a cancelled task would leak its GPU claim for the rest of the
    run -- the exact failure the finally exists to prevent.
    """
    _patch_subprocess_run(monkeypatch)
    released: list[str] = []

    class _RecordingRelease(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_cancel_release"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs) -> None:
            return None

        def release_resources(self, *, task_name, reusable: bool = False) -> None:
            released.append(task_name)

    task = Task(name="server", operator=_RecordingRelease(), logger=logging.getLogger("t"))
    orch = _orchestrator(_HangingLauncher())

    async def _run() -> None:
        launch = asyncio.create_task(orch._launch_task_with_timeout(task))
        await asyncio.sleep(0.05)
        launch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch

    asyncio.run(_run())

    assert released == ["server"], "the cancelled task never released its resources"


def test_launch_finally_release_failure_never_replaces_the_task_outcome(monkeypatch):
    """A release that raises must be logged, not propagated.

    The release now runs in a worker thread, so the swallow has to live inside the
    offloaded call; if it leaked, a teardown error would surface as the task's
    result and hide the real one.
    """
    _patch_subprocess_run(monkeypatch)

    class _ExplodingRelease(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_exploding_release"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs) -> None:
            return None

        def release_resources(self, *, task_name, reusable: bool = False) -> None:
            raise RuntimeError("registry is on fire")

    task = Task(name="server", operator=_ExplodingRelease(), logger=logging.getLogger("t"))

    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))

    assert rc == 0, "a cleanup failure replaced the task's real outcome"


def test_reusable_resources_are_released_before_ready_becomes_observable(monkeypatch):
    """READY is what unblocks the dependent packed onto these GPUs, so the claim
    must already be gone when the transition becomes visible.

    Asserted by observing the task's status from inside release_resources: if the
    release ran after the flip, the dependent could see READY -- and try to
    acquire -- while the GPUs were still held.
    """
    _patch_subprocess_run(monkeypatch)
    seen: list = []

    class _RecordsStatusAtRelease(Operator):
        def __init__(self, task_ref):
            super().__init__(OperatorConfig(type="fake_order"))
            self._task_ref = task_ref

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs) -> None:
            return None

        def release_resources(self, *, task_name, reusable: bool = False) -> None:
            seen.append(self._task_ref[0].status)

    holder: list = []
    op = _RecordsStatusAtRelease(holder)
    task = Task(name="server", operator=op, logger=logging.getLogger("t"))
    holder.append(task)
    task.resource_release_after["gpus"] = "task_ready"
    task.probes = [_AlwaysReadyProbe()]

    _drive_to_ready(_orchestrator(_NoopLauncher()), task)

    assert task.status == TaskStatus.READY
    assert seen, "the reusable release never ran"
    assert TaskStatus.READY not in seen, (
        f"released after READY was already observable (saw {seen}); a dependent "
        f"could acquire while the GPUs were still claimed"
    )


class _RecordsReleaseMode(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="fake_release_mode"))
        self.releases: list[bool] = []

    def build_command(self, *, task_name, script, envs) -> Command:
        return Command(exec="true")

    def acquire_resources(self, *, task_name, envs) -> None:
        return None

    def release_resources(
        self, *, task_name, reusable: bool = False, handover: bool = False
    ) -> None:
        self.releases.append(handover)


def test_completion_hands_over_when_a_later_task_needs_these_gpus(monkeypatch):
    """The planner flagged this task: a successor is scheduled onto its devices but
    has not been submitted yet. Publishing them now lets a concurrent sflow run
    take one, and the successor then fails with "0 free" on a valid placement."""
    _patch_subprocess_run(monkeypatch)
    op = _RecordsReleaseMode()
    task = Task(name="taskx", operator=op, logger=logging.getLogger("t"))
    task.gpus_reused_downstream = True

    assert asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task)) == 0
    assert op.releases == [True], "must hand over, not publish"


def test_completion_frees_the_device_when_nothing_downstream_wants_it(monkeypatch):
    """The last user of a device: its completion genuinely frees the GPU, so it
    must go back to the whole host rather than stay held until the run ends."""
    _patch_subprocess_run(monkeypatch)
    op = _RecordsReleaseMode()
    task = Task(name="tasky", operator=op, logger=logging.getLogger("t"))
    task.gpus_reused_downstream = False

    assert asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task)) == 0
    assert op.releases == [False], "must fully release, not keep holding"


def test_a_flagged_task_still_hands_over_when_it_fails(monkeypatch):
    """Holding is the safe direction on failure: never give away a device the run
    might still need. The run-end sweep clears anything left over."""
    _patch_subprocess_run(monkeypatch)
    op = _RecordsReleaseMode()
    task = Task(name="taskx", operator=op, logger=logging.getLogger("t"))
    task.gpus_reused_downstream = True

    asyncio.run(_orchestrator(_FailingLauncher())._launch_task_with_timeout(task))
    assert op.releases == [True]


def test_an_operator_with_the_older_release_signature_still_works(monkeypatch):
    """`handover` is omitted when it would only restate the default.

    An operator overriding release_resources as `(*, task_name)` predates the
    hand-over. Passing `handover=False` to it would TypeError on EVERY task, and
    because the release is best-effort that would be swallowed to a warning while
    quietly leaking the task's claim for the rest of the run.
    """
    _patch_subprocess_run(monkeypatch)
    released: list[str] = []

    class _LegacySignatureOperator(Operator):
        def __init__(self):
            super().__init__(OperatorConfig(type="fake_legacy_release"))

        def build_command(self, *, task_name, script, envs) -> Command:
            return Command(exec="true")

        def acquire_resources(self, *, task_name, envs) -> None:
            return None

        def release_resources(self, *, task_name) -> None:  # no `reusable`
            released.append(task_name)

    task = Task(name="legacy", operator=_LegacySignatureOperator(),
                logger=logging.getLogger("t"))
    task.gpus_reused_downstream = False

    rc = asyncio.run(_orchestrator(_NoopLauncher())._launch_task_with_timeout(task))

    assert rc == 0
    assert released == ["legacy"], "the ordinary release must still reach the operator"
