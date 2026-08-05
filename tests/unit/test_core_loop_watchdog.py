# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stalled event loop must leave evidence behind.

sflow drives the whole DAG from one event loop, and every sflow log record is emitted
from that same thread -- so anything that occupies it without awaiting freezes the DAG
*and* silences the diagnostics that would explain why. A real MLPerf run sat at 100% CPU
with its harness pod already Completed for 20 minutes, nothing logged, and identifying
it required attaching py-spy to the live process before it recovered. These tests cover
the detector that makes the next occurrence self-documenting.
"""

import asyncio
import gc
import logging
import re
import time

import pytest

from sflow.core.loop_watchdog import EventLoopWatchdog


def _blocking_stall(seconds: float) -> None:
    """Occupy the loop thread without awaiting -- exactly what a stall is."""
    time.sleep(seconds)


def test_stall_dumps_every_thread_stack_to_the_run_dir(tmp_path, caplog):
    dump = tmp_path / "loop_stalls.txt"
    wd = EventLoopWatchdog(dump, stall_seconds=0.15, beat_interval=0.02)

    async def drive():
        wd.start()
        await asyncio.sleep(0.05)      # prove the beat is running first
        _blocking_stall(0.45)          # now wedge the loop
        await asyncio.sleep(0.15)      # let the watcher observe recovery
        wd.stop()

    with caplog.at_level(logging.WARNING):
        asyncio.run(drive())

    assert dump.exists(), "a stall must leave a stack dump behind"
    body = dump.read_text()
    assert "event loop stalled" in body
    # faulthandler writes real Python frames -- this is what identifies the hot path
    # that a single py-spy sample could not.
    assert "Traceback" in body or "File \"" in body, body[:400]
    assert "_blocking_stall" in body, "the stalling frame must be captured"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("has not been scheduled" in m for m in msgs), msgs
    assert any("DAG is not being driven" in m for m in msgs), msgs


def test_healthy_loop_is_never_reported_as_stalled(tmp_path, caplog):
    """No false positives: a responsive loop must stay silent."""
    dump = tmp_path / "loop_stalls.txt"
    wd = EventLoopWatchdog(dump, stall_seconds=0.2, beat_interval=0.02)

    async def drive():
        wd.start()
        for _ in range(20):
            await asyncio.sleep(0.02)   # always yielding
        wd.stop()

    with caplog.at_level(logging.WARNING):
        asyncio.run(drive())

    assert not dump.exists(), "a healthy run must not write a stall dump"
    assert not [r for r in caplog.records if "not been scheduled" in r.getMessage()]


def test_watcher_exits_with_its_loop_instead_of_crying_stall(tmp_path, caplog):
    """A finished loop is not a stalled one.

    The watcher is a daemon thread, so if it kept comparing clocks after its loop went
    away every subsequent second would look like a freeze -- which is exactly what it
    did on first implementation: unrelated later tests logged bogus 49s stalls.
    """
    dump = tmp_path / "loop_stalls.txt"
    wd = EventLoopWatchdog(dump, stall_seconds=0.1, beat_interval=0.02)

    async def drive():
        wd.start()
        await asyncio.sleep(0.05)
        # Return WITHOUT calling stop(): asyncio.run cancels the beat on the way out.

    with caplog.at_level(logging.WARNING):
        asyncio.run(drive())
        time.sleep(0.4)  # far longer than stall_seconds; the watcher must be gone

    assert not dump.exists(), "a completed run must not be reported as a stall"
    assert not [r for r in caplog.records if "not been scheduled" in r.getMessage()]


def test_dumps_are_capped_so_a_long_freeze_cannot_fill_the_disk(tmp_path):
    """An unattended run can stall for many minutes; the evidence file stays bounded."""
    import sflow.core.loop_watchdog as mod

    dump = tmp_path / "loop_stalls.txt"
    wd = EventLoopWatchdog(dump, stall_seconds=0.02, beat_interval=0.01)
    wd._stalled = True
    wd._dumps = mod.MAX_DUMPS
    wd._dump_path = dump
    # Already at the cap: the watcher's guard must skip further dumps.
    assert wd._dumps >= mod.MAX_DUMPS
    assert not dump.exists()


def test_start_refuses_to_arm_without_a_running_loop(tmp_path, recwarn):
    """A synchronous caller must get RuntimeError -- not a watchdog on a dead loop.

    ``asyncio.ensure_future`` is NOT a reliable guard here: with no running loop it falls
    back to ``get_event_loop()``, which on Python 3.10-3.13 creates a brand new (never
    run) loop and returns a Task on it instead of raising. The watchdog would then arm
    against a loop that can never tick, declare a stall after ``stall_seconds``, and
    write a stack dump -- a false alarm in the very file that exists to be trusted --
    while also leaking a "coroutine was never awaited" RuntimeWarning. Dry-run and other
    sync callers rely on this raising so they can skip the watchdog entirely.
    """
    dump = tmp_path / "loop_stalls.txt"
    wd = EventLoopWatchdog(dump, stall_seconds=0.1, beat_interval=0.02)

    with pytest.raises(RuntimeError):
        wd.start()

    time.sleep(0.3)  # well past stall_seconds: a wrongly-armed watcher would fire here
    assert not dump.exists(), "a watchdog that never armed must not report a stall"

    gc.collect()
    assert not [
        w for w in recwarn.list if issubclass(w.category, RuntimeWarning)
    ], "the beat coroutine must not be created before the running-loop check"


def test_a_stopped_watchdog_can_be_armed_again(tmp_path):
    """``stop()`` must leave the object reusable, not silently inert.

    ``_stop`` is a latched Event, so without clearing it on start the watcher thread
    exits on its first wait and the watchdog looks armed while detecting nothing.
    """
    dump = tmp_path / "loop_stalls.txt"
    wd = EventLoopWatchdog(dump, stall_seconds=0.15, beat_interval=0.02)

    async def drive():
        wd.start()
        await asyncio.sleep(0.05)
        wd.stop()
        # Second life: arm again and stall for real.
        wd.start()
        await asyncio.sleep(0.05)
        _blocking_stall(0.45)
        await asyncio.sleep(0.15)
        wd.stop()

    asyncio.run(drive())
    assert dump.exists(), "a re-armed watchdog must still detect a stall"


def test_recovery_reports_the_worst_lag_not_the_lag_at_recovery(tmp_path, caplog):
    """The recovery line must describe the FREEZE, not the moment it ended.

    ``behind`` is re-measured every tick, so by definition it is back under the
    threshold when recovery is noticed. Reporting that number turned a multi-minute
    stall into "recovered after 1s of no scheduling" -- actively misleading in the one
    artifact written to explain the stall.
    """
    dump = tmp_path / "loop_stalls.txt"
    wd = EventLoopWatchdog(dump, stall_seconds=0.15, beat_interval=0.02)

    async def drive():
        wd.start()
        await asyncio.sleep(0.05)
        _blocking_stall(0.9)          # a long freeze
        await asyncio.sleep(0.2)      # let the watcher see it recover
        wd.stop()

    with caplog.at_level(logging.WARNING):
        asyncio.run(drive())

    recovery = [r.getMessage() for r in caplog.records if "recovered" in r.getMessage()]
    assert recovery, "a stall episode needs a visible end, not just a start"
    reported = re.search(r"went (\d+)s", recovery[0])
    assert reported, f"the recovery line must quote the worst lag: {recovery[0]!r}"
    assert int(reported.group(1)) >= 1, (
        f"reported {reported.group(1)}s for a ~0.9s freeze -- that is the lag AT "
        f"recovery, not the worst observed: {recovery[0]!r}"
    )
