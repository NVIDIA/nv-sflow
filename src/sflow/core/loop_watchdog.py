# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect (and capture) an asyncio event loop that has stopped being scheduled.

sflow drives the whole DAG from one event loop. Anything that occupies that thread
without awaiting -- a slow formatter, a synchronous read, a tight loop -- stops the
orchestrator from ticking, so healthy finished tasks go unnoticed and the workflow
appears to hang. The failure is self-concealing: sflow's own log records are emitted
from that same thread, so a stalled run produces NO diagnostic output at all.

That is not hypothetical. A real MLPerf run sat at 100% CPU with its harness pod
already ``Completed`` for 20 minutes, the DAG frozen and not one line logged. Diagnosing
it required attaching py-spy to the live process before it recovered -- a single sample
that, on its own, was too weak to identify the cause. This module makes the next
occurrence self-documenting instead: a stall dumps every thread's Python stack into the
run directory, so the evidence survives the episode.

Deliberately cheap: one coroutine touching a float once a second, and one daemon thread
comparing clocks. The watcher is a plain thread ON PURPOSE -- it must keep running while
the loop is exactly what is blocked.
"""

from __future__ import annotations

import asyncio
import faulthandler
import threading
import time
from pathlib import Path

from sflow.logging import get_logger

_logger = get_logger(__name__)

# How often the loop proves it is still being scheduled.
BEAT_INTERVAL = 1.0
# A loop that has not been scheduled for this long is stalled, not merely busy. Well
# above any legitimate synchronous burst (a large log sanitize, a manifest render) so a
# healthy run never trips it, and far below the ~20 minute freezes seen in the wild.
STALL_SECONDS = 30.0
# Bound the evidence file: one stack dump per episode plus a handful of follow-ups is
# plenty to identify a hot path, and an unattended run must not fill the disk.
MAX_DUMPS = 20


class EventLoopWatchdog:
    """Watches one event loop and dumps all thread stacks when it stops ticking."""

    def __init__(
        self,
        dump_path: Path | str,
        *,
        stall_seconds: float = STALL_SECONDS,
        beat_interval: float = BEAT_INTERVAL,
    ) -> None:
        self._dump_path = Path(dump_path)
        self._stall_seconds = stall_seconds
        self._beat_interval = beat_interval
        self._last_beat = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._beat_task: asyncio.Task | None = None
        self._dumps = 0
        self._stalled = False
        # Worst lag seen during the CURRENT stall episode. The recovery message has to
        # report this, not the lag measured at the moment of recovery -- that one is by
        # definition under the threshold, so a 20-minute freeze would announce itself as
        # "recovered after 1s", in the very artifact written to explain the freeze.
        self._worst_behind = 0.0

    # -- loop side -------------------------------------------------------------
    async def _beat(self) -> None:
        while not self._stop.is_set():
            self._last_beat = time.monotonic()
            await asyncio.sleep(self._beat_interval)

    # -- watcher side (plain thread: runs even when the loop cannot) ------------
    def _watch(self) -> None:
        while not self._stop.wait(self._beat_interval):
            # The loop this watches is gone (asyncio.run returned and cancelled our
            # beat, or the caller never reached finish()). A dead loop is not a stalled
            # one -- keep watching it and every later second looks like a freeze.
            beat = self._beat_task
            if beat is None or beat.done():
                return
            behind = time.monotonic() - self._last_beat
            if behind < self._stall_seconds:
                if self._stalled:
                    self._stalled = False
                    _logger.warning(
                        f"event loop recovered; it went {self._worst_behind:.0f}s "
                        f"without being scheduled at the worst point. Stacks were "
                        f"captured in {self._dump_path.name}"
                    )
                    self._worst_behind = 0.0
                continue
            self._worst_behind = max(self._worst_behind, behind)
            if self._stalled and self._dumps >= MAX_DUMPS:
                continue  # already captured this episode to the cap
            self._stalled = True
            self._dump(behind)

    def _dump(self, behind: float) -> None:
        """Append every thread's Python stack to the dump file.

        ``faulthandler`` writes from C without allocating or taking the GIL-dependent
        locks a logging call would, so it works even against a thread that is wedged --
        which is the whole point.
        """
        self._dumps += 1
        try:
            self._dump_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._dump_path, "a") as fh:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                fh.write(
                    f"\n===== event loop stalled {behind:.1f}s at {stamp} "
                    f"(dump {self._dumps}/{MAX_DUMPS}) =====\n"
                )
                faulthandler.dump_traceback(file=fh, all_threads=True)
                fh.flush()
        except Exception:  # pragma: no cover - diagnostics must never break a run
            return
        # Logged from the WATCHER thread, so it still reaches the console/file handlers
        # while the loop thread is blocked -- the stalled run stops being silent.
        _logger.warning(
            f"event loop has not been scheduled for {behind:.0f}s -- the DAG is not "
            f"being driven and finished tasks cannot be noticed. All thread stacks "
            f"written to {self._dump_path}"
        )

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        """Arm the watchdog. Raises ``RuntimeError`` if there is no running loop.

        The running-loop check is done EXPLICITLY, before the coroutine object exists,
        rather than by letting ``asyncio.ensure_future`` fail. ``ensure_future`` is not a
        reliable guard: with no running loop it falls back to ``get_event_loop()``, which
        on Python 3.10-3.13 creates a brand new (never-run) loop and returns a Task on it
        instead of raising. The watchdog would then arm against a loop that never ticks,
        report a stall after ``stall_seconds``, and write a stack dump -- a false alarm in
        exactly the file that exists to be trusted -- while also leaking a "coroutine was
        never awaited" RuntimeWarning. Callers that may be synchronous (dry-run, tests)
        rely on this raising so they can skip the watchdog entirely.
        """
        if self._thread is not None:
            return
        asyncio.get_running_loop()  # raises RuntimeError when called synchronously
        # Re-arm a previously stopped instance: ``_stop`` stays set after stop(), so
        # without clearing it the watcher thread would exit on its first wait() and the
        # watchdog would be silently dead rather than restarted.
        self._stop.clear()
        self._stalled = False
        self._worst_behind = 0.0
        self._last_beat = time.monotonic()
        self._beat_task = asyncio.ensure_future(self._beat())
        self._thread = threading.Thread(
            target=self._watch, name="sflow-loop-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._beat_task is not None and not self._beat_task.done():
            self._beat_task.cancel()
        self._beat_task = None
        self._thread = None
