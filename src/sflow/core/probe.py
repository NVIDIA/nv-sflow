# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task import Task


class ProbeType(str, Enum):
    READINESS = "readiness"
    FAILURE = "failure"

    def __str__(self) -> str:
        return self.value


@dataclass
class ProbeAttempt:
    """The outcome of the most recent probe check attempt (for the summary log).

    Only the LAST attempt is kept per probe -- enough to see, post-mortem in
    ``sflow_summary.log``, what the probe observed when the task went READY / timed
    out / failed, without flooding the log with every attempt. ``detail`` is a short
    per-type trace (e.g. log_watch: the matched line, or the last line seen on a miss;
    tcp/http: the endpoint + result). ``runtime`` is how long that check took.
    """

    ok: bool
    detail: str
    runtime: float


class ProbeStatus(str, Enum):
    INITIATED = "initiated"
    TRIGGERED = "triggered"

    def __str__(self) -> str:
        return self.value


class ProbeTimeoutError(Exception):
    """Raised when a readiness probe exceeds its overall timeout deadline."""


class Probe(ABC):
    """
    Abstract base class for probe checks.
    """

    # Short probe-type label for the summary's Probe Traces section (overridden
    # by each concrete probe: "log_watch", "tcp_port", "http_get", "http_post").
    kind: str = "probe"

    def __init__(
        self,
        *,
        type: ProbeType,
        delay: int = 0,
        timeout: int = 1200,
        each_check_timeout: int = 30,
        interval: int = 5,
        success_threshold: int = 1,
        failure_threshold: int = 3,
    ):
        # - delay: seconds before first check
        # - timeout: overall deadline (seconds) — for readiness probes, the task
        #   is marked FAILED if not ready within this window
        # - each_check_timeout: per-attempt timeout (seconds) for each individual check
        # - interval: seconds between checks
        # - success_threshold: consecutive successes to trigger readiness
        # - failure_threshold: consecutive failures (for failure probes)
        self.delay = int(delay)
        self.timeout = int(timeout)
        self.each_check_timeout = int(each_check_timeout)
        self.interval = int(interval)
        self.success_threshold = int(success_threshold)
        self.failure_threshold = int(failure_threshold)
        self.type = type
        self.status = ProbeStatus.INITIATED
        self.timed_out = False

        # Internal state for scheduling / thresholds.
        self._started_at = time.time()
        self._next_check_at = self._started_at + max(self.delay, 0)
        self._success_streak = 0
        self._failure_streak = 0

        # Trace of the LAST check attempt only (surfaced in sflow_summary.log). A
        # concrete check() sets ``_attempt_detail`` to a short per-type description;
        # probe() wraps it with ok + runtime into ``last_attempt``. None until the
        # first real check runs.
        self.last_attempt: ProbeAttempt | None = None
        self._attempt_detail: str = ""

    def reset(self) -> None:
        self.status = ProbeStatus.INITIATED
        self.timed_out = False
        self._started_at = time.time()
        self._next_check_at = self._started_at + max(self.delay, 0)
        self._success_streak = 0
        self._failure_streak = 0
        self.last_attempt = None
        self._attempt_detail = ""

    def force_due(self) -> None:
        """Make the next :meth:`probe` run a check immediately, bypassing the
        interval gate (but NOT the overall timeout deadline). Used for a final
        readiness scan when a service's process exits, to close the race between
        process-exit and the interval-gated probe re-scanning the now-complete log.
        """
        self._next_check_at = 0.0

    @property
    def effective_check_timeout(self) -> int:
        """Per-attempt timeout (seconds) for a single check.

        Honors the configured ``each_check_timeout`` directly. ``interval`` is the
        gap *between* checks, not a bound on how long one attempt may take, so it
        must not shrink the per-attempt timeout: with the default ``interval=5`` an
        explicit ``each_check_timeout: 30`` would otherwise be silently capped to
        5s, and a slow-but-valid endpoint (e.g. an LLM whose first token lands past
        the interval) would time out on every attempt and never go ready.
        """
        return self.each_check_timeout

    @abstractmethod
    async def check(self, task: Task) -> bool:
        """
        Performs one probe check attempt.

        Return value means "probe condition is met":
        - readiness probe: True means ready
        - failure probe: True means failed condition detected
        """
        raise NotImplementedError

    async def probe(self, task: Task) -> bool:
        """
        Non-blocking probe tick.

        Called repeatedly by the orchestrator; it enforces delay/interval and uses
        thresholds to determine when to trigger.

        Raises ProbeTimeoutError for readiness probes that exceed their overall
        timeout deadline.
        """
        if self.status != ProbeStatus.INITIATED:
            return False

        now = time.time()
        elapsed = now - self._started_at

        if self.type == ProbeType.READINESS and self.timeout > 0 and elapsed > self.timeout:
            self.timed_out = True
            raise ProbeTimeoutError(
                f"Readiness probe timed out after {int(elapsed)}s "
                f"(deadline: {self.timeout}s)"
            )

        if now < self._next_check_at:
            return False

        self._next_check_at = now + max(self.interval, 0)

        # The concrete check() sets ``_attempt_detail`` as a side effect; reset it so
        # a stale detail can't linger if a check returns early without setting one.
        self._attempt_detail = ""
        check_timeout = max(self.effective_check_timeout, 1)
        started = time.monotonic()
        try:
            ok = await asyncio.wait_for(self.check(task), timeout=check_timeout)
        except asyncio.TimeoutError:
            ok = False
            self._attempt_detail = f"check timed out after {check_timeout}s"
        # Keep ONLY this (latest) attempt's trace for post-mortem debugging in
        # sflow_summary.log -- what the probe last observed, without flooding.
        self.last_attempt = ProbeAttempt(
            ok=bool(ok),
            detail=self._attempt_detail,
            runtime=time.monotonic() - started,
        )

        if self.type == ProbeType.READINESS:
            if ok:
                self._success_streak += 1
            else:
                self._success_streak = 0
            return self._success_streak >= max(self.success_threshold, 1)

        if self.type == ProbeType.FAILURE:
            if ok:
                self._failure_streak += 1
            else:
                self._failure_streak = 0
            return self._failure_streak >= max(self.failure_threshold, 1)

        return False
