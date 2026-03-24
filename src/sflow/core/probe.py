# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task import Task


class ProbeType(str, Enum):
    READINESS = "readiness"
    FAILURE = "failure"

    def __str__(self) -> str:
        return self.value


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

    def reset(self) -> None:
        self.status = ProbeStatus.INITIATED
        self.timed_out = False
        self._started_at = time.time()
        self._next_check_at = self._started_at + max(self.delay, 0)
        self._success_streak = 0
        self._failure_streak = 0

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

        try:
            ok = await asyncio.wait_for(
                self.check(task), timeout=max(self.each_check_timeout, 1)
            )
        except asyncio.TimeoutError:
            ok = False

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
