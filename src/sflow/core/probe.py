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


class Probe(ABC):
    """
    Abstract base class for probe checks.
    """

    def __init__(
        self,
        *,
        type: ProbeType,
        delay: int = 0,
        timeout: int = 60,
        interval: int = 5,
        success_threshold: int = 1,
        failure_threshold: int = 3,
    ):
        # Mirror common K8s-style probe knobs.
        # - delay: seconds before first check
        # - timeout: per-check timeout (seconds)
        # - interval: seconds between checks
        # - success_threshold: consecutive successes to trigger readiness
        # - failure_threshold: consecutive failures (for failure probes)
        self.delay = int(delay)
        self.timeout = int(timeout)
        self.interval = int(interval)
        self.success_threshold = int(success_threshold)
        self.failure_threshold = int(failure_threshold)
        self.type = type
        self.status = ProbeStatus.INITIATED

        # Internal state for scheduling / thresholds.
        self._started_at = time.time()
        self._next_check_at = self._started_at + max(self.delay, 0)
        self._success_streak = 0
        self._failure_streak = 0

    def reset(self) -> None:
        self.status = ProbeStatus.INITIATED
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
        """
        if self.status != ProbeStatus.INITIATED:
            return False

        now = time.time()
        if now < self._next_check_at:
            return False

        self._next_check_at = now + max(self.interval, 0)

        try:
            ok = await asyncio.wait_for(self.check(task), timeout=max(self.timeout, 0))
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
