# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import logging.handlers
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

TaskLogMode = Literal["full", "bounded"]

_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 16


@dataclass(frozen=True)
class TaskLogPolicy:
    mode: TaskLogMode = "bounded"
    keep_lines_per_second: int = 100
    keep_first_lines: int = 1000
    max_bytes: int = _DEFAULT_MAX_BYTES
    backup_count: int = _DEFAULT_BACKUP_COUNT

    def __post_init__(self) -> None:
        if self.mode not in {"full", "bounded"}:
            raise ValueError(f"Invalid task log mode: {self.mode!r}")
        if self.keep_lines_per_second < 0:
            raise ValueError("keep_lines_per_second must be >= 0")
        if self.keep_first_lines < 0:
            raise ValueError("keep_first_lines must be >= 0")
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        if self.backup_count < 0:
            raise ValueError("backup_count must be >= 0")


class BoundedRotatingTaskLogHandler(logging.handlers.RotatingFileHandler):
    """Rotating file handler that rate-limits persisted task output lines."""

    def __init__(
        self,
        filename: str | Path,
        policy: TaskLogPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            filename,
            maxBytes=int(policy.max_bytes),
            backupCount=int(policy.backup_count),
        )
        self._policy = policy
        self._clock = clock
        self._tokens = float(policy.keep_first_lines)
        self._last_refill = self._clock()
        self._suppressed = 0
        self._closed_with_summary = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._allow_record():
                self._emit_suppression_summary(record)
                super().emit(record)
            else:
                self._suppressed += 1
        except Exception:
            self.handleError(record)

    def emit_line(
        self,
        *,
        logger_name: str,
        message: str,
        level: int = logging.INFO,
    ) -> bool:
        """Persist one task output line if the bounded policy allows it."""
        if not self._allow_record():
            self._suppressed += 1
            return False
        record = logging.LogRecord(
            name=logger_name,
            level=level,
            pathname=__file__,
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        self._emit_suppression_summary(record)
        super().emit(record)
        return True

    def close(self) -> None:
        if not self._closed_with_summary:
            try:
                record = logging.LogRecord(
                    name="sflow.task",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=0,
                    msg="",
                    args=(),
                    exc_info=None,
                )
                self._emit_suppression_summary(record)
                self._closed_with_summary = True
            except Exception:
                pass
        super().close()

    def _allow_record(self) -> bool:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _refill(self) -> None:
        rate = float(self._policy.keep_lines_per_second)
        now = self._clock()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        if rate <= 0:
            return
        burst = float(self._policy.keep_first_lines)
        self._tokens = min(burst, self._tokens + elapsed * rate)

    def _emit_suppression_summary(self, record: logging.LogRecord) -> None:
        if self._suppressed <= 0:
            return
        count = self._suppressed
        self._suppressed = 0
        summary = logging.LogRecord(
            name=record.name,
            level=logging.INFO,
            pathname=record.pathname,
            lineno=record.lineno,
            msg=f"sflow: suppressed {count} task log lines due to rate limit",
            args=(),
            exc_info=None,
        )
        super().emit(summary)


def create_task_log_handler(
    log_path: str | Path,
    policy: TaskLogPolicy | None = None,
) -> logging.Handler:
    policy = policy or TaskLogPolicy()
    if policy.mode == "bounded":
        handler: logging.Handler = BoundedRotatingTaskLogHandler(log_path, policy)
    else:
        handler = logging.FileHandler(log_path)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    return handler


class TaskOutputSink:
    """Task output persistence path that can suppress before Logger.info()."""

    def __init__(self, *, logger: logging.Logger, policy: TaskLogPolicy) -> None:
        self._logger = logger
        self._policy = policy
        self._bounded_handler: BoundedRotatingTaskLogHandler | None = None
        if policy.mode == "bounded":
            for handler in getattr(logger, "handlers", []):
                if isinstance(handler, BoundedRotatingTaskLogHandler):
                    self._bounded_handler = handler
                    break

    def emit_line(self, line: str) -> None:
        if self._bounded_handler is not None:
            self._bounded_handler.emit_line(logger_name=self._logger.name, message=line)
            return
        self._logger.info(line)

    def close(self) -> None:
        if self._bounded_handler is not None:
            self._bounded_handler.close()
