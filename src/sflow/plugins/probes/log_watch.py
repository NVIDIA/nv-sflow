# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from pathlib import Path

from sflow.core.probe import Probe, ProbeType


class LogWatchProbe(Probe):
    """
    Watches a task log file for a regex match.

    By default, watches the current task's own log file:
      <SFLOW_WORKFLOW_OUTPUT_DIR>/<task_name>/<task_name>.log

    If logger_task_name is set, watches that task's log file instead.

    match_count: number of times the pattern must be matched (default 1).
    """

    _REGEX_PREFIXES = ("re:", "regex:")

    def __init__(
        self,
        *,
        regex_pattern: str,
        logger_task_name: str | None = None,
        match_count: int = 1,
        type: ProbeType,
        **kwargs,
    ):
        super().__init__(type=type, **kwargs)
        # Default behavior: treat the config value as a literal string to search for.
        # This avoids surprising behavior when users include characters like "()", "[]", ".", "*", etc.
        # If you need true regex semantics, prefix the pattern with "re:" (or "regex:").
        p = str(regex_pattern)
        self._pattern_display = p
        if p.startswith(self._REGEX_PREFIXES):
            p = p.split(":", 1)[1]
            self._regex = re.compile(p)
        else:
            self._regex = re.compile(re.escape(p))
        self._logger_task_name = logger_task_name
        self._match_count = max(int(match_count), 1)
        # Incremental scan state: byte offset of the log already consumed (always
        # at a newline boundary) and the running count of matches found so far.
        # This avoids re-reading and re-scanning the whole file on every check.
        self._offset = 0
        self._match_total = 0

    def reset(self) -> None:
        # The orchestrator calls reset() when a task is (re)submitted/retried. A
        # retry may recreate or truncate the log, so the incremental scan (offset
        # and accumulated match count) must restart from scratch.
        super().reset()
        self._offset = 0
        self._match_total = 0

    def _log_path(self, task) -> Path:  # type: ignore[override]
        wf_out = task.envs.get("SFLOW_WORKFLOW_OUTPUT_DIR")
        if not wf_out:
            # Fall back to current task output dir (can't locate other task logs without workflow dir).
            task_out = task.envs.get("SFLOW_TASK_OUTPUT_DIR", "")
            name = self._logger_task_name or task.name
            if task_out and (
                self._logger_task_name is None or self._logger_task_name == task.name
            ):
                return Path(task_out) / f"{name}.log"
            return Path(f"{name}.log")
        name = self._logger_task_name or task.name
        return Path(wf_out) / name / f"{name}.log"

    async def check(self, task) -> bool:  # type: ignore[override]
        path = self._log_path(task)
        try:
            with path.open("rb") as f:
                # Detect truncation/rotation: if the file shrank below where we
                # last stopped, re-scan it from the beginning.
                f.seek(0, os.SEEK_END)
                if f.tell() < self._offset:
                    self._offset = 0
                    self._match_total = 0
                # Read only the bytes appended since the previous check.
                f.seek(self._offset)
                chunk = f.read()
        except FileNotFoundError:
            return False
        except Exception:
            return False

        # Only consume up to the last newline so a match is never split across
        # reads, and a half-written trailing line isn't counted early. The per-task
        # <task>.log is newline-delimited whether sflow's launcher writes it (stream
        # mode) or srun --output plus the aligned prefixer writes it (offload mode):
        # both emit one record per line ending in "\n". In offload mode the rank
        # label is folded into the message (srun --label is disabled), so the file
        # matches stream mode and patterns behave identically.
        newline = chunk.rfind(b"\n")
        if newline != -1:
            consumed = chunk[: newline + 1]
            self._match_total += len(
                self._regex.findall(consumed.decode("utf-8", errors="ignore"))
            )
            self._offset += len(consumed)

        # Matches accumulate across checks; require at least match_count in total.
        return self._match_total >= self._match_count
