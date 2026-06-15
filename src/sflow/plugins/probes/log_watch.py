# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

from sflow.core.probe import Probe, ProbeStatus, ProbeType


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
            self._literal_pattern: str | None = None
        else:
            self._regex = None
            self._literal_pattern = p
        self._logger_task_name = logger_task_name
        self._match_count = max(int(match_count), 1)
        self._offset = 0
        self._matched_count = 0
        self._received_live_lines = False
        # When True, the next disk read rebaselines ``_offset`` to the current end of
        # the log, so a retry does not recount the previous attempt's output (the
        # per-task log is opened in append mode). Set via ``rebaseline_on_retry``,
        # which the orchestrator calls on re-submission. See ``check``.
        self._rebaseline_offset = False

    def feed_line(self, line: str) -> bool:
        """Feed one live subprocess output line before any persistence policy runs."""
        if not self.is_live_match_active():
            return self._matched_count >= self._match_count
        self._received_live_lines = True
        self._matched_count += self._count_matches(line)
        return self._matched_count >= self._match_count

    def is_live_match_active(self) -> bool:
        return (
            self.status == ProbeStatus.INITIATED
            and self._matched_count < self._match_count
        )

    def _count_matches(self, text: str) -> int:
        if self._literal_pattern is not None:
            return text.count(self._literal_pattern)
        assert self._regex is not None
        return len(self._regex.findall(text))

    def reset(self) -> None:
        super().reset()
        self._offset = 0
        self._matched_count = 0
        self._received_live_lines = False
        self._rebaseline_offset = False

    def rebaseline_on_retry(self) -> None:
        """Skip log content from previous attempts on the next disk read.

        The orchestrator calls this on re-submission (retry). Because the per-task
        log is append-mode, attempt N-1's output is still on disk; without this the
        file fallback in ``check`` could recount stale matches before attempt N
        produces output. (When attempt N produces live output, ``check`` short-
        circuits on ``_received_live_lines`` and never reads the file anyway.)
        """
        self._rebaseline_offset = True

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
        if self._matched_count >= self._match_count:
            return True
        # Once any live line has been fed, rely solely on the live stream: it is a
        # strict superset of what reaches <task>.log (the launcher is the only writer
        # and feeds every persisted line to us via feed_line -- see
        # SubprocessLauncher._emit_subprocess_line). Re-reading the file here would
        # only risk double-counting lines already counted live.
        if self._received_live_lines:
            return False

        path = self._log_path(task)
        try:
            size = path.stat().st_size
            if self._rebaseline_offset:
                # New attempt (after reset): skip everything written by previous
                # attempts so we only count matches produced by this attempt.
                self._offset = size
                self._rebaseline_offset = False
            elif size < self._offset:
                self._offset = 0  # file truncated / rotated
            with path.open("r", errors="ignore") as f:
                f.seek(self._offset)
                data = f.read()
                self._offset = f.tell()
        except FileNotFoundError:
            return False
        except Exception:
            return False

        self._matched_count += self._count_matches(data)
        return self._matched_count >= self._match_count
