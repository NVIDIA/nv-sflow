# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        if p.startswith(self._REGEX_PREFIXES):
            p = p.split(":", 1)[1]
            self._regex = re.compile(p)
        else:
            self._regex = re.compile(re.escape(p))
        self._logger_task_name = logger_task_name
        self._match_count = max(int(match_count), 1)
        self._offset = 0

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
            data = path.read_text(errors="ignore")
        except FileNotFoundError:
            return False
        except Exception:
            return False

        self._offset = len(data)

        # Count total matches in the entire log so far; require at least match_count.
        matches = self._regex.findall(data)
        return len(matches) >= self._match_count
