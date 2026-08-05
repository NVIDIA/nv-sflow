# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from pathlib import Path

from sflow.core.probe import Probe, ProbeType


# Longest line `_clip` flattens in full. Below this the whitespace-collapse is exact (and
# costs at most a few KB); above it, the line is sliced first. See :func:`_clip`.
_CLIP_EXACT_MAX = 4096
# First character a reader would actually see. Used to skip a sliced line's indentation
# without copying the line: `.lstrip()` would allocate the whole remainder, which is the
# very cost the slice path exists to avoid. Scans only the leading whitespace.
_FIRST_VISIBLE_RE = re.compile(r"\S")


def _clip(text: str, limit: int = 160) -> str:
    """One-line, length-capped view of a log line for the summary trace.

    Long lines are SLICED before the whitespace flattening. Both ``.replace()`` calls
    allocate a full copy of their input, and this runs on the event loop once per probe
    tick against the last log line seen -- which for a task that emits a long unbroken
    line (a JSON or base64 dump; ``str.splitlines`` already breaks ``\\r`` progress bars
    into frames) is the whole multi-megabyte line, copied three times over to produce 160
    characters. Measured on a 48 MB line: 6.7 ms per tick, down to 6 microseconds.

    Anything up to ``_CLIP_EXACT_MAX`` still takes the exact path, so the result is
    byte-identical to flattening the whole line -- including for a line whose length is
    over ``limit`` only because of trailing whitespace, which must NOT gain a "..." it
    did not earn. Past that size a slice genuinely drops content, so the marker is right.

    The slice starts at the line's first NON-whitespace character rather than at index
    0, because the ``.strip()`` that used to remove indentation now runs after the
    slice: taking ``text[:limit]`` outright would spend the whole window on a
    deeply-indented line and render it as a bare "...", reporting less than the exact
    path it stands in for. Locating that character is a regex scan over the
    indentation only -- ``.lstrip()`` would copy the entire remainder of the line,
    which is the cost this branch exists to avoid.
    """
    if len(text) <= _CLIP_EXACT_MAX:
        flat = text.replace("\r", " ").replace("\t", " ").strip()
        return flat if len(flat) <= limit else flat[:limit] + "..."
    first = _FIRST_VISIBLE_RE.search(text)
    if first is None:
        return "..."  # nothing visible in the whole line
    start = first.start()
    # Already past the indentation, so rstrip alone finishes the job.
    window = (
        text[start : start + limit].replace("\r", " ").replace("\t", " ").rstrip()
    )
    return window + "..."


class LogWatchProbe(Probe):
    """
    Watches a task log file for a regex match.

    By default, watches the current task's own log file:
      <SFLOW_WORKFLOW_OUTPUT_DIR>/<task_name>/<task_name>.log

    If logger_task_name is set, watches that task's log file instead.

    match_count: number of times the pattern must be matched (default 1).
    """

    kind = "log_watch"

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
        # Incremental scan state: byte offset of the log already consumed (always at
        # a newline boundary) and the running count of matches found so far. Avoids
        # re-reading + re-scanning the whole file on every check. Each probe keeps its
        # OWN offset, so several probes watching one task's log (e.g. a readiness
        # marker + a failure ``Traceback``) each scan EVERY line independently -- they
        # never consume each other's lines.
        self._offset = 0
        self._match_total = 0
        # Last-attempt trace state (surfaced in sflow_summary.log): the most recent
        # non-empty line seen, and the most recent line that matched the pattern.
        self._last_line = ""
        self._last_match_line = ""

    def reset(self) -> None:
        # The orchestrator calls reset() when a task is (re)submitted/retried. A
        # retry may recreate or truncate the log, so the incremental scan (offset
        # and accumulated match count) must restart from scratch.
        super().reset()
        self._offset = 0
        self._match_total = 0
        self._last_line = ""
        self._last_match_line = ""

    def _count_matches(self, text: str) -> int:
        return len(self._regex.findall(text))

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
        # Incrementally scan the task's own ``<task>.log`` from disk, accumulating
        # matches across checks. That file is the single ground truth: under K8s it
        # is written by the offloaded ``kubectl logs -f`` (single pod: a plain
        # redirect; merged pod: the streaming tag demuxer), which follows across
        # kubelet log rotations on modern kubelet (the pre-v1.29 follow-stall bug --
        # kubernetes/kubernetes#115701 -- is fixed) and is rebuilt COMPLETE from a
        # one-shot ``kubectl logs`` re-fetch at pod-terminal (``finalize_*``); local /
        # slurm write it via the launcher. It is newline-delimited (one record per
        # line) on every backend, so matching behaves identically.
        path = self._log_path(task)
        try:
            with path.open("rb") as f:
                # Detect truncation/rotation of <task>.log itself: if the file shrank
                # below where we last stopped, re-scan it from the beginning.
                f.seek(0, os.SEEK_END)
                if f.tell() < self._offset:
                    self._offset = 0
                    self._match_total = 0
                # Read only the bytes appended since the previous check.
                f.seek(self._offset)
                chunk = f.read()
        except FileNotFoundError:
            self._attempt_detail = f"log not found yet: {path}"
            return False
        except Exception as exc:
            self._attempt_detail = f"log read error: {exc}"
            return False

        # Only consume up to the last newline so a match is never split across reads,
        # and a half-written trailing line isn't counted early.
        newline = chunk.rfind(b"\n")
        if newline != -1:
            consumed = chunk[: newline + 1]
            text = consumed.decode("utf-8", errors="ignore")
            self._match_total += self._count_matches(text)
            self._offset += len(consumed)  # advance by BYTES, not decoded chars
            # Track the last line seen and the last matching line for the trace.
            for line in text.splitlines():
                if not line:
                    continue
                self._last_line = line
                if self._regex.search(line):
                    self._last_match_line = line

        # Matches accumulate across checks; require at least match_count in total.
        matched = self._match_total >= self._match_count
        self._attempt_detail = self._trace_detail(matched)
        return matched

    def _trace_detail(self, matched: bool) -> str:
        """Short last-attempt trace for sflow_summary.log: what the probe saw."""
        counts = f"{self._match_total}/{self._match_count}"
        if matched:
            detail = f"matched {self._pattern_display!r} ({counts})"
            if self._last_match_line:
                detail += f" | line: {_clip(self._last_match_line)!r}"
            return detail
        if self._last_line:
            return f"no match ({counts}) | last line: {_clip(self._last_line)!r}"
        return f"no match ({counts}) | (no lines yet)"
