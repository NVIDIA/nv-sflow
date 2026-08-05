# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health telemetry for the external commands sflow shells out to.

sflow's control plane is external processes -- ``kubectl`` for Kubernetes, ``sbatch``
/ ``squeue`` for Slurm -- so when a run stalls the first question is always: *was it
the tool/cluster, or was it sflow?* Answering that used to require guesswork, because
a wedged ``kubectl`` produced no record at all. That is exactly how a ~20-minute
driver hang (a silently-dead TCP connection to the API server, with no call bounded)
stayed undiagnosable: nothing recorded that a call had been issued and never returned.

This records, per invocation: the tool, a REDACTED operation label, how long it took,
its exit code, and whether it hit sflow's timeout. From that a post-mortem can say
"kubectl answered 1,842 times with a mean of 0.2s and 3 timeouts at 14:32" -- which
separates a sick control plane from an sflow logic bug immediately.

EVERY call feeds the in-memory rollup (counts, mean/max latency per operation); only
NOTABLE calls are persisted to the JSONL -- failures (``rc != 0``) and calls slower
than ``SLOW_COMMAND_WARN_S``. Ordinary fast successes are already summarised by the
rollup, and persisting them would flood the file (a 2s poll loop produces ~43k a day
per task) without adding anything a debugger reads. Slow successes DO earn their place:
they show a control plane degrading before it fails, and recovering afterwards.

Distinct from :mod:`sflow.core.command_log`, which records WHAT command text was
issued (into ``bash_cmds.log`` and friends) for reproduction. This records HOW those
invocations behaved. Keeping them apart keeps each one's purpose single.

Deliberately in ``core`` and tool-agnostic: ``kubectl`` is wired up today, and the
Slurm backend's ``sbatch``/``squeue`` calls can record here unchanged.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sflow.logging import get_logger

_logger = get_logger(__name__)

# A call slower than this is surfaced LIVE (not just in the post-mortem), because a
# control plane that has gone slow is the single most useful early warning we can
# give -- the status polls it serves normally answer in ~0.2s.
SLOW_COMMAND_WARN_S = 5.0
# Detail ring for the slowest/failed calls. Aggregate counters cover EVERY call, so
# this bound never hides totals -- it only caps how many individual rows we keep.
_DETAIL_RING = 200
# Hard cap on persisted failure rows. Only failures are written, so this is reached
# only by a run that is already comprehensively broken -- the cap just stops a
# pathological loop from filling the disk. A final note row records the truncation.
_MAX_TRACE_ROWS = 20_000


@dataclass(frozen=True)
class CommandTraceEntry:
    """One external-command invocation, as observed by sflow.

    ``started_at`` is WALL-CLOCK (epoch seconds), deliberately not the monotonic
    clock used for ``duration_s``: durations must be immune to clock steps, but the
    timestamp exists to be lined up against the workflow Timeline (and against
    cluster-side events like pod transitions), so it has to be a real time of day.
    """

    tool: str
    op: str
    duration_s: float
    rc: int
    timed_out: bool
    started_at: float = 0.0

    @property
    def ok(self) -> bool:
        return self.rc == 0

    @property
    def finished_at(self) -> float:
        return self.started_at + self.duration_s

    def clock(self, epoch: float | None = None) -> str:
        """``HH:MM:SS`` of the START, matching the Timeline section's format."""
        return datetime.fromtimestamp(epoch if epoch is not None else self.started_at).strftime(
            "%H:%M:%S"
        )

    def as_json(self) -> str:
        """One self-describing JSONL row: absolute times, so it can be correlated
        with sflow's own log, the summary Timeline, and `kubectl get events`."""
        return json.dumps(
            {
                "started": datetime.fromtimestamp(self.started_at).isoformat(
                    timespec="milliseconds"
                ),
                "finished": datetime.fromtimestamp(self.finished_at).isoformat(
                    timespec="milliseconds"
                ),
                "epoch": round(self.started_at, 3),
                "tool": self.tool,
                "op": self.op,
                "duration_s": round(self.duration_s, 3),
                "rc": self.rc,
                "timed_out": self.timed_out,
            },
            separators=(",", ":"),
        )


@dataclass
class _OpStats:
    """Rolled-up counters for one ``(tool, op)`` pair -- covers every call."""

    count: int = 0
    failures: int = 0
    timeouts: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    @property
    def mean_s(self) -> float:
        return (self.total_s / self.count) if self.count else 0.0


class CommandTrace:
    """Thread-safe recorder of external-command health.

    Aggregates every call; keeps a bounded ring of the individual failures/slow calls
    worth naming in a report. Recording is a couple of dict updates, so it is safe on
    a poll loop's hot path.
    """

    def __init__(self, detail_ring: int = _DETAIL_RING) -> None:
        self._lock = threading.Lock()
        self._stats: dict[tuple[str, str], _OpStats] = {}
        self._notable: deque[CommandTraceEntry] = deque(maxlen=detail_ring)
        self._fh = None
        self._path: Path | None = None
        self._rows = 0
        self._capped = False

    def attach_file(self, path: "str | Path") -> None:
        """Persist NOTABLE calls to ``path`` as JSONL, for the run starting now.

        Rows are written for FAILED calls and for calls slower than
        ``SLOW_COMMAND_WARN_S``. Ordinary fast successes are fully accounted for by the
        in-memory rollup (count / mean / max per operation), so persisting them would
        add nothing a post-mortem uses while flooding the file: a 2s poll loop emits
        ~43k of them a day, ~7.5MB per task.

        The file is opened LAZILY on the first notable call, so a healthy run leaves no
        artifact behind at all -- including every Slurm/local/docker run, which never
        invokes kubectl.

        Attaching does NOT clear counters -- that is :meth:`begin_run`'s job. The two
        are separate on purpose: a run records kubectl during backend allocation
        (reservations, quota checks) BEFORE its output directory, and therefore this
        file, exists. Clearing here would throw that phase away -- exactly the phase
        where reservation stalls and quota rejections happen. Anything notable already
        buffered is flushed into the new file instead.
        """
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:  # pragma: no cover - defensive
                    pass
            self._fh = None
            self._path = Path(path)
            self._rows = 0
            self._capped = False
            # Backfill: notable calls from before the sink existed still belong to
            # this run's file.
            for entry in list(self._notable):
                self._write_notable_locked(entry)

    def begin_run(self) -> None:
        """Mark the start of a run: drop everything the previous one recorded.

        This recorder is a process-wide singleton, so without an explicit boundary a
        second run in the same process inherits the first run's counters and file
        handle -- a report that misattributes calls is worse than no report. Called
        once, early, before any backend allocation.
        """
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:  # pragma: no cover - defensive
                    pass
            self._fh = None
            self._path = None
            self._rows = 0
            self._capped = False
            self._stats.clear()
            self._notable.clear()

    def _write_notable_locked(self, entry: CommandTraceEntry) -> None:
        """Append one NOTABLE call -- failed, or slow enough to matter.

        Same predicate as the in-memory notable ring, so the file and the summary
        agree on what "worth looking at" means. Slow-but-successful calls belong here:
        a 30s timeout and the 6.7s call that follows it are one story, and without the
        latter the file cannot show when the control plane recovered.

        Caller holds the lock. Never raises.
        """
        if self._path is None or self._capped:
            return
        if entry.ok and entry.duration_s < SLOW_COMMAND_WARN_S:
            return  # ordinary fast success: fully covered by the rollup
        try:
            if self._fh is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # Line-buffered on purpose: the runs that most need this file are the
                # ones killed mid-hang, so rows must be on disk as they happen. Cheap
                # now that only failures reach here.
                self._fh = open(self._path, "a", buffering=1, encoding="utf-8")
            if self._rows >= _MAX_TRACE_ROWS:
                self._capped = True
                self._fh.write(
                    json.dumps({"note": f"trace capped at {_MAX_TRACE_ROWS} notable calls"})
                    + "\n"
                )
                return
            self._fh.write(entry.as_json() + "\n")
            self._rows += 1
        except OSError as exc:  # pragma: no cover - defensive
            _logger.debug(f"command trace file unavailable ({self._path}): {exc}")
            self._capped = True

    @property
    def path(self) -> "Path | None":
        """Where notable calls are written -- ``None`` until one actually occurs."""
        return self._path if self._rows else None

    def record(
        self,
        tool: str,
        op: str,
        duration_s: float,
        rc: int,
        *,
        timed_out: bool = False,
        started_at: float | None = None,
    ) -> None:
        """Record one invocation. Never raises -- telemetry must not break a run."""
        try:
            dur = float(duration_s)
            entry = CommandTraceEntry(
                tool=tool, op=op, duration_s=dur, rc=int(rc),
                timed_out=bool(timed_out),
                # Default keeps callers that do not track wall-clock honest: the start
                # is derived from now minus how long the call took.
                started_at=(time.time() - dur) if started_at is None else float(started_at),
            )
            with self._lock:
                st = self._stats.setdefault((tool, op), _OpStats())
                st.count += 1
                st.total_s += entry.duration_s
                st.max_s = max(st.max_s, entry.duration_s)
                if timed_out:
                    st.timeouts += 1
                if not entry.ok:
                    st.failures += 1
                if not entry.ok or entry.duration_s >= SLOW_COMMAND_WARN_S:
                    self._notable.append(entry)
                # Written INSIDE the lock: this class advertises thread-safety, and an
                # unlocked write is the one place two threads could interleave a row.
                # Only failures reach the disk, so this is rare by construction.
                self._write_notable_locked(entry)
            if entry.duration_s >= SLOW_COMMAND_WARN_S and not timed_out:
                # Timeouts already log at their own call site with more context; this
                # covers the "answered, but far too slowly" case that precedes them.
                _logger.warning(
                    f"{tool} {op} took {entry.duration_s:.1f}s "
                    f"(expected well under {SLOW_COMMAND_WARN_S:.0f}s) -- "
                    "control plane may be degraded"
                )
        except Exception:  # pragma: no cover - defensive: never break the caller
            pass

    def totals(self) -> tuple[int, int, int]:
        """``(calls, failures, timeouts)`` across every recorded invocation."""
        with self._lock:
            return (
                sum(s.count for s in self._stats.values()),
                sum(s.failures for s in self._stats.values()),
                sum(s.timeouts for s in self._stats.values()),
            )

    def summary_lines(self, *, top: int = 8, since: float | None = None) -> list[str]:
        """Render a report section, or ``[]`` when nothing was recorded.

        ``since`` (the workflow's start epoch) adds the same ``+elapsed`` column the
        Timeline section uses, so a reader can put a kubectl stall and a task event
        side by side without converting clocks in their head.
        """
        with self._lock:
            if not self._stats:
                return []
            stats = sorted(
                self._stats.items(), key=lambda kv: kv[1].count, reverse=True
            )
            notable = list(self._notable)
        calls, failures, timeouts = self.totals()
        health = "healthy" if (failures == 0 and timeouts == 0) else "DEGRADED"
        lines = [
            "",
            "External Command Health",
            "-----------------------",
            f"  {calls} call(s), {failures} failure(s), {timeouts} timeout(s) -> {health}",
            "",
            f"  {'tool/op':<34} {'calls':>6} {'fail':>5} {'t/out':>6} {'mean':>8} {'max':>8}",
            f"  {'-' * 34} {'-' * 6} {'-' * 5} {'-' * 6} {'-' * 8} {'-' * 8}",
        ]
        for (tool, op), st in stats[:top]:
            lines.append(
                f"  {(tool + ' ' + op)[:34]:<34} {st.count:>6} {st.failures:>5} "
                f"{st.timeouts:>6} {st.mean_s:>7.2f}s {st.max_s:>7.2f}s"
            )
        if len(stats) > top:
            lines.append(f"  ... and {len(stats) - top} more operation(s)")
        if notable:
            # Same clock (and, when we know the run start, the same +elapsed) as the
            # Timeline section above -- that pairing is the whole point: "the poll
            # stalled at 09:17:02 (+2841s)" lines up with "task X still RUNNING".
            head = "  Slow / failed calls (time of day, matches the Timeline above):"
            lines.extend(["", head])
            for e in notable[-top:]:
                why = "TIMEOUT" if e.timed_out else (f"rc={e.rc}" if not e.ok else "slow")
                elapsed = (
                    f"  +{e.started_at - since:06.3f}s"
                    if since is not None and e.started_at >= since
                    else ""
                )
                lines.append(
                    f"    {e.clock()}{elapsed}  {e.tool} {e.op}: "
                    f"{e.duration_s:.1f}s ({why})"
                )
        if self._rows:
            lines.extend(
                ["", f"  Failed/slow-call trace ({self._rows} row(s)): {self._path}"]
            )
        return lines

    def clear(self) -> None:
        with self._lock:
            self._stats.clear()
            self._notable.clear()


_TRACE = CommandTrace()


def get_command_trace() -> CommandTrace:
    """The process-wide recorder (one sflow run == one driver process)."""
    return _TRACE
