# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parsing for the GPU occupancy evidence emitted by the reservation samples.

``examples/gpu_reservation/pipeline.yaml`` makes every GPU task print

    SFLOW_GPUEV <run-label> <task> <START|END> <epoch-ms> <uuid,uuid,...>

from *inside* its container, so the UUIDs are what the container actually got
rather than what sflow meant to give it. Turning those lines into comparable
spans is the part of the e2e suite most likely to be quietly wrong -- and the
part that cannot run on a machine without GPUs -- so it lives here, free of
subprocess and Docker side effects, and is unit-tested directly.

Not named ``test_*``: pytest must not collect it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The trailing group is deliberately permissive about what follows "GPU-": real
# UUIDs are hex-and-dashes, but a truncated or oddly-formatted one should still
# parse so a test fails on the ASSERTION rather than on silently finding no
# evidence at all.
GPUEV_RE = re.compile(
    r"SFLOW_GPUEV\s+(\S+)\s+(\S+)\s+(START|END)\s+(\d+)\s+(GPU-\S*)"
)


@dataclass(frozen=True)
class Span:
    """One task's occupancy of a set of physical GPUs, in wall-clock ms."""

    label: str
    task: str
    start: int
    end: int
    uuids: frozenset

    @property
    def key(self) -> str:
        return f"{self.label}/{self.task}"

    def overlaps(self, other: "Span") -> bool:
        """Whether the two spans were live at the same time.

        Strict on both ends: a span ending exactly as another begins is a clean
        handover, not an overlap. Treating that as a collision would fail the
        invariant check on precisely the behavior it is meant to confirm.
        """
        return self.start < other.end and other.start < self.end

    def __repr__(self) -> str:
        # Trim UUIDs to their tail: full ones are 40+ chars and a failure message
        # comparing several spans becomes unreadable.
        return (
            f"<{self.key} {self.start}..{self.end} "
            f"{sorted(u[-6:] for u in self.uuids)}>"
        )


def start_events(text: str) -> dict[str, tuple[int, frozenset]]:
    """``{<label>/<task>: (epoch-ms, uuids)}`` from START lines alone."""
    events: dict[str, tuple[int, frozenset]] = {}
    for label, task, phase, stamp, uuid_csv in GPUEV_RE.findall(text):
        if phase == "START":
            events[f"{label}/{task}"] = (
                int(stamp),
                frozenset(u for u in uuid_csv.split(",") if u.startswith("GPU-")),
            )
    return events


def end_events(text: str) -> dict[str, int]:
    """``{<label>/<task>: epoch-ms}`` from END lines alone."""
    return {
        f"{label}/{task}": int(stamp)
        for label, task, phase, stamp, _uuids in GPUEV_RE.findall(text)
        if phase == "END"
    }


def _span(key: str, start: int, end: int, uuids: frozenset) -> Span:
    label, task = key.split("/", 1)
    return Span(label=label, task=task, start=start, end=end, uuids=uuids)


def parse_spans(text: str) -> dict[str, Span]:
    """Pair START/END evidence lines into spans, keyed ``<label>/<task>``.

    A task with a START but no END is left out rather than guessed at. Use
    :func:`open_spans` to close those deliberately, with an end time the caller can
    actually justify.
    """
    ends = end_events(text)
    return {
        key: _span(key, start, ends[key], uuids)
        for key, (start, uuids) in start_events(text).items()
        if key in ends
    }


def open_spans(text: str, *, end: int) -> dict[str, Span]:
    """Spans for tasks that started but never logged an END, closed at ``end``.

    This is not a fudge for crashed tasks -- it is the normal shape of a task held
    under ``release_after: workflow_completion``. Such a task is a long-lived
    service: it reaches READY (a terminal status), the workflow finishes without
    waiting for it, and the orchestrator tears its container down mid-hold, so no
    END is ever written. Verified against a real run: a probe-gated service slept
    120s, the workflow completed in 4s, and the task ended READY with no END line.

    Pass the driver process's exit time as ``end``: that is when the run actually
    gave the devices back, so the resulting span is what the task really held.
    Erring late is the safe direction for an exclusivity check -- it can only
    invent a conflict, never conceal one.
    """
    ends = end_events(text)
    return {
        key: _span(key, start, end, uuids)
        for key, (start, uuids) in start_events(text).items()
        if key not in ends
    }


def claimed_uuids(text: str) -> dict[str, frozenset]:
    """Each task's claimed devices, whether or not it ever finished."""
    return {key: uuids for key, (_start, uuids) in start_events(text).items()}


def overlapping_spans(
    spans: list[Span], *, same_run_ok: bool = True
) -> list[tuple[Span, Span, frozenset]]:
    """Pairs of spans that held a GPU at the same time.

    ``same_run_ok`` (the default) ignores pairs from the SAME run. Same-run overlap
    on shared devices is expected -- that is exactly what ``release_after:
    task_ready`` produces, and flagging it would make the reuse the samples
    demonstrate look like a bug. Only cross-run sharing violates the registry's
    guarantee.

    ``same_run_ok=False`` flags those pairs too, for a workflow where every task
    holds to its own exit (``task_completion`` throughout) and so no overlap is
    legitimate.
    """
    ordered = sorted(spans, key=lambda s: s.start)
    conflicts: list[tuple[Span, Span, frozenset]] = []
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            if same_run_ok and first.label == second.label:
                continue
            if not first.overlaps(second):
                continue
            shared = first.uuids & second.uuids
            if shared:
                conflicts.append((first, second, shared))
    return conflicts
