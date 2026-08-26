# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GPU-occupancy evidence parser used by the docker e2e suite.

The e2e tests that consume this only run on a GPU host, so without these the
parsing and the cross-run overlap rule would ship unexercised -- and a parser
that silently finds nothing turns every e2e assertion into a vacuous pass.

Loaded by path rather than imported: ``tests/e2e_tests`` is not a package, and its
test module has import-time side effects (it shells out to `docker info` and
`nvidia-smi`) that a unit test has no business triggering.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "e2e_tests" / "gpu_evidence.py"
)
_spec = importlib.util.spec_from_file_location("gpu_evidence", _MODULE_PATH)
gpu_evidence = importlib.util.module_from_spec(_spec)
# Register BEFORE executing: @dataclass resolves sys.modules[cls.__module__] while
# processing the class, which is None for a module that is not yet registered.
sys.modules.setdefault("gpu_evidence", gpu_evidence)
_spec.loader.exec_module(gpu_evidence)

Span = gpu_evidence.Span
parse_spans = gpu_evidence.parse_spans
overlapping_spans = gpu_evidence.overlapping_spans


def _ev(label, task, phase, stamp, uuids):
    return f"SFLOW_GPUEV {label} {task} {phase} {stamp} {uuids}"


def _span(label, task, start, end, uuids):
    return Span(
        label=label, task=task, start=start, end=end, uuids=frozenset(uuids)
    )


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_start_and_end_pair_into_a_span():
    text = "\n".join(
        [
            _ev("A", "server", "START", 1000, "GPU-aaa,GPU-bbb"),
            _ev("A", "server", "END", 5000, "GPU-aaa,GPU-bbb"),
        ]
    )
    spans = parse_spans(text)
    assert set(spans) == {"A/server"}
    span = spans["A/server"]
    assert (span.label, span.task) == ("A", "server")
    assert (span.start, span.end) == (1000, 5000)
    assert span.uuids == frozenset({"GPU-aaa", "GPU-bbb"})


def test_evidence_is_found_amid_surrounding_log_noise():
    """Real input is a <task>.log full of sflow's own timestamped lines."""
    text = (
        "2026-08-07 10:00:00,123 - sflow.task.server - INFO - starting up\n"
        + "2026-08-07 10:00:01,000 - sflow.task.server - INFO - "
        + _ev("A", "server", "START", 1000, "GPU-aaa")
        + "\n[Vector addition of 50000 elements]\nTest PASSED\n"
        + "2026-08-07 10:00:09,000 - sflow.task.server - INFO - "
        + _ev("A", "server", "END", 9000, "GPU-aaa")
        + "\n"
    )
    spans = parse_spans(text)
    assert spans["A/server"].uuids == frozenset({"GPU-aaa"})
    assert (spans["A/server"].start, spans["A/server"].end) == (1000, 9000)


def test_a_task_that_died_before_END_is_dropped_not_guessed():
    """Inventing an end time would let a half-dead run satisfy the overlap check.

    Dropping it instead surfaces as a missing-key failure that names the task.
    """
    spans = parse_spans(_ev("A", "server", "START", 1000, "GPU-aaa"))
    assert spans == {}


def test_several_runs_and_tasks_are_kept_apart():
    text = "\n".join(
        [
            _ev("A", "server", "START", 100, "GPU-aaa"),
            _ev("A", "server", "END", 900, "GPU-aaa"),
            _ev("A", "client", "START", 300, "GPU-aaa"),
            _ev("A", "client", "END", 500, "GPU-aaa"),
            _ev("B", "server", "START", 150, "GPU-bbb"),
            _ev("B", "server", "END", 800, "GPU-bbb"),
        ]
    )
    spans = parse_spans(text)
    assert set(spans) == {"A/server", "A/client", "B/server"}
    assert spans["B/server"].uuids == frozenset({"GPU-bbb"})


def test_non_uuid_tokens_are_ignored():
    """A blank or malformed device list must not become a phantom GPU."""
    text = "\n".join(
        [
            _ev("A", "t", "START", 1, "GPU-aaa,,notauuid"),
            _ev("A", "t", "END", 2, "GPU-aaa"),
        ]
    )
    assert parse_spans(text)["A/t"].uuids == frozenset({"GPU-aaa"})


def test_lines_without_the_sentinel_are_not_parsed():
    assert parse_spans("nvidia-smi --query-gpu=uuid GPU-aaa\nrandom output") == {}


# ---------------------------------------------------------------------------
# overlap semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_range,b_range,expected",
    [
        ((0, 100), (50, 150), True),    # partial
        ((0, 100), (100, 200), False),  # clean handover: touching, not overlapping
        ((0, 100), (101, 200), False),  # disjoint
        ((0, 100), (10, 20), True),     # contained
        ((50, 150), (0, 100), True),    # symmetric with the first case
    ],
)
def test_overlap_boundaries(a_range, b_range, expected):
    a = _span("A", "t", a_range[0], a_range[1], {"GPU-x"})
    b = _span("B", "t", b_range[0], b_range[1], {"GPU-x"})
    assert a.overlaps(b) is expected
    assert b.overlaps(a) is expected, "overlap must be symmetric"


# ---------------------------------------------------------------------------
# the cross-run invariant
# ---------------------------------------------------------------------------


def test_two_runs_sharing_a_gpu_at_the_same_time_is_a_conflict():
    spans = [
        _span("A", "server", 0, 1000, {"GPU-aaa"}),
        _span("B", "server", 500, 1500, {"GPU-aaa"}),
    ]
    conflicts = overlapping_spans(spans)
    assert len(conflicts) == 1
    first, second, shared = conflicts[0]
    assert {first.label, second.label} == {"A", "B"}
    assert shared == frozenset({"GPU-aaa"})


def test_same_run_sharing_a_gpu_is_the_expected_reuse_not_a_conflict():
    """`release_after: task_ready` produces exactly this shape -- the client runs
    on the server's devices while the server is still on them. Flagging it would
    make the feature the samples demonstrate look like a bug."""
    spans = [
        _span("A", "server", 0, 1000, {"GPU-aaa"}),
        _span("A", "client", 400, 700, {"GPU-aaa"}),
    ]
    assert overlapping_spans(spans) == []


def test_two_runs_on_the_same_gpu_at_different_times_is_correct_queueing():
    """The waiter case: B only got the GPU after A let it go."""
    spans = [
        _span("A", "server", 0, 1000, {"GPU-aaa"}),
        _span("B", "server", 1000, 2000, {"GPU-aaa"}),
    ]
    assert overlapping_spans(spans) == []


def test_two_runs_at_once_on_different_gpus_is_correct_packing():
    spans = [
        _span("A", "server", 0, 1000, {"GPU-aaa"}),
        _span("B", "server", 0, 1000, {"GPU-bbb"}),
    ]
    assert overlapping_spans(spans) == []


def test_partial_overlap_of_multi_gpu_claims_reports_only_the_shared_device():
    spans = [
        _span("A", "server", 0, 1000, {"GPU-aaa", "GPU-bbb"}),
        _span("B", "server", 500, 1500, {"GPU-bbb", "GPU-ccc"}),
    ]
    conflicts = overlapping_spans(spans)
    assert len(conflicts) == 1
    assert conflicts[0][2] == frozenset({"GPU-bbb"})


def test_every_offending_pair_is_reported_not_just_the_first():
    """A failure message naming one pair would hide how wide the breakage is."""
    spans = [
        _span("A", "server", 0, 1000, {"GPU-aaa"}),
        _span("B", "server", 100, 1000, {"GPU-aaa"}),
        _span("C", "server", 200, 1000, {"GPU-aaa"}),
    ]
    assert len(overlapping_spans(spans)) == 3  # AB, AC, BC


def test_a_realistic_clean_contention_timeline_has_no_conflicts():
    """Three runs queueing over two GPUs, each reusing its own server's devices."""
    spans = [
        _span("A", "server", 0, 1000, {"GPU-aaa"}),
        _span("A", "client", 300, 600, {"GPU-aaa"}),
        _span("B", "server", 0, 1000, {"GPU-bbb"}),
        _span("B", "client", 300, 600, {"GPU-bbb"}),
        # C queued until A released, then took A's device.
        _span("C", "server", 1000, 2000, {"GPU-aaa"}),
        _span("C", "client", 1300, 1600, {"GPU-aaa"}),
    ]
    assert overlapping_spans(spans) == []


# ---------------------------------------------------------------------------
# open spans: the workflow_completion service that never logs an END
# ---------------------------------------------------------------------------


def test_a_started_but_unfinished_task_becomes_a_span_closed_at_the_given_end():
    """A `release_after: workflow_completion` service reaches READY, the workflow
    finishes without waiting for it, and its container is torn down mid-hold -- so
    it never writes an END. Closing it at the driver's exit is what makes the
    device it held visible to the exclusivity check at all."""
    text = _ev("A", "pinned_service", "START", 1000, "GPU-aaa")
    opened = gpu_evidence.open_spans(text, end=9000)
    assert set(opened) == {"A/pinned_service"}
    assert (opened["A/pinned_service"].start, opened["A/pinned_service"].end) == (
        1000,
        9000,
    )
    assert opened["A/pinned_service"].uuids == frozenset({"GPU-aaa"})


def test_open_spans_ignores_tasks_that_did_finish():
    """Otherwise a completed task would be closed twice, the second time at a
    later, invented end -- silently widening its span."""
    text = "\n".join(
        [
            _ev("A", "done", "START", 100, "GPU-aaa"),
            _ev("A", "done", "END", 200, "GPU-aaa"),
            _ev("A", "held", "START", 150, "GPU-bbb"),
        ]
    )
    assert set(gpu_evidence.open_spans(text, end=9000)) == {"A/held"}
    assert set(parse_spans(text)) == {"A/done"}


def test_parse_and_open_spans_together_cover_every_started_task():
    text = "\n".join(
        [
            _ev("A", "done", "START", 100, "GPU-aaa"),
            _ev("A", "done", "END", 200, "GPU-aaa"),
            _ev("A", "held", "START", 150, "GPU-bbb"),
        ]
    )
    merged = parse_spans(text)
    merged.update(gpu_evidence.open_spans(text, end=9000))
    assert set(merged) == {"A/done", "A/held"}


def test_a_held_device_still_trips_the_cross_run_check():
    """The regression this guards: if pinned tasks were left out (they have no
    END), the devices held LONGEST would be the ones never checked."""
    a_text = _ev("A", "pinned_service", "START", 0, "GPU-aaa")
    b_text = "\n".join(
        [
            _ev("B", "worker", "START", 500, "GPU-aaa"),
            _ev("B", "worker", "END", 900, "GPU-aaa"),
        ]
    )
    spans = list(gpu_evidence.open_spans(a_text, end=2000).values())
    spans += list(parse_spans(b_text).values())
    conflicts = overlapping_spans(spans)
    assert len(conflicts) == 1
    assert conflicts[0][2] == frozenset({"GPU-aaa"})


# ---------------------------------------------------------------------------
# claimed_uuids
# ---------------------------------------------------------------------------


def test_claimed_uuids_reports_unfinished_tasks_too():
    text = "\n".join(
        [
            _ev("A", "held", "START", 100, "GPU-aaa"),
            _ev("A", "done", "START", 100, "GPU-bbb"),
            _ev("A", "done", "END", 200, "GPU-bbb"),
        ]
    )
    assert gpu_evidence.claimed_uuids(text) == {
        "A/held": frozenset({"GPU-aaa"}),
        "A/done": frozenset({"GPU-bbb"}),
    }


def test_start_and_end_event_accessors_are_keyed_consistently():
    text = "\n".join(
        [
            _ev("A", "t", "START", 100, "GPU-aaa"),
            _ev("A", "t", "END", 200, "GPU-aaa"),
        ]
    )
    assert set(gpu_evidence.start_events(text)) == set(gpu_evidence.end_events(text))


def test_same_run_overlap_is_only_a_conflict_when_asked_for():
    """`task_ready` reuse makes same-run overlap normal, so it is off by default --
    but a workflow that holds every device to its own exit has no such excuse."""
    spans = [
        gpu_evidence.Span("A", "server", 100, 300, frozenset({"GPU-aaa"})),
        gpu_evidence.Span("A", "consumer", 200, 400, frozenset({"GPU-aaa"})),
    ]

    assert gpu_evidence.overlapping_spans(spans) == []
    conflicts = gpu_evidence.overlapping_spans(spans, same_run_ok=False)
    assert [(a.task, b.task, sorted(shared)) for a, b, shared in conflicts] == [
        ("server", "consumer", ["GPU-aaa"])
    ]
