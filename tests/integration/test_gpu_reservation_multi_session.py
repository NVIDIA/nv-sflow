# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Several independent sflow sessions contending for one host's GPUs.

The registry exists to keep *separate `sflow run` processes* off each other's
GPUs, so the interesting behavior only appears across real OS processes: the
flock, the on-disk records, and who wins when a slot frees. These tests spawn
actual subprocesses against a shared registry directory.

Physical GPUs are not required -- the inventory (``discover_gpus``) is stubbed in
each worker, while everything that makes the mechanism work (file locking,
record read/write, free-set computation, release) runs for real. That keeps the
scenario deterministic and lets it run in CI on GPU-less machines; the
hardware-level counterpart is ``examples/gpu_reservation/multi_session.sh``.

The scenario mirrors a shared dev box: sessions arrive a few seconds apart asking
for different amounts (4, 8, 2, 1 GPUs of an 8-GPU board), so some fit right away
and some must wait. Notably there is no queue -- a later, smaller session is
admitted ahead of an earlier one still waiting for a slot big enough.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Each worker: stub the GPU inventory, then poll the REAL registry for its share,
# hold it, and release -- logging a timestamped event per transition.
_WORKER = '''
import json, os, sys, time
import sflow.utils.gpu_reservation as gr

name, count, hold, wait_for = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
n = int(os.environ["FAKE_GPU_COUNT"])
gr.discover_gpus = lambda: [gr.GpuHandle(index=i, uuid=f"GPU-{i:04d}") for i in range(n)]

events = open(os.environ["EVENT_LOG"], "w", buffering=1)
def emit(event, **kw):
    events.write(json.dumps({"name": name, "event": event, "t": time.time(), **kw}) + "\\n")

run_id = f"{os.getpid()}-{name}"
deadline = time.time() + wait_for
emit("start", want=count)
while True:
    try:
        handles = gr.try_reserve_gpus(count, run_id)
        break
    except gr.InsufficientGpusError as e:
        if time.time() >= deadline:
            emit("gave_up", free=e.free)
            sys.exit(3)
        emit("waiting", free=e.free)
        time.sleep(0.1)
emit("acquired", uuids=[h.uuid for h in handles])
time.sleep(hold)
gr.release_gpus(run_id)
emit("released")
'''


def _await_event(log: Path, event: str, *, who: str, timeout: float = 60.0) -> None:
    """Block until ``log`` contains ``event`` (the worker reached that point)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log.exists():
            for line in log.read_text().splitlines():
                if line.strip() and json.loads(line).get("event") == event:
                    return
        time.sleep(0.02)
    pytest.fail(f"session '{who}' never reported '{event}'")


class _Session:
    """One `sflow run`-like process: when it starts, how much it wants."""

    def __init__(self, name, *, want, start_at, hold, wait_for=30.0):
        self.name, self.want = name, want
        self.start_at, self.hold, self.wait_for = start_at, hold, wait_for
        self.proc: subprocess.Popen | None = None
        self.events: list[dict] = []

    def at(self, event: str) -> float | None:
        return next((e["t"] for e in self.events if e["event"] == event), None)

    @property
    def uuids(self) -> list[str]:
        return next(
            (e["uuids"] for e in self.events if e["event"] == "acquired"), []
        )

    def waited(self) -> bool:
        return any(e["event"] == "waiting" for e in self.events)


def _run_sessions(tmp_path: Path, sessions: list[_Session], *, gpus: int) -> None:
    """Start each session at its offset, then collect every process's timeline."""
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER)
    registry = tmp_path / "registry"

    env_base = {
        "SFLOW_GPU_RESERVATION_DIR": str(registry),
        "SFLOW_GPU_IGNORE_FOREIGN": "1",  # the stubbed inventory is all ours
        "FAKE_GPU_COUNT": str(gpus),
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", str(tmp_path)),
    }

    # Arrival order must be CAUSAL, not merely timed: a loaded runner can take
    # longer to boot an interpreter than the gap between two start offsets, which
    # would silently reorder the scenario. Each session is only launched once the
    # previous one has actually reached the registry (its "start" event), then the
    # remaining gap is slept off.
    for i, session in enumerate(sorted(sessions, key=lambda s: s.start_at)):
        log = tmp_path / f"{session.name}.jsonl"
        session.proc = subprocess.Popen(
            [sys.executable, str(worker), session.name, str(session.want),
             str(session.hold), str(session.wait_for)],
            env={**env_base, "EVENT_LOG": str(log)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _await_event(log, "start", who=session.name)
        nxt = sorted(sessions, key=lambda s: s.start_at)[i + 1 : i + 2]
        if nxt:
            time.sleep(max(0.0, nxt[0].start_at - session.start_at))

    for session in sessions:
        try:
            _, err = session.proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            session.proc.kill()
            pytest.fail(f"session '{session.name}' never finished")
        log = tmp_path / f"{session.name}.jsonl"
        assert log.exists(), f"session '{session.name}' produced no events: {err!r}"
        session.events = [
            json.loads(line) for line in log.read_text().splitlines() if line.strip()
        ]


def _assert_never_double_booked(sessions: list[_Session]) -> None:
    """No GPU is held by two sessions whose hold windows overlap in time."""
    for i, a in enumerate(sessions):
        for b in sessions[i + 1 :]:
            a_start, a_end = a.at("acquired"), a.at("released")
            b_start, b_end = b.at("acquired"), b.at("released")
            if None in (a_start, b_start):
                continue
            a_end = a_end or float("inf")
            b_end = b_end or float("inf")
            if a_start < b_end and b_start < a_end:  # overlapping holds
                clash = set(a.uuids) & set(b.uuids)
                assert not clash, (
                    f"'{a.name}' and '{b.name}' held {sorted(clash)} at the same time"
                )


def test_staggered_sessions_share_one_board_without_double_booking(tmp_path):
    """Four sessions arrive seconds apart wanting 4, 8, 2 and 1 of 8 GPUs.

    Asserts the property that matters on a shared box: whatever the arrival order
    and sizes, two sessions never hold the same physical GPU at once -- and the
    big waiter still gets its full board once the others let go.
    """
    big_hold = 6.0
    sessions = [
        _Session("s1_want4", want=4, start_at=0.0, hold=big_hold),
        _Session("s2_want8", want=8, start_at=0.5, hold=0.5),
        _Session("s3_want2", want=2, start_at=1.0, hold=1.0),
        _Session("s4_want1", want=1, start_at=1.5, hold=1.0),
    ]
    _run_sessions(tmp_path, sessions, gpus=8)
    by_name = {s.name: s for s in sessions}

    # Everyone eventually got exactly what they asked for.
    for s in sessions:
        assert len(s.uuids) == s.want, f"{s.name} got {s.uuids}"

    _assert_never_double_booked(sessions)

    # The 8-GPU session could not fit next to the 4-GPU holder, so it waited.
    assert by_name["s2_want8"].waited(), "an 8-GPU ask should not fit beside a 4-GPU hold"


def test_a_later_smaller_session_is_admitted_before_an_earlier_waiting_one(tmp_path):
    """First-fit, not first-come: a small ask takes a slot the big one can't use.

    s2 asks for the whole board while s1 holds half of it, so it blocks. s3 and s4
    arrive *after* s2 but need only 2 and 1 GPUs, which fit in what is left -- so
    they run while s2 is still waiting. This is what makes a shared box usable
    (a big job cannot starve small ones), and it is the behavior most likely to
    regress if reservation ever grows a naive queue.
    """
    sessions = [
        _Session("s1_want4", want=4, start_at=0.0, hold=6.0),
        _Session("s2_want8", want=8, start_at=0.5, hold=0.5),
        _Session("s3_want2", want=2, start_at=1.0, hold=1.0),
        _Session("s4_want1", want=1, start_at=1.5, hold=1.0),
    ]
    _run_sessions(tmp_path, sessions, gpus=8)
    by_name = {s.name: s for s in sessions}
    s2, s3, s4 = by_name["s2_want8"], by_name["s3_want2"], by_name["s4_want1"]

    # Both later, smaller sessions were admitted while the earlier big one waited.
    assert s3.at("acquired") < s2.at("acquired"), "s3 (2 GPUs) should not queue behind s2 (8)"
    assert s4.at("acquired") < s2.at("acquired"), "s4 (1 GPU) should not queue behind s2 (8)"
    assert s3.at("acquired") > s2.at("start"), "s3 must have arrived after s2 was already waiting"

    # ...and the big one still got the whole board once everyone released.
    assert len(s2.uuids) == 8
    for other in (by_name["s1_want4"], s3, s4):
        assert s2.at("acquired") >= other.at("released"), (
            f"s2 took the board before '{other.name}' released"
        )


def test_sessions_that_fit_immediately_never_wait(tmp_path):
    """Disjoint asks that all fit are admitted straight away, no polling."""
    # Ordering is guaranteed causally (each launch waits for the previous
    # session to reach the registry), so the offsets only add spacing. Holds must
    # outlast the whole launch span for all four to be holding at once.
    sessions = [
        _Session("a_want4", want=4, start_at=0.0, hold=5.0),
        _Session("b_want2", want=2, start_at=0.1, hold=5.0),
        _Session("c_want1", want=1, start_at=0.2, hold=5.0),
        _Session("d_want1", want=1, start_at=0.3, hold=5.0),
    ]
    _run_sessions(tmp_path, sessions, gpus=8)

    for s in sessions:
        assert not s.waited(), f"{s.name} waited despite a free slot"
        assert len(s.uuids) == s.want
    _assert_never_double_booked(sessions)
    # All four are holding simultaneously...
    last_acquired = max(s.at("acquired") for s in sessions)
    for s in sessions:
        assert s.at("released") > last_acquired, (
            f"{s.name} released before everyone had acquired; holds too short"
        )
    # ...and together they exactly fill the board, each on its own devices.
    held = [u for s in sessions for u in s.uuids]
    assert len(held) == 8 and len(set(held)) == 8


def test_a_session_gives_up_when_its_wait_budget_expires(tmp_path):
    """A bounded wait ends in failure, and does not strand the board."""
    sessions = [
        _Session("holder_want8", want=8, start_at=0.0, hold=3.0),
        _Session("impatient_want2", want=2, start_at=0.3, hold=0.5, wait_for=1.0),
    ]
    _run_sessions(tmp_path, sessions, gpus=8)
    by_name = {s.name: s for s in sessions}

    impatient = by_name["impatient_want2"]
    assert impatient.at("gave_up") is not None, "bounded wait should have expired"
    assert impatient.uuids == []
    assert impatient.proc.returncode == 3
    # The holder is unaffected and still released cleanly.
    assert len(by_name["holder_want8"].uuids) == 8
    assert by_name["holder_want8"].at("released") is not None
