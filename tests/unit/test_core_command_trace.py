# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External-command health telemetry.

Answers the question every stalled run starts with: was it the tool/cluster, or was
it sflow? A wedged ``kubectl`` used to leave no record at all, which is what made a
~20-minute driver hang undiagnosable.
"""

import asyncio
import logging

from sflow.core.command_trace import CommandTrace, CommandTraceEntry, get_command_trace
from sflow.plugins.k8s import lifecycle as life


def test_records_calls_failures_and_timeouts():
    t = CommandTrace()
    t.record("kubectl", "get pod/a", 0.2, 0)
    t.record("kubectl", "get pod/a", 0.3, 0)
    t.record("kubectl", "get pod/b", 0.1, 1)
    t.record("kubectl", "cp pod/c", 30.0, 124, timed_out=True)
    assert t.totals() == (4, 2, 1)  # calls, failures (rc!=0 incl. timeout), timeouts


def test_summary_reports_health_and_per_op_rows():
    t = CommandTrace()
    for _ in range(3):
        t.record("kubectl", "get pod/a", 0.2, 0)
    t.record("kubectl", "delete pod/a", 40.0, 124, timed_out=True)
    body = "\n".join(t.summary_lines())
    assert "External Command Health" in body
    assert "DEGRADED" in body, "a timeout must not be reported as healthy"
    assert "get pod/a" in body and "delete pod/a" in body
    assert "TIMEOUT" in body


def test_summary_says_healthy_when_all_calls_succeed():
    t = CommandTrace()
    t.record("kubectl", "get pod/a", 0.2, 0)
    assert "healthy" in "\n".join(t.summary_lines())


def test_summary_is_empty_when_nothing_recorded():
    assert CommandTrace().summary_lines() == []


def test_slow_call_is_surfaced_live(caplog):
    """A degraded control plane should warn DURING the run, not only after it."""
    t = CommandTrace()
    with caplog.at_level(logging.WARNING):
        t.record("kubectl", "get pod/a", 12.0, 0)
    assert any("took 12.0s" in r.getMessage() for r in caplog.records)


def test_fast_call_is_quiet(caplog):
    t = CommandTrace()
    with caplog.at_level(logging.WARNING):
        t.record("kubectl", "get pod/a", 0.2, 0)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_record_never_raises_on_bad_input():
    """Telemetry must never be able to break the run it is observing."""
    CommandTrace().record("kubectl", "get x", "not-a-number", 0)  # type: ignore[arg-type]


def test_detail_ring_is_bounded_but_totals_are_not():
    t = CommandTrace(detail_ring=5)
    for i in range(50):
        t.record("kubectl", f"get pod/{i}", 0.1, 1)
    calls, failures, _ = t.totals()
    assert (calls, failures) == (50, 50), "aggregates must cover every call"
    assert len(t.summary_lines()) < 60, "detail rows must stay bounded"


# ---------------------------------------------------------------------------
# wiring: run_kubectl feeds the trace, with a redacted op label
# ---------------------------------------------------------------------------


def test_trace_op_keeps_verb_and_target_only():
    assert life._trace_op(["get", "pod/x", "-o", "jsonpath={.status.phase}"]) == "get pod/x"
    assert life._trace_op(["delete", "--namespace", "ns", "pod/y"]) == "delete pod/y"
    assert life._trace_op([]) == "?"


def test_trace_op_redacts_exec_payload():
    """An exec's script can carry arbitrary content -- it must not reach the report."""
    op = life._trace_op(
        ["exec", "pod/x", "--", "sh", "-c", "touch /secret/path && echo hunter2"]
    )
    assert "hunter2" not in op and "secret" not in op
    assert op == "exec pod/x"


def test_run_kubectl_records_success_and_timeout(monkeypatch):
    trace = get_command_trace()
    trace.clear()

    class _P:
        returncode = 0

        def __init__(self, delay=0.0):
            self._d = delay

        async def communicate(self, input=None):
            await asyncio.sleep(self._d)
            return b"Running", b""

        def kill(self):
            pass

        async def wait(self):
            return 0

    async def _fast(*a, **k):
        return _P(0.0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fast)
    asyncio.run(life.run_kubectl(["get", "pod/x"]))
    assert trace.totals() == (1, 0, 0)

    async def _slow(*a, **k):
        return _P(30.0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _slow)
    asyncio.run(life.run_kubectl(["get", "pod/y"], timeout=0.05))
    calls, failures, timeouts = trace.totals()
    assert (calls, failures, timeouts) == (2, 1, 1)
    assert "get pod/y" in "\n".join(trace.summary_lines())
    trace.clear()


def test_trace_op_does_not_mistake_a_flag_value_for_the_target():
    """Regression: ``delete --namespace ns pod/y`` once labelled itself ``delete ns``.

    That is both wrong (the namespace is not the target) and a leak (a flag VALUE
    reaching a report that gets pasted into tickets).
    """
    assert life._trace_op(["delete", "--namespace", "ns", "pod/y"]) == "delete pod/y"
    assert life._trace_op(["get", "pods", "-l", "role=launcher", "-o", "name"]) == "get pods"
    assert life._trace_op(["get", "-n", "ns", "pods"]) == "get pods"


def test_trace_op_is_length_capped():
    long_ref = "pod/" + "x" * 200
    assert len(life._trace_op(["get", long_ref])) <= 60


# ---------------------------------------------------------------------------
# timestamps: every call is time-stamped for co-debugging against the Timeline
# ---------------------------------------------------------------------------


def test_every_call_carries_a_wall_clock_timestamp():
    import time as _t

    t = CommandTrace()
    # A failure is kept in the detail ring, so it can be inspected.
    t.record("kubectl", "get pod/b", 0.5, 1)
    e = list(t._notable)[-1]
    now = _t.time()
    # With no explicit start, the start is derived as "now minus how long it took",
    # so it lands just under half a second in the past -- not in the future.
    assert now - 5 < e.started_at <= now
    assert abs(e.finished_at - (e.started_at + 0.5)) < 1e-6


def test_explicit_start_is_preserved():
    t = CommandTrace()
    t.record("kubectl", "get pod/a", 2.0, 1, started_at=1_000_000.0)
    e = list(t._notable)[-1]
    assert e.started_at == 1_000_000.0
    assert e.finished_at == 1_000_002.0


def test_jsonl_row_is_self_describing():
    import json as _j

    e = CommandTraceEntry(
        tool="kubectl", op="get pod/a", duration_s=1.25, rc=124, timed_out=True,
        started_at=1_000_000.0,
    )
    row = _j.loads(e.as_json())
    assert row["tool"] == "kubectl" and row["op"] == "get pod/a"
    assert row["rc"] == 124 and row["timed_out"] is True
    assert row["duration_s"] == 1.25
    assert row["epoch"] == 1_000_000.0
    # Absolute start AND finish, so a reader can bracket the stall window exactly.
    assert "started" in row and "finished" in row


def test_only_notable_calls_are_persisted(tmp_path):
    """Fast successes stay in the rollup only; failures AND slow calls reach the file.

    A 2s poll loop emits ~43k fast successes a day per task (~7.5MB). Persisting those
    buys nothing a debugger reads -- the rollup already reports count/mean/max. A SLOW
    success is different: it shows the control plane degrading, and (after a stall) the
    moment it recovered.
    """
    import json as _j

    p = tmp_path / "sub" / "command_trace.jsonl"
    t = CommandTrace()
    t.attach_file(p)
    for _ in range(5):
        t.record("kubectl", "get pod/ok", 0.1, 0)          # fast success: not written
    t.record("kubectl", "get pod/bad", 0.2, 1)             # failure: written
    t.record("kubectl", "get pod/slow", 30.0, 124, timed_out=True)  # timeout: written
    t.record("kubectl", "get pod/recovered", 6.7, 0)       # SLOW success: written
    rows = [_j.loads(x) for x in p.read_text().splitlines()]
    assert [r["op"] for r in rows] == [
        "get pod/bad", "get pod/slow", "get pod/recovered"
    ]
    # The slow one is a success -- kept for its duration, not its rc.
    assert rows[-1]["rc"] == 0 and rows[-1]["duration_s"] == 6.7
    # ...and the rollup still accounts for all 8 calls.
    assert t.totals() == (8, 2, 1)


def test_no_file_is_created_when_nothing_is_notable(tmp_path):
    """A healthy run leaves no artifact -- including every local/Slurm/docker run,
    which never invokes kubectl at all."""
    p = tmp_path / "command_trace.jsonl"
    t = CommandTrace()
    t.attach_file(p)
    for _ in range(10):
        t.record("kubectl", "get pod/ok", 0.1, 0)
    assert not p.exists(), "nothing notable => no file"
    assert t.path is None


def test_attach_keeps_counters_and_backfills_pre_attach_calls(tmp_path):
    """Attaching a sink must NOT discard what the run already recorded.

    kubectl runs during backend allocation -- reservations, quota checks -- BEFORE the
    run's output directory (and therefore this file) exists. Clearing on attach would
    throw away exactly the phase where quota rejections and reservation stalls happen.
    Anything notable recorded beforehand is backfilled into the new file.
    """
    t = CommandTrace()
    t.record("kubectl", "get resourcequota/tenant", 0.2, 1)   # allocation-phase failure
    t.record("kubectl", "get pod/ok", 0.1, 0)                 # fast success
    p = tmp_path / "run" / "command_trace.jsonl"
    t.attach_file(p)

    assert t.totals() == (2, 1, 0), "pre-attach calls must survive attaching a sink"
    rows = p.read_text().splitlines()
    assert len(rows) == 1 and "resourcequota" in rows[0], "the failure is backfilled"


def test_begin_run_clears_everything_and_closes_the_handle(tmp_path):
    """begin_run is the run boundary: a second run must inherit nothing.

    The recorder is a process-wide singleton, so without this a later run's report
    would count an earlier run's calls -- a report that misattributes calls is worse
    than no report.
    """
    p1 = tmp_path / "a" / "t.jsonl"
    t = CommandTrace()
    t.attach_file(p1)
    t.record("kubectl", "get pod/run1", 0.1, 1)
    first_fh = t._fh
    assert first_fh is not None and not first_fh.closed

    t.begin_run()
    assert first_fh.closed, "the previous run's handle must not leak"
    assert t.totals() == (0, 0, 0), "counters must not carry over"
    assert t.path is None
    assert "pod/run1" not in "\n".join(t.summary_lines())
    assert p1.read_text().count("pod/run1") == 1, "run 1's file keeps run 1's row"


def test_notable_rows_are_capped(tmp_path, monkeypatch):
    """A pathologically broken run must not fill the disk."""
    import sflow.core.command_trace as mod

    monkeypatch.setattr(mod, "_MAX_TRACE_ROWS", 3)
    p = tmp_path / "t.jsonl"
    t = CommandTrace()
    t.attach_file(p)
    for _ in range(20):
        t.record("kubectl", "get pod/bad", 0.1, 1)
    lines = p.read_text().splitlines()
    assert len(lines) == 4, "3 rows + one truncation note"
    assert "capped" in lines[-1]
    assert t.totals()[0] == 20, "the rollup still counts every call"


def test_summary_rows_show_time_of_day_and_elapsed():
    t = CommandTrace()
    start = 1_000_000.0
    t.record("kubectl", "get pod/a", 30.0, 124, timed_out=True, started_at=start + 120)
    body = "\n".join(t.summary_lines(since=start))
    assert "+120.000s" in body, "elapsed column must match the Timeline's format"
    assert ":" in body  # HH:MM:SS present
    # Without a run start, rows still render (just no elapsed column).
    assert "get pod/a" in "\n".join(t.summary_lines())


def test_summary_points_at_the_trace_file_only_once_it_exists(tmp_path):
    p = tmp_path / "command_trace.jsonl"
    t = CommandTrace()
    t.attach_file(p)
    t.record("kubectl", "get pod/a", 0.1, 0)
    assert str(p) not in "\n".join(t.summary_lines()), "nothing notable => no pointer"
    t.record("kubectl", "get pod/a", 0.1, 1)
    body = "\n".join(t.summary_lines())
    assert str(p) in body and "Failed/slow-call trace" in body


# ---------------------------------------------------------------------------
# consolidation: preflight/allocation kubectl feeds the SAME trace as task-phase
# ---------------------------------------------------------------------------


def test_backend_kubectl_paths_are_traced(monkeypatch):
    """The backend's allocation/preflight calls must reach the shared trace.

    Before consolidation these ran their own subprocess and were invisible, so a
    report could say "kubectl healthy" while reservations and quota checks were
    failing -- the phase where a lot of real instability lives.
    """
    from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig

    be = KubernetesBackend(
        KubernetesBackendConfig(
            name="k8s", type="kubernetes", namespace="ns", nodes=1, gpus_per_node=8,
            scheduling="device_plugin",
        )
    )
    trace = get_command_trace()
    trace.begin_run()

    # Stub the SUBPROCESS, not run_kubectl -- the tracing lives inside the runner, so
    # replacing the runner would test nothing.
    class _P:
        returncode = 1

        async def communicate(self, input=None):
            return b"", b"exceeded quota"

        def kill(self):
            pass

        async def wait(self):
            return 1

    async def _create(*a, **k):
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create)
    asyncio.run(be._kubectl(["get", "resourcequota/tenant-gpu-quota"]))
    assert trace.totals() == (1, 1, 0), "allocation-phase kubectl must be traced"
    assert "resourcequota" in "\n".join(trace.summary_lines())
    trace.begin_run()


def test_preflight_sync_kubectl_is_traced(monkeypatch):
    """Preflight runs synchronously (no event loop) but must still be traced."""
    import subprocess as _sp

    trace = get_command_trace()
    trace.begin_run()

    class _R:
        returncode = 1
        stdout = ""
        stderr = "Unable to connect to the server"

    monkeypatch.setattr(_sp, "run", lambda *a, **k: _R())
    rc, _out, err = life.run_kubectl_sync(["get", "nodes"], global_args=[])
    assert rc == 1 and "Unable to connect" in err
    assert trace.totals() == (1, 1, 0)
    assert "get nodes" in "\n".join(trace.summary_lines())
    trace.begin_run()


def test_run_kubectl_pipes_stdin(monkeypatch):
    """apply -f - feeds a manifest on stdin through the shared runner."""
    seen = {}

    class _P:
        returncode = 0

        async def communicate(self, input=None):
            seen["stdin"] = input
            return b"created", b""

        def kill(self):
            pass

        async def wait(self):
            return 0

    async def _create(*a, **k):
        seen["stdin_pipe"] = k.get("stdin") is not None
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create)
    rc, out, _ = asyncio.run(life.run_kubectl(["apply", "-f", "-"], input=b'{"kind":"Pod"}'))
    assert (rc, out) == (0, "created")
    assert seen["stdin"] == b'{"kind":"Pod"}' and seen["stdin_pipe"] is True


# ---------------------------------------------------------------------------
# delete paths: a delete that WAITS must not be cut short by the poll ceiling
# ---------------------------------------------------------------------------


def test_waiting_delete_gets_a_far_larger_ceiling_than_a_status_poll():
    """A real run measured the allocation label sweep at 28.4s.

    Unlike the task-teardown delete (``--wait=false``, ~0.2s), that sweep blocks on
    the pods' termination grace period. Bounding it at the 30s poll ceiling would
    abort legitimate cleanup and leak objects.
    """
    assert life.DELETE_KUBECTL_TIMEOUT >= 10 * life.POLL_KUBECTL_TIMEOUT
    assert life.DELETE_KUBECTL_TIMEOUT >= 300


def test_alloc_label_sweep_uses_the_delete_ceiling(monkeypatch):
    from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig

    be = KubernetesBackend(
        KubernetesBackendConfig(
            name="k8s", type="kubernetes", namespace="ns", nodes=1, gpus_per_node=8,
            scheduling="device_plugin",
        )
    )
    seen = {}

    async def fake(args, *, global_args=(), timeout=None, input=None):
        seen["args"], seen["timeout"] = list(args), timeout
        return (0, "", "")

    # Assert at the runner: _kubectl derives the ceiling, so this proves the sweep
    # actually reaches kubectl with the generous delete budget.
    monkeypatch.setattr(life, "run_kubectl", fake)
    asyncio.run(be._delete_by_alloc_label("pod", "abc123"))
    assert seen["timeout"] == life.DELETE_KUBECTL_TIMEOUT
    assert "--wait=false" not in seen["args"], (
        "the sweep must keep waiting -- it is the reclamation backstop"
    )


def test_teardown_delete_keeps_the_short_ceiling_and_does_not_wait(monkeypatch):
    """The other delete is fire-and-forget, so the short ceiling stays correct."""
    seen = {}

    async def fake(args, *, global_args=(), timeout=None, input=None):
        seen["args"], seen["timeout"] = list(args), timeout
        return (0, "", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    asyncio.run(life.delete_objects(["pod/a"], global_args=[], ns_args=[]))
    assert "--wait=false" in seen["args"]
    assert seen["timeout"] == life.POLL_KUBECTL_TIMEOUT


def test_trace_op_labels_the_kind_not_the_selector():
    """Regression from a real trace row: ``delete sflow.ai/allocation=<id>``.

    The selector is a FLAG VALUE; labelling the call with it is both wrong (the
    target is the resource kind) and leaks an identifier into a pasted report.
    """
    op = life._trace_op(
        ["delete", "pod", "-l", "sflow.ai/allocation=206906d7", "-n", "ns",
         "--ignore-not-found"]
    )
    assert op == "delete pod"
    # still prefers a real type/name ref when there is one
    assert life._trace_op(["get", "pod/x", "-o", "jsonpath={.status.phase}"]) == "get pod/x"
    assert life._trace_op(["cp", "podname:/out/f.tgz", "/local/f.tgz"]) == "cp podname:/out/f.tgz"
