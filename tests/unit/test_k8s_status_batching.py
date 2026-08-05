# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pod-status polling collapses into one kubectl call per tick.

The status poll was ~90% of all kubectl traffic (862 of 957 calls in a measured
7-pod run) because every watcher polled its own pod. At 19 concurrent recipes that
self-congests -- mean `get pod` latency rose 0.2s -> 1.3s. Batching keeps the same
per-pod contract while issuing one query for every pod being watched.
"""

import asyncio
import logging
import time

import pytest

from sflow.plugins.k8s import lifecycle as life


@pytest.fixture(autouse=True)
def _clean_batcher():
    b = life._POD_STATUS_BATCHER
    b._active.clear(); b._cache.clear(); b._locks.clear()
    yield
    b._active.clear(); b._cache.clear(); b._locks.clear()


def _register(*refs, ga=(), na=()):
    for r in refs:
        life._POD_STATUS_BATCHER.register(r, global_args=ga, ns_args=na)


def test_single_watcher_uses_the_original_single_pod_query(monkeypatch):
    """Below two watchers nothing changes -- same call shape as before."""
    calls = []

    async def fake(args, *, global_args=(), timeout=None, input=None):
        calls.append(list(args))
        return (0, "Running|x||", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    _register("pod/a")
    asyncio.run(life._pod_terminal_status("pod/a", global_args=[], ns_args=[]))
    assert len(calls) == 1
    assert calls[0][:2] == ["get", "pod/a"], calls[0]
    assert "--ignore-not-found" not in calls[0], "single-pod path must be unchanged"


def test_many_watchers_share_one_kubectl_call(monkeypatch):
    """The headline: N pods, ONE query."""
    calls = []

    async def fake(args, *, global_args=(), timeout=None, input=None):
        calls.append(list(args))
        return (0, "a|Running|x||\nb|Running|x||\nc|Succeeded|||0\n", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    _register("pod/a", "pod/b", "pod/c")

    async def poll_all():
        return await asyncio.gather(*[
            life._pod_terminal_status(r, global_args=[], ns_args=[])
            for r in ("pod/a", "pod/b", "pod/c")
        ])

    got = asyncio.run(poll_all())
    assert len(calls) == 1, f"3 pods must cost 1 call, got {len(calls)}"
    assert got[0][0] == "Running" and got[2][0] == "Succeeded"
    # every watched pod is named in the one query
    assert {"pod/a", "pod/b", "pod/c"} <= set(calls[0])


def test_absent_pod_in_a_successful_batch_is_a_real_deletion(monkeypatch):
    """--ignore-not-found means a deleted pod is simply missing from the output."""
    async def fake(args, *, global_args=(), timeout=None, input=None):
        return (0, "a|Running|x||\n", "")  # 'b' is gone

    monkeypatch.setattr(life, "run_kubectl", fake)
    _register("pod/a", "pod/b")
    phase, done, failed, not_found = asyncio.run(
        life._pod_terminal_status("pod/b", global_args=[], ns_args=[])
    )
    assert not_found is True and phase == ""


def test_failed_batch_is_transient_never_a_deletion(monkeypatch):
    """A failing batch must not make every pod look deleted -- that would fail
    healthy tasks en masse."""
    async def fake(args, *, global_args=(), timeout=None, input=None):
        return (life.KUBECTL_TIMEOUT_RC, "", "timed out")

    monkeypatch.setattr(life, "run_kubectl", fake)
    _register("pod/a", "pod/b")
    for ref in ("pod/a", "pod/b"):
        phase, done, failed, not_found = asyncio.run(
            life._pod_terminal_status(ref, global_args=[], ns_args=[])
        )
        assert not_found is False, f"{ref}: a failed batch is transient"
        assert (phase, done, failed) == ("", False, False)


def test_batch_is_refreshed_once_the_sample_goes_stale(monkeypatch):
    """Staleness is bounded, so terminal detection still converges."""
    calls = []
    outs = iter(["a|Running|x||\nb|Running|x||\n", "a|Succeeded|||0\nb|Running|x||\n"])

    async def fake(args, *, global_args=(), timeout=None, input=None):
        calls.append(1)
        return (0, next(outs), "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    monkeypatch.setattr(life, "_BATCH_FRESH_S", 0.0)  # always stale -> always refresh
    _register("pod/a", "pod/b")
    first = asyncio.run(life._pod_terminal_status("pod/a", global_args=[], ns_args=[]))
    second = asyncio.run(life._pod_terminal_status("pod/a", global_args=[], ns_args=[]))
    assert first[0] == "Running" and second[0] == "Succeeded"
    assert len(calls) == 2


def test_separate_kubectl_contexts_do_not_share_a_batch(monkeypatch):
    """Different namespace/kubeconfig => different query; must not be mixed."""
    seen = []

    async def fake(args, *, global_args=(), timeout=None, input=None):
        seen.append(tuple(global_args))
        return (0, "a|Running|x||\nb|Running|x||\n", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    _register("pod/a", "pod/b", ga=("--context", "one"))
    _register("pod/c", "pod/d", ga=("--context", "two"))
    asyncio.run(life._pod_terminal_status("pod/a", global_args=("--context", "one"), ns_args=()))
    asyncio.run(life._pod_terminal_status("pod/c", global_args=("--context", "two"), ns_args=()))
    assert seen == [("--context", "one"), ("--context", "two")]


def test_unregister_releases_the_context(monkeypatch):
    b = life._POD_STATUS_BATCHER
    _register("pod/a", "pod/b")
    assert b.active_count(global_args=(), ns_args=()) == 2
    b.unregister("pod/a", global_args=(), ns_args=())
    b.unregister("pod/b", global_args=(), ns_args=())
    assert b.active_count(global_args=(), ns_args=()) == 0
    assert not b._cache and not b._locks, "state must not leak between runs"


# ---------------------------------------------------------------------------
# the log stream is no longer on the collect path, and the log re-fetch is bounded
# ---------------------------------------------------------------------------


def test_no_raw_kubectl_spawn_bypasses_the_traced_runners():
    """Guard: a raw create_subprocess_exec("kubectl", ...) is invisible AND unbounded.

    The AST guard on run_kubectl could not see these -- two `kubectl logs` call sites
    slipped through exactly that way.
    """
    import ast
    import pathlib

    root = pathlib.Path(life.__file__).resolve().parents[1]
    allowed = {"start_pod_log_file_stream", "_run_kubectl_to_file", "run_kubectl"}
    offenders = []
    for path in [root / "k8s" / "lifecycle.py", root / "k8s" / "mpi_lifecycle.py",
                 root / "operators" / "k8s_operator.py", root / "backends" / "kubernetes.py"]:
        tree = ast.parse(path.read_text())
        fn_at = {}
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for ln in range(n.lineno, (n.end_lineno or n.lineno) + 1):
                    fn_at[ln] = n.name
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name != "create_subprocess_exec":
                continue
            first = n.args[0] if n.args else None
            if isinstance(first, ast.Constant) and first.value == "kubectl":
                owner = fn_at.get(n.lineno, "?")
                if owner not in allowed:
                    offenders.append(f"{path.name}:{n.lineno} in {owner}()")
    assert not offenders, (
        "raw kubectl spawn bypasses the bounded+traced runners: " + ", ".join(offenders)
    )


def test_batch_query_sends_an_escaped_newline_not_a_raw_one(monkeypatch):
    """Regression: a REAL newline byte in the jsonpath makes kubectl exit 1.

    ``'{"\\n"}'`` in a Python literal is a newline CHARACTER; kubectl's jsonpath
    parser wants the two-character escape and rejects the raw byte with
    ``error parsing jsonpath``. Every stubbed test still passed because none of them
    ran kubectl -- so assert on the argv that would reach it. The live symptom was
    total: no multi-pod task could ever reach a terminal state.
    """
    seen = []

    async def fake(args, *, global_args=(), timeout=None, input=None):
        seen.append(list(args))
        return (0, "a|Running|x||\nb|Running|x||\n", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    asyncio.run(
        life._batched_pod_status(["pod/a", "pod/b"], global_args=[], ns_args=[])
    )
    jsonpath = next(a for a in seen[0] if a.startswith("jsonpath="))
    assert "\n" not in jsonpath, (
        "raw newline in the jsonpath -- kubectl rejects it with 'error parsing "
        f"jsonpath': {jsonpath!r}"
    )
    assert r"{\n}" in jsonpath or r'{"\n"}' in jsonpath, jsonpath


def test_single_ref_batch_never_reports_a_live_pod_as_deleted(monkeypatch):
    """Regression: the LIST-shaped query is invalid for ONE named resource.

    ``kubectl get pod/x -o jsonpath={range .items[*]}...`` returns a bare Pod (no
    ``.items``), so the template yields NOTHING at rc=0 -- and "absent from a
    successful batch" means "confirmed deleted". Verified against a real cluster: a
    healthy Running pod came back as ``('', False, False, True)``. Reachable whenever
    the active set shrinks to one while a batch tick is in flight (fail-fast cancels
    peers together), and only masked from failing a task by ``_GONE_POLLS == 2``.
    """
    calls = []

    async def fake(args, *, global_args=(), timeout=None, input=None):
        calls.append(list(args))
        return (0, "Running|2026-01-01T00:00:00Z||", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    got = asyncio.run(
        life._batched_pod_status(["pod/a"], global_args=[], ns_args=[])
    )
    assert got["pod/a"][0] == "Running", got
    assert got["pod/a"][3] is False, "a live pod must never be reported not_found"
    assert len(calls) == 1
    assert "--ignore-not-found" not in calls[0], "must use the single-pod query shape"
    assert not any("{range" in a for a in calls[0]), "the LIST template needs >=2 refs"


def test_empty_batch_issues_no_query(monkeypatch):
    """The active set can be emptied entirely mid-tick -- ask kubectl nothing."""
    calls = []

    async def fake(args, *, global_args=(), timeout=None, input=None):
        calls.append(list(args))
        return (0, "", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    got = asyncio.run(life._batched_pod_status([], global_args=[], ns_args=[]))
    assert got == {} and calls == []


def test_batch_uses_the_membership_current_at_query_time(monkeypatch):
    """A peer that unregisters while we wait for the lock must not be queried."""
    b = life._POD_STATUS_BATCHER
    seen = []

    async def fake(args, *, global_args=(), timeout=None, input=None):
        seen.append([a for a in args if a.startswith("pod/")])
        return (0, "a|Running|x||\nb|Running|x||\n", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    _register("pod/a", "pod/b", "pod/c")

    async def drive():
        # 'c' leaves before the tick issues its query.
        b.unregister("pod/c", global_args=(), ns_args=())
        return await life._pod_terminal_status("pod/a", global_args=[], ns_args=[])

    asyncio.run(drive())
    assert seen and "pod/c" not in seen[0], f"stale membership queried: {seen[0]}"


# ---------------------------------------------------------------------------
# a stalled control plane must still SAY so once the poll is batched
# ---------------------------------------------------------------------------


def test_batched_poll_failure_warns_once_per_episode_then_recovers(monkeypatch, caplog):
    """Batching must not re-silence the outage the single-pod path already reports.

    A batch is taken whenever >=2 pods share a context -- i.e. essentially every real
    workflow -- so a batched poll that failed quietly would leave a stalled control
    plane looking like a frozen DAG with no log output at all. The event-loop watchdog
    cannot cover this: the poll is an ordinary await, so the loop is idle, not blocked.
    """
    life._timeout_streak.clear()
    state = {"fail": True}

    async def fake(args, *, global_args=(), timeout=None, input=None):
        if state["fail"]:
            return (life.KUBECTL_TIMEOUT_RC, "", "timed out")
        return (0, "a|Running|x||\nb|Running|x||\n", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    monkeypatch.setattr(life, "_BATCH_FRESH_S", 0.0)  # every tick issues a query
    _register("pod/a", "pod/b")

    with caplog.at_level(logging.DEBUG):
        for _ in range(4):
            asyncio.run(life._pod_terminal_status("pod/a", global_args=[], ns_args=[]))
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1, f"expected one WARNING per episode, got {len(warns)}"
        assert "status poll" in warns[0].getMessage()

        caplog.clear()
        state["fail"] = False
        asyncio.run(life._pod_terminal_status("pod/a", global_args=[], ns_args=[]))
        assert any("recovered" in r.getMessage() for r in caplog.records)


def test_batched_poll_reports_a_non_timeout_failure_too(monkeypatch, caplog):
    """rc=1 is a real error for a batch, unlike the single-pod query.

    ``--ignore-not-found`` turns a missing pod into rc=0 with the row absent, so a
    non-zero rc can never mean NotFound here -- it is always a genuine failure, and an
    unreported one would loop every watcher in the batch forever with nothing logged.
    """
    life._timeout_streak.clear()

    async def fake(args, *, global_args=(), timeout=None, input=None):
        return (1, "", "error parsing jsonpath")

    monkeypatch.setattr(life, "run_kubectl", fake)
    _register("pod/a", "pod/b")
    with caplog.at_level(logging.WARNING):
        asyncio.run(life._pod_terminal_status("pod/a", global_args=[], ns_args=[]))
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1, "a non-timeout batch failure must not be silent"
    assert "error parsing jsonpath" in warns[0].getMessage(), warns[0].getMessage()


def test_batch_streak_key_is_stable_as_membership_changes():
    """The streak must not restart when a pod joins or leaves mid-outage.

    Keying on the pod list would reset the streak on every membership change and
    re-warn each tick -- the flooding the warn-once rule exists to prevent.
    """
    assert life._batch_streak_key(["-n", "ns"]) == life._batch_streak_key(["-n", "ns"])
    assert life._batch_streak_key(["-n", "a"]) != life._batch_streak_key(["-n", "b"])
    assert not life._batch_streak_key(["-n", "ns"]).startswith("pod/")


def test_a_successful_batch_clears_the_per_pod_streaks_it_supersedes(monkeypatch):
    """Nothing else clears a per-pod entry once that pod polls via a batch.

    A leftover streak would make the pod's next lone timeout log at debug -- invisible
    in exactly the situation the warning exists for.
    """
    life._timeout_streak.clear()
    life._timeout_streak["pod/a"] = 3  # warned earlier on the single-pod path

    async def fake(args, *, global_args=(), timeout=None, input=None):
        return (0, "a|Running|x||\nb|Running|x||\n", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    _register("pod/a", "pod/b")
    asyncio.run(life._pod_terminal_status("pod/a", global_args=[], ns_args=[]))
    assert "pod/a" not in life._timeout_streak, "stale per-pod streak must be cleared"


# ---------------------------------------------------------------------------
# console tailer must never starve the event loop (live-hang regression)
# ---------------------------------------------------------------------------


def test_a_tick_always_emits_its_first_line_however_long(monkeypatch, tmp_path, caplog):
    """Forward progress must not depend on how the two caps compare.

    A line is echoed whole or not at all, so a per-tick budget smaller than one
    clamped line would reject the FIRST line of every tick and silence the console
    permanently. The clamped length is the per-line cap PLUS a truncation notice whose
    length grows with the task name, so comparing the constants cannot guarantee this.
    """
    monkeypatch.setattr(life, "_console_active", lambda: True)
    monkeypatch.setattr(life, "_TAIL_MAX_CHARS_PER_TICK", 1)  # absurdly small budget
    log = tmp_path / "t.log"
    log.write_bytes(b"")

    async def drive():
        task = asyncio.ensure_future(
            life.tail_file_to_console(str(log), task_name="t")
        )
        await asyncio.sleep(0)
        log.write_bytes(b"x" * 5000 + b"\n" + b"second line\n")
        await asyncio.sleep(life._TAIL_POLL_INTERVAL * 3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    with caplog.at_level(logging.INFO):
        asyncio.run(drive())
    echoed = [
        r.getMessage()
        for r in caplog.records
        if "truncated for the console" in r.getMessage()
    ]
    assert echoed, "the first line of a tick must be emitted even over budget"
