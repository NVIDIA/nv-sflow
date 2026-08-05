# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded kubectl + a collect that never blocks on a dead pod.

Regression cover for a ~20-minute silent driver hang: the node-local output
collector's ``kubectl cp`` races the pod's ``collect_grace_seconds`` window, and
when it loses, the pod has already exited -- so the cp (which execs ``tar`` inside
the container) blocks on a completed pod until the API server finally answers
``cannot exec into a container in a completed pod``. ``execute`` cannot return
until the collector settles, so the whole workflow stalls with zero log output.

Two guarantees are covered here:
  * ``run_kubectl`` accepts an OPTIONAL timeout -- unset keeps the old unbounded
    behavior byte-for-byte (it has 9 callers), set bounds the call and kills the child.
  * ``_collect_via_cp`` checks the pod phase before copying and skips both the cp
    and the done-sentinel exec once the pod is terminal, warning loudly instead of
    hanging (and instead of losing the output silently).
"""

import asyncio
import logging
import os

import pytest

from sflow.plugins.k8s import lifecycle as life
from sflow.plugins.operators.k8s import K8sOperator, K8sOperatorConfig
from sflow.plugins.operators.k8s_operator import (
    _SFLOW_COLLECT_NONE_MARKER,
    _SFLOW_COLLECT_READY_MARKER,
    _sflow_output_collect_trap,
)


def test_no_unbounded_kubectl_call_sites():
    """EVERY ``run_kubectl`` call must pass a timeout -- this is the whole bug.

    A single unbounded call is enough to wedge the driver for the kernel's TCP
    retransmission window (~15-20 min) with no log output, because the caller is
    almost always a poll loop the orchestrator depends on. Bounding four call sites
    is not a fix if a fifth is still unbounded -- notably the k8s_mpi status watch,
    which serves MPI tasks exactly as ``watch_until_terminal`` serves plain pods.

    Static (AST) check so a NEW unbounded call site fails here rather than in
    production 20 silent minutes at a time.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "sflow"
    targets = [
        root / "plugins" / "k8s" / "lifecycle.py",
        root / "plugins" / "k8s" / "mpi_lifecycle.py",
        root / "plugins" / "operators" / "k8s_operator.py",
    ]
    unbounded = []
    for path in targets:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (
                fn.attr if isinstance(fn, ast.Attribute)
                else fn.id if isinstance(fn, ast.Name)
                else None
            )
            if name != "run_kubectl":
                continue
            if not any(kw.arg == "timeout" for kw in node.keywords):
                unbounded.append(f"{path.name}:{node.lineno}")
    assert not unbounded, (
        "unbounded kubectl call(s) can wedge the driver silently: " + ", ".join(unbounded)
    )


def test_mpijob_status_watch_recovers_after_stall(monkeypatch):
    """The MPI watch must survive a wedged control plane, like the plain-pod watch.

    The MLPerf server task is ``k8s_mpi``, so this path -- not just the plain-pod
    one -- has to tolerate a timed-out poll and retry instead of blocking forever.
    """
    from sflow.plugins.k8s import mpi_lifecycle as mpi

    # mpijob_condition issues one query PER terminal condition and returns "" when
    # none is active; the retry lives in watch_mpijob_until_terminal's loop.
    async def all_timeout(args, *, global_args=(), timeout=None):
        assert timeout is not None, "the MPI status poll must be bounded"
        return (life.KUBECTL_TIMEOUT_RC, "", "timed out")

    monkeypatch.setattr(life, "run_kubectl", all_timeout)
    cond = asyncio.run(mpi.mpijob_condition("mpijob/j", global_args=[], ns_args=[]))
    assert cond == "", "a stalled poll must read as 'not terminal yet', never as a verdict"

    # Once the control plane answers again the real condition is returned.
    async def succeeded(args, *, global_args=(), timeout=None):
        assert timeout is not None
        return (0, "True", "")

    monkeypatch.setattr(life, "run_kubectl", succeeded)
    cond = asyncio.run(mpi.mpijob_condition("mpijob/j", global_args=[], ns_args=[]))
    assert cond in ("Succeeded", "Failed")


class _FakeProc:
    """Subprocess stand-in whose ``communicate`` takes ``delay`` seconds."""

    def __init__(self, delay=0.0, out=b"ok", err=b"", rc=0):
        self._delay = delay
        self._out = out
        self._err = err
        self.returncode = rc
        self.killed = False

    async def communicate(self, input=None):
        await asyncio.sleep(self._delay)
        return self._out, self._err

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def _patch_proc(monkeypatch, proc):
    async def _create(*args, **kwargs):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create)
    return proc


# ---------------------------------------------------------------------------
# run_kubectl: optional timeout
# ---------------------------------------------------------------------------


def test_run_kubectl_without_timeout_is_unchanged(monkeypatch):
    """Default (no timeout) keeps the existing contract for all 9 call sites."""
    proc = _patch_proc(monkeypatch, _FakeProc(delay=0.0, out=b"Running", rc=0))
    rc, out, err = asyncio.run(life.run_kubectl(["get", "pod/x"]))
    assert (rc, out, err) == (0, "Running", "")
    assert proc.killed is False


def test_run_kubectl_timeout_returns_nonzero_and_kills_child(monkeypatch):
    """A hung kubectl must not block forever: bound it, kill it, report non-zero."""
    proc = _patch_proc(monkeypatch, _FakeProc(delay=30.0))
    rc, out, err = asyncio.run(life.run_kubectl(["cp", "a", "b"], timeout=0.05))
    assert rc != 0, "a timed-out kubectl must report failure, not success"
    assert "timed out" in err.lower()
    assert proc.killed is True, "the hung child process must be killed, not leaked"


def test_run_kubectl_timeout_not_hit_returns_normally(monkeypatch):
    """A timeout that is not exceeded behaves exactly like the unbounded call."""
    _patch_proc(monkeypatch, _FakeProc(delay=0.0, out=b"done", rc=0))
    rc, out, _ = asyncio.run(life.run_kubectl(["cp", "a", "b"], timeout=5.0))
    assert (rc, out) == (0, "done")


# ---------------------------------------------------------------------------
# pod-status polling: a wedged connection must not stall the driver silently
#
# Root cause of the observed ~20-minute hang (GKE): the driver's TCP connection to
# the API server died silently, and because no kubectl call was bounded, EVERY
# in-flight call blocked until the kernel gave up retransmitting (tcp_retries2=15
# ~= 15-20 min). The pod-status watch is the critical one -- while it is stuck the
# orchestrator cannot notice a finished task, and it logs nothing at all.
# ---------------------------------------------------------------------------


def test_pod_status_poll_is_bounded(monkeypatch):
    """The status poll must pass a timeout, or a dead connection wedges the watch."""
    seen = {}

    async def fake(args, *, global_args=(), timeout=None):
        seen["timeout"] = timeout
        return (0, "Running|x||", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    asyncio.run(life._pod_terminal_status("pod/p", global_args=[], ns_args=[]))
    assert seen["timeout"] is not None and seen["timeout"] > 0


def test_pod_exit_code_poll_is_bounded(monkeypatch):
    seen = {}

    async def fake(args, *, global_args=(), timeout=None):
        seen["timeout"] = timeout
        return (0, "0", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    asyncio.run(life.pod_exit_code("pod/p", global_args=[], ns_args=[], phase="Succeeded"))
    assert seen["timeout"] is not None and seen["timeout"] > 0


def test_timed_out_poll_is_transient_not_a_deleted_pod(monkeypatch):
    """A timeout must NOT be read as 'pod gone' -- that would fail a healthy task."""

    async def fake(args, *, global_args=(), timeout=None):
        return (life.KUBECTL_TIMEOUT_RC, "", "kubectl get timed out after 30.0s")

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase, done, failed, not_found = asyncio.run(
        life._pod_terminal_status("pod/p", global_args=[], ns_args=[])
    )
    assert not_found is False, "a timeout is a transient error, not a confirmed deletion"
    assert (phase, done, failed) == ("", False, False)


def test_watch_recovers_after_connection_stall(monkeypatch):
    """Timeouts must retry, then resolve normally once the API answers again.

    This is the behavior that turns a 20-minute silent wedge into a few retries.
    """
    replies = iter(
        [
            (life.KUBECTL_TIMEOUT_RC, "", "timed out"),
            (life.KUBECTL_TIMEOUT_RC, "", "timed out"),
            (0, "Succeeded|||0", ""),
        ]
    )

    async def fake(args, *, global_args=(), timeout=None):
        return next(replies)

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/p", global_args=[], ns_args=[], interval=0)
    )
    assert phase == "Succeeded"


# ---------------------------------------------------------------------------
# _collect_via_cp: never operate on a terminal pod
# ---------------------------------------------------------------------------


def _op():
    return K8sOperator(K8sOperatorConfig(name="op", image="img:1"))


def _run_collect(op, tmp_path, calls, phase):
    """Drive _collect_via_cp with the ready-marker already present in the log."""
    log = tmp_path / "task.log"
    log.write_text(f"some output\n{_SFLOW_COLLECT_READY_MARKER}\n")

    async def fake_run_kubectl(args, *, global_args=(), timeout=None):
        calls.append((list(args), timeout))
        return (0, "", "")

    async def fake_phase(pod_ref, *, global_args=(), ns_args=(), timeout=None):
        return phase

    return log, fake_run_kubectl, fake_phase


def test_collect_skips_cp_when_pod_already_terminal(monkeypatch, tmp_path, caplog):
    """The pod exited before the driver copied: skip the cp AND the done-exec.

    Both would exec into a completed container and block for ~20 minutes. The
    output is already lost at this point, so the only correct move is to warn.
    """
    op = _op()
    calls = []
    log, fake_run_kubectl, fake_phase = _run_collect(op, tmp_path, calls, "Succeeded")
    monkeypatch.setattr(life, "run_kubectl", fake_run_kubectl)
    monkeypatch.setattr(life, "get_pod_phase", fake_phase)

    with caplog.at_level(logging.WARNING):
        asyncio.run(
            op._collect_via_cp(
                task_name="t",
                pod_ref="pod/p",
                output_dir="/out",
                dest_dir=str(tmp_path / "dest"),
                log_path=str(log),
                global_args=[],
                ns_args=[],
            )
        )

    assert calls == [], f"no kubectl may run against a terminal pod, got {calls}"
    assert any(
        "collect_grace_seconds" in r.message or "collect_grace_seconds" in r.getMessage()
        for r in caplog.records
    ), "the skipped collect must tell the user how to fix it"


def test_collect_timeout_warning_names_the_remedy(monkeypatch, tmp_path, caplog):
    """The branch that actually fires in the wild must name the fix.

    Real-world ordering: the pod is ALIVE when the cp starts (so the terminal-skip
    above does not fire); the cp simply outruns collect_grace_seconds and the pod
    exits mid-copy. That path must say `collect_grace_seconds`, not just "rc=124".
    """
    op = _op()
    log = tmp_path / "task.log"
    log.write_text(f"{_SFLOW_COLLECT_READY_MARKER}\n")

    async def fake_run_kubectl(args, *, global_args=(), timeout=None):
        if args and args[0] == "cp":
            return (life.KUBECTL_TIMEOUT_RC, "", "timed out")
        return (0, "", "")

    async def fake_phase(pod_ref, *, global_args=(), ns_args=(), timeout=None):
        return "Running"

    monkeypatch.setattr(life, "run_kubectl", fake_run_kubectl)
    monkeypatch.setattr(life, "get_pod_phase", fake_phase)

    with caplog.at_level(logging.WARNING):
        asyncio.run(
            op._collect_via_cp(
                task_name="t",
                pod_ref="pod/p",
                output_dir="/out",
                dest_dir=str(tmp_path / "dest"),
                log_path=str(log),
                global_args=[],
                ns_args=[],
            )
        )

    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "collect_grace_seconds" in msgs
    assert "discarded" in msgs.lower() or "abandoned" in msgs.lower()


def test_collect_bounds_the_cp_with_a_timeout(monkeypatch, tmp_path):
    """While the pod is alive the cp still runs -- but bounded, never unbounded."""
    op = _op()
    calls = []
    log, fake_run_kubectl, fake_phase = _run_collect(op, tmp_path, calls, "Running")
    monkeypatch.setattr(life, "run_kubectl", fake_run_kubectl)
    monkeypatch.setattr(life, "get_pod_phase", fake_phase)

    asyncio.run(
        op._collect_via_cp(
            task_name="t",
            pod_ref="pod/p",
            output_dir="/out",
            dest_dir=str(tmp_path / "dest"),
            log_path=str(log),
            global_args=[],
            ns_args=[],
        )
    )

    assert calls, "the cp must still run for a live pod"
    cp_calls = [c for c in calls if c[0] and c[0][0] == "cp"]
    assert cp_calls, f"expected a cp call, got {calls}"
    for args, timeout in calls:
        assert timeout is not None and timeout > 0, (
            f"kubectl {args[0]} must be bounded so it cannot hang the driver"
        )


# ---------------------------------------------------------------------------
# Review follow-ups: coverage for the paths the first pass left untested, and
# for the behaviours added when those findings were fixed.
# ---------------------------------------------------------------------------


def test_launcher_discovery_poll_is_bounded_and_retries(monkeypatch):
    """discover_launcher_pod was entirely uncovered yet is one of the bounded sites.

    Its own deadline cannot fire while an unbounded call is blocked, so the poll must
    carry a timeout and simply retry until the launcher appears.
    """
    from sflow.plugins.k8s import mpi_lifecycle as mpi

    seen, replies = [], iter(
        [
            (life.KUBECTL_TIMEOUT_RC, "", "timed out"),
            (0, "", ""),
            (0, "pod/launcher-abc\n", ""),
        ]
    )

    async def fake(args, *, global_args=(), timeout=None):
        seen.append(timeout)
        return next(replies)

    monkeypatch.setattr(life, "run_kubectl", fake)
    got = asyncio.run(
        mpi.discover_launcher_pod("job", global_args=[], ns_args=[], interval=0)
    )
    assert got == "pod/launcher-abc"
    assert seen and all(t is not None and t > 0 for t in seen), seen


def test_delete_objects_timeout_is_reported_not_swallowed(monkeypatch, caplog):
    """A timed-out teardown delete must not vanish.

    run_kubectl RETURNS a non-zero rc rather than raising, so the surrounding
    ``except Exception`` never fires -- without an explicit check the failure is
    invisible, which is the very thing this work removes.
    """

    async def fake(args, *, global_args=(), timeout=None):
        return (life.KUBECTL_TIMEOUT_RC, "", "timed out")

    monkeypatch.setattr(life, "run_kubectl", fake)
    with caplog.at_level(logging.WARNING):
        asyncio.run(life.delete_objects(["pod/a", "pod/b"], global_args=[], ns_args=[]))
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "teardown delete" in msgs and "timed out" in msgs
    assert "sweep" in msgs, "tell the reader the objects are still reclaimed"


def test_delete_objects_success_is_quiet(monkeypatch, caplog):
    """The happy path must stay silent -- no new noise on every teardown."""

    async def fake(args, *, global_args=(), timeout=None):
        return (0, "", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    with caplog.at_level(logging.WARNING):
        asyncio.run(life.delete_objects(["pod/a"], global_args=[], ns_args=[]))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_status_poll_timeout_warns_once_per_episode_then_recovers(monkeypatch, caplog):
    """One outage = one WARNING per pod, not one per tick, and a visible recovery.

    Every polling task hits the timeout each tick during a control-plane outage, so
    warning unconditionally would emit a line per task per tick.
    """
    life._timeout_streak.clear()
    state = {"fail": True}

    async def fake(args, *, global_args=(), timeout=None):
        if state["fail"]:
            return (life.KUBECTL_TIMEOUT_RC, "", "timed out")
        return (0, "Running|x||", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    with caplog.at_level(logging.DEBUG):
        for _ in range(4):
            asyncio.run(life._pod_terminal_status("pod/p", global_args=[], ns_args=[]))
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1, f"expected one WARNING per episode, got {len(warns)}"

        caplog.clear()
        state["fail"] = False
        asyncio.run(life._pod_terminal_status("pod/p", global_args=[], ns_args=[]))
        assert any("recovered" in r.getMessage() for r in caplog.records)
    assert "pod/p" not in life._timeout_streak, "streak must reset on recovery"


def test_run_kubectl_timeout_survives_a_dead_child(monkeypatch):
    """The kill/wait error branches must not turn a timeout into a crash."""

    class _Stubborn(_FakeProc):
        def kill(self):
            raise ProcessLookupError("already reaped")

        async def wait(self):
            await asyncio.sleep(30)  # never returns -> inner wait_for must bound it

    _patch_proc(monkeypatch, _Stubborn(delay=30.0))
    rc, _out, err = asyncio.run(life.run_kubectl(["get", "pod/x"], timeout=0.05))
    assert rc == life.KUBECTL_TIMEOUT_RC
    assert "timed out" in err.lower()


def _collect_backend_op(grace=120, *, enabled=True):
    """One k8s operator wired to a real backend with the given collect settings."""
    from sflow.core.backend import Allocation
    from sflow.core.compute_node import ComputeNode
    from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig

    backend = KubernetesBackend(
        KubernetesBackendConfig(
            name="k8s", type="kubernetes", namespace="ns", nodes=1, gpus_per_node=8,
            scheduling="device_plugin", collect_grace_seconds=grace,
            collect_node_local_output=enabled,
        )
    )
    backend.allocation = Allocation(
        allocation_id="a1",
        nodes=[ComputeNode(name="node-0", ip_address="10.0.0.1", index=0, num_gpus=8)],
        owned=True,
    )
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=["node-0"], artifacts=[], gpu_count=8,
    )
    return op


def _collect_plan(grace, *, enabled=True):
    """Build a real single-pod execution plan with the given collect settings."""
    op = _collect_backend_op(grace, enabled=enabled)
    return op._build_execution_plan(
        task_name="t",
        script=["echo hi"],
        envs={
            "SFLOW_OUTPUT_DIR": "/out",
            "SFLOW_WORKFLOW_OUTPUT_DIR": "/out/wf",
            "SFLOW_TASK_OUTPUT_DIR": "/out/wf/t",
        },
    )


def test_collect_is_armed_for_a_normal_grace():
    """Baseline: a positive grace arms BOTH sides of the collect handshake."""
    plan = _collect_plan(120)
    assert plan.collect_output is True


def test_collect_disabled_when_grace_is_zero():
    """``collect_grace_seconds: 0`` is an opt-out, not a 0-second copy budget.

    The pod would not wait at all, so arming the trap stages an archive nobody can
    collect and hands the driver a 0s budget -- an instant timeout reported as "did
    not finish within 0s", which reads like a slow copy rather than a disabled
    feature. Neither side may be armed.
    """
    plan = _collect_plan(0)
    assert plan.collect_output is False, "driver-side collector must not run"
    rendered = " ".join(str(a) for a in plan.apply_command.as_list())
    assert _SFLOW_COLLECT_READY_MARKER not in rendered, (
        "the pod-side wait-trap must not be injected either"
    )


def test_collect_fires_once_when_the_task_reaches_its_terminal_state(monkeypatch, tmp_path):
    """The marker (written by the EXIT trap) is the trigger, and it fires one cp."""
    op = _op()
    calls = []
    log, fake_run_kubectl, fake_phase = _run_collect(op, tmp_path, calls, "Running")
    monkeypatch.setattr(life, "run_kubectl", fake_run_kubectl)
    monkeypatch.setattr(life, "get_pod_phase", fake_phase)

    asyncio.run(
        op._collect_via_cp(
            task_name="t", pod_ref="pod/p", output_dir="/out",
            dest_dir=str(tmp_path / "dest"), log_path=str(log),
            global_args=[], ns_args=[],
        )
    )
    verbs = [a[0] for a, _to in calls]
    assert verbs.count("cp") == 1, f"exactly one copy, got {verbs}"
    assert "exec" in verbs, f"the done-sentinel must release the pod, got {verbs}"


# ---------------------------------------------------------------------------
# collect_node_local_output: the master switch
# ---------------------------------------------------------------------------


def test_collect_switch_off_removes_the_mechanism_from_both_sides():
    """``collect_node_local_output: false`` must leave NOTHING of the collect behind.

    The collect is the only part of a task's lifecycle that makes progress depend on
    something other than pod status, readiness probes and the merge-pod done-marker.
    It arms an EXIT trap inside the pod and has the driver poll that pod for a staged
    archive -- which put 42 `kubectl exec` calls into a readiness-probed TRT-LLM server
    during GPU autotuning before it died with SIGTERM. Off must mean off: no trap in
    the rendered script, and no driver-side collector.
    """
    plan = _collect_plan(120, enabled=False)
    assert plan.collect_output is False, "driver-side collector must not run"
    rendered = " ".join(str(a) for a in plan.apply_command.as_list())
    assert "_sflow_collect" not in rendered, "no EXIT trap may be injected into the pod"
    assert _SFLOW_COLLECT_READY_MARKER not in rendered
    # ...and the pod must not be asked to hold itself open for a copy nobody will make.
    assert "awaiting driver copy" not in rendered


def test_collect_switch_on_is_the_unchanged_default():
    """Default stays armed, so turning the switch off is an explicit opt-out."""
    assert _collect_plan(120).collect_output is True
    assert _collect_plan(120, enabled=True).collect_output is True


def test_dag_still_progresses_with_the_collect_switched_off(monkeypatch, tmp_path):
    """The point of the switch: task completion must not depend on the collect.

    With it off, the ONLY things that decide a task's outcome are its pod's status
    (here: terminal + exit code) and its probes. execute() must still return that
    exit code, and must not exec into the pod on the way.
    """
    from sflow.core.backend import Allocation
    from sflow.core.compute_node import ComputeNode
    from sflow.plugins.backends.kubernetes import (
        KubernetesBackend,
        KubernetesBackendConfig,
    )

    execs: list = []

    async def fake_run_kubectl(args, *, global_args=(), timeout=None, input=None):
        execs.append(list(args))
        return (0, "", "")

    async def fake_stream(log_command, dest_path):
        class _P:
            returncode = None

            async def wait(self):
                return 0

            def terminate(self):
                self.returncode = -15

        return _P()

    async def fake_tail(path, *, task_name):
        await asyncio.sleep(3600)

    async def fake_watch(pod_ref, **kw):
        return "Succeeded"

    async def fake_exit(pod_ref, *, phase="", **kw):
        return 0

    async def fake_delete(refs, **kw):
        pass

    def fake_sanitize(paths):
        pass

    monkeypatch.setattr(life, "run_kubectl", fake_run_kubectl)
    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_stream)
    monkeypatch.setattr(life, "tail_file_to_console", fake_tail)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "pod_exit_code", fake_exit)
    monkeypatch.setattr(life, "delete_objects", fake_delete)
    monkeypatch.setattr(life, "sanitize_streamed_logs", fake_sanitize)

    backend = KubernetesBackend(
        KubernetesBackendConfig(
            name="k8s", type="kubernetes", namespace="ns", nodes=1, gpus_per_node=8,
            scheduling="device_plugin", collect_node_local_output=False,
        )
    )
    backend.allocation = Allocation(
        allocation_id="a1",
        nodes=[ComputeNode(name="node-0", ip_address="10.0.0.1", index=0, num_gpus=8)],
        owned=True,
    )
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=["node-0"], artifacts=[], gpu_count=8
    )

    class _Launcher:
        async def run_async(self, command, **kw):
            return 0

    rc = asyncio.run(
        op.execute(
            launcher=_Launcher(), output_logger=None,
            env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
            task_name="t", script=["run"],
        )
    )
    assert rc == 0, "the DAG must still get the task's real exit code"
    assert not [a for a in execs if a and a[0] == "exec"], (
        f"nothing may exec into the pod when the collect is off: {execs}"
    )


def test_collect_issues_no_kubectl_until_the_marker_appears(monkeypatch, tmp_path):
    """The collect trigger is the log marker: LOCAL reads only, no kubectl.

    This is the property that protects a live pod. Polling the pod instead (a
    `kubectl exec test -f` per interval) put 42 execs into a readiness-probed
    TRT-LLM server during GPU autotuning, and it died with SIGTERM. Nothing may
    touch the cluster until the pod itself says its archive is staged.

    The known cost is delivery lag -- `kubectl logs -f` once surfaced the marker
    20 minutes late and the collect window had closed. `collect_grace_seconds`
    widens that window; `collect_node_local_output: false` removes the mechanism.
    """
    op = _op()
    calls = []
    log = tmp_path / "task.log"
    log.write_text("still running, no marker yet\n")

    async def fake_run_kubectl(args, *, global_args=(), timeout=None, input=None):
        calls.append(list(args))
        return (0, "", "")

    async def fake_phase(pod_ref, *, global_args=(), ns_args=(), timeout=None):
        calls.append(["get_pod_phase", pod_ref])
        return "Running"

    monkeypatch.setattr(life, "run_kubectl", fake_run_kubectl)
    monkeypatch.setattr(life, "get_pod_phase", fake_phase)

    async def drive():
        task = asyncio.ensure_future(
            op._collect_via_cp(
                task_name="t", pod_ref="pod/p", output_dir="/out",
                dest_dir=str(tmp_path / "dest"), log_path=str(log),
                global_args=[], ns_args=[],
            )
        )
        for _ in range(50):
            await asyncio.sleep(0)
        assert not task.done(), "collect must still be waiting for the marker"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(drive())
    assert calls == [], f"nothing may exec into a running pod: {calls}"


class _FakeCollectProc:
    """Stand-in for the offloaded `kubectl logs -f` process."""

    def __init__(self):
        self.returncode = None

    def terminate(self):
        self.returncode = -15

    async def wait(self):
        return self.returncode


def _op_with_collect():
    """Operator on a backend with node-local collect ARMED (the default)."""
    return _collect_backend_op()


def test_marker_wait_heartbeats_say_whether_the_log_is_still_growing(monkeypatch, tmp_path, caplog):
    """A long collect wait must explain itself, with the one fact that discriminates.

    The failure this guards is silent by nature: a real run sat 20 minutes between its
    pod finishing and the workflow ending with NOTHING logged, so the post-mortem had to
    be reconstructed from file mtimes. Whether <task>.log is still growing separates
    "task still running" from "the stream stalled and a printed marker may never be
    delivered".
    """
    from sflow.plugins.operators import k8s_operator as k8sop

    log = tmp_path / "task.log"
    log.write_bytes(b"working...\n")

    async def drive():
        task = asyncio.ensure_future(
            k8sop._wait_for_marker(
                str(log), b"[[never-appears]]",
                interval=0.005, task_name="t", heartbeat=0.05,
            )
        )
        await asyncio.sleep(0.3)   # real time: several heartbeat periods
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    with caplog.at_level(logging.INFO):
        asyncio.run(drive())

    beats = [r.getMessage() for r in caplog.records if "still waiting" in r.getMessage()]
    assert beats, "a long wait must report what it is waiting for"
    assert any("has not grown" in m for m in beats), beats
    assert any(str(log) in m for m in beats), "the watched path must be named"


def test_terminal_pod_with_no_sentinel_at_all_is_reported(monkeypatch, tmp_path, caplog):
    """Neither sentinel arriving is genuinely anomalous, and must be reported.

    Distinct from a task that simply produced no output -- that case announces itself
    (see the quiet test below). Reaching terminal with NEITHER sentinel means the trap's
    lines never made it to <task>.log: the container died before the trap ran, or the
    log stream never delivered them.
    """
    from sflow.plugins.operators import k8s_operator as k8sop

    async def _noop_stream(log_command, dest_path):
        return _FakeCollectProc()

    async def _noop_tail(path, *, task_name):
        await asyncio.sleep(3600)

    async def _watch(pod_ref, **kw):
        return "Succeeded"

    async def _exit(pod_ref, *, phase="", **kw):
        return 0

    async def _delete(refs, **kw):
        pass

    def _sanitize(paths):
        pass

    async def _never(*a, **k):
        await asyncio.sleep(3600)   # marker never arrives

    monkeypatch.setattr(life, "start_pod_log_file_stream", _noop_stream)
    monkeypatch.setattr(life, "tail_file_to_console", _noop_tail)
    monkeypatch.setattr(life, "watch_until_terminal", _watch)
    monkeypatch.setattr(life, "pod_exit_code", _exit)
    monkeypatch.setattr(life, "delete_objects", _delete)
    monkeypatch.setattr(life, "sanitize_streamed_logs", _sanitize)
    monkeypatch.setattr(k8sop, "_wait_for_marker", _never)

    op = _op_with_collect()

    class _Launcher:
        async def run_async(self, command, **kw):
            return 0

    with caplog.at_level(logging.WARNING):
        rc = asyncio.run(
            op.execute(
                launcher=_Launcher(), output_logger=None,
                # SFLOW_OUTPUT_DIR is what arms the collect (see the gate in
                # _build_execution_plan) -- without it no collector is created and the
                # test would pass vacuously.
                env={
                    "SFLOW_OUTPUT_DIR": str(tmp_path),
                    "SFLOW_WORKFLOW_OUTPUT_DIR": str(tmp_path / "wf"),
                    "SFLOW_TASK_OUTPUT_DIR": str(tmp_path / "wf" / "t"),
                },
                task_name="t", script=["run"],
            )
        )
    assert rc == 0
    msgs = [r.getMessage() for r in caplog.records]
    assert any("was NOT collected" in m for m in msgs), msgs
    # ...and it must say WHICH causes to look for, plus where to look.
    assert any(
        "without either collect sentinel" in m and "heartbeat lines" in m
        for m in msgs
    ), msgs


def test_task_with_no_output_finishes_the_collect_without_warning(monkeypatch, tmp_path, caplog):
    """A task that legitimately produces nothing must resolve QUIETLY and promptly.

    The trap only prints the ready marker when it staged something; the no-output branch
    prints its own sentinel. Before that sentinel existed the collector waited forever
    for a marker that was never coming, so it was still pending at pod-terminal and
    EVERY no-output task drew a "node-local output was NOT collected" warning -- exactly
    the noise that teaches people to ignore the warning that matters.

    Driven at ``_collect_via_cp`` level on purpose: that is where the sentinel is wired
    in, and without it this call never returns (the wait_for below is what fails).
    """
    from sflow.plugins.operators import k8s_operator as k8sop

    op = _op()
    calls: list = []
    log = tmp_path / "task.log"
    log.write_text(
        "some output\n"
        f"sflow: no new in-pod output files {k8sop._SFLOW_COLLECT_NONE_MARKER}\n"
    )

    async def fake_run_kubectl(args, *, global_args=(), timeout=None, input=None):
        calls.append(list(args))
        return (0, "", "")

    async def fake_phase(pod_ref, *, global_args=(), ns_args=(), timeout=None):
        calls.append(["get_pod_phase", pod_ref])
        return "Running"

    monkeypatch.setattr(life, "run_kubectl", fake_run_kubectl)
    monkeypatch.setattr(life, "get_pod_phase", fake_phase)

    with caplog.at_level(logging.WARNING):
        asyncio.run(
            asyncio.wait_for(
                op._collect_via_cp(
                    task_name="t", pod_ref="pod/p", output_dir="/out",
                    dest_dir=str(tmp_path / "dest"), log_path=str(log),
                    global_args=[], ns_args=[],
                ),
                timeout=2,
            )
        )
    assert calls == [], f"nothing to collect must cost no kubectl: {calls}"
    noisy = [r.getMessage() for r in caplog.records]
    assert not noisy, f"a no-output task must not warn: {noisy}"


def test_none_sentinel_ends_the_wait_without_copying(monkeypatch, tmp_path, caplog):
    """The sentinel resolves the wait directly, with no kubectl and no warning."""
    from sflow.plugins.operators import k8s_operator as k8sop

    log = tmp_path / "task.log"
    log.write_text(
        f"sflow: no new in-pod output files {k8sop._SFLOW_COLLECT_NONE_MARKER}\n"
    )
    with caplog.at_level(logging.INFO):
        staged = asyncio.run(
            asyncio.wait_for(
                k8sop._wait_for_marker(
                    str(log),
                    k8sop._SFLOW_COLLECT_READY_MARKER.encode(),
                    interval=0.01,
                    task_name="t",
                    stop_marker=k8sop._SFLOW_COLLECT_NONE_MARKER.encode(),
                ),
                timeout=2,
            )
        )
    assert staged is False, "the stop sentinel must end the wait"
    assert any("no collectable output" in r.getMessage() for r in caplog.records)


def test_trap_flushes_a_dangling_partial_line_before_each_sentinel():
    """Both collect sentinels must be preceded by a bare newline.

    THE post-run hang. `kubectl logs -f` reassembles CRI partial-line entries and only
    emits once a line TERMINATES. A task whose tail is an unterminated `\\r` progress bar
    (tqdm/aiperf/pip) therefore freezes the follow, and everything printed after it --
    including these sentinels -- is withheld with it.

    Measured live on an MLPerf harness mid-run: `kubectl logs` (one-shot) had 271,494
    bytes while the followed <task>.log had 17,623; the 253,871-byte difference held
    2,149 carriage returns and ZERO newlines, and the follow process was alive the whole
    time. The marker only surfaced ~20 minutes later when the stream closed, and the DAG
    waited out the entire gap.

    Emitting a newline first closes the pending line so the sentinel is delivered
    promptly. Without it the sentinel queues behind the progress bar and the hang
    returns, so this ordering is load-bearing, not cosmetic.
    """
    from sflow.plugins.operators.k8s_operator import (
        _SFLOW_COLLECT_NONE_MARKER,
        _SFLOW_COLLECT_READY_MARKER,
        _sflow_output_collect_trap,
    )

    lines = _sflow_output_collect_trap(10 * 1024 * 1024, 120, ()).splitlines()
    for sentinel in (_SFLOW_COLLECT_READY_MARKER, _SFLOW_COLLECT_NONE_MARKER):
        idx = next(i for i, ln in enumerate(lines) if sentinel in ln)
        # walk back to the echo that carries the sentinel (its command may wrap)
        start = idx
        while start > 0 and not lines[start].strip().startswith("echo "):
            start -= 1
        assert lines[start - 1].strip() == 'echo ""', (
            f"{sentinel} must be preceded by a bare newline echo to flush any "
            f"unterminated progress-bar line; got {lines[start - 1]!r}"
        )


# ---------------------------------------------------------------------------
# A container-log ROTATION does not just delay the marker, it DESTROYS it: once a
# container log passes containerLogMaxSize (10 MiB default) the kubelet renames the
# whole file and starts an empty one, and `kubectl logs`/`logs -f` only ever serve the
# CURRENT file. Measured on a GB300 cluster (kubelet v1.34.3) with two pods differing
# only in output volume: the one that wrote ~30 MB left a 29,772,304-byte `0.log.<ts>`
# beside a 0-byte `0.log` and `kubectl logs` returned 0 BYTES -- no task output and no
# marker -- while a 4.4 MB pod delivered everything including an unterminated `\r` line
# and the marker behind it. So the partial-line flush above cannot cover this: the
# announcement is gone, and the archive died with the pod (the observed symptom was an
# empty <task>.log and "node-local output was NOT collected").
# ---------------------------------------------------------------------------


def test_pod_reannounces_the_ready_marker_while_it_waits():
    """One announcement is not enough -- a rotation can discard it outright.

    The pod must keep saying it while it holds the container open, so a rotation that
    ate the first copy still leaves one in the POST-rotation log for the follow to
    deliver. This is the fix for the empty-log case above, and it costs no kubectl.
    """
    from sflow.plugins.operators.k8s_operator import (
        _SFLOW_COLLECT_REANNOUNCE_SECONDS,
        _sflow_output_collect_trap,
    )

    lines = _sflow_output_collect_trap(10 * 1024 * 1024, 120, ()).splitlines()
    body = "\n".join(lines)
    assert f"$((_i % {_SFLOW_COLLECT_REANNOUNCE_SECONDS})) -eq 0" in body, (
        "the wait loop must re-announce on an interval, not once"
    )
    # The repeat has to carry the marker the driver actually scans for, and has to sit
    # INSIDE the wait loop (after it, a rotation still orphans the archive).
    start = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("while [ ! -f")
    )
    end = next(i for i, ln in enumerate(lines[start:], start) if ln.strip() == "done")
    assert any(_SFLOW_COLLECT_READY_MARKER in ln for ln in lines[start:end]), (
        f"no re-announcement inside the wait loop: {lines[start:end]}"
    )



def _usable_bash():
    """A bash that actually runs, else skip.

    On Windows ``shutil.which("bash")`` usually resolves to the WSL launcher stub in
    ``WindowsApps``, which is not bash and exits non-zero -- probe before trusting it,
    and fall back to the bash shipped alongside git. CI is Linux, where the first
    candidate is the real thing.
    """
    import shutil
    import subprocess
    from pathlib import Path

    candidates = [shutil.which("bash")]
    git = shutil.which("git")
    if git:
        candidates.append(str(Path(git).resolve().parent.parent / "bin" / "bash.exe"))
    for cand in candidates:
        if not cand or not Path(cand).exists():
            continue
        try:
            probe = subprocess.run(
                [cand, "-c", "echo ok"], capture_output=True, timeout=60
            )
        except OSError:
            continue
        if probe.returncode == 0 and probe.stdout.strip() == b"ok":
            return cand
    pytest.skip("no working bash available")


def test_generated_collect_trap_is_valid_bash(tmp_path, fake_process):
    """The trap is assembled as text and only ever executes inside a pod.

    A syntax error therefore surfaces as a cluster run that dies in the entrypoint, with
    the real failure buried in a pod log -- so it is checked here instead. The collect
    trap's wait loop in particular is now multi-line with a nested `if`, which is exactly
    the shape that breaks when lines are appended without their terminators.
    """
    import subprocess

    fake_process.allow_unregistered(True)  # run real bash (conftest fakes subprocess)
    script = tmp_path / "trap.sh"
    script.write_text(
        _sflow_output_collect_trap(10 * 1024 * 1024, 30, ["./artifact.py"]),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [_usable_bash(), "-n", str(script)], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"collect trap is not valid bash:\n{proc.stderr}"


def test_reannounce_interval_always_fits_inside_the_grace_window():
    """A window shorter than the interval would produce ZERO re-announcements.

    That silently removes the container-log-rotation protection from precisely the
    configuration least able to lose the marker (a short copy window). A modulus of 0
    would also be a shell arithmetic error rather than a skipped branch.
    """
    import re as _re

    for grace in (1, 2, 3, 5, 9, 10, 11, 30, 120, 3600):
        body = _sflow_output_collect_trap(1024, grace, ())
        m = _re.search(r"\$\(\(_i % (\d+)\)\)", body)
        assert m, f"no re-announce modulus rendered for grace={grace}"
        interval = int(m.group(1))
        assert interval >= 1, f"modulus 0 for grace={grace} -- shell arithmetic error"
        assert interval <= max(1, grace), (
            f"grace={grace}s with a {interval}s interval never re-announces"
        )


# ---------------------------------------------------------------------------
# every path through the collect trap announces itself
# ---------------------------------------------------------------------------


def _run_trap(tmp_path, fake_process, *, output_dir, workflow_dir, files=()):
    """Execute the rendered EXIT trap under REAL bash and return its stdout.

    Really running it matters: the branch under test is shell control flow, and the
    failure it guards against was a path through that flow which printed nothing. An
    assertion over the rendered TEXT would have passed while the pod stayed silent.
    The unit suite stubs out subprocess globally, so opt this one call back in.
    """
    import subprocess

    fake_process.allow_unregistered(True)

    from sflow.plugins.operators.k8s_operator import _sflow_output_collect_trap

    for name in files:
        target = workflow_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("payload")
    script = tmp_path / "trap.sh"
    # grace=3 so the ready path finishes quickly when it waits for a driver copy.
    script.write_text(_sflow_output_collect_trap(10 * 1024 * 1024, 3, ()) + "\nexit 0\n")
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "SFLOW_OUTPUT_DIR": str(output_dir),
            "SFLOW_WORKFLOW_OUTPUT_DIR": str(workflow_dir),
        },
    )
    return proc.stdout


@pytest.mark.parametrize(
    "case,files,expected",
    [
        ("staged output", ("result.json",), _SFLOW_COLLECT_READY_MARKER),
        ("nothing to collect", (), _SFLOW_COLLECT_NONE_MARKER),
    ],
)
def test_trap_announces_the_outcome_it_reached(
    tmp_path, fake_process, case, files, expected
):
    out_dir = tmp_path / "out"
    wf_dir = out_dir / "wf"
    wf_dir.mkdir(parents=True)
    stdout = _run_trap(
        tmp_path, fake_process, output_dir=out_dir, workflow_dir=wf_dir, files=files
    )
    assert expected in stdout, f"{case}: {stdout!r}"


def test_trap_announces_even_when_the_output_dir_is_unusable(tmp_path, fake_process):
    """The trap RAN but found no output directory -- that must still be announced.

    The driver treats "neither sentinel appeared" as a hard diagnostic: the container
    died before its EXIT trap ran, or the trap's output was never delivered. A missing
    output dir (the preamble's `mkdir -p` lost to a read-only mount) is neither, so
    without a sentinel here the collector waits out the whole grace window and then
    reports a cause that did not happen. Every path must emit exactly one sentinel, so
    that "no sentinel" keeps its precise meaning.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stdout = _run_trap(
        tmp_path, fake_process, output_dir=out_dir,
        workflow_dir=tmp_path / "never-created",
    )
    assert _SFLOW_COLLECT_NONE_MARKER in stdout, stdout
    assert _SFLOW_COLLECT_READY_MARKER not in stdout, (
        "there is nothing staged, so the driver must not be told to copy"
    )


def test_every_trap_path_emits_exactly_one_sentinel(tmp_path, fake_process):
    """Two sentinels in one run would make the driver both copy and give up."""
    out_dir = tmp_path / "out"
    wf_dir = out_dir / "wf"
    wf_dir.mkdir(parents=True)
    for files in ((), ("a.txt",)):
        stdout = _run_trap(
            tmp_path, fake_process, output_dir=out_dir, workflow_dir=wf_dir, files=files
        )
        seen = stdout.count(_SFLOW_COLLECT_NONE_MARKER)
        # The ready marker is deliberately RE-announced while waiting (log rotation),
        # so count distinct sentinel KINDS rather than occurrences.
        kinds = sum(
            1
            for m in (_SFLOW_COLLECT_READY_MARKER, _SFLOW_COLLECT_NONE_MARKER)
            if m in stdout
        )
        assert kinds == 1, f"files={files!r} produced {kinds} sentinel kinds: {stdout!r}"
        assert seen <= 1
