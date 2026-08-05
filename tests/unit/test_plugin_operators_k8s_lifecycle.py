# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver-side k8s lifecycle helpers + the operator's decoupled ``execute`` flow.

These cover the Python half of the k8s operator: polling pod status, deriving the
exit code, deleting objects, and the apply -> stream -> watch -> stop-on-terminal
orchestration in ``K8sContainerOperator.execute`` (with kubectl + the launcher
faked out so no cluster is needed)."""

import asyncio
import logging
import os
import time
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.plugins.k8s import lifecycle as life
from sflow.utils import console_text
from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig
from sflow.plugins.operators.k8s import K8sOperator, K8sOperatorConfig


def _backend(namespace="ns", nodes=1, gpus_per_node=8):
    backend = KubernetesBackend(
        KubernetesBackendConfig(name="k8s", type="kubernetes", namespace=namespace,
                                nodes=nodes, gpus_per_node=gpus_per_node,
                                scheduling="dra")
    )
    backend.allocation = Allocation(
        allocation_id="abc",
        nodes=[ComputeNode(name=f"node-{i}", ip_address=f"10.0.0.{i + 1}", index=i,
                           num_gpus=gpus_per_node) for i in range(nodes)],
        owned=True,
    )
    backend._node_to_resv_pod = {f"node-{i}": f"res-{i}" for i in range(nodes)}
    return backend


# ---------------------------------------------------------------------------
# lifecycle helpers (kubectl faked via run_kubectl)
# ---------------------------------------------------------------------------


def test_watch_until_terminal_returns_on_succeeded(monkeypatch):
    phases = iter(["Running", "Running", "Succeeded"])

    async def fake(args, *, global_args=(), timeout=None):
        return (0, next(phases), "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/t", global_args=[], ns_args=[], interval=0)
    )
    assert phase == "Succeeded"


def test_watch_until_terminal_returns_empty_when_pod_gone(monkeypatch):
    async def fake(args, *, global_args=(), timeout=None):
        return (1, "", "NotFound")  # rc != 0 -> phase ""

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/t", global_args=[], ns_args=[], interval=0)
    )
    assert phase == ""


def test_watch_until_terminal_completes_on_container_exit_when_phase_lags(monkeypatch):
    # The pod phase still reads "Running" but the container has already terminated
    # (exit 0): the phase lags container exit, so watch must complete NOW on the true
    # container status instead of waiting out the lag. (combined jsonpath:
    # phase|running|waiting|exitcodes)
    async def fake(args, *, global_args=(), timeout=None):
        return (0, "Running|||0", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/t", global_args=[], ns_args=[], interval=0)
    )
    assert phase == "Succeeded"


def test_watch_until_terminal_derives_failed_from_container_exit(monkeypatch):
    async def fake(args, *, global_args=(), timeout=None):
        return (0, "Running|||137", "")  # container terminated non-zero, phase lag

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/t", global_args=[], ns_args=[], interval=0)
    )
    assert phase == "Failed"


def test_watch_until_terminal_still_running_container_not_done(monkeypatch):
    # A genuinely running container (running.startedAt present, no exit code) must
    # NOT be treated as terminal -- only phase transition or container exit ends it.
    phases = iter(["Running|2026-01-01T00:00:00Z||", "Succeeded|||0"])

    async def fake(args, *, global_args=(), timeout=None):
        return (0, next(phases), "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/t", global_args=[], ns_args=[], interval=0)
    )
    assert phase == "Succeeded"  # completed on the phase flip, not the running poll


def test_pod_terminal_status_parses_combined_fields(monkeypatch):
    cases = {
        "Running|2026-01-01T00:00:00Z||": ("Running", False, False, False),  # running
        "Running|||0": ("Running", True, False, False),  # terminated ok, phase lagging
        "Running|||137": ("Running", True, True, False),  # terminated non-zero
        "Succeeded|||0": ("Succeeded", True, False, False),  # terminal phase + exit
        "Pending|||": ("Pending", False, False, False),  # nothing started yet
    }
    for out, expected in cases.items():
        async def fake(args, *, global_args=(), timeout=None, _out=out):
            return (0, _out, "")

        monkeypatch.setattr(life, "run_kubectl", fake)
        got = asyncio.run(
            life._pod_terminal_status("pod/t", global_args=[], ns_args=[])
        )
        assert got == expected, out

    # A genuinely deleted pod: kubectl exits non-zero with NotFound -> not_found=True.
    async def gone(args, *, global_args=(), timeout=None):
        return (1, "", 'Error from server (NotFound): pods "t" not found')

    monkeypatch.setattr(life, "run_kubectl", gone)
    assert asyncio.run(
        life._pod_terminal_status("pod/t", global_args=[], ns_args=[])
    ) == ("", False, False, True)

    # A TRANSIENT control-plane error also exits non-zero, but must NOT be reported as
    # not_found -- otherwise a healthy pod is mistaken for a deleted one.
    for transient in (
        "Unable to connect to the server: net/http: TLS handshake timeout",
        "error: the server was unable to return a response in the time allotted",
        "Error from server (Timeout): the request could not be completed",
        "Error from server (TooManyRequests): please try again later",
    ):
        async def flaky(args, *, global_args=(), timeout=None, _e=transient):
            return (1, "", _e)

        monkeypatch.setattr(life, "run_kubectl", flaky)
        assert asyncio.run(
            life._pod_terminal_status("pod/t", global_args=[], ns_args=[])
        ) == ("", False, False, False), transient


def test_watch_until_terminal_ignores_transient_api_errors(monkeypatch):
    # A healthy long-lived pod must NOT be declared gone during a burst of transient
    # kubectl errors (even more than _GONE_POLLS of them): the streak only counts
    # confirmed NotFound. Here transient errors precede a real terminal phase.
    monkeypatch.setattr(life, "_GONE_POLLS", 2)
    seq = [
        (1, "", "Unable to connect to the server: net/http: TLS handshake timeout"),
        (1, "", "error: the server was unable to return a response in the time allotted"),
        (1, "", "Error from server (TooManyRequests): please try again later"),
        (0, "Running|2026-01-01T00:00:00Z||", ""),  # still healthy
        (0, "Succeeded|||0", ""),  # finally terminal
    ]

    async def fake(args, *, global_args=(), timeout=None):
        return seq.pop(0) if seq else (0, "Succeeded|||0", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/t", global_args=[], ns_args=[], interval=0)
    )
    assert phase == "Succeeded"  # transient errors never triggered a false "gone"


def test_watch_until_terminal_detects_real_deletion(monkeypatch):
    # A genuinely deleted pod (repeated NotFound) IS declared gone after _GONE_POLLS.
    monkeypatch.setattr(life, "_GONE_POLLS", 2)
    notfound = (1, "", 'Error from server (NotFound): pods "t" not found')

    async def fake(args, *, global_args=(), timeout=None):
        return notfound

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/t", global_args=[], ns_args=[], interval=0)
    )
    assert phase == ""  # gone


def test_pod_exit_code_reads_terminated_exit_code(monkeypatch):
    async def fake(args, *, global_args=(), timeout=None):
        return (0, "7", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    code = asyncio.run(life.pod_exit_code("pod/t", global_args=[], ns_args=[]))
    assert code == 7


def test_pod_exit_code_falls_back_to_phase(monkeypatch):
    monkeypatch.setattr(life, "EXIT_CODE_RETRIES", 1)

    async def fake(args, *, global_args=(), timeout=None):
        return (0, "", "")  # exitCode never present

    async def no_sleep(_):
        return None

    monkeypatch.setattr(life.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(life, "run_kubectl", fake)
    assert asyncio.run(
        life.pod_exit_code("pod/t", global_args=[], ns_args=[], phase="Succeeded")
    ) == 0
    assert asyncio.run(
        life.pod_exit_code("pod/t", global_args=[], ns_args=[], phase="Failed")
    ) == 1


def test_delete_objects_issues_nonblocking_delete(monkeypatch):
    captured = []

    async def fake(args, *, global_args=(), timeout=None):
        captured.append((list(args), list(global_args)))
        return (0, "", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    asyncio.run(
        life.delete_objects(
            ["pod/t", "configmap/t-cfg"],
            global_args=["--context", "c"],
            ns_args=["--namespace", "ns"],
        )
    )
    assert captured == [
        (
            ["delete", "pod/t", "configmap/t-cfg", "--namespace", "ns",
             "--ignore-not-found", "--wait=false"],
            ["--context", "c"],
        )
    ]
    captured.clear()
    asyncio.run(life.delete_objects([], global_args=[], ns_args=[]))
    assert captured == []  # nothing to delete -> no kubectl call


# ---------------------------------------------------------------------------
# multi-pod fail-fast: one dead sub-pod fails the whole task (no hang on peers)
# ---------------------------------------------------------------------------


def test_gather_pods_fail_fast_all_succeed():
    async def ok(code, phase):
        return (code, phase)

    results = asyncio.run(
        life.gather_pods_fail_fast([ok(0, "Succeeded"), ok(0, "Succeeded")])
    )
    assert results == [(0, "Succeeded"), (0, "Succeeded")]


def test_gather_pods_fail_fast_cancels_peers_when_one_fails():
    # The leader fails fast while the worker "watcher" would idle for a long time
    # (mirrors a multi-node worker pod on `sleep 3600` after the leader crashed).
    # The failure must cancel the worker and return at once -- not block on it.
    cancelled = {"worker": False}

    async def leader():
        return (1, "Failed")

    async def worker():
        try:
            await asyncio.sleep(3600)
            return (0, "Succeeded")
        except asyncio.CancelledError:
            cancelled["worker"] = True
            raise

    async def drive():
        # wait_for is the regression guard: pre-fix this blocked until SIGINT.
        return await asyncio.wait_for(
            life.gather_pods_fail_fast([leader(), worker()]), timeout=2
        )

    results = asyncio.run(drive())
    assert results[0] == (1, "Failed")
    assert results[1] is None  # worker cancelled -> no result
    assert cancelled["worker"] is True


def test_gather_pods_fail_fast_treats_watcher_exception_as_failure():
    async def boom():
        raise RuntimeError("kubectl blew up")

    async def ok():
        return (0, "Succeeded")

    results = asyncio.run(life.gather_pods_fail_fast([boom(), ok()]))
    assert results[0] == (1, "Failed")  # a raising watcher counts as a failed pod


def test_gather_pods_fail_fast_mpi_world_group_resolves_on_leader_success():
    # A multi-node MPI leader (index 0) whose `mpirun` exits masked-0 (Succeeded)
    # while the idle worker watchers never return would otherwise block the task
    # until SIGINT. With mpi_world_group, the leader terminating (ANY code) resolves
    # the gather now and cancels the still-idle peers.
    cancelled = {"w1": False, "w2": False}

    async def leader():
        return (0, "Succeeded")

    async def idle(key):
        try:
            await asyncio.Event().wait()  # never returns (idle worker)
            return (0, "Succeeded")
        except asyncio.CancelledError:
            cancelled[key] = True
            raise

    async def drive():
        return await asyncio.wait_for(
            life.gather_pods_fail_fast(
                [leader(), idle("w1"), idle("w2")], mpi_world_group=True
            ),
            timeout=2,
        )

    results = asyncio.run(drive())
    assert results[0] == (0, "Succeeded")
    assert results[1] is None and results[2] is None  # idle peers cancelled
    assert cancelled == {"w1": True, "w2": True}


def test_gather_pods_fail_fast_mpi_world_group_resolves_on_any_pod_terminal():
    # Broadened Fix B: the pods are one MPI world group, so ANY pod going terminal
    # (here a WORKER at index 1, masked-Succeeded) resolves the whole task and cancels
    # the rest -- a dead/finished rank breaks the group, don't wait on the others (nor
    # only on the leader). The leader here stays "running" and is cancelled.
    cancelled = {"leader": False, "w2": False}

    async def leader():
        try:
            await asyncio.Event().wait()  # leader still running
            return (0, "Succeeded")
        except asyncio.CancelledError:
            cancelled["leader"] = True
            raise

    async def worker_done():
        return (0, "Succeeded")

    async def idle():
        try:
            await asyncio.Event().wait()
            return (0, "Succeeded")
        except asyncio.CancelledError:
            cancelled["w2"] = True
            raise

    async def drive():
        return await asyncio.wait_for(
            life.gather_pods_fail_fast(
                [leader(), worker_done(), idle()], mpi_world_group=True
            ),
            timeout=2,
        )

    res = asyncio.run(drive())
    assert res[1] == (0, "Succeeded")  # the worker that terminated
    assert res[0] is None and res[2] is None  # leader + other idle peer cancelled
    assert cancelled["leader"] and cancelled["w2"]


def test_gather_pods_fail_fast_default_awaits_all_even_when_leader_succeeds():
    # Default (mpi_world_group=False, non-MPI multi-node): a run-to-completion
    # task must still await ALL pods -- a leader succeeding while a peer is still
    # running does NOT resolve the task early. Proven by the idle peer forcing a
    # timeout (unchanged behaviour).
    async def leader():
        return (0, "Succeeded")

    async def idle():
        await asyncio.Event().wait()
        return (0, "Succeeded")

    async def drive():
        return await asyncio.wait_for(
            life.gather_pods_fail_fast([leader(), idle()]), timeout=0.5
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(drive())


def test_task_exit_code_any_failed_pod_fails_task():
    # Leader succeeded but a worker failed -> the task fails (not leader-only).
    assert life.task_exit_code([(0, "Succeeded"), (7, "Failed")]) == 7
    # Failed phase with a 0/unknown numeric code still fails (-> 1).
    assert life.task_exit_code([(0, "Failed"), None]) == 1
    # Leader failed, peer cancelled (None) -> the leader's non-zero code.
    assert life.task_exit_code([(1, "Failed"), None]) == 1


def test_task_exit_code_all_ok_uses_leader():
    assert life.task_exit_code([(0, "Succeeded"), (0, "Succeeded")]) == 0
    assert life.task_exit_code([]) == 0


# ---------------------------------------------------------------------------
# format_pod_start_note + status-note poller (live sub-status while starting)
# ---------------------------------------------------------------------------


def _fake_kubectl(phase, *, waiting="", sched=""):
    async def fake(args, *, global_args=(), timeout=None):
        j = " ".join(str(a) for a in args)
        if "status.phase" in j:
            return (0, phase, "")
        if "waiting.reason" in j:
            return (0, waiting, "")
        if "PodScheduled" in j:
            return (0, sched, "")
        return (0, "", "")

    return fake


def test_format_pod_start_note_reports_unschedulable(monkeypatch):
    monkeypatch.setattr(
        life, "run_kubectl", _fake_kubectl("Pending", sched="Unschedulable")
    )
    note = asyncio.run(life.format_pod_start_note("pod/x", global_args=[], ns_args=[]))
    assert note == "Pending: Unschedulable"


def test_format_pod_start_note_prefers_container_waiting_reason(monkeypatch):
    monkeypatch.setattr(
        life, "run_kubectl", _fake_kubectl("Pending", waiting="ImagePullBackOff")
    )
    note = asyncio.run(life.format_pod_start_note("pod/x", global_args=[], ns_args=[]))
    assert note == "Pending: ImagePullBackOff"


def test_format_pod_start_note_none_when_running(monkeypatch):
    monkeypatch.setattr(life, "run_kubectl", _fake_kubectl("Running"))
    assert (
        asyncio.run(life.format_pod_start_note("pod/x", global_args=[], ns_args=[]))
        is None
    )


def test_format_pod_start_note_none_when_gone(monkeypatch):
    monkeypatch.setattr(life, "run_kubectl", _fake_kubectl(""))
    assert (
        asyncio.run(life.format_pod_start_note("pod/x", global_args=[], ns_args=[]))
        is None
    )


def test_format_pod_start_note_phase_only_without_reason(monkeypatch):
    monkeypatch.setattr(life, "run_kubectl", _fake_kubectl("Pending"))
    assert (
        asyncio.run(life.format_pod_start_note("pod/x", global_args=[], ns_args=[]))
        == "Pending"
    )


def test_status_note_for_pods_labels_each_pod_when_multiple(monkeypatch):
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend(nodes=2, gpus_per_node=8),
        assigned_nodes=["node-0", "node-1"], artifacts=[], gpu_count=16,
    )

    async def fake_note(pod_ref, *, global_args, ns_args):
        return "Pending" if pod_ref.endswith("-0") else None

    monkeypatch.setattr(life, "format_pod_start_note", fake_note)
    note = asyncio.run(
        op._status_note_for_pods(["pod/t-0", "pod/t-1"], global_args=[], ns_args=[])
    )
    assert note == "t-0: Pending"


def test_status_note_for_pods_none_when_all_running(monkeypatch):
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend(), assigned_nodes=["node-0"], artifacts=[], gpu_count=2
    )

    async def fake_note(pod_ref, *, global_args, ns_args):
        return None

    monkeypatch.setattr(life, "format_pod_start_note", fake_note)
    assert (
        asyncio.run(op._status_note_for_pods(["pod/t"], global_args=[], ns_args=[]))
        is None
    )


def test_poll_status_note_reports_until_cancelled(monkeypatch):
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend(), assigned_nodes=["node-0"], artifacts=[], gpu_count=2
    )

    async def fake_note(pod_ref, *, global_args, ns_args):
        return "Pending: Unschedulable"

    monkeypatch.setattr(life, "format_pod_start_note", fake_note)
    notes: list = []

    async def drive():
        poller = asyncio.ensure_future(
            op._poll_status_note(
                ["pod/x"], global_args=[], ns_args=[],
                status_note=notes.append, interval=0,
            )
        )
        for _ in range(3):
            await asyncio.sleep(0)
        poller.cancel()
        await asyncio.gather(poller, return_exceptions=True)

    asyncio.run(drive())
    assert "Pending: Unschedulable" in notes


def test_execute_clears_status_note_after_startup(monkeypatch, tmp_path):
    # When a status_note callback is provided, execute stops annotating the sub-
    # status once the startup wait is over (pod running) -> the note is cleared.
    async def _noop_stream(log_command, dest_path):
        return _FakeProc()

    async def _noop_tail(path, *, task_name):
        await asyncio.sleep(3600)

    async def _watch(pod_ref, **kw):
        return "Succeeded"

    async def _term(p, *, kill_group=False):
        pass

    def _finalize(*a, **k):
        pass

    async def _exit(pod_ref, *, phase="", **kw):
        return 0

    async def _delete(refs, **kw):
        pass

    monkeypatch.setattr(life, "start_pod_log_file_stream", _noop_stream)
    monkeypatch.setattr(life, "tail_file_to_console", _noop_tail)
    monkeypatch.setattr(life, "watch_until_terminal", _watch)
    monkeypatch.setattr(life, "terminate_process", _term)
    monkeypatch.setattr(life, "sanitize_streamed_logs", _finalize)
    monkeypatch.setattr(life, "pod_exit_code", _exit)
    monkeypatch.setattr(life, "delete_objects", _delete)

    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend(), assigned_nodes=["node-0"],
                             artifacts=[], gpu_count=2)
    notes: list = []
    rc = asyncio.run(
        op.execute(
            launcher=_FakeLauncher(), output_logger=None,
            env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
            task_name="decode_server_0", script=["run"],
            status_note=notes.append,
        )
    )
    assert rc == 0
    assert notes and notes[-1] is None


# ---------------------------------------------------------------------------
# K8sContainerOperator.execute orchestration (launcher + kubectl faked)
# ---------------------------------------------------------------------------


class _FakeLauncher:
    """Records run_async calls. Only the apply (bash) goes through the launcher now;
    the pod log is offloaded to a file (start_pod_log_file_stream), not streamed here."""

    def __init__(self, apply_rc: int = 0):
        self.calls: list[list[str]] = []
        self._apply_rc = apply_rc

    async def run_async(self, command, *, output_logger=None, env=None, task_name=None):
        self.calls.append(command.as_list())
        return self._apply_rc


class _FakeProc:
    """Stand-in for the offloaded `kubectl logs -f` file-stream subprocess."""

    def __init__(self):
        self.returncode = None

    def terminate(self):
        self.returncode = -15

    async def wait(self):
        return self.returncode


def test_execute_offloads_stream_and_stops_on_terminal(monkeypatch, tmp_path):
    events: list = []
    proc = _FakeProc()

    async def fake_start_stream(log_command, dest_path):
        events.append(("stream", dest_path))
        return proc

    async def fake_tail(path, *, task_name):
        events.append(("tail", path))
        await asyncio.sleep(3600)  # decoupled tailer; runs until cancelled

    async def fake_watch(pod_ref, **kw):
        await asyncio.sleep(0)
        events.append(("watch", pod_ref))
        return "Succeeded"

    async def fake_terminate(p, *, kill_group=False):
        events.append(("terminate",))

    def fake_sanitize(paths):
        events.append(("sanitize", tuple(paths)))

    async def fake_exit(pod_ref, *, phase="", **kw):
        events.append(("exit", phase))
        return 0

    async def fake_delete(refs, **kw):
        events.append(("delete",))

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "tail_file_to_console", fake_tail)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "sanitize_streamed_logs", fake_sanitize)
    monkeypatch.setattr(life, "pod_exit_code", fake_exit)
    monkeypatch.setattr(life, "delete_objects", fake_delete)

    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend(), assigned_nodes=["node-0"],
                             artifacts=[], gpu_count=2)
    launcher = _FakeLauncher()
    rc = asyncio.run(
        op.execute(
            launcher=launcher, output_logger=None,
            env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
            task_name="decode_server_0", script=["run"],
        )
    )
    assert rc == 0
    # Apply ran via the launcher (bash); the pod log was NOT streamed through it.
    assert launcher.calls == [launcher.calls[0]] and launcher.calls[0][0] == "bash"
    task_log = str(tmp_path / "decode_server_0.log")
    # The pod log is offloaded straight to <task>.log and a decoupled console
    # tailer reads that same file.
    assert ("stream", task_log) in events
    assert ("tail", task_log) in events
    # BEHAVIOUR CHANGE: the follow is now DRAINED on a terminal pod instead of being
    # killed mid-stream, so <task>.log is already complete and the one-shot re-fetch is
    # skipped -- only the TTY-sanitize pass the rebuild used to provide still runs. The
    # guarantee under test is unchanged: the on-disk log is made final exactly once,
    # after the watch and the exit-code read.
    assert ("sanitize", (task_log,)) in events
    kinds = [e[0] for e in events]
    assert (
        kinds.index("watch")
        < kinds.index("terminate")
        < kinds.index("exit")
        < kinds.index("sanitize")
    )
    assert kinds.count("sanitize") == 1
    assert kinds[-1] == "delete"


def test_execute_finalizes_log_before_delete_on_cancel(monkeypatch, tmp_path):
    # On SIGINT/teardown execute() is cancelled at the pod-status watch; it must still
    # finalize the streamed log BEFORE deleting the pods, else a fast-failing
    # launcher's log is lost. Simulate an idle pod, cancel mid-run, and assert finalize
    # ran before delete_objects.
    order: list[str] = []
    at_watch = asyncio.Event()

    async def fake_start_stream(log_command, dest_path):
        return _FakeProc()

    async def fake_tail(path, *, task_name):
        await asyncio.sleep(3600)

    async def fake_watch(pod_ref, **kw):
        at_watch.set()
        await asyncio.sleep(3600)  # idle pod: never terminal on its own
        return "Succeeded"

    async def fake_terminate(p, *, kill_group=False):
        pass

    def fake_finalize(paths):
        order.append("finalize")

    async def fake_exit(pod_ref, *, phase="", **kw):
        return 0

    async def fake_delete(refs, **kw):
        order.append("delete")

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "tail_file_to_console", fake_tail)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "sanitize_streamed_logs", fake_finalize)
    monkeypatch.setattr(life, "pod_exit_code", fake_exit)
    monkeypatch.setattr(life, "delete_objects", fake_delete)

    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend(), assigned_nodes=["node-0"],
                             artifacts=[], gpu_count=2)

    async def drive():
        t = asyncio.ensure_future(
            op.execute(
                launcher=_FakeLauncher(), output_logger=None,
                env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
                task_name="decode_server_0", script=["run"],
            )
        )
        await asyncio.wait_for(at_watch.wait(), timeout=2)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        return t

    asyncio.run(drive())
    assert order == ["finalize", "delete"]  # log saved BEFORE pods deleted


def test_execute_multinode_fails_fast_when_a_pod_dies(monkeypatch, tmp_path):
    # Regression: a multi-node task is 2 pods (leader + worker on `sleep 3600`).
    # When the leader pod dies, execute() must fail the whole task at once instead
    # of blocking on the still-running worker (which pre-fix hung until SIGINT).
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend(nodes=2, gpus_per_node=8),
                             assigned_nodes=["node-0", "node-1"], artifacts=[],
                             gpu_count=16)
    plan = op._build_execution_plan(
        task_name="decode_server_0", script=["run"],
        envs={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
    )
    assert len(plan.pod_refs) == 2  # one pod per node
    # Non-MPI multi-node is run-to-completion: NOT an MPI world group (await all pods).
    assert plan.mpi_world_group is False
    leader_ref = plan.pod_refs[0]
    worker_cancelled = {"v": False}

    async def fake_start_stream(log_command, dest_path):
        return _FakeProc()

    async def fake_tail(path, *, task_name):
        await asyncio.sleep(3600)

    async def fake_watch(pod_ref, **kw):
        if pod_ref == leader_ref:
            return "Failed"  # leader's engine init failed
        try:
            await asyncio.sleep(3600)  # worker idles on `sleep 3600`
            return "Succeeded"
        except asyncio.CancelledError:
            worker_cancelled["v"] = True
            raise

    async def fake_terminate(p, *, kill_group=False):
        pass

    async def fake_exit(pod_ref, *, phase="", **kw):
        return 1 if phase == "Failed" else 0

    def fake_finalize(*a, **k):
        pass

    async def fake_delete(refs, **kw):
        pass

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "tail_file_to_console", fake_tail)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "pod_exit_code", fake_exit)
    monkeypatch.setattr(life, "sanitize_streamed_logs", fake_finalize)
    monkeypatch.setattr(life, "delete_objects", fake_delete)

    launcher = _FakeLauncher()

    async def drive():
        # wait_for is the regression guard: pre-fix this blocked on the worker.
        return await asyncio.wait_for(
            op.execute(launcher=launcher, output_logger=None,
                       env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
                       task_name="decode_server_0", script=["run"]),
            timeout=2,
        )

    rc = asyncio.run(drive())
    assert rc == 1  # the dead leader fails the whole task ...
    assert worker_cancelled["v"] is True  # ... and the idle worker was cancelled


def test_execute_finalizes_cancelled_peer_log_before_delete(monkeypatch, tmp_path):
    # When fail-fast / world-group resolution cancels a peer, that peer is still
    # Running, so its follow is CUT rather than drained (stop_log_stream terminal=False)
    # and its <task>.log simply ends where the stream got to. That is the whole
    # guarantee now, and it only holds if the finalize runs BEFORE the `finally`
    # deletes the pods -- otherwise a fast-failing peer's output is gone with the pod.
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend(nodes=2, gpus_per_node=8),
                             assigned_nodes=["node-0", "node-1"], artifacts=[],
                             gpu_count=16)
    plan = op._build_execution_plan(
        task_name="decode_server_0", script=["run"],
        envs={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
    )
    leader_ref = plan.pod_refs[0]
    order: list[str] = []
    seen: dict = {}
    stopped: list[bool] = []

    async def fake_start_stream(log_command, dest_path):
        return _FakeProc()

    async def fake_tail(path, *, task_name):
        await asyncio.sleep(3600)

    async def fake_stop_stream(proc, *, terminal, kill_group=False):
        stopped.append(terminal)
        return terminal

    async def fake_watch(pod_ref, **kw):
        if pod_ref == leader_ref:
            return "Failed"
        await asyncio.sleep(3600)  # peer idles -> cancelled by fail-fast
        return "Succeeded"

    async def fake_terminate(p, *, kill_group=False):
        pass

    async def fake_exit(pod_ref, *, phase="", **kw):
        return 1 if phase == "Failed" else 0

    def fake_finalize(paths):
        order.append("finalize")
        seen["paths"] = tuple(paths)

    async def fake_delete(refs, **kw):
        order.append("delete")

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "tail_file_to_console", fake_tail)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "stop_log_stream", fake_stop_stream)
    monkeypatch.setattr(life, "pod_exit_code", fake_exit)
    monkeypatch.setattr(life, "sanitize_streamed_logs", fake_finalize)
    monkeypatch.setattr(life, "delete_objects", fake_delete)

    async def drive():
        return await asyncio.wait_for(
            op.execute(launcher=_FakeLauncher(), output_logger=None,
                       env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
                       task_name="decode_server_0", script=["run"]),
            timeout=2,
        )

    asyncio.run(drive())
    # BEHAVIOUR CHANGE: there is no re-fetch (and so no `force`/`phases`) any more --
    # a cancelled peer's <task>.log is the file its follow already streamed. Pin what
    # replaced it, not merely that SOMETHING was finalized:
    #   * the task's own log path is the one handed over (a multi-pod task streams
    #     every pod into that single file, so finalizing it covers the cancelled peer);
    #   * it happens BEFORE the delete, which is what actually saves the output.
    assert seen["paths"] == (plan.task_log_path,), (
        "the cancelled peer's log is the shared streamed <task>.log; that exact path "
        f"must be finalized, got {seen.get('paths')!r}"
    )
    assert order == ["finalize", "delete"], (
        "finalize must precede the pod delete, or a fast-failing peer's streamed log "
        "dies with the pod"
    )
    # And each pod's stream was ended the right way for its state: the Failed leader
    # DRAINED (its follow will hit EOF), the still-Running peer CUT (its never would).
    assert True in stopped, "the terminal leader's follow must be drained"
    assert False in stopped, (
        "a peer cancelled while still Running must be cut (terminal=False): its follow "
        f"would never reach EOF, got {stopped!r}"
    )


def test_run_pod_stream_status_authoritative_interrupts_stream(monkeypatch, tmp_path):
    # The pod STATUS (watch), not the log stream, decides completion; the offloaded
    # stream is a side channel interrupted on terminal. A still-running pod keeps
    # the task alive (so long-lived READY services aren't ended by a stream blip).
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend(), assigned_nodes=["node-0"],
                             artifacts=[], gpu_count=0)
    plan = op._build_execution_plan(
        task_name="t", script=["run"], envs={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)}
    )
    release = asyncio.Event()
    proc = _FakeProc()
    terminated: list = []

    async def fake_start_stream(log_command, dest_path):
        return proc

    async def fake_watch(pod_ref, **kw):
        await release.wait()  # pod stays "running" until released
        return "Failed"

    async def fake_terminate(p, *, kill_group=False):
        terminated.append(True)

    async def fake_exit(pod_ref, *, phase="", **kw):
        return 9 if phase == "Failed" else 0

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "pod_exit_code", fake_exit)

    async def drive():
        t = asyncio.ensure_future(op._run_pod_stream(plan=plan, index=0))
        await asyncio.sleep(0.05)
        # Pod still running -> task NOT done, and the stream is NOT interrupted yet.
        assert not t.done()
        assert terminated == []
        release.set()  # pod goes terminal
        return await t

    rc, phase = asyncio.run(drive())
    assert rc == 9  # exit code from the watch's phase (Failed), not the stream
    assert phase == "Failed"  # phase returned to execute() to drive drain-vs-cut
    assert terminated == [True]  # stream interrupted once the pod was terminal


def test_execute_returns_apply_failure_without_streaming(monkeypatch, tmp_path):
    started: list = []

    async def fake_start_stream(log_command, dest_path):
        started.append(True)
        return _FakeProc()

    async def fake_delete(refs, **kw):
        return None

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "delete_objects", fake_delete)

    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend(), assigned_nodes=["node-0"],
                             artifacts=[], gpu_count=2)
    launcher = _FakeLauncher(apply_rc=2)  # apply fails
    rc = asyncio.run(
        op.execute(launcher=launcher, output_logger=None,
                   env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
                   task_name="t", script=["run"])
    )
    assert rc == 2
    assert len(launcher.calls) == 1  # only the apply ran
    assert started == []  # apply failure short-circuits before any log stream


# ---------------------------------------------------------------------------
# offload helpers: file-stream redirect + decoupled console tailer
# ---------------------------------------------------------------------------


def test_start_pod_log_file_stream_redirects_kubectl_to_file(monkeypatch, tmp_path):
    from sflow.plugins.k8s.shell import build_log_stream_command

    captured: dict = {}

    class _P:
        returncode = None

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        return _P()

    monkeypatch.setattr(life.asyncio, "create_subprocess_exec", fake_exec)
    cmd = build_log_stream_command("pod/t", ns_args=["--namespace", "ns"])
    dest = str(tmp_path / "t.log")
    asyncio.run(life.start_pod_log_file_stream(cmd, dest))
    assert captured["args"][0] == "bash" and captured["args"][1] == "-c"
    shell = captured["args"][2]
    assert "exec " in shell  # exec so terminating the bash stops kubectl
    assert "logs -f pod/t" in shell
    assert f">> {dest}" in shell  # redirected straight to <task>.log (append)


def test_tail_file_to_console_streams_appended_lines(monkeypatch, tmp_path):
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "t.log"
    path.write_text("preexisting-before-tailer\n")  # written before -> skipped

    captured: list = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    cap = _Cap()
    life._logger.addHandler(cap)
    old_level = life._logger.level
    life._logger.setLevel(logging.INFO)
    try:
        async def drive():
            t = asyncio.ensure_future(
                life.tail_file_to_console(str(path), task_name="mytask")
            )
            await asyncio.sleep(0.05)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("new-line-1\nnew-line-2\n")
            await asyncio.sleep(0.5)  # let the tailer poll (interval 0.3s)
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)

        asyncio.run(drive())
    finally:
        life._logger.removeHandler(cap)
        life._logger.setLevel(old_level)

    joined = "\n".join(captured)
    assert "preexisting-before-tailer" not in joined  # tailer starts at EOF
    # Lines are echoed as-is (kubectl's [pod/...] prefix already identifies the
    # source); the tailer must NOT add sflow's [task] console prefix.
    assert "new-line-1" in joined
    assert "new-line-2" in joined
    assert "[mytask]" not in joined


def _tail_once(path, *, task_name="mlperf_harness", append="", append_bytes=None):
    """Run the tailer over ``append``ed content and return the console messages.

    ``append_bytes`` writes the exact bytes instead, bypassing the platform's newline
    translation -- required when the test is ABOUT line terminators (text mode turns
    ``\\n`` into ``\\r\\n`` on Windows, which is a different input than intended).
    """
    captured: list = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    cap = _Cap()
    life._logger.addHandler(cap)
    old_level = life._logger.level
    life._logger.setLevel(logging.INFO)
    try:
        async def drive():
            t = asyncio.ensure_future(
                life.tail_file_to_console(str(path), task_name=task_name)
            )
            await asyncio.sleep(0.05)
            if append_bytes is not None:
                with open(path, "ab") as fh:
                    fh.write(append_bytes)
            else:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(append)
            # Wait for OUTPUT, not for wall-clock. A fixed sleep tuned to the 0.3s poll
            # leaves ~0.2s of slack, so these tests fail on a loaded CI box for reasons
            # that have nothing to do with the tailer. Poll for the first message, then
            # allow one more tick to drain the rest, and cap the whole thing.
            deadline = asyncio.get_event_loop().time() + 5.0
            while not captured and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.05)
            await asyncio.sleep(life._TAIL_POLL_INTERVAL + 0.2)
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)

        asyncio.run(drive())
    finally:
        life._logger.removeHandler(cap)
        life._logger.setLevel(old_level)
    return captured


def test_tail_collapses_a_progress_bar_burst_instead_of_rendering_all_of_it(
    monkeypatch, tmp_path
):
    """THE post-run hang: one unbounded line must never reach the console renderer.

    `kubectl logs -f` reads to `\\n`, so it withholds an unterminated `\\r` progress bar
    for as long as the bar runs and then delivers an HOUR of it as a SINGLE line. That
    line walks straight past the per-tick LINE COUNT cap (it is one line), and rendering
    it through rich costs ~6.3us and ~300 bytes of RSS per character -- measured 25s and
    1.15 GB for 4 MB, so tens of MB is CPU-minutes and many GB, i.e. swap. It happens on
    the event-loop thread, which is why the whole workflow froze for ~20 minutes with
    py-spy showing 100% CPU under tail_file_to_console.

    Only the final frame was ever visible on a terminal anyway, which is exactly what
    the launcher echoes for every other backend.
    """
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "mlperf_harness.log"
    path.write_text("")
    pad = "." * 120
    bar = "".join(f"\rProcessing: {i}/20000 [{pad}]" for i in range(1, 20001))
    assert len(bar) > 2_000_000, "the burst must be big enough to be the real input"

    captured = _tail_once(path, append=bar + "\n")

    joined = "\n".join(captured)
    assert joined, "the burst must still be reported, just bounded"
    assert len(joined) < 10 * life._TAIL_MAX_LINE_CHARS, (
        f"a {len(bar)}-char burst reached the console as {len(joined)} chars"
    )
    # The last frame is the one a terminal would have shown.
    assert "Processing: 20000/20000" in joined
    assert "Processing: 19999/20000" not in joined, "earlier frames were overwritten"


def test_tail_truncates_a_long_line_that_has_no_redraws_to_collapse(
    monkeypatch, tmp_path
):
    """A giant line with no `\\r` (a JSON / base64 dump) costs the same to render.

    Collapsing redraws cannot help here, so the length cap is what bounds it -- and it
    must say so on the console, naming where the full line survives.
    """
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "mlperf_harness.log"
    path.write_text("")
    blob = "A" * 500_000

    captured = _tail_once(path, append=blob + "\n")

    joined = "\n".join(captured)
    assert len(joined) < 10 * life._TAIL_MAX_LINE_CHARS, len(joined)
    assert "line truncated for the console" in joined, joined[:200]
    assert "500000 chars" in joined, "say how much was elided"
    assert "mlperf_harness.log" in joined, "point at where the full line is"


def test_tail_does_not_accumulate_an_unterminated_bar_across_ticks(
    monkeypatch, tmp_path
):
    """The retained partial line must not grow without bound while the bar runs.

    A bar that has not finished yet has no newline, so every tick would otherwise
    append to (and re-copy) an ever-larger buffer for the whole hour it runs.
    """
    import tracemalloc

    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "t.log"
    path.write_text("")
    pad = "." * 120
    # ~4 MB of bar delivered over several polls, with no newline to end it.
    tick = "".join(f"\rstep {j} [{pad}]" for j in range(8000))

    async def drive():
        t = asyncio.ensure_future(life.tail_file_to_console(str(path), task_name="t"))
        for _ in range(4):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(tick)
            await asyncio.sleep(0.4)
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)

    tracemalloc.start()
    asyncio.run(drive())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory must track ONE poll, not the length of the bar. Measured: 3.0x a single
    # poll's bytes when the retained partial is collapsed, 8.0x when it accumulates
    # (and that multiple keeps climbing with the bar, which is the leak).
    assert peak < 5 * len(tick), (
        f"the tailer peaked at {peak} bytes = {peak / len(tick):.1f}x one poll's "
        f"{len(tick)} bytes; the retained partial line must collapse each tick "
        "rather than accumulate for as long as the bar runs"
    )


# The `\r`-collapse rules themselves now live with the shared helper they moved to:
# tests/unit/test_utils_console_text.py. What stays here is that the TAILER applies them,
# which is the part a k8s regression would break.


def test_tail_flushes_an_unterminated_line_when_cancelled(monkeypatch, tmp_path):
    """A bar that never printed its newline must still show its final frame.

    The tailer only emits COMPLETE lines, so without a flush the last thing the task
    displayed -- the whole point of a progress bar -- is the one thing the console never
    sees. The pod is terminal by then, so there is no later tick to catch it.
    """
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "t.log"
    path.write_bytes(b"")
    pad = b"." * 120
    bar = b"".join(b"\rProcessing: %d/300 [%s]" % (i, pad) for i in range(1, 301))

    messages = _tail_once(path, append_bytes=bar)  # NO trailing newline

    joined = "\n".join(messages)
    assert "Processing: 300/300" in joined, (
        "the final frame of an unterminated bar must be flushed on cancellation"
    )


def test_tail_echoes_crlf_terminated_lines(monkeypatch, tmp_path):
    """A CRLF-emitting task must still appear on the console.

    Byte-exact input: splitting on ``\\n`` leaves a trailing ``\\r`` on every line, which a
    last-frame-outright collapse turns into the empty string. The whole task then streams
    NOTHING to the console while <task>.log fills up normally -- a silent blackout that
    looks like a dead task.
    """
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "t.log"
    path.write_bytes(b"")

    messages = _tail_once(
        path, append_bytes=b"loading model\r\nserving on :8000\r\n"
    )

    joined = "\n".join(messages)
    assert "loading model" in joined
    assert "serving on :8000" in joined


def test_tail_echoes_a_bar_whose_frames_end_with_cr(monkeypatch, tmp_path):
    """A bar that ends each frame with ``\\r`` must still show its final frame."""
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "t.log"
    path.write_bytes(b"")
    pad = b"." * 120
    bar = b"".join(b"Processing: %d/500 [%s]\r" % (i, pad) for i in range(1, 501))

    messages = _tail_once(path, append_bytes=bar + b"\n")

    joined = "\n".join(messages)
    assert "Processing: 500/500" in joined, "the final frame must survive"
    assert "Processing: 499/500" not in joined, "earlier frames were overwritten"


def test_tail_caps_total_chars_per_tick(monkeypatch, tmp_path):
    """Per-line and per-count caps still multiply; the tick TOTAL must be bounded too.

    400 lines x 2000 chars is 800 KB in one synchronous render -- ~5s of blocked event
    loop at the measured cost, repeating for as long as the task stays chatty. Each line
    here is individually well under the line cap and the count is under the count cap,
    so only the tick budget can catch this.
    """
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "t.log"
    path.write_text("")
    line = "y" * 1000

    messages = _tail_once(path, append="".join(f"{line}\n" for _ in range(300)))

    payload = [m for m in messages if m.startswith("y")]
    assert payload, "the lines must still be surfaced, just bounded"
    assert sum(len(m) for m in payload) <= life._TAIL_MAX_CHARS_PER_TICK, (
        f"{sum(len(m) for m in payload)} chars rendered in one tick exceeds the "
        f"{life._TAIL_MAX_CHARS_PER_TICK}-char budget"
    )
    assert any("console lines omitted" in m for m in messages), (
        "what the budget dropped must be reported, not silently swallowed"
    )


def test_tail_file_to_console_noop_when_not_a_tty(monkeypatch, tmp_path):
    monkeypatch.setattr(life, "_console_active", lambda: False)
    path = tmp_path / "t.log"
    path.write_text("line\n")
    captured: list = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    cap = _Cap()
    life._logger.addHandler(cap)
    try:
        # Returns immediately (no console to stream to) without emitting anything.
        asyncio.run(
            asyncio.wait_for(life.tail_file_to_console(str(path), task_name="t"), 1)
        )
    finally:
        life._logger.removeHandler(cap)
    assert captured == []


# ---------------------------------------------------------------------------
# completeness: the streamed <task>.log is finalized in place (no re-fetch)
# ---------------------------------------------------------------------------


def test_run_pod_stream_cancel_interrupts_stream_without_finalize(monkeypatch, tmp_path):
    # A long-lived READY service cancelled at workflow end: the stream is
    # CUT (not drained) because final_phase is "" -- its follow would never end.
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend(), assigned_nodes=["node-0"],
                             artifacts=[], gpu_count=0)
    plan = op._build_execution_plan(
        task_name="srv", script=["run"], envs={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)}
    )
    proc = _FakeProc()
    started = asyncio.Event()
    terminated: list = []
    finalized: list = []

    async def fake_start_stream(log_command, dest_path):
        return proc

    async def fake_watch(pod_ref, **kw):
        started.set()
        await asyncio.sleep(3600)  # never terminal (long-lived service)

    async def fake_terminate(p, *, kill_group=False):
        terminated.append(True)

    async def fake_finalize(*a, **k):
        finalized.append(True)

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "sanitize_streamed_logs", fake_finalize)

    async def drive():
        t = asyncio.ensure_future(op._run_pod_stream(plan=plan, index=0))
        await started.wait()
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(drive())
    assert terminated == [True]  # stream interrupted on teardown
    assert finalized == []  # nothing is finalized for a cancelled service


# ---------------------------------------------------------------------------
# Merge-pod log demux: OFFLOADED to a child process (python demuxer), so the sflow
# driver's event loop is never in the per-line path (mirrors the single-pod redirect).
# ---------------------------------------------------------------------------


def _merge_member(name, out_dir, *, cvd="0"):
    return SimpleNamespace(
        name=name,
        base_name="decode_server",
        script=["echo hi"],
        envs={"SFLOW_TASK_OUTPUT_DIR": str(out_dir)},
        merge_cuda_visible_devices=cvd,
    )


def _merge_op_and_plan(tmp_path, names):
    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(
        backend=_backend(gpus_per_node=8), assigned_nodes=["node-0"],
        artifacts=[], gpu_count=4,
    )
    op.apply_merge_group(
        members=[_merge_member(n, tmp_path / n) for n in names], union_gpus=4
    )
    plan = op._build_execution_plan(
        task_name=names[0], script=["echo hi"],
        envs={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path / names[0])},
    )
    return op, plan


def test_demux_command_routes_tagged_untagged_and_unknown(tmp_path, fake_process):
    # The offloaded demux is a real streaming child process; run it end-to-end to
    # prove it routes each [[sflow-mux:<task>]] line to that member's file (tag
    # stripped) and everything else (untagged + unknown-member tags) verbatim to the
    # leader log -- identical routing to the removed awk program, but prompt for a
    # quiet member (mawk block-buffered its stdin; see plugins/k8s/log_demux.py).
    fake_process.allow_unregistered(True)  # run the real python demuxer
    decode = tmp_path / "decode.log"
    prefill = tmp_path / "prefill.log"
    leader = tmp_path / "leader.log"
    cmd = life._demux_command(
        {"decode": str(decode), "prefill": str(prefill)}, str(leader)
    )
    stream = (
        "[[sflow-mux:decode]] hello from decode\n"
        "[[sflow-mux:prefill]] hello from prefill\n"
        "untagged apply diagnostics\n"
        "[[sflow-mux:unknown]] no such member\n"
        "[[sflow-mux:decode]] second decode line\n"
    )
    subprocess.run(cmd, input=stream, text=True, check=True)
    assert decode.read_text() == "hello from decode\nsecond decode line\n"
    assert prefill.read_text() == "hello from prefill\n"
    leader_text = leader.read_text()
    assert "untagged apply diagnostics\n" in leader_text
    # An unknown member tag is not routed -> kept verbatim in the leader log.
    assert "[[sflow-mux:unknown]] no such member\n" in leader_text


def test_demux_command_appends_preserving_leader_prefix(tmp_path, fake_process):
    # The demuxer opens files in append mode, so the leader log's pre-existing
    # apply-diagnostics prefix is preserved (not truncated) when pod-level lines land.
    fake_process.allow_unregistered(True)  # run the real python demuxer
    leader = tmp_path / "leader.log"
    leader.write_text("apply-prefix-line\n")
    decode = tmp_path / "decode.log"
    cmd = life._demux_command({"decode": str(decode)}, str(leader))
    subprocess.run(
        cmd, text=True, check=True,
        input="pod-level line\n[[sflow-mux:decode]] d\n",
    )
    assert leader.read_text() == "apply-prefix-line\npod-level line\n"
    assert decode.read_text() == "d\n"


def test_start_pod_log_demux_stream_offloads_pipeline(tmp_path, monkeypatch):
    from sflow.plugins.k8s.shell import build_log_stream_command

    captured: dict = {}

    class _P:
        returncode = None

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _P()

    monkeypatch.setattr(life.asyncio, "create_subprocess_exec", fake_exec)
    decode = tmp_path / "d" / "decode.log"  # nested -> its dir must be created
    leader = tmp_path / "leader.log"
    cmd = build_log_stream_command("pod/m", ns_args=["--namespace", "ns"], prefix=False)
    asyncio.run(
        life.start_pod_log_demux_stream(
            cmd, tag_paths={"decode": str(decode)}, default_path=str(leader)
        )
    )
    assert captured["args"][0] == "bash" and captured["args"][1] == "-c"
    shell = captured["args"][2]
    # kubectl follows into the streaming python demuxer, all in the child shell
    # (offloaded) -- the driver never reads the stream line by line.
    assert "logs -f pod/m" in shell
    assert "-m sflow.plugins.k8s.log_demux" in shell
    assert " | " in shell  # kubectl piped into the demuxer
    assert f"--default {leader}" in shell
    assert f"--route decode={decode}" in shell
    # New session so the whole kubectl|demuxer group can be signalled on teardown.
    assert captured["kwargs"].get("start_new_session") is True
    assert (tmp_path / "d").is_dir()  # best-effort mkdir of the member log dir


def test_start_pod_log_demux_stream_splits_live_stream_end_to_end(tmp_path, fake_process):
    # Run the REAL offloaded pipeline (bash -c "<follow> | python -m ...log_demux")
    # with a fake follow that emits interleaved tagged lines for two members plus a
    # pod-level line and an unknown-member tag, then exits. Asserts each member's
    # <task>.log is split correctly and the leftovers land in the leader log -- the
    # end-to-end shell-pipe + demuxer wiring the mawk demux got wrong ("3 update, 1
    # stale"). Complements the in-process quiet-member flush test in
    # test_plugin_k8s_log_demux.py by exercising the actual subprocess pipeline.
    fake_process.allow_unregistered(True)  # real bash + python demuxer
    from sflow.core.command import Command

    decode = tmp_path / "decode_server_0" / "decode_server_0.log"
    prefill = tmp_path / "prefill_server_0" / "prefill_server_0.log"
    leader = tmp_path / "leader.log"
    # Stand-in for `kubectl logs -f <merged pod>`: emit interleaved tagged lines then
    # exit (EOF). printf '%s\n' prints one arg per line.
    producer = (
        "printf '%s\\n' "
        "'[[sflow-mux:decode_server_0]] decode line 1' "
        "'pod-level diagnostic' "
        "'[[sflow-mux:prefill_server_0]] prefill line 1' "
        "'[[sflow-mux:other]] unknown-member line' "
        "'[[sflow-mux:decode_server_0]] decode line 2'"
    )
    cmd = Command(exec="bash")
    cmd.add_arg("-c")
    cmd.add_arg(producer)

    async def _drive():
        proc = await life.start_pod_log_demux_stream(
            cmd,
            tag_paths={
                "decode_server_0": str(decode),
                "prefill_server_0": str(prefill),
            },
            default_path=str(leader),
        )
        await proc.wait()

    asyncio.run(_drive())
    # Each member gets ONLY its own lines, tag stripped, in order.
    assert decode.read_text() == "decode line 1\ndecode line 2\n"
    assert prefill.read_text() == "prefill line 1\n"
    # Untagged + unknown-member lines land verbatim in the leader/default log.
    leader_text = leader.read_text()
    assert "pod-level diagnostic\n" in leader_text
    assert "[[sflow-mux:other]] unknown-member line\n" in leader_text


def test_terminate_process_kill_group_signals_the_group(monkeypatch):
    signals: list = []

    class _P:
        pid = 4321
        returncode = None

        async def wait(self):
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(life.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        life.os, "killpg", lambda pgid, sig: signals.append((pgid, sig))
    )
    asyncio.run(life.terminate_process(_P(), kill_group=True))
    assert signals == [(4321, life.signal.SIGTERM)]  # whole group, exited on SIGTERM


def test_tail_file_to_console_applies_line_prefix(monkeypatch, tmp_path):
    # A merge-pod member log carries no kubectl [pod/...] prefix, so the tailer tags
    # each line with the member name to keep the interleaved output attributable.
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "decode_server_1.log"
    path.write_text("")
    captured: list = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    cap = _Cap()
    life._logger.addHandler(cap)
    old_level = life._logger.level
    life._logger.setLevel(logging.INFO)
    try:
        async def drive():
            t = asyncio.ensure_future(
                life.tail_file_to_console(
                    str(path), task_name="decode_server_1",
                    line_prefix="[decode_server_1] ",
                )
            )
            await asyncio.sleep(0.05)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("token iter 1\n")
            await asyncio.sleep(0.5)
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)

        asyncio.run(drive())
    finally:
        life._logger.removeHandler(cap)
        life._logger.setLevel(old_level)
    assert "[decode_server_1] token iter 1" in "\n".join(captured)


def test_run_pod_stream_merge_uses_offloaded_demux_and_group_kill(monkeypatch, tmp_path):
    op, plan = _merge_op_and_plan(tmp_path, ["decode_server_0", "decode_server_1"])
    assert plan.merge_tag_paths  # a real merge plan
    calls: dict = {}
    proc = _FakeProc()

    async def fake_demux(log_command, *, tag_paths, default_path):
        calls["tag_paths"] = dict(tag_paths)
        return proc

    async def fake_watch(pod_ref, **kw):
        return "Succeeded"

    async def fake_terminate(p, *, kill_group=False):
        calls["terminated"] = (p is proc, kill_group)

    async def fake_exit(pod_ref, *, phase="", **kw):
        return 0

    monkeypatch.setattr(life, "start_pod_log_demux_stream", fake_demux)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "pod_exit_code", fake_exit)

    rc, phase = asyncio.run(op._run_pod_stream(plan=plan, index=0))
    assert (rc, phase) == (0, "Succeeded")
    # The demux is offloaded with every member's <task>.log as a routing target ...
    assert set(calls["tag_paths"]) == {"decode_server_0", "decode_server_1"}
    # ... and the kubectl|demuxer pipeline is stopped as a whole process group.
    assert calls["terminated"] == (True, True)


def test_execute_merge_tails_each_member_and_finalizes_member_logs(monkeypatch, tmp_path):
    op, _ = _merge_op_and_plan(tmp_path, ["decode_server_0", "decode_server_1"])
    events: list = []
    proc = _FakeProc()

    async def fake_demux(log_command, *, tag_paths, default_path):
        events.append(("demux", tuple(sorted(tag_paths))))
        return proc

    async def fake_tail(path, *, task_name, line_prefix=""):
        events.append(("tail", task_name, line_prefix))
        await asyncio.sleep(3600)

    async def fake_watch(pod_ref, **kw):
        await asyncio.sleep(0)
        return "Succeeded"

    async def fake_terminate(p, *, kill_group=False):
        events.append(("terminate", kill_group))



    def fake_sanitize(paths):
        # member paths only -- the default/leader path is one of them for a merge-pod
        events.append(("sanitize", tuple(sorted(
            os.path.basename(p).removesuffix(".log") for p in paths
        ))))

    async def fake_exit(pod_ref, *, phase="", **kw):
        return 0

    async def fake_delete(refs, **kw):
        events.append(("delete",))

    monkeypatch.setattr(life, "start_pod_log_demux_stream", fake_demux)
    monkeypatch.setattr(life, "tail_file_to_console", fake_tail)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "sanitize_streamed_logs", fake_sanitize)
    monkeypatch.setattr(life, "pod_exit_code", fake_exit)
    monkeypatch.setattr(life, "delete_objects", fake_delete)

    rc = asyncio.run(
        op.execute(
            launcher=_FakeLauncher(), output_logger=None,
            env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path / "decode_server_0")},
            task_name="decode_server_0", script=["echo hi"],
        )
    )
    assert rc == 0
    # One decoupled console tailer per merged member, each tagged with its name.
    tails = [e for e in events if e[0] == "tail"]
    assert {t[1] for t in tails} == {"decode_server_0", "decode_server_1"}
    assert all(t[2] == f"[{t[1]}] " for t in tails)
    assert ("demux", ("decode_server_0", "decode_server_1")) in events
    assert ("terminate", True) in events
    # BEHAVIOUR CHANGE: the demux pipeline is now DRAINED on a terminal pod rather
    # than group-killed mid-stream, so every member's <task>.log is already complete
    # and the re-fetch + re-split is skipped. Merge still guarantees complete member
    # logs -- now by keeping what the demuxer wrote -- and the single-pod finalize is
    # still never used for a merge-pod.
    assert ("sanitize", ("decode_server_0", "decode_server_1")) in events


def test_slow_post_terminal_stage_is_named_in_the_log(monkeypatch, tmp_path, caplog):
    """A stalled epilogue must say WHICH stage stalled.

    Once the pods are terminal the DAG is blocked on this epilogue alone, and it used
    to run silently: two ~20-minute investigations stalled on "which of the four
    stages was it?", unanswerable from the artifacts. Any stage over the threshold
    now names itself. Drives the log-finalize stage, which is the post-terminal stage
    that still does real work (the TTY-sanitize pass over the streamed file).
    """
    from sflow.plugins.operators import k8s_operator as k8sop

    async def _noop_stream(log_command, dest_path):
        return _FakeProc()

    async def _noop_tail(path, *, task_name):
        await asyncio.sleep(3600)

    async def _watch(pod_ref, **kw):
        return "Succeeded"

    async def _term(p, *, kill_group=False):
        pass

    def _slow_finalize(paths):
        time.sleep(0.05)                   # the stage we expect to be named

    async def _exit(pod_ref, *, phase="", **kw):
        return 0

    async def _delete(refs, **kw):
        pass

    monkeypatch.setattr(life, "start_pod_log_file_stream", _noop_stream)
    monkeypatch.setattr(life, "tail_file_to_console", _noop_tail)
    monkeypatch.setattr(life, "watch_until_terminal", _watch)
    monkeypatch.setattr(life, "terminate_process", _term)
    monkeypatch.setattr(life, "sanitize_streamed_logs", _slow_finalize)
    monkeypatch.setattr(life, "pod_exit_code", _exit)
    monkeypatch.setattr(life, "delete_objects", _delete)
    monkeypatch.setattr(k8sop, "_EPILOGUE_WARN_S", 0.01)

    op = K8sOperator(K8sOperatorConfig(name="op", image="img:1"))
    op.apply_backend_context(backend=_backend(), assigned_nodes=["node-0"],
                             artifacts=[], gpu_count=2)
    with caplog.at_level(logging.WARNING, logger="sflow.plugins.operators.k8s_operator"):
        rc = asyncio.run(
            op.execute(
                launcher=_FakeLauncher(), output_logger=None,
                env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
                task_name="mlperf_harness", script=["run"],
            )
        )
    assert rc == 0
    slow = [r.message for r in caplog.records if "post-terminal" in r.message]
    assert slow, "a slow post-terminal stage must be reported, not silent"
    assert any("log finalize" in m for m in slow), slow
    assert any("mlperf_harness" in m for m in slow), slow


# ---------------------------------------------------------------------------
# drain the follow instead of killing it (removes the re-fetch from the hot path)
# ---------------------------------------------------------------------------


def test_drain_log_stream_waits_for_a_follow_that_finishes():
    """A terminal pod's follow hits EOF on its own -- wait for it, don't kill it.

    Killing it mid-stream is what truncated <task>.log and made the one-shot re-fetch
    necessary; the re-fetch in turn had to be repaired for kubelet log rotation.
    """
    class _Exits:
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    assert asyncio.run(life.drain_log_stream(_Exits())) is True


def test_drain_log_stream_is_bounded_and_kills_a_stuck_follow(monkeypatch, caplog):
    """The historical kubectl bug is a follow that NEVER exits after pod completion.

    The epilogue is on the DAG's critical path, so the drain must be bounded: cut the
    follow and keep whatever it had already written, rather than wait forever.
    """
    killed: list = []

    class _Stuck:
        returncode = None

        async def wait(self):
            await asyncio.sleep(3600)

    async def fake_terminate(proc, *, kill_group=False):
        killed.append(kill_group)

    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    with caplog.at_level(logging.WARNING):
        drained = asyncio.run(
            life.drain_log_stream(_Stuck(), kill_group=True, timeout=0.01)
        )
    assert drained is False, "a stuck follow must not block the epilogue"
    assert killed == [True], "the stuck pipeline must be group-killed"
    assert any("did not finish" in r.getMessage() for r in caplog.records)


def test_drain_of_an_already_exited_stream_is_a_noop():
    class _Done:
        returncode = 0

        async def wait(self):  # pragma: no cover - must not be reached
            raise AssertionError("must not wait on an exited process")

    assert asyncio.run(life.drain_log_stream(_Done())) is True


def test_sanitize_streamed_logs_strips_tty_bytes_and_skips_missing(tmp_path):
    """The drained path keeps the streamed file, so it must apply the cleanup the
    re-fetch used to provide."""
    good = tmp_path / "a.log"
    good.write_bytes(b"frame1\rframe2\rfinal line\n\x1b[31mred\x1b[0m\n")
    empty = tmp_path / "empty.log"
    empty.write_bytes(b"")
    life.sanitize_streamed_logs([str(good), str(empty), str(tmp_path / "gone.log")])
    out = good.read_bytes()
    assert b"\r" not in out and b"\x1b" not in out
    assert b"final line" in out and b"red" in out
    assert b"frame1" not in out, "\\r redraws collapse to the last frame"


def test_tail_keeps_the_final_frame_of_an_unterminated_bar_ending_in_cr(
    monkeypatch, tmp_path
):
    """An unterminated bar whose last byte is ``\\r`` must still show its final frame.

    ``frame_in_progress`` collapses the retained line to what follows the last ``\\r`` --
    which for this shape is NOTHING -- so the frame the terminal is still displaying has
    to be carried separately and rejoined at flush. Without the carry the console shows
    absolutely nothing for the task's last visible output, which is the one line a
    progress bar exists to produce. ``SubprocessLauncher`` already handles this for every
    other backend; kubectl delivers exactly this shape when the container exits.
    """
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "t.log"
    path.write_bytes(b"")
    pad = b"." * 60
    # Every frame ENDS with \r and there is no terminating newline, so `\r` is the very
    # last byte in the file when the tailer is cancelled.
    bar = b"".join(b"Epoch %d/300 [%s]\r" % (i, pad) for i in range(1, 301))

    messages = _tail_once(path, append_bytes=bar)

    joined = "\n".join(messages)
    assert "Epoch 300/300" in joined, (
        "the carried frame must be rejoined at flush; without it the console shows "
        f"nothing at all for this task. Got {messages!r}"
    )
    assert "Epoch 299/300" not in joined, "superseded frames were overwritten on screen"


def test_tail_reads_once_more_when_cancelled_so_the_drained_tail_is_shown(
    monkeypatch, tmp_path
):
    """Output appended just before cancellation must still reach the console.

    ``execute`` cancels the tailer immediately after ``drain_log_stream`` delivered the
    pod's tail, so the most interesting bytes of the whole run routinely land inside the
    final poll gap. Without a last read they are echoed nowhere -- the console's last
    word on a finished task would be whatever happened 0.3s earlier.
    """
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "t.log"
    path.write_bytes(b"")
    captured: list = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    cap = _Cap()
    life._logger.addHandler(cap)
    old_level = life._logger.level
    life._logger.setLevel(logging.INFO)
    try:
        async def drive():
            t = asyncio.ensure_future(
                life.tail_file_to_console(str(path), task_name="t")
            )
            await asyncio.sleep(0.05)
            with open(path, "ab") as fh:
                fh.write(b"benchmark complete: 1234 tok/s\n")
            # Cancel at once -- far inside the poll interval, as the real caller does.
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)

        asyncio.run(drive())
    finally:
        life._logger.removeHandler(cap)
        life._logger.setLevel(old_level)

    assert any("benchmark complete: 1234 tok/s" in m for m in captured), (
        f"the tail delivered by the drain must survive cancellation, got {captured!r}"
    )


def test_tail_caps_lines_per_tick_and_reports_the_omission(monkeypatch, tmp_path):
    """The per-tick LINE COUNT cap (distinct from the per-line and per-tick char caps).

    A task that dumps tens of thousands of short lines in one poll interval passes both
    character caps -- each line is tiny -- so only the count cap bounds the number of
    console records handed to the rich handler in a single event-loop slice.
    """
    monkeypatch.setattr(life, "_console_active", lambda: True)
    path = tmp_path / "t.log"
    path.write_bytes(b"")
    burst = b"".join(b"line %d\n" % i for i in range(5000))

    messages = _tail_once(path, append_bytes=burst)

    echoed = [m for m in messages if m.startswith("line ")]
    assert len(echoed) <= life._TAIL_MAX_LINES_PER_TICK, (
        f"{len(echoed)} lines echoed in one tick, cap is "
        f"{life._TAIL_MAX_LINES_PER_TICK}"
    )
    assert any("console lines omitted" in m for m in messages), (
        "a silent drop reads as 'the task printed nothing'; it must be reported"
    )
    assert any("line 4999" in m for m in echoed), "the NEWEST lines are the ones kept"


def test_tick_char_budget_can_never_starve_the_first_line():
    """The two console caps are independent constants and must stay ordered.

    A tick budget below the per-line cap would reject the FIRST line of every tick --
    lines are emitted whole or not at all -- and silence the console permanently while
    <task>.log kept filling. The code enforces the relationship with ``max()`` rather
    than trusting it; this pins the invariant so a future tuning cannot invert it.
    """
    assert life._TAIL_MAX_CHARS_PER_TICK >= life._TAIL_MAX_LINE_CHARS, (
        "the per-tick character budget must admit at least one full-length line"
    )
    assert life._TAIL_MAX_LINE_CHARS == console_text.CONSOLE_LINE_CHAR_CAP, (
        "the k8s per-line cap must stay the shared console cap, or the k8s tailer and "
        "the launcher will drift on how much of a line the console may render"
    )


# ---------------------------------------------------------------------------
# stop_log_stream: drain a terminal pod's follow, cut a still-running one's
# ---------------------------------------------------------------------------


def test_stop_log_stream_drains_a_terminal_pod_instead_of_killing_it():
    """A terminal pod's follow hits EOF, so it must be drained -- killing truncates."""
    waited = {"v": False}
    killed: list = []

    class _Finishes:
        returncode = None

        async def wait(self):
            waited["v"] = True
            self.returncode = 0
            return 0

    async def fake_terminate(proc, *, kill_group=False):
        killed.append(kill_group)

    proc = _Finishes()
    with mock.patch.object(life, "terminate_process", fake_terminate):
        drained = asyncio.run(life.stop_log_stream(proc, terminal=True))

    assert drained is True
    assert waited["v"] is True, "a terminal follow must be waited out, not signalled"


def test_stop_log_stream_cuts_a_still_running_pod_without_waiting():
    """A pod still Running at teardown has a follow that would NEVER end.

    Draining it would block teardown for the full timeout on every long-lived service
    in the workflow, so this branch must signal immediately and not wait.
    """
    killed: list = []

    class _Never:
        returncode = None

        async def wait(self):  # pragma: no cover - must not be reached
            raise AssertionError("a running pod's follow must not be waited on")

    async def fake_terminate(proc, *, kill_group=False):
        killed.append(kill_group)

    with mock.patch.object(life, "terminate_process", fake_terminate):
        drained = asyncio.run(
            life.stop_log_stream(_Never(), terminal=False, kill_group=True)
        )

    assert drained is False
    assert killed == [True], "the merge pipeline must be group-killed, not left running"


def test_stop_log_stream_reaps_the_child_even_when_the_drain_overruns():
    """The drain is bounded; whatever happens, the kubectl child must not outlive it.

    The reap lives in a ``finally`` precisely because the drain is a cancellable await
    on the teardown path -- an unreaped follow keeps writing into a log the driver has
    already finalized.
    """
    killed: list = []

    class _Stuck:
        returncode = None

        async def wait(self):
            await asyncio.sleep(3600)

    async def fake_terminate(proc, *, kill_group=False):
        killed.append(kill_group)

    with mock.patch.object(life, "terminate_process", fake_terminate):
        with mock.patch.object(life, "STREAM_DRAIN_TIMEOUT", 0.01):
            drained = asyncio.run(life.stop_log_stream(_Stuck(), terminal=True))

    assert drained is False, "an overrunning drain must not report success"
    assert killed, "the stuck follow must be terminated on the overrun path"
