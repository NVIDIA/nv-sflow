# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver-side k8s lifecycle helpers + the operator's decoupled ``execute`` flow.

These cover the Python half of the k8s operator: polling pod status, deriving the
exit code, deleting objects, and the apply -> stream -> watch -> stop-on-terminal
orchestration in ``K8sContainerOperator.execute`` (with kubectl + the launcher
faked out so no cluster is needed)."""

import asyncio
import logging

import pytest

from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.plugins.operators import _k8s_lifecycle as life
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

    async def fake(args, *, global_args=()):
        return (0, next(phases), "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/t", global_args=[], ns_args=[], interval=0)
    )
    assert phase == "Succeeded"


def test_watch_until_terminal_returns_empty_when_pod_gone(monkeypatch):
    async def fake(args, *, global_args=()):
        return (1, "", "NotFound")  # rc != 0 -> phase ""

    monkeypatch.setattr(life, "run_kubectl", fake)
    phase = asyncio.run(
        life.watch_until_terminal("pod/t", global_args=[], ns_args=[], interval=0)
    )
    assert phase == ""


def test_pod_exit_code_reads_terminated_exit_code(monkeypatch):
    async def fake(args, *, global_args=()):
        return (0, "7", "")

    monkeypatch.setattr(life, "run_kubectl", fake)
    code = asyncio.run(life.pod_exit_code("pod/t", global_args=[], ns_args=[]))
    assert code == 7


def test_pod_exit_code_falls_back_to_phase(monkeypatch):
    monkeypatch.setattr(life, "EXIT_CODE_RETRIES", 1)

    async def fake(args, *, global_args=()):
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

    async def fake(args, *, global_args=()):
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
    async def fake(args, *, global_args=()):
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

    async def _term(p):
        pass

    async def _finalize(*a, **k):
        pass

    async def _exit(pod_ref, *, phase="", **kw):
        return 0

    async def _delete(refs, **kw):
        pass

    monkeypatch.setattr(life, "start_pod_log_file_stream", _noop_stream)
    monkeypatch.setattr(life, "tail_file_to_console", _noop_tail)
    monkeypatch.setattr(life, "watch_until_terminal", _watch)
    monkeypatch.setattr(life, "terminate_process", _term)
    monkeypatch.setattr(life, "finalize_complete_log", _finalize)
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

    async def fake_terminate(p):
        events.append(("terminate",))

    async def fake_finalize(pod_refs, dest, *, prefix_size, phases, **kw):
        events.append(("finalize", dest, tuple(pod_refs), tuple(phases)))

    async def fake_exit(pod_ref, *, phase="", **kw):
        events.append(("exit", phase))
        return 0

    async def fake_delete(refs, **kw):
        events.append(("delete",))

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "tail_file_to_console", fake_tail)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "finalize_complete_log", fake_finalize)
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
    # Complete log re-fetched once after the watch, over all pods, with their phases.
    assert ("finalize", task_log, ("pod/decode-server-0",), ("Succeeded",)) in events
    kinds = [e[0] for e in events]
    # Status is authoritative: watch -> interrupt the stream -> read exit code
    # (per pod) -> re-fetch the complete on-disk log (once) -> cleanup.
    assert (
        kinds.index("watch")
        < kinds.index("terminate")
        < kinds.index("exit")
        < kinds.index("finalize")
    )
    assert kinds[-1] == "delete"


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

    async def fake_terminate(p):
        pass

    async def fake_exit(pod_ref, *, phase="", **kw):
        return 1 if phase == "Failed" else 0

    async def fake_finalize(*a, **k):
        pass

    async def fake_delete(refs, **kw):
        pass

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "tail_file_to_console", fake_tail)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "pod_exit_code", fake_exit)
    monkeypatch.setattr(life, "finalize_complete_log", fake_finalize)
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

    async def fake_terminate(p):
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
    assert phase == "Failed"  # phase returned to execute() for the complete re-fetch
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
    from sflow.plugins.operators._k8s_shell import build_log_stream_command

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
# completeness: re-fetch the whole container log on terminal (stream was cut)
# ---------------------------------------------------------------------------


def test_finalize_complete_log_splices_prefix_plus_complete_dump(monkeypatch, tmp_path):
    dest = tmp_path / "t.log"
    # <task>.log mid-run: [apply diagnostics prefix][partial live-streamed tail].
    prefix = b"2026 - sflow.task.t - INFO - apply-diag-1\n"
    dest.write_bytes(prefix + b"[pod/t] partial-live-tail\n")

    async def fake_exec(*args, stdout=None, stderr=None, **kw):
        # `kubectl logs <pod> --all-containers --prefix` returns the COMPLETE log.
        stdout.write(b"[pod/t] complete-1\n[pod/t] complete-2\n[pod/t] complete-3\n")
        stdout.flush()

        class _P:
            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(life.asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(
        life.finalize_complete_log(
            ["pod/t"], str(dest), prefix_size=len(prefix), phases=["Succeeded"],
            global_args=[], ns_args=[],
        )
    )
    # Apply prefix preserved; the partial live tail is replaced by the complete
    # dump (no duplication, nothing missing).
    assert dest.read_bytes() == (
        prefix + b"[pod/t] complete-1\n[pod/t] complete-2\n[pod/t] complete-3\n"
    )
    assert not (tmp_path / "t.log.complete.0").exists()
    assert not (tmp_path / "t.log.final").exists()


def test_finalize_complete_log_multi_pod_concatenates_per_pod(monkeypatch, tmp_path):
    dest = tmp_path / "worker.log"
    prefix = b"2026 - sflow.task.worker - INFO - apply-diag\n"
    # Mid-run the file has the prefix + interleaved partial live content.
    dest.write_bytes(prefix + b"[pod/worker-0] a-live\n[pod/worker-1] b-live\n")

    async def fake_exec(*args, stdout=None, stderr=None, **kw):
        argv = [str(a) for a in args]
        pod = next((a for a in argv if a.startswith("pod/")), "")
        if "worker-0" in pod:
            stdout.write(b"[pod/worker-0] a1\n[pod/worker-0] a2\n")
        elif "worker-1" in pod:
            stdout.write(b"[pod/worker-1] b1\n[pod/worker-1] b2\n")
        stdout.flush()

        class _P:
            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(life.asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(
        life.finalize_complete_log(
            ["pod/worker-0", "pod/worker-1"], str(dest), prefix_size=len(prefix),
            phases=["Succeeded", "Succeeded"], global_args=[], ns_args=[],
        )
    )
    # Prefix preserved, then each pod's COMPLETE log grouped per pod (no interleave
    # garble, nothing missing, no duplication of the partial live content).
    assert dest.read_bytes() == (
        prefix
        + b"[pod/worker-0] a1\n[pod/worker-0] a2\n"
        + b"[pod/worker-1] b1\n[pod/worker-1] b2\n"
    )
    assert not (tmp_path / "worker.log.complete.0").exists()
    assert not (tmp_path / "worker.log.complete.1").exists()
    assert not (tmp_path / "worker.log.final").exists()


def test_finalize_complete_log_keeps_live_content_on_dump_failure(monkeypatch, tmp_path):
    dest = tmp_path / "t.log"
    original = b"apply\n[pod/t] live-content\n"
    dest.write_bytes(original)

    async def fake_exec(*args, stdout=None, stderr=None, **kw):
        class _P:
            async def wait(self):
                return 1  # dump failed (e.g. pod already deleted)

        return _P()

    monkeypatch.setattr(life.asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(
        life.finalize_complete_log(
            ["pod/t"], str(dest), prefix_size=6, phases=["Succeeded"],
            global_args=[], ns_args=[],
        )
    )
    assert dest.read_bytes() == original  # untouched: don't wipe what we have


def test_finalize_complete_log_skips_when_a_pod_is_not_terminal(monkeypatch, tmp_path):
    # If any pod is gone/unknown (phase ""), we can't re-fetch it, so we keep the
    # live-streamed content rather than rebuild a partial log (and never dump).
    dest = tmp_path / "worker.log"
    original = b"apply\n[pod/worker-0] live-0\n[pod/worker-1] live-1\n"
    dest.write_bytes(original)
    calls: list = []

    async def fake_exec(*args, **kw):
        calls.append(args)

        class _P:
            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(life.asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(
        life.finalize_complete_log(
            ["pod/worker-0", "pod/worker-1"], str(dest), prefix_size=6,
            phases=["Succeeded", ""], global_args=[], ns_args=[],
        )
    )
    assert dest.read_bytes() == original  # kept as-is
    assert calls == []  # short-circuited before dumping anything


def test_run_pod_stream_cancel_interrupts_stream_without_finalize(monkeypatch, tmp_path):
    # A long-lived READY service cancelled at workflow end: the stream is
    # interrupted but the complete-log re-fetch does NOT run (final_phase is "").
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

    async def fake_terminate(p):
        terminated.append(True)

    async def fake_finalize(*a, **k):
        finalized.append(True)

    monkeypatch.setattr(life, "start_pod_log_file_stream", fake_start_stream)
    monkeypatch.setattr(life, "watch_until_terminal", fake_watch)
    monkeypatch.setattr(life, "terminate_process", fake_terminate)
    monkeypatch.setattr(life, "finalize_complete_log", fake_finalize)

    async def drive():
        t = asyncio.ensure_future(op._run_pod_stream(plan=plan, index=0))
        await started.wait()
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(drive())
    assert terminated == [True]  # stream interrupted on teardown
    assert finalized == []  # no complete-log re-fetch for a cancelled service
