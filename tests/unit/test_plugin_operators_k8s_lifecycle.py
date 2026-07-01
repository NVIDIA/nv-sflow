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
