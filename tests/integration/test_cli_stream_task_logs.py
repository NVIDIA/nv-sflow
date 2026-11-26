# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise live task output through a real CLI subprocess without a terminal."""

import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import sys
import tempfile
import time

import pytest
import yaml


# External backends are opt-in and run in the existing hardware CI jobs. A bare
# pytest still exercises local execution on every supported Python version.
BACKEND = os.environ.get("SFLOW_E2E_STREAM_BACKEND", "local")
if BACKEND != "local":
    pytestmark = pytest.mark.e2e

_MARKERS = (
    "NTTY_SERVER_STARTUP_0",
    "NTTY_CLIENT_PROGRESS_0",
    "NTTY_CLIENT_PROGRESS_3",
    "NTTY_CLIENT_STDERR_0",
    "NTTY_CLIENT_DONE",
)


def _backend_override(backend):
    """Run the same server/client example on a CI job's real backend."""
    if backend in ("local", "ssh"):
        return {}
    config = {"name": "stream_ci", "default": True, "nodes": 1, "gpus_per_node": 0}
    operator = {"name": "stream_ci"}
    tasks = [{"name": name, "operator": "stream_ci"} for name in ("server", "client")]
    if backend in ("docker", "docker_multinode"):
        config["type"] = "docker"
        operator.update(type="docker_run", image="python:3.12-slim")
        if backend == "docker_multinode":
            hosts = os.environ["SFLOW_E2E_DOCKER_HOSTS"].split(",")
            assert len(hosts) >= 2, "stream test needs two Docker hosts"
            user = os.environ.get("SFLOW_E2E_DOCKER_SSH_USER", "")
            config.update(
                nodes=2,
                hosts=[
                    {"name": f"host{i}", "address": host.strip(), "user": user or None}
                    for i, host in enumerate(hosts[:2])
                ],
            )
            for i, task in enumerate(tasks):
                task["resources"] = {"nodes": {"indices": [i]}}
    elif backend == "slurm":
        config.update(
            type="slurm",
            partition=os.environ["SLURM_E2E_PARTITION"],
            account=os.environ.get("SLURM_E2E_ACCOUNT", ""),
            time="00:05:00",
            gpus_per_node=int(os.environ.get("GPUS_PER_NODE", "0")),
        )
        extras = shlex.split(os.environ.get("SLURM_E2E_SALLOC_EXTRA", ""))
        segment = os.environ.get("SLURM_E2E_SEGMENT", "")
        if segment:
            extras.append(f"--segment={'1' if segment == 'auto' else segment}")
        exclude = os.environ.get("SLURM_E2E_EXCLUDE_NODES", "")
        if exclude:
            extras.append(f"--exclude={exclude}")
        config["extra_args"] = extras
        operator["type"] = "srun"
    elif backend in ("k8s", "k8s_mpi"):
        config.update(
            type="kubernetes",
            scheduling="device_plugin",
            namespace=os.environ.get("K8S_NAMESPACE", "default"),
        )
        operator.update(type=backend, image="python:3.12-slim")
        if backend == "k8s_mpi":
            config["nodes"] = 2
            operator.update(
                image=os.environ.get(
                    "SFLOW_E2E_STREAM_MPI_IMAGE",
                    "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
                ),
                run_as_root=True,
                mpi={"route": "pods", "slots_per_worker": 1},
            )
            for task in tasks:
                task["resources"] = {"nodes": {"count": 2}}
                task["script"] = [
                    "mpirun -np 2 --allow-run-as-root python3 -u "
                    "${{ artifacts.EMITTER.path }} " + task["name"]
                ]
    else:
        raise ValueError(f"Unknown streaming e2e backend: {backend}")
    return {
        "version": "0.1",
        "backends": [config],
        "operators": [operator],
        "workflow": {"tasks": tasks},
    }


@pytest.mark.parametrize(
    "backend", ["local", "docker", "docker_multinode", "slurm", "k8s", "k8s_mpi"]
)
def test_stream_backend_dry_run(tmp_path, monkeypatch, backend):
    from typer.testing import CliRunner
    from sflow.cli import app

    monkeypatch.setenv("SFLOW_E2E_DOCKER_HOSTS", "host0,host1")
    monkeypatch.setenv("SLURM_E2E_PARTITION", "test")
    monkeypatch.setenv("SLURM_E2E_ACCOUNT", "test")
    repo = Path(__file__).resolve().parents[2]
    args = [
        "run",
        "-f",
        str(repo / "examples/self_contained/local/stream_task_logs.yaml"),
    ]
    override = _backend_override(backend)
    if override:
        fragment = tmp_path / "backend.yaml"
        fragment.write_text(yaml.safe_dump(override))
        args += ["-f", str(fragment)]
    result = CliRunner().invoke(
        app,
        [
            *args,
            "--stream-task-logs",
            "--dry-run",
            "--verbose",
            "--workspace-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # Generated kubectl manifests contain inline artifacts. A command echo must
    # never satisfy the live-output assertions (or make default silence fail).
    assert not any(marker in result.output for marker in _MARKERS), result.output


@pytest.mark.parametrize(
    "flags, streaming",
    [
        ([], False),
        (["--stream-task-logs"], True),
        (["--stream-task-logs", "--offload-task-logs"], True),
    ],
)
def test_non_tty_task_logs(tmp_path, flags, streaming):
    repo = Path(__file__).resolve().parents[2]
    recipe = repo / "examples/self_contained/local/stream_task_logs.yaml"
    artifact_root = os.environ.get("SFLOW_E2E_ARTIFACT_DIR")
    if artifact_root:
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(tempfile.mkdtemp(prefix=f"stream-{BACKEND}-", dir=root))
    backend_args = []
    override = _backend_override(BACKEND)
    if override:
        fragment = tmp_path / "backend.yaml"
        fragment.write_text(yaml.safe_dump(override))
        backend_args += ["-f", str(fragment)]
    if BACKEND.startswith("k8s"):
        backend_args += shlex.split(os.environ.get("NODE_SELECTOR", ""))
        backend_args += ["--kube-rdma", "disable"]
    if BACKEND == "ssh":
        backend_args += [
            "--ssh",
            os.environ["SFLOW_E2E_SSH"],
            "--ssh-tty",
            "never",
            "--ssh-follow",
            "none",
        ]
        if os.environ.get("SFLOW_E2E_REMOTE_ROOT"):
            backend_args += ["--ssh-remote-root", os.environ["SFLOW_E2E_REMOTE_ROOT"]]
    env = dict(os.environ)
    for key in ("SFLOW_OFFLOAD_TASK_LOGS", "SFLOW_SLURM_OFFLOAD_TASK_LOGS"):
        env.pop(key, None)
    markers = _MARKERS
    output = ""
    live = set()
    startup_before_client = False
    progress_before_done = False
    with (tmp_path / "stderr.txt").open("w+") as stderr:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sflow",
                "run",
                "-f",
                str(recipe),
                *backend_args,
                "--workspace-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                *flags,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            start_new_session=True,
        )
        try:
            assert not os.isatty(proc.stdout.fileno())
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + (35 if BACKEND == "local" else 900)
                while selector.get_map():
                    assert time.monotonic() < deadline, output
                    for key, _ in selector.select(timeout=0.1):
                        data = os.read(key.fd, 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        output += data.decode(errors="replace")
                        if proc.poll() is None:
                            live.update(m for m in markers if m in output)
                            startup_before_client |= (
                                markers[0] in output and markers[1] not in output
                            )
                            progress_before_done |= (
                                markers[1] in output and markers[-1] not in output
                            )
            stderr.seek(0)
            assert proc.wait(timeout=5) == 0, output + stderr.read()
        finally:
            (tmp_path / "stdout.txt").write_text(output)
            # Let sflow release allocations/pods/containers before killing leftovers.
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
            # Also reap offload pipeline children if the driver has already exited.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            proc.stdout.close()
    assert live == (set(markers) if streaming else set()), output
    assert startup_before_client == streaming, output
    assert progress_before_done == streaming, output
    assert all((m in output) == streaming for m in markers), output
    task_logs = list((tmp_path / "out").glob("**/server.log"))
    task_logs += list((tmp_path / "out").glob("**/client.log"))
    persisted = "".join(path.read_text() for path in task_logs)
    assert all(m in persisted for m in markers)
    driver_log = next((tmp_path / "out").glob("**/sflow.log")).read_text()
    assert all(m not in driver_log for m in markers)
    print(
        f"backend={BACKEND} flags={flags} live_markers={sorted(live)} artifacts={tmp_path}"
    )
