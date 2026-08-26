# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-Docker end-to-end tests for the docker backend.

Unlike the unit/integration suites (which never touch a daemon), these actually
`docker run` containers through `sflow run` and assert on real behavior: task
output is captured, and no container is leaked after the run.

Marked ``e2e``, so a bare ``pytest`` deselects them (they launch real containers)
and additionally gated on a reachable Docker daemon. Run them on a machine with
Docker by asking for the marker explicitly:

    pytest tests/e2e_tests/test_docker_backend_e2e.py -v -m e2e

The workload image must contain `bash` (the docker_run operator launches
`bash -lc <script>`). Override the default via:

    SFLOW_E2E_DOCKER_IMAGE=ubuntu:22.04 pytest tests/e2e_tests/test_docker_backend_e2e.py

GPU cases additionally require `nvidia-smi` on PATH and the NVIDIA Container
Toolkit; they self-skip otherwise.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from gpu_evidence import (
    Span,
    claimed_uuids,
    overlapping_spans,
    open_spans,
    parse_spans,
)
from run_evidence import archive_run_dir, echo_run_evidence

# Image for the CPU-only plumbing tests. Must ship bash. Overridable for mirrors.
E2E_IMAGE = os.environ.get("SFLOW_E2E_DOCKER_IMAGE", "ubuntu:22.04")
# Image for the GPU tests: the NVIDIA vectorAdd sample (~240 MB). Small and quick
# to pull, but it really launches a CUDA kernel, so a container handed the wrong
# devices -- or none -- fails loudly instead of passing the way an nvidia-smi-only
# check would. NOTE: it sets ENTRYPOINT=/cuda-samples/sample, which would swallow
# sflow's `bash -lc <script>`; the samples clear it with `--entrypoint=`.
E2E_GPU_IMAGE = os.environ.get(
    "SFLOW_E2E_DOCKER_GPU_IMAGE", "nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0"
)

# Image for the monitor-window test: the CUDA nbody sample (~500 MB). Unlike
# vectorAdd (which finishes instantly) `nbody -benchmark` sustains GPU load for a
# controllable number of iterations, which is what gives the marker window a
# measurable span to clip to.
E2E_NBODY_IMAGE = os.environ.get(
    "SFLOW_E2E_DOCKER_NBODY_IMAGE", "nvcr.io/nvidia/k8s/cuda-sample:nbody"
)

# A run's container timeout in seconds (first run may pull the image).
_RUN_TIMEOUT = 600


def _docker_daemon_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        return result.returncode == 0
    except Exception:
        return False


_HAS_DOCKER = _docker_daemon_available()
_HAS_NVIDIA = shutil.which("nvidia-smi") is not None


def _gpu_count() -> int:
    """Physical GPUs on this host, or 0 when nvidia-smi is unavailable."""
    if not _HAS_NVIDIA:
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20
        )
    except Exception:
        return 0
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


_GPU_COUNT = _gpu_count()

# The shipped samples these tests drive. Exercising the real files (rather than
# YAML written inline by the test) is the point: it keeps the examples users copy
# honest, and a change that breaks them fails here.
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_GPU_EXAMPLES = _EXAMPLES / "gpu_reservation"
_HOG_SAMPLE = _GPU_EXAMPLES / "hog.yaml"
_SCHEDULING_SAMPLE = _GPU_EXAMPLES / "scheduling_smoke.yaml"

# `e2e` makes these opt-in: they are inside `testpaths`, so without it a bare
# `pytest` on any machine with a Docker daemon would pull images and launch real
# containers. Deselected by default via addopts (`-m "not e2e"`); run them with
# `pytest tests/e2e_tests/test_docker_backend_e2e.py -m e2e`. The skipif still
# guards the case where they ARE selected but no daemon is reachable.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _HAS_DOCKER, reason="requires a reachable Docker daemon"
    ),
]


# Set by CI to a path inside $CI_PROJECT_DIR so `artifacts:` can pick the tree up.
# Unset locally, where tmp_path is already on disk and easy to poke at.
_ARTIFACT_ROOT = os.environ.get("SFLOW_E2E_ARTIFACT_DIR")


@pytest.fixture(autouse=True)
def _publish_run_evidence(request, tmp_path: Path):
    """Echo + archive every e2e test's run output, pass or fail.

    Without this a GREEN run leaves nothing to check it against: tmp_path is under
    /tmp, outside the CI workspace, so the sflow summary and task logs that would
    show whether the workflow really did what it claims are thrown away and only a
    pass count survives. See tests/e2e_tests/run_evidence.py.

    Autouse so it cannot be forgotten by a test added later -- the whole point is
    that the evidence survives without each test opting in. Runs on teardown, so
    a failing assertion still gets its output published too (the assertion message
    shows what broke; this shows what the run did).
    """
    yield
    echo_run_evidence(tmp_path, request.node.name)
    archive_run_dir(tmp_path, request.node.name, _ARTIFACT_ROOT)


def _sflow_container_names(prefix: str) -> list[str]:
    """Names of all containers (running or exited) whose name starts with prefix."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [n for n in result.stdout.split() if n]


def _force_remove(prefix: str) -> None:
    for name in _sflow_container_names(prefix):
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )


def _run_sflow(
    cfg: Path, tmp_path: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    out_dir = tmp_path / "out"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sflow",
            "run",
            "-f",
            str(cfg),
            "--workspace-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT,
        env={**os.environ, **(env or {})},
    )


def _output_haystack(result: subprocess.CompletedProcess, tmp_path: Path) -> str:
    """Everything the run emitted: stdout + stderr + every file under the run dir.

    Robust to log-offload being on (sentinel lands in <task>.log) or off (sentinel
    streams to the driver's stdout).
    """
    parts = [result.stdout or "", result.stderr or ""]
    out_dir = tmp_path / "out"
    if out_dir.exists():
        for path in out_dir.rglob("*"):
            if path.is_file():
                try:
                    parts.append(path.read_text(errors="replace"))
                except OSError:
                    pass
    return "\n".join(parts)


def test_docker_e2e_single_node_runs_and_reaps_container(tmp_path: Path):
    sentinel = "SFLOW_E2E_SINGLE_OK"
    prefix = "e2esingle"  # matches sflow-p<pid>-e2esingle-<node> (task name is mid-name)
    cfg = tmp_path / "single.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            version: "0.1"
            backends:
              - name: docker
                type: docker
                default: true
                image: {E2E_IMAGE}
                nodes: 1
            workflow:
              name: e2e_docker_single
              tasks:
                - name: e2esingle
                  script:
                    - echo "{sentinel}"
            """
        ).strip()
    )

    _force_remove(prefix)
    try:
        result = _run_sflow(cfg, tmp_path)
    finally:
        leaked = _sflow_container_names(prefix)
        _force_remove(prefix)

    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel in _output_haystack(result, tmp_path)
    # The daemon-managed container must not outlive the run.
    assert leaked == [], f"leaked containers: {leaked}"


def test_docker_e2e_local_multi_node_runs_both_and_reaps_all(tmp_path: Path):
    """Two synthetic localhost nodes must each launch a real container and all of
    them must be reaped (exercises the local multi-node bash wrapper for real)."""
    sentinel = "SFLOW_E2E_MULTI_OK"
    prefix = "e2emulti"
    cfg = tmp_path / "multi.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            version: "0.1"
            backends:
              - name: docker
                type: docker
                default: true
                image: {E2E_IMAGE}
                nodes: 2
            workflow:
              name: e2e_docker_multi
              tasks:
                - name: e2emulti
                  resources:
                    nodes:
                      count: 2
                  script:
                    - echo "{sentinel} on ${{SFLOW_TASK_ASSIGNED_NODE_NAMES:-?}}"
            """
        ).strip()
    )

    _force_remove(prefix)
    try:
        result = _run_sflow(cfg, tmp_path)
    finally:
        leaked = _sflow_container_names(prefix)
        _force_remove(prefix)

    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel in _output_haystack(result, tmp_path)
    assert leaked == [], f"leaked containers: {leaked}"


@pytest.mark.skipif(
    not _HAS_NVIDIA, reason="requires nvidia-smi and the NVIDIA Container Toolkit"
)
def test_docker_e2e_gpu_narrowing_exposes_single_device(tmp_path: Path):
    """`--gpus device=<uuid>` must reach the container: with 1 GPU requested,
    `nvidia-smi -L` inside the container lists exactly one GPU.

    Runs with foreign-busy detection off. Any ordinary workstation GPU carries a
    display server (hundreds of MiB + a compute context), so the reservation
    would correctly report zero free GPUs and this test would fail on developer
    machines rather than exercising what it is here to check. The foreign-busy
    path itself is covered by the unit suite.
    """
    prefix = "e2egpu"
    cfg = tmp_path / "gpu.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            version: "0.1"
            backends:
              - name: docker
                type: docker
                default: true
                image: {E2E_GPU_IMAGE}
                nodes: 1
                gpus_per_node: 1
                extra_args:
                  - --entrypoint=
            workflow:
              name: e2e_docker_gpu
              tasks:
                - name: e2egpu
                  resources:
                    gpus:
                      count: 1
                  script:
                    - nvidia-smi -L | tee /tmp/gpus.txt
                    - echo "SFLOW_E2E_GPU_COUNT=$(nvidia-smi -L | wc -l)"
                    # Narrowing is only real if the device is usable, so run a
                    # kernel on it rather than just counting what is visible.
                    - /cuda-samples/vectorAdd
            """
        ).strip()
    )

    _force_remove(prefix)
    try:
        result = _run_sflow(cfg, tmp_path, env={"SFLOW_GPU_IGNORE_FOREIGN": "1"})
    finally:
        _force_remove(prefix)

    haystack = _output_haystack(result, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SFLOW_E2E_GPU_COUNT=1" in haystack
    assert "Test PASSED" in haystack, "the reserved device did not actually compute"


@pytest.mark.skipif(
    not _HAS_NVIDIA, reason="requires nvidia-smi and the NVIDIA Container Toolkit"
)
def test_docker_e2e_monitor_log_window_clips_the_gpu_report(tmp_path: Path):
    """The shipped docker GPU monitor example resolves its marker window.

    Drives examples/self_contained/docker/gpu_monitor.yaml for real: idle, an
    nbody GPU burst bracketed by MEASURE_START/MEASURE_END, idle. Only a real run
    can prove marker resolution -- a dry-run never produces a task log, and the
    unit tests feed synthetic logs. This asserts the window both RESOLVED and
    actually narrowed the report relative to the unclipped workflow baseline.

    Foreign-busy detection off for the same reason as the narrowing test above.
    """
    cfg = _EXAMPLES / "self_contained" / "docker" / "gpu_monitor.yaml"
    prefix = "gpu_work"

    _force_remove(prefix)
    try:
        result = _run_sflow(
            cfg,
            tmp_path,
            env={"SFLOW_GPU_IGNORE_FOREIGN": "1"},
        )
    finally:
        _force_remove(prefix)

    assert result.returncode == 0, result.stdout + result.stderr

    runs = sorted((tmp_path / "out").glob("docker_gpu_monitor-*"))
    assert runs, "no run directory was produced"
    monitor_dir = runs[-1] / "sflow_monitor"

    # 1. The marker window resolved (an unresolved one writes window_not_found.json).
    window_path = monitor_dir / "windowed" / "gpu_work" / "window.json"
    assert window_path.is_file(), sorted(
        p.name for p in (monitor_dir / "windowed" / "gpu_work").glob("*")
    )
    window = json.loads(window_path.read_text())
    assert window["status"] == "matched", window
    assert window["start"]["line"] == "MEASURE_START"
    assert window["end"]["line"] == "MEASURE_END"
    # Markers bracketed real elapsed time, not two adjacent log lines.
    assert window["duration_seconds"] > 1.0, window

    # 2. The window actually clipped: the marker report is a strict subset of the
    #    unclipped whole-run baseline written from the SAME samples.
    clipped = (
        (monitor_dir / "windowed" / "gpu_work" / "timeline.csv").read_text().splitlines()
    )
    baseline = (
        (monitor_dir / "lifecycle" / "workflow" / "timeline.csv").read_text().splitlines()
    )
    assert len(clipped) > 1, "clipped report has no sample rows"
    assert len(clipped) < len(baseline), (
        f"marker window did not narrow the report: {len(clipped)} rows clipped "
        f"vs {len(baseline)} baseline"
    )

    # 3. The overview reports the resolution for the operator to see.
    overview = (runs[-1] / "sflow_monitor.log").read_text()
    assert "window=matched" in overview, overview


# ---------------------------------------------------------------------------
# Concurrent `sflow run` sessions on one host, driving the shipped sample
#
# This is the scenario the machine-local GPU registry exists for, and it only
# appears across separate driver processes -- so these run the real
# examples/gpu_reservation/hog.yaml through real containers, several at a time.
# ---------------------------------------------------------------------------


def _await_reservations(tmp_path: Path, count: int, *, timeout: float = 300.0) -> None:
    """Block until ``count`` reservation records exist in the shared registry.

    The sessions' ordering has to be CAUSAL, not timed: sleeping a fixed few
    seconds assumes the image is already cached and the driver started promptly,
    which is false on a cold CI runner -- the "holder" would still be pulling when
    the contender starts, and the contender would sail in. The driver writes its
    record on the host before it launches any container, so the record appearing
    is the exact signal these tests need.
    """
    registry = tmp_path / "registry"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(list(registry.glob("*.json"))) >= count:
            return
        time.sleep(0.1)
    pytest.fail(f"expected {count} GPU reservation(s) in {registry}, timed out")


def _await_record(
    registry: Path, pattern: str, *, present: bool, timeout: float = 300.0
) -> None:
    """Block until a record matching ``pattern`` exists (or is gone).

    Records are named ``<pid>-<task>.json``, so this waits on ONE task's claim
    rather than a count -- which is what a test needs when the interesting moment
    is a particular task letting go while others keep holding.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if bool(list(registry.glob(pattern))) is present:
            return
        time.sleep(0.1)
    pytest.fail(f"{pattern} in {registry}: waited for present={present}, timed out")


@pytest.fixture(scope="module", autouse=True)
def _prepull_gpu_image():
    """Pull the GPU sample once, before any timing-sensitive test runs.

    Keeps a cold registry from turning into a mid-test flake, and surfaces a
    missing/unauthorized image as one clear error instead of several odd ones.
    """
    if _GPU_COUNT < 1 or not _HAS_DOCKER:
        return
    for image in (E2E_GPU_IMAGE, E2E_NBODY_IMAGE):
        subprocess.run(["docker", "pull", image], capture_output=True, timeout=900)


def _start_sflow(
    cfg: Path,
    out_dir: Path,
    *,
    sets: dict[str, object],
    extra: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Launch `sflow run` in the background (concurrency needs several at once)."""
    cmd = [sys.executable, "-m", "sflow", "run", "-f", str(cfg)]
    for key, value in sets.items():
        cmd += ["--set", f"{key}={value}"]
    cmd += ["--workspace-dir", str(out_dir.parent), "--output-dir", str(out_dir), *extra]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, **(env or {})},
    )


def _shared_registry_env(tmp_path: Path) -> dict[str, str]:
    """Env for sessions that must contend with each other, and only each other.

    They share one registry directory (that is what makes them contend) but a
    private one, so the test never disturbs -- or is disturbed by -- real runs on
    the host. Foreign-busy detection is off because an ordinary workstation GPU
    carries a display server, which would correctly read as busy and leave the
    test nothing to reserve.
    """
    return {
        "SFLOW_GPU_RESERVATION_DIR": str(tmp_path / "registry"),
        "SFLOW_GPU_IGNORE_FOREIGN": "1",
    }


def _gpu_vanished_mid_run(text: str) -> bool:
    """Whether a container proved its GPU worked and then lost it.

    On this shared box a co-tenant resetting a device drops every container's CUDA
    context at once. Safe to treat differently from the bug that matters: a task
    handed the wrong devices fails on its FIRST kernel, never after a passing one.
    """
    return "Test PASSED" in text and "no CUDA-capable device is detected" in text


def _devices_reserved(output: str) -> set[int]:
    """Physical device indices from the driver's `Reserved N GPU(s) ... devices [..]`."""
    found: set[int] = set()
    for match in re.finditer(r"devices \[([0-9,\s]*)\]", output):
        found |= {int(tok) for tok in match.group(1).split(",") if tok.strip()}
    return found


@pytest.mark.skipif(
    _GPU_COUNT < 2, reason="needs >= 2 GPUs to prove two sessions land on different ones"
)
def test_concurrent_runs_of_the_sample_get_disjoint_gpus(tmp_path: Path):
    """Two `sflow run` processes on one box must not land on the same GPU.

    Drives the shipped examples/gpu_reservation/hog.yaml twice, concurrently.
    """
    env = _shared_registry_env(tmp_path)
    common = {"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT, "CLAIM": 1, "HOLD": 20}
    _force_remove("hog")
    try:
        first = _start_sflow(_HOG_SAMPLE, tmp_path / "a", sets=common, env=env)
        _await_reservations(tmp_path, 1)  # first has its GPU before the second asks
        second = _start_sflow(_HOG_SAMPLE, tmp_path / "b", sets=common, env=env)
        _await_reservations(tmp_path, 2)  # both hold at once -> the overlap we assert on
        out_a, _ = first.communicate(timeout=_RUN_TIMEOUT)
        out_b, _ = second.communicate(timeout=_RUN_TIMEOUT)
    finally:
        _force_remove("hog")

    # The placement is the subject here, and it is decided (and logged) before the
    # workload runs -- so check it first, whatever the containers went on to do.
    gpus_a, gpus_b = _devices_reserved(out_a), _devices_reserved(out_b)
    assert len(gpus_a) == 1 and len(gpus_b) == 1, f"a={gpus_a} b={gpus_b}"
    assert gpus_a.isdisjoint(gpus_b), f"both sessions took {gpus_a & gpus_b}"

    for label, proc, out in (("a", first, out_a), ("b", second, out_b)):
        if proc.returncode != 0 and _gpu_vanished_mid_run(_run_dir_text(tmp_path / label)):
            pytest.skip(
                f"run {label} held its GPU, ran kernels on it, then the device "
                f"disappeared: a host/driver event on this shared box, not a "
                f"placement bug (the disjointness above still held)"
            )
        assert proc.returncode == 0, out


@pytest.mark.skipif(_GPU_COUNT < 1, reason="needs a GPU and the NVIDIA Container Toolkit")
def test_a_second_run_fails_fast_while_the_first_holds_every_gpu(tmp_path: Path):
    """Default is fail-fast: an oversubscribed box refuses rather than sharing.

    Without the cross-process registry the second run would happily pack from
    device 0 again and both containers would fight over the same hardware.
    """
    env = _shared_registry_env(tmp_path)
    _force_remove("hog")
    try:
        holder = _start_sflow(
            _HOG_SAMPLE,
            tmp_path / "holder",
            sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT,
                  "CLAIM": _GPU_COUNT, "HOLD": 25},
            env=env,
        )
        _await_reservations(tmp_path, 1)  # holder now owns every GPU
        contender = _start_sflow(
            _HOG_SAMPLE,
            tmp_path / "contender",
            sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT, "CLAIM": 1, "HOLD": 0},
            env=env,
        )
        out_contender, _ = contender.communicate(timeout=_RUN_TIMEOUT)
        out_holder, _ = holder.communicate(timeout=_RUN_TIMEOUT)
    finally:
        _force_remove("hog")

    assert contender.returncode != 0, "second run should not have got a GPU"
    assert re.search(r"only \d+ of \d+ are free", out_contender), out_contender
    # The holder is untouched by the refusal and completes normally.
    assert holder.returncode == 0, out_holder
    assert len(_devices_reserved(out_holder)) == _GPU_COUNT


@pytest.mark.skipif(_GPU_COUNT < 1, reason="needs a GPU and the NVIDIA Container Toolkit")
def test_a_second_run_waits_for_gpus_then_succeeds(tmp_path: Path):
    """`--wait-for-gpus` turns the refusal above into a queue-and-proceed.

    The waiter must also stay responsive: the wait happens on the driver's event
    loop, not inside an uncancellable worker thread.
    """
    env = _shared_registry_env(tmp_path)
    _force_remove("hog")
    try:
        holder = _start_sflow(
            _HOG_SAMPLE,
            tmp_path / "holder",
            sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT,
                  "CLAIM": _GPU_COUNT, "HOLD": 12},
            env=env,
        )
        _await_reservations(tmp_path, 1)  # holder owns the board; the waiter must queue
        waiter = _start_sflow(
            _HOG_SAMPLE,
            tmp_path / "waiter",
            sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT, "CLAIM": 1, "HOLD": 0},
            extra=("--wait-for-gpus", "180"),
            env=env,
        )
        out_waiter, _ = waiter.communicate(timeout=_RUN_TIMEOUT)
        out_holder, _ = holder.communicate(timeout=_RUN_TIMEOUT)
    finally:
        _force_remove("hog")

    assert holder.returncode == 0, out_holder
    assert waiter.returncode == 0, out_waiter
    assert "waiting for" in out_waiter, "waiter should have blocked on a full board"
    assert len(_devices_reserved(out_waiter)) == 1, out_waiter


# ---------------------------------------------------------------------------
# Dynamic occupancy: GPUs moving WITHIN a run, while other runs compete for them
#
# The tests above hold a fixed set of GPUs for a whole run. These drive
# examples/gpu_reservation/pipeline.yaml, whose occupancy changes mid-workflow:
# the server hands its board back at READY (`release_after: task_ready`) and the
# client is placed on those same physical devices while the server still computes
# on them. Sizing makes that load-bearing -- without the hand-back the client
# cannot be scheduled at all (the planner rejects it outright).
#
# Evidence comes from `nvidia-smi` INSIDE each container (physical UUIDs, so
# comparable across runs) rather than from the driver's own log, so what is
# asserted is what the containers really got.
# ---------------------------------------------------------------------------

_PIPELINE_SAMPLE = _GPU_EXAMPLES / "pipeline.yaml"

def _run_dir_text(out_dir: Path) -> str:
    """Everything a background run wrote under its output dir.

    Task output lands in <task>.log rather than the driver's stdout whenever log
    offload is on (the default off a TTY), so the evidence lines these tests parse
    are only reliably found by reading the run directory.
    """
    parts: list[str] = []
    if out_dir.exists():
        for path in out_dir.rglob("*"):
            if path.is_file():
                try:
                    parts.append(path.read_text(errors="replace"))
                except OSError:
                    pass
    return "\n".join(parts)


def _await_reusable_record(tmp_path: Path, *, timeout: float = 300.0) -> set:
    """Block until some run has handed GPUs back at READY; return those UUIDs.

    This is the causal signal the timing-sensitive tests need. Sleeping instead
    would race the image pull and the server's own startup: the contender could
    arrive before the hand-back (testing nothing) or after the whole run finished
    (also testing nothing). The record flipping to reusable=True is exactly the
    moment the in-run hand-back became visible to other processes.
    """
    registry = tmp_path / "registry"
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in registry.glob("*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if record.get("state") in ("lent", "handover"):
                return set(record.get("gpu_uuids") or [])
        time.sleep(0.1)
    pytest.fail(f"no reusable GPU reservation appeared in {registry}")



_PIPELINE_GPU_TASKS = ("pinned_service", "server_a", "server_b", "merged_consumer")

# Smallest board the sample's shape fits on: 1 pinned + 2 (server_a) + 1
# (server_b), with merged_consumer reclaiming server_a's share.
_PIPELINE_BOARD = 4


def _server_a_gpus(board: int) -> int:
    """server_a takes whatever is left after the pinned GPU and server_b's one."""
    return board - 2


def _pipeline_sets(
    *, label: str, board: int | None = None, hold_a: int = 40, hold_b: int = 2,
    pin_hold: int = 200, tail_hold: int = 0,
) -> dict:
    """Sample knobs for one run.

    ``board`` defaults to the WHOLE host, not the sample's 4-GPU default. Tests
    that assert a contender is refused, or that it lands on the one device this
    run freed, only hold when run A actually owns every GPU -- on a bigger host a
    fixed 4-GPU board leaves spares lying around and the contender takes one of
    those instead. server_a absorbs the extra so the shape (1 pinned, 1 held to
    exit, the rest handed over at READY) is unchanged at any size.

    The multi-run contention test passes an explicit smaller board, since there
    the point is for several runs to fit at once.
    """
    board = board if board is not None else _GPU_COUNT
    return {
        "IMAGE": E2E_GPU_IMAGE,
        "GPUS_PER_NODE": board,
        "PINNED_GPUS": 1,
        "SERVER_A_GPUS": _server_a_gpus(board),
        "SERVER_B_GPUS": 1,
        "CONSUMER_GPUS": _server_a_gpus(board),
        "HOLD_A": hold_a,
        "HOLD_B": hold_b,
        "PIN_HOLD": pin_hold,
        "TAIL_HOLD": tail_hold,
        "RUN_LABEL": label,
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


def _host_gpu_uuid_by_index() -> dict[int, str]:
    """Physical index -> UUID, as the driver host sees it.

    The registry logs the devices a run reserved by INDEX, while the containers
    report UUIDs. Comparing what one run reserved against what another run's task
    actually sat on needs this bridge.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    mapping: dict[int, str] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            mapping[int(parts[0])] = parts[1]
    return mapping


def _all_spans(text: str, *, run_end_ms: int) -> dict:
    """Completed spans plus open ones closed at the driver's exit.

    pinned_service is held under `workflow_completion`: it reaches READY, the
    workflow finishes without waiting for it, and its container is torn down -- so
    it never logs an END and only appears via open_spans. Leaving it out would
    exempt the device held LONGEST from the exclusivity check.
    """
    spans = parse_spans(text)
    spans.update(open_spans(text, end=run_end_ms))
    return spans


def _assert_pipeline_shape(text: str, label: str, *, board: int | None = None) -> dict:
    """The per-run invariants of the sample, shared by every pipeline test."""
    board = board if board is not None else _GPU_COUNT
    claims = claimed_uuids(text)
    missing = [t for t in _PIPELINE_GPU_TASKS if f"{label}/{t}" not in claims]
    # A container the driver hands a device it cannot open prints the error where
    # the UUID belongs, which looks identical to a task sflow never placed. Say
    # which it was: only one of the two is a bug in this repo.
    host_fault = any(
        s in text for s in ("Failed to initialize NVML", "No devices were found")
    )
    assert not missing, (
        f"run {label}: no GPU evidence from {missing}"
        + (" -- their containers could not open the device they were given, a "
           "host/driver fault rather than a placement one" if host_fault else "")
    )

    pinned = claims[f"{label}/pinned_service"]
    a = claims[f"{label}/server_a"]
    b = claims[f"{label}/server_b"]
    consumer = claims[f"{label}/merged_consumer"]

    assert (len(pinned), len(a), len(b)) == (1, _server_a_gpus(board), 1), (
        f"run {label}: sizes pinned={sorted(pinned)} a={sorted(a)} b={sorted(b)} "
        f"on a {board}-GPU board"
    )
    # Three distinct halves of one board.
    assert pinned.isdisjoint(a) and pinned.isdisjoint(b) and a.isdisjoint(b), (
        f"run {label}: overlapping claims pinned={sorted(pinned)} "
        f"a={sorted(a)} b={sorted(b)}"
    )
    # Three GPUs were handed back but the consumer takes only server_a's pair, so
    # server_b's device is left over. Asserting the exact identity (not just "2
    # GPUs") is what makes the leftover deterministic enough to test against.
    assert consumer == a, (
        f"run {label}: merged_consumer got {sorted(consumer)}, expected server_a's "
        f"pair {sorted(a)}"
    )
    assert pinned.isdisjoint(consumer), (
        f"run {label}: merged_consumer landed on the pinned device {sorted(pinned)}"
    )
    return claims


@pytest.mark.skipif(
    _GPU_COUNT < _PIPELINE_BOARD,
    reason=f"the pipeline sample needs a {_PIPELINE_BOARD}-GPU board",
)
def test_one_run_pins_one_gpu_and_reshapes_the_rest_as_it_goes(tmp_path: Path):
    """The whole sample in one go, on a board with no slack anywhere.

    pinned_service holds 1 GPU under `workflow_completion`, so nothing may ever be
    placed there. server_a (2) releases at READY and merged_consumer runs on that
    pair WHILE server_a is still computing on it. server_b (1) is held under
    task_completion, so its device stays reserved until that task exits -- which
    is what leaves one GPU idle mid-run for a competing run to pick up.

    Asserting the exact devices (rather than just "got 2 GPUs") is what separates a
    working reuse path from one that wandered onto the pinned device; asserting the
    overlap with server_a is what shows the hand-back happened at READY rather than
    at its exit.
    """
    env = _shared_registry_env(tmp_path)
    out_dir = tmp_path / "pipeline"
    for name in _PIPELINE_GPU_TASKS:
        _force_remove(name)
    try:
        run = _start_sflow(
            _PIPELINE_SAMPLE, out_dir, sets=_pipeline_sets(label="A"), env=env,
        )
        out, _ = run.communicate(timeout=_RUN_TIMEOUT)
        run_end = _now_ms()
    finally:
        for name in _PIPELINE_GPU_TASKS:
            _force_remove(name)

    text = _run_dir_text(out_dir)
    assert run.returncode == 0, out + text

    # The CPU-only steps ran and saw no GPUs at all (sflow's default-none).
    assert "PREP_OK visible_gpus=0" in text, "prep should see zero GPUs"
    assert "REPORT_OK visible_gpus=0" in text, "report should see zero GPUs"
    assert f"MERGED_CONSUMER_GPUS={_server_a_gpus(_GPU_COUNT)}" in text, text

    _assert_pipeline_shape(text, "A")

    spans = _all_spans(text, run_end_ms=run_end)
    for task in ("server_a", "server_b", "merged_consumer"):
        assert f"A/{task}" in spans, f"{task} never finished (no END evidence)"

    consumer = spans["A/merged_consumer"]
    assert consumer.overlaps(spans["A/server_a"]), (
        "merged_consumer did not overlap server_a, so its GPUs came back at that "
        f"server's EXIT rather than at READY: {consumer} vs {spans['A/server_a']}"
    )
    # server_b reports ready later by construction, so the consumer waits for it.
    assert consumer.start > spans["A/server_b"].start, (
        "merged_consumer started before server_b even began"
    )


@pytest.mark.skipif(
    _GPU_COUNT < _PIPELINE_BOARD,
    reason=f"the pipeline sample needs a {_PIPELINE_BOARD}-GPU board",
)
def test_a_gpu_freed_mid_run_is_taken_by_a_concurrent_run(tmp_path: Path):
    """A run that frees a GPU WITHOUT finishing hands it to a competing run.

    server_b exits early, which is a hard release: sflow deletes its registry
    record and the device becomes free to anyone, while run A carries on holding
    the other three (one pinned, two under a live server_a).

    The assertion is the IDENTITY of the GPU the contender gets, and that makes it
    race-free: reservation packs from the lowest free index, so had run A already
    finished, the contender would have taken index 0. Taking server_b's device
    instead is only possible if the other three were still held at that moment.
    """
    env = _shared_registry_env(tmp_path)
    out_dir = tmp_path / "pipeline"
    for name in (*_PIPELINE_GPU_TASKS, "hog"):
        _force_remove(name)
    try:
        holder = _start_sflow(
            _PIPELINE_SAMPLE, out_dir,
            # server_b exits ~2s after reporting ready; server_a keeps its pair for
            # much longer; the CPU tail keeps run A alive well past the handover.
            sets=_pipeline_sets(label="A", hold_a=120, hold_b=2, tail_hold=90),
            env=env,
        )
        _await_reusable_record(tmp_path)
        contender = _start_sflow(
            _HOG_SAMPLE, tmp_path / "contender",
            sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT,
                  "CLAIM": 1, "HOLD": 0},
            extra=("--wait-for-gpus", "300"),
            env=env,
        )
        out_contender, _ = contender.communicate(timeout=_RUN_TIMEOUT)
        holder_alive = holder.poll() is None
        out_holder, _ = holder.communicate(timeout=_RUN_TIMEOUT)
    finally:
        for name in (*_PIPELINE_GPU_TASKS, "hog"):
            _force_remove(name)

    text = _run_dir_text(out_dir)
    assert holder.returncode == 0, out_holder + text
    assert contender.returncode == 0, out_contender
    assert "waiting for" in out_contender, (
        "the contender should have queued: every GPU was taken when it started"
    )
    assert holder_alive, (
        "run A had already finished, so this proves nothing about a mid-run release"
    )

    claims = _assert_pipeline_shape(text, "A")
    taken = _devices_reserved(out_contender)
    assert len(taken) == 1, out_contender
    taken_uuid = _host_gpu_uuid_by_index().get(next(iter(taken)))
    assert taken_uuid in claims["A/server_b"], (
        f"the contender took device {sorted(taken)} ({taken_uuid}), not the one "
        f"server_b freed ({sorted(claims['A/server_b'])}). Devices still held by "
        f"run A: pinned={sorted(claims['A/pinned_service'])} "
        f"server_a={sorted(claims['A/server_a'])}"
    )


@pytest.mark.skipif(
    _GPU_COUNT < _PIPELINE_BOARD,
    reason=f"the pipeline sample needs a {_PIPELINE_BOARD}-GPU board",
)
def test_another_run_cannot_take_gpus_handed_back_for_in_run_reuse(tmp_path: Path):
    """The `task_ready` hand-back is scoped to the run that made it.

    The counterpart to the test above: server_a's pair was also "released", but
    only within run A -- server_a is still computing on it. A contender asking for
    2 GPUs must be refused, even though run A's own consumer was free to take them.
    """
    env = _shared_registry_env(tmp_path)
    out_dir = tmp_path / "pipeline"
    for name in (*_PIPELINE_GPU_TASKS, "hog"):
        _force_remove(name)
    try:
        holder = _start_sflow(
            _PIPELINE_SAMPLE, out_dir,
            # server_b holds its GPU too, so the whole board stays occupied.
            sets=_pipeline_sets(label="A", hold_a=90, hold_b=90, tail_hold=30),
            env=env,
        )
        handed_back = _await_reusable_record(tmp_path)
        outsider = _start_sflow(
            _HOG_SAMPLE, tmp_path / "outsider",
            sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT,
                  "CLAIM": 2, "HOLD": 0},
            env=env,
        )
        out_outsider, _ = outsider.communicate(timeout=_RUN_TIMEOUT)
        out_holder, _ = holder.communicate(timeout=_RUN_TIMEOUT)
    finally:
        for name in (*_PIPELINE_GPU_TASKS, "hog"):
            _force_remove(name)

    assert handed_back, "run A never handed any GPU back at READY"
    assert outsider.returncode != 0, (
        "an unrelated run took GPUs still under run A's live servers:\n"
        + out_outsider
    )
    assert re.search(r"only \d+ of \d+ are free", out_outsider), out_outsider

    text = _run_dir_text(out_dir)
    assert holder.returncode == 0, out_holder + text
    _assert_pipeline_shape(text, "A")


@pytest.mark.skipif(
    _GPU_COUNT < _PIPELINE_BOARD,
    reason=f"the pipeline sample needs a {_PIPELINE_BOARD}-GPU board",
)
def test_competing_runs_with_dynamic_release_never_share_a_gpu(tmp_path: Path):
    """The full contention scenario, checked as an invariant rather than a schedule.

    Every run occupies a whole 4-GPU board and reshapes it as it goes (1 pinned,
    2 handed back at READY and reclaimed, 1 held to exit then freed), and more
    runs are started than the hardware can hold, so some must queue. Which run wins
    the board, and in what order, is genuinely nondeterministic -- asserting a
    particular assignment would just be asserting one legal schedule. What must
    hold for EVERY legal schedule is:

      * every run completes (queued ones proceed; none deadlock or fail),
      * within a run, the consumer takes exactly server_a's pair, never the pinned,
      * across runs, two spans that overlap in time never share a GPU.

    Pinned tasks take part in that last check via their open spans, closed at their
    driver's exit -- otherwise the devices held longest would be the ones left out.
    """
    concurrent = _GPU_COUNT // _PIPELINE_BOARD
    runs = concurrent + 1  # one more than fits -> at least one must queue
    env = _shared_registry_env(tmp_path)
    labels = [chr(ord("A") + i) for i in range(runs)]
    dirs = {label: tmp_path / f"run{label}" for label in labels}

    for name in _PIPELINE_GPU_TASKS:
        _force_remove(name)
    try:
        procs = {}
        for label in labels:
            procs[label] = _start_sflow(
                _PIPELINE_SAMPLE, dirs[label],
                # Short holds: with several runs queueing, each server only needs to
                # outlive its own consumer, not the whole suite.
                sets=_pipeline_sets(
                    label=label, board=_PIPELINE_BOARD,
                    hold_a=12, hold_b=2, pin_hold=90,
                ),
                extra=("--wait-for-gpus", "900"),
                env=env,
            )
            # Stagger so the contention is staged rather than a thundering herd.
            time.sleep(0.5)
        outputs, ends = {}, {}
        for label, proc in procs.items():
            outputs[label] = proc.communicate(timeout=_RUN_TIMEOUT)[0]
            ends[label] = _now_ms()
    finally:
        for name in _PIPELINE_GPU_TASKS:
            _force_remove(name)

    all_spans: dict[str, Span] = {}
    for label in labels:
        text = _run_dir_text(dirs[label])
        assert procs[label].returncode == 0, (
            f"run {label} failed:\n{outputs[label]}\n{text}"
        )
        _assert_pipeline_shape(text, label, board=_PIPELINE_BOARD)
        all_spans.update(_all_spans(text, run_end_ms=ends[label]))

    # At least one run genuinely had to wait, or the scenario never contended.
    assert any("waiting for" in out for out in outputs.values()), (
        "no run ever queued; the board was never oversubscribed so this test did "
        "not exercise contention"
    )

    conflicts = overlapping_spans(list(all_spans.values()))
    assert not conflicts, "\n".join(
        f"runs {first.key} and {second.key} held {sorted(shared)} at the same "
        f"time: {first} vs {second}"
        for first, second, shared in conflicts
    )


# ---------------------------------------------------------------------------
# The hand-over race: a completing task must not publish GPUs its own run needs
# ---------------------------------------------------------------------------
#
# A task released under `task_completion` used to have its registry record DELETED
# the moment it finished. Its successor is not submitted until the next poll tick,
# and a concurrent `sflow run` sitting in --wait-for-gpus takes the device in that
# gap -- failing the successor with "0 free" on a placement the planner had
# already validated. Reproduced before the fix: the workflow failed every time.
#
# Only needs a single GPU, so unlike the pipeline cases this runs on almost any
# GPU host.

_RACE_WORKFLOW = """
version: "0.1"
backends:
  - name: docker
    type: docker
    default: true
    image: {image}
    nodes: 1
    gpus_per_node: {board}
    extra_args: ["--entrypoint="]
workflow:
  name: handover_race
  tasks:
    # No readiness probe -> release_after is inferred task_completion. Holds the
    # WHOLE board, then exits.
    - name: raceholder
      resources:
        gpus:
          count: {board}
      script:
        - 'echo "HOLDER_START $(date +%s%3N)"'
        - '/cuda-samples/vectorAdd'
        - 'sleep {hold}'
        - 'echo "HOLDER_END $(date +%s%3N)"'
    # Planned onto raceholder's devices -- and it needs ALL of them, so a single
    # GPU lost to a contender in the hand-over window is enough to sink it.
    - name: racesuccessor
      depends_on: [raceholder]
      resources:
        gpus:
          count: {board}
      script:
        - 'echo "SUCCESSOR_GOT_GPU $(date +%s%3N)"'
        - '/cuda-samples/vectorAdd'
"""

_RACE_TASKS = ("raceholder", "racesuccessor")


# A single contender is NOT enough to catch this: it retries on a ~5s clock while
# the vulnerable window is only about a poll tick wide, so it usually misses and
# the test passes even against the broken build (verified -- the first draft of
# this test did exactly that). Several contenders started a second apart put their
# retry phases out of step, so between them they probe the window every ~1s.
_RACE_CONTENDERS = 5
_RACE_CONTENDER_STAGGER_S = 1.0


@pytest.mark.skipif(_GPU_COUNT < 1, reason="needs a GPU and the NVIDIA Container Toolkit")
def test_a_completing_task_does_not_lose_its_successors_gpu_to_waiting_runs(
    tmp_path: Path,
):
    """The successor must win the device its own planner reserved for it.

    Several contenders are already queued on --wait-for-gpus when raceholder
    completes, deliberately out of phase so one of them is polling during the
    hand-over window. Against the pre-fix build the workflow loses the GPU and
    fails; with the hand-over it keeps it, and the contenders are served after.
    """
    env = _shared_registry_env(tmp_path)
    cfg = tmp_path / "race.yaml"
    cfg.write_text(
        _RACE_WORKFLOW.format(
            image=E2E_GPU_IMAGE, hold=12, board=_GPU_COUNT
        ).strip()
    )
    out_dir = tmp_path / "holder_run"

    for name in (*_RACE_TASKS, "hog"):
        _force_remove(name)
    contenders = []
    try:
        holder = _start_sflow(cfg, out_dir, sets={}, env=env)
        # The workflow owns the board before any contender is allowed to ask.
        _await_reservations(tmp_path, 1)
        for i in range(_RACE_CONTENDERS):
            contenders.append(
                _start_sflow(
                    _HOG_SAMPLE, tmp_path / f"contender{i}",
                    sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT,
                          "CLAIM": 1, "HOLD": 0},
                    extra=("--wait-for-gpus", "240"),
                    env=env,
                )
            )
            time.sleep(_RACE_CONTENDER_STAGGER_S)
        out_holder, _ = holder.communicate(timeout=_RUN_TIMEOUT)
        contender_out = [c.communicate(timeout=_RUN_TIMEOUT)[0] for c in contenders]
    finally:
        for name in (*_RACE_TASKS, "hog"):
            _force_remove(name)

    text = _run_dir_text(out_dir)
    # The regression this exists for: the workflow itself fails, because its own
    # successor could not get the GPU its predecessor had just released.
    assert holder.returncode == 0, (
        "the workflow lost its successor's GPU to a concurrent run:\n"
        + out_holder + text
    )
    assert "SUCCESSOR_GOT_GPU" in text, (
        "racesuccessor never ran, so the hand-over did not hold the device"
    )
    # The contenders really were competing, not just idling past the window.
    assert sum("waiting for" in out for out in contender_out) >= 1, (
        "no contender ever queued, so none was competing for the window"
    )
    # ...and it is a hand-over, not a lockout: they are all served afterwards.
    for i, (proc, out) in enumerate(zip(contenders, contender_out)):
        assert proc.returncode == 0, f"contender {i} was starved:\n{out}"


@pytest.mark.skipif(_GPU_COUNT < 1, reason="needs a GPU and the NVIDIA Container Toolkit")
def test_a_waiting_run_only_gets_the_gpu_after_the_workflow_is_done_with_it(
    tmp_path: Path,
):
    """Ordering, checked from the container's own clock: the successor goes first,
    and only then does an outside run get the device."""
    env = _shared_registry_env(tmp_path)
    cfg = tmp_path / "race.yaml"
    cfg.write_text(
        _RACE_WORKFLOW.format(
            image=E2E_GPU_IMAGE, hold=8, board=_GPU_COUNT
        ).strip()
    )
    out_dir = tmp_path / "holder_run"

    for name in (*_RACE_TASKS, "hog"):
        _force_remove(name)
    try:
        holder = _start_sflow(cfg, out_dir, sets={}, env=env)
        _await_reservations(tmp_path, 1)
        contender = _start_sflow(
            _HOG_SAMPLE, tmp_path / "contender",
            sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT,
                  "CLAIM": 1, "HOLD": 0},
            extra=("--wait-for-gpus", "180"),
            env=env,
        )
        out_holder, _ = holder.communicate(timeout=_RUN_TIMEOUT)
        successor_done = _now_ms()
        out_contender, _ = contender.communicate(timeout=_RUN_TIMEOUT)
    finally:
        for name in (*_RACE_TASKS, "hog"):
            _force_remove(name)

    text = _run_dir_text(out_dir)
    assert holder.returncode == 0, out_holder + text
    assert contender.returncode == 0, out_contender

    stamps = {
        key: int(value)
        for key, value in re.findall(r"(HOLDER_END|SUCCESSOR_GOT_GPU) (\d+)", text)
    }
    assert {"HOLDER_END", "SUCCESSOR_GOT_GPU"} <= set(stamps), f"got {stamps}"
    assert stamps["SUCCESSOR_GOT_GPU"] > stamps["HOLDER_END"], (
        "the successor cannot have started before its predecessor finished"
    )
    assert stamps["SUCCESSOR_GOT_GPU"] <= successor_done


# ---------------------------------------------------------------------------
# A READY hand-back gives away the claim, not the occupancy
# ---------------------------------------------------------------------------
#
# `release_after: task_ready` lets a later task of the SAME run take the devices
# while the server keeps serving on them. Once that consumer finished, its
# ordinary hard release used to publish the devices to the whole host -- the
# server's own record had been dropped when the consumer superseded it, so
# nothing was left saying "a live task is sitting here". A concurrent run then
# landed on a GPU an inference server was still using.
#
# The pipeline sample covers this on a 4-GPU board; this is the same invariant
# on one GPU, so it runs on almost any GPU host (see the hand-over race above).

_LIVE_REUSE_WORKFLOW = """
version: "0.1"
backends:
  - name: docker
    type: docker
    default: true
    image: {image}
    nodes: 1
    gpus_per_node: {board}
    extra_args: ["--entrypoint="]
workflow:
  name: live_server_reuse
  tasks:
    # Hands its devices back at READY and keeps serving on them well past the
    # consumer, so anything that takes them afterwards is taking a live GPU.
    - name: liveserver
      resources:
        gpus:
          count: {board}
          release_after: task_ready
      probes:
        readiness:
          log_watch:
            match_pattern: "SERVER_READY"
          interval: 1
          timeout: 300
      script:
        - '/cuda-samples/vectorAdd'
        - 'echo SERVER_READY'
        - 'sleep {hold}'
    # Planned onto liveserver's devices while it is still serving -- the reuse
    # `release_after: task_ready` exists for.
    - name: reuser
      depends_on: [liveserver]
      resources:
        gpus:
          count: {board}
      script:
        - '/cuda-samples/vectorAdd'
        - 'echo REUSER_DONE'
    # CPU-only, so it holds no devices: it exists to keep the run alive after the
    # consumer is gone. Without it the workflow would end there and the devices
    # really would be free, which is a different situation entirely.
    #
    # Named distinctively for the same reason every other task in this file is:
    # cleanup matches container names by SUBSTRING, so a plain `tail` would make
    # _force_remove kill any container on this host with "tail" in its name --
    # including a co-tenant's on the shared CI box.
    - name: reusetail
      depends_on: [reuser]
      script:
        - 'sleep {tail}'
"""

# Every task the workflow launches, including the CPU-only one: it outlives the
# consumer by design, so an early exit (a failed await, a timeout) would leave it
# running for its whole sleep and leak into the tests that follow.
_LIVE_REUSE_TASKS = ("liveserver", "reuser", "reusetail")


@pytest.mark.skipif(_GPU_COUNT < 1, reason="needs a GPU and the NVIDIA Container Toolkit")
def test_a_live_servers_gpu_is_not_published_when_its_reuser_finishes(tmp_path: Path):
    """The device stays this run's until the SERVER exits, not until the reuser does.

    The contender is only allowed to ask once the reuser has been and gone, which
    is exactly the window the bug opened: against the pre-fix build it is handed
    liveserver's GPU immediately (verified -- the registry is empty at that
    moment). It must instead wait out its budget and fail, because the only task
    still on that device has not finished.
    """
    env = _shared_registry_env(tmp_path)
    cfg = tmp_path / "reuse.yaml"
    cfg.write_text(
        _LIVE_REUSE_WORKFLOW.format(
            image=E2E_GPU_IMAGE, board=_GPU_COUNT, hold=90, tail=75
        ).strip()
    )
    out_dir = tmp_path / "server_run"

    for name in (*_LIVE_REUSE_TASKS, "hog"):
        _force_remove(name)
    try:
        holder = _start_sflow(cfg, out_dir, sets={}, env=env)
        # Causal, not timed: the reuser must have claimed the handed-back device
        # AND released it. Its release is the exact moment the bug opened -- before
        # it the device is held by a claim of its own and the test would prove
        # nothing; killing the container to get there would just fail the run.
        registry = tmp_path / "registry"
        _await_record(registry, "*-reuser.json", present=True)
        _await_record(registry, "*-reuser.json", present=False)

        contender = _start_sflow(
            _HOG_SAMPLE, tmp_path / "contender",
            sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _GPU_COUNT,
                  "CLAIM": 1, "HOLD": 0},
            extra=("--wait-for-gpus", "20"),
            env=env,
        )
        out_contender, _ = contender.communicate(timeout=_RUN_TIMEOUT)
        server_alive = holder.poll() is None
        out_holder, _ = holder.communicate(timeout=_RUN_TIMEOUT)
    finally:
        for name in (*_LIVE_REUSE_TASKS, "hog"):
            _force_remove(name)

    text = _run_dir_text(out_dir)
    assert server_alive, (
        "the run had already finished, so this proves nothing about a live server"
    )
    assert contender.returncode != 0, (
        "a concurrent run took a GPU the live server is still serving on:\n"
        + out_contender
    )
    assert re.search(r"only \d+ of \d+ are free", out_contender), out_contender
    # The reuse really happened -- otherwise the contender was refused by a
    # device nobody ever handed back, which is a different (passing) story.
    assert "REUSER_DONE" in text, "the reuser never ran on its predecessor's GPU"
    # The run itself is unharmed: holding is not a deadlock.
    assert holder.returncode == 0, out_holder + text


# ---------------------------------------------------------------------------
# The scheduler under sustained contention, on a full board
# ---------------------------------------------------------------------------

# What each task asks for, which is also the task list and the board size.
_SCHEDULING_GPUS = {
    "tp2_a": 2, "tp2_b": 2, "tp2_c": 2, "tp2_d": 2,
    "tp2_e": 2, "tp2_f": 2, "tp2_g": 2, "tp2_h": 2,
    "tp4_a": 4, "tp4_b": 4,
    "tp8": 8,
}
_SCHEDULING_TASKS = tuple(_SCHEDULING_GPUS)
_SCHEDULING_BOARD = _SCHEDULING_GPUS["tp8"]


@pytest.mark.skipif(
    _GPU_COUNT < _SCHEDULING_BOARD,
    reason=f"the scheduling smoke needs a {_SCHEDULING_BOARD}-GPU board",
)
def test_the_scheduler_never_double_books_a_gpu_across_four_waves(tmp_path: Path):
    """Eleven tasks drain and refill a full board; no two may share a device.

    Every task holds until it exits, so at no point is a device legitimately
    shared -- unlike the pipeline sample, where `task_ready` reuse makes same-run
    overlap the expected behavior. That makes the invariant flat: for any two
    tasks whose spans overlap in time, their device sets are disjoint.

    Checked on UUIDs reported from inside each container, because
    CUDA_VISIBLE_DEVICES is renumbered to 0..N-1 there and would make every task
    look like it got the same cards.
    """
    env = _shared_registry_env(tmp_path)
    out_dir = tmp_path / "scheduling"

    for name in _SCHEDULING_TASKS:
        _force_remove(name)
    try:
        run = _start_sflow(
            _SCHEDULING_SAMPLE, out_dir,
            # Pin the board rather than following the host: the sample's waves are
            # sized for exactly 8, and on a bigger box the later waves would have
            # spare devices to spread onto instead of having to reuse.
            sets={"IMAGE": E2E_GPU_IMAGE, "GPUS_PER_NODE": _SCHEDULING_BOARD},
            env=env,
        )
        out, _ = run.communicate(timeout=_RUN_TIMEOUT)
    finally:
        for name in _SCHEDULING_TASKS:
            _force_remove(name)

    text = _run_dir_text(out_dir)
    assert run.returncode == 0, out + text

    spans = parse_spans(text)
    missing = [t for t in _SCHEDULING_TASKS if f"S/{t}" not in spans]
    assert not missing, f"no start/end GPU evidence from {missing}"

    for task, want in _SCHEDULING_GPUS.items():
        got = spans[f"S/{task}"].uuids
        assert len(got) == want, f"{task} got {len(got)} GPU(s), asked for {want}"

    conflicts = overlapping_spans(list(spans.values()), same_run_ok=False)
    assert not conflicts, "\n".join(
        f"{first.task} and {second.task} both held {sorted(shared)} at the same "
        f"time: {first} vs {second}"
        for first, second, shared in conflicts
    )
