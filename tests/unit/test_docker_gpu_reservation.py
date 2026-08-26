# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-task GPU reservation for the docker backend.

GPUs are reserved by the docker_run operator's ``acquire_resources`` just before
a task launches and freed in ``release_resources`` after it ends (the k8s per-pod
model), not for the whole run. nvidia-smi is stubbed via ``discover_gpus`` and the
flock registry runs against a temp dir, so reserve/release run for real w/o a GPU.
"""

import asyncio
import json

import pytest

import sflow.utils.gpu_reservation as gr
import sflow.plugins.operators.docker_run as docker_run_mod
from sflow.core.operator import ResourcesUnavailable
from sflow.utils.gpu_reservation import GpuHandle
from sflow.plugins.backends.docker import DockerBackend, DockerBackendConfig


@pytest.fixture(autouse=True)
def _registry(tmp_path, monkeypatch):
    monkeypatch.setenv("SFLOW_GPU_RESERVATION_DIR", str(tmp_path / "reg"))
    monkeypatch.setenv("SFLOW_GPU_IGNORE_FOREIGN", "1")
    monkeypatch.delenv("SFLOW_GPU_RESERVATION", raising=False)
    monkeypatch.delenv("SFLOW_WAIT_FOR_GPUS", raising=False)


def _patch_gpus(monkeypatch, n):
    gpus = [GpuHandle(index=i, uuid=f"GPU-{i:04d}") for i in range(n)]
    monkeypatch.setattr(gr, "discover_gpus", lambda: list(gpus))


def _records(tmp_path):
    return list((tmp_path / "reg").glob("*.json"))


def _local_op(gpu_count=2, *, name="docker", node="localhost", **backend_kw):
    """A docker_run operator wired to a local single node needing ``gpu_count`` GPUs."""
    backend = DockerBackend(
        DockerBackendConfig(name=name, type="docker", image="ubuntu:22.04", **backend_kw)
    )
    op = backend.default_operator(name=f"{name}_op", assigned_nodes=[node])
    op.apply_backend_context(
        backend=backend, assigned_nodes=[node], artifacts=[], gpu_count=gpu_count
    )
    return op


# ---------------------------------------------------------------------------
# acquire / release
# ---------------------------------------------------------------------------


def test_acquire_reserves_and_pins_gpus(monkeypatch, tmp_path):
    _patch_gpus(monkeypatch, 8)
    op = _local_op(gpu_count=2)
    op.acquire_resources(task_name="t", envs={})
    assert op.config.gpus == "device=GPU-0000,GPU-0001"
    assert len(_records(tmp_path)) == 1


def test_release_frees_the_reservation(monkeypatch, tmp_path):
    _patch_gpus(monkeypatch, 8)
    op = _local_op(gpu_count=2)
    op.acquire_resources(task_name="t", envs={})
    assert _records(tmp_path)
    op.release_resources(task_name="t")
    assert _records(tmp_path) == []
    assert op._reservation_run_id is None


def test_release_is_idempotent(monkeypatch):
    _patch_gpus(monkeypatch, 2)
    op = _local_op(gpu_count=1)
    op.release_resources(task_name="t")  # never acquired -> no-op
    op.acquire_resources(task_name="t", envs={})
    op.release_resources(task_name="t")
    op.release_resources(task_name="t")  # double release ok


def test_two_concurrent_tasks_get_disjoint_gpus(monkeypatch):
    # Distinct task names -> distinct run_ids; each reserves its own GPUs.
    _patch_gpus(monkeypatch, 8)
    a = _local_op(gpu_count=2, name="a")
    b = _local_op(gpu_count=2, name="b")
    a.acquire_resources(task_name="task_a", envs={})
    b.acquire_resources(task_name="task_b", envs={})
    assert a.config.gpus == "device=GPU-0000,GPU-0001"
    assert b.config.gpus == "device=GPU-0002,GPU-0003"


def test_task_gpus_freed_for_the_next_task(monkeypatch):
    # The whole point of per-task: a task's GPUs return to the pool when it ends,
    # so a later task can reuse them even though the board was momentarily full.
    _patch_gpus(monkeypatch, 2)
    a = _local_op(gpu_count=2, name="a")
    a.acquire_resources(task_name="task_a", envs={})
    b = _local_op(gpu_count=2, name="b")
    with pytest.raises(RuntimeError):
        b.acquire_resources(task_name="task_b", envs={})  # board full
    a.release_resources(task_name="task_a")
    b.acquire_resources(task_name="task_b", envs={})  # now fits
    assert b.config.gpus == "device=GPU-0000,GPU-0001"


def test_insufficient_gpus_raises_runtime_error(monkeypatch):
    _patch_gpus(monkeypatch, 2)
    op = _local_op(gpu_count=8)
    with pytest.raises(RuntimeError, match="only 2 of 2 are free"):
        op.acquire_resources(task_name="t", envs={})


def test_zero_gpu_task_reserves_nothing(monkeypatch, tmp_path):
    _patch_gpus(monkeypatch, 8)
    op = _local_op(gpu_count=0)
    op.acquire_resources(task_name="t", envs={})
    assert op.config.gpus is None  # build_command then hides GPUs via void
    assert _records(tmp_path) == []


def test_disabled_reservation_falls_back_to_plan_time_slice(monkeypatch, tmp_path):
    # SFLOW_GPU_RESERVATION=0 must fall back to the plan-time numeric slice (old
    # behavior), NOT leave a local GPU task pinned to no GPUs.
    monkeypatch.setenv("SFLOW_GPU_RESERVATION", "0")
    _patch_gpus(monkeypatch, 8)
    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04")
    )
    op = backend.default_operator(name="op", assigned_nodes=["localhost"])
    op.apply_backend_context(
        backend=backend,
        assigned_nodes=["localhost"],
        artifacts=[],
        cuda_visible_devices="0,1",
        gpu_count=2,
    )
    assert op.config.gpus == "device=0,1"  # plan-time slice, not void
    op.acquire_resources(task_name="t", envs={})  # no-op when disabled
    assert op.config.gpus == "device=0,1"
    assert _records(tmp_path) == []


def test_remote_host_task_keeps_plan_time_slice(monkeypatch, tmp_path):
    # Remote host: no local registry -> --gpus is the plan-time numeric slice,
    # set in apply_backend_context; acquire is a no-op there.
    _patch_gpus(monkeypatch, 8)
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[{"name": "dgx-a", "docker_host": "ssh://dgx-a"}],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    op = backend.default_operator(name="op")
    op.apply_backend_context(
        backend=backend,
        assigned_nodes=["dgx-a"],
        artifacts=[],
        cuda_visible_devices="0,1",
        gpu_count=2,
    )
    assert op.config.gpus == "device=0,1"
    op.acquire_resources(task_name="t", envs={})
    assert op.config.gpus == "device=0,1"
    assert _records(tmp_path) == []


def test_allocate_does_not_reserve(monkeypatch, tmp_path):
    # Run-level allocate no longer touches the registry (per-task now).
    _patch_gpus(monkeypatch, 8)
    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04", gpus_per_node=2)
    )
    allocation = asyncio.run(backend.allocate())
    assert allocation.nodes[0].num_gpus == 2  # planning capacity only
    assert _records(tmp_path) == []
    assert asyncio.run(backend.release(allocation)) is None


def test_resource_env_exposes_virtual_indices_not_uuids():
    # A task asking for N GPUs sees CUDA_VISIBLE_DEVICES=0..N-1 inside the
    # container; the physical UUIDs stay on the host-side --gpus flag.
    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04", gpus_per_node=4)
    )
    assert backend.resource_env(cuda_visible_devices="GPU-aaaa,GPU-bbbb")[
        "CUDA_VISIBLE_DEVICES"
    ] == "0,1"
    assert backend.resource_env(cuda_visible_devices=None) == {}


# ---------------------------------------------------------------------------
# wait_for_gpus (config field + env override), via the operator
# ---------------------------------------------------------------------------


def test_wait_options_config_value_semantics():
    assert gr.wait_options(None) == (False, None)  # fail fast
    assert gr.wait_options(0) == (True, None)  # wait forever
    assert gr.wait_options(30) == (True, 30.0)  # wait 30s
    with pytest.raises(ValueError):
        gr.wait_options(-1)


def test_wait_options_env_valid(monkeypatch):
    monkeypatch.setenv("SFLOW_WAIT_FOR_GPUS", "")  # empty = wait forever
    assert gr.wait_options() == (True, None)
    monkeypatch.setenv("SFLOW_WAIT_FOR_GPUS", "30")
    assert gr.wait_options() == (True, 30.0)


def test_wait_options_env_rejects_malformed(monkeypatch):
    # A typo like `--wait-for-gpus 600s` must fail loudly, not wait forever.
    monkeypatch.setenv("SFLOW_WAIT_FOR_GPUS", "600s")
    with pytest.raises(ValueError, match="non-negative number"):
        gr.wait_options()
    monkeypatch.setenv("SFLOW_WAIT_FOR_GPUS", "-5")
    with pytest.raises(ValueError, match=">= 0"):
        gr.wait_options()


def _full_board(monkeypatch, gpus=1):
    """Occupy every GPU, and return a fresh operator that wants one."""
    _patch_gpus(monkeypatch, gpus)
    holder = _local_op(gpu_count=gpus, name="holder")
    holder.acquire_resources(task_name="holder", envs={})
    return _local_op(gpu_count=1, name="waiter")


def test_no_wait_configured_fails_immediately(monkeypatch):
    # Default (fail fast): a full board is terminal, never retryable.
    waiter = _full_board(monkeypatch)
    with pytest.raises(RuntimeError, match="only 0 of 1 are free") as ei:
        waiter.acquire_resources(task_name="waiter", envs={})
    assert not isinstance(ei.value, ResourcesUnavailable)


def test_config_wait_for_gpus_makes_a_full_board_retryable(monkeypatch):
    """The recipe's wait_for_gpus turns "full" into "try again", not a failure.

    acquire_resources must NOT sleep -- it runs in an uncancellable worker
    thread. It signals, and the orchestrator waits on the event loop.
    """
    waiter = _full_board(monkeypatch)
    waiter._wait_for_gpus = 30  # recipe-level wait_for_gpus: 30
    with pytest.raises(ResourcesUnavailable) as ei:
        waiter.acquire_resources(task_name="waiter", envs={})
    assert ei.value.retry_after > 0


def test_env_wait_overrides_config(monkeypatch):
    # --wait-for-gpus "" (forever) wins over a recipe that said fail-fast.
    monkeypatch.setenv("SFLOW_WAIT_FOR_GPUS", "")
    waiter = _full_board(monkeypatch)
    assert waiter._wait_for_gpus is None  # recipe: fail fast
    with pytest.raises(ResourcesUnavailable):
        waiter.acquire_resources(task_name="waiter", envs={})


def test_wait_budget_is_shared_across_retries_and_then_terminal(monkeypatch):
    """The deadline starts on the first attempt, so retries can't extend it."""
    monkeypatch.setenv("SFLOW_WAIT_FOR_GPUS", "60")
    waiter = _full_board(monkeypatch)

    now = [1000.0]
    monkeypatch.setattr(docker_run_mod.time, "monotonic", lambda: now[0])
    with pytest.raises(ResourcesUnavailable):
        waiter.acquire_resources(task_name="waiter", envs={})  # starts the budget
    deadline = waiter._wait_deadline
    assert deadline == 1060.0

    now[0] = 1030.0                                            # half-way: still retryable
    with pytest.raises(ResourcesUnavailable):
        waiter.acquire_resources(task_name="waiter", envs={})
    assert waiter._wait_deadline == deadline, "retry must not extend the budget"

    now[0] = 1061.0                                            # past it: terminal
    with pytest.raises(RuntimeError, match="only 0 of 1 are free") as ei:
        waiter.acquire_resources(task_name="waiter", envs={})
    assert not isinstance(ei.value, ResourcesUnavailable)


def test_acquire_never_sleeps(monkeypatch):
    """A blocking sleep here would detach from the task and hang driver exit."""
    waiter = _full_board(monkeypatch)
    waiter._wait_for_gpus = 0  # wait forever
    for mod in (gr, docker_run_mod):
        monkeypatch.setattr(
            mod.time,
            "sleep",
            lambda s: pytest.fail(f"acquire_resources slept {s}s in a worker thread"),
        )
    with pytest.raises(ResourcesUnavailable):
        waiter.acquire_resources(task_name="waiter", envs={})


# ---------------------------------------------------------------------------
# fallbacks: no nvidia-smi, multi-node, and what acquire reports back
# ---------------------------------------------------------------------------


def test_acquire_returns_the_physical_device_indices(monkeypatch):
    # The orchestrator records these on the task so run reporting can name the
    # GPUs actually used (the container only ever sees virtual 0..N-1).
    _patch_gpus(monkeypatch, 8)
    gr.try_reserve_gpus(2, "someone-else")  # pushes us onto GPUs 2,3
    op = _local_op(gpu_count=2)
    assert op.acquire_resources(task_name="t", envs={}) == [2, 3]
    assert op.config.gpus == "device=GPU-0002,GPU-0003"


def test_no_nvidia_smi_falls_back_to_plan_slice_instead_of_failing(monkeypatch):
    # The driver may not see nvidia-smi (e.g. it runs in a container with only
    # the docker socket) while the daemon still exposes GPUs fine. That must
    # degrade to the planned slice, not fail a workload that would have run.
    monkeypatch.setattr(gr, "discover_gpus", lambda: [])
    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04")
    )
    op = backend.default_operator(name="op", assigned_nodes=["localhost"])
    op.apply_backend_context(
        backend=backend,
        assigned_nodes=["localhost"],
        artifacts=[],
        cuda_visible_devices="0,1",
        gpu_count=2,
    )
    assert op.acquire_resources(task_name="t", envs={}) is None
    assert op.config.gpus == "device=0,1"


def test_local_multi_node_keeps_the_plan_time_slice(monkeypatch):
    # Synthetic multi-node placements can't use this host's registry (every
    # "node" is the same machine), so they keep the planner's numeric slice.
    _patch_gpus(monkeypatch, 8)
    backend = DockerBackend(
        DockerBackendConfig(
            name="d", type="docker", image="ubuntu:22.04", nodes=2, gpus_per_node=4
        )
    )
    op = backend.default_operator(name="op")
    op.apply_backend_context(
        backend=backend,
        assigned_nodes=["d-node0", "d-node1"],
        artifacts=[],
        cuda_visible_devices="0,1",
        gpu_count=2,
    )
    assert op.config.gpus == "device=0,1"
    assert op.acquire_resources(task_name="t", envs={}) is None
    assert op.config.gpus == "device=0,1"


def test_release_restores_the_plan_time_slice(monkeypatch):
    # After release the UUID pin names GPUs the task no longer holds; a reused
    # operator instance must never launch a second container against them.
    _patch_gpus(monkeypatch, 8)
    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04")
    )
    op = backend.default_operator(name="op", assigned_nodes=["localhost"])
    op.apply_backend_context(
        backend=backend,
        assigned_nodes=["localhost"],
        artifacts=[],
        cuda_visible_devices="0,1",
        gpu_count=2,
    )
    op.acquire_resources(task_name="t", envs={})
    assert op.config.gpus == "device=GPU-0000,GPU-0001"
    op.release_resources(task_name="t")
    assert op.config.gpus == "device=0,1"


def test_dry_run_shows_the_gpu_pinning_a_real_run_applies(monkeypatch):
    # apply_backend_context runs at plan time; --dry-run renders operator config,
    # so --gpus must already be populated there rather than only at launch.
    _patch_gpus(monkeypatch, 8)
    op = _local_op(gpu_count=0)  # no reservation involved
    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04")
    )
    op.apply_backend_context(
        backend=backend,
        assigned_nodes=["localhost"],
        artifacts=[],
        cuda_visible_devices="4,5",
        gpu_count=2,
    )
    assert op.config.gpus == "device=4,5"


def test_a_host_with_no_gpus_is_never_retried(monkeypatch):
    """`--wait-for-gpus` must not poll a host that reports no GPUs at all.

    Waiting could never succeed there, so the operator falls back to the planned
    slice instead of signalling a retry -- otherwise a driver-side nvidia-smi
    problem would look like a busy board and spin until the budget expired.
    """
    monkeypatch.setenv("SFLOW_WAIT_FOR_GPUS", "")  # wait forever, if we waited
    monkeypatch.setattr(gr, "discover_gpus", lambda: [])
    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04")
    )
    op = backend.default_operator(name="op", assigned_nodes=["localhost"])
    op.apply_backend_context(
        backend=backend,
        assigned_nodes=["localhost"],
        artifacts=[],
        cuda_visible_devices="0,1",
        gpu_count=2,
    )
    assert op.acquire_resources(task_name="t", envs={}) is None  # not retryable
    assert op.config.gpus == "device=0,1"  # fell back to the planned slice


def test_cpu_only_task_never_touches_the_gpu_registry(monkeypatch, tmp_path):
    """A container that asks for no GPUs should leave no trace on the host.

    Releasing unconditionally would take the registry lock, and taking the lock
    creates the (world-writable, sticky) registry directory -- on every docker
    run, GPUs or not.
    """
    reg = tmp_path / "reg"
    monkeypatch.setenv("SFLOW_GPU_RESERVATION_DIR", str(reg))
    op = _local_op(gpu_count=0)
    op.acquire_resources(task_name="cpu_task", envs={})
    op.release_resources(task_name="cpu_task")
    assert not reg.exists(), "CPU-only task created the GPU reservation registry"


# ---------------------------------------------------------------------------
# run-end backstop (DockerBackend.release_resources)
# ---------------------------------------------------------------------------


def test_backend_release_still_performs_the_base_allocation_teardown(monkeypatch):
    """Overriding release_resources must ADD to the base, not replace it.

    Every backend's release clears `allocation` (and calls `release()`); the
    docker override only bolts the GPU-registry sweep on top. Dropping the
    `super()` call left docker reporting itself as still allocated after
    teardown -- an invariant the other backends' tests already assert.
    """
    _patch_gpus(monkeypatch, 8)
    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04")
    )
    backend.allocation = asyncio.run(backend.allocate())
    assert backend.allocation is not None

    asyncio.run(backend.release_resources())

    assert backend.allocation is None, "base teardown was skipped"


def test_backend_release_sweeps_records_this_run_left_behind(monkeypatch, tmp_path):
    """The backstop clears reservations a cancelled task could not release."""
    _patch_gpus(monkeypatch, 8)
    op = _local_op(gpu_count=2)
    op.acquire_resources(task_name="t", envs={})
    assert len(_records(tmp_path)) == 1
    op._reservation_run_id = None  # simulate: cancelled before release could see it

    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04")
    )
    backend.allocation = asyncio.run(backend.allocate())
    asyncio.run(backend.release_resources())

    assert _records(tmp_path) == [], "run-end sweep left a reservation behind"


def test_backend_release_leaves_other_processes_reservations_alone(monkeypatch, tmp_path):
    """The sweep is scoped to THIS pid -- a co-tenant's records must survive."""
    _patch_gpus(monkeypatch, 8)
    d = gr._ensure_registry_dir()
    (d / "other.json").write_text(
        json.dumps(
            {"run_id": "other", "pid": 999999998, "gpu_uuids": ["GPU-0007"]}
        )
    )
    backend = DockerBackend(
        DockerBackendConfig(name="d", type="docker", image="ubuntu:22.04")
    )
    backend.allocation = asyncio.run(backend.allocate())
    asyncio.run(backend.release_resources())

    assert (d / "other.json").exists(), "swept another process's reservation"


def test_run_end_sweep_does_not_create_the_registry(monkeypatch, tmp_path):
    """A GPU-less docker run must not leave an empty registry dir behind."""
    reg = tmp_path / "never_used"
    monkeypatch.setenv("SFLOW_GPU_RESERVATION_DIR", str(reg))
    assert gr.release_all_for_pid() == 0
    assert not reg.exists()


# ---------------------------------------------------------------------------
# Telling the user when the safety net is NOT under them
# ---------------------------------------------------------------------------


def test_skipped_reservation_on_a_multi_node_placement_is_announced(
    monkeypatch, caplog
):
    """A silent `return None` reads as "reservation applied" to anyone following
    the docs. Every container of a synthetic `nodes: 2` placement runs on THIS
    daemon, so the user is unprotected on the very host they expect coverage on --
    say so."""
    _patch_gpus(monkeypatch, 8)
    backend = DockerBackend(
        DockerBackendConfig(
            name="d", type="docker", image="ubuntu:22.04", nodes=2, gpus_per_node=4
        )
    )
    op = backend.default_operator(name="op")
    op.apply_backend_context(
        backend=backend,
        assigned_nodes=["d-node0", "d-node1"],
        artifacts=[],
        cuda_visible_devices="0,1",
        gpu_count=2,
    )
    with caplog.at_level("WARNING"):
        assert op.acquire_resources(task_name="t", envs={}) is None
    assert "GPU reservation skipped" in caplog.text
    assert "spans 2 nodes" in caplog.text

    # Warned once, not on every retry/relaunch.
    caplog.clear()
    with caplog.at_level("WARNING"):
        op.acquire_resources(task_name="t", envs={})
    assert "GPU reservation skipped" not in caplog.text


def test_remote_host_skip_is_announced_too(monkeypatch, caplog):
    _patch_gpus(monkeypatch, 8)
    backend = DockerBackend(
        DockerBackendConfig(
            name="d",
            type="docker",
            image="ubuntu:22.04",
            hosts=[{"name": "gpu-a", "docker_host": "ssh://gpu-a"}],
            gpus_per_node=8,
        )
    )
    op = backend.default_operator(name="op", assigned_nodes=["gpu-a"])
    op.apply_backend_context(
        backend=backend,
        assigned_nodes=["gpu-a"],
        artifacts=[],
        cuda_visible_devices="0,1",
        gpu_count=2,
    )
    with caplog.at_level("WARNING"):
        assert op.acquire_resources(task_name="t", envs={}) is None
    assert "targets a remote Docker host" in caplog.text


def test_extra_args_that_grant_gpus_are_flagged_against_the_reservation(
    monkeypatch, caplog
):
    """`docker run --gpus "device=UUID" ... --gpus all` does NOT let the last flag
    win -- docker accumulates device requests, so the container quietly gets every
    GPU while sflow believes it holds a 2-GPU claim. Nothing downstream can catch
    that, so the build must say it."""
    _patch_gpus(monkeypatch, 8)
    op = _local_op(2, extra_args=["--gpus all"])
    op.acquire_resources(task_name="t", envs={})

    with caplog.at_level("WARNING"):
        cmd = op.build_command(task_name="t", script=["echo hi"], envs={})
    args = cmd.as_list()

    assert "extra_args grant GPUs directly" in caplog.text
    # Both really are on the command line -- the warning is not hypothetical.
    assert args.count("--gpus") == 2
    assert "all" in args

    # Warned once per operator, not on every rebuild.
    caplog.clear()
    with caplog.at_level("WARNING"):
        op.build_command(task_name="t", script=["echo hi"], envs={})
    assert "extra_args grant GPUs directly" not in caplog.text


def test_no_warning_when_extra_args_leave_gpus_alone(monkeypatch, caplog):
    _patch_gpus(monkeypatch, 8)
    op = _local_op(2, extra_args=["--shm-size=1g"])
    op.acquire_resources(task_name="t", envs={})
    with caplog.at_level("WARNING"):
        op.build_command(task_name="t", script=["echo hi"], envs={})
    assert "extra_args grant GPUs" not in caplog.text


def test_a_long_unbounded_wait_is_called_out_as_possible_deadlock(
    monkeypatch, caplog
):
    """Two runs each holding part of the pool and each waiting for the rest never
    resolve. With an unbounded wait the only output is the per-retry INFO line,
    which reads like healthy queueing -- surface it instead."""
    _patch_gpus(monkeypatch, 1)
    op = _local_op(1, wait_for_gpus=0)  # 0 == wait forever
    # Someone else owns the only GPU.
    gr.try_reserve_gpus(1, "someone-else")
    monkeypatch.setattr(gr, "_pid_alive", lambda pid, start_ticks=None: True)

    clock = {"t": 0.0}
    monkeypatch.setattr(docker_run_mod.time, "monotonic", lambda: clock["t"])

    with pytest.raises(ResourcesUnavailable):
        op.acquire_resources(task_name="t", envs={})

    clock["t"] = docker_run_mod._STALLED_WAIT_WARN_S + 1
    with caplog.at_level("WARNING"):
        with pytest.raises(ResourcesUnavailable):
            op.acquire_resources(task_name="t", envs={})
    assert "waiting for the other to finish" in caplog.text
    assert "will not time out on its own" in caplog.text

    # Once only, however long the wait drags on.
    caplog.clear()
    clock["t"] += 10_000
    with caplog.at_level("WARNING"):
        with pytest.raises(ResourcesUnavailable):
            op.acquire_resources(task_name="t", envs={})
    assert "waiting for the other to finish" not in caplog.text


def test_a_short_wait_is_not_flagged_as_deadlock(monkeypatch, caplog):
    _patch_gpus(monkeypatch, 1)
    op = _local_op(1, wait_for_gpus=0)
    gr.try_reserve_gpus(1, "someone-else")
    monkeypatch.setattr(gr, "_pid_alive", lambda pid, start_ticks=None: True)
    with caplog.at_level("WARNING"):
        with pytest.raises(ResourcesUnavailable):
            op.acquire_resources(task_name="t", envs={})
    assert "waiting for the other to finish" not in caplog.text


def _probe_explodes(monkeypatch, exc):
    def _boom(count, run_id):
        raise exc

    monkeypatch.setattr(gr, "try_reserve_gpus", _boom)


@pytest.mark.parametrize(
    "exc",
    [
        gr.GpuProbeError("nvidia-smi did not answer"),
        gr.RegistryLockBusy("registry stayed locked"),
    ],
)
def test_a_transient_probe_failure_fails_closed_when_waiting_is_not_allowed(
    monkeypatch, exc
):
    """Falling back to the plan-time slice here would launch on devices starting
    at index 0 with no claim -- exactly the collision the registry prevents. An
    unanswerable driver is NOT the same as "this host has no GPUs"."""
    _patch_gpus(monkeypatch, 8)
    op = _local_op(2)  # no wait configured -> fail fast
    _probe_explodes(monkeypatch, exc)

    with pytest.raises(RuntimeError) as excinfo:
        op.acquire_resources(task_name="t", envs={})
    assert "could not determine GPU availability" in str(excinfo.value)
    assert "SFLOW_GPU_RESERVATION=0" in str(excinfo.value)


@pytest.mark.parametrize(
    "exc",
    [
        gr.GpuProbeError("nvidia-smi did not answer"),
        gr.RegistryLockBusy("registry stayed locked"),
    ],
)
def test_a_transient_probe_failure_is_retried_when_waiting_is_allowed(
    monkeypatch, exc
):
    """Both are transient by nature, so a task with a wait budget should retry
    rather than fail -- the orchestrator drives that on the event loop."""
    _patch_gpus(monkeypatch, 8)
    op = _local_op(2, wait_for_gpus=60)
    _probe_explodes(monkeypatch, exc)

    with pytest.raises(ResourcesUnavailable):
        op.acquire_resources(task_name="t", envs={})
