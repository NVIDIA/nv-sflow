# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU list helpers shared by planning, monitoring and run reporting."""

from dataclasses import dataclass, field

from sflow.utils.gpu import (
    count_device_tokens,
    count_visible_devices,
    parse_cuda_visible_devices,
    planned_gpu_indices,
    task_gpu_indices,
)


@dataclass
class _Task:
    """Minimal stand-in carrying the three fields task_gpu_indices consults."""

    reserved_gpu_indices: list[int] | None = None
    cuda_visible_devices: str | None = None
    envs: dict = field(default_factory=dict)


def test_count_device_tokens_counts_without_interpreting():
    assert count_device_tokens("0,1,2") == 3
    # UUIDs are not indices, but they are still devices -- parse_cuda_visible_devices
    # deliberately drops them, so counting must not go through it.
    assert count_device_tokens("GPU-aaaa,GPU-bbbb") == 2
    assert parse_cuda_visible_devices("GPU-aaaa,GPU-bbbb") == []
    assert count_device_tokens("") == 0
    assert count_device_tokens(None) == 0
    assert count_device_tokens("0, ,1") == 2


def test_count_visible_devices_expands_ranges_and_still_counts_uuids():
    # The range form names 4 devices, not 1 -- counting raw tokens alone would
    # under-size both the docker GPU claim and the container-visible slice.
    assert count_visible_devices("0-3") == 4
    assert count_device_tokens("0-3") == 1
    # Explicit lists and UUIDs keep the plain token count.
    assert count_visible_devices("0,1,2") == 3
    assert count_visible_devices("GPU-aaaa,GPU-bbbb") == 2
    assert count_visible_devices("") == 0
    assert count_visible_devices(None) == 0


def test_reserved_indices_win_over_everything():
    # Devices actually claimed at launch are the only fully accurate source.
    task = _Task(
        reserved_gpu_indices=[4, 5],
        cuda_visible_devices="2,3",
        envs={"CUDA_VISIBLE_DEVICES": "0,1"},
    )
    assert task_gpu_indices(task) == [4, 5]


def test_planner_slice_beats_the_container_visible_env():
    # THE docker case: the container env is virtual (0..N-1 for every task), so
    # reading it would report every task as sitting on GPU 0.
    task = _Task(cuda_visible_devices="2,3", envs={"CUDA_VISIBLE_DEVICES": "0,1"})
    assert task_gpu_indices(task) == [2, 3]


def test_env_is_the_last_resort():
    task = _Task(envs={"CUDA_VISIBLE_DEVICES": "6,7"})
    assert task_gpu_indices(task) == [6, 7]


def test_no_gpu_information_yields_empty():
    assert task_gpu_indices(_Task()) == []
    assert task_gpu_indices(_Task(envs={})) == []


def test_planner_slice_alone_is_not_reported_as_devices():
    """No injected GPU env -> the slice is planning-only, not device identity.

    Kubernetes computes a slice for capacity planning and conflict detection but
    never injects CUDA_VISIBLE_DEVICES: the device plugin / DRA decides which
    physical GPUs the pod gets. Reporting the slice would invent device numbers
    the pod never touched.
    """
    k8s_task = _Task(cuda_visible_devices="2,3", envs={})
    assert task_gpu_indices(k8s_task) == []


def test_a_real_reservation_is_reported_even_without_gpu_env():
    # Devices actually claimed are always trustworthy, env or no env.
    task = _Task(reserved_gpu_indices=[6, 7], cuda_visible_devices="2,3", envs={})
    assert task_gpu_indices(task) == [6, 7]


def test_planned_slice_is_still_reported_where_the_run_view_stays_silent():
    """The two views differ exactly where the plan is all there is: kubernetes.

    task_gpu_indices answers "what ran" and must stay quiet when the device plugin
    picked the GPUs. planned_gpu_indices answers "what was planned", which is
    precisely what the dry-run allocation map is showing -- so the same task must
    yield [] from one and the slice from the other. Collapsing them into one helper
    would silently empty the k8s allocation map.
    """
    k8s_task = _Task(cuda_visible_devices="2,3", envs={})
    assert task_gpu_indices(k8s_task) == []
    assert planned_gpu_indices(k8s_task) == [2, 3]


def test_planned_slice_prefers_the_planner_over_the_container_visible_env():
    # Docker re-indexes to 0..N-1 inside the container; the plan view must name the
    # host devices, or the allocation map draws every task on GPU 0.
    task = _Task(cuda_visible_devices="4,5", envs={"CUDA_VISIBLE_DEVICES": "0,1"})
    assert planned_gpu_indices(task) == [4, 5]


def test_planned_slice_falls_back_to_the_env_and_ignores_reservations():
    # No planner slice -> the env is all there is.
    assert planned_gpu_indices(_Task(envs={"CUDA_VISIBLE_DEVICES": "1,2"})) == [1, 2]
    # A launch-time reservation is deliberately NOT consulted: the allocation map
    # is a dry-run view, rendered before anything has launched.
    reserved = _Task(reserved_gpu_indices=[6, 7], cuda_visible_devices="2,3", envs={})
    assert planned_gpu_indices(reserved) == [2, 3]


def test_planned_slice_is_empty_without_any_gpu_information():
    assert planned_gpu_indices(_Task()) == []


def test_two_docker_tasks_do_not_collide_on_gpu_zero():
    # Regression: both tasks' containers see CUDA_VISIBLE_DEVICES=0,1 while their
    # planner slices (and real devices) are disjoint. Reporting must reflect the
    # disjointness, otherwise the GPU usage chart and monitor overlap them.
    a = _Task(cuda_visible_devices="0,1", envs={"CUDA_VISIBLE_DEVICES": "0,1"})
    b = _Task(cuda_visible_devices="2,3", envs={"CUDA_VISIBLE_DEVICES": "0,1"})
    assert set(task_gpu_indices(a)).isdisjoint(task_gpu_indices(b))
