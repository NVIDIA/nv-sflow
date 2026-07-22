# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for sflow dry-run output helpers.
"""

import logging
from io import StringIO
from types import SimpleNamespace

import pytest

from sflow.app.sflow import SflowApp
from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.utils.container import extract_container_mounts_from_extra_args
from sflow.utils.gpu import parse_cuda_visible_devices
from sflow.utils.logging import (
    build_allocation_map_lines,
    build_resource_rehearsal_lines,
)


_COLLIDING_VAR_CONFIG = """\
version: "0.1"
variables:
  SFLOW_TASK_OUTPUT_DIR:
    value: "/tmp/should-not-be-a-variable"
  MODEL_PATH:
    value: "/models/demo"
backends:
  - name: local
    type: local
    default: true
    nodes: 1
workflow:
  name: reserved_env_collision
  tasks:
    - name: t1
      script:
        - echo hi
"""


def test_dry_run_warns_on_reserved_env_variable_collision(tmp_path):
    cfg = tmp_path / "sflow.yaml"
    cfg.write_text(_COLLIDING_VAR_CONFIG)

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("sflow")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        SflowApp().run(
            file=cfg,
            dry_run=True,
            workspace_dir=tmp_path,
            output_dir=tmp_path / "out",
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    messages = stream.getvalue()
    assert "reserved sflow env var" in messages
    assert "SFLOW_TASK_OUTPUT_DIR" in messages
    # A regular variable name must not be flagged.
    assert "MODEL_PATH" not in messages


class TestExtractContainerMountsFromExtraArgs:
    """Tests for extract_container_mounts_from_extra_args function."""

    def test_empty_extra_args(self):
        """Empty extra_args returns empty list."""
        result = extract_container_mounts_from_extra_args([])
        assert result == []

    def test_no_container_mounts_in_extra_args(self):
        """Extra args without --container-mounts returns empty list."""
        result = extract_container_mounts_from_extra_args(
            ["--some-flag", "--other-option", "value"]
        )
        assert result == []

    def test_container_mounts_separate_arg(self):
        """--container-mounts as separate arg and value."""
        result = extract_container_mounts_from_extra_args(
            ["--container-mounts", "/host:/container"]
        )
        assert result == ["/host:/container"]

    def test_container_mounts_equals_syntax(self):
        """--container-mounts=value syntax."""
        result = extract_container_mounts_from_extra_args(
            ["--container-mounts=/host:/container"]
        )
        assert result == ["/host:/container"]

    def test_container_mounts_comma_separated(self):
        """Multiple comma-separated mounts are split."""
        result = extract_container_mounts_from_extra_args(
            ["--container-mounts", "/path1:/cpath1,/path2:/cpath2"]
        )
        assert result == ["/path1:/cpath1", "/path2:/cpath2"]

    def test_container_mounts_equals_comma_separated(self):
        """Equals syntax with comma-separated mounts."""
        result = extract_container_mounts_from_extra_args(
            ["--container-mounts=/path1:/cpath1,/path2:/cpath2,/path3:/cpath3"]
        )
        assert result == ["/path1:/cpath1", "/path2:/cpath2", "/path3:/cpath3"]

    def test_container_mounts_mixed_with_other_args(self):
        """--container-mounts mixed with other arguments."""
        result = extract_container_mounts_from_extra_args(
            ["--some-flag", "--container-mounts", "/host:/container", "--other-flag"]
        )
        assert result == ["/host:/container"]

    def test_multiple_container_mounts_entries(self):
        """Multiple --container-mounts entries are all collected."""
        result = extract_container_mounts_from_extra_args(
            [
                "--container-mounts", "/path1:/cpath1",
                "--other-flag",
                "--container-mounts=/path2:/cpath2",
            ]
        )
        assert result == ["/path1:/cpath1", "/path2:/cpath2"]

    def test_container_mounts_at_end_without_value(self):
        """--container-mounts at end without value is skipped."""
        result = extract_container_mounts_from_extra_args(
            ["--other-flag", "--container-mounts"]
        )
        assert result == []

    def test_container_mounts_with_rw_mode(self):
        """Mounts with :rw or :ro mode suffix."""
        result = extract_container_mounts_from_extra_args(
            ["--container-mounts", "/host:/container:rw,/data:/data:ro"]
        )
        assert result == ["/host:/container:rw", "/data:/data:ro"]


class TestParseCudaVisibleDevices:
    def test_none_returns_empty(self):
        assert parse_cuda_visible_devices(None) == []

    def test_comma_separated_indices(self):
        assert parse_cuda_visible_devices("0,2,3") == [0, 2, 3]

    def test_range_syntax(self):
        assert parse_cuda_visible_devices("0-3") == [0, 1, 2, 3]

    def test_mixed_tokens_ignores_invalid(self):
        assert parse_cuda_visible_devices("0,abc,2-3") == [0, 2, 3]


class TestBuildAllocationMapLines:
    def test_builds_node_and_gpu_chart(self):
        backend = SimpleNamespace(
            allocation=Allocation(
                allocation_id="job-1",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1, num_gpus=4),
                ],
            )
        )
        tasks = [
            SimpleNamespace(
                name="prefill_0",
                backend_name="slurm_cluster",
                assigned_nodes=["n1"],
                envs={"CUDA_VISIBLE_DEVICES": "0,1"},
                operator=None,
            ),
            SimpleNamespace(
                name="decode_0",
                backend_name="slurm_cluster",
                assigned_nodes=["n1"],
                envs={"CUDA_VISIBLE_DEVICES": "2"},
                operator=None,
            ),
            SimpleNamespace(
                name="frontend",
                backend_name="slurm_cluster",
                assigned_nodes=["n1"],
                envs={},
                operator=None,
            ),
            SimpleNamespace(
                name="decode_1",
                backend_name="slurm_cluster",
                assigned_nodes=["n2"],
                envs={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
                operator=None,
            ),
        ]

        lines = build_allocation_map_lines(tasks, {"slurm_cluster": backend})
        rendered = "\n".join(lines)

        assert "backend 'slurm_cluster'" in rendered
        assert "node n1" in rendered
        assert "node n2" in rendered
        assert "GPU 0: prefill_0" in rendered
        assert "GPU 1: prefill_0" in rendered
        assert "GPU 2: decode_0" in rendered
        assert "GPU 3: ." in rendered
        assert "Tasks: prefill_0, decode_0, frontend" in rendered
        assert "GPU 0: decode_1" in rendered
        assert "GPU 3: decode_1" in rendered

    def test_uses_planner_slice_when_not_injected_into_env_kubernetes(self):
        # Cluster-scheduled backends (Kubernetes) compute CUDA_VISIBLE_DEVICES the
        # same way as every backend but do not inject it into the env. The map reads
        # the planner slice carried on the task so GPU occupancy is still shown.
        backend = SimpleNamespace(
            allocation=Allocation(
                allocation_id="k8s",
                nodes=[
                    ComputeNode(name="node-0", ip_address="10.0.0.1", index=0, num_gpus=4),
                ],
            )
        )
        tasks = [
            SimpleNamespace(
                name="etcd",
                backend_name="k8s",
                assigned_nodes=["node-0"],
                envs={},
                cuda_visible_devices=None,
                operator=None,
            ),
            SimpleNamespace(
                name="prefill_server_0",
                backend_name="k8s",
                assigned_nodes=["node-0"],
                envs={},  # not injected into the env on k8s
                cuda_visible_devices="0,1",
                operator=None,
            ),
            SimpleNamespace(
                name="decode_server_0",
                backend_name="k8s",
                assigned_nodes=["node-0"],
                envs={},
                cuda_visible_devices="2,3",
                operator=None,
            ),
        ]

        rendered = "\n".join(build_allocation_map_lines(tasks, {"k8s": backend}))

        assert "GPU 0: prefill_server_0" in rendered
        assert "GPU 1: prefill_server_0" in rendered
        assert "GPU 2: decode_server_0" in rendered
        assert "GPU 3: decode_server_0" in rendered
        assert "Tasks: etcd, prefill_server_0, decode_server_0" in rendered

    def test_planner_slice_applies_per_node_for_multinode(self):
        # The planner's per-node slice is applied to each assigned node (one pod
        # per node), so a multi-node task fills its GPUs on every node it spans.
        backend = SimpleNamespace(
            allocation=Allocation(
                allocation_id="k8s",
                nodes=[
                    ComputeNode(name="node-0", ip_address="10.0.0.1", index=0, num_gpus=8),
                    ComputeNode(name="node-1", ip_address="10.0.0.2", index=1, num_gpus=8),
                ],
            )
        )
        tasks = [
            SimpleNamespace(
                name="train",
                backend_name="k8s",
                assigned_nodes=["node-0", "node-1"],
                envs={},
                cuda_visible_devices="0,1,2,3,4,5,6,7",
                operator=None,
            ),
        ]

        rendered = "\n".join(build_allocation_map_lines(tasks, {"k8s": backend}))

        # 8 GPUs on each of the 2 nodes, all owned by the task.
        assert rendered.count("train") >= 16
        assert "GPU 7: train" in rendered

    def test_does_not_truncate_long_gpu_owner_names(self):
        backend = SimpleNamespace(
            allocation=Allocation(
                allocation_id="job-2",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=1),
                ],
            )
        )
        tasks = [
            SimpleNamespace(
                name="check_entire_env",
                backend_name="slurm_cluster",
                assigned_nodes=["n1"],
                envs={"CUDA_VISIBLE_DEVICES": "0"},
                operator=None,
            ),
            SimpleNamespace(
                name="worker_0",
                backend_name="slurm_cluster",
                assigned_nodes=["n1"],
                envs={"CUDA_VISIBLE_DEVICES": "0"},
                operator=None,
            ),
        ]

        rendered = "\n".join(
            build_allocation_map_lines(tasks, {"slurm_cluster": backend})
        )

        assert "GPU 0: check_entire_env -> worker_0" in rendered
        assert "check_entire_en..." not in rendered


class TestBuildResourceRehearsalLines:
    def test_describes_release_policies_as_lifecycle_actions(self):
        tasks = [
            SimpleNamespace(
                name="check_entire_env",
                resource_release_after={"gpus": "task_completion"},
            ),
            SimpleNamespace(
                name="server",
                resource_release_after={
                    "gpus": "task_ready",
                    "nodes": "workflow_completion",
                },
            ),
        ]

        lines = build_resource_rehearsal_lines(tasks)

        assert lines == [
            "  - check_entire_env: releases GPUs after task completion",
            "  - server: releases GPUs after task readiness; keeps nodes until workflow completion",
        ]
