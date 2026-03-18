# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for sflow dry-run output helpers.
"""

from types import SimpleNamespace

import pytest

from sflow.app.sflow import (
    build_allocation_map_lines,
    extract_container_mounts_from_extra_args,
    parse_cuda_visible_devices,
)
from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode


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
