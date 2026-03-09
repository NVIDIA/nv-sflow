# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for sflow dry-run output helpers.
"""

import pytest

from sflow.app.sflow import extract_container_mounts_from_extra_args


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
