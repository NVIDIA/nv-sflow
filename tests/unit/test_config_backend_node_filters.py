# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for BackendConfig include_nodes/exclude_nodes fields + merge/validation."""

import pytest
from pydantic import ValidationError

from sflow.config.schema import BackendConfig


def test_backend_config_node_filters_default_none():
    b = BackendConfig(name="b", type="slurm")
    assert b.include_nodes is None and b.exclude_nodes is None


def test_merge_node_filters_unions_cli_over_yaml():
    b = BackendConfig(name="b", type="slurm", exclude_nodes=["bad-1"])
    merged = b.merge_node_filters(["want-1"], ["bad-2"])
    assert merged.include_nodes == ["want-1"]
    assert merged.exclude_nodes == ["bad-1", "bad-2"]


def test_merge_node_filters_noop_returns_self():
    b = BackendConfig(name="b", type="slurm")
    assert b.merge_node_filters(None, None) is b


def test_merge_node_filters_dedups_exact():
    b = BackendConfig(name="b", type="slurm", include_nodes=["a"])
    assert b.merge_node_filters(["a", "c"], None).include_nodes == ["a", "c"]


def test_overlap_rejected_at_validation():
    with pytest.raises(ValidationError):
        BackendConfig(
            name="b", type="slurm", include_nodes=["a", "b"], exclude_nodes=["b"]
        )


def test_overlap_validation_ignores_expressions():
    # Unresolved expressions can't be compared literally; must not raise.
    BackendConfig(
        name="b",
        type="slurm",
        include_nodes=["${{ variables.X }}"],
        exclude_nodes=["${{ variables.X }}"],
    )


def test_merge_node_filters_raises_on_resulting_overlap():
    b = BackendConfig(name="b", type="slurm", include_nodes=["a"])
    with pytest.raises(ValueError):
        b.merge_node_filters(None, ["a"])
