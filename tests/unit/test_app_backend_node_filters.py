# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for _merge_backend_node_filters (CLI node filters -> all backends)."""

import pytest

from sflow.app.sflow import _merge_backend_node_filters
from sflow.config.schema import BackendConfig


class _FakeConfig:
    def __init__(self, backends):
        self.backends = backends

    def model_copy(self, *, update):
        return _FakeConfig(update["backends"])


def test_merge_applies_filters_to_every_backend():
    cfg = _FakeConfig(
        [BackendConfig(name="a", type="slurm"), BackendConfig(name="b", type="kubernetes")]
    )
    out = _merge_backend_node_filters(cfg, ["want"], ["bad"])
    assert [b.include_nodes for b in out.backends] == [["want"], ["want"]]
    assert [b.exclude_nodes for b in out.backends] == [["bad"], ["bad"]]


def test_merge_unions_over_existing_yaml():
    cfg = _FakeConfig([BackendConfig(name="a", type="slurm", exclude_nodes=["bad-1"])])
    out = _merge_backend_node_filters(cfg, None, ["bad-2"])
    assert out.backends[0].exclude_nodes == ["bad-1", "bad-2"]


def test_merge_noop_when_no_filters_returns_same_config():
    cfg = _FakeConfig([BackendConfig(name="a", type="slurm")])
    assert _merge_backend_node_filters(cfg, None, None) is cfg


def test_merge_raises_on_overlap_after_merge():
    cfg = _FakeConfig([BackendConfig(name="a", type="slurm", include_nodes=["x"])])
    with pytest.raises(ValueError):
        _merge_backend_node_filters(cfg, None, ["x"])
