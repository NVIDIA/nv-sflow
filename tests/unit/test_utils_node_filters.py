# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared node include/exclude list helpers."""

from sflow.utils.node_filters import (
    filter_by_node_names,
    find_node_filter_overlap,
    merge_node_lists,
    normalize_node_list,
)


class _Named:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, _Named) and other.name == self.name

    def __repr__(self):
        return f"_Named({self.name!r})"


def test_normalize_splits_comma_and_whitespace_and_dedups():
    assert normalize_node_list(["a,b", "b c", " d "]) == ["a", "b", "c", "d"]


def test_normalize_none_and_empty_entries():
    assert normalize_node_list(None) == []
    assert normalize_node_list([]) == []
    assert normalize_node_list(["", "  "]) == []


def test_merge_unions_preserving_order_and_dedups_exact():
    assert merge_node_lists(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_merge_handles_none_inputs():
    assert merge_node_lists(None, ["a"]) == ["a"]
    assert merge_node_lists(["a"], None) == ["a"]
    assert merge_node_lists(None, None) == []


def test_merge_preserves_expression_entries_verbatim():
    # YAML entries may be unresolved ${{ }} expressions; merge must not split them.
    merged = merge_node_lists(["${{ variables.NODES }}"], ["extra"])
    assert merged == ["${{ variables.NODES }}", "extra"]


def test_overlap_detected_after_normalization():
    assert find_node_filter_overlap(["a,b", "c"], ["c d"]) == ["c"]


def test_overlap_empty_when_disjoint():
    assert find_node_filter_overlap(["a"], ["b"]) == []


def test_filter_by_node_names_include_only_keeps_listed():
    items = [_Named("a"), _Named("b"), _Named("c")]
    assert filter_by_node_names(items, ["a", "c"], None) == [_Named("a"), _Named("c")]


def test_filter_by_node_names_exclude_drops_listed():
    items = [_Named("a"), _Named("b"), _Named("c")]
    assert filter_by_node_names(items, None, ["b"]) == [_Named("a"), _Named("c")]


def test_filter_by_node_names_include_and_exclude_intersect():
    items = [_Named("a"), _Named("b"), _Named("c")]
    assert filter_by_node_names(items, ["a", "b"], ["b"]) == [_Named("a")]


def test_filter_by_node_names_no_filters_returns_shallow_copy():
    items = [_Named("a"), _Named("b")]
    out = filter_by_node_names(items, None, None)
    assert out == items and out is not items


def test_filter_by_node_names_normalizes_raw_entries():
    # Raw comma/whitespace-joined lists are normalized (order-preserving).
    items = [_Named("a"), _Named("b"), _Named("c")]
    assert filter_by_node_names(items, ["a, c"], None) == [_Named("a"), _Named("c")]


def test_filter_by_node_names_custom_key():
    items = [{"host": "a"}, {"host": "b"}]
    assert filter_by_node_names(items, ["b"], None, key=lambda d: d["host"]) == [
        {"host": "b"}
    ]
