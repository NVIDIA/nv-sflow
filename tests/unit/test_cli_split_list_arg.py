# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""split_list_arg: comma/whitespace/repeated CLI list normalization."""

import pytest
import typer

from sflow.cli._args import parse_key_value_args
from sflow.cli.batch import split_list_arg


@pytest.mark.parametrize(
    "values, expected",
    [
        (None, None),
        ([], []),
        (["a,b,c"], ["a", "b", "c"]),
        (["a b c"], ["a", "b", "c"]),
        (["a, b ,c"], ["a", "b", "c"]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a,b", "c"], ["a", "b", "c"]),
        (["a,b", "b,c"], ["a", "b", "c"]),  # dedup, order preserved
        (["  a  ", "", "  "], ["a"]),  # strip + drop empties
    ],
)
def test_split_list_arg(values, expected):
    assert split_list_arg(values) == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        (None, {}),
        ([], {}),
        (["tenant=perflab"], {"tenant": "perflab"}),
        (["k1=v1,k2=v2"], {"k1": "v1", "k2": "v2"}),  # comma-separated
        (["k1=v1 k2=v2"], {"k1": "v1", "k2": "v2"}),  # whitespace-separated
        (["k1=v1", "k2=v2"], {"k1": "v1", "k2": "v2"}),  # repeated flags
        (["nvidia.com/gpu.present=true"], {"nvidia.com/gpu.present": "true"}),
        (["k=a=b"], {"k": "a=b"}),  # only the first '=' splits
        (["k2=v2", "k2=v3"], {"k2": "v3"}),  # later wins
    ],
)
def test_parse_key_value_args(values, expected):
    assert parse_key_value_args(values, flag="--kube-node-selector") == expected


def test_parse_key_value_args_rejects_missing_equals():
    with pytest.raises(typer.BadParameter):
        parse_key_value_args(["tenant"], flag="--kube-node-selector")
