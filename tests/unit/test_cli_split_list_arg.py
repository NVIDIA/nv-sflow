# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""split_list_arg: comma/whitespace/repeated CLI list normalization."""

import pytest

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
