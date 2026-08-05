# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run ``resources.gpus.indices`` end to end through the CLI.

Runs the real user-facing example (``examples/self_contained/slurm/gpu_indices.yaml``)
so the placement documented in that file's header stays true: the example is the
thing users copy, and a silent drift there is worse than a broken unit test.

Exercises the whole YAML -> config -> planner -> report path (no cluster needed)
and asserts the per-node GPU map, which is what a user actually reads to confirm
where their pinned devices landed.
"""

import re
from pathlib import Path

from typer.testing import CliRunner

from sflow.cli import app

_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "self_contained/slurm/gpu_indices.yaml"
)


def _dry_run(tmp_path: Path):
    assert _EXAMPLE.exists(), _EXAMPLE
    return CliRunner().invoke(
        app,
        [
            "run",
            "-f",
            str(_EXAMPLE),
            "--workspace-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--verbose",
        ],
        catch_exceptions=False,
    )


def _gpu_map(output: str) -> list[tuple[int, str]]:
    """``[(gpu_index, owner), ...]`` in printed order (node0 GPUs 0-3, node1 ...)."""
    return [
        (int(m.group(1)), m.group(2).strip())
        for m in re.finditer(r"GPU (\d+): (\S+)", output)
    ]


def test_gpu_indices_example_places_pinned_slots_per_node(tmp_path: Path):
    result = _dry_run(tmp_path)
    assert result.exit_code == 0, result.output

    # 4 nodes x 4 GPUs, printed node by node. This is exactly the map documented
    # in the example's header comment.
    assert _gpu_map(result.output) == [
        # node0: pinned low slots, plus the backfill task on the idle 2,3
        (0, "pin_low"),
        (1, "pin_low"),
        (2, "backfill_high"),
        (3, "backfill_high"),
        # node1: node0's 0,1 were taken, so the second 0,1 request skipped here;
        # the count-only task then packed into the remaining contiguous pair
        (0, "pin_low_again"),
        (1, "pin_low_again"),
        (2, "flexible"),
        (3, "flexible"),
        # node2 + node3: count=4 over indices [0,1] -> 2 nodes, same slots
        (0, "fanout"),
        (1, "fanout"),
        (2, "."),
        (3, "."),
        (0, "fanout"),
        (1, "fanout"),
        (2, "."),
        (3, "."),
    ]


def test_gpu_indices_example_exports_pinned_cuda_visible_devices(tmp_path: Path):
    result = _dry_run(tmp_path)
    assert result.exit_code == 0, result.output

    devices = re.findall(r"CUDA_VISIBLE_DEVICES: (\S+)", result.output)
    # pin_low, pin_low_again, backfill_high, fanout, flexible -- in task order.
    assert devices == ["0,1", "0,1", "2,3", "0,1", "2,3"]
