# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the ``mpi_smoke.yaml`` rank/host self-checks.

The recipe asserts it saw exactly ``NUM_NODES*GPUS_PER_NODE`` ranks across ``NUM_NODES``
hosts. Those asserts must count only REAL rank lines (``[SFLOW_RANK] r=<digit>``) -- the
transparent mpirun shim tee's its OWN launch echo (``[SFLOW_RANK] r=${OMPI_COMM_WORLD_RANK}
... host=$(hostname)``) into the same ``ranks.txt``, so an unanchored grep over-counts both
ranks and hosts and the smoke test would falsely FAIL on every cluster run. This is the
CI-side guard for that fix (the recipe itself only runs on a real multi-node cluster).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MPI_SMOKE = _REPO_ROOT / "examples/self_contained/kubernetes/mpi_smoke.yaml"

# A realistic ranks.txt: the shim's tee'd launch echo (literal r=${...}, host=$(hostname))
# followed by 4 real ranks across 2 hosts.
_RANKS_FIXTURE = (
    """[sflow-mpi] launch: mpirun -np 4 --allow-run-as-root bash -c 'echo "[SFLOW_RANK] r=${OMPI_COMM_WORLD_RANK} n=${OMPI_COMM_WORLD_SIZE} host=$(hostname)"'
[SFLOW_RANK] r=0 n=4 host=node0
[SFLOW_RANK] r=1 n=4 host=node0
[SFLOW_RANK] r=2 n=4 host=node1
[SFLOW_RANK] r=3 n=4 host=node1
"""
)

_ANCHORED = r"\[SFLOW_RANK\] r=[0-9]"


def test_mpi_smoke_uses_anchored_rank_pattern():
    """Contract: the recipe counts anchored rank lines (``r=<digit>``), guarding against a
    revert to a bare, over-counting ``[SFLOW_RANK]`` grep."""
    text = _MPI_SMOKE.read_text()
    assert _ANCHORED in text
    # Both smoke tasks (compound-line + multiline) each use it in the echo, the rank-count
    # assert, and the host-count assert -> many occurrences; require a safe lower bound.
    assert text.count(_ANCHORED) >= 4


@pytest.mark.skipif(not shutil.which("grep"), reason="grep not available")
def test_anchored_pattern_excludes_shim_launch_echo(tmp_path, fake_process):
    """Behavior: the anchored pattern counts the 4 real ranks / 2 hosts and excludes the
    shim's launch echo; the old bare pattern would over-count (5)."""
    fake_process.allow_unregistered(True)  # run real grep (conftest fakes subprocess)
    ranks = tmp_path / "ranks.txt"
    ranks.write_text(_RANKS_FIXTURE)

    anchored_count = subprocess.run(
        ["grep", "-cE", _ANCHORED, str(ranks)], capture_output=True, text=True
    ).stdout.strip()
    assert anchored_count == "4"  # shim launch echo (r=$) excluded

    distinct_hosts = subprocess.run(
        f'grep -E "{_ANCHORED}" "{ranks}" | grep -oE "host=[^ ]+" | sort -u | wc -l',
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert distinct_hosts == "2"  # node0, node1 -- not host=$(hostname) from the shim line

    # The old unanchored count would have over-counted the shim echo -> 5 (the bug fixed).
    bare_count = subprocess.run(
        ["grep", "-c", r"\[SFLOW_RANK\]", str(ranks)], capture_output=True, text=True
    ).stdout.strip()
    assert bare_count == "5"
