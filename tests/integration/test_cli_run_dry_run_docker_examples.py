# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end dry-run tests for the docker backend example configs.

These exercise the whole CLI pipeline for docker (YAML parse -> schema validate
-> resolve_config -> synthetic allocation -> operator selection -> dry-run
render) without a Docker daemon, mirroring the slurm dynamo dry-run test. The
unit suite covers the backend/operator in isolation; this guards that a real,
shipped example config still parses and plans end-to-end.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sflow.cli import app


@pytest.mark.parametrize(
    "example,backend_name",
    [
        ("self_contained/docker/hello_world.yaml", "docker"),
        ("self_contained/docker/multi_node.yaml", "docker_cluster"),
        ("self_contained/docker/sglang_qwen3.yaml", "local_docker"),
    ],
    ids=["hello_world", "multi_node", "sglang_qwen3"],
)
def test_cli_run_dry_run_docker_example_exits_zero_without_daemon(
    tmp_path: Path, example: str, backend_name: str
):
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "examples" / example
    assert cfg.exists(), cfg

    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "-f",
            str(cfg),
            "--workspace-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ],
        catch_exceptions=False,
    )

    # Full pipeline must validate and plan even with no Docker daemon present.
    assert result.exit_code == 0, result.output
    assert "── Allocation map" in result.output
    assert f"backend '{backend_name}'" in result.output
    # The docker backend must plan through the docker_run operator.
    assert "operator=docker_run" in result.output
    # Dry-run must never materialize an output dir.
    assert not out_dir.exists()
