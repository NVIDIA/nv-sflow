# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sflow.cli import app


@pytest.mark.parametrize(
    "num_ctx_servers,ctx_tp_size,num_gen_servers,gen_tp_size",
    [
        (1, 8, 2, 2),
        (6, 1, 6, 1),
        (1, 4, 1, 8),
        (2, 2, 2, 4),
    ],
    ids=[
        "ctx1_tp8_gen2_tp2",
        "ctx6_tp1_gen6_tp1",
        "ctx1_tp4_gen1_tp8",
        "ctx2_tp2_gen2_tp4",
    ],
)
def test_cli_run_dry_run_dynamo_example_exits_zero_and_does_not_create_output_dirs(
    tmp_path: Path,
    num_ctx_servers: int,
    ctx_tp_size: int,
    num_gen_servers: int,
    gen_tp_size: int,
):
    """
    End-to-end dry-run test using the real example config + CLI parsing.

    This intentionally includes duplicate --set for the same variable to ensure
    overrides are applied in-order and do not crash (last one wins).
    """
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "examples" / "slurm_dynamo_trtllm_disagg.yaml"
    assert cfg.exists()

    out_dir = tmp_path / "out"

    # Create a dummy model directory so the fs:// artifact path validation passes
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "-f",
            str(cfg),
            "--set",
            "SLURM_NODES=3",
            "--set",
            "GPUS_PER_NODE=4",
            "--set",
            f"NUM_CTX_SERVERS={num_ctx_servers}",
            "--set",
            f"CTX_TP_SIZE={ctx_tp_size}",
            "--set",
            f"NUM_GEN_SERVERS={num_gen_servers}",
            "--set",
            f"GEN_TP_SIZE={gen_tp_size}",
            "--artifact",
            f"LOCAL_MODEL_PATH=fs://{model_dir}",
            "--workspace-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Allocation map (finalized node/GPU assignment):" in result.output
    assert "backend 'slurm_cluster'" in result.output

    # Dry-run should not create any output dirs/files.
    assert not out_dir.exists()
    # Inline file:// artifacts should not be materialized during dry-run.
    assert not (tmp_path / "prefill_config.yaml").exists()
    assert not (tmp_path / "decode_config.yaml").exists()


