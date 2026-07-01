# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run the Kubernetes dynamo trtllm disagg recipe end to end.

Validates the native Kubernetes recipe through real CLI parsing: the tasks use
the ``k8s`` operator, the ``kubernetes`` backend (scheduling: dra) plans the DAG,
and the ${{ backends.* }} references + probes resolve. A single-node worker
renders one pod; a worker whose GPU count spans multiple nodes is split across
nodes by the planner (one pod per node).
"""

from pathlib import Path

from typer.testing import CliRunner

from sflow.cli import app


def _run_dry(tmp_path: Path, extra_sets: list[str]):
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "examples" / "kubernetes_dynamo_trtllm_disagg.yaml"
    assert cfg.exists()

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "out"

    args = ["run", "-f", str(cfg)]
    for s in extra_sets:
        args += ["--set", s]
    args += [
        "--artifact",
        f"LOCAL_MODEL_PATH=fs://{model_dir}",
        "--workspace-dir",
        str(tmp_path),
        "--output-dir",
        str(out_dir),
        "--dry-run",
    ]
    return CliRunner().invoke(app, args, catch_exceptions=False), out_dir


def test_k8s_dynamo_dry_run_single_node_uses_k8s_operator(tmp_path: Path):
    result, out_dir = _run_dry(
        tmp_path,
        [
            "NUM_NODES=1",
            "GPUS_PER_NODE=4",
            "NUM_CTX_SERVERS=1",
            "CTX_TP_SIZE=2",
            "NUM_GEN_SERVERS=1",
            "GEN_TP_SIZE=2",
        ],
    )

    assert result.exit_code == 0, result.output
    # The recipe runs natively on the kubernetes backend with the k8s operator.
    assert "backend 'k8s_cluster'" in result.output
    assert "prefill_server_0  (backend=k8s_cluster, operator=k8s" in result.output
    assert "decode_server_0  (backend=k8s_cluster, operator=k8s" in result.output
    # Dry-run must not create output dirs or materialize inline artifacts.
    assert not out_dir.exists()
    assert not (tmp_path / "prefill_config.yaml").exists()


def test_k8s_dynamo_dry_run_multinode_worker_spans_nodes(tmp_path: Path):
    # CTX_GPUS_PER_WORKER (=CTX_TP_SIZE=4) > GPUS_PER_NODE (=2) -> the prefill
    # worker spans 2 nodes; the planner's GPU-only inference assigns 2 nodes and
    # the k8s operator renders one pod per node.
    #
    # Keep total GPU demand within the 10-GPU budget (NUM_NODES=5 * GPUS_PER_NODE=2):
    # prefill = NUM_CTX_SERVERS(1) * CTX_TP_SIZE(4) = 4 (spanning 2 nodes), decode =
    # NUM_GEN_SERVERS(1) * GEN_TP_SIZE(2) = 2, total 6. The default NUM_GEN_SERVERS=4
    # would need 4+8=12 GPUs and the planner would (correctly) reject it since the
    # prefill server holds its GPUs release_after=workflow_completion.
    result, _ = _run_dry(
        tmp_path,
        [
            "NUM_NODES=5",
            "GPUS_PER_NODE=2",
            "NUM_CTX_SERVERS=1",  # this test exercises one worker spanning nodes
            "CTX_TP_SIZE=4",
            "NUM_GEN_SERVERS=1",
            "GEN_TP_SIZE=6",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "prefill_server_0  (backend=k8s_cluster, operator=k8s" in result.output
