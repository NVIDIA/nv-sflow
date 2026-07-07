# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run the standalone native-Kubernetes example recipes end to end.

Covers the two Kubernetes examples that carry cross-task / apply-validate
behavior but are not part of the Dynamo recipe matrix
(``test_cli_run_dry_run_k8s_dynamo_example``):

* ``kubernetes_pd_smoke`` -- a PD-separation layout that reserves 3 nodes and
  resolves ``${{ backends.k8s.nodes[0].ip_address }}`` across tasks (leader on
  node 0, worker on node 1, decode client on node 2). A clean dry-run proves the
  cross-task node-IP references resolve and the roles pin via ``resources.nodes``.
* ``kubernetes_apply_launch`` -- the single-pod apply/validate recipe.

Both render through the ``k8s`` operator on the ``kubernetes`` backend named
``k8s``.
"""

from pathlib import Path

from typer.testing import CliRunner

from sflow.cli import app


def _dry_run(cfg_name: str, tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "examples" / cfg_name
    assert cfg.exists()
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
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
    return result, out_dir


def test_k8s_pd_smoke_dry_run_resolves_cross_task_node_ips(tmp_path: Path):
    result, out_dir = _dry_run("kubernetes_pd_smoke.yaml", tmp_path)

    assert result.exit_code == 0, result.output
    # Single kubernetes backend named 'k8s'; tasks render via the k8s operator.
    assert "backend 'k8s'" in result.output
    assert "operator=k8s" in result.output
    # All three PD roles are planned; a clean exit means the cross-task
    # ${{ backends.k8s.nodes[0].ip_address }} references resolved.
    for task in ("prefill_leader", "prefill_worker", "decode_client"):
        assert task in result.output, result.output
    # Dry-run must not create the output dir.
    assert not out_dir.exists()


def test_k8s_apply_launch_dry_run_uses_k8s_operator(tmp_path: Path):
    result, out_dir = _dry_run("kubernetes_apply_launch.yaml", tmp_path)

    assert result.exit_code == 0, result.output
    assert "backend 'k8s'" in result.output
    assert "operator=k8s" in result.output
    assert not out_dir.exists()
