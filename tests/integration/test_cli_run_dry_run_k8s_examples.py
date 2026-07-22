# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run the native-Kubernetes SMOKE recipes (test fixtures) end to end.

These recipes are not user-facing workload samples -- they are mimic-script smoke
tests for sflow's Kubernetes mechanics, so they live under
``tests/integration/recipes/kubernetes/`` (not ``examples/``):

* ``pd_smoke`` -- a PD-separation layout that reserves 3 nodes and resolves
  ``${{ backends.k8s.nodes[0].ip_address }}`` across tasks (leader/worker/decode).
* ``apply_launch`` -- a single-pod apply/validate + live-log-streaming check.
* ``multinode_smoke`` -- multi-node reservation + rendezvous over host_network.
* ``log_offload_smoke`` -- single-pod + merged-pod log offload and DAG cut-over.

All render through the ``k8s`` operator on the ``kubernetes`` backend named ``k8s``.
"""

from pathlib import Path

from typer.testing import CliRunner

from sflow.cli import app

_RECIPES = Path(__file__).resolve().parent / "recipes"


def _dry_run(cfg_name: str, tmp_path: Path):
    cfg = _RECIPES / cfg_name
    assert cfg.exists(), cfg
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
    result, out_dir = _dry_run("kubernetes/pd_smoke.yaml", tmp_path)

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
    result, out_dir = _dry_run("kubernetes/apply_launch.yaml", tmp_path)

    assert result.exit_code == 0, result.output
    assert "backend 'k8s'" in result.output
    assert "operator=k8s" in result.output
    assert not out_dir.exists()


def test_k8s_multinode_smoke_dry_run_plans_rendezvous(tmp_path: Path):
    result, out_dir = _dry_run("kubernetes/multinode_smoke.yaml", tmp_path)

    assert result.exit_code == 0, result.output
    assert "backend 'k8s'" in result.output
    assert "operator=k8s" in result.output
    assert "rendezvous" in result.output, result.output
    assert not out_dir.exists()


def test_k8s_log_offload_smoke_dry_runs(tmp_path: Path):
    result, out_dir = _dry_run("kubernetes/log_offload_smoke.yaml", tmp_path)

    assert result.exit_code == 0, result.output
    assert "backend 'k8s'" in result.output
    assert "operator=k8s" in result.output
    assert not out_dir.exists()
