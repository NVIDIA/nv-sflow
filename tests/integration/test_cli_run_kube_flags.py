# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-level Kubernetes access flags (--kubeconfig / --kube-context /
--kube-namespace / --extra-kubectl-args) flow through ``sflow run`` to the kubernetes
backend, keeping the recipe cluster-agnostic."""

from pathlib import Path

from typer.testing import CliRunner

from sflow.cli import app


def test_run_dry_run_threads_kube_cli_flags(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "examples" / "kubernetes_hello_world.yaml"
    assert cfg.exists()

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\nkind: Config\n")

    result = CliRunner().invoke(
        app,
        [
            "run",
            "-f",
            str(cfg),
            "--kubeconfig",
            str(kubeconfig),
            "--kube-context",
            "my-remote",
            "--kube-namespace",
            "team-ns",
            "--extra-kubectl-args=--request-timeout=30s",
            "--kube-exclude-node",
            "bad-node-1",
            "--kube-exclude-node",
            "bad-node-2",
            "--workspace-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--verbose",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    # Surfaced in the backend's dry-run details.
    assert "kubeconfig:" in result.output and str(kubeconfig) in result.output
    assert "context: my-remote" in result.output
    # --kube-namespace overrides the recipe's namespace.
    assert "namespace: team-ns" in result.output
    assert "--request-timeout=30s" in result.output
    # --kube-exclude-node (repeatable) threads through as a cluster-agnostic CLI flag.
    assert "bad-node-1" in result.output and "bad-node-2" in result.output
