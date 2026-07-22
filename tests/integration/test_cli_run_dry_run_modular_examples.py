# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run the backend-agnostic modular samples (``examples/modular``) end to end.

Each case composes a backend fragment + workload(s) + the shared benchmark and
proves that: (1) the same workload runs on Slurm / plain K8s / K8s-MPI unchanged,
(2) the shared backend name ``cluster`` and logical operators resolve, and
(3) ``required_by: [benchmark]`` on the server folds into the benchmark's
``depends_on`` (no ``--missable-tasks``).
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sflow.cli import app

_MODULAR = (
    Path(__file__).resolve().parents[2] / "examples" / "modular" / "backend_agnostic"
)


# name -> (relative file list, extra -s overrides)
_CASES = {
    "trtllm_k8s": (
        ["backends/k8s.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_trtllm.yaml", "benchmark.yaml"],
        [],
    ),
    "trtllm_k8s_mpi": (
        ["backends/k8s_mpi.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_trtllm.yaml", "benchmark.yaml"],
        ["NUM_NODES=2", "SERVER_GPUS=16"],
    ),
    "vllm_slurm": (
        ["backends/slurm.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_vllm.yaml", "benchmark.yaml"],
        [],
    ),
    "sglang_k8s": (
        ["backends/k8s.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_sglang.yaml", "benchmark.yaml"],
        [],
    ),
    "serve_k8s": (
        ["backends/k8s.yaml", "workloads/trtllm_serve.yaml", "benchmark.yaml"],
        [],
    ),
    "serve_slurm": (
        ["backends/slurm.yaml", "workloads/trtllm_serve.yaml", "benchmark.yaml"],
        [],
    ),
    "vllm_serve_k8s": (
        ["backends/k8s.yaml", "workloads/vllm_serve.yaml", "benchmark.yaml"],
        [],
    ),
    "vllm_serve_slurm": (
        ["backends/slurm.yaml", "workloads/vllm_serve.yaml", "benchmark.yaml"],
        [],
    ),
    "sglang_serve_k8s": (
        ["backends/k8s.yaml", "workloads/sglang_serve.yaml", "benchmark.yaml"],
        [],
    ),
    "sglang_serve_slurm": (
        ["backends/slurm.yaml", "workloads/sglang_serve.yaml", "benchmark.yaml"],
        [],
    ),
}


def _dry_run(files, overrides, tmp_path):
    args = ["run"]
    for f in files:
        cfg = _MODULAR / f
        assert cfg.exists(), cfg
        args += ["-f", str(cfg)]
    for o in overrides:
        args += ["-s", o]
    args += [
        "--workspace-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--dry-run",
    ]
    return CliRunner().invoke(app, args, catch_exceptions=False)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_modular_composition_dry_runs(name, tmp_path):
    files, overrides = _CASES[name]
    result = _dry_run(files, overrides, tmp_path)

    assert result.exit_code == 0, result.output
    # All backend fragments name their backend `cluster`.
    assert "backend 'cluster'" in result.output, result.output
    assert "server" in result.output and "benchmark" in result.output
    # required_by folded: the benchmark hub gained a depends_on the server.
    assert "depends_on=['server']" in result.output, result.output


def test_modular_dynamo_dag_chains_through_infra(tmp_path):
    """The Dynamo chain wires nats/etcd -> frontend -> server -> benchmark."""
    files, overrides = _CASES["trtllm_k8s"]
    result = _dry_run(files, overrides, tmp_path)
    assert result.exit_code == 0, result.output
    for task in ("nats_server", "etcd_server", "frontend_server", "server", "benchmark"):
        assert task in result.output, result.output
    # server depends on the frontend (from dynamo_common) ...
    assert "depends_on=['frontend_server']" in result.output
    # ... and the benchmark depends on the server (from required_by).
    assert "depends_on=['server']" in result.output
