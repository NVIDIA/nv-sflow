# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil

import pytest

from sflow.app.assembly import preflight_validate_backends
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.backends.slurm import SlurmBackend, SlurmBackendConfig


def test_preflight_validation_fails_if_slurm_commands_missing(monkeypatch):
    # Simulate missing Slurm client tooling.
    monkeypatch.setattr(shutil, "which", lambda _: None)

    wf = Workflow(name="wf", task_graph=TaskGraph())
    state = SflowState(workflow=wf)
    state.backends["slurm_cluster"] = SlurmBackend(
        SlurmBackendConfig(
            name="slurm_cluster",
            account="acct",
            partition="part",
            time="00:01:00",
            nodes=1,
            gpus_per_node=1,
        )
    )

    with pytest.raises(
        ValueError, match="Pre-flight validation failed.*Missing required commands"
    ):
        preflight_validate_backends(state)
