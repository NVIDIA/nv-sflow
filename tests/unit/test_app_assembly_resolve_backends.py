# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from sflow.app.assembly import resolve_backends, resolve_global_variables
from sflow.config.schema import (
    SflowConfig,
    TaskConfig,
    WorkflowConfig,
)
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.variable import Variable, VariableType
from sflow.core.workflow import Workflow
from sflow.plugins.backends.slurm import SlurmBackend


def _state_with_vars(vars_dict: dict[str, object]) -> SflowState:
    tg = TaskGraph()
    workflow = Workflow(name="wf", task_graph=tg)
    state = SflowState(workflow=workflow)
    state.variables = {
        k: Variable(name=k, value=v, type=VariableType.STRING)
        for k, v in vars_dict.items()
    }
    return state


def _minimal_workflow() -> WorkflowConfig:
    return WorkflowConfig(
        name="wf",
        tasks=[TaskConfig(name="t1", script=["echo hi"])],
    )


def test_resolve_backends_instantiates_slurm_backend_with_resolved_fields():
    config = SflowConfig(
        version="0.1",
        variables=[
            {"name": "ACC", "value": "acct", "type": "string"},
            {"name": "PART", "value": "batch", "type": "string"},
            {"name": "NODES", "value": 2, "type": "integer"},
            {"name": "TIME", "value": "00:10:00", "type": "string"},
            {"name": "GPUS_PER_NODE", "value": 4, "type": "integer"},
        ],
        backends=[
            {
                "name": "b1",
                "type": "slurm",
                "default": True,
                "gpus_per_node": "${{ variables.GPUS_PER_NODE }}",
                "account": "${{ variables.ACC }}",
                "partition": "${{ PART }}",
                "time": "${{ variables.TIME }}",
                "nodes": "${{ NODES }}",
                "extra_args": ["--exclusive", "--comment=${{ ACC }}"],
            }
        ],
        workflow=_minimal_workflow(),
    )

    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state = resolve_global_variables(config, state)
    state = resolve_backends(config, state)

    assert "b1" in state.backends
    backend = state.backends["b1"]
    assert isinstance(backend, SlurmBackend)

    # Verify resolved fields (private attrs are OK in unit tests)
    assert backend._account == "acct"
    assert backend._partition == "batch"
    assert backend._nodes == 2
    assert backend._time == "00:10:00"
    assert backend._job_name == "wf"
    assert backend._extra_args == ["--exclusive", "--comment=acct"]
    assert backend._gpu_per_node == 4


def test_sflow_config_rejects_slurm_backend_when_required_fields_missing():
    with pytest.raises(ValidationError):
        SflowConfig.model_validate(
            {
                "version": "0.1",
                "backends": [{"name": "b1", "type": "slurm", "default": True}],
                "workflow": _minimal_workflow().model_dump(),
            }
        )
