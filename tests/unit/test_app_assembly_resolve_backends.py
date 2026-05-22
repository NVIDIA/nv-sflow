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


def test_resolve_backends_rejects_deferred_global_variable_reference():
    config = SflowConfig(
        version="0.1",
        variables=[
            {
                "name": "HEAD_NODE_IP",
                "value": "${{ backends.b1.nodes[0].ip_address }}",
                "type": "string",
            }
        ],
        backends=[
            {
                "name": "b1",
                "type": "slurm",
                "default": True,
                "gpus_per_node": 4,
                "account": "acct",
                "partition": "${{ variables.HEAD_NODE_IP }}",
                "time": "00:10:00",
                "nodes": 1,
            }
        ],
        workflow=_minimal_workflow(),
    )

    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state = resolve_global_variables(config, state)

    with pytest.raises(ValueError) as exc_info:
        resolve_backends(config, state)

    message = str(exc_info.value)
    assert "Deferred global variable 'HEAD_NODE_IP'" in message
    assert "pre-allocation field 'backends.b1'" in message
    assert "cannot be used while resolving backends" in message


def test_resolve_backends_rejects_bracket_deferred_global_variable_reference():
    config = SflowConfig(
        version="0.1",
        variables=[
            {
                "name": "HEAD_NODE_IP",
                "value": "${{ backends.b1.nodes[0].ip_address }}",
                "type": "string",
            }
        ],
        backends=[
            {
                "name": "b1",
                "type": "slurm",
                "default": True,
                "gpus_per_node": 4,
                "account": "acct",
                "partition": "${{ variables['HEAD_NODE_IP'] }}",
                "time": "00:10:00",
                "nodes": 1,
            }
        ],
        workflow=_minimal_workflow(),
    )

    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state = resolve_global_variables(config, state)

    with pytest.raises(ValueError, match="Deferred global variable 'HEAD_NODE_IP'"):
        resolve_backends(config, state)


def test_sflow_config_rejects_slurm_backend_when_required_fields_missing():
    with pytest.raises(ValidationError):
        SflowConfig.model_validate(
            {
                "version": "0.1",
                "backends": [{"name": "b1", "type": "slurm", "default": True}],
                "workflow": _minimal_workflow().model_dump(),
            }
        )


def test_sflow_config_accepts_slurm_backend_with_zero_gpus_per_node():
    """CPU-only Slurm partitions: gpus_per_node=0 is allowed."""
    config = SflowConfig.model_validate(
        {
            "version": "0.1",
            "backends": [
                {
                    "name": "cpu_cluster",
                    "type": "slurm",
                    "default": True,
                    "account": "acct",
                    "partition": "cpu",
                    "time": "00:10:00",
                    "nodes": 1,
                    "gpus_per_node": 0,
                }
            ],
            "workflow": _minimal_workflow().model_dump(),
        }
    )
    backend_conf = config.backends[0]
    assert backend_conf.type == "slurm"
    assert backend_conf.gpus_per_node == 0


def test_sflow_config_rejects_negative_gpus_per_node():
    """Negative `gpus_per_node` remains a hard error."""
    with pytest.raises(ValidationError):
        SflowConfig.model_validate(
            {
                "version": "0.1",
                "backends": [
                    {
                        "name": "bad",
                        "type": "slurm",
                        "default": True,
                        "account": "acct",
                        "partition": "cpu",
                        "time": "00:10:00",
                        "nodes": 1,
                        "gpus_per_node": -1,
                    }
                ],
                "workflow": _minimal_workflow().model_dump(),
            }
        )


def test_resolve_backends_instantiates_cpu_only_slurm_backend():
    """resolve_backends should accept gpus_per_node=0 and propagate it as 0."""
    config = SflowConfig(
        version="0.1",
        backends=[
            {
                "name": "cpu_cluster",
                "type": "slurm",
                "default": True,
                "account": "acct",
                "partition": "cpu",
                "time": "00:10:00",
                "nodes": 1,
                "gpus_per_node": 0,
            }
        ],
        workflow=_minimal_workflow(),
    )

    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state = resolve_global_variables(config, state)
    state = resolve_backends(config, state)

    backend = state.backends["cpu_cluster"]
    assert isinstance(backend, SlurmBackend)
    assert backend._gpu_per_node == 0
