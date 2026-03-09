# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sflow.app.assembly import resolve_global_variables, resolve_workflow_variables
from sflow.config.schema import SflowConfig, TaskConfig, WorkflowConfig
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


def _minimal_state() -> SflowState:
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    return SflowState(workflow=wf)


def _minimal_config(*, global_vars=None, workflow_vars=None) -> SflowConfig:
    return SflowConfig(
        version="0.1",
        variables=global_vars,
        workflow=WorkflowConfig(
            name="wf",
            variables=workflow_vars,
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )


def test_resolve_workflow_variables_noop_when_none():
    cfg = _minimal_config(global_vars=None, workflow_vars=None)
    state = _minimal_state()
    out = resolve_workflow_variables(cfg, state)
    assert out is state
    assert out.variables == {}


def test_resolve_workflow_variables_can_reference_global_and_casts():
    cfg = _minimal_config(
        global_vars=[{"name": "A", "value": 3, "type": "integer"}],
        workflow_vars=[
            {"name": "B", "value": "${{ variables.A }}", "type": "integer"},
            {"name": "C", "value": "${{ B }}", "type": "integer"},
        ],
    )
    state = _minimal_state()
    state = resolve_global_variables(cfg, state)
    state = resolve_workflow_variables(cfg, state)

    assert state.variables["A"].value == 3
    assert state.variables["B"].value == 3
    assert state.variables["C"].value == 3


def test_resolve_workflow_variables_raises_on_missing_ref():
    cfg = _minimal_config(
        global_vars=None,
        workflow_vars=[
            {"name": "B", "value": "${{ variables.MISSING }}", "type": "string"}
        ],
    )
    state = _minimal_state()

    with pytest.raises(ValueError, match="Unresolved variable expressions"):
        resolve_workflow_variables(cfg, state)
