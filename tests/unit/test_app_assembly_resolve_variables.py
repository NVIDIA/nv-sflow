# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sflow.app.assembly import resolve_global_variables
from sflow.config.schema import SflowConfig, TaskConfig, WorkflowConfig
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


def _minimal_state() -> SflowState:
    tg = TaskGraph()
    workflow = Workflow(name="wf", task_graph=tg)
    return SflowState(workflow=workflow)


def _minimal_config(*, variables) -> SflowConfig:
    return SflowConfig(
        version="0.1",
        variables=variables,
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo hi"],
                )
            ],
        ),
    )


def test_resolve_variables_no_variables_is_noop():
    state = _minimal_state()
    config = _minimal_config(variables=None)

    out = resolve_global_variables(config, state)
    assert out is state
    assert out.variables == {}


def test_resolve_variables_resolves_dependencies_and_casts_int():
    config = _minimal_config(
        variables=[
            {"name": "A", "value": 1, "type": "integer"},
            {"name": "B", "value": "${{ variables.A }}", "type": "integer"},
        ]
    )
    state = _minimal_state()

    out = resolve_global_variables(config, state)
    assert out.variables["A"].value == 1
    assert out.variables["B"].value == 1  # rendered as "1", then cast to int


def test_resolve_variables_allows_bare_variable_name_reference():
    config = _minimal_config(
        variables=[
            {"name": "A", "value": 2, "type": "integer"},
            {"name": "B", "value": "${{ A }}", "type": "integer"},
        ]
    )
    state = _minimal_state()

    out = resolve_global_variables(config, state)
    assert out.variables["B"].value == 2


def test_resolve_variables_casts_boolean_strings():
    config = _minimal_config(
        variables=[
            {"name": "FLAG", "value": "true", "type": "boolean"},
            {"name": "FLAG2", "value": "0", "type": "boolean"},
        ]
    )
    state = _minimal_state()

    out = resolve_global_variables(config, state)
    assert out.variables["FLAG"].value is True
    assert out.variables["FLAG2"].value is False


def test_resolve_variables_raises_on_unresolved_reference():
    config = _minimal_config(
        variables=[
            {"name": "A", "value": "${{ variables.MISSING }}", "type": "string"},
        ]
    )
    state = _minimal_state()

    with pytest.raises(ValueError, match="Unresolved variable expressions"):
        resolve_global_variables(config, state)


def test_resolve_variables_raises_on_cycle():
    config = _minimal_config(
        variables=[
            {"name": "A", "value": "${{ variables.B }}", "type": "string"},
            {"name": "B", "value": "${{ variables.A }}", "type": "string"},
        ]
    )
    state = _minimal_state()

    with pytest.raises(ValueError, match=r"Unresolved variable expressions.*A.*B"):
        resolve_global_variables(config, state)
