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


def test_resolve_variables_resolves_two_hop_arithmetic_with_default_string_type():
    config = _minimal_config(
        variables=[
            {"name": "ORIGIN", "value": 2},
            {"name": "FIRST_LEVEL", "value": "${{ variables.ORIGIN * 7 }}"},
            {"name": "SECOND_LEVEL", "value": "${{ variables.FIRST_LEVEL - 10 }}"},
        ]
    )
    state = _minimal_state()

    out = resolve_global_variables(config, state)

    assert out.variables["FIRST_LEVEL"].value == "14"
    assert out.variables["SECOND_LEVEL"].value == "4"


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


def test_resolve_variables_exposes_casted_values_to_dependent_expressions():
    config = _minimal_config(
        variables=[
            {"name": "FLAG", "value": "false", "type": "boolean"},
            {"name": "INVERTED", "value": "${{ not variables.FLAG }}", "type": "boolean"},
            {"name": "RATIO", "value": "2.5", "type": "float"},
            {"name": "DOUBLE_RATIO", "value": "${{ variables.RATIO * 2 }}", "type": "float"},
        ]
    )
    state = _minimal_state()

    out = resolve_global_variables(config, state)

    assert out.variables["FLAG"].value is False
    assert out.variables["INVERTED"].value is True
    assert out.variables["RATIO"].value == 2.5
    assert out.variables["DOUBLE_RATIO"].value == 5.0


def test_resolve_variables_raises_on_unresolved_reference():
    config = _minimal_config(
        variables=[
            {"name": "A", "value": "${{ variables.MISSING }}", "type": "string"},
        ]
    )
    state = _minimal_state()

    with pytest.raises(ValueError, match="Unresolved variable expressions"):
        resolve_global_variables(config, state)


def test_resolve_global_variables_defers_backend_context():
    config = _minimal_config(
        variables=[
            {
                "name": "IP_FROM_GLOBAL_VARIABLES",
                "value": "${{ backends.cluster.nodes[0].ip_address }}",
                "type": "string",
            },
        ]
    )
    state = _minimal_state()

    out = resolve_global_variables(config, state)

    assert (
        out.variables["IP_FROM_GLOBAL_VARIABLES"].value
        == "${{ backends.cluster.nodes[0].ip_address }}"
    )


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


def test_resolve_variables_raises_on_get_self_cycle():
    config = _minimal_config(
        variables=[
            {"name": "A", "value": "${{ variables.get('A') }}", "type": "string"},
        ]
    )
    state = _minimal_state()

    with pytest.raises(ValueError, match=r"Unresolved variable expressions.*A"):
        resolve_global_variables(config, state)


def test_resolve_variables_exposes_domain_metadata_in_expressions():
    # A global variable can reference another variable's `domain` list (e.g. to
    # size a value off the largest sweep point), mirroring what already works in
    # task scripts and `compose --resolve`.
    config = _minimal_config(
        variables=[
            {
                "name": "CONCURRENCY",
                "value": 512,
                "type": "integer",
                "domain": [128, 512],
            },
            {"name": "NUM_SERVERS", "value": 4, "type": "integer"},
            {
                "name": "BATCH_SIZE",
                "value": "${{ (variables.CONCURRENCY.domain | max) // variables.NUM_SERVERS }}",
                "type": "integer",
            },
        ]
    )
    state = _minimal_state()

    out = resolve_global_variables(config, state)

    assert out.variables["BATCH_SIZE"].value == 128  # max([128, 512]) // 4
