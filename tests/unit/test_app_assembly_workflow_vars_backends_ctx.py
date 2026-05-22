# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from sflow.app.assembly import build_state
from sflow.config.schema import (
    SflowConfig,
    TaskConfig,
    WorkflowConfig,
)


def test_build_state_allocate_false_seeds_placeholder_backend_nodes_for_workflow_vars():
    config = SflowConfig(
        version="0.1",
        variables=[
            {"name": "ACC", "value": "acct", "type": "string"},
            {"name": "PART", "value": "batch", "type": "string"},
            {"name": "TIME", "value": "00:10:00", "type": "string"},
            {"name": "NODES", "value": 2, "type": "integer"},
            {"name": "GPUS_PER_NODE", "value": 4, "type": "integer"},
        ],
        backends=[
            {
                "name": "b1",
                "type": "slurm",
                "default": True,
                "account": "${{ variables.ACC }}",
                "partition": "${{ variables.PART }}",
                "time": "${{ variables.TIME }}",
                "nodes": "${{ variables.NODES }}",
                "gpus_per_node": "${{ variables.GPUS_PER_NODE }}",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            variables=[
                {
                    "name": "HEAD_NODE_IP",
                    "value": "${{ backends.b1.nodes[0].ip_address }}",
                    "type": "string",
                }
            ],
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    state = asyncio.run(build_state(config, allocate=False))
    assert state.variables["HEAD_NODE_IP"].value == "0.0.0.1"


def test_build_state_allocate_false_resolves_deferred_global_backend_variable():
    config = SflowConfig(
        version="0.1",
        variables=[
            {"name": "ACC", "value": "acct", "type": "string"},
            {"name": "PART", "value": "batch", "type": "string"},
            {"name": "TIME", "value": "00:10:00", "type": "string"},
            {"name": "NODES", "value": 2, "type": "integer"},
            {"name": "GPUS_PER_NODE", "value": 4, "type": "integer"},
            {
                "name": "HEAD_NODE_IP",
                "value": "${{ backends.b1.nodes[0].ip_address }}",
                "type": "string",
            },
        ],
        backends=[
            {
                "name": "b1",
                "type": "slurm",
                "default": True,
                "account": "${{ variables.ACC }}",
                "partition": "${{ variables.PART }}",
                "time": "${{ variables.TIME }}",
                "nodes": "${{ variables.NODES }}",
                "gpus_per_node": "${{ variables.GPUS_PER_NODE }}",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=[
                        "echo jinja=${{ variables.HEAD_NODE_IP }}",
                        "echo shell=${HEAD_NODE_IP}",
                    ],
                )
            ],
        ),
    )

    state = asyncio.run(build_state(config, allocate=False))
    t1 = state.workflow.task_graph.get_task("t1")

    assert state.variables["HEAD_NODE_IP"].value == "0.0.0.1"
    assert t1.script[0] == "echo jinja=0.0.0.1"
    assert t1.envs["HEAD_NODE_IP"] == "0.0.0.1"


def test_build_state_resolves_indirect_deferred_global_backend_variable():
    config = SflowConfig(
        version="0.1",
        variables=[
            {"name": "ACC", "value": "acct", "type": "string"},
            {"name": "PART", "value": "batch", "type": "string"},
            {"name": "TIME", "value": "00:10:00", "type": "string"},
            {
                "name": "HEAD_NODE_IP",
                "value": "${{ backends.b1.nodes[0].ip_address }}",
                "type": "string",
            },
            {
                "name": "SERVER_URL",
                "value": "http://${{ variables.HEAD_NODE_IP }}:8000",
                "type": "string",
            },
        ],
        backends=[
            {
                "name": "b1",
                "type": "slurm",
                "default": True,
                "account": "${{ variables.ACC }}",
                "partition": "${{ variables.PART }}",
                "time": "${{ variables.TIME }}",
                "nodes": 1,
                "gpus_per_node": 4,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo ${{ variables.SERVER_URL }}"],
                )
            ],
        ),
    )

    state = asyncio.run(build_state(config, allocate=False))
    t1 = state.workflow.task_graph.get_task("t1")

    assert state.variables["SERVER_URL"].value == "http://0.0.0.1:8000"
    assert t1.script[0] == "echo http://0.0.0.1:8000"


def test_build_state_resolves_get_alias_deferred_global_backend_variable():
    config = SflowConfig(
        version="0.1",
        variables=[
            {
                "name": "HEAD_NODE_IP",
                "value": "${{ backends.b1.nodes[0].ip_address }}",
                "type": "string",
            },
            {
                "name": "SERVER_URL",
                "value": "http://${{ variables.get('HEAD_NODE_IP') }}:8000",
                "type": "string",
            },
        ],
        backends=[
            {
                "name": "b1",
                "type": "slurm",
                "default": True,
                "account": "acct",
                "partition": "batch",
                "time": "00:10:00",
                "nodes": 1,
                "gpus_per_node": 4,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo ${{ variables.SERVER_URL }}"])],
        ),
    )

    state = asyncio.run(build_state(config, allocate=False))

    assert state.variables["SERVER_URL"].value == "http://0.0.0.1:8000"
