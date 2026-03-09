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
