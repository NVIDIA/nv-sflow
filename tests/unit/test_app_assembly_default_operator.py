# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from sflow.app.assembly import build_state
from sflow.config.schema import SflowConfig, WorkflowConfig


def test_build_state_creates_default_operator_when_operators_enabled_and_task_omits_operator():
    config = SflowConfig(
        version="0.1",
        operators={"op_bash": {"type": "bash"}},
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                # No backend, no operator: should default to local backend and default bash operator.
                {"name": "t1", "script": ["echo 1"]},
            ],
        ),
    )

    state = asyncio.run(build_state(config, allocate=False))
    t1 = state.workflow.task_graph.get_task("t1")
    assert t1.operator is not None
    assert t1.operator.config.type == "bash"
