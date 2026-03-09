# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from sflow.app.assembly import build_state
from sflow.config.schema import SflowConfig, TaskConfig, VariableConfig, WorkflowConfig


def test_task_script_resolves_jinja_expressions_and_injects_variables_env():
    config = SflowConfig(
        version="0.1",
        variables=[VariableConfig(name="GPU_COUNT", type="integer", value=2)],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=[
                        # Jinja-style expression should be resolved at plan time.
                        "echo jinja=${{ variables.GPU_COUNT }}",
                        # Shell-style $GPU_COUNT should work via env injection.
                        "echo shell=${GPU_COUNT}",
                    ],
                )
            ],
        ),
    )

    state = asyncio.run(build_state(config, allocate=False))
    t1 = state.workflow.task_graph.get_task("t1")

    assert t1.script[0] == "echo jinja=2"
    assert t1.envs["GPU_COUNT"] == "2"
