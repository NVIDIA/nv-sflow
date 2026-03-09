# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from sflow.app.assembly import build_state
from sflow.config.schema import SflowConfig, TaskConfig, WorkflowConfig


def test_build_state_defaults_to_local_backend_and_bash_operator_when_backends_missing():
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(name="t1", script=["echo 1"]),
                TaskConfig(name="t2", script=["echo 2"], depends_on=["t1"]),
            ],
        ),
    )

    state = asyncio.run(build_state(config, allocate=False))

    assert state.default_backend is not None
    assert state.default_backend.name == "local"
    assert set(state.backends.keys()) == {"local"}

    t1 = state.workflow.task_graph.get_task("t1")
    t2 = state.workflow.task_graph.get_task("t2")
    assert t1.operator.config.type == "bash"
    assert t2.operator.config.type == "bash"


def test_build_state_attaches_probes_from_config():
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="svc",
                    script=["echo boot"],
                    probes={
                        "readiness": {
                            "tcp_port": {"port": 12345, "host": "127.0.0.1"},
                            "interval": 1,
                            "timeout": 1,
                        }
                    },
                )
            ],
        ),
    )

    state = asyncio.run(build_state(config, allocate=False))
    svc = state.workflow.task_graph.get_task("svc")
    assert len(svc.probes) == 1
