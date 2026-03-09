# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging

from sflow.core.outputs import collect_task_outputs
from sflow.core.task import OutputSpec, Task
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.command import Command


class _NoopOperator(Operator):
    def __init__(self):
        super().__init__(OperatorConfig(type="noop"))

    def build_command(
        self, *, task_name: str, script, envs
    ) -> Command:  # pragma: no cover
        return Command(exec="true")


def test_outputs_mvp_parses_log_and_writes_outputs_json(tmp_path):
    task_out = tmp_path / "t1"
    task_out.mkdir(parents=True)

    t = Task(
        name="t1",
        logger=logging.getLogger("sflow.tests.outputs"),
        operator=_NoopOperator(),
        script=["echo hi"],
    )
    t.envs["SFLOW_TASK_OUTPUT_DIR"] = str(task_out)
    t.output_specs = [
        OutputSpec(pattern="TTFT: {ttft:f} ms"),
        OutputSpec(pattern="tok/s: {tps:f}"),
    ]

    # Write a task log as sflow would.
    (task_out / "t1.log").write_text(
        "\n".join(
            [
                "hello",
                "TTFT: 42.5 ms",
                "tok/s: 123.0",
                "",
            ]
        )
    )

    parsed = asyncio.run(collect_task_outputs(t))
    assert parsed["ttft"] == 42.5
    assert parsed["tps"] == 123.0
    assert t.outputs == parsed

    payload = json.loads((task_out / "outputs.json").read_text())
    assert payload["task"] == "t1"
    assert payload["outputs"]["ttft"] == 42.5
    assert payload["outputs"]["tps"] == 123.0
