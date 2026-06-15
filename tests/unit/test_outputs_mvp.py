# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging

from sflow.core.outputs import collect_task_outputs
from sflow.core.task_logging import TaskLogPolicy, create_task_log_handler
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


def test_live_output_collector_parses_line_suppressed_from_bounded_log(tmp_path):
    from sflow.core.outputs import TaskOutputCollector, write_task_outputs

    task_out = tmp_path / "producer"
    task_out.mkdir(parents=True)

    t = Task(
        name="producer",
        logger=logging.getLogger("sflow.tests.outputs.live"),
        operator=_NoopOperator(),
        script=["echo hi"],
    )
    t.envs["SFLOW_TASK_OUTPUT_DIR"] = str(task_out)
    t.output_specs = [OutputSpec(pattern="RESULT value={value:d}")]

    log_path = task_out / "producer.log"
    handler = create_task_log_handler(
        log_path,
        TaskLogPolicy(
            mode="bounded",
            keep_lines_per_second=0,
            keep_first_lines=1,
            max_bytes=1024 * 1024,
            backup_count=1,
        ),
    )
    logger = logging.getLogger("sflow.tests.outputs.live.task")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    collector = TaskOutputCollector(t.output_specs)
    for line in ["noise line kept", "RESULT value=42"]:
        collector.feed_line(line)
        logger.info(line)
    handler.close()

    assert "RESULT value=42" not in log_path.read_text()
    parsed = collector.parsed()
    assert parsed["value"] == 42

    asyncio.run(write_task_outputs(t, parsed))
    payload = json.loads((task_out / "outputs.json").read_text())
    assert payload["outputs"]["value"] == 42
