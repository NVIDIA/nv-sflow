# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Assembly integration tests for the new ``result`` task entry.

Verifies that ``build_task_graph`` translates the validated
``TaskConfig.result`` into ``Task.result_config`` (a ``ResultConfigRuntime``)
preserving names, regexes, types, units, aggregates, and ``required`` flags.

These tests do NOT exercise actual result file writing — that is covered by
``tests/unit/test_core_results.py``.
"""

from collections.abc import Sequence

from sflow.app.assembly import build_task_graph
from sflow.config.schema import (
    ResultConfig,
    ResultPatternConfig,
    SflowConfig,
    TaskConfig,
    WorkflowConfig,
)
from sflow.core.backend import Allocation, Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.core.state import SflowState
from sflow.core.task import ResultConfigRuntime, ResultSpec
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig


class _LocalLikeBackend(Backend):
    def __init__(self, name: str, allocation: Allocation | None):
        super().__init__(name=name)
        self.allocation = allocation

    async def allocate(self) -> Allocation:  # pragma: no cover
        raise RuntimeError("not used in this unit test")

    async def release(self, allocation: Allocation) -> None:  # pragma: no cover
        raise RuntimeError("not used in this unit test")

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        return BashOperator(BashOperatorConfig(name=name))


def _state_with_local_backend() -> SflowState:
    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    state.backends = {
        "local": _LocalLikeBackend(
            "local",
            allocation=Allocation(
                allocation_id="0",
                nodes=[ComputeNode(name="localhost", ip_address="127.0.0.1", index=0)],
            ),
        )
    }
    state.default_backend = state.backends["local"]
    return state


def _config_with_result(result_obj) -> SflowConfig:
    return SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="benchmark",
                    script=["python run.py"],
                    result=result_obj,
                ),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Translation: TaskConfig.result -> Task.result_config
# ---------------------------------------------------------------------------


def test_simple_map_translated_to_result_config_runtime():
    state = _state_with_local_backend()
    config = _config_with_result(
        {
            "ttft": r"TTFT:\s*([0-9.]+)\s*ms",
            "tps": r"tok/s:\s*([0-9.]+)",
        }
    )

    tg = build_task_graph(config, state)
    t = tg.get_task("benchmark")

    assert isinstance(t.result_config, ResultConfigRuntime)
    assert t.result_config.file is None
    spec_by_name = {s.name: s for s in t.result_config.specs}
    assert set(spec_by_name) == {"ttft", "tps"}

    # Defaults: type=auto, aggregate=last, source=log, required=False
    for spec in t.result_config.specs:
        assert isinstance(spec, ResultSpec)
        assert spec.engine == "regex"
        assert spec.type == "auto"
        assert spec.aggregate == "last"
        assert spec.required is False
        assert spec.source == "log"
        assert spec.regex is not None


def test_advanced_patterns_translated_with_full_metadata():
    state = _state_with_local_backend()
    result_obj = ResultConfig(
        patterns=[
            ResultPatternConfig(
                name="ttft",
                regex=r"TTFT:\s*(?P<value>[0-9.]+)\s*ms",
                type="float",
                unit="ms",
                aggregate="last",
                required=True,
                group="value",
            ),
            ResultPatternConfig(
                name="best_tps",
                regex=r"tok/s:\s*([0-9.]+)",
                type="float",
                aggregate="max",
            ),
        ]
    )
    config = _config_with_result(result_obj)

    tg = build_task_graph(config, state)
    t = tg.get_task("benchmark")

    assert t.result_config is not None
    spec_by_name = {s.name: s for s in t.result_config.specs}

    ttft = spec_by_name["ttft"]
    assert ttft.type == "float"
    assert ttft.unit == "ms"
    assert ttft.aggregate == "last"
    assert ttft.required is True
    assert ttft.group == "value"

    best_tps = spec_by_name["best_tps"]
    assert best_tps.type == "float"
    assert best_tps.aggregate == "max"
    assert best_tps.required is False


def test_file_form_translated_to_runtime_file_field():
    state = _state_with_local_backend()
    config = _config_with_result({"file": "tmp.json"})

    tg = build_task_graph(config, state)
    t = tg.get_task("benchmark")

    assert t.result_config is not None
    assert t.result_config.file == "tmp.json"
    assert t.result_config.specs == []


def test_no_result_entry_leaves_result_config_none():
    state = _state_with_local_backend()
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="benchmark", script=["python run.py"])],
        ),
    )

    tg = build_task_graph(config, state)
    t = tg.get_task("benchmark")

    assert t.result_config is None


def test_legacy_outputs_still_works_alongside_result_absent():
    """Ensure adding ``result_config`` did not break the legacy ``outputs`` MVP path."""
    state = _state_with_local_backend()
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="benchmark",
                    script=["python run.py"],
                    outputs=[{"pattern": "TTFT: {ttft:f} ms"}],
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t = tg.get_task("benchmark")

    assert t.result_config is None
    assert len(t.output_specs) == 1
    assert t.output_specs[0].pattern == "TTFT: {ttft:f} ms"


def test_per_pattern_source_inherits_from_result_source():
    """Per-pattern ``source`` falls back to the parent ``ResultConfig.source``."""
    state = _state_with_local_backend()
    result_obj = ResultConfig(
        patterns=[
            ResultPatternConfig(name="x", regex="x"),  # source unset -> inherits
            ResultPatternConfig(name="y", regex="y", source="log"),
        ],
        source="log",
    )
    config = _config_with_result(result_obj)

    tg = build_task_graph(config, state)
    t = tg.get_task("benchmark")
    assert t.result_config is not None
    sources = {s.name: s.source for s in t.result_config.specs}
    assert sources == {"x": "log", "y": "log"}
