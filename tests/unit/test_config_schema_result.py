# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Schema tests for the new ``result`` task entry.

Exercises the three accepted shapes (simple map, advanced patterns, file source)
plus validation rules from
``docs/developer/dev-notes/result-parsing.md``.
"""

import pytest
from pydantic import ValidationError

from sflow.config.schema import (
    RESULT_AGGREGATES,
    RESULT_TYPES,
    ResultConfig,
    ResultPatternConfig,
    SflowConfig,
    TaskConfig,
)


# ---------------------------------------------------------------------------
# ResultPatternConfig field validation
# ---------------------------------------------------------------------------


class TestResultPatternConfigFieldValidation:
    def test_minimal_fields(self):
        p = ResultPatternConfig(name="ttft", regex=r"TTFT:\s*([0-9.]+)\s*ms")
        assert p.name == "ttft"
        assert p.aggregate == "last"
        assert p.type == "auto"
        assert p.required is False
        assert p.unit is None
        assert p.source is None
        assert p.group is None

    def test_full_advanced_pattern(self):
        p = ResultPatternConfig(
            name="ttft",
            regex=r"TTFT:\s*(?P<value>[0-9.]+)\s*ms",
            type="float",
            unit="ms",
            aggregate="last",
            required=True,
            source="log",
            group="value",
        )
        assert p.aggregate == "last"
        assert p.unit == "ms"
        assert p.required is True
        assert p.group == "value"

    @pytest.mark.parametrize("agg", list(RESULT_AGGREGATES))
    def test_all_aggregates_accepted(self, agg):
        p = ResultPatternConfig(name="x", regex="x", aggregate=agg)
        assert p.aggregate == agg

    def test_invalid_aggregate_rejected(self):
        with pytest.raises(ValidationError, match="aggregate"):
            ResultPatternConfig(name="x", regex="x", aggregate="median")

    @pytest.mark.parametrize("ty", list(RESULT_TYPES))
    def test_all_types_accepted(self, ty):
        p = ResultPatternConfig(name="x", regex="x", type=ty)
        assert p.type == ty

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError, match="type"):
            ResultPatternConfig(name="x", regex="x", type="datetime")

    def test_source_log_only(self):
        ResultPatternConfig(name="x", regex="x", source="log")
        with pytest.raises(ValidationError, match="source"):
            ResultPatternConfig(name="x", regex="x", source="stdout")

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            ResultPatternConfig(name="x")  # missing regex
        with pytest.raises(ValidationError):
            ResultPatternConfig(regex="x")  # missing name


# ---------------------------------------------------------------------------
# ResultConfig validation rules
# ---------------------------------------------------------------------------


class TestResultConfigValidationRules:
    def test_must_have_patterns_or_file(self):
        with pytest.raises(ValidationError, match="patterns.*file|file.*patterns"):
            ResultConfig()

    def test_patterns_only_is_valid(self):
        c = ResultConfig(
            patterns=[ResultPatternConfig(name="t", regex="t")],
        )
        assert c.patterns is not None
        assert len(c.patterns) == 1
        assert c.file is None
        assert c.source == "log"

    def test_file_only_is_valid(self):
        c = ResultConfig(file="tmp.json")
        assert c.file == "tmp.json"
        assert c.patterns is None

    def test_patterns_and_file_together_rejected(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ResultConfig(
                patterns=[ResultPatternConfig(name="t", regex="t")],
                file="tmp.json",
            )

    def test_unknown_source_rejected(self):
        with pytest.raises(ValidationError, match="source"):
            ResultConfig(file="tmp.json", source="stdout")

    def test_duplicate_pattern_names_rejected(self):
        with pytest.raises(ValidationError, match="duplicate.*ttft"):
            ResultConfig(
                patterns=[
                    ResultPatternConfig(name="ttft", regex=r"TTFT:\s*([0-9.]+)"),
                    ResultPatternConfig(name="ttft", regex=r"tok/s:\s*([0-9.]+)"),
                ],
            )

    def test_file_source_requires_json_path(self):
        with pytest.raises(ValidationError, match="result.file.*JSON"):
            ResultConfig(file=r"source file:\s*(\S+)")


# ---------------------------------------------------------------------------
# TaskConfig.result entry – three accepted shapes
# ---------------------------------------------------------------------------


class TestTaskConfigResult:
    def test_simple_map_normalized_to_patterns(self):
        t = TaskConfig(
            name="benchmark",
            script=["python run.py"],
            result={
                "ttft": r"TTFT:\s*([0-9.]+)\s*ms",
                "tps": r"tok/s:\s*([0-9.]+)",
            },
        )
        assert t.result is not None
        assert t.result.patterns is not None
        names = sorted(p.name for p in t.result.patterns)
        assert names == ["tps", "ttft"]
        # Defaults from the spec: type=auto, aggregate=last, source=log, required=False
        for p in t.result.patterns:
            assert p.type == "auto"
            assert p.aggregate == "last"
            assert p.required is False

    def test_advanced_patterns_form(self):
        t = TaskConfig(
            name="benchmark",
            script=["python run.py"],
            result={
                "patterns": [
                    {
                        "name": "ttft",
                        "regex": r"TTFT:\s*(?P<value>[0-9.]+)\s*ms",
                        "type": "float",
                        "unit": "ms",
                        "aggregate": "last",
                        "required": True,
                    },
                    {
                        "name": "best_tps",
                        "regex": r"tok/s:\s*([0-9.]+)",
                        "type": "float",
                        "aggregate": "max",
                    },
                ]
            },
        )
        assert t.result is not None
        names = [p.name for p in t.result.patterns]
        assert names == ["ttft", "best_tps"]
        assert t.result.patterns[0].required is True
        assert t.result.patterns[1].aggregate == "max"

    def test_file_form(self):
        t = TaskConfig(
            name="benchmark",
            script=["python run.py --out $SFLOW_TASK_OUTPUT_DIR/tmp.json"],
            result={"file": "tmp.json"},
        )
        assert t.result is not None
        assert t.result.file == "tmp.json"
        assert t.result.patterns is None

    def test_single_file_key_simple_map_is_rejected_with_guidance(self):
        with pytest.raises(ValidationError, match="metric named 'file'"):
            TaskConfig(
                name="benchmark",
                script=["python run.py"],
                result={"file": r"source file:\s*(\S+)"},
            )

    def test_simple_map_with_invalid_value_type_falls_through_to_patterns_validation(
        self,
    ):
        # When user writes `result: { foo: 123 }` this is not a simple map (non-string value)
        # and should fail because it doesn't match patterns/file shape.
        with pytest.raises(ValidationError):
            TaskConfig(
                name="bad",
                script=["echo"],
                result={"foo": 123},  # type: ignore[dict-item]
            )

    def test_empty_result_dict_rejected(self):
        with pytest.raises(ValidationError):
            TaskConfig(name="bad", script=["echo"], result={})

    def test_in_full_workflow_config(self):
        cfg = SflowConfig.model_validate(
            {
                "version": "0.1",
                "workflow": {
                    "name": "wf",
                    "tasks": [
                        {
                            "name": "benchmark",
                            "script": ["python run.py"],
                            "result": {
                                "ttft": r"TTFT:\s*([0-9.]+)\s*ms",
                            },
                        }
                    ],
                },
            }
        )
        t = cfg.workflow.tasks[0]
        assert t.result is not None
        assert t.result.patterns is not None
        assert t.result.patterns[0].name == "ttft"
