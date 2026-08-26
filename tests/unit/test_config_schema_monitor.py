# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema validation for the ``monitor:`` key (workflow + task scope)."""

import re

import pytest

from sflow.config.schema import MONITOR_BUILTIN_SCOPES, SflowConfig


def _cfg(workflow: dict) -> dict:
    return {"version": "0.1", "workflow": workflow}


def test_workflow_and_task_monitor_parse():
    c = SflowConfig.model_validate(
        _cfg(
            {
                "name": "wf",
                "monitor": {
                    "interval": 3000,
                    "scopes": {"gpu": {"interval": 1000}, "cpu": {}},
                    "report": {"enabled": True, "format": ["csv", "png"]},
                },
                "tasks": [
                    {"name": "server", "script": ["sleep 1"]},
                    {
                        "name": "bench",
                        "script": ["echo hi"],
                        "depends_on": ["server"],
                        "monitor": {
                            "resources": {
                                "used_by_tasks": ["server"],
                                "gpus": {"count": 2},
                            },
                            "scopes": {
                                "gpu": {},
                                "custom": {"script": ["echo custom"]},
                            },
                        },
                    },
                ],
            }
        )
    )
    wf_monitor = c.workflow.monitor
    assert wf_monitor is not None
    assert wf_monitor.scopes.active_builtin_scopes() == ["cpu", "gpu"]
    assert wf_monitor.report.format == ["csv", "png"]

    bench = c.workflow.tasks[1]
    assert bench.monitor.resources.used_by_tasks == ["server"]
    assert bench.monitor.resources.gpus.count == 2
    assert bench.monitor.scopes.custom.script == ["echo custom"]


def test_report_format_defaults_to_csv_and_svg():
    c = SflowConfig.model_validate(
        _cfg(
            {
                "name": "wf",
                "tasks": [
                    {
                        "name": "a",
                        "script": ["x"],
                        "monitor": {"report": {"enabled": True}},
                    }
                ],
            }
        )
    )
    report = c.workflow.tasks[0].monitor.report
    assert report.enabled is True
    # Lightweight, dependency-free defaults; png is opt-in (needs matplotlib).
    assert report.format == ["csv", "svg"]


def test_report_format_allows_png():
    c = SflowConfig.model_validate(
        _cfg(
            {
                "name": "wf",
                "tasks": [
                    {
                        "name": "a",
                        "script": ["x"],
                        "monitor": {"report": {"enabled": True, "format": ["csv", "png"]}},
                    }
                ],
            }
        )
    )
    assert c.workflow.tasks[0].monitor.report.format == ["csv", "png"]


def test_task_monitor_log_window_parses_scalar_list_and_regex():
    c = SflowConfig.model_validate(
        _cfg(
            {
                "name": "wf",
                "tasks": [
                    {
                        "name": "bench",
                        "script": ["x"],
                        "monitor": {
                            "report": {"enabled": True},
                            "window": {
                                "start": "WARMUP_FINISHED",
                                "end": {
                                    "pattern": [
                                        "BENCHMARK_FINISHED",
                                        r"re:^done \d+$",
                                    ],
                                    "select": "first",
                                },
                            },
                        },
                    }
                ],
            }
        )
    )
    window = c.workflow.tasks[0].monitor.window
    assert window.start.pattern == "WARMUP_FINISHED"
    assert window.start.select is None
    assert window.end.pattern == ["BENCHMARK_FINISHED", r"re:^done \d+$"]
    assert window.end.select == "first"


def test_monitor_log_window_rejects_invalid_placement_and_patterns():
    base_window = {"start": "START", "end": "END"}
    with pytest.raises(ValueError, match="requires report.enabled"):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "tasks": [
                        {
                            "name": "bench",
                            "script": ["x"],
                            "monitor": {
                                "report": {"enabled": False},
                                "window": base_window,
                            },
                        }
                    ],
                }
            )
        )

    with pytest.raises(ValueError, match=re.escape("workflow.monitor.window")):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "monitor": {
                        "report": {"enabled": True},
                        "window": base_window,
                    },
                    "tasks": [{"name": "bench", "script": ["x"]}],
                }
            )
        )

    with pytest.raises(ValueError, match=re.escape("workflow.monitor.window")):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "monitor": {"window": base_window},
                    "tasks": [{"name": "bench", "script": ["x"]}],
                }
            )
        )

    for pattern in ([], ["re:("]):
        with pytest.raises(ValueError, match="pattern cannot be empty|invalid.*regex"):
            SflowConfig.model_validate(
                _cfg(
                    {
                        "name": "wf",
                        "tasks": [
                            {
                                "name": "bench",
                                "script": ["x"],
                                "monitor": {
                                    "report": {"enabled": True},
                                    "window": {
                                        "start": {"pattern": pattern},
                                        "end": "END",
                                    },
                                },
                            }
                        ],
                    }
                )
            )


def test_monitor_without_scopes_defaults_to_all():
    c = SflowConfig.model_validate(
        _cfg({"name": "wf", "monitor": {}, "tasks": [{"name": "a", "script": ["x"]}]})
    )
    # scopes omitted -> no explicit scopes object; planner treats it as all built-ins.
    assert c.workflow.monitor.scopes is None
    assert set(MONITOR_BUILTIN_SCOPES) == {"cpu", "gpu", "memory", "disk", "network"}


def test_unknown_used_by_tasks_rejected():
    with pytest.raises(ValueError, match="used_by_tasks refers to unknown task 'nope'"):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "tasks": [
                        {
                            "name": "a",
                            "script": ["x"],
                            "monitor": {"resources": {"used_by_tasks": ["nope"]}},
                        }
                    ],
                }
            )
        )


def test_workflow_monitor_unknown_used_by_tasks_rejected():
    with pytest.raises(
        ValueError, match=re.escape("workflow.monitor.resources.used_by_tasks")
    ):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "monitor": {"resources": {"used_by_tasks": ["ghost"]}},
                    "tasks": [{"name": "a", "script": ["x"]}],
                }
            )
        )


def test_scopes_present_but_all_disabled_rejected():
    with pytest.raises(ValueError, match="no scope is active"):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "tasks": [
                        {
                            "name": "a",
                            "script": ["x"],
                            "monitor": {"scopes": {"gpu": {"enabled": False}}},
                        }
                    ],
                }
            )
        )


def test_custom_script_must_not_be_empty():
    with pytest.raises(ValueError, match="custom.script cannot be empty"):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "tasks": [
                        {
                            "name": "a",
                            "script": ["x"],
                            "monitor": {"scopes": {"custom": {"script": []}}},
                        }
                    ],
                }
            )
        )


def test_gpu_fields_must_lead_with_index():
    # Valid: leads with index.
    SflowConfig.model_validate(
        _cfg(
            {
                "name": "wf",
                "monitor": {
                    "scopes": {"gpu": {"fields": "index,utilization.gpu,power.draw"}}
                },
                "tasks": [{"name": "a", "script": ["x"]}],
            }
        )
    )
    # Invalid: does not start with index -> rejected up front (not at plan time).
    with pytest.raises(ValueError, match="must start with 'index'"):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "monitor": {"scopes": {"gpu": {"fields": "utilization.gpu,power.draw"}}},
                    "tasks": [{"name": "a", "script": ["x"]}],
                }
            )
        )


def test_monitor_interval_rejects_expression():
    # interval is a concrete int (monitor fields are not expression-resolved), so a
    # ${{ }} string is rejected at validation time rather than crashing at plan time.
    with pytest.raises(ValueError):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "monitor": {"interval": "${{ variables.INTERVAL }}"},
                    "tasks": [{"name": "a", "script": ["x"]}],
                }
            )
        )


@pytest.mark.parametrize("bad", [0, -5, 1, 50, 99])
def test_monitor_interval_below_floor_rejected(bad):
    # Sub-100ms intervals would spin the collector hot, so they are rejected at
    # validation time instead of being silently clamped at plan/run time.
    with pytest.raises(ValueError, match="interval must be >= 100ms"):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "monitor": {"interval": bad},
                    "tasks": [{"name": "a", "script": ["x"]}],
                }
            )
        )


def test_monitor_scope_interval_below_floor_rejected():
    with pytest.raises(ValueError, match="interval must be >= 100ms"):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "monitor": {"scopes": {"gpu": {"interval": 10}}},
                    "tasks": [{"name": "a", "script": ["x"]}],
                }
            )
        )


def test_monitor_interval_at_floor_accepted():
    c = SflowConfig.model_validate(
        _cfg(
            {
                "name": "wf",
                "monitor": {"interval": 100, "scopes": {"gpu": {"interval": 100}}},
                "tasks": [{"name": "a", "script": ["x"]}],
            }
        )
    )
    assert c.workflow.monitor.interval == 100
    assert c.workflow.monitor.scopes.gpu.interval == 100


def test_monitor_operator_override_removed():
    # The unimplemented monitor.operator override was removed; passing it is an
    # error (StrictBaseModel forbids unknown keys) instead of being silently ignored.
    with pytest.raises(ValueError):
        SflowConfig.model_validate(
            _cfg(
                {
                    "name": "wf",
                    "monitor": {"operator": "bash"},
                    "tasks": [{"name": "a", "script": ["x"]}],
                }
            )
        )
