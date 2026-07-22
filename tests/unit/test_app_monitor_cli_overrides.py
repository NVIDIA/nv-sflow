# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI `--enable-workflow-monitor` / `--enable-task-monitor` injection."""

import pytest

from sflow.app.monitor_cli import (
    apply_monitor_cli_overrides as _apply_monitor_cli_overrides,
)
from sflow.app.monitor_cli import inject_cli_monitors_into_dict
from sflow.config.schema import MonitorConfig, SflowConfig


def _config() -> SflowConfig:
    return SflowConfig.model_validate(
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {"name": "warmup", "script": ["echo a"]},
                    {"name": "work", "script": ["echo b"], "depends_on": ["warmup"]},
                ],
            },
        }
    )


def _task(config: SflowConfig, name: str):
    return next(t for t in config.workflow.tasks if t.name == name)


def test_enable_workflow_monitor_injects_default():
    out = _apply_monitor_cli_overrides(
        _config(), enable_workflow_monitor=True, enable_task_monitors=None
    )
    assert isinstance(out.workflow.monitor, MonitorConfig)
    # Default = all scopes (scopes None) + report enabled.
    assert out.workflow.monitor.scopes is None
    assert out.workflow.monitor.report is not None
    assert out.workflow.monitor.report.enabled is True
    # Tasks untouched.
    assert _task(out, "work").monitor is None


def test_enable_task_monitor_injects_named_tasks_only():
    out = _apply_monitor_cli_overrides(
        _config(), enable_workflow_monitor=False, enable_task_monitors=["work"]
    )
    assert out.workflow.monitor is None
    assert _task(out, "work").monitor is not None
    assert _task(out, "work").monitor.report.enabled is True
    assert _task(out, "warmup").monitor is None


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="unknown task"):
        _apply_monitor_cli_overrides(
            _config(), enable_workflow_monitor=False, enable_task_monitors=["ghost"]
        )


def test_recipe_monitor_not_overridden():
    config = SflowConfig.model_validate(
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "monitor": {"scopes": {"gpu": {}}},  # recipe-defined workflow monitor
                "tasks": [
                    {
                        "name": "work",
                        "script": ["echo b"],
                        "monitor": {"scopes": {"cpu": {}}},  # recipe-defined task monitor
                    }
                ],
            },
        }
    )
    out = _apply_monitor_cli_overrides(
        config, enable_workflow_monitor=True, enable_task_monitors=["work"]
    )
    # Recipe monitors win (CLI is a no-op where a monitor already exists).
    assert out.workflow.monitor.scopes.active_builtin_scopes() == ["gpu"]
    assert _task(out, "work").monitor.scopes.active_builtin_scopes() == ["cpu"]


def test_no_flags_returns_config_unchanged():
    config = _config()
    out = _apply_monitor_cli_overrides(
        config, enable_workflow_monitor=False, enable_task_monitors=None
    )
    assert out is config


def test_inject_dict_treats_null_monitor_as_missing():
    # Parity with the model path (`monitor is not None`): a present-but-null
    # monitor must be injected so the compose snapshot reproduces CLI monitoring.
    merged = {
        "workflow": {
            "name": "wf",
            "monitor": None,
            "tasks": [
                {"name": "warmup", "script": ["echo a"]},
                {"name": "work", "script": ["echo b"], "monitor": None},
            ],
        }
    }
    inject_cli_monitors_into_dict(
        merged, enable_workflow_monitor=True, enable_task_monitors=["work"]
    )
    assert merged["workflow"]["monitor"] == {"report": {"enabled": True}}
    tasks = {t["name"]: t for t in merged["workflow"]["tasks"]}
    assert tasks["work"]["monitor"] == {"report": {"enabled": True}}
    assert "monitor" not in tasks["warmup"]


def test_inject_dict_preserves_existing_monitor():
    merged = {
        "workflow": {
            "name": "wf",
            "monitor": {"scopes": {"gpu": {}}},
            "tasks": [
                {
                    "name": "work",
                    "script": ["echo b"],
                    "monitor": {"scopes": {"cpu": {}}},
                }
            ],
        }
    }
    inject_cli_monitors_into_dict(
        merged, enable_workflow_monitor=True, enable_task_monitors=["work"]
    )
    # Existing (non-null) monitors win; the CLI only fills gaps.
    assert merged["workflow"]["monitor"] == {"scopes": {"gpu": {}}}
    assert merged["workflow"]["tasks"][0]["monitor"] == {"scopes": {"cpu": {}}}
