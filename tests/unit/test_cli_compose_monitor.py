# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composed-snapshot injection of CLI-enabled monitors (`_compose_files`)."""

import textwrap

import pytest
import yaml

from sflow.cli.compose import _compose_files


def _write(tmp_path, text):
    p = tmp_path / "wf.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def _recipe(tmp_path):
    return _write(
        tmp_path,
        """
        version: "0.1"
        workflow:
          name: wf
          tasks:
            - name: warmup
              script: [echo a]
            - name: work
              depends_on: [warmup]
              script: [echo b]
        """,
    )


def test_compose_injects_workflow_and_task_monitor(tmp_path):
    out = _compose_files(
        [_recipe(tmp_path)],
        None,
        None,
        "info",
        enable_workflow_monitor=True,
        enable_task_monitors=["work"],
    )
    data = yaml.safe_load(out)
    # Workflow-level monitor present.
    assert data["workflow"]["monitor"] == {"report": {"enabled": True}}
    # Task-level monitor only on the named task.
    tasks = {t["name"]: t for t in data["workflow"]["tasks"]}
    assert tasks["work"]["monitor"] == {"report": {"enabled": True}}
    assert "monitor" not in tasks["warmup"]


def test_compose_without_flags_has_no_monitor(tmp_path):
    out = _compose_files([_recipe(tmp_path)], None, None, "info")
    data = yaml.safe_load(out)
    assert "monitor" not in data["workflow"]
    for t in data["workflow"]["tasks"]:
        assert "monitor" not in t


def test_compose_does_not_override_recipe_monitor(tmp_path):
    recipe = _write(
        tmp_path,
        """
        version: "0.1"
        workflow:
          name: wf
          monitor:
            scopes:
              gpu: {}
          tasks:
            - name: work
              script: [echo b]
              monitor:
                scopes:
                  cpu: {}
        """,
    )
    out = _compose_files(
        [recipe],
        None,
        None,
        "info",
        enable_workflow_monitor=True,
        enable_task_monitors=["work"],
    )
    data = yaml.safe_load(out)
    # Recipe-defined monitors are preserved (CLI is a no-op where one exists).
    assert data["workflow"]["monitor"] == {"scopes": {"gpu": {}}}
    work = data["workflow"]["tasks"][0]
    assert work["monitor"] == {"scopes": {"cpu": {}}}


def test_compose_unknown_task_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown task"):
        _compose_files(
            [_recipe(tmp_path)],
            None,
            None,
            "info",
            enable_task_monitors=["ghost"],
        )
