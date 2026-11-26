# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
from typer.testing import CliRunner

from sflow.cli import app
from sflow.core.launcher import _console_streams_task_output
from sflow.core.log_offload import (
    OFFLOAD_TASK_LOGS_ENV,
    STREAM_TASK_LOGS,
    offload_enabled,
)


@pytest.mark.parametrize("fail", [False, True])
def test_stream_task_logs_is_scoped_to_cli_invocation(tmp_path, monkeypatch, fail):
    run_mod = importlib.import_module("sflow.cli.run")
    recipe = tmp_path / "unused.yaml"
    recipe.write_text('version: "0.1"\nworkflow: {name: test, tasks: []}\n')
    monkeypatch.setenv(OFFLOAD_TASK_LOGS_ENV, "1")
    monkeypatch.setattr("sflow.core.log_offload.stdout_is_tty", lambda: False)
    observed = []

    def run(*args, **kwargs):
        observed.append((_console_streams_task_output(), offload_enabled(True)))
        if fail:
            raise ValueError("test workflow failure")
        return None

    monkeypatch.setattr(run_mod._sflow_app, "run", run)
    for flags in (["--stream-task-logs"], []):
        result = CliRunner().invoke(
            app,
            ["run", "-f", str(recipe), *flags],
        )
        assert result.exit_code == (1 if fail else 0), result.output
        assert not STREAM_TASK_LOGS.get()
    assert observed == [(True, False), (False, True)]
