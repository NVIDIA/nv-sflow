# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Task.runnable_script`` fail-fast behavior.

sflow runs a task's SHELL script fail-fast by default (``set -e`` prepended
to the runnable form) so a failed command fails the task instead of a later
successful command (a trailing ``echo``) masking it. ``script`` itself stays the
user's resolved lines -- only the runnable form carries the prelude. The ``python``
operator (script is Python source) and ``fail_fast: false`` opt out.
"""

from __future__ import annotations

import logging

from sflow.core.task import Task
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.plugins.operators.python import PythonOperator, PythonOperatorConfig

_LOG = logging.getLogger("test.fail_fast")

_PRELUDE = ["# sflow: fail-fast", "set -e"]


def _task(operator, script, *, fail_fast=True):
    task = Task(name="t", logger=_LOG, operator=operator, script=list(script))
    task.fail_fast = fail_fast
    return task


def test_shell_operator_runnable_script_is_fail_fast():
    task = _task(BashOperator(BashOperatorConfig(name="b")), ["pip install x", "run"])
    assert task.runnable_script[:2] == _PRELUDE
    assert task.runnable_script[2:] == ["pip install x", "run"]
    # `script` itself is untouched (fail-fast is an execution-time concern).
    assert task.script == ["pip install x", "run"]


def test_python_operator_runnable_script_is_unchanged():
    # The python operator runs `python -c <script>`; a shell `set -e`
    # prelude would be a SyntaxError, so it must NOT be injected.
    task = _task(PythonOperator(PythonOperatorConfig(name="p")), ["import sys", "sys.exit(1)"])
    assert task.runnable_script == ["import sys", "sys.exit(1)"]


def test_fail_fast_false_opts_out():
    task = _task(BashOperator(BashOperatorConfig(name="b")), ["run"], fail_fast=False)
    assert task.runnable_script == ["run"]


def test_launch_command_carries_the_fail_fast_prelude():
    # The build_command path runs the runnable_script, so the prelude reaches the
    # actual `bash -c` payload.
    task = _task(BashOperator(BashOperatorConfig(name="b")), ["false", "echo done"])
    payload = task.launch_command.as_list()[-1]
    assert "set -e" in payload
    assert payload.index("set -e") < payload.index("false")
