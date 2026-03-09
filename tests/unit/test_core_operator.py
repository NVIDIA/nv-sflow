# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sflow.plugins.operators import BashOperator, BashOperatorConfig


def test_bash_operator_build_command_does_not_inline_env_exports():
    op = BashOperator(BashOperatorConfig(name="bash"))

    cmd = op.build_command(
        task_name="t1",
        script=["echo hi"],
        envs={"FOO": "bar"},
    )

    # Env is injected by SubprocessLauncher(env=...) to avoid leaking values into command logs.
    assert str(cmd) == "bash -c 'echo hi'"
