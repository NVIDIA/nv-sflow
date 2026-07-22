# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

from sflow.utils.script import ensure_line_buffered, prepend_fail_fast


class TestPrependFailFast:
    def test_prepends_marker_and_set_e(self):
        out = prepend_fail_fast(["pip install aiperf", "aiperf profile", "echo done"])
        assert out[0] == "# sflow: fail-fast"
        assert out[1] == "set -e"
        # The user's lines follow unchanged.
        assert out[2:] == ["pip install aiperf", "aiperf profile", "echo done"]

    def test_is_idempotent(self):
        out1 = prepend_fail_fast(["run"])
        out2 = prepend_fail_fast(out1)
        assert out1 == out2

    def test_empty_script_is_unchanged(self):
        assert prepend_fail_fast([]) == []

    def test_failed_command_is_not_masked_by_trailing_echo(self, fake_process):
        # The reported bug: a failed command (`false`) followed by a successful
        # `echo` (exit 0) makes the script exit 0, so the workflow "succeeds"
        # despite the failure. Prepending `set -e` surfaces the failure.
        fake_process.allow_unregistered(True)  # run the real bash
        masked = subprocess.run(
            ["bash", "-c", "\n".join(["false", "echo done"])]
        ).returncode
        assert masked == 0  # without fail-fast, the trailing echo masks `false`

        fixed = subprocess.run(
            ["bash", "-c", "\n".join(prepend_fail_fast(["false", "echo done"]))]
        ).returncode
        assert fixed != 0  # with fail-fast, the task exits non-zero


class TestEnsureLineBuffered:
    def test_prepends_prologue_and_wraps_simple_commands(self):
        script = [
            "echo hello",
            "python -c 'print(1)'",
            "VAR=1 python -c 'print(2)'",
            "cmd | other",
            "if true; then",
            "  echo ok",
            "fi",
            "# comment",
            "",
        ]

        out = ensure_line_buffered(script)

        assert out[0] == "# sflow: line-buffered"
        assert "export PYTHONUNBUFFERED=1" in out
        assert "__sflow_linebuf echo hello" in out
        assert "__sflow_linebuf python -c 'print(1)'" in out

        # Conservative: don't rewrite env-prefix commands, pipelines, or control structures.
        assert "VAR=1 python -c 'print(2)'" in out
        assert "cmd | other" in out
        assert "if true; then" in out
        assert "  __sflow_linebuf echo ok" in out

    def test_is_idempotent(self):
        script = ["echo hello"]
        out1 = ensure_line_buffered(script)
        out2 = ensure_line_buffered(out1)
        assert out1 == out2
