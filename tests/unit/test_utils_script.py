# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sflow.utils.script import ensure_line_buffered


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
