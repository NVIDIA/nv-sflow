# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the srun per-task log offload (aligned `--output` mode)."""

import io
import logging
import os
import re

from sflow.logging import CoalescingFileHandler, DeferredTaskLogHandler
from sflow.plugins.operators.srun import (
    _LOG_PREFIX_HELPER_SRC,
    OFFLOAD_TASK_LOGS_ENV,
    SrunOperator,
    SrunOperatorConfig,
)
from sflow.utils.parser import strip_sflow_log_prefix

# Byte-format parity with logging.Formatter's default asctime, incl. millis,
# plus the folded srun rank ("<procid>: ").
_ALIGNED_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - sflow\.task\.t - INFO - \d+: "
)


def _envs(tmp_path):
    task_out = tmp_path / "t"
    task_out.mkdir(parents=True, exist_ok=True)
    return {
        "SFLOW_TASK_OUTPUT_DIR": str(task_out),
        "SFLOW_WORKFLOW_OUTPUT_DIR": str(tmp_path),
    }


def _force_non_tty(monkeypatch):
    # Make the offload decision deterministic regardless of how pytest is run
    # (e.g. `pytest -s` from a terminal would otherwise trigger the TTY fallback).
    monkeypatch.setattr("sflow.core.log_offload.stdout_is_tty", lambda: False)


def test_stream_mode_has_label_and_no_output(tmp_path, monkeypatch):
    monkeypatch.delenv(OFFLOAD_TASK_LOGS_ENV, raising=False)
    _force_non_tty(monkeypatch)
    # Offload is the default now, so stream mode must be requested explicitly.
    op = SrunOperator(SrunOperatorConfig(name="t", log_to_file=False))
    parts = op.build_command(
        task_name="t", script=["echo hi"], envs=_envs(tmp_path)
    ).as_list()
    assert "--output" not in parts
    assert "--label" in parts
    assert "PIPESTATUS" not in parts[-1]
    assert op.writes_own_task_log() is False


def test_offload_emits_output_drops_label_and_wraps(tmp_path, monkeypatch):
    monkeypatch.delenv(OFFLOAD_TASK_LOGS_ENV, raising=False)
    _force_non_tty(monkeypatch)
    op = SrunOperator(SrunOperatorConfig(name="t", log_to_file=True))
    envs = _envs(tmp_path)
    parts = op.build_command(task_name="t", script=["echo hi"], envs=envs).as_list()
    assert "--label" not in parts
    out_idx = parts.index("--output")
    assert parts[out_idx + 1] == os.path.join(envs["SFLOW_TASK_OUTPUT_DIR"], "t.log")
    wrapped = parts[-1]
    assert wrapped.startswith("{")
    assert "2>&1 |" in wrapped
    assert wrapped.rstrip().endswith('exit "${PIPESTATUS[0]}"')
    assert op.writes_own_task_log() is True
    # The python3 prefixer helper is materialized next to the workflow output dir.
    assert (tmp_path / ".sflow" / "log_prefix.py").is_file()


def test_offload_requires_task_output_dir(monkeypatch):
    monkeypatch.delenv(OFFLOAD_TASK_LOGS_ENV, raising=False)
    _force_non_tty(monkeypatch)
    op = SrunOperator(SrunOperatorConfig(name="t", log_to_file=True))
    # Without SFLOW_TASK_OUTPUT_DIR there is nowhere to point --output, so offload
    # cannot engage and we fall back to streaming.
    parts = op.build_command(task_name="t", script=["echo hi"], envs={}).as_list()
    assert "--output" not in parts
    assert "set -o pipefail" not in parts[-1]


def test_env_override_precedence(tmp_path, monkeypatch):
    _force_non_tty(monkeypatch)
    # config off, env on -> offload engaged
    monkeypatch.setenv(OFFLOAD_TASK_LOGS_ENV, "1")
    assert SrunOperator(SrunOperatorConfig(name="t", log_to_file=False)).writes_own_task_log()
    # config on, env off -> stream
    monkeypatch.setenv(OFFLOAD_TASK_LOGS_ENV, "0")
    assert not SrunOperator(SrunOperatorConfig(name="t", log_to_file=True)).writes_own_task_log()


def test_tty_session_falls_back_to_stream(monkeypatch):
    monkeypatch.setenv(OFFLOAD_TASK_LOGS_ENV, "1")
    monkeypatch.setattr("sflow.core.log_offload.stdout_is_tty", lambda: True)
    assert SrunOperator(SrunOperatorConfig(name="t")).writes_own_task_log() is False


def test_python3_prefixer_reproduces_driver_format(monkeypatch):
    # Drive the materialized python3 prefixer logic directly (the unit suite blocks
    # real subprocesses). It must reproduce logging.Formatter's asctime byte-for-byte
    # (incl. milliseconds) and fold the Slurm rank to mirror srun --label.
    namespace: dict = {}
    exec(compile(_LOG_PREFIX_HELPER_SRC, "log_prefix.py", "exec"), namespace)

    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setattr("sys.stdin", io.StringIO("hello\nworld\n"))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.argv", ["log_prefix.py", "sflow.task.t"])

    namespace["main"]()

    lines = out.getvalue().splitlines()
    assert _ALIGNED_RE.match(lines[0]), lines[0]
    assert lines[0].endswith("2: hello")
    assert lines[1].endswith("2: world")


def test_offload_wrapper_uses_pipestatus_for_exit_code(tmp_path, monkeypatch):
    # ${PIPESTATUS[0]} ensures the task's exit status (not the prefixer's) reaches
    # srun; full bash execution is covered by the e2e suite (real subprocesses are
    # blocked here). We assert the wrapper structure that guarantees it.
    monkeypatch.delenv(OFFLOAD_TASK_LOGS_ENV, raising=False)
    _force_non_tty(monkeypatch)
    op = SrunOperator(SrunOperatorConfig(name="t", log_to_file=True))
    wrapped = op.build_command(
        task_name="t", script=["echo before", "exit 7"], envs=_envs(tmp_path)
    ).as_list()[-1]
    assert wrapped.startswith("{")
    assert wrapped.rstrip().endswith('exit "${PIPESTATUS[0]}"')
    assert "} 2>&1 | {" in wrapped


def test_runtime_warning_when_offload_enabled():
    cfg = SrunOperatorConfig(name="t", log_to_file=True)
    assert any("offload" in w.lower() for w in cfg.runtime_warnings())
    cfg_off = SrunOperatorConfig(name="t", log_to_file=False)
    assert not any("offload" in w.lower() for w in cfg_off.runtime_warnings())


def test_strip_sflow_log_prefix_only_when_present():
    # Prefixed lines: prefix removed, message (incl. embedded " - ") preserved.
    assert (
        strip_sflow_log_prefix(
            "2026-06-16 16:20:51,938 - sflow.task.t - INFO - 0: a - b"
        )
        == "0: a - b"
    )
    # Second-resolution prefix (no millis) also handled.
    assert (
        strip_sflow_log_prefix("2026-06-16 16:20:51 - sflow.task.t - INFO - hi")
        == "hi"
    )
    # Raw line that merely contains " - " is left untouched.
    assert strip_sflow_log_prefix("raw - with - dashes") == "raw - with - dashes"
    assert strip_sflow_log_prefix("") == ""


def test_coalescing_file_handler_writes_and_flushes(tmp_path):
    path = tmp_path / "c.log"
    handler = CoalescingFileHandler(str(path), flush_interval=0.0)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test.coalesce")
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info("line1")
    logger.info("line2")
    handler.close()
    assert path.read_text().splitlines() == ["line1", "line2"]


def test_deferred_handler_appends_to_existing_task_log_only_on_flush(tmp_path):
    # Simulates offload: the operator (srun --output) writes <task>.log first;
    # the handler must NOT touch it until flush, then APPEND (never truncate) so
    # the driver-side diagnostics land in the same file -- no scattered sidecar.
    task_log = tmp_path / "printer.log"
    task_log.write_text("0: task body line A\n0: task body line B\n")

    handler = DeferredTaskLogHandler(str(task_log))
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test.deferred")
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info("srun: error: nodeX: task 0: Exited with exit code 1")
    # Before flush the operator-owned file is untouched (single-writer safe).
    assert task_log.read_text() == "0: task body line A\n0: task body line B\n"

    handler.flush()
    assert task_log.read_text().splitlines() == [
        "0: task body line A",
        "0: task body line B",
        "srun: error: nodeX: task 0: Exited with exit code 1",
    ]

    # Idempotent: a second flush with no new records must not duplicate anything
    # (the launcher's finally + logging shutdown can both flush).
    handler.flush()
    handler.close()
    assert task_log.read_text().count("Exited with exit code 1") == 1


def test_deferred_handler_no_diagnostics_leaves_file_untouched(tmp_path):
    # A clean offload task emits no driver-side diagnostics: the handler must not
    # create or modify <task>.log (the operator's content is the whole story).
    task_log = tmp_path / "printer.log"
    handler = DeferredTaskLogHandler(str(task_log))
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.flush()
    handler.close()
    assert not task_log.exists()
