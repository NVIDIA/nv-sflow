# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end matrix: per-task log offload ON/OFF x ``--tui`` ON/OFF.

Runs a real local-backend workflow for all four combinations of
``--offload-task-logs`` / ``--no-offload-task-logs`` and ``--tui`` / no-tui and
asserts the invariants the user cares about:

1. the run completes successfully (a run dir is produced),
2. the per-task ``<task>.log`` is ALWAYS written with the task's stdout,
3. no scattered ``<task>.orchestration.log`` sidecar is left behind (offload
   merges the driver-side diagnostics into ``<task>.log`` itself), and
4. console streaming is correct for the scenario:
     * ``--tui`` / interactive TTY -> task output is streamed to the console /
       TUI log buffer (offload auto-falls back to streaming on a TTY);
     * batch / non-TTY -> nothing is echoed to the console; in offload mode the
       operator writes ``<task>.log`` itself and the driver's captured
       diagnostics are appended to that same file.

The TTY-dependent behavior is pinned deterministically (rather than relying on
how pytest is invoked) so the four cases are reproducible.
"""

import logging
import threading
from collections import deque

import pytest

import sflow.core.launcher as launcher_mod
import sflow.core.log_offload as log_offload
from sflow.app.sflow import SflowApp
from sflow.core.log_offload import OFFLOAD_TASK_LOGS_ENV
from sflow.logging import SFLOW_TASK_STREAM_ATTR
from sflow.ui.rich_tui import attach_tui_log_buffer, detach_tui_log_buffer

# printf concatenates its two args, so the joined marker only ever appears as
# real task stdout -- never inside the logged command text (which keeps the args
# separated by a space). This makes the "did task output reach X?" assertions
# precise.
_MARKER = "OFFLOADMATRIXSENTINEL"

_CONFIG = """\
version: "0.1"
backends:
  - name: local
    type: local
    default: true
workflow:
  name: offload_matrix
  tasks:
    - name: printer
      script:
        - printf '%s%s\\n' OFFLOADMATRIX SENTINEL
"""


class _CaptureHandler(logging.Handler):
    """Records everything that reaches the root ``sflow`` logger (the source the
    console handler / TUI buffer draw from)."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# (case_id, offload_env, tui, isatty)
_CASES = [
    ("offload_off__tui_off", "0", False, False),
    ("offload_on__tui_off", "1", False, False),
    ("offload_off__tui_on", "0", True, True),
    ("offload_on__tui_on", "1", True, True),
]


@pytest.mark.parametrize("case_id, offload_env, tui, isatty", _CASES)
def test_offload_tui_logging_matrix(
    tmp_path, monkeypatch, case_id, offload_env, tui, isatty
):
    cfg_path = tmp_path / "offload_matrix.yaml"
    cfg_path.write_text(_CONFIG)
    out_dir = tmp_path / "out"

    # The per-invocation offload decision via the env channel -- exactly what the
    # `--offload-task-logs / --no-offload-task-logs` CLI flag sets.
    monkeypatch.setenv(OFFLOAD_TASK_LOGS_ENV, offload_env)
    # Pin the TTY-dependent behavior the scenario represents:
    #   tui ON  -> interactive terminal (isatty True): offload auto-falls back to
    #              streaming and task output is echoed to the console / TUI buffer.
    #   tui OFF -> batch/non-interactive (isatty False): offload (when on) engages
    #              and writes <task>.log itself; nothing is echoed to the console.
    monkeypatch.setattr(log_offload, "stdout_is_tty", lambda: isatty)
    monkeypatch.setattr(launcher_mod, "_console_streams_task_output", lambda: isatty)
    # The streaming capture below relies on INFO records flowing through the
    # `sflow` logger chain (the CLI would set this via configure_logging).
    monkeypatch.setattr(
        logging.getLogger("sflow"), "level", logging.INFO, raising=False
    )

    cap = _CaptureHandler()
    root_logger = logging.getLogger("sflow")
    root_logger.addHandler(cap)

    # Mirror the CLI's TUI wiring: attach the shared log buffer ourselves and pass
    # it in (so RichTui does not double-attach a handler).
    buf: deque[logging.LogRecord] = deque(maxlen=4000)
    lock = threading.Lock()
    tui_handler = attach_tui_log_buffer(buf, log_lock=lock) if tui else None

    try:
        workflow_out_dir = SflowApp().run(
            file=cfg_path,
            dry_run=False,
            workspace_dir=tmp_path,
            output_dir=out_dir,
            tui=tui,
            tui_log_buffer=buf if tui else None,
            tui_log_lock=lock if tui else None,
        )
    finally:
        root_logger.removeHandler(cap)
        if tui_handler is not None:
            detach_tui_log_buffer(tui_handler)

    # (1) The run completed and produced a single run dir.
    assert workflow_out_dir is not None and workflow_out_dir.is_dir(), case_id

    # (2) The per-task <task>.log is ALWAYS written with the task's stdout.
    task_log = workflow_out_dir / "printer" / "printer.log"
    assert task_log.is_file(), f"{case_id}: missing per-task log {task_log}"
    assert _MARKER in task_log.read_text(), f"{case_id}: marker absent from {task_log}"

    # (3) No scattered sidecar: in offload mode the driver-side diagnostics are
    # merged into <task>.log itself, so a <task>.orchestration.log must NEVER be
    # left behind in any mode.
    sidecar = workflow_out_dir / "printer" / "printer.orchestration.log"
    assert not sidecar.exists(), f"{case_id}: unexpected scattered sidecar {sidecar}"

    # (4) Console-streaming correctness for the scenario.
    streamed = [r for r in cap.records if getattr(r, SFLOW_TASK_STREAM_ATTR, False)]
    streamed_has_marker = any(_MARKER in r.getMessage() for r in streamed)
    buffered_has_marker = any(_MARKER in r.getMessage() for r in buf)

    if isatty:
        # Stream mode (incl. offload's TTY fallback): task output reaches the
        # console / TUI buffer.
        assert streamed_has_marker, f"{case_id}: task output was not streamed"
        assert buffered_has_marker, f"{case_id}: task output missing from TUI buffer"
    else:
        # Batch / non-TTY: task output must NOT be echoed to the console.
        assert not streamed_has_marker, f"{case_id}: task output leaked to console"
