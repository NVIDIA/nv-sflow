# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sflow.log must capture allocation-phase warnings, not only the live console.

Backends emit capability warnings (RDMA, ComputeDomain/NVLink, DRA, node
reservation) during ``build_state`` (allocation). The sflow.log file handler must
already be attached at that point, otherwise those warnings reach only the console
and are lost from the persistent log -- exactly the debug info needed after a run.
"""

import logging
from pathlib import Path

import sflow.app.sflow as sflow_app_mod
from sflow.config.schema import SflowConfig
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.logging import _PY_WARNINGS_LOGGER, get_logger


def test_build_state_warnings_are_persisted_to_sflow_log(tmp_path, monkeypatch):
    f = tmp_path / "sflow.yaml"
    f.write_text(
        'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n'
        "    - name: t1\n      script:\n        - echo hi\n"
    )

    state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
    observed = {"file_handler_attached": None}

    async def _fake_build_state(
        config: SflowConfig,
        *,
        allocate: bool = True,
        workspace_dir=None,
        output_dir=None,
        source_files=None,
        kubectl_config=None,
    ) -> SflowState:
        # Mimic a Kubernetes backend capability warning emitted during allocation.
        sflow_logger = logging.getLogger("sflow")
        observed["file_handler_attached"] = any(
            isinstance(h, logging.FileHandler)
            and str(getattr(h, "baseFilename", "")).endswith("sflow.log")
            for h in sflow_logger.handlers
        )
        get_logger("sflow.plugins.backends.kubernetes").warning(
            "RDMA_ALLOC_MARKER: RDMA HCAs detected but no usable provider matched"
        )
        return state

    logger = logging.getLogger("sflow")
    warnings_logger = logging.getLogger(_PY_WARNINGS_LOGGER)
    saved = (list(logger.handlers), logger.level, logger.propagate)
    saved_warn = (
        list(warnings_logger.handlers),
        warnings_logger.level,
        warnings_logger.propagate,
    )
    capture_was_on = logging._warnings_showwarning is not None
    monkeypatch.setattr(sflow_app_mod, "build_state", _fake_build_state)
    try:
        workflow_out_dir = sflow_app_mod.SflowApp().run(
            file=Path(f), dry_run=False, output_dir=tmp_path / "out"
        )
        assert workflow_out_dir is not None
        sflow_log = workflow_out_dir / "sflow.log"
        assert sflow_log.is_file()
        content = sflow_log.read_text()
    finally:
        # run() attaches the sflow.log file handler to BOTH the sflow logger and the
        # process-global py.warnings logger (add_log_file mirrors sflow's sinks onto
        # it). Detach/close it from both and restore each logger's saved state so this
        # test never leaks a closed handler onto py.warnings for later tests.
        for lg, saved_state in ((logger, saved), (warnings_logger, saved_warn)):
            for h in list(lg.handlers):
                if isinstance(h, logging.FileHandler):
                    lg.removeHandler(h)
                    h.close()
            lg.handlers, lg.level, lg.propagate = saved_state
        if not capture_was_on:
            logging.captureWarnings(False)

    # The file handler was already wired up when build_state ran (ordering fix)...
    assert observed["file_handler_attached"] is True
    # ...so the allocation-phase warning is persisted, not console-only.
    assert "RDMA_ALLOC_MARKER" in content
