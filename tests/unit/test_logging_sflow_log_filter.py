# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the sflow.log file handler drops per-task subprocess output.

``add_log_file`` attaches a filter so that records tagged with
``SFLOW_TASK_STREAM_ATTR`` (per-task subprocess output) never land in sflow.log,
which is reserved for orchestration lines and command/status hints.
"""

import logging

from sflow.logging import SFLOW_TASK_STREAM_ATTR, add_log_file


def test_add_log_file_drops_task_stream_records(tmp_path):
    log_file = tmp_path / "sflow.log"
    logger = logging.getLogger("sflow")
    saved = (list(logger.handlers), logger.level, logger.propagate)

    try:
        add_log_file(str(log_file))
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.info("ORCHESTRATION_MARKER from orchestrator")
        logger.info(
            "TASKSTREAM_MARKER from a noisy task",
            extra={SFLOW_TASK_STREAM_ATTR: True},
        )

        for h in logger.handlers:
            h.flush()
        content = log_file.read_text()
    finally:
        for h in list(logger.handlers):
            if (
                isinstance(h, logging.FileHandler)
                and getattr(h, "baseFilename", None) == str(log_file)
            ):
                logger.removeHandler(h)
                h.close()
        logger.handlers, logger.level, logger.propagate = saved

    # Orchestration lines are kept; per-task stream lines are dropped.
    assert "ORCHESTRATION_MARKER" in content
    assert "TASKSTREAM_MARKER" not in content
