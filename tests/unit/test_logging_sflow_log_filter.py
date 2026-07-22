# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that the sflow.log file handler drops per-task subprocess output.

Both ``add_log_file`` and ``configure_logging`` attach a filter so that records
tagged with ``SFLOW_TASK_STREAM_ATTR`` (per-task subprocess output) never land in
sflow.log, which is reserved for orchestration lines and command/status hints.
"""

import logging

from sflow.logging import (
    SFLOW_TASK_STREAM_ATTR,
    _PY_WARNINGS_LOGGER,
    add_log_file,
    configure_logging,
)


def _read_then_detach(logger, log_file, saved):
    """Flush + read the log file, then detach/close its handler and restore state."""
    for h in logger.handlers:
        h.flush()
    content = log_file.read_text()
    for h in list(logger.handlers):
        if (
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_file)
        ):
            logger.removeHandler(h)
            h.close()
    logger.handlers, logger.level, logger.propagate = saved
    return content


def test_configure_logging_file_handler_drops_task_stream_records(tmp_path):
    log_file = tmp_path / "sflow.log"
    logger = logging.getLogger("sflow")
    saved = (list(logger.handlers), logger.level, logger.propagate)

    try:
        # console=False isolates the assertion to the file handler.
        configure_logging(level="INFO", log_file=str(log_file), console=False)

        logger.info("ORCHESTRATION_MARKER from orchestrator")
        logger.info(
            "TASKSTREAM_MARKER from a noisy task",
            extra={SFLOW_TASK_STREAM_ATTR: True},
        )

        content = _read_then_detach(logger, log_file, saved)
    finally:
        logger.handlers, logger.level, logger.propagate = saved

    # Orchestration lines are kept; per-task stream lines are dropped.
    assert "ORCHESTRATION_MARKER" in content
    assert "TASKSTREAM_MARKER" not in content


def test_python_warnings_logger_is_routed_to_sflow_log(tmp_path):
    """Captured Python warnings are persisted to sflow.log, not only to stderr.

    ``logging.captureWarnings`` sends ``warnings.warn`` to the ``py.warnings``
    logger, so configure_logging + add_log_file must (a) enable capture and (b)
    wire that logger to sflow.log. (A true end-to-end ``warnings.warn`` cannot be
    exercised here because pytest wraps each test in its own ``catch_warnings``
    recorder; we instead assert the wiring and emit through the target logger.)
    """
    log_file = tmp_path / "sflow.log"
    sflow_logger = logging.getLogger("sflow")
    warnings_logger = logging.getLogger(_PY_WARNINGS_LOGGER)
    saved_sflow = (list(sflow_logger.handlers), sflow_logger.level, sflow_logger.propagate)
    saved_warn = (
        list(warnings_logger.handlers),
        warnings_logger.level,
        warnings_logger.propagate,
    )
    capture_was_on = logging._warnings_showwarning is not None

    try:
        # console=False isolates the assertion to the file handler; add_log_file
        # then mirrors that handler onto the py.warnings logger.
        configure_logging(level="INFO", console=False)
        add_log_file(str(log_file))

        # (a) capture is enabled and (b) the warnings logger points at sflow.log.
        assert logging._warnings_showwarning is not None
        assert any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_file)
            for h in warnings_logger.handlers
        )
        assert warnings_logger.propagate is False

        # A captured warning is emitted as py.warnings.warning(...); confirm it lands.
        warnings_logger.warning("PYWARN_MARKER deprecated behavior")
        for h in warnings_logger.handlers:
            h.flush()
        content = log_file.read_text()
    finally:
        for lg, saved in (
            (sflow_logger, saved_sflow),
            (warnings_logger, saved_warn),
        ):
            for h in list(lg.handlers):
                if (
                    isinstance(h, logging.FileHandler)
                    and getattr(h, "baseFilename", None) == str(log_file)
                ):
                    lg.removeHandler(h)
                    h.close()
            lg.handlers, lg.level, lg.propagate = saved
        if not capture_was_on:
            logging.captureWarnings(False)

    assert "PYWARN_MARKER" in content
    assert _PY_WARNINGS_LOGGER in content
    assert "WARNING" in content


def test_add_log_file_supersedes_previous_sflow_log(tmp_path):
    """A second ``add_log_file`` replaces the first instead of accumulating.

    A long-lived process runs several workflows in-process (``sflow batch`` bulk/rows,
    ``sflow compose``); each calls ``add_log_file`` with a new per-run sflow.log. The
    handlers must not pile up -- on the sflow logger *or* the mirrored py.warnings
    logger -- or each run's records get duplicated into earlier runs' logs and, once
    an earlier output dir is gone, ``FileHandler`` reopen raises on the next record.
    """
    log_a = tmp_path / "a" / "sflow.log"
    log_b = tmp_path / "b" / "sflow.log"
    log_a.parent.mkdir()
    log_b.parent.mkdir()

    sflow_logger = logging.getLogger("sflow")
    warnings_logger = logging.getLogger(_PY_WARNINGS_LOGGER)
    saved_sflow = (list(sflow_logger.handlers), sflow_logger.level, sflow_logger.propagate)
    saved_warn = (
        list(warnings_logger.handlers),
        warnings_logger.level,
        warnings_logger.propagate,
    )
    capture_was_on = logging._warnings_showwarning is not None

    def _file_targets(lg):
        return [
            getattr(h, "baseFilename", None)
            for h in lg.handlers
            if isinstance(h, logging.FileHandler)
        ]

    try:
        configure_logging(level="INFO", console=False)
        add_log_file(str(log_a))
        add_log_file(str(log_b))

        # Only the most recent sflow.log is attached -- no accumulation -- on both
        # the sflow logger and the mirrored py.warnings logger.
        assert _file_targets(sflow_logger) == [str(log_b)]
        assert _file_targets(warnings_logger) == [str(log_b)]

        # A record after the switch lands only in the new log, not the superseded one.
        sflow_logger.info("ONLY_IN_B_MARKER")
        for h in sflow_logger.handlers:
            h.flush()
        assert "ONLY_IN_B_MARKER" in log_b.read_text()
        assert "ONLY_IN_B_MARKER" not in log_a.read_text()
    finally:
        for lg, saved in (
            (sflow_logger, saved_sflow),
            (warnings_logger, saved_warn),
        ):
            for h in list(lg.handlers):
                if isinstance(h, logging.FileHandler):
                    lg.removeHandler(h)
                    h.close()
            lg.handlers, lg.level, lg.propagate = saved
        if not capture_was_on:
            logging.captureWarnings(False)


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
