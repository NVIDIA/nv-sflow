# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest


@pytest.fixture
def mock_config():
    return {}


@pytest.fixture
def image_warnings():
    """Collect the warnings emitted for unrecognised container image references.

    An unrecognised ``--container-image`` / ``container_image`` value warns instead of
    aborting the run, so the assertions that used to be ``pytest.raises`` live here.

    The handler is attached directly to the emitting logger rather than relying on
    propagation to root (which is what ``caplog`` observes): ``configure_logging`` sets
    ``propagate = False`` on the ``sflow`` logger, so root-based capture would silently
    depend on whether an earlier test in the session had configured logging.
    """
    logger = logging.getLogger("sflow.utils.container")
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Capture()
    saved_level, saved_disabled = logger.level, logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.disabled = False
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved_level)
        logger.disabled = saved_disabled
