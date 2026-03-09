# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def temporary_handler(
    logger: logging.Logger,
    handler: logging.Handler,
) -> Iterator[logging.Logger]:
    """Temporarily attach a handler to a logger.

    The handler is removed on exit (even if an exception is raised). If the handler
    is already attached to the logger, it will not be removed on exit.

    Args:
        logger: Logger to attach the handler to.
        handler: Handler to attach.

    Yields:
        The same logger (for convenience).
    """
    already_attached = handler in logger.handlers
    old_level = logger.level
    old_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not already_attached:
        logger.addHandler(handler)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        if not already_attached:
            logger.removeHandler(handler)
