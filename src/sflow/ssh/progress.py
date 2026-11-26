# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSH orchestration diagnostics, using sflow's Rich log style on stderr."""

import logging

from sflow.logging import create_console_handler, get_logger

_logger = get_logger(__name__)


def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    _logger.handlers = [create_console_handler(stderr=True)]
    _logger.setLevel(numeric_level)
    # Remote stdout may be YAML; never inherit sflow's ordinary stdout handler.
    _logger.propagate = False


def log(level: int, message: str, *args: object) -> None:
    if not _logger.handlers:
        configure_logging()
    _logger.log(level, message, *args, stacklevel=2)
