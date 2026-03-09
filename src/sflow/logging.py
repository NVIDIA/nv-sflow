# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# Default width for non-interactive terminals (piping, file output, etc.)
_DEFAULT_NON_TTY_WIDTH = 200


def configure_logging(
    level: str = "INFO", log_file: Optional[str] = None, *, console: bool = True
):
    """
    Configures the global logger for sflow.

    Args:
        level (str): The logging level (DEBUG, INFO, WARNING, ERROR).
        log_file (Optional[str]): Path to a file to write logs to.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Create root logger configuration
    handlers = []

    # Console handler (Rich)
    if console:
        # Use a wider default width when output is not a TTY (e.g. piped to file).
        if sys.stdout.isatty():
            rich_console = Console()
        else:
            rich_console = Console(width=_DEFAULT_NON_TTY_WIDTH, force_terminal=False)
        console_handler = RichHandler(console=rich_console, rich_tracebacks=True)
        # RichHandler handles formatting internally
        handlers.append(console_handler)

    # File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Configure the sflow logger
    logger = logging.getLogger("sflow")
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    for handler in handlers:
        logger.addHandler(handler)

    # Ensure propagation is handled correctly (usually True, but we are setting handlers)
    logger.propagate = False


def add_log_file(log_file: str) -> None:
    """
    Add a file handler to the `sflow` logger without resetting existing handlers.
    Useful once output directories are known (after config load).
    """
    logger = logging.getLogger("sflow")
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(
            h, "baseFilename", None
        ) == str(log_file):
            return

    fh = logging.FileHandler(log_file)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
