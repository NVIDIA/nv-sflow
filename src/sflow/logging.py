# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import time
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# Default width for non-interactive terminals (piping, file output, etc.)
_DEFAULT_NON_TTY_WIDTH = 200

# Log records carrying this attribute (set to True) are per-task subprocess
# output. They may be streamed to an interactive console, but must never be
# written to sflow.log (or the Slurm stdout/err files). Those sinks are reserved
# for orchestration lines and command/status hints; full per-task content lives
# in <task>/<task>.log.
SFLOW_TASK_STREAM_ATTR = "sflow_task_stream"

# ``logging.captureWarnings`` redirects Python ``warnings.warn`` output to this
# logger. It sits outside the ``sflow`` namespace, so sflow's console + file
# handlers must be mirrored onto it or warnings would print only to stderr and
# never reach sflow.log.
_PY_WARNINGS_LOGGER = "py.warnings"


def _mirror_sflow_handlers_to_py_warnings(*, min_level: int) -> None:
    """Point the ``py.warnings`` logger at exactly the ``sflow`` logger's handlers.

    ``logging.captureWarnings`` sends ``warnings.warn`` / ``DeprecationWarning``
    (sflow's own and third-party) to the process-global ``py.warnings`` logger, which
    lives outside the ``sflow`` namespace. Mirroring sflow's handlers onto it persists
    those warnings to the same console + sflow.log sinks instead of only stderr.

    The handler list is *replaced* (never appended to) so this process-global logger
    tracks sflow's current sinks exactly and never accumulates a stale ``FileHandler``
    from a previous run. A leftover handler would try to reopen a since-removed output
    dir and raise ``FileNotFoundError`` on the next warning -- the reopen in
    ``FileHandler.emit`` is outside its own ``try``/``except`` -- and would also write
    one run's warnings into a prior run's sflow.log. ``propagate`` is disabled so the
    root last-resort stderr handler does not emit a duplicate line.

    ``min_level`` only lowers the level (never raises it) so warnings keep flowing even
    when the console ``--log-level`` is higher; warning records are ``>= WARNING``
    regardless.
    """
    logging.captureWarnings(True)
    sflow_logger = logging.getLogger("sflow")
    warnings_logger = logging.getLogger(_PY_WARNINGS_LOGGER)
    warnings_logger.handlers = list(sflow_logger.handlers)
    if warnings_logger.level == logging.NOTSET or warnings_logger.level > min_level:
        warnings_logger.setLevel(min_level)
    warnings_logger.propagate = False


class _DropTaskStreamFilter(logging.Filter):
    """Drop per-task subprocess output records so they never reach a file handler.

    Records produced by streaming a task's stdout/stderr are tagged with
    ``SFLOW_TASK_STREAM_ATTR``. This filter keeps them out of sflow.log while
    still allowing them through to the interactive console handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, SFLOW_TASK_STREAM_ATTR, False)


class CoalescingFileHandler(logging.FileHandler):
    """A ``FileHandler`` that coalesces flushes to cut per-line syscalls.

    The stock ``FileHandler`` flushes after every record, so a chatty task
    becomes one ``write()``+``flush()`` per line. This handler writes every
    record immediately but flushes at most once per ``flush_interval`` seconds
    (and always on ``flush``/``close``). The interval is short enough that
    ``LogWatchProbe`` (which polls the file) still observes lines promptly.
    """

    def __init__(self, *args, flush_interval: float = 0.2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flush_interval = max(float(flush_interval), 0.0)
        self._last_flush = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        # Mirror logging.StreamHandler.emit but defer the flush to an interval.
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            now = time.monotonic()
            if now - self._last_flush >= self._flush_interval:
                self.flush()
                self._last_flush = now
        except RecursionError:  # pragma: no cover - matches stdlib behavior
            raise
        except Exception:
            self.handleError(record)


class DeferredTaskLogHandler(logging.Handler):
    """Buffer task-logger records and append them to ``<task>.log`` only after
    the task's own writer has released the file.

    In offload mode the operator writes ``<task>.log`` itself (srun ``--output``
    or a host-side shell redirect), so sflow must not write the same file
    concurrently (single-writer invariant). Any driver-side diagnostics the
    launcher captures for the task (e.g. ``srun: error: ... Exited with exit
    code 1``) are buffered here and *appended* to ``<task>.log`` on
    :meth:`flush`, which the launcher calls in its ``finally`` block once the
    subprocess has exited -- so they land in the per-task log itself instead of
    a scattered ``<task>.orchestration.log`` sidecar. If the task produced no
    driver-side diagnostics, nothing is written and the file is left untouched.
    """

    def __init__(self, target: str) -> None:
        super().__init__()
        # Mirror logging.FileHandler's attribute so dedup-by-path checks (and any
        # tooling that inspects handler targets) recognize this handler too.
        self.baseFilename = str(target)
        self._buffer: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.append(self.format(record) + "\n")
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        # Append (never truncate) so the operator-written content is preserved.
        # Called post-exit (single writer), so this does not race the operator.
        self.acquire()
        try:
            if not self._buffer:
                return
            try:
                with open(self.baseFilename, "a", encoding="utf-8") as fh:
                    fh.write("".join(self._buffer))
                self._buffer.clear()
            except Exception:
                # Diagnostics are best-effort; never break the run over them.
                pass
        finally:
            self.release()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            super().close()


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
        console_handler.setLevel(numeric_level)
        handlers.append(console_handler)

    # File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        # Per-task subprocess output must never land in the log file.
        file_handler.addFilter(_DropTaskStreamFilter())
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

    # Capture Python warnings (warnings.warn) into the same sinks. Clamp to WARNING
    # so warnings are persisted even when the console --log-level is higher.
    _mirror_sflow_handlers_to_py_warnings(min_level=min(numeric_level, logging.WARNING))


def enable_console_logging(level: Optional[str] = None) -> None:
    """Attach a Rich console handler to the ``sflow`` logger if none is present.

    Idempotent. Used to *resume* console logging after an interactive TUI tears
    down -- the TUI runs with the console handler disabled (``console=False``), so
    deferred work such as hardware-monitor post-processing would otherwise emit
    nothing to the terminal. Existing handlers (e.g. the sflow.log file handler or
    the TUI buffer handler) are left untouched.
    """
    logger = logging.getLogger("sflow")
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            return  # console logging already active
    numeric_level = (
        getattr(logging, level.upper(), logging.INFO)
        if level
        else (logger.level or logging.INFO)
    )
    if sys.stdout.isatty():
        rich_console = Console()
    else:
        rich_console = Console(width=_DEFAULT_NON_TTY_WIDTH, force_terminal=False)
    console_handler = RichHandler(console=rich_console, rich_tracebacks=True)
    console_handler.setLevel(numeric_level if isinstance(numeric_level, int) else logging.INFO)
    logger.addHandler(console_handler)


def add_log_file(log_file: str) -> None:
    """
    Point the ``sflow`` logger at ``log_file`` (its console handler is left intact).
    Useful once output directories are known (after config load).

    A run has exactly one sflow.log. If this exact path is already attached this is a
    no-op; otherwise any previously-attached sflow.log file handler is detached and
    closed first. That matters for a long-lived process that runs several workflows
    in-process (e.g. ``sflow batch`` bulk/rows, ``sflow compose``): without it the
    file handlers accumulate, so each run's lines get duplicated into every earlier
    run's sflow.log and -- once an earlier output dir is gone -- ``FileHandler``'s
    reopen raises on the next record. Runs are sequential, so the prior run's handler
    is safe to close here.

    The file handler always logs at INFO level so the sflow.log captures
    the full orchestration timeline regardless of the console --log-level.
    """
    logger = logging.getLogger("sflow")
    stale_file_handlers: list[logging.FileHandler] = []
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            if getattr(h, "baseFilename", None) == str(log_file):
                return
            stale_file_handlers.append(h)
    for h in stale_file_handlers:
        logger.removeHandler(h)
        h.close()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    # Keep per-task subprocess output out of sflow.log (orchestration only).
    fh.addFilter(_DropTaskStreamFilter())
    logger.addHandler(fh)

    # Ensure the logger itself accepts INFO messages even if the console
    # handler was configured at a higher level (e.g. WARNING).
    if logger.level > logging.INFO:
        logger.setLevel(logging.INFO)

    # Mirror the sflow logger's sinks (now including this file handler) onto the
    # Python-warnings logger so captured warnings (warnings.warn / DeprecationWarning)
    # are persisted to sflow.log too, without letting it accumulate stale handlers.
    _mirror_sflow_handlers_to_py_warnings(min_level=logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
