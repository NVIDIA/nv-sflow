# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import logging
import os
import re
import shlex
import subprocess
import sys
from typing import Mapping, Optional

try:
    import pty
except ImportError:  # pragma: no cover - exercised by Windows import tests.
    pty = None

from sflow.logging import SFLOW_TASK_STREAM_ATTR, get_logger

from .command import Command, format_command
from .command_log import record_active_command

_logger = get_logger(__name__)


def _console_streams_task_output() -> bool:
    """Whether per-task subprocess output should be echoed to the root logger.

    True only for interactive TTY sessions. In batch mode (Slurm redirects
    stdout/stderr to files) or when output is piped to a file, this returns
    False so task content never floods sflow.log or the Slurm stdout/err files.
    Per-task content is always written to the per-task log regardless.
    """
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _emit_task_output_line(
    line_str: str,
    *,
    pfx: str,
    output_logger: Optional[logging.Logger],
    stream_console: bool,
    to_file: bool = True,
) -> None:
    """Route a single line of task (subprocess) output.

    Writes to the per-task logger (``output_logger`` -> ``<task>.log``) unless
    ``to_file`` is False. Only additionally echoes to the root ``sflow`` logger
    (console / Slurm stdout / TUI) when ``stream_console`` is True, tagging the
    record with ``SFLOW_TASK_STREAM_ATTR`` so sflow.log's file handler drops it.

    Transient progress-bar snapshots pass ``to_file=False`` so they animate live
    on the console/TUI without polluting the persistent per-task log, which keeps
    only completed (final-state) lines.
    """
    if output_logger and to_file:
        output_logger.info(line_str)
    if stream_console:
        _logger.info(f"{pfx}{line_str}", extra={SFLOW_TASK_STREAM_ATTR: True})


# How often an in-progress (carriage-return) line is surfaced to the console/TUI
# while it keeps redrawing without a newline (e.g. docker pull / pip / aiperf).
_PARTIAL_FLUSH_INTERVAL = 0.4

# Max 64KB PTY reads drained per event-loop wakeup before yielding back to the
# loop. Draining the whole buffer in one wakeup lets a single chatty stream (e.g.
# a high-volume server log delivering a backlog burst) monopolize the loop and
# starve every other coroutine -- including the K8s pod-status watches and the
# orchestrator's DAG poll, which is what delays detecting that a task (and the
# workflow) has finished. Capping reads per wakeup keeps those responsive; the
# reader stays registered, so any remaining data is drained on the next tick.
_MAX_PTY_READS_PER_LOOP = 8


# Compiled once at import time: recompiling per line was a measurable cost when
# task output is chatty (this runs for every line of every task's stdout/stderr).
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


class SubprocessLauncher:
    """A command launcher for launching commands."""

    def _console_prefix(self, task_name: str | None) -> str:
        """
        Prefix applied to lines printed to the terminal (console logger) so users can
        tell which workflow task produced them.
        """
        return f"[{task_name}] " if task_name else ""

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        """
        Best-effort terminate a running subprocess.
        """
        try:
            if process.returncode is not None:
                return
            process.terminate()
        except ProcessLookupError:
            return
        except Exception:
            # Fall back to kill below.
            pass

        try:
            await asyncio.wait_for(process.wait(), timeout=5)
            return
        except Exception:
            pass

        try:
            if process.returncode is None:
                process.kill()
        except ProcessLookupError:
            return
        except Exception:
            return

        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except Exception:
            return

    async def run_async(
        self,
        command: Command | str | list[str],
        shell: bool = False,
        output_logger: Optional[logging.Logger] = None,
        env: Mapping[str, str] | None = None,
        task_name: str | None = None,
    ) -> int:
        pfx = self._console_prefix(task_name)
        # Orchestration hint: goes to sflow.log + console (and Slurm stdout/err).
        # Intentionally NOT written to the per-task log: LogWatchProbe scans the
        # per-task log file and would false-match patterns inside the command text.
        _logger.info(f"{pfx}========== Command ==========")
        _logger.info(f"{pfx}{format_command(command)}")
        _logger.info(f"{pfx}=============================")

        # Decide once whether to echo per-task output to the root logger/console.
        # Per-task content always goes to the per-task log (output_logger); it is
        # only additionally streamed to the console for interactive TTY sessions.
        stream_console = _console_streams_task_output()

        if isinstance(command, Command):
            command = command.as_list()

        # Prepare command arguments
        if shell:
            if isinstance(command, list):
                cmd_str = shlex.join(command)
            else:
                cmd_str = command
            args = ["/bin/sh", "-c", cmd_str]
        else:
            if isinstance(command, str):
                args = shlex.split(command)
            else:
                args = list(command)

        # Prepare environment
        proc_env = os.environ.copy()
        if env is not None:
            proc_env.update({str(k): str(v) for k, v in env.items()})

        # Use PTY to make subprocess think it's connected to a terminal
        # This prevents output buffering issues with progress bars
        if pty is None:
            raise OSError("PTY subprocess launching is not available on this platform")
        master_fd, slave_fd = pty.openpty()

        try:
            process = subprocess.Popen(
                args,
                stdout=slave_fd,
                stderr=slave_fd,
                stdin=subprocess.DEVNULL,
                env=proc_env,
                close_fds=True,
            )
            record_active_command(command, task_name=task_name, shell=shell)
        except Exception as e:
            os.close(master_fd)
            os.close(slave_fd)
            _logger.error(f"Failed to start command {command}: {e}")
            raise

        # Close slave in parent - child has its own copy
        os.close(slave_fd)

        # Set master to non-blocking for async reading
        import fcntl

        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        loop = asyncio.get_event_loop()
        # The PTY master becomes readable (and yields EOF / EIO) once the child
        # closes the slave on exit. We drive reads off event-loop readiness via
        # ``add_reader`` instead of hopping every chunk to a thread-pool and
        # polling on a 50ms sleep, which serialized/contended under many chatty
        # tasks. ``done`` is set when the PTY reaches EOF.
        done: asyncio.Event = asyncio.Event()
        buffer = b""
        # The most recent in-progress (carriage-return) frame surfaced to the
        # console/TUI, and when, so live progress bars keep updating on a throttle
        # without emitting every redraw frame (and without writing to <task>.log).
        partial_text: str | None = None
        partial_at = 0.0

        def _feed(data: bytes) -> None:
            # Append new bytes and emit any newly-completed lines, keeping the
            # incomplete trailing line in ``buffer`` for the next read.
            nonlocal buffer, partial_text, partial_at
            buffer += data
            # Only split on real newlines; a carriage return rewrites the CURRENT
            # line in place (progress bars), it does not start a new one.
            text = buffer.decode("utf-8", errors="replace").replace("\r\n", "\n")
            lines = text.split("\n")
            # Keep the trailing incomplete physical line for the next read. Collapse
            # its in-place \r redraws now so a pure progress bar (which only emits
            # \r until it finishes) can't grow the buffer without bound.
            buffer = lines[-1].rsplit("\r", 1)[-1].encode("utf-8")
            for line_str in lines[:-1]:
                # Terminal semantics: \r returns to the start of the line and the
                # text after it overwrites what came before. Keep only the final
                # state so a completed progress line renders once, not per redraw.
                line_str = _strip_ansi(line_str.rsplit("\r", 1)[-1]).rstrip()
                if not line_str:
                    continue
                if line_str == partial_text:
                    # Identical to the snapshot already streamed to the console:
                    # still record the final state in <task>.log (snapshots are not
                    # written there), but don't duplicate it on the console.
                    if output_logger is not None:
                        output_logger.info(line_str)
                    partial_text = None
                    continue
                _emit_task_output_line(
                    line_str,
                    pfx=pfx,
                    output_logger=output_logger,
                    stream_console=stream_console,
                )
                # A completed line supersedes any tracked in-progress snapshot.
                partial_text = None

            # Surface the still-incomplete line periodically so progress bars that
            # only redraw with \r (no newline) keep animating on the console/TUI.
            # These transient snapshots are NOT written to <task>.log (to_file=False);
            # the file gets the final line when the real newline arrives. They exist
            # purely for live display, so skip the decode/strip work entirely when
            # nothing consumes them (batch / non-TTY / offload, where stream_console
            # is False and the emit would be a no-op anyway).
            if stream_console:
                snapshot = _strip_ansi(
                    buffer.decode("utf-8", errors="replace")
                ).rstrip()
                now = loop.time()
                if (
                    snapshot
                    and snapshot != partial_text
                    and (now - partial_at) >= _PARTIAL_FLUSH_INTERVAL
                ):
                    _emit_task_output_line(
                        snapshot,
                        pfx=pfx,
                        output_logger=output_logger,
                        stream_console=stream_console,
                        to_file=False,
                    )
                    partial_text = snapshot
                    partial_at = now

        def _flush_tail() -> None:
            nonlocal buffer, partial_text
            if buffer:
                line_str = _strip_ansi(
                    buffer.decode("utf-8", errors="replace").rsplit("\r", 1)[-1]
                ).rstrip()
                # Persist the final state to <task>.log (unless it just duplicates
                # the last streamed snapshot).
                if line_str and line_str != partial_text:
                    _emit_task_output_line(
                        line_str,
                        pfx=pfx,
                        output_logger=output_logger,
                        stream_console=stream_console,
                    )
                elif line_str and output_logger is not None:
                    # Already streamed to the console as a snapshot; just record the
                    # final state in the per-task log so the file is complete.
                    output_logger.info(line_str)
                buffer = b""
                partial_text = None

        def _on_readable() -> None:
            # Drain a BOUNDED amount, then return so the event loop can service
            # other coroutines (pod-status watches, the orchestrator, other tasks'
            # streams) before we read more. Draining the whole PTY buffer in one
            # wakeup lets a single chatty stream monopolize the loop and starve
            # everything else. A read of b"" (or EIO on the PTY master) means the
            # child closed the slave: stop watching and signal completion.
            try:
                for _ in range(_MAX_PTY_READS_PER_LOOP):
                    try:
                        chunk = os.read(master_fd, 65536)
                    except BlockingIOError:
                        # No more data right now; wait for the next readiness.
                        return
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            chunk = b""  # PTY slave closed (child exited).
                        else:
                            raise
                    if not chunk:
                        break
                    _feed(chunk)
                else:
                    # Hit the per-wakeup read cap with data likely still buffered.
                    # Yield to the loop; the reader stays registered, so this fires
                    # again next tick after other tasks have had a turn.
                    return
            except OSError:
                # Treat any unexpected read error as EOF so we never hang.
                pass
            try:
                loop.remove_reader(master_fd)
            except Exception:
                pass
            _flush_tail()
            done.set()

        try:
            loop.add_reader(master_fd, _on_readable)
            await done.wait()
            # PTY EOF means the child closed its stdio (i.e. it is exiting). Poll
            # briefly until it is reaped so we return the real exit code; using
            # poll() (not a blocking wait) keeps the event loop responsive.
            while process.poll() is None:
                await asyncio.sleep(0.01)
            return process.returncode
        except asyncio.CancelledError:
            try:
                loop.remove_reader(master_fd)
            except Exception:
                pass
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            # The per-task FileHandler coalesces flushes, so a just-finished
            # task may still have its tail buffered. Flush now (the subprocess
            # has exited) so post-completion readers - result/output parsing in
            # the orchestrator's finalize step - see the complete log.
            if output_logger is not None:
                for handler in list(getattr(output_logger, "handlers", []) or []):
                    try:
                        handler.flush()
                    except Exception:
                        pass
            try:
                loop.remove_reader(master_fd)
            except Exception:
                pass
            try:
                os.close(master_fd)
            except OSError:
                pass

    # async def run_pipe_async(
    #     self,
    #     commands: list[Command | str | list[str]],
    #     shell: bool = False,
    #     output_logger: Optional[logging.Logger] = None,
    #     env: Mapping[str, str] | None = None,
    #     task_name: str | None = None,
    # ) -> int:
    #     """Execute multiple commands connected by pipes asynchronously.

    #     Args:
    #         commands: List of commands to pipe together
    #         shell: Whether to use shell for individual commands
    #         output_logger: Optional logger to use for output logging

    #     Returns:
    #         int: Exit code of the final command
    #     """
    #     pfx = self._console_prefix(task_name)
    #     _logger.info(f"{pfx}========== Commands ==========")
    #     _logger.info(f"{pfx}{' | '.join([format_command(cmd) for cmd in commands])}")
    #     _logger.info(f"{pfx}=============================")

    #     if not commands:
    #         raise ValueError("At least one command required")

    #     processes = []
    #     pipes = []
    #     proc_env = None
    #     if env is not None:
    #         proc_env = os.environ.copy()
    #         proc_env.update({str(k): str(v) for k, v in env.items()})

    #     try:
    #         for i, cmd in enumerate(commands):
    #             if isinstance(cmd, Command):
    #                 cmd = cmd.as_list()

    #             stdin = None
    #             if i > 0:
    #                 stdin = pipes[-1][0]  # Read end of previous pipe

    #             if i < len(commands) - 1:
    #                 r, w = os.pipe()
    #                 pipes.append((r, w))
    #                 stdout = w
    #             else:
    #                 stdout = asyncio.subprocess.PIPE

    #             # Intermediate processes: inherit stderr (goes to console)
    #             # Last process: merge stderr to stdout (captured in output)
    #             stderr = asyncio.subprocess.STDOUT if i == len(commands) - 1 else None

    #             create_subprocess = (
    #                 asyncio.create_subprocess_shell
    #                 if shell
    #                 else asyncio.create_subprocess_exec
    #             )

    #             args = []
    #             if shell:
    #                 if isinstance(cmd, list):
    #                     args = [shlex.join(cmd)]
    #                 else:
    #                     args = [cmd]
    #             else:
    #                 if isinstance(cmd, str):
    #                     args = shlex.split(cmd)
    #                 else:
    #                     args = cmd

    #             proc = await create_subprocess(
    #                 *args,
    #                 stdin=stdin,
    #                 stdout=stdout,
    #                 stderr=stderr,
    #                 env=proc_env,
    #             )

    #             processes.append(proc)

    #             # Close parent's copy of the write pipe end immediately
    #             if i < len(commands) - 1:
    #                 os.close(stdout)

    #             # Close parent's copy of the read pipe end used as stdin
    #             if i > 0:
    #                 os.close(stdin)

    #         # Read output from last process
    #         last_proc = processes[-1]
    #         try:
    #             # Read output in chunks to handle progress bars and special characters
    #             # that use \r without \n (which would cause readline() to hang)
    #             buffer = b""
    #             while True:
    #                 try:
    #                     # Read available data in chunks (non-blocking when data is available)
    #                     chunk = await last_proc.stdout.read(4096)
    #                     if not chunk:
    #                         # Process any remaining data in buffer
    #                         if buffer:
    #                             line_str = _strip_ansi(buffer.decode("utf-8", errors="replace")).rstrip()
    #                             if line_str:
    #                                 _logger.info(f"{pfx}{line_str}")
    #                                 if output_logger:
    #                                     output_logger.info(line_str)
    #                         break

    #                     buffer += chunk

    #                     # Split on both \n and \r to handle progress bars
    #                     # Replace \r\n with \n first to avoid double processing
    #                     text = buffer.decode("utf-8", errors="replace")
    #                     text = text.replace("\r\n", "\n").replace("\r", "\n")

    #                     # Split into lines, keeping incomplete line in buffer
    #                     lines = text.split("\n")
    #                     buffer = lines[-1].encode("utf-8")  # Keep incomplete line

    #                     for line_str in lines[:-1]:
    #                         # Strip ANSI escape sequences for cleaner logs
    #                         line_str = _strip_ansi(line_str).rstrip()
    #                         if line_str:  # Skip empty lines from progress bar overwrites
    #                             _logger.info(f"{pfx}{line_str}")
    #                             if output_logger:
    #                                 output_logger.info(line_str)
    #                 except Exception as e:
    #                     _logger.warning(f"{pfx}Error reading output: {e}")
    #                     break

    #             # Wait for all processes
    #             exit_codes = await asyncio.gather(*[p.wait() for p in processes])
    #             last_exit_code = exit_codes[-1]

    #             return last_exit_code
    #         except asyncio.CancelledError:
    #             # Terminate the whole pipeline on cancellation.
    #             for p in processes:
    #                 try:
    #                     await self._terminate_process(p)
    #                 except Exception:
    #                     pass
    #             raise

    #     except Exception:
    #         # Cleanup pipes if error
    #         for r, w in pipes:
    #             try:
    #                 os.close(r)
    #             except OSError:
    #                 pass
    #             try:
    #                 os.close(w)
    #             except OSError:
    #                 pass
    #         raise
