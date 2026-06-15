# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import pty
import shlex
import subprocess
import sys
from typing import Any, Callable, Mapping, Optional

from sflow.logging import IN_TASK_OUTPUT_ATTR, get_logger

from .command import Command, format_command

_logger = get_logger(__name__)

OutputLineCallback = Callable[[str | None, str], None]


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class SubprocessLauncher:
    """A command launcher for launching commands."""

    def __init__(self, *, echo_to_console: bool | None = None) -> None:
        """
        Args:
            echo_to_console: Whether to echo in-task (subprocess) output lines to
                the interactive console. Defaults to autodetect: echo only when
                stdout is a real TTY, so live output is shown for ``sflow run`` in a
                terminal but never written to disk under ``sflow batch`` / Slurm
                (where stdout is redirected to a file) or when stdout is piped.
        """
        if echo_to_console is None:
            try:
                echo_to_console = sys.stdout.isatty()
            except Exception:
                echo_to_console = False
        self._echo_to_console = bool(echo_to_console)

    def _console_prefix(self, task_name: str | None) -> str:
        """
        Prefix applied to subprocess lines echoed to the interactive console so users
        can tell which workflow task produced them. The full, unprefixed output is
        always written to the per-task log via ``output_logger``.
        """
        return f"[{task_name}] " if task_name else ""

    def _emit_subprocess_line(
        self,
        line_str: str,
        *,
        prefix: str,
        output_logger: Optional[logging.Logger],
        on_output_line: OutputLineCallback | None = None,
        task_name: str | None = None,
        task_output_sink: Any | None = None,
    ) -> None:
        """
        Route one line of subprocess (in-task) output.

        The full output is always written to the per-task log via ``output_logger``
        (configured non-propagating, so it stays in ``<task>.log``) -- this is the
        single on-disk home for in-task content.

        INVARIANT: this method is the only writer of the per-task log, and it always
        notifies ``on_output_line`` for the same line it persists. So the live stream
        seen by probes / output parsing is a strict superset of ``<task>.log``. This
        is what lets ``LogWatchProbe`` stop re-reading the file once it has seen any
        live line (see ``LogWatchProbe.check``). Do not write to a task's logger from
        anywhere else, or those lines would be invisible to live probes.

        It is additionally echoed to the interactive console only when stdout is a
        real TTY (never under ``sflow batch`` / redirected stdout) and the per-task
        logger does not itself propagate to the shared logger (which would already
        surface the line -- as on the Slurm allocation path). The echo carries the
        ``IN_TASK_OUTPUT_ATTR`` marker so the ``sflow.log`` file handler filters it
        out: in-task output is never persisted to ``sflow.log`` at any log level.
        """
        if not line_str:
            return
        if on_output_line is not None:
            try:
                on_output_line(task_name, line_str)
            except Exception as e:
                _logger.warning(f"{prefix}Output line callback failed: {e}")
        if task_output_sink is not None:
            task_output_sink.emit_line(line_str)
        elif output_logger is not None:
            output_logger.info(line_str)
        if self._echo_to_console and (
            output_logger is None or not output_logger.propagate
        ):
            _logger.info(f"{prefix}{line_str}", extra={IN_TASK_OUTPUT_ATTR: True})

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
        on_output_line: OutputLineCallback | None = None,
        task_output_sink: Any | None = None,
    ) -> int:
        pfx = self._console_prefix(task_name)
        _logger.info(f"{pfx}========== Command ==========")
        _logger.info(f"{pfx}{format_command(command)}")
        _logger.info(f"{pfx}=============================")
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

        try:
            buffer = b""
            while True:
                # Check if process has exited
                ret = process.poll()

                try:
                    # Non-blocking read from PTY master
                    chunk = await loop.run_in_executor(
                        None, lambda: self._read_pty_chunk(master_fd)
                    )
                except OSError:
                    # PTY closed (process exited)
                    chunk = b""

                if chunk:
                    buffer += chunk

                    # Split on both \n and \r to handle progress bars
                    text = buffer.decode("utf-8", errors="replace")
                    text = text.replace("\r\n", "\n").replace("\r", "\n")

                    # Split into lines, keeping incomplete line in buffer
                    lines = text.split("\n")
                    buffer = lines[-1].encode("utf-8")

                    for line_str in lines[:-1]:
                        # Strip ANSI escape sequences for cleaner logs
                        line_str = _strip_ansi(line_str).rstrip()
                        self._emit_subprocess_line(
                            line_str,
                            prefix=pfx,
                            output_logger=output_logger,
                            on_output_line=on_output_line,
                            task_name=task_name,
                            task_output_sink=task_output_sink,
                        )

                if ret is not None and not chunk:
                    # Process exited and no more data
                    if buffer:
                        line_str = _strip_ansi(
                            buffer.decode("utf-8", errors="replace")
                        ).rstrip()
                        self._emit_subprocess_line(
                            line_str,
                            prefix=pfx,
                            output_logger=output_logger,
                            on_output_line=on_output_line,
                            task_name=task_name,
                            task_output_sink=task_output_sink,
                        )
                    break

                if not chunk:
                    # No data available, yield control briefly
                    await asyncio.sleep(0.05)

            return process.returncode
        except asyncio.CancelledError:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass

    def _read_pty_chunk(self, fd: int, size: int = 4096) -> bytes:
        """Read a chunk from PTY file descriptor, returns empty bytes if no data."""
        try:
            return os.read(fd, size)
        except BlockingIOError:
            return b""
        except OSError as e:
            if e.errno == 5:  # EIO - PTY closed
                return b""
            raise

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
