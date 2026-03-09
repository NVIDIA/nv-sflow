# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
from typing import Any, Iterator

from sflow.logging import get_logger

_logger = get_logger(__name__)


class Command:
    """A command builder for constructing shell commands with options and arguments."""

    def __init__(
        self,
        exec: str,
        args: list[str] = None,
        opts: list[tuple[str, Any | None]] = None,
    ):
        """Initialize a Command.

        Args:
            exec: Executable path/name
            args: List of positional arguments
            opts: Dictionary of options
        """
        if not exec:
            raise ValueError("Executable is required")
        self._exec = str(exec)
        self._args = [str(arg) for arg in args] if args else []
        self._opts: list[tuple[str, str | None]] = opts or []

    def add_opt(self, name: str, value: Any = None, append: bool = False) -> Command:
        """Add an option to the command.

        Args:
            name: Option name (must include prefix like - or --)
            value: Option value (None for flag options)
            append: Whether to append the option (allow duplicates) or overwrite existing ones.

        Returns:
            Self for method chaining
        """
        val_str = str(value) if value is not None else None

        if not append:
            # Remove all existing options with this name to simulate overwrite
            self._opts = [opt for opt in self._opts if opt[0] != name]

        self._opts.append((name, val_str))
        return self

    def add_arg(self, arg: Any) -> Command:
        """Add an argument to the command.

        Args:
            arg: Argument to add

        Returns:
            Self for method chaining
        """
        self._args.append(str(arg))
        return self

    def as_list(self) -> list[str]:
        """Convert command to list format suitable for subprocess.

        Returns:
            List of command parts
        """
        parts = []

        # Add program
        parts.append(self._exec)

        # Add options
        for name, value in self._opts:
            if value is not None:
                parts.append(f"{name}")
                parts.append(f"{value}")
            else:
                parts.append(f"{name}")

        # Add arguments
        parts.extend(self._args)

        return parts

    def as_str(self) -> str:
        """Convert command to string format.

        Returns:
            Shell-quoted string representation
        """
        return shlex.join(self.as_list())

    def __iter__(self) -> Iterator[str]:
        """Enable list(command) syntax."""
        return iter(self.as_list())

    def __str__(self) -> str:
        """Enable str(command) syntax.

        Returns:
            Shell-quoted string representation
        """
        return self.as_str()


def format_command(command: Command | str | list[str]) -> str:
    """Format a command for logging."""
    if isinstance(command, Command):
        return str(command)
    elif isinstance(command, str):
        return command
    elif isinstance(command, list):
        return shlex.join(command)
