# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one place SSH delegation shells out.

Every remote invocation goes through a :class:`Transport`, so the delegate
itself contains no ``subprocess`` calls and can be driven by a test double.
:class:`SubprocessTransport` is the production implementation and reproduces
the argv, redirections and timeouts the delegate used before this seam existed.
"""

from __future__ import annotations

import subprocess
import sys
from typing import IO, Protocol


class Transport(Protocol):
    """The remote-invocation shapes SSH delegation needs."""

    def capture(
        self, ssh: list[str], command: str, *, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run ``command`` remotely and return its captured text output."""

    def run_to_stderr(
        self, ssh: list[str], command: str, *, stdin: IO[bytes] | None = None
    ) -> subprocess.CompletedProcess:
        """Run ``command`` remotely with both streams inherited onto stderr."""

    def run_to_file(
        self, ssh: list[str], command: str, stdout: IO[bytes]
    ) -> subprocess.CompletedProcess:
        """Run ``command`` remotely, writing its raw stdout to ``stdout``."""

    def start(self, argv: list[str], command: str) -> subprocess.Popen:
        """Start ``command`` remotely with this process's stdio inherited."""

    def start_to_stderr(self, ssh: list[str], command: str) -> subprocess.Popen:
        """Start ``command`` remotely with its output folded onto stderr."""


class SubprocessTransport:
    """Run remote commands with the local ``ssh`` client."""

    def capture(
        self, ssh: list[str], command: str, *, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*ssh, command],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def run_to_stderr(
        self, ssh: list[str], command: str, *, stdin: IO[bytes] | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*ssh, command],
            stdin=subprocess.DEVNULL if stdin is None else stdin,
            stdout=sys.stderr,
            stderr=sys.stderr,
            check=False,
        )

    def run_to_file(
        self, ssh: list[str], command: str, stdout: IO[bytes]
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*ssh, command],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            check=False,
        )

    def start(self, argv: list[str], command: str) -> subprocess.Popen:
        return subprocess.Popen([*argv, command])

    def start_to_stderr(self, ssh: list[str], command: str) -> subprocess.Popen:
        return subprocess.Popen(
            [*ssh, command],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )


DEFAULT_TRANSPORT = SubprocessTransport()
