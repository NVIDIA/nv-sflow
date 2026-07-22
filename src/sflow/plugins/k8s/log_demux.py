# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming stdin demultiplexer for a merged-pod's Kubernetes log.

A merge-pod's one container carries several members' logs, each line tagged
``[[sflow-mux:<task>]] `` by the launcher (see ``k8s.shell.merged_launcher_lines``).
This tiny child process reads that single stream and routes each line to that
member's ``<task>.log`` (tag stripped); every other line (pod-level output or an
unknown tag) goes verbatim to the leader/default log.

Why a Python child and not the previous ``awk`` demux
-----------------------------------------------------
The default ``awk`` on Debian/Ubuntu is ``mawk``, which **block-buffers its
stdin**: it does not process a line until its input buffer fills or the stream
ends. In ``kubectl logs -f | awk`` this means that when the merged stream goes
quiet -- a member reaches its readiness marker and stops logging -- that member's
buffered tail is never processed, so its ``<task>.log`` **freezes at its last
lines** while chattier members keep crossing the buffer boundary and appear live.
A fresh ``kubectl logs`` (no ``awk``) shows every member reached the marker, which
is exactly the reported "3 of 4 tasks update, 1 is stale" symptom. ``fflush``/
``close`` in the awk program cannot fix this -- the stall is on the *input* side.

This reader instead uses ``os.read`` (which returns whatever bytes are available,
never withholding a quiet tail) and flushes each touched file on a short interval,
so both chatty and quiet members land promptly. It still runs as its own process,
so the sflow driver's event loop is never in the per-line byte path -- the whole
point of the offload.

The same routing is used for the post-terminal rebuild (``... < complete.dump``),
so the live split and the finalize re-split can never diverge.
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import time
from typing import BinaryIO

from sflow.plugins.k8s.render import MERGE_MUX_CLOSE, MERGE_MUX_OPEN

_OPEN = MERGE_MUX_OPEN.encode()
_CLOSE = MERGE_MUX_CLOSE.encode()

# 64 KiB reads keep a chatty/large stream cheap (few syscalls per MB).
_READ_SIZE = 1 << 16
# Per-file buffer; a chatty member auto-flushes to the OS whenever this fills, so
# high-volume output never waits on the interval below.
_WRITE_BUFFER = 1 << 16
# Upper bound on how long a quiet member's buffered tail waits before it is on
# disk (and thus visible to the console tailer, probes, and humans). Small enough
# to feel live, large enough that the flush is negligible next to the log volume.
_FLUSH_INTERVAL = 0.5


class _Router:
    """Routes tagged lines to per-member append files, batching + interval flush."""

    def __init__(self, default_path: str, routes: dict[bytes, str]):
        self._default_path = default_path
        self._routes = routes
        # One handle per distinct path (the leader's own mux tag and the pod-level
        # default share a path -> one handle), opened lazily on first write.
        self._files: dict[str, BinaryIO] = {}
        self._dirty: set[str] = set()

    def _handle(self, path: str) -> BinaryIO:
        fh = self._files.get(path)
        if fh is None:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Append (never truncate) so the leader file's apply-diagnostics prefix
            # and any prior content are preserved, mirroring the old awk ``>>``.
            fh = open(path, "ab", buffering=_WRITE_BUFFER)
            self._files[path] = fh
        return fh

    def route(self, line: bytes) -> None:
        """Write ONE line (newline already stripped) to its destination file.

        A ``[[sflow-mux:<task>]] `` line for a known member -> that member's file,
        tag stripped. Anything else (untagged pod-level line, or an unknown member
        tag) -> the default/leader file, verbatim -- identical to the awk routing.
        """
        path = self._default_path
        payload = line
        if line.startswith(_OPEN):
            rest = line[len(_OPEN) :]
            idx = rest.find(_CLOSE)
            if idx >= 0:
                target = self._routes.get(rest[:idx])
                if target is not None:
                    path = target
                    payload = rest[idx + len(_CLOSE) :]
        fh = self._handle(path)
        fh.write(payload)
        fh.write(b"\n")
        self._dirty.add(path)

    def flush(self) -> None:
        for path in self._dirty:
            try:
                self._files[path].flush()
            except OSError:
                pass
        self._dirty.clear()

    def close(self) -> None:
        for fh in self._files.values():
            try:
                fh.flush()
                fh.close()
            except OSError:
                pass
        self._files.clear()
        self._dirty.clear()


def demux_stream(
    src_fd: int, router: _Router, *, flush_interval: float = _FLUSH_INTERVAL
) -> None:
    """Route ``src_fd`` line by line until EOF, flushing at least every interval.

    ``select`` wakes us on the interval even while the stream is quiet, so a
    member's last line is flushed to disk within ``flush_interval`` instead of
    sitting in a buffer indefinitely (the mawk failure mode). ``os.read`` returns
    whatever is available, so a quiet stream is never withheld.
    """
    pending = b""
    last_flush = time.monotonic()
    while True:
        try:
            ready, _, _ = select.select([src_fd], [], [], flush_interval)
        except (OSError, ValueError):
            break
        if ready:
            try:
                chunk = os.read(src_fd, _READ_SIZE)
            except OSError:
                break
            if not chunk:
                break  # EOF: writer closed (kubectl exited / dump fully read)
            pending += chunk
            parts = pending.split(b"\n")
            pending = parts.pop()  # trailing (incomplete) line waits for its newline
            for line in parts:
                router.route(line)
        now = time.monotonic()
        if now - last_flush >= flush_interval:
            router.flush()
            last_flush = now
    if pending:  # final line with no trailing newline (stream cut mid-line)
        router.route(pending)
    router.flush()


def _parse_route(value: str) -> tuple[bytes, str]:
    task, sep, path = value.partition("=")
    if not sep or not task:
        raise argparse.ArgumentTypeError("--route must be TASK=PATH")
    return task.encode(), path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sflow merged-pod log demuxer")
    parser.add_argument(
        "--default", required=True, help="leader/default log path (untagged lines)"
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        type=_parse_route,
        metavar="TASK=PATH",
        help="route one merged member's tagged lines to PATH (repeatable)",
    )
    args = parser.parse_args(argv)
    router = _Router(args.default, dict(args.route))

    def _stop(_signum: int, _frame: object) -> None:
        # Flush + exit before the parent's SIGTERM->SIGKILL escalation, so a
        # teardown never drops the buffered tail. Default SIGTERM would skip this.
        router.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        demux_stream(sys.stdin.fileno(), router)
    finally:
        router.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
