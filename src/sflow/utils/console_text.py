# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turning raw task output into console-safe text (ANSI + ``\\r`` handling).

Every path that echoes a task's own output has to answer the same two questions, and
they used to be answered separately -- and inconsistently -- in each one:

* ``SubprocessLauncher._feed`` / ``_flush_tail`` (local, docker, slurm: a PTY read loop),
* ``tail_file_to_console`` (kubernetes: an offloaded ``<task>.log`` tailer),
* ``_sanitize_log_line`` (the post-run on-disk ``<task>.log`` rebuild).

The divergence was not academic. A terminal treats ``\\r`` as "return to column 0, what
follows overwrites what came before", so a completed line's only visible content is its
last non-empty frame -- but ``rsplit("\\r", 1)[-1]`` returns the last frame whether or not
it is empty, which silently renders as NOTHING for a CRLF line (splitting on ``\\n``
leaves the ``\\r``) or a bar whose frames END with ``\\r``. The launcher got away with it
only because it normalises ``\\r\\n`` to ``\\n`` several lines earlier; the k8s tailer,
which splits raw bytes, did not, and dropped those lines from the console entirely.

Hence two explicitly-named operations instead of one ambiguous idiom:

* :func:`last_visible_frame` -- for a COMPLETE line (terminated, or being persisted).
* :func:`frame_in_progress` -- for the INCOMPLETE trailing line still being written.

Both accept ``str`` or ``bytes`` and return the same type, so callers can collapse before
decoding (which matters when the line is megabytes of superseded progress-bar frames).
"""

import re
from typing import AnyStr

# Compiled once at import: recompiling per line was a measurable cost when task output is
# chatty (this runs for every line of every task's stdout/stderr).
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences (colour, cursor movement) from ``text``."""
    return _ANSI_ESCAPE_RE.sub("", text)


def _cr(sample: AnyStr) -> AnyStr:
    return b"\r" if isinstance(sample, bytes) else "\r"


def last_visible_frame(raw: AnyStr) -> AnyStr:
    """The only frame of a COMPLETE line a terminal would have left on screen.

    Collapses ``\\r`` redraws to the last NON-EMPTY frame, which is what makes an
    hour-long progress bar -- one physical line holding megabytes of superseded frames --
    into the single short line it always looked like. Taking the last frame outright
    instead would return empty (and so drop the line) for every one of:

      * ``"text\\r"``          -- a CRLF line, once the ``\\n`` split has been done;
      * ``"f1\\rf2\\rf3\\r"``    -- a bar that ends each frame with ``\\r``;
      * ``"Downloading\\rDone\\r"`` -- a bar that clears itself before finishing.

    Implemented as strip-trailing-separators + one split rather than splitting every
    frame: identical result, but O(1) allocations instead of one object per frame, which
    matters because this runs on the event loop against multi-megabyte lines.
    """
    cr = _cr(raw)
    if cr not in raw:
        return raw
    return raw.rstrip(cr).rsplit(cr, 1)[-1]


def frame_in_progress(raw: AnyStr) -> AnyStr:
    """The in-flight frame of an INCOMPLETE line: everything after the last ``\\r``.

    Deliberately NOT :func:`last_visible_frame`. This line is still being written, so a
    trailing ``\\r`` means "nothing drawn yet since the last redraw" -- resurrecting the
    superseded frame would splice it onto the bytes that arrive next read. Collapsing here
    is also what stops a bar that never emits a newline from growing the retained buffer
    without bound for as long as it runs.

    When the caller needs to keep that superseded frame anyway (so an unterminated bar's
    last state is not lost at EOF), it must hold it SEPARATELY and rejoin it with
    :func:`rejoin_carried_frame` -- never by plain concatenation, which splices.
    """
    return raw.rsplit(_cr(raw), 1)[-1]


def rejoin_carried_frame(carried: AnyStr, incoming: AnyStr) -> AnyStr:
    """Put a carried-over redraw frame back in front of the bytes that follow it.

    ``carried`` is the frame a reader kept because :func:`frame_in_progress` collapsed the
    trailing line to nothing (it ended on a ``\\r``), so the terminal is still displaying
    it while the retained buffer is empty. The next read has to reconstruct that physical
    line -- but CONCATENATING is wrong, because ``\\r`` means "return to column 0 and
    overwrite": ``"50%\\r"`` then ``"60%\\r"`` displays ``60%``, not ``50%60%``.

    Re-inserting the separator restores the overwrite semantics, so the collapse functions
    resolve it correctly whichever way the line ends up going:

      * more redraws  -> ``"50%\\r60%\\r"``  -> last_visible_frame -> ``"60%"``
      * plain text    -> ``"50%\\rabc"``     -> frame_in_progress  -> ``"abc"``
      * late newline  -> ``"50%\\r"`` + ``""`` -> last_visible_frame -> ``"50%"``

    Returns ``incoming`` untouched when there is nothing carried, so callers can apply it
    unconditionally.
    """
    if not carried:
        return incoming
    return carried + _cr(carried) + incoming


# Max characters of ONE line echoed to the console, shared by every backend's console
# path. Rendering an unbounded line through the rich console handler is what froze the
# driver: measured at ~6.3us and ~300 BYTES OF RSS PER CHARACTER (a 4 MB line -> 25s and
# 1.15 GB), so a 48 MB line is ~5 CPU-minutes and ~14 GB -- swap, on a normal workstation.
# It runs on the event-loop thread, so nothing else in the driver ticks meanwhile.
#
# Capped at a few console rows' worth: enough to read, ~15 KB of render work at worst.
# The console is best-effort observability; the per-task log on disk still holds every
# byte, so this bounds only what is echoed, never what is persisted.
CONSOLE_LINE_CHAR_CAP = 2000


def clamp_for_console(text: str, *, cap: int = CONSOLE_LINE_CHAR_CAP, source: str = "") -> str:
    """Bound ONE line's length for the console, naming where the full line still lives.

    Applied only on the way to the console/TUI -- never to the text written to
    ``<task>.log``, which must stay complete. ``\\r`` collapsing (:func:`last_visible_frame`)
    handles progress bars; this is the backstop for a long line with no ``\\r`` to collapse
    (a JSON or base64 dump), which the console cannot usefully show anyway.
    """
    if len(text) <= cap:
        return text
    where = f"full line in {source}" if source else "full line in the task log"
    return (
        f"{text[:cap]} ... [sflow: line truncated for the console at "
        f"{cap} of {len(text)} chars; {where}]"
    )
