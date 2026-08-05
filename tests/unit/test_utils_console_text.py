# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ANSI/``\\r`` handling for everything that echoes task output to the console.

These two collapses look interchangeable and are not. Getting them the wrong way round
does not raise -- it silently deletes lines from the console (``last_visible_frame``
semantics applied as last-frame-outright) or splices a superseded frame onto the next
read (``frame_in_progress`` semantics applied to a completed line). Both failures are
invisible in the on-disk log, so they are pinned here.
"""

import pytest

from sflow.utils.console_text import (
    CONSOLE_LINE_CHAR_CAP,
    clamp_for_console,
    frame_in_progress,
    last_visible_frame,
    rejoin_carried_frame,
    strip_ansi,
)

# (raw, the single frame a terminal would have left on screen)
COMPLETE_LINES = [
    (b"server started on port 8000", b"server started on port 8000"),
    # A CRLF line: splitting the stream on \n leaves the \r behind. Taking the last
    # frame outright yields b"" here, so every line of a CRLF-emitting container
    # would vanish from the console while <task>.log filled up normally.
    (b"server started on port 8000\r", b"server started on port 8000"),
    (b"\rProc 1/3\rProc 2/3\rProc 3/3", b"Proc 3/3"),  # \r starts each frame
    (b"Proc 1/3\rProc 2/3\rProc 3/3\r", b"Proc 3/3"),  # \r ends each frame
    (b"Downloading 99%\rDone\r", b"Done"),  # bar clears itself before finishing
    (b"a\r\r\rb", b"b"),  # consecutive redraws
    (b"\r\r\r", b""),  # nothing was ever drawn
    (b"", b""),
]


@pytest.mark.parametrize("raw,expected", COMPLETE_LINES)
def test_last_visible_frame_keeps_the_last_non_empty_frame(raw, expected):
    assert last_visible_frame(raw) == expected


@pytest.mark.parametrize("raw,expected", COMPLETE_LINES)
def test_last_visible_frame_matches_on_str(raw, expected):
    """Same rule for str: the on-disk log rebuild collapses decoded text, the live
    tailers collapse raw bytes, and the two must agree on which frame survived."""
    assert last_visible_frame(raw.decode()) == expected.decode()


def test_frame_in_progress_returns_only_what_follows_the_last_cr():
    """An INCOMPLETE line is still being written, so a trailing ``\\r`` means "nothing
    drawn yet" -- returning the superseded frame would splice it onto the next read."""
    assert frame_in_progress(b"Proc 1/3\rProc 2/3\r") == b""
    assert frame_in_progress(b"Proc 1/3\rProc 2/") == b"Proc 2/"
    assert frame_in_progress(b"no redraws yet") == b"no redraws yet"
    assert frame_in_progress("Proc 1/3\rProc 2/") == "Proc 2/"


def test_the_two_collapses_differ_exactly_where_it_matters():
    """The whole reason both exist: they disagree on a line ending in ``\\r``, and each
    is correct only for its own case. A single shared helper would be wrong for one."""
    ends_on_redraw = b"Processing: 99%\r"
    assert last_visible_frame(ends_on_redraw) == b"Processing: 99%"  # complete -> show it
    assert frame_in_progress(ends_on_redraw) == b""  # in flight -> nothing drawn yet


def test_collapsing_a_megabyte_bar_is_allocation_bounded():
    """This runs on the event-loop thread against multi-MB lines, so it must not
    materialise one object per frame."""
    frames = 20000
    bar = b"".join(b"\rProcessing: %d/%d [%s]" % (i, frames, b"." * 120)
                   for i in range(1, frames + 1))
    assert len(bar) > 2_000_000
    assert last_visible_frame(bar) == b"Processing: %d/%d [%s]" % (
        frames, frames, b"." * 120
    )


def test_strip_ansi_removes_colour_and_cursor_sequences():
    assert strip_ansi("\x1b[32mgreen\x1b[0m") == "green"
    assert strip_ansi("\x1b[2K\x1b[1Gredraw") == "redraw"
    assert strip_ansi("plain") == "plain"


# ---------------------------------------------------------------------------
# rejoin_carried_frame: putting a carried frame back WITHOUT splicing
# ---------------------------------------------------------------------------


def test_rejoin_restores_overwrite_semantics_instead_of_concatenating():
    """A carried frame must be OVERWRITTEN by what follows, not prefixed to it.

    When a read ends on a `\\r` the retained buffer collapses to nothing, so the frame the
    terminal is still showing has to be carried separately. Putting it back by plain
    concatenation was a real bug: two consecutive redraw-terminated reads rendered as
    "50%60%" -- a line the task never displayed -- and because the launcher's tail flush
    persists that same string, the splice reached <task>.log too.
    """
    joined = rejoin_carried_frame("Epoch  50%", "Epoch  60%\r")
    assert last_visible_frame(joined) == "Epoch  60%", (
        "the newer frame overwrites the carried one; concatenation would give "
        f"'Epoch  50%Epoch  60%', got {last_visible_frame(joined)!r}"
    )


def test_rejoin_keeps_the_carried_frame_when_only_a_newline_follows():
    """The case the carry exists for: the bar's terminating newline arrives later.

    Nothing overwrites the frame here, so it must survive -- this is why the carried
    frame cannot simply be dropped when new bytes arrive.
    """
    assert last_visible_frame(rejoin_carried_frame("Done 100%", "")) == "Done 100%"


def test_rejoin_lets_plain_text_overwrite_a_carried_frame():
    assert frame_in_progress(rejoin_carried_frame("50%", "abc")) == "abc"


def test_rejoin_is_a_noop_with_nothing_carried():
    """Callers apply it unconditionally, so the empty case must not inject a `\\r`."""
    assert rejoin_carried_frame("", "plain text") == "plain text"
    assert rejoin_carried_frame(b"", b"plain bytes") == b"plain bytes"


def test_rejoin_works_on_bytes_too():
    """The k8s tailer carries bytes so it can collapse before decoding."""
    joined = rejoin_carried_frame(b"step 1", b"step 2\r")
    assert isinstance(joined, bytes)
    assert last_visible_frame(joined) == b"step 2"


# ---------------------------------------------------------------------------
# clamp_for_console: the shared console length cap
# ---------------------------------------------------------------------------


def test_clamp_passes_a_normal_line_through_untouched():
    assert clamp_for_console("a readable log line") == "a readable log line"
    exact = "x" * CONSOLE_LINE_CHAR_CAP
    assert clamp_for_console(exact) == exact, "a line exactly at the cap is not truncated"


def test_clamp_bounds_a_line_with_no_cr_to_collapse():
    """The case `\\r` collapsing cannot help with: one huge line, no redraws.

    Rendering it through the console handler costs ~300 bytes of RSS per character on
    the event-loop thread, which is what froze the driver. The cap is the only thing
    that bounds it, so the OUTPUT length -- not just the input -- must be bounded.
    """
    out = clamp_for_console("j" * (48 * 1024 * 1024), source="mlperf_harness.log")
    assert len(out) < CONSOLE_LINE_CHAR_CAP + 200, f"still unbounded at {len(out)} chars"
    assert out.startswith("j" * 100), "the readable head must survive"
    assert "mlperf_harness.log" in out, "the reader must be told where the full line is"
    assert "48" in out or "50331648" in out, "the true length should be reported"


def test_clamp_names_the_task_log_when_no_source_is_given():
    out = clamp_for_console("z" * 5000)
    assert "full line in the task log" in out


def test_clamp_is_cheap_enough_to_run_on_the_event_loop():
    """Bounding the render is pointless if the clamp itself is what blocks the loop."""
    import time

    big = "j" * (48 * 1024 * 1024)
    start = time.perf_counter()
    clamp_for_console(big)
    assert time.perf_counter() - start < 0.5, "the clamp must be a slice, not a scan"
