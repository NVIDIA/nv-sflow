# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the merged-pod streaming log demuxer (plugins/k8s/log_demux.py).

The demuxer replaces the former ``awk`` split because ``mawk`` block-buffers its
stdin -- a quiet member's tail never reached disk until the stream got chattier.
The key regression here (:func:`test_demux_stream_flushes_quiet_member_before_eof`)
proves a member's line lands on disk while the stream stays OPEN and idle.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

import pytest

from sflow.plugins.k8s import log_demux


def test_router_routes_tagged_untagged_and_unknown(tmp_path):
    leader = tmp_path / "leader.log"
    decode = tmp_path / "decode.log"
    router = log_demux._Router(str(leader), {b"decode": str(decode)})
    router.route(b"[[sflow-mux:decode]] body one")  # known member -> tag stripped
    router.route(b"plain pod-level line")            # untagged -> leader verbatim
    router.route(b"[[sflow-mux:other]] x")           # unknown tag -> leader verbatim
    router.route(b"[[sflow-mux:]] empty")            # empty tag -> leader verbatim
    router.route(b"[[sflow-mux:decode]] body two")
    router.close()
    assert decode.read_text() == "body one\nbody two\n"
    assert leader.read_text() == (
        "plain pod-level line\n[[sflow-mux:other]] x\n[[sflow-mux:]] empty\n"
    )


def test_router_appends_preserving_prefix(tmp_path):
    # Files open in append mode, so a pre-existing apply-diagnostics prefix survives.
    leader = tmp_path / "leader.log"
    leader.write_text("apply-prefix\n")
    router = log_demux._Router(str(leader), {})
    router.route(b"pod line")
    router.close()
    assert leader.read_text() == "apply-prefix\npod line\n"


def test_router_creates_missing_parent_dir(tmp_path):
    leader = tmp_path / "nested" / "dir" / "leader.log"
    router = log_demux._Router(str(leader), {})
    router.route(b"x")
    router.close()
    assert leader.read_text() == "x\n"


def test_demux_stream_writes_trailing_line_without_newline(tmp_path):
    # A stream cut mid-line (no trailing newline) must still write the final line --
    # matching the launcher's own `read ... || [ -n ]` last-line capture.
    leader = tmp_path / "leader.log"
    router = log_demux._Router(str(leader), {})
    r, w = os.pipe()
    os.write(w, b"line without newline")
    os.close(w)  # EOF with a partial last line
    try:
        log_demux.demux_stream(r, router, flush_interval=0.05)
    finally:
        os.close(r)
        router.close()
    assert leader.read_text() == "line without newline\n"


def test_demux_stream_flushes_quiet_member_before_eof(tmp_path):
    # THE regression: with the stream still OPEN and idle, a member's line must reach
    # disk within a flush interval. mawk would withhold it (block-buffered stdin)
    # until the stream got chattier or ended, freezing that member's <task>.log.
    leader = tmp_path / "leader.log"
    member = tmp_path / "m.log"
    router = log_demux._Router(str(leader), {b"m": str(member)})
    r, w = os.pipe()
    t = threading.Thread(
        target=log_demux.demux_stream,
        args=(r,),
        kwargs={"router": router, "flush_interval": 0.05},
        daemon=True,
    )
    t.start()
    try:
        os.write(w, b"[[sflow-mux:m]] ready marker\n")
        got = ""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if member.exists():
                got = member.read_text()
                if got:
                    break
            time.sleep(0.02)
        assert got == "ready marker\n"  # landed while the stream is still open
    finally:
        os.close(w)  # EOF -> demux_stream returns
        t.join(timeout=2.0)
        os.close(r)
        router.close()
    assert not t.is_alive()


def test_main_cli_end_to_end(tmp_path, fake_process):
    # Exercises argument parsing + main() through the real interpreter, the same way
    # the offloaded pipeline (`kubectl logs -f | python -m ...log_demux`) invokes it.
    fake_process.allow_unregistered(True)  # run the real python demuxer
    leader = tmp_path / "leader.log"
    decode = tmp_path / "decode.log"
    subprocess.run(
        [
            sys.executable, "-m", "sflow.plugins.k8s.log_demux",
            "--default", str(leader), "--route", f"decode={decode}",
        ],
        input="[[sflow-mux:decode]] hi\nuntagged line\n",
        text=True,
        check=True,
    )
    assert decode.read_text() == "hi\n"
    assert leader.read_text() == "untagged line\n"


def test_parse_route_rejects_missing_equals():
    with pytest.raises(argparse.ArgumentTypeError):
        log_demux._parse_route("no-equals-sign")
    task, path = log_demux._parse_route("decode=/tmp/decode.log")
    assert task == b"decode" and path == "/tmp/decode.log"
