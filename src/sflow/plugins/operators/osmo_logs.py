# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream one OSMO task's log into stdout, ready for sflow's ``<task>.log``.

Usage: ``python -m sflow.plugins.operators.osmo_logs <workflow_id> <task> <state_file>``

- Drops OSMO's ``Workflow <id> has logs:`` header and the
  ``YYYY/MM/DD HH:MM:SS [<task>] `` prefix on workload lines, so result and probe
  patterns see the same text as on other backends. OSMO's own ``[<task>][osmo]``
  lines keep their prefix.
- OSMO replays the log from the start on every attach. ``state_file`` holds how
  many lines earlier attaches already wrote, and those are skipped, so the
  operator can re-attach after a cut stream without duplicating lines.
- Drops the CLI's "New client available" banner from stderr; other stderr passes through.
- Exits ``CUT`` (75) when the CLI reports the stream timed out, else the CLI's code.
"""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

CUT = 75
_HEADER = re.compile(r"^Workflow \S+ has logs:$")
_TIMED_OUT = "Log stream has timed out or failed."
# The CLI's "new client available" banner, printed to stderr on every call.
_BANNER = re.compile(
    r"^(WARNING: New client \S+ available\.|Current version: \S+|"
    r"Please run the following command:|curl -fsSL \S+/cli/install\.sh \| bash)$"
)


def main(wf_id: str, task: str, state_file: str) -> int:
    """Stream the log once; return CUT, or the ``osmo workflow logs`` exit code."""
    state = Path(state_file)
    done = int(state.read_text() or 0) if state.exists() else 0
    prefix = re.compile(
        rf"^\d{{4}}/\d{{2}}/\d{{2}} \d{{2}}:\d{{2}}:\d{{2}} \[{re.escape(task)}\] "
    )
    # stderr goes to a file (no pipe to deadlock on) and is relayed without the banner.
    err = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(
        ["osmo", "workflow", "logs", wf_id, "--task", task],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=err,
        text=True,
        bufsize=1,
    )
    signal.signal(signal.SIGTERM, lambda *_: (proc.kill(), sys.exit(143)))
    seen, blank, cut = 0, False, False
    try:
        for i, line in enumerate(proc.stdout or []):
            text = line.rstrip("\n")
            if i == 0 and _HEADER.match(text):
                continue
            if text.startswith(_TIMED_OUT):
                cut = True  # the CLI prints a blank line first: drop both
                break
            if blank:  # emit a held blank line once the next line shows it is real
                seen += 1
                if seen > done:
                    sys.stdout.write("\n")
            blank = text == ""
            if blank:
                continue
            seen += 1
            if seen > done:
                sys.stdout.write(prefix.sub("", line))
                sys.stdout.flush()
        rc = proc.wait()
    finally:  # also on SIGTERM, so a teardown keeps the CLI's error output
        state.write_text(str(max(seen, done)))
        err.seek(0)
        sys.stderr.writelines(x for x in err if not _BANNER.match(x.rstrip("\n")))
    return CUT if cut else rc


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
