# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recovery receipt an SSH delegation leaves behind.

One module owns the schema so the writer (``ssh/delegate.py``) and the readers
(``sflow watch`` / ``sflow fetch``) cannot drift: a key added on one side but
not the other used to surface only as a ``KeyError`` mid-recovery.

The receipt is rewritten in place as the run advances through the phases in
:data:`PHASES`, in that order -- ``launch`` and ``fetch`` are skipped when the
run never gets that far, but the sequence never runs backwards.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import typer

PHASES = ("bootstrap", "execution", "launch", "postprocessing", "fetch", "finished")

#: Keys every receipt carries, all of them non-empty strings. Kept minimal so a
#: hand-written or older receipt is still usable for ``watch``/``fetch``
#: recovery -- those commands only need to reach the remote host, not replay
#: the full phase history.
REQUIRED_KEYS = (
    "command",
    "connection",
    "remote_session",
)
#: Keys a receipt may carry once the run reaches the phase that sets them.
OPTIONAL_KEYS = (
    "session",
    "session_dir",
    "status",
    "execution_status",
    "fetch_status",
    "phase",
    "local_output",
    "return_code",
    "backend_status",
)


def initial_receipt(
    *,
    command: str,
    connection: str,
    session: str,
    session_dir: Path,
    remote_session: str,
    local_output: Path | None,
    fetch: str,
) -> dict[str, object]:
    """The receipt as it stands before the remote host is touched."""
    return {
        "command": command,
        "connection": connection,
        "session": session,
        "session_dir": str(session_dir),
        "remote_session": remote_session,
        "local_output": str(local_output) if local_output else None,
        "status": "bootstrapping",
        "execution_status": "pending",
        "fetch_status": "skipped" if fetch == "none" else "pending",
        "phase": "bootstrap",
    }


def write_receipt(path: Path, receipt: dict[str, object]) -> None:
    """Atomically publish a mode-0600 recovery receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)


def load_receipt(path: Path) -> dict[str, object]:
    """Read a receipt, rejecting anything the recovery commands cannot use."""
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"could not read SSH receipt {path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise typer.BadParameter(f"invalid SSH receipt: {path}")
    for key in REQUIRED_KEYS:
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise typer.BadParameter(f"SSH receipt is missing {key!r}: {path}")
    if not str(receipt["remote_session"]).startswith("/"):
        raise typer.BadParameter(f"SSH receipt has an invalid remote_session: {path}")
    if "phase" in receipt and receipt["phase"] not in PHASES:
        raise typer.BadParameter(
            f"SSH receipt has an unknown phase {receipt['phase']!r}: {path}"
        )
    local_output = receipt.get("local_output")
    if local_output is not None and not isinstance(local_output, str):
        raise typer.BadParameter(f"SSH receipt has an invalid local_output: {path}")
    if "return_code" in receipt and not isinstance(receipt["return_code"], int):
        raise typer.BadParameter(f"SSH receipt has an invalid return_code: {path}")
    return receipt
