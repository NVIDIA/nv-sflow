# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sflow watch`` and ``sflow fetch``: recover a delegated run from its receipt."""

import shlex
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from sflow.cli import DOCS_URL, app
from sflow.ssh import delegate as ssh_delegate
from sflow.ssh.receipt import load_receipt
from sflow.ssh.transport import DEFAULT_TRANSPORT


def _receipt_endpoints(data: dict[str, object]) -> tuple[list[str], str, str]:
    session = str(data["remote_session"])
    return (
        ssh_delegate.ssh_argv(str(data["connection"])),
        f"{session}/remote_helper.py",
        f"{session}/output",
    )


@app.command(name="watch", epilog=f"Documentation: {DOCS_URL}")
def watch(
    receipt: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    once: Annotated[
        bool,
        typer.Option("--once", help="Print one snapshot and exit."),
    ] = False,
) -> None:
    """Follow logs from an SSH receipt; Ctrl-C does not cancel the remote run."""
    data = load_receipt(receipt)
    ssh, helper, output = _receipt_endpoints(data)
    if once:
        ssh_delegate.snapshot(ssh, helper, output, None, 0, connection_hint=True)
        return
    process = DEFAULT_TRANSPORT.start_to_stderr(
        ssh, f"exec python3 {shlex.quote(helper)} follow {shlex.quote(output)}"
    )
    try:
        return_code = process.wait()
        if return_code == 255:
            ssh_delegate.ssh_failure_hint(ssh)
        raise typer.Exit(code=return_code)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        typer.echo("Stopped watching; the remote run was not cancelled.", err=True)
        raise typer.Exit(code=130) from None


@app.command(name="fetch", epilog=f"Documentation: {DOCS_URL}")
def fetch(
    receipt: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, resolve_path=True),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Override the receipt destination."),
    ] = None,
    all_files: Annotated[
        bool,
        typer.Option("--all", help="Fetch the complete remote output directory."),
    ] = False,
) -> None:
    """Fetch current logs or complete output from an SSH receipt."""
    data = load_receipt(receipt)
    command = str(data["command"])
    saved_output = data.get("local_output")
    saved_destination = output_dir is None
    if output_dir is None:
        if not isinstance(saved_output, str) or not saved_output:
            raise typer.BadParameter(
                "receipt has no local output; pass --output-dir explicitly"
            )
        output_dir = Path(saved_output)
    destination = output_dir.expanduser().resolve()
    fetch_destination = (
        destination.parent
        if command == "compose" and saved_destination
        else destination
    )
    mode = {"compose": "all", "batch": "batch"}.get(command, "logs")
    if all_files:
        mode = "all"
    ssh, helper, output = _receipt_endpoints(data)
    if not ssh_delegate.fetch(
        ssh,
        helper,
        output,
        mode,
        fetch_destination,
    ):
        raise typer.Exit(code=1)
    typer.echo(f"Fetched remote output to {destination}", err=True)
