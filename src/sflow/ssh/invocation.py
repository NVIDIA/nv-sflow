# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What ``sflow <command> --ssh ...`` was asked to do, as one value.

Delegation runs the command's own click parser but stops before Typer's type
conversion: options like ``--kubeconfig`` name files on the *remote* host, so
resolving them locally would fail before delegation ever starts. Parsing with
the real parser (rather than a hand-rolled scan) is what keeps ``-fflow.yaml``,
``--file=flow.yaml`` and positional arguments meaning here exactly what they
mean to a local run.

:class:`Invocation` carries ``raw_args`` too, so a delegation is fully
described by this value and :func:`sflow.ssh.delegate.delegate` never has to
reach back into ``sys.argv``.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import typer

_DELEGATED_COMMANDS = {"run", "batch", "compose"}
# typer vendors click, so click's UsageError is only reachable through a class
# typer re-exports.
_UsageError = typer.BadParameter.__mro__[1]
# What the parser leaves in a slot nobody filled: None up to click 8.2, an UNSET
# sentinel from 8.3 on. Reached through the same class as above, because the
# click typer parses with may be the vendored copy rather than the package.
try:
    _UNSET = getattr(
        importlib.import_module(
            f"{sys.modules[_UsageError.__module__].__package__}._utils"
        ),
        "UNSET",
        None,
    )
except ImportError:  # click < 8.3 has no such module, and no such sentinel
    _UNSET = None
_CONFIG_SUFFIXES = {".yaml", ".yml"}


@dataclass(frozen=True)
class Invocation:
    """One delegated CLI invocation."""

    command: str
    connection: str
    raw_args: list[str] = field(default_factory=list)
    follow: str = "auto"
    fetch: str = "logs"
    remote_root: str | None = None
    tty: str = "auto"
    workspace_dir: Path | None = None
    output_dir: Path | None = None
    input_files: list[Path] = field(default_factory=list)
    config_files: list[Path] = field(default_factory=list)
    variable_overrides: list[str] = field(default_factory=list)
    artifact_overrides: list[str] = field(default_factory=list)
    missable_tasks: list[str] = field(default_factory=list)
    skip_artifact_check: bool = False
    bulk_input: Path | None = None
    compose_output: Path | None = None
    compose_bulk: bool = False
    batch_sbatch_path: Path | None = None
    batch_runtime_explicit: bool = False
    log_level: str = "INFO"


def parse_command_options(command: str, args: list[str]) -> dict[str, object] | None:
    """Split ``args`` with the command's own parser, without converting values.

    Returns what was actually given, by parameter name, or None when the
    invocation is a help request rather than a run. Only the ``--ssh-*``
    switches fall back to their declared defaults: the path options default to
    values already resolved against *this* process's cwd, which is not what an
    omitted ``--output-dir`` means to the delegate.
    """
    import typer.main

    from sflow.cli import app  # imported here: sflow.cli imports this package back

    group = typer.main.get_command(app)
    click_command = group.commands[command]
    context = type(group).context_class(click_command, info_name=command)
    try:
        values, _positional, _order = click_command.make_parser(context).parse_args(
            list(args)
        )
    except _UsageError as exc:
        raise typer.BadParameter(str(exc)) from None
    # An empty slot means the parameter was not given. Dropping it here is what
    # keeps ``options.get(...)`` meaning "absent": left in, click >= 8.3's UNSET
    # would reach the callers as the file name "Sentinel.UNSET".
    given = {
        name: value
        for name, value in values.items()
        if value is not None and value is not _UNSET
    }
    if given.get("help"):
        return None
    switch_defaults = {
        param.name: param.default
        for param in click_command.params
        if param.name.startswith("ssh_")
    }
    return {**switch_defaults, **given}


def _strings(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _staged_path(value: str) -> Path:
    """A path that has to be uploaded, so it has to exist here.

    Typer would reject a missing one before the command body runs; without this
    the delegate silently dropped it and fell back to ``sflow.yaml``.
    """
    path = Path(value).expanduser()
    if not path.exists():
        raise typer.BadParameter(f"File {str(path)!r} does not exist.")
    return path


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_path(value: object) -> Path | None:
    return Path(str(value)).expanduser() if value else None


def argv_to_invocation(argv: list[str]) -> Invocation | None:
    """Parse ``argv`` (without ``sys.argv[0]``), or None if it is not delegated."""
    if not argv or argv[0] not in _DELEGATED_COMMANDS:
        return None
    command = argv[0]
    args = argv[1:]
    options = parse_command_options(command, args)
    if options is None or not options.get("ssh"):
        return None

    given = [
        *_strings(options.get("file")),
        *_strings(options.get("src_files")),
    ]
    recipes = [_staged_path(value) for value in given]
    config_files = list(
        dict.fromkeys(
            path for path in recipes if path.suffix.lower() in _CONFIG_SUFFIXES
        )
    )
    if not recipes and command == "run":
        config_files = [Path("sflow.yaml").resolve()]

    bulk_input = _optional_path(options.get("bulk_input"))
    input_files = list(
        dict.fromkeys(
            _staged_path(value)
            for value in [
                *_strings(options.get("file")),
                *_strings(options.get("bulk_submit")),
                *_strings(options.get("sflow_source_path")),
                *_strings(options.get("bulk_input")),
                *_strings(options.get("src_files")),
            ]
        )
    )

    workspace = _optional_path(options.get("workspace_dir")) or Path.cwd()
    compose_output = (
        _optional_path(options.get("output")) if command == "compose" else None
    )
    return Invocation(
        command=command,
        connection=str(options["ssh"]),
        raw_args=list(args),
        follow=str(options["ssh_follow"]),
        fetch=str(options["ssh_fetch"]),
        remote_root=_optional_string(options.get("ssh_remote_root")),
        tty=str(options["ssh_tty"]),
        workspace_dir=workspace,
        output_dir=_optional_path(options.get("output_dir")),
        input_files=input_files,
        config_files=config_files,
        variable_overrides=_strings(options.get("set_var")),
        artifact_overrides=_strings(options.get("artifact")),
        missable_tasks=_strings(options.get("missable_tasks")),
        skip_artifact_check=bool(options.get("skip_artifact_check", False)),
        log_level=str(options.get("log_level", "INFO")),
        bulk_input=bulk_input,
        compose_output=compose_output,
        compose_bulk=command == "compose" and bulk_input is not None,
        batch_sbatch_path=_optional_path(options.get("sbatch_path"))
        if command == "batch"
        else None,
        batch_runtime_explicit=any(
            options.get(name)
            for name in ("sflow_source_path", "sflow_version", "sflow_index_url")
        ),
    )
