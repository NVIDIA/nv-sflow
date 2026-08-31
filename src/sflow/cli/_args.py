# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI argument helpers + option definitions (used by `run` and `batch`)."""

from __future__ import annotations

from typing import Annotated, List, Optional

import typer

# Shared `--enable-*-monitor` option definitions, so `sflow run` and `sflow batch`
# expose identical flags + help (single source of truth -- no drift). Used as the
# parameter annotations in both command signatures.
EnableWorkflowMonitorOption = Annotated[
    bool,
    typer.Option(
        "--enable-workflow-monitor",
        help="Enable a default workflow-level hardware monitor (all scopes, whole "
        "pool, for the full run) without editing the recipe. No-op if the recipe "
        "already defines workflow.monitor.",
    ),
]

EnableTaskMonitorOption = Annotated[
    Optional[List[str]],
    typer.Option(
        "--enable-task-monitor",
        help="Enable a default hardware monitor (all scopes) bound to the named "
        "task(s) without editing the recipe. Accepts comma-separated names "
        '(--enable-task-monitor a,b), a quoted whitespace-separated list '
        '(--enable-task-monitor "a b"), and/or repeated flags. No-op for tasks '
        "that already define a monitor.",
    ),
]

# Shared `--include-nodes` / `--exclude-nodes` definitions (single source of truth
# for `sflow run` and `sflow batch`). Backend-agnostic host lists that each backend
# translates to its native node selection: Slurm `--nodelist`/`--exclude`, K8s
# `kubernetes.io/hostname` In/NotIn nodeAffinity, Docker host-pool filtering.
IncludeNodesOption = Annotated[
    Optional[List[str]],
    typer.Option(
        "--include-nodes",
        help="Restrict the candidate node pool to these hostnames across all "
        "backends. Accepts comma-separated (--include-nodes a,b), a quoted "
        'whitespace-separated list (--include-nodes "a b"), and/or repeated flags.',
    ),
]

ExcludeNodesOption = Annotated[
    Optional[List[str]],
    typer.Option(
        "--exclude-nodes",
        help="Remove these hostnames from the candidate node pool across all "
        "backends. Accepts comma-separated (--exclude-nodes a,b), a quoted "
        'whitespace-separated list (--exclude-nodes "a b"), and/or repeated flags.',
    ),
]

SshOption = Annotated[
    Optional[str],
    typer.Option(
        "--ssh",
        help="Run this command on an SSH host. The value may include SSH options, "
        "for example 'user@login -p 22 -i ~/.ssh/id_ed25519'.",
    ),
]

SshFollowOption = Annotated[
    str,
    typer.Option(
        "--ssh-follow",
        help="Remote progress display: auto (default), logs (5-second tail), "
        "status, or none. auto keeps the remote process as the only writer on "
        "the terminal when it has a PTY (so --tui renders cleanly) and falls "
        "back to the log tail when it does not, because a PTY-less remote run "
        "offloads task logs to files.",
    ),
]

SshFetchOption = Annotated[
    str,
    typer.Option(
        "--ssh-fetch",
        help="Files copied back after completion: logs, all, or none.",
    ),
]

SshTtyOption = Annotated[
    str,
    typer.Option(
        "--ssh-tty",
        help="Remote PTY allocation, which keeps sflow's terminal formatting "
        "and colors: auto (only when the local terminal is interactive), "
        "always, or never. Applies to the remote run only; the payload upload "
        "and output fetch always stay byte-clean.",
    ),
]

SshRemoteRootOption = Annotated[
    Optional[str],
    typer.Option(
        "--ssh-remote-root",
        help="Remote cache/run root. Default: $XDG_CACHE_HOME/sflow/ssh or ~/.cache/sflow/ssh.",
    ),
]


def parse_key_value_args(values: list[str] | None, *, flag: str) -> dict[str, str]:
    """Parse repeatable ``KEY=VALUE`` CLI options into a dict (later keys win).

    Accepts comma- and/or whitespace-separated tokens within each entry and across
    repeated flags (same normalization as :func:`split_list_arg`), so all of these
    yield ``{"a": "1", "b": "2"}``:
        --opt a=1,b=2
        --opt "a=1 b=2"
        --opt a=1 --opt b=2
    Raises ``typer.BadParameter`` on a token without ``=`` or with an empty key.
    Returns ``{}`` when falsy (None / empty).
    """
    result: dict[str, str] = {}
    for token in split_list_arg(values) or []:
        if "=" not in token:
            raise typer.BadParameter(f"{flag} expects KEY=VALUE, got: {token!r}")
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"{flag} has an empty key: {token!r}")
        result[key] = value.strip()
    return result


def split_list_arg(values: list[str] | None) -> list[str] | None:
    """Normalize a repeatable CLI list option into individual tokens.

    Accepts comma- and/or whitespace-separated values within each entry and across
    repeated flags, e.g. all of these yield ``["a", "b", "c"]``:
        --opt a,b,c
        --opt "a b c"
        --opt a --opt b --opt c
        --opt a,b --opt c
    Order is preserved and duplicates are dropped. Returns the input unchanged when
    falsy (None / empty) so "not provided" stays distinguishable.
    """
    if not values:
        return values
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in value.replace(",", " ").split():
            if token not in seen:
                seen.add(token)
                out.append(token)
    return out
