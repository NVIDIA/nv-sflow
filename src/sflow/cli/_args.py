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
