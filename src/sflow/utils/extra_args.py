# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared de-dup/merge for backend ``extra_args`` (salloc/sbatch/docker flags).

A single option-key merge is reused everywhere CLI-provided extra args are
combined with a recipe's backend ``extra_args``:

* ``sflow run --extra-salloc-args`` (via ``BackendConfig.merge_extra_args``),
* ``sflow batch`` ``#SBATCH`` directives + the per-backend ``salloc`` merge.

De-duping by option (not exact string) means a CLI ``--gres=gpu:4`` overrides a
recipe ``--gres=gpu:8`` instead of both being passed. The de-dup is "recursive"
(keys on everything except the final value), so repeatable ``key=value`` flags
such as ``--env=FOO=1`` / ``--env=BAR=2`` are kept as distinct entries rather
than collapsed.
"""

from __future__ import annotations


def extra_arg_key(arg: str) -> str:
    """De-dup identity of a flag arg: everything except its final value.

    De-dup is "recursive" rather than just the top-level flag name, so repeatable
    ``key=value`` flags survive while single-valued flags still collapse:

    * ``--exclusive`` -> ``--exclusive`` (bare flag).
    * ``--network=host`` / ``--network=bridge`` -> ``--network`` (single-valued; a
      later value overrides the earlier one).
    * ``--gres=gpu:8`` / ``--gres=gpu:4`` -> ``--gres`` (value has no ``=``).
    * ``--env=FOO=1`` -> ``--env=FOO`` and ``--env=BAR=2`` -> ``--env=BAR`` (distinct
      keys coexist); ``--env=FOO=9`` -> ``--env=FOO`` so it overrides ``--env=FOO=1``.

    Implementation: split on the LAST ``=`` (the value); an arg with no ``=`` keys
    on its first whitespace token (the flag itself).
    """
    s = arg.strip()
    if not s:
        return arg
    if "=" not in s:
        return s.split()[0]
    return s.rsplit("=", 1)[0]


def dedup_merge_extra_args(base: list[str], override: list[str]) -> list[str]:
    """Merge two extra-arg lists, de-duped by option name.

    De-dups by :func:`extra_arg_key`. ``override`` wins on a conflicting option
    (e.g. a CLI flag overriding a recipe backend default), and a later entry
    wins within either list. Order: ``base`` options first (keeping their
    position, value overridden by ``override`` on a collision), then
    override-only options.
    """
    merged: dict[str, str] = {}
    for arg in [*base, *override]:
        merged[extra_arg_key(arg)] = arg
    return list(merged.values())
