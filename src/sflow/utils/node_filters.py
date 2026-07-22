# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for backend node include/exclude host lists.

The ``--include-nodes`` / ``--exclude-nodes`` CLI flags and the matching
``include_nodes`` / ``exclude_nodes`` backend config fields carry plain hostnames
that each backend translates to its own node-selection mechanism (Slurm
``--nodelist``/``--exclude``, Kubernetes ``nodeAffinity`` In/NotIn, Docker host
pool filtering).

* :func:`normalize_node_list` turns raw entries (which may be comma- or
  whitespace-joined) into an order-preserving deduped list of individual hosts.
  Backends call it *after* expression resolution, when every entry is concrete.
* :func:`merge_node_lists` unions two lists by exact string (order preserved,
  deduped) WITHOUT splitting -- used to merge CLI values over YAML entries that
  may still be unresolved ``${{ }}`` expressions.
* :func:`find_node_filter_overlap` reports hosts present in both lists.
* :func:`filter_by_node_names` keeps objects whose host name passes the
  include/exclude filter -- the shared core of each backend's pool filtering
  (Slurm reused-allocation nodes, Docker host pool); backends wrap it with their
  own empty-pool / all-removed messaging.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence, TypeVar

_T = TypeVar("_T")


def normalize_node_list(values: Iterable[str] | None) -> list[str]:
    """Split comma/whitespace-joined entries into a deduped, ordered host list."""
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in str(value).replace(",", " ").split():
            if token not in seen:
                seen.add(token)
                out.append(token)
    return out


def merge_node_lists(
    base: Sequence[str] | None, override: Sequence[str] | None
) -> list[str]:
    """Union two host lists by exact string, order-preserving and deduped.

    Does NOT split entries, so unresolved ``${{ }}`` expressions survive intact.
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in [*(base or []), *(override or [])]:
        s = str(value)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def find_node_filter_overlap(
    include: Iterable[str] | None, exclude: Iterable[str] | None
) -> list[str]:
    """Return the sorted hosts present in both lists (after normalization)."""
    inc = set(normalize_node_list(include))
    exc = set(normalize_node_list(exclude))
    return sorted(inc & exc)


def filter_by_node_names(
    items: Sequence[_T],
    include: Iterable[str] | None,
    exclude: Iterable[str] | None,
    *,
    key: Callable[[_T], str] = lambda item: item.name,  # type: ignore[attr-defined]
) -> list[_T]:
    """Keep ``items`` whose host name passes the include/exclude filter.

    An item is kept when ``key(item)`` is in ``include`` (only enforced when
    ``include`` is non-empty) AND not in ``exclude``. Order-preserving.
    ``include``/``exclude`` are normalized (comma/whitespace split, deduped) so
    callers may pass raw or already-normalized lists. Returns a shallow copy when
    both filters are empty. Backends wrap this with their own empty-pool /
    all-removed warnings + errors, which differ per backend.
    """
    inc = set(normalize_node_list(include))
    exc = set(normalize_node_list(exclude))
    if not inc and not exc:
        return list(items)
    return [
        item
        for item in items
        if (not inc or key(item) in inc) and key(item) not in exc
    ]


def resolve_node_filters(resolver: Any, conf: Any, ctx: Any) -> tuple[list | None, list | None]:
    """Resolve ``conf.include_nodes`` / ``conf.exclude_nodes`` expressions to strings.

    Returns ``(include, exclude)`` where each is a list of resolved strings (kept
    un-split so a resolved ``"a,b"`` is normalized later by the backend) or None
    when the field is unset. Used by each backend's ``resolve_config`` so the
    shared fields survive the field-by-field config rebuild.
    """

    def _resolve(values: Sequence[Any] | None) -> list | None:
        if not values:
            return None
        return [str(resolver.resolve(v, ctx)) for v in values]

    return _resolve(getattr(conf, "include_nodes", None)), _resolve(
        getattr(conf, "exclude_nodes", None)
    )
