# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pull each container image once per node, before any task starts.

Backend-agnostic: this module decides *which* image goes to *which* nodes and
owns the retry/failure policy. A backend only knows how to pull one image onto
some of its nodes (:meth:`sflow.core.backend.Backend.prefetch_image`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sflow.logging import get_logger
from sflow.utils.container import is_local_container_image

if TYPE_CHECKING:
    from sflow.core.backend import Backend

_logger = get_logger(__name__)

# Extra attempts after a failed pull; only the nodes that failed are retried.
# Layers that did download stay in the node's cache, so a retry is cheap.
PREFETCH_RETRIES = 2
PREFETCH_RETRY_DELAY_S = 10.0


@dataclass
class PrefetchResult:
    """What one pull attempt left undone: the nodes that failed, and why."""

    failed_nodes: list[str] = field(default_factory=list)
    output: str = ""

    @classmethod
    def from_node_errors(
        cls, errors: Sequence[tuple[str, str | None]]
    ) -> PrefetchResult:
        """Build from per-node ``(node, error-or-None)`` pairs, in node order."""
        failed = [(node, error) for node, error in errors if error]
        return cls(
            failed_nodes=[node for node, _ in failed],
            output="\n".join(f"[{node}] {error}" for node, error in failed),
        )


class ImagePrefetchError(RuntimeError):
    """A container image could not be pulled onto a node after every retry."""


@dataclass
class ImagePrefetch:
    """One image, the nodes that will run it, and the tasks that use it."""

    image: str
    nodes: list[str]
    tasks: list[Any] = field(default_factory=list)


def operator_config(task: Any) -> Any:
    """The task's resolved operator config, or None."""
    return getattr(getattr(task, "operator", None), "config", None)


def plan_image_prefetch(
    tasks: Sequence[Any], allocation_nodes: Sequence[str]
) -> list[ImagePrefetch]:
    """Distinct container images of *tasks*, each with the nodes that run it.

    Images are in order of first use; nodes follow the allocation order. A task
    with no assigned nodes may run anywhere, so it claims every allocation node.
    Local images (``.sqsh`` files, paths) and tasks that run no container are
    skipped.
    """
    rank = {name: index for index, name in enumerate(allocation_nodes)}
    by_image: dict[str, tuple[set[str], list[Any]]] = {}
    for task in tasks:
        container_images = getattr(operator_config(task), "container_images", None)
        if not callable(container_images):
            continue
        nodes = list(getattr(task, "assigned_nodes", None) or allocation_nodes)
        for raw in container_images():
            image = str(raw).strip()
            if not image or is_local_container_image(image):
                continue
            node_set, users = by_image.setdefault(image, (set(), []))
            node_set.update(nodes)
            if task not in users:
                users.append(task)
    return [
        ImagePrefetch(
            image=image,
            nodes=sorted(node_set, key=lambda n: (rank.get(n, len(rank)), n)),
            tasks=users,
        )
        for image, (node_set, users) in by_image.items()
    ]


def supports_image_prefetch(backend: Backend) -> bool:
    """Whether *backend* implements :meth:`Backend.prefetch_image`."""
    from sflow.core.backend import Backend

    return type(backend).prefetch_image is not Backend.prefetch_image


async def prefetch_image_with_retries(backend: Backend, entry: ImagePrefetch) -> None:
    """Pull ``entry.image`` onto ``entry.nodes``, retrying only the nodes that failed.

    Raises :class:`ImagePrefetchError` naming the backend, image, nodes and the
    runtime's output once the last attempt fails.
    """
    attempts = 1 + PREFETCH_RETRIES
    pending = list(entry.nodes)
    for attempt in range(1, attempts + 1):
        result = await backend.prefetch_image(entry.image, pending, entry.tasks)
        reported = set(result.failed_nodes)
        # A failure naming no node we know of still failed somewhere: retry all.
        failed = [n for n in pending if n in reported] or (
            list(pending) if reported else []
        )
        if not failed:
            _logger.info(
                f"Backend '{backend.name}': pulled {entry.image!r} on "
                f"{','.join(pending)} (attempt {attempt}/{attempts})."
            )
            return
        output = result.output.strip() or "(no output)"
        if attempt < attempts:
            _logger.warning(
                f"Backend '{backend.name}': pulling {entry.image!r} failed on "
                f"{','.join(failed)} (attempt {attempt}/{attempts}); retrying in "
                f"{PREFETCH_RETRY_DELAY_S:g}s. Output:\n{output}"
            )
            await asyncio.sleep(PREFETCH_RETRY_DELAY_S)
            pending = failed
            continue
        message = (
            f"Backend '{backend.name}': could not pull container image "
            f"{entry.image!r} on node(s) {','.join(failed)} after {attempts} "
            f"attempt(s); no task was started. Output:\n{output}"
        )
        _logger.error(message)
        raise ImagePrefetchError(message)


async def prefetch_backend_images(
    backend: Backend, plan: Sequence[ImagePrefetch]
) -> None:
    """Carry out *plan* on *backend*, one image at a time (each on all its nodes)."""
    if not plan:
        return
    if not supports_image_prefetch(backend):
        _logger.info(
            f"Backend '{backend.name}': prefetch_images is on, but this backend "
            "type cannot pull images ahead of time; skipping."
        )
        return
    _logger.info(
        f"Backend '{backend.name}': pulling {len(plan)} container image(s) onto "
        "the nodes that run them, before any task starts."
    )
    for entry in plan:
        await prefetch_image_with_retries(backend, entry)
