# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OSMO backend: each task runs as its own OSMO workflow (see the ``osmo`` operator).

There is no allocation to hold: OSMO schedules every task on submit, so the
backend only carries the pool/platform and checks the ``osmo`` CLI is usable.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from typing import Any, Literal

from sflow.config.schema import BackendConfig, Resolvable
from sflow.core.backend import Allocation, Backend, BackendCapabilities
from sflow.core.backend_registry import register_backend
from sflow.core.operator import Operator


class OsmoBackendConfig(BackendConfig):
    """Configuration for the ``osmo`` backend."""

    type: Literal["osmo"] = "osmo"
    # OSMO pool to submit to; None uses the pool set in the user's OSMO profile.
    pool: Resolvable[str] | None = None
    # Default OSMO platform (node type) for tasks, e.g. ``ovx-l40``.
    platform: Resolvable[str] | None = None

    def executes_offhost(self) -> bool:
        """Tasks run on OSMO, not on the controller."""
        return True


@register_backend("osmo", OsmoBackendConfig)
class OsmoBackend(Backend):
    """Submit each task as a single-task OSMO workflow through the ``osmo`` CLI."""

    # Like Kubernetes: a failed command in the container should fail the task.
    default_fail_fast: bool = True

    def __init__(self, config: OsmoBackendConfig):
        """Build the backend; OSMO places tasks itself, so no node placement or GPU env."""
        super().__init__(name=config.name)
        self.config = config
        self.capabilities = BackendCapabilities(
            supports_node_placement=False,
            supports_gpu_env=False,
            supports_host_path_mounts=False,
            has_runtime_node_addresses=False,
            supports_gpu_sharing=False,
            supports_host_monitoring=False,
        )

    def preflight_validate(self, *, artifacts: Sequence[Any] = ()) -> None:
        """Fail fast when the ``osmo`` CLI is missing or not logged in."""
        if shutil.which("osmo") is None:
            raise RuntimeError(
                f"OSMO backend '{self.name}': the 'osmo' CLI is not on PATH. Install it "
                "and run 'osmo login <url>' (CI: --method token --token-file <file>)."
            )
        try:
            probe = subprocess.run(
                ["osmo", "workflow", "list", "--count", "1", "-t", "json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"OSMO backend '{self.name}': could not run 'osmo workflow list' "
                f"({exc}). Check the 'osmo' CLI is executable and OSMO is reachable, "
                "then run 'osmo login <url>' (CI: --method token --token-file <file>)."
            ) from exc
        if probe.returncode != 0:
            raise RuntimeError(
                f"OSMO backend '{self.name}': 'osmo workflow list' failed, so the CLI "
                "is not logged in or cannot reach OSMO. Run 'osmo login <url>' "
                f"(CI: --method token --token-file <file>).\n{probe.stdout}{probe.stderr}"
            )

    async def allocate(self) -> Allocation:
        """Nothing to reserve: each task is scheduled when it is submitted."""
        return self.placeholder_allocation()

    async def release(self, allocation: Allocation) -> None:
        """Nothing to release: tasks end with their OSMO workflows."""
        return None

    def dry_run_details(self) -> list[tuple[str, str]]:
        """Show the pool and platform in dry-run output."""
        return [
            ("pool", str(self.config.pool or "<osmo profile default>")),
            ("platform", str(self.config.platform or "<pool default>")),
        ]

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        """Reject tasks without an explicit ``osmo`` operator (it carries the image)."""
        raise ValueError(
            f"OSMO backend '{self.name}': task has no operator. OSMO tasks need an "
            "explicit 'osmo' operator with an 'image:'."
        )

    @classmethod
    def resolve_config(
        cls,
        conf: OsmoBackendConfig,
        *,
        resolver: Any,
        ctx: dict[str, Any],
        workflow_name: str,
    ) -> OsmoBackendConfig:
        """Resolve ``${{ }}`` expressions in ``pool`` and ``platform``."""

        def _str(value: Any) -> str | None:
            """Resolve one optional field; null or empty means unset (OSMO default)."""
            resolved = None if value is None else resolver.resolve(value, ctx)
            return None if resolved in (None, "") else str(resolved)

        return conf.model_copy(
            update={"pool": _str(conf.pool), "platform": _str(conf.platform)}
        )
