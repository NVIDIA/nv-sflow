# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from sflow.config.schema import BackendConfig, Resolvable
from sflow.core.backend import Allocation, Backend
from sflow.core.backend_registry import register_backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig


class LocalBackendConfig(BackendConfig):
    type: Literal["local"] = "local"
    nodes: Resolvable[int] = 1


@register_backend("local", LocalBackendConfig)
class LocalBackend(Backend):
    """
    Local backend implementation.

    This backend does not allocate external resources. It returns a synthetic allocation
    representing the local machine.
    """

    def __init__(self, config: LocalBackendConfig):
        super().__init__(name=config.name)
        self.config = config
        self._nodes = int(config.nodes) if config.nodes is not None else 1
        self._gpu_per_node = (
            int(config.gpus_per_node) if config.gpus_per_node is not None else None
        )

    async def allocate(self) -> Allocation:
        count = max(int(self._nodes), 1)
        nodes = [
            ComputeNode(
                name="localhost" if i == 0 else f"localhost-{i}",
                ip_address="127.0.0.1",
                index=i,
                num_gpus=self._gpu_per_node,
            )
            for i in range(count)
        ]
        # Synthetic allocation, not owned.
        return Allocation(allocation_id="local", nodes=nodes, owned=False)

    async def release(self, allocation: Allocation) -> None:
        # Nothing to release for local execution.
        return

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        # Local execution defaults to bash operator.
        return BashOperator(BashOperatorConfig(name=name))

    @classmethod
    def resolve_config(
        cls,
        conf: LocalBackendConfig,
        *,
        resolver: Any,
        ctx: dict[str, Any],
        workflow_name: str,
    ) -> LocalBackendConfig:
        nodes = resolver.resolve(conf.nodes, ctx) if conf.nodes is not None else 1
        try:
            nodes_i = int(nodes)
        except Exception as e:
            raise ValueError(
                f"Backend '{conf.name}' nodes must resolve to int, got {nodes!r}"
            ) from e
        nodes_i = max(nodes_i, 1)

        gpus_per_node = None
        if conf.gpus_per_node is not None:
            resolved = resolver.resolve(conf.gpus_per_node, ctx)
            try:
                gpus_per_node = int(resolved)
            except Exception as e:
                raise ValueError(
                    f"Backend '{conf.name}' gpus_per_node must resolve to int, got {resolved!r}"
                ) from e
            if gpus_per_node <= 0:
                raise ValueError(
                    f"Backend '{conf.name}' gpus_per_node must be > 0, got {gpus_per_node}"
                )

        return LocalBackendConfig(
            name=conf.name,
            type="local",
            default=bool(getattr(conf, "default", False)),
            nodes=nodes_i,
            gpus_per_node=gpus_per_node,
        )
