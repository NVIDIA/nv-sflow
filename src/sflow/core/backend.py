# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from sflow.logging import get_logger

from .compute_node import ComputeNode
from .operator import Operator

_logger = get_logger(__name__)


@dataclass
class Allocation:
    allocation_id: str | None
    nodes: list[ComputeNode]
    # Whether this allocation is owned by sflow and should be released on exit.
    # Example: if sflow reuses an existing Slurm allocation from env (SLURM_JOB_ID),
    # it must NOT scancel it on exit.
    owned: bool = True


class Backend(ABC):
    """
    Abstract base class for compute resource providers.
    """

    def __init__(self, name: str):
        self.name = name
        self.allocation: Allocation | None = None

    @abstractmethod
    async def allocate(self) -> Allocation:
        """
        Acquires resources

        Returns:
            AllocationInfo: Allocation information.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def release(self, allocation: Allocation) -> None:
        """
        Releases all resources.

        Args:
            allocation: Allocation information.
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def allocate_resources(self) -> None:
        """
        Allocates resources.
        """
        _logger.info(f"Allocating resources for backend {self.name}")
        self.allocation = await self.allocate()

    async def release_resources(self) -> None:
        """
        Releases resources.
        """
        _logger.info(f"Releasing resources for backend {self.name}")
        if not self.allocation:
            return
        if not getattr(self.allocation, "owned", True):
            _logger.info(
                f"Skipping release for backend {self.name} (allocation not owned by sflow)"
            )
            self.allocation = None
            return
        await self.release(self.allocation)
        self.allocation = None

    @abstractmethod
    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        """
        Construct the default Operator for this backend.

        This is backend-owned behavior (not user-configurable via YAML). The assembly layer passes
        in late-bound context such as assigned_nodes and backend-level extra_args.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """
        Converts the backend to a dictionary.
        """
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.allocation.nodes]
            if self.allocation
            else [],
        }
