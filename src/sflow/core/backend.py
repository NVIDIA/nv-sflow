# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from sflow.logging import get_logger

from .compute_node import ComputeNode
from .operator import Operator

if TYPE_CHECKING:
    from .probe_transport import ProbeTransport

_logger = get_logger(__name__)


def configure_bare_monitor_operator(
    operator: Operator,
    *,
    backend: "Backend",
    assigned_nodes: Sequence[str] | None,
) -> Operator:
    """Configure an operator for bare-node host monitoring and apply backend context.

    A monitor collector must run as a single, overlapping host process with no GPU
    reservation, streaming its (tiny) startup line rather than offloading to a
    per-step output file. Centralizing these knobs here keeps the monitor planner
    out of operator-config internals: every backend's :meth:`Backend.monitor_operator`
    funnels through this one place, and the only backend-specific bits that differ
    are which attributes the operator's config actually exposes (guarded below).
    """
    cfg = operator.config
    if hasattr(cfg, "log_to_file"):
        cfg.log_to_file = False
    if hasattr(cfg, "ntasks_per_node"):
        cfg.ntasks_per_node = 1
    if hasattr(cfg, "ntasks"):
        cfg.ntasks = None
    if hasattr(cfg, "overlap"):
        cfg.overlap = True
    operator.apply_backend_context(
        backend=backend,
        assigned_nodes=list(assigned_nodes or []),
        artifacts=[],
        cuda_visible_devices=None,
        gpu_count=None,
    )
    return operator


@dataclass
class Allocation:
    allocation_id: str | None
    nodes: list[ComputeNode]
    # Whether this allocation is owned by sflow and should be released on exit.
    # Example: if sflow reuses an externally managed allocation, it must not
    # release that allocation on exit.
    owned: bool = True


@dataclass(frozen=True)
class BackendCapabilities:
    """Backend behavior that core planning can depend on without type checks."""

    supports_node_placement: bool = True
    supports_gpu_env: bool = True
    supports_host_path_mounts: bool = True
    has_runtime_node_addresses: bool = True
    # Whether the backend can hand a still-running task's GPUs to another task
    # (Slurm fakes this via CUDA_VISIBLE_DEVICES). Kubernetes (DRA/device-plugin)
    # hard-enforces one pod per physical GPU, so it sets this False and the
    # planner coerces GPU ``release_after: task_ready`` -> ``task_completion``.
    supports_gpu_sharing: bool = True
    # Whether sflow can run the bare-host hardware monitor (hardware_monitor.py)
    # directly on the backend's reserved nodes. Slurm/local/Docker run on the
    # physical host, so True. Kubernetes has no node-level collector mechanism yet
    # (a privileged DCGM/nvidia-smi DaemonSet would be the proper path) -- running
    # it on the sflow driver host would only sample the driver, not the reserved
    # GPU nodes -- so it sets this False and the monitor planner skips it.
    supports_host_monitoring: bool = True


class Backend(ABC):
    """
    Abstract base class for compute resource providers.
    """

    def __init__(self, name: str):
        self.name = name
        self.allocation: Allocation | None = None
        self.capabilities = BackendCapabilities()

    def preflight_validate(self) -> None:
        """Validate backend-specific prerequisites before allocation."""
        return None

    def planning_capacity(self) -> tuple[int, int | None]:
        """Return node count and per-node GPU capacity for placeholder planning."""
        return (1, None)

    def placeholder_allocation(self) -> Allocation:
        """Return deterministic planning nodes when real allocation is not available."""
        nodes_count, num_gpus = self.planning_capacity()
        nodes_count = max(nodes_count, 1)
        has_addresses = self.capabilities.has_runtime_node_addresses

        nodes: list[ComputeNode] = []
        for i in range(nodes_count):
            name = f"{self.name}-node{i}"
            ip = f"0.0.0.{i + 1}" if has_addresses else ""
            nodes.append(
                ComputeNode(name=name, ip_address=ip, index=i, num_gpus=num_gpus)
            )
        return Allocation(allocation_id="0", nodes=nodes, owned=False)

    def resource_env(self, *, cuda_visible_devices: str | None = None) -> dict[str, str]:
        """Return backend-specific resource environment variables for a task."""
        if cuda_visible_devices is None or not self.capabilities.supports_gpu_env:
            return {}
        return {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}

    def dry_run_details(self) -> list[tuple[str, str]]:
        """Return backend-specific config details for dry-run summaries."""
        return []

    def probe_transport(self) -> "ProbeTransport | None":
        """Transport used to run this backend's network probes, or None.

        Returning None (the default) means TCP/HTTP probes run directly from the
        sflow driver host. Backends whose driver host may not reach the workload
        network (e.g. Kubernetes) override this to run the checks from inside the
        backend's network instead.
        """
        return None

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

    def emergency_release(self, allocation: Allocation) -> None:
        """Best-effort synchronous cleanup for atexit fallback paths."""
        return None

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

    def monitor_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        """Operator for bare-node hardware monitoring (runs on the host).

        Hardware monitors must observe the *physical* node, so the collector runs
        directly on the host -- never inside the workload's container, which would
        only see the container's (cgroup-limited) view and could not read the
        materialized collector script from the host filesystem.

        For host-level backends (Slurm ``srun``, local ``bash``) the default
        operator already executes on the host, so this builds it and applies the
        bare-monitor configuration. Container backends (e.g. Docker) override this
        to bypass their container and run on the host.
        """
        operator = self.default_operator(name=name, assigned_nodes=assigned_nodes)
        return configure_bare_monitor_operator(
            operator, backend=self, assigned_nodes=assigned_nodes
        )

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """
        Converts the backend to a dictionary.
        """
        nodes: list[dict[str, Any]] = []
        if self.allocation:
            for node in self.allocation.nodes:
                node_dict = node.to_dict()
                if not node_dict.get("ip_address"):
                    node_dict.pop("ip_address", None)
                nodes.append(node_dict)
        return {
            "name": self.name,
            "nodes": nodes,
        }
