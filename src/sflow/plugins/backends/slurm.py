# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from typing import Any, Literal

from sflow.config.schema import BackendConfig, Resolvable
from sflow.core.backend import Allocation, Backend
from sflow.core.backend_registry import register_backend
from sflow.core.command import Command
from sflow.core.compute_node import ComputeNode
from sflow.core.launcher import SubprocessLauncher
from sflow.core.operator import Operator
from sflow.logging import get_logger
from sflow.plugins.operators.srun import SrunOperator, SrunOperatorConfig
from sflow.utils.logging import temporary_handler
from sflow.utils.parser import ParseLogHandler

_logger = get_logger(__name__)


class SlurmBackendConfig(BackendConfig):
    type: Literal["slurm"] = "slurm"
    account: Resolvable[str]
    partition: Resolvable[str]
    time: Resolvable[str]
    nodes: Resolvable[int]
    # Slurm backend requires GPUs-per-node to be specified so we can correctly
    # populate ComputeNode.num_gpus for scheduling/packing/validation.
    gpus_per_node: Resolvable[int]
    extra_args: list[Resolvable[str]] | None = None
    job_name: str | None = None


@register_backend("slurm", SlurmBackendConfig)
class SlurmBackend(Backend):
    """
    Slurm implementation of the backend interface.
    """

    def __init__(self, config: SlurmBackendConfig):
        super().__init__(name=config.name)
        self.config = config
        self._account = str(config.account)
        self._partition = str(config.partition)
        self._nodes = int(config.nodes)
        self._time = str(config.time)
        self._job_name = str(config.job_name or config.name)
        self._extra_args = [str(a) for a in (config.extra_args or [])]
        try:
            self._gpu_per_node = int(config.gpus_per_node)
        except Exception as e:
            raise ValueError(
                f"Backend '{config.name}' gpus_per_node must be an int after resolution, got {config.gpus_per_node!r}"
            ) from e
        self._subprocess_launcher = SubprocessLauncher()

    async def _resolve_nodes_via_scontrol(self, *, nodelist: str) -> list[ComputeNode] | None:
        """
        Try to resolve hostnames + IPs using `scontrol getaddrs`.
        Returns None if scontrol getaddrs is not available or fails.
        """
        parser = ParseLogHandler(patterns=["{hostname}: {ip_address}:{port}"])
        with temporary_handler(_logger, parser):
            exit_code = await self._subprocess_launcher.run_async(
                ["scontrol", "getaddrs", nodelist], output_logger=_logger
            )
            if exit_code != 0:
                _logger.warning(
                    f"scontrol getaddrs failed with exit code {exit_code}, "
                    "will try fallback method using srun"
                )
                return None

        parsed_result = parser.get_parsed_dict()
        hostnames = parsed_result.get("hostname")
        ip_addresses = parsed_result.get("ip_address")
        if not hostnames or not ip_addresses:
            _logger.warning(
                f"Failed to parse scontrol getaddrs output for nodelist={nodelist!r}, "
                "will try fallback method using srun"
            )
            return None

        if isinstance(hostnames, str):
            hostnames = [hostnames]
        if isinstance(ip_addresses, str):
            ip_addresses = [ip_addresses]

        nodes: list[ComputeNode] = []
        for index, (hostname, ip_address) in enumerate(
            zip(hostnames, ip_addresses, strict=False)
        ):
            nodes.append(
                ComputeNode(
                    name=hostname,
                    ip_address=ip_address,
                    index=index,
                    num_gpus=self._gpu_per_node,
                )
            )
        return nodes

    async def _resolve_nodes_via_srun(
        self, *, nodelist: str, job_id: str | None = None
    ) -> list[ComputeNode]:
        """
        Fallback method to resolve node hostnames and IPs using srun.
        Runs `srun --nodelist=<nodelist> --ntasks-per-node=1 bash -c 'echo $(hostname):$(hostname -i)'`
        and parses the output.
        """
        # Build srun command
        cmd: list[str] = ["srun", "--nodelist", nodelist, "--ntasks-per-node=1"]

        # Add job_id if available (for existing allocations)
        if job_id:
            cmd.extend(["--jobid", job_id])

        # Add --overlap to allow running within existing srun jobs
        cmd.append("--overlap")

        # The command to run on each node
        cmd.extend(["bash", "-c", "echo $(hostname):$(hostname -i)"])

        _logger.info(f"Resolving node addresses via srun: {' '.join(cmd)}")

        # Capture output
        output_lines: list[str] = []

        class OutputCaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord):
                output_lines.append(record.getMessage())

        capture_handler = OutputCaptureHandler()
        with temporary_handler(_logger, capture_handler):
            exit_code = await self._subprocess_launcher.run_async(
                cmd, output_logger=_logger
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"Failed to resolve node addresses via srun (exit code {exit_code}). "
                    f"Output: {output_lines}"
                )

        # Parse output lines in format "hostname:ip_address"
        nodes: list[ComputeNode] = []
        seen_hostnames: set[str] = set()

        for line in output_lines:
            line = line.strip()
            if not line or ":" not in line:
                continue
            # Handle potential srun prefix like "node1: " in output
            # Format could be "hostname:ip" or "srun: hostname:ip"
            parts = line.split(":")
            if len(parts) >= 2:
                # Take the last two parts as hostname:ip
                hostname = parts[-2].strip()
                ip_address = parts[-1].strip()
                # Skip if it looks like a srun message
                if hostname.startswith("srun") or not ip_address:
                    continue
                # Extract short hostname from FQDN (e.g., "node01.cluster.example.com" -> "node01")
                if "." in hostname:
                    hostname = hostname.split(".")[0]
                # Skip duplicate hostnames (srun might output multiple times)
                if hostname in seen_hostnames:
                    continue
                seen_hostnames.add(hostname)
                nodes.append(
                    ComputeNode(
                        name=hostname,
                        ip_address=ip_address,
                        index=len(nodes),
                        num_gpus=self._gpu_per_node,
                    )
                )

        if not nodes:
            raise RuntimeError(
                f"Failed to parse any node addresses from srun output for nodelist={nodelist!r}. "
                f"Output lines: {output_lines}"
            )

        _logger.info(
            f"Resolved {len(nodes)} nodes via srun: "
            f"{[(n.name, n.ip_address) for n in nodes]}"
        )
        return nodes

    async def _resolve_nodes_from_nodelist(
        self, *, nodelist: str, job_id: str | None = None
    ) -> list[ComputeNode]:
        """
        Resolve hostnames + IPs for a Slurm nodelist.

        First tries `scontrol getaddrs`, falls back to `srun hostname -i` if that fails.
        Returns a list of ComputeNode.
        """
        # Try scontrol getaddrs first (faster, no job overhead)
        nodes = await self._resolve_nodes_via_scontrol(nodelist=nodelist)
        if nodes:
            return nodes

        # Fallback to srun method
        _logger.info(
            "Falling back to srun method for resolving node addresses"
        )
        return await self._resolve_nodes_via_srun(nodelist=nodelist, job_id=job_id)

    async def allocate(self) -> Allocation:
        import os

        # If we're already inside a Slurm allocation, reuse it instead of creating a new one.
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
        nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get(
            "SLURM_NODELIST"
        )
        if job_id and nodelist:
            _logger.info(
                "Detected existing Slurm allocation in env; skipping salloc "
                f"(SLURM_JOB_ID={job_id!r}, SLURM_JOB_NODELIST={nodelist!r})"
            )

            nodes = await self._resolve_nodes_from_nodelist(nodelist=nodelist, job_id=job_id)
            if self._nodes and len(nodes) != self._nodes:
                _logger.warning(
                    "Slurm allocation node count differs from backend config (continuing): "
                    f"config_nodes={self._nodes} env_nodes={len(nodes)}"
                )

            # Important: we do NOT own this allocation; do not scancel on exit.
            return Allocation(allocation_id=str(job_id), nodes=nodes, owned=False)

        command = (
            Command(exec="salloc")
            .add_opt("--account", self._account)
            .add_opt("--partition", self._partition)
            .add_opt("--nodes", self._nodes)
            .add_opt("--time", self._time)
            .add_opt("--job-name", self._job_name)
            .add_opt("--no-shell")
        )
        for arg in self._extra_args:
            command.add_opt(arg)

        parser = ParseLogHandler(
            patterns=[
                "salloc: Granted job allocation {job_id}",
                "salloc: Nodes {nodelist} are ready for job",
            ]
        )
        # Capture all output lines for error reporting
        output_lines: list[str] = []

        class OutputCaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord):
                output_lines.append(record.getMessage())

        capture_handler = OutputCaptureHandler()
        with temporary_handler(_logger, parser), temporary_handler(
            _logger, capture_handler
        ):
            exit_code = await self._subprocess_launcher.run_async(
                command, output_logger=_logger
            )
            if exit_code != 0:
                output_log = "\n".join(output_lines) if output_lines else "(no output)"
                _logger.error(
                    f"salloc failed with exit code {exit_code}. Output:\n{output_log}"
                )
                raise RuntimeError(
                    f"Failed to allocate nodes (exit code {exit_code}). "
                    f"salloc output:\n{output_log}\n"
                    f"Please check if you are using the correct Slurm account / partition / nodes in the yaml file, or --gpus-per-node=N is correctly set/removed according to your cluster's documentation"
                )

        parsed_result = parser.get_parsed_dict()

        allocation_id = parsed_result["job_id"]
        slurm_nodelist = parsed_result["nodelist"]

        nodes = await self._resolve_nodes_from_nodelist(
            nodelist=slurm_nodelist, job_id=allocation_id
        )
        return Allocation(
            allocation_id=allocation_id,
            nodes=nodes,
            owned=True,
        )

    async def release(self, allocation: Allocation) -> None:
        await self._subprocess_launcher.run_async(
            ["scancel", allocation.allocation_id], output_logger=_logger
        )

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        # Slurm execution defaults to srun operator. If allocated, surface job_id/nodelist.
        job_id = "0"
        nodelist: list[str] = []
        if self.allocation and self.allocation.allocation_id:
            job_id = str(self.allocation.allocation_id)
            nodelist = [n.name for n in (self.allocation.nodes or [])]

        # Prefer assembly-assigned nodes if present.
        if assigned_nodes:
            nodelist = list(assigned_nodes)

        cfg = SrunOperatorConfig(
            name=name,
            job_id=job_id,
            nodelist=nodelist,
        )
        return SrunOperator(cfg)

    @classmethod
    def resolve_config(
        cls,
        conf: SlurmBackendConfig,
        *,
        resolver: Any,
        ctx: dict[str, Any],
        workflow_name: str,
    ) -> SlurmBackendConfig:
        account = resolver.resolve(conf.account, ctx)
        partition = resolver.resolve(conf.partition, ctx)
        time = resolver.resolve(conf.time, ctx)
        nodes = resolver.resolve(conf.nodes, ctx)

        try:
            nodes_i = int(nodes)
        except Exception as e:
            raise ValueError(
                f"Backend '{conf.name}' nodes must resolve to int, got {nodes!r}"
            ) from e

        extra_args_raw = conf.extra_args or []
        extra_args = [str(resolver.resolve(arg, ctx)) for arg in extra_args_raw]

        if conf.gpus_per_node is None:
            # Defensive (schema should enforce), but keeps the error clear if called directly.
            raise ValueError(
                f"Backend '{conf.name}' requires 'gpus_per_node' to be set for slurm backends"
            )
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

        return SlurmBackendConfig(
            name=conf.name,
            type="slurm",
            default=bool(getattr(conf, "default", False)),
            account=str(account),
            partition=str(partition),
            time=str(time),
            nodes=nodes_i,
            extra_args=extra_args,
            job_name=str(workflow_name),
            gpus_per_node=gpus_per_node,
        )
