# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from sflow.config.schema import BackendConfig, Resolvable
from sflow.core.backend import (
    Allocation,
    Backend,
    BackendCapabilities,
    configure_bare_monitor_operator,
)
from sflow.core.backend_registry import register_backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.logging import get_logger
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.plugins.operators.docker_run import (
    DockerRunOperator,
    DockerRunOperatorConfig,
)
from sflow.utils.extra_args import normalize_extra_args
from sflow.utils.node_filters import (
    filter_by_node_names,
    normalize_node_list,
    resolve_node_filters,
)

_logger = get_logger(__name__)


class DockerHostConfig(BaseModel):
    name: str
    docker_host: Resolvable[str] | None = None
    context: Resolvable[str] | None = None
    ip_address: Resolvable[str] | None = None
    gpus_per_node: Resolvable[int] | None = None
    mounts: list[Resolvable[str]] | None = None
    extra_args: list[Resolvable[str]] | None = None

    @model_validator(mode="after")
    def exactly_one_endpoint(self) -> "DockerHostConfig":
        if bool(self.docker_host) == bool(self.context):
            raise ValueError(
                "docker host config requires exactly one of 'docker_host' or 'context'"
            )
        return self


class DockerBackendConfig(BackendConfig):
    type: Literal["docker"] = "docker"
    image: Resolvable[str]
    nodes: Resolvable[int] = 1
    hosts: list[DockerHostConfig] | None = None
    mounts: list[Resolvable[str]] | None = None
    workdir: Resolvable[str] | None = None
    extra_args: list[Resolvable[str]] | None = None
    # Per-task log offload is ON by default (CLI flag / env override take
    # precedence; resolved in the docker_run operator). When enabled, the docker
    # output is redirected on the host through a compute-side prefixer instead of
    # streaming it through the sflow driver. Auto-falls back to streaming on an
    # interactive TTY / --tui session.
    offload_task_logs: bool = True

    def container_images(self) -> list[str]:
        return [str(self.image)] if self.image else []

    def planning_node_count(self) -> Resolvable[int] | None:
        if self.hosts:
            return len(self.hosts)
        return self.nodes


@register_backend("docker", DockerBackendConfig)
class DockerBackend(Backend):
    """Docker backend using local synthetic allocation and docker run launch."""

    def __init__(self, config: DockerBackendConfig):
        super().__init__(name=config.name)
        self.config = config
        self._include_nodes = normalize_node_list(config.include_nodes)
        self._exclude_nodes = normalize_node_list(config.exclude_nodes)
        self._hosts = self._filter_hosts(list(config.hosts or []))
        self._host_by_name = {host.name: host for host in self._hosts}
        self.capabilities = BackendCapabilities(
            supports_node_placement=True,
            supports_gpu_env=True,
            supports_host_path_mounts=True,
            has_runtime_node_addresses=bool(self._hosts)
            and any(bool(host.ip_address) for host in self._hosts),
        )
        self._image = str(config.image)
        self._nodes = len(self._hosts) if self._hosts else int(config.nodes or 1)
        self._gpu_per_node = (
            int(config.gpus_per_node) if config.gpus_per_node is not None else None
        )
        self._mounts = [str(m) for m in (config.mounts or [])]
        self._workdir = str(config.workdir) if config.workdir is not None else None
        # Shell-split each entry into clean argv tokens (shared with the Slurm
        # backend): a bundled/whitespace-laden entry can't survive as one
        # unparsable token for `docker run`. Never worse than verbatim passthrough.
        self._extra_args = normalize_extra_args(config.extra_args)

    def _filter_hosts(
        self, hosts: list[DockerHostConfig]
    ) -> list[DockerHostConfig]:
        """Restrict/steer the ``hosts:`` pool by include/exclude host names.

        No-op (with a warning) when there is no ``hosts:`` pool: containers then
        run on the local Docker daemon, where hostname filters are meaningless.
        """
        if not (self._include_nodes or self._exclude_nodes):
            return hosts
        if not hosts:
            _logger.warning(
                "Docker backend '%s': --include-nodes/--exclude-nodes have no effect "
                "without a 'hosts:' pool (containers run on the local daemon).",
                self.name,
            )
            return hosts
        filtered = filter_by_node_names(
            hosts, self._include_nodes, self._exclude_nodes
        )
        if not filtered:
            raise ValueError(
                f"Docker backend '{self.name}': include/exclude node filters removed "
                "all hosts from the pool"
            )
        if len(filtered) != len(hosts):
            _logger.info(
                "Docker backend '%s': node filters selected %d of %d host(s).",
                self.name,
                len(filtered),
                len(hosts),
            )
        return filtered

    def preflight_validate(self) -> None:
        if shutil.which("docker") is None:
            raise ValueError(
                f"Pre-flight validation failed for Docker backend '{self.name}'. "
                "Missing required command: docker. Ensure Docker is installed and available on PATH."
            )

    async def allocate(self) -> Allocation:
        return self.placeholder_allocation()

    def placeholder_allocation(self) -> Allocation:
        if self._hosts:
            nodes = []
            for index, host in enumerate(self._hosts):
                gpus_per_node = (
                    int(host.gpus_per_node)
                    if host.gpus_per_node is not None
                    else self._gpu_per_node
                )
                nodes.append(
                    ComputeNode(
                        name=host.name,
                        ip_address=str(host.ip_address or ""),
                        index=index,
                        num_gpus=gpus_per_node,
                    )
                )
            return Allocation(allocation_id="docker", nodes=nodes, owned=False)

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
        return Allocation(allocation_id="docker", nodes=nodes, owned=False)

    def host_for_node(self, node_name: str) -> DockerHostConfig | None:
        return self._host_by_name.get(node_name)

    def dry_run_details(self) -> list[tuple[str, str]]:
        details = [("image", self._image)]
        if self._hosts:
            details.append(("hosts", ", ".join(host.name for host in self._hosts)))
        else:
            details.append(("nodes", str(self._nodes)))
        if self._gpu_per_node is not None:
            details.append(("gpus_per_node", str(self._gpu_per_node)))
        if self._workdir is not None:
            details.append(("workdir", self._workdir))
        if self._mounts:
            details.append(("mounts", str(list(self._mounts))))
        if self._extra_args:
            details.append(("extra_args", str(list(self._extra_args))))
        return details

    async def release(self, allocation: Allocation) -> None:
        return None

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        return DockerRunOperator(
            DockerRunOperatorConfig(
                name=name,
                image=self._image,
                workdir=self._workdir,
                mounts=list(self._mounts),
                extra_args=list(self._extra_args),
                log_to_file=bool(self.config.offload_task_logs),
            )
        )

    def monitor_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        # Bare-node monitoring must observe the physical host, not a container.
        # Running the collector inside the workload image would (a) hide the
        # materialized hardware_monitor.py, which lives on the host filesystem,
        # and (b) report a cgroup-limited view instead of real host hardware.
        # So run it directly on the host via bash.
        for node in assigned_nodes or []:
            host = self.host_for_node(node)
            if host is not None and (host.docker_host or host.context):
                _logger.warning(
                    f"Monitor '{name}': backend '{self.name}' targets remote Docker "
                    f"host '{host.name}', but bare-node monitoring runs on the local "
                    "driver host; reported hardware metrics reflect the driver, not "
                    "the remote host."
                )
                break
        operator = BashOperator(BashOperatorConfig(name=name, log_to_file=False))
        return configure_bare_monitor_operator(
            operator, backend=self, assigned_nodes=assigned_nodes
        )

    @classmethod
    def resolve_config(
        cls,
        conf: DockerBackendConfig,
        *,
        resolver: Any,
        ctx: dict[str, Any],
        workflow_name: str,
    ) -> DockerBackendConfig:
        image = resolver.resolve(conf.image, ctx)
        nodes = resolver.resolve(conf.nodes, ctx) if conf.nodes is not None else 1
        try:
            nodes_i = max(int(nodes), 1)
        except Exception as e:
            raise ValueError(
                f"Backend '{conf.name}' nodes must resolve to int, got {nodes!r}"
            ) from e

        gpus_per_node = None
        if conf.gpus_per_node is not None:
            resolved = resolver.resolve(conf.gpus_per_node, ctx)
            try:
                gpus_per_node = int(resolved)
            except Exception as e:
                raise ValueError(
                    f"Backend '{conf.name}' gpus_per_node must resolve to int, got {resolved!r}"
                ) from e
            if gpus_per_node < 0:
                raise ValueError(
                    f"Backend '{conf.name}' gpus_per_node must be >= 0, got {gpus_per_node}"
                )

        mounts = [str(resolver.resolve(m, ctx)) for m in (conf.mounts or [])]
        extra_args = [str(resolver.resolve(a, ctx)) for a in (conf.extra_args or [])]
        hosts: list[DockerHostConfig] | None = None
        if conf.hosts:
            hosts = []
            for host in conf.hosts:
                gpus_per_node_host = None
                if host.gpus_per_node is not None:
                    gpus_per_node_host = int(resolver.resolve(host.gpus_per_node, ctx))
                hosts.append(
                    DockerHostConfig(
                        name=str(resolver.resolve(host.name, ctx)),
                        docker_host=str(resolver.resolve(host.docker_host, ctx))
                        if host.docker_host is not None
                        else None,
                        context=str(resolver.resolve(host.context, ctx))
                        if host.context is not None
                        else None,
                        ip_address=str(resolver.resolve(host.ip_address, ctx))
                        if host.ip_address is not None
                        else None,
                        gpus_per_node=gpus_per_node_host,
                        mounts=[
                            str(resolver.resolve(m, ctx)) for m in (host.mounts or [])
                        ],
                        extra_args=[
                            str(resolver.resolve(a, ctx))
                            for a in (host.extra_args or [])
                        ],
                    )
                )
        workdir = (
            str(resolver.resolve(conf.workdir, ctx))
            if conf.workdir is not None
            else None
        )
        include_nodes, exclude_nodes = resolve_node_filters(resolver, conf, ctx)

        return DockerBackendConfig(
            name=conf.name,
            type="docker",
            default=bool(getattr(conf, "default", False)),
            image=str(image),
            nodes=nodes_i,
            hosts=hosts,
            gpus_per_node=gpus_per_node,
            mounts=mounts,
            workdir=workdir,
            extra_args=extra_args,
            include_nodes=include_nodes,
            exclude_nodes=exclude_nodes,
            offload_task_logs=bool(getattr(conf, "offload_task_logs", True)),
        )
