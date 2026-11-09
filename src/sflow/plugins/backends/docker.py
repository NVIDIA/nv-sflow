# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal
from urllib.parse import urlsplit

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
from sflow.core.monitor import MonitorDelivery
from sflow.core.operator import Operator
from sflow.logging import get_logger
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.plugins.operators.docker_run import (
    DockerRunOperator,
    DockerRunOperatorConfig,
    make_container_name,
)
from sflow.utils import gpu_reservation
from sflow.utils.container import local_artifact_mounts
from sflow.utils.extra_args import normalize_extra_args
from sflow.utils.gpu import count_visible_devices
from sflow.utils.gpu_reservation import validate_config_wait_for_gpus
from sflow.utils.node_filters import (
    filter_by_node_names,
    normalize_node_list,
    resolve_node_filters,
)

_logger = get_logger(__name__)


# Bounds every host-touching pre-flight call. Long enough for a cold ssh
# handshake, short enough that a wedged host does not hold up the whole run.
_PREFLIGHT_TIMEOUT_S = 20

# How many hosts are probed at once. Every probe is an idle wait on the network,
# so this is not CPU-bound -- the cap exists so a very large pool does not open
# a hundred ssh connections in the same instant.
_PREFLIGHT_MAX_PARALLEL = 16

# Pre-flight must never turn into a prompt. Up to _PREFLIGHT_MAX_PARALLEL probes
# run at once and they all inherit the driver's terminal, so an unknown host key
# or a passphrase-protected key would otherwise put sixteen ssh processes on the
# same TTY at the same instant -- interleaved questions nobody can answer,
# followed by sixteen timeouts. BatchMode turns each of those into an immediate,
# attributable failure, which is what pre-flight exists to produce.
#
# Both probes therefore run `ssh` THEMSELVES. `docker -H ssh://` spawns ssh with
# no way to pass options (the CLI reads no DOCKER_SSH_OPTS or similar; only
# ~/.ssh/config applies), so a probe routed through it could still prompt.
_SSH_BATCH_OPTS = ("-o", "BatchMode=yes", "-o", f"ConnectTimeout={_PREFLIGHT_TIMEOUT_S}")

_DAEMON_PROBE = ("version", "--format", "{{.Server.Version}}")


def _ssh_target(host: "DockerHostConfig") -> tuple[str, ...] | None:
    """The ssh argv tail (`[-p PORT] [user@]address`) reaching this host's
    machine, or None when the daemon is not behind an ssh:// endpoint."""
    args = list(host.control_args())
    if len(args) == 2 and args[0] == "--host" and args[1].startswith("ssh://"):
        url = urlsplit(args[1])
        target = f"{url.username}@{url.hostname}" if url.username else str(url.hostname)
        return ("-p", str(url.port), target) if url.port else (target,)
    return None


def _describe_host(host: DockerHostConfig) -> str:
    """The concrete machine behind a host, as a dry-run shows it.

    The control endpoint (`user@address`, else the daemon URL or context), plus
    the rendezvous address only when peers dial something other than that
    endpoint, and the GPU count only when this host overrides the backend's.
    """
    target = _ssh_target(host)
    if target is None:
        target = str(host.docker_host) if host.docker_host else f"context {host.context}"
    else:
        target = target[-1]
    extras = []
    rendezvous = host.rendezvous()
    if rendezvous and rendezvous != target.rsplit("@", 1)[-1]:
        extras.append(f"rendezvous={rendezvous}")
    if host.gpus_per_node is not None:
        extras.append(f"gpus={host.gpus_per_node}")
    return f"{target} ({', '.join(extras)})" if extras else target


def _mount_sources(mounts: Sequence[str]) -> list[str]:
    """Host-side paths of `host:container[:mode]` specs, deduped, order kept.

    Only absolute paths: a bare name is a named docker volume, which the daemon
    owns and which no filesystem check applies to.
    """
    sources: list[str] = []
    for spec in mounts:
        source = str(spec).split(":", 1)[0].strip()
        if source.startswith("/") and source not in sources:
            sources.append(source)
    return sources


def _is_unresolved(value: Any) -> bool:
    """Whether ``value`` still holds an unsubstituted ``${{ ... }}`` expression."""
    return value is not None and "${{" in str(value)


class DockerHostConfig(BaseModel):
    """One machine in the declared pool.

    A host answers two questions that look alike and are not: which daemon the
    *driver* sends `docker run` to (the control endpoint), and which address
    *peer containers* dial to rendezvous. ``address`` supplies both, because on
    an ordinary machine they are the same. They diverge behind a jump host, on a
    Docker context, or when peers talk over a separate fabric -- hence the
    overrides, which exist so the simple case stays one line.
    """

    name: str
    # The machine. Everything else on this model is an exception to it.
    address: Resolvable[str] | None = None
    # SSH user for the derived endpoint. Port/key/proxy stay in ~/.ssh/config
    # rather than being mirrored here, so there is one source of truth for them.
    user: Resolvable[str] | None = None
    # Control-endpoint overrides (at most one).
    docker_host: Resolvable[str] | None = None
    context: Resolvable[str] | None = None
    # Rendezvous override, for a data plane that is not `address`.
    rendezvous_address: Resolvable[str] | None = None
    # Deprecated spelling of rendezvous_address.
    ip_address: Resolvable[str] | None = None
    gpus_per_node: Resolvable[int] | None = None
    mounts: list[Resolvable[str]] | None = None
    extra_args: list[Resolvable[str]] | None = None

    @model_validator(mode="after")
    def check_identity(self) -> "DockerHostConfig":
        # These fields are Resolvable, so this runs on the raw template first.
        # A `${{ ... }}` is made of exactly the characters the checks below
        # reject, so judging one here would reject every recipe that
        # parameterises a host. resolve_config rebuilds this model with the
        # substituted values, and the checks bite on that pass.
        if any(
            _is_unresolved(value)
            for value in (self.address, self.user, self.rendezvous_address)
        ):
            return self
        if self.address is not None:
            # `address` feeds two consumers: an ssh:// endpoint, where a scheme
            # or `user@` would be silently absorbed, and the rendezvous address
            # handed to peers, where the same text is simply wrong. Decorations
            # that look right in the endpoint are the failure mode -- an
            # `ssh://host` here yields `ssh://ssh://host` and a peer address no
            # one can dial -- so reject them rather than guess.
            address = str(self.address).strip()
            for bad, hint in (
                ("://", "drop the scheme; 'address' is a host, not a URL"),
                ("@", "put the login in 'user' instead"),
                (" ", "an address cannot contain whitespace"),
            ):
                if bad in address:
                    raise ValueError(
                        f"docker host '{self.name}' has address {self.address!r}: {hint}"
                    )
            self.address = address
        if self.user is not None and "@" in str(self.user):
            raise ValueError(
                f"docker host '{self.name}' has user {self.user!r}: a login name "
                "cannot contain '@'"
            )
        if self.docker_host and self.context:
            raise ValueError(
                f"docker host '{self.name}' sets both 'docker_host' and "
                "'context'; they are alternative endpoint overrides, so set at "
                "most one"
            )
        if not (self.address or self.docker_host or self.context):
            raise ValueError(
                f"docker host '{self.name}' needs an 'address' (the machine), "
                "or a 'docker_host'/'context' endpoint override"
            )
        if self.user and not self.address:
            # Silently ignoring it would hide a real misconfiguration: the user
            # believes they set an SSH identity that is never applied.
            raise ValueError(
                f"docker host '{self.name}' sets 'user' without 'address'. "
                "'user' decorates the endpoint derived from 'address'; with an "
                "explicit 'docker_host' put the user in the URL instead"
            )
        if self.ip_address is not None:
            # A log line, not DeprecationWarning: Python silences that category
            # by default, so it would never reach the person running sflow.
            _logger.warning(
                f"docker host '{self.name}': 'ip_address' is deprecated, use "
                "'rendezvous_address' (or just 'address' when the machine is "
                "reachable at one address)"
            )
        return self

    def control_args(self) -> list[str]:
        """Docker client flags selecting this host's daemon."""
        if self.docker_host:
            return ["--host", str(self.docker_host)]
        if self.context:
            return ["--context", str(self.context)]
        target = str(self.address)
        if self.user:
            target = f"{self.user}@{target}"
        return ["--host", f"ssh://{target}"]

    def rendezvous(self) -> str | None:
        """Address peers dial, or None when this host declares no machine.

        A `context:`-only host names a daemon, not a machine, so there is
        nothing to hand a peer -- callers must treat None as "cannot take part
        in a multi-node task" rather than substituting a guess.
        """
        # `address` verbatim. If it is an ssh alias that peers cannot resolve,
        # `rendezvous_address` is how you say so -- sflow does not guess by
        # reading your ~/.ssh/config.
        for value in (self.rendezvous_address, self.ip_address, self.address):
            if value:
                return str(value)
        # A `docker_host: ssh://...` names the machine just as `address` does.
        if self.docker_host and str(self.docker_host).startswith("ssh://"):
            return urlsplit(str(self.docker_host)).hostname
        return None


class DockerBackendConfig(BackendConfig):
    type: Literal["docker"] = "docker"
    # The image for tasks that declare no `operator:` (the backend's default
    # docker_run), and for the monitor collector's container on a host that has
    # no ssh endpoint. A recipe whose tasks all name an operator -- the usual
    # shape once there is more than one image in play -- has no use for it, so
    # it is optional; the two consumers fail with a message naming it if they
    # are reached without one.
    image: Resolvable[str] | None = None
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
    # Wait behavior when not enough GPUs are free at reserve time: None (default)
    # = fail fast; 0 = wait indefinitely; N > 0 = wait up to N seconds. The
    # --wait-for-gpus CLI flag / SFLOW_WAIT_FOR_GPUS env override this.
    wait_for_gpus: Resolvable[int] | None = None

    def container_images(self) -> list[str]:
        return [str(self.image)] if self.image else []

    def planning_node_count(self) -> Resolvable[int] | None:
        if self.hosts:
            return len(self.hosts)
        return self.nodes

    def executes_offhost(self) -> bool:
        # Off-host only CONDITIONALLY: the same type runs on this machine with
        # no `hosts:` and on other machines with one.
        return bool(self.hosts)


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
            # A declared pool runs every container on another machine, so the
            # controller's filesystem is not the one that matters -- a local
            # fs:// path must not be hard-validated against it. Derived from
            # the config's own answer so the pre-resolution check
            # (config_uses_offhost_backend) and this one cannot disagree.
            supports_host_path_mounts=not config.executes_offhost(),
            has_runtime_node_addresses=bool(self._hosts)
            # `any`, not `all`: this reports whether the backend can supply node
            # addresses at all, and consumers ask it per task. Requiring every
            # host to qualify would blank the address of a host that *has* one
            # because some unrelated pool member lacks it -- silently costing a
            # single-node task its service URL. The multi-node requirement that
            # *every assigned* host resolve is enforced where the assignment is
            # known, in the operator, and names the offending hosts.
            and any(host.rendezvous() for host in self._hosts),
        )
        # Declared artifacts, handed to preflight_validate by the backend
        # lifecycle. Operators turn local ones into bind mounts, so their paths
        # must exist on every host -- but operators do not exist yet at
        # pre-flight time, hence the hand-off.
        self._preflight_artifacts: list[Any] = []
        self._image = str(config.image) if config.image is not None else None
        self._nodes = len(self._hosts) if self._hosts else int(config.nodes or 1)
        self._reject_untrue_node_count(config)
        self._gpu_per_node = (
            int(config.gpus_per_node) if config.gpus_per_node is not None else None
        )
        self._mounts = [str(m) for m in (config.mounts or [])]
        self._workdir = str(config.workdir) if config.workdir is not None else None
        # Shell-split each entry into clean argv tokens (shared with the Slurm
        # backend): a bundled/whitespace-laden entry can't survive as one
        # unparsable token for `docker run`. Never worse than verbatim passthrough.
        self._extra_args = normalize_extra_args(config.extra_args)
        # Recipe-level GPU wait behavior (None = fail fast; 0 = forever; N = N s).
        # The --wait-for-gpus CLI flag / env overrides this at reserve time.
        self._wait_for_gpus = (
            int(config.wait_for_gpus) if config.wait_for_gpus is not None else None
        )

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

    def preflight_validate(self, *, artifacts: Sequence[Any] = ()) -> None:
        # Stashed rather than threaded: the per-host checks run through
        # _in_parallel, which maps a single-argument probe over the pool.
        self._preflight_artifacts = list(artifacts)
        if shutil.which("docker") is None:
            raise ValueError(
                f"Pre-flight validation failed for Docker backend '{self.name}'. "
                "Missing required command: docker. Ensure Docker is installed and available on PATH."
            )
        if not self._hosts or self._preflight_skipped():
            return
        # Everything below fails the workflow anyway if it is wrong. Checking it
        # here costs a second and names the host; discovering it at launch costs
        # the image pulls that already succeeded and names a container instead.
        _logger.info(
            f"Pre-flight: validating {len(self._hosts)} declared Docker host(s) "
            f"for backend '{self.name}', in parallel "
            f"(set SFLOW_SKIP_DOCKER_PREFLIGHT=1 to skip)"
        )
        per_host = self._in_parallel(self._check_one_daemon)
        daemon_problems = [p for host_problems in per_host for p in host_problems]
        # The mount check is an ssh to the same machine the daemon probe just
        # reached. For a host that did not answer, repeating it proves nothing
        # and costs another full timeout, so only reachable hosts are checked.
        reachable = [
            host
            for host, host_problems in zip(self._hosts, per_host)
            if not host_problems
        ]
        mount_problems = [
            p
            for host_problems in self._in_parallel(
                self._check_one_hosts_mounts, reachable
            )
            for p in host_problems
        ]
        problems = [*daemon_problems, *mount_problems]
        if not problems:
            _logger.info(
                f"Pre-flight: all {len(self._hosts)} Docker host(s) reachable"
            )
        if problems:
            listed = "\n  - ".join(problems)
            raise ValueError(
                f"Pre-flight validation failed for Docker backend "
                f"'{self.name}':\n  - {listed}"
            )

    @staticmethod
    def _preflight_skipped() -> bool:
        """True when SFLOW_SKIP_DOCKER_PREFLIGHT bypasses host-touching checks."""
        skip = os.environ.get("SFLOW_SKIP_DOCKER_PREFLIGHT", "")
        return bool(skip and skip.lower() not in ("0", "false", "no"))

    def _in_parallel(
        self,
        probe: "Callable[[DockerHostConfig], list[str]]",
        hosts: "Sequence[DockerHostConfig] | None" = None,
    ) -> list[list[str]]:
        """Run ``probe`` against every host at once; one result list per host.

        Kept per-host rather than flattened so callers can tell WHICH hosts
        failed -- the mount check uses that to skip hosts already known dead.

        Each probe is a bounded-but-slow network call, so serially the pool's
        pre-flight cost was the SUM of its hosts -- minutes for a large pool,
        and worst on exactly the unreachable hosts the check exists to find.
        Run together, it costs the slowest host instead.

        Results stay in the declared host order -- ``map`` yields by input, not
        by completion -- so an error list does not reshuffle between runs
        according to which host happened to answer first.
        """
        targets = list(self._hosts if hosts is None else hosts)
        if not targets:
            return []
        if len(targets) == 1:
            return [probe(targets[0])]
        workers = min(len(targets), _PREFLIGHT_MAX_PARALLEL)
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="sflow-preflight"
        ) as pool:
            return list(pool.map(probe, targets))

    def _check_one_daemon(self, host: "DockerHostConfig") -> list[str]:
        """One bounded `docker version` on the host.

        Proves the whole control path in a single call -- ssh auth, a shell on
        the machine, and a daemon that answers -- which is exactly what every
        later command on that host depends on. For an ssh endpoint it runs
        `ssh <target> docker version`, the same hop `docker -H ssh://` makes
        (`ssh <target> docker system dial-stdio`) but with _SSH_BATCH_OPTS on
        the argv, which the docker client offers no way to pass.
        """
        args = list(host.control_args())
        endpoint = " ".join(args)
        target = _ssh_target(host)
        if target is not None:
            argv = ["ssh", *_SSH_BATCH_OPTS, *target, shlex.join(("docker", *_DAEMON_PROBE))]
        else:
            argv = ["docker", *args, *_DAEMON_PROBE]
        # Announced BEFORE the call, not after: each probe can sit for
        # _PREFLIGHT_TIMEOUT_S on an unreachable host, and a silent driver is
        # indistinguishable from a hung one.
        _logger.info(
            f"Pre-flight host '{host.name}': contacting "
            f"Docker daemon via {endpoint} (up to {_PREFLIGHT_TIMEOUT_S}s)..."
        )
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=_PREFLIGHT_TIMEOUT_S,
                check=False,
                # capture_output does NOT redirect stdin; without this the probe
                # inherits the driver's terminal. See _SSH_BATCH_OPTS.
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            _logger.warning(
                f"Pre-flight host '{host.name}': no response within "
                f"{_PREFLIGHT_TIMEOUT_S}s"
            )
            return [
                f"host '{host.name}' ({endpoint}): no response within "
                f"{_PREFLIGHT_TIMEOUT_S}s"
            ]
        except OSError as exc:
            _logger.warning(f"Pre-flight host '{host.name}': {exc}")
            return [f"host '{host.name}' ({endpoint}): {exc}"]
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            reason = detail[-1] if detail else f"exit {result.returncode}"
            _logger.warning(f"Pre-flight host '{host.name}': unreachable ({reason})")
            return [f"host '{host.name}' ({endpoint}) unreachable: {reason}"]
        _logger.info(
            f"Pre-flight host '{host.name}': daemon OK "
            f"(server {result.stdout.strip() or 'unknown'})"
        )
        return []

    def _check_one_hosts_mounts(self, host: "DockerHostConfig") -> list[str]:
        """Check one host actually has the paths it is asked to bind-mount.

        Docker creates a missing bind source as an empty directory rather than
        failing, so a typo'd or unmounted path becomes an empty mount and the
        workload reports missing data instead of a bad config. Checked over ssh,
        which needs an ssh endpoint -- a `context:` host names a daemon, not a
        machine sflow knows how to shell into, so it is skipped.
        """
        target = _ssh_target(host)
        declared = _mount_sources([*self._mounts, *(host.mounts or [])])
        # Operators bind these too, but they are not the same promise: a
        # declared mount asserts data that already exists, while an artifact
        # may be PRODUCED by the workflow -- a model a download task writes
        # on first run. Missing declared mounts fail; missing artifacts are
        # reported and allowed through.
        from_artifacts = [
            src
            for src in _mount_sources(
                local_artifact_mounts(self._preflight_artifacts)
            )
            if src not in declared
        ]
        sources = [*declared, *from_artifacts]
        if not (target and sources):
            return []
        probe = "; ".join(
            f'[ -e {shlex.quote(src)} ] || printf "%s\\n" {shlex.quote(src)}'
            for src in sources
        )
        _logger.info(
            f"Pre-flight host '{host.name}': checking {len(sources)} mount "
            f"source(s) over ssh ({', '.join(sources)})..."
        )
        try:
            result = subprocess.run(
                ["ssh", *_SSH_BATCH_OPTS, *target, probe],
                capture_output=True,
                text=True,
                timeout=_PREFLIGHT_TIMEOUT_S,
                check=False,
                # capture_output does NOT redirect stdin; without this the probe
                # inherits the driver's terminal. See _SSH_BATCH_OPTS.
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired):
            # Reachability is the daemon check's job; do not report it twice.
            _logger.info(
                f"Pre-flight host '{host.name}': mount check skipped "
                "(ssh did not complete)"
            )
            return []
        missing = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        missing_artifacts = [m for m in missing if m in from_artifacts]
        missing_declared = [m for m in missing if m not in from_artifacts]
        if missing_artifacts:
            _logger.warning(
                f"Pre-flight host '{host.name}': artifact path(s) not present "
                f"yet: {', '.join(missing_artifacts)}. Docker will create them "
                "empty; a task in this workflow is expected to populate them."
            )
        if missing_declared:
            _logger.warning(
                f"Pre-flight host '{host.name}': missing mount source(s): "
                f"{', '.join(missing_declared)}"
            )
            return [
                f"host '{host.name}': mount source(s) not found on that "
                f"machine: {', '.join(missing_declared)}"
            ]
        if not missing:
            _logger.info(f"Pre-flight host '{host.name}': mount sources OK")
        return []

    async def allocate(self) -> Allocation:
        # GPUs are reserved per task at launch (the docker_run operator's
        # acquire_resources), not for the whole run here, so a task holds GPUs
        # only while it runs. gpus_per_node is just the planning capacity.
        return self.placeholder_allocation()

    @property
    def wait_for_gpus_setting(self) -> int | None:
        """Recipe-level wait_for_gpus, read by the operator at reserve time."""
        return self._wait_for_gpus

    def placeholder_allocation(self) -> Allocation:
        if self._hosts:
            nodes = []
            for index, host in enumerate(self._hosts):
                gpus_per_node = self._gpus_on(host)
                nodes.append(
                    ComputeNode(
                        name=host.name,
                        ip_address=str(host.rendezvous() or ""),
                        index=index,
                        num_gpus=gpus_per_node,
                    )
                )
            return Allocation(allocation_id="docker", nodes=nodes, owned=False)

        # No pool -> this machine, one node. `nodes: >1` never reaches here;
        # _reject_untrue_node_count refuses it at construction.
        node = ComputeNode(
            name="localhost", ip_address="127.0.0.1", index=0,
            num_gpus=self._gpu_per_node,
        )
        return Allocation(allocation_id="docker", nodes=[node], owned=False)

    def _reject_untrue_node_count(self, config: DockerBackendConfig) -> None:
        """`nodes: >1` must name as many machines as `hosts:` declares.

        One rule, two ways to break it. `hosts:` IS the node list, so a
        different `nodes:` was never an override -- the pool silently won and
        the recipe ran with fewer workers than it sizes its parallelism for. No
        pool at all is the same rule with zero hosts: `nodes: 2` used to invent
        `localhost-1` from this machine and report the box's full GPU count for
        each, so the planner handed the same devices out twice.

        Either way the configuration claimed something untrue, so it is refused
        rather than quietly reinterpreted. Only a count above the default is
        certainly a request: `nodes` defaults to 1, so an unset field and a
        deliberate `nodes: 1` are indistinguishable here and neither costs
        anything.
        """
        if config.nodes is None:
            return
        try:
            asked = int(config.nodes)
        except (TypeError, ValueError):
            # Unresolved or non-numeric: resolve_config rebuilds this model with
            # the substituted value, and this check bites on that pass.
            return
        if asked <= 1:
            return
        # The declared list, not self._hosts: --include/--exclude-nodes narrows
        # a run, it does not make the recipe inconsistent.
        declared = config.hosts or []
        if not declared:
            raise ValueError(
                f"Docker backend '{self.name}' asks for {asked} nodes but "
                "declares no 'hosts:'. Multi-node docker runs one container per "
                "machine, so list the machines under 'hosts:' -- or set "
                "'nodes: 1' to run several containers on this one."
            )
        if asked != len(declared):
            raise ValueError(
                f"Docker backend '{self.name}' declares {len(declared)} host(s) "
                f"but asks for nodes: {asked}. 'hosts:' is the node list, so the "
                f"run would use {len(declared)} and ignore the rest. Drop "
                "'nodes:', or declare a host per node."
            )

    def host_for_node(self, node_name: str) -> DockerHostConfig | None:
        return self._host_by_name.get(node_name)

    def _gpus_on(self, host: DockerHostConfig) -> int | None:
        """This host's GPU count: its own if declared, else the backend's.

        A pool need not be uniform, so the per-host value wins -- but only when
        it is set, since 0 is a meaningful "this box has none". One definition
        because the allocation and the monitor delivery must agree on it: they
        disagreeing is how a node gets planned for GPUs its collector is then
        started without.
        """
        if host.gpus_per_node is not None:
            return int(host.gpus_per_node)
        return self._gpu_per_node

    def monitor_delivery(
        self, *, node_name: str, scopes: Sequence[str]
    ) -> MonitorDelivery:
        """Run the collector on the node's OWN machine, bare when ssh can reach it.

        Only a declared host needs this. Its machine is not the driver's, so the
        bare-host collector would sample the driver and file the result under
        this node's name; and since the two share no filesystem, the collector
        can neither read the materialized script nor write to the raw dir. Both
        transports below solve all three: the source rides in on stdin and the
        samples ride back out on stdout.
        """
        host = self.host_for_node(node_name)
        if host is None:
            return MonitorDelivery()

        container = make_container_name("sflow-monitor", node_name)
        target = _ssh_target(host)
        if target is not None:
            # Straight on the host over ssh: it already sees the real NICs, disk
            # and GPUs, so none of the container's namespace or --gpus caveats
            # apply, the workload image need not carry python3 (or clear its
            # entrypoint), and no image is pulled just to sample. The host needs
            # python3 and, for the gpu scope, nvidia-smi.
            ssh = ("ssh", *_SSH_BATCH_OPTS, *target)
            # Killing the local ssh client does not signal the remote process
            # (no pty), so record its pid -- `exec` keeps the shell's $$ -- and
            # kill that on teardown, as the container path removes by name.
            # ponytail: a SIGKILLed driver never runs teardown, so this path can
            # leave the collector and its pidfile on the host; the container path
            # is reap-parseable by name, this one is not. Add an ssh sweep if
            # that leak is observed.
            pidfile = f"/tmp/{container}.pid"
            return MonitorDelivery(
                launch_prefix=(*ssh, f"echo $$ > {pidfile}; exec"),
                offhost=True,
                stop_argv=(
                    *ssh,
                    f"kill $(cat {pidfile} 2>/dev/null) 2>/dev/null; rm -f {pidfile}",
                ),
            )

        # A `context:` or tcp:// endpoint names a daemon, not a machine sflow can
        # log into, so run the collector as a container on that daemon instead.
        prefix = [
            "docker", *host.control_args(), "run", "--rm",
            # The collector source arrives on stdin.
            "-i",
            # Reap-parseable, so docker_run's orphan sweep reclaims this
            # container if the driver is killed before it can stop it.
            "--name", container,
            # The host's network namespace. `/proc/net/dev` is namespaced, so
            # without this the collector measures a veth pair instead of the
            # real NICs -- and `socket.gethostname()` returns the container id.
            "--network", "host",
        ]
        if "disk" in scopes:
            # The mount namespace is NOT shared, so the collector's "/" is the
            # image overlay -- it would report the image's free space as the
            # node's. Read-only, and only when the scope is actually asked for.
            prefix += ["-v", "/:/sflow-host-root:ro"]
            prefix += ["-e", "SFLOW_MONITOR_DISK_ROOT=/sflow-host-root"]
        # `--gpus all` is refused at daemon admission on a host with no NVIDIA
        # runtime, and that failure takes the WHOLE container down -- so asking
        # for it speculatively costs the cpu/memory/disk/network scopes too.
        # Scope alone is not enough of a signal: the default monitor requests
        # every scope, `gpu` included, so a CPU-only pool would always ask. Pair
        # it with the pool actually declaring GPUs on this node. (nvidia-smi
        # itself is injected by the container toolkit, so the image needs no
        # CUDA -- only the host needs the runtime.)
        declared_gpus = self._gpus_on(host)
        if declared_gpus is None and "gpu" in scopes:
            # None is "nobody said", which is NOT the same as a declared 0 -- and
            # falling back to 0 silently costs a GPU box its GPU metrics. Cannot
            # be resolved by asking for them anyway (see above), so name the one
            # thing the user can do about it.
            # ponytail: declared count only; probe `docker info` for the nvidia
            # runtime in the pre-flight sweep if this proves too easy to forget.
            _logger.warning(
                f"Docker host '{host.name}' declares no 'gpus_per_node', so sflow "
                "cannot tell whether it has GPUs and starts the monitor without "
                "them -- this node will report no GPU metrics. Set 'gpus_per_node' "
                "on the host or the backend to collect them."
            )
        with_gpus = (declared_gpus or 0) > 0
        if "gpu" in scopes and with_gpus:
            prefix += ["--gpus", "all"]
        if self._image is None:
            # Not the bare default: that would sample the DRIVER and file it
            # under this node's name -- the exact misreport the pool delivery
            # exists to prevent. A plan-time error names the one knob to turn.
            raise ValueError(
                f"Docker backend '{self.name}': host '{host.name}' has no ssh "
                "endpoint, so its monitor collector runs as a container and needs "
                "the backend's 'image:' -- set it, or give the host an 'address'."
            )
        prefix.append(self._image)

        return MonitorDelivery(
            launch_prefix=tuple(prefix),
            offhost=True,
            provides_gpus=with_gpus,
            # `docker run` is a client, not the container's parent: killing it
            # leaves the collector sampling on the daemon (observed leaking a
            # live container past the end of a run). Remove it by name instead.
            stop_argv=("docker", *host.control_args(), "rm", "-f", container),
        )

    def dry_run_details(self) -> list[tuple[str, str]]:
        details = [("image", self._image)] if self._image else []
        if self._hosts:
            # One row per host naming the machine sflow will drive, so a dry-run
            # shows where each node lands before anything touches it.
            details += [(f"host {host.name}", _describe_host(host)) for host in self._hosts]
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

    def resource_env(self, *, cuda_visible_devices: str | None = None) -> dict[str, str]:
        # The host-side `--gpus device=<uuid>` already isolates the container to
        # the reserved physical GPUs; inside the container they re-index to
        # 0..N-1. Expose those simple virtual indices (not the UUIDs) so a user
        # who asked for N GPUs just sees CUDA_VISIBLE_DEVICES=0,..,N-1 and never
        # has to deal with physical UUIDs. N is the size of the planned slice.
        #
        # Because this env is *virtual*, run reporting must not read it as a
        # physical device list -- see sflow.utils.gpu.task_gpu_indices.
        # Must match the operator's GPU claim exactly, so use the shared counter:
        # it expands the range form ("0-3" -> 4), which a raw token count reads as
        # a single device -- leaving the container told about 1 GPU while 4 were
        # reserved for it.
        n = count_visible_devices(cuda_visible_devices)
        if not n:
            return {}
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(n))}

    async def release(self, allocation: Allocation) -> None:
        # GPUs are reserved/released per task by the operator, so there is no
        # run-level allocation to release here.
        return None

    async def release_resources(self) -> None:
        # The base does the allocation teardown every backend needs (clearing
        # self.allocation, calling release()); this only ADDS the GPU-registry
        # backstop on top, so it must delegate rather than replace.
        await super().release_resources()
        # Run-end backstop for the machine-local GPU registry. Per-task release is
        # the normal path; this catches the narrow case where a task was cancelled
        # at the instant its reservation record was being written, which its own
        # release could not have seen. Only records owned by THIS pid are touched.
        gpu_reservation.release_all_for_pid()

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        if self._image is None:
            raise ValueError(
                f"Task '{name}' declares no 'operator:' and Docker backend "
                f"'{self.name}' has no 'image:' to run it with -- set one of them."
            )
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
        # Always a plain host shell, for both kinds of node. Without `hosts:` the
        # node is this machine and the collector runs here directly. With a pool,
        # the shell's job is to run the `docker run` that monitor_delivery()
        # built -- so the container lands on the right daemon. Either way this
        # must NOT be the workload's DockerRunOperator: that would nest the
        # collector inside the task's own container and report a cgroup-limited
        # view of it rather than the machine.
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
        image = resolver.resolve(conf.image, ctx) if conf.image is not None else None
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
            def _resolved(value: Any) -> str | None:
                return str(resolver.resolve(value, ctx)) if value is not None else None

            for host in conf.hosts:
                gpus_per_node_host = None
                if host.gpus_per_node is not None:
                    gpus_per_node_host = int(resolver.resolve(host.gpus_per_node, ctx))
                hosts.append(
                    DockerHostConfig(
                        name=str(resolver.resolve(host.name, ctx)),
                        address=_resolved(host.address),
                        user=_resolved(host.user),
                        docker_host=_resolved(host.docker_host),
                        context=_resolved(host.context),
                        rendezvous_address=_resolved(host.rendezvous_address),
                        ip_address=_resolved(host.ip_address),
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

        wait_for_gpus = None
        if conf.wait_for_gpus is not None:
            resolved_wait = resolver.resolve(conf.wait_for_gpus, ctx)
            try:
                wait_for_gpus = int(resolved_wait)
            except Exception as e:
                raise ValueError(
                    f"Backend '{conf.name}' wait_for_gpus must resolve to int, "
                    f"got {resolved_wait!r}"
                ) from e
            # Same >= 0 rule the reservation layer enforces, stated once there.
            validate_config_wait_for_gpus(
                wait_for_gpus, where=f"Backend '{conf.name}'"
            )

        return DockerBackendConfig(
            name=conf.name,
            type="docker",
            default=bool(getattr(conf, "default", False)),
            image=str(image) if image is not None else None,
            nodes=nodes_i,
            hosts=hosts,
            gpus_per_node=gpus_per_node,
            mounts=mounts,
            workdir=workdir,
            extra_args=extra_args,
            include_nodes=include_nodes,
            exclude_nodes=exclude_nodes,
            offload_task_logs=bool(getattr(conf, "offload_task_logs", True)),
            wait_for_gpus=wait_for_gpus,
        )
