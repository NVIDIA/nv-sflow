# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import shlex
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from typing import Literal

from pydantic import Field, PrivateAttr, field_validator

from sflow.core.command import Command
from sflow.core.command_log import record_active_command
from sflow.core.log_offload import offload_enabled, task_log_path, wrap_with_prefixer
from sflow.core.operator import Operator, OperatorConfig, ResourcesUnavailable
from sflow.core.operator_registry import register_operator
from sflow.logging import get_logger
from sflow.utils import gpu_reservation as gpu_res
from sflow.utils.container import (
    append_runtime_mounts as append_runtime_mount_specs,
    local_artifact_mounts,
    validate_container_image_reference,
)
from sflow.utils.extra_args import normalize_extra_args
from sflow.utils.gpu import count_visible_devices

_logger = get_logger(__name__)


# Every sflow-managed container name starts with this, followed by ``p<pid>``.
# The driver PID makes names unique per `sflow run` (so concurrent runs on one
# host don't collide) while still being parseable, so orphan reaping can find a
# crashed run's leftovers and check whether the owning PID is still alive.
_CONTAINER_PREFIX = "sflow"

# Args in `extra_args` that themselves grant the container GPUs. When one is
# present sflow must not also inject NVIDIA_VISIBLE_DEVICES=void, which would
# override the user's explicit request and hide every device.
_GPU_GRANTING_ARG_PREFIXES = (
    "--gpus",
    "--runtime=nvidia",
    "--device=/dev/nvidia",
    "NVIDIA_VISIBLE_DEVICES=",
)
# `--device` takes its value as a separate token too (`--device /dev/nvidia0`),
# which extra_args normalization shell-splits, so the pair must be matched.
_DEVICE_FLAGS = ("--device", "-e", "--env")

# How long a task may wait for GPUs before the wait is called out as possibly
# deadlocked (two runs each holding part of the pool). Well past normal queueing
# -- a task that is merely behind a long-running neighbour should not trip it.
_STALLED_WAIT_WARN_S = 600.0

def _host_args(host: Any | None) -> list[str]:
    """Client flags selecting ``host``'s daemon -- empty for the local one.

    One definition because launch, teardown and the orphan sweep must all reach
    the *same* daemon; a second copy that drifts reaps on one host what it
    launched on another. The host config owns the derivation (address, user, and
    the endpoint overrides); this only adapts it for a ``Command``.
    """
    if host is None:
        return []
    return list(host.control_args())


def _docker_cmd(host: Any | None, *args: str) -> Command:
    """``docker [--host …|--context …] <args>`` for ``host``'s daemon.

    Every docker client sflow runs goes through here, so launch, teardown and
    the orphan sweep cannot drift onto different daemons -- reaping on one host
    what was launched on another is the failure this prevents.
    """
    cmd = Command(exec="docker")
    for arg in (*_host_args(host), *args):
        cmd.add_arg(arg)
    return cmd


def _daemon_label(host: Any | None) -> str:
    """Which Docker daemon a member targets, for human review.

    ``control_args()`` is ``["--host", "ssh://..."]`` or ``["--context", name]``.
    """
    return " ".join(_host_args(host)[1:]) or "local daemon"


def _driver_host_token() -> str:
    """Short, stable id for the machine this driver runs on.

    A PID identifies an owner only *within* one machine, but a remote daemon is
    routinely shared by several drivers -- so two machines can both produce
    `sflow-p1234-*`, and a reaper checking its own /proc would judge a
    co-tenant's live container dead. Tagging the name with the driver's machine
    identity makes ownership decidable from the name alone, which is what lets
    the sweep run against remote daemons at all.

    machine-id is preferred over the hostname because it survives renames and is
    actually unique; the hostname is only a fallback for hosts without one.

    ``SFLOW_DRIVER_ID`` overrides both. A driver running *inside a container*
    gets a fresh machine-id and hostname per run, so its token would change every
    time and it could never reclaim the previous run's orphans -- exactly the
    containerised-CI case the sweep is most needed for. Setting a stable value
    there restores it.
    """
    override = os.environ.get("SFLOW_DRIVER_ID", "").strip()
    if override:
        return hashlib.blake2s(override.encode(), digest_size=3).hexdigest()
    seed = ""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(path) as handle:
                seed = handle.read().strip()
        except OSError:
            continue
        if seed:
            break
    return hashlib.blake2s(
        (seed or socket.gethostname()).encode(), digest_size=3
    ).hexdigest()


# Names of containers owned by a driver on THIS machine. Everything after this
# prefix up to the next '-' is the owning PID.
_OWNER_PREFIX = f"{_CONTAINER_PREFIX}-{_driver_host_token()}p"


def orphan_reap_script(host: Any | None = None) -> str:
    """The sweep reclaiming containers left by *crashed* drivers on ``host``.

    Safe against a shared daemon: the name filter restricts it to containers this
    machine created, so the /proc and `kill -0` checks below are asking about
    local PIDs -- the only process table this script can actually read.

    Liveness must also be user-agnostic: `kill -0 <pid>` fails with EPERM for a
    process owned by ANOTHER user, so on its own it would classify a co-tenant's
    live `sflow run` as dead and force-remove its running containers. /proc/<pid>
    exists regardless of ownership, so check that first and keep kill -0 as the
    fallback for non-procfs systems.

    ``*-kept-*`` is exempt. Those are the containers `rm: false` deliberately
    preserved for `docker logs`/`inspect`; their owning driver is *expected* to
    be gone by the time anyone looks, so the dead-owner rule would have the very
    next run on that host delete the evidence the flag exists to keep.
    """
    docker = " ".join(shlex.quote(p) for p in ("docker", *_host_args(host)))
    return (
        f'for c in $({docker} ps -a --filter name={_OWNER_PREFIX} '
        '--format "{{.Names}}" 2>/dev/null); do '
        'case "$c" in *-kept-*) continue;; esac; '
        f'pid=${{c#{_OWNER_PREFIX}}}; pid=${{pid%%-*}}; '
        'case "$pid" in ""|*[!0-9]*) continue;; esac; '
        '[ -d "/proc/$pid" ] && continue; '
        'kill -0 "$pid" 2>/dev/null && continue; '
        f'{docker} rm -f "$c" >/dev/null 2>&1 || true; '
        'done; true'
    )


# The local-daemon sweep, as a constant because `.gitlab-ci.yml` runs the exact
# same predicate in its after_script -- two hand-maintained copies of something
# this sharp (it force-removes containers) will drift, and the drift is only
# visible when it wrongly kills a live co-tenant's run. CI executes it via:
#
#     python -c "from sflow.plugins.operators.docker_run import ORPHAN_REAP_SCRIPT
#                print(ORPHAN_REAP_SCRIPT)" | bash
ORPHAN_REAP_SCRIPT = orphan_reap_script()


def _safe_container_name(*parts: str) -> str:
    raw = "-".join(part for part in parts if part)
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw).strip("-_.")
    return (sanitized or "sflow-task")[:128]


def make_container_name(task_name: str, node_name: str) -> str:
    """``sflow-<machine>p<pid>-<task>-<node>`` -- reap-parseable, globally unique.

    The machine token is part of the prefix because a PID identifies an owner
    only WITHIN one machine, and a remote daemon is routinely shared by several
    drivers -- two of which can produce the same PID.

    Public because the docker BACKEND names its monitor container with it too:
    that container must be parseable by this module's orphan sweep, so the two
    cannot use different conventions. Named ``make_`` rather than
    ``container_name`` so it does not shadow the local of that name that several
    functions here already bind.
    """
    return _safe_container_name(
        f"{_OWNER_PREFIX}{os.getpid()}", task_name, node_name
    )


def _gpus_device_spec(device_ids: Sequence[str]) -> str:
    """The ``--gpus`` value pinning a container to ``device_ids``.

    Plain: the quoting docker's parser needs is applied by :func:`_quoted_gpus_arg`
    at the one point the value reaches the command line, so it covers a hand-written
    ``gpus:`` too and never ends up baked into a config that gets dumped and reloaded.
    """
    ids = [str(d).strip() for d in device_ids if str(d).strip()]
    return f"device={','.join(ids)}"


def _quoted_gpus_arg(value: str) -> str:
    """``--gpus`` as docker's flag parser needs it.

    `docker run --gpus device=0,1` splits the value on the comma as CSV, reads the
    `1` as a *count*, and dies with "cannot set both Count and DeviceIDs on device
    request". Double-quoting keeps the list one device specification.

    Narrow on purpose: only a lone ``device=`` list is wrapped. In anything else --
    ``count=2,capabilities=gpu`` -- the commas separate OPTIONS, and quoting those
    would break the very parse this exists to fix.
    """
    spec = value.strip()
    prefix = "device="
    if not spec.startswith(prefix) or "=" in spec[len(prefix):]:
        return value
    return f'"{spec}"'


@dataclass(frozen=True)
class _Member:
    """One container of a task: its rank, its node, and the daemon it runs on.

    ``index`` is carried rather than recomputed downstream: deriving the rank a
    second time is only correct while two functions happen to agree on ordering,
    and a rank that disagrees with the leader assignment is a wedged collective
    rather than an error.
    """

    index: int
    node_name: str
    host: Any | None
    container_name: str


class _NodeLogger:
    """The task's logger with member identity prefixed onto every line.

    ``info`` and ``handlers`` are the whole surface -- that is all the launcher
    uses of an ``output_logger``.
    """

    def __init__(self, logger: Any, node_name: str, node_index: int):
        self._logger = logger
        # A newline in a node name would forge extra log lines and break the
        # one-record-per-line shape the `[node rank=N]` label depends on.
        # Substituted outside the f-string: a backslash inside an f-string
        # expression is a syntax error before Python 3.12, and this package
        # supports 3.10.
        safe_name = re.sub(r"[\r\n]", "?", node_name)
        self._prefix = f"[{safe_name} rank={node_index}] "
        self.handlers = getattr(logger, "handlers", [])

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(f"{self._prefix}{message}", *args, **kwargs)


def _grants_gpus(extra_args: Sequence[str]) -> bool:
    """Whether raw docker args already expose GPUs to the container.

    Matches both the joined spelling (``--gpus=all``, ``--device=/dev/nvidia0``)
    and the split one (``--device /dev/nvidia0``, ``-e NVIDIA_VISIBLE_DEVICES=all``),
    since extra_args normalization shell-splits every entry into argv tokens.
    """
    args = [str(a) for a in extra_args]
    for i, arg in enumerate(args):
        if arg.startswith(_GPU_GRANTING_ARG_PREFIXES):
            return True
        if arg in _DEVICE_FLAGS and i + 1 < len(args):
            value = args[i + 1]
            if value.startswith("/dev/nvidia") or value.startswith(
                "NVIDIA_VISIBLE_DEVICES="
            ):
                return True
    return False


class DockerRunOperatorConfig(OperatorConfig):
    name: str
    type: Literal["docker_run"] = "docker_run"

    image: str
    workdir: str | None = None
    mounts: list[str] = Field(default_factory=list)  # e.g. ["/host:/ctr:rw"]
    gpus: str | None = None  # e.g. "all" or "device=0"
    extra_args: list[str] = Field(default_factory=list)
    # Pass `--rm` so the daemon discards the container once it exits. Turning this
    # off also downgrades teardown from `docker rm -f` to `docker stop`, because
    # otherwise the driver would delete the very container this flag preserves.
    rm: bool = True
    pass_envs: bool = True
    auto_mount_runtime_dirs: bool = True
    # Per-task log offload, ON by default. When enabled (also via the
    # SFLOW_OFFLOAD_TASK_LOGS env / --offload-task-logs flag, which take
    # precedence), the task's output is redirected on the host through a
    # compute-side prefixer into <task>.log instead of streaming through the
    # sflow driver's pump. Auto-falls back to streaming on an interactive
    # TTY / --tui session.
    log_to_file: bool = True
    # Accumulated rather than derived, unlike the srun operator's: these depend
    # on the *placement* (how many nodes this task got), which the config cannot
    # see -- so the operator appends them once the assignment is known.
    _runtime_warnings: list[str] = PrivateAttr(default_factory=list)

    def runtime_warnings(self) -> list[str]:
        return list(dict.fromkeys(self._runtime_warnings))

    def container_images(self) -> list[str]:
        return [self.image] if self.image else []

    def mount_specs(self) -> list[str]:
        return list(self.mounts or [])

    def append_runtime_mounts(self, mounts: Sequence[str]) -> None:
        if not self.auto_mount_runtime_dirs:
            return
        self.mounts = append_runtime_mount_specs(list(self.mounts or []), list(mounts))

    @field_validator("image")
    @classmethod
    def image_must_be_valid(cls, value: str) -> str:
        validate_container_image_reference(
            value,
            source="docker_run operator config: 'image'",
        )
        return value


@register_operator("docker_run", DockerRunOperatorConfig)
class DockerRunOperator(Operator):
    def __init__(self, config: DockerRunOperatorConfig):
        super().__init__(config)
        self.config: DockerRunOperatorConfig
        self._assigned_nodes: list[str] = []
        self._node_hosts: dict[str, Any] = {}
        # Per-task GPU reservation state, resolved at launch (acquire_resources).
        self._gpu_count: int = 0
        self._targets_remote_host: bool = False
        self._wait_for_gpus: int | None = None
        self._reservation_run_id: str | None = None
        # The planner's numeric --gpus slice, kept so release_resources can undo
        # a launch-time UUID pin (and so a fallback has something to restore).
        self._plan_time_gpus: str | None = None
        # Deadline for --wait-for-gpus, set on the first attempt so the retries
        # the orchestrator drives all share one budget.
        self._wait_deadline: float | None = None
        # When the wait actually started, for the stalled-wait warning below.
        self._wait_started: float | None = None
        # One-shot warning latches: these fire per launch attempt / per retry, and
        # the conditions do not change between them, so without a latch the same
        # line would repeat every few seconds for the life of the task.
        self._warned_extra_args_gpus: bool = False
        self._warned_reservation_skipped: bool = False
        self._warned_wait_stalled: bool = False

    def _offload_enabled(self) -> bool:
        # Multi-node members already stream through the shared launcher, which
        # owns their output and labels it per rank; a second host-side redirect
        # would be a competing writer on the same <task>.log.
        return offload_enabled(self.config.log_to_file) and len(self._assigned_nodes) <= 1

    def writes_own_task_log(self) -> bool:
        # In offload mode the host-side redirect owns <task>.log, so sflow must
        # not also attach a FileHandler to that path (single-writer invariant).
        return self._offload_enabled()

    def _maybe_offload(
        self, cmd: Command, *, task_name: str, envs: Mapping[str, str]
    ) -> Command:
        """Wrap the docker invocation so its output is prefixed and written to
        <task>.log on the host, taking the driver out of the per-line pump.

        Works uniformly for the single-node ``docker run`` and the multi-node
        ``bash -lc`` forms by piping the whole (shlex-joined) command through the
        prefixer; ``${PIPESTATUS[0]}`` preserves the container/script exit code.
        """
        log_path = task_log_path(envs, task_name)
        if not (self._offload_enabled() and log_path):
            return cmd
        wrapped = wrap_with_prefixer(
            shlex.join(cmd.as_list()),
            workflow_out_dir=envs.get("SFLOW_WORKFLOW_OUTPUT_DIR"),
            task_name=task_name,
            redirect_to=log_path,
        )
        offloaded = Command(exec="bash")
        offloaded.add_arg("-c")
        offloaded.add_arg(wrapped)
        return offloaded

    def apply_backend_context(
        self,
        *,
        backend: Any,
        assigned_nodes: Sequence[str],
        artifacts: Sequence[Any],
        cuda_visible_devices: str | None = None,
        gpu_count: int | None = None,
    ) -> None:
        self._assigned_nodes = list(assigned_nodes or [])
        host_for_node = getattr(backend, "host_for_node", None)
        self._node_hosts = {
            node_name: host_for_node(node_name)
            for node_name in self._assigned_nodes
            if callable(host_for_node)
        }
        # Any declared host is a daemon reached over the network; only the
        # synthetic local node has no host entry at all.
        self._targets_remote_host = any(
            host is not None for host in self._node_hosts.values()
        )
        # Capture the user's own setting first: the flag is also flipped below
        # for remote hosts, and that is sflow's decision about workspace/output
        # dirs, not an instruction to drop the recipe's artifacts.
        opted_out_of_auto_mounts = not self.config.auto_mount_runtime_dirs
        if self._targets_remote_host:
            self.config.auto_mount_runtime_dirs = False

        # `fs://`/`file://` artifacts become same-path bind mounts, so a task can
        # read ${{ artifacts.X.path }} without the recipe repeating the path
        # under `mounts:`. srun has always done this; docker_run accepted the
        # artifacts and ignored them, so the same recipe worked on Slurm and
        # failed here with a missing file.
        #
        # Applied to remote hosts too, bypassing the runtime-dirs flag above.
        # The path does have to exist on those machines -- but that is
        # pre-flight's job: it checks every mount source on every host and names
        # the ones missing. Refusing to mount would instead leave the recipe
        # working on Slurm and quietly broken here.
        auto_mounts = local_artifact_mounts(artifacts)
        if auto_mounts and not opted_out_of_auto_mounts:
            self.config.mounts = append_runtime_mount_specs(
                list(self.config.mounts or []), auto_mounts
            )
        if len(self._assigned_nodes) > 1:
            # Every member is handed the leader's address, so one host that
            # cannot supply one makes the whole assignment unusable. Failing here
            # names the host; launching anyway spends the job's wall clock in a
            # rendezvous that never completes and reports nothing.
            # Only declared hosts can fail this. Synthetic local nodes have no
            # host entry at all and every container lands on this machine, so
            # they rendezvous over loopback and need nothing declared.
            unaddressed = [
                node_name
                for node_name, host in self._node_hosts.items()
                if host is not None and not host.rendezvous()
            ]
            if offload_enabled(self.config.log_to_file):
                message = (
                    "Docker multi-node members stream through the shared "
                    "launcher, so log_to_file offload is skipped; member output "
                    "is labelled by node and merged into the task log instead."
                )
                # Both surfaces on purpose: runtime_warnings() is only collected
                # on --dry-run, so on a real run the log line is the only thing
                # that tells the user their setting was not applied.
                self.config._runtime_warnings.append(message)
                _logger.warning(f"Task uses {len(self._assigned_nodes)} nodes: {message}")
            if unaddressed:
                raise ValueError(
                    f"Docker backend '{getattr(backend, 'name', '?')}': "
                    f"multi-node task cannot resolve a rendezvous address for "
                    f"host(s) {', '.join(sorted(unaddressed))}. Give each host an "
                    "'address' (or 'rendezvous_address'); peers need it to find "
                    "the leader."
                )
        # How many GPUs this task needs. The actual physical GPUs are reserved at
        # launch (acquire_resources), so a task holds them only while it runs
        # rather than for the whole run. The container still sees them as
        # CUDA_VISIBLE_DEVICES=0..N-1 (set by the backend's resource_env).
        if gpu_count is not None:
            self._gpu_count = int(gpu_count)
        elif cuda_visible_devices:
            # The planner emits an explicit numeric slice; the shared counter also
            # handles the range form ("0-3") and non-numeric device ids (e.g.
            # UUIDs). The backend's resource_env sizes the container-visible slice
            # with the same helper, so the two can never disagree.
            self._gpu_count = count_visible_devices(cuda_visible_devices)
        else:
            self._gpu_count = 0
        self._wait_for_gpus = getattr(backend, "wait_for_gpus_setting", None)

        # Pin to the planner's device slice up front for *every* placement. A
        # launch-time reservation (acquire_resources) overwrites it with the
        # physical GPUs it claimed; when there is none -- declared hosts,
        # SFLOW_GPU_RESERVATION=0, or no readable nvidia-smi -- this
        # numeric slice stays as the fallback. Setting it here (rather than only
        # at launch) also keeps `--dry-run` honest: the plan shows the GPU pinning
        # a real run would apply instead of omitting --gpus entirely.
        if cuda_visible_devices:
            self._plan_time_gpus = _gpus_device_spec(cuda_visible_devices.split(","))
            self.config.gpus = self._plan_time_gpus

    def acquire_resources(
        self, *, task_name: str, envs: Mapping[str, str]
    ) -> list[int] | None:
        """Reserve this task's GPUs from the machine-local registry and pin the
        container to them (``--gpus "device=<uuid,...>"``).

        Returns the reserved *physical* device indices (for run reporting), or
        ``None`` when no reservation was made. Local single-node docker only:
        declared hosts (can't read this host's nvidia-smi) and
        reservation-disabled runs skip it and keep the plan-time device slice
        set in :meth:`apply_backend_context`.

        Makes exactly ONE attempt. When the board is full and this task is allowed
        to wait, it raises :class:`ResourcesUnavailable` so the orchestrator can
        retry on the event loop -- sleeping here would block an uncancellable
        worker thread and hang the driver on Ctrl-C.
        """
        if self._gpu_count <= 0 or not gpu_res.reservation_enabled():
            return None
        if self._targets_remote_host:
            # The registry describes this machine's devices, so it cannot speak
            # for a container placed on another one. (Multi-node requires a
            # declared pool, so it is covered by the same test.)
            #
            # Said out loud rather than silently returned: the user read about
            # the reservation in the docs and is not getting it.
            if not self._warned_reservation_skipped:
                self._warned_reservation_skipped = True
                _logger.warning(
                    f"Task '{task_name}': GPU reservation skipped because it runs "
                    f"on declared Docker host(s) whose devices this machine's "
                    f"registry does not track; launching on the planned device "
                    f"slice ({self.config.gpus}) without a cross-process claim. "
                    f"Another run on those hosts can pick the same devices."
                )
            return None
        wait, timeout = gpu_res.wait_options(self._wait_for_gpus)
        # Start the wait budget on the first attempt so retries share one deadline.
        if wait and self._wait_deadline is None:
            self._wait_deadline = (
                time.monotonic() + timeout if timeout is not None else math.inf
            )
            self._wait_started = time.monotonic()
        run_id = gpu_res.make_run_id(task_name)
        try:
            handles = gpu_res.try_reserve_gpus(self._gpu_count, run_id)
        except (gpu_res.GpuProbeError, gpu_res.RegistryLockBusy) as e:
            # Transient: the driver did not answer, or the registry lock is held
            # by something wedged. Crucially NOT the same as "this host has no
            # GPUs" -- degrading to the plan-time slice here would launch
            # unreserved on devices starting at index 0, which is exactly the
            # collision this module exists to prevent. Retry if allowed to wait,
            # otherwise fail closed.
            if wait and time.monotonic() < (self._wait_deadline or 0):
                self._warn_if_wait_stalled(task_name)
                raise ResourcesUnavailable(
                    str(e), retry_after=self._retry_after()
                ) from e
            raise RuntimeError(
                f"Task '{task_name}': could not determine GPU availability ({e}). "
                f"Refusing to launch without a reservation; retry, or set "
                f"{gpu_res.GPU_RESERVATION_ENV}=0 to opt out of reservation."
            ) from e
        except OSError as e:
            # The registry itself is unusable (read-only /tmp, full disk, a bad
            # SFLOW_GPU_RESERVATION_DIR). Reservation is a safety layer, not a
            # prerequisite for running, so warn loudly and fall back rather than
            # failing a task that would otherwise work.
            _logger.warning(
                f"Task '{task_name}': GPU reservation registry unusable ({e}); "
                f"falling back to the planned device slice ({self.config.gpus}) "
                f"without a cross-process reservation."
            )
            return None
        except gpu_res.InsufficientGpusError as e:
            if e.total == 0:
                # nvidia-smi is unreadable from the driver (e.g. it runs in a
                # container that only has the docker socket) while the daemon may
                # still expose GPUs fine. Degrade to the plan-time slice rather
                # than failing a workload that would otherwise run. Waiting could
                # never help here, so this is terminal either way.
                _logger.warning(
                    f"Task '{task_name}': no GPUs visible to nvidia-smi on the "
                    f"driver host; falling back to the planned device slice "
                    f"({self.config.gpus}) without a cross-process reservation."
                )
                return None
            if wait and time.monotonic() < (self._wait_deadline or 0):
                delay = self._retry_after()
                _logger.info(
                    f"Task '{task_name}' waiting for {self._gpu_count} GPU(s); "
                    f"{e.free}/{e.total} free. Retrying in {delay:.1f}s..."
                )
                self._warn_if_wait_stalled(task_name)
                raise ResourcesUnavailable(str(e), retry_after=delay) from e
            raise RuntimeError(f"Task '{task_name}': {e}") from e
        if not handles:
            return None
        self._reservation_run_id = run_id
        self.config.gpus = _gpus_device_spec([h.uuid for h in handles])
        return [h.index for h in handles]

    def _retry_after(self) -> float:
        """Poll delay, never overshooting the remaining wait budget.

        A flat poll interval makes a short budget overrun it -- ``--wait-for-gpus
        1`` would sleep the full 5s before noticing it was out of time. Clamp so
        the last poll lands on the deadline instead of past it.
        """
        remaining = (self._wait_deadline or 0.0) - time.monotonic()
        return max(0.0, min(gpu_res.DEFAULT_POLL_INTERVAL_S, remaining))

    def _warn_if_wait_stalled(self, task_name: str) -> None:
        """Warn once when a GPU wait has gone on long enough to look like deadlock.

        Two runs that each hold part of the pool and each wait for the rest never
        resolve: neither can finish, so neither releases. With an unbounded wait
        (``wait_for_gpus: 0`` / ``--wait-for-gpus 0``) that is silent forever
        apart from the per-retry INFO line, which reads like healthy queueing.
        Nothing here can safely break the tie -- releasing another run's GPUs is
        exactly what the registry exists to prevent -- so name the situation and
        let the user decide.
        """
        if self._warned_wait_stalled or self._wait_started is None:
            return
        waited = time.monotonic() - self._wait_started
        if waited < _STALLED_WAIT_WARN_S:
            return
        self._warned_wait_stalled = True
        unbounded = self._wait_deadline == math.inf
        _logger.warning(
            f"Task '{task_name}' has waited {waited / 60:.0f} min for "
            f"{self._gpu_count} GPU(s) and is still short. If another sflow run on "
            f"this host is also waiting while holding GPUs, neither can proceed "
            f"(each is waiting for the other to finish). "
            + (
                "This wait is unbounded, so it will not time out on its own; "
                "bound it with --wait-for-gpus <seconds>, or stop one of the runs."
                if unbounded
                else "This wait is bounded and will fail when the budget runs out."
            )
        )

    def release_resources(
        self, *, task_name: str, reusable: bool = False, handover: bool = False
    ) -> None:
        # Release by the deterministic run id rather than only when we know we
        # hold one: a task cancelled at the instant its record was written would
        # otherwise leave it behind. release_gpus is idempotent.
        #
        # Skipped for tasks that could never have reserved (no GPUs asked for,
        # reservation disabled) so an ordinary CPU-only container does not pay
        # registry I/O on every teardown.
        if self._gpu_count > 0 and gpu_res.reservation_enabled():
            gpu_res.release_gpus(
                gpu_res.make_run_id(task_name), reusable=reusable, handover=handover
            )
        if reusable:
            # The task is still running on these GPUs -- only its *claim* was
            # handed back. Keep the UUID pin and the wait budget so nothing about
            # the live container changes.
            return
        self._reservation_run_id = None
        self._wait_deadline = None
        self._wait_started = None
        self._warned_wait_stalled = False
        # Drop the UUID pin so a reused operator instance can never launch a
        # second container against GPUs this task no longer holds.
        if self._plan_time_gpus is not None:
            self.config.gpus = self._plan_time_gpus

    def _build_docker_command(
        self,
        *,
        task_name: str,
        node_name: str | None,
        host: Any | None,
        script: Sequence[str],
        envs: Mapping[str, str],
        container_name: str | None = None,
        per_node_envs: Mapping[str, str] | None = None,
    ) -> Command:
        c = self.config
        gpus = c.gpus
        cmd = _docker_cmd(host)

        cmd.add_arg("run")
        if c.rm:
            cmd.add_arg("--rm")
        if container_name:
            cmd.add_arg("--name")
            cmd.add_arg(container_name)

        host_extra_args = list(getattr(host, "extra_args", None) or []) if host else []
        merged_extra_args = normalize_extra_args([*c.extra_args, *host_extra_args])

        if gpus is not None:
            # A `--gpus` (or NVIDIA_VISIBLE_DEVICES / --device=/dev/nvidia*) in
            # extra_args is appended AFTER this pin, and docker *accumulates*
            # device requests rather than letting the last one win -- so a raw
            # `--gpus all` silently widens the container to every GPU on the host
            # while sflow believes it holds a 2-GPU reservation. Nothing
            # downstream can detect that, so say so loudly. Not fatal: the raw arg
            # is still honored (passthrough is the contract for extra_args), and
            # rejecting it would break recipes that predate the reservation.
            if not self._warned_extra_args_gpus and _grants_gpus(merged_extra_args):
                self._warned_extra_args_gpus = True
                _logger.warning(
                    f"Task '{task_name}': extra_args grant GPUs directly "
                    f"({list(merged_extra_args)}) "
                    f"while sflow pinned this task to {gpus}. Docker ADDS both "
                    f"requests, so the container will see more GPUs than were "
                    f"reserved and may collide with another task. Drop the raw GPU "
                    f"arg and use resources.gpus instead."
                )
            cmd.add_arg("--gpus")
            cmd.add_arg(_quoted_gpus_arg(gpus))
        elif not _grants_gpus(merged_extra_args):
            # No GPUs requested: hide all devices. Many CUDA images bake
            # NVIDIA_VISIBLE_DEVICES=all, so without a --gpus flag the container
            # would otherwise see every GPU on the host. "void" exposes none.
            # Skipped when extra_args already grant GPUs (e.g. `--gpus all` passed
            # raw), where injecting void would silently revoke what was asked for.
            cmd.add_arg("-e")
            cmd.add_arg("NVIDIA_VISIBLE_DEVICES=void")
        if c.workdir is not None:
            cmd.add_arg("-w")
            cmd.add_arg(c.workdir)

        host_mounts = list(getattr(host, "mounts", None) or []) if host else []
        for m in [*c.mounts, *host_mounts]:
            cmd.add_arg("-v")
            cmd.add_arg(m)

        if c.pass_envs:
            for k in dict(envs):
                cmd.add_arg("-e")
                cmd.add_arg(str(k))

        # Passed by value, not by name: these differ per member and so are not in
        # the driver's own environment for a bare `-e NAME` to inherit.
        for key, value in (per_node_envs or {}).items():
            cmd.add_arg("-e")
            cmd.add_arg(f"{key}={value}")

        for a in merged_extra_args:
            cmd.add_arg(a)

        cmd.add_arg(c.image)
        cmd.add_arg("bash")
        cmd.add_arg("-lc")
        cmd.add_arg("\n".join(list(script)))
        return cmd

    def _launch_specs(self, task_name: str) -> list[_Member]:
        """Every container this task runs, in rank order.

        ``build_command`` and :meth:`teardown_commands` both derive container names
        from here so the driver can always reap a container by its ``--name`` even
        after the ``docker run`` client was killed. The name carries the driver
        PID, so it is stable within a run yet unique across concurrent runs.
        """
        launch_nodes = self._assigned_nodes or [""]
        return [
            _Member(
                index=index,
                node_name=node_name,
                host=self._node_hosts.get(node_name),
                container_name=make_container_name(task_name, node_name),
            )
            for index, node_name in enumerate(launch_nodes)
        ]

    def dry_run_launch_preview(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> list[tuple[str, Command]]:
        """One entry per member, so the whole fan-out is reviewable before launch.

        ``build_command`` returns rank 0 alone as a representative, and the
        command log only exists once commands are actually issued -- so without
        this a dry run shows one machine's worth of a job that spans several,
        and never shows which daemon any of them targets.

        Deliberately NOT via ``build_command``: that records every member into
        the active command log, which on a dry run would fabricate a record of
        commands nothing ran.
        """
        members = self._launch_specs(task_name)
        if len(members) <= 1:
            return []
        return [
            (
                f"node {member.node_name} (rank {member.index}) "
                f"-> {_daemon_label(member.host)}",
                command,
            )
            for member, command in self._multi_node_commands(
                task_name=task_name, script=script, envs=envs
            )
        ]

    def teardown_commands(self, *, task_name: str) -> list[Command]:
        """Force-remove this task's containers (one per node).

        Killing the foreground ``docker run`` client never stops the
        daemon-managed container, so the orchestrator runs these after the task
        exits (and before relaunch) to guarantee no container outlives the run.
        """
        commands: list[Command] = []
        for member in self._launch_specs(task_name):
            cmd = _docker_cmd(member.host)
            if self.config.rm:
                cmd.add_arg("rm")
                cmd.add_arg("-f")
                cmd.add_arg(member.container_name)
                commands.append(cmd)
                continue

            # Stopped, never deleted: the point of `rm: false` is to leave the
            # container for `docker logs`/`inspect`, which a force-remove here
            # would undo. Stopping still happens, so a READY service cannot
            # outlive the workflow.
            cmd.add_arg("stop")
            cmd.add_arg(member.container_name)
            commands.append(cmd)

            # Then free the name. `docker run --name X` refuses while ANY
            # container called X exists, in any state -- so a kept container
            # would make every retry of this task die with "name already in
            # use" instead of re-running. Renaming keeps the evidence and
            # unblocks the next attempt.
            rename = _docker_cmd(
                member.host, "rename", member.container_name
            )
            # Nanoseconds, not seconds: two retries of the same task inside one
            # second would otherwise pick the same suffix, `docker rename` would
            # fail with "name already in use", and the stale container would go
            # on blocking the next `--name` -- the exact failure this rename is
            # here to prevent.
            rename.add_arg(f"{member.container_name}-kept-{time.time_ns()}")
            commands.append(rename)
        return commands

    def _multi_node_commands(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> list[tuple[_Member, Command]]:
        """One `docker run` per assigned host, each carrying its own identity.

        Only the rank is per-member. ``SFLOW_TASK_NUM_NODES`` and
        ``SFLOW_LEADER_ADDRESS`` are the same for every member, so they arrive
        with the rest of the task env (assembly sets them for every backend)
        and are forwarded by name like any other variable.
        """
        members = self._launch_specs(task_name)
        commands: list[tuple[_Member, Command]] = []
        for member in members:
            per_node_envs = {"SFLOW_TASK_NODE_INDEX": str(member.index)}
            commands.append(
                (
                    member,
                    self._build_docker_command(
                        task_name=task_name,
                        node_name=member.node_name,
                        host=member.host,
                        script=script,
                        envs=envs,
                        container_name=member.container_name,
                        per_node_envs=per_node_envs,
                    ),
                )
            )
        return commands

    def manages_own_execution(self) -> bool:
        # One container is an ordinary subprocess the orchestrator can launch;
        # several need supervising together, which is what execute() does.
        return len(self._assigned_nodes) > 1

    async def execute(
        self,
        *,
        launcher: Any,
        output_logger: Any,
        env: Mapping[str, str],
        task_name: str,
        script: Sequence[str],
        status_note: Any = None,
    ) -> int:
        """Run every member concurrently; the first failure ends the task.

        Peers are cancelled rather than left running: a job whose rank 1 died is
        not going to finish, and the remaining containers would otherwise hold
        their GPUs on every other machine until someone noticed.
        """
        commands = self._multi_node_commands(
            task_name=task_name, script=script, envs=env
        )
        members = {
            asyncio.ensure_future(
                launcher.run_async(
                    command,
                    output_logger=_NodeLogger(
                        output_logger, member.node_name, member.index
                    ),
                    env=env,
                    task_name=f"{task_name}:{member.node_name}",
                )
            ): member
            for member, command in commands
        }
        pending = set(members)
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done:
                    rc = finished.result()
                    if rc != 0:
                        member = members[finished]
                        output_logger.info(
                            f"sflow: Docker node '{member.node_name}' exited with "
                            f"status {rc}; stopping peer containers."
                        )
                        return rc
            return 0
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    def stale_reap_commands(self, *, task_name: str) -> list[Command]:
        """Reap *orphaned* sflow containers -- ones whose owning driver PID is
        dead -- on each host this task targets.

        Unlike :meth:`teardown_commands` (which removes this run's own containers
        by exact name), this must never touch a container owned by a live driver:
        that could be another ``sflow run`` executing concurrently on the same
        host. So it lists every ``sflow-p<pid>-*`` container, parses the PID from
        the name, and force-removes only those whose PID is no longer alive --
        the same dead-owner reclaim the GPU reservation registry uses.

        Remote daemons are swept too, once each. That is only sound because the
        name carries the driver's machine identity: without it a PID read off a
        co-tenant's container would be checked against *this* host's process
        table and a live container reaped. Exact-name teardown cannot cover this
        case -- a SIGKILLed driver never runs teardown at all, leaving remote
        containers with no other route back.
        """
        commands: list[Command] = []
        swept: set[tuple[str, ...]] = set()
        for member in self._launch_specs(task_name):
            key = tuple(_host_args(member.host))
            if key in swept:
                continue
            swept.add(key)
            commands.append(self._orphan_reap_command(member.host))
        return commands

    def _orphan_reap_command(self, host: Any | None = None) -> Command:
        # For each of this machine's containers on `host`: strip the owner prefix,
        # take the pid up to the next '-', and rm -f only when the owning driver
        # is really gone. Built by orphan_reap_script so CI's after_script sweep
        # (which imports ORPHAN_REAP_SCRIPT) cannot drift from this one.
        cmd = Command(exec="bash")
        cmd.add_arg("-c")
        cmd.add_arg(orphan_reap_script(host))
        return cmd

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        launch_specs = self._launch_specs(task_name)

        if len(launch_specs) == 1:
            member = launch_specs[0]
            # Name the container so the driver can force-remove it on teardown:
            # killing the foreground `docker run` client does not stop the
            # daemon-managed container, so `--rm` alone can leak it.
            cmd = self._build_docker_command(
                task_name=task_name,
                node_name=member.node_name or None,
                host=member.host,
                script=script,
                envs=envs,
                container_name=member.container_name,
            )
            return self._maybe_offload(cmd, task_name=task_name, envs=envs)

        # Multi-node runs through execute(), not this command: a generated shell
        # supervisor could only report a generic failure, could not name the rank
        # that died, and had no way to label output per member. Every member's
        # command is recorded to the command log, so the fan-out stays reviewable;
        # this returns rank 0 as the representative for single-command callers.
        commands = self._multi_node_commands(
            task_name=task_name, script=script, envs=envs
        )
        for member, command in commands:
            record_active_command(
                command,
                task_name=f"{task_name}:{member.node_name}",
                shell=False,
            )
        return commands[0][1]
