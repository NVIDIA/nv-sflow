# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import re
import shlex
import time
from collections.abc import Mapping, Sequence
from typing import Any
from typing import Literal

from pydantic import Field, field_validator

from sflow.core.command import Command
from sflow.core.log_offload import offload_enabled, task_log_path, wrap_with_prefixer
from sflow.core.operator import Operator, OperatorConfig, ResourcesUnavailable
from sflow.core.operator_registry import register_operator
from sflow.logging import get_logger
from sflow.utils import gpu_reservation as gpu_res
from sflow.utils.container import (
    append_runtime_mounts as append_runtime_mount_specs,
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

# Reaps orphaned sflow containers: every `sflow-p<pid>-*` whose owning driver
# process is gone. Defined here, as data, because `.gitlab-ci.yml` runs the exact
# same sweep in its after_script -- two hand-maintained copies of a predicate this
# sharp (it force-removes containers) will drift, and the drift is only visible
# when it wrongly kills a live co-tenant's run. CI executes it via:
#
#     python -c "from sflow.plugins.operators.docker_run import ORPHAN_REAP_SCRIPT
#                print(ORPHAN_REAP_SCRIPT)" | bash
#
# Liveness must be user-agnostic: `kill -0 <pid>` fails with EPERM for a process
# owned by ANOTHER user, so on its own it would classify a co-tenant's live
# `sflow run` as dead and force-remove its running containers. /proc/<pid> exists
# regardless of ownership, so check it first and keep kill -0 as the fallback for
# non-procfs systems.
ORPHAN_REAP_SCRIPT = (
    f'for c in $(docker ps -a --filter name={_CONTAINER_PREFIX}-p '
    '--format "{{.Names}}" 2>/dev/null); do '
    'pid=${c#' + _CONTAINER_PREFIX + '-p}; pid=${pid%%-*}; '
    'case "$pid" in ""|*[!0-9]*) continue;; esac; '
    '[ -d "/proc/$pid" ] && continue; '
    'kill -0 "$pid" 2>/dev/null && continue; '
    'docker rm -f "$c" >/dev/null 2>&1 || true; '
    'done; true'
)


def _safe_container_name(*parts: str) -> str:
    raw = "-".join(part for part in parts if part)
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw).strip("-_.")
    return (sanitized or "sflow-task")[:128]


def _container_name(task_name: str, node_name: str) -> str:
    """``sflow-p<pid>-<task>-<node>`` -- unique per driver process, reap-parseable."""
    return _safe_container_name(
        _CONTAINER_PREFIX, f"p{os.getpid()}", task_name, node_name
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
    pass_envs: bool = True
    auto_mount_runtime_dirs: bool = True
    # Per-task log offload, ON by default. When enabled (also via the
    # SFLOW_OFFLOAD_TASK_LOGS env / --offload-task-logs flag, which take
    # precedence), the task's output is redirected on the host through a
    # compute-side prefixer into <task>.log instead of streaming through the
    # sflow driver's pump. Auto-falls back to streaming on an interactive
    # TTY / --tui session.
    log_to_file: bool = True

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
        return offload_enabled(self.config.log_to_file)

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
        self._targets_remote_host = any(
            host is not None
            and (getattr(host, "docker_host", None) or getattr(host, "context", None))
            for host in self._node_hosts.values()
        )
        if self._targets_remote_host:
            self.config.auto_mount_runtime_dirs = False
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
        # physical GPUs it claimed; when there is none -- remote hosts, synthetic
        # multi-node, SFLOW_GPU_RESERVATION=0, or no readable nvidia-smi -- this
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
        remote hosts (can't read this host's nvidia-smi), synthetic multi-node
        placements, and reservation-disabled runs skip it and keep the plan-time
        device slice set in :meth:`apply_backend_context`.

        Makes exactly ONE attempt. When the board is full and this task is allowed
        to wait, it raises :class:`ResourcesUnavailable` so the orchestrator can
        retry on the event loop -- sleeping here would block an uncancellable
        worker thread and hang the driver on Ctrl-C.
        """
        if self._gpu_count <= 0 or not gpu_res.reservation_enabled():
            return None
        if self._targets_remote_host or len(self._assigned_nodes) > 1:
            # Multi-container placements pin every container to the SAME plan-time
            # device slice (`--gpus` is one operator-config value shared by all
            # launch specs), so there is no per-container claim to make here.
            # Remote hosts additionally can't be judged from this host's
            # nvidia-smi. Worth saying out loud rather than silently returning:
            # with synthetic `nodes: > 1` on the LOCAL daemon every container runs
            # on this very host, so the user reasonably expects the reservation
            # they read about in the docs, and gets none.
            if not self._warned_reservation_skipped:
                self._warned_reservation_skipped = True
                why = (
                    "it targets a remote Docker host"
                    if self._targets_remote_host
                    else f"it spans {len(self._assigned_nodes)} nodes"
                )
                _logger.warning(
                    f"Task '{task_name}': GPU reservation skipped because {why}; "
                    f"launching on the planned device slice ({self.config.gpus}) "
                    f"without a cross-process claim. A concurrent sflow run on the "
                    f"same host can pick the same devices."
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
    ) -> Command:
        c = self.config
        cmd = Command(exec="docker")
        if host is not None:
            if getattr(host, "docker_host", None):
                cmd.add_arg("--host")
                cmd.add_arg(str(host.docker_host))
            elif getattr(host, "context", None):
                cmd.add_arg("--context")
                cmd.add_arg(str(host.context))

        cmd.add_arg("run")
        cmd.add_arg("--rm")
        if container_name:
            cmd.add_arg("--name")
            cmd.add_arg(container_name)

        host_extra_args = list(getattr(host, "extra_args", None) or []) if host else []
        merged_extra_args = normalize_extra_args([*c.extra_args, *host_extra_args])

        if c.gpus is not None:
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
                    f"while sflow pinned this task to {c.gpus}. Docker ADDS both "
                    f"requests, so the container will see more GPUs than were "
                    f"reserved and may collide with another task. Drop the raw GPU "
                    f"arg and use resources.gpus instead."
                )
            cmd.add_arg("--gpus")
            cmd.add_arg(_quoted_gpus_arg(c.gpus))
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
            for k in dict(envs).keys():
                cmd.add_arg("-e")
                cmd.add_arg(str(k))

        for a in merged_extra_args:
            cmd.add_arg(a)

        cmd.add_arg(c.image)
        cmd.add_arg("bash")
        cmd.add_arg("-lc")
        cmd.add_arg("\n".join(list(script)))
        return cmd

    def _cleanup_command(self, host: Any | None, container_name: str) -> str:
        parts = ["docker"]
        if host is not None:
            if getattr(host, "docker_host", None):
                parts.extend(["--host", str(host.docker_host)])
            elif getattr(host, "context", None):
                parts.extend(["--context", str(host.context)])
        parts.extend(["rm", "-f", container_name])
        return shlex.join(parts)

    def _launch_specs(self, task_name: str) -> list[tuple[str, Any | None, str]]:
        """``(node_name, host, container_name)`` for every container this task runs.

        ``build_command`` and :meth:`teardown_commands` both derive container names
        from here so the driver can always reap a container by its ``--name`` even
        after the ``docker run`` client was killed. The name carries the driver
        PID, so it is stable within a run yet unique across concurrent runs.
        """
        launch_nodes = self._assigned_nodes or [""]
        return [
            (
                node_name,
                self._node_hosts.get(node_name),
                _container_name(task_name, node_name),
            )
            for node_name in launch_nodes
        ]

    def teardown_commands(self, *, task_name: str) -> list[Command]:
        """Force-remove this task's containers (one per node).

        Killing the foreground ``docker run`` client never stops the
        daemon-managed container, so the orchestrator runs these after the task
        exits (and before relaunch) to guarantee no container outlives the run.
        """
        commands: list[Command] = []
        for _node_name, host, container_name in self._launch_specs(task_name):
            cmd = Command(exec="docker")
            if host is not None:
                if getattr(host, "docker_host", None):
                    cmd.add_arg("--host")
                    cmd.add_arg(str(host.docker_host))
                elif getattr(host, "context", None):
                    cmd.add_arg("--context")
                    cmd.add_arg(str(host.context))
            cmd.add_arg("rm")
            cmd.add_arg("-f")
            cmd.add_arg(container_name)
            commands.append(cmd)
        return commands

    def stale_reap_commands(self, *, task_name: str) -> list[Command]:
        """Reap *orphaned* sflow containers -- ones whose owning driver PID is
        dead -- on each host this task targets.

        Unlike :meth:`teardown_commands` (which removes this run's own containers
        by exact name), this must never touch a container owned by a live driver:
        that could be another ``sflow run`` executing concurrently on the same
        host. So it lists every ``sflow-p<pid>-*`` container, parses the PID from
        the name, and force-removes only those whose PID is no longer alive --
        the same dead-owner reclaim the GPU reservation registry uses.

        Local daemon only: the ``kill -0`` check reads *this* driver host's
        process table, so on a shared *remote* daemon it could wrongly judge
        another machine's live container dead and reap it. Remote containers still
        get their exact-name teardown after each task.
        """
        has_local = any(
            host is None for _node, host, _c in self._launch_specs(task_name)
        )
        return [self._orphan_reap_command()] if has_local else []

    def _orphan_reap_command(self) -> Command:
        # For each local sflow-p<pid>-* container: strip the prefix, take the pid
        # up to the next '-', and rm -f only when its owning driver is really gone.
        # The script itself is ORPHAN_REAP_SCRIPT so CI's after_script sweep and
        # this one cannot drift apart.
        cmd = Command(exec="bash")
        cmd.add_arg("-c")
        cmd.add_arg(ORPHAN_REAP_SCRIPT)
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
            node_name, host, container_name = launch_specs[0]
            # Name the container so the driver can force-remove it on teardown:
            # killing the foreground `docker run` client does not stop the
            # daemon-managed container, so `--rm` alone can leak it.
            cmd = self._build_docker_command(
                task_name=task_name,
                node_name=node_name or None,
                host=host,
                script=script,
                envs=envs,
                container_name=container_name,
            )
            return self._maybe_offload(cmd, task_name=task_name, envs=envs)

        lines = [
            "set -euo pipefail",
            "status=0",
            "pids=\"\"",
            "cleanup() {",
        ]
        run_lines: list[str] = []
        for node_name, host, container_name in launch_specs:
            lines.append(
                f"  {self._cleanup_command(host, container_name)} >/dev/null 2>&1 || true"
            )
            docker_cmd = self._build_docker_command(
                task_name=task_name,
                node_name=node_name,
                host=host,
                script=script,
                envs=envs,
                container_name=container_name,
            )
            run_lines.append(f"{shlex.join(docker_cmd.as_list())} &")
            run_lines.append('pids="$pids $!"')
        lines.extend(
            [
                "}",
                "trap cleanup EXIT",
                "trap 'cleanup; exit 143' HUP INT TERM",
                *run_lines,
                "for pid in $pids; do",
                "  if ! wait \"$pid\"; then status=1; fi",
                "done",
                "exit \"$status\"",
            ]
        )
        cmd = Command(exec="bash")
        cmd.add_arg("-lc")
        cmd.add_arg("\n".join(lines))
        return self._maybe_offload(cmd, task_name=task_name, envs=envs)
