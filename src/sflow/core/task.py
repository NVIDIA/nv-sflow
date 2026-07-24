# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from logging import Logger
from typing import TYPE_CHECKING, Any, Literal

from sflow.core.command import Command
from sflow.core.operator import Operator
from sflow.core.probe import Probe
from sflow.utils.script import prepend_fail_fast

if TYPE_CHECKING:
    from sflow.core.monitor import MonitorConsumer


class TaskStatus(str, Enum):
    INITIATED = "INITIATED"  # The task has just been initiated, not yet submitted

    RUNNING = "RUNNING"  # The task is running
    READY = "READY"  # The task is ready, indicated by probes, this status is for service type task
    FINALIZING = "FINALIZING"  # The process exited; post-processing is still running

    COMPLETED = "COMPLETED"  # The task has completed successfully
    FAILED = "FAILED"  # The task has failed
    TIMEOUT = "TIMEOUT"  # The task has timed out
    CANCELLED = "CANCELLED"  # The task has been cancelled

    def __str__(self) -> str:
        return self.value

    def is_terminal(self) -> bool:
        return self in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.CANCELLED,
            TaskStatus.READY,
        }


class TaskType(str, Enum):
    SERVICE = "SERVICE"
    BATCH = "BATCH"

    def __str__(self) -> str:
        return self.value


@dataclass
class RetryPolicy:
    """
    Retry policy for a task.

    Semantics:
    - `count` is the number of retries after the initial attempt (total attempts = 1 + count).
    - `interval` is the initial delay (seconds) before the first retry.
    - `backoff` multiplies the delay for each subsequent retry attempt.
    """

    count: int
    interval: float
    backoff: float = 1.0


@dataclass(frozen=True)
class OutputSpec:
    """
    Output parsing specification for a task.

    `pattern` follows the `parse` library format used elsewhere in sflow, e.g.:
      "TTFT: {ttft:f} ms"
    """

    pattern: str
    source: str = "stdout"  # MVP: logs are merged; kept for schema parity.


@dataclass(frozen=True)
class ResultSpec:
    """
    Per-pattern specification for the new ``result`` parsing entry.

    Mirrors the YAML ``result.patterns[]`` schema and the legacy ``outputs[]``
    rows so a single internal type can serve both contracts during migration.

    Attributes:
        name: Result key written under ``values`` in ``result.json``.
        regex: Python regex; preferred form. Either one positional capture group
            or a named ``value`` capture group.
        pattern: Legacy ``parse``-style pattern (used when ``engine == "parse"``).
        engine: ``"regex"`` (new contract) or ``"parse"`` (legacy ``outputs`` migration).
        source: Source selector. Initially only ``"log"`` is implemented.
        type: One of ``auto``, ``string``, ``int``, ``float``, ``bool``, ``json``.
        unit: Optional metadata unit, e.g. ``ms``, ``tok/s``.
        aggregate: One of ``first``, ``last``, ``list``, ``count``, ``min``, ``max``,
            ``avg``, ``sum``. Default ``last``.
        required: When True, missing matches make parsing for this spec unsuccessful.
        group: Capture group name or index to extract from the regex.
    """

    name: str
    regex: str | None = None
    # Reserved for the future `parse` engine migration of legacy `outputs[*].pattern`.
    # Unused by the regex engine; kept here so a single ResultSpec serves both contracts.
    pattern: str | None = None
    engine: Literal["regex", "parse"] = "regex"
    source: str = "log"
    type: str = "auto"
    unit: str | None = None
    aggregate: str = "last"
    required: bool = False
    group: str | int | None = None


@dataclass(frozen=True)
class ResultConfigRuntime:
    """
    Runtime representation of the consolidated ``result`` task entry.

    Either ``specs`` (regex/parse patterns) or ``file`` (source JSON path) is set.
    For the first implementation, mixing both is rejected at schema validation time.
    """

    specs: list[ResultSpec] = field(default_factory=list)
    file: str | None = None
    source: str = "log"


@dataclass(frozen=True)
class ResolvedUpload:
    """
    A per-task upload spec attached to a runtime Task.

    `from_expr` and `to_expr` may contain ${{ }} expressions; they are resolved
    at upload time (after the task completes) so references like
    ${{ task.output_dir }} have values.
    """

    target: str
    from_expr: str
    to_expr: str | None = None
    on_error: Literal["warn", "fail"] = "warn"
    # When set, the remote key's basename is auto-suffixed with `_<disambiguate_with>`
    # (inserted before the file extension) so replicas of the same task don't
    # overwrite each other on the storage target. Decided at assembly time: it holds
    # the replica's name for replicated tasks, and stays None for non-replicated tasks
    # or when the user already references ${{ task.name }} in `to:`.
    disambiguate_with: str | None = None


@dataclass(frozen=True)
class TaskPort:
    """A resolved service port (from task.ports)."""

    port: int
    name: str | None = None


@dataclass
class Task:
    """
    Execution representation of a task in the workflow.
    """

    name: str
    logger: Logger
    operator: Operator

    status: TaskStatus = TaskStatus.INITIATED
    # Optional live sub-status shown next to `status` (e.g. a k8s pod's
    # "Pending: Unschedulable" while the task is RUNNING but not yet started).
    # Backend-agnostic: operators set it via the execute() status_note callback.
    status_detail: str | None = None
    type: TaskType = TaskType.BATCH
    envs: dict[str, str] = field(default_factory=dict)
    script: list[str] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)

    # Output parsing (MVP): parse from task log and persist outputs.json
    output_specs: list[OutputSpec] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)

    # New consolidated result parsing (see docs/developer/dev-notes/result-parsing.md).
    # When set, sflow writes a per-task ${SFLOW_TASK_OUTPUT_DIR}/result.json after the task
    # completes successfully and updates the workflow-level results.json index.
    result_config: "ResultConfigRuntime | None" = None
    result: dict[str, Any] = field(default_factory=dict)

    # Post-completion uploads to named storage targets (S3 etc.).
    uploads: list[ResolvedUpload] = field(default_factory=list)

    # Planning metadata (helps with dry-run plan rendering / observability).
    backend_name: str | None = None  # backend used for execution/resources
    operator_name: str | None = None  # operator config name (if any)
    # Config task name this runtime task derives from: == name for non-replicated
    # tasks, the base task name for replicas (e.g. "server_0" -> "server"). Set at
    # assembly time so consumers don't re-derive it from the replica name string.
    base_name: str | None = None
    # Best-effort assigned nodes for this task (may be empty for local or when not pinned).
    assigned_nodes: list[str] = field(default_factory=list)
    # The planner-computed GPU slice (CUDA_VISIBLE_DEVICES), calculated uniformly
    # for every backend. Kept here for the dry-run allocation map even when it is
    # NOT injected into the execution env (e.g. Kubernetes, where the cluster/DRA
    # assigns the physical devices); env injection is gated by Backend.resource_env.
    cuda_visible_devices: str | None = None
    # task.ports; feeds task.<name>.service.
    ports: list[TaskPort] = field(default_factory=list)
    # Sweep variable names for this replica (empty if not a sweep replica).
    sweep_variables: list[str] = field(default_factory=list)
    # Resource lifetime policies used by dry-run/rehearsal reporting.
    resource_release_after: dict[str, str] = field(default_factory=dict)

    # Task names that should mirror this task's readiness/failure probe result.
    # Populated when HTTP probes are deduplicated across replicas with identical check info.
    readiness_followers: list[str] = field(default_factory=list)
    failure_followers: list[str] = field(default_factory=list)

    # Optional retry configuration (see SRD REQ-3.6).
    retries: RetryPolicy | None = None
    # Number of launch attempts made so far (includes the initial attempt).
    attempts: int = 0
    # Wall clock timestamp (time.time()) before which the task must not be re-submitted.
    next_retry_at: float = 0.0
    # Exit code from the most recent subprocess execution (None if never finished yet).
    exit_code: int | None = None
    # Set to True when the task was terminated by a failure probe (not a process crash).
    failed_by_probe: bool = False
    # Opt in to fail-fast for the task's shell script: assembly prepends ``set -e``
    # for shell operators so a failed command fails the task (instead of a later
    # successful command masking it). Default False keeps the shell default (only the
    # LAST command's exit code counts), so existing recipes are unchanged. See
    # ``config.schema.TaskConfig.fail_fast`` + ``utils.script``.
    fail_fast: bool = False

    # Resolved hardware monitor bound to this task (plan-time). The orchestrator
    # acquires it when the task starts and releases it when the task's process
    # exits / at workflow teardown.
    monitor: "MonitorConsumer | None" = None

    # --- Merge-pod mode (Kubernetes): co-located GPU tasks share one pod ---
    # Stable id shared by every member of one merge group (backend+node scoped).
    merge_group_id: str | None = None
    # Set on a merge FOLLOWER to the leader task's name (the task that owns and
    # launches the shared pod); None on the leader and on non-merged tasks. A
    # follower is never launched on its own -- the leader runs its script as a
    # background process in the merged container and the orchestrator mirrors the
    # follower's lifecycle off the leader while keeping its own log/probes.
    merge_leader: str | None = None
    # Non-empty on the merge LEADER: ordered member task names (leader first) that
    # run in the leader's single merged pod/container. Empty otherwise.
    merge_members: list[str] = field(default_factory=list)
    # Packed CUDA_VISIBLE_DEVICES slice for this member within the merged
    # container's union GPU range (e.g. "0,1"); None when not merged.
    merge_cuda_visible_devices: str | None = None
    # Set on a merged member to the sorted names of the DIRECT in-group members it
    # depends on. Its in-pod subshell (see merged_launcher_lines) blocks until each
    # is met -- COMPLETED (its exit-code file) or READY (a driver-touched marker) --
    # before running. Empty for non-dependent members. See _plan_merge_groups.
    merge_gate_after: list[str] = field(default_factory=list)

    @property
    def is_merge_leader(self) -> bool:
        """True when this task owns a merged pod running several members' scripts."""
        return bool(self.merge_members)

    @property
    def is_merge_follower(self) -> bool:
        """True when this task runs inside another (leader) task's merged pod."""
        return self.merge_leader is not None

    @property
    def runnable_script(self) -> list[str]:
        """The task's script as it will actually RUN (execution paths use this).

        For a shell operator (see :meth:`Operator.runs_shell_script`) with
        ``fail_fast`` opted in (off by default), prepends ``set -e`` so a failed
        command fails the task instead of a later successful command (e.g. a trailing
        ``echo``) masking it. The ``python`` operator (script is Python source) and
        the default ``fail_fast: false`` return the script unchanged. ``script`` itself stays the
        user's resolved lines -- only the runnable form carries the prelude, so the
        build_command path, the orchestrator's ``execute`` call, and the k8s
        merged-member gather all read THIS.
        """
        script = list(self.script)
        if (
            self.fail_fast
            and self.operator is not None
            and self.operator.runs_shell_script()
        ):
            return prepend_fail_fast(script)
        return script

    @cached_property
    def launch_command(self) -> Command:
        return self.operator.build_command(
            task_name=self.name,
            script=self.runnable_script,
            envs=self.envs,
        )
