# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from logging import Logger
from typing import Any

from sflow.core.command import Command
from sflow.core.operator import Operator
from sflow.core.probe import Probe


class TaskStatus(str, Enum):
    INITIATED = "INITIATED"  # The task has just been initiated, not yet submitted

    RUNNING = "RUNNING"  # The task is running
    READY = "READY"  # The task is ready, indicated by probes, this status is for service type task

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


@dataclass
class Task:
    """
    Execution representation of a task in the workflow.
    """

    name: str
    logger: Logger
    operator: Operator

    status: TaskStatus = TaskStatus.INITIATED
    type: TaskType = TaskType.BATCH
    envs: dict[str, str] = field(default_factory=dict)
    script: list[str] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)

    # Output parsing (MVP): parse from task log and persist outputs.json
    output_specs: list[OutputSpec] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)

    # Planning metadata (helps with dry-run plan rendering / observability).
    backend_name: str | None = None  # backend used for execution/resources
    operator_name: str | None = None  # operator config name (if any)
    # Best-effort assigned nodes for this task (may be empty for local or when not pinned).
    assigned_nodes: list[str] = field(default_factory=list)
    # Sweep variable names for this replica (empty if not a sweep replica).
    sweep_variables: list[str] = field(default_factory=list)

    # Optional retry configuration (see SRD REQ-3.6).
    retries: RetryPolicy | None = None
    # Number of launch attempts made so far (includes the initial attempt).
    attempts: int = 0
    # Wall clock timestamp (time.time()) before which the task must not be re-submitted.
    next_retry_at: float = 0.0
    # Exit code from the most recent subprocess execution (None if never finished yet).
    exit_code: int | None = None

    @cached_property
    def launch_command(self) -> Command:
        return self.operator.build_command(
            task_name=self.name,
            script=self.script,
            envs=self.envs,
        )
