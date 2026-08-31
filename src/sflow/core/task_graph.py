# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from sflow.core.dag import DAG
from sflow.core.task import Task, TaskStatus
from sflow.logging import get_logger

_logger = get_logger(__name__)


class TaskGraph:
    """
    Represents the execution DAG of a workflow.
    """

    def __init__(self):
        self.dag = DAG()

    def get_task(self, task_name: str) -> Task:
        """Get task object by name."""
        return self.dag.nodes[task_name]

    def _get_dependencies(self, task_name: str) -> List[Task]:
        """Get dependencies for a task."""
        return [
            self.dag.nodes[dep_name]
            for dep_name in self.dag.get_dependencies(task_name)
        ]

    def get_tasks(self) -> List[Task]:
        """Get all tasks in the graph."""
        return list(self.dag.nodes.values())

    def is_finished(self) -> bool:
        """Check if all tasks have reached a terminal state."""
        for task in self.get_tasks():
            if not task.status.is_terminal():
                return False
        return True

    def get_submittable_tasks(self) -> List[Task]:
        """Get tasks that are submittable.

        A task is submittable when:
        - Its status is UNSUBMITTED
        - All its dependencies are in a terminal state (COMPLETED or READY)
        """
        to_submit = []
        for task in self.get_tasks():
            if task.status != TaskStatus.INITIATED:
                continue

            # Check if all dependencies are satisfied
            dependencies = self._get_dependencies(task.name)
            all_deps_satisfied = True
            for dep_task in dependencies:
                # A dependency is satisfied if it's COMPLETED or READY
                if dep_task.status not in (TaskStatus.COMPLETED, TaskStatus.READY):
                    all_deps_satisfied = False
                    break

            if all_deps_satisfied:
                to_submit.append(task)

        return to_submit

    def update_task_status(
        self,
        task_name: str,
        status: TaskStatus | str,
    ) -> None:
        """Update the status of a task.

        Args:
            task_name: Name of the task to update
            status: New status (can be TaskStatus enum or string)
        """
        task = self.dag.nodes[task_name]

        if isinstance(status, str):
            status = TaskStatus(status)

        task.status = status

        _logger.debug(f"Task '{task_name}' status updated to {status}")

    def mark_all_cancelled(self) -> None:
        """Mark all non-terminal tasks as cancelled."""
        for task in self.get_tasks():
            if task.status not in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.TIMEOUT,
                TaskStatus.CANCELLED,
            ]:
                task.status = TaskStatus.CANCELLED
                _logger.debug(f"Task '{task.name}' marked as CANCELLED")
