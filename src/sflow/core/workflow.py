# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .task import Task, TaskStatus
from .task_graph import TaskGraph


class Workflow:
    def __init__(self, name: str, task_graph: TaskGraph):
        self.name = name
        self.task_graph = task_graph

    def is_finished(self) -> bool:
        return self.task_graph.is_finished()

    def get_task(self, name: str) -> Task:
        return self.task_graph.get_task(name)

    def get_tasks(self) -> list[Task]:
        return self.task_graph.get_tasks()

    def get_tasks_to_submit(self) -> list[Task]:
        return self.task_graph.get_submittable_tasks()

    def get_tasks_to_sync(self) -> list[Task]:
        return [task for task in self.get_tasks() if task.status == TaskStatus.RUNNING]
