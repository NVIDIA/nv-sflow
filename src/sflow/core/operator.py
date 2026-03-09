# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from pydantic import BaseModel

from sflow.core.command import Command


class OperatorConfig(BaseModel):
    """
    Base configuration for an Operator.

    This is intentionally minimal for Step A: we are introducing the concept without
    wiring it into the main YAML schema yet.
    """

    type: str


class Operator(ABC):
    """
    Abstract base class for task execution operators (Airflow-style).

    Contract:
    - Operator instances are configured ONLY via an OperatorConfig object.
    - Operators must be able to build a launch Command from script/envs.
    """

    def __init__(self, config: OperatorConfig):
        self.config = config

    @abstractmethod
    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        """
        Build the command that launches the task.
        """
        raise NotImplementedError
