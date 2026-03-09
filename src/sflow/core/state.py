# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from .artifact import Artifact
from .backend import Backend
from .variable import Variable
from .workflow import Workflow


@dataclass
class SflowState:
    """
    Execution state of a workflow execution.
    """

    workflow: Workflow
    variables: dict[str, Variable] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    backends: dict[str, Backend] = field(default_factory=dict)
    default_backend: Backend | None = None

    def add_variable(self, variable: Variable) -> None:
        """
        Add a variable to the state.
        """
        self.variables[variable.name] = variable

    def add_artifact(self, artifact: Artifact) -> None:
        """
        Add an artifact to the state.
        """
        self.artifacts[artifact.name] = artifact

    def add_backend(self, backend: Backend) -> None:
        """
        Add a backend to the state.
        """
        self.backends[backend.name] = backend

    def to_context_dict(self) -> dict[str, Any]:
        """
        Convert the state to a dictionary for the context.
        """
        return {
            "variables": self.variables,
            "artifacts": self.artifacts,
            "backends": self.backends,
        }
