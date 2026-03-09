# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict

from .backend import Allocation


@dataclass
class GlobalContext:
    """
    Global context for the workflow execution.
    This class helps constructing the dictionary context required by ExpressionResolver.
    """

    variables: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    backends: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    workflow: Dict[str, Any] = field(default_factory=dict)

    def update_variable_value(self, variable_name: str, value: Any) -> None:
        """
        Update variable value.
        """
        self.variables[variable_name] = value

    def update_backend_allocation(
        self, backend_name: str, allocation: Allocation
    ) -> None:
        """
        Update backend with allocation information after backend allocation completes.

        Args:
            backend_name: Name of the backend
            allocation: Allocation containing allocation_id and node details
        """
        if backend_name not in self.backends:
            self.backends[backend_name] = {}
        # Update the backends dict with allocation info
        self.backends[backend_name]["nodes"] = [n.to_dict() for n in allocation.nodes]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for expression resolution."""
        return {
            "variables": self.variables,
            "artifacts": self.artifacts,
            "backends": self.backends,
            "workflow": self.workflow,
        }


global_context = GlobalContext()
