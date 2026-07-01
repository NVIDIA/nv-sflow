# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from sflow.core.operator_registry import register_operator
from sflow.plugins.operators._k8s_operator import (
    K8sContainerOperator,
    K8sContainerOperatorConfig,
)


class K8sOperatorConfig(K8sContainerOperatorConfig):
    type: Literal["k8s"] = "k8s"


@register_operator("k8s", K8sOperatorConfig)
class K8sOperator(K8sContainerOperator):
    """Render a task into pinned pod(s) and ``kubectl apply`` them.

    A single-node task becomes one Pod; a multi-node task (``resources.nodes``)
    becomes N Pods (leader = index 0) pinned one-per-node, with each pod's GPUs
    from a DRA ResourceClaimTemplate (``scheduling: dra``) or an ``nvidia.com/gpu``
    limit (``scheduling: device_plugin``). Multi-node pods receive
    ``SFLOW_TASK_NODE_INDEX`` and ``SFLOW_LEADER_ADDRESS`` (plus the shared
    ``SFLOW_TASK_ASSIGNED_NODE_IPS``) so peers can rendezvous. Any failed pod
    fails the whole task (a dead sub-pod must not leave the task hanging on a
    still-running peer); otherwise the leader pod (index 0) determines the exit
    code.
    """

    config: K8sOperatorConfig
