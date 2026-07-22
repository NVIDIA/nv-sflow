# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Operator plugin implementations for sflow.
"""

from .bash import BashOperator, BashOperatorConfig
from .docker_run import DockerRunOperator, DockerRunOperatorConfig
from .k8s import K8sOperator, K8sOperatorConfig
from .k8s_mpi import K8sMpiOperator, K8sMpiOperatorConfig
from .python import PythonOperator, PythonOperatorConfig
from .srun import SrunOperator, SrunOperatorConfig
from .ssh import SshOperator, SshOperatorConfig

__all__ = [
    "BashOperator",
    "BashOperatorConfig",
    "DockerRunOperator",
    "DockerRunOperatorConfig",
    "K8sOperator",
    "K8sOperatorConfig",
    "K8sMpiOperator",
    "K8sMpiOperatorConfig",
    "PythonOperator",
    "PythonOperatorConfig",
    "SshOperator",
    "SshOperatorConfig",
    "SrunOperator",
    "SrunOperatorConfig",
]
