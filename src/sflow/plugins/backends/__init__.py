# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend plugin implementations for sflow.

Importing this package registers built-in backends.
"""

from .local import LocalBackend  # noqa: F401
from .slurm import SlurmBackend  # noqa: F401

"""
Backend plugins.
"""
