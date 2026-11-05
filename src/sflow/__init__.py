# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SFLOW - Workflow Orchestrator with Pluggable Backends
"""

from sflow.utils.distribution import installed_distribution

try:
    from sflow._version import version as __version__
except ImportError:
    _dist = installed_distribution()
    __version__ = _dist.version if _dist is not None else "unknown"
