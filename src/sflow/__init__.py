# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SFLOW - Workflow Orchestrator with Pluggable Backends
"""

try:
    from sflow._version import version as __version__
except ImportError:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("sflow")
    except PackageNotFoundError:
        __version__ = "unknown"
