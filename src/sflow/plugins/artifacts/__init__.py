# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Artifact plugins.
"""

from __future__ import annotations

# Import modules to register built-in artifact scheme resolvers.
#
# The registry is populated lazily via:
#   sflow.core.artifact_registry.ensure_builtin_artifacts_registered()
#
# Keep imports here (not in core) to avoid core depending on plugins.
from . import docker as _docker  # noqa: F401
from . import http as _http  # noqa: F401
from . import huggingface as _huggingface  # noqa: F401
from . import local as _local  # noqa: F401
