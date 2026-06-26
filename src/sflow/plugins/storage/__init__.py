# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Storage plugins.
"""

from __future__ import annotations

# Import modules to register built-in storage target implementations.
#
# The registry is populated lazily via:
#   sflow.core.storage_registry.ensure_builtin_storage_registered()
#
# Keep imports here (not in core) to avoid core depending on plugins.
from . import s3 as _s3  # noqa: F401
