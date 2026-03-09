# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

class AllocationRequiredError(Exception):
    """
    Exception raised when an allocation is required but not available.
    """
    pass