# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def parse_cuda_visible_devices(cuda_visible: str | None) -> list[int]:
    """
    Parse CUDA_VISIBLE_DEVICES into a list of GPU indices.

    Supports comma-separated indices and simple ranges like ``0-3``.
    Non-numeric tokens are ignored.
    """
    if not cuda_visible:
        return []

    indices: list[int] = []
    for part in str(cuda_visible).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            try:
                start_i = int(start_s)
                end_i = int(end_s)
            except ValueError:
                continue
            if start_i <= end_i:
                indices.extend(range(start_i, end_i + 1))
            continue
        try:
            indices.append(int(token))
        except ValueError:
            continue
    return indices
