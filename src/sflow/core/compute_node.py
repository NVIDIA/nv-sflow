# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any


@dataclass
class ComputeNode:
    name: str
    ip_address: str
    index: int
    # GPU count available on this node (if known). Used for CUDA_VISIBLE_DEVICES packing/validation.
    num_gpus: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ip_address": self.ip_address,
            "index": self.index,
            "num_gpus": self.num_gpus,
        }
