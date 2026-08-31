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
    # Physical GPU UUIDs on this node, ordered by HOST device index, as read on
    # bare metal before anything was carved. This is ground truth: it is the only
    # way to tell "the right number of GPUs" from "the right GPUs", and it is what
    # lets a task step name the indices its planned cards turned out to have,
    # whatever layer renumbered them. None when the backend could not
    # probe (no GPUs, no nvidia-smi, probe failed) -- callers must degrade, not
    # assume. Per node, not per backend: nodes in one allocation can differ, and
    # two Slurm backends can have different gpus_per_node entirely.
    gpu_uuids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ip_address": self.ip_address,
            "index": self.index,
            "num_gpus": self.num_gpus,
            "gpu_uuids": self.gpu_uuids,
        }
