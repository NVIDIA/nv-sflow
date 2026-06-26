# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Slurm messaging helpers."""

from collections.abc import Callable


def format_gpus_per_node_semantics_warning(gpus_per_node: int) -> str:
    return (
        f"backend.gpus_per_node={gpus_per_node} is sflow planning only "
        "(topology/resource planning and GPU index assignment); it does not add "
        "Slurm --gpus-per-node to salloc, srun, or sbatch. If your cluster "
        "requires that Slurm flag, set it explicitly in backend.extra_args for "
        "salloc, operator.extra_args for srun, or sflow batch "
        "-e/--sbatch-extra-args for sbatch."
    )


def emit_gpus_per_node_semantics_warning(
    gpus_per_node: int | None,
    emit: Callable[[str], None],
    *,
    prefix: str = "",
) -> bool:
    if gpus_per_node is None:
        return False
    emit(f"{prefix}{format_gpus_per_node_semantics_warning(gpus_per_node)}")
    return True


def emit_gpus_per_node_semantics_warning_once(
    gpus_per_node: int | None,
    emit: Callable[[str], None],
    *,
    already_warned: bool,
    prefix: str = "",
) -> bool:
    if already_warned:
        return True
    return emit_gpus_per_node_semantics_warning(
        gpus_per_node,
        emit,
        prefix=prefix,
    )
