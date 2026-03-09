# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sflow.config.schema import SflowConfig
from sflow.core.backend_registry import (
    ensure_builtin_backends_registered,
    get_backend_registry,
)


def _minimal_cfg_dict(**extra):
    base = {
        "version": "0.1",
        "workflow": {"name": "wf", "tasks": [{"name": "t1", "script": ["echo 1"]}]},
    }
    base.update(extra)
    return base


def test_backend_registry_has_builtins():
    ensure_builtin_backends_registered()
    reg = get_backend_registry()
    for t in ["local", "slurm"]:
        assert t in reg


def test_config_rejects_unknown_backend_type():
    with pytest.raises(ValueError):
        SflowConfig.model_validate(
            _minimal_cfg_dict(
                backends={"b1": {"type": "not_a_real_backend", "default": True}}
            )
        )
