# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sflow.config.schema import SflowConfig
from sflow.core.operator_registry import (
    ensure_builtin_operators_registered,
    get_operator_registry,
)


def _minimal_cfg_dict(**extra):
    base = {
        "version": "0.1",
        "workflow": {"name": "wf", "tasks": [{"name": "t1", "script": ["echo 1"]}]},
    }
    base.update(extra)
    return base


def test_operator_registry_has_builtins():
    ensure_builtin_operators_registered()
    reg = get_operator_registry()
    for t in ["bash", "srun", "ssh", "docker", "python"]:
        assert t in reg


def test_config_validates_registered_operator_config_from_dict_form():
    cfg = SflowConfig.model_validate(
        _minimal_cfg_dict(
            operators={
                "op_bash": {"type": "bash"},
                "op_python": {"type": "python", "python_exec": "python3"},
            }
        )
    )

    assert cfg.operators is not None
    assert {o.name for o in cfg.operators} == {"op_bash", "op_python"}
    assert {o.type for o in cfg.operators} == {"bash", "python"}


def test_config_rejects_unknown_operator_type():
    with pytest.raises(ValueError):
        SflowConfig.model_validate(
            _minimal_cfg_dict(operators={"bad": {"type": "not_a_real_operator"}})
        )
