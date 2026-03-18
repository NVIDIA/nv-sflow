# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from sflow.app.assembly import preflight_validate_container_images
from sflow.config.schema import (
    SflowConfig,
    TaskConfig,
    TaskOperatorOverrideConfig,
    WorkflowConfig,
)
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.variable import Variable, VariableType
from sflow.core.workflow import Workflow


def _state(**variables: str) -> SflowState:
    wf = Workflow(name="wf", task_graph=TaskGraph())
    st = SflowState(workflow=wf)
    for name, value in variables.items():
        st.variables[name] = Variable(name=name, value=value, type=VariableType.STRING)
    return st


def _config(
    *,
    container_image: str | None = None,
    extra_args: list[str] | None = None,
    task_override_image: str | None = None,
    task_override_extra_args: list[str] | None = None,
) -> SflowConfig:
    op_kwargs: dict = {
        "name": "op_srun",
        "type": "srun",
    }
    if container_image is not None:
        op_kwargs["container_image"] = container_image
    if extra_args is not None:
        op_kwargs["extra_args"] = extra_args

    task_kwargs: dict = {"name": "t1", "script": ["echo hi"], "operator": "op_srun"}
    if task_override_image is not None or task_override_extra_args is not None:
        override: dict = {"name": "op_srun"}
        if task_override_image is not None:
            override["container_image"] = task_override_image
        if task_override_extra_args is not None:
            override["extra_args"] = task_override_extra_args
        task_kwargs["operator"] = TaskOperatorOverrideConfig(**override)

    return SflowConfig(
        version="0.1",
        operators=[op_kwargs],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(**task_kwargs)],
        ),
    )


# -- valid cases (should pass without error) ----------------------------------


def test_valid_registry_image():
    preflight_validate_container_images(
        _config(container_image="nvcr.io/nvidia/pytorch:24.01-py3"),
        _state(),
    )


def test_valid_sqsh_image():
    preflight_validate_container_images(
        _config(container_image="/opt/images/container.sqsh"),
        _state(),
    )


def test_valid_image_in_extra_args():
    preflight_validate_container_images(
        _config(extra_args=["--container-image=nvcr.io/nvidia/pytorch:24.01-py3"]),
        _state(),
    )


def test_template_resolves_to_valid_image():
    preflight_validate_container_images(
        _config(container_image="${{ variables.IMG }}"),
        _state(IMG="nvcr.io/nvidia/pytorch:24.01-py3"),
    )


def test_unresolvable_template_is_skipped():
    preflight_validate_container_images(
        _config(container_image="${{ variables.UNKNOWN }}"),
        _state(),
    )


def test_no_container_image_passes():
    preflight_validate_container_images(
        _config(),
        _state(),
    )


def test_placeholder_image_rejected_at_parse_time():
    with pytest.raises(ValidationError, match="container_image.*does not look like"):
        _config(container_image="<your-container-image>")


# -- literal invalid values are caught at config parse time by the Pydantic
#    model validator in SrunOperatorConfig (before preflight even runs). --------


def test_literal_invalid_image_rejected_at_parse_time():
    with pytest.raises(ValidationError, match="container_image.*does not look like"):
        _config(container_image="not a valid image!!")


def test_literal_invalid_extra_args_image_rejected_at_parse_time():
    with pytest.raises(ValidationError, match="extra_args.*does not look like"):
        _config(extra_args=["--container-image=bad image!!"])


# -- template expressions that resolve to invalid values are caught by the
#    preflight check (the unique value-add over the Pydantic validator). --------


def test_template_resolves_to_invalid_image_raises():
    with pytest.raises(ValueError, match="Pre-flight validation failed.*invalid container image"):
        preflight_validate_container_images(
            _config(container_image="${{ variables.IMG }}"),
            _state(IMG="not a valid image!!"),
        )


def test_template_in_extra_args_resolves_to_invalid_raises():
    cfg = _config(extra_args=["--container-image=${{ variables.IMG }}"])
    with pytest.raises(ValueError, match="Pre-flight validation failed.*extra_args.*invalid container image"):
        preflight_validate_container_images(
            cfg,
            _state(IMG="not a valid image!!"),
        )


def test_template_in_extra_args_equals_resolves_to_valid():
    cfg = _config(extra_args=["--container-image=${{ variables.IMG }}"])
    preflight_validate_container_images(
        cfg,
        _state(IMG="nvcr.io/nvidia/pytorch:24.01-py3"),
    )


# -- task-level operator overrides (TaskOperatorOverrideConfig uses extra="allow"
#    so the Pydantic SrunOperatorConfig validator does NOT run for them) --------


def test_invalid_task_override_image_raises():
    with pytest.raises(ValueError, match="Pre-flight validation failed.*task.*operator override.*invalid container image"):
        preflight_validate_container_images(
            _config(
                container_image="nvcr.io/valid:latest",
                task_override_image="not valid!!",
            ),
            _state(),
        )


def test_invalid_task_override_extra_args_raises():
    with pytest.raises(ValueError, match="Pre-flight validation failed.*task.*operator override.*extra_args.*invalid container image"):
        preflight_validate_container_images(
            _config(
                container_image="nvcr.io/valid:latest",
                task_override_extra_args=["--container-image=not valid!!"],
            ),
            _state(),
        )


def test_valid_task_override_image_passes():
    preflight_validate_container_images(
        _config(
            container_image="nvcr.io/valid:latest",
            task_override_image="nvcr.io/also-valid:v2",
        ),
        _state(),
    )
