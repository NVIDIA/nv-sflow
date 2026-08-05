# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace


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


def test_placeholder_image_warns_at_parse_time(image_warnings):
    _config(container_image="<your-container-image>")
    assert any("<your-container-image>" in m for m in image_warnings), image_warnings


# -- literal unrecognised values are reported at config parse time by the Pydantic
#    model validator in SrunOperatorConfig (before preflight even runs).
#
#    BEHAVIOR CHANGE: these WARN rather than abort. is_valid_container_image is a
#    heuristic and the references a runtime accepts outnumber the shapes it models
#    (pyxis/enroot `registry#path:tag`, `docker://` URIs, site-local schemes), so a
#    rejected recipe had no override short of editing sflow. The runtime reports a
#    genuinely bad reference itself, with a better message than this regex can give.
#    The cost of the change: a typo'd image is no longer caught before allocation. ----


def test_literal_unrecognised_image_warns_at_parse_time(image_warnings):
    _config(container_image="not a valid image!!")
    assert any("container_image" in m for m in image_warnings), image_warnings


def test_literal_unrecognised_extra_args_image_warns_at_parse_time(image_warnings):
    _config(extra_args=["--container-image=bad image!!"])
    assert any("extra_args" in m for m in image_warnings), image_warnings


# -- template expressions that resolve to invalid values are caught by the
#    preflight check (the unique value-add over the Pydantic validator). --------


def test_template_resolves_to_unrecognised_image_warns(image_warnings):
    preflight_validate_container_images(
        _config(container_image="${{ variables.IMG }}"),
        _state(IMG="not a valid image!!"),
    )
    assert any(
        "Pre-flight validation failed" in m and "invalid container image" in m
        for m in image_warnings
    ), image_warnings


def test_template_in_extra_args_resolves_to_unrecognised_warns(image_warnings):
    cfg = _config(extra_args=["--container-image=${{ variables.IMG }}"])
    preflight_validate_container_images(cfg, _state(IMG="not a valid image!!"))
    assert any("operator 'op_srun'" in m for m in image_warnings), image_warnings


def test_template_in_extra_args_equals_resolves_to_valid():
    cfg = _config(extra_args=["--container-image=${{ variables.IMG }}"])
    preflight_validate_container_images(
        cfg,
        _state(IMG="nvcr.io/nvidia/pytorch:24.01-py3"),
    )


# -- task-level operator overrides (TaskOperatorOverrideConfig uses extra="allow"
#    so the Pydantic SrunOperatorConfig validator does NOT run for them) --------


def test_unrecognised_task_override_image_warns(image_warnings):
    preflight_validate_container_images(
        _config(
            container_image="nvcr.io/valid:latest",
            task_override_image="not valid!!",
        ),
        _state(),
    )
    assert any(
        "task" in m and "operator override" in m for m in image_warnings
    ), image_warnings


def test_unrecognised_task_override_extra_args_warns(image_warnings):
    preflight_validate_container_images(
        _config(
            container_image="nvcr.io/valid:latest",
            task_override_extra_args=["--container-image=not valid!!"],
        ),
        _state(),
    )
    assert any(
        "task" in m and "operator override" in m for m in image_warnings
    ), image_warnings


def test_valid_task_override_image_passes():
    preflight_validate_container_images(
        _config(
            container_image="nvcr.io/valid:latest",
            task_override_image="nvcr.io/also-valid:v2",
        ),
        _state(),
    )


def test_unrecognised_docker_backend_default_image_warns(image_warnings):
    cfg = SflowConfig(
        version="0.1",
        backends=[
            {
                "name": "docker",
                "type": "docker",
                "default": True,
                "image": "${{ variables.IMG }}",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    preflight_validate_container_images(cfg, _state(IMG="not valid!!"))
    assert any("backend 'docker'" in m for m in image_warnings), image_warnings


def test_kubernetes_backend_contributes_no_image_to_preflight():
    # The kubernetes backend has no `image` field anymore (workload images come
    # from operators), so a stray `image` is dropped and never validated --
    # preflight must NOT raise on it.
    cfg = SflowConfig(
        version="0.1",
        backends=[
            {
                "name": "k8s",
                "type": "kubernetes",
                "default": True,
                "image": "${{ variables.IMG }}",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    assert cfg.backends[0].container_images() == []
    # No raise even with an otherwise-invalid image value.
    preflight_validate_container_images(cfg, _state(IMG="not valid!!"))


def test_backend_config_exposes_container_images_for_default_operator_preflight():
    cfg = SflowConfig(
        version="0.1",
        backends=[
            {
                "name": "docker",
                "type": "docker",
                "default": True,
                "image": "nvcr.io/example/app:1.0",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    assert cfg.backends[0].container_images() == ["nvcr.io/example/app:1.0"]


def test_operator_container_images_hook_is_validated_even_with_primary_image_attr(
    image_warnings,
):
    class CustomImageOperatorConfig:
        name = "op_custom"
        image = "nvcr.io/example/app:1.0"
        extra_args: list[str] = []

        def container_images(self) -> list[str]:
            return [self.image, "not valid!!"]

    cfg = SimpleNamespace(
        operators=[CustomImageOperatorConfig()],
        backends=[],
        workflow=SimpleNamespace(tasks=[]),
    )

    preflight_validate_container_images(cfg, _state())
    assert any(
        "operator 'op_custom'" in m and "invalid container image" in m
        for m in image_warnings
    ), image_warnings
