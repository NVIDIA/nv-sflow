# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from sflow.plugins.operators.srun import (
    SrunOperator,
    SrunOperatorConfig,
)
from sflow.utils.container import is_valid_container_image


def test_srun_operator_supports_pyxis_container_image_flags_and_common_args():
    op = SrunOperator(
        SrunOperatorConfig(
            name="op_srun",
            job_id="123",
            nodelist=["n1", "n2"],
            partition="batch",
            account="acct",
            qos="q1",
            time="00:10:00",
            cpus_per_task=4,
            gpus="1",
            mem="8G",
            container_image="nvcr.io/a/b:1",
            container_mount_home=True,
            container_writable=True,
            container_mounts=["/host:/ctr:rw", "/data:/data:ro"],
            container_workdir="/workspace",
            export="ALL",
        )
    )

    cmd = op.build_command(
        task_name="t1",
        script=["echo hi"],
        envs={"FOO": "bar"},
    )

    # Don't assert exact ordering of every option; just verify key substrings exist.
    s = str(cmd)
    for token in [
        "--jobid 123",
        "--nodes 2",
        "--nodelist n1,n2",
        "--partition batch",
        "--account acct",
        "--qos q1",
        "--time 00:10:00",
        "--cpus-per-task 4",
        "--gpus 1",
        "--mem 8G",
        "--container-image nvcr.io/a/b:1",
        "--container-mount-home",
        "--container-writable",
        "--container-mounts /host:/ctr:rw,/data:/data:ro",
        "--container-workdir /workspace",
    ]:
        assert token in s

    # Payload should be executed via a bash -c wrapper.
    assert " bash -c " in s
    # Env is injected by SubprocessLauncher(env=...) and propagated to Slurm tasks via srun --export=ALL.
    assert "echo hi" in s


def test_srun_operator_supports_pyxis_container_name_flags():
    op = SrunOperator(
        SrunOperatorConfig(
            name="op_srun",
            container_name="cname",
        )
    )

    s = str(
        op.build_command(
            task_name="t1",
            script=["echo hi"],
            envs={},
        )
    )
    assert "--container-name cname" in s


def test_srun_operator_maps_slurm_runtime_envs_to_sflow_aliases():
    op = SrunOperator(SrunOperatorConfig(name="op_srun"))

    command = op.build_command(
        task_name="t1",
        script=["echo aliases"],
        envs={},
    )
    payload = command.as_list()[-1]

    assert 'export SFLOW_BACKEND_JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"' in payload
    assert 'export SFLOW_BACKEND_NODELIST="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"' in payload
    assert 'export SFLOW_BACKEND_NUM_NODES="${SLURM_NNODES:-}"' in payload
    assert 'export SFLOW_BACKEND_STEP_ID="${SLURM_STEP_ID:-}"' in payload
    assert 'export SFLOW_TASK_NODE_NAME="${SLURMD_NODENAME:-}"' in payload
    assert 'export SFLOW_TASK_NODE_INDEX="${SLURM_NODEID:-}"' in payload
    assert 'export SFLOW_TASK_PROCESS_ID="${SLURM_PROCID:-}"' in payload
    assert 'export SFLOW_TASK_LOCAL_PROCESS_ID="${SLURM_LOCALID:-}"' in payload
    assert 'export SFLOW_TASK_NUM_PROCESSES="${SLURM_NTASKS:-}"' in payload
    assert payload.rstrip().endswith("echo aliases")


def test_srun_operator_rejects_container_image_and_name_set_together():
    try:
        SrunOperatorConfig(
            name="op_srun",
            container_image="nvcr.io/a/b:1",
            container_name="cname",
        )
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert "container_image" in str(e) or "container_name" in str(e)


def test_srun_operator_explicit_nodes_is_used():
    """
    When nodes is explicitly set, it should be used.
    """
    op = SrunOperator(
        SrunOperatorConfig(
            name="op_srun",
            nodes=5,  # explicitly set
            ntasks=8,
            ntasks_per_node=4,
        )
    )
    s = str(op.build_command(task_name="t1", script=["echo hi"], envs={}))
    # Explicit nodes=5 should be used
    assert "--nodes 5" in s
    assert "--ntasks 8" in s
    assert "--ntasks-per-node 4" in s


def test_srun_operator_coerces_string_ntasks_to_int():
    """
    Ensure string values for ntasks and ntasks_per_node are converted to int.
    """
    config = SrunOperatorConfig(
        name="string_coercion_op",
        ntasks="8",  # String instead of int
        ntasks_per_node="4",  # String instead of int
        nodes="2",  # String instead of int
        cpus_per_task="16",  # String instead of int
    )
    op = SrunOperator(config)

    # Verify they were coerced to int
    assert config.ntasks == 8
    assert isinstance(config.ntasks, int)
    assert config.ntasks_per_node == 4
    assert isinstance(config.ntasks_per_node, int)
    assert config.nodes == 2
    assert isinstance(config.nodes, int)
    assert config.cpus_per_task == 16
    assert isinstance(config.cpus_per_task, int)

    # Also verify the command is built correctly
    cmd = op.build_command(task_name="t", script=["echo hi"], envs={})
    cmd_str = cmd.as_str()
    assert "--ntasks 8" in cmd_str
    assert "--ntasks-per-node 4" in cmd_str
    assert "--nodes 2" in cmd_str
    assert "--cpus-per-task 16" in cmd_str


def test_srun_operator_nodelist_takes_precedence_over_computed_nodes():
    """
    When nodelist is set but nodes is not, nodelist length should take precedence
    over computed value from ntasks/ntasks_per_node.
    """
    op = SrunOperator(
        SrunOperatorConfig(
            name="op_srun",
            nodelist=["n1", "n2", "n3"],  # 3 nodes
            ntasks=8,
            ntasks_per_node=4,
        )
    )
    s = str(op.build_command(task_name="t1", script=["echo hi"], envs={}))
    # nodelist length=3 should be used, not computed 8/4=2
    assert "--nodes 3" in s
    assert "--nodelist n1,n2,n3" in s


def test_srun_operator_merges_container_mounts_from_extra_args():
    """
    When --container-mounts is provided in extra_args, it should be merged
    with the container_mounts config field into a single --container-mounts flag.
    """
    # Case 1: extra_args with separate arg and value
    op = SrunOperator(
        SrunOperatorConfig(
            name="op_srun",
            container_image="my-image:latest",
            container_mounts=["/host/path1:/container/path1"],
            extra_args=["--container-mounts", "/host/path2:/container/path2"],
        )
    )
    s = str(op.build_command(task_name="t1", script=["echo hi"], envs={}))
    # Should have single --container-mounts with both paths
    assert s.count("--container-mounts") == 1
    assert "/host/path1:/container/path1" in s
    assert "/host/path2:/container/path2" in s


def test_srun_operator_merges_container_mounts_from_extra_args_equals_syntax():
    """
    When --container-mounts=VALUE is provided in extra_args with = syntax,
    it should be merged with the container_mounts config field.
    """
    op = SrunOperator(
        SrunOperatorConfig(
            name="op_srun",
            container_image="my-image:latest",
            container_mounts=["/path1:/cpath1"],
            extra_args=["--container-mounts=/path2:/cpath2,/path3:/cpath3"],
        )
    )
    s = str(op.build_command(task_name="t1", script=["echo hi"], envs={}))
    # Should have single --container-mounts with all three paths
    assert s.count("--container-mounts") == 1
    assert "/path1:/cpath1" in s
    assert "/path2:/cpath2" in s
    assert "/path3:/cpath3" in s


def test_srun_operator_container_mounts_only_from_extra_args():
    """
    When container_mounts is empty but --container-mounts is in extra_args,
    only the extra_args mounts should appear.
    """
    op = SrunOperator(
        SrunOperatorConfig(
            name="op_srun",
            container_image="my-image:latest",
            extra_args=["--container-mounts", "/extra:/cextra"],
        )
    )
    s = str(op.build_command(task_name="t1", script=["echo hi"], envs={}))
    assert "--container-mounts /extra:/cextra" in s
    assert s.count("--container-mounts") == 1


def test_srun_operator_config_exposes_container_protocol_hooks():
    cfg = SrunOperatorConfig(
        name="op_srun",
        container_image="nvcr.io/example/app:1.0",
        container_mounts=["/host:/ctr:ro"],
        extra_args=["--container-mounts", "/extra:/extra:rw,/host:/ctr:rw"],
    )

    assert cfg.uses_container() is True
    assert cfg.container_images() == ["nvcr.io/example/app:1.0"]
    assert cfg.mount_specs() == ["/host:/ctr:ro", "/extra:/extra:rw"]


def test_srun_operator_config_warns_when_enroot_credentials_missing(
    tmp_path, monkeypatch
):
    # log_to_file=False isolates the enroot warning from the (now default-on)
    # offload python3/bash>=5 warning.
    cfg = SrunOperatorConfig(
        name="op_srun",
        container_image="nvcr.io/example/app:1.0",
        log_to_file=False,
    )

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    warnings = cfg.runtime_warnings()

    assert len(warnings) == 1
    assert "enroot credentials" in warnings[0]


# ---------------------------------------------------------------------------
# Container image validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "image",
    [
        "nvcr.io/nvidia/pytorch:24.01-py3",
        "docker.io/library/ubuntu:latest",
        "ghcr.io/owner/image:v1.0",
        "registry.example.com:5000/org/repo:tag",
        "ubuntu:latest",
        "nvidia/cuda:12.0-runtime",
        "python:3.11",
        "my-image",
        "my-image:1",
        "nvcr.io/a/b@sha256:abcdef1234567890",
        "/path/to/image.sqsh",
        "./relative/image.sqsh",
        "image.sqsh",
        "/opt/containers/my-container.sqsh",
        "${{ variables.IMG }}",
        "${MY_IMAGE}",
    ],
)
def test_valid_container_images(image: str):
    assert is_valid_container_image(image), f"Expected valid: {image}"


@pytest.mark.parametrize(
    "image",
    [
        "",
        "   ",
        "has space:latest",
        "!invalid",
        "*glob*",
    ],
)
def test_invalid_container_images(image: str):
    assert not is_valid_container_image(image), f"Expected invalid: {image}"


def test_srun_operator_warns_on_unrecognised_container_image(image_warnings):
    """BEHAVIOR CHANGE: config parsing no longer fails on an unrecognised image."""
    cfg = SrunOperatorConfig(name="op", container_image="not a valid image!!")
    assert cfg.container_image == "not a valid image!!", "the value is kept as given"
    assert any("container_image" in m for m in image_warnings), image_warnings


def test_srun_operator_accepts_valid_registry_image():
    cfg = SrunOperatorConfig(
        name="op", container_image="nvcr.io/nvidia/pytorch:24.01-py3"
    )
    assert cfg.container_image == "nvcr.io/nvidia/pytorch:24.01-py3"


def test_srun_operator_accepts_sqsh_image():
    cfg = SrunOperatorConfig(name="op", container_image="/opt/images/my.sqsh")
    assert cfg.container_image == "/opt/images/my.sqsh"


def test_srun_operator_accepts_template_variable_image():
    cfg = SrunOperatorConfig(
        name="op", container_image="${{ variables.CONTAINER_IMAGE }}"
    )
    assert cfg.container_image == "${{ variables.CONTAINER_IMAGE }}"


def test_srun_operator_warns_on_placeholder_image(image_warnings):
    cfg = SrunOperatorConfig(name="op", container_image="<your-container-image>")
    assert cfg.container_image == "<your-container-image>"
    assert any("container_image" in m for m in image_warnings), image_warnings


def test_srun_operator_warns_on_unrecognised_image_in_extra_args_equals(image_warnings):
    SrunOperatorConfig(
        name="op",
        extra_args=["--container-image=not a valid image!!"],
    )
    assert any("extra_args" in m for m in image_warnings), image_warnings


def test_srun_operator_warns_on_unrecognised_image_in_extra_args_space(image_warnings):
    SrunOperatorConfig(
        name="op",
        extra_args=["--container-image", "not a valid image!!"],
    )
    assert any("extra_args" in m for m in image_warnings), image_warnings


def test_srun_operator_accepts_pyxis_enroot_image_in_extra_args(image_warnings):
    """The reported regression: a valid pyxis/enroot URI aborted the whole run.

    ``registry#path:tag`` is the documented enroot form, so it must be accepted
    outright -- not merely downgraded to a warning.
    """
    image = "nvcr.io#nvidia/ai-dynamo/sglang-runtime:1.2.0-deepseek-v4-cuda13-dev.3"
    cfg = SrunOperatorConfig(
        name="op",
        extra_args=["--container-image", image, "--container-mount-home"],
    )
    assert image in cfg.extra_args
    assert image_warnings == [], "a valid enroot URI must not even warn"


def test_srun_operator_accepts_valid_image_in_extra_args():
    cfg = SrunOperatorConfig(
        name="op",
        extra_args=["--container-image=nvcr.io/nvidia/pytorch:24.01-py3"],
    )
    assert "--container-image=nvcr.io/nvidia/pytorch:24.01-py3" in cfg.extra_args


# ---------------------------------------------------------------------------
# Job / placement targeting
# ---------------------------------------------------------------------------


def test_srun_apply_backend_context_pins_jobid_and_nodelist():
    """apply_backend_context binds the step to the backend's allocation via
    --jobid / --nodelist (login-node / single-allocation path)."""
    from sflow.core.backend import Allocation
    from sflow.core.compute_node import ComputeNode

    class _FakeBackend:
        def __init__(self):
            self.allocation = Allocation(
                allocation_id="777",
                nodes=[
                    ComputeNode(
                        name="n1", ip_address="10.0.0.1", index=0, num_gpus=0
                    )
                ],
                owned=True,
            )

    op = SrunOperator(SrunOperatorConfig(name="op", ntasks_per_node=1))
    op.apply_backend_context(
        backend=_FakeBackend(),
        assigned_nodes=["n1"],
        artifacts=[],
    )

    assert op.config.job_id == "777"
    assert op.config.nodelist == ["n1"]

    s = str(op.build_command(task_name="t", script=["echo hi"], envs={}))
    assert "--jobid 777" in s
    assert "--nodelist n1" in s


def test_srun_step_exports_all_without_overrides():
    """A step keeps a plain --export ALL: no enroot runtime relocation and no
    SLURM_JOB_ID override."""
    op = SrunOperator(
        SrunOperatorConfig(
            name="op", job_id="777", nodelist=["n1"], ntasks_per_node=1
        )
    )

    s = str(op.build_command(task_name="t", script=["echo hi"], envs={}))
    assert "--export ALL" in s
    assert "ENROOT_RUNTIME_PATH=" not in s
    assert "SLURM_JOB_ID=" not in s
    assert "--jobid 777" in s
