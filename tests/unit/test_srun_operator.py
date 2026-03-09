# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ValidationError

from sflow.plugins.operators.srun import SrunOperator, SrunOperatorConfig


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
