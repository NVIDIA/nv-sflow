# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from pydantic import ValidationError

from sflow.app.run_support import configure_task_runtime
from sflow.plugins.backends.docker import DockerBackend, DockerBackendConfig
from sflow.plugins.operators.docker_run import DockerRunOperatorConfig


def test_docker_backend_allocates_synthetic_local_nodes():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker",
            type="docker",
            image="ubuntu:22.04",
            nodes=2,
            gpus_per_node=4,
        )
    )

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "docker"
    assert allocation.owned is False
    assert [n.name for n in allocation.nodes] == ["localhost", "localhost-1"]
    assert [n.ip_address for n in allocation.nodes] == ["127.0.0.1", "127.0.0.1"]
    assert [n.num_gpus for n in allocation.nodes] == [4, 4]


def test_docker_backend_config_merges_extra_args_without_duplicates():
    config = DockerBackendConfig(
        name="docker",
        type="docker",
        image="ubuntu:22.04",
        extra_args=["--pull=always"],
    )

    merged = config.merge_extra_args(["--pull=always", "--network=host"])

    assert merged is not config
    assert merged.extra_args == ["--pull=always", "--network=host"]
    assert config.extra_args == ["--pull=always"]


def test_docker_backend_normalizes_bundled_extra_args():
    # Docker extra_args are shell-split via the shared normalize_extra_args (as on the
    # Slurm backend), so a bundled/whitespace-laden entry can't reach `docker run` as
    # one unparsable token.
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker",
            type="docker",
            image="ubuntu:22.04",
            extra_args=["--cap-add=SYS_PTRACE --shm-size=1g", "--network=host"],
        )
    )
    assert backend._extra_args == [
        "--cap-add=SYS_PTRACE",
        "--shm-size=1g",
        "--network=host",
    ]


def test_docker_backend_placeholder_allocation_uses_localhost_addresses():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker",
            type="docker",
            image="ubuntu:22.04",
            nodes=1,
        )
    )

    allocation = backend.placeholder_allocation()

    assert [(n.name, n.ip_address) for n in allocation.nodes] == [
        ("localhost", "127.0.0.1")
    ]


def test_docker_backend_uses_docker_run_operator_as_default():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker",
            type="docker",
            image="ubuntu:22.04",
            gpus_per_node=1,
        )
    )

    operator = backend.default_operator(name="default_docker", assigned_nodes=["localhost"])

    assert operator.__class__.__name__ == "DockerRunOperator"
    assert operator.config.name == "default_docker"
    assert operator.config.type == "docker_run"
    assert operator.config.image == "ubuntu:22.04"
    assert operator.config.gpus is None


def test_docker_run_operator_narrows_gpu_devices_from_resource_placement():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker",
            type="docker",
            image="ubuntu:22.04",
            gpus_per_node=8,
        )
    )
    operator = backend.default_operator(name="default_docker", assigned_nodes=["localhost"])

    operator.apply_backend_context(
        backend=backend,
        assigned_nodes=["localhost"],
        artifacts=[],
        cuda_visible_devices="2,3",
    )

    assert operator.config.gpus == "device=2,3"


def test_docker_backend_allocates_configured_remote_hosts():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {
                    "name": "dgx-a",
                    "docker_host": "ssh://dgx-a",
                    "ip_address": "10.0.0.11",
                    "gpus_per_node": 8,
                },
                {
                    "name": "dgx-b",
                    "context": "dgx-b-context",
                    "ip_address": "10.0.0.12",
                    "gpus_per_node": 4,
                },
            ],
        )
    )

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "docker"
    assert allocation.owned is False
    assert [(n.name, n.ip_address, n.index, n.num_gpus) for n in allocation.nodes] == [
        ("dgx-a", "10.0.0.11", 0, 8),
        ("dgx-b", "10.0.0.12", 1, 4),
    ]
    assert backend.capabilities.has_runtime_node_addresses is True
    assert backend.host_for_node("dgx-a").docker_host == "ssh://dgx-a"
    assert backend.host_for_node("dgx-b").context == "dgx-b-context"


def _docker_hosts_cfg(**extra):
    return DockerBackendConfig(
        name="docker_cluster",
        type="docker",
        image="ubuntu:22.04",
        hosts=[
            {"name": "dgx-a", "docker_host": "ssh://dgx-a", "ip_address": "10.0.0.11"},
            {"name": "dgx-b", "docker_host": "ssh://dgx-b", "ip_address": "10.0.0.12"},
            {"name": "dgx-c", "docker_host": "ssh://dgx-c", "ip_address": "10.0.0.13"},
        ],
        **extra,
    )


def test_docker_backend_include_restricts_host_pool():
    backend = DockerBackend(_docker_hosts_cfg(include_nodes=["dgx-a", "dgx-c"]))
    allocation = asyncio.run(backend.allocate())
    assert [n.name for n in allocation.nodes] == ["dgx-a", "dgx-c"]
    assert backend.host_for_node("dgx-b") is None


def test_docker_backend_exclude_drops_hosts():
    backend = DockerBackend(_docker_hosts_cfg(exclude_nodes=["dgx-b"]))
    allocation = asyncio.run(backend.allocate())
    assert [n.name for n in allocation.nodes] == ["dgx-a", "dgx-c"]
    assert backend.host_for_node("dgx-b") is None


def test_docker_backend_include_removing_all_hosts_raises():
    with pytest.raises(ValueError):
        DockerBackend(
            DockerBackendConfig(
                name="d",
                type="docker",
                image="ubuntu:22.04",
                include_nodes=["nope"],
                hosts=[{"name": "dgx-a", "docker_host": "ssh://dgx-a"}],
            )
        )


def test_docker_backend_context_keeps_configured_ips_per_node():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a"},
                {
                    "name": "dgx-b",
                    "context": "dgx-b-context",
                    "ip_address": "10.0.0.12",
                },
            ],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())

    nodes = backend.to_dict()["nodes"]

    assert "ip_address" not in nodes[0]
    assert nodes[1]["ip_address"] == "10.0.0.12"


def test_docker_backend_dry_run_details_describe_multi_host_config():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a"},
                {"name": "dgx-b", "context": "dgx-b-context"},
            ],
            gpus_per_node=8,
        )
    )

    details = dict(backend.dry_run_details())

    assert details["image"] == "ubuntu:22.04"
    assert details["hosts"] == "dgx-a, dgx-b"
    assert details["gpus_per_node"] == "8"


def test_docker_run_operator_rejects_invalid_literal_image_at_parse_time():
    with pytest.raises(ValidationError, match="image.*does not look like"):
        DockerRunOperatorConfig(name="docker", image="<replace-me>")


def test_docker_run_operator_allows_templated_image_at_parse_time():
    config = DockerRunOperatorConfig(name="docker", image="${{ variables.IMAGE }}")

    assert config.image == "${{ variables.IMAGE }}"


def test_docker_run_operator_config_exposes_container_protocol_hooks():
    config = DockerRunOperatorConfig(
        name="docker",
        image="ubuntu:22.04",
        mounts=["/host:/ctr:ro"],
    )

    config.append_runtime_mounts(["/host:/ctr:rw", "/extra:/extra:rw"])

    assert config.uses_container() is True
    assert config.container_images() == ["ubuntu:22.04"]
    assert config.mount_specs() == ["/host:/ctr:ro", "/extra:/extra:rw"]


def test_configure_task_runtime_skips_implicit_mounts_for_remote_docker_hosts(tmp_path):
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a"},
            ],
        )
    )
    operator = backend.default_operator(name="default_docker")
    operator.apply_backend_context(
        backend=backend,
        assigned_nodes=["dgx-a"],
        artifacts=[],
    )
    task = type("Task", (), {})()
    task.name = "t1"
    task.envs = {}
    task.operator = operator
    out_dir = tmp_path / "out"
    workflow_out_dir = out_dir / "wf"

    configure_task_runtime(
        task,
        ws_dir=tmp_path,
        out_dir=out_dir,
        workflow_out_dir=workflow_out_dir,
        dry_run=True,
    )

    assert operator.config.mounts == []


def test_docker_run_operator_builds_multi_host_wrapper_without_inlining_env_values():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a", "gpus_per_node": 8},
                {"name": "dgx-b", "context": "dgx-b-context", "gpus_per_node": 8},
            ],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    operator = backend.default_operator(name="default_docker")

    operator.apply_backend_context(
        backend=backend,
        assigned_nodes=["dgx-a", "dgx-b"],
        artifacts=[],
        cuda_visible_devices="0",
    )
    cmd = operator.build_command(
        task_name="distributed_job",
        script=["./run_worker.sh"],
        envs={"TOKEN": "sensitivevalue"},
    )

    assert cmd.as_list()[:2] == ["bash", "-lc"]
    script = cmd.as_list()[2]
    assert "docker --host ssh://dgx-a run" in script
    assert "docker --context dgx-b-context run" in script
    assert "--gpus device=0" in script
    assert "-e TOKEN" in script
    assert "sensitivevalue" not in script
    assert "wait \"$pid\"" in script
    assert "trap 'cleanup; exit 143' HUP INT TERM" in script
    assert "docker --host ssh://dgx-a rm -f" in script
    assert "docker --context dgx-b-context rm -f" in script


def test_docker_run_single_node_names_container_and_teardown_matches():
    """Single-node `docker run` must carry a deterministic --name, and teardown
    must force-remove exactly that container (killing the client can leak it)."""
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker", type="docker", image="ubuntu:22.04", gpus_per_node=1
        )
    )
    operator = backend.default_operator(name="default_docker", assigned_nodes=["localhost"])
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["localhost"], artifacts=[]
    )

    cmd = operator.build_command(task_name="server", script=["echo hi"], envs={})
    parts = cmd.as_list()
    # Still a bare `docker run` (not a bash wrapper) for the single-node case ...
    assert parts[:2] == ["docker", "run"]
    # ... but now with a deterministic container name the driver can reap.
    assert "--name" in parts
    assert parts[parts.index("--name") + 1] == "sflow-server-localhost"

    teardown = operator.teardown_commands(task_name="server")
    assert [c.as_list() for c in teardown] == [
        ["docker", "rm", "-f", "sflow-server-localhost"]
    ]


def test_docker_run_single_remote_host_teardown_targets_that_host():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[{"name": "dgx-a", "docker_host": "ssh://dgx-a"}],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    operator = backend.default_operator(name="default_docker")
    operator.apply_backend_context(backend=backend, assigned_nodes=["dgx-a"], artifacts=[])

    # build_command names the container; teardown reaps the same name on the host.
    assert "sflow-job-dgx-a" in operator.build_command(
        task_name="job", script=["echo hi"], envs={}
    ).as_list()
    teardown = operator.teardown_commands(task_name="job")
    assert [c.as_list() for c in teardown] == [
        ["docker", "--host", "ssh://dgx-a", "rm", "-f", "sflow-job-dgx-a"]
    ]


def test_docker_run_multi_host_teardown_covers_every_node():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a"},
                {"name": "dgx-b", "context": "dgx-b-context"},
            ],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    operator = backend.default_operator(name="default_docker")
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["dgx-a", "dgx-b"], artifacts=[]
    )

    teardown = [c.as_list() for c in operator.teardown_commands(task_name="job")]
    assert ["docker", "--host", "ssh://dgx-a", "rm", "-f", "sflow-job-dgx-a"] in teardown
    assert [
        "docker",
        "--context",
        "dgx-b-context",
        "rm",
        "-f",
        "sflow-job-dgx-b",
    ] in teardown


def test_docker_run_operator_builds_single_remote_host_command():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a", "gpus_per_node": 8},
            ],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    operator = backend.default_operator(name="default_docker")

    operator.apply_backend_context(
        backend=backend,
        assigned_nodes=["dgx-a"],
        artifacts=[],
        cuda_visible_devices="1",
    )
    cmd = operator.build_command(
        task_name="single_host_job",
        script=["./run_worker.sh"],
        envs={"TOKEN": "sensitivevalue"},
    )

    assert cmd.as_list()[:4] == ["docker", "--host", "ssh://dgx-a", "run"]
    assert "--gpus" in cmd.as_list()
    assert "device=1" in cmd.as_list()
    assert "-e" in cmd.as_list()
    assert "TOKEN" in cmd.as_list()
    assert "sensitivevalue" not in cmd.as_str()
