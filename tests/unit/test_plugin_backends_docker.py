# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shlex
import unittest.mock as mock

import pytest
from pydantic import ValidationError

from sflow.app.run_support import configure_task_runtime
from sflow.core.backend import Allocation
from sflow.core.command import format_command
from sflow.plugins.backends.docker import (
    DockerBackend,
    DockerBackendConfig,
    DockerHostConfig,
)
from sflow.plugins.operators.bash import BashOperator
from sflow.plugins.operators.docker_run import (
    _OWNER_PREFIX,
    DockerRunOperator,
    DockerRunOperatorConfig,
    _safe_container_name,
)


def _cname(task: str, node: str) -> str:
    """Expected name: <owner-prefix><pid>-<task>-<node> (same process = same pid)."""
    return f"{_OWNER_PREFIX}{os.getpid()}-{task}-{node}"


class _IdentityResolver:
    """resolve() returns its input unchanged (mirrors the k8s backend tests)."""

    def resolve(self, value, ctx):
        return value


def test_docker_backend_without_hosts_allocates_one_local_node():
    """No pool means this machine, and this machine is one node.

    `nodes: >1` without `hosts:` is refused at construction -- see
    tests/unit/test_docker_allocation_honesty.py.
    """
    backend = DockerBackend(
        DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
    )

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "docker"
    assert allocation.owned is False
    assert [n.name for n in allocation.nodes] == ["localhost"]
    assert [n.ip_address for n in allocation.nodes] == ["127.0.0.1"]


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
        gpu_count=2,
    )

    # The planner's device slice is pinned up front for every placement, so
    # --dry-run shows the pinning a real run applies. A launch-time reservation
    # (acquire_resources) replaces it with the physical GPUs actually claimed.
    # Plain here: docker's quoting is applied when the value is put on the command
    # line, so it does not leak into the config (or into --dry-run output).
    assert operator.config.gpus == "device=2,3"
    assert operator._gpu_count == 2


@pytest.mark.parametrize(
    "written, expected",
    [
        # The reported bug: docker reads the comma as a CSV separator, takes the
        # `1` for a COUNT, and dies with "cannot set both Count and DeviceIDs on
        # device request". Verified against a real daemon, before and after.
        ("device=0,1", '"device=0,1"'),
        ("device=GPU-aaa,GPU-bbb", '"device=GPU-aaa,GPU-bbb"'),
        # Harmless without a comma, but kept identical to what the reservation
        # emits so both spellings of the same request look the same.
        ("device=0", '"device=0"'),
        # Not a device list -- passed through exactly as written.
        ("all", "all"),
        ("2", "2"),
        # Here the commas separate OPTIONS, not devices. Quoting would break the
        # very parse this normalization exists to fix.
        ("count=2,capabilities=gpu", "count=2,capabilities=gpu"),
        ("device=0,capabilities=gpu", "device=0,capabilities=gpu"),
    ],
)
def test_hand_written_gpu_device_lists_are_quoted_for_docker(written, expected):
    """Quoting happens where the value meets docker, not in the config.

    A `gpus:` a user typed reaches `docker run` exactly as typed, so it needs the
    same quoting sflow's own reservation gets -- applied at the single point both
    of them pass through, which also leaves the config reading as it was written.
    """
    config = DockerRunOperatorConfig(name="op", image="ubuntu:22.04", gpus=written)

    assert config.gpus == written, "the config keeps what the user wrote"
    rendered = format_command(
        DockerRunOperator(config).build_command(
            task_name="t", script=["echo hi"], envs={}
        )
    )
    assert f"--gpus {shlex.quote(expected)}" in rendered


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


def test_docker_backend_exclude_keeps_declared_node_count_valid():
    # nodes: matches the three declared hosts; steering the run down to two
    # with --exclude-nodes must not turn that into a config error.
    backend = DockerBackend(_docker_hosts_cfg(nodes=3, exclude_nodes=["dgx-b"]))
    assert [n.name for n in asyncio.run(backend.allocate()).nodes] == ["dgx-a", "dgx-c"]


def test_docker_host_ssh_url_gives_target_with_port_and_rendezvous():
    from sflow.plugins.backends.docker import _ssh_target

    host = DockerHostConfig(name="h", docker_host="ssh://me@gpu-box:2222")
    assert _ssh_target(host) == ("-p", "2222", "me@gpu-box")
    assert host.rendezvous() == "gpu-box"
    assert _ssh_target(DockerHostConfig(name="h", address="box", user="u")) == ("u@box",)
    assert DockerHostConfig(name="h", context="ctx").rendezvous() is None


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
                # A context names a daemon, not a machine -> nothing for peers
                # to dial, so the node carries no ip_address.
                {"name": "dgx-a", "context": "dgx-a-context"},
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
                {"name": "dgx-a", "docker_host": "ssh://dgx-a", "address": "10.0.0.11"},
                {"name": "dgx-b", "context": "dgx-b-context", "address": "10.0.0.12"},
            ],
            gpus_per_node=8,
        )
    )

    details = dict(backend.dry_run_details())

    assert details["image"] == "ubuntu:22.04"
    # Each host names its concrete endpoint; the rendezvous is shown because
    # peers dial it rather than the endpoint.
    assert details["host dgx-a"] == "dgx-a (rendezvous=10.0.0.11)"
    assert details["host dgx-b"] == "context dgx-b-context (rendezvous=10.0.0.12)"
    assert details["gpus_per_node"] == "8"


def test_dry_run_details_show_each_hosts_login_and_address():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "node-0", "address": "10.0.0.11", "user": "svc"},
                {"name": "node-1", "address": "10.0.0.12", "gpus_per_node": 4},
                {
                    "name": "node-2",
                    "address": "10.0.0.13",
                    "rendezvous_address": "192.168.1.13",
                },
            ],
        )
    )

    details = dict(backend.dry_run_details())

    # Rendezvous equal to the address adds nothing, so only the login shows.
    assert details["host node-0"] == "svc@10.0.0.11"
    assert details["host node-1"] == "10.0.0.12 (gpus=4)"
    assert details["host node-2"] == "10.0.0.13 (rendezvous=192.168.1.13)"


def test_docker_run_operator_warns_on_unrecognised_literal_image(image_warnings):
    """BEHAVIOR CHANGE: an unrecognised image warns instead of failing config parsing."""
    config = DockerRunOperatorConfig(name="docker", image="<replace-me>")

    assert config.image == "<replace-me>", "the value is kept as given"
    assert any("<replace-me>" in m for m in image_warnings), image_warnings


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
                {"name": "dgx-a", "docker_host": "ssh://dgx-a", "address": "10.0.0.11"},
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


def test_docker_run_operator_builds_multi_host_members_without_inlining_env_values():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a", "address": "10.0.0.11", "gpus_per_node": 8},
                {"name": "dgx-b", "context": "dgx-b-context", "address": "10.0.0.12", "gpus_per_node": 8},
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
    argv = [
        command.as_list()
        for _member, command in operator._multi_node_commands(
            task_name="distributed_job",
            script=["./run_worker.sh"],
            envs={"TOKEN": "sensitivevalue"},
        )
    ]

    assert argv[0][:3] == ["docker", "--host", "ssh://dgx-a"]
    assert argv[1][:3] == ["docker", "--context", "dgx-b-context"]
    assert all('"device=0"' in args for args in argv)
    # Env is forwarded BY NAME; a value in the argv would leak a secret into
    # every process listing and command log on the host.
    assert all("TOKEN" in args for args in argv)
    assert all("sensitivevalue" not in args for args in argv)

    # Each member is reaped on the daemon that ran it.
    teardown = [c.as_list() for c in operator.teardown_commands(task_name="distributed_job")]
    assert teardown[0][:5] == ["docker", "--host", "ssh://dgx-a", "rm", "-f"]
    assert teardown[1][:5] == ["docker", "--context", "dgx-b-context", "rm", "-f"]


def test_docker_run_rm_defaults_to_on():
    """The default has to stay `--rm` -- existing recipes rely on it."""
    assert DockerRunOperatorConfig(name="op", image="ubuntu:22.04").rm is True


@pytest.mark.parametrize(
    "rm, expect_flag, expect_teardown",
    [
        (True, True, ["rm", "-f"]),
        (False, False, ["stop"]),
    ],
)
def test_docker_run_rm_gates_both_the_flag_and_teardown_removal(
    rm, expect_flag, expect_teardown
):
    """`rm: false` must survive teardown to mean anything.

    The driver force-removes every container it named, so omitting `--rm` alone
    would still leave nothing to inspect. Teardown therefore downgrades to
    `docker stop`, which keeps the container while still guaranteeing it is not
    left running after the workflow.
    """
    backend = DockerBackend(
        DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
    )
    operator = DockerRunOperator(
        DockerRunOperatorConfig(name="op", image="ubuntu:22.04", rm=rm)
    )
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["localhost"], artifacts=[]
    )

    rendered = format_command(
        operator.build_command(task_name="server", script=["echo hi"], envs={})
    )
    assert ("--rm" in rendered) is expect_flag

    teardown = [c.as_list() for c in operator.teardown_commands(task_name="server")]
    assert teardown[0] == ["docker", *expect_teardown, _cname("server", "localhost")]


def test_a_kept_container_frees_its_name_for_a_retry():
    """`docker run --name X` refuses while ANY container X exists, in any state.

    Without the rename, a kept container makes every retry of the task die with
    "name already in use" -- the name is deterministic within a run -- so
    `rm: false` would be unusable with `retries:`.
    """
    backend = DockerBackend(
        DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
    )
    operator = DockerRunOperator(
        DockerRunOperatorConfig(name="op", image="ubuntu:22.04", rm=False)
    )
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["localhost"], artifacts=[]
    )

    teardown = [c.as_list() for c in operator.teardown_commands(task_name="server")]
    name = _cname("server", "localhost")

    assert teardown[0] == ["docker", "stop", name]
    assert teardown[1][:3] == ["docker", "rename", name]
    assert teardown[1][3].startswith(f"{name}-kept-"), "kept under a findable name"


def test_removing_containers_needs_no_rename():
    operator = DockerRunOperator(
        DockerRunOperatorConfig(name="op", image="ubuntu:22.04", rm=True)
    )
    operator.apply_backend_context(
        backend=DockerBackend(
            DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
        ),
        assigned_nodes=["localhost"],
        artifacts=[],
    )

    teardown = [c.as_list() for c in operator.teardown_commands(task_name="server")]

    assert len(teardown) == 1, "rm -f already frees the name"


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
    assert parts[parts.index("--name") + 1] == _cname("server", "localhost")

    teardown = operator.teardown_commands(task_name="server")
    assert [c.as_list() for c in teardown] == [
        ["docker", "rm", "-f", _cname("server", "localhost")]
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
    assert _cname("job", "dgx-a") in operator.build_command(
        task_name="job", script=["echo hi"], envs={}
    ).as_list()
    teardown = operator.teardown_commands(task_name="job")
    assert [c.as_list() for c in teardown] == [
        ["docker", "--host", "ssh://dgx-a", "rm", "-f", _cname("job", "dgx-a")]
    ]


def test_stale_reap_command_removes_only_dead_pid_containers():
    # Single local node -> one bash sweep that lists sflow-p* containers, parses
    # the pid from the name, and rm -f only those whose pid is dead (kill -0).
    backend = DockerBackend(
        DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
    )
    operator = backend.default_operator(name="d", assigned_nodes=["localhost"])
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["localhost"], artifacts=[]
    )

    reap = operator.stale_reap_commands(task_name="job")
    assert len(reap) == 1
    script = reap[0].as_list()[2]
    assert reap[0].as_list()[:2] == ["bash", "-c"]
    # lists only sflow-p* containers, checks liveness, force-removes the dead ones
    assert f"ps -a --filter name={_OWNER_PREFIX}" in script
    assert "kill -0" in script
    assert "rm -f" in script


def test_stale_reap_covers_remote_hosts():
    """The sweep reaches every daemon the task targets, not just the local one.

    See tests/unit/test_docker_multinode.py for why that is safe (the name
    carries the driver's machine identity) and why it is necessary (a SIGKILLed
    driver never runs teardown, so remote containers have no other route back).
    """
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a", "address": "10.0.0.11"},
                {"name": "dgx-b", "context": "dgx-b-context", "address": "10.0.0.12"},
            ],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    operator = backend.default_operator(name="d")
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["dgx-a", "dgx-b"], artifacts=[]
    )

    scripts = [c.as_list()[2] for c in operator.stale_reap_commands(task_name="job")]

    assert len(scripts) == 2, "one sweep per distinct daemon"
    assert any("docker --host ssh://dgx-a ps -a" in s for s in scripts)
    assert any("docker --context dgx-b-context ps -a" in s for s in scripts)


def test_docker_run_multi_host_teardown_covers_every_node():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a", "address": "10.0.0.11"},
                {"name": "dgx-b", "context": "dgx-b-context", "address": "10.0.0.12"},
            ],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    operator = backend.default_operator(name="default_docker")
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["dgx-a", "dgx-b"], artifacts=[]
    )

    teardown = [c.as_list() for c in operator.teardown_commands(task_name="job")]
    assert [
        "docker", "--host", "ssh://dgx-a", "rm", "-f", _cname("job", "dgx-a")
    ] in teardown
    assert [
        "docker",
        "--context",
        "dgx-b-context",
        "rm",
        "-f",
        _cname("job", "dgx-b"),
    ] in teardown


def test_docker_run_operator_builds_single_remote_host_command():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {"name": "dgx-a", "docker_host": "ssh://dgx-a", "address": "10.0.0.11", "gpus_per_node": 8},
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
    assert '"device=1"' in cmd.as_list()
    assert "-e" in cmd.as_list()
    assert "TOKEN" in cmd.as_list()
    assert "sensitivevalue" not in cmd.as_str()


# ---------------------------------------------------------------------------
# preflight_validate
# ---------------------------------------------------------------------------


def test_preflight_raises_when_docker_missing():
    backend = DockerBackend(
        DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
    )
    with mock.patch("shutil.which", return_value=None):
        with pytest.raises(ValueError, match="docker"):
            backend.preflight_validate()


def test_preflight_passes_when_docker_present():
    backend = DockerBackend(
        DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
    )
    with mock.patch("shutil.which", return_value="/usr/bin/docker"):
        assert backend.preflight_validate() is None


# ---------------------------------------------------------------------------
# resolve_config
# ---------------------------------------------------------------------------


def test_resolve_config_resolves_scalars_and_hosts():
    conf = DockerBackendConfig(
        name="dk",
        type="docker",
        image="ubuntu:22.04",
        nodes=3,
        gpus_per_node=4,
        mounts=["/a:/b"],
        workdir="/w",
        extra_args=["--net=host"],
        hosts=[
            {
                "name": "dgx-a",
                "docker_host": "ssh://dgx-a",
                "ip_address": "10.0.0.11",
                "gpus_per_node": 8,
            },
            {"name": "dgx-b", "context": "ctx", "ip_address": "10.0.0.12"},
        ],
    )

    resolved = DockerBackend.resolve_config(
        conf, resolver=_IdentityResolver(), ctx={}, workflow_name="wf"
    )

    assert resolved.image == "ubuntu:22.04"
    assert resolved.nodes == 3
    assert resolved.gpus_per_node == 4
    assert resolved.mounts == ["/a:/b"]
    assert resolved.workdir == "/w"
    assert resolved.extra_args == ["--net=host"]
    assert [h.name for h in resolved.hosts] == ["dgx-a", "dgx-b"]
    assert resolved.hosts[0].docker_host == "ssh://dgx-a"
    assert resolved.hosts[0].gpus_per_node == 8
    assert resolved.hosts[1].context == "ctx"


def test_resolve_config_nodes_non_int_raises():
    conf = DockerBackendConfig(
        name="dk", type="docker", image="ubuntu:22.04", nodes="abc"
    )
    with pytest.raises(ValueError, match="nodes must resolve to int"):
        DockerBackend.resolve_config(
            conf, resolver=_IdentityResolver(), ctx={}, workflow_name="wf"
        )


def test_config_rejects_non_int_gpus_at_construction():
    # Concrete (non-template) gpus_per_node is validated by the schema itself,
    # before resolve_config ever runs.
    with pytest.raises(ValidationError, match="int or expression"):
        DockerBackendConfig(
            name="dk", type="docker", image="ubuntu:22.04", gpus_per_node="x"
        )


def test_config_rejects_negative_gpus_at_construction():
    with pytest.raises(ValidationError, match=">= 0"):
        DockerBackendConfig(
            name="dk", type="docker", image="ubuntu:22.04", gpus_per_node=-1
        )


class _MapResolver:
    """Resolves only the templated values it's given; everything else passes through.

    Lets us exercise resolve_config's runtime guards for a template that resolves
    to a bad value (the schema only checks *concrete* gpus_per_node at construction).
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def resolve(self, value, ctx):
        return self.mapping.get(value, value)


def test_resolve_config_gpus_template_non_int_raises():
    conf = DockerBackendConfig(
        name="dk",
        type="docker",
        image="ubuntu:22.04",
        gpus_per_node="${{ variables.G }}",
    )
    resolver = _MapResolver({"${{ variables.G }}": "notanint"})
    with pytest.raises(ValueError, match="gpus_per_node must resolve to int"):
        DockerBackend.resolve_config(
            conf, resolver=resolver, ctx={}, workflow_name="wf"
        )


def test_resolve_config_gpus_template_negative_raises():
    conf = DockerBackendConfig(
        name="dk",
        type="docker",
        image="ubuntu:22.04",
        gpus_per_node="${{ variables.G }}",
    )
    resolver = _MapResolver({"${{ variables.G }}": -1})
    with pytest.raises(ValueError, match=">= 0"):
        DockerBackend.resolve_config(
            conf, resolver=resolver, ctx={}, workflow_name="wf"
        )


def test_resolve_config_clamps_nodes_min_one():
    conf = DockerBackendConfig(
        name="dk", type="docker", image="ubuntu:22.04", nodes=0
    )
    resolved = DockerBackend.resolve_config(
        conf, resolver=_IdentityResolver(), ctx={}, workflow_name="wf"
    )
    assert resolved.nodes == 1


# ---------------------------------------------------------------------------
# DockerHostConfig.exactly_one_endpoint
# ---------------------------------------------------------------------------


def test_host_config_requires_a_machine_or_an_endpoint_override():
    """`address` identifies the machine; the endpoints are overrides of it.

    See tests/unit/test_docker_host_identity.py for the derivation rules; this
    only pins that a host cannot be anonymous and cannot set both overrides.
    """
    # both endpoint overrides -> reject
    with pytest.raises(ValidationError, match="docker_host.*context"):
        DockerHostConfig(name="dgx-a", docker_host="ssh://dgx-a", context="ctx")
    # nothing at all -> reject
    with pytest.raises(ValidationError, match="address"):
        DockerHostConfig(name="dgx-a")
    # any one of the three identifies the host
    assert DockerHostConfig(name="dgx-a", address="10.0.0.11").address
    assert DockerHostConfig(name="dgx-a", docker_host="ssh://dgx-a").docker_host
    assert DockerHostConfig(name="dgx-b", context="ctx").context


# ---------------------------------------------------------------------------
# DockerBackendConfig helpers
# ---------------------------------------------------------------------------


def test_config_container_images():
    config = DockerBackendConfig(name="dk", type="docker", image="ubuntu:22.04")
    assert config.container_images() == ["ubuntu:22.04"]


def test_planning_node_count_hosts_vs_nodes():
    with_hosts = DockerBackendConfig(
        name="dk",
        type="docker",
        image="ubuntu:22.04",
        nodes=5,
        hosts=[
            {"name": "dgx-a", "docker_host": "ssh://dgx-a", "address": "10.0.0.11"},
            {"name": "dgx-b", "context": "ctx"},
        ],
    )
    # hosts take priority over the nodes scalar
    assert with_hosts.planning_node_count() == 2

    without_hosts = DockerBackendConfig(
        name="dk", type="docker", image="ubuntu:22.04", nodes=3
    )
    assert without_hosts.planning_node_count() == 3


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


def test_release_is_noop():
    backend = DockerBackend(
        DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
    )
    alloc = Allocation(allocation_id="docker", nodes=[], owned=False)
    # Docker nodes are not owned; release must be a side-effect-free no-op.
    assert asyncio.run(backend.release(alloc)) is None


# ---------------------------------------------------------------------------
# monitor_operator
# ---------------------------------------------------------------------------


def test_monitor_operator_returns_bash():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker", type="docker", image="ubuntu:22.04", nodes=1
        )
    )
    operator = backend.monitor_operator(name="mon", assigned_nodes=["localhost"])
    # Monitoring must run on the bare host, not inside the workload container.
    assert isinstance(operator, BashOperator)
    assert operator.config.log_to_file is False


def test_endpoint_only_host_still_gets_a_container_collector():
    """A tcp:// `docker_host:` names a daemon and no machine. It has no address
    to ssh to, but the collector does not need one -- it rides the same endpoint
    the workload does."""
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[{"name": "dgx-a", "docker_host": "tcp://dgx-a:2376"}],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())

    assert backend.capabilities.supports_host_monitoring is True
    prefix = list(
        backend.monitor_delivery(
            node_name="dgx-a",
            scopes=["cpu"],
        ).launch_prefix
    )
    assert prefix[1:3] == ["--host", "tcp://dgx-a:2376"]


# ---------------------------------------------------------------------------
# dry_run_details (synthetic-node branch)
# ---------------------------------------------------------------------------


def test_dry_run_details_synthetic_node_config():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker",
            type="docker",
            image="ubuntu:22.04",
            nodes=1,
            gpus_per_node=4,
            workdir="/work",
            mounts=["/data:/data:ro"],
            extra_args=["--network=host"],
        )
    )

    details = dict(backend.dry_run_details())

    assert details["image"] == "ubuntu:22.04"
    assert details["nodes"] == "1"
    assert details["gpus_per_node"] == "4"
    assert details["workdir"] == "/work"
    assert "/data:/data:ro" in details["mounts"]
    assert "--network=host" in details["extra_args"]
    # synthetic-node config must not describe a hosts row
    assert "hosts" not in details


# ---------------------------------------------------------------------------
# build_command flag flow
# ---------------------------------------------------------------------------


def test_build_command_includes_workdir_mounts_extra_args():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker",
            type="docker",
            image="ubuntu:22.04",
            workdir="/work",
            mounts=["/data:/data:ro"],
            extra_args=["--network=host"],
        )
    )
    operator = backend.default_operator(
        name="default_docker", assigned_nodes=["localhost"]
    )
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["localhost"], artifacts=[]
    )

    # envs={} keeps this a bare `docker run` (no log offload wrapper).
    parts = operator.build_command(
        task_name="job", script=["echo hi"], envs={}
    ).as_list()

    assert parts[parts.index("-w") + 1] == "/work"
    assert parts[parts.index("-v") + 1] == "/data:/data:ro"
    assert "--network=host" in parts


def test_build_command_pass_envs_false_omits_task_env_keys():
    operator = DockerRunOperator(
        DockerRunOperatorConfig(
            name="t", image="ubuntu:22.04", pass_envs=False, log_to_file=False
        )
    )
    parts = operator.build_command(
        task_name="job", script=["echo hi"], envs={"TOKEN": "x"}
    ).as_list()
    # The task env key is not forwarded ...
    assert "TOKEN" not in parts
    # ... but GPU isolation is independent of pass_envs: no GPUs claimed means
    # the container is explicitly pinned to none.
    assert "NVIDIA_VISIBLE_DEVICES=void" in parts


def test_build_command_no_gpus_hides_all_devices():
    operator = DockerRunOperator(
        DockerRunOperatorConfig(name="t", image="ubuntu:22.04", log_to_file=False)
    )
    parts = operator.build_command(
        task_name="job", script=["echo hi"], envs={}
    ).as_list()
    # Default-none: absent an explicit GPU request, expose zero GPUs (overriding
    # images that bake NVIDIA_VISIBLE_DEVICES=all).
    assert "--gpus" not in parts
    assert parts[parts.index("-e") + 1] == "NVIDIA_VISIBLE_DEVICES=void"


def test_build_command_with_gpus_does_not_add_void():
    operator = DockerRunOperator(
        DockerRunOperatorConfig(
            name="t", image="ubuntu:22.04", gpus="device=GPU-abc", log_to_file=False
        )
    )
    parts = operator.build_command(
        task_name="job", script=["echo hi"], envs={}
    ).as_list()
    # Quoted on the way out: a bare `device=` list is what docker mis-parses as a
    # count (see test_hand_written_gpu_device_lists_are_quoted_for_docker). What
    # this test is actually about is the line below -- an explicit GPU request
    # must not also get the default-none `void`.
    assert parts[parts.index("--gpus") + 1] == '"device=GPU-abc"'
    assert "NVIDIA_VISIBLE_DEVICES=void" not in parts


def test_build_command_gpus_all_literal():
    operator = DockerRunOperator(
        DockerRunOperatorConfig(
            name="t", image="ubuntu:22.04", gpus="all", log_to_file=False
        )
    )
    parts = operator.build_command(
        task_name="job", script=["echo hi"], envs={}
    ).as_list()
    assert parts[parts.index("--gpus") + 1] == "all"


# ---------------------------------------------------------------------------
# apply_backend_context: remote host disables auto-mount
# ---------------------------------------------------------------------------


def test_apply_backend_context_disables_auto_mount_for_remote_host():
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
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["dgx-a"], artifacts=[]
    )
    # Remote host paths don't exist on the driver, so auto-mounting is disabled.
    assert operator.config.auto_mount_runtime_dirs is False


def test_apply_backend_context_keeps_auto_mount_for_local_node():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker", type="docker", image="ubuntu:22.04", nodes=1
        )
    )
    operator = backend.default_operator(
        name="default_docker", assigned_nodes=["localhost"]
    )
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["localhost"], artifacts=[]
    )
    assert operator.config.auto_mount_runtime_dirs is True


# ---------------------------------------------------------------------------
# build_command: local multi-node wrapper (bare docker, no --host/--context)
# ---------------------------------------------------------------------------


def test_build_command_injects_per_host_mounts_and_extra_args():
    """A host's own mounts/extra_args (not just backend-level ones) must be
    merged into that node's `docker run`."""
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=[
                {
                    "name": "dgx-a",
                    "docker_host": "ssh://dgx-a",
                    "mounts": ["/nvme/dgx-a:/data:rw"],
                    "extra_args": ["--shm-size=32g"],
                }
            ],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    operator = backend.default_operator(name="default_docker")
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["dgx-a"], artifacts=[]
    )

    parts = operator.build_command(
        task_name="job", script=["echo hi"], envs={}
    ).as_list()

    assert parts[parts.index("-v") + 1] == "/nvme/dgx-a:/data:rw"
    assert "--shm-size=32g" in parts


# ---------------------------------------------------------------------------
# dry_run_details: gpus row omitted when unset
# ---------------------------------------------------------------------------


def test_dry_run_details_omits_gpus_when_unset():
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker", type="docker", image="ubuntu:22.04", nodes=1
        )
    )
    details = dict(backend.dry_run_details())
    assert "gpus_per_node" not in details
    assert details["nodes"] == "1"


# ---------------------------------------------------------------------------
# _safe_container_name: sanitization / truncation / fallback
# ---------------------------------------------------------------------------


def test_safe_container_name_sanitizes_illegal_chars():
    # Spaces and '/' are illegal in a docker --name; each run collapses to '-'.
    assert _safe_container_name("sflow", "my job/v2", "node") == "sflow-my-job-v2-node"


def test_safe_container_name_truncates_to_128():
    name = _safe_container_name("sflow", "x" * 200)
    assert len(name) == 128


def test_safe_container_name_falls_back_when_empty():
    assert _safe_container_name("", "") == "sflow-task"


# ---------------------------------------------------------------------------
# orphan reap: the generated shell, executed for real against a stub `docker`
# ---------------------------------------------------------------------------


def _run_reap_script(tmp_path, fake_process, container_names: list[str]) -> list[str]:
    """Execute the real orphan-reap script with `docker` stubbed on PATH.

    Returns the container names the script chose to `rm -f`, so the shell's
    liveness logic is exercised rather than merely pattern-matched.
    """
    import subprocess

    # The unit suite's autouse fake_process blocks real subprocesses; this test
    # needs the actual shell to run so the liveness logic is genuinely executed.
    fake_process.allow_unregistered(True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)  # helper may run twice in one test
    removed = tmp_path / "removed.txt"
    removed.unlink(missing_ok=True)
    listing = "\n".join(container_names)
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "ps" ]; then\n'
        f"  printf '%s\\n' {shlex.quote(listing)}\n"
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = "rm" ]; then\n'
        f'  echo "$3" >> {shlex.quote(str(removed))}\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    stub.chmod(0o755)

    backend = DockerBackend(
        DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
    )
    operator = backend.default_operator(name="d", assigned_nodes=["localhost"])
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["localhost"], artifacts=[]
    )
    cmd = operator.stale_reap_commands(task_name="job")[0].as_list()

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    if not removed.exists():
        return []
    return [line for line in removed.read_text().split() if line]


def test_reap_removes_containers_whose_owner_is_dead(tmp_path, fake_process):
    # PID 999999999 cannot exist -> its container is an orphan from a crashed run.
    removed = _run_reap_script(tmp_path, fake_process, [f"{_OWNER_PREFIX}999999999-oldtask-localhost"])
    assert removed == [f"{_OWNER_PREFIX}999999999-oldtask-localhost"]


def test_reap_keeps_containers_of_a_live_owner(tmp_path, fake_process):
    # Our own PID is alive, so our own containers must survive the sweep.
    removed = _run_reap_script(tmp_path, fake_process, [f"{_OWNER_PREFIX}{os.getpid()}-mytask-localhost"])
    assert removed == []


def test_reap_keeps_containers_owned_by_another_users_live_run(tmp_path, fake_process):
    # THE cross-user hazard: `kill -0 <pid>` fails with EPERM for a process owned
    # by a different user, so a liveness check built on it alone would call a
    # co-tenant's running `sflow run` dead and force-remove its live containers.
    # PID 1 (init, root-owned) stands in for that other user's driver.
    removed = _run_reap_script(tmp_path, fake_process, [f"{_OWNER_PREFIX}1-othertask-localhost"])
    assert removed == [], "must not reap a live container owned by another user"


def test_reap_spares_containers_kept_on_purpose(tmp_path, fake_process):
    # `rm: false` renames a finished container to `<name>-kept-<ns>` so it stays
    # around for `docker logs`/`inspect`. Its owning driver is EXPECTED to be
    # gone by then, so the dead-owner rule would have the next run on this host
    # delete exactly the evidence the flag exists to preserve.
    kept = f"{_OWNER_PREFIX}999999999-server-localhost-kept-1700000000000000000"
    assert _run_reap_script(tmp_path, fake_process, [kept]) == []


def test_reap_ignores_names_without_a_numeric_pid(tmp_path, fake_process):
    removed = _run_reap_script(
        tmp_path, fake_process, [f"{_OWNER_PREFIX}notapid-x-localhost", f"{_OWNER_PREFIX}-localhost"]
    )
    assert removed == []


def test_reap_handles_a_mixed_and_empty_container_list(tmp_path, fake_process):
    assert _run_reap_script(tmp_path, fake_process, []) == []
    removed = _run_reap_script(
        tmp_path,
        fake_process,
        [
            f"{_OWNER_PREFIX}{os.getpid()}-live-localhost",
            f"{_OWNER_PREFIX}999999999-dead-localhost",
            f"{_OWNER_PREFIX}1-otheruser-localhost",
        ],
    )
    assert removed == [f"{_OWNER_PREFIX}999999999-dead-localhost"]


# ---------------------------------------------------------------------------
# NVIDIA_VISIBLE_DEVICES=void must not override an explicit GPU request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--gpus", "all"],
        ["--gpus=all"],
        ["--runtime=nvidia"],
        ["-e", "NVIDIA_VISIBLE_DEVICES=all"],
        ["--device=/dev/nvidia0"],
        # extra_args normalization shell-splits entries, so the value of a flag
        # can land in the next token -- both spellings must be recognized.
        ["--device", "/dev/nvidia0"],
        ["--device /dev/nvidiactl"],
        ["--env", "NVIDIA_VISIBLE_DEVICES=all"],
    ],
    ids=[
        "gpus-split",
        "gpus-joined",
        "runtime-nvidia",
        "explicit-env",
        "device-joined",
        "device-split",
        "device-bundled",
        "env-long-flag",
    ],
)
def test_void_is_not_injected_when_extra_args_already_grant_gpus(extra_args):
    # A recipe may expose GPUs through raw docker args rather than the operator's
    # `gpus:` field. Injecting NVIDIA_VISIBLE_DEVICES=void there would silently
    # revoke exactly what the user asked for.
    op = DockerRunOperator(
        DockerRunOperatorConfig(
            name="t", image="busybox", log_to_file=False, extra_args=extra_args
        )
    )
    parts = op.build_command(task_name="t", script=["echo hi"], envs={}).as_list()
    assert "NVIDIA_VISIBLE_DEVICES=void" not in parts


def test_void_is_still_injected_for_an_ordinary_cpu_task():
    op = DockerRunOperator(
        DockerRunOperatorConfig(
            name="t", image="busybox", log_to_file=False, extra_args=["--network=host"]
        )
    )
    parts = op.build_command(task_name="t", script=["echo hi"], envs={}).as_list()
    assert "NVIDIA_VISIBLE_DEVICES=void" in parts


# ---------------------------------------------------------------------------
# wait_for_gpus: recipe field resolution + validation
# ---------------------------------------------------------------------------


def _resolve_wait(wait_for_gpus, resolver=None):
    """Run wait_for_gpus through the real config-resolution path."""
    conf = DockerBackendConfig(
        name="docker", type="docker", image="ubuntu:22.04", wait_for_gpus=wait_for_gpus
    )
    return DockerBackend.resolve_config(
        conf, resolver=resolver or _IdentityResolver(), ctx={}, workflow_name="wf"
    )


def test_resolve_config_reads_wait_for_gpus():
    resolved = _resolve_wait(120)
    assert resolved.wait_for_gpus == 120
    assert DockerBackend(resolved).wait_for_gpus_setting == 120


def test_resolve_config_wait_for_gpus_absent_is_none():
    resolved = _resolve_wait(None)
    assert resolved.wait_for_gpus is None
    assert DockerBackend(resolved).wait_for_gpus_setting is None


def test_resolve_config_wait_for_gpus_zero_means_wait_forever():
    resolved = _resolve_wait(0)
    assert resolved.wait_for_gpus == 0


def test_resolve_config_wait_for_gpus_template_non_int_raises():
    class _Resolver:
        def resolve(self, value, ctx):
            return "soon" if value == "${{ variables.W }}" else value

    with pytest.raises(ValueError, match="wait_for_gpus must resolve to int"):
        _resolve_wait("${{ variables.W }}", _Resolver())


def test_resolve_config_wait_for_gpus_negative_raises():
    with pytest.raises(ValueError, match="wait_for_gpus must be >= 0"):
        _resolve_wait(-1)


def test_resolve_config_wait_for_gpus_resolves_a_template():
    class _Resolver:
        def resolve(self, value, ctx):
            return 45 if value == "${{ variables.W }}" else value

    resolved = _resolve_wait("${{ variables.W }}", _Resolver())
    assert resolved.wait_for_gpus == 45


def test_unrelated_flag_values_do_not_look_like_gpu_grants():
    # The paired-token scan must not treat any `-e`/`--device` as a GPU grant.
    op = DockerRunOperator(
        DockerRunOperatorConfig(
            name="t",
            image="busybox",
            log_to_file=False,
            extra_args=["-e", "FOO=bar", "--device", "/dev/fuse"],
        )
    )
    parts = op.build_command(task_name="t", script=["echo hi"], envs={}).as_list()
    assert "NVIDIA_VISIBLE_DEVICES=void" in parts


def test_resolve_config_resolves_every_declared_host_field():
    """`resolve_config` rebuilds DockerHostConfig by hand-listing its fields.

    That list and the model are two places holding one fact. Add an eleventh
    field and it parses, validates, and is accepted -- but its `${{ }}` is never
    substituted, so the raw template string reaches a docker flag. This pins
    them together so the drift fails here instead of at runtime.
    """
    resolver = _MapResolver(
        {
            "${{ v.name }}": "h0",
            "${{ v.address }}": "10.0.0.1",
            "${{ v.user }}": "svc",
            "${{ v.rendezvous }}": "10.1.0.1",
            "${{ v.gpus }}": 4,
            "${{ v.mount }}": "/data:/data",
            "${{ v.extra }}": "--shm-size=1g",
        }
    )
    conf = DockerBackendConfig(
        name="cluster",
        type="docker",
        image="img:1",
        hosts=[
            {
                "name": "${{ v.name }}",
                "address": "${{ v.address }}",
                "user": "${{ v.user }}",
                "rendezvous_address": "${{ v.rendezvous }}",
                "gpus_per_node": "${{ v.gpus }}",
                "mounts": ["${{ v.mount }}"],
                "extra_args": ["${{ v.extra }}"],
            }
        ],
    )
    resolved = DockerBackend.resolve_config(
        conf, resolver=resolver, ctx={}, workflow_name="wf"
    )
    host = resolved.hosts[0]

    # Every field the model declares must come back substituted. Fields this
    # case leaves unset (docker_host/context/ip_address) are mutually exclusive
    # with the ones above, so they are asserted absent rather than resolved.
    for field_name in DockerHostConfig.model_fields:
        value = getattr(host, field_name)
        assert "${{" not in str(value), (
            f"DockerHostConfig.{field_name} was not resolved by "
            f"DockerBackend.resolve_config: got {value!r}"
        )


def test_a_node_count_that_contradicts_the_host_pool_is_rejected():
    """`hosts:` defines the nodes, so a different `nodes:` is not an override --
    it is a number the run will not honour.

    Asking for 4 nodes against a 2-host pool silently produced 2: fewer workers
    than the recipe sizes its parallelism for. That is the same class of untrue
    configuration `_reject_invented_gpus` already refuses.
    """
    with pytest.raises(ValueError) as excinfo:
        DockerBackend(
            DockerBackendConfig(
                name="cluster",
                type="docker",
                image="img:1",
                nodes=4,
                hosts=[
                    {"name": "h1", "address": "10.0.0.1"},
                    {"name": "h2", "address": "10.0.0.2"},
                ],
            )
        )
    message = str(excinfo.value)
    assert "4" in message and "2" in message
    assert "hosts" in message


def test_a_node_count_that_agrees_with_the_pool_is_accepted():
    """Redundant is not wrong: the number is true, just already implied."""
    backend = DockerBackend(
        DockerBackendConfig(
            name="cluster",
            type="docker",
            image="img:1",
            nodes=2,
            hosts=[
                {"name": "h1", "address": "10.0.0.1"},
                {"name": "h2", "address": "10.0.0.2"},
            ],
        )
    )
    assert backend._nodes == 2


def test_a_pool_without_an_explicit_node_count_is_unaffected():
    backend = DockerBackend(
        DockerBackendConfig(
            name="cluster",
            type="docker",
            image="img:1",
            hosts=[{"name": "h1", "address": "10.0.0.1"}],
        )
    )
    assert backend._nodes == 1


class TestBackendImageIsOptional:
    """The backend `image:` is only the default for operator-less tasks and the
    monitor container on non-ssh hosts. A recipe whose tasks all name an operator
    had to repeat an image nothing used; now it can leave it out, and the two
    consumers that do need one say so instead of failing somewhere downstream."""

    @staticmethod
    def _backend(**kwargs):
        from sflow.plugins.backends.docker import DockerBackend, DockerBackendConfig

        return DockerBackend(DockerBackendConfig(name="pool", type="docker", **kwargs))

    def test_a_backend_without_image_builds_and_reports_none(self):
        backend = self._backend(hosts=[{"name": "a", "address": "10.0.0.11"}])
        assert backend.config.container_images() == []
        assert ("image", None) not in backend.dry_run_details()
        assert not [k for k, _ in backend.dry_run_details() if k == "image"]

    def test_an_operator_less_task_names_both_knobs(self):
        backend = self._backend(nodes=1)
        with pytest.raises(ValueError, match="'hello'.*no 'operator:'.*'pool'.*no 'image:'"):
            backend.default_operator(name="hello")

    def test_ssh_hosts_monitor_bare_without_an_image(self):
        backend = self._backend(hosts=[{"name": "a", "address": "10.0.0.11"}])
        delivery = backend.monitor_delivery(node_name="a", scopes=["cpu", "gpu"])
        assert delivery.launch_prefix[0] == "ssh"

    def test_a_context_host_needs_the_image_for_its_monitor_container(self):
        backend = self._backend(hosts=[{"name": "ctx", "context": "lab"}])
        with pytest.raises(ValueError, match="no ssh endpoint.*'image:'"):
            backend.monitor_delivery(node_name="ctx", scopes=["cpu"])
