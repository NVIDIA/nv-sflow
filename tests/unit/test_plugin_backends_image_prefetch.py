# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each backend's ``prefetch_image`` primitive: one image onto some nodes."""

import asyncio
import json
import logging

import pytest

from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.core.task import Task
from sflow.plugins.backends.docker import DockerBackend, DockerBackendConfig
from sflow.plugins.backends.kubernetes import (
    KubernetesBackend,
    KubernetesBackendConfig,
)
from sflow.plugins.backends.slurm import SlurmBackend, SlurmBackendConfig
from sflow.plugins.operators.k8s import K8sOperator, K8sOperatorConfig

NIM = "nvcr.io/nim/meta/llama:1.0"


class _FakeLauncher:
    """Replays (exit_code, output_lines) per call and records each command."""

    def __init__(self, script):
        self._script = list(script)
        self.commands: list[list[str]] = []

    async def run_async(self, command, shell=False, output_logger=None, **_kw):
        await asyncio.sleep(0)
        exit_code, lines = self._script.pop(0)
        self.commands.append([str(c) for c in command])
        if output_logger:
            for line in lines:
                output_logger.info(line)
        return exit_code


class _RoutingLauncher:
    """Answers each command by the first matching rule; order-independent."""

    def __init__(self, rules):
        self._rules = rules  # [(predicate(cmd) -> bool, exit_code, lines)]
        self.commands: list[list[str]] = []

    async def run_async(self, command, shell=False, output_logger=None, **_kw):
        await asyncio.sleep(0)
        cmd = [str(c) for c in command]
        self.commands.append(cmd)
        for matches, exit_code, lines in self._rules:
            if matches(cmd):
                if output_logger:
                    for line in lines:
                        output_logger.info(line)
                return exit_code
        return 0


def _allocation(name, nodes):
    return Allocation(
        allocation_id=name,
        nodes=[
            ComputeNode(name=n, ip_address=f"10.0.0.{i}", index=i)
            for i, n in enumerate(nodes)
        ],
    )


# -- Slurm --------------------------------------------------------------------


def _slurm(launcher):
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="slurm",
            account="a",
            partition="p",
            time="00:10:00",
            nodes=2,
            gpus_per_node=8,
        )
    )
    backend.allocation = _allocation("42", ["n1", "n2"])
    backend._subprocess_launcher = launcher
    return backend


def test_slurm_pulls_with_one_overlapping_srun_step_per_image():
    launcher = _FakeLauncher([(0, [])])

    result = asyncio.run(_slurm(launcher).prefetch_image(NIM, ["n1", "n2"], []))

    assert result.failed_nodes == []
    assert launcher.commands == [
        [
            "srun",
            "--jobid",
            "42",
            "--nodelist",
            "n1,n2",
            "--ntasks-per-node=1",
            "--overlap",
            "--job-name=sflow_image_prefetch",
            f"--container-image={NIM}",
            "true",
        ]
    ]


def test_slurm_reports_the_nodes_srun_names_as_failed_with_pyxis_output():
    launcher = _FakeLauncher(
        [
            (
                1,
                [
                    "pyxis: [ERROR] URL https://nvcr.io/v2/... returned error code: 401",
                    "srun: error: n2: task 1: Exited with exit code 1",
                ],
            )
        ]
    )

    result = asyncio.run(_slurm(launcher).prefetch_image(NIM, ["n1", "n2"], []))

    assert result.failed_nodes == ["n2"]
    assert "401" in result.output


def test_slurm_failure_naming_no_node_fails_every_node():
    launcher = _FakeLauncher([(1, ["srun: error: Unable to create step"])])

    result = asyncio.run(_slurm(launcher).prefetch_image(NIM, ["n1", "n2"], []))

    assert result.failed_nodes == ["n1", "n2"]


# -- Docker -------------------------------------------------------------------


def _docker(launcher):
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker",
            hosts=[
                {"name": "h1", "address": "10.0.0.1"},
                {"name": "h2", "address": "10.0.0.2"},
            ],
        )
    )
    backend.allocation = backend.placeholder_allocation()
    backend._subprocess_launcher = launcher
    return backend


def _on(host, verb):
    return lambda cmd: f"ssh://{host}" in cmd and verb in cmd


def test_docker_pulls_only_on_hosts_missing_the_image():
    launcher = _RoutingLauncher(
        [
            (_on("10.0.0.1", "inspect"), 0, []),
            (_on("10.0.0.2", "inspect"), 1, ["Error: No such image"]),
            (_on("10.0.0.2", "pull"), 0, []),
        ]
    )

    result = asyncio.run(_docker(launcher).prefetch_image(NIM, ["h1", "h2"], []))

    assert result.failed_nodes == []
    assert sorted(launcher.commands) == sorted(
        [
            ["docker", "--host", "ssh://10.0.0.1", "image", "inspect", NIM],
            ["docker", "--host", "ssh://10.0.0.2", "image", "inspect", NIM],
            ["docker", "--host", "ssh://10.0.0.2", "pull", NIM],
        ]
    )


def test_docker_reports_a_host_whose_pull_fails_with_its_output():
    launcher = _RoutingLauncher(
        [
            (lambda cmd: "inspect" in cmd, 1, ["Error: No such image"]),
            (_on("10.0.0.1", "pull"), 1, ["unauthorized: authentication required"]),
        ]
    )

    result = asyncio.run(_docker(launcher).prefetch_image(NIM, ["h1", "h2"], []))

    assert result.failed_nodes == ["h1"]
    assert "h1" in result.output
    assert "unauthorized: authentication required" in result.output


def test_docker_without_a_pool_pulls_on_the_local_daemon():
    launcher = _RoutingLauncher([(lambda cmd: "inspect" in cmd, 1, [])])
    backend = DockerBackend(DockerBackendConfig(name="docker", image="ubuntu:22.04"))
    backend.allocation = backend.placeholder_allocation()
    backend._subprocess_launcher = launcher

    result = asyncio.run(backend.prefetch_image(NIM, ["localhost"], []))

    assert result.failed_nodes == []
    assert launcher.commands == [
        ["docker", "image", "inspect", NIM],
        ["docker", "pull", NIM],
    ]


# -- Kubernetes ---------------------------------------------------------------


class _FakeCluster:
    """Stands in for kubectl: records applied pods, reports a scripted status."""

    def __init__(self, statuses=None, apply_error=None, get_error=None):
        # node -> containerStatus dict reported once the pod exists
        self._statuses = statuses or {}
        self._apply_error = apply_error
        self._get_error = get_error
        self.applied: dict[str, dict] = {}
        self.deleted: list[str] = []

    async def apply_manifest(self, manifest):
        if self._apply_error:
            raise RuntimeError(self._apply_error)
        self.applied[manifest["metadata"]["name"]] = manifest

    async def kubectl(self, args, *, timeout=None, namespaced=True):
        args = [str(a) for a in args]
        if args[:2] == ["get", "pod"]:
            if self._get_error:
                return 1, "", self._get_error
            pod = self.applied[args[2]]
            node = pod["spec"]["nodeSelector"]["kubernetes.io/hostname"]
            status = self._statuses.get(node, {"imageID": "sha256:abc"})
            return 0, json.dumps({"status": {"containerStatuses": [status]}}), ""
        if args[:2] == ["delete", "pod"]:
            self.deleted.append(args[2])
            return 0, "", ""
        raise AssertionError(f"unexpected kubectl {args}")


def _k8s(cluster, monkeypatch, **cfg):
    backend = KubernetesBackend(
        KubernetesBackendConfig(name="k8s", type="kubernetes", namespace="ns", **cfg)
    )
    backend.allocation = _allocation("abc123", ["gpu-a", "gpu-b"])
    monkeypatch.setattr(backend, "_apply_manifest", cluster.apply_manifest)
    monkeypatch.setattr(backend, "_kubectl", cluster.kubectl)
    return backend


def _k8s_task(name, **op_kwargs):
    op = K8sOperator(K8sOperatorConfig(name=name, image=NIM, **op_kwargs))
    return Task(name=name, logger=logging.getLogger(name), operator=op)


def _pods_by_node(cluster):
    return {
        pod["spec"]["nodeSelector"]["kubernetes.io/hostname"]: pod
        for pod in cluster.applied.values()
    }


def test_k8s_pulls_with_one_short_pinned_pod_per_node_then_deletes_it(monkeypatch):
    cluster = _FakeCluster()
    backend = _k8s(cluster, monkeypatch, image_pull_policy="IfNotPresent")
    tasks = [
        _k8s_task("nim_0", image_pull_secrets=["nvcr"]),
        _k8s_task("nim_1", image_pull_secrets=["nvcr", "gitlab"]),
    ]

    result = asyncio.run(backend.prefetch_image(NIM, ["gpu-a", "gpu-b"], tasks))

    assert result.failed_nodes == []
    pods = _pods_by_node(cluster)
    assert sorted(pods) == ["gpu-a", "gpu-b"]
    pod = pods["gpu-a"]
    (container,) = pod["spec"]["containers"]
    assert container["image"] == NIM
    assert container["imagePullPolicy"] == "IfNotPresent"
    # Holds no GPU, so it schedules beside the placeholder that reserves the node.
    assert "nvidia.com/gpu" not in json.dumps(container.get("resources", {}))
    assert pod["spec"]["restartPolicy"] == "Never"
    assert pod["spec"]["imagePullSecrets"] == [{"name": "nvcr"}, {"name": "gitlab"}]
    assert pod["metadata"]["labels"]["sflow.ai/allocation"] == "abc123"
    assert sorted(cluster.deleted) == sorted(cluster.applied)


def test_k8s_task_pull_policy_wins_over_the_backend_default(monkeypatch):
    cluster = _FakeCluster()
    backend = _k8s(cluster, monkeypatch, image_pull_policy="IfNotPresent")

    asyncio.run(
        backend.prefetch_image(
            NIM, ["gpu-a"], [_k8s_task("nim", image_pull_policy="Always")]
        )
    )

    (pod,) = cluster.applied.values()
    assert pod["spec"]["containers"][0]["imagePullPolicy"] == "Always"


@pytest.mark.parametrize(
    "reason",
    ["ErrImagePull", "ImagePullBackOff", "InvalidImageName", "ErrImageNeverPull"],
)
def test_k8s_reports_a_node_stuck_on_a_pull_error_as_failed(monkeypatch, reason):
    cluster = _FakeCluster(
        statuses={
            "gpu-b": {
                "imageID": "",
                "state": {"waiting": {"reason": reason, "message": "401 Unauthorized"}},
            }
        }
    )
    backend = _k8s(cluster, monkeypatch)

    result = asyncio.run(backend.prefetch_image(NIM, ["gpu-a", "gpu-b"], []))

    assert result.failed_nodes == ["gpu-b"]
    assert reason in result.output
    assert "401 Unauthorized" in result.output
    assert sorted(cluster.deleted) == sorted(cluster.applied)


def test_k8s_pod_rejected_at_admission_fails_its_node(monkeypatch):
    cluster = _FakeCluster(apply_error="exceeded quota: pods")
    backend = _k8s(cluster, monkeypatch)

    result = asyncio.run(backend.prefetch_image(NIM, ["gpu-a"], []))

    assert result.failed_nodes == ["gpu-a"]
    assert "exceeded quota" in result.output


@pytest.mark.parametrize(
    "error",
    [
        'Error from server (NotFound): pods "sflow-pull-x" not found',
        "Unable to connect to the server: dial tcp: i/o timeout",
    ],
)
def test_k8s_pull_pod_that_cannot_be_read_fails_instead_of_polling_forever(
    monkeypatch, error
):
    import sflow.plugins.backends.kubernetes as k8s_mod

    monkeypatch.setattr(k8s_mod, "_IMAGE_PREFETCH_POLL_S", 0)
    cluster = _FakeCluster(get_error=error)
    backend = _k8s(cluster, monkeypatch)

    result = asyncio.run(backend.prefetch_image(NIM, ["gpu-a"], []))

    assert result.failed_nodes == ["gpu-a"]
    assert error in result.output
    assert sorted(cluster.deleted) == sorted(cluster.applied)
