# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognising a backend that runs tasks somewhere other than the controller.

A local `fs://` path means nothing on a machine that is not this one, so sflow
must not hard-validate it against the controller's filesystem when the work runs
elsewhere. Two checks implement that, at different stages, and they have to
agree:

  * ``config_uses_offhost_backend`` -- before resolution, each backend config
    answers ``executes_offhost()``;
  * ``backends_execute_offhost``    -- after resolution, from capabilities.

Docker is the case neither originally handled. It is off-host *conditionally*:
the same backend type runs on this machine with no `hosts:`, and on other
machines with one. Missing that made a run from a controller outside the pool
fail on artifact paths that exist on every host that actually matters.
"""

import asyncio

from sflow.app.run_support import (
    backends_execute_offhost,
    backends_inject_inline_content,
    config_uses_offhost_backend,
)
from sflow.config.schema import BackendConfig
from sflow.plugins.backends.docker import DockerBackend, DockerBackendConfig
from sflow.plugins.backends.kubernetes import KubernetesBackendConfig


class _Config:
    def __init__(self, backends):
        self.backends = backends


class _State:
    def __init__(self, backends):
        self.backends = backends


def _docker_config(**kwargs) -> DockerBackendConfig:
    return DockerBackendConfig(
        name="docker", type="docker", image="ubuntu:22.04", **kwargs
    )


def _docker_backend(**kwargs) -> DockerBackend:
    backend = DockerBackend(_docker_config(**kwargs))
    backend.allocation = asyncio.run(backend.allocate())
    return backend


_HOSTS = [
    {"name": "node-a", "address": "10.0.0.11"},
    {"name": "node-b", "address": "10.0.0.12"},
]


class TestBeforeResolution:
    def test_a_docker_pool_runs_off_host(self):
        config = _Config([_docker_config(hosts=_HOSTS)])

        assert config_uses_offhost_backend(config) is True

    def test_a_local_docker_backend_does_not(self):
        """Without `hosts:` the containers run on this very machine."""
        config = _Config([_docker_config()])

        assert config_uses_offhost_backend(config) is False

    def test_an_empty_hosts_list_is_still_local(self):
        config = _Config([_docker_config(hosts=[])])

        assert config_uses_offhost_backend(config) is False

    def test_kubernetes_is_unchanged(self):
        config = _Config([KubernetesBackendConfig(name="k", type="kubernetes")])

        assert config_uses_offhost_backend(config) is True

    def test_local_and_slurm_are_unchanged(self):
        config = _Config(
            [BackendConfig(name="l", type="local"), BackendConfig(name="s", type="slurm")]
        )

        assert config_uses_offhost_backend(config) is False


class TestAfterResolution:
    def test_a_docker_pool_reports_no_host_path_mounts(self):
        backend = _docker_backend(hosts=_HOSTS)

        assert backend.capabilities.supports_host_path_mounts is False
        assert backends_execute_offhost(_State({"docker": backend})) is True

    def test_a_local_docker_backend_keeps_host_path_mounts(self):
        backend = _docker_backend(nodes=1)

        assert backend.capabilities.supports_host_path_mounts is True
        assert backends_execute_offhost(_State({"docker": backend})) is False

    def test_a_docker_pool_is_still_host_monitorable(self):
        """Off-host execution does NOT imply unmonitorable: the collector runs
        as a container on each host's own daemon. Unlike mounts, this one does
        not follow from `hosts:`."""
        backend = _docker_backend(hosts=_HOSTS)

        assert backend.capabilities.supports_host_monitoring is True


class TestMountValidationSkipsOffhostTasks:
    """Stat-ing the controller says nothing about a path on another machine.

    Running sflow from a box outside the pool produced one "mount source path
    does not exist" warning per task for a model present on every host that
    actually runs it. The host-side pre-flight is what can answer this.
    """

    @staticmethod
    def _task(name, backend_name, mount):
        class _Conf:
            def __init__(self, m):
                self.mounts = [m]

        class _Op:
            def __init__(self, m):
                self.config = _Conf(m)

        class _Task:
            def __init__(self):
                self.name = name
                self.backend_name = backend_name
                self.operator = _Op(mount)

        return _Task()

    def test_an_offhost_task_is_not_stat_ed_locally(self, tmp_path):
        from sflow.app.run_support import validate_container_mounts

        task = self._task("t", "docker_cluster", "/not/here:/not/here:ro")

        warnings = validate_container_mounts(
            [task],
            sflow_output_dir=tmp_path,
            offhost_backends={"docker_cluster"},
        )

        assert warnings == []

    def test_a_local_task_is_still_checked(self, tmp_path):
        from sflow.app.run_support import validate_container_mounts

        task = self._task("t", "local_docker", "/not/here:/not/here:ro")

        warnings = validate_container_mounts(
            [task], sflow_output_dir=tmp_path, offhost_backends=set()
        )

        assert len(warnings) == 1 and "/not/here" in warnings[0]


def test_both_checks_agree_for_every_docker_shape():
    """The two stages must not disagree, or validation flips mid-run."""
    for kwargs in ({"hosts": _HOSTS}, {"nodes": 1}):
        backend = _docker_backend(**kwargs)
        pre = config_uses_offhost_backend(_Config([_docker_config(**kwargs)]))
        post = backends_execute_offhost(_State({"docker": backend}))

        assert pre == post, f"stages disagree for {kwargs}"


def test_a_docker_pool_is_off_host_but_injects_nothing():
    """Off-host must not be read as "someone else delivers the file".

    `remote_filesystem` makes an ABSOLUTE file:// inline artifact skip being
    written on the controller, on the promise that the backend injects the
    content itself (k8s does, via a ConfigMap). A Docker pool runs off-host with
    no such mechanism, so conflating the two wrote that artifact nowhere at all
    -- and since the flag is global, it did so to the slurm tasks of a mixed
    workflow too.
    """
    state = _State({"docker": _docker_backend(hosts=_HOSTS)})

    assert backends_execute_offhost(state) is True
    assert backends_inject_inline_content(state) is False
