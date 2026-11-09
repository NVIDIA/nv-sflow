# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`fs://` artifacts reach a docker_run container.

The srun operator auto-mounts local artifact paths so a task can read
`${{ artifacts.X.path }}` without the recipe repeating the path as a mount.
docker_run accepted the same `artifacts` argument and ignored it, so the identical
recipe worked on Slurm and failed on Docker with a missing file.

The remote case is deliberately different: a driver-local path is not known to
exist on another machine, and docker CREATES a missing bind source as an empty
directory -- so auto-mounting there would turn a portable recipe into a silently
empty one. That case warns instead.
"""

import asyncio

import pytest

from sflow.plugins.backends.docker import DockerBackend, DockerBackendConfig


class _Artifact:
    def __init__(self, uri: str, path: str):
        self.uri = uri
        self.path = path


def _local_operator(artifacts, **operator_kw):
    backend = DockerBackend(
        DockerBackendConfig(name="docker", type="docker", image="ubuntu:22.04")
    )
    operator = backend.default_operator(name="d", assigned_nodes=["localhost"])
    for key, value in operator_kw.items():
        setattr(operator.config, key, value)
    operator.apply_backend_context(
        backend=backend, assigned_nodes=["localhost"], artifacts=artifacts
    )
    return operator


def _mounts(operator) -> list[str]:
    parts = operator.build_command(
        task_name="t", script=["echo hi"], envs={}
    ).as_list()
    return [parts[i + 1] for i, arg in enumerate(parts) if arg == "-v"]


class TestLocalDaemon:
    def test_an_fs_artifact_directory_is_mounted(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()

        operator = _local_operator([_Artifact(f"fs://{model}", str(model))])

        assert f"{model}:{model}:rw" in _mounts(operator)

    def test_a_file_artifact_mounts_its_parent(self, tmp_path):
        weights = tmp_path / "data" / "weights.bin"
        weights.parent.mkdir()
        weights.write_text("x")

        operator = _local_operator([_Artifact(f"file://{weights}", str(weights))])

        assert f"{weights.parent}:{weights.parent}:rw" in _mounts(operator)

    def test_a_remote_uri_artifact_is_not_mounted(self, tmp_path):
        """s3:// and friends are fetched, not bind-mounted."""
        operator = _local_operator(
            [_Artifact("s3://bucket/model", str(tmp_path / "model"))]
        )

        assert not any(str(tmp_path) in mount for mount in _mounts(operator))

    def test_opting_out_of_runtime_mounts_also_opts_out_of_artifacts(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()

        operator = _local_operator(
            [_Artifact(f"fs://{model}", str(model))],
            auto_mount_runtime_dirs=False,
        )

        assert not any(str(model) in mount for mount in _mounts(operator))


class TestDeclaredHosts:
    @staticmethod
    def _remote_operator(artifacts):
        backend = DockerBackend(
            DockerBackendConfig(
                name="cluster",
                type="docker",
                image="ubuntu:22.04",
                hosts=[{"name": "node-a", "address": "10.0.0.11"}],
            )
        )
        backend.allocation = asyncio.run(backend.allocate())
        operator = backend.default_operator(name="d")
        operator.apply_backend_context(
            backend=backend, assigned_nodes=["node-a"], artifacts=artifacts
        )
        return operator

    def test_artifacts_are_mounted_on_remote_hosts_too(self, tmp_path):
        """The path must exist there, but that is pre-flight's job to say.

        Refusing to mount would leave the same recipe working on Slurm and
        quietly broken here, which is the bug this whole behaviour fixes.
        """
        model = tmp_path / "model"
        model.mkdir()

        operator = self._remote_operator([_Artifact(f"fs://{model}", str(model))])

        assert f"{model}:{model}:rw" in _mounts(operator)

    def test_remote_hosts_still_skip_the_runtime_workspace_dirs(self, tmp_path):
        """Artifacts bypass that flag; workspace/output dirs do not."""
        model = tmp_path / "model"
        model.mkdir()

        operator = self._remote_operator([_Artifact(f"fs://{model}", str(model))])

        assert operator.config.auto_mount_runtime_dirs is False
        assert f"{model}:{model}:rw" in _mounts(operator)


@pytest.mark.parametrize("uri", ["fs:///data/model.sqsh", "file:///data/model.sqsh"])
def test_a_squashfs_image_is_not_mounted(uri):
    """It is an image to load, not a directory to bind."""
    operator = _local_operator([_Artifact(uri, "/data/model.sqsh")])

    assert not any("model.sqsh" in mount for mount in _mounts(operator))
