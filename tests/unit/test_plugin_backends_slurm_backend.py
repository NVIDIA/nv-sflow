# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import unittest.mock

import pytest

import sflow.plugins.backends.slurm as slurm_mod
from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.plugins.backends.slurm import SlurmBackend, SlurmBackendConfig


class _FakeSubprocessLauncher:
    def __init__(self, script: list[tuple[int, list[str]]]):
        # script is a list of (exit_code, output_lines) tuples, consumed per call
        self._script = list(script)
        self.calls: list[dict] = []

    async def run_async(self, command, shell: bool = False, output_logger=None) -> int:
        await asyncio.sleep(0)
        if not self._script:
            raise AssertionError("Unexpected extra run_async() call")

        exit_code, lines = self._script.pop(0)
        self.calls.append(
            {
                "command": command,
                "shell": shell,
                "output_logger": output_logger,
            }
        )

        if output_logger:
            for line in lines:
                output_logger.info(line)

        return exit_code


@pytest.fixture
def slurm_test_logger(monkeypatch) -> logging.Logger:
    logger = logging.getLogger("sflow.tests.plugins.backends.slurm_backend")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)

    monkeypatch.setattr(slurm_mod, "_logger", logger)
    return logger


def test_slurm_backend_config_merges_extra_args_without_duplicates():
    config = SlurmBackendConfig(
        name="test_backend",
        type="slurm",
        account="test_account",
        partition="batch",
        nodes=1,
        time="00:10:00",
        extra_args=["--exclusive"],
        gpus_per_node=8,
    )

    merged = config.merge_extra_args(["--exclusive", "--constraint=gpu"])

    assert merged is not config
    assert merged.extra_args == ["--exclusive", "--constraint=gpu"]
    assert config.extra_args == ["--exclusive"]


def test_merge_extra_args_dedups_by_option_cli_wins():
    """The shared option-key de-dup: a CLI --gres overrides the recipe's --gres
    (same mechanism sflow batch uses), rather than keeping both."""
    config = SlurmBackendConfig(
        name="b",
        type="slurm",
        account="a",
        partition="p",
        nodes=1,
        time="00:10:00",
        extra_args=["--exclusive", "--gres=gpu:8"],
        gpus_per_node=8,
    )

    merged = config.merge_extra_args(["--gres=gpu:4"])

    assert merged.extra_args == ["--exclusive", "--gres=gpu:4"]


def test_salloc_includes_nodelist_and_exclude_flags(monkeypatch, slurm_test_logger):
    # Host include/exclude become salloc --nodelist / --exclude, placed before the
    # user's extra_args so an explicit extra_args override still wins.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="b",
            type="slurm",
            account="acct",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="job",
            include_nodes=["node001", "node002"],
            exclude_nodes=["bad001"],
            gpus_per_node=8,
        )
    )
    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (0, ["salloc: Granted job allocation 1", "salloc: Nodes node001 are ready for job"]),
            (0, ["node001: 10.0.0.1:123"]),
        ]
    )
    backend._subprocess_launcher = fake_launcher
    asyncio.run(backend.allocate())

    salloc_cmd = list(fake_launcher.calls[0]["command"])
    assert "--nodelist" in salloc_cmd
    assert salloc_cmd[salloc_cmd.index("--nodelist") + 1] == "node001,node002"
    assert "--exclude" in salloc_cmd
    assert salloc_cmd[salloc_cmd.index("--exclude") + 1] == "bad001"
    # Node filters precede the user's extra_args.
    assert salloc_cmd.index("--nodelist") < salloc_cmd.index("--no-shell") + 2


def test_salloc_tokenizes_extra_args_with_bundled_whitespace(
    monkeypatch, slurm_test_logger
):
    # A --extra-salloc-args value can arrive as one token carrying stray leading
    # whitespace (e.g. the CI's " --exclude=n1,n2"). It must be tokenized into
    # clean argv, else salloc gets one unparsable " --exclude=..." arg and
    # silently ignores the node exclude (the multi-backend regression this guards).
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="b",
            type="slurm",
            account="acct",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="job",
            extra_args=[" --exclude=bad001,bad002", "--gpus-per-node=4"],
            gpus_per_node=8,
        )
    )
    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (0, ["salloc: Granted job allocation 1", "salloc: Nodes node001 are ready for job"]),
            (0, ["node001: 10.0.0.1:123"]),
        ]
    )
    backend._subprocess_launcher = fake_launcher
    asyncio.run(backend.allocate())

    salloc_cmd = list(fake_launcher.calls[0]["command"])
    # Clean, parseable tokens reach salloc (no leading-space variant survives).
    assert "--exclude=bad001,bad002" in salloc_cmd
    assert "--gpus-per-node=4" in salloc_cmd
    assert not any(arg != arg.strip() for arg in salloc_cmd)


def test_salloc_extra_args_keep_repeated_values(monkeypatch, slurm_test_logger):
    # `-e '-G 1 -p polar4 -A acct -N 1'` tokenizes to space-separated flag/value
    # pairs. Feeding each token through Command.add_opt() treated the bare values
    # as option names and de-duped them, so the second "1" deleted the first and
    # salloc received `-G -p polar4 -A acct -N 1` -- a silent, wrong allocation.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="b",
            type="slurm",
            account="acct",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="job",
            extra_args=["-G 1 -p polar4 -A general_perflab -N 1"],
            gpus_per_node=8,
        )
    )
    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (0, ["salloc: Granted job allocation 1", "salloc: Nodes node001 are ready for job"]),
            (0, ["node001: 10.0.0.1:123"]),
        ]
    )
    backend._subprocess_launcher = fake_launcher
    asyncio.run(backend.allocate())

    salloc_cmd = list(fake_launcher.calls[0]["command"])
    assert salloc_cmd[-8:] == [
        "-G",
        "1",
        "-p",
        "polar4",
        "-A",
        "general_perflab",
        "-N",
        "1",
    ]


def test_env_reuse_applies_exclude_filter(monkeypatch, slurm_test_logger):
    # A reused Slurm allocation can't take salloc flags, so exclude filters the
    # resolved node pool instead.
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[001-003]")

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="b",
            type="slurm",
            account="acct",
            partition="batch",
            nodes=3,
            time="00:10:00",
            exclude_nodes=["node002"],
            gpus_per_node=8,
        )
    )

    async def fake_resolve(*, nodelist, job_id):
        from sflow.core.compute_node import ComputeNode

        return [
            ComputeNode(name=f"node00{i}", ip_address=f"10.0.0.{i}", index=i - 1, num_gpus=8)
            for i in (1, 2, 3)
        ]

    monkeypatch.setattr(backend, "_resolve_nodes_from_nodelist", fake_resolve)
    allocation = asyncio.run(backend.allocate())
    assert [n.name for n in allocation.nodes] == ["node001", "node003"]
    assert allocation.owned is False


def test_resolve_config_preserves_node_filters():
    class _Id:
        def resolve(self, v, ctx):
            return v

    conf = SlurmBackendConfig(
        name="b",
        type="slurm",
        account="a",
        partition="p",
        nodes=1,
        time="1",
        gpus_per_node=1,
        include_nodes=["want-1"],
        exclude_nodes=["bad-1"],
    )
    resolved = SlurmBackend.resolve_config(
        conf, resolver=_Id(), ctx={}, workflow_name="wf"
    )
    assert resolved.include_nodes == ["want-1"]
    assert resolved.exclude_nodes == ["bad-1"]


def test_allocate_success_single_node(monkeypatch, slurm_test_logger):
    # Ensure we don't accidentally take the "reuse existing allocation" path.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            extra_args=["--exclusive"],
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (
                0,
                [
                    "salloc: Granted job allocation 1111111",
                    "salloc: Nodes node001 are ready for job",
                ],
            ),
            (
                0,
                [
                    "node001: 10.0.0.1:123",
                ],
            ),
        ]
    )
    backend._subprocess_launcher = fake_launcher
    warning_messages: list[str] = []

    class WarningCapture(logging.Handler):
        def emit(self, record: logging.LogRecord):
            if record.levelno >= logging.WARNING:
                warning_messages.append(record.getMessage())

    slurm_test_logger.addHandler(WarningCapture())

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "1111111"
    assert [n.name for n in allocation.nodes] == ["node001"]
    assert [n.ip_address for n in allocation.nodes] == ["10.0.0.1"]
    assert [n.index for n in allocation.nodes] == [0]
    assert [n.num_gpus for n in allocation.nodes] == [8]

    assert len(fake_launcher.calls) == 2
    first_cmd = fake_launcher.calls[0]["command"]
    assert list(first_cmd) == [
        "salloc",
        "--account",
        "test_account",
        "--partition",
        "batch",
        "--nodes",
        "1",
        "--time",
        "00:10:00",
        "--job-name",
        "test_job",
        "--no-shell",
        "--exclusive",
    ]
    assert fake_launcher.calls[1]["command"] == ["scontrol", "getaddrs", "node001"]
    assert any(
        "backend.gpus_per_node=8" in msg
        and "salloc" in msg
        and "srun" in msg
        and "sbatch" in msg
        for msg in warning_messages
    )


def test_allocate_success_multiple_nodes(monkeypatch, slurm_test_logger):
    # Ensure we don't accidentally take the "reuse existing allocation" path.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=2,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (
                0,
                [
                    "salloc: Granted job allocation 2222222",
                    "salloc: Nodes node001,node002 are ready for job",
                ],
            ),
            (
                0,
                [
                    "node001: 10.0.0.1:123",
                    "node002: 10.0.0.2:123",
                ],
            ),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "2222222"
    assert [(n.name, n.ip_address, n.index) for n in allocation.nodes] == [
        ("node001", "10.0.0.1", 0),
        ("node002", "10.0.0.2", 1),
    ]
    assert fake_launcher.calls[1]["command"] == [
        "scontrol",
        "getaddrs",
        "node001,node002",
    ]


def test_allocate_raises_when_salloc_fails(slurm_test_logger):
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(script=[(42, [])])
    backend._subprocess_launcher = fake_launcher

    with pytest.raises(RuntimeError, match=r"Failed to allocate nodes \(exit code 42\)"):
        asyncio.run(backend.allocate())

    assert len(fake_launcher.calls) == 1


def test_release_calls_scancel(slurm_test_logger):
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(script=[(0, [])])
    backend._subprocess_launcher = fake_launcher

    allocation = Allocation(allocation_id="1111111", nodes=[])
    asyncio.run(backend.release(allocation))

    assert len(fake_launcher.calls) == 1
    assert fake_launcher.calls[0]["command"] == ["scancel", "1111111"]


def test_emergency_release_calls_scancel(monkeypatch):
    calls = []

    class _Result:
        returncode = 0

    def _fake_run(command, stdout=None, stderr=None, check=None):
        calls.append(
            {
                "command": command,
                "stdout": stdout,
                "stderr": stderr,
                "check": check,
            }
        )
        return _Result()

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    monkeypatch.setattr(slurm_mod.shutil, "which", lambda cmd: "/usr/bin/scancel")
    monkeypatch.setattr(slurm_mod.subprocess, "run", _fake_run)

    backend.emergency_release(Allocation(allocation_id="1111111", nodes=[]))

    assert calls == [
        {
            "command": ["/usr/bin/scancel", "1111111"],
            "stdout": slurm_mod.subprocess.DEVNULL,
            "stderr": slurm_mod.subprocess.DEVNULL,
            "check": False,
        }
    ]


def test_allocate_reuses_env_allocation_and_skips_salloc(
    monkeypatch, slurm_test_logger
):
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=2,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    monkeypatch.setenv("SLURM_JOB_ID", "9999999")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node001,node002")

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (
                0,
                [
                    "node001: 10.0.0.1:123",
                    "node002: 10.0.0.2:123",
                ],
            )
        ]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "9999999"
    assert [(n.name, n.ip_address, n.index, n.num_gpus) for n in allocation.nodes] == [
        ("node001", "10.0.0.1", 0, 8),
        ("node002", "10.0.0.2", 1, 8),
    ]
    assert len(fake_launcher.calls) == 1
    assert fake_launcher.calls[0]["command"] == [
        "scontrol",
        "getaddrs",
        "node001,node002",
    ]


def test_allocate_multi_backend_wrapper_reuses_env_allocation(
    monkeypatch, slurm_test_logger
):
    """In per-backend-salloc mode (multi-backend `sflow batch`), the wrapper
    backend reuses the driver sbatch allocation instead of running salloc."""
    monkeypatch.delenv("SLURM_HET_SIZE", raising=False)
    monkeypatch.setenv("SFLOW_SLURM_MULTI_BACKEND_SALLOC", "1")
    monkeypatch.setenv("SFLOW_SLURM_WRAPPER_BACKEND", "cluster_a")
    monkeypatch.setenv("SLURM_JOB_ID", "9999999")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node001,node002")

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="cluster_a",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=2,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )
    fake_launcher = _FakeSubprocessLauncher(
        script=[(0, ["node001: 10.0.0.1:123", "node002: 10.0.0.2:123"])]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "9999999"
    assert allocation.owned is False
    # No salloc: only this backend's node resolution ran.
    assert len(fake_launcher.calls) == 1
    assert fake_launcher.calls[0]["command"] == [
        "scontrol",
        "getaddrs",
        "node001,node002",
    ]


def test_allocate_multi_backend_non_wrapper_sallocs_despite_env_allocation(
    monkeypatch, slurm_test_logger
):
    """In per-backend-salloc mode, a non-wrapper backend must run its OWN salloc
    (its own Slurm job id) even though the driver's SLURM_JOB_ID is in the env.

    This is what makes pyxis/enroot work on every partition: each backend's
    container steps key their per-job runtime dir on a job id that matches the
    node that provisioned it.
    """
    monkeypatch.delenv("SLURM_HET_SIZE", raising=False)
    monkeypatch.setenv("SFLOW_SLURM_MULTI_BACKEND_SALLOC", "1")
    monkeypatch.setenv("SFLOW_SLURM_WRAPPER_BACKEND", "cluster_a")
    # Driver allocation env is present, but cluster_b is NOT the wrapper, so it
    # must not reuse it.
    monkeypatch.setenv("SLURM_JOB_ID", "9999999")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node001,node002")

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="cluster_b",
            type="slurm",
            account="test_account",
            partition="other",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )
    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (
                0,
                [
                    "salloc: Granted job allocation 1002",
                    "salloc: Nodes node011 are ready for job",
                ],
            ),
            (0, ["node011: 10.0.0.11:123"]),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "1002"
    assert allocation.owned is True
    assert [n.name for n in allocation.nodes] == ["node011"]
    # salloc must have run (not a reuse): first call is salloc for this backend.
    salloc_cmd = list(fake_launcher.calls[0]["command"])
    assert salloc_cmd[0] == "salloc"
    assert "--partition" in salloc_cmd


def test_allocate_fallback_to_srun_when_scontrol_fails(monkeypatch, slurm_test_logger):
    """When scontrol getaddrs fails (non-zero exit), fall back to srun hostname -i."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=2,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            # salloc succeeds
            (
                0,
                [
                    "salloc: Granted job allocation 3333333",
                    "salloc: Nodes node001,node002 are ready for job",
                ],
            ),
            # scontrol getaddrs fails (e.g., not available on this cluster)
            (1, []),
            # srun fallback succeeds
            (
                0,
                [
                    "node001:10.0.0.1",
                    "node002:10.0.0.2",
                ],
            ),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "3333333"
    assert [(n.name, n.ip_address, n.index) for n in allocation.nodes] == [
        ("node001", "10.0.0.1", 0),
        ("node002", "10.0.0.2", 1),
    ]
    assert len(fake_launcher.calls) == 3
    # First call: salloc (Command object)
    salloc_cmd = fake_launcher.calls[0]["command"]
    assert "salloc" in salloc_cmd.as_str()
    # Second call: scontrol getaddrs (failed)
    assert fake_launcher.calls[1]["command"] == [
        "scontrol",
        "getaddrs",
        "node001,node002",
    ]
    # Third call: srun fallback
    srun_cmd = fake_launcher.calls[2]["command"]
    assert srun_cmd[0] == "srun"
    assert "--nodelist" in srun_cmd
    assert "node001,node002" in srun_cmd


def test_allocate_fallback_to_srun_when_scontrol_output_unparseable(
    monkeypatch, slurm_test_logger
):
    """When scontrol getaddrs returns success but output is unparseable, fall back to srun."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=4,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            # salloc succeeds
            (
                0,
                [
                    "salloc: Granted job allocation 4444444",
                    "salloc: Nodes node01 are ready for job",
                ],
            ),
            # scontrol getaddrs returns 0 but with unparseable output
            (0, ["some unexpected output format"]),
            # srun fallback succeeds
            (
                0,
                [
                    "node01:192.168.1.10",
                ],
            ),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "4444444"
    assert [(n.name, n.ip_address, n.num_gpus) for n in allocation.nodes] == [
        ("node01", "192.168.1.10", 4),
    ]
    assert len(fake_launcher.calls) == 3


def test_allocate_srun_fallback_raises_on_failure(monkeypatch, slurm_test_logger):
    """When both scontrol and srun fallback fail, raise an error."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=4,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            # salloc succeeds
            (
                0,
                [
                    "salloc: Granted job allocation 5555555",
                    "salloc: Nodes node01 are ready for job",
                ],
            ),
            # scontrol getaddrs fails
            (1, []),
            # srun fallback also fails
            (1, ["srun: error: Unable to create job step"]),
            # scancel of the orphaned allocation (salloc had granted job 5555555)
            (0, []),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    with pytest.raises(RuntimeError, match=r"Failed to resolve node addresses via srun"):
        asyncio.run(backend.allocate())

    # The granted job must be cancelled so it doesn't leak until --time expires.
    assert ["scancel", "5555555"] in [
        list(c["command"]) for c in fake_launcher.calls
    ]


def test_allocate_srun_fallback_with_env_allocation(monkeypatch, slurm_test_logger):
    """When using existing env allocation and scontrol fails, srun fallback includes job_id."""
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=2,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    monkeypatch.setenv("SLURM_JOB_ID", "7777777")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "gpu001,gpu002")

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            # scontrol getaddrs fails
            (1, []),
            # srun fallback succeeds
            (
                0,
                [
                    "gpu001:10.1.1.1",
                    "gpu002:10.1.1.2",
                ],
            ),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "7777777"
    assert [(n.name, n.ip_address) for n in allocation.nodes] == [
        ("gpu001", "10.1.1.1"),
        ("gpu002", "10.1.1.2"),
    ]

    # Verify srun fallback includes --jobid for the existing allocation
    srun_cmd = fake_launcher.calls[1]["command"]
    assert "--jobid" in srun_cmd
    assert "7777777" in srun_cmd


def test_srun_fallback_handles_duplicate_hostnames(monkeypatch, slurm_test_logger):
    """srun fallback should deduplicate hostnames if srun outputs them multiple times."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=2,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            # salloc succeeds
            (
                0,
                [
                    "salloc: Granted job allocation 8888888",
                    "salloc: Nodes n1,n2 are ready for job",
                ],
            ),
            # scontrol fails
            (1, []),
            # srun outputs duplicates
            (
                0,
                [
                    "n1:10.0.0.1",
                    "n1:10.0.0.1",  # duplicate
                    "n2:10.0.0.2",
                ],
            ),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    # Should have only 2 nodes, not 3
    assert len(allocation.nodes) == 2
    assert [(n.name, n.ip_address, n.index) for n in allocation.nodes] == [
        ("n1", "10.0.0.1", 0),
        ("n2", "10.0.0.2", 1),
    ]


def test_srun_fallback_raises_when_no_valid_output(monkeypatch, slurm_test_logger):
    """srun fallback should raise if it can't parse any valid hostname:ip pairs."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=4,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            # salloc succeeds
            (
                0,
                [
                    "salloc: Granted job allocation 9999999",
                    "salloc: Nodes node01 are ready for job",
                ],
            ),
            # scontrol fails
            (1, []),
            # srun succeeds but with unparseable output
            (
                0,
                [
                    "some garbage output",
                    "no colons here",
                ],
            ),
            # scancel of the orphaned allocation (salloc had granted job 9999999)
            (0, []),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    with pytest.raises(RuntimeError, match=r"Failed to parse any node addresses"):
        asyncio.run(backend.allocate())

    # The granted job must be cancelled so it doesn't leak until --time expires.
    assert ["scancel", "9999999"] in [
        list(c["command"]) for c in fake_launcher.calls
    ]


def test_srun_fallback_extracts_short_hostname_from_fqdn(monkeypatch, slurm_test_logger):
    """srun fallback should extract short hostname from FQDN (e.g., node01.cluster.example.com -> node01)."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=2,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            # salloc succeeds
            (
                0,
                [
                    "salloc: Granted job allocation 1234567",
                    "salloc: Nodes node01,node02 are ready for job",
                ],
            ),
            # scontrol fails
            (1, []),
            # srun outputs FQDNs
            (
                0,
                [
                    "node01.cluster.example.com:10.0.0.14",
                    "node02.cluster.example.com:10.0.0.15",
                ],
            ),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    # Should extract short hostnames
    assert [(n.name, n.ip_address) for n in allocation.nodes] == [
        ("node01", "10.0.0.14"),
        ("node02", "10.0.0.15"),
    ]


def test_allocate_omits_gpus_per_node_when_capacity_unknown(
    monkeypatch, slurm_test_logger
):
    """When backend GPU capacity is unknown (e.g. CPU-only partition declared
    with no gpus_per_node), salloc must not receive --gpus-per-node. Slurm
    rejects --gpus-per-node=0 on most clusters."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="cpu_backend",
            type="slurm",
            account="test_account",
            partition="cpu",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )
    # Simulate capacity-unknown state regardless of which branch's schema
    # policy applies; this exercises the `is not None` guard in allocate().
    backend._gpu_per_node = None

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (
                0,
                [
                    "salloc: Granted job allocation 6666666",
                    "salloc: Nodes cpu001 are ready for job",
                ],
            ),
            (0, ["cpu001: 10.0.0.5:123"]),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    asyncio.run(backend.allocate())

    salloc_cmd = list(fake_launcher.calls[0]["command"])
    assert "--gpus-per-node" not in salloc_cmd
    assert not any(a.startswith("--gpus-per-node=") for a in salloc_cmd)


def test_allocate_defers_to_user_extra_args_gpus_per_node_equals_form(
    monkeypatch, slurm_test_logger
):
    """User-provided --gpus-per-node=N in extra_args is passed through."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            extra_args=["--gpus-per-node=4"],
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (
                0,
                [
                    "salloc: Granted job allocation 1010101",
                    "salloc: Nodes node001 are ready for job",
                ],
            ),
            (0, ["node001: 10.0.0.1:123"]),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    asyncio.run(backend.allocate())

    salloc_cmd = list(fake_launcher.calls[0]["command"])
    gpn_args = [
        a
        for a in salloc_cmd
        if a == "--gpus-per-node" or a.startswith("--gpus-per-node=")
    ]
    assert gpn_args == ["--gpus-per-node=4"]


def test_allocate_defers_to_user_extra_args_gpus_per_node_separated_form(
    monkeypatch, slurm_test_logger
):
    """User-provided ['--gpus-per-node', 'N'] in extra_args is passed through."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            extra_args=["--gpus-per-node", "4"],
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (
                0,
                [
                    "salloc: Granted job allocation 2020202",
                    "salloc: Nodes node001 are ready for job",
                ],
            ),
            (0, ["node001: 10.0.0.1:123"]),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    asyncio.run(backend.allocate())

    salloc_cmd = list(fake_launcher.calls[0]["command"])
    assert salloc_cmd.count("--gpus-per-node") == 1
    gpn_idx = salloc_cmd.index("--gpus-per-node")
    assert salloc_cmd[gpn_idx + 1] == "4"


def test_cpu_only_slurm_backend_allocates_with_zero_gpu_capacity(
    monkeypatch, slurm_test_logger
):
    """`gpus_per_node=0` declares a CPU-only partition.

    Allocated ComputeNodes carry num_gpus=0, which downstream packing logic
    treats as "no GPUs available" (skip pack, reject `resources.gpus`).
    """
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="cpu_backend",
            type="slurm",
            account="test_account",
            partition="cpu",
            nodes=2,
            time="00:10:00",
            job_name="cpu_job",
            gpus_per_node=0,
        )
    )
    assert backend._gpu_per_node == 0

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            (
                0,
                [
                    "salloc: Granted job allocation 6666666",
                    "salloc: Nodes cpu001,cpu002 are ready for job",
                ],
            ),
            (
                0,
                [
                    "cpu001: 10.0.0.1:123",
                    "cpu002: 10.0.0.2:123",
                ],
            ),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "6666666"
    assert [(n.name, n.ip_address, n.index, n.num_gpus) for n in allocation.nodes] == [
        ("cpu001", "10.0.0.1", 0, 0),
        ("cpu002", "10.0.0.2", 1, 0),
    ]


def test_cpu_only_slurm_backend_via_env_allocation(monkeypatch, slurm_test_logger):
    """Reusing an env-provided allocation with `gpus_per_node=0` keeps num_gpus=0."""
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="cpu_backend",
            type="slurm",
            account="test_account",
            partition="cpu",
            nodes=1,
            time="00:10:00",
            job_name="cpu_job",
            gpus_per_node=0,
        )
    )

    monkeypatch.setenv("SLURM_JOB_ID", "1010101")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "cpu001")

    fake_launcher = _FakeSubprocessLauncher(
        script=[(0, ["cpu001: 10.0.0.1:123"])]
    )
    backend._subprocess_launcher = fake_launcher

    allocation = asyncio.run(backend.allocate())

    assert allocation.allocation_id == "1010101"
    assert [(n.name, n.num_gpus) for n in allocation.nodes] == [("cpu001", 0)]


def test_allocate_cancels_orphaned_salloc_job_on_resolution_failure(
    monkeypatch, slurm_test_logger
):
    """If salloc grants a job but node resolution later fails, allocate() must
    scancel the granted job before raising.

    Regression: ``allocate_resources`` only stores ``self.allocation`` once
    ``allocate()`` returns, so a failure *after* salloc granted a job leaves the
    job untracked. Neither the ``allocate_backends`` failure cleanup nor the
    atexit fallback can see it (both key off ``backend.allocation``), so it
    leaked until ``--time`` expired.
    """
    for var in ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_JOB_NODELIST", "SLURM_NODELIST"):
        monkeypatch.delenv(var, raising=False)

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=1,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )

    fake_launcher = _FakeSubprocessLauncher(
        script=[
            # salloc grants a job
            (
                0,
                [
                    "salloc: Granted job allocation 5550001",
                    "salloc: Nodes node01 are ready for job",
                ],
            ),
            # scontrol getaddrs fails
            (1, []),
            # srun fallback also fails -> allocate() raises
            (1, ["srun: error: Unable to create job step"]),
            # scancel of the orphaned allocation
            (0, []),
        ]
    )
    backend._subprocess_launcher = fake_launcher

    with pytest.raises(RuntimeError, match=r"Failed to resolve node addresses via srun"):
        asyncio.run(backend.allocate())

    scancel_calls = [
        list(c["command"])
        for c in fake_launcher.calls
        if list(c["command"])[:1] == ["scancel"]
    ]
    assert scancel_calls == [["scancel", "5550001"]]


def test_concurrent_slurm_allocations_do_not_crosstalk(monkeypatch, slurm_test_logger):
    """Allocating two Slurm backends concurrently must not contaminate each
    other's parsed salloc output.

    Regression: ``allocate_backends`` runs all backends in parallel via
    ``asyncio.gather``. Each ``allocate()`` attached its ``ParseLogHandler`` to the
    *shared* module logger and passed that same logger as ``output_logger``, so
    every parser received every backend's salloc lines. ``LinesParser`` then
    returned a *list* of both backends' values, turning the nodelist into a list
    that crashed ``scontrol getaddrs`` via ``shlex.join`` with
    "expected string or bytes-like object, got 'list'".
    """
    for var in ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_JOB_NODELIST", "SLURM_NODELIST"):
        monkeypatch.delenv(var, raising=False)

    class _Coordinator:
        def __init__(self):
            self.both_inside = asyncio.Event()
            self.both_emitted = asyncio.Event()
            self.entered = 0
            self.emitted = 0

    class _CrossTalkLauncher:
        """Forces both salloc emissions to happen while both backends' parser
        handlers are attached, deterministically reproducing the cross-talk."""

        def __init__(self, coord: _Coordinator, job_id: str, node: str):
            self.coord = coord
            self.job_id = job_id
            self.node = node
            self.calls: list[list] = []

        async def run_async(self, command, shell: bool = False, output_logger=None):
            cmd = list(command)
            self.calls.append(cmd)
            if cmd[:1] == ["salloc"]:
                self.coord.entered += 1
                if self.coord.entered >= 2:
                    self.coord.both_inside.set()
                await self.coord.both_inside.wait()

                if output_logger:
                    output_logger.info(
                        f"salloc: Granted job allocation {self.job_id}"
                    )
                    output_logger.info(
                        f"salloc: Nodes {self.node} are ready for job"
                    )

                # Keep both parsers attached until both have emitted.
                self.coord.emitted += 1
                if self.coord.emitted >= 2:
                    self.coord.both_emitted.set()
                await self.coord.both_emitted.wait()
                return 0

            # scontrol getaddrs <nodelist>
            assert cmd[:2] == ["scontrol", "getaddrs"], cmd
            if output_logger:
                output_logger.info(f"{self.node}: 10.0.0.{self.job_id[-1]}:123")
            return 0

    def _make_backend(name: str) -> SlurmBackend:
        return SlurmBackend(
            SlurmBackendConfig(
                name=name,
                type="slurm",
                account="test_account",
                partition="cpu",
                nodes=1,
                time="00:10:00",
                job_name="multi_backend_alloc_repro",
                gpus_per_node=0,
            )
        )

    backend_a = _make_backend("cluster_a")
    backend_b = _make_backend("cluster_b")

    async def _run_both():
        coord = _Coordinator()
        backend_a._subprocess_launcher = _CrossTalkLauncher(coord, "1001", "cpu-0001")
        backend_b._subprocess_launcher = _CrossTalkLauncher(coord, "1002", "cpu-0002")
        return await asyncio.gather(backend_a.allocate(), backend_b.allocate())

    alloc_a, alloc_b = asyncio.run(_run_both())

    assert alloc_a.allocation_id == "1001"
    assert alloc_b.allocation_id == "1002"
    assert [n.name for n in alloc_a.nodes] == ["cpu-0001"]
    assert [n.name for n in alloc_b.nodes] == ["cpu-0002"]

    # Each backend must resolve only its own (string) nodelist via scontrol.
    a_getaddrs = [
        c for c in backend_a._subprocess_launcher.calls if c[:2] == ["scontrol", "getaddrs"]
    ]
    b_getaddrs = [
        c for c in backend_b._subprocess_launcher.calls if c[:2] == ["scontrol", "getaddrs"]
    ]
    assert a_getaddrs == [["scontrol", "getaddrs", "cpu-0001"]]
    assert b_getaddrs == [["scontrol", "getaddrs", "cpu-0002"]]


def test_core_allocation_has_no_slurm_specific_fields():
    """The backend-agnostic Allocation must not carry Slurm-specific placement.

    ``het_group`` is a Slurm heterogeneous-job concept; it must live on the
    Slurm-owned allocation type, not on the core ``Allocation`` that
    docker / kubernetes / local backends also return.
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(Allocation)}
    assert "het_group" not in field_names


def test_slurm_backend_resource_env_preserves_controller_slurm_envs(monkeypatch):
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="test_backend",
            type="slurm",
            account="test_account",
            partition="batch",
            nodes=2,
            time="00:10:00",
            job_name="test_job",
            gpus_per_node=8,
        )
    )
    backend.allocation = Allocation(
        allocation_id="1111111",
        nodes=[],
    )

    monkeypatch.setenv("SLURM_JOB_ID", "2222222")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[001-002]")
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURMD_NODENAME", "node001")

    env = backend.resource_env(cuda_visible_devices="0,1")

    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    # INTENTIONAL CHANGE: NVIDIA_VISIBLE_DEVICES is no longer handed to an srun
    # step. It used to mirror the slice, but the container runtime reads it at
    # container CREATION and carves + RENUMBERS those devices from 0 -- leaving
    # the CUDA_VISIBLE_DEVICES we exported beside it, in host numbering,
    # addressing nothing. See SlurmBackend.resource_env.
    assert "NVIDIA_VISIBLE_DEVICES" not in env
    assert env["SLURM_JOB_ID"] == "2222222"
    assert env["SLURM_JOB_NODELIST"] == "node[001-002]"
    assert env["SLURM_NNODES"] == "2"
    assert env["SLURM_PROCID"] == "3"
    assert env["SLURMD_NODENAME"] == "node001"
    assert env["SFLOW_BACKEND_JOB_ID"] == "2222222"
    assert env["SFLOW_BACKEND_NODELIST"] == "node[001-002]"
    assert env["SFLOW_BACKEND_NUM_NODES"] == "2"


def test_srun_steps_are_not_handed_nvidia_visible_devices():
    """The variable that carves the container must not name a subset.

    pyxis/enroot reads NVIDIA_VISIBLE_DEVICES when it CREATES the container: a
    subset there makes it expose only those devices and renumber them from 0, so
    the CUDA_VISIBLE_DEVICES exported alongside -- in host numbering -- then
    addresses nothing. A worker planned for host 2,3 landed in a 2-GPU container
    numbered 0,1 and died with "No CUDA GPUs are available"; one planned for 0,1
    survived only by coincidence.

    Not exporting it lets the container see the node's GPUs with host numbering
    intact, so the planned slice is directly addressable and CUDA_VISIBLE_DEVICES
    -- the variable CUDA actually reads -- is the only thing sflow sets.

    Docker is unaffected: it overrides resource_env and isolates with
    `--gpus device=<uuid>`.
    """
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="s",
            type="slurm",
            account="acct",
            partition="batch",
            nodes=1,
            time="00:10:00",
            gpus_per_node=8,
        )
    )
    env = backend.resource_env(cuda_visible_devices="2,3")
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "NVIDIA_VISIBLE_DEVICES" not in env

    # A task with no GPU slice gets neither.
    assert "NVIDIA_VISIBLE_DEVICES" not in backend.resource_env(cuda_visible_devices=None)


def _discovery_backend(script):
    """A backend whose only subprocess call is the GPU topology probe."""
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="b", type="slurm", account="acct", partition="batch",
            nodes=1, time="00:10:00", job_name="job", gpus_per_node=4,
        )
    )
    backend._subprocess_launcher = _FakeSubprocessLauncher(script=script)
    return backend


def test_gpu_topology_probe_records_each_nodes_index_to_uuid_map():
    """The probe's output is the ground truth every later check rests on.

    A step can only ask "am I on the cards I was planned for?" by UUID, and this
    bare srun is where those UUIDs come from. Parsing it wrong does not fail
    loudly -- it leaves the maps empty and silently drops every task in the run to
    index arithmetic, so the parse needs a test of its own.
    """
    nodes = [ComputeNode(name="n0", ip_address="10.0.0.1", index=0),
             ComputeNode(name="n1", ip_address="10.0.0.2", index=1)]
    backend = _discovery_backend([(0, [
        "n0 0,GPU-aaa", "n0 1,GPU-bbb",
        "n1 0,GPU-ccc", "n1 1,GPU-ddd",
    ])])

    asyncio.run(backend._discover_gpu_uuids(nodes=nodes))

    # Ordered by HOST index, per node: the same planned slot is a different
    # physical card on n1 than on n0, which is why this is not one flat list.
    assert nodes[0].gpu_uuids == ["GPU-aaa", "GPU-bbb"]
    assert nodes[1].gpu_uuids == ["GPU-ccc", "GPU-ddd"]

    cmd = backend._subprocess_launcher.calls[0]["command"]
    # --overlap: shares the allocation instead of queueing behind a real step.
    assert "--overlap" in cmd and "--nodelist" in cmd
    assert cmd[cmd.index("--nodelist") + 1] == "n0,n1"


def test_gpu_topology_probe_rejects_a_partial_reading():
    """A gap means indices are missing, and a partial map resolves to the WRONG card.

    With 0 and 2 reported, index 1 is unaccounted for; treating the two as a
    0,1 run would map planned slot 1 onto the physical card at slot 2. Better to
    say nothing and let the step fall back.
    """
    nodes = [ComputeNode(name="n0", ip_address="10.0.0.1", index=0)]
    backend = _discovery_backend([(0, ["n0 0,GPU-aaa", "n0 2,GPU-ccc"])])

    asyncio.run(backend._discover_gpu_uuids(nodes=nodes))

    assert nodes[0].gpu_uuids is None


def test_gpu_topology_probe_ignores_noise_and_survives_failure():
    """Never the reason a workflow cannot start."""
    nodes = [ComputeNode(name="n0", ip_address="10.0.0.1", index=0)]
    # Slurm prologue chatter, a malformed pair, and a non-UUID value.
    backend = _discovery_backend([(0, [
        "srun: job 42 queued and waiting for resources",
        "n0 notanindex,GPU-zzz",
        "n0 0,NOT-A-UUID",
        "n0 0,GPU-aaa",
    ])])
    asyncio.run(backend._discover_gpu_uuids(nodes=nodes))
    assert nodes[0].gpu_uuids == ["GPU-aaa"]

    # A failed probe leaves the map untouched rather than raising.
    other = [ComputeNode(name="n0", ip_address="10.0.0.1", index=0)]
    asyncio.run(_discovery_backend([(1, ["srun: error"])])._discover_gpu_uuids(nodes=other))
    assert other[0].gpu_uuids is None


def test_resource_env_hands_the_step_its_planned_uuids():
    """The seam the whole feature hangs on: driver map -> step env.

    _planned_gpu_uuids and gpu_placement.sh were each covered in isolation, but
    nothing asserted resource_env actually JOINS them. Drop this line and every
    other test still passes while every step silently falls back to index
    arithmetic -- the failure mode is invisible.
    """
    backend = SlurmBackend(
        SlurmBackendConfig(
            name="b", type="slurm", account="acct", partition="batch",
            nodes=1, time="00:10:00", job_name="job", gpus_per_node=4,
        )
    )
    backend.allocation = Allocation(
        allocation_id="1",
        nodes=[ComputeNode(name="n0", ip_address="10.0.0.1", index=0,
                           gpu_uuids=["GPU-a", "GPU-b", "GPU-c", "GPU-d"])],
        owned=False,
    )
    env = backend.resource_env(cuda_visible_devices="2,3")
    assert env["SFLOW_PLANNED_GPU_UUIDS"] == "n0=GPU-c,GPU-d"

    # Nothing trustworthy to say -> say nothing. Absent means "fall back", and an
    # empty string would read as "this task was planned for no GPUs".
    backend.allocation.nodes[0].gpu_uuids = None
    assert "SFLOW_PLANNED_GPU_UUIDS" not in backend.resource_env(cuda_visible_devices="2,3")


def test_planned_uuids_warns_when_a_slot_is_past_the_nodes_device_count():
    """The warning IS the deliverable here.

    Returning "" already drops the task to index arithmetic; the log line is the
    only thing that tells the user their gpus_per_node is bigger than the node.
    Silence would leave them with weaker checking and no idea why.
    """
    alloc = Allocation(
        allocation_id="1",
        nodes=[ComputeNode(name="n0", ip_address="10.0.0.1", index=0,
                           gpu_uuids=["GPU-a", "GPU-b"])],
        owned=False,
    )
    with unittest.mock.patch.object(slurm_mod._logger, "warning") as warn:
        assert slurm_mod._planned_gpu_uuids("0,9", alloc) == ""
    assert warn.called
    assert "gpus_per_node" in warn.call_args.args[0]


def test_allocation_probes_the_gpu_topology():
    """Both allocate() paths must actually call the probe.

    It is best-effort and never raises, so a missing call site is silent: the
    maps stay None and every task drops to index arithmetic while the suite
    stays green.
    """
    import inspect

    # Once for the pre-existing (unowned) allocation, once for the salloc path.
    assert inspect.getsource(SlurmBackend).count("await self._discover_gpu_uuids(") == 2
