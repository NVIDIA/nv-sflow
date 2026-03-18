# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import pytest

import sflow.plugins.backends.slurm as slurm_mod
from sflow.core.backend import Allocation
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
        ]
    )
    backend._subprocess_launcher = fake_launcher

    with pytest.raises(RuntimeError, match=r"Failed to resolve node addresses via srun"):
        asyncio.run(backend.allocate())


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
        ]
    )
    backend._subprocess_launcher = fake_launcher

    with pytest.raises(RuntimeError, match=r"Failed to parse any node addresses"):
        asyncio.run(backend.allocate())


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
