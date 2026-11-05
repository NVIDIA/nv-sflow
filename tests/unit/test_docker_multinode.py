# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One sflow task spanning several Docker hosts.

Two seams, deliberately:

  * **plan -> commands** -- a pure function of config plus assignment, so the
    whole fan-out can be asserted as exact argv with no daemon anywhere;
  * **managed execution** -- the orchestration semantics (fail fast, stop the
    peers, label the output) driven against a fake launcher.
"""

from __future__ import annotations

import asyncio

import pytest

from sflow.plugins.backends.docker import DockerBackend, DockerBackendConfig


def _backend(*hosts: dict, **kwargs) -> DockerBackend:
    backend = DockerBackend(
        DockerBackendConfig(
            name="cluster",
            type="docker",
            image="ubuntu:22.04",
            hosts=list(hosts),
            **kwargs,
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    return backend


def _two_host_operator(nodes=("node-a", "node-b")):
    backend = _backend(
        {"name": "node-a", "address": "10.0.0.11"},
        {"name": "node-b", "address": "10.0.0.12"},
    )
    operator = backend.default_operator(name="d")
    operator.apply_backend_context(
        backend=backend, assigned_nodes=list(nodes), artifacts=[]
    )
    return operator


def _argv(operator, **kwargs) -> list[list[str]]:
    commands = operator._multi_node_commands(
        task_name=kwargs.pop("task_name", "job"),
        script=kwargs.pop("script", ["echo hi"]),
        envs=kwargs.pop("envs", {}),
    )
    return [command.as_list() for _member, command in commands]


# ---------------------------------------------------------------------------
# Seam A -- plan to commands
# ---------------------------------------------------------------------------


class TestFanOut:
    def test_one_container_per_assigned_host(self):
        argv = _argv(_two_host_operator())

        assert len(argv) == 2
        assert argv[0][:3] == ["docker", "--host", "ssh://10.0.0.11"]
        assert argv[1][:3] == ["docker", "--host", "ssh://10.0.0.12"]

    def test_a_multi_node_task_manages_its_own_execution(self):
        """The orchestrator launches one subprocess per task by default; a task
        spanning hosts has to supervise several, so it opts out."""
        assert _two_host_operator().manages_own_execution() is True

    def test_a_single_node_task_does_not(self):
        assert _two_host_operator(nodes=("node-a",)).manages_own_execution() is False

    def test_each_member_gets_its_rank_by_value(self):
        """The rank differs per member, so it cannot ride in on the shared task
        env; it is the ONE variable inlined. World size and leader address are
        the same for every member and come from assembly like any other env."""
        argv = _argv(_two_host_operator())

        assert "SFLOW_TASK_NODE_INDEX=0" in argv[0]
        assert "SFLOW_TASK_NODE_INDEX=1" in argv[1]
        assert not any("SFLOW_TASK_NUM_NODES=" in a for args in argv for a in args)
        assert not any("SFLOW_LEADER_ADDRESS=" in a for args in argv for a in args)

    def test_build_command_is_not_the_execution_path_for_multi_node(self):
        """`build_command` returns rank 0 ALONE, so it must never be launched.

        The orchestrator chooses execute() over the single-command path purely on
        `manages_own_execution()`. If that ever regressed, this one command would
        start 1 container instead of N and the task would report success on rank
        0's exit code -- silently running a fraction of the job with no error.
        """
        operator = _two_host_operator()
        members = _argv(operator)
        representative = operator.build_command(
            task_name="job", script=["echo hi"], envs={}
        ).as_list()

        assert operator.manages_own_execution() is True, "execute() must own this"
        assert len(members) == 2
        assert representative == members[0], "representative is rank 0, not a fan-out"

    def test_container_names_are_unique_per_member(self):
        argv = _argv(_two_host_operator())
        names = [args[args.index("--name") + 1] for args in argv]

        assert len(set(names)) == 2


class TestRendezvousIsMandatory:
    def test_a_multi_node_task_errors_when_a_host_has_no_rendezvous_address(self):
        """The alternative is a silent hang.

        Without every address the leader cannot be named, and the workload sits
        in rendezvous until someone kills it -- so this has to fail at planning,
        naming the host, rather than launch and wait.
        """
        backend = _backend(
            {"name": "node-a", "address": "10.0.0.11"},
            {"name": "node-b", "context": "lab-ctx"},
        )
        operator = backend.default_operator(name="d")

        with pytest.raises(ValueError, match="node-b"):
            operator.apply_backend_context(
                backend=backend, assigned_nodes=["node-a", "node-b"], artifacts=[]
            )

    def test_a_single_node_task_on_such_a_host_is_still_fine(self):
        """One container has nobody to rendezvous with, so it needs no address."""
        backend = _backend({"name": "node-b", "context": "lab-ctx"})
        operator = backend.default_operator(name="d")

        operator.apply_backend_context(
            backend=backend, assigned_nodes=["node-b"], artifacts=[]
        )

        parts = operator.build_command(
            task_name="t", script=["echo hi"], envs={}
        ).as_list()
        assert parts[:3] == ["docker", "--context", "lab-ctx"]


# ---------------------------------------------------------------------------
# Seam B -- managed execution against a fake launcher
# ---------------------------------------------------------------------------


class _FakeLauncher:
    """Returns a scripted exit code per member, recording what it was given."""

    def __init__(self, codes: dict[str, int], hang: set[str] | None = None):
        self._codes = codes
        self._hang = hang or set()
        self.started: list[str] = []
        self.cancelled: list[str] = []

    async def run_async(self, command, *, output_logger, env, task_name):
        member = task_name.split(":", 1)[-1]
        self.started.append(member)
        output_logger.info("hello from the workload")
        try:
            if member in self._hang:
                await asyncio.sleep(3600)
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled.append(member)
            raise
        return self._codes.get(member, 0)


class _RecordingLogger:
    def __init__(self):
        self.lines: list[str] = []
        self.handlers: list = []

    def info(self, message, *args, **kwargs):
        self.lines.append(str(message))


def _run(operator, launcher, logger=None):
    logger = logger or _RecordingLogger()
    rc = asyncio.run(
        operator.execute(
            launcher=launcher,
            output_logger=logger,
            env={},
            task_name="job",
            script=["echo hi"],
        )
    )
    return rc, logger


class TestManagedExecution:
    def test_all_members_run_and_success_is_zero(self):
        launcher = _FakeLauncher({})

        rc, _ = _run(_two_host_operator(), launcher)

        assert rc == 0
        assert sorted(launcher.started) == ["node-a", "node-b"]

    def test_a_failing_member_fails_the_task_with_its_exit_code(self):
        launcher = _FakeLauncher({"node-b": 7})

        rc, _ = _run(_two_host_operator(), launcher)

        assert rc == 7, "the failing member's code, not a generic 1"

    def test_a_failing_member_stops_its_peers(self):
        """Otherwise a dead job keeps burning GPUs on every other machine."""
        launcher = _FakeLauncher({"node-b": 7}, hang={"node-a"})

        rc, _ = _run(_two_host_operator(), launcher)

        assert rc == 7
        assert launcher.cancelled == ["node-a"]

    def test_member_output_is_labelled_with_node_and_rank(self):
        _, logger = _run(_two_host_operator(), _FakeLauncher({}))

        assert any(line.startswith("[node-a rank=0] ") for line in logger.lines)
        assert any(line.startswith("[node-b rank=1] ") for line in logger.lines)

    def test_the_failure_is_reported_on_the_task_log(self):
        _, logger = _run(_two_host_operator(), _FakeLauncher({"node-b": 7}))

        assert any("node-b" in line and "7" in line for line in logger.lines)


# ---------------------------------------------------------------------------
# GPU device slices
# ---------------------------------------------------------------------------


def _gpu_arg(args: list[str]) -> str | None:
    return args[args.index("--gpus") + 1] if "--gpus" in args else None


def _capture_docker_run_warnings():
    """sflow loggers set propagate=False, so caplog cannot see them."""
    import logging

    import sflow.plugins.operators.docker_run as docker_run_mod

    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    handler = _Capture()
    docker_run_mod._logger.addHandler(handler)
    return handler, messages


def _remove_docker_run_handler(handler):
    import sflow.plugins.operators.docker_run as docker_run_mod

    docker_run_mod._logger.removeHandler(handler)


class TestPerMemberGpuSlices:
    """Which physical devices each member is pinned to.

    The rule is about the *daemon*, not the rank: two machines each have their
    own GPU 0, so members on different hosts may hold the same device numbers.
    Members sharing one daemon may not -- that is two containers pinned to the
    same physical GPUs, contending silently with no error anywhere.
    """

    def test_members_on_different_hosts_may_share_device_numbers(self):
        backend = _backend(
            {"name": "node-a", "address": "10.0.0.11", "gpus_per_node": 8},
            {"name": "node-b", "address": "10.0.0.12", "gpus_per_node": 8},
        )
        operator = backend.default_operator(name="d")
        operator.apply_backend_context(
            backend=backend,
            assigned_nodes=["node-a", "node-b"],
            artifacts=[],
            cuda_visible_devices="0,1",
            gpu_count=2,
        )

        argv = _argv(operator)

        # Each host has its own device 0 and 1 -- the same slice is correct here.
        assert [_gpu_arg(args) for args in argv] == ['"device=0,1"', '"device=0,1"']

    def test_synthetic_local_nodes_with_gpus_never_get_this_far(self):
        """The unsound configuration is refused where the allocation is built.

        It used to reach the operator, which could only warn about it. Rejecting
        it at construction keeps the resource planner free of any docker-shaped
        exception -- see tests/unit/test_docker_allocation_honesty.py.
        """
        with pytest.raises(ValueError, match="hosts"):
            DockerBackend(
                DockerBackendConfig(
                    name="local",
                    type="docker",
                    image="ubuntu:22.04",
                    nodes=2,
                    gpus_per_node=4,
                )
            )

    def test_a_single_node_task_is_unchanged(self):
        backend = _backend({"name": "node-a", "address": "10.0.0.11"})
        operator = backend.default_operator(name="d")
        operator.apply_backend_context(
            backend=backend,
            assigned_nodes=["node-a"],
            artifacts=[],
            cuda_visible_devices="2,3",
            gpu_count=2,
        )

        parts = operator.build_command(
            task_name="t", script=["echo hi"], envs={}
        ).as_list()

        assert _gpu_arg(parts) == '"device=2,3"'

    def test_non_numeric_devices_are_not_offset(self):
        """UUID pins have no arithmetic -- offsetting them would invent a device."""
        backend = _backend(
            {"name": "node-a", "address": "10.0.0.11"},
            {"name": "node-b", "address": "10.0.0.12"},
        )
        operator = backend.default_operator(name="d")
        operator.apply_backend_context(
            backend=backend,
            assigned_nodes=["node-a", "node-b"],
            artifacts=[],
            cuda_visible_devices="GPU-aaa,GPU-bbb",
            gpu_count=2,
        )

        argv = _argv(operator)

        assert all(_gpu_arg(args) == '"device=GPU-aaa,GPU-bbb"' for args in argv)


# ---------------------------------------------------------------------------
# Reclaiming containers a crashed driver left behind
# ---------------------------------------------------------------------------


class TestOrphanReclamationAcrossHosts:
    """A SIGKILLed driver never runs teardown, so the next run has to clean up.

    Teardown cannot cover this: it runs in the orchestrator's `finally`, which a
    SIGKILL bypasses entirely. Without a sweep that reaches remote daemons, those
    containers have no route back at all and hold their GPUs indefinitely.
    """

    def test_every_distinct_daemon_is_swept(self):
        scripts = [
            cmd.as_list()[2]
            for cmd in _two_host_operator().stale_reap_commands(task_name="job")
        ]

        assert len(scripts) == 2
        assert any("docker --host ssh://10.0.0.11 ps -a" in s for s in scripts)
        assert any("docker --host ssh://10.0.0.12 ps -a" in s for s in scripts)

    def test_two_nodes_on_one_daemon_are_swept_once(self):
        backend = _backend(
            {"name": "a", "docker_host": "ssh://same", "address": "10.0.0.11"},
            {"name": "b", "docker_host": "ssh://same", "address": "10.0.0.12"},
        )
        operator = backend.default_operator(name="d")
        operator.apply_backend_context(
            backend=backend, assigned_nodes=["a", "b"], artifacts=[]
        )

        assert len(operator.stale_reap_commands(task_name="job")) == 1

    def test_the_sweep_only_considers_this_machines_containers(self):
        """The safety property that makes remote sweeping sound at all.

        The liveness check reads the *driver's* process table, so a container
        created by a different machine must be excluded by name before its PID is
        ever consulted -- otherwise a co-tenant's live run looks dead.
        """
        from sflow.plugins.operators.docker_run import _OWNER_PREFIX

        scripts = [
            cmd.as_list()[2]
            for cmd in _two_host_operator().stale_reap_commands(task_name="job")
        ]

        assert all(f"--filter name={_OWNER_PREFIX}" in s for s in scripts)
        assert _OWNER_PREFIX.startswith("sflow-")
        assert _OWNER_PREFIX.endswith("p"), "the owning PID follows the prefix"

    def test_container_names_start_with_the_owner_prefix(self):
        from sflow.plugins.operators.docker_run import _OWNER_PREFIX

        argv = _argv(_two_host_operator())
        names = [args[args.index("--name") + 1] for args in argv]

        assert all(name.startswith(_OWNER_PREFIX) for name in names)


class TestDryRunFanOutPreview:
    """Issue #1 US29/US30: the whole fan-out must be reviewable BEFORE launching.

    `build_command` returns rank 0 alone, and `docker_cmds.log` is only written
    once commands are actually issued -- so a dry run used to show one machine's
    worth of a job spanning several, and never which daemon any member targeted.
    """

    def test_one_preview_entry_per_member_naming_its_daemon(self):
        operator = _two_host_operator()

        entries = operator.dry_run_launch_preview(
            task_name="job", script=["echo hi"], envs={}
        )

        assert len(entries) == 2
        labels = [label for label, _cmd in entries]
        assert "node node-a (rank 0) -> ssh://10.0.0.11" in labels[0]
        assert "node node-b (rank 1) -> ssh://10.0.0.12" in labels[1]

    def test_preview_commands_match_what_would_really_run(self):
        """A preview that drifts from the launch path is worse than none."""
        operator = _two_host_operator()

        previewed = [
            cmd.as_list()
            for _label, cmd in operator.dry_run_launch_preview(
                task_name="job", script=["echo hi"], envs={}
            )
        ]

        assert previewed == _argv(operator)

    def test_a_single_node_task_adds_nothing(self):
        """The plan's one-task-one-command shape already says everything."""
        backend = _backend({"name": "solo", "address": "10.0.0.11"})
        operator = backend.default_operator(name="d")
        operator.apply_backend_context(
            backend=backend, assigned_nodes=["solo"], artifacts=[]
        )

        assert (
            operator.dry_run_launch_preview(
                task_name="job", script=["echo hi"], envs={}
            )
            == []
        )

    def test_preview_records_nothing_to_the_command_log(self):
        """A dry run must not fabricate a record of commands nothing ran."""
        from sflow.core import command_log

        recorded: list = []

        class _Router:
            def record(self, *args, **kwargs):
                recorded.append((args, kwargs))

        token = command_log.set_active_command_log_router(_Router())
        try:
            _two_host_operator().dry_run_launch_preview(
                task_name="job", script=["echo hi"], envs={}
            )
        finally:
            command_log.reset_active_command_log_router(token)

        assert recorded == []
