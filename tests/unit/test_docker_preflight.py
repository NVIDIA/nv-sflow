# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight for a declared Docker host pool.

Everything checked here fails the workflow anyway if it is wrong -- an
unreachable daemon, a mount source that does not exist on the remote machine.
The only question is whether the user finds out in a second, before anything
launches, or several minutes in with half the containers already started and an
error that names a container rather than the host that was never reachable.

Kubernetes already front-loads this. Docker did not: its whole pre-flight was a
`which docker` on the *driver*, which says nothing about the machines the work
actually runs on.
"""

import shlex
import shutil

import pytest

from sflow.plugins.backends.docker import (
    _DAEMON_PROBE,
    _SSH_BATCH_OPTS,
    DockerBackend,
    DockerBackendConfig,
)


@pytest.fixture(autouse=True)
def _docker_binary_present(monkeypatch):
    """Pretend the `docker` client is installed.

    Pre-flight's first act is `shutil.which("docker")`, and that is a real
    filesystem lookup -- pytest-subprocess fakes `subprocess.run`, not PATH. So
    every test below passed on a developer machine with Docker installed and all
    22 failed on a CI runner without it, none of them for the reason they exist.

    Only "docker" is intercepted; everything else still resolves for real, so
    this cannot mask a different missing binary.
    """
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda cmd, *a, **kw: "/usr/bin/docker"
        if cmd == "docker"
        else real_which(cmd, *a, **kw),
    )


def _backend(**kwargs) -> DockerBackend:
    return DockerBackend(
        DockerBackendConfig(
            name="cluster", type="docker", image="ubuntu:22.04", **kwargs
        )
    )


def _two_hosts(**extra):
    return [
        {"name": "dgx-a", "address": "10.0.0.11", **extra},
        {"name": "dgx-b", "address": "10.0.0.12", **extra},
    ]


# The daemon probe for an address host: `ssh <opts> <address> docker version`.
# ssh is run by pre-flight itself, not via `docker -H ssh://`, because the
# docker client offers no way to put BatchMode on the ssh it spawns.
_DAEMON_REMOTE_CMD = shlex.join(("docker", *_DAEMON_PROBE))


def _daemon_ok(fake_process, address):
    fake_process.register(
        ["ssh", *_SSH_BATCH_OPTS, address, _DAEMON_REMOTE_CMD],
        stdout="27.0.3\n",
    )


def _daemon_unreachable(fake_process, address):
    fake_process.register(
        ["ssh", *_SSH_BATCH_OPTS, address, _DAEMON_REMOTE_CMD],
        stderr="ssh: connect to host timed out\n",
        returncode=1,
    )


def _paths_ok(fake_process, occurrences=2):
    fake_process.register(["ssh", fake_process.any()], stdout="", occurrences=occurrences)


def _ssh_to(fake_process, address):
    """The mount probe's argv for one host.

    Built from the real option tuple so these stubs keep matching if the options
    change; what those options must BE is pinned separately and literally in
    TestNonInteractive, which is the assertion that would catch a regression.
    """
    return ["ssh", *_SSH_BATCH_OPTS, address, fake_process.any()]


def _mount_probe_calls(fake_process):
    """The ssh calls that checked mount sources (not the daemon probes).

    Both probes are ssh now, so "was a path checked" has to look at the remote
    command: the mount probe is a chain of `[ -e <path> ]` tests.
    """
    return [
        call
        for call in fake_process.calls
        if call and str(call[0]) == "ssh" and "[ -e " in str(call[-1])
    ]


class TestDaemonReachability:
    def test_all_hosts_reachable_passes(self, fake_process):
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)

        _backend(hosts=_two_hosts()).preflight_validate()

    def test_an_unreachable_host_fails_before_anything_launches(self, fake_process):
        _daemon_ok(fake_process, "10.0.0.11")
        _daemon_unreachable(fake_process, "10.0.0.12")

        with pytest.raises(ValueError) as excinfo:
            _backend(hosts=_two_hosts()).preflight_validate()

        message = str(excinfo.value)
        assert "dgx-b" in message, "the failing host must be named"
        assert "timed out" in message, "and why it failed"

    def test_every_unreachable_host_is_reported_at_once(self, fake_process):
        """One round trip per fix is a bad loop when a whole pool is misconfigured."""
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_unreachable(fake_process, address)

        with pytest.raises(ValueError) as excinfo:
            _backend(hosts=_two_hosts()).preflight_validate()

        message = str(excinfo.value)
        assert "dgx-a" in message and "dgx-b" in message

    def test_a_reachable_host_is_not_re_probed_per_mount(self, fake_process):
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)
        _paths_ok(fake_process)

        _backend(hosts=_two_hosts(), mounts=["/data:/data:ro"]).preflight_validate()

        daemon_probes = [
            call for call in fake_process.calls if str(call[-1]) == _DAEMON_REMOTE_CMD
        ]
        assert len(daemon_probes) == 2, "one version probe per host, not per mount"


class TestRemoteMountPaths:
    def test_a_missing_mount_source_fails_with_the_host_and_path(self, fake_process):
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)
        # The probe prints back whichever paths are absent.
        fake_process.register(
            _ssh_to(fake_process, "10.0.0.11"), stdout="/data\n"
        )
        fake_process.register(_ssh_to(fake_process, "10.0.0.12"), stdout="")

        with pytest.raises(ValueError) as excinfo:
            _backend(
                hosts=_two_hosts(), mounts=["/data:/data:ro"]
            ).preflight_validate()

        message = str(excinfo.value)
        assert "dgx-a" in message and "/data" in message
        assert "dgx-b" not in message, "the host that has the path is not blamed"

    def test_per_host_mounts_are_checked_on_that_host(self, fake_process):
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)
        fake_process.register(_ssh_to(fake_process, "10.0.0.11"), stdout="")
        fake_process.register(
            _ssh_to(fake_process, "10.0.0.12"), stdout="/scratch\n"
        )

        hosts = _two_hosts()
        hosts[1]["mounts"] = ["/scratch:/scratch:rw"]

        with pytest.raises(ValueError, match="/scratch"):
            _backend(hosts=hosts).preflight_validate()

    def test_a_missing_artifact_path_is_reported_but_allowed(self, fake_process):
        """An artifact may be PRODUCED by the workflow, unlike a declared mount.

        A model directory is created by a download task on the first run, so
        failing here would make the recipe unrunnable until someone fetched the
        weights by hand. Declared `mounts:` keep failing -- those assert data
        that already exists.
        """

        class _Artifact:
            uri = "fs:///models/qwen"
            path = "/models/qwen"

        backend = _backend(hosts=_two_hosts())
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)
        fake_process.register(_ssh_to(fake_process, "10.0.0.11"), stdout="")
        fake_process.register(
            _ssh_to(fake_process, "10.0.0.12"), stdout="/models/qwen\n"
        )

        handler, messages, docker_mod = _capture_docker_logs()
        try:
            backend.preflight_validate(artifacts=[_Artifact()])  # must not raise
        finally:
            docker_mod._logger.removeHandler(handler)

        assert any("/models/qwen" in m and "not present yet" in m for m in messages)

    def test_a_declared_mount_still_fails_even_alongside_artifacts(self, fake_process):
        class _Artifact:
            uri = "fs:///models/qwen"
            path = "/models/qwen"

        backend = _backend(hosts=_two_hosts(), mounts=["/data:/data:ro"])
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)
        fake_process.register(_ssh_to(fake_process, "10.0.0.11"), stdout="")
        # Both absent on this host; only the declared one is fatal.
        fake_process.register(
            _ssh_to(fake_process, "10.0.0.12"), stdout="/data\n/models/qwen\n"
        )

        with pytest.raises(ValueError) as excinfo:
            backend.preflight_validate(artifacts=[_Artifact()])

        message = str(excinfo.value)
        assert "/data" in message
        assert "/models/qwen" not in message, "an artifact is not a hard failure"

    def test_no_mounts_means_no_path_probe(self, fake_process):
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)

        _backend(hosts=_two_hosts()).preflight_validate()

        assert not _mount_probe_calls(fake_process)

    def test_a_context_host_skips_the_path_probe(self, fake_process):
        """A context names a daemon, not a machine sflow knows how to shell into."""
        fake_process.register(
            ["docker", "--context", "lab", "version", "--format", "{{.Server.Version}}"],
            stdout="27.0.3\n",
        )

        _backend(
            hosts=[{"name": "ctx", "context": "lab"}], mounts=["/data:/data:ro"]
        ).preflight_validate()

        assert not _mount_probe_calls(fake_process)


def _capture_docker_logs():
    """sflow loggers set propagate=False, so caplog cannot see them."""
    import logging

    import sflow.plugins.backends.docker as docker_mod

    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Capture()
    handler.setLevel(logging.DEBUG)
    docker_mod._logger.addHandler(handler)
    # The module logger may be configured above INFO; these are INFO lines.
    docker_mod._logger.setLevel(logging.DEBUG)
    return handler, messages, docker_mod


class TestProgressIsVisible:
    """A silent driver is indistinguishable from a hung one.

    Each probe can sit for the full timeout on an unreachable host, so the
    announcement has to come *before* the call, not after it.
    """

    def test_each_host_is_announced_before_it_is_contacted(self, fake_process):
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)

        handler, messages, docker_mod = _capture_docker_logs()
        try:
            _backend(hosts=_two_hosts()).preflight_validate()
        finally:
            docker_mod._logger.removeHandler(handler)

        assert any("validating 2 declared Docker host(s)" in m for m in messages)
        assert any("dgx-a" in m and "contacting Docker daemon" in m for m in messages)
        assert any("dgx-b" in m and "contacting Docker daemon" in m for m in messages)
        # The endpoint is named, so a wrong address is obvious from the log.
        assert any("ssh://10.0.0.11" in m for m in messages)

    def test_the_mount_check_says_what_it_is_checking(self, fake_process):
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)
        _paths_ok(fake_process)

        handler, messages, docker_mod = _capture_docker_logs()
        try:
            _backend(hosts=_two_hosts(), mounts=["/data:/data:ro"]).preflight_validate()
        finally:
            docker_mod._logger.removeHandler(handler)

        assert any("mount source(s) over ssh" in m and "/data" in m for m in messages)

    def test_a_clean_pool_reports_completion(self, fake_process):
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)

        handler, messages, docker_mod = _capture_docker_logs()
        try:
            _backend(hosts=_two_hosts()).preflight_validate()
        finally:
            docker_mod._logger.removeHandler(handler)

        assert any("all 2 Docker host(s) reachable" in m for m in messages)


class TestScope:
    def test_a_local_backend_touches_no_network(self, fake_process):
        """No declared hosts -> nothing remote to check."""
        _backend(nodes=1).preflight_validate()

        assert list(fake_process.calls) == []

    def test_the_skip_switch_bypasses_remote_checks(self, fake_process, monkeypatch):
        monkeypatch.setenv("SFLOW_SKIP_DOCKER_PREFLIGHT", "1")

        _backend(hosts=_two_hosts(), mounts=["/data:/data:ro"]).preflight_validate()

        assert list(fake_process.calls) == []


class TestHostsAreCheckedInParallel:
    """Pre-flight cost must scale with the slowest host, not the host count.

    Every probe is a bounded-but-slow network call (a cold ssh handshake, or the
    full timeout on a host that never answers). Run one at a time, an 8-host
    pool spent minutes before anything launched.
    """

    @staticmethod
    def _concurrency_tracking_run(monkeypatch, docker_mod, hold_s=0.15):
        """Replace subprocess.run with one that records peak overlap."""
        import subprocess as real_subprocess
        import threading
        import time

        state = {"now": 0, "peak": 0}
        lock = threading.Lock()

        def _fake_run(args, **kwargs):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(hold_s)
            with lock:
                state["now"] -= 1
            return real_subprocess.CompletedProcess(args, 0, stdout="27.0.3\n", stderr="")

        monkeypatch.setattr(docker_mod.subprocess, "run", _fake_run)
        return state

    def test_daemon_probes_overlap(self, monkeypatch):
        import sflow.plugins.backends.docker as docker_mod

        hosts = [
            {"name": f"dgx-{i}", "address": f"10.0.0.{i}"} for i in range(1, 6)
        ]
        state = self._concurrency_tracking_run(monkeypatch, docker_mod)

        _backend(hosts=hosts).preflight_validate()

        assert state["peak"] == 5, (
            f"expected all 5 daemon probes in flight at once, peak was {state['peak']}"
        )

    def test_five_slow_hosts_take_about_one_probe(self, monkeypatch):
        """The wall-clock claim, stated directly."""
        import time

        import sflow.plugins.backends.docker as docker_mod

        hosts = [
            {"name": f"dgx-{i}", "address": f"10.0.0.{i}"} for i in range(1, 6)
        ]
        self._concurrency_tracking_run(monkeypatch, docker_mod, hold_s=0.2)

        started = time.monotonic()
        _backend(hosts=hosts).preflight_validate()
        elapsed = time.monotonic() - started

        # Sequential would be 5 x 0.2s = 1.0s.
        assert elapsed < 0.6, f"pre-flight took {elapsed:.2f}s; probes are serialized"

    def test_problems_are_reported_in_host_order(self, monkeypatch):
        """Completion order is nondeterministic once parallel; the message the
        user reads must not be."""
        import subprocess as real_subprocess
        import time

        import sflow.plugins.backends.docker as docker_mod

        # Later hosts answer FIRST, so completion order is the reverse of config.
        delays = {"10.0.0.1": 0.30, "10.0.0.2": 0.20, "10.0.0.3": 0.05}

        def _fake_run(args, **kwargs):
            address = next(
                (a for a in delays if any(a in str(part) for part in args)), None
            )
            time.sleep(delays.get(address, 0))
            return real_subprocess.CompletedProcess(
                args, 1, stdout="", stderr="nope\n"
            )

        monkeypatch.setattr(docker_mod.subprocess, "run", _fake_run)
        hosts = [
            {"name": "dgx-1", "address": "10.0.0.1"},
            {"name": "dgx-2", "address": "10.0.0.2"},
            {"name": "dgx-3", "address": "10.0.0.3"},
        ]

        with pytest.raises(ValueError) as excinfo:
            _backend(hosts=hosts).preflight_validate()

        message = str(excinfo.value)
        assert message.index("dgx-1") < message.index("dgx-2") < message.index("dgx-3")


class TestUnreachableHostsAreNotProbedTwice:
    def test_a_dead_host_gets_no_mount_check(self, fake_process):
        """The mount check is an ssh to the same machine the daemon probe just
        failed to reach. Running it anyway bought nothing and cost a second full
        timeout per dead host -- doubling the wait for the worst case."""
        _daemon_unreachable(fake_process, "10.0.0.11")
        _daemon_ok(fake_process, "10.0.0.12")
        _paths_ok(fake_process, occurrences=1)

        with pytest.raises(ValueError):
            _backend(hosts=_two_hosts(), mounts=["/data:/data:ro"]).preflight_validate()

        ssh_calls = _mount_probe_calls(fake_process)
        # Exactly one: the reachable host. The dead one is not tried again.
        assert len(ssh_calls) == 1, ssh_calls
        assert "10.0.0.12" in " ".join(str(part) for part in ssh_calls[0])

    def test_a_healthy_pool_still_checks_every_host(self, fake_process):
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)
        _paths_ok(fake_process, occurrences=2)

        _backend(hosts=_two_hosts(), mounts=["/data:/data:ro"]).preflight_validate()

        ssh_calls = _mount_probe_calls(fake_process)
        assert len(ssh_calls) == 2, ssh_calls


class TestNonInteractive:
    """Pre-flight must fail, never prompt.

    Up to _PREFLIGHT_MAX_PARALLEL probes run at once and each inherits the
    driver's terminal, so an unknown host key or a passphrase-protected key
    would put sixteen ssh processes on one TTY at the same instant. The user
    sees interleaved questions nobody can answer, then sixteen timeouts -- the
    opposite of the fast, attributable failure pre-flight exists to give.
    """

    def test_mount_probe_is_batch_mode_and_cannot_read_the_terminal(
        self, fake_process, monkeypatch
    ):
        import sflow.plugins.backends.docker as docker_mod

        seen = {}
        real_run = docker_mod.subprocess.run

        def _record(args, **kwargs):
            if args and str(args[0]) == "ssh":
                seen["argv"] = [str(a) for a in args]
                seen["stdin"] = kwargs.get("stdin")
            return real_run(args, **kwargs)

        monkeypatch.setattr(docker_mod.subprocess, "run", _record)
        for address in ("10.0.0.11", "10.0.0.12"):
            _daemon_ok(fake_process, address)
        _paths_ok(fake_process, occurrences=2)

        _backend(hosts=_two_hosts(), mounts=["/data:/data:ro"]).preflight_validate()

        argv = seen["argv"]
        assert "BatchMode=yes" in argv, argv
        assert f"ConnectTimeout={docker_mod._PREFLIGHT_TIMEOUT_S}" in argv, argv
        # capture_output alone leaves fd 0 attached to the driver's terminal.
        assert seen["stdin"] is docker_mod.subprocess.DEVNULL

    def test_daemon_probe_owns_its_ssh_argv_and_is_batch_mode(
        self, fake_process, monkeypatch
    ):
        """The probe runs `ssh ... docker version` ITSELF.

        `docker -H ssh://` spawns ssh with no way to pass -o (the client reads
        no DOCKER_SSH_OPTS or similar), so routing the probe through it could
        still prompt. Owning the ssh argv is the only way to guarantee BatchMode.
        """
        import sflow.plugins.backends.docker as docker_mod

        seen = {}
        real_run = docker_mod.subprocess.run

        def _record(args, **kwargs):
            argv = [str(a) for a in args]
            if argv and argv[0] == "ssh" and argv[-1] == _DAEMON_REMOTE_CMD:
                seen["argv"] = argv
                seen["stdin"] = kwargs.get("stdin")
            return real_run(args, **kwargs)

        monkeypatch.setattr(docker_mod.subprocess, "run", _record)
        _daemon_ok(fake_process, "10.0.0.11")

        _backend(hosts=[{"name": "dgx-a", "address": "10.0.0.11"}]).preflight_validate()

        argv = seen["argv"]
        assert "BatchMode=yes" in argv, argv
        assert f"ConnectTimeout={docker_mod._PREFLIGHT_TIMEOUT_S}" in argv, argv
        assert "10.0.0.11" in argv
        assert not [a for a in argv if a.startswith("ssh://")], "no docker tunnel"
        assert seen["stdin"] is docker_mod.subprocess.DEVNULL

    def test_a_user_decorated_target_keeps_the_login(self, fake_process):
        """`user:` must reach the probe's ssh target, as it does the endpoint."""
        fake_process.register(
            ["ssh", *_SSH_BATCH_OPTS, "ubuntu@10.0.0.11", _DAEMON_REMOTE_CMD],
            stdout="27.0.3\n",
        )

        _backend(
            hosts=[{"name": "dgx-a", "address": "10.0.0.11", "user": "ubuntu"}]
        ).preflight_validate()
