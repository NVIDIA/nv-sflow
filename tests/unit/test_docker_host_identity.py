# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a Docker host in the `hosts:` pool is identified.

A host answers two different questions, and conflating them is the bug this
shape exists to prevent:

  * which daemon does the *driver* send `docker run` to -- the control endpoint;
  * which address do *peer containers* dial to rendezvous.

They are usually the same machine, so `address` supplies both. They diverge on
a jump host, a Docker context, or a separate high-speed fabric, which is what
the overrides are for.
"""

import asyncio

import pytest
from pydantic import ValidationError

from sflow.plugins.backends.docker import (
    DockerBackend,
    DockerBackendConfig,
    DockerHostConfig,
)


def _host(**kwargs) -> DockerHostConfig:
    return DockerHostConfig(name="node-a", **kwargs)


class TestDerivedFromAddress:
    """The common case: one field per machine."""

    def test_address_alone_derives_an_ssh_endpoint(self):
        host = _host(address="10.0.0.11")
        assert host.control_args() == ["--host", "ssh://10.0.0.11"]

    def test_address_alone_is_also_the_rendezvous_address(self):
        assert _host(address="10.0.0.11").rendezvous() == "10.0.0.11"

    def test_user_is_applied_to_the_endpoint_only(self):
        """`user` is an SSH detail; peers dial a bare address, not `user@host`."""
        host = _host(address="10.0.0.11", user="ubuntu")

        assert host.control_args() == ["--host", "ssh://ubuntu@10.0.0.11"]
        assert host.rendezvous() == "10.0.0.11"

    def test_a_hostname_works_as_well_as_an_ip(self):
        host = _host(address="node-a.lab.internal")

        assert host.control_args() == ["--host", "ssh://node-a.lab.internal"]
        assert host.rendezvous() == "node-a.lab.internal"


class TestAliasAddresses:
    """An alias is used verbatim; sflow does not read your ~/.ssh/config.

    Resolving one via `ssh -G` was tried and cut: no shipped recipe names an
    alias, and `rendezvous_address` already states the peer address explicitly
    when the control name is not reachable from other hosts.
    """

    def test_an_alias_is_the_endpoint_and_the_peer_address(self):
        host = _host(address="dgx-a")

        assert host.control_args() == ["--host", "ssh://dgx-a"]
        assert host.rendezvous() == "dgx-a"

    def test_rendezvous_address_is_how_you_say_peers_need_something_else(self):
        host = _host(address="dgx-a", rendezvous_address="192.168.50.11")

        assert host.control_args() == ["--host", "ssh://dgx-a"]
        assert host.rendezvous() == "192.168.50.11"


class TestOverrides:
    """Escape hatches, for when control and data planes genuinely differ."""

    def test_docker_host_overrides_the_derived_endpoint(self):
        host = _host(address="10.0.0.11", docker_host="ssh://admin@bastion")

        assert host.control_args() == ["--host", "ssh://admin@bastion"]
        # The override is about reaching the daemon; peers still use `address`.
        assert host.rendezvous() == "10.0.0.11"

    def test_context_overrides_the_derived_endpoint(self):
        host = _host(address="10.0.0.11", context="lab-ctx")

        assert host.control_args() == ["--context", "lab-ctx"]
        assert host.rendezvous() == "10.0.0.11"

    def test_rendezvous_address_overrides_the_peer_address(self):
        """A separate fabric: control over management, peers over InfiniBand."""
        host = _host(address="10.0.0.11", rendezvous_address="192.168.50.11")

        assert host.control_args() == ["--host", "ssh://10.0.0.11"]
        assert host.rendezvous() == "192.168.50.11"


class TestBackCompat:
    """Recipes written against the shipped shape must keep working."""

    def test_ip_address_is_accepted_as_the_rendezvous_address(self):
        host = _host(docker_host="ssh://node-a", ip_address="10.0.0.11")

        assert host.control_args() == ["--host", "ssh://node-a"]
        assert host.rendezvous() == "10.0.0.11"

    def test_the_deprecation_reaches_the_user_through_the_log(self):
        """DeprecationWarning is silenced by default in Python, so the log line
        is the only channel that reaches the person running sflow."""
        import logging

        import sflow.plugins.backends.docker as docker_mod

        records: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture()
        docker_mod._logger.addHandler(handler)
        try:
            _host(docker_host="ssh://node-a", ip_address="10.0.0.11")
        finally:
            docker_mod._logger.removeHandler(handler)

        assert any("ip_address" in r and "rendezvous_address" in r for r in records)

    def test_rendezvous_address_wins_over_the_deprecated_alias(self):
        host = _host(
            address="10.0.0.11",
            ip_address="10.0.0.99",
            rendezvous_address="192.168.50.11",
        )
        assert host.rendezvous() == "192.168.50.11"

    def test_endpoint_only_host_still_validates(self):
        """`docker_host` with no `address` was the previously shipped shape."""
        host = _host(docker_host="ssh://node-a")
        assert host.control_args() == ["--host", "ssh://node-a"]

    def test_a_context_only_host_has_no_rendezvous_address(self):
        """A context names a daemon, not a machine -- nothing to dial."""
        assert _host(context="lab-ctx").rendezvous() is None


class TestValidation:
    def test_docker_host_and_context_are_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="docker_host.*context"):
            _host(address="10.0.0.11", docker_host="ssh://x", context="y")

    def test_a_host_with_no_address_and_no_endpoint_is_rejected(self):
        with pytest.raises(ValidationError, match="address"):
            _host()

    @pytest.mark.parametrize(
        "address, why",
        [
            # The obvious migration from the documented `docker_host: ssh://x`.
            # Accepting it yields `--host ssh://ssh://x` and a peer address no
            # one can dial -- both silent.
            ("ssh://dgx-a", "://"),
            ("tcp://dgx-a:2375", "://"),
            # Looks right in the endpoint, is wrong as a rendezvous address.
            ("root@dgx-a", "@"),
            ("dgx a", "whitespace"),
        ],
    )
    def test_decorated_addresses_are_rejected(self, address, why):
        with pytest.raises(ValidationError):
            _host(address=address)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("address", "${{ variables.ADDR_0 }}"),
            ("user", "${{ variables.SSH_USER }}"),
            ("rendezvous_address", "${{ variables.RDV }}"),
        ],
    )
    def test_unresolved_expressions_are_not_judged_yet(self, field, value):
        """These fields are Resolvable, so the model sees the template first.

        Validation runs at parse time, before `${{ ... }}` is substituted. A
        template is full of the very characters the checks reject -- spaces, and
        often `@` or `://` in the resolved value -- so judging it here rejects
        every recipe that parameterises a host. The resolved value is validated
        on the second pass, when resolve_config rebuilds this model.
        """
        host = _host(**{field: value})

        assert getattr(host, field) == value

    def test_surrounding_whitespace_is_stripped(self):
        assert _host(address="  10.0.0.11  ").rendezvous() == "10.0.0.11"

    def test_a_user_containing_an_at_sign_is_rejected(self):
        with pytest.raises(ValidationError, match="user"):
            _host(address="10.0.0.11", user="ops@corp")

    def test_the_deprecated_alias_still_names_the_peer_address(self):
        """`address` is the endpoint, `ip_address` the address peers dial --
        a legitimate control/data split, and the shape `rendezvous_address`
        replaces, so it is honoured rather than rejected."""
        host = _host(address="10.0.0.11", ip_address="10.0.0.99")

        assert host.control_args() == ["--host", "ssh://10.0.0.11"]
        assert host.rendezvous() == "10.0.0.99"

    def test_user_without_address_is_rejected(self):
        """`user` decorates a derived endpoint; with an override it is a no-op,
        and silently ignoring it would hide a real misconfiguration."""
        with pytest.raises(ValidationError, match="user"):
            _host(docker_host="ssh://node-a", user="ubuntu")


class TestIdentityReachesTheCommand:
    """The derived endpoint has to survive all the way to the argv.

    Every command sflow issues about a container -- launch, teardown, orphan
    sweep -- must reach the same daemon. A host that resolves correctly in the
    config but loses its flag on the way to `docker` runs the workload on the
    driver instead, which looks like success until the results are wrong.
    """

    @staticmethod
    def _operator(host_spec: dict, node: str = "node-a"):
        backend = DockerBackend(
            DockerBackendConfig(
                name="cluster",
                type="docker",
                image="ubuntu:22.04",
                hosts=[host_spec],
            )
        )
        backend.allocation = asyncio.run(backend.allocate())
        operator = backend.default_operator(name="d")
        operator.apply_backend_context(
            backend=backend, assigned_nodes=[node], artifacts=[]
        )
        return operator

    def test_an_address_only_host_is_targeted_over_ssh(self):
        operator = self._operator({"name": "node-a", "address": "10.0.0.11"})
        parts = operator.build_command(
            task_name="t", script=["echo hi"], envs={}
        ).as_list()

        assert parts[:3] == ["docker", "--host", "ssh://10.0.0.11"]

    def test_user_reaches_the_command(self):
        operator = self._operator(
            {"name": "node-a", "address": "10.0.0.11", "user": "ubuntu"}
        )
        parts = operator.build_command(
            task_name="t", script=["echo hi"], envs={}
        ).as_list()

        assert parts[:3] == ["docker", "--host", "ssh://ubuntu@10.0.0.11"]

    def test_teardown_reaches_the_same_daemon_as_the_launch(self):
        operator = self._operator({"name": "node-a", "address": "10.0.0.11"})

        teardown = [c.as_list() for c in operator.teardown_commands(task_name="t")]

        assert all(c[:3] == ["docker", "--host", "ssh://10.0.0.11"] for c in teardown)
