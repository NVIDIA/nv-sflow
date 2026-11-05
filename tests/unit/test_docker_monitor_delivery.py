# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Docker backend's monitor delivery: a collector on the node being sampled.

A declared `hosts:` pool runs on other machines, so the bare-host collector on
the driver would report the driver. Instead the collector runs on each host --
bare over ssh when the host has an ssh endpoint, else as a container on its own
daemon -- reading that machine's hardware and streaming samples back over
stdout. Neither transport needs the host to share a filesystem with the driver.
"""

import asyncio
import os
import subprocess
import time

from sflow.plugins.backends.docker import DockerBackend, DockerBackendConfig
from sflow.plugins.operators.docker_run import _OWNER_PREFIX

_ALL = ["cpu", "gpu", "memory", "disk", "network"]
_CPU_ONLY = ["cpu"]


def _pool(**kwargs) -> DockerBackend:
    """Hosts sflow can ssh to -- the collector runs bare on them."""
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="nvcr.io/example/img:1",
            hosts=[
                {"name": "node-0", "address": "10.0.0.11", "user": "svc"},
                {"name": "node-1", "address": "10.0.0.12"},
            ],
            **kwargs,
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    return backend


def _daemon_pool(**kwargs) -> DockerBackend:
    """Hosts that name only a daemon -- no machine to ssh to, so a container."""
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="nvcr.io/example/img:1",
            hosts=[
                {"name": "node-0", "docker_host": "tcp://10.0.0.11:2376"},
                {"name": "node-1", "docker_host": "tcp://10.0.0.12:2376"},
            ],
            **kwargs,
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    return backend


def _local() -> DockerBackend:
    backend = DockerBackend(
        DockerBackendConfig(
            name="local_docker", type="docker", image="img:1", nodes=1
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    return backend


def test_a_pool_is_monitorable_again():
    """Phase 0 skipped these nodes because only the driver could be sampled.
    With a per-host collector that limitation is gone."""
    assert _pool().capabilities.supports_host_monitoring is True


# --- ssh-reachable hosts: bare collector -----------------------------------


def test_an_ssh_host_is_sampled_bare_not_in_a_container():
    """No image involved: the workload image need not carry python3 or have its
    entrypoint cleared, and nothing is pulled just to sample."""
    prefix = list(_pool().monitor_delivery(node_name="node-0", scopes=_ALL).launch_prefix)

    assert prefix[0] == "ssh"
    assert "svc@10.0.0.11" in prefix
    assert "docker" not in prefix
    assert "nvcr.io/example/img:1" not in prefix
    # Never a prompt: a pool's worth of collectors start at once on one TTY.
    assert "BatchMode=yes" in prefix


def test_each_host_gets_its_own_endpoint():
    backend = _pool()
    first = backend.monitor_delivery(node_name="node-0", scopes=_ALL).launch_prefix
    second = backend.monitor_delivery(node_name="node-1", scopes=_ALL).launch_prefix
    assert "svc@10.0.0.11" in first
    # No `user:` on the second host -> the target carries no login.
    assert "10.0.0.12" in second


def test_an_ssh_docker_host_endpoint_is_sampled_bare_too():
    """`docker_host: ssh://x` names a machine as well as a daemon."""
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="img:1",
            hosts=[{"name": "dgx-a", "docker_host": "ssh://dgx-a"}],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())
    prefix = list(backend.monitor_delivery(node_name="dgx-a", scopes=_ALL).launch_prefix)
    assert prefix[0] == "ssh" and "dgx-a" in prefix


def test_the_bare_collector_streams_back_and_is_torn_down_on_the_same_host():
    delivery = _pool().monitor_delivery(node_name="node-0", scopes=_ALL)
    assert delivery.offhost is True
    stop = list(delivery.stop_argv)
    assert stop[0] == "ssh" and "svc@10.0.0.11" in stop


def test_teardown_kills_exactly_the_process_the_launch_started(fp):
    """Killing the local ssh client does not signal the remote collector (no
    pty), so the launch records the pid that `exec` hands to python3 and the
    teardown kills it. Run both snippets through a real shell -- what the remote
    login shell does with them -- with `sleep` standing in for python3."""
    fp.allow_unregistered(True)  # real processes, not pytest-subprocess fakes
    delivery = _pool().monitor_delivery(node_name="node-0", scopes=_ALL)
    launch, stop = delivery.launch_prefix[-1], delivery.stop_argv[-1]
    pidfile = launch.split("> ", 1)[1].split(";", 1)[0]

    collector = subprocess.Popen(["sh", "-c", f"{launch} sleep 30"])
    try:
        # `>` creates the file before echo fills it, so wait for content.
        deadline = time.monotonic() + 5
        recorded = ""
        while not recorded and time.monotonic() < deadline:
            if os.path.exists(pidfile):
                with open(pidfile) as handle:
                    recorded = handle.read().strip()
            time.sleep(0.01)
        # `exec` replaced the shell, so the recorded pid IS the collector.
        assert int(recorded) == collector.pid

        subprocess.run(["sh", "-c", stop], check=True)

        assert collector.wait(timeout=5) < 0  # killed by signal
        assert not os.path.exists(pidfile)
    finally:
        collector.kill()
        if os.path.exists(pidfile):
            os.remove(pidfile)


def test_teardown_of_a_collector_that_never_started_is_harmless(fp):
    fp.allow_unregistered(True)
    stop = _pool().monitor_delivery(node_name="node-0", scopes=_ALL).stop_argv[-1]
    result = subprocess.run(["sh", "-c", stop], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and result.stderr == ""


def test_a_bare_collector_assumes_gpus_are_visible(caplog):
    """On the host nvidia-smi is there or it isn't; `gpus_per_node` only gated
    the container's --gpus request, so it neither drops the scope nor warns."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert _pool().monitor_delivery(node_name="node-0", scopes=_ALL).provides_gpus
    assert not [r for r in caplog.records if "gpus_per_node" in r.message]


# --- daemon-only endpoints: container collector ----------------------------


def test_a_daemon_endpoint_runs_the_collector_on_that_daemon():
    delivery = _daemon_pool().monitor_delivery(node_name="node-0", scopes=_ALL)
    prefix = list(delivery.launch_prefix)

    assert prefix[0] == "docker"
    # Routed to THAT host's daemon -- the whole point; the driver's daemon would
    # sample the driver.
    assert prefix[1:3] == ["--host", "tcp://10.0.0.11:2376"]
    assert "run" in prefix
    assert prefix[-1] == "nvcr.io/example/img:1"


def test_host_namespace_flags_are_present():
    """/proc/net/dev is namespaced and gethostname() returns the container id,
    so without --network host the collector reports a veth pair and files its
    samples under a name nothing downstream recognises."""
    prefix = list(
        _daemon_pool().monitor_delivery(node_name="node-0", scopes=_ALL).launch_prefix
    )
    assert "--network" in prefix and prefix[prefix.index("--network") + 1] == "host"


def test_gpus_are_requested_only_when_the_gpu_scope_is():
    """--gpus all fails outright on a host with no toolkit, which would take
    the CPU scopes down with it. Ask only when there is something to read."""
    backend = _daemon_pool(gpus_per_node=8)
    with_gpu = list(backend.monitor_delivery(node_name="node-0", scopes=_ALL).launch_prefix)
    assert "--gpus" in with_gpu and with_gpu[with_gpu.index("--gpus") + 1] == "all"

    without = list(
        backend.monitor_delivery(node_name="node-0", scopes=_CPU_ONLY).launch_prefix
    )
    assert "--gpus" not in without


def test_the_container_is_named_so_a_crashed_driver_can_be_reaped():
    """docker_run's orphan sweep parses the owning pid out of the name; a
    monitor container that does not follow it would outlive a killed driver and
    keep sampling forever."""
    prefix = list(
        _daemon_pool().monitor_delivery(node_name="node-0", scopes=_ALL).launch_prefix
    )
    name = prefix[prefix.index("--name") + 1]

    assert name.startswith(_OWNER_PREFIX)
    pid = name[len(_OWNER_PREFIX):].split("-")[0]
    assert pid.isdigit(), name
    assert "node-0" in name
    # --rm so the ordinary path leaves nothing behind at all.
    assert "--rm" in prefix


def test_the_container_is_torn_down_by_name_not_just_cancelled():
    """`docker run` is a client, not the container's parent. Cancelling it left
    a live container sampling on the daemon after the workflow finished, so the
    delivery carries an explicit removal."""
    delivery = _daemon_pool().monitor_delivery(node_name="node-0", scopes=_ALL)
    stop = list(delivery.stop_argv)
    name = list(delivery.launch_prefix)[list(delivery.launch_prefix).index("--name") + 1]

    assert stop[0] == "docker"
    # Aimed at the same daemon that was told to start it.
    assert stop[1:3] == ["--host", "tcp://10.0.0.11:2376"]
    assert stop[-3:] == ["rm", "-f", name]


def test_the_source_is_inlined_and_samples_stream_back():
    delivery = _daemon_pool().monitor_delivery(node_name="node-0", scopes=_ALL)
    # The remote host can neither see the driver's materialized script nor
    # write to the driver's raw dir.
    assert delivery.offhost is True
    # -i so the redirect on the driver reaches the container's stdin.
    assert "-i" in delivery.launch_prefix


def test_a_cpu_only_pool_never_asks_for_gpus():
    """The default monitor requests EVERY scope, gpu included, so scope alone
    never rules `--gpus all` out. On a pool with no GPUs declared, asking for
    them is refused at daemon admission and takes the whole container down --
    losing cpu/memory/disk/network too."""
    backend = _daemon_pool()  # no gpus_per_node anywhere
    prefix = list(backend.monitor_delivery(node_name="node-0", scopes=_ALL).launch_prefix)
    assert "--gpus" not in prefix


def test_a_host_declaring_its_own_gpus_still_gets_them():
    """gpus_per_node is per-host as well as per-backend; a pool may be mixed."""
    backend = DockerBackend(
        DockerBackendConfig(
            name="docker_cluster",
            type="docker",
            image="img:1",
            hosts=[
                {"name": "gpu-host", "docker_host": "tcp://10.0.0.11:2376", "gpus_per_node": 8},
                {"name": "cpu-host", "docker_host": "tcp://10.0.0.12:2376"},
            ],
        )
    )
    backend.allocation = asyncio.run(backend.allocate())

    gpu_host = list(
        backend.monitor_delivery(node_name="gpu-host", scopes=_ALL).launch_prefix
    )
    cpu_host = list(
        backend.monitor_delivery(node_name="cpu-host", scopes=_ALL).launch_prefix
    )
    assert "--gpus" in gpu_host
    assert "--gpus" not in cpu_host


def test_a_pool_without_gpus_says_so_to_the_planner():
    """Declining `--gpus` and still asking for the gpu scope produced a log of
    'nvidia-smi not found' rows. The delivery reports what the node can read."""
    assert _daemon_pool().monitor_delivery(
        node_name="node-0", scopes=_ALL
    ).provides_gpus is False
    assert _daemon_pool(gpus_per_node=8).monitor_delivery(
        node_name="node-0", scopes=_ALL
    ).provides_gpus is True


def test_an_undeclared_gpu_count_is_said_out_loud(caplog):
    """`None` is "nobody said", not "this box has none".

    Both fall back to running without GPUs, so a GPU box whose pool never
    declared `gpus_per_node` reported no GPU metrics at all -- and the only
    trace was an INFO line saying it "declares no GPUs", which reads like the
    box genuinely has none.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        _daemon_pool().monitor_delivery(node_name="node-0", scopes=_ALL)

    assert any("gpus_per_node" in r.message for r in caplog.records)

    # A declared count -- either way -- is an answer, so nothing is warned.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _daemon_pool(gpus_per_node=8).monitor_delivery(node_name="node-0", scopes=_ALL)
        _daemon_pool(gpus_per_node=0).monitor_delivery(node_name="node-0", scopes=_ALL)
    assert not [r for r in caplog.records if "gpus_per_node" in r.message]


def test_the_disk_scope_measures_the_node_not_the_image():
    """The collector's container shares the host's NET namespace but not its
    MOUNT namespace, so a bare `statvfs('/')` reports the image overlay's free
    space and files it as the node's root."""
    prefix = _daemon_pool().monitor_delivery(node_name="node-0", scopes=_ALL).launch_prefix

    assert "-v" in prefix and "/:/sflow-host-root:ro" in prefix
    assert "SFLOW_MONITOR_DISK_ROOT=/sflow-host-root" in prefix


def test_no_host_root_is_mounted_when_disk_is_not_sampled():
    """Read-only or not, binding the node's whole filesystem is not something
    to do for a scope nobody asked for."""
    prefix = _daemon_pool().monitor_delivery(
        node_name="node-0", scopes=_CPU_ONLY
    ).launch_prefix

    assert "/:/sflow-host-root:ro" not in prefix
    assert not any("SFLOW_MONITOR_DISK_ROOT" in token for token in prefix)


# --- no hosts: in place ----------------------------------------------------


def test_a_single_machine_docker_backend_is_unchanged():
    """Without `hosts:` the only node is this machine, where the collector has
    always run in place. Nothing about that should move."""
    delivery = _local().monitor_delivery(node_name="localhost", scopes=_ALL)
    assert delivery.launch_prefix == ()
    assert delivery.offhost is False


def test_a_single_machine_backend_needs_no_teardown():
    assert _local().monitor_delivery(node_name="localhost", scopes=_ALL).stop_argv == ()


def test_a_host_delivery_assumes_gpus_are_visible():
    """On a host the collector runs directly: nvidia-smi is there or it isn't,
    and nothing sflow does changes that."""
    assert _local().monitor_delivery(
        node_name="localhost", scopes=_ALL
    ).provides_gpus is True
