# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator surfaces an in-pod unusable RDMA NIC when a task goes READY.

The k8s operator writes the ``[sflow-rdma]`` decision to the offloaded task log;
on readiness the orchestrator asks the operator (duck-typed
``network_fallback_status``) and, if a pod's RDMA NIC was unusable, emits a WARNING
hint (sflow does not force a fallback -- the libraries auto-select, so the user is
told what to set if their cluster has no NVLink fabric) and records it in the run
summary. RDMA-active / unknown / MNNVL tasks stay silent.
"""

import asyncio
import logging

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.orchestrator import Orchestrator
from sflow.core.probe import Probe, ProbeType
from sflow.core.task import Task, TaskStatus
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.k8s.rdma_preamble import RdmaRuntimeStatus


class _FakeOp(Operator):
    def __init__(self, status):
        super().__init__(OperatorConfig(type="fake"))
        self._status = status

    def build_command(self, *, task_name, script, envs) -> Command:
        return Command(exec="echo").add_arg("x")

    def network_fallback_status(self, task):
        return self._status


class _AlwaysReadyProbe(Probe):
    def __init__(self):
        super().__init__(
            type=ProbeType.READINESS, failure_threshold=1, interval=0, timeout=1
        )

    async def check(self, task) -> bool:
        return True


class _RecordingSummary:
    def __init__(self):
        self.ready: list[str] = []
        self.network: list[tuple[str, str]] = []

    def task_ready(self, task, **_):
        self.ready.append(task.name)

    def record_network_warning(self, task, message, **_):
        self.network.append((task.name, message))


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, *, containing: str) -> list[str]:
        return [r.getMessage() for r in self.records if containing in r.getMessage()]


def _mk(status):
    tg = TaskGraph()
    wf = Workflow(name="wf", task_graph=tg)
    server = Task(
        name="decode_server_0",
        operator=_FakeOp(status),
        logger=logging.getLogger("sflow.task.decode_server_0"),
        status=TaskStatus.RUNNING,
        probes=[_AlwaysReadyProbe()],
    )
    tg.dag.add_node("decode_server_0", server)
    summary = _RecordingSummary()
    orch = Orchestrator(
        workflow=wf, poll_interval=0.01, fail_fast=True, execution_summary=summary
    )
    return orch, server, summary


def test_readiness_surfaces_rdma_nic_unusable_as_hint():
    status = RdmaRuntimeStatus(
        rdma_nic_unusable=True,
        reason="all ports DOWN",
        pods_degraded=1,
        pods_total=1,
    )
    orch, server, summary = _mk(status)

    capture = _LogCapture()
    orch_logger = logging.getLogger("sflow.core.orchestrator")
    orch_logger.addHandler(capture)
    try:
        asyncio.run(orch._run_probe(server.probes[0], server))
    finally:
        orch_logger.removeHandler(capture)

    assert server.status == TaskStatus.READY
    assert summary.ready == ["decode_server_0"]
    assert len(summary.network) == 1
    name, msg = summary.network[0]
    assert name == "decode_server_0"
    # A hint, not a forced-fallback claim: it names the cause and tells the user
    # which envs THEY may set (sflow did not set them).
    assert "unusable" in msg and "all ports DOWN" in msg
    assert "did not force" in msg
    assert "NCCL_IB_DISABLE=1" in msg
    warnings = [
        r for r in capture.records if r.levelno == logging.WARNING
    ]
    assert any("decode_server_0" in r.getMessage() for r in warnings)


def test_readiness_no_warning_when_rdma_active():
    status = RdmaRuntimeStatus(
        rdma_nic_unusable=False, reason="", pods_degraded=0, pods_total=1
    )
    orch, server, summary = _mk(status)
    asyncio.run(orch._run_probe(server.probes[0], server))
    assert server.status == TaskStatus.READY
    assert summary.network == []


def test_readiness_surfaces_ucx_intra_node_tcp():
    status = RdmaRuntimeStatus(
        rdma_nic_unusable=False,
        reason="",
        pods_degraded=0,
        pods_total=0,
        ucx_intra_node_tcp=True,
        ucx_transport="rma_am(tcp/enP5p9s0)",
    )
    orch, server, summary = _mk(status)
    asyncio.run(orch._run_probe(server.probes[0], server))
    assert server.status == TaskStatus.READY
    assert len(summary.network) == 1
    _name, msg = summary.network[0]
    assert "UCX selected TCP for intra-node transport" in msg
    assert "tcp/enP5p9s0" in msg


def test_readiness_ucx_intra_node_tcp_includes_remedy_hint():
    # App-agnostic remedy guidance: cuda_ipc/NVLink not carrying KV means either an
    # IMEX ComputeDomain channel is missing OR the framework's KV memory is not
    # fabric/VMM-capable. The message names the vLLM knob only as an example.
    status = RdmaRuntimeStatus(
        rdma_nic_unusable=False,
        reason="",
        pods_degraded=0,
        pods_total=0,
        ucx_intra_node_tcp=True,
        ucx_transport="rma_am(tcp/enP5p9s0)",
    )
    orch, server, summary = _mk(status)
    asyncio.run(orch._run_probe(server.probes[0], server))
    _name, msg = summary.network[0]
    assert "IMEX" in msg or "compute_domain.channel" in msg
    assert "--enable-sleep-mode" in msg  # named only as an example


def test_readiness_surfaces_gpudirect_rdma_unavailable():
    status = RdmaRuntimeStatus(
        rdma_nic_unusable=False,
        reason="",
        pods_degraded=0,
        pods_total=1,
        gpudirect_rdma_unavailable=True,
        gpudirect_rdma_reason="nvidia_peermem/nv_peer_mem kernel module not visible in pod",
    )
    orch, server, summary = _mk(status)
    asyncio.run(orch._run_probe(server.probes[0], server))
    assert server.status == TaskStatus.READY
    assert len(summary.network) == 1
    _name, msg = summary.network[0]
    assert "GPUDirect RDMA unavailable" in msg
    assert "nvidia_peermem" in msg


def test_readiness_no_slow_tcp_warning_when_mnnvl_crossnode():
    # IB/RoCE NIC unusable, but the task is in a rack-scale NVLink (MNNVL)
    # ComputeDomain -> NCCL cross-node rides rack NVLink, so the misleading "slow
    # TCP / low performance" warning is suppressed (an INFO breadcrumb is logged).
    status = RdmaRuntimeStatus(
        rdma_nic_unusable=True,
        reason="all ports DOWN",
        pods_degraded=4,
        pods_total=4,
        mnnvl_crossnode=True,
    )
    orch, server, summary = _mk(status)

    capture = _LogCapture()
    orch_logger = logging.getLogger("sflow.core.orchestrator")
    prev_level = orch_logger.level
    orch_logger.addHandler(capture)
    orch_logger.setLevel(logging.INFO)  # so the INFO breadcrumb is emitted
    try:
        asyncio.run(orch._run_probe(server.probes[0], server))
    finally:
        orch_logger.removeHandler(capture)
        orch_logger.setLevel(prev_level)

    assert server.status == TaskStatus.READY
    # No "slow TCP" warning recorded in the summary, and no WARNING-level record.
    assert summary.network == []
    assert not [r for r in capture.records if r.levelno == logging.WARNING]
    # An INFO breadcrumb explains IB is down but MNNVL carries cross-node.
    assert capture.messages(containing="MNNVL")


def test_readiness_no_warning_when_status_unknown():
    orch, server, summary = _mk(None)
    asyncio.run(orch._run_probe(server.probes[0], server))
    assert server.status == TaskStatus.READY
    assert summary.network == []
