# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sflow.app.task_context import build_task_expression_hint, compute_task_service
from sflow.core.backend import Allocation, BackendCapabilities
from sflow.core.compute_node import ComputeNode
from sflow.core.task import TaskPort

_TASK_CTX_WITH_SERVICE = {
    "frontend": {
        "nodes": [],
        "gpus": [],
        "backend": "b",
        "operator": "o",
        "service": {"host": "", "port": "", "url": ""},
    }
}


class _Backend:
    def __init__(self, *, has_addresses: bool):
        self.capabilities = BackendCapabilities(
            has_runtime_node_addresses=has_addresses
        )
        self.allocation = Allocation(
            allocation_id="a",
            nodes=[ComputeNode(name="n1", ip_address="10.0.0.5", index=0)],
        )


def test_compute_task_service_full_for_runtime_address_backend():
    svc = compute_task_service(
        backend=_Backend(has_addresses=True),
        assigned_nodes=["n1"],
        ports=[TaskPort(port=8000, name="http")],
    )
    assert svc == {"host": "10.0.0.5", "port": 8000, "url": "http://10.0.0.5:8000"}


def test_compute_task_service_empty_host_without_runtime_addresses():
    # e.g. Kubernetes: no node IP, so host/url stay empty.
    svc = compute_task_service(
        backend=_Backend(has_addresses=False),
        assigned_nodes=["n1"],
        ports=[TaskPort(port=8000)],
    )
    assert svc["host"] == ""
    assert svc["port"] == 8000
    assert svc["url"] == ""


def test_compute_task_service_no_ports_yields_no_url():
    svc = compute_task_service(
        backend=_Backend(has_addresses=True),
        assigned_nodes=["n1"],
        ports=[],
    )
    assert svc["host"] == "10.0.0.5"
    assert svc["port"] == ""
    assert svc["url"] == ""


def test_compute_task_service_no_backend_or_nodes():
    assert compute_task_service(backend=None, assigned_nodes=[], ports=[]) == {
        "host": "",
        "port": "",
        "url": "",
    }


def test_task_hint_treats_service_as_valid_attribute():
    # A valid service ref must not produce a (misleading) "service is invalid" hint.
    assert (
        build_task_expression_hint(
            ["${{ task.frontend.service.url }}"], _TASK_CTX_WITH_SERVICE, None
        )
        is None
    )


def test_task_hint_flags_bad_service_subattribute():
    hint = build_task_expression_hint(
        ["${{ task.frontend.service.urls }}"], _TASK_CTX_WITH_SERVICE, None
    )
    assert "not an available service attribute" in hint
    assert "host, port, url" in hint
    assert "not an available task attribute" not in hint
