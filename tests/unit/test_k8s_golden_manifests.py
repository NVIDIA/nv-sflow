# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline golden-manifest tests for the Kubernetes operators.

CI usually cannot reach a real cluster, so instead of applying manifests we
render them from the operator build path with a deterministic placeholder
allocation (fixed node names/IPs, allocation id ``abc``) and a seeded SSH
keypair, then compare the full pod/MPIJob manifest against a checked-in golden
YAML. Any drift in the rendered manifest fails the test.

These cases mirror the backend-agnostic samples in ``examples/modular``:
Dynamo-style GPU servers + a CPU helper on the plain ``k8s`` operator, and a
cross-node server on the ``k8s_mpi`` operator (both routes).

Regenerate goldens after an intentional manifest change:

    SFLOW_UPDATE_GOLDEN=1 pytest tests/unit/test_k8s_golden_manifests.py
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig
from sflow.plugins.k8s.capabilities import CapabilityState
from sflow.plugins.operators.k8s import K8sOperator, K8sOperatorConfig
from sflow.plugins.operators.k8s_mpi import K8sMpiOperator, K8sMpiOperatorConfig

_MARK = "SFLOW_K8S_MANIFEST"
_GOLDEN_DIR = Path(__file__).parent / "golden" / "k8s_manifests"


# ---------------------------------------------------------------------------
# deterministic fixtures
# ---------------------------------------------------------------------------
def _backend(scheduling="device_plugin", gpus_per_node=8, nodes=1, namespace="golden-ns"):
    backend = KubernetesBackend(
        KubernetesBackendConfig(
            name="cluster",
            type="kubernetes",
            namespace=namespace,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            scheduling=scheduling,
            host_network=True,
        )
    )
    backend.allocation = Allocation(
        allocation_id="abc",
        nodes=[
            ComputeNode(name=f"node-{i}", ip_address=f"10.0.0.{i + 1}", index=i,
                        num_gpus=gpus_per_node)
            for i in range(nodes)
        ],
        owned=True,
    )
    backend._node_to_resv_pod = {f"node-{i}": f"res-{i}" for i in range(nodes)}
    return backend


def _extract_manifest(shell: str) -> dict:
    after = shell.split(_MARK, 1)[1]
    body = after.split("\n" + _MARK, 1)[0]
    return json.loads(body.split("\n", 1)[1])


def _seed_mpi_keypair(monkeypatch):
    monkeypatch.setattr(
        "sflow.plugins.operators.k8s_mpi._generate_ssh_keypair_b64",
        lambda: ("PRIVB64", "PUBB64"),
    )


# ---------------------------------------------------------------------------
# case builders (each returns the rendered manifest dict)
# ---------------------------------------------------------------------------
def _case_k8s_dynamo_server_device_plugin(monkeypatch):
    backend = _backend("device_plugin", gpus_per_node=8, nodes=1)
    op = K8sOperator(K8sOperatorConfig(name="server", image="dynamo:trtllm"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=["node-0"], artifacts=[], gpu_count=4,
    )
    cmd = op.build_command(
        task_name="server",
        script=["trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /models/m"],
        envs={"SERVED_MODEL_NAME": "demo"},
    )
    return _extract_manifest(cmd.as_list()[-1])


def _case_k8s_dynamo_server_dra(monkeypatch):
    backend = _backend("dra", gpus_per_node=8, nodes=1)
    op = K8sOperator(K8sOperatorConfig(name="server", image="dynamo:trtllm"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=["node-0"], artifacts=[], gpu_count=4,
    )
    cmd = op.build_command(
        task_name="server",
        script=["trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /models/m"],
        envs={"SERVED_MODEL_NAME": "demo"},
    )
    return _extract_manifest(cmd.as_list()[-1])


def _case_k8s_dynamo_helper_cpu(monkeypatch):
    backend = _backend("device_plugin", gpus_per_node=8, nodes=1)
    op = K8sOperator(K8sOperatorConfig(name="helper", image="dynamo:trtllm"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=["node-0"], artifacts=[], gpu_count=None,
    )
    cmd = op.build_command(
        task_name="frontend_server",
        script=["python3 -m dynamo.frontend --http-port 8000"],
        envs={},
    )
    return _extract_manifest(cmd.as_list()[-1])


def _mpi_op(backend, *, route, has_mpi_operator=None):
    op = K8sMpiOperator(
        K8sMpiOperatorConfig(
            name="server", image="trtllm:latest", run_as_root=True,
            mpi={"route": route, "ensure_sshd": True, "ssh_port": 2222},
        )
    )
    op.apply_backend_context(
        backend=backend, assigned_nodes=["node-0", "node-1"], artifacts=[], gpu_count=16,
    )
    if has_mpi_operator is True:
        op._mpi_state = CapabilityState.USABLE
    elif has_mpi_operator is False:
        op._mpi_state = CapabilityState.ABSENT
    return op


def _case_k8s_mpi_server_pods(monkeypatch):
    _seed_mpi_keypair(monkeypatch)
    backend = _backend("device_plugin", gpus_per_node=8, nodes=2)
    op = _mpi_op(backend, route="pods")
    envs = op._inject_pods_keypair_env(
        {"SFLOW_TASK_ASSIGNED_NODE_IPS": "10.0.0.1,10.0.0.2"}
    )
    plan = op._build_execution_plan(
        task_name="server",
        script=["exec mpirun -np 16 trtllm-serve /models/m"],
        envs=envs,
    )
    return _extract_manifest(plan.apply_command.as_list()[-1])


def _case_k8s_mpi_server_operator(monkeypatch):
    _seed_mpi_keypair(monkeypatch)
    backend = _backend("device_plugin", gpus_per_node=8, nodes=2)
    op = _mpi_op(backend, route="operator", has_mpi_operator=True)
    plan = op._build_mpijob_execution_plan(
        task_name="server",
        script=["exec mpirun -np 16 trtllm-serve /models/m"],
        envs={},
    )
    return _extract_manifest(plan.apply_command.as_list()[-1])


_CASES = {
    "k8s_dynamo_server_device_plugin": _case_k8s_dynamo_server_device_plugin,
    "k8s_dynamo_server_dra": _case_k8s_dynamo_server_dra,
    "k8s_dynamo_helper_cpu": _case_k8s_dynamo_helper_cpu,
    "k8s_mpi_server_pods": _case_k8s_mpi_server_pods,
    "k8s_mpi_server_operator": _case_k8s_mpi_server_operator,
}


def _dump(manifest: dict) -> str:
    return yaml.safe_dump(manifest, sort_keys=True, default_flow_style=False, width=100)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_k8s_manifest_matches_golden(name, monkeypatch):
    manifest = _CASES[name](monkeypatch)
    rendered = _dump(manifest)

    path = _GOLDEN_DIR / f"{name}.yaml"
    if os.environ.get("SFLOW_UPDATE_GOLDEN"):
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"updated golden {path.name}")

    assert path.exists(), (
        f"missing golden {path}; regenerate with SFLOW_UPDATE_GOLDEN=1"
    )
    assert rendered == path.read_text(), (
        f"rendered k8s manifest for '{name}' drifted from its golden; "
        f"if intended, regenerate with SFLOW_UPDATE_GOLDEN=1"
    )
