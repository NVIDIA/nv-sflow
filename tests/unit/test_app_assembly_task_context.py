# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for task context resolution in assembly.py.

The `task` context allows task scripts to reference information about other tasks
(or the same task), such as:
- ${{ task.task_name.nodes[0].ip_address }}
- ${{ task.task_name.nodes[0].name }}
- ${{ task.task_name.gpus }}

For replicated tasks, you can also use indexed access:
- ${{ task.task_name[0].nodes[0].ip_address }} - access replica 0
- ${{ task.task_name[1].gpus }} - access replica 1's GPUs
"""

from collections.abc import Sequence

import pytest

from sflow.app.assembly import build_task_graph, _build_tasks_ctx, _build_task_info
from sflow.config.schema import ReplicaConfig
from sflow.config.schema import (
    GpuResourceConfig,
    NodeResourceConfig,
    ResourcesConfig,
    SflowConfig,
    TaskConfig,
    WorkflowConfig,
)
from sflow.core.backend import Allocation, Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.operator import Operator
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.plugins.operators.srun import SrunOperator, SrunOperatorConfig


class _FakeBackend(Backend):
    def __init__(self, name: str, allocation: Allocation | None):
        super().__init__(name=name)
        self.allocation = allocation

    async def allocate(self) -> Allocation:
        raise RuntimeError("not used in this unit test")

    async def release(self, allocation: Allocation) -> None:
        raise RuntimeError("not used in this unit test")

    def default_operator(
        self,
        *,
        name: str,
        assigned_nodes: Sequence[str] | None = None,
    ) -> Operator:
        if self.name == "local":
            return BashOperator(BashOperatorConfig(name=name))
        job_id = str(self.allocation.allocation_id) if self.allocation else "0"
        nodelist = []
        if assigned_nodes:
            nodelist = list(assigned_nodes)
        elif self.allocation:
            nodelist = [n.name for n in (self.allocation.nodes or [])]
        return SrunOperator(
            SrunOperatorConfig(name=name, job_id=job_id, nodelist=nodelist)
        )


def _state() -> SflowState:
    wf = Workflow(name="wf", task_graph=TaskGraph())
    return SflowState(workflow=wf)


def test_build_tasks_ctx_populates_node_info_and_gpus():
    """_build_tasks_ctx should return task info with nodes and GPUs."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo t1"],
                    resources=ResourcesConfig(
                        nodes=NodeResourceConfig(indices=[0]),
                        gpus=GpuResourceConfig(count=2),
                    ),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    tasks_ctx = _build_tasks_ctx(tg, state.backends)

    assert "t1" in tasks_ctx
    t1_ctx = tasks_ctx["t1"]
    
    # Check nodes
    assert len(t1_ctx["nodes"]) == 1
    assert t1_ctx["nodes"][0]["name"] == "n1"
    assert t1_ctx["nodes"][0]["ip_address"] == "10.0.0.1"
    assert t1_ctx["nodes"][0]["num_gpus"] == 4
    
    # Check GPUs
    assert t1_ctx["gpus"] == [0, 1]
    
    # Check metadata
    assert t1_ctx["backend"] == "b1"


def test_task_context_resolves_node_ip_in_script():
    """Task scripts can reference other tasks' node IPs via ${{ task.name.nodes[0].ip_address }}."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="server",
                    script=["echo starting server"],
                    resources=ResourcesConfig(nodes=NodeResourceConfig(indices=[0])),
                ),
                TaskConfig(
                    name="client",
                    script=[
                        "echo connecting to ${{ task.server.nodes[0].ip_address }}",
                        "connect --host=${{ task.server.nodes[0].name }}",
                    ],
                    resources=ResourcesConfig(nodes=NodeResourceConfig(indices=[1])),
                    depends_on=["server"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    client = tg.get_task("client")
    
    # The script should have the server's IP resolved
    assert client.script[0] == "echo connecting to 10.0.0.1"
    assert client.script[1] == "connect --host=n1"


def test_task_context_resolves_gpus_in_script():
    """Task scripts can reference other tasks' GPU assignments via ${{ task.name.gpus }}."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=8),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="prefill",
                    script=["echo prefill"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=4)),
                ),
                TaskConfig(
                    name="decode",
                    script=[
                        "echo prefill uses GPUs: ${{ task.prefill.gpus }}",
                    ],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=4)),
                    depends_on=["prefill"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    decode = tg.get_task("decode")
    
    # The script should have the prefill's GPU list resolved
    assert decode.script[0] == "echo prefill uses GPUs: [0, 1, 2, 3]"


def test_task_context_self_reference():
    """Task can reference its own nodes and GPUs via task context."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="worker",
                    script=[
                        "echo my IP is ${{ task.worker.nodes[0].ip_address }}",
                        "echo my GPUs are ${{ task.worker.gpus }}",
                    ],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    worker = tg.get_task("worker")
    
    assert worker.script[0] == "echo my IP is 10.0.0.1"
    assert worker.script[1] == "echo my GPUs are [0, 1]"


def test_task_context_with_replicas():
    """Task context works correctly with replicated tasks."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="server",
                    script=["echo server"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=1)),
                ),
                TaskConfig(
                    name="client",
                    script=[
                        # Reference the server task's node
                        "connect ${{ task.server.nodes[0].ip_address }}",
                    ],
                    replicas={"count": 2, "policy": "parallel"},
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=1)),
                    depends_on=["server"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    
    # Both replicas should have the server's IP resolved
    client_0 = tg.get_task("client_0")
    client_1 = tg.get_task("client_1")
    
    assert client_0.script[0] == "connect 10.0.0.1"
    assert client_1.script[0] == "connect 10.0.0.1"


def test_task_context_replica_references_other_replica():
    """Replicas can reference other replicas by their full name."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=8),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="worker",
                    script=[
                        # worker_0 references itself, worker_1 references worker_0
                        "echo worker_0 GPUs: ${{ task.worker_0.gpus }}",
                    ],
                    replicas={"count": 2, "policy": "parallel"},
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    
    worker_0 = tg.get_task("worker_0")
    worker_1 = tg.get_task("worker_1")
    
    # Both should resolve worker_0's GPUs
    assert worker_0.script[0] == "echo worker_0 GPUs: [0, 1]"
    assert worker_1.script[0] == "echo worker_0 GPUs: [0, 1]"


def test_task_context_mixed_with_other_contexts():
    """Task context works alongside other contexts (variables, backends, artifacts)."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]
    
    from sflow.core.variable import Variable, VariableType
    state.variables = {
        "PORT": Variable(name="PORT", value=8080, type=VariableType.INTEGER),
    }

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="server",
                    script=["echo server"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                ),
                TaskConfig(
                    name="client",
                    script=[
                        # Mix task context with variable context
                        "connect http://${{ task.server.nodes[0].ip_address }}:${{ PORT }}",
                    ],
                    depends_on=["server"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    client = tg.get_task("client")
    
    assert client.script[0] == "connect http://10.0.0.1:8080"


def test_task_context_undefined_task_raises_error():
    """Referencing an undefined task in task context should raise an error."""
    state = _state()
    state.backends = {"local": _FakeBackend("local", allocation=None)}
    state.default_backend = state.backends["local"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="client",
                    script=[
                        "connect ${{ task.nonexistent.nodes[0].ip_address }}",
                    ],
                ),
            ],
        ),
    )

    with pytest.raises(ValueError, match=r"Failed to resolve task expression"):
        build_task_graph(config, state)


def test_task_context_multi_node_task():
    """Task context correctly handles tasks assigned to multiple nodes."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1, num_gpus=4),
                    ComputeNode(name="n3", ip_address="10.0.0.3", index=2, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="distributed",
                    script=["echo distributed task"],
                    resources=ResourcesConfig(nodes=NodeResourceConfig(count=2)),
                ),
                TaskConfig(
                    name="monitor",
                    script=[
                        "echo first node: ${{ task.distributed.nodes[0].ip_address }}",
                        "echo second node: ${{ task.distributed.nodes[1].ip_address }}",
                    ],
                    depends_on=["distributed"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    monitor = tg.get_task("monitor")
    
    assert monitor.script[0] == "echo first node: 10.0.0.1"
    assert monitor.script[1] == "echo second node: 10.0.0.2"


def test_task_context_replica_indexed_access():
    """Replicated tasks can be accessed by base name with replica index."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=8),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="prefill",
                    script=["echo prefill"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                ),
                TaskConfig(
                    name="client",
                    script=[
                        # Access prefill replicas by index
                        "echo prefill_0 IP: ${{ task.prefill[0].nodes[0].ip_address }}",
                        "echo prefill_1 GPUs: ${{ task.prefill[1].gpus }}",
                        "echo prefill_0 GPUs: ${{ task.prefill[0].gpus }}",
                    ],
                    depends_on=["prefill"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    client = tg.get_task("client")
    
    # Verify indexed access to replicas works
    assert client.script[0] == "echo prefill_0 IP: 10.0.0.1"
    assert client.script[1] == "echo prefill_1 GPUs: [2, 3]"
    assert client.script[2] == "echo prefill_0 GPUs: [0, 1]"


def test_task_context_replica_indexed_and_full_name_access():
    """Both indexed and full name access work for replicated tasks."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=8),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="worker",
                    script=["echo worker"],
                    replicas=ReplicaConfig(count=3, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                ),
                TaskConfig(
                    name="monitor",
                    script=[
                        # Access by full name
                        "echo worker_0: ${{ task.worker_0.gpus }}",
                        "echo worker_1: ${{ task.worker_1.gpus }}",
                        "echo worker_2: ${{ task.worker_2.gpus }}",
                        # Access by indexed base name
                        "echo worker[0]: ${{ task.worker[0].gpus }}",
                        "echo worker[1]: ${{ task.worker[1].gpus }}",
                        "echo worker[2]: ${{ task.worker[2].gpus }}",
                    ],
                    depends_on=["worker"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    monitor = tg.get_task("monitor")
    
    # Both access patterns should give the same results
    assert monitor.script[0] == "echo worker_0: [0, 1]"
    assert monitor.script[1] == "echo worker_1: [2, 3]"
    assert monitor.script[2] == "echo worker_2: [4, 5]"
    assert monitor.script[3] == "echo worker[0]: [0, 1]"
    assert monitor.script[4] == "echo worker[1]: [2, 3]"
    assert monitor.script[5] == "echo worker[2]: [4, 5]"


def test_task_context_replica_indexed_access_multi_node():
    """Indexed access works correctly for replicas assigned to different nodes."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="server",
                    script=["echo server"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(
                        nodes=NodeResourceConfig(count=1),
                        gpus=GpuResourceConfig(count=4),
                    ),
                ),
                TaskConfig(
                    name="client",
                    script=[
                        # Access each replica's node info
                        "echo server_0 is at ${{ task.server[0].nodes[0].ip_address }}",
                        "echo server_1 is at ${{ task.server[1].nodes[0].ip_address }}",
                        "echo server_0 host: ${{ task.server[0].nodes[0].name }}",
                        "echo server_1 host: ${{ task.server[1].nodes[0].name }}",
                    ],
                    depends_on=["server"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    client = tg.get_task("client")
    
    # Each replica should be on a different node
    assert client.script[0] == "echo server_0 is at 10.0.0.1"
    assert client.script[1] == "echo server_1 is at 10.0.0.2"
    assert client.script[2] == "echo server_0 host: n1"
    assert client.script[3] == "echo server_1 host: n2"


def test_task_context_variable_sweep_replica_indexed_access():
    """Variable sweep replicas can be accessed by index."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=8),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]
    
    from sflow.core.variable import Variable, VariableType
    state.variables = {
        "BATCH_SIZE": Variable(
            name="BATCH_SIZE",
            value=32,
            type=VariableType.INTEGER,
            domain=[32, 64],
        ),
    }

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="train",
                    script=["echo training with batch size"],
                    replicas=ReplicaConfig(variables=["BATCH_SIZE"], policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                ),
                TaskConfig(
                    name="monitor",
                    script=[
                        # Access variable sweep replicas by index
                        "echo train[0] GPUs: ${{ task.train[0].gpus }}",
                        "echo train[1] GPUs: ${{ task.train[1].gpus }}",
                    ],
                    depends_on=["train"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    monitor = tg.get_task("monitor")
    
    assert monitor.script[0] == "echo train[0] GPUs: [0, 1]"
    assert monitor.script[1] == "echo train[1] GPUs: [2, 3]"


def test_task_assigned_node_env_vars_single_node():
    """Tasks get SFLOW_TASK_ASSIGNED_NODE_NAMES and SFLOW_TASK_ASSIGNED_NODE_IPS env vars."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo t1"],
                    resources=ResourcesConfig(nodes=NodeResourceConfig(indices=[0])),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    
    assert t1.envs["SFLOW_TASK_ASSIGNED_NODE_NAMES"] == "n1"
    assert t1.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] == "10.0.0.1"


def test_task_assigned_node_env_vars_multi_node():
    """Tasks assigned to multiple nodes get comma-separated node names and IPs."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1, num_gpus=4),
                    ComputeNode(name="n3", ip_address="10.0.0.3", index=2, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="distributed",
                    script=["echo distributed"],
                    resources=ResourcesConfig(nodes=NodeResourceConfig(count=2)),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    distributed = tg.get_task("distributed")
    
    assert distributed.envs["SFLOW_TASK_ASSIGNED_NODE_NAMES"] == "n1,n2"
    assert distributed.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] == "10.0.0.1,10.0.0.2"


def test_task_assigned_node_env_vars_with_replicas():
    """Each replica gets its own assigned node env vars."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="worker",
                    script=["echo worker"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(
                        nodes=NodeResourceConfig(count=1),
                        gpus=GpuResourceConfig(count=4),
                    ),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    
    worker_0 = tg.get_task("worker_0")
    worker_1 = tg.get_task("worker_1")
    
    # Each replica should have its own assigned node
    assert worker_0.envs["SFLOW_TASK_ASSIGNED_NODE_NAMES"] == "n1"
    assert worker_0.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] == "10.0.0.1"
    
    assert worker_1.envs["SFLOW_TASK_ASSIGNED_NODE_NAMES"] == "n2"
    assert worker_1.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] == "10.0.0.2"


def test_task_assigned_node_env_vars_multi_node_replicas():
    """Replicas with multi-node assignments get correct comma-separated values."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1, num_gpus=4),
                    ComputeNode(name="n3", ip_address="10.0.0.3", index=2, num_gpus=4),
                    ComputeNode(name="n4", ip_address="10.0.0.4", index=3, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="distributed",
                    script=["echo distributed"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(nodes=NodeResourceConfig(count=2)),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    
    d0 = tg.get_task("distributed_0")
    d1 = tg.get_task("distributed_1")
    
    # Each replica gets 2 nodes
    assert d0.envs["SFLOW_TASK_ASSIGNED_NODE_NAMES"] == "n1,n2"
    assert d0.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] == "10.0.0.1,10.0.0.2"
    
    assert d1.envs["SFLOW_TASK_ASSIGNED_NODE_NAMES"] == "n3,n4"
    assert d1.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] == "10.0.0.3,10.0.0.4"
