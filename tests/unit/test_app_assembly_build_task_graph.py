# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from sflow.app.assembly import build_task_graph, resolve_artifacts
from collections.abc import Sequence
from sflow.core.operator import Operator
from sflow.config.schema import (
    GpuResourceConfig,
    NodeResourceConfig,
    ReplicaConfig,
    ResourcesConfig,
    SflowConfig,
    TaskConfig,
    WorkflowConfig,
)
from sflow.core.backend import Allocation, Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.variable import Variable, VariableType
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.plugins.operators.srun import SrunOperator, SrunOperatorConfig
from sflow.plugins.probes import HttpPostProbe, TcpPortProbe


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
        # Behave like a slurm backend by default (srun), unless explicitly local.
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


def test_build_task_graph_creates_nodes_edges_and_default_operator():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[ComputeNode(name="n1", ip_address="10.0.0.1", index=0)],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        backends=[
            {
                "name": "b1",
                "type": "slurm",
                "default": True,
                "account": "acct",
                "partition": "batch",
                "time": "00:10:00",
                "nodes": 1,
                "gpus_per_node": 1,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(name="t1", script=["echo 1"]),
                TaskConfig(name="t2", script=["echo 2"], depends_on=["t1"]),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    assert set(tg.dag.nodes.keys()) == {"t1", "t2"}
    assert tg.dag.get_dependencies("t2") == ["t1"]

    t1 = tg.get_task("t1")
    assert t1.operator.config.type == "srun"
    assert t1.operator.config.job_id == "111"
    assert t1.operator.config.nodelist == ["n1"]


def test_build_task_graph_srun_operator_resolves_container_image_and_extra_args():
    state = _state()
    state.variables = {
        "IMG": Variable(name="IMG", value="img:1", type=VariableType.STRING),
    }
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="222",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        operators=[
            {
                "name": "ctr",
                "type": "srun",
                "container_image": "${{ variables.IMG }}",
                "container_writable": True,
                "extra_args": ["--foo=${{ IMG }}"],
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", operator="ctr", script=["echo hi"])],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    assert t1.operator.config.type == "srun"
    assert t1.operator.config.job_id == "222"
    assert t1.operator.config.nodelist == ["n1", "n2"]
    assert t1.operator.config.container_image == "img:1"
    assert t1.operator.config.extra_args == ["--foo=img:1"]


def test_build_task_graph_operator_override_allows_task_level_srun_overrides_ntasks():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="222",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        operators=[
            {
                "name": "ctr",
                "type": "srun",
                "container_image": "img:1",
                "container_writable": True,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    operator={"name": "ctr", "ntasks": 4, "ntasks_per_node": 2},
                    script=["echo hi"],
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    assert t1.operator.config.type == "srun"
    assert t1.operator.config.ntasks == 4
    assert t1.operator.config.ntasks_per_node == 2


def test_build_task_graph_srun_operator_resolves_container_image_from_artifact_path(
    tmp_path,
):
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="222",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    img = tmp_path / "img.sqsh"
    img.write_text("x")

    config = SflowConfig(
        version="0.1",
        artifacts=[{"name": "IMG", "uri": f"fs://{img}"}],
        operators=[
            {
                "name": "ctr",
                "type": "srun",
                "container_image": "${{ artifacts.IMG.path }}",
                "container_writable": True,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", operator="ctr", script=["echo hi"])],
        ),
    )

    state = resolve_artifacts(config, state, workspace_dir=tmp_path, materialize=False)
    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    assert t1.operator.config.type == "srun"
    assert t1.operator.config.container_image == str(img)


def test_build_task_graph_srun_operator_auto_adds_container_mounts_for_local_artifacts(
    tmp_path,
):
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="222",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("x")
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    config = SflowConfig(
        version="0.1",
        artifacts=[
            {"name": "CFG", "uri": "file://cfg.yaml"},
            {"name": "MODEL_DIR", "uri": f"fs://{model_dir}"},
            # `.sqsh` should be skipped for auto-mount.
            {"name": "IMG_SQSH", "uri": "file://image.sqsh"},
        ],
        operators=[
            {
                "name": "ctr",
                "type": "srun",
                "container_image": "img:1",
                "container_writable": True,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", operator="ctr", script=["echo hi"])],
        ),
    )

    state = resolve_artifacts(config, state, workspace_dir=tmp_path, materialize=False)
    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    mounts = list(t1.operator.config.container_mounts or [])

    # file:// artifacts mount their parent dir so the file path stays valid inside container.
    assert f"{tmp_path}:{tmp_path}:rw" in mounts
    # fs:// dir artifacts mount the directory itself (same path inside container).
    assert f"{model_dir}:{model_dir}:rw" in mounts
    # `.sqsh` artifacts should not contribute mounts.
    assert not any("image.sqsh" in m for m in mounts)


def test_build_task_graph_defaults_to_bash_when_backend_is_local():
    state = _state()
    state.backends = {"local": _FakeBackend("local", allocation=None)}
    state.default_backend = state.backends["local"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    assert t1.operator.config.type == "bash"


def test_build_task_graph_parallel_replicas_expand_and_depends_on_all_replicas():
    state = _state()
    state.backends = {"local": _FakeBackend("local", allocation=None)}
    state.default_backend = state.backends["local"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo 1"],
                    replicas=ReplicaConfig(count=3, policy="parallel"),
                ),
                TaskConfig(name="t2", script=["echo 2"], depends_on=["t1"]),
            ],
        ),
    )

    tg = build_task_graph(config, state)

    assert set(tg.dag.nodes.keys()) == {"t1_0", "t1_1", "t1_2", "t2"}
    assert set(tg.dag.get_dependencies("t2")) == {"t1_0", "t1_1", "t1_2"}

    assert tg.get_task("t1_0").envs["SFLOW_REPLICA_INDEX"] == "0"
    assert tg.get_task("t1_1").envs["SFLOW_REPLICA_INDEX"] == "1"
    assert tg.get_task("t1_2").envs["SFLOW_REPLICA_INDEX"] == "2"


def test_build_task_graph_sequential_replicas_chain_and_depends_on_last_replica():
    state = _state()
    state.backends = {"local": _FakeBackend("local", allocation=None)}
    state.default_backend = state.backends["local"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo 1"],
                    replicas=ReplicaConfig(count=3, policy="sequential"),
                ),
                TaskConfig(name="t2", script=["echo 2"], depends_on=["t1"]),
            ],
        ),
    )

    tg = build_task_graph(config, state)

    assert set(tg.dag.nodes.keys()) == {"t1_0", "t1_1", "t1_2", "t2"}
    # Sequential policy creates an explicit chain.
    assert set(tg.dag.get_dependencies("t1_1")) == {"t1_0"}
    assert set(tg.dag.get_dependencies("t1_2")) == {"t1_1"}
    # Downstream depends only on last replica.
    assert set(tg.dag.get_dependencies("t2")) == {"t1_2"}


def test_build_task_graph_replica_policy_is_resolved_from_expression():
    state = _state()
    state.backends = {"local": _FakeBackend("local", allocation=None)}
    state.default_backend = state.backends["local"]
    state.variables = {
        "REPLICA_POLICY": Variable(name="REPLICA_POLICY", value="sequential")
    }

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo 1"],
                    replicas=ReplicaConfig(
                        count=3, policy="${{ variables.REPLICA_POLICY }}"
                    ),
                ),
                TaskConfig(name="t2", script=["echo 2"], depends_on=["t1"]),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    assert set(tg.dag.nodes.keys()) == {"t1_0", "t1_1", "t1_2", "t2"}
    # Resolved "sequential" policy creates an explicit chain.
    assert set(tg.dag.get_dependencies("t1_1")) == {"t1_0"}
    assert set(tg.dag.get_dependencies("t1_2")) == {"t1_1"}
    # Downstream depends only on last replica.
    assert set(tg.dag.get_dependencies("t2")) == {"t1_2"}


def test_build_task_graph_sequential_replicas_only_first_replica_depends_on_upstream():
    state = _state()
    state.backends = {"local": _FakeBackend("local", allocation=None)}
    state.default_backend = state.backends["local"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(name="up", script=["echo up"]),
                TaskConfig(
                    name="seq",
                    script=["echo seq"],
                    replicas=ReplicaConfig(count=3, policy="sequential"),
                    depends_on=["up"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    assert set(tg.dag.nodes.keys()) == {"up", "seq_0", "seq_1", "seq_2"}

    # Only first replica depends on upstream; later replicas depend on the chain only.
    assert set(tg.dag.get_dependencies("seq_0")) == {"up"}
    assert set(tg.dag.get_dependencies("seq_1")) == {"seq_0"}
    assert set(tg.dag.get_dependencies("seq_2")) == {"seq_1"}


def test_build_task_graph_tcp_probe_defaults_to_assigned_node_ip_for_slurm_backend():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="111",
                nodes=[ComputeNode(name="n1", ip_address="10.0.0.1", index=0)],
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
                    name="svc",
                    script=["echo hi"],
                    resources=ResourcesConfig(nodes=NodeResourceConfig(indices=[0])),
                    probes={"readiness": {"tcp_port": {"port": 8000}}},
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    svc = tg.get_task("svc")
    assert len(svc.probes) == 1
    p = svc.probes[0]
    assert isinstance(p, TcpPortProbe)
    assert p._host == "10.0.0.1"


def test_build_task_graph_replica_sweep_uses_variable_domain_and_injects_envs():
    state = _state()
    state.variables = {
        "CONCURRENCY": Variable(
            name="CONCURRENCY",
            value=1,
            type=VariableType.INTEGER,
            domain=[1, 2, 4],
        )
    }
    state.backends = {"local": _FakeBackend("local", allocation=None)}
    state.default_backend = state.backends["local"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="benchmark",
                    script=["echo ${CONCURRENCY}"],
                    replicas=ReplicaConfig(
                        variables=["CONCURRENCY"], policy="sequential"
                    ),
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    # Replica names now use variable values: benchmark_1, benchmark_2, benchmark_4
    assert set(tg.dag.nodes.keys()) == {"benchmark_1", "benchmark_2", "benchmark_4"}

    # Sequential chain exists.
    assert set(tg.dag.get_dependencies("benchmark_2")) == {"benchmark_1"}
    assert set(tg.dag.get_dependencies("benchmark_4")) == {"benchmark_2"}

    # Per-replica envs include sweep variable and replica index.
    assert tg.get_task("benchmark_1").envs["SFLOW_REPLICA_INDEX"] == "0"
    assert tg.get_task("benchmark_1").envs["CONCURRENCY"] == "1"
    assert tg.get_task("benchmark_2").envs["SFLOW_REPLICA_INDEX"] == "1"
    assert tg.get_task("benchmark_2").envs["CONCURRENCY"] == "2"
    assert tg.get_task("benchmark_4").envs["SFLOW_REPLICA_INDEX"] == "2"
    assert tg.get_task("benchmark_4").envs["CONCURRENCY"] == "4"


def test_build_task_graph_resources_nodes_indices_selects_subset_of_allocation_nodes():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="333",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1),
                    ComputeNode(name="n3", ip_address="10.0.0.3", index=2),
                    ComputeNode(name="n4", ip_address="10.0.0.4", index=3),
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
                    script=["echo 1"],
                    resources=ResourcesConfig(nodes=NodeResourceConfig(indices=[1, 3])),
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    assert t1.operator.config.type == "srun"
    assert t1.operator.config.nodelist == ["n2", "n4"]
    assert t1.operator.config.nodes == 2


def test_build_task_graph_resources_nodes_count_compact_allocation_for_parallel_replicas():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="444",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1),
                    ComputeNode(name="n3", ip_address="10.0.0.3", index=2),
                    ComputeNode(name="n4", ip_address="10.0.0.4", index=3),
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
                    script=["echo 1"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(nodes=NodeResourceConfig(count=2)),
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t10 = tg.get_task("t1_0")
    t11 = tg.get_task("t1_1")
    assert t10.operator.config.type == "srun"
    assert t11.operator.config.type == "srun"
    assert t10.operator.config.nodelist == ["n1", "n2"]
    assert t11.operator.config.nodelist == ["n3", "n4"]


def test_build_task_graph_resources_gpus_count_sets_cuda_visible_devices_with_offset():
    state = _state()
    state.backends = {"local": _FakeBackend("local", allocation=None)}
    state.default_backend = state.backends["local"]

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo 1"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    assert tg.get_task("t1_0").envs["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert tg.get_task("t1_1").envs["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_build_task_graph_resources_gpus_respects_compute_node_num_gpus():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="555",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    ok = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo 1"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                )
            ],
        ),
    )
    tg = build_task_graph(ok, state)
    assert tg.get_task("t1_0").envs["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert tg.get_task("t1_1").envs["CUDA_VISIBLE_DEVICES"] == "2,3"

    bad = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=["echo 1"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=3)),
                )
            ],
        ),
    )
    # With planning-time GPU allocation, the second replica may fail either due to exceeding
    # node capacity or because no GPUs remain available on the node. Match either message.
    with pytest.raises(
        ValueError,
        match=r"(has only 4 GPUs|only 1 GPUs remain available|remain available)",
    ):
        build_task_graph(bad, state)


def test_build_task_graph_resources_gpus_count_requires_more_nodes_when_multiple_but_allocation_is_too_small():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="777",
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
                    name="t1",
                    script=["echo 1"],
                    # 8 is a multiple of per-node capacity (4), so auto-expansion would require 2 nodes,
                    # but the allocation contains only 1 node -> fail fast.
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=8)),
                )
            ],
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"requests 8 GPUs which requires 2 nodes.*allocation has only 1 nodes",
    ):
        build_task_graph(config, state)


def test_build_task_graph_resources_gpus_count_auto_expands_nodes_when_total_exceeds_single_node():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="888",
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
                    script=["echo 1"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=8)),
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    assert t1.operator.config.type == "srun"
    assert t1.operator.config.nodelist == ["n1", "n2"]
    assert t1.operator.config.nodes == 2
    # Multi-node request exposes a per-node slice (env is evaluated per node).
    assert t1.envs["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_build_task_graph_resources_gpus_count_requires_multiple_of_per_node_capacity_for_auto_expand():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="889",
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
                    script=["echo 1"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=6)),
                )
            ],
        ),
    )

    with pytest.raises(ValueError, match=r"multiple of per-node GPU capacity"):
        build_task_graph(config, state)


def test_build_task_graph_resources_with_multiple_tasks_pins_distinct_nodes_and_sets_gpu_envs():
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="666",
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
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(
                        gpus=GpuResourceConfig(count=2),
                    ),
                ),
                TaskConfig(
                    name="t2",
                    script=["echo t2"],
                    replicas=ReplicaConfig(count=4, policy="parallel"),
                    resources=ResourcesConfig(
                        gpus=GpuResourceConfig(count=1),
                    ),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    assert set(tg.dag.nodes.keys()) == {"t1_0", "t1_1", "t2_0", "t2_1", "t2_2", "t2_3"}

    t10 = tg.get_task("t1_0")
    t11 = tg.get_task("t1_1")
    t20 = tg.get_task("t2_0")
    t21 = tg.get_task("t2_1")
    t22 = tg.get_task("t2_2")
    t23 = tg.get_task("t2_3")

    assert t10.operator.config.nodelist == ["n1"]
    assert t11.operator.config.nodelist == ["n1"]
    assert t20.operator.config.nodelist == ["n2"]
    assert t21.operator.config.nodelist == ["n2"]
    assert t22.operator.config.nodelist == ["n2"]
    assert t23.operator.config.nodelist == ["n2"]

    assert t10.envs["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert t11.envs["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert t20.envs["CUDA_VISIBLE_DEVICES"] == "0"
    assert t21.envs["CUDA_VISIBLE_DEVICES"] == "1"
    assert t22.envs["CUDA_VISIBLE_DEVICES"] == "2"
    assert t23.envs["CUDA_VISIBLE_DEVICES"] == "3"


def test_build_task_graph_gpu_packing_allows_multiple_base_tasks_to_share_remaining_gpus_on_same_node():
    """
    Ensure GPU allocation is treated as a global pool across workflow tasks:
    if a task only consumes part of a node's GPU capacity, later tasks can use the remainder.
    """
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="667",
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
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                ),
                TaskConfig(
                    name="t2",
                    script=["echo t2"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=1)),
                ),
                TaskConfig(
                    name="t3",
                    script=["echo t3"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=4)),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)

    t1 = tg.get_task("t1")
    t20 = tg.get_task("t2_0")
    t21 = tg.get_task("t2_1")
    t3 = tg.get_task("t3")

    # t1 consumes 2 GPUs on n1 -> leaves 2 GPUs (2,3) available on n1.
    assert t1.operator.config.nodelist == ["n1"]
    assert t1.envs["CUDA_VISIBLE_DEVICES"] == "0,1"

    # t2 replicas should use the remaining GPUs on n1.
    assert t20.operator.config.nodelist == ["n1"]
    assert t21.operator.config.nodelist == ["n1"]
    assert t20.envs["CUDA_VISIBLE_DEVICES"] == "2"
    assert t21.envs["CUDA_VISIBLE_DEVICES"] == "3"

    # t3 requests a full node (4 GPUs) -> should land on n2.
    assert t3.operator.config.nodelist == ["n2"]
    assert t3.envs["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_build_task_graph_multi_node_gpu_request_reserves_gpus_so_later_tasks_use_other_nodes():
    """
    Regression: multi-node GPU requests must advance the global planning cursor (`gpu_next`)
    so later GPU-packed tasks don't get placed onto already-fully-consumed nodes.
    """
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="668",
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
                # Needs 8 GPUs total -> 2 nodes (n1, n2) with 4 GPUs each.
                TaskConfig(
                    name="prefill",
                    script=["echo prefill"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=8)),
                ),
                # Two decode replicas needing 2 GPUs each should land on n3 after prefill
                # fully consumes n1+n2 (via reservation).
                TaskConfig(
                    name="decode",
                    script=["echo decode"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=2)),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    prefill = tg.get_task("prefill")
    d0 = tg.get_task("decode_0")
    d1 = tg.get_task("decode_1")

    assert prefill.operator.config.nodelist == ["n1", "n2"]
    assert prefill.envs["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

    assert d0.operator.config.nodelist == ["n3"]
    assert d1.operator.config.nodelist == ["n3"]
    assert d0.envs["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert d1.envs["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_build_task_graph_multi_node_decode_avoids_node_fully_consumed_by_sequential_prefill_replicas():
    """
    Match the reported dry-run scenario:
    - 3 nodes with 4 GPUs each
    - prefill replicas (sequential) consume all 4 GPUs on node0 (1 GPU per replica, 4 replicas)
    - decode needs 8 GPUs total -> should use node1 + node2 (skip node0)
    """
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="669",
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
                    name="prefill",
                    script=["echo prefill"],
                    replicas=ReplicaConfig(count=4, policy="sequential"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=1)),
                ),
                TaskConfig(
                    name="decode",
                    script=["echo decode"],
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=8)),
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)

    # Prefill fully consumes n1.
    assert tg.get_task("prefill_0").operator.config.nodelist == ["n1"]
    assert tg.get_task("prefill_1").operator.config.nodelist == ["n1"]
    assert tg.get_task("prefill_2").operator.config.nodelist == ["n1"]
    assert tg.get_task("prefill_3").operator.config.nodelist == ["n1"]
    assert tg.get_task("prefill_0").envs["CUDA_VISIBLE_DEVICES"] == "0"
    assert tg.get_task("prefill_1").envs["CUDA_VISIBLE_DEVICES"] == "1"
    assert tg.get_task("prefill_2").envs["CUDA_VISIBLE_DEVICES"] == "2"
    assert tg.get_task("prefill_3").envs["CUDA_VISIBLE_DEVICES"] == "3"

    # Decode should avoid n1 and use n2+n3.
    decode = tg.get_task("decode")
    assert decode.operator.config.nodelist == ["n2", "n3"]
    assert decode.envs["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_build_task_graph_replica_tasks_have_independent_container_mounts(tmp_path):
    """
    Each replica task should have its own independent container_mounts list.
    Modifying one replica's mounts should not affect other replicas.
    """
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="333",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    config = SflowConfig(
        version="0.1",
        artifacts=[
            {"name": "MODEL_DIR", "uri": f"fs://{model_dir}"},
        ],
        operators=[
            {
                "name": "ctr",
                "type": "srun",
                "container_image": "img:1",
                "container_writable": True,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="server",
                    operator="ctr",
                    script=["echo server"],
                    replicas=ReplicaConfig(count=3, policy="parallel"),
                    resources=ResourcesConfig(gpus=GpuResourceConfig(count=1)),
                ),
            ],
        ),
    )

    state = resolve_artifacts(config, state, workspace_dir=tmp_path, materialize=False)
    tg = build_task_graph(config, state)

    # Get all replica tasks
    server_0 = tg.get_task("server_0")
    server_1 = tg.get_task("server_1")
    server_2 = tg.get_task("server_2")

    # Each replica should have the artifact mount
    assert f"{model_dir}:{model_dir}:rw" in server_0.operator.config.container_mounts
    assert f"{model_dir}:{model_dir}:rw" in server_1.operator.config.container_mounts
    assert f"{model_dir}:{model_dir}:rw" in server_2.operator.config.container_mounts

    # CRITICAL: Verify that each replica has an INDEPENDENT container_mounts list.
    # Modifying one should not affect others.
    original_mounts_0 = list(server_0.operator.config.container_mounts)
    original_mounts_1 = list(server_1.operator.config.container_mounts)
    original_mounts_2 = list(server_2.operator.config.container_mounts)

    # The lists should be equal in content
    assert original_mounts_0 == original_mounts_1 == original_mounts_2

    # But they should be different list objects (independent)
    assert server_0.operator.config.container_mounts is not server_1.operator.config.container_mounts
    assert server_1.operator.config.container_mounts is not server_2.operator.config.container_mounts

    # Now modify one replica's mounts and verify others are unaffected
    server_0.operator.config.container_mounts.append("/extra/mount:/extra/mount:rw")

    # server_0 should have the new mount
    assert "/extra/mount:/extra/mount:rw" in server_0.operator.config.container_mounts

    # server_1 and server_2 should NOT have the new mount
    assert "/extra/mount:/extra/mount:rw" not in server_1.operator.config.container_mounts
    assert "/extra/mount:/extra/mount:rw" not in server_2.operator.config.container_mounts


def test_build_task_graph_srun_operator_auto_adds_container_mounts_when_using_container_name(
    tmp_path,
):
    """
    Auto-mount should also work when using container_name instead of container_image.
    This is common with Pyxis when reusing a cached container.
    """
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="444",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    config = SflowConfig(
        version="0.1",
        artifacts=[
            {"name": "MODEL_DIR", "uri": f"fs://{model_dir}"},
        ],
        operators=[
            {
                "name": "ctr",
                "type": "srun",
                # Using container_name instead of container_image
                "container_name": "my_container",
                "container_writable": True,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", operator="ctr", script=["echo hi"])],
        ),
    )

    state = resolve_artifacts(config, state, workspace_dir=tmp_path, materialize=False)
    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    mounts = list(t1.operator.config.container_mounts or [])

    # fs:// dir artifacts should be auto-mounted even when using container_name
    assert f"{model_dir}:{model_dir}:rw" in mounts


def test_build_task_graph_variable_sweep_replicas_have_artifact_mounts(tmp_path):
    """
    When replicas are created via variable sweep (replicas.variables),
    each replica should have artifact mounts.
    """
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="555",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]
    # Add a variable with domain for sweep
    state.variables = {
        "CONCURRENCY": Variable(
            name="CONCURRENCY",
            value=64,
            type=VariableType.INTEGER,
            domain=[64, 128, 256],
        ),
    }

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    config = SflowConfig(
        version="0.1",
        artifacts=[
            {"name": "MODEL_DIR", "uri": f"fs://{model_dir}"},
        ],
        operators=[
            {
                "name": "ctr",
                "type": "srun",
                "container_name": "bench_container",
                "container_writable": True,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="benchmark",
                    operator="ctr",
                    script=["echo $CONCURRENCY"],
                    replicas=ReplicaConfig(
                        variables=["CONCURRENCY"],
                        policy="sequential",
                    ),
                ),
            ],
        ),
    )

    state = resolve_artifacts(config, state, workspace_dir=tmp_path, materialize=False)
    tg = build_task_graph(config, state)

    # Get all variable-sweep replicas
    bench_64 = tg.get_task("benchmark_64")
    bench_128 = tg.get_task("benchmark_128")
    bench_256 = tg.get_task("benchmark_256")

    # Each replica should have the artifact mount
    assert f"{model_dir}:{model_dir}:rw" in bench_64.operator.config.container_mounts
    assert f"{model_dir}:{model_dir}:rw" in bench_128.operator.config.container_mounts
    assert f"{model_dir}:{model_dir}:rw" in bench_256.operator.config.container_mounts

    # Verify the sweep variable is set correctly in each replica's env
    assert bench_64.envs.get("CONCURRENCY") == "64"
    assert bench_128.envs.get("CONCURRENCY") == "128"
    assert bench_256.envs.get("CONCURRENCY") == "256"


def test_build_task_graph_srun_uses_all_nodes_by_default():
    """
    When task does not explicitly define resources, it should use ALL nodes
    from the backend allocation.
    """
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="666",
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
        operators=[
            {
                "name": "multi_node_op",
                "type": "srun",
                "ntasks": 8,
                "ntasks_per_node": 4,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="multi_task",
                    operator="multi_node_op",
                    script=["echo hi"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("multi_task")

    # Without explicit resources, task should use ALL nodes from allocation
    assert t1.operator.config.nodes == 4
    assert t1.operator.config.ntasks == 8
    assert t1.operator.config.ntasks_per_node == 4
    # nodelist should have all 4 nodes
    assert len(t1.operator.config.nodelist) == 4
    assert t1.operator.config.nodelist == ["n1", "n2", "n3", "n4"]


def test_build_task_graph_srun_all_nodes_with_string_values():
    """
    When ntasks and ntasks_per_node are strings (from expression resolution),
    and no explicit resources are set, task should use all allocation nodes.
    """
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="777",
                nodes=[
                    ComputeNode(name="node-0", ip_address="10.0.0.1", index=0, num_gpus=4),
                    ComputeNode(name="node-1", ip_address="10.0.0.2", index=1, num_gpus=4),
                    ComputeNode(name="node-2", ip_address="10.0.0.3", index=2, num_gpus=4),
                ],
            ),
        )
    }
    state.default_backend = state.backends["b1"]

    config = SflowConfig(
        version="0.1",
        operators=[
            {
                "name": "string_values_op",
                "type": "srun",
                "ntasks": "3",  # String, like from ${{ variables.SLURM_NODES }}
                "ntasks_per_node": "1",  # String
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="string_task",
                    operator="string_values_op",
                    script=["echo hi"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("string_task")

    # Without explicit resources, task should use ALL 3 nodes
    assert t1.operator.config.nodes == 3
    assert len(t1.operator.config.nodelist) == 3
    assert t1.operator.config.nodelist == ["node-0", "node-1", "node-2"]


def test_build_task_graph_srun_explicit_nodes_overrides_computed():
    """
    When operator has explicit nodes set, it should NOT be overridden
    by the ntasks/ntasks_per_node computation.
    """
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="777",
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
        operators=[
            {
                "name": "explicit_nodes_op",
                "type": "srun",
                "nodes": 3,  # Explicitly set
                "ntasks": 8,
                "ntasks_per_node": 4,
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="explicit_task",
                    operator="explicit_nodes_op",
                    script=["echo hi"],
                ),
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("explicit_task")

    # Explicit nodes=3 should be preserved, not overridden by 8/4=2
    assert t1.operator.config.nodes == 3


# ---------------------------------------------------------------------------
# resources.nodes.exclude tests
# ---------------------------------------------------------------------------


def test_build_task_graph_resources_nodes_exclude_single_int():
    """exclude: 0 removes the first node, task gets remaining nodes."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="exc1",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1),
                    ComputeNode(name="n3", ip_address="10.0.0.3", index=2),
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
                    script=["echo 1"],
                    resources=ResourcesConfig(nodes=NodeResourceConfig(exclude=0)),
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    assert t1.operator.config.nodelist == ["n2", "n3"]


def test_build_task_graph_resources_nodes_exclude_list():
    """exclude: [0, 2] removes first and third nodes."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="exc2",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1),
                    ComputeNode(name="n3", ip_address="10.0.0.3", index=2),
                    ComputeNode(name="n4", ip_address="10.0.0.4", index=3),
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
                    script=["echo 1"],
                    resources=ResourcesConfig(nodes=NodeResourceConfig(exclude=[0, 2])),
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    assert t1.operator.config.nodelist == ["n2", "n4"]


def test_build_task_graph_resources_nodes_exclude_with_count():
    """exclude + count: count operates on the filtered pool."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="exc3",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
                    ComputeNode(name="n2", ip_address="10.0.0.2", index=1),
                    ComputeNode(name="n3", ip_address="10.0.0.3", index=2),
                    ComputeNode(name="n4", ip_address="10.0.0.4", index=3),
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
                    script=["echo 1"],
                    resources=ResourcesConfig(
                        nodes=NodeResourceConfig(exclude=[0], count=2)
                    ),
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    # Pool after exclude: [n2, n3, n4], count=2 takes first 2
    assert t1.operator.config.nodelist == ["n2", "n3"]
    assert t1.operator.config.nodes == 2


def test_build_task_graph_resources_nodes_exclude_with_gpus():
    """exclude + gpus.count: GPU packing runs on filtered pool."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="exc4",
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
                    script=["echo 1"],
                    resources=ResourcesConfig(
                        nodes=NodeResourceConfig(exclude=[0]),
                        gpus=GpuResourceConfig(count=2),
                    ),
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    t1 = tg.get_task("t1")
    # Pool after exclude: only n2. GPU packing should place task on n2.
    assert t1.operator.config.nodelist == ["n2"]
    assert t1.envs.get("CUDA_VISIBLE_DEVICES") == "0,1"


def test_build_task_graph_resources_nodes_exclude_all_raises():
    """Excluding all nodes should raise an error."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="exc5",
                nodes=[
                    ComputeNode(name="n1", ip_address="10.0.0.1", index=0),
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
                    script=["echo 1"],
                    resources=ResourcesConfig(
                        nodes=NodeResourceConfig(exclude=[0]),
                    ),
                )
            ],
        ),
    )

    with pytest.raises(ValueError, match="removed all nodes"):
        build_task_graph(config, state)


# ---------------------------------------------------------------------------
# HTTP probe replica deduplication
# ---------------------------------------------------------------------------


def _state_with_slurm_backend() -> SflowState:
    """Convenience: SflowState with a single slurm-like backend and one node."""
    state = _state()
    state.backends = {
        "b1": _FakeBackend(
            "b1",
            allocation=Allocation(
                allocation_id="probe-dedup",
                nodes=[ComputeNode(name="n1", ip_address="10.0.0.1", index=0)],
            ),
        )
    }
    state.default_backend = state.backends["b1"]
    return state


def test_http_probe_skipped_on_non_first_replica_when_no_sweep_var_referenced():
    """HTTP readiness probe that doesn't reference sweep vars should only appear on
    the first replica — non-first replicas should have no probes but the first
    replica should list them as readiness_followers."""
    state = _state_with_slurm_backend()
    state.variables = {
        "CONCURRENCY": Variable(
            name="CONCURRENCY", value=4, type=VariableType.INTEGER, domain=[4, 8]
        ),
    }

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="bench",
                    script=["echo run"],
                    replicas=ReplicaConfig(
                        variables=["CONCURRENCY"], policy="sequential"
                    ),
                    probes={
                        "readiness": {
                            "http_post": {
                                "url": "http://10.0.0.1:8888/v1/chat/completions",
                                "body": '{"model": "m", "messages": []}',
                            },
                            "timeout": 60,
                            "interval": 5,
                        }
                    },
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    first = tg.get_task("bench_4")
    second = tg.get_task("bench_8")

    assert len(first.probes) == 1
    assert isinstance(first.probes[0], HttpPostProbe)
    assert len(second.probes) == 0
    assert first.readiness_followers == ["bench_8"]


def test_http_probe_kept_on_all_replicas_when_sweep_var_referenced():
    """HTTP readiness probe that references a sweep variable should be present on
    every replica."""
    state = _state_with_slurm_backend()
    state.variables = {
        "PORT": Variable(
            name="PORT", value=8000, type=VariableType.INTEGER, domain=[8000, 9000]
        ),
    }

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="svc",
                    script=["echo run"],
                    replicas=ReplicaConfig(
                        variables=["PORT"], policy="parallel"
                    ),
                    probes={
                        "readiness": {
                            "http_post": {
                                "url": "http://10.0.0.1:${{ variables.PORT }}/health",
                                "body": '{"check": true}',
                            },
                            "timeout": 30,
                            "interval": 5,
                        }
                    },
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    first = tg.get_task("svc_8000")
    second = tg.get_task("svc_9000")

    assert len(first.probes) == 1
    assert len(second.probes) == 1
    assert isinstance(first.probes[0], HttpPostProbe)
    assert isinstance(second.probes[0], HttpPostProbe)


def test_tcp_probe_always_per_replica():
    """TCP probes should never be deduplicated — they inherently differ per replica
    (different assigned hosts)."""
    state = _state_with_slurm_backend()
    state.variables = {
        "CONCURRENCY": Variable(
            name="CONCURRENCY", value=4, type=VariableType.INTEGER, domain=[4, 8]
        ),
    }

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="svc",
                    script=["echo run"],
                    replicas=ReplicaConfig(
                        variables=["CONCURRENCY"], policy="parallel"
                    ),
                    probes={
                        "readiness": {
                            "tcp_port": {"port": 8888},
                            "timeout": 30,
                            "interval": 5,
                        }
                    },
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    first = tg.get_task("svc_4")
    second = tg.get_task("svc_8")

    assert len(first.probes) == 1
    assert len(second.probes) == 1
    assert isinstance(first.probes[0], TcpPortProbe)
    assert isinstance(second.probes[0], TcpPortProbe)


def test_http_probe_followers_multiple_replicas():
    """When 3+ replicas share a deduplicated HTTP probe, all non-first replicas
    should appear in the first replica's readiness_followers."""
    state = _state_with_slurm_backend()
    state.variables = {
        "CONCURRENCY": Variable(
            name="CONCURRENCY",
            value=4,
            type=VariableType.INTEGER,
            domain=[4, 8, 16],
        ),
    }

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="bench",
                    script=["echo run"],
                    replicas=ReplicaConfig(
                        variables=["CONCURRENCY"], policy="sequential"
                    ),
                    probes={
                        "readiness": {
                            "http_post": {
                                "url": "http://10.0.0.1:8888/health",
                                "body": "{}",
                            },
                            "timeout": 60,
                            "interval": 5,
                        }
                    },
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    first = tg.get_task("bench_4")
    second = tg.get_task("bench_8")
    third = tg.get_task("bench_16")

    assert len(first.probes) == 1
    assert len(second.probes) == 0
    assert len(third.probes) == 0
    assert first.readiness_followers == ["bench_8", "bench_16"]
    assert second.readiness_followers == []
    assert third.readiness_followers == []


def test_failure_http_probe_followers():
    """Deduplicated failure HTTP probes should populate failure_followers."""
    state = _state_with_slurm_backend()
    state.variables = {
        "CONCURRENCY": Variable(
            name="CONCURRENCY", value=4, type=VariableType.INTEGER, domain=[4, 8]
        ),
    }

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="bench",
                    script=["echo run"],
                    replicas=ReplicaConfig(
                        variables=["CONCURRENCY"], policy="sequential"
                    ),
                    probes={
                        "failure": {
                            "http_get": {
                                "url": "http://10.0.0.1:8888/health",
                            },
                            "timeout": 60,
                            "interval": 5,
                        }
                    },
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    first = tg.get_task("bench_4")
    second = tg.get_task("bench_8")

    assert len(first.probes) == 1
    assert len(second.probes) == 0
    assert first.failure_followers == ["bench_8"]
    assert first.readiness_followers == []


def test_http_probe_kept_when_referencing_sflow_replica_index():
    """HTTP probe referencing SFLOW_REPLICA_INDEX should NOT be skipped on any
    replica, since each replica has a different index value."""
    state = _state_with_slurm_backend()

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="svc",
                    script=["echo run"],
                    replicas=ReplicaConfig(count=3, policy="parallel"),
                    probes={
                        "readiness": {
                            "http_get": {
                                "url": "http://10.0.0.1:${SFLOW_REPLICA_INDEX}/health",
                            },
                            "timeout": 30,
                            "interval": 5,
                        }
                    },
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    for i in range(3):
        task = tg.get_task(f"svc_{i}")
        assert len(task.probes) == 1, (
            f"svc_{i} should have its own probe since URL references SFLOW_REPLICA_INDEX"
        )
        assert task.readiness_followers == []


def test_http_probe_skipped_when_no_replica_var_referenced():
    """HTTP probe that doesn't reference any per-replica variable (neither sweep
    vars nor SFLOW_REPLICA_INDEX) should be skipped on non-first replicas."""
    state = _state_with_slurm_backend()

    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="svc",
                    script=["echo run"],
                    replicas=ReplicaConfig(count=2, policy="parallel"),
                    probes={
                        "readiness": {
                            "http_post": {
                                "url": "http://10.0.0.1:8888/health",
                                "body": "{}",
                            },
                            "timeout": 30,
                            "interval": 5,
                        }
                    },
                )
            ],
        ),
    )

    tg = build_task_graph(config, state)
    first = tg.get_task("svc_0")
    second = tg.get_task("svc_1")

    assert len(first.probes) == 1
    assert len(second.probes) == 0
    assert first.readiness_followers == ["svc_1"]
