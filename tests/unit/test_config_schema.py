# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from sflow.config.schema import (
    SflowConfig,
    WorkflowConfig,
    TaskConfig,
    VariableConfig,
    ArtifactConfig,
    BackendConfig,
    ReplicaConfig,
    ReplicaPolicy,
    ProbesConfig,
    ProbeConfig,
    TcpPortProbeConfig,
    HttpProbeConfig,
    LogWatchProbeConfig,
    RetryConfig,
    ResourcesConfig,
    NodeResourceConfig,
    GpuResourceConfig,
    OutputConfig,
    OutputMetricConfig,
)


class TestSflowConfigSchema:
    """
    Tests for sflow.config.schema based on requirements from SRD/SADD.
    """

    def test_full_valid_configuration(self):
        """
        REQ-1.1: The system shall accept a YAML-based configuration file defining top-level variables,
        artifacts, backends, operators, and the workflow (tasks/DAG).

        This test uses a structure similar to Appendix A of the SRD.
        """
        config_data = {
            "version": "0.1",
            "variables": {
                "SLURM_PARTITION": {"description": "SLURM partition", "value": "debug"},
                "GPUS_PER_NODE": {"description": "GPUs per node", "value": 4},
                "CONCURRENCY": {"value": 16, "domain": [16, 32]},
            },
            "artifacts": [
                {"name": "MODEL_PATH", "uri": "fs:///data/model"},
                {
                    "name": "INLINE_CONFIG",
                    "uri": "file://config.yaml",
                    "content": "batch_size: 32",
                },
            ],
            "backends": [
                {
                    "name": "slurm_cluster",
                    "type": "slurm",
                    "default": True,
                    "account": "test_account",
                    "partition": "debug",
                    "time": "10:00",
                    "nodes": 1,
                    "gpus_per_node": "${{ variables.GPUS_PER_NODE }}",
                }
            ],
            "operators": [
                {
                    "name": "my_container",
                    "type": "srun",
                    "container_image": "docker://alpine:latest",
                    "container_writable": True,
                }
            ],
            "workflow": {
                "name": "test_workflow",
                "timeout": "1h",
                "variables": {"WORKFLOW_VAR": {"value": 123}},
                "tasks": [
                    # Dependency target must exist for schema validation
                    {
                        "name": "other_task",
                        "script": ["echo other"],
                    },
                    {
                        "name": "task1",
                        "script": ["echo hello"],
                        "operator": "my_container",
                        "resources": {"nodes": {"count": 1}, "gpus": {"count": 1}},
                        "probes": {"readiness": {"tcp_port": {"port": 8080}}},
                        "replicas": {"count": 2, "policy": "parallel"},
                        "depends_on": ["other_task"],
                    },
                ],
            },
        }

        config = SflowConfig(**config_data)
        assert config.version == "0.1"
        assert config.variables[0].value == "debug"
        assert config.workflow.tasks[1].name == "task1"
        assert config.workflow.tasks[1].depends_on == ["other_task"]
        assert config.backends[0].default is True

    def test_variable_config(self):
        """
        REQ-1.3: Variable System. Support strongly typed variables.
        """
        # Test basic value
        v = VariableConfig(name="V", value="test")
        assert v.value == "test"

        # Test with domain
        v = VariableConfig(name="V2", value=10, domain=[10, 20, 30])
        assert v.value == 10
        assert v.domain == [10, 20, 30]

        # Test with description
        v = VariableConfig(name="V3", value=1.5, description="A float")
        assert v.value == 1.5

    def test_artifact_config(self):
        """
        REQ-1.5, REQ-1.6, REQ-1.8: Artifact Management.
        """
        # Remote artifact
        a = ArtifactConfig(name="HF_MODEL", uri="huggingface://model")
        assert a.uri == "huggingface://model"

        # Inline artifact with content
        a = ArtifactConfig(name="INLINE", uri="file://conf", content="data")
        assert a.content == "data"

        # Missing required fields
        with pytest.raises(ValidationError):
            ArtifactConfig(name="Bad")  # Missing uri

    def test_probe_config(self):
        """
        REQ-4.1: Readiness Probing.
        """
        # TCP Probe
        p = ProbeConfig(tcp_port=TcpPortProbeConfig(port=8080))
        assert p.tcp_port.port == 8080

        # Http Probe
        p = ProbeConfig(http_get=HttpProbeConfig(url="http://localhost"))
        assert str(p.http_get.url) == "http://localhost/"

        # Log Watch Probe
        p = ProbeConfig(
            log_watch=LogWatchProbeConfig(regex_pattern="Ready", logger="other_task")
        )
        assert p.log_watch.regex_pattern == "Ready"
        assert p.log_watch.logger == "other_task"

        # Defaults
        assert p.timeout == 60
        assert p.interval == 5

    def test_task_config_required_fields(self):
        """
        REQ-3.1: Task Definition. Name and script are minimal requirements effectively?
        Actually script is required in the Pydantic model.
        """
        with pytest.raises(ValidationError):
            TaskConfig(name="no_script")

        t = TaskConfig(name="basic", script=["echo hi"])
        assert t.name == "basic"
        assert t.script == ["echo hi"]

        # operator can be a string or an inline override object
        t2 = TaskConfig(
            name="with_operator_override",
            script=["echo hi"],
            operator={"name": "op", "ntasks": 4, "ntasks_per_node": 2},
        )
        assert t2.operator is not None
        assert t2.operator.name == "op"
        assert t2.operator.ntasks == 4
        assert t2.operator.ntasks_per_node == 2

    def test_task_resources(self):
        """
        REQ-3.1: Resources schema.
        """
        r = ResourcesConfig(
            nodes=NodeResourceConfig(count=2), gpus=GpuResourceConfig(count=4)
        )
        t = TaskConfig(name="res_task", script=["run"], resources=r)
        assert t.resources.nodes.count == 2
        assert t.resources.gpus.count == 4

    def test_replica_policy(self):
        """
        REQ-3.3: Task Replication policies.
        """
        # Valid enum values
        r = ReplicaConfig(policy=ReplicaPolicy.PARALLEL)
        assert r.policy == "parallel"

        r = ReplicaConfig(policy=ReplicaPolicy.SEQUENTIAL)
        assert r.policy == "sequential"

        # String conversion works if it matches enum value
        r = ReplicaConfig(policy="parallel")
        assert r.policy == ReplicaPolicy.PARALLEL

        # Expression strings are allowed (resolved later during assembly)
        r = ReplicaConfig(policy="${{ variables.REPLICA_POLICY }}")
        assert r.policy == "${{ variables.REPLICA_POLICY }}"

    def test_output_config(self):
        """
        REQ-4.3: Output Parsing.
        """
        o = OutputConfig(
            pattern="Loss: {loss:f}", metrics={"loss": OutputMetricConfig(type="float")}
        )
        assert o.pattern == "Loss: {loss:f}"
        assert o.metrics["loss"].type == "float"
        assert o.source == "stdout"  # Default

    def test_backend_and_operator_config(self):
        """
        REQ-2.1: Backend abstraction and operator-based execution.
        """
        cfg = SflowConfig.model_validate(
            {
                "version": "0.1",
                "backends": [
                    {
                        "name": "slurm",
                        "type": "slurm",
                        "account": "acct",
                        "partition": "batch",
                        "time": "1h",
                        "nodes": 1,
                        "gpus_per_node": 1,
                    },
                    {"name": "local", "type": "local"},
                ],
                "workflow": {
                    "name": "wf",
                    "tasks": [{"name": "t1", "script": ["echo 1"]}],
                },
            }
        )
        assert cfg.backends is not None
        assert {b.type for b in cfg.backends} == {"slurm", "local"}

    def test_retry_config(self):
        """
        REQ-3.6: Task Retry Policy.
        """
        retry = RetryConfig(count=3, interval=10, backoff=2)
        assert retry.count == 3
        assert retry.interval == 10
        assert retry.backoff == 2

        t = TaskConfig(name="retry_task", script=["fail"], retries=retry)
        assert t.retries.count == 3
