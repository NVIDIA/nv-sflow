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
    validate_node_exclude_indices,
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

    def test_resource_release_after_accepts_independent_node_and_gpu_policies(self):
        """resources.nodes/gpus.release_after should allow different policies."""
        config = SflowConfig.model_validate({
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t1",
                        "script": ["echo 1"],
                        "resources": {
                            "nodes": {
                                "indices": [0],
                                "release_after": "workflow_completion",
                            },
                            "gpus": {
                                "count": 8,
                                "release_after": "task_completion",
                            },
                        },
                    }
                ],
            },
        })

        resources = config.workflow.tasks[0].resources
        assert resources.nodes.release_after == "workflow_completion"
        assert resources.gpus.release_after == "task_completion"

    def test_resource_release_after_is_unset_until_a_policy_is_named(self):
        """An unannotated resource states no policy, and still states none after a dump.

        The planner reads the *value* to decide whether a task claims what it is placed
        on, so "no policy" has to be a value, not the absence of an assignment.
        """
        config = SflowConfig.model_validate({
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t1",
                        "script": ["echo 1"],
                        "resources": {
                            "nodes": {"count": 1},
                            "gpus": {"count": 1},
                        },
                    }
                ],
            },
        })

        # No policy named: carried as None rather than as a materialized default, so
        # "the recipe did not ask for a reservation" survives a dump and reload. The
        # planner turns None into the effective lifetime (see ReservationPolicyPlanner).
        resources = config.workflow.tasks[0].resources
        assert resources.nodes.release_after is None
        assert resources.gpus.release_after is None

        # and it stays None through a round trip, which is the point
        again = SflowConfig.model_validate(config.model_dump(mode="json", exclude_none=True))
        assert again.workflow.tasks[0].resources.nodes.release_after is None
        assert again.workflow.tasks[0].resources.gpus.release_after is None

    def test_resource_release_after_rejects_invalid_policy(self):
        """release_after only accepts documented resource lifetime policies."""
        with pytest.raises(ValidationError, match="release_after"):
            SflowConfig.model_validate({
                "version": "0.1",
                "workflow": {
                    "name": "wf",
                    "tasks": [
                        {
                            "name": "t1",
                            "script": ["echo 1"],
                            "resources": {
                                "gpus": {
                                    "count": 1,
                                    "release_after": "after_launch",
                                },
                            },
                        }
                    ],
                },
            })

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
        assert str(p.http_get.url) == "http://localhost"

        # Http GET Probe with expression (resolved later by assembly)
        p = ProbeConfig(http_get=HttpProbeConfig(url="http://${{ backends.slurm_cluster.nodes[0].ip_address }}:8000/health"))
        assert "${{" in str(p.http_get.url)

        # Http POST Probe with body and headers
        p = ProbeConfig(http_post=HttpProbeConfig(
            url="http://${{ backends.slurm_cluster.nodes[0].ip_address }}:8000/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body='{"model": "${{ variables.SERVED_MODEL_NAME }}", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}',
        ))
        assert "${{" in str(p.http_post.url)
        assert p.http_post.headers == {"Content-Type": "application/json"}
        assert "${{ variables.SERVED_MODEL_NAME }}" in p.http_post.body

        # Log Watch Probe with regex_pattern
        p = ProbeConfig(
            log_watch=LogWatchProbeConfig(regex_pattern="Ready", logger="other_task")
        )
        assert p.log_watch.regex_pattern == "Ready"
        assert p.log_watch.logger == "other_task"

        # Log Watch Probe with match_pattern (alias)
        p = ProbeConfig(
            log_watch=LogWatchProbeConfig(match_pattern="server started")
        )
        assert p.log_watch.regex_pattern == "server started"
        assert p.log_watch.match_pattern == "server started"

        # Log Watch Probe: two different patterns is rejected
        with pytest.raises(ValidationError, match="Only one of"):
            LogWatchProbeConfig(regex_pattern="a", match_pattern="b")

        # Log Watch Probe: neither field set is rejected
        with pytest.raises(ValidationError, match="Either"):
            LogWatchProbeConfig()

        # Log Watch Probe: the same pattern under both spellings is what this model
        # itself produces, so it has to load again -- see the round-trip test below.
        both = LogWatchProbeConfig(regex_pattern="server started", match_pattern="server started")
        assert both.regex_pattern == "server started"

        # Defaults
        assert p.timeout == 1200
        assert p.each_check_timeout == 30
        assert p.interval == 5

        # Backwards compatibility: the old single readiness probe object is still valid.
        single_probe = ProbeConfig(tcp_port=TcpPortProbeConfig(port=8080))
        probes = ProbesConfig(readiness=single_probe)
        assert probes.readiness == single_probe

        # Multiple readiness probes are allowed and evaluated as an AND at runtime.
        probes = ProbesConfig(
            readiness=[
                ProbeConfig(tcp_port=TcpPortProbeConfig(port=8080)),
                ProbeConfig(http_get=HttpProbeConfig(url="http://localhost/health")),
            ]
        )
        assert len(probes.readiness) == 2

        # Backwards compatibility: the old single failure probe object is still valid.
        failure_probe = ProbeConfig(
            log_watch=LogWatchProbeConfig(match_pattern="Traceback")
        )
        probes = ProbesConfig(failure=failure_probe)
        assert probes.failure == failure_probe

        # Multiple failure probes are allowed and evaluated as an OR at runtime.
        probes = ProbesConfig(
            failure=[
                ProbeConfig(log_watch=LogWatchProbeConfig(match_pattern="Traceback")),
                ProbeConfig(
                    log_watch=LogWatchProbeConfig(match_pattern="RuntimeError")
                ),
            ]
        )
        assert len(probes.failure) == 2

        with pytest.raises(ValidationError, match="failure probe list cannot be empty"):
            ProbesConfig(failure=[])

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

        expr_resources = ResourcesConfig(
            nodes=NodeResourceConfig(
                indices="${{ range(variables.INFRA_NODE_INDEX, variables.INFRA_NODE_INDEX + variables.NUM_FRONTENDS) | list }}"
            )
        )
        expr_task = TaskConfig(name="expr_task", script=["run"], resources=expr_resources)
        assert expr_task.resources.nodes.indices == (
            "${{ range(variables.INFRA_NODE_INDEX, variables.INFRA_NODE_INDEX + variables.NUM_FRONTENDS) | list }}"
        )

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

    def test_legacy_docker_operator_type_rejected_with_hint(self):
        """A legacy operator `type: docker` should raise a hint to use `docker_run`."""
        with pytest.raises(ValidationError, match="docker_run"):
            SflowConfig.model_validate(
                {
                    "version": "0.1",
                    "operators": [
                        {"name": "legacy", "type": "docker", "image": "nvcr.io/x/y:1"}
                    ],
                    "workflow": {
                        "name": "wf",
                        "tasks": [{"name": "t1", "script": ["echo 1"]}],
                    },
                }
            )

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


# ---------------------------------------------------------------------------
# validate_node_exclude_indices tests
# ---------------------------------------------------------------------------

def _make_config(
    nodes_val,
    exclude_val,
    *,
    variables=None,
    task_name="t1",
):
    """Build a minimal SflowConfig for exclude-index validation tests."""
    data = {
        "version": "0.1",
        "backends": [
            {
                "name": "slurm",
                "type": "slurm",
                "default": True,
                "account": "a",
                "partition": "p",
                "time": "1h",
                "nodes": nodes_val,
                "gpus_per_node": 1,
            },
        ],
        "workflow": {
            "name": "wf",
            "tasks": [
                {
                    "name": task_name,
                    "script": ["echo 1"],
                    "resources": {"nodes": {"exclude": exclude_val}},
                },
            ],
        },
    }
    if variables:
        data["variables"] = variables
    return SflowConfig.model_validate(data)


class TestValidateNodeExcludeIndices:

    def test_concrete_nodes_valid_exclude(self):
        """Valid exclude index should not raise."""
        cfg = _make_config(nodes_val=3, exclude_val=[0, 2])
        validate_node_exclude_indices(cfg)

    def test_concrete_nodes_out_of_range_exclude(self):
        """Exclude index >= total nodes should raise."""
        cfg = _make_config(nodes_val=2, exclude_val=[2])
        with pytest.raises(ValueError, match="out of range for 2 allocated"):
            validate_node_exclude_indices(cfg)

    def test_concrete_nodes_negative_exclude_wraps(self):
        """Negative exclude index wraps Python-style: -1 is last node."""
        cfg = _make_config(nodes_val=3, exclude_val=[-1])
        validate_node_exclude_indices(cfg)  # -1 → index 2, valid for 3 nodes

    def test_concrete_nodes_negative_exclude_out_of_range(self):
        """Negative exclude index too large should raise."""
        cfg = _make_config(nodes_val=3, exclude_val=[-4])
        with pytest.raises(ValueError, match="out of range for 3 allocated"):
            validate_node_exclude_indices(cfg)

    def test_concrete_nodes_single_int_exclude(self):
        """Single int exclude (not a list) should work."""
        cfg = _make_config(nodes_val=2, exclude_val=0)
        validate_node_exclude_indices(cfg)

    def test_concrete_nodes_single_int_out_of_range(self):
        """Single int exclude out of range should raise."""
        cfg = _make_config(nodes_val=1, exclude_val=1)
        with pytest.raises(ValueError, match="out of range for 1 allocated"):
            validate_node_exclude_indices(cfg)

    def test_variable_resolved_nodes_out_of_range(self):
        """Exclude validated against variable-resolved backend nodes."""
        cfg = _make_config(
            nodes_val="${{ variables.SLURM_NODES }}",
            exclude_val=[2],
            variables={"SLURM_NODES": {"value": 2}},
        )
        with pytest.raises(ValueError, match="out of range for 2 allocated"):
            validate_node_exclude_indices(cfg)

    def test_variable_resolved_nodes_valid(self):
        """Valid exclude after resolving variable should not raise."""
        cfg = _make_config(
            nodes_val="${{ variables.SLURM_NODES }}",
            exclude_val=[1],
            variables={"SLURM_NODES": {"value": 4}},
        )
        validate_node_exclude_indices(cfg)

    def test_backend_planning_node_count_hook_used(self, monkeypatch):
        """Exclude validation should use the backend config hook, not raw attr access."""
        cfg = _make_config(nodes_val=3, exclude_val=[3])
        backend = cfg.backends[0]
        monkeypatch.setattr(
            type(backend),
            "planning_node_count",
            lambda self: 3,
            raising=False,
        )
        backend.nodes = None

        with pytest.raises(ValueError, match="out of range for 3 allocated"):
            validate_node_exclude_indices(cfg)

    def test_unresolvable_variable_skipped(self):
        """When the variable is not defined, validation is skipped."""
        cfg = _make_config(
            nodes_val="${{ variables.UNKNOWN_VAR }}",
            exclude_val=[99],
        )
        validate_node_exclude_indices(cfg)

    def test_nested_expression_skipped(self):
        """Complex expressions that can't be resolved are skipped."""
        cfg = _make_config(
            nodes_val="${{ variables.A + variables.B }}",
            exclude_val=[99],
        )
        validate_node_exclude_indices(cfg)

    def test_exclude_expression_skipped(self):
        """Exclude values that are expressions are skipped."""
        cfg = _make_config(
            nodes_val=4,
            exclude_val="${{ variables.EXCLUDE_IDX }}",
        )
        validate_node_exclude_indices(cfg)

    def test_no_backends_skipped(self):
        """No backends → no validation."""
        cfg = SflowConfig.model_validate({
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo 1"]}],
            },
        })
        validate_node_exclude_indices(cfg)

    def test_no_exclude_skipped(self):
        """No exclude configured → no validation."""
        cfg = SflowConfig.model_validate({
            "version": "0.1",
            "backends": [{
                "name": "slurm", "type": "slurm", "default": True,
                "account": "a", "partition": "p", "time": "1h",
                "nodes": 2, "gpus_per_node": 1,
            }],
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo 1"]}],
            },
        })
        validate_node_exclude_indices(cfg)

    def test_workflow_level_variable_resolved(self):
        """Workflow-level variables should also be resolved."""
        cfg = SflowConfig.model_validate({
            "version": "0.1",
            "backends": [{
                "name": "slurm", "type": "slurm", "default": True,
                "account": "a", "partition": "p", "time": "1h",
                "nodes": "${{ variables.N }}", "gpus_per_node": 1,
            }],
            "workflow": {
                "name": "wf",
                "variables": {"N": {"value": 2}},
                "tasks": [{
                    "name": "t1", "script": ["echo 1"],
                    "resources": {"nodes": {"exclude": [2]}},
                }],
            },
        })
        with pytest.raises(ValueError, match="out of range for 2 allocated"):
            validate_node_exclude_indices(cfg)

    def test_multiple_tasks_partial_error(self):
        """Only the task with bad exclude raises; other tasks are fine."""
        cfg = SflowConfig.model_validate({
            "version": "0.1",
            "backends": [{
                "name": "slurm", "type": "slurm", "default": True,
                "account": "a", "partition": "p", "time": "1h",
                "nodes": 2, "gpus_per_node": 1,
            }],
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "good_task", "script": ["echo 1"],
                        "resources": {"nodes": {"exclude": [0]}},
                    },
                    {
                        "name": "bad_task", "script": ["echo 2"],
                        "resources": {"nodes": {"exclude": [5]}},
                    },
                ],
            },
        })
        with pytest.raises(ValueError, match="bad_task.*out of range"):
            validate_node_exclude_indices(cfg)


def test_task_ports_accepts_int_expression_and_optional_name():
    task = TaskConfig(
        name="frontend",
        script=["python serve.py"],
        ports=[
            {"name": "http", "port": 8000},
            {"port": "${{ variables.PORT }}"},
        ],
    )
    assert task.ports is not None
    assert task.ports[0].name == "http"
    assert task.ports[0].port == 8000
    assert task.ports[1].name is None
    assert task.ports[1].port == "${{ variables.PORT }}"


def test_task_ports_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TaskConfig(
            name="frontend",
            script=["echo hi"],
            ports=[{"port": 8000, "protocol": "tcp"}],
        )


class TestLogWatchPatternRoundTrip:
    """A validated config has to survive being dumped and loaded again.

    `sflow compose` writes one, and `sflow batch` writes one for its dry run whenever
    `--nodes` / `--gpus-per-node` re-plan the backends. `normalize_pattern` fills
    `regex_pattern` in from `match_pattern`, so both spellings are present in that
    dump -- which used to be rejected on the way back in, failing every recipe whose
    readiness probe is written with `match_pattern`.
    """

    @staticmethod
    def _dump(probe):
        return probe.model_dump(mode="json", exclude_none=True)

    def test_match_pattern_survives_a_dump_and_reload(self):
        probe = LogWatchProbeConfig(match_pattern="Image Loaded", match_count=2)
        dumped = self._dump(probe)
        assert dumped["regex_pattern"] == "Image Loaded"
        assert dumped["match_pattern"] == "Image Loaded"

        reloaded = LogWatchProbeConfig.model_validate(dumped)
        assert reloaded.regex_pattern == "Image Loaded"
        assert reloaded.match_count == 2
        assert self._dump(reloaded) == dumped  # and again, without drifting

    def test_regex_pattern_alone_still_round_trips(self):
        probe = LogWatchProbeConfig(regex_pattern="Maximum concurrency for")
        dumped = self._dump(probe)
        assert "match_pattern" not in dumped
        assert LogWatchProbeConfig.model_validate(dumped).regex_pattern == "Maximum concurrency for"

    def test_conflicting_patterns_are_still_rejected(self):
        with pytest.raises(ValidationError, match="Only one of"):
            LogWatchProbeConfig.model_validate(
                {"regex_pattern": "Image Loaded", "match_pattern": "Ready"}
            )

    def test_a_whole_workflow_round_trips(self):
        config = SflowConfig.model_validate(
            {
                "version": "0.1",
                "workflow": {
                    "name": "wf",
                    "tasks": [
                        {
                            "name": "load_image",
                            "script": ["echo hi"],
                            "probes": {
                                "readiness": {
                                    "log_watch": {
                                        "match_pattern": "Image Loaded",
                                        "match_count": 2,
                                    }
                                }
                            },
                        }
                    ],
                },
            }
        )
        dumped = config.model_dump(mode="json", exclude_none=True)
        SflowConfig.model_validate(dumped)


class TestRequiredBy:
    """`required_by` is the reverse pointer of `depends_on` and is folded into
    the targets' `depends_on` at validation time."""

    @staticmethod
    def _cfg(tasks):
        return SflowConfig.model_validate(
            {"version": "0.1", "workflow": {"name": "wf", "tasks": tasks}}
        )

    def _deps(self, cfg):
        return {t.name: (t.depends_on or []) for t in cfg.workflow.tasks}

    def test_required_by_folds_into_downstream_depends_on(self):
        cfg = self._cfg(
            [
                {"name": "server", "script": ["s"], "required_by": ["benchmark"]},
                {"name": "benchmark", "script": ["b"]},
            ]
        )
        assert self._deps(cfg) == {"server": [], "benchmark": ["server"]}

    def test_required_by_equivalent_to_depends_on(self):
        """A workflow using required_by yields the same depends_on graph as the
        equivalent depends_on workflow."""
        via_required_by = self._deps(
            self._cfg(
                [
                    {"name": "a", "script": ["a"], "required_by": ["c"]},
                    {"name": "b", "script": ["b"], "required_by": ["c"]},
                    {"name": "c", "script": ["c"]},
                ]
            )
        )
        via_depends_on = self._deps(
            self._cfg(
                [
                    {"name": "a", "script": ["a"]},
                    {"name": "b", "script": ["b"]},
                    {"name": "c", "script": ["c"], "depends_on": ["a", "b"]},
                ]
            )
        )
        assert via_required_by == via_depends_on

    def test_absent_target_is_skipped(self):
        """A required_by target absent from the merged task set is dropped
        silently (no validation error) -- this is what removes the need for
        --missable-tasks in the forward direction."""
        cfg = self._cfg(
            [
                {
                    "name": "server",
                    "script": ["s"],
                    "required_by": ["benchmark", "not_included"],
                },
                {"name": "benchmark", "script": ["b"]},
            ]
        )
        assert self._deps(cfg) == {"server": [], "benchmark": ["server"]}

    def test_dedup_with_existing_depends_on(self):
        cfg = self._cfg(
            [
                {"name": "a", "script": ["a"], "required_by": ["b"]},
                {"name": "b", "script": ["b"], "depends_on": ["a"]},
            ]
        )
        assert self._deps(cfg)["b"] == ["a"]

    def test_self_reference_is_skipped(self):
        """A task requiring itself must not create a self-cycle."""
        cfg = self._cfg(
            [{"name": "a", "script": ["a"], "required_by": ["a"]}]
        )
        assert self._deps(cfg)["a"] == []

    def test_dict_form_tasks(self):
        cfg = self._cfg(
            {
                "a": {"script": ["a"], "required_by": ["b"]},
                "b": {"script": ["b"]},
            }
        )
        assert self._deps(cfg)["b"] == ["a"]

    def test_required_by_field_preserved(self):
        cfg = self._cfg(
            [
                {"name": "a", "script": ["a"], "required_by": ["b"]},
                {"name": "b", "script": ["b"]},
            ]
        )
        by_name = {t.name: t for t in cfg.workflow.tasks}
        assert by_name["a"].required_by == ["b"]
