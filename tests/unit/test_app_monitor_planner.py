# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-time monitor resolution + dedup (via build_state, no real allocation)."""

import asyncio
import logging

import pytest

from sflow.app.assembly import build_state
from sflow.config.schema import SflowConfig


def _build(workflow: dict, tmp_path, *, nodes: int = 2):
    config = SflowConfig.model_validate(
        {
            "version": "0.1",
            "backends": [
                {
                    "name": "slurm",
                    "type": "slurm",
                    "default": True,
                    "nodes": nodes,
                    "gpus_per_node": 8,
                    "partition": "p",
                    "account": "a",
                    "time": "01:00:00",
                }
            ],
            "workflow": workflow,
        }
    )
    # allocate=False -> placeholder allocation with N nodes: slurm-node0..N-1.
    return asyncio.run(build_state(config, allocate=False, output_dir=tmp_path))


def _build_k8s(workflow: dict, tmp_path, *, nodes: int = 2):
    # k8s tasks require an explicit `k8s` operator carrying a workload image;
    # inject one onto every task so the test workflows stay terse.
    wf = dict(workflow)
    wf["tasks"] = [{**t, "operator": t.get("operator", "work")} for t in wf.get("tasks", [])]
    config = SflowConfig.model_validate(
        {
            "version": "0.1",
            "backends": [
                {
                    "name": "k8s",
                    "type": "kubernetes",
                    "default": True,
                    "namespace": "ns",
                    "nodes": nodes,
                    "gpus_per_node": 8,
                }
            ],
            "operators": [{"name": "work", "type": "k8s", "image": "img:1"}],
            "workflow": wf,
        }
    )
    return asyncio.run(build_state(config, allocate=False, output_dir=tmp_path))


def _build_docker(workflow: dict, tmp_path):
    config = SflowConfig.model_validate(
        {
            "version": "0.1",
            "backends": [
                {
                    "name": "local_docker",
                    "type": "docker",
                    "default": True,
                    "image": "nvcr.io/example/img:1",
                    "gpus_per_node": 1,
                    "nodes": 1,
                }
            ],
            "workflow": workflow,
        }
    )
    # allocate=False -> placeholder allocation with one "localhost" node.
    return asyncio.run(build_state(config, allocate=False, output_dir=tmp_path))


def test_workflow_monitor_covers_all_pool_nodes(tmp_path):
    state = _build(
        {
            "name": "wf",
            "monitor": {"interval": 1000},
            "tasks": [{"name": "a", "script": ["echo a"]}],
        },
        tmp_path,
    )
    reg = state.monitor_registry
    assert reg is not None
    # One collector per node (2 nodes), singleton.
    assert reg.collector_count == 2
    wf = state.workflow_monitor
    assert wf is not None
    assert wf.nodes == ["slurm-node0", "slurm-node1"]
    # No explicit scopes -> all built-ins.
    assert set(wf.scopes) == {"cpu", "gpu", "memory", "disk", "network"}


def test_workflow_and_task_monitor_dedup_same_node(tmp_path):
    state = _build(
        {
            "name": "wf",
            "monitor": {"scopes": {"gpu": {}}},  # workflow: gpu only, all nodes
            "tasks": [
                {
                    "name": "work",
                    "script": ["echo work"],
                    "resources": {"nodes": {"indices": [0]}},
                    "monitor": {"scopes": {"cpu": {}}},  # task: cpu only, node0
                }
            ],
        },
        tmp_path,
    )
    reg = state.monitor_registry
    # Still only 2 collectors (node0 shared by workflow + task -> singleton).
    assert reg.collector_count == 2
    # Node0 collector collects the UNION of scopes (gpu + cpu).
    node0 = reg._collectors[("slurm", "slurm-node0")]
    cmd = node0.command.as_str()
    assert "--scopes" in cmd
    scopes_token = cmd.split("--scopes ")[1].split()[0]
    assert "gpu" in scopes_token and "cpu" in scopes_token
    # Node1 collector only has the workflow's gpu scope.
    node1 = reg._collectors[("slurm", "slurm-node1")]
    cmd1 = node1.command.as_str()
    scopes1 = cmd1.split("--scopes ")[1].split()[0]
    assert "gpu" in scopes1 and "cpu" not in scopes1


def test_used_by_tasks_union_of_nodes_and_gpus(tmp_path):
    state = _build(
        {
            "name": "wf",
            "tasks": [
                {
                    "name": "server",
                    "script": ["sleep 1"],
                    "resources": {"nodes": {"indices": [1]}, "gpus": {"count": 4}},
                },
                {
                    "name": "bench",
                    "script": ["echo bench"],
                    "depends_on": ["server"],
                    "monitor": {"resources": {"used_by_tasks": ["server"]}},
                },
            ],
        },
        tmp_path,
    )
    reg = state.monitor_registry
    assert reg is not None
    # The bench monitor targets server's node (node1) -> 1 collector.
    assert reg.collector_count == 1
    assert ("slurm", "slurm-node1") in reg._collectors
    bench = state.workflow.get_task("bench")
    assert bench.monitor is not None
    assert bench.monitor.nodes == ["slurm-node1"]
    # GPU filter inherited from server's CUDA_VISIBLE_DEVICES (0,1,2,3).
    assert bench.monitor.gpus == [0, 1, 2, 3]


def test_no_monitor_means_no_registry(tmp_path):
    state = _build(
        {"name": "wf", "tasks": [{"name": "a", "script": ["echo a"]}]},
        tmp_path,
    )
    assert state.monitor_registry is None
    assert state.workflow_monitor is None


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_kubernetes_backend_skips_hardware_monitor(tmp_path):
    # Node-level monitoring is not implemented on k8s (the collector would run on
    # the driver host, not the reserved GPU nodes), so the planner skips it
    # entirely instead of producing misleading driver-host samples.
    state = _build_k8s(
        {
            "name": "wf",
            "monitor": {"interval": 1000, "report": {"enabled": True}},
            "tasks": [{"name": "a", "script": ["echo a"]}],
        },
        tmp_path,
    )
    assert state.monitor_registry is None
    assert state.workflow_monitor is None


def test_kubernetes_backend_used_by_tasks_monitor_skipped(tmp_path):
    state = _build_k8s(
        {
            "name": "wf",
            "tasks": [
                {
                    "name": "server",
                    "script": ["sleep 1"],
                    "resources": {"nodes": {"indices": [1]}, "gpus": {"count": 4}},
                },
                {
                    "name": "bench",
                    "script": ["echo bench"],
                    "depends_on": ["server"],
                    "monitor": {
                        "report": {"enabled": True},
                        "resources": {"used_by_tasks": ["server"]},
                    },
                },
            ],
        },
        tmp_path,
    )
    assert state.monitor_registry is None
    bench = state.workflow.get_task("bench")
    assert bench.monitor is None


def test_kubernetes_backend_monitor_logs_skip_hint(tmp_path):
    logger = logging.getLogger("sflow.app.monitor_planner")
    handler = _ListHandler()
    handler.setLevel(logging.INFO)
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        _build_k8s(
            {
                "name": "wf",
                "monitor": {"interval": 1000},
                "tasks": [{"name": "a", "script": ["echo a"]}],
            },
            tmp_path,
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    joined = "\n".join(handler.messages).lower()
    assert "monitor" in joined
    assert "not implemented" in joined or "kubernetes" in joined or "skip" in joined


def test_used_by_tasks_merges_across_replicas(tmp_path):
    """used_by_tasks on a replicated task unions ALL its replicas' nodes + GPUs."""
    state = _build(
        {
            "name": "wf",
            "tasks": [
                {
                    "name": "server",
                    "script": ["sleep 1"],
                    # 2 parallel replicas, one node + 4 GPUs each -> node0/node1.
                    "replicas": {"count": 2, "policy": "parallel"},
                    "resources": {"nodes": {"count": 1}, "gpus": {"count": 4}},
                },
                {
                    "name": "bench",
                    "script": ["echo bench"],
                    "depends_on": ["server"],
                    "monitor": {"resources": {"used_by_tasks": ["server"]}},
                },
            ],
        },
        tmp_path,
    )
    reg = state.monitor_registry
    bench = state.workflow.get_task("bench")
    assert bench.monitor is not None
    # Covers BOTH replica nodes (singleton collector per node).
    assert bench.monitor.nodes == ["slurm-node0", "slurm-node1"]
    assert reg.collector_count == 2
    # GPU filter is the union of both replicas' CUDA_VISIBLE_DEVICES (0-3 each).
    assert bench.monitor.gpus == [0, 1, 2, 3]


def test_used_by_tasks_does_not_overmatch_prefix_collision(tmp_path):
    """Referencing 'server' must NOT pull in a distinct 'server_extra' task."""
    state = _build(
        {
            "name": "wf",
            "tasks": [
                {
                    "name": "server",
                    "script": ["sleep 1"],
                    "replicas": {"count": 2, "policy": "parallel"},
                    "resources": {"nodes": {"count": 1}},  # -> node0, node1
                },
                {
                    "name": "server_extra",
                    "script": ["sleep 1"],
                    "resources": {"nodes": {"indices": [2]}},  # -> node2
                },
                {
                    "name": "bench",
                    "script": ["echo bench"],
                    "depends_on": ["server", "server_extra"],
                    "monitor": {"resources": {"used_by_tasks": ["server"]}},
                },
            ],
        },
        tmp_path,
        nodes=3,
    )
    bench = state.workflow.get_task("bench")
    assert bench.monitor is not None
    # Only server's two replica nodes; server_extra's node2 is excluded.
    assert bench.monitor.nodes == ["slurm-node0", "slurm-node1"]
    assert "slurm-node2" not in bench.monitor.nodes


def test_monitor_on_replicated_owner_attaches_per_replica(tmp_path):
    """A monitor on a replicated task is attached to each replica task."""
    state = _build(
        {
            "name": "wf",
            "tasks": [
                {
                    "name": "worker",
                    "script": ["echo w"],
                    "replicas": {"count": 2, "policy": "parallel"},
                    "resources": {"nodes": {"count": 1}},
                    "monitor": {"scopes": {"cpu": {}}},
                }
            ],
        },
        tmp_path,
    )
    w0 = state.workflow.get_task("worker_0")
    w1 = state.workflow.get_task("worker_1")
    assert w0.monitor is not None and w1.monitor is not None
    # Distinct consumers, one per replica, each on its own node.
    assert w0.monitor.owner == "task:worker_0"
    assert w1.monitor.owner == "task:worker_1"
    assert w0.monitor.nodes == ["slurm-node0"]
    assert w1.monitor.nodes == ["slurm-node1"]
    # Both replicas' nodes get a (deduped) collector.
    assert state.monitor_registry.collector_count == 2


def test_docker_monitor_runs_on_bare_host_not_container(tmp_path):
    """A Docker-backend collector runs on the host (bash), never inside the
    workload container -- otherwise it can't read the host-materialized
    hardware_monitor.py and would only see a cgroup-limited view."""
    state = _build_docker(
        {
            "name": "wf",
            "monitor": {"interval": 1000},
            "tasks": [{"name": "a", "script": ["echo a"]}],
        },
        tmp_path,
    )
    reg = state.monitor_registry
    assert reg is not None
    assert reg.collector_count == 1
    node = reg._collectors[("local_docker", "localhost")]
    cmd = node.command.as_str()
    # Bare-host monitoring: not a `docker run`, and references the host script path.
    assert "docker run" not in cmd
    assert "hardware_monitor.py" in cmd
    assert node.command._exec == "bash"


def _task_reports(state):
    return {r["label"]: r for r in state.monitor_registry.report_spec()["task_reports"]}


def test_workflow_monitor_emits_per_task_and_replica_views(tmp_path):
    """A workflow monitor expands into one natural view per covered task, with
    per-replica views (+ a combined view) for replicated tasks."""
    state = _build(
        {
            "name": "wf",
            "monitor": {"interval": 1000, "report": {"enabled": True}},
            "tasks": [
                {
                    "name": "server",
                    "script": ["sleep 1"],
                    "replicas": {"count": 2, "policy": "parallel"},
                    "resources": {"nodes": {"count": 1}, "gpus": {"count": 4}},
                },
                {
                    "name": "bench",
                    "script": ["echo b"],
                    "depends_on": ["server"],
                    "resources": {"nodes": {"indices": [2]}},
                },
            ],
        },
        tmp_path,
        nodes=3,
    )
    reports = _task_reports(state)
    # Replicated task -> per-replica views, each on its own node + GPUs.
    assert reports["server_0"]["nodes"] == ["slurm-node0"]
    assert reports["server_1"]["nodes"] == ["slurm-node1"]
    assert reports["server_0"]["gpus"] == [0, 1, 2, 3]
    # ...plus a combined view spanning both replicas.
    assert reports["server"]["nodes"] == ["slurm-node0", "slurm-node1"]
    assert set(reports["server"]["window_tasks"]) == {"server_0", "server_1"}
    assert reports["server_0"]["window_tasks"] == ["server_0"]
    # Non-replicated task -> single view, no per-replica duplicate.
    assert reports["bench"]["nodes"] == ["slurm-node2"]
    assert "bench_0" not in reports
    # All workflow-derived views are natural (not cross-task).
    assert all(not r["cross"] for r in reports.values())


def test_task_monitor_emits_only_that_task(tmp_path):
    """With no workflow monitor, only tasks with their own monitor get views."""
    state = _build(
        {
            "name": "wf",
            "tasks": [
                {"name": "a", "script": ["echo a"], "resources": {"nodes": {"indices": [0]}}},
                {
                    "name": "b",
                    "script": ["echo b"],
                    "resources": {"nodes": {"indices": [1]}},
                    "monitor": {"report": {"enabled": True}},
                },
            ],
        },
        tmp_path,
    )
    reports = _task_reports(state)
    assert set(reports) == {"b"}
    assert reports["b"]["nodes"] == ["slurm-node1"]


def test_task_log_window_survives_workflow_report_merge(tmp_path):
    state = _build(
        {
            "name": "wf",
            "monitor": {
                "scopes": {"cpu": {}},
                "report": {"enabled": True, "format": ["csv"]},
            },
            "tasks": [
                {
                    "name": "bench",
                    "script": ["echo b"],
                    "resources": {"nodes": {"indices": [0]}},
                    "monitor": {
                        "scopes": {"gpu": {}},
                        "report": {"enabled": True, "format": ["svg"]},
                        "window": {
                            "start": {"pattern": ["WARMUP_DONE", "re:^ready$"]},
                            "end": {
                                "pattern": "BENCHMARK_DONE",
                                "select": "first",
                            },
                        },
                    },
                }
            ],
        },
        tmp_path,
    )
    report = _task_reports(state)["bench"]
    assert report["scopes"] == ["cpu", "gpu"]
    assert report["formats"] == ["csv", "svg"]
    assert report["window_tasks"] == ["bench"]
    assert report["log_window"] == {
        "start": {
            "pattern": ["WARMUP_DONE", "re:^ready$"],
            "select": "first",
        },
        "end": {"pattern": ["BENCHMARK_DONE"], "select": "first"},
    }


def test_used_by_tasks_emits_cross_views_windowed_by_owner(tmp_path):
    """A used_by_tasks monitor (owner A watching B) yields B__monitored_by__A
    views over B's resources, windowed to A's run."""
    state = _build(
        {
            "name": "wf",
            "tasks": [
                {
                    "name": "server",
                    "script": ["sleep 1"],
                    "resources": {"nodes": {"indices": [1]}, "gpus": {"count": 4}},
                },
                {
                    "name": "bench",
                    "script": ["echo b"],
                    "depends_on": ["server"],
                    "monitor": {
                        "report": {"enabled": True},
                        "resources": {"used_by_tasks": ["server"]},
                        "window": {"start": "WARMUP_DONE", "end": "RUN_DONE"},
                    },
                },
            ],
        },
        tmp_path,
    )
    reports = _task_reports(state)
    label = "server__monitored_by__bench"
    assert label in reports
    view = reports[label]
    assert view["cross"] is True
    assert view["nodes"] == ["slurm-node1"]
    assert view["gpus"] == [0, 1, 2, 3]
    # The reporting window comes from the OWNER (bench), not the monitored task.
    assert view["window_tasks"] == ["bench"]
    assert view["log_window"] == {
        "start": {"pattern": ["WARMUP_DONE"], "select": "first"},
        "end": {"pattern": ["RUN_DONE"], "select": "last"},
    }
    assert "monitored by bench" in view["title"]


def test_report_disabled_emits_no_task_views(tmp_path):
    """A monitor with report.enabled=false samples but produces no task views."""
    state = _build(
        {
            "name": "wf",
            "monitor": {"interval": 1000, "report": {"enabled": False}},
            "tasks": [{"name": "a", "script": ["echo a"]}],
        },
        tmp_path,
    )
    assert state.monitor_registry is not None
    assert state.monitor_registry.report_spec()["task_reports"] == []


def test_report_is_enabled_by_default(tmp_path):
    """Declaring `monitor:` is enough: reports no longer need an opt-in block."""
    state = _build(
        {
            "name": "wf",
            "monitor": {"interval": 1000},  # no report block at all
            "tasks": [{"name": "a", "script": ["echo a"]}],
        },
        tmp_path,
    )
    reports = state.monitor_registry.report_spec()["task_reports"]
    assert [r["name"] for r in reports] == ["a"]


def test_task_monitor_default_targets_bound_task_nodes(tmp_path):
    state = _build(
        {
            "name": "wf",
            "tasks": [
                {
                    "name": "work",
                    "script": ["echo work"],
                    "resources": {"nodes": {"indices": [1]}},
                    "monitor": {},
                }
            ],
        },
        tmp_path,
    )
    reg = state.monitor_registry
    assert reg.collector_count == 1
    work = state.workflow.get_task("work")
    assert work.monitor is not None
    assert work.monitor.nodes == ["slurm-node1"]


def test_docker_monitor_views_use_physical_gpus_not_container_indices(tmp_path):
    """Two docker tasks sharing a host must be monitored on disjoint GPUs.

    The docker backend exposes CUDA_VISIBLE_DEVICES=0..N-1 *inside* each
    container, so deriving monitored GPUs from the task env alone would point
    every task at GPU 0/1 -- the monitor would then report one task's utilization
    against another's devices. Views must follow the planner's slice instead.
    """
    config = SflowConfig.model_validate(
        {
            "version": "0.1",
            "backends": [
                {
                    "name": "local_docker",
                    "type": "docker",
                    "default": True,
                    "image": "nvcr.io/example/img:1",
                    "gpus_per_node": 8,
                    "nodes": 1,
                }
            ],
            "workflow": {
                "name": "wf",
                "monitor": {"interval": 1000, "report": {"enabled": True}},
                "tasks": [
                    {
                        "name": "a",
                        "script": ["sleep 1"],
                        "resources": {"gpus": {"count": 2}},
                    },
                    {
                        "name": "b",
                        "script": ["sleep 1"],
                        "resources": {"gpus": {"count": 2}},
                    },
                ],
            },
        }
    )
    state = asyncio.run(build_state(config, allocate=False, output_dir=tmp_path))

    tasks = dict(state.workflow.task_graph.dag.nodes)
    # Both containers really do see the same virtual indices...
    assert tasks["a"].envs["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert tasks["b"].envs["CUDA_VISIBLE_DEVICES"] == "0,1"

    reports = _task_reports(state)
    # ...but the monitor targets the disjoint physical devices.
    assert reports["a"]["gpus"] == [0, 1]
    assert reports["b"]["gpus"] == [2, 3]
    assert set(reports["a"]["gpus"]).isdisjoint(reports["b"]["gpus"])


def test_monitor_gpu_indices_watch_exactly_those_devices(tmp_path):
    """``monitor.resources.gpus.indices`` selects specific devices.

    ``docs/user/monitor.md`` documents ``gpus: { count | indices }``, but the
    schema only accepted ``count`` until ``resources.gpus.indices`` was added --
    this pins the documented form to real behavior.
    """
    state = _build(
        {
            "name": "wf",
            "tasks": [
                {
                    "name": "work",
                    "script": ["echo work"],
                    "resources": {"nodes": {"indices": [0]}},
                    "monitor": {"resources": {"gpus": {"indices": [2, 5]}}},
                }
            ],
        },
        tmp_path,
    )
    work = state.workflow.get_task("work")
    assert work.monitor is not None
    # Not range(count) -- the exact devices asked for, order preserved.
    assert work.monitor.gpus == [2, 5]


def test_monitor_gpu_count_still_means_first_n_devices(tmp_path):
    state = _build(
        {
            "name": "wf",
            "tasks": [
                {
                    "name": "work",
                    "script": ["echo work"],
                    "resources": {"nodes": {"indices": [0]}},
                    "monitor": {"resources": {"gpus": {"count": 3}}},
                }
            ],
        },
        tmp_path,
    )
    work = state.workflow.get_task("work")
    assert work.monitor.gpus == [0, 1, 2]


def test_monitor_gpu_indices_expression_is_rejected_with_a_clear_message(tmp_path):
    # Monitor resources are not run through the expression resolver, so an
    # expression here would silently mean "no GPUs" without this guard.
    with pytest.raises(ValueError, match="do not support"):
        _build(
            {
                "name": "wf",
                "tasks": [
                    {
                        "name": "work",
                        "script": ["echo work"],
                        "resources": {"nodes": {"indices": [0]}},
                        "monitor": {
                            "resources": {"gpus": {"indices": "${{ variables.IDS }}"}}
                        },
                    }
                ],
            },
            tmp_path,
        )


def test_workflow_monitor_spans_every_backend_not_just_the_default(tmp_path):
    """`workflow.monitor` is the whole-POOL monitor.

    Resolving it against the default backend alone leaves a second pool
    unsampled, so tasks running there get a report folder with no data of their
    own -- which the post-processor's name-mismatch fallback then fills with the
    other backend's nodes and GPUs.
    """
    config = SflowConfig.model_validate(
        {
            "version": "0.1",
            "backends": [
                {
                    "name": "pool_a", "type": "slurm", "default": True, "nodes": 2,
                    "gpus_per_node": 8, "partition": "p", "account": "a",
                    "time": "01:00:00",
                },
                {
                    "name": "pool_b", "type": "slurm", "nodes": 1, "gpus_per_node": 0,
                    "partition": "p2", "account": "a", "time": "01:00:00",
                },
            ],
            "workflow": {
                "name": "wf",
                "monitor": {"report": {"enabled": True}},
                "tasks": [
                    {"name": "on_a", "script": ["echo a"]},
                    {"name": "on_b", "backend": "pool_b", "script": ["echo b"]},
                ],
            },
        }
    )
    state = asyncio.run(build_state(config, allocate=False, output_dir=tmp_path))
    spec = state.monitor_registry.report_spec()

    workflow_nodes = spec["consumers"][0]["nodes"]
    assert any(n.startswith("pool_b") for n in workflow_nodes), workflow_nodes
    assert any(n.startswith("pool_a") for n in workflow_nodes), workflow_nodes
    # The second pool's task gets a report of its OWN hardware.
    on_b = [r for r in spec["task_reports"] if r["name"] == "on_b"]
    assert on_b, [r["name"] for r in spec["task_reports"]]
    assert all(n.startswith("pool_b") for n in on_b[0]["nodes"]), on_b[0]["nodes"]


def test_reports_never_mix_in_another_tasks_gpus_on_a_shared_node(tmp_path):
    """Two tasks on ONE node must report only their own half of its GPUs.

    A collector is a per-node singleton sampling every GPU on that node, so the
    only thing keeping one task's report free of its neighbour's devices is the
    plan-time GPU subset. A task that reserved NO GPUs must likewise report none
    -- an empty subset is not "all of them".
    """
    state = _build(
        {
            "name": "wf",
            "monitor": {"report": {"enabled": True}},
            "tasks": [
                {
                    "name": "srv_a", "script": ["x"],
                    "resources": {"nodes": {"indices": [0]}, "gpus": {"count": 4}},
                },
                {
                    "name": "srv_b", "script": ["x"],
                    "resources": {"nodes": {"indices": [0]}, "gpus": {"count": 4}},
                },
                {
                    "name": "cpu_only", "script": ["x"],
                    "resources": {"nodes": {"indices": [1]}},
                },
            ],
        },
        tmp_path,
    )
    reports = {r["name"]: r for r in state.monitor_registry.report_spec()["task_reports"]}

    # Same node, disjoint GPU halves -- neither may contain the other's.
    assert reports["srv_a"]["nodes"] == reports["srv_b"]["nodes"] == ["slurm-node0"]
    a, b = set(reports["srv_a"]["gpus"]), set(reports["srv_b"]["gpus"])
    assert a and b and not (a & b), (a, b)
    # No GPUs reserved -> report none, NOT every GPU on its node.
    assert reports["cpu_only"]["gpus"] == [], reports["cpu_only"]["gpus"]



def test_shared_view_credits_both_the_workflow_and_the_task_monitor(tmp_path):
    """A folder both monitors ask for must count for BOTH of them.

    Views are deduped by label and the workflow monitor emits first, so the merged
    view keeps `triggered_by="workflow"`. Counting folders by that alone credits
    the task monitor with none and the dry-run then reports it as `report=no` --
    i.e. "this monitor writes nothing", for the monitor that owns the window.
    """
    state = _build(
        {
            "name": "wf",
            "monitor": {"interval": 1000},
            "tasks": [
                {
                    "name": "work",
                    "script": ["echo work"],
                    "monitor": {
                        "interval": 1000,
                        "window": {"start": "GO", "end": "STOP"},
                    },
                },
            ],
        },
        tmp_path,
    )
    views = {v.label: v for v in state.monitor_registry.task_views}
    work = views["work"]
    # One folder, but two monitors put it there.
    assert work.triggered_by == "workflow"
    assert work.contributors == ["workflow", "task:work"]
    # The task monitor's window survived the merge onto the workflow's view.
    assert work.log_window is not None

    folders_by_owner: dict[str, int] = {}
    for view in state.monitor_registry.task_views:
        for owner in view.contributors:
            folders_by_owner[owner] = folders_by_owner.get(owner, 0) + 1
    assert folders_by_owner["task:work"] == 1
