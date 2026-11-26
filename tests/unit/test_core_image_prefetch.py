# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-node, per-image container image prefetch: the backend-agnostic core."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

import sflow.core.image_prefetch as image_prefetch_mod
from sflow.app.backend_lifecycle import prefetch_images
from sflow.core.backend import Allocation, Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.image_prefetch import PrefetchResult, plan_image_prefetch
from sflow.core.state import SflowState
from sflow.core.task import Task
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow
from sflow.plugins.operators.bash import BashOperator, BashOperatorConfig
from sflow.plugins.operators.srun import SrunOperator, SrunOperatorConfig

NIM = "nvcr.io/nim/meta/llama:1.0"
CLIENT = "gitlab-master.nvidia.com:5005/team/client:latest"


def _srun_task(name: str, nodes: list[str], **op_kwargs) -> Task:
    op = SrunOperator(SrunOperatorConfig(name=name, **op_kwargs))
    task = Task(name=name, logger=logging.getLogger(name), operator=op)
    task.assigned_nodes = list(nodes)
    return task


def _bash_task(name: str) -> Task:
    op = BashOperator(BashOperatorConfig(name=name))
    return Task(name=name, logger=logging.getLogger(name), operator=op)


def _summary(plan):
    return [(p.image, p.nodes, [t.name for t in p.tasks]) for p in plan]


def test_three_tasks_on_two_images_plan_one_entry_per_image():
    tasks = [
        _srun_task("nim_0", ["n2"], container_image=NIM),
        _srun_task("benchmark", ["n1"], container_image=CLIENT),
        _srun_task("nim_1", ["n1"], container_image=NIM),
    ]

    plan = plan_image_prefetch(tasks, ["n1", "n2"])

    # Images in order of first use; nodes in allocation order, each once.
    assert _summary(plan) == [
        (NIM, ["n1", "n2"], ["nim_0", "nim_1"]),
        (CLIENT, ["n1"], ["benchmark"]),
    ]


def test_plan_skips_local_images_and_containerless_tasks():
    tasks = [
        _srun_task("sqsh", ["n1"], container_image="/lustre/images/nim.sqsh"),
        _srun_task("relative", ["n1"], container_image="./nim.sqsh"),
        _srun_task("bare_sqsh", ["n1"], container_image="nim.sqsh"),
        _srun_task("no_image", ["n1"]),
        _bash_task("bash"),
        # Images given via extra_args count too.
        _srun_task("via_extra_args", [], extra_args=[f"--container-image={CLIENT}"]),
    ]

    # A task with no assigned nodes may run anywhere in the allocation.
    assert _summary(plan_image_prefetch(tasks, ["n1", "n2"])) == [
        (CLIENT, ["n1", "n2"], ["via_extra_args"]),
    ]


# -- the run step: prefetch_images(state) ------------------------------------


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setattr(image_prefetch_mod, "PREFETCH_RETRY_DELAY_S", 0)


class _FakeBackend(Backend):
    """Pulls by script: each call pops the next list of nodes that fail."""

    def __init__(self, name, *, failures=(), enabled=True, nodes=("n1", "n2")):
        super().__init__(name=name)
        self.config = SimpleNamespace(prefetch_images=enabled)
        self.allocation = Allocation(
            allocation_id=name,
            nodes=[
                ComputeNode(name=n, ip_address=f"10.0.0.{i}", index=i)
                for i, n in enumerate(nodes)
            ],
        )
        self._failures = [list(f) for f in failures]
        self.calls: list[tuple[str, list[str], list[str]]] = []

    async def allocate(self):  # pragma: no cover - not used
        raise NotImplementedError

    async def release(self, allocation):  # pragma: no cover - not used
        return None

    def default_operator(self, **_kw):  # pragma: no cover - not used
        raise NotImplementedError

    async def prefetch_image(self, image, nodes, tasks):
        await asyncio.sleep(0)
        self.calls.append((image, list(nodes), [t.name for t in tasks]))
        failed = self._failures.pop(0) if self._failures else []
        return PrefetchResult(
            failed_nodes=[n for n in nodes if n in failed],
            output=f"pull {image} failed on {','.join(failed)}" if failed else "",
        )


def _state(backends, tasks_by_backend) -> SflowState:
    tg = TaskGraph()
    for backend_name, tasks in tasks_by_backend.items():
        for task in tasks:
            task.backend_name = backend_name
            tg.dag.add_node(task.name, task)
    state = SflowState(workflow=Workflow(name="wf", task_graph=tg))
    state.backends = {b.name: b for b in backends}
    return state


def test_failed_pull_is_retried_on_the_failed_nodes_only():
    backend = _FakeBackend("b", failures=[["n2"]])
    state = _state(
        [backend], {"b": [_srun_task("nim", ["n1", "n2"], container_image=NIM)]}
    )

    asyncio.run(prefetch_images(state))

    assert [(image, nodes) for image, nodes, _ in backend.calls] == [
        (NIM, ["n1", "n2"]),
        (NIM, ["n2"]),
    ]


def test_exhausted_retries_fail_naming_backend_image_node_and_output():
    backend = _FakeBackend("gpu", failures=[["n1"], ["n1"], ["n1"]])
    state = _state(
        [backend], {"gpu": [_srun_task("nim", ["n1", "n2"], container_image=NIM)]}
    )

    with pytest.raises(image_prefetch_mod.ImagePrefetchError) as excinfo:
        asyncio.run(prefetch_images(state))

    message = str(excinfo.value)
    assert "Backend 'gpu'" in message
    assert NIM in message
    assert "node(s) n1 " in message
    assert "after 3 attempt(s)" in message
    assert f"pull {NIM} failed on n1" in message
    assert [nodes for _, nodes, _ in backend.calls] == [["n1", "n2"], ["n1"], ["n1"]]


class _NoPrimitiveBackend(_FakeBackend):
    prefetch_image = Backend.prefetch_image


def test_each_enabled_backend_pulls_only_its_own_tasks_images():
    gpu = _FakeBackend("gpu")
    cpu = _FakeBackend("cpu")
    off = _FakeBackend("off", enabled=False)
    unsupported = _NoPrimitiveBackend("local")
    state = _state(
        [gpu, cpu, off, unsupported],
        {
            "gpu": [
                _srun_task("nim_0", ["n1"], container_image=NIM),
                _srun_task("nim_1", ["n2"], container_image=NIM),
            ],
            "cpu": [_srun_task("bench", ["n1"], container_image=CLIENT)],
            "off": [_srun_task("other", ["n1"], container_image=CLIENT)],
            "local": [_srun_task("local", ["n1"], container_image=CLIENT)],
        },
    )

    asyncio.run(prefetch_images(state))

    assert gpu.calls == [(NIM, ["n1", "n2"], ["nim_0", "nim_1"])]
    assert cpu.calls == [(CLIENT, ["n1"], ["bench"])]
    assert off.calls == []
    assert unsupported.calls == []


def test_a_backend_pulls_one_image_at_a_time():
    in_flight = 0
    peak = 0

    class _Tracking(_FakeBackend):
        async def prefetch_image(self, image, nodes, tasks):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return PrefetchResult()

    backend = _Tracking("b")
    state = _state(
        [backend],
        {
            "b": [
                _srun_task("nim", ["n1"], container_image=NIM),
                _srun_task("bench", ["n2"], container_image=CLIENT),
            ]
        },
    )

    asyncio.run(prefetch_images(state))

    assert peak == 1


# -- config: the switch survives each backend's resolve_config ----------------


@pytest.mark.parametrize(
    "backend_conf",
    [
        {
            "type": "slurm",
            "account": "a",
            "partition": "p",
            "time": "00:10:00",
            "nodes": 1,
            "gpus_per_node": 8,
        },
        {"type": "docker", "image": "ubuntu:22.04", "nodes": 1, "gpus_per_node": 0},
        {"type": "kubernetes", "namespace": "ns", "nodes": 1, "gpus_per_node": 0},
        {"type": "local"},
    ],
    ids=lambda c: c["type"],
)
def test_prefetch_images_survives_resolve_and_defaults_off(backend_conf):
    from sflow.app.assembly import resolve_backends
    from sflow.config.schema import SflowConfig, TaskConfig, WorkflowConfig

    def _resolved(**extra):
        config = SflowConfig(
            version="0.1",
            backends=[{"name": "b", "default": True, **backend_conf, **extra}],
            workflow=WorkflowConfig(
                name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
            ),
        )
        state = SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))
        return resolve_backends(config, state).backends["b"].config

    assert _resolved(prefetch_images=True).prefetch_images is True
    assert _resolved().prefetch_images is False


# -- build_state: the step runs last, and only on a real run ------------------


def _patched_build_state(monkeypatch, backend, tasks):
    import sflow.app.assembly as assembly_mod
    from sflow.config.schema import SflowConfig, TaskConfig, WorkflowConfig

    async def _allocate(state):
        state.backends = {backend.name: backend}
        state.default_backend = backend
        return state

    def _graph(config, state, *, workspace_dir=None):
        tg = TaskGraph()
        for task in tasks:
            task.backend_name = backend.name
            tg.dag.add_node(task.name, task)
        return tg

    monkeypatch.setattr(assembly_mod, "resolve_backends", lambda c, s, **_: s)
    monkeypatch.setattr(
        assembly_mod, "preflight_validate_task_graph", lambda *a, **k: None
    )
    monkeypatch.setattr(assembly_mod, "allocate_backends", _allocate)
    monkeypatch.setattr(assembly_mod, "build_task_graph", _graph)
    config = SflowConfig(
        version="0.1",
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )
    return lambda **kw: asyncio.run(assembly_mod.build_state(config, **kw))


class _ReleasingBackend(_FakeBackend):
    released = False

    async def release(self, allocation):
        self.released = True


def test_build_state_prefetches_after_planning_and_releases_on_failure(monkeypatch):
    backend = _ReleasingBackend("b", failures=[["n1"]] * 3)
    build = _patched_build_state(
        monkeypatch, backend, [_srun_task("nim", ["n1"], container_image=NIM)]
    )

    with pytest.raises(image_prefetch_mod.ImagePrefetchError):
        build(allocate=True)

    assert len(backend.calls) == 3
    assert backend.released is True


def test_build_state_does_not_prefetch_without_allocation(monkeypatch):
    backend = _FakeBackend("b")
    build = _patched_build_state(
        monkeypatch, backend, [_srun_task("nim", ["n1"], container_image=NIM)]
    )

    build(allocate=False)

    assert backend.calls == []


# -- dry-run: shows the plan, pulls nothing -----------------------------------

_PREFETCH_DRY_RUN_CONFIG = f"""\
version: "0.1"
backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: a
    partition: p
    time: "00:10:00"
    nodes: 2
    gpus_per_node: 8
    prefetch_images: true
operators:
  - name: nim
    type: srun
    container_image: {NIM}
  - name: client
    type: srun
    container_image: {CLIENT}
workflow:
  name: prefetch_dry_run
  tasks:
    - name: server
      operator: nim
      resources:
        nodes:
          count: 2
      script:
        - echo serve
    - name: bench
      operator: client
      depends_on: [server]
      script:
        - echo bench
"""


def test_sflow_run_dry_run_lists_planned_prefetch_per_backend(tmp_path, caplog):
    from typer.testing import CliRunner

    from sflow.cli import app

    cfg = tmp_path / "sflow.yaml"
    cfg.write_text(_PREFETCH_DRY_RUN_CONFIG)

    with caplog.at_level(logging.INFO, logger="sflow"):
        result = CliRunner().invoke(
            app,
            [
                "run",
                "-f",
                str(cfg),
                "--workspace-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    messages = caplog.text + result.output
    assert "Image prefetch" in messages
    assert f"[slurm_cluster] {NIM} -> " in messages
    assert f"[slurm_cluster] {CLIENT} -> " in messages


def test_failed_nodes_the_plan_does_not_know_still_count_as_a_failure():
    # e.g. a backend that reports an FQDN for a node planned by its short name
    backend = _FakeBackend("b")
    backend.prefetch_image = _reporting(backend, ["n1.cluster.local"])
    state = _state(
        [backend], {"b": [_srun_task("nim", ["n1", "n2"], container_image=NIM)]}
    )

    asyncio.run(prefetch_images(state))

    assert [nodes for _, nodes, _ in backend.calls] == [["n1", "n2"], ["n1", "n2"]]


def _reporting(backend, first_failure):
    """prefetch_image that reports *first_failure* verbatim once, then succeeds."""
    pending = [list(first_failure)]

    async def _prefetch(image, nodes, tasks):
        backend.calls.append((image, list(nodes), [t.name for t in tasks]))
        failed = pending.pop(0) if pending else []
        return PrefetchResult(failed_nodes=failed, output="pull failed")

    return _prefetch
