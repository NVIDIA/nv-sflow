# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-step GPU remap must survive slurmstepd overwriting CUDA_VISIBLE_DEVICES.

These run the emitted shell for real: the logic only earns its keep if the
script sflow ships actually resolves to the right devices.
"""

import subprocess

import pytest

from sflow.utils.gpu import GPU_MARKER_FILE
from sflow.plugins.operators.srun import (
    SrunOperator,
    SrunOperatorConfig,
    _gpu_placement_prelude,
)


@pytest.fixture(autouse=True)
def allow_real_bash(fake_process):
    """tests/unit/conftest.py fakes all subprocesses; this suite needs a real shell."""
    fake_process.allow_unregistered(True)


def _run(plan: str, observed: str | None) -> subprocess.CompletedProcess:
    script = "\n".join(_gpu_placement_prelude(plan) + ['echo "$CUDA_VISIBLE_DEVICES"'])
    env = {"PATH": "/usr/bin:/bin"}
    if observed is not None:
        env["CUDA_VISIBLE_DEVICES"] = observed
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env
    )


@pytest.mark.parametrize(
    "plan, observed, expected",
    [
        # GRES cluster: slurmstepd hands every step the whole allocation, so each
        # task must narrow to its own planned slots. This is the OOM bug.
        ("0,1", "0,1,2,3", "0,1"),
        ("2,3", "0,1,2,3", "2,3"),
        ("3", "0,1,2,3", "3"),
        # Slots are positions, not raw ids: a partial allocation renumbers.
        ("0,1", "3,5,6,7", "3,5"),
        ("2,3", "3,5,6,7", "6,7"),
        # Non-GRES cluster: the step sees exactly what sflow exported. Unchanged.
        ("2,3", "2,3", "2,3"),
        # Slurm (or a container runtime) already carved the step to exactly this
        # task's share: keep its numbering.
        ("2,3", "6,7", "6,7"),
        # Nothing visible at all: fall back to the plan, as today.
        ("2,3", "", "2,3"),
    ],
)
def test_remap_selects_planned_devices(plan, observed, expected):
    result = _run(plan, observed)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_concurrent_tasks_get_disjoint_devices():
    """The whole point: two steps on one node must not land on the same GPU."""
    a_result = _run("0,1", "0,1,2,3")
    b_result = _run("2,3", "0,1,2,3")
    assert a_result.returncode == 0, a_result.stderr
    assert b_result.returncode == 0, b_result.stderr
    a = a_result.stdout.strip().split(",")
    b = b_result.stdout.strip().split(",")
    assert not set(a) & set(b)


def test_slot_outside_visible_devices_aborts():
    """Guessing a device is worse than failing loudly."""
    result = _run("6,7", "0,1,2,3,4")
    assert result.returncode == 97
    assert "outside CUDA_VISIBLE_DEVICES" in result.stderr


def test_fewer_devices_than_planned_aborts():
    """Under-provisioning must fail, not silently run on half the GPUs.

    A 4-GPU task handed 2 devices used to be accepted as "already carved" -- the
    same silent-wrong-placement failure this prelude exists to prevent.
    """
    result = _run("0,1,2,3", "0,1")
    assert result.returncode == 97
    assert "planned for 4" in result.stderr


def test_no_prelude_without_planned_gpus():
    assert _gpu_placement_prelude(None) == []
    assert _gpu_placement_prelude("") == []


def test_build_command_splices_prelude_into_step_body():
    op = SrunOperator(SrunOperatorConfig(name="t", log_to_file=False))
    cmd = op.build_command(
        task_name="worker",
        script=["python -c 'import torch'"],
        envs={"CUDA_VISIBLE_DEVICES": "2,3"},
    )
    body = cmd.as_list()[-1]
    assert "__sflow_plan='2,3'" in body
    # Must precede the user script: the remap is an export the task then inherits.
    assert body.index("__sflow_plan") < body.index("import torch")


def test_build_command_omits_prelude_for_cpu_task():
    op = SrunOperator(SrunOperatorConfig(name="t", log_to_file=False))
    cmd = op.build_command(task_name="cpu", script=["echo hi"], envs={})
    assert "__sflow_plan" not in cmd.as_list()[-1]


# The plan is interpolated into shell text. Every value sflow's planner emits is
# comma-joined non-negative ints, but `build_command` reads it from `envs`, where a
# workflow variable of the same name lands verbatim for a task with no
# `resources.gpus` -- so the value is not trusted just because sflow usually writes it.
@pytest.mark.parametrize(
    "hostile",
    [
        "0'; echo INJECTED; :'",   # closes the single quote and appends a command
        "0,1; rm -rf /",
        "$(touch /tmp/sflow_pwned)",
        "0,`id`",
        "GPU-8a1b2c3d",            # a real CUDA UUID form -- valid to CUDA, not a plan
        "0 1",
        "-1",
        "0,,1",
        "0,",
    ],
)
def test_non_numeric_plan_never_reaches_the_shell(hostile):
    assert _gpu_placement_prelude(hostile) == []


@pytest.mark.parametrize("plan", ["0", "2,3", "0,1,2,3", "10,11"])
def test_real_plans_are_still_emitted(plan):
    assert _gpu_placement_prelude(plan) != []


def test_injected_value_cannot_execute_a_command():
    """The end-to-end proof: run the emitted body and confirm nothing extra fires."""
    op = SrunOperator(SrunOperatorConfig(name="t", log_to_file=False))
    body = op.build_command(
        task_name="victim",
        script=["echo task-body"],
        envs={"CUDA_VISIBLE_DEVICES": "0'; echo INJECTED-COMMAND-RAN; :'"},
    ).as_list()[-1]
    result = subprocess.run(
        ["bash", "-c", body],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    )
    assert "INJECTED-COMMAND-RAN" not in result.stdout
    assert "task-body" in result.stdout, "the task itself must still run"


def test_step_reports_the_devices_it_actually_selected(tmp_path):
    """Run reporting must name the real cards, not the plan.

    On a GRES partition the step picks its devices positionally from whatever
    slurmstepd handed it, so a partial allocation makes plan `0,1` mean physical
    `3,5`. The driver only learns that if the step writes it down.
    """
    from types import SimpleNamespace

    from sflow.utils.gpu import GPU_MARKER_FILE, planned_gpu_indices, task_gpu_indices

    plan = "0,1"
    body = "\n".join(_gpu_placement_prelude(plan))
    result = subprocess.run(
        ["bash", "-c", body],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "CUDA_VISIBLE_DEVICES": "3,5,6,7",   # partial allocation
            "SFLOW_TASK_OUTPUT_DIR": str(tmp_path),
        },
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / GPU_MARKER_FILE).read_text().strip() == "3,5"

    task = SimpleNamespace(
        cuda_visible_devices=plan,
        envs={"CUDA_VISIBLE_DEVICES": plan, "SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
    )
    assert task_gpu_indices(task) == [3, 5], "reporting must use the real devices"
    assert planned_gpu_indices(task) == [0, 1], "the plan view stays the plan"


def test_reporting_falls_back_to_the_plan_without_a_marker(tmp_path):
    """Docker/k8s and pre-existing runs write no marker and must be unaffected."""
    from types import SimpleNamespace

    from sflow.utils.gpu import task_gpu_indices

    task = SimpleNamespace(
        cuda_visible_devices="4,5",
        envs={"CUDA_VISIBLE_DEVICES": "4,5", "SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
    )
    assert task_gpu_indices(task) == [4, 5]

    docker = SimpleNamespace(
        reserved_gpu_indices=[6, 7],
        cuda_visible_devices="0,1",
        envs={"CUDA_VISIBLE_DEVICES": "0,1", "SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
    )
    assert task_gpu_indices(docker) == [6, 7], "a launch-time reservation still wins"


def test_unparseable_marker_falls_back_to_the_plan(tmp_path):
    """A marker sflow cannot parse must not delete the task from run reporting.

    `parse_cuda_visible_devices` ignores UUID forms, and the step copies whatever it
    observed -- so an empty parse is "I don't know", not "no GPUs". The summary skips
    any task with no ids, so returning [] here silently drops it from both the GPU
    Assignment table and the usage chart.
    """
    from types import SimpleNamespace

    from sflow.utils.gpu import GPU_MARKER_FILE, task_gpu_indices

    (tmp_path / GPU_MARKER_FILE).write_text("GPU-8a1b2c3d\n")
    task = SimpleNamespace(
        cuda_visible_devices="2,3",
        envs={"CUDA_VISIBLE_DEVICES": "2,3", "SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
        assigned_nodes=["node01"],
    )
    assert task_gpu_indices(task) == [2, 3]


def test_only_rank_zero_writes_the_marker(tmp_path):
    """Every rank runs this body; with --gpus-per-task they hold different devices.

    Letting them all truncate one path makes the reported devices a coin flip.
    """
    plan = "0,1"
    body = "\n".join(_gpu_placement_prelude(plan))

    def rank(procid, seen):
        return subprocess.run(
            ["bash", "-c", body],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "CUDA_VISIBLE_DEVICES": seen,
                "SLURM_PROCID": procid,
                "SFLOW_TASK_OUTPUT_DIR": str(tmp_path),
            },
        )

    marker = tmp_path / GPU_MARKER_FILE
    assert rank("1", "4,5,6,7").returncode == 0
    assert not marker.exists(), "a non-zero rank must not write the marker"

    assert rank("0", "0,1,2,3").returncode == 0
    assert marker.read_text().strip() == "0,1"

    # A later non-zero rank must not clobber rank 0's value.
    assert rank("2", "4,5,6,7").returncode == 0
    assert marker.read_text().strip() == "0,1"


def test_multi_node_task_reports_the_plan_not_one_node(tmp_path):
    """The marker is one step's view; a flat list cannot speak for several nodes."""
    from types import SimpleNamespace

    from sflow.utils.gpu import GPU_MARKER_FILE, task_gpu_indices

    (tmp_path / GPU_MARKER_FILE).write_text("6,7\n")
    envs = {"CUDA_VISIBLE_DEVICES": "0,1", "SFLOW_TASK_OUTPUT_DIR": str(tmp_path)}
    single = SimpleNamespace(
        cuda_visible_devices="0,1", envs=envs, assigned_nodes=["node01"]
    )
    multi = SimpleNamespace(
        cuda_visible_devices="0,1", envs=envs, assigned_nodes=["node01", "node02"]
    )
    assert task_gpu_indices(single) == [6, 7], "single node: trust what the step saw"
    assert task_gpu_indices(multi) == [0, 1], "multi node: fall back to the plan"


def test_multi_node_step_writes_no_marker(tmp_path):
    """The writer's rule must match the reader's, or the file is a lie on disk.

    Reporting discounts the marker for a multi-node task, so writing it there leaves
    an artifact that is right for node 0 and wrong for every other node -- and looks
    authoritative to anyone who opens the task output dir.
    """
    body = "\n".join(_gpu_placement_prelude("0,1,2,3"))

    def step(nnodes):
        return subprocess.run(
            ["bash", "-c", body],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "SLURM_PROCID": "0",
                "SLURM_NNODES": nnodes,
                "SFLOW_TASK_OUTPUT_DIR": str(tmp_path),
            },
        )

    assert step("2").returncode == 0, "a multi-node step must still run normally"
    assert not (tmp_path / GPU_MARKER_FILE).exists()

    assert step("1").returncode == 0
    assert (tmp_path / GPU_MARKER_FILE).read_text().strip() == "0,1,2,3"


def test_prelude_carries_a_short_banner_but_not_the_rationale():
    """The generated command explains itself in a few lines, not ten.

    Someone reading a failing srun line needs to know why sflow is rewriting
    CUDA_VISIBLE_DEVICES; they do not need the marker-write design notes, which
    live in the module instead.
    """
    body = _gpu_placement_prelude("1,3")[0]
    comments = [line for line in body.splitlines() if line.startswith("#")]

    # Wrapped, so it is obvious where sflow's block stops and the task's starts.
    assert body.startswith("# --- sflow GPU placement (begin)")
    assert body.rstrip("\n").endswith("# --- sflow GPU placement (end) ---------------------------------------------")
    assert any("slurmstepd rewrites CUDA_VISIBLE_DEVICES" in c for c in comments)
    # Thin: a banner, not an essay.
    assert len(comments) <= 8, comments
    # The long rationale stays out of the shipped shell text.
    assert "rank 0 only" not in body
    assert "Best-effort" not in body


def test_banner_does_not_break_the_emitted_shell():
    """A comment block is inert, but it is spliced into a `bash -c` body."""
    body = _gpu_placement_prelude("2")[0]
    result = subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

