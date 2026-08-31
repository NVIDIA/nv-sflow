# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-step GPU remap must survive slurmstepd overwriting CUDA_VISIBLE_DEVICES.

These run the emitted shell for real: the logic only earns its keep if the
script sflow ships actually resolves to the right devices.
"""

import atexit
import itertools
import shutil
import subprocess
import tempfile
from pathlib import Path

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


def _no_driver_dir() -> str:
    """A PATH prefix whose `nvidia-smi` reports nothing.

    The prelude ALWAYS asks the driver now, so without this every test here would
    measure whatever GPU the developer's machine happens to have -- this one has
    an RTX 3090, which silently rewrote five expectations. Tests that want a
    driver state it explicitly via `_run_in_container`.
    """
    d = Path(tempfile.mkdtemp(prefix="sflow_no_driver_"))
    smi = d / "nvidia-smi"
    smi.write_text("#!/bin/bash\nexit 1\n")
    smi.chmod(0o755)
    return str(d)


_NO_DRIVER = _no_driver_dir()
# The placement script is staged into the workflow output dir and sourced from
# there -- that is the only delivery path -- so tests need a dir to stage into.
_STAGE = tempfile.mkdtemp(prefix="sflow_stage_")
# Module-level, so pytest's tmp_path machinery never sees them: clean up by hand
# or every run leaks two /tmp directories.
atexit.register(shutil.rmtree, _NO_DRIVER, ignore_errors=True)
atexit.register(shutil.rmtree, _STAGE, ignore_errors=True)


@pytest.fixture
def no_driver():
    return _NO_DRIVER


def _run(plan: str, observed: str | None, path_prefix=None) -> subprocess.CompletedProcess:
    script = "\n".join(_gpu_placement_prelude(plan, workflow_out_dir=_STAGE) + ['echo "$CUDA_VISIBLE_DEVICES"'])
    prefix = f"{path_prefix or _NO_DRIVER}:"
    env = {"PATH": f"{prefix}/usr/bin:/bin"}
    if observed is not None:
        env["CUDA_VISIBLE_DEVICES"] = observed
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env
    )


def _run_in_container(
    plan: str, visible_gpus: int, tmp_path, observed: str | None = None
) -> subprocess.CompletedProcess:
    """Run the prelude against a driver reporting N GPUs numbered 0..N-1.

    This is the pyxis/enroot shape: the runtime passes through only this task's
    devices and renumbers them from 0. ``observed`` is what --export=ALL carried
    into the step, which is usually sflow's own plan in HOST ordinals.
    """
    smi = tmp_path / "nvidia-smi"
    lines = "\n".join(
        f"GPU {i}: NVIDIA GB200 (UUID: GPU-{i:08x})" for i in range(visible_gpus)
    )
    pairs = "\n".join(f"{i}, GPU-{i:08x}" for i in range(visible_gpus))
    # Answers both forms: `-L` for the legacy path, `--query-gpu=index,uuid` for
    # the in-step probe that records the index the driver actually reports.
    smi.write_text(
        "#!/bin/bash\n"
        'if [ "$*" != "${*/index,uuid/}" ]; then\n'
        f"cat <<'EOF'\n{pairs}\nEOF\n"
        "else\n"
        f"cat <<'EOF'\n{lines}\nEOF\n"
        "fi\n"
    )
    smi.chmod(0o755)
    script = "\n".join(_gpu_placement_prelude(plan, workflow_out_dir=_STAGE) + ['echo "$CUDA_VISIBLE_DEVICES"'])
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin"}
    if observed is not None:
        env["CUDA_VISIBLE_DEVICES"] = observed
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env
    )


@pytest.mark.parametrize(
    "plan, visible_gpus, expected",
    [
        # A container carved to this task's 2 GPUs numbers them 0,1, so host
        # ordinals 2,3 name devices that do not exist here.
        ("2,3", 2, "0,1"),
        # The task that accidentally worked: its plan already matched the
        # renumbering, which is why only the decode server ever failed.
        ("0,1", 2, "0,1"),
        # Single-GPU carve, high host ordinal.
        ("3", 1, "0"),
        # NOT carved: the whole node is visible with the variable unset, so the
        # positional narrowing must still happen or concurrent tasks collide.
        ("2,3", 4, "2,3"),
        ("0,1", 4, "0,1"),
    ],
)
def test_container_carve_keeps_the_containers_own_numbering(
    plan, visible_gpus, expected, tmp_path
):
    result = _run_in_container(plan, visible_gpus, tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    "plan, observed, visible_gpus, expected",
    [
        # THE BUG, exactly as it reached the cluster. srun runs --export=ALL, so
        # sflow's own plan arrives in the step as CUDA_VISIBLE_DEVICES. Trusting it
        # as an observation is circular: the counts match the plan by construction,
        # so the remap concluded "already carved, keep what I see" and kept host
        # ordinals 2,3 inside a container holding only 0,1 -> no GPU at all.
        ("2,3", "2,3", 2, "0,1"),
        # The task that accidentally worked: plan already matched the renumbering.
        ("0,1", "0,1", 2, "0,1"),
        # Single GPU carve, high host ordinal echoed back.
        ("3", "3", 1, "0"),
        # NOT a container: slurmstepd set a real in-range slice on a 4-GPU node.
        # That IS an observation and must be honoured, or the GRES fix regresses.
        ("0,1", "0,1", 4, "0,1"),
        ("2,3", "2,3", 4, "2,3"),
        # Whole allocation visible and handed over: still narrow positionally.
        ("2,3", "0,1,2,3", 4, "2,3"),
        ("0,1", "3,5,6,7", 8, "3,5"),
        # UUID-form CUDA_VISIBLE_DEVICES carries no positions to remap against.
        ("2,3", "GPU-abc,GPU-def", 2, "0,1"),
    ],
)
def test_inherited_cuda_visible_devices_is_validated_against_the_driver(
    plan, observed, visible_gpus, expected, tmp_path
):
    result = _run_in_container(plan, visible_gpus, tmp_path, observed=observed)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_no_driver_falls_back_to_the_plan(no_driver):
    """With no driver to ask, behave exactly as before this probe existed."""
    result = _run("2,3", None, path_prefix=no_driver)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2,3"


def test_no_driver_says_so_instead_of_degrading_silently(no_driver):
    """A GPU task with no nvidia-smi loses the container-renumbering check.

    That is the ptyche failure restored: the inherited value is trusted, and a
    container holding 0,1 keeps a plan of 2,3 and sees no GPU. It is the right
    fallback -- guessing is worse -- but a slim image must not lose the protection
    without saying so, or the only symptom is a task that dies inside CUDA.
    """
    result = _run("2,3", "2,3", path_prefix=no_driver)

    assert result.returncode == 0
    assert "no nvidia-smi here" in result.stderr
    assert "placement may be wrong" in result.stderr
    # The warning must not pollute what the task reads.
    assert result.stdout.strip() == "2,3"


def test_a_known_plan_never_narrows_less_than_an_unknown_one(no_driver, tmp_path):
    """No nvidia-smi + planned UUIDs must still narrow, not bail.

    Knowing MORE about a step must never make sflow do LESS to it. Without a
    probe the planned UUIDs cannot be checked, and it is right to record the
    placement as unproven -- but returning early there left
    CUDA_VISIBLE_DEVICES exactly as inherited, which on a GRES partition is the
    whole allocation. Every concurrent worker then picks ordinal 0 and collides
    on one physical GPU: precisely the OOM this prelude exists to prevent, and it
    only happened when a UUID map was available. A task with NO map, in the same
    container on the same partition, was narrowed correctly.

    So the no-probe case degrades to the same index arithmetic instead of
    stopping. `action` still says `unverified`, so the e2e audit keeps counting
    it as unproven.
    """
    out = tmp_path / "out"
    out.mkdir()
    script = "\n".join(
        _gpu_placement_prelude("2,3", workflow_out_dir=_STAGE)
        + ['echo "CVD=$CUDA_VISIBLE_DEVICES"']
    )
    env = {
        "PATH": f"{no_driver}:/usr/bin:/bin",
        "SLURMD_NODENAME": "n0",
        "SFLOW_TASK_OUTPUT_DIR": str(out),
        # Slurm handed this step the whole 4-GPU allocation, as a GRES step with
        # no --gres of its own always is.
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "SLURM_STEP_GPUS": "0,1,2,3",
        "SFLOW_PLANNED_GPU_UUIDS": "n0=GPU-c,GPU-d",
    }
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)

    assert r.returncode == 0, r.stderr
    assert "CVD=2,3" in r.stdout, "must narrow to its own slice, not keep all four"
    lines = (out / GPU_MARKER_FILE).read_text().splitlines()
    assert "action=unverified" in lines, "narrowed, but not PROVEN -- both are true"


def test_a_fallback_record_keeps_the_uuids_it_could_not_find(tmp_path, fp):
    """`fallback` must say what was wanted, not claim it never knew.

    When Slurm grants cards the planner never saw, the planned UUIDs WERE
    resolved -- they just are not in the grant. A record saying
    `planned_uuids=(not resolved)` sends the reader after the driver probe, which
    worked fine, instead of at the grant, which is the actual finding.
    """
    fp.allow_unregistered(True)

    out = tmp_path / "out"
    out.mkdir()
    script = "\n".join(_gpu_placement_prelude("2,3", workflow_out_dir=_STAGE))
    r = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{_smi_stub(tmp_path, ['GPU-x', 'GPU-y'])}:/usr/bin:/bin",
            "SLURMD_NODENAME": "n0",
            "SFLOW_TASK_OUTPUT_DIR": str(out),
            "CUDA_VISIBLE_DEVICES": "0,1",
            "SLURM_STEP_GPUS": "6,7",
            "SFLOW_PLANNED_GPU_UUIDS": "n0=GPU-c,GPU-d",
        },
    )
    assert r.returncode == 0, r.stderr

    lines = (out / GPU_MARKER_FILE).read_text().splitlines()
    assert "action=fallback" in lines
    assert "planned_uuids=GPU-c,GPU-d" in lines, "resolved, just not present"
    # The reason belongs in the record, not only in a step log nobody greps.
    assert any("not among the devices Slurm granted" in ln for ln in lines)
    # ...and what it landed on instead is right there to compare against.
    assert "selected=0 GPU-x" in lines and "selected=1 GPU-y" in lines


def test_no_warning_when_the_driver_answers(tmp_path):
    result = _run_in_container("2,3", 2, tmp_path, observed="2,3")

    assert result.returncode == 0
    assert "no nvidia-smi" not in result.stderr


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
def test_remap_selects_planned_devices(plan, observed, expected, no_driver):
    result = _run(plan, observed, path_prefix=no_driver)
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

    This is only a real fault while the step holds the task's WHOLE slice. Under
    --gpus-per-task it is the normal shape, which is why no prelude is emitted
    there (see test_gpus_per_task_leaves_placement_to_slurm).
    """
    result = _run("0,1,2,3", "0,1")
    assert result.returncode == 97
    assert "planned for 4" in result.stderr


@pytest.mark.parametrize("gpus_per_task", ["1", "2"])
def test_gpus_per_task_leaves_placement_to_slurm(gpus_per_task):
    """Per-rank carving breaks the prelude's premise, so it must step aside.

    --gpus-per-task makes the step REQUEST GRES, so Slurm hands each rank only its
    own devices instead of handing the step the whole allocation. Counting the
    task's full slice against one rank's view then aborts every rank of a perfectly
    valid config: 8 ranks at 1 GPU each died with "step has 1 GPU(s) but this task
    was planned for 8". Slurm keeps those per-rank sets disjoint itself, so there is
    nothing to re-apply.
    """
    assert _gpu_placement_prelude("0,1,2,3,4,5,6,7", gpus_per_task=gpus_per_task, workflow_out_dir=_STAGE) == []
    # Without the flag the same plan is still enforced.
    assert _gpu_placement_prelude("0,1,2,3,4,5,6,7", workflow_out_dir=_STAGE) != []


def test_build_command_omits_prelude_under_gpus_per_task():
    op = SrunOperator(
        SrunOperatorConfig(
            name="t", log_to_file=False, ntasks_per_node=8, gpus_per_task="1"
        )
    )
    body = op.build_command(
        task_name="ranks",
        script=["torchrun train.py"],
        envs={
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            # Staging must be POSSIBLE here, or this would pass via the
            # "nowhere to stage it" path instead of the gpus_per_task guard.
            "SFLOW_WORKFLOW_OUTPUT_DIR": _STAGE,
        },
    ).as_list()[-1]

    assert "SFLOW_GPU_PLAN" not in body
    assert "torchrun train.py" in body, "the task itself must still run"


def test_many_ranks_sharing_the_task_slice_all_get_it():
    """The shape the samples actually use: ntasks_per_node with no per-rank carving.

    Every rank sees the whole step allocation and must narrow to the same planned
    slice -- the app then picks its device by local rank.
    """
    selected = {
        _run("0,1,2,3", "0,1,2,3,4,5,6,7").stdout.strip() for _ in range(4)
    }
    assert selected == {"0,1,2,3"}


def test_no_prelude_without_planned_gpus():
    assert _gpu_placement_prelude(None, workflow_out_dir=_STAGE) == []
    assert _gpu_placement_prelude("", workflow_out_dir=_STAGE) == []


def test_build_command_splices_prelude_into_step_body():
    op = SrunOperator(SrunOperatorConfig(name="t", log_to_file=False))
    cmd = op.build_command(
        task_name="worker",
        script=["python -c 'import torch'"],
        envs={
            "CUDA_VISIBLE_DEVICES": "2,3",
            # The script is staged here and sourced from there.
            "SFLOW_WORKFLOW_OUTPUT_DIR": _STAGE,
        },
    )
    body = cmd.as_list()[-1]
    assert "export SFLOW_GPU_PLAN='2,3'" in body
    # Must precede the user script: the placement is an export the task inherits.
    assert body.index("SFLOW_GPU_PLAN") < body.index("import torch")


def test_build_command_omits_prelude_for_cpu_task():
    op = SrunOperator(SrunOperatorConfig(name="t", log_to_file=False))
    cmd = op.build_command(task_name="cpu", script=["echo hi"], envs={})
    assert "SFLOW_GPU_PLAN" not in cmd.as_list()[-1]


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
    assert _gpu_placement_prelude(hostile, workflow_out_dir=_STAGE) == []


@pytest.mark.parametrize("plan", ["0", "2,3", "0,1,2,3", "10,11"])
def test_real_plans_are_still_emitted(plan):
    assert _gpu_placement_prelude(plan, workflow_out_dir=_STAGE) != []


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
        env={"PATH": f"{_NO_DRIVER}:/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
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
    body = "\n".join(_gpu_placement_prelude(plan, workflow_out_dir=_STAGE))
    result = subprocess.run(
        ["bash", "-c", body],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{_NO_DRIVER}:/usr/bin:/bin",
            "CUDA_VISIBLE_DEVICES": "3,5,6,7",   # partial allocation
            "SFLOW_TASK_OUTPUT_DIR": str(tmp_path),
        },
    )
    assert result.returncode == 0, result.stderr
    assert _marker_devices(tmp_path) == "3,5"

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
    """Every rank runs this body, so letting them all truncate one path is a race.

    Ranks that were handed different partial allocations resolve to different
    devices, which makes the reported value a coin flip.
    """
    plan = "0,1"
    body = "\n".join(_gpu_placement_prelude(plan, workflow_out_dir=_STAGE))

    def rank(procid, seen):
        return subprocess.run(
            ["bash", "-c", body],
            capture_output=True,
            text=True,
            env={
                "PATH": f"{_NO_DRIVER}:/usr/bin:/bin",
                "CUDA_VISIBLE_DEVICES": seen,
                "SLURM_PROCID": procid,
                "SFLOW_TASK_OUTPUT_DIR": str(tmp_path),
            },
        )

    marker = tmp_path / GPU_MARKER_FILE
    assert rank("1", "4,5,6,7").returncode == 0
    assert not marker.exists(), "a non-zero rank must not write the marker"

    assert rank("0", "0,1,2,3").returncode == 0
    assert marker.read_text().splitlines()[0].strip() == "0,1"

    # A later non-zero rank must not clobber rank 0's value.
    assert rank("2", "4,5,6,7").returncode == 0
    assert marker.read_text().splitlines()[0].strip() == "0,1"


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


def _marker_step(tmp_path, **slurm_env) -> subprocess.CompletedProcess:
    body = "\n".join(_gpu_placement_prelude("0,1,2,3", workflow_out_dir=_STAGE))
    return subprocess.run(
        ["bash", "-c", body],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{_NO_DRIVER}:/usr/bin:/bin",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "SLURM_PROCID": "0",
            "SFLOW_TASK_OUTPUT_DIR": str(tmp_path),
            **slurm_env,
        },
    )


def test_multi_node_step_writes_no_marker(tmp_path):
    """The writer's rule must match the reader's, or the file is a lie on disk.

    Reporting discounts the marker for a multi-node task, so writing it there leaves
    an artifact that is right for node 0 and wrong for every other node -- and looks
    authoritative to anyone who opens the task output dir.
    """
    assert (
        _marker_step(tmp_path, SLURM_STEP_NUM_NODES="2").returncode == 0
    ), "a multi-node step must still run normally"
    assert not (tmp_path / GPU_MARKER_FILE).exists()

    assert _marker_step(tmp_path, SLURM_STEP_NUM_NODES="1").returncode == 0
    assert _marker_devices(tmp_path) == "0,1,2,3"


def test_allocation_wide_node_count_does_not_suppress_the_marker(tmp_path):
    """The guard is about the STEP's nodes, not the allocation's.

    SLURM_NNODES is the backwards-compat alias of SLURM_JOB_NUM_NODES, and
    SlurmBackend.resource_env copies the driver's SLURM_* into the env handed to
    srun -- so a 4-node allocation puts SLURM_NNODES=4 in every step, including
    single-node ones. Reading it here silently killed the marker for every task on
    any multi-node allocation, which is exactly where concurrent single-node tasks
    get devices the plan cannot predict.
    """
    result = _marker_step(tmp_path, SLURM_NNODES="4", SLURM_STEP_NUM_NODES="1")

    assert result.returncode == 0, result.stderr
    assert _marker_devices(tmp_path) == "0,1,2,3"


def test_multi_node_task_applies_its_per_node_slice_on_every_node(tmp_path):
    """A multi-node plan is ONE node's slice, and each node resolves it alone.

    The planner guarantees this: pinned indices repeat per node, the multi-node
    count path divides by the node count and refuses nodes with different
    allocation cursors. So the same flat plan is evaluated independently in every
    node's step, and each must land on its own node's devices -- including when the
    nodes were handed different partial allocations.
    """
    plan = "0,1"  # gpus.count=4 over 2 nodes -> 2 per node
    body = "\n".join(_gpu_placement_prelude(plan, workflow_out_dir=_STAGE) + ['echo "$CUDA_VISIBLE_DEVICES"'])

    def node(seen, procid, nodeid):
        return subprocess.run(
            ["bash", "-c", body],
            capture_output=True,
            text=True,
            env={
                "PATH": f"{_NO_DRIVER}:/usr/bin:/bin",
                "CUDA_VISIBLE_DEVICES": seen,
                "SLURM_PROCID": procid,
                "SLURM_NODEID": nodeid,
                "SLURM_STEP_NUM_NODES": "2",
                "SFLOW_TASK_OUTPUT_DIR": str(tmp_path),
            },
        )

    # Both nodes handed the whole 4-GPU node: narrow to the planned slots.
    assert node("0,1,2,3", "0", "0").stdout.strip() == "0,1"
    assert node("0,1,2,3", "1", "1").stdout.strip() == "0,1"
    # Node 1 handed a partial allocation: slots are positions, so it follows.
    assert node("4,5,6,7", "1", "1").stdout.strip() == "4,5"
    # And no node leaves a marker that would speak for the others.
    assert not (tmp_path / GPU_MARKER_FILE).exists()


def test_prelude_carries_a_short_banner_but_not_the_rationale():
    """The generated command explains itself in a few lines, not ten.

    Someone reading a failing srun line needs to know why sflow is rewriting
    CUDA_VISIBLE_DEVICES; they do not need the marker-write design notes, which
    live in the module instead.
    """
    body = _gpu_placement_prelude("1,3", workflow_out_dir=_STAGE)[0]
    comments = [line for line in body.splitlines() if line.startswith("#")]

    # Wrapped, so it is obvious where sflow's block stops and the task's starts.
    assert body.startswith("# --- sflow GPU placement (begin)")
    assert body.rstrip("\n").endswith("# --- sflow GPU placement (end) ---------------------------------------------")
    # Names what it does and the two shapes that need it -- a reader staring at a
    # failing srun line must not conclude this block is what broke their GPUs.
    assert any("planned GPUs" in c for c in comments)
    assert any("GRES" in c for c in comments)
    assert any("container" in c for c in comments)
    # Thin: a banner, not an essay.
    assert len(comments) <= 8, comments
    # The long rationale stays out of the shipped shell text.
    assert "rank 0 only" not in body
    assert "Best-effort" not in body


def test_banner_does_not_break_the_emitted_shell():
    """A comment block is inert, but it is spliced into a `bash -c` body."""
    body = _gpu_placement_prelude("2", workflow_out_dir=_STAGE)[0]
    result = subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr



# ---------------------------------------------------------------------------
# UUID-verified placement: compare against the plan, and act only if it differs.
# ---------------------------------------------------------------------------


def _marker_devices(out_dir) -> str:
    """The device list from a marker file.

    Line 1 is the bare list; the lines after it are the audit record (node, the
    action taken, every visible index -> UUID, and what CUDA ended up selecting).
    Keeping the list first is the contract utils.gpu.task_gpu_indices relies on.
    """
    return (out_dir / GPU_MARKER_FILE).read_text().splitlines()[0].strip()


def _smi_stub(tmp_path, uuids: list[str]):
    """An nvidia-smi that answers BOTH forms the prelude uses."""
    smi = tmp_path / "nvidia-smi"
    listing = "\n".join(f"GPU {i}: Fake (UUID: {u})" for i, u in enumerate(uuids))
    plain = "\n".join(uuids)
    pairs = "\n".join(f"{i}, {u}" for i, u in enumerate(uuids))
    smi.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-L" ]; then\n'
        f"cat <<'EOF'\n{listing}\nEOF\n"
        'elif [ "$*" != "${*/index,uuid/}" ]; then\n'
        f"cat <<'EOF'\n{pairs}\nEOF\n"
        "else\n"
        f"cat <<'EOF'\n{plain}\nEOF\n"
        "fi\n"
    )
    smi.chmod(0o755)
    return smi.parent


_RUN_SEQ = [0]


def _run_verified(
    tmp_path, *, plan: str, visible: list[str], observed: str | None, planned_map: str,
    node: str = "nodeA",
):
    """Render the real prelude and report the resulting CVD and the branch taken.

    `action` from the placement record is the authoritative signal for which
    branch ran -- verified / fallback / unverified. (It used to be
    inferred from NVIDIA_VISIBLE_DEVICES being set, but the prelude no longer
    writes that: the container runtime owns it, and it is consumed at container
    creation, so writing it afterwards states something false about another layer.)
    """
    _RUN_SEQ[0] += 1
    out_dir = tmp_path / f"run{_RUN_SEQ[0]}"
    out_dir.mkdir()
    script = "\n".join(
        _gpu_placement_prelude(plan, workflow_out_dir=_STAGE)
        + ['echo "CVD=${CUDA_VISIBLE_DEVICES-<unset>}"']
    )
    env = {
        "PATH": f"{_smi_stub(tmp_path, visible)}:/usr/bin:/bin",
        "SLURMD_NODENAME": node,
        "SFLOW_PLANNED_GPU_UUIDS": planned_map,
        "SFLOW_TASK_OUTPUT_DIR": str(out_dir),
    }
    if observed is not None:
        env["CUDA_VISIBLE_DEVICES"] = observed
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    cvd = next(
        (ln.split("=", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("CVD=")), None
    )
    action = None
    marker = out_dir / GPU_MARKER_FILE
    if marker.exists():
        action = next(
            (ln.split("=", 1)[1] for ln in marker.read_text().splitlines()
             if ln.startswith("action=")), None
        )
    return r, cvd, action


def test_placement_names_the_planned_cards_by_uuid_in_every_shape(tmp_path, fp):
    """One rule covers every shape: look the planned UUIDs up, name their indices.

    There is deliberately no separate no-op / pin / narrow / re-select branch.
    Each of those was the SAME question -- "which indices do the planned cards
    have HERE?" -- answered against a different number of visible devices, so
    they collapse into one lookup. Whether the answer differs from what was
    inherited is a fact the record shows (inherited vs final), not a mode.

    Identity is the UUID, never the index: a device index stops being an identity
    the moment a container renumbers from 0.
    """
    fp.allow_unregistered(True)  # drives a real `bash` subprocess

    want = "nodeA=GPU-c,GPU-d"

    # Whole node visible, inherited value already correct -> same answer back.
    _, cvd, action = _run_verified(
        tmp_path, plan="2,3", visible=["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
        observed="2,3", planned_map=want,
    )
    assert (cvd, action) == ("2,3", "verified"), "an already-correct slice must survive"

    # A carved container whose plan happens to start at 0. (This is the shape
    # that survived by coincidence before any of this existed.)
    _, cvd, action = _run_verified(
        tmp_path, plan="0,1", visible=["GPU-c", "GPU-d"],
        observed="0,1", planned_map="nodeA=GPU-c,GPU-d",
    )
    assert (cvd, action) == ("0,1", "verified")

    # Carved, and CUDA_VISIBLE_DEVICES not set at all. The cards are right --
    # unset means "every visible device" -- but leaving it unset is not safe:
    # recipes read this variable to derive ranks, port offsets and device counts,
    # and under `set -u` an unset one is a hard error (this killed bare_count_8
    # on perfwg). The same cards get named explicitly, which changes nothing
    # about which are used.
    _, cvd, action = _run_verified(
        tmp_path, plan="2,3", visible=["GPU-c", "GPU-d"], observed=None, planned_map=want,
    )
    assert (cvd, action) == ("0,1", "verified"), "an unset CVD must be named, not left unset"

    # The regression this exists for: host ordinals inherited into a 2-GPU
    # container name nothing there, so they resolve to 0,1.
    _, cvd, action = _run_verified(
        tmp_path, plan="2,3", visible=["GPU-c", "GPU-d"], observed="2,3", planned_map=want,
    )
    assert (cvd, action) == ("0,1", "verified")

    # More visible than planned, nothing selected: the lookup narrows.
    _, cvd, action = _run_verified(
        tmp_path, plan="2,3", visible=["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
        observed=None, planned_map=want,
    )
    assert (cvd, action) == ("2,3", "verified")

    # Holding the right NUMBER of the WRONG cards is the failure a count-based
    # check cannot see. It must be loud, not a silent pass.
    r, _, _ = _run_verified(
        tmp_path, plan="2,3", visible=["GPU-a", "GPU-b"], observed="2,3", planned_map=want,
    )
    assert r.returncode == 97
    assert "is not visible" in r.stderr


def test_planned_uuids_are_resolved_per_node_not_per_task(tmp_path, fp):
    """One flat plan, different physical cards on each node.

    The plan is a list of HOST indices applied identically on every node a task
    spans, so slot 2 is a different card on node B than on node A. A step must
    read its OWN node's entry; reading another node's would "verify" against
    hardware it is not running on.
    """
    fp.allow_unregistered(True)

    both = "nodeA=GPU-a2,GPU-a3;nodeB=GPU-b2,GPU-b3"
    # On node B the same plan must resolve to B's cards -- and since they are
    # already the visible ones, nothing is rewritten.
    _, cvd, action = _run_verified(
        tmp_path, plan="2,3", visible=["GPU-b2", "GPU-b3"], observed="2,3",
        planned_map=both, node="nodeB",
    )
    assert (cvd, action) == ("0,1", "verified")

    # Node B seeing node A's cards is a real placement error.
    r, _, _ = _run_verified(
        tmp_path, plan="2,3", visible=["GPU-a2", "GPU-a3"], observed="2,3",
        planned_map=both, node="nodeB",
    )
    assert r.returncode == 97


def test_without_a_uuid_map_the_previous_behaviour_is_unchanged(tmp_path, fp):
    """Clusters that cannot be probed must keep working exactly as before.

    No SFLOW_PLANNED_GPU_UUIDS -> the old index arithmetic runs, untouched.
    """
    fp.allow_unregistered(True)

    _, cvd, action = _run_verified(
        tmp_path, plan="2,3", visible=["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
        observed="2,3", planned_map="",
    )
    assert (cvd, action) == ("2,3", "fallback"), "legacy path still runs"

    # And the legacy container remap still happens.
    _, cvd, _ = _run_verified(
        tmp_path, plan="2,3", visible=["GPU-c", "GPU-d"], observed="2,3", planned_map="",
    )
    assert cvd == "0,1"


def test_planned_uuid_map_is_per_backend_and_refuses_partial_answers():
    """Encoding the plan as physical cards, per node.

    Two Slurm backends can have different gpus_per_node and different node sets,
    so this is built from the calling backend's OWN allocation. A node whose
    topology is unknown, or whose device count cannot contain a planned slot, is
    omitted entirely: a partial map would let a step "verify" against a reading
    that cannot hold the card it was planned for, which is worse than falling back.
    """
    from sflow.core.backend import Allocation
    from sflow.core.compute_node import ComputeNode
    from sflow.plugins.backends.slurm import _planned_gpu_uuids

    def node(name, uuids):
        return ComputeNode(
            name=name, ip_address="1.2.3.4", index=0, num_gpus=len(uuids or []),
            gpu_uuids=uuids,
        )

    alloc = Allocation(
        allocation_id="1",
        nodes=[node("n0", ["A0", "A1", "A2", "A3"]), node("n1", ["B0", "B1", "B2", "B3"])],
        owned=False,
    )
    # The SAME flat plan resolves to different physical cards per node.
    assert _planned_gpu_uuids("2,3", alloc) == "n0=A2,A3;n1=B2,B3"
    # Order is the plan's order, not the device order.
    assert _planned_gpu_uuids("3,0", alloc) == "n0=A3,A0;n1=B3,B0"

    # Nothing trustworthy to say -> say nothing, and the step falls back.
    assert _planned_gpu_uuids(None, alloc) == ""
    assert _planned_gpu_uuids("2,3", None) == ""
    assert _planned_gpu_uuids("9", alloc) == "", "a slot past the device count"
    assert _planned_gpu_uuids("GPU-abc", alloc) == "", "UUID-form plan"
    assert _planned_gpu_uuids("2,3", Allocation(
        allocation_id="1", nodes=[node("n0", None)], owned=False
    )) == "", "unprobed node"

    # A heterogeneous allocation contributes only the nodes it can vouch for.
    mixed = Allocation(
        allocation_id="1",
        nodes=[node("small", ["S0", "S1"]), node("big", ["G0", "G1", "G2", "G3"])],
        owned=False,
    )
    assert _planned_gpu_uuids("2,3", mixed) == "big=G2,G3"


def test_marker_is_an_audit_record_whose_first_line_stays_the_device_list(tmp_path, fp):
    """A bare index list cannot settle "was this placed right?".

    An index means nothing once a container renumbers from 0, so the marker also
    records what the step actually SAW (index -> UUID), what CUDA ended up
    selecting, and which branch ran. Compared against the allocation topology in
    the summary, that distinguishes a bad placement from a recipe using the wrong
    device -- after the run, without reproducing it.

    Line 1 stays the plain device list: utils.gpu.task_gpu_indices reads it, and
    markers written by an older sflow must still parse.
    """
    fp.allow_unregistered(True)  # drives a real `bash` subprocess

    out = tmp_path / "task_out"
    out.mkdir()
    script = "\n".join(_gpu_placement_prelude("2,3", workflow_out_dir=_STAGE))
    env = {
        "PATH": f"{_smi_stub(tmp_path, ['GPU-c', 'GPU-d'])}:/usr/bin:/bin",
        "SLURMD_NODENAME": "nodeA",
        "SFLOW_PLANNED_GPU_UUIDS": "nodeA=GPU-c,GPU-d",
        "SFLOW_TASK_OUTPUT_DIR": str(out),
        # Host ordinals inherited into a carved container -> a re-select. Both
        # variables, as sflow's driver exports them (Backend.resource_env).
        "CUDA_VISIBLE_DEVICES": "2,3",
        "NVIDIA_VISIBLE_DEVICES": "2,3",
    }
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr

    text = (out / GPU_MARKER_FILE).read_text()
    lines = text.splitlines()
    # The contract the reader depends on.
    assert lines[0] == "0,1"
    from sflow.utils.gpu import parse_cuda_visible_devices

    assert parse_cuda_visible_devices(lines[0]) == [0, 1]

    record = dict(
        line.split("=", 1) for line in lines[1:] if "=" in line and not line.startswith("visible=")
    )
    assert record["node"] == "nodeA"
    assert record["action"] == "verified"
    # The record must show what ARRIVED, not just the post-state: inherited "2,3"
    # next to a final "0,1" is the whole story of the container renumbering, and
    # it is also how "did sflow change anything?" is answered now that there is
    # no separate no-op action to read.
    assert record["cuda_visible_devices_inherited"] == "2,3"
    assert "located by UUID" in record["reason"]
    # NVIDIA_VISIBLE_DEVICES as it ARRIVED, so a diff can be attributed. sflow
    # never exports it (Backend.resource_env pops it), so whatever is here came
    # from the container runtime -- which is exactly what makes it evidence.
    assert record["nvidia_visible_devices_inherited"] == "2,3"
    assert record["planned_host_indices"] == "2,3"
    assert record["planned_uuids"] == "GPU-c,GPU-d"
    assert record["cuda_visible_devices"] == "0,1"
    # Everything the step could see, and what CUDA will really use.
    assert "visible=0 GPU-c" in lines and "visible=1 GPU-d" in lines
    assert "selected=0 GPU-c" in lines and "selected=1 GPU-d" in lines


def test_marker_shows_an_unchanged_slice_as_inherited_equals_final(tmp_path, fp):
    """When the lookup confirms what arrived, the record has to say so.

    There is no "noop" action to read any more, so the record carries the fact
    instead: an inherited value equal to the final one means the step was already
    on the planned cards and nothing was rewritten.
    """
    fp.allow_unregistered(True)

    out = tmp_path / "task_out"
    out.mkdir()
    script = "\n".join(_gpu_placement_prelude("2,3", workflow_out_dir=_STAGE))
    env = {
        "PATH": f"{_smi_stub(tmp_path, ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d'])}:/usr/bin:/bin",
        "SLURMD_NODENAME": "nodeA",
        "SFLOW_PLANNED_GPU_UUIDS": "nodeA=GPU-c,GPU-d",
        "SFLOW_TASK_OUTPUT_DIR": str(out),
        "CUDA_VISIBLE_DEVICES": "2,3",
    }
    assert subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env).returncode == 0

    lines = (out / GPU_MARKER_FILE).read_text().splitlines()
    assert lines[0] == "2,3"
    assert "action=verified" in lines
    assert "cuda_visible_devices_inherited=2,3" in lines
    assert "cuda_visible_devices=2,3" in lines, "unchanged: inherited == final"
    assert "visible_gpu_count=4" in lines
    # It selected the planned cards out of the whole node.
    assert "selected=2 GPU-c" in lines and "selected=3 GPU-d" in lines


def test_multi_node_step_records_per_node_instead_of_racing_one_file(tmp_path, fp):
    """Every node of a multi-node task has its own devices.

    One shared marker would be a race whose winner is arbitrary, which is why the
    plain marker stays single-node-only. The per-node record is suffixed with the
    node name so each node's evidence survives.
    """
    fp.allow_unregistered(True)

    out = tmp_path / "task_out"
    out.mkdir()
    script = "\n".join(_gpu_placement_prelude("2,3", workflow_out_dir=_STAGE))
    env = {
        "PATH": f"{_smi_stub(tmp_path, ['GPU-c', 'GPU-d'])}:/usr/bin:/bin",
        "SLURMD_NODENAME": "nodeB",
        "SFLOW_PLANNED_GPU_UUIDS": "nodeB=GPU-c,GPU-d",
        "SFLOW_TASK_OUTPUT_DIR": str(out),
        "SLURM_STEP_NUM_NODES": "2",
        "SLURM_PROCID": "1",
        "SLURM_LOCALID": "0",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    assert subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env).returncode == 0

    assert not (out / GPU_MARKER_FILE).exists(), "the flat marker stays single-node"
    # <name>.<node>.log, so .log stays the extension for viewers and globs.
    per_node = out / GPU_MARKER_FILE.replace(".log", ".nodeB.log")
    assert per_node.exists()
    assert "node=nodeB" in per_node.read_text()


def test_slurm_reports_discovered_gpu_topology_to_the_summary():
    """The allocation's bare-metal topology belongs in the run record.

    Without it the per-task records have nothing to be compared against: knowing a
    task held GPU-c only answers "was that the right card?" if the run also says
    which cards the nodes had.
    """
    from sflow.core.backend import Allocation
    from sflow.core.compute_node import ComputeNode
    from sflow.plugins.backends.slurm import SlurmBackend, SlurmBackendConfig

    backend = SlurmBackend(
        SlurmBackendConfig(
            name="s",
            type="slurm",
            account="acct",
            partition="batch",
            nodes=2,
            time="00:10:00",
            gpus_per_node=2,
        )
    )
    assert backend.node_topology_report is None, "nothing to say before allocation"

    backend.allocation = Allocation(
        allocation_id="1",
        nodes=[
            ComputeNode(name="n0", ip_address="1", index=0, num_gpus=2,
                        gpu_uuids=["GPU-a", "GPU-b"]),
            ComputeNode(name="n1", ip_address="2", index=1, num_gpus=0, gpu_uuids=None),
        ],
        owned=False,
    )
    report = backend.node_topology_report
    assert "n0: 2 GPU(s)" in report
    assert "[0] GPU-a" in report and "[1] GPU-b" in report
    # A node with no probe contributes nothing rather than a misleading empty entry.
    assert "n1" not in report


def test_placement_logic_is_staged_once_and_sourced_not_pasted(tmp_path, fp):
    """The step body should point at the logic, not carry it.

    ~150 lines of shell in every srun command made a failing command line
    unreadable and duplicated the same text per task. It is staged once into the
    run's output dir -- shared storage on Slurm, so every node can read it -- and
    sourced. Sourced, not executed: it exports CUDA_VISIBLE_DEVICES into the
    task's own shell, which a child process could not.
    """
    fp.allow_unregistered(True)  # drives a real `bash` subprocess

    out = tmp_path / "run"
    out.mkdir()
    body = "\n".join(_gpu_placement_prelude("2,3", workflow_out_dir=str(out)))

    staged = out / ".sflow" / "gpu_placement.sh"
    assert staged.exists(), "the script must be staged where every node can read it"
    assert body.count("\n") < 12, f"the step body should stay short:\n{body}"
    assert f'. "{staged}"' in body, "sourced"
    assert "bash " + str(staged) not in body, "must not be run as a child process"
    # Inputs travel as environment, so nothing is interpolated into shell text.
    assert "export SFLOW_GPU_PLAN='2,3'" in body
    assert f"export SFLOW_GPU_MARKER='{GPU_MARKER_FILE}'" in body

    # Staging twice is fine (many tasks launch concurrently).
    again = "\n".join(_gpu_placement_prelude("0,1", workflow_out_dir=str(out)))
    assert staged.exists() and f'. "{staged}"' in again

    # And the staged script still does the job when sourced.
    smi = _smi_stub(tmp_path, ["GPU-c", "GPU-d"])
    r = subprocess.run(
        ["bash", "-c", f'{body}\necho "CVD=$CUDA_VISIBLE_DEVICES"'],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{smi}:/usr/bin:/bin",
            "SLURMD_NODENAME": "nodeA",
            "SFLOW_PLANNED_GPU_UUIDS": "nodeA=GPU-c,GPU-d",
            "CUDA_VISIBLE_DEVICES": "2,3",
        },
    )
    assert r.returncode == 0, r.stderr
    assert "CVD=0,1" in r.stdout


def test_placement_is_skipped_when_the_script_cannot_be_staged(caplog):
    """Nowhere the nodes can read it from -> do nothing, loudly.

    The alternative was pasting a second copy of the logic into the command line,
    which meant two delivery paths to keep honest (and a source rewrite so
    `return` stayed valid outside a sourced file). It was also unreachable in
    practice: run_support always sets SFLOW_WORKFLOW_OUTPUT_DIR, so the only
    trigger is an unwritable output dir -- by which point the run's own logs are
    already broken.

    Skipping leaves CUDA_VISIBLE_DEVICES exactly as exported, which is the
    behaviour from before this prelude existed.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        assert _gpu_placement_prelude("2,3", workflow_out_dir=None) == []
    assert "Could not stage the GPU placement script" in caplog.text


def test_the_record_states_detected_values_not_echoed_ones(tmp_path, fp):
    """Everything in the record that claims to be runtime must BE runtime.

    The point of the record is to settle "did this run on the cards we meant?"
    after the fact, so a field that merely echoes what the driver passed in would
    be worse than absent -- it would agree with the plan by construction.

    Probed inside the step: the visible UUIDs AND their indices (nvidia-smi
    ignores CUDA_VISIBLE_DEVICES, so this is the namespace's real view), plus the
    effective CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES after the decision.
    Passed in by the driver: planned_host_indices and planned_uuids -- named as
    "planned" precisely so they are not mistaken for observations.
    """
    fp.allow_unregistered(True)  # drives a real `bash` subprocess

    out = tmp_path / "task_out"
    out.mkdir()
    # The step sees cards the PLAN never mentions, so nothing here can be an echo
    # of the plan: a real probe is the only way these names appear.
    visible = ["GPU-zz0", "GPU-zz1", "GPU-c", "GPU-d"]
    script = "\n".join(_gpu_placement_prelude("2,3", workflow_out_dir=_STAGE))
    r = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{_smi_stub(tmp_path, visible)}:/usr/bin:/bin",
            "SLURMD_NODENAME": "nodeA",
            "SFLOW_PLANNED_GPU_UUIDS": "nodeA=GPU-c,GPU-d",
            "SFLOW_TASK_OUTPUT_DIR": str(out),
            "CUDA_VISIBLE_DEVICES": "2,3",
        },
    )
    assert r.returncode == 0, r.stderr
    lines = (out / GPU_MARKER_FILE).read_text().splitlines()

    # Detected: every visible card, with the index nvidia-smi reported.
    assert "visible_gpu_count=4" in lines
    for index, uuid in enumerate(visible):
        assert f"visible={index} {uuid}" in lines
    # Cards the plan never named still show up -> this is an observation.
    assert "visible=0 GPU-zz0" in lines

    # Detected: the effective environment after the decision. Nothing changed
    # here (the plan already selected the right cards).
    assert "action=verified" in lines
    assert "cuda_visible_devices=2,3" in lines
    # There is no nvidia_visible_devices= post-state, and that is deliberate:
    # sflow never writes that variable, so such a line could only ever repeat
    # nvidia_visible_devices_inherited= and would read as a second, independent
    # observation that it is not.
    assert "nvidia_visible_devices_inherited=<env-not-set>" in lines
    assert not any(ln.startswith("nvidia_visible_devices=") for ln in lines)

    # Passed in, and labelled as such.
    assert "planned_host_indices=2,3" in lines
    assert "planned_uuids=GPU-c,GPU-d" in lines

    # selected= is the join of the two: real CVD resolved through the real probe.
    assert "selected=2 GPU-c" in lines and "selected=3 GPU-d" in lines


def test_an_unset_cuda_visible_devices_is_named_not_left_unset(tmp_path, fp):
    """The whole-node case: right cards, but nothing names them.

    `gpus.count` equal to a node's device count plans every GPU, so an
    already-correct step can arrive with CUDA_VISIBLE_DEVICES unset -- "all of
    them" is the right answer. Leaving it unset is what "nothing to do" would
    look like to a naive check, and it broke a real run: assert_placement.sh dereferences the variable under
    `set -u` and bare_count_8 died with "CUDA_VISIBLE_DEVICES: unbound variable"
    on both nodes. Recipes also derive ranks and port offsets from it.

    So the cards get named explicitly. Same devices, contract intact.
    """
    fp.allow_unregistered(True)  # drives a real `bash` subprocess

    out = tmp_path / "task_out"
    out.mkdir()
    whole_node = ["GPU-w", "GPU-x", "GPU-y", "GPU-z"]
    script = "\n".join(_gpu_placement_prelude("0,1,2,3", workflow_out_dir=_STAGE))
    r = subprocess.run(
        # `set -u` on purpose: that is how the failure surfaced.
        ["bash", "-c", f'set -u\n{script}\necho "CVD=${{CUDA_VISIBLE_DEVICES}}"'],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{_smi_stub(tmp_path, whole_node)}:/usr/bin:/bin",
            "SLURMD_NODENAME": "c01",
            "SFLOW_PLANNED_GPU_UUIDS": "c01=" + ",".join(whole_node),
            "SFLOW_TASK_OUTPUT_DIR": str(out),
            # no CUDA_VISIBLE_DEVICES in the environment at all
        },
    )
    assert r.returncode == 0, r.stderr
    assert "unbound variable" not in r.stderr
    assert "CVD=0,1,2,3" in r.stdout

    lines = (out / GPU_MARKER_FILE).read_text().splitlines()
    assert "action=verified" in lines
    assert "cuda_visible_devices_inherited=<env-not-set>" in lines, "named, not left unset"
    assert "cuda_visible_devices=0,1,2,3" in lines
    # Still the planned cards -- naming them changed nothing about which.
    for index, uuid in enumerate(whole_node):
        assert f"selected={index} {uuid}" in lines


def test_record_distinguishes_unset_from_set_but_empty(tmp_path, fp):
    """Unset and empty are different states, and reporting them alike hides a bug.

    Both variables are recorded VERBATIM -- only <env-not-set> and
    <set-as-empty> are substituted, so the two can be told apart. sflow does not annotate what a
    value means to CUDA or to a container runtime: it does not own those
    semantics, and a gloss would be wrong on any stack that differs. A reader
    seeing `all` can conclude the runtime carved nothing; that is their call to
    make from the raw value.
    """
    fp.allow_unregistered(True)  # drives a real `bash` subprocess

    def record(env_extra):
        out = tmp_path / f"t{abs(hash(tuple(sorted(env_extra.items()))))}"
        out.mkdir()
        script = "\n".join(_gpu_placement_prelude("0", workflow_out_dir=_STAGE))
        env = {
            "PATH": f"{_smi_stub(tmp_path, ['GPU-a'])}:/usr/bin:/bin",
            "SLURMD_NODENAME": "n0",
            "SFLOW_PLANNED_GPU_UUIDS": "n0=GPU-a",
            "SFLOW_TASK_OUTPUT_DIR": str(out),
        }
        env.update(env_extra)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        return dict(
            line.split("=", 1)
            for line in (out / GPU_MARKER_FILE).read_text().splitlines()
            if "=" in line and not line.startswith(("visible=", "selected="))
        )

    # Neither variable present at all.
    rec = record({})
    assert rec["cuda_visible_devices_inherited"] == "<env-not-set>"
    assert rec["nvidia_visible_devices_inherited"] == "<env-not-set>"

    # Present but empty -- a different state, and it must not read as "unset".
    rec = record({"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "all"})
    assert rec["cuda_visible_devices_inherited"] == "<set-as-empty>"
    # Recorded verbatim: "all" is the runtime's own value, not sflow's gloss on it.
    assert rec["nvidia_visible_devices_inherited"] == "all"


def test_gres_clusters_resolve_by_uuid_first_and_fall_back_only_on_a_miss(tmp_path, fp):
    """On a GRES cluster the UUID lookup still runs first, and usually wins.

    Where GRES allocates the GPUs, slurmstepd sets CUDA_VISIBLE_DEVICES itself and
    a `--overlap` step can be handed the whole node's grant. The lookup handles
    that with no special case: the planned cards are in the grant, so their
    indices are found and the step narrows to exactly them -- by identity, not by
    counting positions.

    Only when a planned card is genuinely ABSENT does who-chose-the-devices
    matter. Slurm may have granted cards the planner never saw, in which case the
    plan can only mean a POSITION into the grant; resolving it as a host index
    would abort a healthy run with exit 97. Two independent signals say Slurm
    owns the devices, either sufficient:
      * SLURM_STEP_GPUS -- set by Slurm only when THIS STEP took GRES
      * the CUDA_VISIBLE_DEVICES we exported did not survive into the step

    Both are STEP-scoped. SLURM_JOB_GPUS deliberately is NOT one of them: it says
    the JOB has GPUs, and Backend.resource_env copies every SLURM_* var from the
    DRIVER's environment into every task, so on the `batch --submit` path it was
    set for every step on any GRES cluster -- and because steps run --overlap
    (Slurm does not carve per step, so SLURM_STEP_GPUS is unset there) it was the
    ONLY signal in play. The hard fail below could not fire on the very clusters
    it was written for. Asserted here so it is not quietly reinstated.
    """
    fp.allow_unregistered(True)  # drives a real `bash` subprocess

    seq = itertools.count()

    def run(env_extra, visible):
        out = tmp_path / f"g{next(seq)}"
        out.mkdir()
        script = "\n".join(
            _gpu_placement_prelude("2,3", workflow_out_dir=_STAGE)
            + ['echo "CVD=$CUDA_VISIBLE_DEVICES"']
        )
        env = {
            "PATH": f"{_smi_stub(tmp_path, visible)}:/usr/bin:/bin",
            "SLURMD_NODENAME": "n0",
            # The driver resolved the plan against a bare-metal probe of the WHOLE
            # node; Slurm then granted something else entirely.
            "SFLOW_PLANNED_GPU_UUIDS": "n0=GPU-c,GPU-d",
            "SFLOW_TASK_OUTPUT_DIR": str(out),
        }
        env.update(env_extra)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        action = next(
            (ln.split("=", 1)[1] for ln in (out / GPU_MARKER_FILE).read_text().splitlines()
             if ln.startswith("action=")), None
        ) if (out / GPU_MARKER_FILE).exists() else None
        return r, action

    granted = ["GPU-x", "GPU-y"]  # not the planned cards -- Slurm chose these

    # The `--overlap` shape from the field report: every concurrent worker is
    # handed the WHOLE node and they all race on device 0 unless something
    # narrows them. The planned cards are in the grant, so the lookup finds them
    # and narrows by identity -- no fallback, no counting.
    r, action = run(
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_STEP_GPUS": "0,1,2,3"},
        ["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
    )
    assert (r.returncode, action) == (0, "verified"), r.stderr
    assert "CVD=2,3" in r.stdout, "must land on the planned cards, not keep all four"

    # Now the miss: Slurm granted cards the planner never saw. Deferring keeps a
    # healthy run alive instead of aborting on a plan that was never a host index.
    r, action = run({"CUDA_VISIBLE_DEVICES": "0,1", "SLURM_STEP_GPUS": "3,5"}, granted)
    assert r.returncode == 0, f"must not abort a GRES step\n{r.stderr}"
    assert "CVD=0,1" in r.stdout, "both granted devices are the planned slice here"
    assert action == "fallback"
    assert "not among the devices Slurm granted this step" in r.stderr

    # ...but the JOB-level variable alone must NOT defer: it is what the driver
    # leaks into every task, and treating it as "Slurm chose this step's devices"
    # is what made the hard fail unreachable. CVD here still equals the plan, so
    # nothing step-scoped says Slurm touched it -> this is a real mis-placement.
    r, action = run({"CUDA_VISIBLE_DEVICES": "2,3", "SLURM_JOB_GPUS": "0,1,2,3"}, granted)
    assert r.returncode == 97, f"a leaked job-level var must not excuse a miss\n{r.stderr}"
    assert action == "missing"
    assert "is not visible on" in r.stderr

    # No GRES variables, but our export was replaced anyway -> same deference.
    r, action = run({"CUDA_VISIBLE_DEVICES": "0,1"}, granted)
    assert (r.returncode, action) == (0, "fallback"), r.stderr

    # Deferring still narrows: 4 granted, none of them planned, a 2-slot plan
    # -> positions 2,3 of the grant.
    r, action = run(
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_STEP_GPUS": "0,1,2,3"},
        ["GPU-p", "GPU-q", "GPU-r", "GPU-s"],
    )
    assert (r.returncode, action) == (0, "fallback"), r.stderr
    assert "CVD=2,3" in r.stdout

    # And the container case is untouched: our export DID survive, so the UUID
    # lookup still corrects the renumbering.
    r, action = run({"CUDA_VISIBLE_DEVICES": "2,3"}, ["GPU-c", "GPU-d"])
    assert (r.returncode, action) == (0, "verified"), r.stderr
    assert "CVD=0,1" in r.stdout


def test_placement_is_skipped_when_staging_raises(tmp_path):
    """Staging raises -> no placement, not a second inlined copy of the logic.

    The unstageable dir is a path UNDER A REGULAR FILE, so mkdir raises
    NotADirectoryError (an OSError). Not `chmod(0o500)`: CI runs the suite as
    root, and root ignores directory permission bits -- the write simply
    succeeded there and the test failed on a premise that was never true, rather
    than on the behaviour it meant to pin.
    """
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")

    assert _gpu_placement_prelude("2,3", workflow_out_dir=str(blocker / "out")) == []


def test_a_hard_placement_failure_still_leaves_the_record(tmp_path):
    """exit 97 is exactly when someone needs to see what the step held.

    The writer used to sit below every exit, so the one run worth diagnosing was
    the one that produced no record at all.
    """
    from types import SimpleNamespace

    from sflow.utils.gpu import task_gpu_record

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    script = "\n".join(_gpu_placement_prelude("2,3", workflow_out_dir=_STAGE))
    r = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            # Planned cards resolved, but neither is present here.
            "PATH": f"{_smi_stub(tmp_path, ['GPU-x', 'GPU-y'])}:/usr/bin:/bin",
            "SLURMD_NODENAME": "nodeA",
            "SFLOW_PLANNED_GPU_UUIDS": "nodeA=GPU-a,GPU-b",
            "SFLOW_TASK_OUTPUT_DIR": str(out_dir),
        },
    )
    assert r.returncode == 97, r.stdout + r.stderr
    record = task_gpu_record(SimpleNamespace(envs={"SFLOW_TASK_OUTPUT_DIR": str(out_dir)}))
    assert record["action"] == "missing"
    assert record["planned_uuids"] == "GPU-a,GPU-b"
    assert "not visible" in record["reason"]


def _drive(tmp_path, name, *, plan, env_extra, visible=None):
    """Run the real staged script and hand back (result, record dict)."""
    out = tmp_path / name
    out.mkdir()
    script = "\n".join(_gpu_placement_prelude(plan, workflow_out_dir=_STAGE))
    env = {
        "PATH": f"{_smi_stub(tmp_path, visible) if visible else _NO_DRIVER}:/usr/bin:/bin",
        "SLURMD_NODENAME": "nodeA",
        "SFLOW_TASK_OUTPUT_DIR": str(out),
    }
    env.update(env_extra)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    from types import SimpleNamespace

    from sflow.utils.gpu import task_gpu_record

    return r, task_gpu_record(SimpleNamespace(envs={"SFLOW_TASK_OUTPUT_DIR": str(out)}))


def test_a_node_missing_from_the_planned_map_says_so_once(tmp_path):
    """The FQDN-vs-short-name mismatch silently switches verification off.

    The map is keyed by the name the DRIVER saw; the step keys by
    $SLURMD_NODENAME. When they disagree the step finds no entry, falls back to
    index arithmetic, and every task in the run quietly stops being verified --
    with nothing failing. The warning is the only signal, so it needs a test.
    """
    r, record = _drive(
        tmp_path, "mismatch", plan="0,1",
        env_extra={
            # Map names the short form; the step reports the FQDN.
            "SLURMD_NODENAME": "nodeA.cluster.example.com",
            "SFLOW_PLANNED_GPU_UUIDS": "nodeA=GPU-a,GPU-b",
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
        visible=["GPU-a", "GPU-b"],
    )
    assert r.returncode == 0, r.stderr
    assert "no planned-GPU entry for node 'nodeA.cluster.example.com'" in r.stderr
    assert "falling back to device-index placement" in r.stderr
    # Unverified, and the record must say why rather than claim a proven placement.
    assert record["action"] == "fallback"
    assert record["planned_uuids"] == "(not resolved)"


def test_every_abort_path_leaves_a_record(tmp_path):
    """exit 97 without a record is the worst outcome: a run that failed on
    placement and cannot be diagnosed. Proven for all three aborts, not just one.
    """
    # too-few: 2 visible, 4 planned.
    r, record = _drive(
        tmp_path, "toofew", plan="0,1,2,3",
        env_extra={"CUDA_VISIBLE_DEVICES": "0,1"}, visible=["GPU-x", "GPU-y"],
    )
    assert r.returncode == 97, r.stderr
    assert record["action"] == "too-few", record
    assert "planned for 4" in record["reason"]

    # out-of-range: slot 9 is outside a 4-device visible set.
    r, record = _drive(
        tmp_path, "oor", plan="9",
        env_extra={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        visible=["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
    )
    assert r.returncode == 97, r.stderr
    assert record["action"] == "out-of-range", record
    assert "outside the visible devices" in record["reason"]


def test_the_record_names_every_selected_device_or_says_it_cannot(tmp_path):
    """`selected=` must resolve each device CUDA will use back to a UUID.

    An index the step cannot see, and a device named as a UUID rather than an
    ordinal, are both real shapes -- the record must state them rather than drop
    the line, or a reader silently sees fewer devices than the task used.
    """
    out = tmp_path / "sel"
    out.mkdir()
    # Force a post-state CUDA cannot resolve: no driver, so placement falls back
    # to the plan verbatim, and the plan names devices nvidia-smi never reported.
    script = "\n".join(_gpu_placement_prelude("0,1", workflow_out_dir=_STAGE))
    r = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        env={"PATH": f"{_NO_DRIVER}:/usr/bin:/bin", "SLURMD_NODENAME": "nodeA",
             "SFLOW_TASK_OUTPUT_DIR": str(out)},
    )
    assert r.returncode == 0, r.stderr
    lines = (out / GPU_MARKER_FILE).read_text().splitlines()
    selected = [ln for ln in lines if ln.startswith("selected=")]
    assert len(selected) == 2, lines
    # No driver -> nothing to resolve against, so each says so rather than lying.
    assert all("(not visible here)" in ln for ln in selected), selected


def test_the_marker_format_keys_are_the_contract(tmp_path):
    """Three independent parsers read this file: utils/gpu.py (Python),
    sample_test.sh::gpu_placement_verified (shell sed), and a human. Renaming a
    key breaks the shell reader silently, so the key set is pinned here.
    """
    r, record = _drive(
        tmp_path, "keys", plan="0,1",
        env_extra={"SFLOW_PLANNED_GPU_UUIDS": "nodeA=GPU-a,GPU-b",
                   "CUDA_VISIBLE_DEVICES": "0,1"},
        visible=["GPU-a", "GPU-b"],
    )
    assert r.returncode == 0, r.stderr
    assert set(record) == {
        "devices",                          # line 1, the bare device list
        "node", "action", "reason",
        "cuda_visible_devices_inherited", "nvidia_visible_devices_inherited",
        "cuda_visible_devices", "planned_host_indices", "planned_uuids",
        "visible_gpu_count",
    }, sorted(record)
    # The two keys the shell reader greps for must carry parseable values.
    assert record["action"] == "verified"
    assert record["planned_uuids"] == "GPU-a,GPU-b"
