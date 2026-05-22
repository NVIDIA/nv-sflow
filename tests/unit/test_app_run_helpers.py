from types import SimpleNamespace

import pytest

from sflow.app.run_support import (
    build_run_paths,
    check_enroot_credentials,
    configure_task_runtime,
    ensure_sflow_dir_mounts_for_srun_container,
    preflight_validate_artifacts,
    validate_container_mounts,
)


def test_build_run_paths_uses_dry_run_workflow_dir(tmp_path):
    paths = build_run_paths(
        workflow_name="wf",
        dry_run=True,
        workspace_dir=tmp_path,
        output_dir=None,
        run_id_factory=lambda: "unused",
    )

    assert paths.workspace_dir == tmp_path
    assert paths.output_dir == tmp_path / "sflow_output"
    assert paths.workflow_output_dir == tmp_path / "sflow_output" / "_dry_run" / "wf"
    assert paths.run_id is None


def test_build_run_paths_prefixes_slurm_job_id(tmp_path):
    paths = build_run_paths(
        workflow_name="wf",
        dry_run=False,
        workspace_dir=tmp_path,
        output_dir=tmp_path / "out",
        run_id_factory=lambda: "wf-20260101-000000-abcdef",
        slurm_job_id="12345",
    )

    assert paths.run_id == "12345-wf-20260101-000000-abcdef"
    assert paths.workflow_output_dir == tmp_path / "out" / paths.run_id


def test_preflight_validate_artifacts_resolves_variables_and_warns_on_dry_run(
    tmp_path,
):
    missing = tmp_path / "missing-model"
    artifact = SimpleNamespace(name="MODEL", uri="fs://${{ variables.MODEL_DIR }}")
    variable = SimpleNamespace(name="MODEL_DIR", value=str(missing))

    warnings = preflight_validate_artifacts(
        [artifact],
        [variable],
        workspace_dir=tmp_path,
        dry_run=True,
    )

    assert warnings == [
        f"Artifact 'MODEL' (fs://) path does not exist: {missing}"
    ]


def test_preflight_validate_artifacts_raises_for_missing_fs_path(tmp_path):
    missing = tmp_path / "missing-model"
    artifact = SimpleNamespace(name="MODEL", uri=f"fs://{missing}")

    with pytest.raises(ValueError, match="Artifact path validation failed"):
        preflight_validate_artifacts(
            [artifact],
            [],
            workspace_dir=tmp_path,
            dry_run=False,
        )


def test_configure_task_runtime_sets_sflow_envs_without_overriding(tmp_path):
    task = SimpleNamespace(name="t1", envs={"SFLOW_OUTPUT_DIR": "custom"})
    workflow_out_dir = tmp_path / "out" / "wf"

    configure_task_runtime(
        task,
        ws_dir=tmp_path,
        out_dir=tmp_path / "out",
        workflow_out_dir=workflow_out_dir,
        dry_run=True,
    )

    assert task.envs["SFLOW_WORKSPACE_DIR"] == str(tmp_path)
    assert task.envs["SFLOW_OUTPUT_DIR"] == "custom"
    assert task.envs["SFLOW_WORKFLOW_OUTPUT_DIR"] == str(workflow_out_dir)
    assert task.envs["SFLOW_TASK_OUTPUT_DIR"] == str(workflow_out_dir / "t1")
    assert (workflow_out_dir / "t1").exists() is False


def test_ensure_sflow_dir_mounts_for_srun_container_appends_missing_dirs(tmp_path):
    op_conf = SimpleNamespace(
        type="srun",
        container_image="docker://alpine:latest",
        container_name=None,
        container_mounts=[f"{tmp_path}:{tmp_path}:ro"],
    )
    task = SimpleNamespace(operator=SimpleNamespace(config=op_conf))
    out_dir = tmp_path / "out"
    workflow_out_dir = out_dir / "_dry_run" / "wf"
    task_out_dir = workflow_out_dir / "t1"

    ensure_sflow_dir_mounts_for_srun_container(
        task=task,
        ws_dir=tmp_path,
        out_dir=out_dir,
        workflow_out_dir=workflow_out_dir,
        task_out_dir=task_out_dir,
    )

    assert op_conf.container_mounts == [
        f"{tmp_path}:{tmp_path}:ro",
        f"{out_dir}:{out_dir}:rw",
        f"{workflow_out_dir}:{workflow_out_dir}:rw",
        f"{task_out_dir}:{task_out_dir}:rw",
    ]


def test_validate_container_mounts_skips_output_and_env_paths(tmp_path):
    missing = tmp_path / "missing"
    output_dir = tmp_path / "out"
    op_conf = SimpleNamespace(
        container_mounts=[
            f"{missing}:/container:rw",
            f"{output_dir / 'future'}:/future:rw",
            "$DATA:/data:ro",
        ],
        mounts=[],
    )
    task = SimpleNamespace(name="t1", operator=SimpleNamespace(config=op_conf))

    warnings = validate_container_mounts([task], sflow_output_dir=output_dir)

    assert warnings == [
        f"Task 't1': mount source path does not exist: {missing}"
    ]


def test_check_enroot_credentials_warns_only_for_srun_containers(tmp_path):
    op_conf = SimpleNamespace(
        type="srun",
        container_image="docker://alpine:latest",
        container_name=None,
    )
    task = SimpleNamespace(operator=SimpleNamespace(config=op_conf))

    warning = check_enroot_credentials(
        [task],
        credentials_path=tmp_path / ".config" / "enroot" / ".credentials",
    )

    assert warning is not None
    assert "enroot credentials" in warning
