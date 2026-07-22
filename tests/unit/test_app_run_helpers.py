from types import SimpleNamespace

import pytest

from sflow.app.run_support import (
    SFLOW_RESERVED_ENV_VARS,
    build_run_paths,
    collect_operator_runtime_warnings,
    configure_task_runtime,
    config_uses_offhost_backend,
    find_reserved_env_collisions,
    preflight_validate_artifacts,
    validate_container_mounts,
)


def test_find_reserved_env_collisions_flags_reserved_names_sorted():
    names = [
        "MY_VAR",
        "SFLOW_TASK_OUTPUT_DIR",
        "CUDA_VISIBLE_DEVICES",
        "SFLOW_TASK_RESULT_FILE",
        "ANOTHER",
    ]

    assert find_reserved_env_collisions(names) == [
        "CUDA_VISIBLE_DEVICES",
        "SFLOW_TASK_OUTPUT_DIR",
        "SFLOW_TASK_RESULT_FILE",
    ]


def test_find_reserved_env_collisions_returns_empty_without_overlap():
    assert find_reserved_env_collisions(["MODEL_PATH", "TP_SIZE"]) == []


def test_merge_backend_extra_args_routes_by_backend_type(tmp_path):
    """--extra-salloc-args reaches only Slurm backends and --extra-docker-args only
    docker backends; each de-dups by option against its own config extra_args."""
    from sflow.app.sflow import _merge_backend_extra_args
    from sflow.config.loader import ConfigLoader

    f = tmp_path / "mixed.yaml"
    f.write_text(
        """
version: "0.1"
backends:
  - name: cluster_a
    type: slurm
    default: true
    account: acct
    partition: part_a
    nodes: 1
    gpus_per_node: 8
    time: "00:10:00"
    extra_args: ["--exclusive"]
  - name: local_docker
    type: docker
    image: ubuntu:22.04
    extra_args: ["--network=bridge"]
operators:
  - name: w
    type: srun
    ntasks_per_node: 1
workflow:
  name: mixed
  tasks:
    - name: a
      operator: w
      script: ["echo a"]
"""
    )
    config = ConfigLoader().load_configs([f], None, None, None)
    merged = _merge_backend_extra_args(
        config,
        {"slurm": ["--gpus-per-node=4"], "docker": ["--network=host"]},
    )
    by_name = {b.name: [str(a) for a in (b.extra_args or [])] for b in merged.backends}

    # Slurm backend gets the salloc args (and keeps its own), not the docker args.
    assert "--gpus-per-node=4" in by_name["cluster_a"]
    assert "--exclusive" in by_name["cluster_a"]
    assert "--network=host" not in by_name["cluster_a"]
    # Docker backend gets the docker args (option-key: --network=host overrides the
    # recipe's --network=bridge), not the salloc args.
    assert by_name["local_docker"] == ["--network=host"]
    assert "--gpus-per-node=4" not in by_name["local_docker"]


def test_sflow_reserved_env_vars_covers_injected_paths():
    # The result-parsing direct-write targets and core dirs must be reserved.
    for name in (
        "SFLOW_WORKSPACE_DIR",
        "SFLOW_OUTPUT_DIR",
        "SFLOW_WORKFLOW_OUTPUT_DIR",
        "SFLOW_TASK_OUTPUT_DIR",
        "SFLOW_TASK_RESULT_FILE",
        "SFLOW_WORKFLOW_RESULT_FILE",
        "CUDA_VISIBLE_DEVICES",
    ):
        assert name in SFLOW_RESERVED_ENV_VARS


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


def test_build_run_paths_prefixes_generic_run_id(tmp_path):
    paths = build_run_paths(
        workflow_name="wf",
        dry_run=False,
        workspace_dir=tmp_path,
        output_dir=tmp_path / "out",
        run_id_factory=lambda: "wf-20260101-000000-abcdef",
        run_id_prefix="12345",
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


def test_preflight_validate_artifacts_warns_for_missing_fs_path_when_offhost(tmp_path):
    # Off-host backend (e.g. Kubernetes): a missing fs:// path must NOT block the run
    # (the path lives on the remote cluster/image); it only warns.
    missing = tmp_path / "missing-model"
    artifact = SimpleNamespace(name="MODEL", uri=f"fs://{missing}")

    warnings = preflight_validate_artifacts(
        [artifact],
        [],
        workspace_dir=tmp_path,
        dry_run=False,
        skip_local_fs_validation=True,
    )

    assert len(warnings) == 1
    assert "off-host backend" in warnings[0]


def test_config_uses_offhost_backend_detects_kubernetes():
    k8s = SimpleNamespace(backends=[SimpleNamespace(type="kubernetes")])
    mixed = SimpleNamespace(
        backends=[SimpleNamespace(type="slurm"), {"type": "kubernetes"}]
    )
    local = SimpleNamespace(
        backends=[SimpleNamespace(type="slurm"), SimpleNamespace(type="local")]
    )
    assert config_uses_offhost_backend(k8s) is True
    assert config_uses_offhost_backend(mixed) is True
    assert config_uses_offhost_backend(local) is False
    assert config_uses_offhost_backend(SimpleNamespace(backends=None)) is False


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


def test_configure_task_runtime_appends_sflow_dirs_via_operator_hook(tmp_path):
    class _MountingConfig:
        def __init__(self):
            self.mounts = [f"{tmp_path}:{tmp_path}:ro"]

        def append_runtime_mounts(self, mounts):
            for mount in mounts:
                if mount not in self.mounts:
                    self.mounts.append(mount)

    op_conf = _MountingConfig()
    task = SimpleNamespace(name="t1", envs={}, operator=SimpleNamespace(config=op_conf))
    out_dir = tmp_path / "out"
    workflow_out_dir = out_dir / "_dry_run" / "wf"

    configure_task_runtime(
        task,
        ws_dir=tmp_path,
        out_dir=out_dir,
        workflow_out_dir=workflow_out_dir,
        dry_run=True,
    )

    assert op_conf.mounts == [
        f"{tmp_path}:{tmp_path}:ro",
        f"{tmp_path}:{tmp_path}:rw",
        f"{out_dir}:{out_dir}:rw",
        f"{workflow_out_dir}:{workflow_out_dir}:rw",
        f"{workflow_out_dir / 't1'}:{workflow_out_dir / 't1'}:rw",
    ]


def test_configure_task_runtime_appends_sflow_dirs_to_docker_mounts(tmp_path):
    def _append_runtime_mounts(mounts):
        for mount in mounts:
            if mount not in op_conf.mounts:
                op_conf.mounts.append(mount)

    op_conf = SimpleNamespace(
        type="docker",
        image="alpine:latest",
        mounts=[f"{tmp_path}:{tmp_path}:ro"],
        append_runtime_mounts=_append_runtime_mounts,
    )
    task = SimpleNamespace(
        name="t1",
        envs={},
        operator=SimpleNamespace(config=op_conf),
    )
    out_dir = tmp_path / "out"
    workflow_out_dir = out_dir / "_dry_run" / "wf"

    configure_task_runtime(
        task,
        ws_dir=tmp_path,
        out_dir=out_dir,
        workflow_out_dir=workflow_out_dir,
        dry_run=True,
    )

    assert op_conf.mounts == [
        f"{tmp_path}:{tmp_path}:ro",
        f"{tmp_path}:{tmp_path}:rw",
        f"{out_dir}:{out_dir}:rw",
        f"{workflow_out_dir}:{workflow_out_dir}:rw",
        f"{workflow_out_dir / 't1'}:{workflow_out_dir / 't1'}:rw",
    ]


def test_configure_task_runtime_appends_sflow_dirs_to_docker_run_test_double(tmp_path):
    def _append_runtime_mounts(mounts):
        for mount in mounts:
            if mount not in op_conf.mounts:
                op_conf.mounts.append(mount)

    op_conf = SimpleNamespace(
        type="docker_run",
        image="alpine:latest",
        mounts=[f"{tmp_path}:{tmp_path}:ro"],
        append_runtime_mounts=_append_runtime_mounts,
    )
    task = SimpleNamespace(
        name="t1",
        envs={},
        operator=SimpleNamespace(config=op_conf),
    )
    out_dir = tmp_path / "out"
    workflow_out_dir = out_dir / "_dry_run" / "wf"

    configure_task_runtime(
        task,
        ws_dir=tmp_path,
        out_dir=out_dir,
        workflow_out_dir=workflow_out_dir,
        dry_run=True,
    )

    assert op_conf.mounts == [
        f"{tmp_path}:{tmp_path}:ro",
        f"{tmp_path}:{tmp_path}:rw",
        f"{out_dir}:{out_dir}:rw",
        f"{workflow_out_dir}:{workflow_out_dir}:rw",
        f"{workflow_out_dir / 't1'}:{workflow_out_dir / 't1'}:rw",
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


def test_validate_container_mounts_uses_operator_mount_specs_hook(tmp_path):
    missing = tmp_path / "missing"
    output_dir = tmp_path / "out"
    op_conf = SimpleNamespace(
        mount_specs=lambda: [
            f"{missing}:/container:rw",
            f"{output_dir / 'future'}:/future:rw",
            "$DATA:/data:ro",
        ],
        container_mounts=[],
        mounts=[],
    )
    task = SimpleNamespace(name="t1", operator=SimpleNamespace(config=op_conf))

    warnings = validate_container_mounts([task], sflow_output_dir=output_dir)

    assert warnings == [
        f"Task 't1': mount source path does not exist: {missing}"
    ]


def test_collect_operator_runtime_warnings_uses_operator_config_hook():
    op_conf = SimpleNamespace(runtime_warnings=lambda: ["backend-specific warning"])
    task = SimpleNamespace(operator=SimpleNamespace(config=op_conf))

    assert collect_operator_runtime_warnings([task]) == ["backend-specific warning"]


def test_collect_operator_runtime_warnings_dedupes_with_affected_task_count():
    # Same task-agnostic warning emitted by many tasks/replicas collapses into one
    # line with an "(affects N tasks)" suffix.
    op_conf = SimpleNamespace(runtime_warnings=lambda: ["enroot creds missing"])
    tasks = [
        SimpleNamespace(operator=SimpleNamespace(config=op_conf)) for _ in range(10)
    ]

    assert collect_operator_runtime_warnings(tasks) == [
        "enroot creds missing (affects 10 tasks)"
    ]


def test_collect_operator_runtime_warnings_preserves_distinct_messages_in_order():
    op_a = SimpleNamespace(runtime_warnings=lambda: ["warn A"])
    op_b = SimpleNamespace(runtime_warnings=lambda: ["warn B"])
    tasks = [
        SimpleNamespace(operator=SimpleNamespace(config=op_a)),
        SimpleNamespace(operator=SimpleNamespace(config=op_b)),
        SimpleNamespace(operator=SimpleNamespace(config=op_a)),
    ]

    # "warn A" affects 2 tasks; "warn B" affects 1 (no suffix); first-seen order kept.
    assert collect_operator_runtime_warnings(tasks) == [
        "warn A (affects 2 tasks)",
        "warn B",
    ]
