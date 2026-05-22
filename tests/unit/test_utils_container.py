from sflow.utils.container import (
    append_missing_mounts,
    extract_container_mounts_from_extra_args,
    local_artifact_mounts,
    mount_key,
)


def test_mount_key_ignores_mode_suffix():
    assert mount_key("/host:/container:rw") == ("/host", "/container")
    assert mount_key("/host:/container") == ("/host", "/container")
    assert mount_key("invalid") is None


def test_append_missing_mounts_deduplicates_by_source_and_destination():
    mounts = append_missing_mounts(
        ["/host:/container:ro"],
        ["/host:/container:rw", "/other:/other:rw"],
    )

    assert mounts == ["/host:/container:ro", "/other:/other:rw"]


def test_extract_container_mounts_from_extra_args_supports_both_spellings():
    mounts = extract_container_mounts_from_extra_args(
        [
            "--container-mounts",
            "/path1:/cpath1,/path2:/cpath2:ro",
            "--container-mounts=/path3:/cpath3",
        ]
    )

    assert mounts == ["/path1:/cpath1", "/path2:/cpath2:ro", "/path3:/cpath3"]


def test_local_artifact_mounts_mounts_file_parent_and_skips_sqsh(tmp_path):
    data_file = tmp_path / "data.txt"
    data_file.write_text("x")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    artifacts = [
        {"uri": f"file://{data_file}", "path": str(data_file)},
        {"uri": f"fs://{model_dir}", "path": str(model_dir)},
        {
            "uri": f"file://{tmp_path / 'image.sqsh'}",
            "path": str(tmp_path / "image.sqsh"),
        },
        {"uri": "s3://bucket/key", "path": str(tmp_path / "remote")},
    ]

    mounts = local_artifact_mounts(artifacts)

    assert f"{tmp_path}:{tmp_path}:rw" in mounts
    assert f"{model_dir}:{model_dir}:rw" in mounts
    assert not any("image.sqsh" in mount for mount in mounts)
    assert not any("remote" in mount for mount in mounts)
