from types import SimpleNamespace

import pytest

from sflow.utils.container import (
    CONTAINER_IMAGE_INVALID_HINT,
    append_missing_mounts,
    append_runtime_mounts,
    collect_container_mounts,
    extract_container_images_from_extra_args,
    extract_container_mounts_from_extra_args,
    local_artifact_mounts,
    mount_key,
    validate_container_image_reference,
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


def test_extract_container_images_from_extra_args_supports_both_spellings():
    images = extract_container_images_from_extra_args(
        [
            "--container-image",
            "nvcr.io/nvidia/pytorch:24.01-py3",
            "--container-image=${{ variables.IMG }}",
            "--unrelated",
        ]
    )

    assert images == [
        "nvcr.io/nvidia/pytorch:24.01-py3",
        "${{ variables.IMG }}",
    ]


def test_validate_container_image_reference_uses_shared_hint():
    with pytest.raises(ValueError) as exc_info:
        validate_container_image_reference("not a valid image!!", source="operator image")

    message = str(exc_info.value)
    assert "operator image does not look like a valid container image" in message
    assert CONTAINER_IMAGE_INVALID_HINT in message


def test_validate_container_image_reference_skips_deferred_values():
    validate_container_image_reference("${{ variables.IMAGE }}", source="operator image")
    validate_container_image_reference("${IMAGE}", source="operator image")


def test_append_runtime_mounts_uses_source_destination_dedupe():
    mounts = append_runtime_mounts(
        ["/host:/container:ro"],
        ["/host:/container:rw", "/new:/new:rw"],
    )

    assert mounts == ["/host:/container:ro", "/new:/new:rw"]


def test_collect_container_mounts_prefers_mount_specs_hook():
    op_conf = SimpleNamespace(
        mount_specs=lambda: ["/hook:/hook:rw"],
        container_mounts=["/ignored:/ignored:rw"],
        mounts=["/also-ignored:/also-ignored:rw"],
        extra_args=["--container-mounts", "/extra:/extra:rw"],
    )

    assert collect_container_mounts(op_conf) == ["/hook:/hook:rw"]


def test_collect_container_mounts_falls_back_to_known_mount_fields_and_extra_args():
    op_conf = SimpleNamespace(
        container_mounts=["/a:/b:ro"],
        mounts=["/c:/d:rw"],
        extra_args=["--container-mounts", "/e:/f:rw,/a:/b:rw"],
    )

    assert collect_container_mounts(op_conf) == [
        "/a:/b:ro",
        "/c:/d:rw",
        "/e:/f:rw",
    ]


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
