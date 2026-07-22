# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from sflow.config.schema import SflowConfig, UploadSpec


def _base_cfg(**overrides):
    cfg = {
        "version": "0.1",
        "workflow": {
            "name": "wf",
            "tasks": [{"name": "t1", "script": ["echo hi"]}],
        },
    }
    cfg.update(overrides)
    return cfg


def test_storage_block_parses_with_s3_target():
    cfg = _base_cfg(
        storage=[
            {
                "name": "results",
                "type": "s3",
                "bucket": "my-bucket",
                "region": "us-west-2",
                "prefix": "runs/",
            }
        ]
    )
    parsed = SflowConfig(**cfg)
    assert parsed.storage is not None
    assert len(parsed.storage) == 1
    s = parsed.storage[0]
    assert s.name == "results"
    assert s.type == "s3"
    assert getattr(s, "bucket") == "my-bucket"


def test_upload_spec_parses_with_from_alias_and_defaults_on_error_warn():
    spec = UploadSpec.model_validate({"target": "x", "from": "/tmp/a.txt"})
    assert spec.target == "x"
    assert spec.from_ == "/tmp/a.txt"
    assert spec.on_error == "warn"
    assert spec.to is None


def test_upload_spec_extra_field_forbidden():
    with pytest.raises(ValidationError):
        UploadSpec.model_validate(
            {"target": "x", "from": "/tmp/a.txt", "bogus": True}
        )


def test_upload_spec_empty_from_rejected():
    with pytest.raises(ValidationError, match="at least 1"):
        UploadSpec.model_validate({"target": "x", "from": ""})


def test_upload_spec_empty_to_rejected():
    with pytest.raises(ValidationError, match="at least 1"):
        UploadSpec.model_validate({"target": "x", "from": "/tmp/a.txt", "to": ""})


def test_task_uploads_must_reference_known_storage_target():
    cfg = _base_cfg(
        storage=[
            {"name": "bucketA", "type": "s3", "bucket": "a"},
        ],
        workflow={
            "name": "wf",
            "tasks": [
                {
                    "name": "t1",
                    "script": ["echo hi"],
                    "uploads": [
                        {"target": "bucketB", "from": "x.txt"},
                    ],
                }
            ],
        },
    )
    with pytest.raises(ValidationError) as ei:
        SflowConfig(**cfg)
    assert "unknown storage target 'bucketB'" in str(ei.value)


def test_storage_target_names_must_be_unique():
    cfg = _base_cfg(
        storage=[
            {"name": "results", "type": "s3", "bucket": "bucket-a"},
            {"name": "results", "type": "s3", "bucket": "bucket-b"},
        ],
        workflow={
            "name": "wf",
            "tasks": [
                {
                    "name": "t1",
                    "script": ["echo hi"],
                    "uploads": [{"target": "results", "from": "x.txt"}],
                }
            ],
        },
    )

    with pytest.raises(ValidationError) as ei:
        SflowConfig(**cfg)

    assert "Duplicate storage target names" in str(ei.value)
    assert "results" in str(ei.value)


def test_task_uploads_validates_against_declared_targets():
    cfg = _base_cfg(
        storage=[
            {"name": "bucketA", "type": "s3", "bucket": "a"},
        ],
        workflow={
            "name": "wf",
            "tasks": [
                {
                    "name": "t1",
                    "script": ["echo hi"],
                    "uploads": [
                        {
                            "target": "bucketA",
                            "from": "${{ task.output_dir }}/a.json",
                            "on_error": "fail",
                        },
                    ],
                }
            ],
        },
    )
    parsed = SflowConfig(**cfg)
    assert parsed.workflow.tasks[0].uploads[0].on_error == "fail"


def test_uploads_with_no_storage_block_raises():
    cfg = _base_cfg(
        workflow={
            "name": "wf",
            "tasks": [
                {
                    "name": "t1",
                    "script": ["echo hi"],
                    "uploads": [{"target": "anywhere", "from": "x.txt"}],
                }
            ],
        }
    )
    with pytest.raises(ValidationError):
        SflowConfig(**cfg)


def test_workflow_upload_all_parses_and_validates_target():
    cfg = _base_cfg(
        storage=[
            {"name": "results", "type": "s3", "bucket": "a"},
        ],
        workflow={
            "name": "wf",
            "upload_all": {
                "target": "results",
                "to": "archive/${{ workflow.run_id }}.zip",
                "on_error": "fail",
            },
            "tasks": [{"name": "t1", "script": ["echo hi"]}],
        },
    )
    parsed = SflowConfig(**cfg)
    assert parsed.workflow.upload_all is not None
    assert parsed.workflow.upload_all.target == "results"
    assert parsed.workflow.upload_all.on_error == "fail"


def test_workflow_upload_all_defaults_on_error_to_warn():
    cfg = _base_cfg(
        storage=[{"name": "results", "type": "s3", "bucket": "a"}],
        workflow={
            "name": "wf",
            "upload_all": {"target": "results"},
            "tasks": [{"name": "t1", "script": ["echo hi"]}],
        },
    )
    parsed = SflowConfig(**cfg)
    assert parsed.workflow.upload_all.on_error == "warn"
    assert parsed.workflow.upload_all.to is None


def test_workflow_upload_all_rejects_unknown_target():
    cfg = _base_cfg(
        storage=[{"name": "results", "type": "s3", "bucket": "a"}],
        workflow={
            "name": "wf",
            "upload_all": {"target": "ghost"},
            "tasks": [{"name": "t1", "script": ["echo hi"]}],
        },
    )
    with pytest.raises(ValidationError) as ei:
        SflowConfig(**cfg)
    assert "workflow.upload_all references unknown storage target 'ghost'" in str(
        ei.value
    )
