# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test: real local-backend workflow with a top-level `storage:` block
and per-task `uploads:`. moto intercepts S3 calls so no network is required.
"""

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from sflow.app.sflow import SflowApp


@pytest.fixture(autouse=True)
def _aws_creds(monkeypatch):
    # moto needs *some* credentials in the environment; values are ignored.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


YAML = """
version: "0.1"

variables:
  - name: RUN_ID
    value: run-42

storage:
  - name: results_bucket
    type: s3
    bucket: e2e-results
    region: us-east-1
    prefix: "runs/${{ variables.RUN_ID }}/"

workflow:
  name: storage_e2e
  tasks:
    - name: produce
      script:
        - mkdir -p "$SFLOW_TASK_OUTPUT_DIR"
        - echo "k,v" > "$SFLOW_TASK_OUTPUT_DIR/results.csv"
        - echo "n=1" > "$SFLOW_TASK_OUTPUT_DIR/a.json"
        - echo "n=2" > "$SFLOW_TASK_OUTPUT_DIR/b.json"
      uploads:
        - target: results_bucket
          from: "${{ task.output_dir }}/results.csv"
          to: "main/results.csv"
        - target: results_bucket
          from: "${{ task.output_dir }}/*.json"
          on_error: warn
"""


def test_storage_uploads_run_after_task_completion(tmp_path: Path):
    cfg_path = tmp_path / "sflow.yaml"
    cfg_path.write_text(YAML)
    out_dir = tmp_path / "out"

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="e2e-results")

        SflowApp().run(
            file=cfg_path,
            dry_run=False,
            workspace_dir=tmp_path,
            output_dir=out_dir,
        )

        keys = sorted(
            obj["Key"]
            for obj in s3.list_objects_v2(Bucket="e2e-results").get("Contents", [])
        )
        assert keys == sorted(
            [
                "runs/run-42/main/results.csv",
                "runs/run-42/a.json",
                "runs/run-42/b.json",
            ]
        )

        # Spot-check content of one object.
        obj = s3.get_object(Bucket="e2e-results", Key="runs/run-42/main/results.csv")
        assert obj["Body"].read().decode().strip() == "k,v"


YAML_BAD_UPLOAD_FAIL = """
version: "0.1"

storage:
  - name: results_bucket
    type: s3
    bucket: does-not-exist

workflow:
  name: storage_fail_e2e
  tasks:
    - name: produce
      script:
        - mkdir -p "$SFLOW_TASK_OUTPUT_DIR"
        - echo "x" > "$SFLOW_TASK_OUTPUT_DIR/x.txt"
      uploads:
        - target: results_bucket
          from: "${{ task.output_dir }}/x.txt"
          on_error: fail
"""


def test_on_error_fail_propagates_to_workflow_failure(tmp_path: Path):
    cfg_path = tmp_path / "sflow.yaml"
    cfg_path.write_text(YAML_BAD_UPLOAD_FAIL)
    out_dir = tmp_path / "out"

    with mock_aws():
        # Note: we deliberately do NOT create the bucket — boto3 will raise NoSuchBucket.
        with pytest.raises(RuntimeError, match="failed"):
            SflowApp().run(
                file=cfg_path,
                dry_run=False,
                workspace_dir=tmp_path,
                output_dir=out_dir,
            )
