# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import builtins
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from sflow.plugins.storage.s3 import S3StorageConfig, S3StorageTarget


@pytest.fixture
def aws_credentials(monkeypatch):
    # moto needs *some* credentials in env; values are ignored.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def test_s3_upload_writes_object(tmp_path: Path, aws_credentials):
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="my-bucket")

        cfg = S3StorageConfig(
            name="results", bucket="my-bucket", prefix="runs/abc/"
        )
        target = S3StorageTarget(cfg)

        local = tmp_path / "data.txt"
        local.write_text("hello sflow")

        asyncio.run(target.upload(local, "runs/abc/data.txt"))

        resp = boto3.client("s3").get_object(Bucket="my-bucket", Key="runs/abc/data.txt")
        assert resp["Body"].read().decode() == "hello sflow"


def test_s3_plan_returns_uri(tmp_path: Path):
    cfg = S3StorageConfig(name="r", bucket="bk", prefix="p/")
    target = S3StorageTarget(cfg)
    assert target.plan(tmp_path / "x", "p/x") == "s3://bk/p/x"


def test_s3_upload_raises_friendly_error_without_boto3(tmp_path: Path, monkeypatch, aws_credentials):
    # Construction must NOT require boto3; only the actual upload path does.
    cfg = S3StorageConfig(name="r", bucket="bk")
    target = S3StorageTarget(cfg)

    original_import = builtins.__import__

    def _no_boto3(name, *args, **kwargs):
        if name == "boto3" or name.startswith("boto3."):
            raise ImportError("simulated missing boto3")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_boto3)
    local = tmp_path / "x.txt"
    local.write_text("hi")
    with pytest.raises(ImportError, match="S3 storage requires boto3"):
        asyncio.run(target.upload(local, "x.txt"))


def test_s3_endpoint_url_for_compatible_stores(tmp_path: Path, aws_credentials):
    # moto via mock_aws also intercepts custom endpoints.
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="custom-bucket")
        cfg = S3StorageConfig(
            name="r",
            bucket="custom-bucket",
            endpoint_url="http://minio.example:9000",
        )
        # Construction must not raise even with a non-AWS endpoint; the moto mock
        # intercepts the call regardless of endpoint_url.
        target = S3StorageTarget(cfg)
        assert target.bucket == "custom-bucket"


def test_s3_endpoint_url_defaults_to_path_addressing(aws_credentials):
    """When endpoint_url is set (MinIO/Ceph), addressing_style defaults to 'path'."""
    with mock_aws():
        cfg = S3StorageConfig(
            name="r",
            bucket="bk",
            endpoint_url="http://minio.example:9000",
        )
        target = S3StorageTarget(cfg)
        client = asyncio.run(target._client())
        # boto3 stores per-service style on client.meta.config.s3
        assert client.meta.config.s3.get("addressing_style") == "path"


def test_s3_explicit_addressing_style_overrides_default(aws_credentials):
    with mock_aws():
        cfg = S3StorageConfig(
            name="r",
            bucket="bk",
            endpoint_url="http://minio.example:9000",
            addressing_style="virtual",
        )
        target = S3StorageTarget(cfg)
        client = asyncio.run(target._client())
        assert client.meta.config.s3.get("addressing_style") == "virtual"


def test_s3_aws_default_has_no_addressing_override(aws_credentials):
    """No endpoint_url + no addressing_style => let boto3 use its default."""
    with mock_aws():
        cfg = S3StorageConfig(name="r", bucket="bk")
        target = S3StorageTarget(cfg)
        client = asyncio.run(target._client())
        # boto3's default is "auto" when not explicitly set
        s3_cfg = client.meta.config.s3 or {}
        assert "addressing_style" not in s3_cfg or s3_cfg["addressing_style"] in (
            "auto",
            None,
        )
