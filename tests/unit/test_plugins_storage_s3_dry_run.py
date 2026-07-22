# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run credential/SDK preflight warnings for the S3 storage target.

These cover ``S3StorageTarget.dry_run_warnings()`` (surfaced in
``sflow run --dry-run``). They need ``boto3`` importable for the
credential-present branches, but not ``moto`` (no network/mock S3 involved).
"""

import builtins

import pytest

pytest.importorskip("boto3")

from sflow.plugins.storage.s3 import S3StorageConfig, S3StorageTarget


def _target(name: str = "bucket") -> S3StorageTarget:
    return S3StorageTarget(S3StorageConfig(name=name, bucket="b"))


def _clear_aws_env(monkeypatch) -> None:
    for var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_dry_run_warns_when_no_credentials(tmp_path, monkeypatch):
    _clear_aws_env(monkeypatch)
    # Point the shared-credentials path at a file that does not exist.
    monkeypatch.setenv(
        "AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing" / "credentials")
    )

    warnings = _target("results").dry_run_warnings()

    assert len(warnings) == 1
    assert "results" in warnings[0]
    assert "no AWS credentials detected" in warnings[0]


def test_dry_run_no_warning_with_env_keys(tmp_path, monkeypatch):
    _clear_aws_env(monkeypatch)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing"))

    assert _target().dry_run_warnings() == []


def test_dry_run_no_warning_with_profile(tmp_path, monkeypatch):
    _clear_aws_env(monkeypatch)
    monkeypatch.setenv("AWS_PROFILE", "dev")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing"))

    assert _target().dry_run_warnings() == []


def test_dry_run_no_warning_with_credentials_file(tmp_path, monkeypatch):
    _clear_aws_env(monkeypatch)
    creds = tmp_path / "credentials"
    creds.write_text("[default]\naws_access_key_id=x\naws_secret_access_key=y\n")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds))

    assert _target().dry_run_warnings() == []


def test_dry_run_warns_when_credentials_file_malformed(tmp_path, monkeypatch):
    _clear_aws_env(monkeypatch)
    # Exists but is env-var style (no INI '[profile]' header), so boto3's
    # credential loader raises ConfigParseError ("Unable to parse config file")
    # at upload time. The dry-run check must surface this offline.
    creds = tmp_path / "credentials"
    creds.write_text(
        "AWS_ACCESS_KEY_ID=AKIAEXAMPLE\nAWS_SECRET_ACCESS_KEY=secret\n"
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds))

    warnings = _target("results").dry_run_warnings()

    assert len(warnings) == 1
    assert "results" in warnings[0]
    assert "cannot parse" in warnings[0]
    assert str(creds) in warnings[0]


def test_dry_run_warns_when_boto3_missing(monkeypatch):
    real_import = builtins.__import__

    def _no_boto3(name, *args, **kwargs):
        if name == "boto3" or name.startswith("boto3."):
            raise ImportError("simulated missing boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_boto3)

    warnings = _target("results").dry_run_warnings()

    assert len(warnings) == 1
    assert "requires boto3" in warnings[0]
