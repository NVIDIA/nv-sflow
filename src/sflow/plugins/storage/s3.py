# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Literal, Optional

from sflow.config.schema import Resolvable, StorageConfig
from sflow.core.storage import StorageTarget
from sflow.core.storage_registry import register_storage
from sflow.logging import get_logger

_logger = get_logger(__name__)


class S3StorageConfig(StorageConfig):
    type: Literal["s3"] = "s3"
    bucket: Resolvable[str]
    region: Optional[Resolvable[str]] = None
    prefix: Optional[Resolvable[str]] = None
    # For S3-compatible stores (MinIO, Ceph RGW, ...). Leave unset for AWS S3.
    endpoint_url: Optional[Resolvable[str]] = None
    # Optional S3 storage class (STANDARD, STANDARD_IA, GLACIER, ...).
    storage_class: Optional[Resolvable[str]] = None
    # Most S3-compatible endpoints (MinIO, Ceph RGW) require "path"-style.
    # AWS S3 accepts both but defaults to virtual-hosted. When endpoint_url is
    # set, we default to "path" so IP/hostname-addressed clusters work; users
    # can override explicitly.
    addressing_style: Optional[Literal["auto", "virtual", "path"]] = None


@register_storage("s3", S3StorageConfig)
class S3StorageTarget(StorageTarget):
    """
    Upload files to AWS S3 (or an S3-compatible endpoint) using boto3.

    Credentials are resolved by boto3's default credential chain:
    env vars (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN),
    `~/.aws/credentials`, or the instance / pod IAM role. No secrets are
    accepted inline in YAML.
    """

    def __init__(self, config: S3StorageConfig):
        super().__init__(name=config.name)
        self.config = config
        self._bucket = str(config.bucket)
        self._region = str(config.region) if config.region else None
        self._prefix = str(config.prefix) if config.prefix else ""
        self._endpoint_url = str(config.endpoint_url) if config.endpoint_url else None
        self._storage_class = (
            str(config.storage_class) if config.storage_class else None
        )
        # If endpoint_url is set and the user hasn't specified, prefer path-style
        # so IP/hostname-addressed MinIO/Ceph clusters work out of the box.
        if config.addressing_style:
            self._addressing_style: Optional[str] = str(config.addressing_style)
        elif self._endpoint_url:
            self._addressing_style = "path"
        else:
            self._addressing_style = None
        self._client_cache: Any = None
        self._client_lock = asyncio.Lock()

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    async def _client(self) -> Any:
        if self._client_cache is not None:
            return self._client_cache
        async with self._client_lock:
            if self._client_cache is not None:
                return self._client_cache
            try:
                import boto3  # type: ignore[import-not-found]
                from botocore.config import Config  # type: ignore[import-not-found]
            except ImportError as e:
                raise ImportError(
                    "S3 storage requires boto3. Install with: pip install 'sflow[s3]'"
                ) from e

            client_kwargs: dict[str, Any] = {}
            if self._region:
                client_kwargs["region_name"] = self._region
            if self._endpoint_url:
                client_kwargs["endpoint_url"] = self._endpoint_url
            if self._addressing_style:
                client_kwargs["config"] = Config(
                    s3={"addressing_style": self._addressing_style}
                )
            self._client_cache = boto3.client("s3", **client_kwargs)
            return self._client_cache

    async def upload(self, local_path: Path, remote_key: str) -> None:
        client = await self._client()
        extra_args: dict[str, Any] = {}
        if self._storage_class:
            extra_args["StorageClass"] = self._storage_class

        def _do_upload() -> None:
            kwargs: dict[str, Any] = {}
            if extra_args:
                kwargs["ExtraArgs"] = extra_args
            client.upload_file(str(local_path), self._bucket, remote_key, **kwargs)

        await asyncio.to_thread(_do_upload)

    def plan(self, local_path: Path, remote_key: str) -> str:
        return f"s3://{self._bucket}/{remote_key}"

    def dry_run_warnings(self) -> list[str]:
        """Offline preflight: flag a missing boto3 install or absent AWS credentials.

        Runs at dry-run time so users see likely upload failures before a real run.
        No network calls are made (boto3's chain — incl. IAM roles — is only
        consulted at upload time), so the credential check is best-effort: it looks
        for the documented sources (AWS_* env vars and the shared credentials file)
        and notes that an IAM role would also satisfy them at runtime.
        """
        try:
            import boto3  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            return [
                f"storage target '{self.name}' (s3) requires boto3, which is not "
                f"installed — uploads will fail. Install with: pip install 'sflow[s3]'"
            ]

        creds_file = Path(
            os.environ.get(
                "AWS_SHARED_CREDENTIALS_FILE",
                str(Path.home() / ".aws" / "credentials"),
            )
        )
        has_env_keys = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        has_profile = bool(os.environ.get("AWS_PROFILE"))
        creds_file_exists = creds_file.exists()
        if not (has_env_keys or has_profile or creds_file_exists):
            return [
                f"storage target '{self.name}' (s3): no AWS credentials detected "
                f"(no AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE env, "
                f"and no {creds_file}). Uploads use boto3's default credential chain "
                f"at runtime (env vars, ~/.aws/credentials, or an attached IAM role); "
                f"if none is available they will fail."
            ]

        # A credentials file that exists but is malformed is worse than a missing
        # one: boto3 parses it lazily on the first authenticated call and aborts
        # with "Unable to parse config file", which never surfaces during planning.
        # Mirror boto3's own INI parsing here so the dry run catches it offline.
        if creds_file_exists and self._credentials_parse_error(creds_file):
            return [
                f"storage target '{self.name}' (s3): the AWS shared credentials "
                f"file {creds_file} exists but boto3 cannot parse it — uploads will "
                f'fail at runtime with "Unable to parse config file". It must be '
                f"valid INI with a profile section header, e.g.: [default] / "
                f"aws_access_key_id=<KEY> / aws_secret_access_key=<SECRET>."
            ]
        return []

    @staticmethod
    def _credentials_parse_error(creds_file: Path) -> Optional[str]:
        """Return boto3's parse-error message for ``creds_file``, else None.

        Uses ``botocore``'s own INI loader so the dry-run check matches exactly
        what the runtime credential chain does. Best-effort: if botocore is not
        importable or the parse succeeds, returns None (no warning).
        """
        try:
            from botocore.configloader import (  # type: ignore[import-not-found]
                raw_config_parse,
            )
            from botocore.exceptions import (  # type: ignore[import-not-found]
                ConfigParseError,
            )
        except ImportError:
            return None
        try:
            raw_config_parse(str(creds_file))
        except ConfigParseError as e:
            return str(e)
        except Exception:
            # Never let a parser edge case turn a planning hint into a crash.
            return None
        return None
