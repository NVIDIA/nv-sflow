# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from sflow.core.probe import Probe, ProbeType
from sflow.core.probe_transport import ProbeTransport, default_probe_transport


def _validate_http_url(url: str) -> None:
    """
    Prevent surprising/unsafe schemes like file://, ftp://, etc.

    These probes are intended for HTTP(S) endpoints only.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme for HTTP probe: {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError(f"Invalid HTTP(S) URL for probe: {url!r}")


def _status_ok(status: Optional[int]) -> bool:
    return status is not None and 200 <= status < 400


class HttpGetProbe(Probe):
    def __init__(
        self,
        *,
        url: str,
        headers: Optional[dict[str, str]] = None,
        type: ProbeType,
        transport: ProbeTransport | None = None,
        **kwargs,
    ):
        super().__init__(type=type, **kwargs)
        self._url = str(url)
        self._headers = dict(headers or {})
        _validate_http_url(self._url)
        self._transport = transport or default_probe_transport()

    async def check(self, task) -> bool:  # type: ignore[override]
        status = await self._transport.http_request(
            method="GET",
            url=self._url,
            headers=self._headers,
            body=None,
            timeout=self.effective_check_timeout,
        )
        return _status_ok(status)


class HttpPostProbe(Probe):
    def __init__(
        self,
        *,
        url: str,
        body: str | None = None,
        headers: Optional[dict[str, str]] = None,
        type: ProbeType,
        transport: ProbeTransport | None = None,
        **kwargs,
    ):
        super().__init__(type=type, **kwargs)
        self._url = str(url)
        self._body = "" if body is None else str(body)
        self._headers = dict(headers or {})
        _validate_http_url(self._url)
        self._transport = transport or default_probe_transport()

    async def check(self, task) -> bool:  # type: ignore[override]
        status = await self._transport.http_request(
            method="POST",
            url=self._url,
            headers=self._headers,
            body=self._body,
            timeout=self.effective_check_timeout,
        )
        return _status_ok(status)
