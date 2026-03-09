# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
from urllib import request
from urllib.parse import urlparse

from sflow.core.probe import Probe, ProbeType


@dataclass(frozen=True)
class _HttpResponse:
    status: int
    body: bytes


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


class HttpGetProbe(Probe):
    def __init__(
        self,
        *,
        url: str,
        headers: Optional[dict[str, str]] = None,
        type: ProbeType,
        **kwargs,
    ):
        super().__init__(type=type, **kwargs)
        self._url = str(url)
        self._headers = dict(headers or {})
        _validate_http_url(self._url)

    async def check(self, task) -> bool:  # type: ignore[override]
        def _do() -> _HttpResponse:
            req = request.Request(self._url, headers=self._headers, method="GET")
            with request.urlopen(req, timeout=max(self.timeout, 1)) as resp:  # nosec B310
                return _HttpResponse(status=int(resp.status), body=resp.read())

        try:
            resp = await asyncio.to_thread(_do)
            return 200 <= resp.status < 400
        except Exception:
            return False


class HttpPostProbe(Probe):
    def __init__(
        self,
        *,
        url: str,
        body: str | None = None,
        headers: Optional[dict[str, str]] = None,
        type: ProbeType,
        **kwargs,
    ):
        super().__init__(type=type, **kwargs)
        self._url = str(url)
        self._body = "" if body is None else str(body)
        self._headers = dict(headers or {})
        _validate_http_url(self._url)

    async def check(self, task) -> bool:  # type: ignore[override]
        def _do() -> _HttpResponse:
            data = self._body.encode("utf-8")
            headers = dict(self._headers)
            headers.setdefault("Content-Type", "text/plain; charset=utf-8")
            req = request.Request(self._url, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=max(self.timeout, 1)) as resp:  # nosec B310
                return _HttpResponse(status=int(resp.status), body=resp.read())

        try:
            resp = await asyncio.to_thread(_do)
            return 200 <= resp.status < 400
        except Exception:
            return False
