# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pod log tailer honors the same streaming override as the launcher."""

import asyncio
import logging

import pytest

from sflow.core.log_offload import STREAM_TASK_LOGS
from sflow.logging import SFLOW_TASK_STREAM_ATTR
from sflow.plugins.k8s import lifecycle


@pytest.mark.parametrize("force", [False, True])
def test_non_tty_pod_logs_honor_stream_flag(tmp_path, monkeypatch, caplog, force):
    monkeypatch.setattr(lifecycle.sys.stdout, "isatty", lambda: False)
    path = tmp_path / "server.log"
    path.write_text("before-tailer\n")
    token = STREAM_TASK_LOGS.set(force)

    async def drive():
        task = asyncio.create_task(
            lifecycle.tail_file_to_console(str(path), task_name="server")
        )
        try:
            await asyncio.sleep(0)  # start the tailer before appending pod output
            assert task.done() is not force
            with path.open("a") as output:
                output.write("server-startup\nprogress-10%\rprogress-100%\n")
            await asyncio.sleep(0.4)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        with caplog.at_level(logging.INFO, logger=lifecycle._logger.name):
            asyncio.run(drive())
    finally:
        STREAM_TASK_LOGS.reset(token)
    streamed = [
        r.getMessage()
        for r in caplog.records
        if getattr(r, SFLOW_TASK_STREAM_ATTR, False)
    ]
    assert streamed == (["server-startup", "progress-100%"] if force else [])
    assert "before-tailer" in path.read_text()
