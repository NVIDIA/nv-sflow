# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest


@pytest.fixture(autouse=True)
def _sflow_logger_caplog_friendly():
    """Keep pytest's ``caplog`` reliable regardless of test order.

    ``sflow.logging.configure_logging()`` sets ``logging.getLogger("sflow").propagate =
    False`` (so production runs don't double-log through the root logger) and pins the
    sflow logger's level. Once ANY test triggers that, it persists on the shared logger
    and silently breaks ``caplog`` for every later test that asserts on sflow log output
    -- ``caplog`` captures via a handler on the ROOT logger, reached only by propagation.
    That is why e.g. the kubelet-preflight warning tests pass in isolation but fail in the
    full suite. Reset the sflow logger to a capture-friendly state (propagation on, level
    inherited) around each test and restore whatever was there afterwards.
    """
    lg = logging.getLogger("sflow")
    saved_propagate, saved_level = lg.propagate, lg.level
    lg.propagate = True
    lg.setLevel(logging.NOTSET)
    try:
        yield
    finally:
        lg.propagate = saved_propagate
        lg.setLevel(saved_level)


@pytest.fixture(autouse=True)
def setup_fake_process(fake_process):
    fake_process.register(
        "sacctmgr show user $(whoami) format=DefaultAccount -nP",
        stdout="test_account",
        returncode=0,
    )
    fake_process.register(
        'sinfo -o "%P" | grep "*" | sed "s/*//"',
        stdout="batch",
        returncode=0,
    )