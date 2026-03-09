# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


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