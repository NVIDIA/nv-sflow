# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sflow run --wait-for-gpus`` threading: CLI flag -> env -> reservation.

The flag is consumed by the docker operator at reserve time via an env var, so
the CLI is the only place that connects them -- a flag that parses but never
reaches the env is a silent no-op. It is also validated here rather than at
reserve time, so a typo fails immediately instead of after a long run has
already started.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sflow.cli import app
from sflow.utils.gpu_reservation import WAIT_FOR_GPUS_ENV, wait_options

_CONFIG = """\
version: "0.1"
backends:
  - name: docker
    type: docker
    default: true
    image: ubuntu:22.04
    nodes: 1
    gpus_per_node: 8
workflow:
  name: wf
  tasks:
    - name: gpu_task
      resources:
        gpus:
          count: 2
      script:
        - echo hi
"""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(WAIT_FOR_GPUS_ENV, raising=False)


def _dry_run(tmp_path: Path, *extra: str):
    cfg = tmp_path / "wf.yaml"
    cfg.write_text(_CONFIG)
    return CliRunner().invoke(
        app,
        [
            "run",
            "-f",
            str(cfg),
            "--workspace-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
            *extra,
        ],
        catch_exceptions=False,
    )


def test_flag_reaches_the_reservation_env(tmp_path, monkeypatch):
    result = _dry_run(tmp_path, "--wait-for-gpus", "600")
    assert result.exit_code == 0, result.output
    import os

    assert os.environ[WAIT_FOR_GPUS_ENV] == "600"
    # ...and the reservation layer reads it as "wait up to 600s", overriding the
    # recipe-level default of fail-fast.
    assert wait_options(None) == (True, 600.0)


def test_empty_value_means_wait_forever(tmp_path):
    assert _dry_run(tmp_path, "--wait-for-gpus", "").exit_code == 0
    assert wait_options(None) == (True, None)


def test_absent_flag_leaves_the_env_untouched(tmp_path):
    import os

    assert _dry_run(tmp_path).exit_code == 0
    assert WAIT_FOR_GPUS_ENV not in os.environ
    assert wait_options(None) == (False, None)  # fail fast


@pytest.mark.parametrize("bad", ["600s", "abc", "-5"], ids=["suffix", "word", "negative"])
def test_malformed_value_fails_at_parse_time(tmp_path, bad):
    # Must fail here, not at the first GPU task of a real run: `--wait-for-gpus
    # 600s` used to sail through argument parsing AND --dry-run.
    import os

    result = _dry_run(tmp_path, "--wait-for-gpus", bad)
    assert result.exit_code != 0
    assert "--wait-for-gpus" in result.output
    assert WAIT_FOR_GPUS_ENV not in os.environ
