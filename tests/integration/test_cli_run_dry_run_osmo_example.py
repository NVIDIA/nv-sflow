# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OSMO example dry-runs without the ``osmo`` CLI (no submit, no login)."""

from pathlib import Path

from typer.testing import CliRunner

from sflow.cli import app


def test_osmo_hello_world_dry_run(tmp_path: Path, monkeypatch):
    """Dry-run validates the example and shows the backend pool and platform."""
    monkeypatch.setenv("PATH", str(tmp_path))  # no osmo CLI on PATH, as on CI runners
    repo_root = Path(__file__).resolve().parents[2]
    cfg = repo_root / "examples" / "self_contained/osmo/hello_world.yaml"

    result = CliRunner().invoke(
        app,
        ["run", "-f", str(cfg), "--dry-run", "--workspace-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "pool: dht-l40-05" in result.output
    assert "platform: ovx-l40" in result.output
