# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sflow.app.sflow import SflowApp


def test_integration_artifacts_inline_file_materializes_and_is_consumed(tmp_path: Path):
    """
    Integration-style test: run a real workflow on the local backend and verify:
    - file:// inline content is materialized under the workflow output dir
    - artifacts are available via expression ctx and env injection
    """

    # Load the guide YAML and run it from a temp workspace so relative paths are deterministic.
    guide = Path(__file__).parent / "guide" / "sflow_artifacts.yaml"
    cfg_path = tmp_path / "sflow.yaml"
    cfg_path.write_text(guide.read_text())

    out_dir = tmp_path / "out"
    SflowApp().run(
        file=cfg_path,
        dry_run=False,
        workspace_dir=tmp_path,
        output_dir=out_dir,
    )

    # Find the single run directory under out_dir.
    runs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]

    # The inline artifact should have been materialized under the workflow output dir.
    inline_path = run_dir / "artifacts" / "inline.txt"
    assert inline_path.exists() is True
    assert inline_path.read_text().strip() == "hello from inline artifact"

    result = run_dir / "check" / "result.txt"
    assert result.exists() is True
    txt = result.read_text()
    assert f"inline_ctx={inline_path}" in txt
    assert f"inline_env={inline_path}" in txt
    assert "hello from inline artifact" in txt
    assert f"var_inline_path={inline_path}" in txt
