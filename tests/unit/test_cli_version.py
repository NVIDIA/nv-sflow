# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from sflow import runtime_info
from sflow.cli import _git_info, app


runner = CliRunner()


def test_version_outputs_runtime_info_box_only(monkeypatch):
    monkeypatch.setattr(
        runtime_info,
        "format_runtime_info",
        lambda: "runtime info box",
    )

    result = runner.invoke(app, ["--version"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert result.output == "runtime info box\n"


def test_version_output_includes_runtime_info_fields():
    result = runner.invoke(app, ["--version"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert result.output.startswith("╭─ sflow executable ")
    assert not result.output.startswith("sflow ")
    for field in ("version :", "bin     :", "python  :", "package :", "install :"):
        assert field in result.output


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )


def _init_repo(repo: Path) -> None:
    _git(repo, "init")
    (repo / "tracked.txt").write_text("initial\n")
    _git(repo, "add", "tracked.txt")
    _git(
        repo,
        "-c",
        "user.name=sflow-test",
        "-c",
        "user.email=sflow-test@example.com",
        "commit",
        "-m",
        "initial",
    )


def test_git_info_marks_staged_changes_dirty(tmp_path: Path, monkeypatch, fp):
    fp.allow_unregistered(True)

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tracked.txt").write_text("changed\n")
    _git(tmp_path, "add", "tracked.txt")

    assert ", dirty" in _git_info()


def test_git_info_marks_untracked_files_dirty(tmp_path: Path, monkeypatch, fp):
    fp.allow_unregistered(True)

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "untracked.txt").write_text("new\n")

    assert ", dirty" in _git_info()
