# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

from sflow.cli import _git_info


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
