# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from sflow.cli import app
from sflow import runtime_info


runner = CliRunner()


def test_run_logs_executable_info_at_startup(tmp_path):
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        """
version: "0.1"
workflow:
  name: test
  tasks:
    - name: hello
      script:
        - echo hello
""".lstrip()
    )

    with patch("sflow.cli.run._sflow_app") as mock_app:
        mock_app.run = MagicMock(return_value=None)
        mock_app.last_workflow_output_dir = None

        result = runner.invoke(
            app,
            ["run", "-f", str(workflow), "--dry-run"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert "sflow executable" in result.output
    assert "╭─ sflow executable " in result.output
    assert "╰" in result.output
    assert "version :" in result.output
    assert "bin     :" in result.output
    assert "python  :" in result.output
    assert "package :" in result.output
    assert "install :" in result.output


def test_run_prints_summary_and_command_log_paths(tmp_path):
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        """
version: "0.1"
workflow:
  name: test
  tasks:
    - name: hello
      script:
        - echo hello
""".lstrip()
    )
    workflow_out_dir = tmp_path / "sflow_output" / "test-run"
    workflow_out_dir.mkdir(parents=True)
    (workflow_out_dir / "bash_cmds.log").write_text("cmd\n")

    with patch("sflow.cli.run._sflow_app") as mock_app:
        mock_app.run = MagicMock(return_value=workflow_out_dir)
        mock_app.last_workflow_output_dir = workflow_out_dir

        result = runner.invoke(
            app,
            ["run", "-f", str(workflow)],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert f"Summary: {workflow_out_dir / 'sflow_summary.log'}" in result.output
    assert f"Command logs: {workflow_out_dir / 'bash_cmds.log'}" in result.output


def test_format_runtime_info_reports_editable_direct_url(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    class FakeDistribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return (
                '{"url": "file://'
                + str(repo)
                + '", "dir_info": {"editable": true}}'
            )

    monkeypatch.setattr(
        runtime_info.importlib_metadata,
        "distribution",
        lambda _name: FakeDistribution(),
    )
    monkeypatch.setattr(runtime_info, "_git_info_for_repo", lambda _repo: "(main abc123)")

    info = runtime_info.format_runtime_info()

    assert "install : editable" in info
    assert "source  : local editable dev" in info
    assert "repo    : " in info
    assert repo.name in info
    assert "git     : (main abc123)" in info


def test_format_runtime_info_reports_local_direct_url(monkeypatch, tmp_path):
    wheel_source = tmp_path / "dist" / "sflow"
    wheel_source.mkdir(parents=True)

    class FakeDistribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return '{"url": "file://' + str(wheel_source) + '"}'

    monkeypatch.setattr(
        runtime_info.importlib_metadata,
        "distribution",
        lambda _name: FakeDistribution(),
    )
    monkeypatch.setattr(runtime_info, "_git_info_for_repo", lambda _repo: "")

    info = runtime_info.format_runtime_info()

    assert "install : direct-url" in info
    assert "source  : local build" in info


def test_format_runtime_info_reports_vcs_direct_url_source(monkeypatch):
    class FakeDistribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return (
                '{"url": "https://github.com/NVIDIA/nv-sflow.git", '
                '"vcs_info": {"vcs": "git", "requested_revision": "develop", '
                '"commit_id": "0858dce39"}}'
            )

    monkeypatch.setattr(
        runtime_info.importlib_metadata,
        "distribution",
        lambda _name: FakeDistribution(),
    )
    monkeypatch.setattr(runtime_info, "_git_info_for_repo", lambda _repo: "")

    info = runtime_info.format_runtime_info()

    assert "install : direct-url" in info
    assert "source  : https://github.com/NVIDIA/nv-sflow.git@develop" in info
    assert "repo    :" not in info


def test_format_runtime_info_reports_imported_repo_source_as_editable(
    monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    package = repo / "src" / "sflow"
    package.mkdir(parents=True)
    (repo / ".git").mkdir()

    class FakeDistribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return None

    monkeypatch.setattr(
        runtime_info.importlib_metadata,
        "distribution",
        lambda _name: FakeDistribution(),
    )
    monkeypatch.setattr(runtime_info.sflow, "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(runtime_info, "_git_info_for_repo", lambda _repo: "")

    info = runtime_info.format_runtime_info()

    assert "install : editable" in info
    assert "source  : local editable dev" in info
    assert "repo    : " in info
    assert repo.name in info


def test_format_runtime_info_omits_parent_repo_git_for_regular_install(
    monkeypatch, tmp_path
):
    project = tmp_path / "other-project"
    package = project / ".venv" / "lib" / "python3.12" / "site-packages" / "sflow"
    package.mkdir(parents=True)

    class FakeDistribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return None

    monkeypatch.setattr(
        runtime_info.importlib_metadata,
        "distribution",
        lambda _name: FakeDistribution(),
    )
    monkeypatch.setattr(runtime_info.sflow, "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(runtime_info, "_find_repo_root", lambda _path: project)
    monkeypatch.setattr(
        runtime_info,
        "_git_info_for_repo",
        lambda repo: "(main abc123, dirty)" if repo is not None else "",
    )

    info = runtime_info.format_runtime_info()

    assert "install : installed" in info
    assert "source  : remote pypi package" in info
    assert "git     :" not in info


def test_source_label_identifies_remote_pypi_develop():
    assert (
        runtime_info._source_label(
            "installed", None, "0.2.3.dev390+develop.7e546f38"
        )
        == "remote pypi develop"
    )


def test_source_label_identifies_remote_pypi_feature_branch():
    assert (
        runtime_info._source_label(
            "installed",
            None,
            "0.2.3.dev357+feature.sflow.162.allow.backend.d5a31f78",
        )
        == "remote pypi branch"
    )


def test_source_label_identifies_remote_pypi_release():
    assert runtime_info._source_label("installed", None, "0.2.2") == "remote pypi release"


def test_format_runtime_info_uses_multiline_fields(monkeypatch):
    monkeypatch.setattr(runtime_info, "_install_info", lambda: ("installed", None))
    monkeypatch.setattr(runtime_info, "_git_info_for_repo", lambda _repo: "")
    monkeypatch.setattr(runtime_info, "_current_console_width", lambda: 120)

    info = runtime_info.format_runtime_info()
    lines = info.splitlines()

    assert lines[0].startswith("╭─ sflow executable ")
    assert lines[-1].startswith("╰")
    assert any("│ version : " in line for line in lines)
    assert any("│ bin     : " in line for line in lines)
    assert any("│ python  : " in line for line in lines)
    assert all(line.startswith(("╭", "│", "╰")) for line in lines)
    assert all(len(line) <= 72 for line in lines)


def test_format_runtime_info_expands_with_rich_console_width(monkeypatch):
    monkeypatch.setattr(runtime_info, "_install_info", lambda: ("installed", None))
    monkeypatch.setattr(runtime_info, "_git_info_for_repo", lambda _repo: "")

    monkeypatch.setattr(runtime_info, "_current_console_width", lambda: 120)
    narrow_width = len(runtime_info.format_runtime_info().splitlines()[0])

    monkeypatch.setattr(runtime_info, "_current_console_width", lambda: 160)
    wide_width = len(runtime_info.format_runtime_info().splitlines()[0])

    assert narrow_width == 72
    assert wide_width > narrow_width
    assert wide_width <= runtime_info.RUNTIME_INFO_MAX_BOX_WIDTH


def test_format_runtime_info_wraps_long_fields(monkeypatch):
    long_repo = "/very/long/" + "/".join(f"segment{i}" for i in range(20))
    monkeypatch.setattr(runtime_info, "_install_info", lambda: ("editable", long_repo))
    monkeypatch.setattr(runtime_info, "_resolve_bin_path", lambda: long_repo + "/bin/sflow")
    monkeypatch.setattr(runtime_info, "_git_info_for_repo", lambda _repo: "")
    monkeypatch.setattr(runtime_info, "_current_console_width", lambda: 120)

    info = runtime_info.format_runtime_info()
    lines = info.splitlines()

    assert len(lines) > 8
    assert all(len(line) <= 72 for line in lines)
    assert any("│ repo    : " in line for line in lines)
    assert any("│          " in line for line in lines)
