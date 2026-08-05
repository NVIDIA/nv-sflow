# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sflow upgrade`` / ``sflow update``: install-route resolution and safeguards."""

import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sflow.cli import app
from sflow.utils.install import DEFAULT_SFLOW_GIT_URL

runner = CliRunner()


@pytest.fixture
def fake_env(monkeypatch):
    """Pretend sflow is a normal installed package, with uv available.

    Records the argv the command would run instead of installing anything.
    """
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "sflow.cli.upgrade.install_info", lambda: ("installed", None)
    )
    monkeypatch.setattr("sflow.cli.upgrade.shutil.which", lambda name: "/usr/bin/uv")

    def _run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("sflow.cli.upgrade.subprocess.run", _run)
    return calls


def _invoke(*args):
    return runner.invoke(app, list(args))


# ---------------------------------------------------------------------------
# Default + explicit sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["upgrade", "update"])
def test_defaults_to_oss_main_under_both_names(fake_env, name):
    result = _invoke(name)

    assert result.exit_code == 0, result.output
    assert len(fake_env) == 1
    argv = fake_env[0]
    # uv, pinned to the interpreter running sflow (not whatever venv is active).
    assert argv[:4] == ["/usr/bin/uv", "pip", "install", "--python"]
    assert argv[4] == sys.executable
    assert argv[5:] == [
        "--reinstall-package",
        "sflow",
        "--prerelease=allow",
        f"sflow @ git+{DEFAULT_SFLOW_GIT_URL}@main",
    ]


def test_branch_only_uses_the_default_repo(fake_env):
    assert _invoke("upgrade", "--branch", "develop").exit_code == 0
    assert fake_env[0][-1] == f"sflow @ git+{DEFAULT_SFLOW_GIT_URL}@develop"


def test_repo_only_uses_main(fake_env):
    assert _invoke("upgrade", "--repo", "https://git.example.com/x/sflow.git").exit_code == 0
    assert fake_env[0][-1] == "sflow @ git+https://git.example.com/x/sflow.git@main"


def test_repo_and_branch_together(fake_env):
    assert (
        _invoke(
            "upgrade",
            "--repo",
            "https://git.example.com/x/sflow.git",
            "--branch",
            "topic",
        ).exit_code
        == 0
    )
    assert fake_env[0][-1] == "sflow @ git+https://git.example.com/x/sflow.git@topic"


def test_sflow_version_ref_matches_batch_syntax(fake_env):
    assert _invoke("upgrade", "--sflow-version", "v0.1.0").exit_code == 0
    assert fake_env[0][-1] == f"sflow @ git+{DEFAULT_SFLOW_GIT_URL}@v0.1.0"


def test_sflow_version_repo_at_ref_matches_batch_syntax(fake_env):
    assert (
        _invoke("upgrade", "--sflow-version", "https://git.example.com/x/sflow.git@dev").exit_code
        == 0
    )
    assert fake_env[0][-1] == "sflow @ git+https://git.example.com/x/sflow.git@dev"


def test_index_url_route_builds_a_pypi_requirement(fake_env):
    assert (
        _invoke(
            "upgrade",
            "--sflow-index-url",
            "https://host/simple",
            "--sflow-version",
            ">=0.2,<0.3",
        ).exit_code
        == 0
    )
    argv = fake_env[0]
    assert "--extra-index-url" in argv and "https://host/simple" in argv
    assert argv[-1] == "sflow>=0.2,<0.3"


def test_source_path_installs_editable(fake_env, tmp_path):
    assert _invoke("upgrade", "--sflow-source-path", str(tmp_path)).exit_code == 0
    argv = fake_env[0]
    assert argv[-1] == f"-e{tmp_path}"
    # A local editable install must not be forced/reinstalled like a moving ref.
    assert "--reinstall-package" not in argv


# ---------------------------------------------------------------------------
# Safeguards
# ---------------------------------------------------------------------------


def test_refuses_to_overwrite_an_editable_dev_install(monkeypatch):
    monkeypatch.setattr(
        "sflow.cli.upgrade.install_info", lambda: ("editable", "/repo/sflow")
    )
    result = _invoke("upgrade")

    assert result.exit_code == 1
    assert "development) install" in result.output
    assert "/repo/sflow" in result.output
    assert "--force" in result.output


def test_force_overrides_the_dev_install_refusal(fake_env, monkeypatch):
    monkeypatch.setattr(
        "sflow.cli.upgrade.install_info", lambda: ("editable", "/repo/sflow")
    )
    assert _invoke("upgrade", "--force").exit_code == 0
    assert fake_env, "install should have run"


def test_dry_run_prints_the_plan_without_installing(fake_env, monkeypatch):
    monkeypatch.setattr(
        "sflow.cli.upgrade.install_info", lambda: ("editable", "/repo/sflow")
    )
    result = _invoke("upgrade", "--dry-run")

    assert result.exit_code == 0
    assert "Dry run: nothing installed." in result.output
    # Dry run is safe on a dev install: it inspects, it does not overwrite.
    assert fake_env == []


def test_dry_run_command_is_copy_pasteable(fake_env):
    result = _invoke("upgrade", "--dry-run")
    # The requirement contains a space, so it must be quoted in the echoed command.
    assert f"'sflow @ git+{DEFAULT_SFLOW_GIT_URL}@main'" in result.output


def test_falls_back_to_pip_when_uv_is_missing(fake_env, monkeypatch):
    monkeypatch.setattr("sflow.cli.upgrade.shutil.which", lambda name: None)
    # This repo's venv is uv-managed and has no pip, so the fallback must be told
    # pip exists -- otherwise the availability guard fires instead.
    monkeypatch.setattr(
        "sflow.cli.upgrade.importlib.util.find_spec", lambda name: object()
    )
    assert _invoke("upgrade").exit_code == 0
    argv = fake_env[0]
    assert argv[1:3] == ["-m", "pip"]
    # pip has no per-package reinstall, so it force-reinstalls instead.
    assert "--force-reinstall" in argv and "--pre" in argv


def test_installer_failure_propagates_the_exit_code(monkeypatch):
    monkeypatch.setattr("sflow.cli.upgrade.install_info", lambda: ("installed", None))
    monkeypatch.setattr("sflow.cli.upgrade.shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        "sflow.cli.upgrade.subprocess.run",
        lambda argv, **kw: SimpleNamespace(returncode=7),
    )
    result = _invoke("upgrade")

    assert result.exit_code == 7
    assert "install failed" in result.output


# ---------------------------------------------------------------------------
# Conflicting / invalid flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--sflow-version", "v1", "--branch", "x"], "already encodes the source"),
        (["--sflow-version", "v1", "--repo", "https://h/x.git"], "already encodes the source"),
        (
            ["--repo", "https://h/x.git", "--sflow-index-url", "https://i/simple"],
            "cannot be combined with --sflow-index-url",
        ),
    ],
)
def test_conflicting_source_flags_are_rejected(fake_env, args, expected):
    result = _invoke("upgrade", *args)
    assert result.exit_code == 1
    assert expected in result.output
    assert fake_env == []


def test_source_path_conflicts_are_rejected(fake_env, tmp_path):
    result = _invoke(
        "upgrade", "--sflow-source-path", str(tmp_path), "--branch", "develop"
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output
    assert fake_env == []


def test_index_url_with_embedded_credentials_is_rejected(fake_env):
    result = _invoke("upgrade", "--sflow-index-url", "https://u:p@host/simple")
    assert result.exit_code == 1
    assert "embedded credentials" in result.output
    assert fake_env == []


def test_git_route_rejects_a_version_specifier(fake_env):
    result = _invoke("upgrade", "--sflow-version", ">=0.2, <0.3")
    assert result.exit_code == 1
    assert "not a valid git ref" in result.output
    assert fake_env == []


def test_pypi_route_rejects_a_direct_reference(fake_env):
    result = _invoke(
        "upgrade",
        "--sflow-index-url",
        "https://host/simple",
        "--sflow-version",
        "https://h/x.git",
    )
    assert result.exit_code == 1
    assert "not a valid PyPI version specifier" in result.output
    assert fake_env == []


def test_errors_clearly_when_neither_uv_nor_pip_is_available(monkeypatch):
    # uv-created venvs often ship without pip; the fallback must say so rather
    # than dying with a bare "No module named pip".
    monkeypatch.setattr("sflow.cli.upgrade.install_info", lambda: ("installed", None))
    monkeypatch.setattr("sflow.cli.upgrade.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "sflow.cli.upgrade.importlib.util.find_spec", lambda name: None
    )
    result = _invoke("upgrade")

    assert result.exit_code == 1
    assert "neither 'uv' nor 'pip' is available" in result.output
    assert "ensurepip" in result.output
