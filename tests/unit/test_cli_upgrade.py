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
    # Never let a test edit the real environment's install records.
    monkeypatch.setattr("sflow.cli.upgrade.reconcile", lambda: None)
    monkeypatch.setattr("sflow.cli.upgrade.non_editable_records", lambda: [])

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
def test_defaults_to_latest_pypi_release_under_both_names(fake_env, name):
    result = _invoke(name)

    assert result.exit_code == 0, result.output
    assert len(fake_env) == 1
    argv = fake_env[0]
    # uv, pinned to the interpreter running sflow (not whatever venv is active).
    assert argv[:4] == ["/usr/bin/uv", "pip", "install", "--python"]
    assert argv[4] == sys.executable
    # A release install: no pre-releases (PyPI carries a 0.0.0a0 name-reservation
    # placeholder that must never win), floored at the first migration-aware
    # release (pypi.org also holds a pre-rename 0.3.1), only nv-sflow upgraded --
    # uv's bare --upgrade would upgrade every dependency -- and nv-sflow itself
    # reinstalled, so a dev build versioned above the release is replaced.
    assert argv[5:] == [
        "--upgrade-package",
        "nv-sflow",
        "--reinstall-package",
        "nv-sflow",
        "nv-sflow>=0.3.2",
    ]


def test_sflow_version_pins_a_pypi_release(fake_env):
    assert _invoke("upgrade", "--sflow-version", "0.3.1").exit_code == 0
    assert fake_env[0][-1] == "nv-sflow==0.3.1"


def test_sflow_version_specifier_is_passed_through(fake_env):
    assert _invoke("upgrade", "--sflow-version", ">=0.3,<0.4").exit_code == 0
    assert fake_env[0][-1] == "nv-sflow>=0.3,<0.4"


def test_branch_switches_to_the_git_route(fake_env):
    assert _invoke("upgrade", "--branch", "develop").exit_code == 0
    argv = fake_env[0]
    assert argv[-1] == f"git+{DEFAULT_SFLOW_GIT_URL}@develop"
    # A moving ref needs the forced reinstall and .devN versions allowed.
    assert "--reinstall-package" in argv and "--prerelease=allow" in argv
    assert "--upgrade-package" not in argv
    # The ref may build either name (pre-rename refs build 'sflow').
    assert argv[argv.index("sflow") - 1] == "--reinstall-package"


def test_repo_only_uses_main(fake_env):
    assert _invoke("upgrade", "--repo", "https://git.example.com/x/sflow.git").exit_code == 0
    assert fake_env[0][-1] == "git+https://git.example.com/x/sflow.git@main"


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
    assert fake_env[0][-1] == "git+https://git.example.com/x/sflow.git@topic"


def test_index_url_is_searched_alongside_pypi(fake_env):
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
    # A private index must not widen what gets installed: no .devN builds unless
    # pinned, and no cross-index version comparison for the dependencies.
    assert "--prerelease=allow" not in argv
    assert "--index-strategy" not in argv
    assert argv[-1] == "nv-sflow>=0.2,<0.3"


def test_source_path_installs_editable(fake_env, tmp_path):
    assert _invoke("upgrade", "--sflow-source-path", str(tmp_path)).exit_code == 0
    argv = fake_env[0]
    assert argv[-1] == f"-e{tmp_path}"
    # A local editable install must not be forced/reinstalled like a moving ref.
    assert "--reinstall-package" not in argv


@pytest.mark.parametrize("uv", [True, False])
def test_source_path_removes_a_shadowing_wheel_first(fake_env, monkeypatch, tmp_path, uv):
    # A GitHub checkout builds 'sflow'; an nv-sflow wheel already in site-packages
    # would keep winning the import. It is uninstalled before the editable install.
    monkeypatch.setattr("sflow.cli.upgrade.non_editable_records", lambda: ["nv-sflow"])
    if not uv:
        monkeypatch.setattr("sflow.cli.upgrade.shutil.which", lambda name: None)
        monkeypatch.setattr("sflow.cli.upgrade.importlib.util.find_spec", lambda name: object())
    result = _invoke("upgrade", "--sflow-source-path", str(tmp_path))
    assert result.exit_code == 0, result.output
    uninstall, install = fake_env
    assert "uninstall" in uninstall and uninstall[-1] == "nv-sflow"
    assert ("-y" in uninstall) is (not uv)
    assert "install" in install and install[-1] == f"-e{tmp_path}"
    assert "Removing first" in result.output


def test_source_path_dry_run_shows_the_removal_without_running_it(fake_env, monkeypatch, tmp_path):
    monkeypatch.setattr("sflow.cli.upgrade.non_editable_records", lambda: ["nv-sflow"])
    result = _invoke("upgrade", "--sflow-source-path", str(tmp_path), "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Removing first" in result.output and "uninstall" in result.output
    assert fake_env == []


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
    result = _invoke("upgrade", "--sflow-version", ">=0.3,<0.4", "--dry-run")
    # Specifier operators are shell metacharacters, so the echoed command must quote them.
    assert "'nv-sflow>=0.3,<0.4'" in result.output


def test_git_route_installs_a_bare_vcs_url(fake_env):
    # No 'nv-sflow @' prefix: refs from before the rename build a distribution
    # named 'sflow', and uv/pip refuse a named requirement whose metadata differs.
    assert _invoke("upgrade", "--branch", "v0.3.1").exit_code == 0
    assert fake_env[0][-1] == f"git+{DEFAULT_SFLOW_GIT_URL}@v0.3.1"
    assert not fake_env[0][-1].startswith("nv-sflow")


def test_falls_back_to_pip_when_uv_is_missing(fake_env, monkeypatch):
    monkeypatch.setattr("sflow.cli.upgrade.shutil.which", lambda name: None)
    # This repo's venv is uv-managed and has no pip, so the fallback must be told
    # pip exists -- otherwise the availability guard fires instead.
    monkeypatch.setattr(
        "sflow.cli.upgrade.importlib.util.find_spec", lambda name: object()
    )
    assert _invoke("upgrade").exit_code == 0
    assert fake_env[0][1:] == [
        "-m", "pip", "install", "--upgrade", "--force-reinstall", "nv-sflow>=0.3.2"
    ]
    # pip has no per-package reinstall, so the git route force-reinstalls instead.
    assert _invoke("upgrade", "--branch", "develop").exit_code == 0
    assert "--force-reinstall" in fake_env[1] and "--pre" in fake_env[1]


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
        (["--sflow-version", "0.3.1", "--branch", "x"], "cannot be combined with --sflow-version"),
        (["--sflow-version", "0.3.1", "--repo", "https://h/x.git"], "cannot be combined with --sflow-version"),
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


@pytest.mark.parametrize("value", ["https://h/x.git", "repo@develop", "main"])
def test_sflow_version_rejects_git_style_values(fake_env, value):
    # The batch spelling of a git ref no longer applies here; --repo/--branch does.
    result = _invoke("upgrade", "--sflow-version", value)
    assert result.exit_code == 1
    assert "not a valid PyPI version specifier" in result.output
    assert "use --branch/--repo" in result.output
    assert fake_env == []


def test_flag_conflict_is_reported_before_the_version_check(fake_env):
    # 'main' is not a PyPI version, but the real mistake is pairing it with --branch.
    result = _invoke("upgrade", "--branch", "x", "--sflow-version", "main")
    assert result.exit_code == 1
    assert "cannot be combined with --sflow-version" in result.output
    assert fake_env == []


@pytest.mark.parametrize("args", [[], ["--branch", "v0.3.1"]])
def test_install_then_reconcile_never_uninstall(fake_env, monkeypatch, args):
    # Crossing the rename leaves both records over one set of files. The stale
    # record is dropped after the install; never uninstall + reinstall, which
    # leaves the files missing in between and needs the network to recover.
    monkeypatch.setattr(
        "sflow.cli.upgrade.reconcile", lambda: fake_env.append(["reconcile"])
    )
    assert _invoke("upgrade", *args).exit_code == 0
    install, reconciled = fake_env
    assert install[2] == "install"
    assert reconciled == ["reconcile"]


def test_dry_run_neither_installs_nor_reconciles(fake_env, monkeypatch):
    monkeypatch.setattr(
        "sflow.cli.upgrade.reconcile", lambda: fake_env.append(["reconcile"])
    )
    assert _invoke("upgrade", "--dry-run").exit_code == 0
    assert fake_env == []


@pytest.mark.parametrize("mode", ["installed", "direct-url"])
def test_plain_upgrade_returns_a_dev_build_to_pypi(fake_env, monkeypatch, mode):
    # A dev build -- a git checkout (direct-url) or a .devN wheel from a private
    # index (installed) -- is versioned above the latest release; without a
    # reinstall the upgrade would keep it instead of going to PyPI.
    monkeypatch.setattr(
        "sflow.cli.upgrade.install_info", lambda: (mode, "https://h/x.git@main")
    )
    assert _invoke("upgrade").exit_code == 0
    argv = fake_env[0]
    assert argv[argv.index("--reinstall-package") + 1] == "nv-sflow"
    assert argv[-1] == "nv-sflow>=0.3.2"


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
