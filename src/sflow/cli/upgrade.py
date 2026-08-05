# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sflow upgrade`` / ``sflow update`` -- reinstall sflow into the current env.

Accepts the same install-route flags as ``sflow batch`` (``--sflow-version``,
``--sflow-index-url``, ``--sflow-source-path``) so the version you pin for a
batch job and the version you install locally are described the same way. The
shared parsing/validation lives in :mod:`sflow.utils.install`.

With no flags this installs ``main`` from the public OSS GitHub repo. That is
deliberately different from ``sflow batch``, which defaults to whatever ref the
*running* environment was installed from -- ``upgrade`` is an explicit "get me
the latest" action.
"""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from sflow.cli import DOCS_URL, app
from sflow.runtime_info import install_info
from sflow.utils.install import (
    DEFAULT_SFLOW_GIT_BRANCH,
    DEFAULT_SFLOW_GIT_URL,
    sflow_git_install_url,
    sflow_git_spec,
    sflow_index_url_error,
    sflow_pypi_requirement,
    sflow_version_error,
)

# Install modes that mean "this env points at a working tree". Overwriting one
# replaces a developer's editable checkout with a released build.
_DEV_INSTALL_MODES = {"editable", "source-tree"}


def _installer_argv() -> tuple[list[str], bool]:
    """Return the base install command and whether it is uv.

    uv is preferred (it is what the batch job bootstrap uses and it is far
    faster); ``--python`` pins it to the interpreter running sflow rather than
    whatever venv happens to be active. Plain pip is the fallback.
    """
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable], True
    if importlib.util.find_spec("pip") is None:
        # uv-created venvs routinely ship without pip, so the fallback would die
        # with a bare "No module named pip" and a useless exit code 1.
        _fail(
            "neither 'uv' nor 'pip' is available in this environment, so sflow "
            "cannot install anything. Install uv (https://docs.astral.sh/uv/) or "
            f"add pip with '{sys.executable} -m ensurepip --upgrade', then retry."
        )
    return [sys.executable, "-m", "pip", "install"], False


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _resolve_target(
    *,
    repo: str | None,
    branch: str | None,
    sflow_version: str | None,
    sflow_index_url: str | None,
    sflow_source_path: Path | None,
) -> tuple[str, str]:
    """Return ``(requirement, human_description)`` for the selected route.

    Rejects flag combinations that describe two different sources at once --
    ``--sflow-version`` already encodes ``repo-url@ref``, so pairing it with
    ``--repo``/``--branch`` is ambiguous rather than additive.
    """
    explicit_git = repo is not None or branch is not None

    if sflow_source_path is not None:
        conflicting = [
            name
            for name, value in (
                ("--sflow-version", sflow_version),
                ("--sflow-index-url", sflow_index_url),
                ("--repo", repo),
                ("--branch", branch),
            )
            if value is not None
        ]
        if conflicting:
            _fail(
                f"--sflow-source-path is mutually exclusive with {', '.join(conflicting)}; "
                "it installs from a local checkout instead of a remote source."
            )
        return f"-e{sflow_source_path}", f"editable install from {sflow_source_path}"

    if sflow_index_url is not None:
        if explicit_git:
            _fail(
                "--repo/--branch describe a git source and cannot be combined with "
                "--sflow-index-url; with an index URL, use --sflow-version as the "
                "PyPI version specifier."
            )
        requirement = sflow_pypi_requirement(sflow_version)
        return requirement, f"{requirement} from index {sflow_index_url}"

    if sflow_version is not None and explicit_git:
        _fail(
            "--sflow-version already encodes the source ('ref' or 'repo-url@ref') and "
            "cannot be combined with --repo/--branch. Use one form or the other."
        )

    spec = sflow_version if sflow_version is not None else sflow_git_spec(repo, branch)
    url = sflow_git_install_url(spec)
    return f"sflow @ {url}", url


def upgrade(
    repo: Annotated[
        Optional[str],
        typer.Option(
            "--repo",
            help="Git repository URL to install sflow from. "
            f"Defaults to the public OSS repo ({DEFAULT_SFLOW_GIT_URL}). "
            "Mutually exclusive with --sflow-version, which encodes the repo and "
            "ref together.",
        ),
    ] = None,
    branch: Annotated[
        Optional[str],
        typer.Option(
            "--branch",
            help="Git branch or tag to install "
            f"(default: '{DEFAULT_SFLOW_GIT_BRANCH}'). Combined with --repo. "
            "Mutually exclusive with --sflow-version.",
        ),
    ] = None,
    sflow_version: Annotated[
        Optional[str],
        typer.Option(
            "--sflow-version",
            help="Git ref (branch or tag) to install from the sflow repo (e.g. 'main', "
            "'v0.1.0'), or a repository URL with an @ref suffix (e.g. "
            "'https://git.example.com/example/sflow.git@develop'). Same syntax as "
            "'sflow batch --sflow-version'. When --sflow-index-url is set, this is "
            "instead interpreted as a PyPI version specifier (e.g. '0.2.1' or "
            "'>=0.2,<0.3').",
        ),
    ] = None,
    sflow_index_url: Annotated[
        Optional[str],
        typer.Option(
            "--sflow-index-url",
            help="Install sflow from a private PyPI index instead of from git. "
            "--sflow-version is then a PyPI version specifier: a bare version is "
            "pinned ('0.2.1' -> 'sflow==0.2.1'), an operator spec is passed through, "
            "and omitting it installs the latest. Credentials must come from ~/.netrc "
            "or a credential helper; URLs with embedded credentials are rejected.",
        ),
    ] = None,
    sflow_source_path: Annotated[
        Optional[Path],
        typer.Option(
            "--sflow-source-path",
            help="Path to a local sflow checkout to install editable ('pip install -e'). "
            "Mutually exclusive with the remote-source options.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Upgrade even when sflow is currently an editable/source-tree (dev) "
            "install. Without this, upgrading is refused so a developer's working "
            "checkout is not silently replaced by a released build.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the resolved install command and exit without installing.",
        ),
    ] = False,
) -> None:
    """Upgrade sflow in the current environment.

    Defaults to the 'main' branch of the public OSS GitHub repo.
    """
    index_url_error = sflow_index_url_error(
        sflow_index_url, hint="use ~/.netrc or a credential helper instead."
    )
    if index_url_error:
        _fail(index_url_error)

    version_error = sflow_version_error(
        sflow_version, registry=sflow_index_url is not None
    )
    if version_error:
        _fail(version_error)

    requirement, description = _resolve_target(
        repo=repo,
        branch=branch,
        sflow_version=sflow_version,
        sflow_index_url=sflow_index_url,
        sflow_source_path=sflow_source_path,
    )

    mode, source = install_info()
    if mode in _DEV_INSTALL_MODES and not force and not dry_run:
        where = f" ({source})" if source else ""
        _fail(
            f"sflow is currently a '{mode}' (development) install{where}. Upgrading "
            "would replace it with a released build and your local changes would no "
            "longer be used. Re-run with --force if that is what you want, or "
            "reinstall from your checkout with 'pip install -e .'."
        )

    argv, is_uv = _installer_argv()
    argv = list(argv)
    if sflow_source_path is None:
        # A branch head moves without the version string changing, so a plain
        # install can decide sflow is already satisfied and do nothing. Force
        # exactly sflow to be reinstalled; uv can scope that to one package,
        # pip cannot.
        argv += ["--reinstall-package", "sflow"] if is_uv else ["--force-reinstall"]
        argv += ["--prerelease=allow"] if is_uv else ["--pre"]
    if sflow_index_url is not None:
        argv += ["--extra-index-url", sflow_index_url]
    argv.append(requirement)

    typer.echo(f"Current install : {mode}{f' ({source})' if source else ''}")
    typer.echo(f"Installing      : {description}")
    # shlex.join, not ' '.join: the requirement is a single argv entry that
    # contains spaces ("sflow @ git+..."), so an unquoted echo would not be
    # copy-pasteable.
    typer.echo(f"Command         : {shlex.join(argv)}")

    if dry_run:
        typer.echo("Dry run: nothing installed.")
        return

    try:
        result = subprocess.run(argv, check=False)
    except OSError as exc:
        _fail(f"could not run the installer ({argv[0]}): {exc}")
        return
    if result.returncode != 0:
        typer.echo(
            f"Error: install failed with exit code {result.returncode}.", err=True
        )
        raise typer.Exit(code=result.returncode)

    typer.echo("Upgrade complete. Run 'sflow --version' to confirm.")


# Registered under both names: 'upgrade' is the primary spelling, 'update' is
# accepted because both are muscle memory depending on the tool.
app.command(name="upgrade", epilog=f"Documentation: {DOCS_URL}")(upgrade)
app.command(name="update", epilog=f"Documentation: {DOCS_URL}")(upgrade)
