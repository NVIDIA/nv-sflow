# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sflow upgrade`` / ``sflow update`` -- reinstall sflow into the current env.

With no flags this installs the latest ``nv-sflow`` release from PyPI, like
``pip install --upgrade`` would. ``--sflow-version`` narrows that to a PEP 440
version or specifier, ``--sflow-index-url`` adds a private index (an Artifactory
PyPI repo) that is searched ahead of PyPI, and ``--repo`` / ``--branch`` switch
to installing a git ref instead -- the route for pre-release branches.

That is deliberately different from ``sflow batch``, which defaults to whatever
ref the *running* environment was installed from, and whose ``--sflow-version``
is a git ref unless an index URL is given: ``upgrade`` is an explicit "get me the
latest release" action. The shared validation lives in :mod:`sflow.utils.install`.
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

from sflow.utils.distribution import (
    DIST_NAME,
    FIRST_PYPI_VERSION,
    LEGACY_DIST_NAME,
    non_editable_records,
    reconcile,
)
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
) -> tuple[str, str, str]:
    """Return ``(route, requirement, human_description)`` for the selected route.

    ``route`` is ``"editable"``, ``"git"`` or ``"pypi"``; the caller shapes the
    installer flags from it rather than re-deriving the decision from the flags.

    Rejects flag combinations that describe two different sources at once: a
    PyPI version pin or index URL cannot be applied to a git checkout.
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
        return "editable", f"-e{sflow_source_path}", f"editable install from {sflow_source_path}"

    if explicit_git:
        conflicting = [
            name
            for name, value in (
                ("--sflow-version", sflow_version),
                ("--sflow-index-url", sflow_index_url),
            )
            if value is not None
        ]
        if conflicting:
            _fail(
                f"--repo/--branch install a git ref and cannot be combined with "
                f"{', '.join(conflicting)}; a PyPI version or index does not apply "
                "to a git checkout."
            )
        # Bare VCS URL, no 'nv-sflow @' prefix: a named requirement fails on any
        # ref that still builds the pre-rename 'sflow' distribution.
        url = sflow_git_install_url(sflow_git_spec(repo, branch))
        return "git", url, url

    if sflow_version is None:
        # "Latest" never reaches below the first migration-aware release: the
        # pre-rename uploads on pypi.org must not be what a plain upgrade picks.
        requirement = f"{DIST_NAME}>={FIRST_PYPI_VERSION}"
    else:
        requirement = sflow_pypi_requirement(sflow_version)
    where = f"index {sflow_index_url}" if sflow_index_url else "PyPI"
    return "pypi", requirement, f"{requirement} from {where}"


def upgrade(
    repo: Annotated[
        Optional[str],
        typer.Option(
            "--repo",
            help="Install a git checkout instead of a PyPI release: the repository "
            f"URL (default: the public OSS repo, {DEFAULT_SFLOW_GIT_URL}). "
            "Mutually exclusive with --sflow-version/--sflow-index-url.",
        ),
    ] = None,
    branch: Annotated[
        Optional[str],
        typer.Option(
            "--branch",
            help="Install a git checkout instead of a PyPI release: the branch or tag "
            f"(default: '{DEFAULT_SFLOW_GIT_BRANCH}'). Combined with --repo. "
            "Mutually exclusive with --sflow-version/--sflow-index-url.",
        ),
    ] = None,
    sflow_version: Annotated[
        Optional[str],
        typer.Option(
            "--sflow-version",
            help="PyPI version to install: a bare version is pinned ('0.3.2' -> "
            "'nv-sflow==0.3.2'), an operator spec is passed through ('>=0.3,<0.4'). "
            "Default: the latest release. For a git ref use --repo/--branch.",
        ),
    ] = None,
    sflow_index_url: Annotated[
        Optional[str],
        typer.Option(
            "--sflow-index-url",
            help="Also search a private PyPI index (e.g. an Artifactory PyPI repo), "
            "picking the best version across it and PyPI. Credentials must come from "
            "~/.netrc or a credential helper; URLs with embedded credentials are "
            "rejected.",
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

    Defaults to the latest nv-sflow release on PyPI.
    """
    index_url_error = sflow_index_url_error(
        sflow_index_url, hint="use ~/.netrc or a credential helper instead."
    )
    if index_url_error:
        _fail(index_url_error)

    route, requirement, description = _resolve_target(
        repo=repo,
        branch=branch,
        sflow_version=sflow_version,
        sflow_index_url=sflow_index_url,
        sflow_source_path=sflow_source_path,
    )

    version_error = sflow_version_error(sflow_version, registry=True)
    if version_error:
        _fail(f"{version_error} For a git branch or tag use --branch/--repo.")

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
    uninstall: list[str] = []
    if route == "editable":
        # A wheel under either name keeps its sflow/ files in site-packages, ahead
        # of the editable .pth, so the checkout would be shadowed. pip replaces
        # only the same name; the other one has to go first. Its console script
        # is recreated by the editable install.
        shadowing = non_editable_records()
        if shadowing:
            uninstall = [a if a != "install" else "uninstall" for a in argv]
            uninstall += ([] if is_uv else ["-y"]) + shadowing
    if route == "git":
        # A branch head moves without the version string changing, so a plain
        # install can decide sflow is already satisfied and do nothing. Force
        # exactly sflow to be reinstalled under either name the ref may build;
        # uv can scope that to those packages, pip cannot. Git checkouts build
        # as .devN versions, so pre-releases must be allowed for the result to
        # be installable at all.
        argv += (
            ["--reinstall-package", DIST_NAME, "--reinstall-package", LEGACY_DIST_NAME]
            if is_uv
            else ["--force-reinstall"]
        )
        argv += ["--prerelease=allow"] if is_uv else ["--pre"]
    elif route == "pypi":
        # uv's --upgrade upgrades every dependency too; pip's only-if-needed
        # default leaves them alone. Keep a shared env's other packages as-is.
        # nv-sflow itself is always reinstalled: a dev build -- a git checkout or
        # a .devN wheel from a private index -- is versioned above the latest
        # release, so a plain upgrade would keep it instead of returning to PyPI.
        # pip has no per-package reinstall, so there it is --force-reinstall.
        argv += (
            ["--upgrade-package", DIST_NAME, "--reinstall-package", DIST_NAME]
            if is_uv
            else ["--upgrade", "--force-reinstall"]
        )
    if sflow_index_url is not None:
        # No --prerelease: an extra index must not turn "latest release" into the
        # newest .devN build; pin one with --sflow-version to install it. No
        # unsafe-best-match either: every release reaches the private index before
        # PyPI, and best-match would compare every dependency across indexes.
        argv += ["--extra-index-url", sflow_index_url]
    argv.append(requirement)

    typer.echo(f"Current install : {mode}{f' ({source})' if source else ''}")
    typer.echo(f"Installing      : {description}")
    if uninstall:
        typer.echo(f"Removing first  : {shlex.join(uninstall)}")
    # shlex.join so the echoed command stays copy-pasteable whatever the
    # requirement contains (specifier operators, local paths with spaces).
    typer.echo(f"Command         : {shlex.join(argv)}")

    if dry_run:
        typer.echo("Dry run: nothing installed.")
        return

    if uninstall:
        _run(uninstall)
    _run(argv)
    # Crossing the rename (nv-sflow over a GitHub 'sflow' build, or the reverse
    # via --branch) leaves both records over one set of files. Install first,
    # then drop the stale record: the files are never missing in between.
    reconcile()

    typer.echo("Upgrade complete. Run 'sflow --version' to confirm.")


def _run(argv: list[str]) -> None:
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


# Registered under both names: 'upgrade' is the primary spelling, 'update' is
# accepted because both are muscle memory depending on the tool.
app.command(name="upgrade", epilog=f"Documentation: {DOCS_URL}")(upgrade)
app.command(name="update", epilog=f"Documentation: {DOCS_URL}")(upgrade)
