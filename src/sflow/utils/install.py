# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for building an sflow install specification.

Two commands need to answer the same question -- "which sflow should be
installed, from where?" -- and must answer it identically:

* ``sflow batch`` bakes the install into each generated Slurm job script, so a
  compute node bootstraps the same sflow the submitter intended.
* ``sflow upgrade`` installs into the environment the user is running right now.

Both accept ``--sflow-version`` / ``--sflow-index-url``, so the ref parsing,
requirement building and validation live here rather than being duplicated (or
imported across CLI modules).

Two mutually exclusive install routes are supported:

* **git** -- a branch/tag, or a ``repo-url@ref``, installed as ``sflow @ git+...``.
* **PyPI index** -- a PEP 440 version/specifier installed as ``sflow==...`` from
  ``--sflow-index-url``.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Public OSS repository. The default source for both the batch job bootstrap and
# `sflow upgrade`.
DEFAULT_SFLOW_GIT_URL = "https://github.com/NVIDIA/nv-sflow.git"

# Default git ref used when none is given.
DEFAULT_SFLOW_GIT_BRANCH = "main"


def sflow_git_install_url(sflow_version_or_url: str | None) -> str:
    """Return a pip-compatible git URL for installing sflow."""
    value = sflow_version_or_url or DEFAULT_SFLOW_GIT_BRANCH
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        if parsed.scheme.startswith("git+"):
            return value
        return f"git+{value}"
    return f"git+{DEFAULT_SFLOW_GIT_URL}@{value}"


def sflow_git_spec(repo: str | None, branch: str | None) -> str:
    """Combine an explicit ``--repo`` / ``--branch`` pair into a single ref spec.

    The result is whatever :func:`sflow_git_install_url` expects: a bare ref when
    the default repo is in play, otherwise ``repo-url@ref``. Either side may be
    omitted and falls back to its default.
    """
    ref = (branch or "").strip() or DEFAULT_SFLOW_GIT_BRANCH
    repo_url = (repo or "").strip() or DEFAULT_SFLOW_GIT_URL
    if repo_url == DEFAULT_SFLOW_GIT_URL:
        return ref
    return f"{repo_url}@{ref}"


def sflow_pypi_requirement(sflow_version: str | None) -> str:
    """Build the ``sflow`` requirement for a PyPI-index install (PyPI route).

    ``sflow_version`` is a PEP 440 version/specifier already validated by
    :func:`sflow_version_error`: a bare version is pinned (``0.2.1`` ->
    ``sflow==0.2.1``), an operator-led spec is kept (``>=0.2,<0.3`` ->
    ``sflow>=0.2,<0.3``), and an empty value installs the latest (``sflow``).
    """
    spec = (sflow_version or "").strip()
    if not spec:
        return "sflow"
    if spec[0] in "=<>!~":
        return f"sflow{spec}"
    return f"sflow=={spec}"


def sflow_index_url_error(
    sflow_index_url: str | None,
    *,
    hint: str = "use ~/.netrc or a credential helper on the compute node instead.",
) -> str | None:
    """Reject an index URL that smuggles credentials into the install command.

    The URL is embedded verbatim into a ``uv pip install --extra-index-url ...``
    invocation, so inline credentials would leak into generated scripts and logs.
    Query strings and fragments are rejected for the same reason. Returns an error
    message, or ``None`` when the URL is acceptable.
    """
    if sflow_index_url is None:
        return None
    parsed = urlparse(sflow_index_url)
    if parsed.username is not None or parsed.password is not None:
        return f"--sflow-index-url must not contain embedded credentials; {hint}"
    if parsed.query or parsed.fragment:
        return (
            f"--sflow-index-url must not include query parameters or fragments; {hint}"
        )
    return None


def sflow_version_error(
    sflow_version: str | None,
    *,
    registry: bool,
    option: str = "--sflow-version",
) -> str | None:
    """Sanity-check a version value for the active install route.

    * **git** (``registry=False``, i.e. no ``--sflow-index-url``): a git
      branch/tag or a ``repo-url@ref``, installed as ``sflow @ git+...``.
    * **PyPI index** (``registry=True``, ``--sflow-index-url`` set): a PEP 440
      version (``0.2.1``) or specifier (``>=0.2,<0.3``), installed as
      ``sflow==...`` from that index.

    Returns an error message, or ``None`` when the value is acceptable. An empty
    value is always accepted -- each route has a sensible default (git falls back
    to the caller's default ref; PyPI installs the latest). ``option`` names the
    flag in the message so callers can reuse this for differently-named flags.
    """
    spec = (sflow_version or "").strip()
    if not spec:
        return None

    if not registry:
        # Git route: refs and repo URLs never contain whitespace -- reject it as a
        # likely mistake (a stray version spec, shell-quoted args, ...).
        if any(ch.isspace() for ch in spec):
            return (
                f"{option} '{sflow_version}' is not a valid git ref or repo URL "
                "(whitespace is not allowed). Pass a branch/tag like 'main' or a "
                "'repo-url@ref'; for a PyPI version, add --sflow-index-url."
            )
        return None

    # PyPI route: a plain PEP 440 version/specifier only -- never a package name,
    # URL or PEP 508 direct reference, which would otherwise be embedded verbatim
    # into the generated install command. The structural reject of '@' / '://' /
    # whitespace is unconditional; the PEP 440 parse needs ``packaging`` and is
    # skipped (best-effort) when it is unavailable.
    pypi_message = (
        f"{option} '{sflow_version}' is not a valid PyPI version specifier "
        "(required with --sflow-index-url). Use a plain version like '0.2.1' or a "
        "specifier like '>=0.2,<0.3'; package names, URLs and '@' direct references "
        "are not allowed."
    )
    if "@" in spec or "://" in spec or any(ch.isspace() for ch in spec):
        return pypi_message
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
    except ImportError:
        return None
    # Bare versions ('0.2.1', '0.2.*') aren't specifiers on their own, so pin them
    # with '==' before parsing; operator-led values are validated as written.
    candidate = spec if spec[0] in "=<>!~" else f"=={spec}"
    try:
        SpecifierSet(candidate)
    except InvalidSpecifier:
        return pypi_message
    return None
