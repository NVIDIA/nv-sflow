# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which installed distribution is the running sflow, across the nv-sflow rename.

The import package is ``sflow``, but its distribution has two names: ``nv-sflow``
(PyPI, every GitLab build) and ``sflow`` (GitHub builds and every release up to
v0.3.1). Anything that reads install metadata must find whichever one owns the
package, and an environment that ends up with both -- ``pip install nv-sflow``
over an old git install -- is reconciled down to one by :func:`reconcile`.

Stdlib only: ``sflow/__init__.py`` imports this for ``__version__``.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import importlib.metadata as importlib_metadata
import os
import shutil
import sys
from pathlib import Path

DIST_NAME = "nv-sflow"
LEGACY_DIST_NAME = "sflow"
# First release published as nv-sflow and the first that carries reconcile().
# pypi.org also holds earlier uploads under the name (a 0.0.0a0 placeholder, a
# 0.3.1 built before this module existed); nothing may resolve to those.
FIRST_PYPI_VERSION = "0.3.2"
NO_AUTO_MIGRATE_ENV = "SFLOW_NO_AUTO_MIGRATE"
_PACKAGE = "sflow"
_CONSOLE_SCRIPT = ("console_scripts", "sflow", "sflow.cli:app")


def _find(name: str) -> importlib_metadata.Distribution | None:
    try:
        return importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def installed_distribution() -> importlib_metadata.Distribution | None:
    """The distribution that owns the ``sflow`` package, or ``None`` (source tree).

    ``nv-sflow`` wins while both are installed (before :func:`reconcile` has run).
    """
    dist = _find(DIST_NAME)
    if dist is None:
        # 'sflow' on pypi.org is an unrelated project: a record under the old
        # name is ours only if it installs our console script.
        legacy = _find(LEGACY_DIST_NAME)
        dist = legacy if legacy is not None and _is_ours(legacy) else None
    if dist is not None:
        return dist
    # Only a fork that renamed the distribution again gets here. Checked last
    # because it reads every installed RECORD (~1s in a large env) where the two
    # lookups above take under a millisecond, and this runs on every CLI start.
    for name in importlib_metadata.packages_distributions().get(_PACKAGE, []):
        dist = _find(name)
        if dist is not None:
            return dist
    return None


def _is_editable(dist: importlib_metadata.Distribution) -> bool:
    try:
        info = json.loads(dist.read_text("direct_url.json") or "{}")
    except (OSError, ValueError):
        return False
    return bool((info.get("dir_info") or {}).get("editable"))


def non_editable_records() -> list[str]:
    """Names among ``nv-sflow``/``sflow`` installed as a wheel (not editable).

    Their ``sflow/`` files sit in site-packages, ahead of any editable ``.pth``,
    so they shadow a checkout installed under the other name.
    """
    found = []
    for name in (DIST_NAME, LEGACY_DIST_NAME):
        dist = _find(name)
        if dist is not None and not _is_editable(dist):
            found.append(name)
    return found


def _record_matches_disk(dist: importlib_metadata.Distribution) -> bool:
    """Whether every hashed ``sflow/`` file in ``dist``'s RECORD is the file on disk."""
    checked = 0
    for entry in dist.files or []:
        if entry.hash is None or not entry.parts or entry.parts[0] != _PACKAGE:
            continue
        try:
            data = Path(entry.locate()).read_bytes()
            digest = hashlib.new(entry.hash.mode, data).digest()
        except (OSError, ValueError):
            return False
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != entry.hash.value:
            return False
        checked += 1
    return checked > 0


def _is_ours(dist: importlib_metadata.Distribution) -> bool:
    return any(
        (ep.group, ep.name, ep.value) == _CONSOLE_SCRIPT for ep in dist.entry_points
    )


def _record_in(site: Path, name: str) -> tuple[importlib_metadata.Distribution, Path] | None:
    """``name``'s distribution and its ``*.dist-info`` dir, looked up in ``site`` only."""
    for dist in importlib_metadata.distributions(name=name, path=[str(site)]):
        path = getattr(dist, "_path", None)
        if path is not None and Path(path).suffix == ".dist-info":
            return dist, Path(path)
    return None


def reconcile(site: Path | None = None) -> str | None:
    """Drop the stale record when both ``sflow`` and ``nv-sflow`` are installed.

    Both distributions own the same ``sflow/`` files, so the one whose RECORD
    hashes match the files on disk is the one actually installed; the other is a
    leftover record. Deleting only that record's ``*.dist-info`` leaves one
    distribution, touches no package file, needs no network or pip -- and makes a
    later ``pip uninstall sflow`` a no-op instead of deleting the live files.

    Only the site-packages directory the running ``sflow`` was imported from is
    examined (``site`` overrides it for tests), so an editable checkout or a
    record on another ``sys.path`` entry is never touched.

    Returns the name of the dropped distribution, or ``None`` when nothing was done.
    Never raises: anything unexpected (read-only site-packages, a concurrent run
    that got there first) leaves the environment as it was.
    """
    try:
        return _reconcile(site)
    except Exception:  # unreadable/corrupt metadata: leave the environment alone
        return None


def _reconcile(site: Path | None) -> str | None:
    if os.environ.get(NO_AUTO_MIGRATE_ENV) == "1":
        return None
    if site is None:
        import sflow

        site = Path(sflow.__file__).resolve().parent.parent
    found = {name: _record_in(site, name) for name in (DIST_NAME, LEGACY_DIST_NAME)}
    if None in found.values():
        return None
    dists = {name: rec[0] for name, rec in found.items()}
    editable = {name for name, dist in dists.items() if _is_editable(dist)}
    if editable:
        # An editable record lists no hashed sflow/ files, so it can never "match
        # the disk"; it is a different install, not a leftover, and is kept. The
        # wheel under the other name is what shadows it (its files are the ones
        # this process imported), and only uninstalling it fixes that.
        if len(editable) == 1:
            (checkout,) = editable
            (wheel,) = set(dists) - editable
            sys.stderr.write(
                f"sflow: the '{wheel}' wheel shadows the editable '{checkout}' checkout; "
                f"remove it with: pip uninstall {wheel}\n"
            )
        return None
    current = {name: _record_matches_disk(dist) for name, dist in dists.items()}
    if current[DIST_NAME]:
        stale = LEGACY_DIST_NAME
    elif current[LEGACY_DIST_NAME]:
        stale = DIST_NAME
    else:
        sys.stderr.write(
            "sflow: both 'sflow' and 'nv-sflow' are installed and neither matches the "
            "files on disk; repair with: pip install --force-reinstall nv-sflow\n"
        )
        return None

    dist_info = found[stale][1]
    if not _is_ours(dists[stale]) or not os.access(site, os.W_OK):
        return None
    # Renamed first so the record disappears in one atomic step; a name not ending
    # in .dist-info is invisible to importlib.metadata and pip if rmtree is cut short.
    trash = dist_info.with_name(f".{dist_info.name}.sflow-migrate-{os.getpid()}")
    try:
        os.rename(dist_info, trash)
    except OSError:
        return None
    shutil.rmtree(trash, ignore_errors=True)
    importlib.invalidate_caches()
    kept = DIST_NAME if stale == LEGACY_DIST_NAME else LEGACY_DIST_NAME
    sys.stderr.write(
        f"sflow: removed the stale '{stale}' install record; this install is "
        f"'{kept}' ({NO_AUTO_MIGRATE_ENV}=1 disables this)\n"
    )
    return stale
