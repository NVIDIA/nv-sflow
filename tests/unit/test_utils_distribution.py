# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sflow.utils.distribution``: finding the install across the nv-sflow rename,
and reconciling an environment that holds both records."""

import base64
import hashlib
import os
from types import SimpleNamespace

import pytest

from sflow.utils import distribution as dist_mod
from sflow.utils.distribution import DIST_NAME, LEGACY_DIST_NAME, reconcile

OURS = "[console_scripts]\nsflow = sflow.cli:app\n"


def _install(site, name, files, version="1.0", entry_points=OURS, editable_from=None):
    """Lay down ``files`` and a wheel-style record for ``name``, as pip would.

    ``editable_from`` writes an editable record instead: direct_url.json with
    ``dir_info.editable`` and a RECORD that lists no package files.
    """
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (info / "entry_points.txt").write_text(entry_points)
    if editable_from is not None:
        (info / "direct_url.json").write_text(
            f'{{"url": "file://{editable_from}", "dir_info": {{"editable": true}}}}'
        )
    rows = []
    for rel, data in files.items():
        path = site / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        rows.append(f"{rel},sha256={digest.decode()},{len(data)}")
    rows.append(f"{info.name}/RECORD,,")
    (info / "RECORD").write_text("\n".join(rows) + "\n")
    return info


OLD = {"sflow/__init__.py": b"old", "sflow/cli.py": b"old cli"}
NEW = {"sflow/__init__.py": b"new", "sflow/cli.py": b"new cli"}


@pytest.fixture(autouse=True)
def _migration_enabled(monkeypatch):
    monkeypatch.delenv(dist_mod.NO_AUTO_MIGRATE_ENV, raising=False)


@pytest.mark.parametrize(
    "first, second",
    [
        (LEGACY_DIST_NAME, DIST_NAME),  # pip install nv-sflow over an old git install
        (DIST_NAME, LEGACY_DIST_NAME),  # sflow upgrade --branch <GitHub ref>
    ],
)
def test_the_record_written_last_is_kept(tmp_path, capsys, first, second):
    stale = _install(tmp_path, first, OLD)
    kept = _install(tmp_path, second, NEW)

    assert reconcile(site=tmp_path) == first

    assert not stale.exists() and kept.exists()
    # Only the record goes: the live files are exactly what the kept record says.
    assert (tmp_path / "sflow/__init__.py").read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and f"stale '{first}'" in err


def test_identical_content_keeps_nv_sflow(tmp_path):
    _install(tmp_path, LEGACY_DIST_NAME, NEW)
    _install(tmp_path, DIST_NAME, NEW)

    assert reconcile(site=tmp_path) == LEGACY_DIST_NAME


@pytest.mark.parametrize("checkout, wheel", [(LEGACY_DIST_NAME, DIST_NAME), (DIST_NAME, LEGACY_DIST_NAME)])
def test_an_editable_record_is_never_stale(tmp_path, capsys, checkout, wheel):
    # `pip install -e <checkout>` next to a wheel of the other name: the editable
    # RECORD has no hashed sflow/ files, which must not read as "stale". Nothing
    # is deleted; the user is told which wheel shadows the checkout.
    _install(tmp_path, wheel, NEW)
    _install(tmp_path, checkout, {}, editable_from="/src/checkout")

    assert reconcile(site=tmp_path) is None

    assert len(list(tmp_path.glob("*.dist-info"))) == 2
    err = capsys.readouterr().err
    assert f"'{wheel}' wheel shadows the editable '{checkout}'" in err
    assert f"pip uninstall {wheel}" in err


def test_non_editable_records_lists_wheels_only(monkeypatch):
    def distribution(name):
        if name == DIST_NAME:
            return SimpleNamespace(read_text=lambda _n: None)  # a wheel: no direct_url.json
        if name == LEGACY_DIST_NAME:
            return SimpleNamespace(
                read_text=lambda _n: '{"url": "file:///src", "dir_info": {"editable": true}}'
            )
        raise dist_mod.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(dist_mod.importlib_metadata, "distribution", distribution)
    assert dist_mod.non_editable_records() == [DIST_NAME]


def test_a_single_record_is_left_alone(tmp_path, capsys):
    _install(tmp_path, DIST_NAME, NEW)

    assert reconcile(site=tmp_path) is None
    assert capsys.readouterr().err == ""


def test_neither_record_matching_changes_nothing_and_says_how_to_repair(tmp_path, capsys):
    _install(tmp_path, LEGACY_DIST_NAME, OLD)
    _install(tmp_path, DIST_NAME, NEW)
    (tmp_path / "sflow/cli.py").write_bytes(b"edited by hand")

    assert reconcile(site=tmp_path) is None
    assert len(list(tmp_path.glob("*.dist-info"))) == 2
    assert "pip install --force-reinstall nv-sflow" in capsys.readouterr().err


def test_opt_out(tmp_path, monkeypatch):
    _install(tmp_path, LEGACY_DIST_NAME, OLD)
    _install(tmp_path, DIST_NAME, NEW)
    monkeypatch.setenv(dist_mod.NO_AUTO_MIGRATE_ENV, "1")

    assert reconcile(site=tmp_path) is None
    assert len(list(tmp_path.glob("*.dist-info"))) == 2


def test_a_foreign_sflow_distribution_is_never_deleted(tmp_path):
    # 'sflow' on pypi.org is an unrelated project; only a record that installs our
    # console script is ours to remove.
    _install(tmp_path, LEGACY_DIST_NAME, OLD, entry_points="[console_scripts]\nsflow = other:main\n")
    _install(tmp_path, DIST_NAME, NEW)

    assert reconcile(site=tmp_path) is None
    assert len(list(tmp_path.glob("*.dist-info"))) == 2


def test_a_lost_race_leaves_the_environment_as_it_was(tmp_path, monkeypatch):
    _install(tmp_path, LEGACY_DIST_NAME, OLD)
    _install(tmp_path, DIST_NAME, NEW)

    def rename(src, dst):
        raise FileNotFoundError(src)  # another sflow process already moved it

    monkeypatch.setattr(dist_mod.os, "rename", rename)
    assert reconcile(site=tmp_path) is None
    assert len(list(tmp_path.glob("*.dist-info"))) == 2


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_site_packages_is_skipped(tmp_path):
    _install(tmp_path, LEGACY_DIST_NAME, OLD)
    _install(tmp_path, DIST_NAME, NEW)
    tmp_path.chmod(0o555)
    try:
        assert reconcile(site=tmp_path) is None
    finally:
        tmp_path.chmod(0o755)
    assert len(list(tmp_path.glob("*.dist-info"))) == 2


def test_records_elsewhere_on_sys_path_are_not_considered(tmp_path):
    # e.g. an editable checkout: the package is imported from one directory while
    # a record sits in another. Only the package's own directory is examined.
    site, other = tmp_path / "site", tmp_path / "other"
    site.mkdir()
    other.mkdir()
    _install(other, LEGACY_DIST_NAME, OLD)
    _install(site, DIST_NAME, NEW)

    assert reconcile(site=site) is None
    assert len(list(other.glob("*.dist-info"))) == 1


# ---------------------------------------------------------------------------
# installed_distribution
# ---------------------------------------------------------------------------


OURS_EP = [SimpleNamespace(group="console_scripts", name="sflow", value="sflow.cli:app")]
FOREIGN_EP = [SimpleNamespace(group="console_scripts", name="sflow", value="other:main")]


def _fake_metadata(monkeypatch, installed, packages=None, entry_points=OURS_EP):
    def distribution(name):
        if name in installed:
            return SimpleNamespace(
                metadata={"Name": name}, version=installed[name], entry_points=entry_points
            )
        raise dist_mod.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(dist_mod.importlib_metadata, "distribution", distribution)
    monkeypatch.setattr(
        dist_mod.importlib_metadata, "packages_distributions", lambda: packages or {}
    )


@pytest.mark.parametrize(
    "installed, expected",
    [
        ({DIST_NAME: "0.3.2"}, DIST_NAME),
        ({LEGACY_DIST_NAME: "0.3.2"}, LEGACY_DIST_NAME),  # a GitHub build
        ({DIST_NAME: "0.3.2", LEGACY_DIST_NAME: "0.3.1"}, DIST_NAME),  # before reconcile
    ],
)
def test_installed_distribution_recognises_either_name(monkeypatch, installed, expected):
    _fake_metadata(monkeypatch, installed)

    assert dist_mod.installed_distribution().metadata["Name"] == expected


def test_a_foreign_sflow_distribution_is_not_our_install(monkeypatch):
    # pypi.org's unrelated 'sflow' must not be mistaken for a legacy install.
    _fake_metadata(monkeypatch, {LEGACY_DIST_NAME: "1.0"}, entry_points=FOREIGN_EP)

    assert dist_mod.installed_distribution() is None


def test_a_renamed_fork_is_found_through_the_package_it_owns(monkeypatch):
    _fake_metadata(monkeypatch, {"acme-sflow": "9.9"}, {"sflow": ["acme-sflow"]})

    assert dist_mod.installed_distribution().version == "9.9"


def test_no_install_means_a_source_tree(monkeypatch):
    _fake_metadata(monkeypatch, {})

    assert dist_mod.installed_distribution() is None
