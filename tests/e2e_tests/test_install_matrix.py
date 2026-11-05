# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install matrix for the nv-sflow rename: every starting point ends as ONE
``nv-sflow`` distribution with a working ``sflow`` whose plain ``upgrade`` targets
PyPI. Gates the release (CI job ``install_matrix``).

Real installs into throwaway venvs, with every remote replaced by a local stand-in:

* "PyPI" -- a find-links wheelhouse holding this tree built as ``nv-sflow``
  900.0.0 and 900.0.1 (far above any real release, so pypi.org can never outrank
  them; dependencies still resolve from the real index), plus 900.0.2.dev1 as
  the stand-in for a private index's ``.devN`` build.
* "GitHub main" -- a git repo of this tree named ``sflow``, as the OSS sync
  publishes it, reached through ``url.insteadOf`` so even the unmodified v0.3.1
  ``sflow upgrade`` lands on it.
* "≤0.3.1" -- this repository's ``v0.3.1`` tree, re-committed and re-tagged in a
  repo of its own (CI clones shallowly, and git cannot clone a tag out of a
  shallow repo), installed from git as it was.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
GITHUB_URL = "https://github.com/NVIDIA/nv-sflow.git"
OLD, NEW, DEV = "900.0.0", "900.0.1", "900.0.2.dev1"


def _run(argv, env, cwd=None):
    result = subprocess.run(
        argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, (
        f"{' '.join(map(str, argv))} exited {result.returncode}\n"
        f"--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}"
    )
    return result


def _copy_tree(dest: Path) -> None:
    """This working tree as git sees it (tracked + untracked, minus ignored)."""
    files = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    for rel in filter(None, files):
        src = REPO / rel
        if src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True,
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    if not shutil.which("uv") or not shutil.which("git"):
        pytest.skip("needs uv and git")
    tags = subprocess.run(
        ["git", "tag", "-l", "v0.3.1"], cwd=REPO, capture_output=True, text=True
    ).stdout
    if "v0.3.1" not in tags:
        pytest.skip("needs the v0.3.1 tag (git fetch --tags)")

    root = tmp_path_factory.mktemp("install_matrix")
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "VIRTUAL_ENV", "PIP_", "PYTHONPATH"))}

    # Build from a copy: setuptools-scm writes _version.py and the egg-info into
    # the tree it builds, which would leave the developer's checkout reporting
    # 900.0.x until the next editable install.
    source = root / "source"
    source.mkdir()
    _copy_tree(source)
    wheelhouse = root / "pypi"
    for version in (OLD, NEW, DEV):
        _run(
            ["uv", "build", "--wheel", "--out-dir", str(wheelhouse), str(source)],
            {**base_env, "SETUPTOOLS_SCM_PRETEND_VERSION": version},
        )

    github = root / "github"
    github.mkdir()
    _copy_tree(github)
    pyproject = github / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace('name = "nv-sflow"', 'name = "sflow"', 1)
    )
    # A dev build differs from the release it is versioned like; identical files
    # would make the two records indistinguishable (and either one correct).
    with open(github / "src" / "sflow" / "__init__.py", "a") as fh:
        fh.write("\n# development build\n")
    _git(github, "init", "-q", "-b", "main")
    _git(github, "add", "-A")
    _git(github, "commit", "-q", "-m", "oss sync")
    _git(github, "tag", f"v{OLD}")

    legacy = root / "legacy"
    legacy.mkdir()
    archive = subprocess.run(
        ["git", "archive", "v0.3.1"], cwd=REPO, capture_output=True, check=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(legacy)], input=archive, check=True)
    _git(legacy, "init", "-q", "-b", "main")
    _git(legacy, "add", "-A")
    _git(legacy, "commit", "-q", "-m", "v0.3.1")
    _git(legacy, "tag", "v0.3.1")

    gitconfig = root / "gitconfig"
    gitconfig.write_text(f'[url "{github.as_uri()}"]\n\tinsteadOf = {GITHUB_URL}\n')

    env = {
        **base_env,
        "UV_FIND_LINKS": str(wheelhouse),
        "GIT_CONFIG_GLOBAL": str(gitconfig),
        "GIT_CONFIG_NOSYSTEM": "1",
        # uv resolves github.com refs through the GitHub API before fetching; that
        # would pick the real repo's commit, which the local stand-in lacks.
        "UV_NO_GITHUB_FAST_PATH": "1",
        # Its own cache, so the stand-in's commits never enter the user's git cache
        # under the real GitHub URL.
        "UV_CACHE_DIR": str(root / "uv-cache"),
    }
    return {"root": root, "env": env, "github": github, "legacy": f"sflow @ git+{legacy.as_uri()}@v0.3.1"}


class Venv:
    def __init__(self, world, name, seed=False):
        self.dir = world["root"] / name
        self.env = dict(world["env"])
        # --seed adds pip, which uv venvs otherwise lack; the pip-only row needs it.
        _run(
            ["uv", "venv", "-q", *(["--seed"] if seed else []), "--python", sys.executable, str(self.dir)],
            self.env,
        )
        self.python = self.dir / "bin" / "python"
        self.env["VIRTUAL_ENV"] = str(self.dir)
        self.env["PATH"] = f"{self.dir / 'bin'}{os.pathsep}{self.env['PATH']}"

    def pip_install(self, *args):
        return _run(["uv", "pip", "install", "-q", "--python", str(self.python), *args], self.env)

    def sflow(self, *args):
        return _run([str(self.dir / "bin" / "sflow"), *args], self.env)

    def dists(self) -> dict[str, str]:
        out = _run(
            [str(self.python), "-c", (
                "import importlib.metadata as m, json\n"
                "found = {}\n"
                "for n in ('nv-sflow', 'sflow'):\n"
                "    try: found[n] = m.version(n)\n"
                "    except m.PackageNotFoundError: pass\n"
                "print(json.dumps(found))"
            )],
            self.env,
        )
        return json.loads(out.stdout)

    def assert_settled(self, version=None):
        """One nv-sflow, a working CLI, and a plain upgrade that targets PyPI."""
        self.sflow("--version")  # first CLI run: reconciles any leftover record
        dists = self.dists()
        assert list(dists) == ["nv-sflow"], dists
        if version:
            assert dists["nv-sflow"] == version
        plan = self.sflow("upgrade", "--dry-run").stdout
        command = next(line for line in plan.splitlines() if line.startswith("Command"))
        assert command.endswith(" 'nv-sflow>=0.3.2'") and "git+" not in command, plan


def test_fresh_install(world):
    venv = Venv(world, "fresh")
    venv.pip_install("nv-sflow")
    venv.assert_settled(NEW)


def test_legacy_git_install_then_pip_install_nv_sflow(world):
    venv = Venv(world, "legacy_then_pip")
    venv.pip_install(world["legacy"])
    assert venv.dists() == {"sflow": "0.3.1"}

    venv.pip_install("nv-sflow")
    assert set(venv.dists()) == {"sflow", "nv-sflow"}  # until sflow first runs
    venv.assert_settled(NEW)


def test_legacy_git_install_then_its_own_upgrade(world):
    venv = Venv(world, "legacy_old_upgrade")
    venv.pip_install(world["legacy"])

    # The v0.3.1 code, unmodified: installs 'sflow @ git+<GitHub>@main'. This only
    # works because GitHub still builds 'sflow'; it lands on migration-aware code.
    venv.sflow("upgrade")
    assert venv.dists() == {"sflow": OLD}

    venv.sflow("upgrade")  # now the new code: GitHub build -> PyPI
    venv.assert_settled(NEW)


def test_nv_sflow_upgrades_to_the_newer_release(world):
    venv = Venv(world, "upgrade_release")
    venv.pip_install(f"nv-sflow=={OLD}")
    venv.sflow("upgrade")
    venv.assert_settled(NEW)


def test_dev_build_from_github_is_kept_until_a_plain_upgrade(world):
    venv = Venv(world, "github_dev")
    venv.pip_install(f"nv-sflow=={OLD}")
    venv.pip_install(f"git+{GITHUB_URL}@main")

    venv.sflow("--version")  # reconcile keeps the build that owns the files
    assert venv.dists() == {"sflow": OLD}

    venv.sflow("upgrade")
    venv.assert_settled(NEW)


def test_index_dev_build_returns_to_the_release_on_a_plain_upgrade(world):
    # The internal setup: a .devN wheel of nv-sflow from a private index (install
    # mode 'installed', no direct_url). It is versioned above the latest release,
    # so only a forced reinstall of nv-sflow brings a plain upgrade back to PyPI.
    venv = Venv(world, "index_dev")
    venv.pip_install(f"nv-sflow=={DEV}")
    assert venv.dists() == {"nv-sflow": DEV}

    venv.sflow("upgrade")
    venv.assert_settled(NEW)


def test_branch_over_a_release_then_a_plain_upgrade(world):
    # The reverse crossing: `sflow upgrade --branch` lays a GitHub build (named
    # 'sflow') over the release, and upgrade's own reconcile drops the stale
    # nv-sflow record right away. A plain upgrade later returns to PyPI.
    venv = Venv(world, "branch_over_release")
    venv.pip_install(f"nv-sflow=={OLD}")

    venv.sflow("upgrade", "--branch", "main")
    assert venv.dists() == {"sflow": OLD}

    venv.sflow("upgrade")
    venv.assert_settled(NEW)


def test_pip_only_host_upgrades_through_pip(world):
    # No uv on PATH: upgrade falls back to pip, whose only per-package knob is
    # --force-reinstall. pip needs its own pointer at the stand-in wheelhouse.
    venv = Venv(world, "pip_only", seed=True)
    venv.pip_install(f"nv-sflow=={OLD}")
    venv.env["PATH"] = str(venv.dir / "bin")
    venv.env["PIP_FIND_LINKS"] = str(world["root"] / "pypi")

    venv.sflow("upgrade")
    venv.assert_settled(NEW)


def test_source_path_over_a_release_switches_to_the_checkout(world):
    # `sflow upgrade --sflow-source-path <GitHub checkout>` installs an editable
    # distribution named 'sflow'. The nv-sflow wheel's files would shadow it, so
    # upgrade removes the wheel first, and a later run must not treat the editable
    # record as stale.
    venv = Venv(world, "source_path")
    venv.pip_install(f"nv-sflow=={OLD}")

    venv.sflow("upgrade", "--sflow-source-path", str(world["github"]))
    assert venv.dists() == {"sflow": OLD}
    origin = _run([str(venv.python), "-c", "import sflow; print(sflow.__file__)"], venv.env).stdout
    assert origin.strip().startswith(str(world["github"])), origin

    venv.sflow("--version")  # reconcile leaves the editable alone
    assert venv.dists() == {"sflow": OLD}
