# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CI_CONFIG = REPO_ROOT / ".gitlab-ci.yml"


def _job_script(job_name: str) -> list[str]:
    return yaml.safe_load(CI_CONFIG.read_text())[job_name]["script"]


def _publish_release_job() -> dict:
    return yaml.safe_load(CI_CONFIG.read_text())["publish_release"]


def _release_validation_and_export_script() -> list[str]:
    script = _job_script("publish_release")
    validation = next(item for item in script if "CI_COMMIT_TAG" in item)
    export = next(
        item for item in script if item.startswith("export SETUPTOOLS_SCM_PRETEND_VERSION")
    )
    return [validation, export]


def _run_release_version_script(tag: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        [
            "set -euo pipefail",
            *_release_validation_and_export_script(),
            'printf "\\nEXPORT=%s\\n" "$SETUPTOOLS_SCM_PRETEND_VERSION"',
        ]
    )
    env = {
        **os.environ,
        "CI_COMMIT_TAG": tag,
        "CI_PROJECT_DIR": str(tmp_path),
    }
    return subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_publish_release_exports_uppercase_v_tag_as_validated_version(tmp_path: Path, fp):
    fp.allow_unregistered(True)

    result = _run_release_version_script("V1.2.3", tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Publishing tag 'V1.2.3' as VERSION '1.2.3'" in result.stdout
    assert "EXPORT=1.2.3" in result.stdout


def test_publish_release_accepts_new_minor_tag_as_validated_version(tmp_path: Path, fp):
    fp.allow_unregistered(True)

    result = _run_release_version_script("v0.2.0", tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Publishing tag 'v0.2.0' as VERSION '0.2.0'" in result.stdout
    assert "EXPORT=0.2.0" in result.stdout


def test_publish_release_rejects_multiple_v_prefixes(tmp_path: Path, fp):
    fp.allow_unregistered(True)

    result = _run_release_version_script("vv1.2.3", tmp_path)

    assert result.returncode != 0
    assert "at most one leading" in result.stderr + result.stdout


def test_feature_branch_version_strips_uppercase_base_tag(fp):
    fp.allow_unregistered(True)

    publish_script = _job_script("publish_feature_branch")
    version_block = next(item for item in publish_script if "BASE_TAG=" in item)
    export_line = next(
        line.strip()
        for line in version_block.splitlines()
        if line.strip().startswith("export SETUPTOOLS_SCM_PRETEND_VERSION")
    )
    script = "\n".join(
        [
            "set -euo pipefail",
            "BASE_TAG=V1.2.3",
            "CI_PIPELINE_IID=42",
            "CI_COMMIT_REF_SLUG=feature",
            "CI_COMMIT_SHORT_SHA=abc1234",
            export_line,
            'printf "%s" "$SETUPTOOLS_SCM_PRETEND_VERSION"',
        ]
    )

    result = subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout == "1.2.3.dev42+feature.abc1234"


def test_publish_release_does_not_publish_stale_dist_artifacts():
    job = _publish_release_job()

    assert job.get("dependencies") == []

    script = job["script"]
    clean_dist_index = script.index("rm -rf dist")
    build_index = script.index("python -m build")
    publish_index = script.index('uv publish --check-url "$UV_PUBLISH_CHECK_URL" dist/*')

    assert clean_dist_index < build_index < publish_index


def test_publish_release_skips_existing_artifacts_on_rerun():
    script = _publish_release_job()["script"]

    assert 'export UV_PUBLISH_INDEX="ct-ppp-shto-pypi-local"' not in script
    assert (
        'export UV_PUBLISH_URL="https://urm.nvidia.com/artifactory/api/pypi/ct-ppp-shto-pypi-local/"'
        in script
    )
    assert (
        'export UV_PUBLISH_CHECK_URL="https://urm.nvidia.com/artifactory/api/pypi/ct-ppp-shto-pypi-local/simple/"'
        in script
    )
    assert 'uv publish --check-url "$UV_PUBLISH_CHECK_URL" dist/*' in script
