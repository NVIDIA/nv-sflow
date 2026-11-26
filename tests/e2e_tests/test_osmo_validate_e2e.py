# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The spec the ``osmo`` operator renders passes OSMO's own server-side validation.

``osmo workflow validate`` checks the spec against the live OSMO service and pool
without submitting anything, so it catches drift between what sflow renders and
what the installed OSMO accepts (field names, resource format, pool/platform).

Opt-in (``e2e`` marker) and skipped unless the ``osmo`` CLI is installed and logged
in. ``SFLOW_E2E_OSMO_POOL`` / ``SFLOW_E2E_OSMO_PLATFORM`` pick the target; unset
uses the OSMO profile's pool and the pool's default platform.

    pytest tests/e2e_tests/test_osmo_validate_e2e.py -m e2e
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from sflow.plugins.backends.osmo import OsmoBackend, OsmoBackendConfig
from sflow.plugins.operators.osmo import OsmoOperator, OsmoOperatorConfig, render_spec

pytestmark = pytest.mark.e2e

POOL = os.environ.get("SFLOW_E2E_OSMO_POOL") or None
PLATFORM = os.environ.get("SFLOW_E2E_OSMO_PLATFORM") or None


@pytest.fixture(scope="module", autouse=True)
def _osmo_logged_in() -> None:
    """Skip, not fail, where there is no usable OSMO CLI session."""
    if shutil.which("osmo") is None:
        pytest.skip("osmo CLI not installed")
    probe = subprocess.run(
        ["osmo", "workflow", "list", "--count", "1", "-t", "json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode != 0:
        pytest.skip("osmo CLI not logged in (run 'osmo login <url>')")


@pytest.mark.parametrize("gpus", [0, 1], ids=["cpu", "gpu"])
def test_rendered_spec_passes_osmo_validate(tmp_path, gpus):
    """Validate a spec with Jinja-like script text, env, output dirs and GPUs."""
    op = OsmoOperator(
        OsmoOperatorConfig(
            name="op", image="ubuntu:24.04", cpu=1, memory="1Gi", storage="1Gi"
        )
    )
    backend = OsmoBackend(OsmoBackendConfig(name="osmo", pool=POOL, platform=PLATFORM))
    op.apply_backend_context(
        backend=backend, assigned_nodes=[], artifacts=[], gpu_count=gpus
    )
    spec = tmp_path / "workflow.yaml"
    spec.write_text(
        render_spec(
            workflow_name="sflow-validate",
            image=op.config.image,
            script=[
                "echo 'literal {{ not_jinja }} {% raw %}'",
                "nvidia-smi -L || true",
            ],
            envs={"A": "it's", "SFLOW_TASK_OUTPUT_DIR": "/tmp/sflow/out/t"},
            resources=op.resources(),
        )
    )

    args = ["osmo", "workflow", "validate", str(spec)]
    if POOL:
        args += ["--pool", POOL]
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stdout + result.stderr
