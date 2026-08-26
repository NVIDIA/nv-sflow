# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`--skip-artifact-check` demotes a missing local fs:// path from error to warning.

The demotion itself is `preflight_validate_artifacts`' existing off-host path
(covered in test_app_run_helpers.py). What is new -- and what these cover -- is the
wiring: the flag has to survive CLI -> SflowApp.run -> preflight, and for `sflow
batch` it has to reach the `sflow run` INSIDE the submitted job, which is where the
non-dry-run validation actually happens.
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from sflow.app.run_support import preflight_validate_artifacts
from sflow.cli import app

runner = CliRunner()

WORKFLOW = """
version: "0.1"
artifacts:
  - name: MODEL
    uri: fs:///definitely/not/here
workflow:
  name: test
  tasks:
    - name: hello
      script:
        - echo hello
"""


@pytest.fixture
def workflow_file(tmp_path):
    f = tmp_path / "wf.yaml"
    f.write_text(WORKFLOW)
    return f


@pytest.fixture
def mock_sflow_app():
    with patch("sflow.cli.batch._sflow_app") as mock_app:
        mock_app.run = MagicMock()
        yield mock_app


def _batch(workflow_file, sbatch_path, *extra):
    return runner.invoke(
        app,
        [
            "batch", "--file", str(workflow_file),
            "--partition", "batch", "--account", "acct", "--nodes", "1",
            "--sbatch-path", str(sbatch_path), *extra,
        ],
    )


def test_batch_forwards_flag_into_the_submitted_job(
    mock_sflow_app, workflow_file, tmp_path
):
    sbatch_path = tmp_path / "job.sh"
    result = _batch(workflow_file, sbatch_path, "--skip-artifact-check")
    assert result.exit_code == 0, result.output
    script = sbatch_path.read_text()
    assert "--skip-artifact-check" in script
    # It belongs on the inner `sflow run`, not as an #SBATCH directive.
    assert "#SBATCH --skip-artifact-check" not in script


def test_batch_omits_flag_by_default(mock_sflow_app, workflow_file, tmp_path):
    sbatch_path = tmp_path / "job.sh"
    result = _batch(workflow_file, sbatch_path)
    assert result.exit_code == 0, result.output
    assert "--skip-artifact-check" not in sbatch_path.read_text()


@pytest.mark.parametrize("flag, expected", [(["--skip-artifact-check"], True), ([], False)])
def test_run_flag_reaches_preflight(workflow_file, flag, expected):
    """CLI -> SflowApp.run -> preflight_validate_artifacts, unbroken."""
    with patch("sflow.app.sflow.preflight_validate_artifacts", return_value=[]) as pf:
        runner.invoke(app, ["run", str(workflow_file), "--dry-run", *flag])
    assert pf.call_args is not None, "preflight was never reached"
    assert pf.call_args.kwargs["skip_local_fs_validation"] is expected


def test_offhost_detection_still_skips_without_the_flag(tmp_path):
    """The flag ORs with off-host detection; it must not replace it."""
    k8s = tmp_path / "k8s.yaml"
    k8s.write_text(
        WORKFLOW.replace(
            "workflow:",
            "backends:\n"
            "  - name: k8s_cluster\n"
            "    type: kubernetes\n"
            "    default: true\n"
            "    namespace: default\n"
            "    nodes: 1\n"
            "    gpus_per_node: 0\n"
            "workflow:",
        )
    )
    with patch("sflow.app.sflow.preflight_validate_artifacts", return_value=[]) as pf:
        runner.invoke(app, ["run", str(k8s), "--dry-run"])
    assert pf.call_args is not None, "preflight was never reached"
    assert pf.call_args.kwargs["skip_local_fs_validation"] is True


def test_missing_fs_path_warns_instead_of_raising(tmp_path):
    """The behaviour the flag buys, at the function that owns it."""
    conf = MagicMock(name="MODEL", uri="fs:///definitely/not/here")
    conf.name = "MODEL"

    with pytest.raises(ValueError, match="does not exist"):
        preflight_validate_artifacts([conf], [], tmp_path, dry_run=False)

    warnings = preflight_validate_artifacts(
        [conf], [], tmp_path, dry_run=False, skip_local_fs_validation=True
    )
    assert len(warnings) == 1
    assert "--skip-artifact-check" in warnings[0]


def test_missing_fs_path_fails_the_real_run_without_the_flag(tmp_path, fake_process):
    """The other half of the pair below: without the flag the run must still FAIL.

    Every other CLI-level test here either mocks preflight or passes the flag, so
    without this one a bypass that became unconditional (default flipped, condition
    inverted) would leave the whole suite green.
    """
    fake_process.allow_unregistered(True)

    missing = tmp_path / "models" / "llama"
    wf = tmp_path / "wf.yaml"
    wf.write_text(WORKFLOW.replace("fs:///definitely/not/here", f"fs://{missing}"))

    result = runner.invoke(
        app, ["run", str(wf), "--output-dir", str(tmp_path / "out")]
    )
    assert result.exit_code != 0, f"a missing fs:// path must fail the run:\n{result.output}"
    assert "does not exist" in result.output
    assert not missing.exists(), "a failed validation must not have created the path"


def test_flag_does_not_create_a_phantom_directory(tmp_path, fake_process):
    """Same workflow as above, now WITH the flag: the run succeeds instead.

    And the missing fs:// path is passed through, not replaced with an empty dir --
    otherwise `--skip-artifact-check` would mkdir the model path it was told to stop
    checking, masking the real problem with an empty directory.
    """
    # This one runs the workflow for real; tests/unit/conftest.py fakes subprocesses.
    fake_process.allow_unregistered(True)

    missing = tmp_path / "models" / "llama"
    wf = tmp_path / "wf.yaml"
    wf.write_text(WORKFLOW.replace("fs:///definitely/not/here", f"fs://{missing}"))

    result = runner.invoke(
        app,
        ["run", str(wf), "--skip-artifact-check", "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert not missing.exists(), "flag must not materialize the missing artifact path"


def test_flag_does_not_disable_inline_file_content_materialization(tmp_path, fake_process):
    """The bypass is scoped to the fs:// existence check, nothing else.

    `remote_filesystem` ALSO means "an off-host backend injects file:// inline content
    into the pod", which is why an absolute file:// is deliberately not written on the
    controller. On slurm/local there is no pod, so routing --skip-artifact-check through
    that flag would leave the content written nowhere at all.
    """
    fake_process.allow_unregistered(True)

    missing = tmp_path / "models" / "llama"
    inline = tmp_path / "generated" / "helper.sh"
    wf = tmp_path / "wf.yaml"
    wf.write_text(f"""
version: "0.1"
artifacts:
  - name: MODEL
    uri: fs://{missing}
  - name: HELPER
    uri: file://{inline}
    content: |
      echo helper-ran
workflow:
  name: test
  tasks:
    - name: hello
      script:
        - echo hello
""")

    result = runner.invoke(
        app,
        ["run", str(wf), "--skip-artifact-check", "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert not missing.exists(), "the fs:// path must still be left alone"
    assert inline.exists(), (
        "an absolute file:// inline artifact must still be materialized on the "
        "controller; --skip-artifact-check must not reach off-host content injection"
    )
    assert "helper-ran" in inline.read_text()


def test_batch_bulk_submit_forwards_flag_to_every_job(mock_sflow_app, workflow_file, tmp_path):
    """Bulk mode threads the flag too -- it goes through its own launcher path."""
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "batch", "--bulk-submit", str(workflow_file),
            "--partition", "batch", "--account", "acct", "--nodes", "1",
            "--output-dir", str(out), "--skip-artifact-check",
        ],
    )
    assert result.exit_code == 0, result.output
    scripts = list(out.rglob("*.sh"))
    assert scripts, "bulk submit generated no job scripts"
    for script in scripts:
        assert "--skip-artifact-check" in script.read_text(), script
