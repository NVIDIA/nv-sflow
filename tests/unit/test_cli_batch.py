# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sflow batch CLI command."""

import logging
import logging.handlers
import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from sflow.cli import app
from sflow.cli.batch import (
    _build_var_map,
    _classify_csv_columns,
    _dedup_words,
    _derive_nodes,
    _derive_row_name,
    _normalize_col_value,
    _resolve_backend_int_field,
    _resolve_sbatch_extra_args,
    _sanitize_name,
    _scan_sflow_yamls,
    build_row_naming_ctx,
    parse_row_selector,
    resolve_row_indices,
)


runner = CliRunner()


@pytest.fixture
def mock_sflow_app():
    """Mock SflowApp.run to skip actual dry-run validation."""
    with patch("sflow.cli.batch._sflow_app") as mock_app:
        mock_app.run = MagicMock()
        yield mock_app


@pytest.fixture
def temp_workflow_file(tmp_path):
    """Create a temporary workflow file for testing."""
    workflow_file = tmp_path / "test_workflow.yaml"
    workflow_file.write_text("""
version: "0.1"
workflow:
  name: test
  tasks:
    - name: hello
      script:
        - echo hello
""")
    return workflow_file


def test_batch_sbatch_extra_args_single(mock_sflow_app, temp_workflow_file, tmp_path):
    """Test that --sbatch-extra-args adds a single directive."""
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
            "--nodes",
            "1",
            "--sbatch-path",
            str(sbatch_path),
            "--sbatch-extra-args",
            "--exclusive",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    script_content = sbatch_path.read_text()
    assert "#SBATCH --exclusive" in script_content


def test_batch_sbatch_extra_args_multiple(mock_sflow_app, temp_workflow_file, tmp_path):
    """Test that multiple --sbatch-extra-args add multiple directives."""
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
            "--nodes",
            "2",
            "--sbatch-path",
            str(sbatch_path),
            "--sbatch-extra-args",
            "--exclusive",
            "--sbatch-extra-args",
            "--segment=2",
            "--sbatch-extra-args",
            "--constraint=gpu",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    script_content = sbatch_path.read_text()
    assert "#SBATCH --exclusive" in script_content
    assert "#SBATCH --segment=2" in script_content
    assert "#SBATCH --constraint=gpu" in script_content


def test_batch_sbatch_extra_args_short_option(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """Test that -e short option works for --sbatch-extra-args."""
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
            "--nodes",
            "1",
            "--sbatch-path",
            str(sbatch_path),
            "-e",
            "--exclusive",
            "-e",
            "--segment=1",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    script_content = sbatch_path.read_text()
    assert "#SBATCH --exclusive" in script_content
    assert "#SBATCH --segment=1" in script_content


def test_batch_sbatch_extra_args_preserves_value(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """Test that sbatch-extra-args preserves the exact value."""
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
            "--nodes",
            "4",
            "--sbatch-path",
            str(sbatch_path),
            "--sbatch-extra-args",
            "--gres=gpu:8",
            "--sbatch-extra-args",
            "--mem-per-cpu=4G",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    script_content = sbatch_path.read_text()
    assert "#SBATCH --gres=gpu:8" in script_content
    assert "#SBATCH --mem-per-cpu=4G" in script_content


def test_batch_without_sbatch_extra_args(mock_sflow_app, temp_workflow_file, tmp_path):
    """Test that batch works without --sbatch-extra-args."""
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
            "--nodes",
            "1",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    script_content = sbatch_path.read_text()
    # Standard directives should be present
    assert "#SBATCH --partition=batch" in script_content
    assert "#SBATCH --account=testaccount" in script_content
    assert "#SBATCH --nodes=1" in script_content
    assert "#SBATCH --gpus-per-node" not in script_content
    # Extra args should not be present (standard directives: job-name, output, error, mem, partition, account, nodes)
    assert (
        script_content.count("#SBATCH") == 7
    )  # job-name, output, error, mem, partition, account, nodes


def test_single_job_with_nodes_succeeds(mock_sflow_app, temp_workflow_file, tmp_path):
    """Single-job mode + --nodes => should succeed."""
    sbatch_path = tmp_path / "test.sh"
    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
            "--nodes",
            "2",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "#SBATCH --nodes=2" in sbatch_path.read_text()


def test_single_job_without_nodes_fails_when_not_derivable(mock_sflow_app, temp_workflow_file):
    """Single-job mode without --nodes and no backend => should fail."""
    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
        ],
    )
    assert result.exit_code == 1
    assert "could not be derived" in (result.output + (result.stderr or ""))


def test_bulk_input_with_nodes_succeeds(mock_sflow_app, tmp_path):
    """Bulk-input mode + --nodes => should succeed (--nodes passed to all scripts)."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    out_dir = tmp_path / "sflow_output"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,TP_SIZE\n{wf},4\n")

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "batch",
            "--account",
            "acct",
            "--nodes",
            "3",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    scripts = list(out_dir.rglob("*.sh"))
    assert len(scripts) == 1
    assert "#SBATCH --nodes=3" in scripts[0].read_text()


def test_bulk_input_without_nodes_fails_if_csv_has_no_node_column(
    mock_sflow_app, tmp_path
):
    """Bulk-input without --nodes and no node column in CSV => should fail."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,TP_SIZE\n{wf},4\n")

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "batch",
            "--account",
            "acct",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "SLURM_NODES" in (result.output + (result.stderr or ""))


def test_bulk_input_without_nodes_succeeds_with_node_column(mock_sflow_app, tmp_path):
    """Bulk-input without --nodes but CSV has SLURM_NODES column => should succeed."""
    wf_path = tmp_path / "wf.yaml"
    wf_path.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: TP_SIZE\n"
        "    value: 2\n"
        "  - name: SLURM_NODES\n"
        "    value: 1\n"
        "artifacts:\n"
        "  - name: MODEL_PATH\n"
        "    uri: fs:///default/model\n"
        "workflow:\n"
        "  name: test_wf\n"
        "  tasks:\n"
        "    - name: serve\n"
        "      script:\n"
        "        - echo ${{ variables.TP_SIZE }}\n"
    )
    out_dir = tmp_path / "sflow_output"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,TP_SIZE,SLURM_NODES\n{wf_path},4,2\n")

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "batch",
            "--account",
            "acct",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    scripts = list(out_dir.rglob("*.sh"))
    assert len(scripts) == 1
    script_text = scripts[0].read_text()
    assert "#SBATCH --nodes=2" in script_text, (
        "Nodes should be set from CSV SLURM_NODES column"
    )
    assert "--set SLURM_NODES=2" in script_text


def test_batch_sbatch_extra_args_order_preserved(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """Test that sbatch-extra-args are appended after standard directives."""
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
            "--nodes",
            "2",
            "--time",
            "01:00:00",
            "--sbatch-path",
            str(sbatch_path),
            "--sbatch-extra-args",
            "--exclusive",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    script_content = sbatch_path.read_text()

    # Check that --exclusive comes after --time (order is preserved)
    time_pos = script_content.find("#SBATCH --time=01:00:00")
    exclusive_pos = script_content.find("#SBATCH --exclusive")
    assert time_pos < exclusive_pos, "Extra args should come after standard directives"


# ---------------------------------------------------------------------------
# Bulk-edit tests
# ---------------------------------------------------------------------------


def _write_workflow_with_vars(path: Path) -> Path:
    """Create a workflow YAML with variables and artifacts for bulk-edit testing."""
    path.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: TP_SIZE\n"
        "    value: 2\n"
        "  - name: MODEL_NAME\n"
        "    value: default-model\n"
        "artifacts:\n"
        "  - name: MODEL_PATH\n"
        "    uri: fs:///default/model\n"
        "workflow:\n"
        "  name: test_wf\n"
        "  tasks:\n"
        "    - name: serve\n"
        "      script:\n"
        "        - echo ${{ variables.TP_SIZE }}\n"
    )
    return path


def _write_csv(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_bulk_edit_generates_scripts_per_row(mock_sflow_app, tmp_path):
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,TP_SIZE,MODEL_PATH\n"
        f"{wf},4,fs://{model_dir}\n"
        f"{wf},8,fs://{model_dir}\n",
    )

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "batch",
            "--account",
            "acct",
            "--nodes",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "2 jobs generated" in result.output

    bulk_dirs = list(out_dir.glob("bulk_*"))
    assert len(bulk_dirs) == 1
    scripts = sorted(bulk_dirs[0].glob("*.sh"))
    assert len(scripts) == 2

    s1 = scripts[0].read_text()
    s2 = scripts[1].read_text()
    all_text = s1 + s2
    assert "--set TP_SIZE=4" in all_text
    assert "--set TP_SIZE=8" in all_text
    assert f"--artifact MODEL_PATH=fs://{model_dir}" in all_text


def test_bulk_edit_rejects_missing_sflow_config_file_column(mock_sflow_app, tmp_path):
    csv_file = _write_csv(
        tmp_path / "bad.csv",
        "TP_SIZE,MODEL_NAME\n4,llama\n",
    )
    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "batch",
            "--account",
            "acct",
            "--nodes",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "sflow_config_file" in result.output


def test_bulk_edit_rejects_unknown_column(mock_sflow_app, tmp_path):
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    csv_file = _write_csv(
        tmp_path / "bad.csv",
        f"sflow_config_file,NONEXISTENT_VAR\n{wf},42\n",
    )
    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "batch",
            "--account",
            "acct",
            "--nodes",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "NONEXISTENT_VAR" in result.output


# --- _classify_csv_columns chained error info tests ---


def test_classify_csv_columns_all_configs_fail_enriches_unknown_column_error(tmp_path):
    """When all config sets fail to load, the unknown-column ValueError includes
    chained error context pointing to config loading as the root cause."""
    base = tmp_path / "base.yaml"
    base.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      depends_on: [missing_task]\n"
        "      script:\n"
        "        - echo hi\n"
    )
    row_configs = [([base], None)]
    with pytest.raises(ValueError, match="all 1 config set.*failed to load"):
        _classify_csv_columns(["SOME_VAR"], row_configs)


def test_classify_csv_columns_partial_failure_no_chained_hint(tmp_path):
    """When some configs load successfully, the unknown-column error does NOT
    include the 'all configs failed' hint — the variable is genuinely missing."""
    good = tmp_path / "good.yaml"
    good.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: TP\n"
        "    value: 1\n"
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo hi\n"
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      depends_on: [nonexistent]\n"
        "      script:\n"
        "        - echo hi\n"
    )
    row_configs = [([good], None), ([bad], None)]
    with pytest.raises(ValueError, match="not a variable or artifact") as exc_info:
        _classify_csv_columns(["MISSING_VAR"], row_configs)
    assert "all" not in str(exc_info.value).lower() or "failed to load" not in str(exc_info.value)


def test_classify_csv_columns_all_configs_fail_logs_warnings(tmp_path):
    """When all config sets fail, warnings are logged listing each failure
    and a hint about --missable-tasks."""
    f1 = tmp_path / "a.yaml"
    f1.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      depends_on: [ghost]\n"
        "      script:\n"
        "        - echo hi\n"
    )
    row_configs = [([f1], None)]

    log_handler = logging.handlers.MemoryHandler(capacity=100)
    logger = logging.getLogger("sflow.cli.batch")
    logger.addHandler(log_handler)
    old_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        with pytest.raises(ValueError):
            _classify_csv_columns(["X"], row_configs)
        log_handler.flush()
        messages = [r.getMessage() for r in log_handler.buffer]
        combined = "\n".join(messages)
        assert "1 config file set(s) failed to load" in combined
        assert "No config sets loaded successfully" in combined
        assert "missable" in combined.lower()
    finally:
        logger.removeHandler(log_handler)
        logger.setLevel(old_level)


def test_classify_csv_columns_succeeds_when_column_valid_despite_partial_failure(tmp_path):
    """A valid column is still recognized even when some config sets fail."""
    good = tmp_path / "good.yaml"
    good.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: TP_SIZE\n"
        "    value: 1\n"
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo hi\n"
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      depends_on: [nonexistent]\n"
        "      script:\n"
        "        - echo hi\n"
    )
    row_configs = [([good], None), ([bad], None)]
    var_cols, art_cols = _classify_csv_columns(["TP_SIZE"], row_configs)
    assert var_cols == {"TP_SIZE"}
    assert art_cols == set()


def test_classify_csv_columns_missable_tasks_prevents_load_failure(tmp_path):
    """Passing missable_tasks for the row avoids the config load failure."""
    f = tmp_path / "wf.yaml"
    f.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: MY_VAR\n"
        "    value: x\n"
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      depends_on: [missing_task]\n"
        "      script:\n"
        "        - echo hi\n"
    )
    row_configs = [([f], ["missing_task"])]
    var_cols, art_cols = _classify_csv_columns(["MY_VAR"], row_configs)
    assert var_cols == {"MY_VAR"}


def test_bulk_edit_with_multiple_config_files(mock_sflow_app, tmp_path):
    f1 = tmp_path / "backends.yaml"
    f1.write_text('version: "0.1"\nvariables:\n  - name: NODES\n    value: 1\n')
    f2 = tmp_path / "workflow.yaml"
    f2.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo ${{ variables.NODES }}\n"
    )
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,NODES\n{f1} {f2},2\n{f1} {f2},4\n",
    )

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "gpu",
            "--account",
            "acct",
            "--nodes",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "2 jobs generated" in result.output

    bulk_dirs = list(out_dir.glob("bulk_*"))
    scripts = sorted(bulk_dirs[0].glob("*.sh"))
    s1 = scripts[0].read_text()
    assert f"--file {shlex.quote(str(f1.resolve()))}" in s1
    assert f"--file {shlex.quote(str(f2.resolve()))}" in s1
    assert "--set NODES=2" in s1


# ---------------------------------------------------------------------------
# Results CSV and dry-run failure tests
# ---------------------------------------------------------------------------


def test_bulk_input_writes_results_csv_with_submit(mock_sflow_app, tmp_path):
    """With --submit, bulk-input writes a results.csv with slurm_job_id and sflow_output_dir."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,TP_SIZE\n{wf},4\n{wf},8\n",
    )

    with patch(
        "sflow.cli.batch._submit_sbatch", return_value="Submitted batch job 99999"
    ):
        result = runner.invoke(
            app,
            [
                "batch",
                "--bulk-input",
                str(csv_file),
                "--partition",
                "batch",
                "--account",
                "acct",
                "--nodes",
                "1",
                "--output-dir",
                str(out_dir),
                "--submit",
            ],
        )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "results.csv" in result.output

    bulk_dirs = list(out_dir.glob("bulk_*"))
    results_csv = bulk_dirs[0] / "results.csv"
    assert results_csv.exists()

    import csv as csv_mod

    with open(results_csv) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert "slurm_job_id" in reader.fieldnames
    assert "sflow_output_dir" in reader.fieldnames
    assert "sflow_batch_dir" in reader.fieldnames
    assert rows[0]["slurm_job_id"] == "99999"
    assert rows[0]["sflow_batch_dir"] == bulk_dirs[0].name


def test_bulk_input_results_csv_without_submit_has_not_submitted(mock_sflow_app, tmp_path):
    """Without --submit, results.csv is still generated with 'not submitted' values."""
    import csv as _csv

    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,TP_SIZE\n{wf},4\n",
    )

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "batch",
            "--account",
            "acct",
            "--nodes",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Results CSV:" in result.output

    bulk_dirs = list(out_dir.glob("bulk_*"))
    results_csv = bulk_dirs[0] / "results.csv"
    assert results_csv.exists()

    with open(results_csv) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["slurm_job_id"] == "not submitted"
    assert rows[0]["sflow_output_dir"] == "not submitted"
    assert rows[0]["sflow_batch_dir"] == bulk_dirs[0].name


def test_bulk_input_results_csv_marks_failed_rows(tmp_path):
    """Rows that fail dry-run get slurm_job_id=FAILED in results.csv."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,TP_SIZE\n{wf},4\n{wf},8\n",
    )

    with (
        patch("sflow.cli.batch._sflow_app") as mock_app,
        patch(
            "sflow.cli.batch._submit_sbatch", return_value="Submitted batch job 11111"
        ),
    ):
        call_count = 0

        def _fail_second_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("GPU over-subscription")

        mock_app.run = MagicMock(side_effect=_fail_second_call)

        result = runner.invoke(
            app,
            [
                "batch",
                "--bulk-input",
                str(csv_file),
                "--partition",
                "batch",
                "--account",
                "acct",
                "--nodes",
                "1",
                "--output-dir",
                str(out_dir),
                "--submit",
            ],
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "1 failed dry-run" in result.output
    assert "ERRORS" in result.output

    bulk_dirs = list(out_dir.glob("bulk_*"))
    results_csv = bulk_dirs[0] / "results.csv"

    import csv as csv_mod

    with open(results_csv) as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["slurm_job_id"] == "11111"
    assert rows[1]["slurm_job_id"] == "FAILED"
    assert rows[1]["sflow_output_dir"] == ""
    assert rows[0]["sflow_batch_dir"] == bulk_dirs[0].name
    assert rows[1]["sflow_batch_dir"] == bulk_dirs[0].name


def test_bulk_input_dry_run_failures_shown_at_end(tmp_path):
    """Dry-run failures are listed in a prominent block at the end."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,TP_SIZE\n{wf},4\n",
    )

    with patch("sflow.cli.batch._sflow_app") as mock_app:
        mock_app.run = MagicMock(side_effect=ValueError("bad config"))

        result = runner.invoke(
            app,
            [
                "batch",
                "--bulk-input",
                str(csv_file),
                "--partition",
                "batch",
                "--account",
                "acct",
                "--nodes",
                "1",
                "--output-dir",
                str(out_dir),
            ],
        )

    assert result.exit_code == 0
    assert "ERRORS: 1 row(s) failed dry-run validation:" in result.output
    assert "bad config" in result.output
    assert "====" in result.output


# ---------------------------------------------------------------------------
# Row naming tests (_derive_row_name / _sanitize_name)
# ---------------------------------------------------------------------------


def _name(rows, idx=1, row_idx=0, fallback_base="sflow", cli_nodes=None):
    """Shorthand: build context + derive name in one call (for tests only)."""
    ctx = build_row_naming_ctx(rows, fallback_base=fallback_base, cli_nodes=cli_nodes)
    return _derive_row_name(rows[row_idx], idx, ctx)


class TestSanitizeName:
    def test_basic(self):
        assert _sanitize_name("hello_world") == "hello_world"

    def test_special_chars(self):
        assert _sanitize_name("foo/bar:baz") == "foo_bar_baz"

    def test_collapses_underscores(self):
        assert _sanitize_name("a___b") == "a_b"

    def test_strips_leading_trailing(self):
        assert _sanitize_name("__hello__") == "hello"

    def test_truncates(self):
        result = _sanitize_name("a" * 100, max_len=10)
        assert len(result) == 10

    def test_no_trailing_underscore_after_truncation(self):
        result = _sanitize_name("abc_def_ghi", max_len=4)
        assert result == "abc"
        assert not result.endswith("_")

    def test_empty_returns_row(self):
        assert _sanitize_name("") == "row"

    def test_preserves_dashes(self):
        assert _sanitize_name("my-name") == "my-name"


class TestDeriveRowName:
    def test_explicit_job_name(self):
        rows = [{"sflow_config_file": "a.yaml", "job_name": "my_job"}]
        assert _name(rows) == "my_job_001"

    def test_explicit_job_name_sanitized(self):
        rows = [{"sflow_config_file": "a.yaml", "job_name": "my job/v2"}]
        assert _name(rows) == "my_job_v2_001"

    def test_auto_derive_unique_stems(self):
        rows = [
            {"sflow_config_file": "common.yaml vllm_prefill.yaml vllm_decode.yaml"},
            {"sflow_config_file": "common.yaml sglang_prefill.yaml sglang_decode.yaml"},
        ]
        assert _name(rows, idx=1, row_idx=0) == "vllm_prefill_decode_001"
        assert _name(rows, idx=2, row_idx=1) == "sglang_prefill_decode_002"

    def test_auto_derive_removes_common_stems(self):
        rows = [
            {"sflow_config_file": "shared.yaml engine_a.yaml"},
            {"sflow_config_file": "shared.yaml engine_b.yaml"},
        ]
        name = _name(rows)
        assert name == "engine_a_001"
        assert "shared" not in name

    def test_fallback_when_all_rows_same_files(self):
        rows = [
            {"sflow_config_file": "workflow.yaml"},
            {"sflow_config_file": "workflow.yaml"},
        ]
        assert _name(rows) == "sflow_001"
        assert _name(rows, idx=2, row_idx=1, fallback_base="myjob") == "myjob_002"

    def test_empty_job_name_triggers_auto_derive(self):
        rows = [
            {"sflow_config_file": "a.yaml b.yaml", "job_name": ""},
            {"sflow_config_file": "a.yaml c.yaml", "job_name": ""},
        ]
        assert _name(rows, idx=1, row_idx=0) == "b_001"
        assert _name(rows, idx=2, row_idx=1) == "c_002"

    def test_no_job_name_column(self):
        rows = [
            {"sflow_config_file": "x.yaml y.yaml"},
            {"sflow_config_file": "x.yaml z.yaml"},
        ]
        assert _name(rows) == "y_001"

    def test_single_row_all_stems_unique(self):
        rows = [{"sflow_config_file": "alpha.yaml beta.yaml"}]
        assert _name(rows) == "sflow_001"

    def test_differing_column_values_appended(self):
        rows = [
            {"sflow_config_file": "wf.yaml", "TP": "4", "BATCH": "128"},
            {"sflow_config_file": "wf.yaml", "TP": "8", "BATCH": "256"},
        ]
        name = _name(rows)
        assert "4" in name
        assert "128" in name
        assert name == "4_128_001"

    def test_common_column_values_skipped(self):
        rows = [
            {"sflow_config_file": "wf.yaml", "TP": "4", "GPU": "8"},
            {"sflow_config_file": "wf.yaml", "TP": "8", "GPU": "8"},
        ]
        name = _name(rows)
        assert "4" in name
        assert "8" not in name.replace("_001", "")

    def test_path_values_skipped(self):
        rows = [
            {"sflow_config_file": "wf.yaml", "MODEL": "fs:///path/a", "TP": "2"},
            {"sflow_config_file": "wf.yaml", "MODEL": "fs:///path/b", "TP": "4"},
        ]
        name = _name(rows)
        assert "fs" not in name
        assert "path" not in name
        assert "2" in name

    def test_node_count_from_csv_always_included(self):
        rows = [
            {"sflow_config_file": "wf.yaml", "SLURM_NODES": "4"},
            {"sflow_config_file": "wf.yaml", "SLURM_NODES": "4"},
        ]
        assert "4n" in _name(rows)

    def test_node_count_from_cli_always_included(self):
        rows = [
            {"sflow_config_file": "wf.yaml"},
            {"sflow_config_file": "wf.yaml"},
        ]
        assert "8n" in _name(rows, cli_nodes=8)

    def test_cli_nodes_overrides_csv_in_name(self):
        rows = [{"sflow_config_file": "wf.yaml", "SLURM_NODES": "2"}]
        name = _name(rows, cli_nodes=4)
        assert "4n" in name
        assert "2n" not in name

    def test_node_column_not_duplicated_in_other_cols(self):
        rows = [
            {"sflow_config_file": "wf.yaml", "SLURM_NODES": "2", "TP": "4"},
            {"sflow_config_file": "wf.yaml", "SLURM_NODES": "4", "TP": "8"},
        ]
        assert _name(rows) == "2n_4_001"

    def test_stems_and_columns_combined(self):
        rows = [
            {"sflow_config_file": "common.yaml vllm.yaml", "TP": "2"},
            {"sflow_config_file": "common.yaml sglang.yaml", "TP": "4"},
        ]
        assert _name(rows, idx=1, row_idx=0) == "vllm_2_001"
        assert _name(rows, idx=2, row_idx=1) == "sglang_4_002"

    def test_name_truncated_to_30_chars(self):
        rows = [
            {
                "sflow_config_file": "a_very_long_config_name.yaml b_another_long_one.yaml"
            },
            {
                "sflow_config_file": "a_very_long_config_name.yaml c_different_long_one.yaml"
            },
        ]
        name = _name(rows)
        base = name.rsplit("_", 1)[0]
        assert len(base) <= 30

    def test_dedup_removes_repeated_words(self):
        rows = [
            {"sflow_config_file": "common.yaml trtllm_prefill.yaml trtllm_decode.yaml"},
            {"sflow_config_file": "common.yaml vllm_prefill.yaml vllm_decode.yaml"},
        ]
        assert _name(rows) == "trtllm_prefill_decode_001"

    def test_relative_paths_handled(self):
        rows = [
            {
                "sflow_config_file": "../../configs/common.yaml ../../configs/engine_a.yaml"
            },
            {
                "sflow_config_file": "../../configs/common.yaml ../../configs/engine_b.yaml"
            },
        ]
        assert _name(rows, idx=1, row_idx=0) == "configs_engine_a_001"
        assert _name(rows, idx=2, row_idx=1) == "configs_engine_b_002"

    def test_mixed_relative_and_bare_paths(self):
        rows = [
            {"sflow_config_file": "shared.yaml ./dir/vllm_task.yaml"},
            {"sflow_config_file": "shared.yaml ./dir/sglang_task.yaml"},
        ]
        assert _name(rows, idx=1, row_idx=0) == "dir_vllm_task_001"
        assert _name(rows, idx=2, row_idx=1) == "dir_sglang_task_002"


class TestDedupWords:
    def test_basic(self):
        assert _dedup_words("trtllm_prefill_trtllm_decode") == "trtllm_prefill_decode"

    def test_no_duplicates(self):
        assert _dedup_words("vllm_prefill_decode") == "vllm_prefill_decode"

    def test_all_same(self):
        assert _dedup_words("a_a_a") == "a"

    def test_preserves_order(self):
        assert _dedup_words("c_b_a_b_c") == "c_b_a"

    def test_empty_string(self):
        assert _dedup_words("") == ""

    def test_single_word(self):
        assert _dedup_words("hello") == "hello"


class TestNormalizeColValue:
    def test_plain_value(self):
        assert _normalize_col_value("42") == "42"

    def test_model_name(self):
        assert _normalize_col_value("Qwen3-8B-FP8") == "Qwen3-8B-FP8"

    def test_uri_skipped(self):
        assert _normalize_col_value("fs:///path/to/model") is None

    def test_s3_uri_skipped(self):
        assert _normalize_col_value("s3://bucket/key") is None

    def test_absolute_path_skipped(self):
        assert _normalize_col_value("/home/user/model") is None

    def test_container_image_skipped(self):
        assert (
            _normalize_col_value("nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0") is None
        )

    def test_container_image_with_org_skipped(self):
        assert _normalize_col_value("lmsysorg/sglang:v0.5.8.post1-cu130") is None

    def test_slash_path_skipped(self):
        assert _normalize_col_value("org/repo") is None


def test_derive_row_name_container_image_skipped():
    """Container image columns should be excluded from derived names entirely."""
    rows = [
        {
            "sflow_config_file": "wf.yaml",
            "IMAGE": "nvcr.io/nvidia/vllm-runtime:0.8.0",
            "SLURM_NODES": "2",
        },
        {
            "sflow_config_file": "wf.yaml",
            "IMAGE": "nvcr.io/nvidia/vllm-runtime:0.9.0",
            "SLURM_NODES": "2",
        },
    ]
    name = _name(rows)
    assert "nvcr" not in name
    assert "nvidia" not in name
    assert "vllm" not in name
    assert "0_8" not in name
    assert name == "2n_001"


def test_bulk_edit_uses_derived_names(mock_sflow_app, tmp_path):
    """Bulk-edit should use auto-derived names from unique config file stems."""
    f_common = tmp_path / "common.yaml"
    f_common.write_text('version: "0.1"\nvariables:\n  - name: NODES\n    value: 1\n')
    f_engine_a = tmp_path / "engine_a.yaml"
    f_engine_a.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo ${{ variables.NODES }}\n"
    )
    f_engine_b = tmp_path / "engine_b.yaml"
    f_engine_b.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo ${{ variables.NODES }}\n"
    )
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,NODES\n"
        f"{f_common} {f_engine_a},2\n"
        f"{f_common} {f_engine_b},4\n",
    )

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "gpu",
            "--account",
            "acct",
            "--nodes",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    bulk_dirs = list(out_dir.glob("bulk_*"))
    scripts = sorted(bulk_dirs[0].glob("*.sh"))
    names = [s.stem for s in scripts]
    # CLI --nodes=1 always included as "1n"; NODES column differs (2 vs 4) and is also appended
    assert "engine_a_1n_2_001" in names
    assert "engine_b_1n_4_002" in names


def test_bulk_edit_explicit_job_name_column(mock_sflow_app, tmp_path):
    """Bulk-edit should use explicit job_name column when present."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,job_name,TP_SIZE\n{wf},small_run,4\n{wf},large_run,8\n",
    )

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input",
            str(csv_file),
            "--partition",
            "batch",
            "--account",
            "acct",
            "--nodes",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    bulk_dirs = list(out_dir.glob("bulk_*"))
    scripts = {s.stem: s for s in bulk_dirs[0].glob("*.sh")}
    assert "small_run_001" in scripts
    assert "large_run_002" in scripts
    assert "#SBATCH --job-name=small_run_001" in scripts["small_run_001"].read_text()


# ---------------------------------------------------------------------------
# Row selector parsing tests (parse_row_selector)
# ---------------------------------------------------------------------------


class TestParseRowSelector:
    def test_single_int(self):
        assert parse_row_selector(["1"]) == [1]

    def test_multiple_singles(self):
        assert parse_row_selector(["1", "3", "5"]) == [1, 3, 5]

    def test_comma_separated(self):
        assert parse_row_selector(["1,3,5"]) == [1, 3, 5]

    def test_slice_two_part(self):
        assert parse_row_selector(["1:4"]) == [1, 2, 3]

    def test_slice_three_part(self):
        assert parse_row_selector(["1:6:2"]) == [1, 3, 5]

    def test_brackets_stripped(self):
        assert parse_row_selector(["[1:4]"]) == [1, 2, 3]

    def test_brackets_comma(self):
        assert parse_row_selector(["[1,3,5]"]) == [1, 3, 5]

    def test_combined(self):
        assert parse_row_selector(["1:3", "7"]) == [1, 2, 7]

    def test_deduplicates(self):
        assert parse_row_selector(["1", "1", "2"]) == [1, 2]

    def test_empty_list(self):
        assert parse_row_selector([]) == []

    def test_mixed_comma_and_slice(self):
        assert parse_row_selector(["1,4:6"]) == [1, 4, 5]

    # -- Negative indices (deferred, no n_rows) --

    def test_negative_single(self):
        assert parse_row_selector(["-1"]) == [-1]

    def test_negative_multiple(self):
        assert parse_row_selector(["-1", "-3"]) == [-3, -1]

    def test_negative_comma(self):
        assert parse_row_selector(["-1,-3"]) == [-3, -1]

    def test_negative_slice_both_bounds(self):
        assert parse_row_selector(["-3:-1"]) == [-3, -2]

    def test_mixed_positive_negative(self):
        result = parse_row_selector(["1", "-1"])
        assert result == [1, -1]

    # -- Negative indices (resolved with n_rows) --

    def test_negative_single_resolved(self):
        assert parse_row_selector(["-1"], n_rows=10) == [10]

    def test_negative_last_three_resolved(self):
        assert parse_row_selector(["-3", "-2", "-1"], n_rows=10) == [8, 9, 10]

    def test_negative_slice_resolved(self):
        assert parse_row_selector(["-3:-1"], n_rows=10) == [8, 9]

    def test_mixed_positive_negative_resolved(self):
        assert parse_row_selector(["1", "-1"], n_rows=5) == [1, 5]

    # -- Open-ended slices (require n_rows) --

    def test_open_end_slice(self):
        assert parse_row_selector(["3:"], n_rows=5) == [3, 4, 5]

    def test_open_start_slice(self):
        assert parse_row_selector([":3"], n_rows=5) == [1, 2]

    def test_negative_open_end_slice(self):
        assert parse_row_selector(["-3:"], n_rows=10) == [8, 9, 10]

    def test_open_end_slice_without_n_rows_raises(self):
        with pytest.raises(Exception, match="Open-ended slice"):
            parse_row_selector(["3:"])

    def test_open_start_slice_without_n_rows_raises(self):
        with pytest.raises(Exception, match="Open-ended slice"):
            parse_row_selector([":3"])

    def test_open_end_with_step(self):
        assert parse_row_selector(["1::2"], n_rows=6) == [1, 3, 5]

    # -- Edge cases --

    def test_negative_out_of_range_warns(self):
        result = parse_row_selector(["-10"], n_rows=5)
        assert result == []

    def test_brackets_negative(self):
        assert parse_row_selector(["[-1]"]) == [-1]

    def test_brackets_negative_resolved(self):
        assert parse_row_selector(["[-1]"], n_rows=5) == [5]


# ---------------------------------------------------------------------------
# resolve_row_indices tests
# ---------------------------------------------------------------------------


class TestResolveRowIndices:
    def test_positive_passthrough(self):
        assert resolve_row_indices([1, 3, 5], 10) == [1, 3, 5]

    def test_negative_last(self):
        assert resolve_row_indices([-1], 10) == [10]

    def test_negative_sequence(self):
        assert resolve_row_indices([-3, -2, -1], 10) == [8, 9, 10]

    def test_mixed(self):
        assert resolve_row_indices([1, -1], 5) == [1, 5]

    def test_out_of_range_dropped(self):
        assert resolve_row_indices([0, 11, -11], 10) == []

    def test_deduplicates(self):
        assert resolve_row_indices([1, 1, -1, -1], 5) == [1, 5]

    def test_empty(self):
        assert resolve_row_indices([], 10) == []


# ---------------------------------------------------------------------------
# CLI integration: negative indices & open-ended slices via sflow batch --row
# ---------------------------------------------------------------------------


def _make_batch_csv(tmp_path, n_rows=5):
    """Create a minimal CSV with *n_rows* data rows for batch --row tests."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    header = "sflow_config_file,TP_SIZE\n"
    rows = "".join(f"{wf},{2 * (i + 1)}\n" for i in range(n_rows))
    return _write_csv(tmp_path / "jobs.csv", header + rows)


class TestBatchRowNegativeIndex:
    """Test sflow batch --bulk-input with negative indices and open-ended slices."""

    def test_batch_row_negative_last(self, mock_sflow_app, tmp_path):
        csv_file = _make_batch_csv(tmp_path, n_rows=5)
        out_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                "batch", "--bulk-input", str(csv_file),
                "--row=-1",
                "--partition", "p", "--account", "a", "--nodes", "1",
                "--output-dir", str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        scripts = list(out_dir.rglob("*.sh"))
        assert len(scripts) == 1

    def test_batch_row_negative_last_three(self, mock_sflow_app, tmp_path):
        csv_file = _make_batch_csv(tmp_path, n_rows=5)
        out_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                "batch", "--bulk-input", str(csv_file),
                "--row=-3:",
                "--partition", "p", "--account", "a", "--nodes", "1",
                "--output-dir", str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        scripts = list(out_dir.rglob("*.sh"))
        assert len(scripts) == 3

    def test_batch_row_open_end_from_3(self, mock_sflow_app, tmp_path):
        csv_file = _make_batch_csv(tmp_path, n_rows=5)
        out_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                "batch", "--bulk-input", str(csv_file),
                "--row=3:",
                "--partition", "p", "--account", "a", "--nodes", "1",
                "--output-dir", str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        scripts = list(out_dir.rglob("*.sh"))
        assert len(scripts) == 3

    def test_batch_row_open_start_to_3(self, mock_sflow_app, tmp_path):
        csv_file = _make_batch_csv(tmp_path, n_rows=5)
        out_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                "batch", "--bulk-input", str(csv_file),
                "--row=:3",
                "--partition", "p", "--account", "a", "--nodes", "1",
                "--output-dir", str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        scripts = list(out_dir.rglob("*.sh"))
        assert len(scripts) == 2  # rows 1, 2 (exclusive end)

    def test_batch_row_negative_slice(self, mock_sflow_app, tmp_path):
        csv_file = _make_batch_csv(tmp_path, n_rows=5)
        out_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                "batch", "--bulk-input", str(csv_file),
                "--row=-3:-1",
                "--partition", "p", "--account", "a", "--nodes", "1",
                "--output-dir", str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        scripts = list(out_dir.rglob("*.sh"))
        assert len(scripts) == 2  # rows 3, 4

    def test_batch_row_mixed_positive_and_negative(self, mock_sflow_app, tmp_path):
        csv_file = _make_batch_csv(tmp_path, n_rows=5)
        out_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                "batch", "--bulk-input", str(csv_file),
                "--row", "1", "--row=-1",
                "--partition", "p", "--account", "a", "--nodes", "1",
                "--output-dir", str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        scripts = list(out_dir.rglob("*.sh"))
        assert len(scripts) == 2  # rows 1 and 5


# ---------------------------------------------------------------------------
# _scan_sflow_yamls tests
# ---------------------------------------------------------------------------

_VALID_SFLOW_YAML = 'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t1\n      script:\n        - echo hi\n'


class TestScanSflowYamls:
    def test_explicit_files(self, tmp_path):
        f1 = tmp_path / "a.yaml"
        f1.write_text(_VALID_SFLOW_YAML)
        f2 = tmp_path / "b.yml"
        f2.write_text(_VALID_SFLOW_YAML)
        result = _scan_sflow_yamls([f1, f2])
        assert len(result) == 2
        assert f1.resolve() in result
        assert f2.resolve() in result

    def test_directory_scan(self, tmp_path):
        (tmp_path / "a.yaml").write_text(_VALID_SFLOW_YAML)
        (tmp_path / "b.yaml").write_text(_VALID_SFLOW_YAML)
        (tmp_path / "not_yaml.txt").write_text("hello")
        result = _scan_sflow_yamls([tmp_path])
        assert len(result) == 2

    def test_skips_invalid_yaml(self, tmp_path):
        valid = tmp_path / "valid.yaml"
        valid.write_text(_VALID_SFLOW_YAML)
        no_version = tmp_path / "no_version.yaml"
        no_version.write_text("key: value\n")
        broken = tmp_path / "broken.yaml"
        broken.write_text("{{invalid yaml")
        result = _scan_sflow_yamls([tmp_path])
        assert len(result) == 1
        assert valid.resolve() in result

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "slurm_a.yaml").write_text(_VALID_SFLOW_YAML)
        (tmp_path / "slurm_b.yaml").write_text(_VALID_SFLOW_YAML)
        (tmp_path / "other.yaml").write_text(_VALID_SFLOW_YAML)
        pattern = tmp_path / "slurm_*"
        result = _scan_sflow_yamls([pattern])
        assert len(result) == 2

    def test_deduplicates(self, tmp_path):
        f = tmp_path / "dup.yaml"
        f.write_text(_VALID_SFLOW_YAML)
        result = _scan_sflow_yamls([f, f, f])
        assert len(result) == 1

    def test_mixed_files_dirs_globs(self, tmp_path):
        subdir = tmp_path / "configs"
        subdir.mkdir()
        f1 = tmp_path / "standalone.yaml"
        f1.write_text(_VALID_SFLOW_YAML)
        (subdir / "cfg_a.yaml").write_text(_VALID_SFLOW_YAML)
        (subdir / "cfg_b.yaml").write_text(_VALID_SFLOW_YAML)
        (tmp_path / "glob_match.yaml").write_text(_VALID_SFLOW_YAML)
        result = _scan_sflow_yamls([f1, subdir, tmp_path / "glob_*"])
        assert len(result) == 4

    def test_nonexistent_path_returns_empty(self, tmp_path):
        result = _scan_sflow_yamls([tmp_path / "does_not_exist.yaml"])
        assert result == []

    def test_skips_non_yaml_files(self, tmp_path):
        (tmp_path / "script.sh").write_text("#!/bin/bash\necho hi")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "valid.yaml").write_text(_VALID_SFLOW_YAML)
        result = _scan_sflow_yamls([tmp_path])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# --bulk-submit CLI integration tests
# ---------------------------------------------------------------------------


def test_bulk_submit_with_directory(mock_sflow_app, tmp_path):
    """--bulk-submit with a directory scans and processes all valid YAML files."""
    (tmp_path / "wf1.yaml").write_text(_VALID_SFLOW_YAML)
    (tmp_path / "wf2.yaml").write_text(_VALID_SFLOW_YAML)
    (tmp_path / "not_sflow.yaml").write_text("key: value\n")
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(tmp_path),
            "--partition", "gpu",
            "--account", "test",
            "--output-dir", str(out_dir),
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 0
    assert "Found 2 sflow YAML config(s)" in result.output
    assert "2/2 configs processed" in result.output


def test_bulk_submit_with_explicit_files(mock_sflow_app, tmp_path):
    """--bulk-submit with explicit file paths."""
    f1 = tmp_path / "a.yaml"
    f1.write_text(_VALID_SFLOW_YAML)
    f2 = tmp_path / "b.yaml"
    f2.write_text(_VALID_SFLOW_YAML)
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "-B", str(f1),
            "-B", str(f2),
            "--partition", "gpu",
            "--account", "test",
            "--output-dir", str(out_dir),
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 0
    assert "Found 2 sflow YAML config(s)" in result.output
    assert "2/2 configs processed" in result.output


def test_bulk_submit_with_shell_expanded_glob(mock_sflow_app, tmp_path):
    """Simulates shell glob expansion: first file via -B, rest as positional args."""
    f1 = tmp_path / "slurm_a.yaml"
    f1.write_text(_VALID_SFLOW_YAML)
    f2 = tmp_path / "slurm_b.yaml"
    f2.write_text(_VALID_SFLOW_YAML)
    f3 = tmp_path / "slurm_c.yaml"
    f3.write_text(_VALID_SFLOW_YAML)
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(f1),
            str(f2), str(f3),
            "--partition", "gpu",
            "--account", "test",
            "--output-dir", str(out_dir),
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 0
    assert "Found 3 sflow YAML config(s)" in result.output
    assert "3/3 configs processed" in result.output


def test_bulk_submit_writes_results_csv(mock_sflow_app, tmp_path):
    """--bulk-submit writes a results.csv with config, job_name, and status."""
    import csv

    (tmp_path / "wf.yaml").write_text(_VALID_SFLOW_YAML)
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(tmp_path),
            "--partition", "gpu",
            "--account", "test",
            "--output-dir", str(out_dir),
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 0
    assert "Results CSV:" in result.output

    csv_line = [l for l in result.output.splitlines() if "Results CSV:" in l][0]
    csv_path = Path(csv_line.split("Results CSV: ")[1].strip())
    assert csv_path.exists()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert "sflow_config_file" in rows[0]
    assert "job_name" in rows[0]
    assert "status" in rows[0]
    assert "sflow_batch_dir" in rows[0]
    assert rows[0]["sflow_batch_dir"].startswith("bulk_submit_")


def test_bulk_submit_no_valid_files(mock_sflow_app, tmp_path):
    """--bulk-submit with no valid YAML files exits with error."""
    (tmp_path / "empty.yaml").write_text("key: value\n")

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(tmp_path),
            "--partition", "gpu",
            "--account", "test",
        ],
    )
    assert result.exit_code == 1
    assert "no valid sflow YAML" in result.output


def test_bulk_submit_dry_run_failure_skips_config(tmp_path):
    """Configs that fail dry-run are skipped but reported."""
    (tmp_path / "bad.yaml").write_text(
        'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t1\n      script:\n        - echo ${{ variables.MISSING }}\n'
    )
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(tmp_path),
            "--partition", "gpu",
            "--account", "test",
            "--output-dir", str(out_dir),
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 0
    assert "0/1 configs processed" in result.output
    assert "1 failed validation" in result.output


def test_bulk_submit_not_submitted_hint(mock_sflow_app, tmp_path):
    """Without --submit, a hint is shown to add --submit."""
    (tmp_path / "wf.yaml").write_text(_VALID_SFLOW_YAML)
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(tmp_path),
            "--partition", "gpu",
            "--account", "test",
            "--output-dir", str(out_dir),
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 0
    assert "not submitted" in result.output.lower() or "--submit" in result.output


def test_bulk_submit_results_csv_not_submitted_values(mock_sflow_app, tmp_path):
    """Without --submit, results.csv shows 'not submitted' for job_id and output_dir."""
    import csv as _csv

    (tmp_path / "wf.yaml").write_text(_VALID_SFLOW_YAML)
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(tmp_path),
            "--partition", "gpu",
            "--account", "test",
            "--output-dir", str(out_dir),
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 0

    csv_line = [l for l in result.output.splitlines() if "Results CSV:" in l][0]
    csv_path = Path(csv_line.split("Results CSV: ")[1].strip())
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["slurm_job_id"] == "not submitted"
    assert rows[0]["sflow_output_dir"] == "not submitted"
    assert rows[0]["sflow_batch_dir"].startswith("bulk_submit_")


def test_bulk_input_generates_merged_yaml(mock_sflow_app, tmp_path):
    """--bulk-input generates merged YAML config files alongside sbatch scripts."""
    f_common = tmp_path / "common.yaml"
    f_common.write_text('version: "0.1"\nvariables:\n  - name: NODES\n    value: 1\n')
    f_task = tmp_path / "task.yaml"
    f_task.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo hello\n"
    )
    out_dir = tmp_path / "sflow_output"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(
        f"sflow_config_file,NODES\n{f_common} {f_task},2\n"
    )

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input", str(csv_file),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "--output-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    bulk_dirs = list(out_dir.glob("bulk_*"))
    assert len(bulk_dirs) == 1
    yaml_files = list(bulk_dirs[0].glob("*.yaml"))
    assert len(yaml_files) == 1
    content = yaml_files[0].read_text()
    assert "version:" in content
    assert "workflow:" in content


def test_single_job_stdout_hint(mock_sflow_app, temp_workflow_file):
    """Without -o, a hint is shown that output is stdout only."""
    result = runner.invoke(
        app,
        [
            "batch",
            str(temp_workflow_file),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 0
    assert "-o" in result.output


def test_sbatch_script_copies_logs_to_output_dir(mock_sflow_app, temp_workflow_file, tmp_path):
    """Generated sbatch script includes commands to copy logs to workflow output dir."""
    out = tmp_path / "out.sh"
    result = runner.invoke(
        app,
        [
            "batch",
            str(temp_workflow_file),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "-o", str(out),
        ],
    )
    assert result.exit_code == 0
    script = out.read_text()
    assert "SFLOW_WF_DIR" in script
    assert "cp " in script
    assert "SLURM_JOB_ID" in script


# ---------------------------------------------------------------------------
# _derive_nodes / _resolve_backend_int_field tests
# ---------------------------------------------------------------------------


class TestDeriveNodes:
    def test_plain_integer(self, tmp_path):
        """Backend with nodes: 2 (plain integer)."""
        f = tmp_path / "wf.yaml"
        f.write_text(
            'version: "0.1"\n'
            "backends:\n"
            "  - name: slurm_cluster\n    type: slurm\n    nodes: 2\n"
            "    partition: gpu\n    account: test\n"
            "workflow:\n  name: wf\n  tasks:\n"
            "    - name: t1\n      script:\n        - echo hi\n"
        )
        assert _derive_nodes([f]) == 2

    def test_expression_with_dict_of_dict_variable(self, tmp_path):
        """Backend nodes: ${{ variables.SLURM_NODES }} with dict-of-dict variable format."""
        f = tmp_path / "wf.yaml"
        f.write_text(
            'version: "0.1"\n'
            "variables:\n"
            "  SLURM_NODES:\n"
            "    description: Number of nodes\n"
            "    value: 4\n"
            "backends:\n"
            "  - name: slurm_cluster\n    type: slurm\n"
            "    nodes: ${{ variables.SLURM_NODES }}\n"
            "    partition: gpu\n    account: test\n"
            "workflow:\n  name: wf\n  tasks:\n"
            "    - name: t1\n      script:\n        - echo hi\n"
        )
        assert _derive_nodes([f]) == 4

    def test_expression_with_list_of_dict_variable(self, tmp_path):
        """Backend nodes: ${{ variables.SLURM_NODES }} with list-of-dict variable format."""
        f = tmp_path / "wf.yaml"
        f.write_text(
            'version: "0.1"\n'
            "variables:\n"
            "  - name: SLURM_NODES\n"
            "    value: 8\n"
            "backends:\n"
            "  - name: slurm_cluster\n    type: slurm\n"
            "    nodes: ${{ variables.SLURM_NODES }}\n"
            "    partition: gpu\n    account: test\n"
            "workflow:\n  name: wf\n  tasks:\n"
            "    - name: t1\n      script:\n        - echo hi\n"
        )
        assert _derive_nodes([f]) == 8

    def test_string_integer(self, tmp_path):
        """Backend with nodes: '3' (string that parses to int)."""
        f = tmp_path / "wf.yaml"
        f.write_text(
            'version: "0.1"\n'
            "backends:\n"
            "  - name: slurm_cluster\n    type: slurm\n    nodes: '3'\n"
            "    partition: gpu\n    account: test\n"
            "workflow:\n  name: wf\n  tasks:\n"
            "    - name: t1\n      script:\n        - echo hi\n"
        )
        assert _derive_nodes([f]) == 3

    def test_no_backend_returns_none(self, tmp_path):
        """No backend defined => returns None."""
        f = tmp_path / "wf.yaml"
        f.write_text(
            'version: "0.1"\n'
            "workflow:\n  name: wf\n  tasks:\n"
            "    - name: t1\n      script:\n        - echo hi\n"
        )
        assert _derive_nodes([f]) is None

    def test_non_slurm_backend_skipped(self, tmp_path):
        """Non-slurm backend is skipped."""
        f = tmp_path / "wf.yaml"
        f.write_text(
            'version: "0.1"\n'
            "backends:\n"
            "  - name: local_dev\n    type: local\n    nodes: 2\n"
            "workflow:\n  name: wf\n  tasks:\n"
            "    - name: t1\n      script:\n        - echo hi\n"
        )
        assert _derive_nodes([f]) is None

    def test_multi_file_merges_variables(self, tmp_path):
        """Variable defined in one file, backend referencing it in another."""
        f1 = tmp_path / "vars.yaml"
        f1.write_text(
            'version: "0.1"\n'
            "variables:\n"
            "  SLURM_NODES:\n"
            "    value: 6\n"
        )
        f2 = tmp_path / "backend.yaml"
        f2.write_text(
            'version: "0.1"\n'
            "backends:\n"
            "  - name: slurm_cluster\n    type: slurm\n"
            "    nodes: ${{ variables.SLURM_NODES }}\n"
            "    partition: gpu\n    account: test\n"
            "workflow:\n  name: wf\n  tasks:\n"
            "    - name: t1\n      script:\n        - echo hi\n"
        )
        assert _derive_nodes([f1, f2]) == 6

    def test_cli_override_wins(self, tmp_path):
        """CLI --set overrides the config value."""
        f = tmp_path / "wf.yaml"
        f.write_text(
            'version: "0.1"\n'
            "variables:\n"
            "  SLURM_NODES:\n"
            "    value: 2\n"
            "backends:\n"
            "  - name: slurm_cluster\n    type: slurm\n"
            "    nodes: ${{ variables.SLURM_NODES }}\n"
            "    partition: gpu\n    account: test\n"
            "workflow:\n  name: wf\n  tasks:\n"
            "    - name: t1\n      script:\n        - echo hi\n"
        )
        assert _derive_nodes([f], cli_overrides=["SLURM_NODES=10"]) == 10


def test_single_job_derives_nodes_from_config(mock_sflow_app, tmp_path):
    """Single-job mode without --nodes but config has backend.nodes => succeeds."""
    wf = tmp_path / "wf.yaml"
    wf.write_text(
        'version: "0.1"\n'
        "backends:\n"
        "  - name: slurm_cluster\n    type: slurm\n    nodes: 2\n"
        "    partition: gpu\n    account: test\n"
        "workflow:\n  name: wf\n  tasks:\n"
        "    - name: t1\n      script:\n        - echo hi\n"
    )
    sbatch_path = tmp_path / "out.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "-f", str(wf),
            "--partition", "batch",
            "--account", "testaccount",
            "-o", str(sbatch_path),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "derived from config: 2" in (result.output + (result.stderr or ""))
    assert "#SBATCH --nodes=2" in sbatch_path.read_text()


# ---------------------------------------------------------------------------
# --resolve tests (compose YAML alongside sbatch script)
# ---------------------------------------------------------------------------


def test_single_job_resolve_generates_yaml(mock_sflow_app, temp_workflow_file, tmp_path):
    """Single-job with --resolve generates a composed YAML next to the sbatch script."""
    sbatch_path = tmp_path / "run.sh"
    result = runner.invoke(
        app,
        [
            "batch",
            "-f", str(temp_workflow_file),
            "--partition", "batch",
            "--account", "test",
            "--nodes", "1",
            "-o", str(sbatch_path),
            "--resolve",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    yaml_path = tmp_path / "run.yaml"
    assert yaml_path.exists(), "Composed YAML should be generated next to sbatch script"
    content = yaml_path.read_text()
    assert "version:" in content
    assert "workflow:" in content


def test_single_job_without_resolve_still_generates_yaml(mock_sflow_app, temp_workflow_file, tmp_path):
    """Single-job without --resolve still generates a composed YAML (unresolved)."""
    sbatch_path = tmp_path / "out.sh"
    result = runner.invoke(
        app,
        [
            "batch",
            "-f", str(temp_workflow_file),
            "--partition", "batch",
            "--account", "test",
            "--nodes", "1",
            "-o", str(sbatch_path),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert (tmp_path / "out.yaml").exists()


def test_single_job_modular_resolve_merges_files(mock_sflow_app, tmp_path):
    """Single-job with multiple files + --resolve merges into one composed YAML."""
    f1 = tmp_path / "vars.yaml"
    f1.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  GREETING:\n"
        "    value: hello\n"
    )
    f2 = tmp_path / "wf.yaml"
    f2.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo ${{ variables.GREETING }}\n"
    )
    sbatch_path = tmp_path / "merged.sh"
    result = runner.invoke(
        app,
        [
            "batch",
            "-f", str(f1), "-f", str(f2),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "-o", str(sbatch_path),
            "--resolve",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    yaml_path = tmp_path / "merged.yaml"
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "hello" in content
    assert "${{" not in content


def test_single_job_no_sbatch_path_skips_yaml(mock_sflow_app, temp_workflow_file):
    """Without -o (stdout mode), no YAML file is generated."""
    result = runner.invoke(
        app,
        [
            "batch",
            "-f", str(temp_workflow_file),
            "--partition", "batch",
            "--account", "test",
            "--nodes", "1",
            "--resolve",
        ],
    )
    assert result.exit_code == 0


def test_bulk_submit_resolve_generates_yamls(mock_sflow_app, tmp_path):
    """--bulk-submit with --resolve generates YAML alongside each sbatch script."""
    (tmp_path / "wf1.yaml").write_text(_VALID_SFLOW_YAML)
    (tmp_path / "wf2.yaml").write_text(_VALID_SFLOW_YAML)
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(tmp_path),
            "--partition", "gpu",
            "--account", "test",
            "--output-dir", str(out_dir),
            "--nodes", "1",
            "--resolve",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    bulk_dirs = list(out_dir.glob("bulk_submit_*"))
    assert len(bulk_dirs) == 1
    yaml_files = list(bulk_dirs[0].glob("*.yaml"))
    sh_files = list(bulk_dirs[0].glob("*.sh"))
    assert len(yaml_files) == 2, f"Expected 2 YAML files, got {len(yaml_files)}"
    assert len(sh_files) == 2
    for yf in yaml_files:
        content = yf.read_text()
        assert "version:" in content
        assert "workflow:" in content


# ---------------------------------------------------------------------------
# --missable-tasks tests
# ---------------------------------------------------------------------------

_MISSABLE_TASK_YAML = 'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t1\n      script:\n        - echo hi\n'
_MISSABLE_TASK_WITH_DEP = 'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n    - name: t2\n      depends_on:\n        - t1\n        - missing_task\n      script:\n        - echo hi\n'


def test_missable_tasks_rejected_with_single_file(mock_sflow_app, tmp_path):
    """--missable-tasks should error with a single input file."""
    f = tmp_path / "wf.yaml"
    f.write_text(_MISSABLE_TASK_YAML)

    result = runner.invoke(
        app,
        [
            "batch",
            str(f),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "--missable-tasks", "missing_task",
        ],
    )
    assert result.exit_code == 1
    assert "multiple input files" in result.output


def test_missable_tasks_allowed_with_multiple_files(mock_sflow_app, tmp_path):
    """--missable-tasks should work with multiple input files."""
    f1 = tmp_path / "base.yaml"
    f1.write_text(_MISSABLE_TASK_YAML)
    f2 = tmp_path / "extra.yaml"
    f2.write_text(_MISSABLE_TASK_WITH_DEP)
    out = tmp_path / "out.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            str(f1), str(f2),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "-M", "missing_task",
            "-o", str(out),
        ],
    )
    assert result.exit_code == 0


def test_missable_tasks_allowed_with_bulk_submit(mock_sflow_app, tmp_path):
    """--missable-tasks should work with --bulk-submit."""
    (tmp_path / "wf.yaml").write_text(_MISSABLE_TASK_YAML)
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(tmp_path),
            "--partition", "gpu",
            "--account", "test",
            "--output-dir", str(out_dir),
            "--nodes", "1",
            "-M", "some_task",
        ],
    )
    assert result.exit_code == 0


def test_missable_tasks_short_flag(mock_sflow_app, tmp_path):
    """-M short flag should work for --missable-tasks."""
    f1 = tmp_path / "base.yaml"
    f1.write_text(_MISSABLE_TASK_YAML)
    f2 = tmp_path / "extra.yaml"
    f2.write_text(_MISSABLE_TASK_WITH_DEP)
    out = tmp_path / "out.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            str(f1), str(f2),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "-M", "missing_task",
            "-o", str(out),
        ],
    )
    assert result.exit_code == 0


def test_missable_tasks_bulk_input_csv_column(mock_sflow_app, tmp_path):
    """missable_tasks CSV column should strip absent tasks per row."""
    f_common = tmp_path / "common.yaml"
    f_common.write_text(
        'version: "0.1"\nvariables:\n  - name: NODES\n    value: 1\n'
        "workflow:\n  name: wf\n  tasks:\n"
        "    - name: shared_task\n      script:\n        - echo shared\n"
    )
    f_task = tmp_path / "task.yaml"
    f_task.write_text(
        'version: "0.1"\nworkflow:\n  name: wf\n  tasks:\n'
        "    - name: bench\n      depends_on:\n        - shared_task\n"
        "        - agg_server\n        - prefill_server\n"
        "      script:\n        - echo bench\n"
    )
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(
        f"sflow_config_file,NODES,missable_tasks\n"
        f"{f_common} {f_task},1,agg_server prefill_server\n"
    )
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input", str(csv_file),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "--output-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "1/1" in result.output


# ---------------------------------------------------------------------------
# sflow run --missable-tasks tests
# ---------------------------------------------------------------------------


def test_run_missable_tasks_rejected_with_single_file(tmp_path):
    """sflow run --missable-tasks should error with a single input file."""
    f = tmp_path / "wf.yaml"
    f.write_text(_MISSABLE_TASK_YAML)

    result = runner.invoke(
        app,
        ["run", "-f", str(f), "--dry-run", "-M", "missing_task"],
    )
    assert result.exit_code == 1
    assert "multiple input files" in result.output


def test_run_missable_tasks_allowed_with_multiple_files(tmp_path):
    """sflow run --missable-tasks should work with multiple files."""
    with patch("sflow.cli.run._sflow_app") as mock_app:
        mock_app.run = MagicMock(return_value=None)

        f1 = tmp_path / "base.yaml"
        f1.write_text(_MISSABLE_TASK_YAML)
        f2 = tmp_path / "extra.yaml"
        f2.write_text(_MISSABLE_TASK_WITH_DEP)

        result = runner.invoke(
            app,
            ["run", "-f", str(f1), "-f", str(f2), "--dry-run", "-M", "missing_task"],
        )
        assert result.exit_code == 0
        mock_app.run.assert_called_once()
        call_kwargs = mock_app.run.call_args.kwargs
        assert call_kwargs["missable_tasks"] == ["missing_task"]


# ---------------------------------------------------------------------------
# E2E-style: mixed disagg/agg CSV with per-row missable_tasks
# ---------------------------------------------------------------------------


def _write_modular_configs(tmp_path):
    """Create a minimal modular config set for testing mixed disagg/agg CSV."""
    base = tmp_path / "base.yaml"
    base.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: SLURM_NODES\n    type: integer\n    value: 1\n"
        "  - name: GPUS_PER_NODE\n    type: integer\n    value: 4\n"
        "backends:\n"
        "  - name: slurm_cluster\n    type: slurm\n    default: true\n"
        "    nodes: 1\n    gpus_per_node: 4\n    time: 60\n"
        "    partition: test\n    account: test\n"
        "operators:\n"
        "  - name: dynamo\n    type: srun\n"
        "workflow:\n  name: wf\n  tasks:\n"
        "    - name: frontend\n      script:\n        - echo frontend\n"
    )
    disagg = tmp_path / "disagg.yaml"
    disagg.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: NUM_CTX\n    type: integer\n    value: 1\n"
        "workflow:\n  name: wf\n  tasks:\n"
        "    - name: prefill_server\n      depends_on: [frontend]\n"
        "      script:\n        - echo prefill\n"
        "    - name: decode_server\n      depends_on: [frontend]\n"
        "      script:\n        - echo decode\n"
    )
    agg = tmp_path / "agg.yaml"
    agg.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: NUM_AGG\n    type: integer\n    value: 1\n"
        "workflow:\n  name: wf\n  tasks:\n"
        "    - name: agg_server\n      depends_on: [frontend]\n"
        "      script:\n        - echo agg\n"
    )
    bench = tmp_path / "bench.yaml"
    bench.write_text(
        'version: "0.1"\n'
        "workflow:\n  name: wf\n  tasks:\n"
        "    - name: benchmark\n"
        "      depends_on: [prefill_server, decode_server, agg_server, frontend]\n"
        "      script:\n        - echo bench\n"
    )
    return base, disagg, agg, bench


def test_batch_bulk_input_mixed_disagg_agg_csv(mock_sflow_app, tmp_path):
    """Mixed CSV with disagg rows (missable agg_server) and agg rows (missable prefill/decode)."""
    base, disagg, agg, bench = _write_modular_configs(tmp_path)
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(
        f"sflow_config_file,NUM_CTX,NUM_AGG,missable_tasks\n"
        f"{base} {disagg} {bench},2,,agg_server\n"
        f"{base} {agg} {bench},,1,prefill_server decode_server\n"
    )
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input", str(csv_file),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "--output-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "2/2" in result.output


def test_compose_bulk_input_mixed_disagg_agg_csv(tmp_path):
    """Compose with mixed CSV: disagg and agg rows with per-row missable_tasks."""
    base, disagg, agg, bench = _write_modular_configs(tmp_path)
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(
        f"sflow_config_file,NUM_CTX,NUM_AGG,missable_tasks\n"
        f"{base} {disagg} {bench},2,,agg_server\n"
        f"{base} {agg} {bench},,1,prefill_server decode_server\n"
    )
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "compose",
            "--bulk-input", str(csv_file),
            "-o", str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    yaml_files = list(out_dir.glob("*/*.yaml"))
    assert len(yaml_files) == 2


def test_run_multiple_files_with_missable(tmp_path):
    """sflow run with multiple files and --missable-tasks strips absent tasks."""
    from unittest.mock import patch, MagicMock

    base, disagg, _, bench = _write_modular_configs(tmp_path)

    with patch("sflow.cli.run._sflow_app") as mock_app:
        mock_app.run = MagicMock(return_value=None)

        result = runner.invoke(
            app,
            [
                "run",
                "-f", str(base),
                "-f", str(disagg),
                "-f", str(bench),
                "--dry-run",
                "-M", "agg_server",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_app.run.call_args.kwargs
        assert call_kwargs["missable_tasks"] == ["agg_server"]


# ---------------------------------------------------------------------------
# --missable-tasks in generated sbatch script tests
# ---------------------------------------------------------------------------


def test_sbatch_script_includes_missable_tasks(mock_sflow_app, tmp_path):
    """Generated sbatch script should include --missable-tasks flags in sflow run command."""
    f1 = tmp_path / "base.yaml"
    f1.write_text(_MISSABLE_TASK_YAML)
    f2 = tmp_path / "extra.yaml"
    f2.write_text(_MISSABLE_TASK_WITH_DEP)
    out = tmp_path / "out.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            str(f1), str(f2),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "-M", "missing_task",
            "-M", "another_*",
            "-o", str(out),
        ],
    )
    assert result.exit_code == 0
    script = out.read_text()
    assert "--missable-tasks" in script
    assert "missing_task" in script
    assert "another_*" in script


def test_bulk_input_sbatch_script_includes_per_row_missable(mock_sflow_app, tmp_path):
    """Bulk-input generated scripts should include per-row missable_tasks from CSV."""
    base, disagg, agg, bench = _write_modular_configs(tmp_path)
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(
        f"sflow_config_file,NUM_CTX,NUM_AGG,missable_tasks\n"
        f"{base} {disagg} {bench},2,,agg_server\n"
        f"{base} {agg} {bench},,1,prefill_server decode_server\n"
    )
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input", str(csv_file),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "--output-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    bulk_dirs = list(out_dir.glob("bulk_*"))
    assert len(bulk_dirs) == 1
    scripts = list(bulk_dirs[0].glob("*.sh"))
    assert len(scripts) == 2

    script_contents = {s.name: s.read_text() for s in scripts}
    disagg_script = [v for k, v in script_contents.items() if "001" in k][0]
    agg_script = [v for k, v in script_contents.items() if "002" in k][0]

    assert "--missable-tasks" in disagg_script
    assert "agg_server" in disagg_script

    assert "--missable-tasks" in agg_script
    assert "prefill_server" in agg_script
    assert "decode_server" in agg_script


# ---------------------------------------------------------------------------
# CLI vs CSV precedence tests
# ---------------------------------------------------------------------------


def test_batch_bulk_input_variable_csv_wins_over_cli(mock_sflow_app, tmp_path):
    """For --set variables, CSV value should take precedence over CLI."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,TP_SIZE\n{wf},8\n",
    )
    result = runner.invoke(
        app,
        [
            "batch", "--bulk-input", str(csv_file),
            "-p", "batch", "-A", "acct", "--nodes", "1",
            "--output-dir", str(out_dir),
            "--set", "TP_SIZE=2",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "CSV value will take precedence" in (result.output + (result.stderr or ""))
    scripts = sorted(list(out_dir.glob("bulk_*"))[0].glob("*.sh"))
    script = scripts[0].read_text()
    assert "--set TP_SIZE=8" in script
    assert "--set TP_SIZE=2" not in script


def test_batch_bulk_input_artifact_cli_wins_over_csv(mock_sflow_app, tmp_path):
    """For --artifact, CLI value should take precedence over CSV."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    csv_model_dir = tmp_path / "csv_models"
    csv_model_dir.mkdir()
    out_dir = tmp_path / "sflow_output"
    csv_file = _write_csv(
        tmp_path / "jobs.csv",
        f"sflow_config_file,MODEL_PATH\n{wf},fs://{csv_model_dir}\n",
    )
    result = runner.invoke(
        app,
        [
            "batch", "--bulk-input", str(csv_file),
            "-p", "batch", "-A", "acct", "--nodes", "1",
            "--output-dir", str(out_dir),
            "--artifact", f"MODEL_PATH=fs://{model_dir}",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "CLI --artifact value will take precedence" in (result.output + (result.stderr or ""))
    scripts = sorted(list(out_dir.glob("bulk_*"))[0].glob("*.sh"))
    script = scripts[0].read_text()
    assert f"--artifact MODEL_PATH=fs://{model_dir}" in script
    assert f"--artifact MODEL_PATH=fs://{csv_model_dir}" not in script


def test_compose_bulk_input_variable_csv_wins_over_cli(tmp_path):
    """For --set variables in compose, CSV value should take precedence over CLI."""
    wf = tmp_path / "wf.yaml"
    wf.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: TP_SIZE\n"
        "    value: 1\n"
        "workflow:\n"
        "  name: test_wf\n"
        "  tasks:\n"
        "    - name: run\n"
        "      script:\n"
        "        - echo ${{ variables.TP_SIZE }}\n"
    )
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,TP_SIZE\n{wf},8\n")
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "compose", "--bulk-input", str(csv_file),
            "-o", str(out_dir),
            "--set", "TP_SIZE=2",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "CSV value will take precedence" in (result.output + (result.stderr or ""))
    yaml_files = list(out_dir.glob("*/*.yaml"))
    assert len(yaml_files) == 1
    content = yaml_files[0].read_text()
    assert "value: '8'" in content or "value: 8" in content


def test_compose_bulk_input_artifact_cli_wins_over_csv(tmp_path):
    """For --artifact in compose, CLI value should take precedence over CSV."""
    cli_path = tmp_path / "cli_model"
    cli_path.mkdir()
    csv_path = tmp_path / "csv_model"
    csv_path.mkdir()
    wf = tmp_path / "wf.yaml"
    wf.write_text(
        'version: "0.1"\n'
        "artifacts:\n"
        "  - name: MY_MODEL\n"
        f"    uri: fs://{csv_path}\n"
        "workflow:\n"
        "  name: test_wf\n"
        "  tasks:\n"
        "    - name: run\n"
        "      script:\n"
        "        - echo done\n"
    )
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,MY_MODEL\n{wf},fs://{csv_path}\n")
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "compose", "--bulk-input", str(csv_file),
            "-o", str(out_dir),
            "--artifact", f"MY_MODEL=fs://{cli_path}",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "CLI --artifact value will take precedence" in (result.output + (result.stderr or ""))
    yaml_files = list(out_dir.glob("*/*.yaml"))
    assert len(yaml_files) == 1
    content = yaml_files[0].read_text()
    assert str(cli_path) in content
    assert str(csv_path) not in content


# --- CSV-without-bulk-input hint tests ---


def test_batch_csv_input_without_bulk_input_flag(tmp_path):
    """sflow batch with a .csv file but no --bulk-input exits with a helpful hint."""
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text("sflow_config_file\nworkflow.yaml\n")

    result = runner.invoke(
        app,
        [
            "batch",
            str(csv_file),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 1
    assert "CSV file(s) detected" in result.output
    assert "--bulk-input" in result.output


def test_batch_csv_via_file_flag_without_bulk_input(tmp_path):
    """sflow batch -f jobs.csv (no --bulk-input) exits with a helpful hint."""
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text("sflow_config_file\nworkflow.yaml\n")

    result = runner.invoke(
        app,
        [
            "batch",
            "-f", str(csv_file),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
        ],
    )
    assert result.exit_code == 1
    assert "CSV file(s) detected" in result.output
    assert "--bulk-input" in result.output


def test_bulk_submit_csv_file_rejected(tmp_path):
    """sflow batch --bulk-submit with a CSV file exits with a helpful hint."""
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text("sflow_config_file\nworkflow.yaml\n")

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit", str(csv_file),
            "--partition", "gpu",
            "--account", "test",
        ],
    )
    assert result.exit_code == 1
    assert "CSV file(s) detected" in result.output
    assert "--bulk-input" in result.output


# --- _resolve_sbatch_extra_args tests ---


def test_resolve_sbatch_extra_args_no_expressions():
    """Args without expressions are returned unchanged."""
    args = ["--exclusive", "--segment=4"]
    result = _resolve_sbatch_extra_args(args, [], None)
    assert result == ["--exclusive", "--segment=4"]


def test_resolve_sbatch_extra_args_with_variable_from_set_var():
    """Expression resolved from --set overrides."""
    args = ["--segment=${{ variables.SLURM_NODES }}"]
    result = _resolve_sbatch_extra_args(
        args, [], ["SLURM_NODES=6"]
    )
    assert result == ["--segment=6"]


def test_resolve_sbatch_extra_args_from_config_file(tmp_path):
    """Expression resolved from config YAML variable defaults."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "version: '0.1'\n"
        "variables:\n"
        "  SLURM_NODES:\n"
        "    value: 3\n"
    )
    args = ["--segment=${{ variables.SLURM_NODES }}"]
    result = _resolve_sbatch_extra_args(args, [cfg], None)
    assert result == ["--segment=3"]


def test_resolve_sbatch_extra_args_set_var_overrides_config(tmp_path):
    """--set overrides take priority over config defaults."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "version: '0.1'\n"
        "variables:\n"
        "  SLURM_NODES:\n"
        "    value: 3\n"
    )
    args = ["--segment=${{ variables.SLURM_NODES }}"]
    result = _resolve_sbatch_extra_args(args, [cfg], ["SLURM_NODES=8"])
    assert result == ["--segment=8"]


def test_resolve_sbatch_extra_args_mixed():
    """Mix of expression and non-expression args."""
    args = [
        "--exclusive",
        "--segment=${{ variables.SLURM_NODES }}",
        "--gres=gpu:8",
    ]
    result = _resolve_sbatch_extra_args(args, [], ["SLURM_NODES=4"])
    assert result == ["--exclusive", "--segment=4", "--gres=gpu:8"]


def test_resolve_sbatch_extra_args_undefined_variable_passthrough():
    """Undefined variables are passed through unchanged."""
    args = ["--segment=${{ variables.UNDEFINED_VAR }}"]
    result = _resolve_sbatch_extra_args(args, [], None)
    assert result == ["--segment=${{ variables.UNDEFINED_VAR }}"]


def test_resolve_sbatch_extra_args_shorthand_without_variables_prefix():
    """${{ SLURM_NODES }} shorthand (no 'variables.' prefix) resolves."""
    args = ["--segment=${{ SLURM_NODES }}"]
    result = _resolve_sbatch_extra_args(args, [], ["SLURM_NODES=4"])
    assert result == ["--segment=4"]


def test_resolve_sbatch_extra_args_shorthand_from_config(tmp_path):
    """Shorthand resolves from config file defaults."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "version: '0.1'\n"
        "variables:\n"
        "  GPUS_PER_NODE:\n"
        "    value: 8\n"
    )
    args = ["--gres=gpu:${{ GPUS_PER_NODE }}"]
    result = _resolve_sbatch_extra_args(args, [cfg], None)
    assert result == ["--gres=gpu:8"]


def test_resolve_sbatch_extra_args_both_syntaxes_in_same_call():
    """Both ${{ variables.X }} and ${{ X }} work in the same invocation."""
    args = [
        "--segment=${{ variables.SLURM_NODES }}",
        "--gres=gpu:${{ GPUS_PER_NODE }}",
    ]
    result = _resolve_sbatch_extra_args(
        args, [], ["SLURM_NODES=3", "GPUS_PER_NODE=8"]
    )
    assert result == ["--segment=3", "--gres=gpu:8"]


# --- CLI integration tests: -e expression in generated sbatch scripts ---


def test_batch_sbatch_extra_args_expression_resolved_in_script(
    mock_sflow_app, tmp_path
):
    """Full CLI: -e with ${{ variables.X }} produces resolved #SBATCH directive."""
    workflow_file = tmp_path / "wf.yaml"
    workflow_file.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  SLURM_NODES:\n"
        "    value: 4\n"
        "workflow:\n"
        "  name: test\n"
        "  tasks:\n"
        "    - name: hello\n"
        "      script:\n"
        "        - echo hello\n"
    )
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file", str(workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "4",
            "--sbatch-path", str(sbatch_path),
            "-e", "--segment=${{ variables.SLURM_NODES }}",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script = sbatch_path.read_text()
    assert "#SBATCH --segment=4" in script
    assert "${{" not in script.split("#SBATCH --segment")[1].split("\n")[0]


def test_batch_sbatch_extra_args_expression_with_set_override(
    mock_sflow_app, tmp_path
):
    """Full CLI: --set overrides variable before -e expression resolution."""
    workflow_file = tmp_path / "wf.yaml"
    workflow_file.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  SLURM_NODES:\n"
        "    value: 2\n"
        "workflow:\n"
        "  name: test\n"
        "  tasks:\n"
        "    - name: hello\n"
        "      script:\n"
        "        - echo hello\n"
    )
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file", str(workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "8",
            "--sbatch-path", str(sbatch_path),
            "--set", "SLURM_NODES=8",
            "-e", "--segment=${{ variables.SLURM_NODES }}",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script = sbatch_path.read_text()
    assert "#SBATCH --segment=8" in script


def test_batch_sbatch_extra_args_expression_mixed_with_plain(
    mock_sflow_app, tmp_path
):
    """Full CLI: mix of plain and expression -e args in generated script."""
    workflow_file = tmp_path / "wf.yaml"
    workflow_file.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  SLURM_NODES:\n"
        "    value: 3\n"
        "workflow:\n"
        "  name: test\n"
        "  tasks:\n"
        "    - name: hello\n"
        "      script:\n"
        "        - echo hello\n"
    )
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file", str(workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "3",
            "--sbatch-path", str(sbatch_path),
            "-e", "--exclusive",
            "-e", "--segment=${{ variables.SLURM_NODES }}",
            "-e", "--gres=gpu:8",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script = sbatch_path.read_text()
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --segment=3" in script
    assert "#SBATCH --gres=gpu:8" in script


def test_batch_sbatch_extra_args_expression_jinja2_arithmetic(
    mock_sflow_app, tmp_path
):
    """Full CLI: Jinja2 arithmetic in -e expression."""
    workflow_file = tmp_path / "wf.yaml"
    workflow_file.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  SLURM_NODES:\n"
        "    type: integer\n"
        "    value: 4\n"
        "  GPUS_PER_NODE:\n"
        "    type: integer\n"
        "    value: 8\n"
        "workflow:\n"
        "  name: test\n"
        "  tasks:\n"
        "    - name: hello\n"
        "      script:\n"
        "        - echo hello\n"
    )
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file", str(workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "4",
            "--sbatch-path", str(sbatch_path),
            "-e", "--gres=gpu:${{ variables.GPUS_PER_NODE }}",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script = sbatch_path.read_text()
    assert "#SBATCH --gres=gpu:8" in script


def test_bulk_input_sbatch_extra_args_expression_per_row(mock_sflow_app, tmp_path):
    """Bulk-input: -e expression resolved independently per CSV row."""
    workflow_file = tmp_path / "wf.yaml"
    workflow_file.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  SLURM_NODES:\n"
        "    type: integer\n"
        "    value: 1\n"
        "workflow:\n"
        "  name: test\n"
        "  tasks:\n"
        "    - name: hello\n"
        "      script:\n"
        "        - echo hello\n"
    )
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(
        "sflow_config_file,SLURM_NODES\n"
        f"{workflow_file.name},2\n"
        f"{workflow_file.name},5\n"
    )
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-input", str(csv_file),
            "--partition", "batch",
            "--account", "testaccount",
            "-e", "--segment=${{ variables.SLURM_NODES }}",
            "--output-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    scripts = sorted(out_dir.rglob("*.sh"))
    assert len(scripts) == 2

    script_1 = scripts[0].read_text()
    script_2 = scripts[1].read_text()
    assert "#SBATCH --segment=2" in script_1
    assert "#SBATCH --segment=5" in script_2
