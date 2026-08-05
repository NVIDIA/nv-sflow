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

import sflow.cli.batch as batch_mod
from sflow.cli import app
from sflow.cli.batch import (
    _batch_launch_strategy,
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


def test_batch_launch_strategy_defers_unimplemented_backends():
    try:
        _batch_launch_strategy("docker")
        assert False, "Expected Docker batch strategy to be deferred"
    except NotImplementedError as e:
        assert "docker" in str(e)
        assert "persistent launcher" in str(e)


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


def test_batch_node_filters_emit_sbatch_directives_and_forward_to_inner_run(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """--include-nodes/--exclude-nodes add #SBATCH --nodelist/--exclude and forward
    to the inner `sflow run`."""
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
            "--include-nodes",
            "node001,node002",
            "--exclude-nodes",
            "bad001",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_content = sbatch_path.read_text()
    assert "#SBATCH --nodelist=node001,node002" in script_content
    assert "#SBATCH --exclude=bad001" in script_content
    # Forwarded to the inner sflow run so every backend applies them.
    assert "--include-nodes node001" in script_content
    assert "--exclude-nodes bad001" in script_content


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


def _slurm_backend_workflow(extra_args: list[str]) -> str:
    """A single-slurm-backend workflow YAML with the given backend extra_args."""
    args_yaml = "".join(f"      - {a}\n" for a in extra_args)
    return (
        'version: "0.1"\n'
        "backends:\n"
        "  - name: slurm_cluster\n"
        "    type: slurm\n"
        "    default: true\n"
        "    account: testaccount\n"
        "    partition: batch\n"
        "    time: '00:10:00'\n"
        "    nodes: 1\n"
        "    gpus_per_node: 8\n"
        "    extra_args:\n"
        f"{args_yaml}"
        "workflow:\n"
        "  name: test\n"
        "  tasks:\n"
        "    - name: hello\n"
        "      script:\n"
        "        - echo hello\n"
    )


def test_batch_includes_backend_extra_args_single(mock_sflow_app, tmp_path):
    """Backend extra_args are emitted as #SBATCH directives in the single-backend script."""
    workflow_file = tmp_path / "wf.yaml"
    workflow_file.write_text(_slurm_backend_workflow(["--exclusive", "--gres=gpu:8"]))
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(workflow_file),
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
    assert "#SBATCH --exclusive" in script_content
    assert "#SBATCH --gres=gpu:8" in script_content


def test_batch_merges_and_dedups_backend_and_cli_extra_args(mock_sflow_app, tmp_path):
    """CLI --sbatch-extra-args and backend extra_args merge and de-dup by option (CLI wins)."""
    workflow_file = tmp_path / "wf.yaml"
    workflow_file.write_text(_slurm_backend_workflow(["--exclusive", "--gres=gpu:8"]))
    sbatch_path = tmp_path / "test.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
            "--nodes",
            "1",
            "--sbatch-path",
            str(sbatch_path),
            "-e",
            "--gres=gpu:4",
            "-e",
            "--constraint=gpu",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_content = sbatch_path.read_text()
    # CLI value wins on the conflicting --gres option; only one --gres directive.
    assert "#SBATCH --gres=gpu:4" in script_content
    assert "#SBATCH --gres=gpu:8" not in script_content
    assert script_content.count("#SBATCH --gres") == 1
    # Non-conflicting args from both sources are present exactly once.
    assert script_content.count("#SBATCH --exclusive") == 1
    assert "#SBATCH --constraint=gpu" in script_content


def test_dedup_merge_extra_args_override_wins():
    """Unit: dedup_merge_extra_args de-dups single-valued flags by option, override wins."""
    from sflow.utils.extra_args import dedup_merge_extra_args

    merged = dedup_merge_extra_args(
        ["--exclusive", "--gres=gpu:8", "--time=01:00:00"],
        ["--gres=gpu:4", "--constraint=gpu"],
    )
    assert merged == ["--exclusive", "--gres=gpu:4", "--time=01:00:00", "--constraint=gpu"]


def test_dedup_merge_extra_args_keeps_repeatable_kv_flags():
    """Repeatable key=value flags (e.g. --env) with distinct value-keys all
    survive; the same value-key is overridden by the later (override) value."""
    from sflow.utils.extra_args import dedup_merge_extra_args

    merged = dedup_merge_extra_args(
        ["--env=FOO=1", "--env=BAR=2"],
        ["--env=FOO=9", "--env=BAZ=3"],
    )
    # FOO overridden in place, BAR kept, BAZ appended.
    assert merged == ["--env=FOO=9", "--env=BAR=2", "--env=BAZ=3"]


def test_dedup_merge_extra_args_repeatable_kv_space_form():
    """Same, for the space form '--env FOO=1' (one token, space-separated)."""
    from sflow.utils.extra_args import dedup_merge_extra_args

    merged = dedup_merge_extra_args(
        ["--env FOO=1", "--env BAR=2"],
        ["--env FOO=9"],
    )
    assert merged == ["--env FOO=9", "--env BAR=2"]


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


def test_batch_dry_run_does_not_pass_slurm_overrides_to_sflow_app(
    mock_sflow_app, tmp_path
):
    workflow_file = tmp_path / "slurm_workflow.yaml"
    workflow_file.write_text(
        'version: "0.1"\n'
        "backends:\n"
        "  - name: slurm_cluster\n"
        "    type: slurm\n"
        "    account: testaccount\n"
        "    partition: batch\n"
        "    time: '00:10:00'\n"
        "    nodes: 1\n"
        "    gpus_per_node: 1\n"
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
            "--file",
            str(workflow_file),
            "--partition",
            "batch",
            "--account",
            "testaccount",
            "--nodes",
            "2",
            "--gpus-per-node",
            "4",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    kwargs = mock_sflow_app.run.call_args.kwargs
    assert "slurm_nodes" not in kwargs
    assert "slurm_gpus_per_node" not in kwargs


def test_batch_gpus_per_node_warns_it_is_not_sbatch_directive(
    mock_sflow_app, temp_workflow_file, tmp_path
):
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
            "--gpus-per-node",
            "4",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "backend.gpus_per_node=4" in result.output
    assert "salloc" in result.output
    assert "srun" in result.output
    assert "sbatch" in result.output

    script_content = sbatch_path.read_text()
    assert "#SBATCH --gpus-per-node" not in script_content
    assert "WARNING: backend.gpus_per_node=4" not in script_content
    assert "sflow planning only" not in script_content


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


def test_bulk_input_set_overrides_backend_nodes_in_generated_script(
    mock_sflow_app, tmp_path
):
    """Repro: `--set` overriding the backend node count reaches the GENERATED sbatch
    script, not just the dry-run. The backend sizes off ``nodes: ${{ variables.NUM_NODES }}``;
    the CSV sets NUM_NODES=2 and ``--set NUM_NODES=8`` must win in ``#SBATCH --nodes``."""
    wf_path = tmp_path / "wf.yaml"
    wf_path.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: NUM_NODES\n"
        "    value: 1\n"
        "backends:\n"
        "  - name: slurm\n"
        "    type: slurm\n"
        "    default: true\n"
        "    nodes: ${{ variables.NUM_NODES }}\n"
        "    gpus_per_node: 8\n"
        "    account: acct\n"
        "    partition: batch\n"
        '    time: "01:00:00"\n'
        "workflow:\n"
        "  name: test_wf\n"
        "  tasks:\n"
        "    - name: serve\n"
        "      script:\n"
        "        - echo hi\n"
    )
    out_dir = tmp_path / "sflow_output"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,NUM_NODES\n{wf_path},2\n")

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
            "--set",
            "NUM_NODES=8",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    scripts = list(out_dir.rglob("*.sh"))
    assert len(scripts) == 1
    script_text = scripts[0].read_text()
    # --set (8) must win over the stale CSV value (2), matching the dry-run.
    assert "#SBATCH --nodes=8" in script_text, script_text
    assert "#SBATCH --nodes=2" not in script_text
    assert "--set NUM_NODES=8" in script_text


def test_bulk_input_set_overrides_node_column_in_generated_script(
    mock_sflow_app, tmp_path
):
    """Same bug via a bare node-count column (no backend ``nodes`` field): ``--set``
    must still win over the CSV SLURM_NODES value in the ``#SBATCH --nodes`` directive."""
    wf_path = tmp_path / "wf.yaml"
    wf_path.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: SLURM_NODES\n"
        "    value: 1\n"
        "workflow:\n"
        "  name: test_wf\n"
        "  tasks:\n"
        "    - name: serve\n"
        "      script:\n"
        "        - echo hi\n"
    )
    out_dir = tmp_path / "sflow_output"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,SLURM_NODES\n{wf_path},2\n")

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
            "--set",
            "SLURM_NODES=8",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    scripts = list(out_dir.rglob("*.sh"))
    assert len(scripts) == 1
    script_text = scripts[0].read_text()
    assert "#SBATCH --nodes=8" in script_text, script_text
    assert "#SBATCH --nodes=2" not in script_text


def test_bulk_input_set_node_override_reflected_in_job_name(mock_sflow_app, tmp_path):
    """The derived job NAME encodes the node count; a ``--set`` node override must be
    reflected there too (not the stale CSV value) so the name matches the allocation."""
    wf_path = tmp_path / "wf.yaml"
    wf_path.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: SLURM_NODES\n"
        "    value: 1\n"
        "workflow:\n"
        "  name: test_wf\n"
        "  tasks:\n"
        "    - name: serve\n"
        "      script:\n"
        "        - echo hi\n"
    )
    out_dir = tmp_path / "sflow_output"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,SLURM_NODES\n{wf_path},2\n")

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
            "--set",
            "SLURM_NODES=8",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_text = list(out_dir.rglob("*.sh"))[0].read_text()
    assert "8n" in script_text  # job name encodes the effective (overridden) node count
    assert "2n_" not in script_text  # not the stale CSV value


def test_resolve_node_count_cli_set_overrides_csv_node_column():
    """`_resolve_node_count` precedence: explicit --nodes > global --set node var >
    the row's CSV node cell (so the job name tracks the effective allocation)."""
    from sflow.cli.batch import _resolve_node_count

    assert _resolve_node_count({"SLURM_NODES": "2"}, None, {"SLURM_NODES": "8"}) == "8n"
    assert _resolve_node_count({"SLURM_NODES": "2"}, None) == "2n"  # no override -> CSV
    assert _resolve_node_count({"SLURM_NODES": "2"}, 4, {"SLURM_NODES": "8"}) == "4n"


def test_first_node_column_int_scans_and_skips_malformed():
    """The consolidated node-column peek returns the first PARSEABLE node-count column,
    skips empty/malformed values (they can't mask a valid column), and returns None when
    nothing parses. Shared by the naming and both bulk paths."""
    from sflow.cli.batch import _first_node_column_int

    assert _first_node_column_int({"NUM_NODES": "4"}) == 4
    assert _first_node_column_int({"SLURM_NODES": "3"}) == 3
    # a malformed value does not mask a good column elsewhere in the scan
    assert _first_node_column_int({"SLURM_NODES": "oops", "NUM_NODES": "2"}) == 2
    # empty / whitespace / non-numeric only -> None
    assert _first_node_column_int({"NUM_NODES": "  "}) is None
    assert _first_node_column_int({"SLURM_NODES": "x"}) is None
    assert _first_node_column_int({}) is None
    # non-node columns are ignored
    assert _first_node_column_int({"MODEL": "5"}) is None


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
    assert "backend_job_id" in reader.fieldnames
    assert "sflow_output_dir" in reader.fieldnames
    assert "sflow_batch_dir" in reader.fieldnames
    assert rows[0]["slurm_job_id"] == "99999"
    assert rows[0]["backend_job_id"] == rows[0]["slurm_job_id"]
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
    assert rows[0]["backend_job_id"] == "not submitted"
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
    assert rows[0]["backend_job_id"] == "11111"
    assert rows[1]["slurm_job_id"] == "FAILED"
    assert rows[1]["backend_job_id"] == "FAILED"
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
    assert rows[0]["backend_job_id"] == "not submitted"
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


def test_bulk_input_with_cli_files_includes_them_in_sbatch_script(
    mock_sflow_app, tmp_path
):
    """CLI -f files should be prepended to each CSV row in generated sbatch scripts."""
    f_common = tmp_path / "common.yaml"
    f_common.write_text('version: "0.1"\nvariables:\n  - name: SHARED\n    value: yes\n')
    f_task = tmp_path / "task.yaml"
    f_task.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo ${{ variables.SHARED }}\n"
    )
    out_dir = tmp_path / "sflow_output"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file\n{f_task}\n")

    result = runner.invoke(
        app,
        [
            "batch",
            "-f", str(f_common),
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
    script = next(bulk_dirs[0].glob("*.sh")).read_text()
    common_arg = f"--file {shlex.quote(str(f_common.resolve()))}"
    task_arg = f"--file {shlex.quote(str(f_task.resolve()))}"
    assert common_arg in script
    assert task_arg in script
    assert script.index(common_arg) < script.index(task_arg)


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
    assert 'export SFLOW_RUN_ID_PREFIX="$SLURM_JOB_ID"' in script


def test_sbatch_script_copies_custom_sbatch_log_paths(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """Custom sbatch log patterns are the source paths copied into workflow output."""
    out = tmp_path / "out.sh"
    stdout_pattern = tmp_path / "slurm_logs" / "%j.custom.out"
    stderr_pattern = tmp_path / "slurm_logs" / "%j.custom.err"
    result = runner.invoke(
        app,
        [
            "batch",
            str(temp_workflow_file),
            "--partition", "gpu",
            "--account", "test",
            "--nodes", "1",
            "--sbatch-output", str(stdout_pattern),
            "--sbatch-error", str(stderr_pattern),
            "-o", str(out),
        ],
    )
    assert result.exit_code == 0
    script = out.read_text()
    assert f"#SBATCH --output={stdout_pattern}" in script
    assert f"#SBATCH --error={stderr_pattern}" in script
    assert f"SBATCH_OUT_PATTERN={shlex.quote(str(stdout_pattern))}" in script
    assert f"SBATCH_ERR_PATTERN={shlex.quote(str(stderr_pattern))}" in script
    assert 'SBATCH_OUT="${SBATCH_OUT_PATTERN//%j/$SLURM_JOB_ID}"' in script
    assert 'SBATCH_ERR="${SBATCH_ERR_PATTERN//%j/$SLURM_JOB_ID}"' in script
    assert 'SBATCH_OUT=' + shlex.quote(str(tmp_path / "sflow_output")) not in script


def test_sbatch_script_always_copies_logs_via_finalize_trap(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """The .out/.err (and config) copy runs from the exit/signal trap, always.

    Putting the copy in a trap (not just a post-run block) means a failed
    bootstrap, a crashed run, or a Slurm cancel/timeout still leaves the sbatch
    logs in the workflow output dir. When the run never created its
    ``<job id>-*`` dir, the copy falls back to a ``<job id>-sflow-submit`` dir so
    the logs are still captured somewhere useful.
    """
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

    # The copy lives in the finalize function, which the trap runs on exit AND
    # on the signals Slurm uses for timeout/cancel/preempt.
    assert "_sflow_finalize() {" in script
    assert "trap _sflow_finalize EXIT INT TERM HUP" in script
    # Fallback dir so logs land somewhere even if no workflow dir was created.
    assert '"${SLURM_JOB_ID}-sflow-submit"' in script
    assert 'mkdir -p "$SFLOW_WF_DIR"' in script

    # The .out/.err copy is inside the finalize function (between its definition
    # and the trap registration), i.e. not gated behind a happy-path-only block.
    fn_idx = script.index("_sflow_finalize() {")
    cp_out_idx = script.index('cp "$SBATCH_OUT" "$SFLOW_WF_DIR/"')
    cp_err_idx = script.index('cp "$SBATCH_ERR" "$SFLOW_WF_DIR/"')
    trap_idx = script.index("trap _sflow_finalize")
    assert fn_idx < cp_out_idx < trap_idx
    assert fn_idx < cp_err_idx < trap_idx

    # A failed copy/cleanup must never change the job's exit status: the finalize
    # fn captures the incoming rc first (before any step), disarms every trap so it
    # runs exactly once (no re-entry from a signal mid-cleanup or the EXIT trap
    # firing after a signal-triggered run), then exits with that rc last -- with
    # errexit disabled and every step guarded so it always reaches the exit.
    assert "_sflow_rc=$?" in script
    assert "trap - EXIT INT TERM HUP" in script
    assert 'exit "$_sflow_rc"' in script
    rc_capture_idx = script.index("_sflow_rc=$?")
    disarm_idx = script.index("trap - EXIT INT TERM HUP")
    rc_exit_idx = script.index('exit "$_sflow_rc"')
    # rc captured before the disarm (so $? is the real incoming rc, not the trap
    # builtin's 0), and the disarm precedes the copy/cleanup steps.
    assert fn_idx < rc_capture_idx < disarm_idx < cp_out_idx
    assert cp_err_idx < rc_exit_idx < trap_idx
    # Cleanup is best-effort too (guarded), so a stuck rm can't fail the job.
    assert (
        'rm -rf "$SFLOW_VENV_DIR" ${SFLOW_SRC_DIR:+"$SFLOW_SRC_DIR"} 2>/dev/null || true'
        in script
    )


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

    def test_compound_expression_resolves_via_full_pipeline(self, tmp_path):
        """A COMPOUND ${{ }} expression (arithmetic) resolves through the SAME full
        config pipeline as the dry-run, so `sflow batch` sizing matches `sflow run`.
        The bespoke regex only matched a bare ``${{ variables.X }}`` and returned None
        here. Requires a complete backend so the pipeline validates (incomplete configs
        fall back to the regex, unchanged)."""
        f = tmp_path / "wf.yaml"
        f.write_text(
            'version: "0.1"\n'
            "variables:\n"
            "  - name: NUM_NODES\n    type: integer\n    value: 3\n"
            "backends:\n"
            "  - name: slurm_cluster\n    type: slurm\n"
            "    nodes: ${{ variables.NUM_NODES * 2 }}\n"
            "    partition: gpu\n    account: test\n    gpus_per_node: 8\n"
            '    time: "01:00:00"\n'
            "workflow:\n  name: wf\n  tasks:\n"
            "    - name: t1\n      script:\n        - echo hi\n"
        )
        assert _derive_nodes([f]) == 6  # 3 * 2, via full resolver
        assert _derive_nodes([f], cli_overrides=["NUM_NODES=5"]) == 10  # 5 * 2


def test_bulk_input_compound_node_expression_matches_dry_run(mock_sflow_app, tmp_path):
    """Backend sizes nodes off a COMPOUND expression of a CSV var; the generated sbatch
    must resolve it like the dry-run (8), not fall back to the raw CSV cell (4)."""
    wf_path = tmp_path / "wf.yaml"
    wf_path.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: NUM_NODES\n    type: integer\n    value: 1\n"
        "backends:\n"
        "  - name: slurm\n    type: slurm\n    default: true\n"
        "    nodes: ${{ variables.NUM_NODES * 2 }}\n"
        "    gpus_per_node: 8\n    account: acct\n    partition: batch\n"
        '    time: "01:00:00"\n'
        "workflow:\n  name: wf\n  tasks:\n    - name: serve\n      script: [echo hi]\n"
    )
    out_dir = tmp_path / "out"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,NUM_NODES\n{wf_path},4\n")

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
    script_text = list(out_dir.rglob("*.sh"))[0].read_text()
    assert "#SBATCH --nodes=8" in script_text, script_text  # 4 * 2, matching dry-run
    assert "#SBATCH --nodes=4" not in script_text  # not the raw CSV cell


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


def test_batch_bulk_input_variable_cli_wins_over_csv(mock_sflow_app, tmp_path):
    """For --set variables, CLI value should take precedence over CSV."""
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
    assert "CLI --set value will take precedence" in (result.output + (result.stderr or ""))
    scripts = sorted(list(out_dir.glob("bulk_*"))[0].glob("*.sh"))
    script = scripts[0].read_text()
    assert "--set TP_SIZE=2" in script
    assert "--set TP_SIZE=8" not in script
    import csv as csv_mod

    results_csv = list(out_dir.glob("bulk_*/results.csv"))[0]
    with open(results_csv) as f:
        rows = list(csv_mod.DictReader(f))
    assert rows[0]["TP_SIZE"] == "2"


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


def test_compose_bulk_input_variable_cli_wins_over_csv(tmp_path):
    """For --set variables in compose, CLI value should take precedence over CSV."""
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
    assert "CLI --set value will take precedence" in (result.output + (result.stderr or ""))
    yaml_files = list(out_dir.glob("*/*.yaml"))
    assert len(yaml_files) == 1
    content = yaml_files[0].read_text()
    assert "value: '2'" in content or "value: 2" in content


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


def test_resolve_sbatch_extra_args_domain_from_config(tmp_path):
    """${{ variables.X.domain }} resolves to the domain list from config."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "version: '0.1'\n"
        "variables:\n"
        "  CONCURRENCY:\n"
        "    value: 16\n"
        "    type: integer\n"
        "    domain: [1, 4, 16, 64]\n"
    )
    args = ["--comment=${{ variables.CONCURRENCY.domain }}"]
    result = _resolve_sbatch_extra_args(args, [cfg], None)
    assert result == ["--comment=[1, 4, 16, 64]"]


def test_resolve_sbatch_extra_args_domain_shorthand(tmp_path):
    """${{ X.domain }} shorthand resolves domain from config."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "version: '0.1'\n"
        "variables:\n"
        "  MODE:\n"
        "    value: fast\n"
        "    domain: [fast, balanced, accurate]\n"
    )
    args = ["--comment=${{ MODE.domain }}"]
    result = _resolve_sbatch_extra_args(args, [cfg], None)
    assert result == ["--comment=['fast', 'balanced', 'accurate']"]


def test_resolve_sbatch_extra_args_domain_empty_when_not_set():
    """${{ variables.X.domain }} returns [] when variable has no domain."""
    args = ["--comment=${{ variables.X.domain }}"]
    result = _resolve_sbatch_extra_args(args, [], ["X=42"])
    assert result == ["--comment=[]"]


def test_resolve_sbatch_extra_args_value_and_domain_together(tmp_path):
    """Value and domain can be accessed in the same arg list."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "version: '0.1'\n"
        "variables:\n"
        "  NODES:\n"
        "    value: 4\n"
        "    domain: [1, 2, 4, 8]\n"
    )
    args = [
        "--segment=${{ variables.NODES }}",
        "--comment=${{ variables.NODES.domain }}",
    ]
    result = _resolve_sbatch_extra_args(args, [cfg], None)
    assert result == ["--segment=4", "--comment=[1, 2, 4, 8]"]


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


def test_batch_sbatch_extra_args_domain_in_script(
    mock_sflow_app, tmp_path
):
    """Full CLI: -e with ${{ variables.X.domain }} produces resolved #SBATCH directive."""
    workflow_file = tmp_path / "wf.yaml"
    workflow_file.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  CONCURRENCY:\n"
        "    value: 16\n"
        "    type: integer\n"
        "    domain: [1, 4, 16, 64]\n"
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
            "--nodes", "1",
            "--sbatch-path", str(sbatch_path),
            "-e", "--comment=${{ variables.CONCURRENCY.domain }}",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script = sbatch_path.read_text()
    assert "#SBATCH --comment=[1, 4, 16, 64]" in script


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


class _FakeSflowDistribution:
    def __init__(self, *, version: str, direct_url_text: str | None = None):
        self.version = version
        self._direct_url_text = direct_url_text

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self._direct_url_text


def test_resolve_effective_sflow_version_uses_requested_revision():
    dist = _FakeSflowDistribution(
        version="0.2.0",
        direct_url_text=(
            '{"url":"https://github.com/NVIDIA/nv-sflow.git",'
            '"vcs_info":{"vcs":"git","requested_revision":"feature/infmax_v3","commit_id":"abc123"}}'
        ),
    )

    with patch("sflow.cli.batch.importlib_metadata.distribution", return_value=dist):
        assert batch_mod._resolve_effective_sflow_version(None) == "feature/infmax_v3"


def test_resolve_effective_sflow_version_uses_editable_repo_branch(tmp_path):
    repo_path = tmp_path / "nv-sflow"
    repo_path.mkdir()
    dist = _FakeSflowDistribution(
        version="0.2.0",
        direct_url_text=(
            '{"url":"file://'
            + str(repo_path)
            + '","dir_info":{"editable":true}}'
        ),
    )

    with (
        patch("sflow.cli.batch.importlib_metadata.distribution", return_value=dist),
        patch("sflow.cli.batch._git_current_ref", return_value="feature/infmax_v3"),
    ):
        assert batch_mod._resolve_effective_sflow_version(None) == "feature/infmax_v3"


def test_resolve_effective_sflow_version_falls_back_to_installed_package_version():
    dist = _FakeSflowDistribution(version="0.2.0", direct_url_text=None)

    with patch("sflow.cli.batch.importlib_metadata.distribution", return_value=dist):
        assert batch_mod._resolve_effective_sflow_version(None) == "0.2.0"


def test_batch_defaults_sflow_version_from_execution_env(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    sbatch_path = tmp_path / "test.sh"

    with patch(
        "sflow.cli.batch._resolve_effective_sflow_version",
        return_value="feature/infmax_v3",
    ):
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
    # With neither --sflow-version nor --sflow-source-path, the install link must be
    # the git ref auto-detected from the running sflow env, installed with
    # --prerelease=allow ...
    expected_install = (
        '"$VIRTUAL_ENV/bin/uv" pip install '
        "'sflow @ git+https://github.com/NVIDIA/nv-sflow.git@feature/infmax_v3' "
        "--prerelease=allow"
    )
    assert expected_install in script_content
    # ... and the editable (--sflow-source-path) install path must NOT be taken.
    assert 'pip install -e ".[dev]"' not in script_content


def test_batch_sflow_version_accepts_repo_url_with_ref(
    mock_sflow_app, temp_workflow_file, tmp_path
):
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
            "--sflow-version",
            "https://git.example.com/example/sflow.git@develop",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_content = sbatch_path.read_text()
    assert (
        "sflow @ git+https://git.example.com/example/sflow.git@develop"
        in script_content
    )
    assert (
        "git+https://github.com/NVIDIA/nv-sflow.git@https://git.example.com"
        not in script_content
    )


# ---------------------------------------------------------------------------
# Install sflow from a private PyPI registry (--sflow-index-url)
# ---------------------------------------------------------------------------


def test_sflow_pypi_requirement_builds_specs():
    """_sflow_pypi_requirement turns a validated version/specifier into a requirement."""
    assert batch_mod._sflow_pypi_requirement(None) == "sflow"
    assert batch_mod._sflow_pypi_requirement("") == "sflow"
    assert batch_mod._sflow_pypi_requirement("   ") == "sflow"
    assert batch_mod._sflow_pypi_requirement("0.2.1") == "sflow==0.2.1"
    assert batch_mod._sflow_pypi_requirement("==0.2.1") == "sflow==0.2.1"
    assert batch_mod._sflow_pypi_requirement(">=0.2,<0.3") == "sflow>=0.2,<0.3"
    assert batch_mod._sflow_pypi_requirement("~=0.2.0") == "sflow~=0.2.0"


def test_batch_sflow_index_url_installs_pinned_version_from_registry(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """--sflow-index-url installs a pinned wheel from the private index, not git."""
    sbatch_path = tmp_path / "test.sh"
    index_url = "https://example.com/artifactory/api/pypi/private/simple"

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
            "--sflow-version",
            "0.2.1",
            "--sflow-index-url",
            index_url,
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_content = sbatch_path.read_text()
    assert (
        '"$VIRTUAL_ENV/bin/uv" pip install sflow==0.2.1 '
        f"--extra-index-url {index_url} --prerelease=allow" in script_content
    )
    assert (
        "set +x\n"
        '"$VIRTUAL_ENV/bin/uv" pip install sflow==0.2.1 '
        f"--extra-index-url {index_url} --prerelease=allow\n"
        "set -x"
        in script_content
    )
    # Registry mode must NOT fall back to the git or editable install paths.
    assert "git+" not in script_content
    assert 'pip install -e ".[dev]"' not in script_content


def test_batch_sflow_index_url_without_version_installs_latest(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """Omitting --sflow-version with --sflow-index-url installs the latest wheel."""
    sbatch_path = tmp_path / "test.sh"
    index_url = "https://example.com/simple"

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
            "--sflow-index-url",
            index_url,
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_content = sbatch_path.read_text()
    assert (
        f'"$VIRTUAL_ENV/bin/uv" pip install sflow --extra-index-url {index_url} '
        "--prerelease=allow" in script_content
    )
    assert "git+" not in script_content


def test_batch_sflow_index_url_passes_through_version_spec(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """An operator-led --sflow-version is passed through as a PEP 508 specifier."""
    sbatch_path = tmp_path / "test.sh"
    index_url = "https://example.com/simple"

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
            "--sflow-version",
            ">=0.2,<0.3",
            "--sflow-index-url",
            index_url,
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_content = sbatch_path.read_text()
    assert "'sflow>=0.2,<0.3'" in script_content
    assert f"--extra-index-url {index_url}" in script_content


def test_batch_sflow_index_url_and_source_path_mutually_exclusive(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """--sflow-index-url and --sflow-source-path cannot be combined."""
    src_dir = tmp_path / "sflow_src"
    src_dir.mkdir()

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
            "--sflow-index-url",
            "https://example.com/simple",
            "--sflow-source-path",
            str(src_dir),
            "--sbatch-path",
            str(tmp_path / "test.sh"),
        ],
    )

    assert result.exit_code != 0
    assert (
        "--sflow-index-url and --sflow-source-path are mutually exclusive"
        in result.output
    )


def test_batch_sflow_index_url_rejects_embedded_credentials(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """Credentials for --sflow-index-url must come from node-local auth config."""
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
            "--sflow-index-url",
            "https://user:token@example.com/simple",
            "--sbatch-path",
            str(tmp_path / "test.sh"),
        ],
    )

    assert result.exit_code != 0
    assert "--sflow-index-url must not contain embedded credentials" in result.output


def test_sflow_version_error_pypi_route_validates_pep440_specifiers():
    """PyPI route: only plain PEP 440 versions/specifiers pass."""
    # Accepted: empty (latest), bare versions, wildcards, pre/dev, specifiers.
    for ok in (
        None,
        "",
        "   ",
        "0.2.1",
        "0.2.*",
        "v0.2.0",
        "0.2.1.dev3",
        "1.0.0rc1",
        "==0.2.1",
        ">=0.2,<0.3",
        "~=0.2.0",
    ):
        assert batch_mod._sflow_version_error(ok, registry=True) is None, ok
    # Rejected: git refs, package names, PEP 508 direct references, markers.
    for bad in (
        "main",
        "feature/x",
        "sflow==0.2.1",
        "sflow @ https://example.com/sflow-0.2.1-py3-none-any.whl",
        "name @ https://example.com/x.whl",
        "https://example.com/x.whl",
        "0.2.1 ; python_version>'3.9'",
    ):
        assert batch_mod._sflow_version_error(bad, registry=True) is not None, bad


def test_sflow_version_error_git_route_allows_refs_rejects_whitespace():
    """Git route: refs/URLs (and empty) pass; whitespace is rejected."""
    for ok in (None, "", "main", "v0.1.0", "feature/x",
               "https://git.example.com/example/sflow.git@develop"):
        assert batch_mod._sflow_version_error(ok, registry=False) is None, ok
    # A PyPI-style spec is fine as a value here (no whitespace); only whitespace
    # -- which a git ref/URL never contains -- is flagged as a likely mistake.
    assert batch_mod._sflow_version_error("main branch", registry=False) is not None
    assert batch_mod._sflow_version_error(">=0.2, <0.3", registry=False) is not None


def test_batch_sflow_index_url_rejects_git_ref_version(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """In registry mode a git-ref-style --sflow-version fails fast at generation."""
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
            "--sflow-version",
            "main",
            "--sflow-index-url",
            "https://example.com/simple",
            "--sbatch-path",
            str(tmp_path / "test.sh"),
        ],
    )

    assert result.exit_code != 0
    assert "not a valid PyPI version specifier" in result.output


def test_batch_sflow_index_url_rejects_direct_reference_version(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """A PEP 508 direct reference as --sflow-version is rejected in registry mode."""
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
            "--sflow-version",
            "sflow @ https://example.com/sflow-0.2.1-py3-none-any.whl",
            "--sflow-index-url",
            "https://example.com/simple",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code != 0
    assert "not a valid PyPI version specifier" in result.output
    assert not sbatch_path.exists()


def test_batch_sflow_index_url_threads_into_bulk_submit(mock_sflow_app, tmp_path):
    """Registry mode applies to every script generated by --bulk-submit."""
    (tmp_path / "wf1.yaml").write_text(_VALID_SFLOW_YAML)
    out_dir = tmp_path / "output"
    index_url = "https://example.com/simple"

    result = runner.invoke(
        app,
        [
            "batch",
            "--bulk-submit",
            str(tmp_path),
            "--partition",
            "gpu",
            "--account",
            "test",
            "--output-dir",
            str(out_dir),
            "--nodes",
            "1",
            "--sflow-version",
            "0.2.1",
            "--sflow-index-url",
            index_url,
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    bulk_dirs = list(out_dir.glob("bulk_*"))
    assert bulk_dirs, "expected a bulk_* output directory"
    scripts = sorted(bulk_dirs[0].glob("*.sh"))
    assert scripts, "expected at least one generated script"
    content = scripts[0].read_text()
    assert (
        f"pip install sflow==0.2.1 --extra-index-url {index_url} --prerelease=allow"
        in content
    )
    assert "git+" not in content


# ---------------------------------------------------------------------------
# Multi-backend (per-backend salloc) driver script generation
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_multi_backend_file(tmp_path):
    """A self-contained two-Slurm-backend workflow for multi-backend tests."""
    f = tmp_path / "multi_backend.yaml"
    f.write_text(
        """
version: "0.1"
variables:
  PARTITION_A: { value: part_a }
  PARTITION_B: { value: part_b }
  ACCT: { value: my_acct }
backends:
  - name: cluster_a
    type: slurm
    default: true
    account: ${{ variables.ACCT }}
    partition: ${{ variables.PARTITION_A }}
    nodes: 2
    gpus_per_node: 4
    time: 30
  - name: cluster_b
    type: slurm
    account: ${{ variables.ACCT }}
    partition: ${{ variables.PARTITION_B }}
    nodes: 1
    gpus_per_node: 0
    time: 30
operators:
  - name: worker_a
    type: srun
    ntasks_per_node: 1
  - name: worker_b
    type: srun
    ntasks_per_node: 1
workflow:
  name: mb
  tasks:
    - name: task_a
      operator: worker_a
      script: ["echo a"]
    - name: task_b
      backend: cluster_b
      operator: worker_b
      script: ["echo b"]
"""
    )
    return f


def test_resolve_slurm_backends_resolves_and_orders(temp_multi_backend_file):
    """_resolve_slurm_backends resolves ${{ }} fields, preserves config order, and
    applies --set overrides."""
    from sflow.cli.batch import _resolve_slurm_backends

    backends = _resolve_slurm_backends([temp_multi_backend_file], ["PARTITION_A=ovr_a"])

    assert [b.name for b in backends] == ["cluster_a", "cluster_b"]
    assert backends[0].partition == "ovr_a"  # --set wins over config default
    assert backends[1].partition == "part_b"
    assert backends[0].account == "my_acct"
    assert backends[0].nodes == 2
    assert backends[1].nodes == 1
    assert backends[0].gpus_per_node == 4
    assert backends[1].gpus_per_node == 0


def test_resolve_slurm_backends_accepts_dict_style_backends(tmp_path):
    """Raw backend detection should honor the schema's dict-to-list normalization."""
    from sflow.cli.batch import _resolve_slurm_backends

    f = tmp_path / "dict_backends.yaml"
    f.write_text(
        """
version: "0.1"
variables:
  ACCT: { value: my_acct }
backends:
  cluster-a:
    type: slurm
    default: true
    account: ${{ variables.ACCT }}
    partition: part_a
    nodes: 2
    gpus_per_node: 4
    time: 30
  cluster_b:
    type: slurm
    account: ${{ variables.ACCT }}
    partition: part_b
    nodes: 1
    gpus_per_node: 0
    time: 30
workflow:
  name: mb
  tasks:
    - name: task_a
      script: ["echo a"]
"""
    )

    backends = _resolve_slurm_backends([f], None)

    assert [b.name for b in backends] == ["cluster-a", "cluster_b"]
    assert [b.partition for b in backends] == ["part_a", "part_b"]


def test_batch_multi_backend_generates_per_backend_salloc_driver(
    mock_sflow_app, temp_multi_backend_file, tmp_path
):
    """A config with two Slurm backends produces a single driver sbatch sized to
    the leader backend (NOT a hetjob); every other backend runs its own salloc
    at runtime. The driver exports the per-backend-salloc markers so each backend
    gets its own Slurm job id (required for pyxis/enroot on every partition)."""
    sbatch_path = tmp_path / "mb.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_multi_backend_file),
            "--partition",
            "ignored_cli_partition",
            "--account",
            "ignored_cli_account",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    s = sbatch_path.read_text()

    # No heterogeneous job is emitted anymore.
    assert "#SBATCH hetjob" not in s
    assert "SFLOW_SLURM_HET_BACKEND" not in s
    # Driver sbatch is sized to the leader backend (cluster_a).
    assert "#SBATCH --partition=part_a" in s
    assert "#SBATCH --nodes=2" in s
    # The non-leader backend's partition is NOT baked into the sbatch; cluster_b
    # allocates itself at runtime via salloc.
    assert "#SBATCH --partition=part_b" not in s
    # Per-backend-salloc markers drive SlurmBackend.allocate().
    assert "export SFLOW_SLURM_MULTI_BACKEND_SALLOC=1" in s
    assert "export SFLOW_SLURM_WRAPPER_BACKEND=cluster_a" in s
    # The single-allocation CLI partition must NOT be emitted.
    assert "ignored_cli_partition" not in s


def test_multi_backend_driver_leader_sbatch_and_inner_run_extra_salloc_args(tmp_path):
    """Multi-backend: #SBATCH carries only the leader's merged set (its extra_args
    + the CLI -e args, de-duped, CLI wins). The CLI args are threaded into the
    inner `sflow run` as --extra-salloc-args so every backend merges them into
    its own salloc at runtime (no magic env var)."""
    f = tmp_path / "mb_extra.yaml"
    f.write_text(
        """
version: "0.1"
backends:
  - name: cluster_a
    type: slurm
    default: true
    account: acct
    partition: part_a
    nodes: 2
    gpus_per_node: 4
    time: 30
    extra_args: ["--exclusive", "--gres=gpu:8"]
  - name: cluster_b
    type: slurm
    account: acct
    partition: part_b
    nodes: 1
    gpus_per_node: 0
    time: 30
operators:
  - name: w
    type: srun
    ntasks_per_node: 1
workflow:
  name: mb
  tasks:
    - name: a
      operator: w
      script: ["echo a"]
    - name: b
      backend: cluster_b
      operator: w
      script: ["echo b"]
"""
    )
    sbatch_path = tmp_path / "mb.sh"
    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(f),
            "--partition",
            "ignored",
            "--account",
            "ignored",
            "--sbatch-path",
            str(sbatch_path),
            "-e",
            "--gres=gpu:4",
            "-e",
            "--constraint=gpu",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    s = sbatch_path.read_text()

    # Leader is cluster_a (heavier). #SBATCH = leader merged set; CLI wins on --gres.
    assert "export SFLOW_SLURM_WRAPPER_BACKEND=cluster_a" in s
    assert "#SBATCH --exclusive" in s
    assert "#SBATCH --gres=gpu:4" in s
    assert "#SBATCH --gres=gpu:8" not in s
    assert s.count("#SBATCH --gres") == 1
    assert "#SBATCH --constraint=gpu" in s

    # No magic env var: the CLI args reach each backend's salloc via the inner
    # `sflow run --extra-salloc-args`, which merges them into every backend.
    assert "SFLOW_SLURM_CLI_EXTRA_ARGS" not in s
    assert "--extra-salloc-args --gres=gpu:4" in s
    assert "--extra-salloc-args --constraint=gpu" in s


def test_select_wrapper_backend_picks_most_resource_heavy():
    """The driver sbatch wraps the heaviest backend (most nodes, then most total
    GPUs, then most GPUs/node); ties keep config-declaration order."""
    from sflow.cli.batch import _ResolvedSlurmBackend, _select_wrapper_backend

    def be(name, nodes, gpn):
        return _ResolvedSlurmBackend(
            name=name,
            partition="p",
            account="a",
            nodes=nodes,
            time=None,
            extra_args=[],
            gpus_per_node=gpn,
        )

    # Most nodes wins even when another backend has more GPUs/node.
    assert (
        _select_wrapper_backend([be("a", 1, 0), be("b", 3, 8), be("c", 2, 8)]).name
        == "b"
    )
    # Tie on nodes -> more total GPUs wins.
    assert _select_wrapper_backend([be("d", 2, 2), be("e", 2, 8)]).name == "e"
    # Full tie -> first declared (config order).
    assert _select_wrapper_backend([be("f", 2, 4), be("g", 2, 4)]).name == "f"


def test_batch_multi_backend_driver_wraps_heaviest_backend(mock_sflow_app, tmp_path):
    """The driver sbatch is sized to the most resource-heavy backend (here the
    second-declared cluster_b), not the first/default; lighter backends salloc."""
    f = tmp_path / "heavy_second.yaml"
    f.write_text(
        """
version: "0.1"
backends:
  - name: cluster_a
    type: slurm
    default: true
    account: acct
    partition: part_a
    nodes: 1
    gpus_per_node: 0
    time: 30
  - name: cluster_b
    type: slurm
    account: acct
    partition: part_b
    nodes: 3
    gpus_per_node: 8
    time: 30
workflow:
  name: mb
  tasks:
    - name: task_a
      script: ["echo a"]
    - name: task_b
      backend: cluster_b
      script: ["echo b"]
"""
    )
    sbatch_path = tmp_path / "mb.sh"
    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(f),
            "--partition",
            "ignored",
            "--account",
            "ignored",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    s = sbatch_path.read_text()
    # Driver wraps the heavier cluster_b (3 nodes / 8 GPUs), not the default cluster_a.
    assert "export SFLOW_SLURM_WRAPPER_BACKEND=cluster_b" in s
    assert "#SBATCH --partition=part_b" in s
    assert "#SBATCH --nodes=3" in s
    # cluster_a (lighter) is NOT baked into the sbatch; it sallocs at runtime.
    assert "#SBATCH --partition=part_a" not in s


def test_batch_multi_backend_exports_wrapper_backend_name_safely(
    mock_sflow_app, tmp_path
):
    """The leader backend name is exported as a value (hyphens are safe) so
    SlurmBackend.allocate can identify which backend reuses the driver alloc."""
    f = tmp_path / "hyphen_backend.yaml"
    f.write_text(
        """
version: "0.1"
backends:
  - name: cluster-a
    type: slurm
    default: true
    account: acct
    partition: part_a
    nodes: 1
    gpus_per_node: 1
    time: 30
  - name: cluster-b
    type: slurm
    account: acct
    partition: part_b
    nodes: 1
    gpus_per_node: 0
    time: 30
workflow:
  name: mb
  tasks:
    - name: task_a
      script: ["echo a"]
"""
    )
    sbatch_path = tmp_path / "mb.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(f),
            "--partition",
            "ignored",
            "--account",
            "ignored",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    s = sbatch_path.read_text()
    assert "export SFLOW_SLURM_WRAPPER_BACKEND=cluster-a" in s
    assert "export SFLOW_SLURM_MULTI_BACKEND_SALLOC=1" in s
    assert "SFLOW_SLURM_HET_BACKEND" not in s


def test_batch_multi_backend_applies_cli_extra_args_to_driver(
    mock_sflow_app, temp_multi_backend_file, tmp_path
):
    """CLI sbatch extras apply to the single driver sbatch (one allocation)."""
    sbatch_path = tmp_path / "mb.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_multi_backend_file),
            "--partition",
            "ignored",
            "--account",
            "ignored",
            "--sbatch-extra-args",
            "--exclusive",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert sbatch_path.read_text().count("#SBATCH --exclusive") == 1


def test_batch_multi_backend_honors_set_overrides(
    mock_sflow_app, temp_multi_backend_file, tmp_path
):
    """--set overrides set the leader's driver partition and flow to non-leader
    backends through the wrapped `sflow run` command (they salloc at runtime)."""
    sbatch_path = tmp_path / "mb.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_multi_backend_file),
            "--partition",
            "ignored",
            "--account",
            "ignored",
            "--set",
            "PARTITION_A=genesisq",
            "--set",
            "PARTITION_B=gamoraq",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    s = sbatch_path.read_text()
    # The driver sbatch is sized to the leader (cluster_a) partition only.
    assert "#SBATCH --partition=genesisq" in s
    assert "#SBATCH --partition=gamoraq" not in s
    # The non-leader partition flows to sflow run, which sallocs it at runtime.
    assert "PARTITION_B=gamoraq" in s


def test_batch_multi_backend_reports_per_backend_plan(
    mock_sflow_app, temp_multi_backend_file, tmp_path
):
    """The CLI reports each backend's partition/nodes/gpus and its allocation
    role (leader reuses the driver; others salloc), not a single derived value."""
    sbatch_path = tmp_path / "mb.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_multi_backend_file),
            "--partition",
            "ignored",
            "--account",
            "ignored",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    out = result.output
    assert "2 Slurm backends detected" in out
    assert "multi-backend driver job" in out
    assert "[cluster_a] partition=part_a, nodes=2, gpus_per_node=4" in out
    assert "[cluster_b] partition=part_b, nodes=1, gpus_per_node=0" in out
    assert "driver/leader" in out
    assert "own salloc" in out
    # The misleading single-value derivation messages must NOT appear here.
    assert "--nodes not specified, derived from config" not in out
    assert "--gpus-per-node not specified, derived from config" not in out


def test_batch_multi_backend_warns_cli_nodes_ignored(
    mock_sflow_app, temp_multi_backend_file, tmp_path
):
    """Passing -N/-G with a multi-backend config warns that they are ignored
    (each backend uses its own config values)."""
    sbatch_path = tmp_path / "mb.sh"

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_multi_backend_file),
            "--partition",
            "ignored",
            "--account",
            "ignored",
            "--nodes",
            "9",
            "--gpus-per-node",
            "9",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    out = result.output
    assert "ignored for multi-backend" in out
    # The CLI single value must not leak into the driver directives; the driver
    # is sized to the leader backend (nodes=2), and the non-leader nodes (1) are
    # not baked into the sbatch (cluster_b sallocs at runtime).
    s = sbatch_path.read_text()
    assert "#SBATCH --nodes=9" not in s
    assert "#SBATCH --nodes=2" in s
    assert "#SBATCH --nodes=1" not in s


def test_batch_single_backend_is_not_hetjob(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """A config without >=2 Slurm backends keeps the single-allocation script."""
    sbatch_path = tmp_path / "single.sh"

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
    s = sbatch_path.read_text()
    assert "#SBATCH hetjob" not in s
    assert "SFLOW_SLURM_HET_GROUP" not in s
    assert "#SBATCH --partition=batch" in s


def test_slurm_strategy_plan_reports_multi_backend_components(temp_multi_backend_file):
    """The Slurm launch strategy owns multi-backend planning: it detects multiple
    Slurm backends and reports each backend's plan + allocation role, instead of
    batch() doing it."""
    from sflow.cli.batch import BatchPlanRequest, SlurmBatchLaunchStrategy

    plan = SlurmBatchLaunchStrategy().plan(
        BatchPlanRequest(
            files=[temp_multi_backend_file],
            set_var=None,
            cli_nodes=None,
            cli_gpus_per_node=None,
        )
    )

    joined = "\n".join(plan.messages)
    assert plan.error is None
    assert "2 Slurm backends detected" in joined
    assert "[cluster_a] partition=part_a, nodes=2, gpus_per_node=4" in joined
    assert "[cluster_b] partition=part_b, nodes=1, gpus_per_node=0" in joined
    assert "driver/leader" in joined
    # Each backend carries its own nodes/gpus, so dry-run must not override them
    # with a single CLI value.
    assert plan.dry_run_nodes is None
    assert plan.dry_run_gpus_per_node is None


def test_slurm_strategy_plan_derives_single_backend_nodes(tmp_path):
    """For a single Slurm backend the strategy derives nodes/gpus from config."""
    from sflow.cli.batch import BatchPlanRequest, SlurmBatchLaunchStrategy

    f = tmp_path / "single.yaml"
    f.write_text(
        """
version: "0.1"
backends:
  - name: only
    type: slurm
    default: true
    account: acct
    partition: part
    nodes: 3
    gpus_per_node: 8
    time: 30
workflow:
  name: single
  tasks:
    - name: t
      script: ["echo hi"]
"""
    )

    plan = SlurmBatchLaunchStrategy().plan(
        BatchPlanRequest(files=[f], set_var=None, cli_nodes=None, cli_gpus_per_node=None)
    )

    assert plan.error is None
    assert plan.nodes == 3
    assert plan.gpus_per_node == 8
    assert any("derived from config: 3" in m for m in plan.messages)


def test_batch_single_job_delegates_planning_to_strategy(
    mock_sflow_app, temp_workflow_file, tmp_path, monkeypatch
):
    """batch() must obtain its node/gpu/hetjob plan from the launch strategy
    rather than deciding hetjob state itself."""
    sbatch_path = tmp_path / "out.sh"
    real_plan = batch_mod.SlurmBatchLaunchStrategy.plan
    calls: list[str] = []

    def spy_plan(self, request):
        calls.append("called")
        return real_plan(self, request)

    monkeypatch.setattr(batch_mod.SlurmBatchLaunchStrategy, "plan", spy_plan)

    result = runner.invoke(
        app,
        [
            "batch",
            "--file",
            str(temp_workflow_file),
            "--partition",
            "batch",
            "--account",
            "acct",
            "--nodes",
            "1",
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert calls == ["called"]


def test_batch_script_bootstraps_venv_with_resolved_system_python(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """The generated sbatch venv bootstrap must not use a bare ``python3``.

    A bare ``python3`` resolves through PATH, so if the submitting shell has a
    virtualenv activated, sbatch's default ``--export=ALL`` leaks that venv into
    the job and ``python3 -m venv`` runs the caller's interpreter -- which fails
    with "Exec format error" when the login node and compute node differ in
    architecture (e.g. x86 login vs aarch64 Grace compute).

    The script must instead clear the inherited virtualenv and resolve a real
    system python3: prefer well-known absolute locations, then fall back to a
    PATH-resolved interpreter so nodes that install python outside /usr/bin work.

    It must also drop a leaked PYTHONPATH: --export=ALL carries the submitter's
    PYTHONPATH into the job, and if it points at an sflow source tree (e.g. the
    under-dev e2e exports PYTHONPATH=<repo>/src) it is prepended to sys.path
    ahead of site-packages and shadows the per-job editable install, so sflow
    would be imported from that shared tree instead of the job's own venv/copy.
    """
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
    # Venv is created via a resolved interpreter, never a bare python3 from PATH.
    assert '"$SFLOW_BOOTSTRAP_PYTHON" -m venv "$SFLOW_VENV_DIR"' in script_content
    assert "python3 -m venv" not in script_content
    # Well-known absolute location is tried first, with a PATH fallback for nodes
    # that install python elsewhere.
    assert "/usr/bin/python3" in script_content
    assert "command -v python3" in script_content
    # An inherited (possibly wrong-arch) virtualenv must be neutralized.
    assert "unset VIRTUAL_ENV" in script_content
    assert 'grep -vxF "$VIRTUAL_ENV/bin"' in script_content
    # A leaked PYTHONPATH would shadow the per-job editable install (import from
    # a shared source tree instead of this job's venv/copy), so it is dropped too.
    assert "PYTHONPATH" in script_content
    assert "unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT PYTHONHOME PYTHONPATH" in script_content


def test_batch_script_creates_fresh_per_job_venv_without_flock(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """Each Slurm job builds its own fresh venv -- no shared venv, no flock.

    Sharing one venv across concurrent batch jobs caused create/install races and
    required flock plus an exists/reuse branch (and, because the bootstrap ran in
    the flock subshell, a fragile parent-side exit-code propagation). Keying the
    venv path on ``$SLURM_JOB_ID`` gives every job a private venv, eliminating the
    race -- and the subshell -- entirely, so ``set -e`` in the main shell can abort
    the job directly on any bootstrap failure.
    """
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

    # With no --sflow-venv-path, the parent defaults to compute-node-local scratch
    # resolved at runtime ($TMPDIR, falling back to /tmp) -- not a baked submit-side
    # path -- so it honors per-node/per-job cluster scratch.
    assert (
        'SFLOW_VENV_PARENT="${TMPDIR:-/tmp}/sflow_compute_node_venv"' in script_content
    )
    # Per-job venv path keyed on the Slurm job id (PID fallback off-Slurm).
    assert (
        'SFLOW_VENV_DIR="$SFLOW_VENV_PARENT/.sflow_venv-${SLURM_JOB_ID:-$$}"'
        in script_content
    )
    # Created fresh with a resolved system python3, and sflow runs from that venv.
    assert '"$SFLOW_BOOTSTRAP_PYTHON" -m venv "$SFLOW_VENV_DIR"' in script_content
    assert '"$SFLOW_VENV_DIR/bin/sflow" run' in script_content

    # No shared-venv concurrency machinery remains (the word "flock" may still
    # appear in an explanatory comment, so assert the command/lock are gone).
    assert "flock -w" not in script_content
    assert "SFLOW_LOCK" not in script_content
    assert ".sflow_venv.lock" not in script_content
    # No "venv already exists -> reuse" branch and no subshell exit-code dance.
    assert "SFLOW_ACTIVATE" not in script_content
    assert "SFLOW_VENV_RC" not in script_content

    # Disposable: the per-job venv (and per-job source copy, if any) is cleaned up
    # by the finalize trap on exit AND on the termination signals Slurm uses
    # (timeout/cancel/preempt), because a bare EXIT trap does not run on an
    # untrapped signal -- otherwise the venvs would leak on every cancelled job.
    assert "trap _sflow_finalize EXIT INT TERM HUP" in script_content
    assert (
        'rm -rf "$SFLOW_VENV_DIR" ${SFLOW_SRC_DIR:+"$SFLOW_SRC_DIR"}' in script_content
    )

    # Bootstrap still fails fast. There is no subshell now, so set -e in the main
    # shell aborts the job directly; it is re-disabled before the run so log
    # copying still happens after a failed workflow.
    assert "\nset -e\n" in script_content
    assert "\nset +e\n" in script_content
    set_e_idx = script_content.index("\nset -e\n")
    venv_create_idx = script_content.index('"$SFLOW_BOOTSTRAP_PYTHON" -m venv')
    set_plus_e_idx = script_content.index("\nset +e\n")
    assert set_e_idx < venv_create_idx < set_plus_e_idx

    # The job exits with the workflow's status (captured right after the run), so a
    # failed workflow is a failed Slurm job rather than always exit 0 (which would
    # mask failures and wrongly satisfy --dependency=afterok).
    assert "SFLOW_RUN_RC=$?" in script_content
    assert 'exit "$SFLOW_RUN_RC"' in script_content
    run_idx = script_content.index('"$SFLOW_VENV_DIR/bin/sflow" run')
    rc_capture_idx = script_content.index("SFLOW_RUN_RC=$?")
    rc_exit_idx = script_content.index('exit "$SFLOW_RUN_RC"')
    assert run_idx < rc_capture_idx < rc_exit_idx


def test_batch_sflow_venv_path_overrides_default_parent(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """An explicit --sflow-venv-path bakes that absolute path as the venv parent.

    The default is compute-node-local scratch resolved at runtime, but passing a
    path (e.g. a shared filesystem) pins SFLOW_VENV_PARENT to that resolved path
    instead, so the default $TMPDIR expression is not emitted.
    """
    venv_parent = tmp_path / "shared_venvs"
    venv_parent.mkdir()
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
            "--sflow-venv-path",
            str(venv_parent),
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_content = sbatch_path.read_text()

    assert f"SFLOW_VENV_PARENT={shlex.quote(str(venv_parent.resolve()))}" in script_content
    assert "${TMPDIR:-/tmp}/sflow_compute_node_venv" not in script_content


def test_batch_script_installs_editable_from_source_path(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """--sflow-source-path installs sflow editable from a *per-job copy*.

    A shared source tree cannot be reused across concurrent jobs: an editable
    build rewrites setuptools-scm's _version.py and *.egg-info back into the tree,
    so concurrent installs would race. Each job instead copies the checkout into
    its own per-job dir (rsync, or a tar fallback when rsync is absent;
    heavy/generated paths excluded) and runs ``uv pip install -e ".[dev]"`` from
    that copy into the fresh per-job venv.
    """
    src_dir = tmp_path / "sflow_src"
    src_dir.mkdir()
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
            "--sflow-source-path",
            str(src_dir),
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_content = sbatch_path.read_text()

    # Per-job source dir keyed on the Slurm job id, populated from the provided
    # checkout, then an editable install from the copy.
    assert (
        'SFLOW_SRC_DIR="$SFLOW_VENV_PARENT/.sflow_src-${SLURM_JOB_ID:-$$}"'
        in script_content
    )
    # rsync preferred, with a tar fallback when the compute node has no rsync.
    assert "if command -v rsync >/dev/null 2>&1; then" in script_content
    assert "rsync -a --exclude=" in script_content
    assert (
        f'{shlex.quote(str(src_dir.resolve()))}/ "$SFLOW_SRC_DIR/"' in script_content
    )
    assert (
        f'tar -C {shlex.quote(str(src_dir.resolve()))} --exclude='
        in script_content
    )
    assert '-cf - . | tar -C "$SFLOW_SRC_DIR" -xf -' in script_content
    assert 'cd "$SFLOW_SRC_DIR"' in script_content
    assert '"$VIRTUAL_ENV/bin/uv" pip install -e ".[dev]"' in script_content
    # The shared source tree is NOT cd'd into directly, and no git-ref install.
    assert f"cd {shlex.quote(str(src_dir.resolve()))}\n" not in script_content
    assert "sflow @ git+" not in script_content
    # Heavy/generated paths are excluded from the copy (and still bootstrapped
    # into the fresh per-job venv, not a bare pip).
    assert "--exclude=sflow_output" in script_content
    assert "--exclude='*.egg-info'" in script_content
    assert '"$VIRTUAL_ENV/bin/pip" install uv' in script_content
    # Correctness-critical: the per-job venv/source dirs must be excluded so that
    # when the venv parent IS the source (e2e passes both = $REPO_DIR) the copy
    # does not recurse into its own destination / sibling jobs' growing copies.
    assert "--exclude='.sflow_venv*'" in script_content
    assert "--exclude='.sflow_src*'" in script_content
    # The per-job source copy is cleaned up with the venv on exit/signal.
    assert '${SFLOW_SRC_DIR:+"$SFLOW_SRC_DIR"}' in script_content


def test_batch_source_copy_excludes_per_job_dirs_when_venv_parent_is_source(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """Regression: venv parent == source must not cause a runaway self-copy.

    The under-dev e2e passes --sflow-venv-path and --sflow-source-path both as
    $REPO_DIR, so SFLOW_SRC_DIR (=$SFLOW_VENV_PARENT/.sflow_src-<job id>) lands
    inside the directory being copied. The copy MUST exclude the per-job venv/src
    dirs; otherwise rsync/tar recurse into the destination (and every concurrent
    job's growing copy), which hung CI for hours. Verify both the source-copy and
    its destination share the same parent and that both per-job globs are excluded.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
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
            "--sflow-venv-path",
            str(repo),
            "--sflow-source-path",
            str(repo),
            "--sbatch-path",
            str(sbatch_path),
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    script_content = sbatch_path.read_text()

    # venv parent and the source being copied are the same dir (the e2e setup).
    assert f"SFLOW_VENV_PARENT={shlex.quote(str(repo.resolve()))}" in script_content
    assert f'{shlex.quote(str(repo.resolve()))}/ "$SFLOW_SRC_DIR/"' in script_content
    # Both per-job globs must be in the exclude args of the copy, so the copy can
    # never descend into its own destination or a sibling job's copy.
    assert "--exclude='.sflow_venv*'" in script_content
    assert "--exclude='.sflow_src*'" in script_content


def test_batch_sflow_version_and_source_path_are_mutually_exclusive(
    mock_sflow_app, temp_workflow_file, tmp_path
):
    """--sflow-version and --sflow-source-path cannot be combined."""
    src_dir = tmp_path / "sflow_src"
    src_dir.mkdir()
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
            "--sflow-version",
            "main",
            "--sflow-source-path",
            str(src_dir),
        ],
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


# ---------------------------------------------------------------------------
# a short CSV row must be a clean CLI error, not a traceback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,content",
    [
        # csv.DictReader PADS a row that has fewer fields than the header with None, so
        # this row reaches the code as {"sflow_config_file": None} -- past the header
        # check, straight into `.split()`.
        ("short row (trailing column padded to None)", "extra,sflow_config_file\nonly_one_field\n"),
        ("explicitly empty value", "sflow_config_file,extra\n,x\n"),
        ("whitespace-only value", "sflow_config_file,extra\n   ,x\n"),
    ],
)
def test_bulk_input_row_missing_config_file_is_a_clean_error(
    mock_sflow_app, tmp_path, case, content
):
    """A CSV whose header has the column but whose ROW leaves it blank must be rejected.

    The header check alone is not enough, and the gap was not cosmetic: every downstream
    ``row["sflow_config_file"].split()`` raised ``AttributeError: 'NoneType' object has no
    attribute 'split'``, which escaped typer as a raw Python traceback rather than a CLI
    error -- so malformed input turned into a crash. Note the asymmetry that hid it:
    ``row_missable`` already guarded its OPTIONAL column with ``or ""`` while the REQUIRED
    one was dereferenced unconditionally.
    """
    csv_file = _write_csv(tmp_path / "bad.csv", content)
    result = runner.invoke(
        app,
        [
            "batch", "--bulk-input", str(csv_file),
            "--partition", "batch", "--account", "acct", "--nodes", "1",
        ],
        catch_exceptions=True,
    )

    assert not isinstance(result.exception, AttributeError), (
        f"{case}: malformed CSV crashed with a traceback instead of a CLI error: "
        f"{result.exception!r}"
    )
    assert result.exit_code == 1, case
    assert "sflow_config_file" in result.output, case
    # The message must locate the offending row -- "some row is bad" is not actionable
    # for a CSV with hundreds of rows.
    assert "row 1" in result.output, f"{case}: no row number in {result.output!r}"


def test_bulk_input_row_check_names_the_offending_row_number(mock_sflow_app, tmp_path):
    """With several rows, the error must point at the bad one, not the first."""
    wf = _write_workflow_with_vars(tmp_path / "wf.yaml")
    csv_file = _write_csv(
        tmp_path / "bad.csv",
        f"sflow_config_file,TP_SIZE\n{wf},4\n{wf},8\n,16\n",
    )
    result = runner.invoke(
        app,
        [
            "batch", "--bulk-input", str(csv_file),
            "--partition", "batch", "--account", "acct", "--nodes", "1",
        ],
        catch_exceptions=True,
    )
    assert not isinstance(result.exception, AttributeError)
    assert "row 3" in result.output, result.output


def test_read_bulk_csv_rejects_blank_config_file_at_the_ingestion_point(tmp_path):
    """Validated in read_bulk_csv so ALL of its callers (batch, compose, run) are covered
    by one guard rather than each dereference site growing its own."""
    csv_file = _write_csv(tmp_path / "bad.csv", "extra,sflow_config_file\nonly_one_field\n")
    with pytest.raises(ValueError, match="sflow_config_file"):
        batch_mod.read_bulk_csv(csv_file)


def test_read_bulk_csv_still_accepts_a_valid_row(tmp_path):
    """The new per-row check must not reject well-formed input, including the
    space-separated multi-file form."""
    csv_file = _write_csv(
        tmp_path / "ok.csv", "sflow_config_file,TP_SIZE\na.yaml b.yaml,4\n"
    )
    columns, rows = batch_mod.read_bulk_csv(csv_file)
    assert columns == ["sflow_config_file", "TP_SIZE"]
    assert rows[0]["sflow_config_file"] == "a.yaml b.yaml"
