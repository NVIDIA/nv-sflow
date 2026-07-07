# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sflow run --bulk-input --row feature."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from sflow.cli import app
from sflow.cli.batch import (
    _parse_kv_list,
    merge_row_overrides,
    read_bulk_csv,
    resolve_csv_row,
    resolve_row_files,
    row_missable,
)
from sflow.cli.run import _resolve_bulk_input_row

runner = CliRunner()


@pytest.fixture
def mock_sflow_app():
    with patch("sflow.cli.run._sflow_app") as mock_app:
        mock_app.run = MagicMock(return_value=None)
        mock_app.last_workflow_output_dir = None
        yield mock_app


@pytest.fixture
def workflow_files(tmp_path):
    """Create minimal workflow YAML files for testing."""
    base = tmp_path / "base.yaml"
    base.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  SERVER_PORT:\n"
        "    type: integer\n"
        "    value: 8000\n"
        "workflow:\n"
        "  name: test\n"
        "  tasks:\n"
        "    - name: server\n"
        "      script:\n"
        "        - echo hello\n"
    )
    variant = tmp_path / "variant.yaml"
    variant.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  MY_VAR:\n"
        "    type: integer\n"
        "    value: 1\n"
    )
    return base, variant


@pytest.fixture
def csv_file(tmp_path, workflow_files):
    """Create a test CSV with 3 rows."""
    base, variant = workflow_files
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "sflow_config_file,MY_VAR,SERVER_PORT,missable_tasks\n"
        f"{base.name} {variant.name},10,8000,\n"
        f"{base.name} {variant.name},20,8001,\n"
        f"{base.name},30,8002,server\n"
    )
    return csv_path


# -- _resolve_bulk_input_row unit tests --


def test_resolve_bulk_input_row_basic(csv_file, workflow_files):
    """Test basic CSV row resolution."""
    base, variant = workflow_files
    files, set_var, artifact, missable = _resolve_bulk_input_row(
        bulk_input=csv_file,
        row_selectors=["1"],
        cli_files=[],
        cli_set_var=None,
        cli_artifact=None,
        cli_missable=None,
    )
    assert len(files) == 2
    assert files[0].name == "base.yaml"
    assert files[1].name == "variant.yaml"
    assert "MY_VAR=10" in set_var
    assert "SERVER_PORT=8000" in set_var
    assert artifact is None
    assert missable is None


def test_resolve_bulk_input_row_with_missable(csv_file, workflow_files):
    """Test that missable_tasks column is picked up."""
    _base, variant = workflow_files
    files, set_var, artifact, missable = _resolve_bulk_input_row(
        bulk_input=csv_file,
        row_selectors=["3"],
        cli_files=[variant],
        cli_set_var=None,
        cli_artifact=None,
        cli_missable=None,
    )
    assert missable == ["server"]


def test_resolve_bulk_input_row_cli_files_prepended(csv_file, tmp_path):
    """Test that CLI -f files are prepended and deduped."""
    extra = tmp_path / "extra.yaml"
    extra.write_text('version: "0.1"\n')

    files, _, _, _ = _resolve_bulk_input_row(
        bulk_input=csv_file,
        row_selectors=["1"],
        cli_files=[extra],
        cli_set_var=None,
        cli_artifact=None,
        cli_missable=None,
    )
    assert files[0].name == "extra.yaml"
    assert len(files) == 3


def test_resolve_bulk_input_row_cli_set_var_merged(csv_file):
    """Test that CLI --set overrides merge with CSV columns (CLI wins)."""
    _, set_var, _, _ = _resolve_bulk_input_row(
        bulk_input=csv_file,
        row_selectors=["2"],
        cli_files=[],
        cli_set_var=["MY_VAR=999", "EXTRA=hello"],
        cli_artifact=None,
        cli_missable=None,
    )
    var_map = dict(v.split("=", 1) for v in set_var)
    assert var_map["MY_VAR"] == "999"
    assert var_map["EXTRA"] == "hello"


def test_resolve_csv_row_out_of_range(csv_file):
    """Test that out-of-range row index raises IndexError from resolve_csv_row."""
    with pytest.raises(IndexError, match="out of range"):
        resolve_csv_row(
            csv_path=csv_file,
            row_idx=99,
        )


def test_resolve_bulk_input_row_out_of_range(csv_file):
    """Test that out-of-range row index raises BadParameter via _resolve_bulk_input_row."""
    import typer

    with pytest.raises(typer.BadParameter):
        _resolve_bulk_input_row(
            bulk_input=csv_file,
            row_selectors=["99"],
            cli_files=[],
            cli_set_var=None,
            cli_artifact=None,
            cli_missable=None,
        )


def test_resolve_bulk_input_row_multiple_rows_rejected(csv_file):
    """Test that multiple row indices are rejected."""
    import typer

    with pytest.raises(typer.BadParameter, match="exactly one row"):
        _resolve_bulk_input_row(
            bulk_input=csv_file,
            row_selectors=["1", "2"],
            cli_files=[],
            cli_set_var=None,
            cli_artifact=None,
            cli_missable=None,
        )


def test_resolve_bulk_input_missing_sflow_config_column(tmp_path):
    """Test error when CSV lacks sflow_config_file column."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("name,value\nfoo,bar\n")
    with pytest.raises(ValueError, match="sflow_config_file"):
        _resolve_bulk_input_row(
            bulk_input=bad_csv,
            row_selectors=["1"],
            cli_files=[],
            cli_set_var=None,
            cli_artifact=None,
            cli_missable=None,
        )


def test_resolve_bulk_input_empty_csv(tmp_path):
    """Test error when CSV has headers but no data rows."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("sflow_config_file,MY_VAR\n")
    with pytest.raises(ValueError, match="no data rows"):
        _resolve_bulk_input_row(
            bulk_input=empty_csv,
            row_selectors=["1"],
            cli_files=[],
            cli_set_var=None,
            cli_artifact=None,
            cli_missable=None,
        )


# -- CLI integration tests --


def test_cli_run_bulk_input_dry_run(mock_sflow_app, csv_file):
    """Test sflow run --bulk-input --row --dry-run invokes SflowApp.run correctly."""
    result = runner.invoke(
        app,
        [
            "run",
            "--bulk-input", str(csv_file),
            "--row", "1",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    mock_sflow_app.run.assert_called_once()
    call_kwargs = mock_sflow_app.run.call_args
    assert call_kwargs.kwargs["dry_run"] is True
    passed_files = call_kwargs.kwargs["file"]
    assert len(passed_files) == 2
    overrides = call_kwargs.kwargs.get("variable_overrides") or []
    override_map = dict(v.split("=", 1) for v in overrides)
    assert override_map.get("MY_VAR") == "10"


def test_cli_run_bulk_input_row2(mock_sflow_app, csv_file):
    """Test selecting row 2 passes correct overrides."""
    result = runner.invoke(
        app,
        [
            "run",
            "--bulk-input", str(csv_file),
            "--row", "2",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    overrides = mock_sflow_app.run.call_args.kwargs.get("variable_overrides") or []
    override_map = dict(v.split("=", 1) for v in overrides)
    assert override_map.get("MY_VAR") == "20"
    assert override_map.get("SERVER_PORT") == "8001"


def test_cli_run_bulk_input_without_row_fails(mock_sflow_app, csv_file):
    """Test that --bulk-input without --row produces an error."""
    result = runner.invoke(
        app,
        ["run", "--bulk-input", str(csv_file), "--dry-run"],
    )
    assert result.exit_code != 0
    assert "--row" in result.output


def test_cli_run_row_without_bulk_input_fails(mock_sflow_app):
    """Test that --row without --bulk-input produces an error."""
    result = runner.invoke(
        app,
        ["run", "--row", "1", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "--bulk-input" in result.output


def test_cli_run_bulk_input_with_cli_files(mock_sflow_app, csv_file, tmp_path):
    """Test that CLI -f files are prepended to CSV config files."""
    extra = tmp_path / "extra.yaml"
    extra.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  EXTRA_VAR:\n"
        "    value: yes\n"
    )
    result = runner.invoke(
        app,
        [
            "run",
            "-f", str(extra),
            "--bulk-input", str(csv_file),
            "--row", "1",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    passed_files = mock_sflow_app.run.call_args.kwargs["file"]
    assert passed_files[0].name == "extra.yaml"
    assert len(passed_files) == 3


def test_cli_run_bulk_input_out_of_range(mock_sflow_app, csv_file):
    """Test that out-of-range row index produces an error."""
    result = runner.invoke(
        app,
        ["run", "--bulk-input", str(csv_file), "--row", "99", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "out of range" in result.output.lower() or "Row 99" in result.output


def test_cli_run_bulk_input_negative_last(mock_sflow_app, csv_file, workflow_files):
    """--row=-1 resolves to the last CSV row (row 3 in a 3-row CSV)."""
    _base, variant = workflow_files
    files, set_var, _artifact, missable = _resolve_bulk_input_row(
        bulk_input=csv_file,
        row_selectors=["-1"],
        cli_files=[variant],
        cli_set_var=None,
        cli_artifact=None,
        cli_missable=None,
    )
    assert any("30" in v for v in set_var)
    assert missable == ["server"]


def test_cli_run_bulk_input_negative_second_to_last(mock_sflow_app, csv_file):
    """--row=-2 resolves to the second-to-last CSV row (row 2)."""
    result = runner.invoke(
        app,
        ["run", "--bulk-input", str(csv_file), "--row=-2", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    overrides = mock_sflow_app.run.call_args.kwargs.get("variable_overrides") or []
    override_map = dict(v.split("=", 1) for v in overrides)
    assert override_map.get("MY_VAR") == "20"


def test_cli_run_bulk_input_negative_first(mock_sflow_app, csv_file):
    """--row=-3 resolves to the first row (row 1 in a 3-row CSV)."""
    result = runner.invoke(
        app,
        ["run", "--bulk-input", str(csv_file), "--row=-3", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    overrides = mock_sflow_app.run.call_args.kwargs.get("variable_overrides") or []
    override_map = dict(v.split("=", 1) for v in overrides)
    assert override_map.get("MY_VAR") == "10"


def test_cli_run_bulk_input_negative_out_of_range(mock_sflow_app, csv_file):
    """--row=-99 is out of range for a 3-row CSV."""
    result = runner.invoke(
        app,
        ["run", "--bulk-input", str(csv_file), "--row=-99", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "out of range" in result.output.lower() or "Row" in result.output


def _min_workflow(tmp_path) -> Path:
    wf = tmp_path / "wf.yaml"
    wf.write_text(
        'version: "0.1"\n'
        "workflow:\n"
        "  name: t\n"
        "  tasks:\n"
        "    - name: hello\n"
        "      script:\n"
        "        - echo hi\n"
    )
    return wf


def test_cli_run_extra_salloc_args_routed_to_slurm(mock_sflow_app, tmp_path):
    """--extra-salloc-args is threaded to SflowApp.run under the 'slurm' type."""
    wf = _min_workflow(tmp_path)
    result = runner.invoke(
        app,
        ["run", "-f", str(wf), "--dry-run", "--extra-salloc-args", "--gpus-per-node=4"],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert mock_sflow_app.run.call_args.kwargs["backend_extra_args_by_type"] == {
        "slurm": ["--gpus-per-node=4"]
    }


def test_cli_run_extra_docker_args_routed_to_docker(mock_sflow_app, tmp_path):
    """--extra-docker-args is threaded to SflowApp.run under the 'docker' type."""
    wf = _min_workflow(tmp_path)
    result = runner.invoke(
        app,
        ["run", "-f", str(wf), "--dry-run", "--extra-docker-args", "--shm-size=16g"],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert mock_sflow_app.run.call_args.kwargs["backend_extra_args_by_type"] == {
        "docker": ["--shm-size=16g"]
    }


def test_cli_run_extra_args_generic_fans_out_to_all_channels(mock_sflow_app, tmp_path):
    """--extra-args is generic: it fans out to the slurm salloc, docker run, and
    kubectl channels so whichever backend the recipe uses picks it up."""
    wf = _min_workflow(tmp_path)
    result = runner.invoke(
        app,
        ["run", "-f", str(wf), "--dry-run", "--extra-args", "--foo=1"],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    kwargs = mock_sflow_app.run.call_args.kwargs
    assert kwargs["backend_extra_args_by_type"] == {
        "slurm": ["--foo=1"],
        "docker": ["--foo=1"],
    }
    assert kwargs["kubectl_config"].extra_args == ["--foo=1"]


def test_cli_run_extra_args_short_flag_is_generic(mock_sflow_app, tmp_path):
    """The -e short flag still works and maps to the generic --extra-args."""
    wf = _min_workflow(tmp_path)
    result = runner.invoke(
        app,
        ["run", "-f", str(wf), "--dry-run", "-e", "--bar=2"],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    kwargs = mock_sflow_app.run.call_args.kwargs
    assert kwargs["backend_extra_args_by_type"] == {
        "slurm": ["--bar=2"],
        "docker": ["--bar=2"],
    }
    assert kwargs["kubectl_config"].extra_args == ["--bar=2"]


def test_cli_run_specific_extra_args_override_generic(mock_sflow_app, tmp_path):
    """A specific --extra-salloc-args overrides --extra-args on a conflicting option;
    channels without a specific override keep the generic value."""
    wf = _min_workflow(tmp_path)
    result = runner.invoke(
        app,
        [
            "run", "-f", str(wf), "--dry-run",
            "--extra-args", "--gres=gpu:1",
            "--extra-salloc-args", "--gres=gpu:8",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    by_type = mock_sflow_app.run.call_args.kwargs["backend_extra_args_by_type"]
    # slurm: the specific --extra-salloc-args wins on the conflicting --gres option.
    assert by_type["slurm"] == ["--gres=gpu:8"]
    # docker: no docker-specific override, so it keeps the generic value.
    assert by_type["docker"] == ["--gres=gpu:1"]


def test_cli_run_generic_extra_args_flagged_generic_for_kubectl(mock_sflow_app, tmp_path):
    """Generic --extra-args routed to kubectl are flagged as generic-origin so the
    k8s backend can warn they are being applied as kubectl global flags."""
    wf = _min_workflow(tmp_path)
    result = runner.invoke(
        app,
        ["run", "-f", str(wf), "--dry-run", "--extra-args", "--gpus-per-node=4"],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    kube_cfg = mock_sflow_app.run.call_args.kwargs["kubectl_config"]
    assert kube_cfg.extra_args == ["--gpus-per-node=4"]
    assert kube_cfg.generic_extra_args == ["--gpus-per-node=4"]


def test_cli_run_explicit_kubectl_args_not_flagged_generic(mock_sflow_app, tmp_path):
    """An explicit --extra-kubectl-args value is intentional, so it is NOT flagged
    as generic-origin (no misrouting warning). A generic arg on a different option
    is still flagged."""
    wf = _min_workflow(tmp_path)
    result = runner.invoke(
        app,
        [
            "run", "-f", str(wf), "--dry-run",
            "--extra-args", "--foo=1",
            "--extra-kubectl-args", "--request-timeout=30s",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    kube_cfg = mock_sflow_app.run.call_args.kwargs["kubectl_config"]
    assert kube_cfg.extra_args == ["--foo=1", "--request-timeout=30s"]
    # Only the generic --foo is flagged; the explicit --request-timeout is not.
    assert kube_cfg.generic_extra_args == ["--foo=1"]


# -- Shared batch helper unit tests --


class TestReadBulkCsv:
    def test_basic(self, csv_file):
        columns, rows = read_bulk_csv(csv_file)
        assert "sflow_config_file" in columns
        assert len(rows) == 3

    def test_missing_column(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("name,value\nfoo,bar\n")
        with pytest.raises(ValueError, match="sflow_config_file"):
            read_bulk_csv(bad)

    def test_empty_csv(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("sflow_config_file,MY_VAR\n")
        with pytest.raises(ValueError, match="no data rows"):
            read_bulk_csv(empty)

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_bulk_csv(empty)


class TestResolveRowFiles:
    def test_resolves_relative_to_csv_dir(self, tmp_path):
        f1 = tmp_path / "a.yaml"
        f1.write_text("version: '0.1'\n")
        row = {"sflow_config_file": "a.yaml"}
        files = resolve_row_files(row, tmp_path, [])
        assert len(files) == 1
        assert files[0] == f1.resolve()

    def test_cli_files_prepended(self, tmp_path):
        f1 = tmp_path / "a.yaml"
        f2 = tmp_path / "b.yaml"
        f1.write_text("")
        f2.write_text("")
        row = {"sflow_config_file": "b.yaml"}
        files = resolve_row_files(row, tmp_path, [f1.resolve()])
        assert files[0] == f1.resolve()
        assert files[1] == f2.resolve()

    def test_deduplicates(self, tmp_path):
        f1 = tmp_path / "a.yaml"
        f1.write_text("")
        row = {"sflow_config_file": "a.yaml"}
        files = resolve_row_files(row, tmp_path, [f1.resolve()])
        assert len(files) == 1

    def test_multiple_csv_files(self, tmp_path):
        for name in ["a.yaml", "b.yaml", "c.yaml"]:
            (tmp_path / name).write_text("")
        row = {"sflow_config_file": "a.yaml b.yaml c.yaml"}
        files = resolve_row_files(row, tmp_path, [])
        assert len(files) == 3


class TestRowMissable:
    def test_csv_only(self):
        row = {"missable_tasks": "task_a task_b"}
        result = row_missable(row, None)
        assert result == ["task_a", "task_b"]

    def test_cli_only(self):
        row = {"missable_tasks": ""}
        result = row_missable(row, ["cli_task"])
        assert result == ["cli_task"]

    def test_merged(self):
        row = {"missable_tasks": "csv_task"}
        result = row_missable(row, ["cli_task"])
        assert "cli_task" in result
        assert "csv_task" in result

    def test_empty(self):
        row = {}
        result = row_missable(row, None)
        assert result is None

    def test_whitespace_stripped(self):
        row = {"missable_tasks": "  task_a  "}
        result = row_missable(row, None)
        assert result == ["task_a"]


class TestParseKvList:
    def test_basic(self):
        assert _parse_kv_list(["A=1", "B=2"]) == {"A": "1", "B": "2"}

    def test_none(self):
        assert _parse_kv_list(None) == {}

    def test_empty(self):
        assert _parse_kv_list([]) == {}

    def test_value_with_equals(self):
        result = _parse_kv_list(["KEY=a=b=c"])
        assert result == {"KEY": "a=b=c"}

    def test_skips_invalid(self):
        result = _parse_kv_list(["GOOD=1", "noequalssign"])
        assert result == {"GOOD": "1"}


class TestMergeRowOverrides:
    def test_cli_vars_win_over_csv(self):
        row = {"VAR1": "csv_val", "VAR2": "csv2"}
        var_cols = {"VAR1", "VAR2"}
        cli_var_map = {"VAR1": "cli_val", "EXTRA": "extra"}
        set_var, _ = merge_row_overrides(row, var_cols, set(), cli_var_map, {})
        var_map = dict(v.split("=", 1) for v in set_var)
        assert var_map["VAR1"] == "cli_val"
        assert var_map["VAR2"] == "csv2"
        assert var_map["EXTRA"] == "extra"

    def test_cli_artifacts_win_over_csv(self):
        row = {"ART1": "csv_uri"}
        art_cols = {"ART1"}
        cli_art_map = {"ART1": "cli_uri"}
        _, artifacts = merge_row_overrides(row, set(), art_cols, {}, cli_art_map)
        art_map = dict(v.split("=", 1) for v in artifacts)
        assert art_map["ART1"] == "cli_uri"

    def test_empty_csv_values_skipped(self):
        row = {"VAR1": "", "VAR2": "val2"}
        var_cols = {"VAR1", "VAR2"}
        set_var, _ = merge_row_overrides(row, var_cols, set(), {}, {})
        var_map = dict(v.split("=", 1) for v in set_var)
        assert "VAR1" not in var_map
        assert var_map["VAR2"] == "val2"

    def test_no_overrides_returns_none(self):
        row = {"VAR1": ""}
        set_var, artifacts = merge_row_overrides(row, {"VAR1"}, set(), {}, {})
        assert set_var is None
        assert artifacts is None
