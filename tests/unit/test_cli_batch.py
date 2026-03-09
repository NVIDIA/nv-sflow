# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sflow batch CLI command."""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path

from sflow.cli import app


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
            "--file", str(temp_workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "1",
            "--sbatch-path", str(sbatch_path),
            "--sbatch-extra-args", "--exclusive",
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
            "--file", str(temp_workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "2",
            "--sbatch-path", str(sbatch_path),
            "--sbatch-extra-args", "--exclusive",
            "--sbatch-extra-args", "--segment=2",
            "--sbatch-extra-args", "--constraint=gpu",
        ],
    )
    
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    
    script_content = sbatch_path.read_text()
    assert "#SBATCH --exclusive" in script_content
    assert "#SBATCH --segment=2" in script_content
    assert "#SBATCH --constraint=gpu" in script_content


def test_batch_sbatch_extra_args_short_option(mock_sflow_app, temp_workflow_file, tmp_path):
    """Test that -e short option works for --sbatch-extra-args."""
    sbatch_path = tmp_path / "test.sh"
    
    result = runner.invoke(
        app,
        [
            "batch",
            "--file", str(temp_workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "1",
            "--sbatch-path", str(sbatch_path),
            "-e", "--exclusive",
            "-e", "--segment=1",
        ],
    )
    
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    
    script_content = sbatch_path.read_text()
    assert "#SBATCH --exclusive" in script_content
    assert "#SBATCH --segment=1" in script_content


def test_batch_sbatch_extra_args_preserves_value(mock_sflow_app, temp_workflow_file, tmp_path):
    """Test that sbatch-extra-args preserves the exact value."""
    sbatch_path = tmp_path / "test.sh"
    
    result = runner.invoke(
        app,
        [
            "batch",
            "--file", str(temp_workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "4",
            "--sbatch-path", str(sbatch_path),
            "--sbatch-extra-args", "--gres=gpu:8",
            "--sbatch-extra-args", "--mem-per-cpu=4G",
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
            "--file", str(temp_workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "1",
            "--sbatch-path", str(sbatch_path),
        ],
    )
    
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    
    script_content = sbatch_path.read_text()
    # Standard directives should be present
    assert "#SBATCH --partition=batch" in script_content
    assert "#SBATCH --account=testaccount" in script_content
    assert "#SBATCH --nodes=1" in script_content
    # Extra args should not be present
    assert script_content.count("#SBATCH") == 7  # job-name, output, error, mem, partition, account, nodes


def test_batch_sbatch_extra_args_order_preserved(mock_sflow_app, temp_workflow_file, tmp_path):
    """Test that sbatch-extra-args are appended after standard directives."""
    sbatch_path = tmp_path / "test.sh"
    
    result = runner.invoke(
        app,
        [
            "batch",
            "--file", str(temp_workflow_file),
            "--partition", "batch",
            "--account", "testaccount",
            "--nodes", "2",
            "--time", "01:00:00",
            "--sbatch-path", str(sbatch_path),
            "--sbatch-extra-args", "--exclusive",
        ],
    )
    
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    
    script_content = sbatch_path.read_text()
    
    # Check that --exclusive comes after --time (order is preserved)
    time_pos = script_content.find("#SBATCH --time=01:00:00")
    exclusive_pos = script_content.find("#SBATCH --exclusive")
    assert time_pos < exclusive_pos, "Extra args should come after standard directives"
