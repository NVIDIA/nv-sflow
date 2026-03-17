# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml
from typer.testing import CliRunner

from sflow.cli import app

runner = CliRunner()


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.dump(data, sort_keys=False))
    return path


def test_compose_two_files_outputs_valid_yaml_to_stdout(tmp_path: Path):
    f1 = _write_yaml(
        tmp_path / "backends.yaml",
        {
            "version": "0.1",
            "backends": [
                {
                    "name": "slurm_cluster",
                    "type": "slurm",
                    "default": True,
                    "account": "acct",
                    "partition": "batch",
                    "time": "00:10:00",
                    "nodes": 1,
                    "gpus_per_node": 4,
                }
            ],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo hi"]}],
            },
        },
    )

    result = runner.invoke(app, ["compose", str(f1), str(f2)], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    merged = yaml.safe_load(result.output)
    assert merged["version"] == "0.1"
    assert merged["workflow"]["name"] == "wf"
    assert any(b["name"] == "slurm_cluster" for b in merged["backends"])
    assert len(merged["workflow"]["tasks"]) == 1


def test_compose_writes_to_output_file(tmp_path: Path):
    f1 = _write_yaml(
        tmp_path / "a.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo 1"]}],
            },
        },
    )
    f2 = _write_yaml(
        tmp_path / "b.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t2", "script": ["echo 2"]}],
            },
        },
    )
    out = tmp_path / "merged.yaml"
    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "-o", str(out)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    merged = yaml.safe_load(out.read_text())
    task_names = [t["name"] for t in merged["workflow"]["tasks"]]
    assert task_names == ["t1", "t2"]


def test_compose_applies_set_overrides_and_resolves(tmp_path: Path):
    f1 = _write_yaml(
        tmp_path / "a.yaml",
        {
            "version": "0.1",
            "variables": [{"name": "X", "value": 1}],
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t1",
                        "script": ["echo ${{ variables.X }}"],
                    }
                ],
            },
        },
    )
    f2 = _write_yaml(tmp_path / "b.yaml", {"version": "0.1"})

    result = runner.invoke(
        app,
        ["compose", str(f1), str(f2), "--set", "X=42", "--resolve"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    merged = yaml.safe_load(result.output)
    assert "variables" not in merged, "X should be resolved and removed"
    assert merged["workflow"]["tasks"][0]["script"] == ["echo 42"]


def test_compose_accepts_single_file(tmp_path: Path):
    f = _write_yaml(
        tmp_path / "solo.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo hi"]}],
            },
        },
    )

    result = runner.invoke(app, ["compose", str(f)])
    assert result.exit_code == 0


def test_compose_rejects_invalid_merged_config(tmp_path: Path):
    f1 = _write_yaml(tmp_path / "a.yaml", {"version": "0.1"})
    f2 = _write_yaml(tmp_path / "b.yaml", {"version": "0.1"})

    result = runner.invoke(app, ["compose", str(f1), str(f2)])
    assert result.exit_code == 1


def test_compose_real_disagg_example_files(tmp_path: Path):
    """Compose the real disagg example files and validate the output is loadable."""
    repo_root = Path(__file__).resolve().parents[2]
    examples_dir = repo_root / "examples" / "inference_x_v2"
    if not examples_dir.exists():
        return

    candidates = sorted(examples_dir.glob("*.yaml"))
    if len(candidates) < 2:
        return

    out = tmp_path / "composed.yaml"
    args = ["compose"] + [str(f) for f in candidates] + ["-o", str(out)]
    result = runner.invoke(app, args)

    if result.exit_code == 0:
        merged = yaml.safe_load(out.read_text())
        assert "version" in merged
        assert "workflow" in merged


# ---------------------------------------------------------------------------
# --resolve on/off tests
# ---------------------------------------------------------------------------


def _make_resolve_test_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create two YAML files with a variable and an expression referencing it."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [{"name": "TP", "value": 4}],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t1",
                        "script": [
                            "run --tp ${{ variables.TP }}",
                            "echo ${TP}",
                        ],
                    }
                ],
            },
        },
    )
    return f1, f2


def test_compose_without_resolve_keeps_variables(tmp_path: Path):
    """Without --resolve, variables and expressions stay as-is."""
    f1, f2 = _make_resolve_test_files(tmp_path)

    result = runner.invoke(app, ["compose", str(f1), str(f2)], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    assert "variables" in composed, "Variables section must be kept without --resolve"
    var_names = [v["name"] for v in composed["variables"]]
    assert "TP" in var_names
    scripts = composed["workflow"]["tasks"][0]["script"]
    assert "${{ variables.TP }}" in scripts[0], "Expression must not be resolved"
    assert "${TP}" in scripts[1], "Shell ref must not be resolved"


def test_compose_with_resolve_inlines_variables(tmp_path: Path):
    """With --resolve, variables are inlined and removed."""
    f1, f2 = _make_resolve_test_files(tmp_path)

    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "--resolve"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    assert "variables" not in composed, "Variables should be resolved and removed"
    scripts = composed["workflow"]["tasks"][0]["script"]
    assert scripts[0] == "run --tp 4", "Expression must be resolved to literal"
    assert scripts[1] == "echo 4", "Shell ref must be resolved to literal"


# ---------------------------------------------------------------------------
# Variable resolution tests (all use --resolve)
# ---------------------------------------------------------------------------


def test_compose_resolves_variable_expressions_inline(tmp_path: Path):
    """${{ variables.X }} expressions are replaced with literal values."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [
                {"name": "TP_SIZE", "value": 4},
                {"name": "MODEL", "value": "llama"},
            ],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "server",
                        "script": ["launch --tp ${{ variables.TP_SIZE }}"],
                        "operator": {
                            "name": "op",
                            "ntasks": "${{ variables.TP_SIZE }}",
                        },
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "--resolve"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    assert "variables" not in composed
    task = composed["workflow"]["tasks"][0]
    assert task["operator"]["ntasks"] == 4
    assert task["script"] == ["launch --tp 4"]


def test_compose_resolves_arithmetic_expressions(tmp_path: Path):
    """${{ variables.A * variables.B }} is computed and inlined."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [
                {"name": "TP", "value": 2},
                {"name": "DP", "value": 4},
            ],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t1",
                        "script": ["echo gpus=${{ variables.TP * variables.DP }}"],
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "--resolve"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    assert composed["workflow"]["tasks"][0]["script"] == ["echo gpus=8"]


def test_compose_keeps_backend_dependent_expressions(tmp_path: Path):
    """Expressions referencing backends.* are preserved (need allocation)."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [
                {"name": "NODES", "value": 2},
            ],
            "backends": [
                {
                    "name": "slurm_cluster",
                    "type": "slurm",
                    "default": True,
                    "account": "acct",
                    "partition": "batch",
                    "time": "00:10:00",
                    "nodes": "${{ variables.NODES }}",
                    "gpus_per_node": 4,
                }
            ],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "variables": [
                    {
                        "name": "HEAD_IP",
                        "value": "${{ backends.slurm_cluster.nodes[0].ip_address }}",
                    }
                ],
                "tasks": [
                    {
                        "name": "t1",
                        "script": [
                            "echo server at ${HEAD_IP}",
                        ],
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "--resolve"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    # NODES is resolved (literal 2), so backends.nodes becomes literal 2
    backend = next(b for b in composed["backends"] if b["name"] == "slurm_cluster")
    assert backend["nodes"] == 2
    # HEAD_IP depends on backends, so it must be kept
    wf_vars = composed["workflow"]["variables"]
    head_ip = next(v for v in wf_vars if v["name"] == "HEAD_IP")
    assert "${{ backends.slurm_cluster.nodes[0].ip_address }}" in str(head_ip["value"])
    # ${HEAD_IP} in scripts should NOT be resolved (it's unresolvable)
    assert composed["workflow"]["tasks"][0]["script"] == ["echo server at ${HEAD_IP}"]


def test_compose_resolves_shell_variable_refs_in_scripts(tmp_path: Path):
    """${NAME} shell references in scripts are inlined for resolved variables."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [
                {"name": "MODEL_NAME", "value": "llama-7b"},
                {"name": "PORT", "value": 8000},
            ],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t1",
                        "script": [
                            "serve --model ${MODEL_NAME} --port ${PORT}",
                            "echo ${CUDA_VISIBLE_DEVICES}",
                        ],
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "--resolve"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    scripts = composed["workflow"]["tasks"][0]["script"]
    assert scripts[0] == "serve --model llama-7b --port 8000"
    assert scripts[1] == "echo ${CUDA_VISIBLE_DEVICES}", "Built-in env vars stay as-is"


def test_compose_resolves_chained_variables(tmp_path: Path):
    """Variable B referencing variable A is resolved transitively."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [
                {"name": "BASE", "value": 4},
                {"name": "DOUBLED", "value": "${{ variables.BASE * 2 }}"},
            ],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t1",
                        "script": ["echo ${{ variables.DOUBLED }}"],
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "--resolve"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    assert "variables" not in composed
    assert composed["workflow"]["tasks"][0]["script"] == ["echo 8"]


def test_compose_mixed_expression_kept_if_unresolvable(tmp_path: Path):
    """A string mixing resolvable and unresolvable expressions is kept as-is."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "backends": [
                {
                    "name": "slurm_cluster",
                    "type": "slurm",
                    "default": True,
                    "account": "acct",
                    "partition": "batch",
                    "time": "00:10:00",
                    "nodes": 1,
                    "gpus_per_node": 4,
                }
            ],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "variables": [
                    {
                        "name": "NATS_URL",
                        "value": "nats://${{ backends.slurm_cluster.nodes[0].ip_address }}:4222",
                    }
                ],
                "tasks": [{"name": "t1", "script": ["echo hi"]}],
            },
        },
    )

    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "--resolve"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    nats = next(v for v in composed["workflow"]["variables"] if v["name"] == "NATS_URL")
    assert "backends.slurm_cluster.nodes[0].ip_address" in nats["value"]


def test_compose_keeps_replica_sweep_variables(tmp_path: Path):
    """Variables listed in replicas.variables must stay in the variables section
    and must NOT be resolved in expressions or shell refs."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [
                {"name": "CONCURRENCY", "value": 64, "domain": [64, 128]},
                {"name": "OTHER", "value": 99},
            ],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "bench",
                        "script": [
                            "run --concurrency ${CONCURRENCY}",
                            "echo ${{ variables.CONCURRENCY }}",
                        ],
                        "replicas": {
                            "variables": ["CONCURRENCY"],
                            "policy": "sequential",
                        },
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "--resolve"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    var_names = [v["name"] for v in composed.get("variables", [])]
    assert "CONCURRENCY" in var_names, "Replica sweep variable must be preserved"
    assert "OTHER" not in var_names, "Non-replica variable should be removed"
    scripts = composed["workflow"]["tasks"][0]["script"]
    assert scripts[0] == "run --concurrency ${CONCURRENCY}", (
        "${CONCURRENCY} must not be inlined"
    )
    assert "${{ variables.CONCURRENCY }}" in scripts[1], (
        "expression ref must not be inlined"
    )


def test_compose_removes_unused_literal_variables(tmp_path: Path):
    """Variables with literal values that are not referenced anywhere are removed."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [
                {"name": "UNUSED_VAR", "value": 42},
            ],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo hi"]}],
            },
        },
    )

    result = runner.invoke(
        app, ["compose", str(f1), str(f2), "--resolve"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output

    composed = yaml.safe_load(result.output)
    assert "variables" not in composed


# ---------------------------------------------------------------------------
# Compose --validate tests
# ---------------------------------------------------------------------------


def test_compose_validate_warns_on_invalid_config(tmp_path: Path):
    """--validate prints a WARNING when dry-run fails (single file mode)."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [{"name": "X", "value": 1}],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo hi"]}],
            },
        },
    )

    from unittest.mock import MagicMock, patch

    with patch("sflow.app.sflow.SflowApp") as MockApp:
        MockApp.return_value.run = MagicMock(
            side_effect=ValueError("GPU over-subscription")
        )
        result = runner.invoke(
            app,
            ["compose", str(f1), str(f2), "--validate"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert "WARNING" in (result.output + (result.stderr or ""))


def test_compose_bulk_validate_shows_warnings_at_end(tmp_path: Path):
    """Bulk compose --validate shows a warning block at the end for failed rows."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [{"name": "TP", "value": 2}],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo ${{ variables.TP }}"]}],
            },
        },
    )
    out_dir = tmp_path / "output"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,TP\n{f1} {f2},4\n{f1} {f2},8\n")

    from unittest.mock import MagicMock, patch

    call_count = 0

    def _fail_second(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ValueError("not enough GPUs")

    with patch("sflow.app.sflow.SflowApp") as MockApp:
        MockApp.return_value.run = MagicMock(side_effect=_fail_second)
        result = runner.invoke(
            app,
            [
                "compose",
                "--bulk-input",
                str(csv_file),
                "--validate",
                "-o",
                str(out_dir),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    output = result.output + (result.stderr or "")
    assert "WARNINGS: 1 config(s) failed dry-run validation:" in output
    assert "not enough GPUs" in output
    assert "====" in output

    composed_files = sorted(out_dir.rglob("*.yaml"))
    assert len(composed_files) == 2, "Both configs should still be generated"


def test_compose_bulk_without_validate_no_warnings(tmp_path: Path):
    """Without --validate, no dry-run warnings appear."""
    f1 = _write_yaml(
        tmp_path / "vars.yaml",
        {
            "version": "0.1",
            "variables": [{"name": "TP", "value": 2}],
        },
    )
    f2 = _write_yaml(
        tmp_path / "workflow.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo ${{ variables.TP }}"]}],
            },
        },
    )
    out_dir = tmp_path / "output"
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(f"sflow_config_file,TP\n{f1} {f2},4\n")

    result = runner.invoke(
        app,
        ["compose", "--bulk-input", str(csv_file), "-o", str(out_dir)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "WARNINGS" not in result.output
    assert "dry-run" not in result.output.lower()


# ---------------------------------------------------------------------------
# --missable-tasks tests for compose
# ---------------------------------------------------------------------------


def test_compose_missable_tasks_rejected_with_single_file(tmp_path: Path):
    """--missable-tasks should error with a single compose file."""
    f = _write_yaml(
        tmp_path / "wf.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo hi"]}],
            },
        },
    )

    result = runner.invoke(
        app, ["compose", str(f), "--missable-tasks", "missing"]
    )
    assert result.exit_code == 1
    assert "multiple input files" in result.output


def test_compose_missable_tasks_removes_absent_dependency(tmp_path: Path):
    """--missable-tasks should remove absent tasks from depends_on."""
    f1 = _write_yaml(
        tmp_path / "base.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo base"]}],
            },
        },
    )
    f2 = _write_yaml(
        tmp_path / "extra.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t2",
                        "depends_on": ["t1", "absent_task"],
                        "script": ["echo extra"],
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        app,
        ["compose", str(f1), str(f2), "-M", "absent_task"],
    )
    assert result.exit_code == 0
    assert "absent_task" not in result.output.split("version:")[1]


def test_compose_missable_tasks_glob_pattern(tmp_path: Path):
    """--missable-tasks should support glob patterns."""
    f1 = _write_yaml(
        tmp_path / "base.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo base"]}],
            },
        },
    )
    f2 = _write_yaml(
        tmp_path / "extra.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "benchmark",
                        "depends_on": ["t1", "prefill_server", "decode_server"],
                        "script": ["echo bench"],
                    }
                ],
            },
        },
    )

    result = runner.invoke(
        app,
        ["compose", str(f1), str(f2), "-M", "prefill_*", "-M", "decode_*"],
    )
    assert result.exit_code == 0
    output_yaml = result.output.split("version:")[1] if "version:" in result.output else ""
    assert "prefill_server" not in output_yaml
    assert "decode_server" not in output_yaml


def test_compose_missable_tasks_short_flag(tmp_path: Path):
    """-M short flag should work for compose --missable-tasks."""
    f1 = _write_yaml(
        tmp_path / "base.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo base"]}],
            },
        },
    )
    f2 = _write_yaml(
        tmp_path / "extra.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t2",
                        "depends_on": ["t1", "gone"],
                        "script": ["echo extra"],
                    }
                ],
            },
        },
    )

    result = runner.invoke(app, ["compose", str(f1), str(f2), "-M", "gone"])
    assert result.exit_code == 0


def test_compose_bulk_input_missable_csv_column(tmp_path: Path):
    """missable_tasks CSV column should work in compose --bulk-input."""
    f_base = _write_yaml(
        tmp_path / "base.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo base"]}],
            },
        },
    )
    f_extra = _write_yaml(
        tmp_path / "extra.yaml",
        {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {
                        "name": "t2",
                        "depends_on": ["t1", "missing_task"],
                        "script": ["echo extra"],
                    }
                ],
            },
        },
    )
    csv_file = tmp_path / "jobs.csv"
    csv_file.write_text(
        f"sflow_config_file,missable_tasks\n"
        f"{f_base} {f_extra},missing_task\n"
    )
    out_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        ["compose", "--bulk-input", str(csv_file), "-o", str(out_dir)],
    )
    assert result.exit_code == 0
