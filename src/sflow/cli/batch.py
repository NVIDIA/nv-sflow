# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for generating sbatch scripts to run sflow in batch mode.
"""

import shlex
from pathlib import Path
from typing import Annotated, List, Optional

import typer

from sflow.app.sflow import SflowApp
from sflow.cli import DOCS_URL, app
from sflow.logging import configure_logging, get_logger

_logger = get_logger(__name__)

_sflow_app = SflowApp()


@app.command(epilog=f"Documentation: {DOCS_URL}")
def batch(
    file: Annotated[
        Path,
        typer.Option(
            "-f",
            "--file",
            help="Path to the sflow.yaml workflow file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("sflow.yaml"),
    set_var: Annotated[
        Optional[List[str]],
        typer.Option(
            "--set",
            "-s",
            help="Override variable value or domain (format: KEY=VALUE or KEY=[1,2,3] for domain). Can be used multiple times.",
        ),
    ] = None,
    artifact: Annotated[
        Optional[List[str]],
        typer.Option(
            "--artifact",
            "-a",
            help="Override artifact URI (format: NAME=URI, can be used multiple times)",
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level for sflow run (debug, info, warning, error, critical). Default: info.",
        ),
    ] = "info",
    workspace_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--workspace-dir",
            help="Workspace root directory. Default: current working directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            help="Global output root directory for sflow. Default: <workspace-dir>/sflow_output",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path.cwd() / "sflow_output",
    # sbatch options
    job_name: Annotated[
        str,
        typer.Option(
            "--job-name",
            "-J",
            help="Slurm job name",
        ),
    ] = "sflow",
    sbatch_output: Annotated[
        str,
        typer.Option(
            "--sbatch-output",
            "-O",
            help="Slurm output file pattern. Default: sflow-%j.out",
        ),
    ] = Path.cwd() / "sflow_output" / "sflow-%j.out",
    sbatch_error: Annotated[
        str,
        typer.Option(
            "--sbatch-error",
            "-E",
            help="Slurm error file pattern. Default: sflow-%j.out",
        ),
    ] = Path.cwd() / "sflow_output" / "sflow-%j.out",
    partition: Annotated[
        str,
        typer.Option(
            "--partition",
            "-p",
            help="Slurm partition (required)",
        ),
    ] = ...,
    account: Annotated[
        str,
        typer.Option(
            "--account",
            "-A",
            help="Slurm account (required)",
        ),
    ] = ...,
    time: Annotated[
        Optional[str],
        typer.Option(
            "--time",
            help="Slurm time limit (e.g., 01:00:00)",
        ),
    ] = None,
    nodes: Annotated[
        int,
        typer.Option(
            "--nodes",
            "-N",
            help="Number of nodes for sbatch (required)",
        ),
    ] = ...,
    gpus_per_node: Annotated[
        Optional[int],
        typer.Option(
            "--gpus-per-node",
            "-G",
            help="Number of GPUs per node (adds #SBATCH --gpus-per-node=N directive), not supported by all clusters, add per your cluster's documentation",
        ),
    ] = None,
    sbatch_extra_args: Annotated[
        Optional[List[str]],
        typer.Option(
            "--sbatch-extra-args",
            "-e",
            help="Additional sbatch directives to append (e.g., '--exclusive', '--segment=NUM_NODES'). Can be used multiple times, will be in script as '#SBATCH directives'.",
        ),
    ] = None,
    # runtime options
    sflow_venv_path: Annotated[
        Optional[Path],
        typer.Option(
            "--sflow-venv-path",
            "-v",
            help="Path to Python virtual environment for sflow (e.g., /path/to/.venv). "
            "The script will activate this venv before running sflow, pay extra attention to the arch of python ( x86 / arm ) when using existing venv.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path.cwd() / "sflow_compute_node_venv",
    sflow_version: Annotated[
        Optional[str],
        typer.Option(
            "--sflow-version",
            help="Version of sflow to install in the generated sflow_compute_node_venv (e.g., '0.1.0', '0.1.0.dev134'). If not specified, reuse the installed version in the existing venv, or create a new venv and install the latest version",
        ),
    ] = None,
    # output options
    sbatch_path: Annotated[
        Optional[Path],
        typer.Option(
            "--sbatch-path",
            "-o",
            help="Write the sbatch script to this file instead of stdout",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            help="Submit the job immediately after generating the script",
        ),
    ] = False,
):
    """
    Generate an sbatch script for running sflow in Slurm batch mode.

    This command creates a bash script with sbatch directives that wraps
    the 'sflow run' command for headless execution on a Slurm cluster.

    Examples:
        # Generate sbatch script to stdout
        sflow batch --file workflow.yaml

        # Generate and save to file
        sflow batch --file workflow.yaml --sbatch-path run_workflow.sh

        # Generate with Slurm options
        sflow batch --file workflow.yaml --partition gpu --time 02:00:00 --account myaccount

        # Generate with GPU allocation
        sflow batch --file workflow.yaml --nodes 2 --gpus-per-node 8

        # Generate and submit immediately
        sflow batch --file workflow.yaml --partition gpu --submit

        # With variable overrides
        sflow batch --file workflow.yaml --set NUM_GPUS=8 --set MODEL=llama

        # With custom virtual environment
        sflow batch --file workflow.yaml --sflow-venv-path /path/to/.venv

        # With extra sbatch directives
        sflow batch --file workflow.yaml --sbatch-extra-args "--exclusive" --sbatch-extra-args "--segment=NUM_NODES"
    """
    # Run dry-run validation before generating sbatch script
    typer.echo("Running dry-run validation before generating sbatch script...")
    configure_logging(level=log_level, console=True)
    try:
        _sflow_app.run(
            file=file,
            dry_run=True,
            variable_overrides=set_var,
            artifact_overrides=artifact,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
        )
        typer.echo("✓ Dry-run validation passed\n")
    except ValueError as e:
        typer.echo(f"✗ Configuration error: {e}", err=True)
        typer.echo("Aborting sbatch generation due to configuration errors.", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        typer.echo(f"✗ File not found: {e}", err=True)
        typer.echo("Aborting sbatch generation due to missing files.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"✗ Dry-run validation failed: {e}", err=True)
        typer.echo("Aborting sbatch generation due to validation errors.", err=True)
        raise typer.Exit(code=1)

    # Build the sflow run command
    sflow_cmd_parts = ["sflow", "run", "--file", shlex.quote(str(file))]

    if set_var:
        for var in set_var:
            sflow_cmd_parts.extend(["--set", shlex.quote(var)])

    if artifact:
        for art in artifact:
            sflow_cmd_parts.extend(["--artifact", shlex.quote(art)])

    if log_level != "info":
        sflow_cmd_parts.extend(["--log-level", log_level])

    if workspace_dir:
        sflow_cmd_parts.extend(["--workspace-dir", shlex.quote(str(workspace_dir))])

    if output_dir:
        sflow_cmd_parts.extend(["--output-dir", shlex.quote(str(output_dir))])

    sflow_cmd = " ".join(sflow_cmd_parts)

    # Build sbatch directives
    sbatch_directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={sbatch_output}",
        f"#SBATCH --error={sbatch_error}",
        f"#SBATCH --mem=0",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={account}",
        f"#SBATCH --nodes={nodes}",
    ]

    if time:
        sbatch_directives.append(f"#SBATCH --time={time}")

    if gpus_per_node:
        sbatch_directives.append(f"#SBATCH --gpus-per-node={gpus_per_node}")

    if sbatch_extra_args:
        for extra_arg in sbatch_extra_args:
            sbatch_directives.append(f"#SBATCH {extra_arg}")

    # Generate the script
    script_lines = [
        "#!/bin/bash",
        "#",
        "# Generated by: sflow batch",
        f"# Workflow file: {file}",
        "#",
        "",
        *sbatch_directives,
        "",
        # "# Change to the submit directory",
        # "cd $SLURM_SUBMIT_DIR",
        "set -x",
        "",
    ]

    # Add venv activation if specified
    if sflow_venv_path:
        activate_script = sflow_venv_path / ".sflow_venv" / "bin" / "activate"
        # script_lines.extend([
        #     "# Activate Python virtual environment",
        #     f"source {shlex.quote(str(activate_script))}",
        #     "",
        # ])
    else:
        activate_script = Path.cwd() / ".sflow_venv" / "bin" / "activate"

    if activate_script.exists():
        script_lines.extend([
            "# Activate existing Python virtual environment for sflow, please make sure this venv is compatible with your compute node arch of x86 / arm64",
            "# Sometimes especially in GB200 clusters, login node is x86 and compute node is arm64, so you need to create a venv for arm64",
            f"source {shlex.quote(str(activate_script))}",
            "",
        ])
        if sflow_version:
            script_lines.extend([
                f"uv pip install sflow=={sflow_version} --prerelease=allow",
            ])
    else:
        script_lines.extend([
            "# Python virtual environment activation script not found; creating from scratch and installing sflow",
            f"mkdir -p {shlex.quote(str(Path(activate_script).resolve().parent.parent.parent))}",
            f"cd {shlex.quote(str(Path(activate_script).resolve().parent.parent.parent))}",
            "",
            "# # By default we use compute node's python to create venv and install uv",
            "# # This is because sometimes especially in GB200 / GB300 clusters, login node is x86 and compute node is arm64, so you need to create a venv for arm64",
            "/usr/bin/python3 -m venv .sflow_venv",
            "source .sflow_venv/bin/activate",
            "pip install uv",
            "uv --version",
            "which uv",
            "",
            "# # You can uncomment below to fall back to sh way if your meet issues with pip install uv in your cluster",
            "# curl -LsSf https://astral.sh/uv/install.sh | sh",
            "# uv --version",
            "# which uv",
            "# uv venv .sflow_venv --python python3",
            "# source .sflow_venv/bin/activate",
            "",
            f"uv pip install sflow{'=='+sflow_version if sflow_version else ' -U'} --prerelease=allow",
            "sflow --help",
            "",
        ])

    script_lines.extend([
        f"cd {workspace_dir}",
        # "# Run sflow workflow dry-run to validate configuration and fail early if configuration error is detected",
        # f"{sflow_cmd} --dry-run | grep -q 'Configuration error' && (echo 'Detected Configuration error in dry run, aborting.'; exit 1;) || echo 'Configuration validated successfully, continuing.'",
        "",
        "# Run sflow workflow",
        shlex.quote(str(Path(activate_script).resolve().parent)) + "/" + sflow_cmd,
        "",
    ])

    script_content = "\n".join(script_lines)

    if sbatch_path:
        # Write to file
        sbatch_path.write_text(script_content)
        sbatch_path.chmod(0o755)
        typer.echo(script_content)
        typer.echo(f"✓ Sbatch script written to: {sbatch_path}")

        if submit:
            import subprocess

            result = subprocess.run(
                ["sbatch", str(sbatch_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                typer.echo(f"✓ Job submitted: {result.stdout.strip()}")
            else:
                typer.echo(f"✗ Failed to submit job: {result.stderr.strip()}", err=True)
                raise typer.Exit(code=1)
    else:
        # Print to stdout
        typer.echo(script_content)

        if submit:
            typer.echo(
                "⚠ Cannot submit without --script-output. "
                "Please specify a file to save the script first.",
                err=True,
            )
            raise typer.Exit(code=1)
