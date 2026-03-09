# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for listing and copying sample workflow files.
"""

import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer

from sflow.cli import DOCS_URL, app
from sflow.logging import get_logger
from sflow.samples import get_sample_path, get_samples_dir, list_samples

_logger = get_logger(__name__)


@app.command(epilog=f"Documentation: {DOCS_URL}")
def sample(
    name: Annotated[
        Optional[str],
        typer.Argument(
            help="Name of the sample to copy (e.g., 'local_hello_world' or 'local_hello_world.yaml'). "
            "Omit to list all available samples.",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Output path for the sample file. Default: ./<sample_name>",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing file if it exists",
        ),
    ] = False,
    list_all: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="List all available samples",
        ),
    ] = False,
):
    """
    List available sample workflows or copy a sample to your project.

    Examples:
        # List all available samples
        sflow sample --list
        sflow sample

        # Copy a sample to current directory
        sflow sample local_hello_world

        # Copy with custom output path
        sflow sample local_hello_world --output my_workflow.yaml

        # Overwrite existing file
        sflow sample local_hello_world --force
    """
    # If no name provided or --list flag, show available samples
    if name is None or list_all:
        _list_samples()
        return

    # Find the sample
    sample_path = get_sample_path(name)
    if sample_path is None:
        available = list_samples()
        typer.echo(f"✗ Sample '{name}' not found.", err=True)
        typer.echo("\nAvailable samples:", err=True)
        for s in available:
            typer.echo(f"  - {s.replace('.yaml', '')}", err=True)
        raise typer.Exit(code=1)

    # Determine output path
    if output is None:
        output = Path.cwd() / sample_path.name

    # Check if output exists
    if output.exists() and not force:
        typer.echo(
            f"✗ File '{output}' already exists. Use --force to overwrite.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Copy the sample
    try:
        shutil.copy2(sample_path, output)
        typer.echo(f"✓ Sample copied to: {output}")
        typer.echo("\nYou can use it with below options:")
        if "slurm" in output.name.lower():
            typer.echo("\nValidate the workflow with:")
            typer.echo(
                f"  sflow run --file {output.name} --set SLURM_ACCOUNT=YOUR_SLURM_ACCOUNT --set SLURM_PARTITION=YOUR_SLURM_PARTITION --set SLURM_NODES=NUMBER_OF_NODES --dry-run"
            )
            typer.echo("\nRun it interactively with:")
            typer.echo(
                f"  sflow run --file {output.name} --set SLURM_ACCOUNT=YOUR_SLURM_ACCOUNT --set SLURM_PARTITION=YOUR_SLURM_PARTITION --set SLURM_NODES=NUMBER_OF_NODES --tui"
            )
            typer.echo("\nSubmit it to Slurm cluster with:")
            typer.echo(
                f"  sflow batch --file {output.name} -J sflow-job-{output.name.replace('.yaml', '')} -A YOUR_SLURM_ACCOUNT -p YOUR_SLURM_PARTITION -N NUMBER_OF_NODES -o sflow-sbatch-{output.name.replace('.yaml', '')}.sh --submit"
            )
        else:
            typer.echo("\nValidate the workflow with:")
            typer.echo(f"  sflow run --file {output.name} --dry-run")
            typer.echo("\nRun it interactively with:")
            typer.echo(f"  sflow run --file {output.name} --tui")
    except Exception as e:
        _logger.exception(f"Failed to copy sample: {e}")
        typer.echo(f"✗ Failed to copy sample: {e}", err=True)
        raise typer.Exit(code=1)


def _list_samples():
    """List all available samples with descriptions."""
    samples = list_samples()
    samples_dir = get_samples_dir()

    typer.echo("Available sample workflows:\n")

    # Group samples by category
    categories = {
        "Local": [],
        "Slurm": [],
        "Dynamo in Slurm (Disaggregated Inference)": [],
        "Other": [],
    }

    for sample in samples:
        sample_name = sample.replace(".yaml", "")
        if sample.startswith("local"):
            categories["Local"].append(sample_name)
        elif sample.startswith("slurm_") or "slurm" in sample.lower():
            categories["Slurm"].append(sample_name)
        elif sample.startswith("dynamo_"):
            categories["Dynamo in Slurm (Disaggregated Inference)"].append(sample_name)
        else:
            categories["Other"].append(sample_name)

    for category, sample_list in categories.items():
        if sample_list:
            typer.echo(f"  {category}:")
            for sample_name in sample_list:
                sample_path = samples_dir / f"{sample_name}.yaml"
                node_info = _get_sample_node_info(sample_path)
                if node_info:
                    typer.echo(f"    - {sample_name:<50} [{node_info}]")
                else:
                    typer.echo(f"    - {sample_name}")
            typer.echo()

    typer.echo("Usage:")
    typer.echo("  sflow sample <name>              # Copy sample to current directory")
    typer.echo("  sflow sample <name> -o out.yaml  # Copy with custom output path")


def _get_sample_node_info(sample_path: Path) -> str | None:
    """Extract node and GPU requirements from the sample file."""
    try:
        content = sample_path.read_text()
        nodes = None
        gpus_per_node = None

        for line in content.split("\n"):
            line_stripped = line.strip()
            # Look for SLURM_NODES variable
            if line_stripped.startswith("SLURM_NODES:"):
                # Multi-line format, look for value in next lines
                continue
            if "value:" in line_stripped and nodes is None:
                # Check if this is under SLURM_NODES by looking at context
                pass
            # Direct value extraction for SLURM_NODES
            if "SLURM_NODES:" in line and "value:" in line:
                # Inline format
                parts = line.split("value:", 1)
                if len(parts) == 2:
                    val = parts[1].strip().strip('"').strip("'")
                    if val.isdigit():
                        nodes = int(val)
            # Look for nodes: X pattern in backends section
            if line_stripped.startswith("nodes:") and "indices" not in line_stripped:
                parts = line_stripped.split(":", 1)
                if len(parts) == 2:
                    val = parts[1].strip()
                    if val.isdigit():
                        nodes = int(val)
                    elif "SLURM_NODES" in val:
                        # It's a variable reference, need to find the variable value
                        pass
            # Look for gpus_per_node
            if (
                line_stripped.startswith("gpus_per_node:")
                or "GPUS_PER_NODE:" in line_stripped
            ):
                parts = line_stripped.split(":", 1)
                if len(parts) == 2:
                    val = parts[1].strip()
                    if val.isdigit():
                        gpus_per_node = int(val)

        # Parse YAML properly for accurate extraction
        import re

        # Find SLURM_NODES value
        nodes_match = re.search(r"SLURM_NODES:[\s\S]*?value:\s*(\d+)", content)
        if nodes_match:
            nodes = int(nodes_match.group(1))

        # Find GPUS_PER_NODE value
        gpus_match = re.search(r"GPUS_PER_NODE:[\s\S]*?value:\s*(\d+)", content)
        if gpus_match:
            gpus_per_node = int(gpus_match.group(1))

        # Build info string
        if nodes is not None and gpus_per_node is not None:
            return (
                f"{nodes} node{'s, ' if nodes > 1 else ',  '}{gpus_per_node} GPUs/node"
            )
        elif nodes is not None:
            return f"{nodes} node{'s' if nodes > 1 else ''}"
        elif gpus_per_node is not None:
            return f"{gpus_per_node} GPUs/node"

        return None
    except Exception:
        return None


def _get_sample_description(sample_path: Path) -> str | None:
    """Extract a brief description from the sample file."""
    try:
        content = sample_path.read_text()
        # Look for workflow name as a simple description indicator
        for line in content.split("\n")[:20]:
            if "name:" in line and "workflow" not in line.lower():
                # Extract the name value
                parts = line.split(":", 1)
                if len(parts) == 2:
                    name = parts[1].strip().strip('"').strip("'")
                    if name and len(name) < 50:
                        return name
        return None
    except Exception:
        return None
