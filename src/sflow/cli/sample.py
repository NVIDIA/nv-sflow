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
from sflow.samples import (
    get_sample_path,
    get_samples_dir,
    list_modular_samples,
    list_samples,
)

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
            f"✗ '{output}' already exists. Use --force to overwrite.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Copy the sample (file or directory)
    try:
        if sample_path.is_dir():
            if output.exists() and force:
                shutil.rmtree(output)
            shutil.copytree(
                sample_path, output,
                ignore=shutil.ignore_patterns("__init__.py", "__pycache__"),
            )
            typer.echo(f"✓ Modular sample folder copied to: {output}/")
            yaml_files = sorted(f.name for f in output.glob("*.yaml"))
            _skip_dirs = {"__pycache__", "sflow_output", ".git"}
            sub_dirs = sorted(
                d.name
                for d in output.iterdir()
                if d.is_dir()
                and not d.name.startswith("_")
                and d.name not in _skip_dirs
            )
            if yaml_files or sub_dirs:
                parts = [f.replace(".yaml", "") for f in yaml_files]
                parts.extend(f"{d}/" for d in sub_dirs)
                typer.echo(f"  Contains: {', '.join(parts)}")
            csv_files = sorted(f.name for f in output.glob("*.csv"))
            # Detect framework subdirectories for concrete compose examples
            _skip_dirs2 = {"__pycache__", "sflow_output", ".git"}
            subdirs = sorted(
                d.name
                for d in output.iterdir()
                if d.is_dir()
                and not d.name.startswith("_")
                and d.name not in _skip_dirs2
            )
            has_common = (output / "slurm_config.yaml").exists() and (
                output / "common_workflow.yaml"
            ).exists()

            typer.echo("\n" + "=" * 65)
            typer.echo("  Option A: Bulk batch (CSV-driven, all-in-one)")
            typer.echo("=" * 65)
            if csv_files:
                typer.echo("\n  Preview scripts (no submission):")
                typer.echo(
                    f"    sflow batch --bulk-input {output.name}/{csv_files[0]} "
                    f"-A ACCOUNT -p PARTITION"
                )
                typer.echo("\n  Generate and submit to Slurm:")
                typer.echo(
                    f"    sflow batch --bulk-input {output.name}/{csv_files[0]} "
                    f"-A ACCOUNT -p PARTITION --submit"
                )
                typer.echo(
                    "\n  Add --resolve to inline all variables into the generated configs:"
                )
                typer.echo(
                    f"    sflow batch --bulk-input {output.name}/{csv_files[0]} "
                    f"-A ACCOUNT -p PARTITION --resolve"
                )

            if has_common and subdirs:
                fw = "trtllm" if "trtllm" in subdirs else subdirs[0]
                typer.echo(f"\n{'=' * 65}")
                typer.echo("  Option B: Compose + Submit (step-by-step)")
                typer.echo("=" * 65)
                typer.echo(
                    "\n  Step 1 - Compose modular files into a complete workflow:"
                )
                typer.echo(
                    f"    sflow compose {output.name}/slurm_config.yaml "
                    f"{output.name}/common_workflow.yaml \\"
                )
                typer.echo(
                    f"                  {output.name}/{fw}/prefill.yaml "
                    f"{output.name}/{fw}/decode.yaml \\"
                )
                typer.echo(
                    f"                  {output.name}/benchmark_aiperf.yaml "
                    f"-o composed.yaml"
                )
                typer.echo("\n  Step 2 - Validate, run, or submit:")
                typer.echo(
                    "    sflow run -f composed.yaml --dry-run                        # validate"
                )
                typer.echo(
                    "    sflow run -f composed.yaml --tui                             # run interactively"
                )
                typer.echo(
                    "    sflow batch -f composed.yaml -N 1 -p PARTITION -A ACCOUNT \\"
                )
                typer.echo(
                    "                -o run.sh --submit                               # submit to Slurm"
                )

            typer.echo(f"\n{'=' * 65}")
            typer.echo("  Tip: --resolve")
            typer.echo("=" * 65)
            typer.echo(
                "\n  By default, composed configs keep ${{ variables.* }} expressions"
            )
            typer.echo("  so you can easily override values with --set at run time.")
            typer.echo(
                "\n  Add --resolve to inline all resolvable variables into literal"
            )
            typer.echo("  values, producing a plain-text config with no expressions.")
            typer.echo("  Use this when you want a self-contained, fully-baked recipe.")
            typer.echo("\n    sflow compose ... --resolve -o resolved.yaml")

            typer.echo(f"\n{'=' * 65}")
            typer.echo("  Tip: Model path")
            typer.echo("=" * 65)
            typer.echo(
                "\n  Sample configs use a placeholder model path (LOCAL_MODEL_PATH)."
            )
            typer.echo("  Override it with --artifact to point to your actual model:")
            typer.echo("\n    -a LOCAL_MODEL_PATH=fs:///path/to/your/model")
            typer.echo("\n  Example:")
            typer.echo(
                f"    sflow batch --bulk-input {output.name}/{csv_files[0] if csv_files else 'bulk_input.csv'} \\"
            )
            typer.echo(
                "                -a LOCAL_MODEL_PATH=fs:///data/models/Llama-3.1-8B \\"
            )
            typer.echo("                -A ACCOUNT -p PARTITION --submit")
        else:
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
    modular = list_modular_samples()
    samples_dir = get_samples_dir()

    typer.echo("Available sample workflows:\n")

    # Group top-level samples by category
    categories = {
        "Local": [],
        "Slurm (self-contained)": [],
        "Other": [],
    }

    for sample in samples:
        sample_name = sample.replace(".yaml", "")
        if sample.startswith("local"):
            categories["Local"].append(sample_name)
        elif sample.startswith("slurm_") or "slurm" in sample.lower():
            categories["Slurm (self-contained)"].append(sample_name)
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

    # Show modular samples (subdirectories)
    if modular:
        typer.echo("  Modular (compose multiple files per workflow):")
        for folder, yamls in modular.items():
            file_list = ", ".join(y.replace(".yaml", "") for y in yamls)
            typer.echo(f"    - {folder + '/':<50} [{file_list}]")
        typer.echo()
        typer.echo("  Modular workflow:")
        typer.echo()
        typer.echo(
            "    ┌─────────────────────────────────────────────────────────────┐"
        )
        typer.echo(
            "    │  Step 1: Compose modular YAML files into complete configs   │"
        )
        typer.echo(
            "    │                                                             │"
        )
        typer.echo(
            "    │    sflow compose <folder>/base.yaml <folder>/task.yaml \\     │"
        )
        typer.echo(
            "    │                  --resolve -o composed.yaml                 │"
        )
        typer.echo(
            "    │                                                             │"
        )
        typer.echo(
            "    │  Or bulk compose via CSV:                                   │"
        )
        typer.echo(
            "    │    sflow compose --bulk-input <folder>/bulk_input.csv \\      │"
        )
        typer.echo(
            "    │                  --resolve -o output_dir/                   │"
        )
        typer.echo(
            "    ├─────────────────────────────────────────────────────────────┤"
        )
        typer.echo(
            "    │                          ↓                                  │"
        )
        typer.echo(
            "    ├─────────────────────────────────────────────────────────────┤"
        )
        typer.echo(
            "    │  Step 2: Submit composed configs to Slurm                   │"
        )
        typer.echo(
            "    │                                                             │"
        )
        typer.echo(
            "    │    sflow batch --bulk-submit output_dir/ \\                   │"
        )
        typer.echo(
            "    │                -p PARTITION -A ACCOUNT --submit              │"
        )
        typer.echo(
            "    │                                                             │"
        )
        typer.echo(
            "    │  Or directly from modular files (compose + batch in one):    │"
        )
        typer.echo(
            "    │    sflow batch --bulk-input <folder>/bulk_input.csv \\         │"
        )
        typer.echo(
            "    │                -p PARTITION -A ACCOUNT --submit              │"
        )
        typer.echo(
            "    └─────────────────────────────────────────────────────────────┘"
        )
        typer.echo()

    typer.echo("Usage:")
    typer.echo("  sflow sample <name>              # Copy a self-contained sample")
    typer.echo("  sflow sample <folder>             # Copy a modular sample folder")
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
