# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for copying AI agent skills into a project.
"""

import shutil
from pathlib import Path
from typing import Annotated, Optional

import typer

from sflow.cli import DOCS_URL, app
from sflow.logging import get_logger
from sflow.skills import get_skills_dir, list_skills

_logger = get_logger(__name__)


@app.command(epilog=f"Documentation: {DOCS_URL}")
def skill(
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Output directory for the skills. Default: ./skills",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing skills directory if it exists",
        ),
    ] = False,
    list_all: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="List all available skills",
        ),
    ] = False,
):
    """
    Copy AI agent skills into your project for Cursor/IDE agent integration.

    Skills teach AI coding agents how to write sflow YAML configs and
    diagnose sflow errors. After copying, point your IDE's agent skill
    configuration to the output directory.

    Examples:
        # List available skills
        sflow skill --list

        # Copy skills to ./skills (default)
        sflow skill

        # Copy to a custom directory
        sflow skill --output .cursor/skills

        # Overwrite existing skills
        sflow skill --force
    """
    if list_all:
        _list_skills()
        return

    if output is None:
        output = Path.cwd() / "skills"

    skills_src = get_skills_dir()

    if output.exists() and not force:
        typer.echo(
            f"✗ '{output}' already exists. Use --force to overwrite.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        if output.exists() and force:
            shutil.rmtree(output)

        _skip = {"__pycache__", ".git", "__init__.py"}
        output.mkdir(parents=True, exist_ok=True)

        copied = []
        for item in sorted(skills_src.iterdir()):
            if item.name in _skip or item.name.startswith("_"):
                continue
            dest = output / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
                copied.append(f"{item.name}/")
            elif item.is_file():
                shutil.copy2(item, dest)
                copied.append(item.name)

        typer.echo(f"✓ Skills copied to: {output}/")
        typer.echo(f"  Contents: {', '.join(copied)}")
        typer.echo()
        typer.echo("Included skills:")
        for s in list_skills():
            skill_md = skills_src / s / "SKILL.md"
            desc = _get_skill_description(skill_md)
            typer.echo(f"  - {s:<30} {desc or ''}")
        agents_md = output / "AGENTS.md"
        if agents_md.exists():
            typer.echo(f"\nAgent guidelines: {agents_md}")
        typer.echo()
        typer.echo("To use with Cursor, add the skills directory to your IDE agent skill config.")
    except Exception as e:
        _logger.exception(f"Failed to copy skills: {e}")
        typer.echo(f"✗ Failed to copy skills: {e}", err=True)
        raise typer.Exit(code=1)


def _list_skills():
    """List all available skills with descriptions."""
    skills = list_skills()
    skills_dir = get_skills_dir()

    typer.echo("Available AI agent skills:\n")
    for s in skills:
        skill_md = skills_dir / s / "SKILL.md"
        desc = _get_skill_description(skill_md)
        typer.echo(f"  - {s:<30} {desc or ''}")

    agents_md = skills_dir / "AGENTS.md"
    if agents_md.exists():
        typer.echo(f"\n  + AGENTS.md (agent workflow guidelines)")

    typer.echo()
    typer.echo("Usage:")
    typer.echo("  sflow skill                      # Copy skills to ./skills")
    typer.echo("  sflow skill -o .cursor/skills    # Copy to custom directory")
    typer.echo("  sflow skill --force              # Overwrite existing skills")


def _get_skill_description(skill_md: Path) -> str | None:
    """Extract description from SKILL.md frontmatter."""
    try:
        content = skill_md.read_text()
        in_frontmatter = False
        desc_lines = []
        collecting_desc = False
        for line in content.split("\n"):
            if line.strip() == "---":
                if in_frontmatter:
                    break
                in_frontmatter = True
                continue
            if in_frontmatter:
                if line.startswith("description:"):
                    val = line.split(":", 1)[1].strip()
                    if val and not val.startswith(">"):
                        return val.strip("'\"")
                    collecting_desc = True
                elif collecting_desc:
                    stripped = line.strip()
                    if stripped and not stripped.startswith(("-", "name:")):
                        desc_lines.append(stripped)
                    else:
                        break
        if desc_lines:
            return " ".join(desc_lines).strip()
    except Exception:
        pass
    return None
