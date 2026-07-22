# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for copying AI agent skills into a project.
"""

import shutil
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from sflow.cli import DOCS_URL, app
from sflow.logging import get_logger
from sflow.skills import get_skills_dir, list_skills

_logger = get_logger(__name__)

_SKIP = {"__pycache__", ".git", "__init__.py"}


class SkillTarget(str, Enum):
    """A coding agent whose skills folder `sflow skill --target` installs into."""

    claude = "claude"
    cursor = "cursor"
    codex = "codex"
    all = "all"


# Each agent's skills folder, relative to the project root (or to $HOME with
# --global). All three auto-discover SKILL.md skill folders dropped in here.
_TARGET_SUBDIR: dict[str, Path] = {
    "claude": Path(".claude") / "skills",  # Claude Code (also ~/.claude/skills)
    "cursor": Path(".cursor") / "skills",  # Cursor (project-scoped only)
    "codex": Path(".codex") / "skills",  # Codex CLI (also ~/.codex/skills)
}
# Agents that also support a user-level (global) skills dir. Cursor is project-only.
_GLOBAL_SUPPORTED = {"claude", "codex"}


@app.command(epilog=f"Documentation: {DOCS_URL}")
def skill(
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Copy skills to a custom directory (prompts first). Default: ./skills",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    target: Annotated[
        Optional[list[SkillTarget]],
        typer.Option(
            "-t",
            "--target",
            help="Install straight into an agent's skills folder (no prompt): "
            "claude, cursor, codex, or all. Repeatable.",
        ),
    ] = None,
    install_global: Annotated[
        bool,
        typer.Option(
            "--global",
            "-g",
            help="With --target, install into the user-level (~) skills dir "
            "instead of the project. Cursor is project-only.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing files when merging into an existing directory",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the confirmation prompt (implied by --target)",
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
    Install AI agent skills for Claude Code, Cursor, and Codex.

    Skills teach coding agents how to write sflow YAML configs, preflight and
    debug workflows, and review sflow changes. Claude Code, Cursor, and Codex
    all auto-discover skills from a per-tool folder, so installing there is
    enough — the agent loads a skill when your request matches its description.

    Use --target to install straight into an agent's folder with no prompt:

    \b
      claude  ->  .claude/skills    (or ~/.claude/skills with --global)
      cursor  ->  .cursor/skills    (project-only; Cursor has no global dir)
      codex   ->  .codex/skills     (or ~/.codex/skills with --global)

    Without --target, skills are copied to ./skills (or -o DIR) after a prompt.
    Either way the copy MERGES: existing files and other skills are preserved;
    use --force to overwrite.

    Examples:
        # List available skills
        sflow skill --list

        # Install into every agent's project skills folder (no prompt)
        sflow skill --target all

        # Just Claude Code + Cursor
        sflow skill -t claude -t cursor

        # Install into Claude Code's user-level folder (all projects)
        sflow skill -t claude --global

        # Copy to a custom directory (prompts first)
        sflow skill -o vendor/skills
    """
    if list_all:
        _list_skills()
        return

    skills_src = get_skills_dir()
    available = [
        s
        for s in sorted(skills_src.iterdir())
        if s.is_dir()
        and not s.name.startswith("_")
        and s.name not in {"__pycache__", ".git"}
    ]

    # Direct install into one or more agent skill folders -- no prompt.
    if target:
        if output is not None:
            typer.echo("✗ Use either --target or --output, not both.", err=True)
            raise typer.Exit(code=1)
        _install_to_targets(
            skills_src, _expand_targets(target), install_global=install_global, force=force
        )
        return

    # Legacy path: copy to -o / ./skills, with a confirmation prompt.
    if output is None:
        output = Path.cwd() / "skills"

    typer.echo(f"Skills will be copied to: {output}/")
    typer.echo(f"  Skills: {', '.join(s.name for s in available)}")
    if output.exists():
        typer.echo(
            f"  Note: directory already exists — files will be merged{' (existing files will be overwritten)' if force else ' (existing files will be preserved)'}."
        )
    if not yes and not typer.confirm("Proceed?", default=True):
        raise typer.Abort()

    try:
        copied = _install_skills(skills_src, output, force=force)
        typer.echo(f"✓ Skills copied to: {output}/")
        typer.echo(f"  Contents: {', '.join(copied)}")
        _print_included(skills_src)
        agents_md = output / "AGENTS.md"
        if agents_md.exists():
            typer.echo(f"\nAgent guidelines: {agents_md}")
        typer.echo()
        typer.echo(
            "Tip: install straight into an agent's auto-discovered folder with "
            "`sflow skill --target all`."
        )
    except Exception as e:
        _logger.exception(f"Failed to copy skills: {e}")
        typer.echo(f"✗ Failed to copy skills: {e}", err=True)
        raise typer.Exit(code=1)


def _expand_targets(targets: list[SkillTarget]) -> list[str]:
    """Normalize the --target list: expand 'all', dedupe, keep order."""
    names = [t.value for t in targets]
    if "all" in names:
        return ["claude", "cursor", "codex"]
    ordered: list[str] = []
    for n in names:
        if n not in ordered:
            ordered.append(n)
    return ordered


def _resolve_target_dir(target: str, *, install_global: bool) -> Path:
    """Map an agent name to its skills folder (project, or ~ with --global)."""
    subdir = _TARGET_SUBDIR[target]
    if install_global:
        if target in _GLOBAL_SUPPORTED:
            return Path.home() / subdir
        typer.echo(
            f"  Note: {target} has no user-level skills dir; installing into the project."
        )
    return Path.cwd() / subdir


def _install_to_targets(
    skills_src: Path, names: list[str], *, install_global: bool, force: bool
) -> None:
    """Install skills into each resolved agent folder with no prompt."""
    try:
        for name in names:
            dest = _resolve_target_dir(name, install_global=install_global)
            existed = dest.exists()
            _install_skills(skills_src, dest, force=force)
            note = ""
            if existed:
                note = (
                    " (merged, existing overwritten)"
                    if force
                    else " (merged, existing preserved)"
                )
            typer.echo(f"✓ {name:<7} → {dest}{note}")
        _print_included(skills_src)
        typer.echo()
        typer.echo(
            "These folders are auto-discovered: the agent loads a skill when your "
            "request matches its description."
        )
    except Exception as e:
        _logger.exception(f"Failed to install skills: {e}")
        typer.echo(f"✗ Failed to install skills: {e}", err=True)
        raise typer.Exit(code=1)


def _install_skills(skills_src: Path, output: Path, *, force: bool) -> list[str]:
    """Merge every skill (and AGENTS.md) from *skills_src* into *output*.

    Existing files are only overwritten when *force* is True. Returns the list
    of top-level items copied.
    """
    output.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for item in sorted(skills_src.iterdir()):
        if item.name in _SKIP or item.name.startswith("_"):
            continue
        dest = output / item.name
        if item.is_dir():
            _merge_tree(item, dest, force=force)
            copied.append(f"{item.name}/")
        elif item.is_file():
            if not dest.exists() or force:
                shutil.copy2(item, dest)
            copied.append(item.name)
    return copied


def _print_included(skills_src: Path) -> None:
    """Echo the installed skills with their descriptions."""
    typer.echo()
    typer.echo("Included skills:")
    for s in list_skills():
        desc = _get_skill_description(skills_src / s / "SKILL.md")
        typer.echo(f"  - {s:<30} {desc or ''}")


def _merge_tree(src: Path, dst: Path, *, force: bool = False) -> None:
    """Recursively copy *src* into *dst*, merging into existing directories.

    Files are only overwritten when *force* is True.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        # Never copy Python bytecode caches into the install target (they can be
        # created by running the skill scripts in place).
        if item.name == "__pycache__" or item.suffix in {".pyc", ".pyo"}:
            continue
        dest = dst / item.name
        if item.is_dir():
            _merge_tree(item, dest, force=force)
        elif item.is_file():
            if not dest.exists() or force:
                shutil.copy2(item, dest)


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
        typer.echo("\n  + AGENTS.md (agent workflow guidelines)")

    typer.echo()
    typer.echo("Usage:")
    typer.echo("  sflow skill --target all         # Install into claude/cursor/codex folders")
    typer.echo("  sflow skill -t claude --global   # Install into ~/.claude/skills")
    typer.echo("  sflow skill                      # Copy to ./skills (prompts)")
    typer.echo("  sflow skill -o vendor/skills     # Copy to a custom directory")


def _get_skill_description(skill_md: Path) -> str | None:
    """Extract description from SKILL.md frontmatter."""
    try:
        # Skill files are authored in UTF-8; decode explicitly so this works on
        # non-UTF-8 locales (e.g. Windows GBK) instead of silently returning None.
        content = skill_md.read_text(encoding="utf-8")
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
