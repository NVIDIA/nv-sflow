# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sflow skill CLI command."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sflow.cli import app
from sflow.skills import get_skills_dir, list_skills

runner = CliRunner()


class TestSkillsList:
    """Tests for sflow skill --list."""

    def test_list_shows_available_skills(self):
        result = runner.invoke(app, ["skill", "--list"])
        assert result.exit_code == 0
        assert "writing-sflow-yaml" in result.output
        assert "sflow-error-analysis" in result.output

    def test_list_shows_agents_md(self):
        result = runner.invoke(app, ["skill", "--list"])
        assert result.exit_code == 0
        assert "AGENTS.md" in result.output

    def test_list_shows_descriptions(self):
        result = runner.invoke(app, ["skill", "--list"])
        assert result.exit_code == 0
        assert "YAML" in result.output or "workflow" in result.output
        assert "error" in result.output.lower() or "troubleshoot" in result.output.lower()


class TestSkillsCopy:
    """Tests for sflow skill (copy to directory)."""

    def test_copy_to_default_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["skill"], input="y\n")
        assert result.exit_code == 0
        assert "Skills copied to" in result.output
        skills_dir = tmp_path / "skills"
        assert skills_dir.exists()
        assert (skills_dir / "AGENTS.md").exists()
        assert (skills_dir / "writing-sflow-yaml" / "SKILL.md").exists()
        assert (skills_dir / "sflow-error-analysis" / "SKILL.md").exists()

    def test_copy_to_custom_dir(self, tmp_path):
        output_dir = tmp_path / "custom_skills"
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="y\n")
        assert result.exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "AGENTS.md").exists()
        assert (output_dir / "writing-sflow-yaml" / "SKILL.md").exists()
        assert (output_dir / "sflow-error-analysis" / "SKILL.md").exists()

    def test_copy_includes_scripts(self, tmp_path):
        output_dir = tmp_path / "skills"
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="y\n")
        assert result.exit_code == 0
        assert (output_dir / "writing-sflow-yaml" / "scripts" / "validate_sflow_yaml.py").exists()
        assert (output_dir / "sflow-error-analysis" / "scripts" / "parse_sflow_errors.py").exists()

    def test_copy_includes_reference_docs(self, tmp_path):
        output_dir = tmp_path / "skills"
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="y\n")
        assert result.exit_code == 0
        assert (output_dir / "writing-sflow-yaml" / "schema-reference.md").exists()
        assert (output_dir / "writing-sflow-yaml" / "examples.md").exists()
        assert (output_dir / "sflow-error-analysis" / "error-catalog.md").exists()

    def test_copy_shows_confirmation_prompt(self, tmp_path):
        """Confirmation prompt shows target directory and skill names."""
        output_dir = tmp_path / "skills"
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="y\n")
        assert result.exit_code == 0
        assert "Skills will be copied to:" in result.output
        assert "Proceed?" in result.output

    def test_copy_aborts_on_decline(self, tmp_path):
        """Declining the confirmation prompt aborts without copying."""
        output_dir = tmp_path / "skills"
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="n\n")
        assert result.exit_code != 0
        assert not output_dir.exists()

    def test_copy_merges_into_existing_dir(self, tmp_path):
        """Copying into an existing directory merges without removing other files."""
        output_dir = tmp_path / "skills"
        output_dir.mkdir()
        (output_dir / "other_skill.txt").write_text("custom")
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="y\n")
        assert result.exit_code == 0
        assert "Skills copied to" in result.output
        assert "already exists" in result.output
        assert (output_dir / "other_skill.txt").read_text() == "custom"
        assert (output_dir / "AGENTS.md").exists()
        assert (output_dir / "writing-sflow-yaml" / "SKILL.md").exists()

    def test_copy_preserves_existing_files_without_force(self, tmp_path):
        """Existing files are not overwritten unless --force is used."""
        output_dir = tmp_path / "skills"
        output_dir.mkdir()
        agents_md = output_dir / "AGENTS.md"
        agents_md.write_text("customized")
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="y\n")
        assert result.exit_code == 0
        assert agents_md.read_text() == "customized"

    def test_copy_overwrites_existing_files_with_force(self, tmp_path):
        """--force overwrites existing files but preserves unrelated ones."""
        output_dir = tmp_path / "skills"
        output_dir.mkdir()
        agents_md = output_dir / "AGENTS.md"
        agents_md.write_text("customized")
        (output_dir / "other_skill.txt").write_text("keep me")
        result = runner.invoke(app, ["skill", "-o", str(output_dir), "--force"], input="y\n")
        assert result.exit_code == 0
        assert agents_md.read_text(encoding="utf-8") != "customized"
        assert (output_dir / "other_skill.txt").read_text() == "keep me"

    def test_copy_merges_into_existing_skill_subdir(self, tmp_path):
        """Merging preserves extra files inside an existing skill subdirectory."""
        output_dir = tmp_path / "skills"
        skill_dir = output_dir / "writing-sflow-yaml"
        skill_dir.mkdir(parents=True)
        (skill_dir / "my_notes.txt").write_text("notes")
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="y\n")
        assert result.exit_code == 0
        assert (skill_dir / "my_notes.txt").read_text() == "notes"
        assert (skill_dir / "SKILL.md").exists()

    def test_copy_does_not_include_init_py(self, tmp_path):
        output_dir = tmp_path / "skills"
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="y\n")
        assert result.exit_code == 0
        assert not (output_dir / "__init__.py").exists()

    def test_copy_excludes_pycache_and_pyc(self, tmp_path):
        """Bytecode caches are never copied, at any depth (they can exist in scripts/)."""
        output_dir = tmp_path / "skills"
        result = runner.invoke(app, ["skill", "-o", str(output_dir)], input="y\n")
        assert result.exit_code == 0
        assert not list(output_dir.rglob("__pycache__"))
        assert not list(output_dir.rglob("*.pyc"))


class TestSkillsTargets:
    """Tests for `sflow skill --target` (direct install into agent folders)."""

    def test_target_claude_installs_to_dot_claude_skills(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # No input piped: --target must not prompt.
        result = runner.invoke(app, ["skill", "--target", "claude"])
        assert result.exit_code == 0, result.output
        skills = tmp_path / ".claude" / "skills"
        assert (skills / "writing-sflow-yaml" / "SKILL.md").exists()
        assert (skills / "AGENTS.md").exists()

    def test_target_skips_confirmation_prompt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["skill", "-t", "claude"])
        assert result.exit_code == 0
        assert "Proceed?" not in result.output

    def test_target_cursor_and_codex_folders(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["skill", "-t", "cursor", "-t", "codex"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / ".cursor" / "skills" / "writing-sflow-yaml" / "SKILL.md").exists()
        assert (tmp_path / ".codex" / "skills" / "writing-sflow-yaml" / "SKILL.md").exists()
        # Only the requested targets are written.
        assert not (tmp_path / ".claude").exists()

    def test_target_all_installs_three(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["skill", "--target", "all"])
        assert result.exit_code == 0, result.output
        for tool in (".claude", ".cursor", ".codex"):
            assert (tmp_path / tool / "skills" / "sflow-error-analysis" / "SKILL.md").exists()

    def test_target_global_uses_home_dir(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "home", lambda: home)
        result = runner.invoke(app, ["skill", "-t", "codex", "--global"])
        assert result.exit_code == 0, result.output
        assert (home / ".codex" / "skills" / "writing-sflow-yaml" / "SKILL.md").exists()
        # Nothing written into the project dir.
        assert not (tmp_path / ".codex").exists()

    def test_target_cursor_global_falls_back_to_project(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(Path, "home", lambda: home)
        result = runner.invoke(app, ["skill", "-t", "cursor", "--global"])
        assert result.exit_code == 0, result.output
        # Cursor has no global dir -> project install, with a note.
        assert (tmp_path / ".cursor" / "skills" / "writing-sflow-yaml" / "SKILL.md").exists()
        assert not (home / ".cursor").exists()
        assert "no user-level skills dir" in result.output

    def test_target_and_output_mutually_exclusive(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app, ["skill", "-t", "claude", "-o", str(tmp_path / "x")]
        )
        assert result.exit_code != 0
        assert "not both" in result.output

    def test_target_merges_and_preserves_existing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        skill_dir = tmp_path / ".claude" / "skills" / "writing-sflow-yaml"
        skill_dir.mkdir(parents=True)
        (skill_dir / "my_notes.txt").write_text("notes")
        result = runner.invoke(app, ["skill", "-t", "claude"])
        assert result.exit_code == 0, result.output
        assert (skill_dir / "my_notes.txt").read_text() == "notes"
        assert (skill_dir / "SKILL.md").exists()


class TestSkillsModule:
    """Tests for the sflow.skills module."""

    def test_get_skills_dir_exists(self):
        skills_dir = get_skills_dir()
        assert skills_dir.exists()
        assert skills_dir.is_dir()

    def test_list_skills_returns_expected(self):
        skills = list_skills()
        assert "writing-sflow-yaml" in skills
        assert "sflow-error-analysis" in skills

    def test_list_skills_excludes_pycache(self):
        skills = list_skills()
        assert "__pycache__" not in skills

    def test_skill_directories_have_skill_md(self):
        skills_dir = get_skills_dir()
        for skill_name in list_skills():
            skill_path = skills_dir / skill_name
            if skill_path.is_dir() and skill_name != "user-docs":
                assert (skill_path / "SKILL.md").exists(), f"{skill_name} missing SKILL.md"

    def test_skill_md_has_frontmatter(self):
        skills_dir = get_skills_dir()
        for skill_name in list_skills():
            skill_md = skills_dir / skill_name / "SKILL.md"
            if skill_md.exists():
                content = skill_md.read_text(encoding="utf-8")
                assert content.startswith("---"), f"{skill_name}/SKILL.md missing frontmatter"
                assert "name:" in content, f"{skill_name}/SKILL.md missing name field"
                assert "description:" in content, f"{skill_name}/SKILL.md missing description field"

    def test_agents_md_exists(self):
        skills_dir = get_skills_dir()
        assert (skills_dir / "AGENTS.md").exists()

    def test_agents_md_references_skills(self):
        skills_dir = get_skills_dir()
        content = (skills_dir / "AGENTS.md").read_text(encoding="utf-8")
        assert "writing-sflow-yaml" in content
        assert "sflow-error-analysis" in content
        assert "nvidia.github.io/nv-sflow" in content
