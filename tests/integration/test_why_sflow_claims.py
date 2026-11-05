# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The "Why sflow" pitch (the /why-sflow page and its slide deck) rests on two checkable
claims: the YAML it shows is the real example, unchanged, and the commands it shows run
the same workload on every backend. Keep both true, or the pitch quietly turns into
marketing."""

import html
import re
import shlex
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sflow.cli import app

_REPO = Path(__file__).resolve().parents[2]
_PAGE = _REPO / "docs-site" / "src" / "pages" / "why-sflow.mdx"
_DECK = _REPO / "docs-site" / "static" / "why_sflow.html"
_EXAMPLE = _REPO / "examples" / "modular" / "backend_agnostic"
# Pages that repeat the swap-one-file commands. README.md is maintained separately on
# GitHub, so it is checked only where it exists in this form.
_COMMAND_PAGES = [_PAGE, _DECK, _REPO / "docs" / "user" / "intro.md", _REPO / "README.md"]


def _quoted_yaml_blocks() -> list[tuple[str, str]]:
    """(source path, excerpt) for every repo file the page or the deck quotes."""
    page = re.findall(r'```yaml title="([^"]+)"\n(.*?)```', _PAGE.read_text(encoding="utf-8"), re.DOTALL)
    deck = [
        (path, html.unescape(body))
        for path, body in re.findall(
            r'<pre class="code" data-source="([^"]+)">(.*?)</pre>',
            _DECK.read_text(encoding="utf-8"),
            re.DOTALL,
        )
    ]
    return page + deck


def _backend_swap_commands() -> list[str]:
    commands: set[str] = set()
    for page in _COMMAND_PAGES:
        if page.exists():
            # The deck wraps the swapped file in markup; drop tags before matching.
            text = html.unescape(re.sub(r"<[^>]+>", "", page.read_text(encoding="utf-8")))
            for line in re.findall(r"sflow run (-f backends/[^`#\n]+)", text):
                commands.add(line.rstrip())
    return sorted(commands)


def test_page_and_deck_show_real_example_excerpts():
    blocks = _quoted_yaml_blocks()
    # the page quotes the workload + one backend per platform; the deck may quote more
    assert {path for path, _ in blocks} >= {
        f"examples/modular/backend_agnostic/{f}"
        for f in ("workloads/vllm_serve.yaml", "backends/docker.yaml", "backends/slurm.yaml", "backends/k8s.yaml")
    }, blocks
    for path, excerpt in blocks:
        lines = (_REPO / path).read_text(encoding="utf-8").splitlines()
        want = textwrap.dedent(excerpt).rstrip("\n")
        n = want.count("\n") + 1
        windows = (
            textwrap.dedent("\n".join(lines[i : i + n])) for i in range(len(lines) - n + 1)
        )
        assert want in windows, f"{path}: excerpt no longer matches the file"


def test_deck_line_counts_match_the_files():
    counts = re.findall(r'data-lines-of="([^"]+)">(\d+)<', _DECK.read_text(encoding="utf-8"))
    assert counts
    for path, shown in counts:
        actual = len((_REPO / path).read_text(encoding="utf-8").splitlines())
        assert int(shown) == actual, f"deck says {path} has {shown} lines; it has {actual}"


def test_pitch_lists_one_workload_on_three_backends():
    commands = [c for c in _backend_swap_commands() if "vllm_serve" in c]
    backends = {re.search(r"backends/(\w+)\.yaml", c).group(1) for c in commands}
    assert backends >= {"docker", "slurm", "k8s"}, commands


@pytest.mark.parametrize("command", _backend_swap_commands())
def test_backend_swap_command_dry_runs(command, tmp_path, monkeypatch):
    monkeypatch.chdir(_EXAMPLE)
    result = CliRunner().invoke(
        app,
        ["run", *shlex.split(command), "--dry-run", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, f"sflow run {command}\n{result.output}"
