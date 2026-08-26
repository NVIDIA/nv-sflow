# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the e2e run-evidence publisher.

The e2e suite that uses this only runs on a GPU host, so without these the
publisher would ship unexercised -- and it exists precisely to be trusted when
nobody is watching: a renderer that silently finds nothing turns a green CI job
back into the "just a pass count" problem it was written to solve.

Loaded by path rather than imported: ``tests/e2e_tests`` is not a package, and its
test module has import-time side effects (it shells out to `docker info` and
`nvidia-smi`) that a unit test has no business triggering.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "e2e_tests" / "run_evidence.py"
_spec = importlib.util.spec_from_file_location("run_evidence", _MODULE_PATH)
run_evidence = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("run_evidence", run_evidence)
_spec.loader.exec_module(run_evidence)


@pytest.fixture
def run_tree(tmp_path: Path) -> Path:
    """A tree shaped like one sflow run: a summary, task logs, a driver log."""
    out = tmp_path / "pipeline" / "wf-20260807-000000-abcdef"
    out.mkdir(parents=True)
    (out / "sflow_summary.log").write_text(
        "Timeline\n--------\nserver_a COMPLETED\nmerged_consumer COMPLETED\n"
    )
    (out / "server_a.log").write_text("\n".join(f"line {i}" for i in range(100)))
    (out / "merged_consumer.log").write_text("MERGED_CONSUMER_GPUS=2\n")
    (out / "sflow.log").write_text("driver chatter nobody asked for\n")
    return tmp_path


def test_render_includes_the_summary_and_the_task_logs(run_tree: Path):
    text = run_evidence.render_run_evidence(run_tree, "test_pipeline")

    assert "run evidence: test_pipeline" in text
    # The summary is the headline: it is what says the workflow reached the end.
    assert "merged_consumer COMPLETED" in text
    # Task logs carry the actual workload evidence.
    assert "MERGED_CONSUMER_GPUS=2" in text
    assert "server_a.log" in text


def test_render_skips_the_driver_log_but_not_the_summary(run_tree: Path):
    """sflow.log is driver chatter, already on stdout; the summary is the point."""
    text = run_evidence.render_run_evidence(run_tree, "t")

    assert "driver chatter nobody asked for" not in text
    assert "Timeline" in text


def test_long_task_logs_are_tailed_and_say_so(run_tree: Path):
    text = run_evidence.render_run_evidence(run_tree, "t", tail=10)

    assert "line 99" in text, "the tail must keep the END of the log, not the start"
    assert "line 5" not in text
    # An elided log must announce it, or a truncated tail reads as a short run.
    assert "last 10 of 100 lines" in text


def test_tail_zero_keeps_nothing_rather_than_everything(run_tree: Path):
    """`lines[-0:]` is the whole file, not an empty tail -- the opposite of the ask.

    Latent today (nothing passes 0), but a caller that dialled the tail down to
    silence a chatty log would have got every line of it instead.
    """
    text = run_evidence.render_run_evidence(run_tree, "t", tail=0)

    assert "line 99" not in text and "line 0" not in text
    assert "last 0 of 100 lines" in text, "an emptied tail must still say so"
    # The summary is not tailed, so it survives regardless.
    assert "merged_consumer COMPLETED" in text


def test_a_negative_tail_keeps_the_whole_log(run_tree: Path):
    text = run_evidence.render_run_evidence(run_tree, "t", tail=-1)
    assert "line 0" in text and "line 99" in text


def test_short_task_logs_are_not_labelled_as_tailed(run_tree: Path):
    text = run_evidence.render_run_evidence(run_tree, "t")
    assert "MERGED_CONSUMER_GPUS=2" in text
    assert "last 1 of 1 lines" not in text


def test_a_tree_with_no_sflow_output_renders_nothing(tmp_path: Path):
    """Tests that never ran a workflow must not add noise to the job log."""
    (tmp_path / "scratch.txt").write_text("not a run")
    assert run_evidence.render_run_evidence(tmp_path, "t") == ""


def test_a_missing_tree_renders_nothing(tmp_path: Path):
    assert run_evidence.render_run_evidence(tmp_path / "nope", "t") == ""


def test_archive_copies_the_whole_tree_under_the_test_name(run_tree: Path, tmp_path: Path):
    artifacts = tmp_path.parent / "artifacts"
    dest = run_evidence.archive_run_dir(run_tree, "test_pipeline", str(artifacts))

    assert dest == artifacts / "test_pipeline"
    assert (dest / "pipeline").is_dir()
    summaries = list(dest.rglob("sflow_summary.log"))
    assert summaries, "the summary is the one file the archive exists for"
    # Full fidelity: the driver log is skipped in the ECHO but kept in the archive,
    # and so is anything else a future test starts writing.
    assert list(dest.rglob("sflow.log"))


def test_archive_is_a_noop_without_an_artifact_root(run_tree: Path):
    """Unset locally, where tmp_path is already on disk -- must not copy anywhere."""
    assert run_evidence.archive_run_dir(run_tree, "t", None) is None
    assert run_evidence.archive_run_dir(run_tree, "t", "") is None


def test_archive_never_raises_when_the_destination_is_unusable(
    run_tree: Path, tmp_path: Path
):
    """A bookkeeping failure must not turn a green e2e red."""
    blocker = tmp_path.parent / "blocked"
    blocker.write_text("I am a file, not a directory")

    assert run_evidence.archive_run_dir(run_tree, "t", str(blocker)) is None


def test_echo_prints_the_rendered_evidence(run_tree: Path, capsys):
    run_evidence.echo_run_evidence(run_tree, "test_pipeline")
    assert "merged_consumer COMPLETED" in capsys.readouterr().out


def test_echo_stays_silent_when_there_is_nothing(tmp_path: Path, capsys):
    run_evidence.echo_run_evidence(tmp_path, "t")
    assert capsys.readouterr().out == ""
