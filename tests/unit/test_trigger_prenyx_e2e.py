# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The infmax e2e trigger must pin THIS branch's sflow build, and nothing else.

The whole point of delegating the infmax suites to prenyx-ci-automation is to test the
wheel this MR just published. Every failure mode here is silent -- the pipeline goes green
having benchmarked some other build, or having erased a setting the recipe needed -- so the
variable payload is pinned rather than eyeballed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "trigger_prenyx_e2e.py"
_spec = importlib.util.spec_from_file_location("trigger_prenyx_e2e", _SCRIPT)
trigger = importlib.util.module_from_spec(_spec)
# Register BEFORE exec_module: `@dataclass` resolves its own module via
# `sys.modules.get(cls.__module__)`, which returns None for a module loaded by
# spec_from_file_location alone -- the class body then dies with
# "'NoneType' object has no attribute '__dict__'" at import time.
sys.modules[_spec.name] = trigger
_spec.loader.exec_module(trigger)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    """Strip the trigger-token env vars from EVERY test in this module.

    ``main()`` gates on a token before it ever calls out, so a developer who happens to
    export ``CI_TRIGGER_TOKEN`` (common -- it is also prenyx's own convention) gets a
    different code path than CI does. That is exactly how the first version of
    ``test_every_known_cluster_tag_is_accepted`` passed locally and failed on the runner.
    Tests that need a token now set one explicitly.
    """
    for name in trigger.TRIGGER_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SFLOW_PUBLISHED_VERSION", raising=False)


def _vars(spec=None, **kw):
    base = dict(
        pipeline=spec if spec is not None else trigger.SMOKE_PIPELINES[0],
        sflow_version="0.2.3.dev42+feat.my.branch.abc1234",
        cluster_tag="gb200-ptyche",
        jobs_csv="smoke_test_jobs.csv",
    )
    base.update(kw)
    return trigger.build_variables(**base)


def test_pins_the_build_under_test_on_the_shared_index():
    """prenyx installs sflow by version from the index sflow publishes to.

    If SFLOW_VERSION did not reach prenyx, it would fall back to its OWN pinned default
    and benchmark a completely different sflow -- a green run that proves nothing about
    this branch.
    """
    v = _vars()
    assert v["SFLOW_VERSION"] == "0.2.3.dev42+feat.my.branch.abc1234"
    assert v["SFLOW_INDEX_URL"].endswith("ct-ppp-shto-pypi-local/simple")


def test_never_sends_sflow_extra_settings():
    """``SFLOW_EXTRA_SETTINGS`` is set PER MATRIX CHILD on prenyx's kimik2.5-vllm row
    (``-s DYNAMO_VERSION=1.0.1``). Trigger variables outrank job-level variables, so
    sending this key would erase that setting and break the one vLLM job this exists to
    run -- while still appearing to succeed."""
    for spec in trigger.SMOKE_PIPELINES:
        assert "SFLOW_EXTRA_SETTINGS" not in _vars(spec)


def test_every_pipeline_is_limited_to_fp4_and_one_cluster():
    """fp8 is out of scope, and a missing cluster filter would create every cluster's
    jobs -- on hardware nobody asked for."""
    for spec in trigger.SMOKE_PIPELINES:
        v = _vars(spec)
        assert v["ONLY_PRECISION"] == "fp4"
        assert v["ONLY_CLUSTER"] == "gb200-ptyche"
        assert v["JOBS_CSV"] == "smoke_test_jobs.csv"


def test_rows_are_split_across_pipelines_because_row_is_pipeline_scoped():
    """``SFLOW_EXTRA_ARGS`` (which carries --row) is a PIPELINE-level trigger variable,
    so one trigger cannot give sglang/vllm row 1 and trtllm row 2. Collapsing these back
    into a single pipeline would silently run kimik2.5 trtllm at row 1 = 10 nodes instead
    of row 2 = 4."""
    rows = [v["SFLOW_EXTRA_ARGS"] for v in (_vars(s) for s in trigger.SMOKE_PIPELINES)]
    assert rows == ["--row 1", "--row 2"]


def test_row_two_pipeline_is_narrowed_to_kimi_trtllm_only():
    """Row 2 is the cheap topology for kimik2.5 trtllm ONLY. Without the model/framework
    filters the other recipes would be re-created at row 2, where their topologies are
    different and more expensive (dsr1 sglang row 2 = 13 nodes vs 3 at row 1)."""
    v = _vars(trigger.SMOKE_PIPELINES[1])
    assert v["ONLY_MODEL"] == "kimik2.5"
    assert v["ONLY_FRAMEWORK"] == "trtllm"


def test_click_list_covers_the_three_requested_recipes_and_their_cost():
    clicks = [c for spec in trigger.SMOKE_PIPELINES for c in spec.clicks]
    labels = [label for label, _ in clicks]
    assert "[dsr1, fp4, sglang]" in labels
    assert "[kimik2.5, fp4, vllm]" in labels
    assert "[kimik2.5, fp4, trtllm]" in labels
    # Least-nodes selection: 3 + 5 + 4. A regression that picked row 1 for kimi trtllm
    # would show up here as 10 instead of 4.
    assert sum(nodes for _, nodes in clicks) == 12


def test_form_encodes_variables_the_way_the_trigger_api_expects():
    form = trigger.build_form("inference_x", "tok", {"A": "1", "B": "2"})
    assert ("token", "tok") in form
    assert ("ref", "inference_x") in form
    assert ("variables[A]", "1") in form and ("variables[B]", "2") in form


def test_unknown_cluster_tag_is_rejected_before_any_network_call():
    """A typo'd tag would otherwise create a pipeline with EVERY cluster's jobs in it,
    on hardware nobody asked for."""
    assert trigger.main(["--cluster-tag", "gb200-typo", "--sflow-version", "1.0"]) == 2


def test_missing_version_fails_instead_of_silently_testing_another_build():
    assert trigger.main(["--dry-run"]) == 2


def test_dry_run_makes_no_network_call(monkeypatch, capsys):
    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("--dry-run must not contact GitLab")

    monkeypatch.setattr(trigger, "trigger_pipeline", _boom)
    assert trigger.main(["--dry-run", "--sflow-version", "9.9.9"]) == 0
    assert "SFLOW_VERSION=9.9.9" in capsys.readouterr().out


def test_token_resolution_order(monkeypatch):
    for name in trigger.TRIGGER_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert trigger.resolve_trigger_token(None) is None
    monkeypatch.setenv("GITLAB_TRIGGER_TOKEN", "last")
    assert trigger.resolve_trigger_token(None) == "last"
    monkeypatch.setenv("PRENYX_TRIGGER_TOKEN", "first")
    assert trigger.resolve_trigger_token(None) == "first"
    assert trigger.resolve_trigger_token("explicit") == "explicit"


@pytest.mark.parametrize("tag", trigger.KNOWN_CLUSTER_TAGS)
def test_every_known_cluster_tag_is_accepted(tag, monkeypatch):
    monkeypatch.setenv("PRENYX_TRIGGER_TOKEN", "tok")
    monkeypatch.setattr(trigger, "trigger_pipeline", lambda **k: {"web_url": "u", "id": 1})
    assert trigger.main(["--cluster-tag", tag, "--sflow-version", "1.0"]) == 0


def test_missing_trigger_token_fails_before_calling_gitlab(monkeypatch):
    """No token -> exit 2, and no network call attempted.

    Pinned explicitly because this is the branch CI takes and a developer with
    CI_TRIGGER_TOKEN exported does not -- the autouse fixture above makes both the same.
    """
    def _boom(**_):  # pragma: no cover - must not be reached
        raise AssertionError("must not contact GitLab without a token")

    monkeypatch.setattr(trigger, "trigger_pipeline", _boom)
    assert trigger.main(["--sflow-version", "1.0"]) == 2
