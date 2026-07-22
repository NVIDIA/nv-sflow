# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden-manifest tests driven by the real Kubernetes recipe files.

Unlike ``test_k8s_golden_manifests.py`` (which hand-builds a few operators), this
renders the **shipped recipes** end to end -- config load -> assembly -> per-task
manifest -- so drift in the recipe YAMLs, the assembly/merge planning, or the
operator rendering is all caught by CI without a cluster:

* ``build_state(allocate=False)`` seeds a deterministic placeholder allocation
  (``KubernetesBackend.placeholder_allocation``: fixed node names ``<backend>-nodeN``,
  IPs ``0.0.0.N``, allocation id ``kubernetes``), so every rendered path is stable
  and machine-independent.
* each task's ``launch_command`` is the exact apply command the launcher would run;
  the pod/List/MPIJob manifest is embedded between ``SFLOW_K8S_MANIFEST`` markers.
* merge FOLLOWERS are omitted -- their scripts run inside the leader's merged pod,
  so the leader's manifest already carries them (and any merge-grouping drift shows
  up as a changed leader manifest or a changed set of rendered tasks).

The offloaded ``kubectl logs`` demux child, the SSH keypair (injected at execute()
for the pods route), and the local output dir are NOT part of the rendered pod
manifest, so goldens stay deterministic. The MPI keypair generator is seeded anyway
for belt-and-suspenders determinism on the operator (MPIJob) route.

The same representative set is exercised by ``scripts/full_sample_tests.sh -P`` so
the ``preflight_cli`` CI job also covers K8s manifest rendering.

Regenerate goldens after an intentional recipe/manifest change:

    SFLOW_UPDATE_GOLDEN=1 pytest tests/unit/test_k8s_recipe_golden_manifests.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from sflow.app.assembly import build_state
from sflow.config.loader import ConfigLoader

_MARK = "SFLOW_K8S_MANIFEST"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECIPE_DIR = _REPO_ROOT / "examples" / "self_contained" / "kubernetes"
_GOLDEN_DIR = Path(__file__).parent / "golden" / "k8s_recipes"

# A fixed (never-touched) workspace dir keeps any path that reaches a manifest
# stable across machines; allocate=False + materialize=False never write here.
_WORKSPACE = Path("/sflow-golden")

# Representative recipes spanning the render paths: plain-k8s agg WITH a merged pod,
# plain-k8s disagg WITH merged prefill+decode pods, and a k8s_mpi (TRT-LLM) disagg.
_RECIPES = {
    "dynamo_vllm_agg": "dynamo_vllm_agg.yaml",
    "dynamo_sglang_disagg": "dynamo_sglang_disagg.yaml",
    "dynamo_trtllm_disagg": "dynamo_trtllm_disagg.yaml",
}

# Modular backend-agnostic compositions (examples/modular/backend_agnostic) rendered onto
# the K8s backends -- the SAME workload fragments the dry-run matrix in
# scripts/full_sample_tests.sh (MODULAR_CASES) composes. Each entry: (relative file list
# under _MODULAR_DIR, variable overrides). This snapshots the deep-merge + fold render path
# (plain k8s and cross-node k8s_mpi), including the operator's OMP_NUM_THREADS default.
_MODULAR_DIR = _REPO_ROOT / "examples" / "modular" / "backend_agnostic"
_MODULAR_GOLDEN_DIR = _GOLDEN_DIR / "modular"
_MODULAR_RECIPES: dict[str, tuple[list[str], list[str] | None]] = {
    "dynamo_trtllm_k8s": (
        ["backends/k8s.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_trtllm.yaml", "benchmark.yaml"], None),
    "dynamo_trtllm_k8s_mpi": (
        ["backends/k8s_mpi.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_trtllm.yaml", "benchmark.yaml"], ["NUM_NODES=2", "SERVER_GPUS=16"]),
    "dynamo_vllm_k8s": (
        ["backends/k8s.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_vllm.yaml", "benchmark.yaml"], None),
    "dynamo_sglang_k8s": (
        ["backends/k8s.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_sglang.yaml", "benchmark.yaml"], None),
    "trtllm_serve_k8s": (
        ["backends/k8s.yaml", "workloads/trtllm_serve.yaml", "benchmark.yaml"], None),
    "trtllm_serve_k8s_mpi": (
        ["backends/k8s_mpi.yaml", "workloads/trtllm_serve.yaml", "benchmark.yaml"],
        ["NUM_NODES=2", "SERVER_GPUS=16"]),
    "vllm_serve_k8s": (
        ["backends/k8s.yaml", "workloads/vllm_serve.yaml", "benchmark.yaml"], None),
    "sglang_serve_k8s": (
        ["backends/k8s.yaml", "workloads/sglang_serve.yaml", "benchmark.yaml"], None),
}


def _extract_manifest(shell: str) -> dict | None:
    """Pull the JSON manifest embedded between ``SFLOW_K8S_MANIFEST`` markers."""
    if _MARK not in shell:
        return None
    after = shell.split(_MARK, 1)[1]
    body = after.split("\n" + _MARK, 1)[0]
    return json.loads(body.split("\n", 1)[1])


def _render_manifests(
    paths: list[Path], overrides: list[str] | None, source_lines: list[str]
) -> str:
    """Render every K8s task's manifest for a (possibly multi-file) config into one
    deterministic doc.

    Loads + merges the config file(s) with the given variable overrides, assembles with
    a placeholder allocation (no cluster), and dumps each non-follower task's rendered
    manifest as sorted YAML, keyed by task name in topological order.
    """
    config = ConfigLoader().load_configs(list(paths), overrides, None, None)
    state = asyncio.run(
        build_state(config, allocate=False, workspace_dir=_WORKSPACE)
    )
    task_graph = state.workflow.task_graph

    sections: list[str] = [
        "# sflow K8s recipe golden manifests -- DO NOT EDIT BY HAND.",
        *source_lines,
        "# Rendered offline with a deterministic placeholder allocation (no cluster).",
        "# Merge followers are omitted; their containers run in the leader's merged pod.",
        "# Regenerate: SFLOW_UPDATE_GOLDEN=1 pytest "
        "tests/unit/test_k8s_recipe_golden_manifests.py",
    ]
    rendered = 0
    for name in task_graph.dag.topological_sort():
        task = task_graph.get_task(name)
        if getattr(task, "is_merge_follower", False):
            continue
        manifest = _extract_manifest(task.launch_command.as_list()[-1])
        if manifest is None:
            continue  # non-K8s task (no manifest) -- nothing to snapshot
        rendered += 1
        op = type(task.operator).__name__
        dumped = yaml.safe_dump(
            manifest, sort_keys=True, default_flow_style=False, width=100
        )
        sections.append(f"---\n# task: {name} (operator: {op})\n{dumped.rstrip()}")

    assert rendered > 0, f"no K8s manifests rendered for {source_lines}"
    return "\n".join(sections) + "\n"


def render_recipe_manifests(recipe_path: Path) -> str:
    """Render a single self-contained recipe file's K8s task manifests."""
    return _render_manifests(
        [recipe_path],
        None,
        [f"# recipe: examples/self_contained/kubernetes/{recipe_path.name}"],
    )


def render_modular_manifests(rel_paths: list[str], overrides: list[str] | None) -> str:
    """Render a modular composition (backend + workload(s) + benchmark) K8s manifests."""
    src = "# composed: " + " + ".join(
        f"examples/modular/backend_agnostic/{p}" for p in rel_paths
    )
    if overrides:
        src += "  (-s " + " -s ".join(overrides) + ")"
    return _render_manifests([_MODULAR_DIR / p for p in rel_paths], overrides, [src])


@pytest.fixture(autouse=True)
def _seed_mpi_keypair(monkeypatch):
    # The MPIJob (operator) route can embed the shared SSH keypair; seed it so the
    # rendered manifest is deterministic regardless of the host's crypto RNG.
    monkeypatch.setattr(
        "sflow.plugins.operators.k8s_mpi._generate_ssh_keypair_b64",
        lambda: ("PRIVB64", "PUBB64"),
    )


@pytest.mark.parametrize("name", sorted(_RECIPES))
def test_k8s_recipe_manifest_matches_golden(name):
    recipe_path = _RECIPE_DIR / _RECIPES[name]
    assert recipe_path.exists(), f"missing recipe {recipe_path}"
    rendered = render_recipe_manifests(recipe_path)

    path = _GOLDEN_DIR / f"{name}.yaml"
    if os.environ.get("SFLOW_UPDATE_GOLDEN"):
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"updated golden {path.name}")

    assert path.exists(), (
        f"missing golden {path}; regenerate with SFLOW_UPDATE_GOLDEN=1"
    )
    assert rendered == path.read_text(), (
        f"rendered k8s manifests for recipe '{name}' drifted from the golden; "
        f"if intended, regenerate with SFLOW_UPDATE_GOLDEN=1 pytest "
        f"tests/unit/test_k8s_recipe_golden_manifests.py"
    )


@pytest.mark.parametrize("name", sorted(_MODULAR_RECIPES))
def test_k8s_modular_manifest_matches_golden(name):
    rel_paths, overrides = _MODULAR_RECIPES[name]
    for rel in rel_paths:
        assert (_MODULAR_DIR / rel).exists(), f"missing modular file {rel}"
    rendered = render_modular_manifests(rel_paths, overrides)

    path = _MODULAR_GOLDEN_DIR / f"{name}.yaml"
    if os.environ.get("SFLOW_UPDATE_GOLDEN"):
        _MODULAR_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"updated golden {path.name}")

    assert path.exists(), (
        f"missing golden {path}; regenerate with SFLOW_UPDATE_GOLDEN=1"
    )
    assert rendered == path.read_text(), (
        f"rendered k8s manifests for modular '{name}' drifted from the golden; "
        f"if intended, regenerate with SFLOW_UPDATE_GOLDEN=1 pytest "
        f"tests/unit/test_k8s_recipe_golden_manifests.py"
    )
