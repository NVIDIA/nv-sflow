# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the writing-sflow-yaml helper scripts."""

import importlib.util
import subprocess
import sys

import pytest

from sflow.skills import get_skills_dir

SCRIPTS = get_skills_dir() / "writing-sflow-yaml" / "scripts"


def _load(name, skill="writing-sflow-yaml"):
    path = get_skills_dir() / skill / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_skill_{name}", path)
    module = importlib.util.module_from_spec(spec)
    # Registered so `dataclasses` can resolve the script's postponed annotations.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gpu_plan_resolves_arithmetic_without_eval():
    resolve = _load("check_gpu_plan")._resolve_value
    variables = {"TP": {"value": 4}, "DP": 2, "N": 7}

    assert resolve("${{ variables.TP * variables.DP }}", variables) == 8
    assert resolve("${{ (variables.N + 1) // 2 }}", variables) == 4
    assert resolve("${{ -variables.DP + 10 }}", variables) == 8
    # Anything beyond + - * / // % on numbers is unresolved, never executed.
    assert resolve("${{ __import__('os').getcwd() }}", variables) is None
    assert resolve("${{ variables.N ** 99999 }}", variables) is None
    assert resolve("${{ variables.N / 0 }}", variables) is None


def test_gpu_plan_counts_a_docker_hosts_pool_as_the_node_list():
    capacity = _load("check_gpu_plan")._get_backend_capacity
    pool = {
        "name": "pool",
        "type": "docker",
        "gpus_per_node": 8,
        "hosts": [{"name": "a"}, {"name": "b", "gpus_per_node": "${{ variables.G }}"}],
    }

    [b] = capacity({"backends": [pool]}, {"G": 4})
    assert (b["nodes"], b["total_gpus"]) == (2, 12)


# One log line per failure signature learned on real hardware; the first matching
# pattern wins, so these also pin the ordering against the generic catch-alls.
@pytest.mark.parametrize(
    "line, description",
    [
        ("RuntimeError: No CUDA GPUs are available", "No GPUs in the allocation (GRES partition, none requested)"),
        ("nvidia-smi: No devices were found", "No GPUs in the allocation (GRES partition, none requested)"),
        ("salloc: error: Job allocation 123 has been revoked.", "Allocation lost to a node prolog/boot failure"),
        ("salloc: error: Nodes n[1-2] are still not ready", "Allocation lost to a node prolog/boot failure"),
        (
            "srun: error: Unable to confirm allocation for job 9: Invalid job id specified",
            "Allocation expired before the tasks launched (partner backend still queued)",
        ),
        (
            "enroot-mount: failed to mount: /dev/nvidia-caps-imex-channels at /r/dev/nvidia-caps-imex-channels: No such file or directory",
            "Cold enroot cache race on first container start (retry succeeds)",
        ),
        ("slurmstepd: error: pyxis: couldn't start container", "Cold enroot cache race on first container start (retry succeeds)"),
        ("ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'", "Container toolchain predates the GPU architecture"),
        ("[TRT-LLM] OOM during general warmup", "GPU memory not yet released by a previous run's pods"),
        ("Processed 0 valid requests and 512 errors", "aiperf measured nothing (every request failed; exit 0 anyway)"),
        ("Docker backend 'd' asks for 2 nodes but declares no 'hosts:'. Multi-node docker ...", "Docker nodes > 1 without a hosts: pool"),
        ("Docker backend 'd' declares 2 host(s) but asks for nodes: 3. 'hosts:' is the node list", "Docker nodes: disagrees with the hosts: pool size"),
        ("docker host 'a' needs an 'address' (the machine), or a 'docker_host'/'context' endpoint override", "Docker host identity misdeclared"),
        ("Pre-flight validation failed for Docker backend 'd':", "Docker pre-flight failed (host unreachable / mount source missing / no docker CLI)"),
        ("Error: --sflow-version 'develop' is not a valid PyPI version specifier. Use ...", "--sflow-version is PEP 440 only (git ref given)"),
        ("Error: sflow is currently a 'editable' (development) install (/src). Upgrading ...", "sflow upgrade refused to replace an editable/dev install"),
        ("Error: install failed with exit code 1.", "Installer failed (release not on the index?)"),
    ],
)
def test_parse_sflow_errors_recognises_hardware_signatures(line, description):
    parse = _load("parse_sflow_errors", "sflow-error-analysis")

    result = parse.parse_log([line])
    assert [m.pattern.description for m in result.matched_errors] == [description]


def test_preflight_dry_run_failure_runs_root_cause_parser(monkeypatch, capsys):
    preflight = _load("preflight")
    failed = subprocess.CompletedProcess([], 1, stdout="", stderr="Configuration validation failed: bad\n")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: failed)

    assert preflight._run_dry_run(["any.yaml"], []) is False
    assert "Root-cause analysis" in capsys.readouterr().out


def test_preflight_parser_crash_keeps_dry_run_verdict(monkeypatch, capsys):
    preflight = _load("preflight")
    failed = subprocess.CompletedProcess([], 1, stdout="", stderr="boom\n")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: failed)

    def broken_import(name):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(preflight.importlib, "import_module", broken_import)

    assert preflight._run_dry_run(["any.yaml"], []) is False
    assert "(skipped: parser exploded)" in capsys.readouterr().out


def test_preflight_fails_closed_when_validator_cannot_import(monkeypatch, capsys):
    preflight = _load("preflight")
    monkeypatch.setitem(sys.modules, "validate_sflow_yaml", None)  # import -> ImportError

    assert preflight._run_validation(["any.yaml"]) is False
    assert "schema/reference checks did not run" in capsys.readouterr().out
