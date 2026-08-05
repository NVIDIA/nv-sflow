# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`kubectl apply` subcommand flags (--extra-kubectl-apply-args).

kubectl takes global flags BEFORE the verb and subcommand flags AFTER it, so an
apply-only flag (--validate=false, --server-side, ...) cannot ride in the global
channel: `kubectl --validate=false apply` is an unknown flag, and it would break
every other kubectl call too. These cover the separate channel that carries them.
"""

import logging
import shlex
import subprocess

from sflow.core.kubectl_config import KubectlConfig
from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig
from sflow.plugins.k8s.shell import kubectl_global_args_prelude


def _cfg(**kwargs) -> KubernetesBackendConfig:
    base = {"name": "k8s", "type": "kubernetes"}
    base.update(kwargs)
    return KubernetesBackendConfig(**base)


def _backend(*, apply_args=(), extra_args=(), namespace="ml") -> KubernetesBackend:
    be = KubernetesBackend(_cfg(namespace=namespace))
    be.apply_kubectl_config(
        KubectlConfig(apply_args=list(apply_args), extra_args=list(extra_args))
    )
    return be


# ---------------------------------------------------------------------------
# the shell prelude
# ---------------------------------------------------------------------------


def test_prelude_unchanged_when_no_apply_args():
    # Backward compatibility: the globals-only wrapper must stay byte-identical, so
    # every existing generated script is unaffected.
    assert kubectl_global_args_prelude(["--context", "prod"]) == [
        'kubectl() { command kubectl --context prod "$@"; }'
    ]
    assert kubectl_global_args_prelude([]) == []


def test_prelude_emitted_for_apply_args_alone():
    # No global flags but apply flags present -> the wrapper is still needed.
    lines = kubectl_global_args_prelude([], ["--validate=false"])
    assert lines and "apply --validate=false" in "\n".join(lines)


def test_prelude_splices_apply_args_after_the_verb():
    script = "\n".join(
        kubectl_global_args_prelude(["--context", "prod"], ["--validate=false"])
    )
    # Flag lands AFTER `apply` (where kubectl accepts it), never before.
    assert "command kubectl --context prod apply --validate=false" in script
    assert "--validate=false apply" not in script


def test_prelude_leaves_non_apply_calls_alone():
    script = "\n".join(kubectl_global_args_prelude(["--context", "prod"], ["--validate=false"]))
    else_branch = script.split("else", 1)[1]
    assert "--validate" not in else_branch  # get/logs/delete stay clean


def test_prelude_quotes_apply_args():
    script = "\n".join(kubectl_global_args_prelude([], ["--field-manager=a b"]))
    assert shlex.quote("--field-manager=a b") in script


def _run_wrapper(script_lines, invocation, fake_kubectl_dir):
    """Execute the generated wrapper against a stub kubectl; return what it saw."""
    script = "\n".join(["set -euo pipefail", *script_lines, invocation])
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        # Give the script a stdin that is always at EOF. The stub kubectl DRAINS stdin
        # (see the test below for why), and without this the non-piped invocations
        # would hand it whatever fd 0 pytest happens to have -- /dev/null under default
        # capture, but the real terminal under `-s`, where the drain would block
        # forever. The piped invocation is unaffected: bash gives that child the pipe.
        stdin=subprocess.DEVNULL,
        env={"PATH": f"{fake_kubectl_dir}:/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_generated_wrapper_actually_routes_flags(tmp_path, fake_process):
    # The wrapper is shell we generate, so assert on real bash behaviour, not just
    # the rendered string. The suite stubs subprocess globally; opt this one out so
    # bash really runs (against a stub kubectl on PATH, never a real cluster).
    fake_process.allow_unregistered(True)
    stub = tmp_path / "kubectl"
    # The stub must DRAIN stdin, like the real kubectl does for `apply -f -`. A reader
    # that exits without consuming the pipe makes the WRITER die of SIGPIPE, and under
    # the wrapper's own `set -o pipefail` that becomes the pipeline's status: exit 141,
    # with `set -e` aborting the script -- even though the routing under test was
    # perfect (stdout still held the expected "GOT: ..." line). It is a race between
    # `echo`'s 3-byte write and the stub's fork+exec+exit, so it passes on an idle box
    # (400/400 here, and 300/300 with every core saturated) and fails on a loaded CI
    # runner. Draining removes the race outright rather than making it rarer.
    stub.write_text('#!/bin/sh\ncat >/dev/null 2>&1 || true\necho "GOT: $*"\n')
    stub.chmod(0o755)
    lines = kubectl_global_args_prelude(["--context", "prod"], ["--validate=false"])

    assert _run_wrapper(lines, "kubectl apply -f -", str(tmp_path)) == (
        "GOT: --context prod apply --validate=false -f -"
    )
    assert _run_wrapper(lines, "kubectl get pods", str(tmp_path)) == (
        "GOT: --context prod get pods"
    )
    # Piped form (`... | kubectl apply -f -`) is how every manifest is applied.
    assert _run_wrapper(lines, 'echo "{}" | kubectl apply -f -', str(tmp_path)) == (
        "GOT: --context prod apply --validate=false -f -"
    )
    # A bare `kubectl` must not trip `set -u` on the unset $1.
    assert _run_wrapper(lines, "kubectl", str(tmp_path)) == "GOT: --context prod"


# ---------------------------------------------------------------------------
# backend + operator plumbing
# ---------------------------------------------------------------------------


def test_backend_exposes_apply_args_to_the_operator():
    be = _backend(apply_args=["--validate=false"])
    assert be.kubectl_apply_args == ["--validate=false"]
    # ...and they are NOT smuggled into the global flags.
    assert "--validate=false" not in be.kubectl_global_args


def test_backend_applies_flags_to_allocate_time_manifests(monkeypatch):
    # The reservation Pod / ComputeDomain / ResourceClaimTemplate are applied during
    # allocate(), BEFORE any task exists -- a cluster that needs the flag would fail
    # the whole run here if these were left unflagged.
    import asyncio

    from sflow.plugins.k8s import lifecycle as k8s_lifecycle

    be = _backend(apply_args=["--validate=false"])
    seen: list[list[str]] = []

    async def _run(args, **kw):
        seen.append(list(args))
        return 0, "", ""

    monkeypatch.setattr(k8s_lifecycle, "run_kubectl", _run)
    asyncio.run(be._apply_manifest({"kind": "Pod", "metadata": {"name": "p"}}))
    assert seen[0][:3] == ["apply", "-f", "-"]
    assert "--validate=false" in seen[0]


def test_no_apply_args_leaves_manifest_apply_unchanged(monkeypatch):
    import asyncio

    from sflow.plugins.k8s import lifecycle as k8s_lifecycle

    be = _backend()
    seen: list[list[str]] = []

    async def _run(args, **kw):
        seen.append(list(args))
        return 0, "", ""

    monkeypatch.setattr(k8s_lifecycle, "run_kubectl", _run)
    asyncio.run(be._apply_manifest({"kind": "Pod", "metadata": {"name": "p"}}))
    assert seen[0] == ["apply", "-f", "-", "--namespace", "ml"]


def test_warns_when_apply_only_flag_passed_as_a_global(caplog):
    # The trap this option exists to close: --extra-kubectl-args=--validate=false
    # makes kubectl reject EVERY call, far from the typo. Name the right option.
    with caplog.at_level(logging.WARNING):
        _backend(extra_args=["--validate=false"])
    assert "--extra-kubectl-apply-args" in caplog.text
    assert "unknown flag" in caplog.text


def test_no_warning_for_legitimate_global_flags(caplog):
    with caplog.at_level(logging.WARNING):
        _backend(extra_args=["--insecure-skip-tls-verify", "--request-timeout=30s"])
    assert "--extra-kubectl-apply-args" not in caplog.text


def test_task_apply_command_carries_the_flags():
    # End of the chain: the generated per-task wrapper must route apply flags too,
    # not just the backend's own manifests.
    from sflow.plugins.k8s.shell import build_apply_command

    cmd = build_apply_command(
        manifest_json='{"kind":"Pod"}',
        ns_seg=" --namespace ml",
        pod_names=["p"],
        secret_name=None,
        envs={},
        handoff_delete_pods=[],
        kubectl_global_args=["--context", "prod"],
        kubectl_apply_args=["--validate=false"],
    )
    script = cmd.as_str()
    assert "apply --validate=false" in script
    # The manifest apply itself stays a plain `kubectl apply -f -`; the wrapper adds
    # the flag, which is what makes this cover every apply line at once.
    assert "kubectl apply -f -" in script


def test_operator_reads_apply_args_off_the_backend():
    from sflow.plugins.operators.k8s_operator import K8sContainerOperator

    op = K8sContainerOperator.__new__(K8sContainerOperator)
    op._kubectl_apply_args = []
    backend = _backend(apply_args=["--validate=false"])
    # Only the attribute lookup matters here (the rest of apply_backend_context
    # needs a full allocation), so assert the contract the operator relies on.
    assert list(getattr(backend, "kubectl_apply_args", [])) == ["--validate=false"]
