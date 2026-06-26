# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side shell builders for the kubernetes operator.

This module owns the *imperative* half of the kubernetes operator: name
sanitization, env-secret handling, and the ``kubectl`` wrapper script that
applies a manifest, runs the create-before-destroy handoff, streams pod logs,
and propagates the pod exit code. The *declarative* half (manifest dicts) lives
in ``_k8s_render``.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence

from sflow.core.command import Command
from sflow.plugins.operators._k8s_render import SFLOW_ALLOC_LABEL

# kubectl logs -f bails with BadRequest on a not-yet-started container, so we
# poll the pod phase first; terminated.exitCode also lags container exit, so we
# poll that too. 300 * 2s readiness budget, 30 * 1s exit-code budget.
READY_RETRIES = 300
EXIT_CODE_RETRIES = 30

_MANIFEST_HEREDOC = "SFLOW_K8S_MANIFEST"


def sanitize_name(name: str, *, max_length: int = 253) -> str:
    """Coerce a task name into a DNS-1123-ish Kubernetes resource name."""
    sanitized = re.sub(r"[^a-z0-9-]+", "-", str(name).lower()).strip("-")
    if not sanitized:
        sanitized = "sflow-task"
    sanitized = sanitized[:max_length].strip("-")
    return sanitized or "sflow-task"


def configmap_data_key(name: str) -> str:
    """Coerce ``name`` into a valid ConfigMap data key (``[-._a-zA-Z0-9]+``).

    Unlike ``sanitize_name`` (DNS-1123 resource names), ConfigMap data keys allow
    upper-case, ``_`` and ``.``, so artifact names like ``PREFILL_CONFIG`` are kept
    readable; only out-of-charset characters are replaced.
    """
    key = re.sub(r"[^-._a-zA-Z0-9]", "_", str(name))
    return key or "artifact"


def namespace_segment(namespace: str | None) -> str:
    """Return ``" --namespace <ns>"`` (quoted) or ``""`` for kubectl commands."""
    return f" --namespace {shlex.quote(namespace)}" if namespace else ""


def secret_printf_lines(envs: Mapping[str, str]) -> list[str]:
    """Lines appending each env var to ``$env_file`` (values read at runtime, never inlined)."""
    return [
        f"printf '%s=%s\\n' {shlex.quote(str(key))} \"${{{str(key)}-}}\" >> \"$env_file\""
        for key in dict(envs).keys()
    ]


def create_secret_cmd(secret_name: str, ns_seg: str) -> str:
    """Idempotent ``kubectl create secret`` from ``$env_file``."""
    return (
        f"kubectl create secret generic {shlex.quote(secret_name)}{ns_seg} "
        '--from-env-file="$env_file" --dry-run=client -o yaml | kubectl apply -f -'
    )


def label_secret_cmd(secret_name: str, ns_seg: str, allocation_id: str) -> str:
    """Label the env Secret with the allocation label so the sweep can delete it."""
    return (
        f"kubectl label secret {shlex.quote(secret_name)}{ns_seg} "
        f"{SFLOW_ALLOC_LABEL}={shlex.quote(allocation_id)} --overwrite "
        ">/dev/null 2>&1 || true"
    )


def delete_secret_cmd(secret_name: str, ns_seg: str) -> str:
    """Best-effort secret cleanup for trap handlers."""
    return (
        f"kubectl delete secret {shlex.quote(secret_name)}{ns_seg} "
        "--ignore-not-found >/dev/null 2>&1 || true"
    )


def wait_for_pod_ready_lines(pod_ref: str, ns_seg: str, *, label: str) -> list[str]:
    """Shell lines that wait for ``pod_ref`` to start, echoing phase/reason.

    Polls the pod phase and a short ``reason`` (container waiting reason, else the
    ``PodScheduled`` condition message) and echoes ``[sflow] <label>: phase=<p>
    reason=<r> <detail> (<n>s)`` to stdout on change or every ~30s, so a slow
    image pull / unschedulable pod is never silent. ``<detail>`` adds live
    insight that ``reason`` alone hides (a bare ``ContainerCreating`` covers image
    pull, volume mounts, and sandbox setup): the container ``waiting.message``
    when set (e.g. ``Back-off pulling image``), otherwise the pod's most recent
    Event (e.g. ``Pulling: Pulling image "..."`` vs ``Failed: ...``) -- the only
    pull signal Kubernetes exposes (the API has no pull-progress percentage). On
    the timeout fall-through it dumps ``kubectl describe`` + recent events.
    """
    reason_jsonpath = (
        "jsonpath={.status.containerStatuses[*].state.waiting.reason}"
    )
    msg_jsonpath = (
        "jsonpath={.status.containerStatuses[*].state.waiting.message}"
    )
    cond_jsonpath = 'jsonpath={.status.conditions[?(@.type=="PodScheduled")].message}'
    # Most recent pod Event as "<reason>: <message>" (sorted ascending, take last).
    # custom-columns avoids embedding jsonpath newlines in the shell line.
    events_latest = (
        f"kubectl get events{ns_seg} --field-selector "
        f"involvedObject.name={shlex.quote(label)} --sort-by=.lastTimestamp "
        "--no-headers -o custom-columns=R:.reason,M:.message 2>/dev/null "
        "| tail -n 1 | tr -s ' ' || true"
    )
    return [
        'last=""',
        "for i in $(seq 1 %d); do" % READY_RETRIES,
        f"  phase=$(kubectl get {pod_ref}{ns_seg} -o jsonpath='{{.status.phase}}' 2>/dev/null || true);",
        f"  reason=$(kubectl get {pod_ref}{ns_seg} -o '{reason_jsonpath}' 2>/dev/null || true);",
        f"  msg=$(kubectl get {pod_ref}{ns_seg} -o '{msg_jsonpath}' 2>/dev/null || true);",
        f'  if [ -z "$reason" ]; then reason=$(kubectl get {pod_ref}{ns_seg} -o \'{cond_jsonpath}\' 2>/dev/null || true); fi;',
        '  cur="$phase|$reason|$msg";',
        '  if [ "$cur" != "$last" ] || [ $((i % 15)) -eq 0 ]; then',
        '    detail="";',
        '    if [ -n "$msg" ]; then detail=" msg=$msg";',
        f'    else ev=$({events_latest}); if [ -n "$ev" ]; then detail=" event=$ev"; fi; fi;',
        f'    echo "[sflow] {label}: phase=${{phase:-?}} reason=${{reason:-}}${{detail}} ($((i * 2))s)";',
        '    last="$cur";',
        "  fi;",
        '  case "$phase" in Running|Succeeded|Failed) break;; esac;',
        "  sleep 2;",
        "done",
        f'echo "[sflow] {label}: final phase=${{phase:-?}}";',
        f'if [ "${{phase:-}}" != "Running" ] && [ "${{phase:-}}" != "Succeeded" ]; then '
        f"kubectl describe {pod_ref}{ns_seg} 2>/dev/null || true; "
        f"kubectl get events{ns_seg} --field-selector involvedObject.name={shlex.quote(label)} --sort-by=.lastTimestamp 2>/dev/null | tail -n 10 || true; fi",
    ]


def stream_logs_line(pod_ref: str, ns_seg: str, *, background: bool) -> str:
    """A ``kubectl logs -f`` line; backgrounded (``&``) for parallel multi-pod streaming."""
    suffix = "&" if background else "|| true"
    return f"kubectl logs -f {pod_ref}{ns_seg} --all-containers --prefix {suffix}"


def exit_code_poll_lines(pod_ref: str, ns_seg: str) -> list[str]:
    """Poll ``pod_ref`` terminated.exitCode, then ``exit`` it (fallback 1)."""
    return [
        "exit_code=''",
        (
            f"for _ in $(seq 1 {EXIT_CODE_RETRIES}); do "
            f"exit_code=$(kubectl get {pod_ref}{ns_seg} -o jsonpath='{{.status.containerStatuses[0].state.terminated.exitCode}}' 2>/dev/null || true); "
            f'if [ -n "$exit_code" ]; then break; fi; sleep 1; done'
        ),
        'exit "${exit_code:-1}"',
    ]


def kubectl_global_args_prelude(args: Sequence[str]) -> list[str]:
    """A shell ``kubectl`` function that prefixes CLI-level global flags.

    Defining a function named ``kubectl`` makes every ``kubectl ...`` line in the
    wrapper (apply / logs / get / delete / secret / handoff) transparently carry
    the ``--kubeconfig`` / ``--context`` / passthrough flags, with no per-call
    plumbing. ``command kubectl`` runs the real binary (avoids recursion). Returns
    no lines when there are no global args (the real kubectl is used directly).
    """
    if not args:
        return []
    joined = " ".join(shlex.quote(str(a)) for a in args)
    return [f'kubectl() {{ command kubectl {joined} "$@"; }}']


def bash_lc_command(lines: Sequence[str]) -> Command:
    """Wrap shell ``lines`` into a ``bash -lc`` Command."""
    cmd = Command(exec="bash")
    cmd.add_arg("-lc")
    cmd.add_arg("\n".join(lines))
    return cmd


def handoff_delete_lines(reservation_pods: Sequence[str], ns_seg: str) -> list[str]:
    """Delete the per-node placeholder pods so their GPUs free up for the task.

    Create-before-destroy handoff: the real task pod(s) are already applied (and
    Pending, since the placeholder still holds the node's GPUs); deleting the
    placeholder blocks (``kubectl delete`` waits by default) until it is gone, at
    which point the already-queued task pod binds with no delete->apply gap.
    ``--ignore-not-found`` makes it idempotent across tasks sharing a node and
    across retries.
    """
    return [
        f"kubectl delete pod {shlex.quote(pod)}{ns_seg} "
        "--ignore-not-found >/dev/null 2>&1 || true"
        for pod in reservation_pods
    ]


def build_k8s_task_command(
    *,
    manifest_json: str,
    ns_seg: str,
    pod_names: Sequence[str],
    configmap_name: str,
    rct_name: str | None,
    secret_name: str | None,
    envs: Mapping[str, str],
    handoff_delete_pods: Sequence[str],
    parallel_logs: bool,
    kubectl_global_args: Sequence[str] = (),
    allocation_id: str | None = None,
    artifacts_configmap_name: str | None = None,
) -> Command:
    """Build the ``kubectl`` wrapper for a (possibly multi-pod) task.

    The script: optionally creates an env Secret, registers a cleanup trap,
    applies ``manifest_json`` (a v1/List of the ConfigMap + optional
    ResourceClaimTemplate + the task pod(s)), runs the create-before-destroy
    handoff (deletes the assigned node(s)' placeholder pods so the already-Pending
    task pod(s) bind), waits for each pod to start before streaming its logs (in
    parallel when ``parallel_logs``), then propagates the exit code from the
    leader pod (``pod_names[0]``).
    """
    use_secret = secret_name is not None
    pod_refs = [f"pod/{shlex.quote(p)}" for p in pod_names]

    cleanup_refs = [*pod_refs, f"configmap/{shlex.quote(configmap_name)}"]
    if artifacts_configmap_name is not None:
        cleanup_refs.append(f"configmap/{shlex.quote(artifacts_configmap_name)}")
    if secret_name is not None:
        cleanup_refs.append(f"secret/{shlex.quote(secret_name)}")
    if rct_name is not None:
        cleanup_refs.append(
            f"resourceclaimtemplate.resource.k8s.io/{shlex.quote(rct_name)}"
        )
    cleanup_delete = (
        f"kubectl delete {' '.join(cleanup_refs)}{ns_seg} "
        "--ignore-not-found >/dev/null 2>&1 || true"
    )
    cleanup_body = "; ".join(
        ['rm -f "$env_file"', cleanup_delete] if use_secret else [cleanup_delete]
    )

    lines = ["set -euo pipefail"]
    # Define a kubectl() wrapper carrying the CLI-level global flags (if any) so
    # every kubectl call below (apply / logs / delete / secret / handoff) uses them.
    lines.extend(kubectl_global_args_prelude(kubectl_global_args))
    if use_secret:
        lines.append("env_file=$(mktemp)")
    # Trap INT/TERM as well as EXIT: a bare EXIT trap does not run when bash is
    # killed by the SIGTERM the launcher sends on Ctrl+C, which would leak the
    # task objects. Disarm inside cleanup so it runs exactly once, then re-exit
    # with the original status.
    lines.append(
        f"cleanup() {{ rc=$?; trap - INT TERM EXIT; {cleanup_body}; exit $rc; }}"
    )
    lines.append("trap cleanup INT TERM EXIT")
    if secret_name is not None:
        lines.extend(secret_printf_lines(envs))
        lines.append(create_secret_cmd(secret_name, ns_seg))
        if allocation_id:
            lines.append(label_secret_cmd(secret_name, ns_seg, allocation_id))

    # Quoted heredoc keeps the JSON literal (the user script may contain single quotes).
    lines.append(
        f"cat <<'{_MANIFEST_HEREDOC}' | kubectl apply -f -\n{manifest_json}\n{_MANIFEST_HEREDOC}"
    )

    # Create-before-destroy: pods are applied (Pending) above; now free the
    # reserved node(s) so they bind. GPU tasks only (the operator passes an
    # empty list otherwise, so CPU-only tasks coexist with the placeholder).
    lines.extend(handoff_delete_lines(handoff_delete_pods, ns_seg))

    # Wait for each pod to leave Pending/ContainerCreating before streaming it,
    # echoing phase/reason while it starts up.
    for pod_ref, pod_name in zip(pod_refs, pod_names, strict=True):
        lines.extend(wait_for_pod_ready_lines(pod_ref, ns_seg, label=pod_name))
        lines.append(stream_logs_line(pod_ref, ns_seg, background=parallel_logs))
    if parallel_logs:
        # `wait` blocks until all backgrounded log streams close.
        lines.append("wait")

    lines.extend(exit_code_poll_lines(pod_refs[0], ns_seg))
    return bash_lc_command(lines)
