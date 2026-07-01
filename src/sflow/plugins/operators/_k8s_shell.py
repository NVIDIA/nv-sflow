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

# kubectl logs -f bails with BadRequest on a not-yet-started container, so the
# apply command polls the pod phase first (READY_RETRIES * POLL_SLEEP_SECS budget)
# before the driver attaches the log stream. Status/exit-code polling and the
# early stop live driver-side in _k8s_lifecycle, not in this bash.
READY_RETRIES = 300
# Seconds between pod-phase polls in the apply wait loop.
POLL_SLEEP_SECS = 2
# Consecutive polls a pod may sit in an unrecoverable start state (Unschedulable,
# ImagePullBackOff, ...) before the apply step aborts the task (exit 1) instead of
# waiting out the full budget + readiness-probe timeout. A grace window (rather
# than failing on the first bad poll) tolerates transient blips -- e.g. a brief
# post-handoff scheduling gap, or an ErrImagePull that resolves on retry.
UNRECOVERABLE_GRACE_POLLS = 15
# Container waiting reasons / scheduling reason that do not clear on their own, so
# a pod stuck in one past the grace window is treated as a hard failure.
_UNRECOVERABLE_WAITING_REASONS = (
    "ImagePullBackOff",
    "ErrImagePull",
    "InvalidImageName",
    "CreateContainerConfigError",
    "CreateContainerError",
    "RunContainerError",
    "CrashLoopBackOff",
)

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
    ``PodScheduled`` reason/message) and echoes ``[sflow] <label>: phase=<p>
    reason=<r> <detail> (<n>s)`` to stdout on change or every ~30s, so a slow
    image pull / unschedulable pod is never silent. ``<detail>`` adds live
    insight that ``reason`` alone hides (a bare ``ContainerCreating`` covers image
    pull, volume mounts, and sandbox setup): the container ``waiting.message``
    when set (e.g. ``Back-off pulling image``), otherwise the pod's most recent
    Event (e.g. ``Pulling: Pulling image "..."`` vs ``Failed: ...``) -- the only
    pull signal Kubernetes exposes (the API has no pull-progress percentage).

    Fail-fast: if the pod sits in an unrecoverable start state
    (``Unschedulable``, ``ImagePullBackOff``, ...) for ``UNRECOVERABLE_GRACE_POLLS``
    consecutive polls, it dumps ``kubectl describe`` + events and exits non-zero --
    aborting the apply (and thus the task) in ~grace seconds instead of hanging
    until the readiness-probe timeout. A grace window (and a streak that resets on
    recovery) tolerates transient blips. On the budget fall-through it likewise
    dumps ``kubectl describe`` + recent events.
    """
    reason_jsonpath = (
        "jsonpath={.status.containerStatuses[*].state.waiting.reason}"
    )
    msg_jsonpath = (
        "jsonpath={.status.containerStatuses[*].state.waiting.message}"
    )
    sched_reason_jsonpath = (
        'jsonpath={.status.conditions[?(@.type=="PodScheduled")].reason}'
    )
    cond_jsonpath = 'jsonpath={.status.conditions[?(@.type=="PodScheduled")].message}'
    unrecoverable_pattern = "|".join(_UNRECOVERABLE_WAITING_REASONS)
    # Most recent pod Event as "<reason>: <message>" (sorted ascending, take last).
    # custom-columns avoids embedding jsonpath newlines in the shell line.
    events_latest = (
        f"kubectl get events{ns_seg} --field-selector "
        f"involvedObject.name={shlex.quote(label)} --sort-by=.lastTimestamp "
        "--no-headers -o custom-columns=R:.reason,M:.message 2>/dev/null "
        "| tail -n 1 | tr -s ' ' || true"
    )
    diagnostics_dump = (
        f"kubectl describe {pod_ref}{ns_seg} 2>/dev/null || true; "
        f"kubectl get events{ns_seg} --field-selector "
        f"involvedObject.name={shlex.quote(label)} --sort-by=.lastTimestamp "
        "2>/dev/null | tail -n 10 || true"
    )
    return [
        'last=""',
        "bad=0",
        "for i in $(seq 1 %d); do" % READY_RETRIES,
        f"  phase=$(kubectl get {pod_ref}{ns_seg} -o jsonpath='{{.status.phase}}' 2>/dev/null || true);",
        f"  wreason=$(kubectl get {pod_ref}{ns_seg} -o '{reason_jsonpath}' 2>/dev/null || true);",
        f"  msg=$(kubectl get {pod_ref}{ns_seg} -o '{msg_jsonpath}' 2>/dev/null || true);",
        f"  sched=$(kubectl get {pod_ref}{ns_seg} -o '{sched_reason_jsonpath}' 2>/dev/null || true);",
        '  reason="$wreason";',
        '  if [ -z "$reason" ] && [ -n "$sched" ]; then reason="$sched"; fi;',
        f'  if [ -z "$reason" ]; then reason=$(kubectl get {pod_ref}{ns_seg} -o \'{cond_jsonpath}\' 2>/dev/null || true); fi;',
        '  cur="$phase|$reason|$msg";',
        '  if [ "$cur" != "$last" ] || [ $((i % 15)) -eq 0 ]; then',
        '    detail="";',
        '    if [ -n "$msg" ]; then detail=" msg=$msg";',
        f'    else ev=$({events_latest}); if [ -n "$ev" ]; then detail=" event=$ev"; fi; fi;',
        f'    echo "[sflow] {label}: phase=${{phase:-?}} reason=${{reason:-}}${{detail}} ($((i * {POLL_SLEEP_SECS}))s)";',
        '    last="$cur";',
        "  fi;",
        '  case "$phase" in Running|Succeeded|Failed) break;; esac;',
        # Unrecoverable-state streak: fail the task early rather than idle until the
        # readiness-probe timeout. Resets whenever the pod is not in a bad state.
        "  isbad=0;",
        f'  case "$wreason" in {unrecoverable_pattern}) isbad=1;; esac;',
        '  if [ "$sched" = "Unschedulable" ]; then isbad=1; fi;',
        '  if [ "$isbad" = "1" ]; then bad=$((bad + 1)); else bad=0; fi;',
        '  if [ "$bad" -ge %d ]; then' % UNRECOVERABLE_GRACE_POLLS,
        f'    echo "[sflow] {label}: FATAL phase=${{phase:-?}} reason=${{reason:-}} — unrecoverable pod start state; aborting task early ($((i * {POLL_SLEEP_SECS}))s)";',
        f"    {diagnostics_dump};",
        "    exit 1;",
        "  fi;",
        f"  sleep {POLL_SLEEP_SECS};",
        "done",
        f'echo "[sflow] {label}: final phase=${{phase:-?}}";',
        f'if [ "${{phase:-}}" != "Running" ] && [ "${{phase:-}}" != "Succeeded" ]; then '
        f"{diagnostics_dump}; fi",
    ]


def build_log_stream_command(
    pod_ref: str,
    *,
    ns_args: Sequence[str] = (),
    kubectl_global_args: Sequence[str] = (),
) -> Command:
    """Build the standalone ``kubectl logs -f`` Command for one pod.

    Run by the sflow driver as its own subprocess (see
    ``K8sContainerOperator.execute``) so its stdout streams live to ``<task>.log``
    through the launcher, and the driver can stop it the moment a background
    status-watch sees the pod go terminal -- instead of blocking on the K8s log
    backlog. The CLI-level global flags are prefixed directly (no shell wrapper).
    """
    cmd = Command(exec="kubectl")
    for arg in kubectl_global_args:
        cmd.add_arg(str(arg))
    cmd.add_arg("logs")
    cmd.add_arg("-f")
    cmd.add_arg(pod_ref)
    for arg in ns_args:
        cmd.add_arg(str(arg))
    cmd.add_arg("--all-containers")
    cmd.add_arg("--prefix")
    return cmd


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


def build_apply_command(
    *,
    manifest_json: str,
    ns_seg: str,
    pod_names: Sequence[str],
    secret_name: str | None,
    envs: Mapping[str, str],
    handoff_delete_pods: Sequence[str],
    kubectl_global_args: Sequence[str] = (),
    allocation_id: str | None = None,
) -> Command:
    """Build the ``kubectl apply`` step for a (possibly multi-pod) task.

    This is the FIRST, fire-and-return half of a k8s task: optionally create the
    env Secret, ``kubectl apply`` the manifest (ConfigMap + optional
    ResourceClaimTemplate + the task pod(s)), run the create-before-destroy
    handoff (delete the assigned node(s)' placeholder pods so the already-Pending
    task pod(s) bind), then wait for each pod to leave Pending/ContainerCreating
    (echoing phase/reason). It then exits -- it does NOT stream logs, poll the
    exit code, or register a cleanup trap. The sflow driver (``execute``) takes
    over from there: it attaches the log stream, watches pod status, and deletes
    the task objects, while the kubernetes backend's allocation-label sweep is the
    backstop on Ctrl+C / crash.
    """
    use_secret = secret_name is not None
    pod_refs = [f"pod/{shlex.quote(p)}" for p in pod_names]

    lines = ["set -euo pipefail"]
    # Define a kubectl() wrapper carrying the CLI-level global flags (if any) so
    # every kubectl call below (secret / apply / handoff / wait) uses them.
    lines.extend(kubectl_global_args_prelude(kubectl_global_args))
    if use_secret:
        lines.append("env_file=$(mktemp)")
        lines.extend(secret_printf_lines(envs))
        lines.append(create_secret_cmd(secret_name, ns_seg))
        if allocation_id:
            lines.append(label_secret_cmd(secret_name, ns_seg, allocation_id))
        lines.append('rm -f "$env_file"')

    # Quoted heredoc keeps the JSON literal (the user script may contain single quotes).
    lines.append(
        f"cat <<'{_MANIFEST_HEREDOC}' | kubectl apply -f -\n{manifest_json}\n{_MANIFEST_HEREDOC}"
    )

    # Create-before-destroy: pods are applied (Pending) above; now free the
    # reserved node(s) so they bind. GPU tasks only (the operator passes an
    # empty list otherwise, so CPU-only tasks coexist with the placeholder).
    lines.extend(handoff_delete_lines(handoff_delete_pods, ns_seg))

    # Wait for each pod to leave Pending/ContainerCreating before returning, so
    # the driver's `kubectl logs -f` does not bail on a not-yet-started container.
    for pod_ref, pod_name in zip(pod_refs, pod_names, strict=True):
        lines.extend(wait_for_pod_ready_lines(pod_ref, ns_seg, label=pod_name))

    return bash_lc_command(lines)
