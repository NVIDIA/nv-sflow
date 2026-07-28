# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side shell builders for the kubernetes operator.

This module owns the *imperative* half of the kubernetes operator: name
sanitization, env-secret handling, and the ``kubectl`` wrapper script that
applies a manifest, runs the create-before-destroy handoff, streams pod logs,
and propagates the pod exit code. The *declarative* half (manifest dicts) lives
in ``k8s.render``.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence

from sflow.core.command import Command
from sflow.plugins.k8s.render import (
    MERGE_DONE_CLOSE,
    MERGE_DONE_OPEN,
    MERGE_MUX_CLOSE,
    MERGE_MUX_OPEN,
    SFLOW_ALLOC_LABEL,
)

# kubectl logs -f bails with BadRequest on a not-yet-started container, so the
# apply command polls the pod phase first (READY_RETRIES * POLL_SLEEP_SECS budget)
# before the driver attaches the log stream. Status/exit-code polling and the
# early stop live driver-side in k8s.lifecycle, not in this bash.
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

# In-pod dependency gate for merge pods. A dependent member's subshell blocks in
# _sflow_gate until each named in-group dependency is met: either the driver touches
# <MERGE_GATE_DIR>/<dep>.open (the dependency reached READY -- only the driver knows
# a service is ready) OR the dependency member's own exit-code file shows success
# (COMPLETED, written in-pod). Fixed path so the driver's `kubectl exec ... touch`
# and the in-pod loop agree; /tmp is per-container so two merged pods on one node
# never collide.
MERGE_GATE_DIR = "/tmp/sflow-merge-gate"
MERGE_GATE_POLL_SECONDS = 2


def merge_gate_marker(dep_name: str) -> str:
    """Absolute path of the gate-open marker the driver touches for ``dep_name``."""
    return f"{MERGE_GATE_DIR}/{dep_name}.open"


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
    prefix: bool = True,
) -> Command:
    """Build the standalone ``kubectl logs -f`` Command for one pod.

    Run by the sflow driver as its own subprocess (see
    ``K8sContainerOperator.execute``) so its stdout streams live to ``<task>.log``
    through the launcher, and the driver can stop it the moment a background
    status-watch sees the pod go terminal -- instead of blocking on the K8s log
    backlog. The CLI-level global flags are prefixed directly (no shell wrapper).

    ``prefix`` toggles kubectl's ``[pod/<pod>/<container>]`` line prefix. Merge-pod
    streams pass ``prefix=False`` so each line begins with the launcher's own
    ``[[sflow-mux:<task>]] `` tag (the driver demuxes on it); everything else keeps
    the prefix for readable multi-pod logs.
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
    if prefix:
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


def apply_then_handoff_lines(
    apply_line: str,
    handoff_delete_pods: Sequence[str],
    ns_seg: str,
    *,
    handoff_before_apply: bool = False,
) -> list[str]:
    """Order the manifest ``apply`` and the placeholder-delete handoff.

    Default (``handoff_before_apply=False``) is **create-before-destroy**: apply the task
    pod (it goes Pending while the placeholder still holds the node's GPUs), then delete
    the placeholder so it binds with no delete->apply node-loss gap.

    ``handoff_before_apply=True`` is **destroy-before-create**: delete the placeholder
    FIRST (freeing its GPU request from the namespace ResourceQuota) and only then apply
    the task pod. Needed under a GPU ResourceQuota, where create-before-destroy
    double-counts the placeholder + task requests so the API rejects the task at admission
    ("exceeded quota"). Trade-off: a brief window where the node is unreserved.
    """
    delete_lines = handoff_delete_lines(handoff_delete_pods, ns_seg)
    if handoff_before_apply:
        return [*delete_lines, apply_line]
    return [apply_line, *delete_lines]


def merged_launcher_lines(
    members: Sequence[tuple[str, ...]],  # (name, cvd, script, env[, gate])
    *,
    preamble_lines: Sequence[str] = (),
) -> list[str]:
    """Entrypoint for a merge-pod: run each member's script as a background process.

    ``members`` is ``[(task_name, cuda_visible_devices, script_path, env_path)]``.
    All members share the one container (so they see every GPU -> NVLink/cuda_ipc),
    but each runs in its own subshell with its packed ``CUDA_VISIBLE_DEVICES`` and
    its own sourced env file. A member with an in-group dependency (a non-empty 5th
    ``gate`` tuple element) blocks in ``_sflow_gate`` until each dependency is met
    (its ``.open`` marker or a success rc file) before running; an empty ``gate``
    waits for nothing. Each member's combined stdout/stderr is tagged
    ``[[sflow-mux:<task>]] `` so the driver can demux the single container log into
    per-task ``<task>.log`` files. The launcher exits non-zero if any member does
    (its container exit code is the merge leader task's), and blocks in ``wait``
    for long-lived services until the pod is torn down. ``preamble_lines`` (e.g. the
    RDMA affinity preamble) run once in the parent shell before any member starts.

    Because that ``wait`` never returns while ANY member is a long-lived service,
    the pod's phase says nothing about an individual member. Each member therefore
    echoes ``[[sflow-member-done:<rc>]]`` on its own tagged stream as soon as its
    script returns, so the driver can resolve a finished one-shot member (e.g. the
    workflow's terminal task) while its sibling services keep the container alive.
    """
    tag_fmt = f"{MERGE_MUX_OPEN}%s{MERGE_MUX_CLOSE}%s\\n"
    lines: list[str] = ["set -uo pipefail"]
    lines.extend(list(preamble_lines))
    lines.extend(
        [
            f"_SFLOW_GATE_DIR={shlex.quote(MERGE_GATE_DIR)}",
            'mkdir -p "$_SFLOW_GATE_DIR"',
            '_sflow_rc_dir="$(mktemp -d)"',
            # Block until each named in-pod dependency is met: the driver touches
            # $_SFLOW_GATE_DIR/<dep>.open when it reaches READY, or the dependency's
            # own rc file shows 0 (COMPLETED). If a dependency FAILED (rc!=0), return
            # that rc so the gated member propagates the failure instead of running.
            "_sflow_gate() {",
            '  for _dep in "$@"; do',
            '    if [ ! -f "$_SFLOW_GATE_DIR/$_dep.open" ] '
            '&& [ ! -f "$_sflow_rc_dir/$_dep" ]; then',
            '      echo "sflow: $_name waiting for merged dependency $_dep..."',
            "    fi",
            "    while :; do",
            '      [ -f "$_SFLOW_GATE_DIR/$_dep.open" ] && break',
            '      if [ -f "$_sflow_rc_dir/$_dep" ]; then',
            '        _drc="$(cat "$_sflow_rc_dir/$_dep" 2>/dev/null || echo 1)"',
            # The rc file is written non-atomically (`echo "$?" > f` truncates then
            # writes), so a poll can read it EMPTY mid-write. Treat empty as "still
            # writing" and keep waiting -- an empty _drc must never reach
            # `return "$_drc"` (bash errors on a non-numeric arg, wrongly failing the
            # gated member as if its dependency had failed).
            f'        [ -z "$_drc" ] && {{ sleep {int(MERGE_GATE_POLL_SECONDS)}; '
            "continue; }",
            '        [ "$_drc" = 0 ] && break',
            '        echo "sflow: merged dependency $_dep failed (rc=$_drc); '
            '$_name will not start" >&2',
            '        return "$_drc"',
            "      fi",
            f"      sleep {int(MERGE_GATE_POLL_SECONDS)}",
            "    done",
            "  done",
            "  return 0",
            "}",
            "_sflow_run() {",
            '  _name="$1"; _cvd="$2"; _script="$3"; _envf="$4"; _gate="$5"',
            "  (",
            '    export CUDA_VISIBLE_DEVICES="$_cvd"',
            # Mirror onto NVIDIA_VISIBLE_DEVICES too -- some runtimes/newer GPU stacks
            # honor only that one; keep the member's slice identical across both.
            '    export NVIDIA_VISIBLE_DEVICES="$_cvd"',
            '    if [ -f "$_envf" ]; then . "$_envf"; fi',
            # _gate unquoted: empty string -> zero args -> _sflow_gate returns 0
            # immediately, so a non-dependent member behaves exactly as before.
            "    if _sflow_gate $_gate; then",
            '      bash -l "$_script"',
            "      _sflow_mrc=$?",
            "    else",
            # _sflow_gate returned the failed dependency's rc; record it as ours.
            "      _sflow_mrc=$?",
            "    fi",
            '    echo "$_sflow_mrc" > "$_sflow_rc_dir/$_name"',
            # Announce THIS member's completion on its own tagged stream. The pod's
            # container cannot exit while a sibling service still runs (the `wait`
            # below), so this marker is the only signal the driver has that a
            # one-shot member finished -- see MERGE_DONE_OPEN in k8s.render.
            f'    echo "{MERGE_DONE_OPEN}${{_sflow_mrc}}{MERGE_DONE_CLOSE}"',
            # `|| [ -n "$_sflow_line" ]`: emit a final line the script wrote without a
            # trailing newline (plain `read` returns non-zero and would drop it), so a
            # member's last line (e.g. a no-newline readiness marker) is never lost.
            '  ) 2>&1 | while IFS= read -r _sflow_line || [ -n "$_sflow_line" ]; do',
            f"      printf '{tag_fmt}' \"$_name\" \"$_sflow_line\"",
            "    done &",
            "}",
        ]
    )
    for member in members:
        name, cvd, script_path, env_path = member[0], member[1], member[2], member[3]
        gate = member[4] if len(member) > 4 else ""
        lines.append(
            "_sflow_run "
            f"{shlex.quote(name)} {shlex.quote(cvd)} "
            f"{shlex.quote(script_path)} {shlex.quote(env_path)} "
            f"{shlex.quote(gate)}"
        )
    lines.extend(
        [
            "wait",
            "_sflow_rc=0",
            'for _f in "$_sflow_rc_dir"/*; do',
            '  _c="$(cat "$_f" 2>/dev/null || echo 1)"',
            '  [ "$_c" = "0" ] || _sflow_rc="$_c"',
            "done",
            'exit "$_sflow_rc"',
        ]
    )
    return lines


def merged_env_secret_lines(
    secret_name: str,
    ns_seg: str,
    key_pairs: Sequence[tuple[str, str]],
    *,
    allocation_id: str | None = None,
) -> list[str]:
    """Lines writing one member's env to a single shell-sourceable Secret file.

    ``key_pairs`` is ``[(original_key, prefixed_shell_var)]``. Values are read from
    the (prefixed) process env at apply time via ``%q`` (never inlined into the
    manifest) and written as ``export KEY=<quoted>`` lines into a Secret's single
    ``envsh`` key, which the pod mounts as a file the member subshell sources. The
    prefix keeps merged members' identically-named vars from colliding in the one
    launcher process env.
    """
    lines = ["menv=$(mktemp)"]
    for orig, pref in key_pairs:
        lines.append(
            "printf 'export %s=%q\\n' "
            + shlex.quote(orig)
            + ' "${'
            + pref
            + '-}" >> "$menv"'
        )
    lines.append(
        f"kubectl create secret generic {shlex.quote(secret_name)}{ns_seg} "
        '--from-file=envsh="$menv" --dry-run=client -o yaml | kubectl apply -f -'
    )
    if allocation_id:
        lines.append(label_secret_cmd(secret_name, ns_seg, allocation_id))
    lines.append('rm -f "$menv"')
    return lines


def build_merged_apply_command(
    *,
    manifest_json: str,
    ns_seg: str,
    pod_name: str,
    member_env_secrets: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    handoff_delete_pods: Sequence[str],
    handoff_before_apply: bool = False,
    kubectl_global_args: Sequence[str] = (),
    allocation_id: str | None = None,
) -> Command:
    """Apply step for a merge-pod: per-member env Secrets, then the single pod.

    Like :func:`build_apply_command` but creates one env Secret per merged member
    (each mounted as a sourceable file, not a shared ``envFrom``) instead of one
    task-wide Secret, then applies the manifest (ConfigMap with the merged launcher
    + per-member scripts, optional union GPU ResourceClaimTemplate, one Pod), runs
    the create-before-destroy handoff, and waits for the single pod to start.
    """
    lines = ["set -euo pipefail"]
    lines.extend(kubectl_global_args_prelude(kubectl_global_args))
    for secret_name, key_pairs in member_env_secrets:
        lines.extend(
            merged_env_secret_lines(
                secret_name, ns_seg, key_pairs, allocation_id=allocation_id
            )
        )
    lines.extend(
        apply_then_handoff_lines(
            f"cat <<'{_MANIFEST_HEREDOC}' | kubectl apply -f -\n{manifest_json}\n{_MANIFEST_HEREDOC}",
            handoff_delete_pods,
            ns_seg,
            handoff_before_apply=handoff_before_apply,
        )
    )
    lines.extend(
        wait_for_pod_ready_lines(f"pod/{shlex.quote(pod_name)}", ns_seg, label=pod_name)
    )
    return bash_lc_command(lines)


def build_manifest_apply_command(
    *,
    manifest_json: str,
    ns_seg: str,
    secret_name: str | None = None,
    envs: Mapping[str, str] | None = None,
    handoff_delete_pods: Sequence[str] = (),
    handoff_before_apply: bool = False,
    kubectl_global_args: Sequence[str] = (),
    allocation_id: str | None = None,
) -> Command:
    """Apply a manifest via heredoc + run the GPU handoff, WITHOUT waiting on a pod.

    Used for the ``k8s_mpi`` operator route: the applied manifest carries an MPIJob
    CR whose launcher/worker pods are created by the mpi-operator controller and
    then discovered + watched by the sflow driver -- so, unlike
    :func:`build_apply_command`, there is no pod-name to poll for readiness here.
    Optionally creates the task env Secret first (same as the pod path), then
    applies the manifest and deletes the reserved node placeholders so the
    controller's Pending pods can bind onto the reserved GPUs.
    """
    lines = ["set -euo pipefail"]
    lines.extend(kubectl_global_args_prelude(kubectl_global_args))
    if secret_name is not None and envs:
        lines.append("env_file=$(mktemp)")
        lines.extend(secret_printf_lines(envs))
        lines.append(create_secret_cmd(secret_name, ns_seg))
        if allocation_id:
            lines.append(label_secret_cmd(secret_name, ns_seg, allocation_id))
        lines.append('rm -f "$env_file"')
    lines.extend(
        apply_then_handoff_lines(
            f"cat <<'{_MANIFEST_HEREDOC}' | kubectl apply -f -\n{manifest_json}\n{_MANIFEST_HEREDOC}",
            handoff_delete_pods,
            ns_seg,
            handoff_before_apply=handoff_before_apply,
        )
    )
    return bash_lc_command(lines)


def build_apply_command(
    *,
    manifest_json: str,
    ns_seg: str,
    pod_names: Sequence[str],
    secret_name: str | None,
    envs: Mapping[str, str],
    handoff_delete_pods: Sequence[str],
    handoff_before_apply: bool = False,
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
    # Apply the task pod(s) + run the placeholder handoff. Default create-before-destroy
    # (apply Pending pods, then free the reserved node(s) so they bind -- no node-loss gap);
    # destroy-before-create (handoff_before_apply) frees the placeholder's GPU quota FIRST so
    # a namespace ResourceQuota does not reject the task at admission. GPU tasks only (the
    # operator passes an empty list otherwise, so CPU-only tasks coexist with the placeholder).
    lines.extend(
        apply_then_handoff_lines(
            f"cat <<'{_MANIFEST_HEREDOC}' | kubectl apply -f -\n{manifest_json}\n{_MANIFEST_HEREDOC}",
            handoff_delete_pods,
            ns_seg,
            handoff_before_apply=handoff_before_apply,
        )
    )

    # Wait for each pod to leave Pending/ContainerCreating before returning, so
    # the driver's `kubectl logs -f` does not bail on a not-yet-started container.
    for pod_ref, pod_name in zip(pod_refs, pod_names, strict=True):
        lines.extend(wait_for_pod_ready_lines(pod_ref, ns_seg, label=pod_name))

    return bash_lc_command(lines)
