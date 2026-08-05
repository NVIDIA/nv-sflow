# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared base for the kubernetes container operator.

The single ``k8s`` operator (see ``k8s.py``) renders each task into its own
scheduler-placed pod(s): one pod for a single-node task, or N pods (one per
assigned node, leader = index 0) for a multi-node task. GPUs are requested via
DRA (``resource.k8s.io`` ResourceClaimTemplate) or the legacy ``nvidia.com/gpu``
device-plugin limit, selected by the backend ``scheduling`` field. The backend's
reserve+discover+pin context (namespace, assigned nodes, scheduling/DRA config,
node IPs, placeholder pods to hand off) is injected via ``apply_backend_context``.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shlex
import shutil
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import ConfigDict, field_validator

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.logging import get_logger
from sflow.plugins.k8s import lifecycle as k8s_lifecycle
from sflow.plugins.k8s.rdma_preamble import (
    RdmaRuntimeStatus,
    build_rdma_affinity_preamble,
    parse_rdma_runtime_status,
)
from sflow.plugins.k8s.render import (
    DEFAULT_GPU_TOLERATION,
    MERGE_DONE_CLOSE,
    MERGE_DONE_OPEN,
    SFLOW_ENTRYPOINT_FILE,
    SFLOW_SCRIPT_DIR,
    render_configmap,
    render_resource_claim_template,
    render_task_pod,
)
from sflow.plugins.k8s.shell import (
    MERGE_GATE_DIR,
    build_apply_command,
    build_log_stream_command,
    build_merged_apply_command,
    configmap_data_key,
    merge_gate_marker,
    merged_launcher_lines,
    namespace_segment,
    sanitize_name,
)
from sflow.utils.container import validate_container_image_reference
from sflow.utils.gpu import parse_cuda_visible_devices

_logger = get_logger(__name__)

# The in-pod RDMA decision marker is printed at container start, so only the
# log's leading bytes need scanning to detect an RDMA->TCP fallback -- bounding
# the read keeps this cheap even for tasks with large (e.g. pip/vllm) logs.
_RDMA_SCAN_MAX_BYTES = 256 * 1024


def _warn_manifest_overrides(task_name: str, conflicts: Sequence[str]) -> None:
    """Warn (once, de-duped) that user-provided k8s fields replaced sflow-managed
    manifest values.

    The curated-field / ``*_overrides`` merges in ``render_task_pod`` record each
    conflict where a user key clobbered a value sflow had set (env vars,
    securityContext, selector labels, or any overridden key), so an unintended
    override of an sflow-intended setting is surfaced instead of silently applied.
    """
    if not conflicts:
        return
    unique = list(dict.fromkeys(conflicts))
    _logger.warning(
        f"[{task_name}] user-provided k8s fields overrode sflow-managed manifest "
        f"values (sflow's intended settings were replaced): {'; '.join(unique)}"
    )


# Node-local output collection (K8s single-pod), via ``kubectl cp`` -- file BYTES never
# transit the log. The pod stages its ``<= cap`` output files into ONE tar.gz on the
# writable output emptyDir, prints the readiness marker to its log, and keeps its
# container alive (bounded) until the driver ``kubectl cp``s that tar.gz out and signals
# completion. (``kubectl logs`` splits lines at 16 KiB, which corrupts any large base64
# blob AND floods the console -- so the log is used only for the tiny readiness marker.)
_SFLOW_COLLECT_READY_MARKER = "[[sflow-collect-ready]]"  # pod -> log: staged, copy me
# pod -> log: the task produced nothing collectable, so no archive is coming. Without
# this the collector waits for a marker that will never be printed, and the driver
# cannot tell a healthy no-output task from output that was staged but never
# announced -- so every no-output task drew a "output was NOT collected" warning.
_SFLOW_COLLECT_NONE_MARKER = "[[sflow-collect-none]]"
_SFLOW_COLLECT_TGZ = ".sflow_collect.tgz"    # staged archive (under SFLOW_OUTPUT_DIR)
_SFLOW_COLLECT_DONE = ".sflow_collect_done"  # driver -> pod: cp done, you may exit
# Max seconds the pod keeps its container alive awaiting the driver's copy before exiting
# on its own (so a dead/slow driver can never hang the pod).
_SFLOW_COLLECT_GRACE_SECONDS = 120
# How often the WAITING pod RE-ANNOUNCES its readiness marker.
#
# The marker rides the pod's stdout, and the kubelet can DESTROY that channel outright:
# once a container log exceeds ``containerLogMaxSize`` (10 MiB default) the kubelet
# renames the whole file and starts an empty one, and ``kubectl logs`` / ``logs -f`` only
# ever serve the CURRENT file -- everything in the rotated file is unreachable forever.
# Measured on a GB300 cluster (kubelet v1.34.3): a pod that wrote ~30 MB left a
# 29,772,304-byte ``0.log.<ts>`` beside a 0-byte ``0.log``, and ``kubectl logs`` returned
# 0 bytes -- no task output, no marker. The rotation monitor polls rather than enforcing
# per-write, so one burst overshoots the cap and then the whole file is rotated at once.
#
# A single announcement is therefore not enough: the pod must keep saying it, so that a
# rotation which ate the first copy still leaves one in the POST-rotation log. Costs
# nothing (one line per interval, only while the handshake is unresolved) and keeps the
# trigger on the log channel, so the healthy path still execs nothing into a live pod.
_SFLOW_COLLECT_REANNOUNCE_SECONDS = 10
# How often the driver rescans the offloaded log for the readiness marker (local file
# reads only -- no kubectl -- so it is cheap for a task of any duration).
_SFLOW_COLLECT_POLL_INTERVAL = 1.0
# How often a still-waiting collector reports what it is waiting for. The collect is the
# one part of a task's lifecycle that can stall the DAG without any error, and it does so
# silently: a real run sat 20 minutes between its pod finishing and the workflow ending
# with nothing logged, so the post-mortem had to be reconstructed from file mtimes. These
# heartbeats make the next one self-explaining -- each line says whether <task>.log is
# still growing, which separates "task is simply still running" from "the log stream
# died" from "the marker was printed but never delivered".
_MARKER_WAIT_HEARTBEAT_S = 30.0
# Ceiling for the PRE-COPY pod-phase check ONLY. Much tighter than the usual poll
# timeout because that check runs INSIDE the pod's collect_grace_seconds countdown --
# a full-length stall there would eat the copy window before the copy starts.
#
# Scoped to that check ON PURPOSE. An earlier design also polled the running pod for its
# staged archive and reused this ceiling for it -- wrong on both counts: that poll raced
# no window, and a 5s cap only served to kill healthy `kubectl exec` calls mid-handshake
# (rc=124 against a serving TRT-LLM pod, which then died with SIGTERM). The collect is
# triggered from the log sentinels now and nothing execs into a running pod.
_SFLOW_COLLECT_PHASE_TIMEOUT = 5.0

# Default per-file cap for auto-collecting a K8s task's node-local output dir back to the
# driver. Files larger than this are skipped with a warning -- sync those via ``uploads:``
# / a PVC. Override per task with ``collect_max_file_size`` (0 disables collection).
_SFLOW_COLLECT_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB

# Bound on the cancel-path log finalize (``execute``'s ``finally``): a SIGINT'd run still
# sanitizes the streamed log before deleting the pods, but that best-effort pass must
# never let teardown hang -- so cap it and fall through to the pod delete on timeout.
_SFLOW_FINALIZE_ON_CANCEL_TIMEOUT = 30.0

# Once a task's pods are terminal, nothing else is driving the workflow: every second
# the post-terminal epilogue spends is a second the DAG is blocked on a task whose work
# is already done, and it spends them without logging anything. Two separate 20-minute
# investigations came down to "which of these four stages was it?" -- unanswerable from
# the artifacts, because the window was silent. Name any stage that outlasts this.
_EPILOGUE_WARN_S = 20.0

# How much of a merged member's log tail ``merged_member_exit_code`` scans for the
# member-done marker. The orchestrator polls it once per tick (1s) for every
# unresolved member, so this MUST stay O(1) in log size -- a serving member's log
# reaches megabytes. The marker is the member's final line, so a few KiB always
# covers it while keeping each check a single short read.
_MERGE_DONE_TAIL_BYTES = 8192


def _parse_size_bytes(value: int | str | None, default: int) -> int:
    """Parse a byte count from an int or a size string (``10Mi``, ``500K``, ``1G``).
    ``None``/empty -> ``default``; unparseable -> ``default``; negatives clamp to 0."""
    if value is None:
        return default
    if isinstance(value, int):
        return max(0, value)
    text = str(value).strip()
    if not text:
        return default
    units = {
        "Ki": 2**10, "Mi": 2**20, "Gi": 2**30,
        "K": 10**3, "M": 10**6, "G": 10**9,
    }
    for suffix, mult in units.items():
        if text.endswith(suffix):
            try:
                return max(0, int(float(text[: -len(suffix)]) * mult))
            except ValueError:
                return default
    try:
        return max(0, int(text))
    except ValueError:
        return default


def _sflow_pod_output_dir(envs: Mapping[str, str]) -> str | None:
    """The resolved ``SFLOW_OUTPUT_DIR`` (driver-host path) to mount a writable
    emptyDir at inside the pod, or ``None`` when unset. Its ``WORKFLOW``/``TASK``
    subdirs all live under it, so one mount makes the whole output tree writable at
    the SAME path the recipe/driver uses (no env remap needed)."""
    out = envs.get("SFLOW_OUTPUT_DIR")
    return out or None


def _sflow_pod_mkdir_preamble(envs: Mapping[str, str]) -> list[str]:
    """Entrypoint line that creates the per-task output dir before the user script.

    K8s mounts a writable emptyDir at the resolved ``SFLOW_OUTPUT_DIR`` (see the
    ``sflow_scratch_dir`` render arg), so the driver-host ``SFLOW_*`` paths are valid
    and writable in the pod WITHOUT remapping the env. Only the per-task subdir under
    that mount still needs creating.
    """
    if not envs.get("SFLOW_TASK_OUTPUT_DIR"):
        return []
    return [
        'mkdir -p "$SFLOW_TASK_OUTPUT_DIR" "$SFLOW_WORKFLOW_OUTPUT_DIR" 2>/dev/null '
        "|| true"
    ]


def _collect_exclude_rel(
    artifacts: Sequence[Any], workflow_dir: str | None
) -> list[str]:
    """``find``-relative paths (``./name``) of injected ``file://`` artifacts that live
    under the workflow output dir. These ConfigMap-mounted files are identical in EVERY
    task pod and are already written on the driver (see the local artifact resolver), so
    excluding them from each pod's collection scan avoids re-tarring/​re-copying the same
    shared files from every pod (de-dup the WORK; the driver's no-overwrite still de-dups
    the result). Deterministically sorted."""
    if not workflow_dir:
        return []
    wf = os.path.abspath(str(workflow_dir))
    rels: set[str] = set()
    for art in artifacts:
        uri = str(getattr(art, "uri", "") or "")
        if not uri.startswith("file://") or getattr(art, "content", None) is None:
            continue
        path = getattr(art, "path", None)
        if not path:
            continue
        ap = os.path.abspath(str(path))
        if ap != wf and ap.startswith(wf + os.sep):  # only if under the workflow dir
            rels.add(f"./{os.path.relpath(ap, wf)}")
    return sorted(rels)


def _sflow_output_collect_trap(
    max_bytes: int, grace_seconds: int, exclude_rel: Sequence[str] = ()
) -> str:
    """Register an ``EXIT`` trap that hands the node-local output to the driver via
    ``kubectl cp`` (no file bytes in the log). Prepended BEFORE the user script so it runs
    however the script ends -- normal completion, a failure, or an explicit ``exit`` in
    the recipe. (A trailing epilogue was silently skipped whenever the recipe ran ``exit``,
    e.g. the mlperf harness's ``exit 0``, so nothing was collected.)

    On exit, it scans the WHOLE ``$SFLOW_WORKFLOW_OUTPUT_DIR`` (the task dir is a subtree
    of it, plus workflow-level dirs a task writes such as aiperf's ``aiperf_concurrency_*``)
    for syncable files (skipping/​warning about files larger than ``max_bytes``, and skipping
    ``exclude_rel`` -- injected ``file://`` artifacts every pod shares and the driver already
    has), stages them into ONE tar.gz on the writable output emptyDir, prints the readiness
    marker, then keeps the container alive up to ``grace_seconds`` for the driver to
    ``kubectl cp`` the archive out and extract the ones missing on the host. While it
    waits it RE-ANNOUNCES that marker every ``_SFLOW_COLLECT_REANNOUNCE_SECONDS``, so a
    container-log rotation that discarded the first announcement cannot orphan the
    archive (see that constant). In the pod's
    emptyDir the workflow dir only holds files THIS pod created, so scanning it is safe.
    ``set +ex`` silences xtrace/errexit so a recipe's ``set -x``/``set -e`` can neither flood
    the log with the wait loop nor abort the block. The task's real exit code is preserved
    (``exit $__sflow_rc``).
    """
    big = f"+{int(max_bytes)}c"  # find: files strictly larger than max_bytes
    limit = int(max_bytes)
    grace = int(grace_seconds)
    marker = _SFLOW_COLLECT_READY_MARKER
    none_marker = _SFLOW_COLLECT_NONE_MARKER
    # Derived, not fixed: `collect_grace_seconds` is user-configurable, and a window
    # shorter than the interval would produce ZERO re-announcements -- silently removing
    # the rotation protection from exactly the configuration that can least afford to
    # lose the marker. Guarantee at least a few repeats inside any window, and never 0
    # (a modulus of 0 is a shell arithmetic error, not a skipped branch).
    reannounce = max(1, min(int(_SFLOW_COLLECT_REANNOUNCE_SECONDS), grace // 3))
    # Skip injected file:// artifacts (identical in every pod, already on the driver) so
    # shared files aren't re-tarred/​re-copied by every task pod.
    excl = "".join(f" ! -path {shlex.quote(p)}" for p in exclude_rel)
    # Staged archive + done-sentinel live at the output ROOT (a sibling of the run dir),
    # NOT under the workflow dir, so the scan below never picks them up.
    tgz = f'"$SFLOW_OUTPUT_DIR"/{_SFLOW_COLLECT_TGZ}'
    done = f'"$SFLOW_OUTPUT_DIR"/{_SFLOW_COLLECT_DONE}'
    scan = '"$SFLOW_WORKFLOW_OUTPUT_DIR"'
    return "\n".join(
        [
            "_sflow_collect() {",
            # Capture the triggering exit code and silence xtrace+errexit in one group
            # (stderr hidden) so neither this line nor the wait loop below is traced.
            "  { __sflow_rc=$?; set +ex; } 2>/dev/null",
            f'  if [ -n "$SFLOW_OUTPUT_DIR" ] && [ -d {scan} ]; then',
            f"    rm -f {tgz} {done} 2>/dev/null || true",
            # Warn (to the log) about files too large to sync back to the driver.
            f"    find {scan} -type f -size {big} 2>/dev/null"
            " | while IFS= read -r _f; do"
            f' echo "sflow: NOT syncing large output file back to driver'
            f' (> {limit} bytes): $_f" >&2; done',
            # Only collect when the pod actually created syncable files (skip empty dirs);
            # {excl} drops injected file:// artifacts already present on the driver.
            f'    _sflow_n=$( cd {scan} &&'
            f' find . -type f ! -size {big}{excl} 2>/dev/null | wc -l | tr -d " " )',
            '    if [ "${_sflow_n:-0}" -gt 0 ]; then',
            # Stage the <= cap files (whole workflow subtree) into ONE tar.gz on the
            # (writable) output emptyDir.
            f'      ( cd {scan} && find . -type f ! -size {big}{excl}'
            # Tar to a .part then RENAME so the archive is never observable while
            # tar is still writing it: a rename is atomic, a half-written tar is a
            # corrupt copy. The readiness marker below is only echoed after this
            # completes, so ordering already implies a whole file -- this also holds if
            # anything ever probes for the archive directly instead of the marker.
            f" -print0 2>/dev/null | tar czf {tgz}.part --null -T - 2>/dev/null )"
            f" && mv -f {tgz}.part {tgz} || true",
            # LOAD-BEARING blank line. `kubectl logs -f` reassembles CRI partial-line
            # entries and only emits once a line TERMINATES, so a task whose tail is an
            # unterminated `\r` progress bar (tqdm, aiperf, pip) freezes the follow: the
            # driver's <task>.log stops growing and everything printed afterwards --
            # including the sentinel below -- is withheld with it. Measured live on an
            # MLPerf harness: kubectl held 253,871 bytes containing 2,149 CRs and ZERO
            # newlines, so the marker only surfaced ~20 minutes later when the stream
            # finally closed, and the DAG waited that whole time. Emitting a newline
            # first closes the pending line and flushes it, so the sentinel is delivered
            # promptly instead of queueing behind a progress bar.
            '      echo ""',
            # Signal readiness (tiny log line -- NOT the file bytes), then wait, bounded,
            # for the driver to copy + acknowledge (touch the done-sentinel).
            '      echo "sflow: ${_sflow_n} output file(s) staged from'
            f' $SFLOW_WORKFLOW_OUTPUT_DIR; awaiting driver copy to the driver-side'
            f' filesystem {marker}"',
            "      _i=0",
            f'      while [ ! -f {done} ] && [ "$_i" -lt {grace} ]; do',
            "        sleep 1; _i=$((_i+1))",
            # Re-announce, because the announcement above may have been DESTROYED rather
            # than merely delayed: a container-log rotation makes everything written
            # before it unreachable to `kubectl logs` (see the constant). Each repeat
            # lands in the post-rotation log, so the handshake recovers within one
            # interval instead of the archive dying with the pod.
            f"        if [ $((_i % {reannounce})) -eq 0 ]; then",
            '          echo "sflow: still awaiting driver copy after ${_i}s;'
            " re-announcing in case a container-log rotation discarded the first"
            f' announcement {marker}"',
            "        fi",
            "      done",
            f'      if [ -f {done} ]; then echo "sflow: driver copy complete; exiting.";'
            f' else echo "sflow: driver copy window ({grace}s) elapsed; exiting."; fi',
            f"      rm -f {tgz} {done} 2>/dev/null || true",
            "    else",
            '      echo ""',  # same partial-line flush as above
            '      echo "sflow: no new in-pod output files in'
            f' $SFLOW_WORKFLOW_OUTPUT_DIR; nothing to collect back to the driver'
            f' {none_marker}"',
            "    fi",
            # The trap RAN but the output dir is unusable -- the preamble's `mkdir -p`
            # failed (read-only mount), or SFLOW_OUTPUT_DIR was unset by the recipe.
            # Announce it anyway: without a sentinel here the driver's collector waits
            # out the whole grace window and then reports "the container either died
            # before its EXIT trap ran, or the trap's output was never delivered" --
            # neither of which happened. Every path through this trap now emits exactly
            # one sentinel, so "no sentinel at all" keeps its precise meaning.
            "  else",
            '    echo ""',  # same partial-line flush as above
            '    echo "sflow: no usable in-pod output directory'
            f' ($SFLOW_WORKFLOW_OUTPUT_DIR); nothing to collect back to the driver'
            f' {none_marker}"',
            "  fi",
            "  exit $__sflow_rc",
            "}",
            "trap _sflow_collect EXIT",
        ]
    )


async def _wait_for_marker(
    log_path: str,
    marker: bytes,
    *,
    interval: float = _SFLOW_COLLECT_POLL_INTERVAL,
    task_name: str = "",
    heartbeat: float = _MARKER_WAIT_HEARTBEAT_S,
    stop_marker: bytes | None = None,
) -> bool:
    """Incrementally scan the offloaded ``<task>.log`` for ``marker`` (local reads only,
    no kubectl). Returns ``True`` once it appears; otherwise loops until the caller
    cancels it (which ``execute`` does the instant the pod is terminal). A small tail
    overlap catches a marker split across two reads.

    Emits a heartbeat every ``heartbeat`` seconds naming what it is still waiting for and
    whether ``<task>.log`` is still growing. That one fact separates the three ways this
    wait ends badly, which previously had to be inferred from file mtimes after the run:

    * log still growing        -> the task is simply still running (normal)
    * log stopped, pod alive   -> the stream stalled; the marker may be printed but
                                  undelivered (`kubectl logs -f` has been observed 20
                                  minutes behind), so the collect window may be closing
    * log never grew at all    -> the log stream never attached

    ``stop_marker`` is a second sentinel meaning "no archive is coming" (the task
    produced nothing collectable). Seeing it returns False, so the caller can stop
    cleanly instead of waiting out a marker that will never be printed.
    """
    pos = 0
    tail = b""
    keep = max(0, max(len(marker), len(stop_marker or b"")) - 1)
    started = time.monotonic()
    last_growth = started
    last_beat = started
    last_size = -1
    while True:
        chunk = b""
        try:
            with open(log_path, "rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
        except OSError:
            pass
        now = time.monotonic()
        if pos != last_size:
            last_size = pos
            last_growth = now
        if chunk:
            data = tail + chunk
            if marker in data:
                _logger.info(
                    f"[{task_name}] collect-ready marker seen after "
                    f"{now - started:.0f}s; copying the staged output archive"
                )
                return True
            if stop_marker is not None and stop_marker in data:
                _logger.info(
                    f"[{task_name}] task reported no collectable output after "
                    f"{now - started:.0f}s; nothing to copy back"
                )
                return False
            tail = data[-keep:] if keep else b""
        if now - last_beat >= heartbeat:
            last_beat = now
            quiet = now - last_growth
            state = (
                f"{log_path} is still growing ({pos} bytes) -- task likely still running"
                if quiet < heartbeat
                else f"{log_path} has not grown for {quiet:.0f}s ({pos} bytes) -- the "
                "log stream may have stalled, so a printed marker could go undelivered"
            )
            _logger.info(
                f"[{task_name}] still waiting for the collect-ready marker after "
                f"{now - started:.0f}s: {state}"
            )
        await asyncio.sleep(interval)


def _unpack_collected_tar(blob: bytes, dest: str) -> tuple[list[str], list[str]]:
    """Unpack a collected tar.gz into ``dest`` WITHOUT overwriting existing files.

    The pod's node-local task dir and the driver's task dir share the same path string
    and basenames but live on different hosts, so a file already present on the driver
    (e.g. one sflow wrote, or output from an earlier step) takes precedence and is NOT
    clobbered by the collected copy. Returns ``(extracted, skipped_existing)`` member
    basenames (``skipped_existing`` also covers any path-traversal entries defended
    against). Directories are created as needed; only regular files are written.
    """
    extracted: list[str] = []
    skipped: list[str] = []
    dest_abs = os.path.abspath(dest)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name[2:] if member.name.startswith("./") else member.name
            target = os.path.abspath(os.path.join(dest, member.name))
            # Path-traversal guard (the tar is ours, but stay defensive): the target
            # must land inside dest.
            if target != dest_abs and not target.startswith(dest_abs + os.sep):
                skipped.append(name)
                continue
            if os.path.exists(target):
                skipped.append(name)  # don't overwrite the driver-side file
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                src = tar.extractfile(member)
                if src is None:
                    continue
                with src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
            except Exception:
                continue
            extracted.append(name)
    return extracted, skipped


def _collect_summary_line(
    dest: str, extracted: Sequence[str], skipped: Sequence[str]
) -> str:
    """One informative log line describing what the driver collected from the pod."""
    line = (
        f"sflow: collected {len(extracted)} file(s) from the pod's node-local workflow "
        f"output dir into the driver-side filesystem at {dest}"
    )
    if skipped:
        shown = sorted(skipped)
        preview = ", ".join(shown[:5]) + (", ..." if len(shown) > 5 else "")
        line += (
            f"; kept {len(skipped)} existing driver file(s) (not overwritten): {preview}"
        )
    return line


@dataclass
class _K8sExecPlan:
    """The decoupled steps for one k8s task, built deterministically from
    ``task_name``/``script``/``envs`` (no cross-task instance state).

    ``execute`` runs ``apply_command`` (start the pod), then per pod streams
    ``log_stream_commands`` while watching pod status, and on terminal/teardown
    dumps the complete log to ``complete_log_paths`` and deletes ``cleanup_refs``.
    """

    apply_command: Command
    pod_refs: list[str]
    log_stream_commands: list[Command]
    # Single per-task log file all pods' `kubectl logs -f` append to (offload);
    # None when the task output dir is unknown (dry-run). The console tailer reads it.
    task_log_path: str | None
    cleanup_refs: list[str]
    global_args: list[str] = field(default_factory=list)
    ns_args: list[str] = field(default_factory=list)
    # Merge-pod mode: when set, the single pod runs several members' scripts and
    # its one (unprefixed) log stream is demuxed -- in the offloaded child shell, not
    # the driver -- into these per-task ``<task>.log`` files (task name -> path).
    # ``merge_launcher_env`` is the prefix-namespaced env the apply command reads to
    # build each member's isolated env Secret.
    merge_tag_paths: dict[str, str] | None = None
    merge_launcher_env: dict[str, str] | None = None
    # Leader task name used to tag pod-level (untagged) merge lines in the console
    # echo; per-member lines are tagged with their own task name by their tailer.
    merge_console_label: str | None = None
    # Operator route (k8s_mpi): the MPIJob CR name. When set, ``execute`` watches
    # the MPIJob + its launcher pod instead of the ``pod_refs`` list (which is
    # empty -- the mpi-operator, not sflow, creates the launcher/worker pods).
    mpijob_name: str | None = None
    # Single-pod K8s only: the entrypoint stages its node-local WORKFLOW output subtree
    # into a tar.gz and waits for the driver to ``kubectl cp`` it out. When True,
    # ``execute`` runs the cp-collector (see ``_collect_via_cp``) concurrently with the
    # pod-status watch.
    collect_output: bool = False
    # Multi-node MPI (pods route): the pods are one MPI COMM_WORLD. When True,
    # ``execute`` passes it to ``gather_pods_fail_fast`` so the task resolves the moment
    # ANY rank pod goes terminal (a finished/dead rank breaks the group) instead of
    # blocking on the survivors until teardown. Off for single-pod and non-MPI
    # multi-node (run-to-completion) tasks, which must await every pod.
    mpi_world_group: bool = False


class K8sContainerOperatorConfig(OperatorConfig):
    """Base config for the kubernetes container operator.

    The concrete subclass sets the ``type`` literal. ``namespace`` is
    intentionally absent: it is backend-owned and injected at runtime (one
    namespace per backend), so ``extra="forbid"`` makes setting it on an operator
    an error.
    """

    model_config = ConfigDict(extra="forbid")

    name: str

    image: str
    image_pull_policy: str | None = None
    restart: str = "Never"
    pass_envs: bool = True
    # Use host networking for the pod (pod IP == node IP). None means inherit
    # from the kubernetes backend at runtime; True/False overrides explicitly.
    host_network: bool | None = None
    # Constrain the pod to nodes matching these labels. None means inherit the
    # backend's node_selector at runtime.
    node_selector: dict[str, str] | None = None
    # DRA overrides (None -> inherit the backend's dra config). The DeviceClass
    # GPUs are requested from, and optional CEL selectors narrowing eligible
    # devices. Ignored under ``scheduling: device_plugin``.
    device_class: str | None = None
    device_selectors: list[str] | None = None
    # Pod tolerations. None -> inherit the backend's tolerations (default:
    # tolerate ``nvidia.com/gpu`` so pods can land on tainted GPU nodes).
    tolerations: list[dict[str, Any]] | None = None
    # Size cap for the pod's RAM-backed /dev/shm (e.g. "16Gi"). None -> a tmpfs
    # bounded by node memory (the K8s 64Mi default is too small for MPI/NCCL and
    # segfaults multi-GPU/multi-node jobs).
    shm_size: str | None = None
    # Force the container to run as root (securityContext.runAsUser/runAsGroup=0),
    # overriding an image whose default USER is non-root. Needed when the workload
    # must write a root-owned NFS/ceph PVC or bootstrap MPI over SSH (sshd host
    # keys, /run/sshd, ~/.ssh). Default False -> use the image's own user.
    run_as_root: bool = False
    # Allocate a pseudo-TTY for the pod container (sets container ``stdin`` + ``tty``).
    # Makes tools that redraw a line with ``\r`` (progress bars: aiperf/pip/docker
    # pull) stream live to ``<task>.log`` via ``kubectl logs``, instead of the
    # container runtime batching newline-less output. Trade-off: merges stderr into
    # stdout and keeps raw ``\r``/ANSI control bytes in the log. Applies to
    # single-pod container tasks; ignored for merge/co-located pods and the
    # ``k8s_mpi`` operator route. Default False.
    tty: bool = False

    # --- Additional Kubernetes pod/container fields --------------------------
    # Curated passthroughs for common v1.Container / v1.PodSpec fields that lack a
    # dedicated knob, plus two raw escape hatches. All are optional; when unset the
    # rendered manifest is unchanged. For the ``k8s_mpi`` operator these apply on
    # the pods route (base render); the MPIJob CR route does not consume them.
    #
    # Container-level:
    # Explicit container env vars as raw k8s ``EnvVar`` entries (list of dicts),
    # e.g. ``{name, value}`` or ``{name, valueFrom: {...}}`` -- the only way to use
    # ``valueFrom`` (fieldRef/secretKeyRef/configMapKeyRef). Merged after the
    # backend-injected env; on a name clash the user entry wins. Plain values are
    # usually better set via workflow ``variables``.
    env: list[dict[str, Any]] | None = None
    # Container ``workingDir``.
    working_dir: str | None = None
    # Container ``securityContext`` (deep-merged over sflow's managed
    # runAsUser/Group + IPC_LOCK; user keys win). E.g. ``privileged``,
    # ``allowPrivilegeEscalation``, ``readOnlyRootFilesystem``, ``runAsNonRoot``,
    # ``capabilities``, ``seccompProfile``. ``capabilities.add`` / ``capabilities.drop``
    # are UNION-merged with sflow's managed caps (so ``IPC_LOCK`` is always kept), not
    # replaced.
    security_context: dict[str, Any] | None = None
    # Container ``ports`` (list of ``ContainerPort`` dicts; informational).
    ports: list[dict[str, Any]] | None = None
    # Container ``lifecycle`` (``postStart`` / ``preStop`` handlers).
    lifecycle: dict[str, Any] | None = None
    #
    # Pod-level:
    # ``imagePullSecrets`` by name (for private registries).
    image_pull_secrets: list[str] | None = None
    # ``serviceAccountName`` (RBAC / cloud workload identity).
    service_account: str | None = None
    # ``runtimeClassName``.
    runtime_class: str | None = None
    # ``priorityClassName`` (scheduling priority / preemption).
    priority_class: str | None = None
    # ``terminationGracePeriodSeconds``.
    termination_grace_period: int | None = None
    # Extra pod ``metadata.labels`` (sflow's own task/allocation labels always win).
    labels: dict[str, str] | None = None
    # Extra pod ``metadata.annotations``.
    annotations: dict[str, str] | None = None
    # Pod-level ``securityContext`` (``fsGroup`` / ``supplementalGroups`` /
    # ``sysctls`` / ``seccompProfile``), deep-merged; user keys win.
    pod_security_context: dict[str, Any] | None = None
    #
    # Escape hatches (raw dicts). Deep-merged into the rendered container /
    # ``pod.spec`` AFTER all managed + curated fields, so they WIN. Use for any k8s
    # field without a dedicated knob above. WARNING: overriding sflow-managed keys
    # (container ``command``, ``resources.claims``, artifact ``volumeMounts``, the
    # env-secret ``envFrom``, the hostname ``nodeSelector`` pin) can break the task.
    container_overrides: dict[str, Any] | None = None
    pod_overrides: dict[str, Any] | None = None
    # Auto-collect the pod's node-local WORKFLOW output subtree back to the driver (K8s
    # only): the pod kubectl-cp's files up to this size to the driver, which unpacks the
    # ones missing on the host under ``$SFLOW_WORKFLOW_OUTPUT_DIR`` -- so both per-task
    # writes ($SFLOW_TASK_OUTPUT_DIR: result.json, CSVs) and workflow-level dirs a task
    # writes (e.g. aiperf_concurrency_*) land locally, and file-based ``result:`` works.
    # Larger files are skipped with a warning; sync those via ``uploads:`` / a PVC.
    # Accepts bytes (int) or a size string ("10Mi", "500K"); ``0`` disables. Default: 10 MiB.
    collect_max_file_size: int | str | None = None
    # Optional per-task CPU/memory for this task's pod(s). ``cpu``/``memory`` set the
    # Kubernetes *requests* (cgroup weight + scheduling reservation); ``cpu_limit``/
    # ``memory_limit`` set the optional hard caps (unset by default -> requests-only,
    # so CPU is shared dynamically and never CFS-throttled). When ``cpu`` is unset,
    # sflow injects a cpu request ONLY if the backend opts in via ``cpu_per_gpu`` /
    # ``cpu_request`` (GPU pods: cpu_per_gpu x per-pod GPUs; CPU-only pods:
    # cpu_request); otherwise no cpu request is set (pod runs BestEffort). Values may
    # be ints (CPU cores) or Kubernetes quantity strings (e.g. "500m", "16Gi"), incl.
    # ``${{ }}`` expressions.
    # These are the pod's container resources -- distinct from ``task.resources``
    # (the planner's node/GPU request); named flatly (cpu/memory) to avoid that
    # collision.
    cpu: int | str | None = None
    memory: str | None = None
    cpu_limit: int | str | None = None
    memory_limit: str | None = None

    def container_images(self) -> list[str]:
        return [self.image] if self.image else []

    @field_validator("image")
    @classmethod
    def image_must_be_valid(cls, value: str) -> str:
        type_field = cls.model_fields.get("type")
        type_name = (
            type_field.default
            if type_field is not None and isinstance(type_field.default, str)
            else "kubernetes"
        )
        validate_container_image_reference(
            value,
            source=f"{type_name} operator config: 'image'",
        )
        return value

    def runtime_warnings(self) -> list[str]:
        # The kubernetes operator always offloads: each pod's log is written
        # straight to <task>.log by `kubectl logs -f` (the sflow driver is never
        # in the per-line path), and a decoupled tailer streams it to the console.
        # So there is nothing to warn about re: offload support.
        return []


class K8sContainerOperator(Operator):
    """Render a task into pinned, scheduler-placed pod(s) and ``kubectl apply`` them."""

    def __init__(self, config: K8sContainerOperatorConfig):
        super().__init__(config)
        self.config: K8sContainerOperatorConfig = config
        self._image: str = config.image
        # Backend-injected context (see apply_backend_context).
        self._namespace: str | None = None
        self._node_count: int = 1
        self._assigned_node_names: list[str] = []
        self._assigned_node_ips: list[str] = []
        self._node_placement: bool = False
        self._scheduling: str = "dra"
        self._gpu_device_class: str = "gpu.nvidia.com"
        self._device_selectors: list[str] | None = config.device_selectors
        # Per-pod GPU count (resources.gpus.count // node_count); 0 == no GPUs.
        self._per_pod_gpus: int = 0
        # Planner-reserved node-local GPU slot for this pod, encoded as a
        # CUDA_VISIBLE_DEVICES string (e.g. "4,5,6,7"). k8s never injects it as
        # env (the device plugin/DRA picks the physical GPUs), but its first index
        # is the pod's node-local slot start -- used to align the RDMA NIC window.
        self._cuda_visible_devices: str | None = None
        self._host_network: bool = (
            bool(config.host_network) if config.host_network is not None else False
        )
        # Share the node IPC namespace + /dev/shm (cross-pod CUDA IPC / NVLink).
        # Backend-driven only (no operator-level override); set in apply_backend_context.
        self._host_ipc: bool = False
        self._node_selector: dict[str, str] | None = config.node_selector
        self._tolerations: list[dict[str, Any]] | None = config.tolerations
        self._shm_size: str | None = config.shm_size
        self._run_as_root: bool = bool(config.run_as_root)
        self._tty: bool = bool(config.tty)
        # Additional pod/container passthrough fields (see the config docstrings).
        self._env_vars: list[dict[str, Any]] | None = config.env
        self._working_dir: str | None = config.working_dir
        self._security_context: dict[str, Any] | None = config.security_context
        self._ports: list[dict[str, Any]] | None = config.ports
        self._lifecycle: dict[str, Any] | None = config.lifecycle
        self._image_pull_secrets: list[str] = config.image_pull_secrets or []
        self._service_account: str | None = config.service_account
        self._runtime_class: str | None = config.runtime_class
        self._priority_class: str | None = config.priority_class
        self._termination_grace_period: int | None = config.termination_grace_period
        self._labels: dict[str, str] | None = config.labels
        self._annotations: dict[str, str] | None = config.annotations
        self._pod_security_context: dict[str, Any] | None = config.pod_security_context
        self._container_overrides: dict[str, Any] | None = config.container_overrides
        self._pod_overrides: dict[str, Any] | None = config.pod_overrides
        # Node-local output-dir collection cap (bytes; 0 disables). See config docstring.
        self._collect_max_file_bytes: int = _parse_size_bytes(
            config.collect_max_file_size, _SFLOW_COLLECT_MAX_FILE_BYTES
        )
        # Max seconds the pod stays alive awaiting the driver's kubectl cp of its output.
        self._collect_grace_seconds: int = _SFLOW_COLLECT_GRACE_SECONDS
        # Backend master switch; False = no collect machinery in the pod at all.
        self._collect_enabled: bool = True
        # Pod CPU/memory (requests-only policy). The cpu/memory overrides come from
        # the operator's flat fields; memory/limits depend only on the override so
        # are resolved here, while the CPU *request* baseline (cpu_per_gpu /
        # cpu_request) is read from the backend in apply_backend_context.
        self._res_cpu: int | str | None = config.cpu
        self._cpu_limit: str | None = (
            None if config.cpu_limit is None else str(config.cpu_limit)
        )
        self._memory_request: str | None = (
            None if config.memory is None else str(config.memory)
        )
        self._memory_limit: str | None = (
            None if config.memory_limit is None else str(config.memory_limit)
        )
        # Backend CPU-request policy (overridden in apply_backend_context). None =>
        # unset, so no cpu request is injected unless the backend opts in.
        self._cpu_per_gpu: int | None = None
        self._cpu_request_default: int | None = None
        # Placeholder pods to delete on the create-before-destroy handoff. Only
        # populated for GPU tasks so CPU-only tasks coexist with the placeholder.
        self._handoff_pods: list[str] = []
        # ComputeDomain channel ResourceClaimTemplate (dra multi-node NVLink).
        self._compute_domain_channel: str | None = None
        # CLI-level kube access flags (from `sflow run`), prefixed onto every
        # kubectl call in the per-task wrapper; read from the backend below.
        self._kubectl_global_args: list[str] = []
        # `kubectl apply` subcommand flags (--extra-kubectl-apply-args).
        self._kubectl_apply_args: list[str] = []
        # Backend allocation id, stamped onto every task object as the allocation
        # label so the backend's label-selector sweep can delete them all.
        self._allocation_id: str | None = None
        # Resolved workflow artifacts (injected via apply_backend_context). Used to
        # mount file:// inline content (ConfigMap) and fs:// paths (hostPath) into
        # each task's pod(s) the K8s-native way.
        self._artifacts: list[Any] = []
        # Pre-existing PVC mounts declared on the backend (shared storage that fs://
        # artifacts can live on); injected via apply_backend_context.
        self._pvc_volumes: list[dict[str, Any]] = []
        # Network env (UCX/NCCL/gloo device + interface vars) the backend detected at
        # reservation; injected into task pod env so IB/NCCL/UCX/NIXL use the fast NICs.
        self._network_env: dict[str, Any] = {}
        # RDMA fast path (from the kubernetes backend): whether GPU pods get scoped
        # RDMA device access, and the per-node (resource_name, hca_name) NIC specs to
        # assign a per-pod slice from (sized to the pod's GPU count).
        self._rdma_enabled: bool = False
        self._rdma_nic_specs: list[tuple[str, str]] = []
        # GKE gIB libs + NCCL tuning script for multi-node NCCL over RDMA.
        self._rdma_lib_mounts: list[tuple[str, str]] = []
        self._rdma_nccl_env_script: str = ""
        # Host device dirs (e.g. /dev/infiniband) to hostPath-mount, and whether
        # RDMA pods need CAP_IPC_LOCK -- from the backend's resolved RDMA plan
        # (host-device provider grants verbs access this way, not via a resource).
        self._rdma_host_device_paths: list[str] = []
        self._rdma_ipc_lock: bool = False
        # Whether GPU pods pick their GPU-local RDMA NIC at runtime (host-device /
        # shared-device-plugin providers): the operator injects the affinity
        # preamble and skips the build-time per-pod NIC pin. See k8s.rdma_preamble.
        self._rdma_runtime_affinity: bool = False
        # DRA GPU<->NIC topology co-allocation (opt-in via backend dra.rdma_*).
        self._dra_rdma_device_class: str | None = None
        self._dra_rdma_match_attribute: str = "resource.kubernetes.io/pcieRoot"
        # Merge-pod mode (set by apply_merge_group when this task leads a merge
        # group): the ordered member Task objects sharing this leader's single pod,
        # and the union GPU count the one container requests. Empty => normal task.
        self._merge_members: list[Any] = []
        self._merge_union_gpus: int = 0

    def apply_merge_group(self, *, members: Sequence[Any], union_gpus: int) -> None:
        """Mark this operator as a merge-pod leader (see ``_plan_merge_groups``).

        ``members`` are the ordered member Task objects (leader first) whose scripts
        run as concurrent background processes in this leader's single container;
        ``union_gpus`` is the total GPUs that container requests so every member
        sees every GPU (NVLink/cuda_ipc). Their live ``script``/``envs`` and packed
        ``merge_cuda_visible_devices`` are read at build/execute time.
        """
        self._merge_members = list(members)
        self._merge_union_gpus = int(union_gpus)

    def merged_member_exit_code(self, task: Any) -> int | None:
        """This merged member's exit code once it announced completion, else None.

        A merge pod's container blocks in ``wait`` while ANY member is a long-lived
        service, so a finished one-shot member (notably the workflow's terminal task)
        is invisible in the pod's phase -- without this the run would hang forever.
        Each member echoes ``[[sflow-member-done:<rc>]]`` on its own tagged stream
        when its script returns (``k8s.shell.merged_launcher_lines``) and the demux
        lands it in that member's ``<task>.log``; parse the LAST marker (a retry
        appends to the same file). Returns None if the log or marker is not there
        yet -- the orchestrator just re-checks on its next tick.
        """
        envs = getattr(task, "envs", None) or {}
        name = getattr(task, "name", "")
        wf_out = envs.get("SFLOW_WORKFLOW_OUTPUT_DIR")
        if wf_out:
            path = Path(wf_out) / name / f"{name}.log"
        else:
            task_out = envs.get("SFLOW_TASK_OUTPUT_DIR")
            if not task_out:
                return None
            path = Path(task_out) / f"{name}.log"
        # Read only the TAIL, never the whole log: the orchestrator re-checks every
        # poll tick (1s) for every unresolved member, and a serving member's log runs
        # to megabytes -- a full read here would re-scan all of it, per member, per
        # second. The marker is the LAST thing a member writes (its subshell echoes it
        # after the script returns), so a small tail always contains it.
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - _MERGE_DONE_TAIL_BYTES))
                text = fh.read().decode(errors="replace")
        except OSError:
            return None
        rc: int | None = None
        for chunk in text.split(MERGE_DONE_OPEN)[1:]:
            end = chunk.find(MERGE_DONE_CLOSE)
            if end < 0:
                continue  # marker still mid-write; ignore this fragment
            try:
                rc = int(chunk[:end].strip())
            except ValueError:
                continue
        return rc

    async def open_merge_gate(self, dep_name: str) -> bool:
        """Open the in-pod gate for ``dep_name`` so a merged member waiting on it
        (via ``_sflow_gate``) proceeds. The orchestrator calls this when an in-group
        dependency reaches READY/COMPLETED. Reuses the release idiom used for
        node-local collect: ``kubectl exec ... touch <marker>``. Returns ``True`` on
        a successful exec (the orchestrator then stops re-touching); ``False`` if
        this operator is not a merge leader or the exec failed (retried next tick).
        """
        if not self._merge_members:
            return False
        pod = self._merged_pod_base()
        ns_args = (
            ["--namespace", self._namespace] if self._namespace else []
        )
        marker = merge_gate_marker(dep_name)
        rc, _out, err = await k8s_lifecycle.run_kubectl(
            [
                "exec", pod, *ns_args, "--", "sh", "-c",
                f"mkdir -p {shlex.quote(MERGE_GATE_DIR)} && "
                f"touch {shlex.quote(marker)}",
            ],
            global_args=list(self._kubectl_global_args),
            # Called from the ORCHESTRATOR's poll loop every tick, so an unbounded
            # exec here stalls the whole DAG, not just one task. Idempotent (mkdir -p
            # + touch), so a timeout is retried on the next tick.
            timeout=k8s_lifecycle.POLL_KUBECTL_TIMEOUT,
        )
        if rc != 0:
            _logger.debug(
                "open_merge_gate(%s) exec on pod %s rc=%s: %s",
                dep_name, pod, rc, err,
            )
        return rc == 0

    def apply_backend_context(
        self,
        *,
        backend: Any,
        assigned_nodes: Sequence[str],
        artifacts: Sequence[Any],
        cuda_visible_devices: str | None = None,
        gpu_count: int | None = None,
    ) -> None:
        self._namespace = getattr(backend, "namespace", None)
        self._node_count = max(len(assigned_nodes), 1)
        self._assigned_node_names = list(assigned_nodes or [])
        self._artifacts = list(artifacts or [])
        self._pvc_volumes = list(getattr(backend, "volumes", None) or [])
        self._network_env = {
            str(k): str(v) for k, v in (getattr(backend, "network_env", None) or {}).items()
        }
        self._rdma_enabled = bool(getattr(backend, "rdma_enabled", False))
        self._rdma_nic_specs = [
            (str(r), str(h))
            for r, h in (getattr(backend, "rdma_nic_specs", None) or [])
        ]
        self._rdma_lib_mounts = [
            (str(h), str(m))
            for h, m in (getattr(backend, "rdma_lib_mounts", None) or [])
        ]
        self._rdma_nccl_env_script = str(
            getattr(backend, "rdma_nccl_env_script", "") or ""
        )
        self._rdma_host_device_paths = [
            str(p) for p in (getattr(backend, "rdma_host_device_paths", None) or [])
        ]
        self._rdma_ipc_lock = bool(getattr(backend, "rdma_ipc_lock", False))
        self._rdma_runtime_affinity = bool(
            getattr(backend, "rdma_runtime_affinity", False)
        )
        dra_nic = getattr(backend, "dra_rdma_device_class", None)
        self._dra_rdma_device_class = str(dra_nic) if dra_nic is not None else None
        self._dra_rdma_match_attribute = str(
            getattr(backend, "dra_rdma_match_attribute", None)
            or "resource.kubernetes.io/pcieRoot"
        )
        self._scheduling = str(getattr(backend, "scheduling", "dra"))
        # Device-plugin GPU resource name (default nvidia.com/gpu); backend-driven so
        # every task pod on the backend requests GPUs under the same name.
        self._gpu_resource_name = str(
            getattr(backend, "gpu_resource_name", "nvidia.com/gpu")
        )
        self._gpu_device_class = self.config.device_class or str(
            getattr(backend, "gpu_device_class", "gpu.nvidia.com")
        )
        if self.config.device_selectors is not None:
            self._device_selectors = self.config.device_selectors
        else:
            self._device_selectors = getattr(backend, "device_selectors", None)

        # Explicit operator value wins; None inherits from the backend.
        if self.config.host_network is not None:
            self._host_network = self.config.host_network
        else:
            self._host_network = bool(getattr(backend, "host_network", False))
        self._host_ipc = bool(getattr(backend, "host_ipc", False))
        # CPU-request policy (requests-only, opt-in): a request is injected only when
        # the backend explicitly sets a knob -- GPU pods get cpu_per_gpu x per-pod
        # GPUs, CPU-only pods get cpu_request. Unset (None) => no request. See
        # _pod_cpu_request.
        cpu_per_gpu = getattr(backend, "cpu_per_gpu", None)
        self._cpu_per_gpu = int(cpu_per_gpu) if cpu_per_gpu is not None else None
        cpu_request = getattr(backend, "cpu_request", None)
        self._cpu_request_default = (
            int(cpu_request) if cpu_request is not None else None
        )
        # Node-local output-collection cap: operator override wins; else inherit the
        # backend default; else the built-in 10 MiB (set in __init__).
        if self.config.collect_max_file_size is None:
            backend_cap = getattr(backend, "collect_max_file_size", None)
            if backend_cap is not None:
                self._collect_max_file_bytes = _parse_size_bytes(
                    backend_cap, _SFLOW_COLLECT_MAX_FILE_BYTES
                )
        # Output-collection grace window (seconds): inherit the backend default
        # (its property already falls back to the built-in 120s).
        self._collect_grace_seconds = int(
            getattr(backend, "collect_grace_seconds", _SFLOW_COLLECT_GRACE_SECONDS)
        )
        self._collect_enabled = bool(
            getattr(backend, "collect_node_local_output", True)
        )
        if self.config.node_selector is not None:
            self._node_selector = self.config.node_selector
        else:
            self._node_selector = getattr(backend, "node_selector", None)
        if self.config.tolerations is not None:
            self._tolerations = self.config.tolerations
        else:
            self._tolerations = getattr(backend, "tolerations", None)

        self._node_placement = bool(
            getattr(
                getattr(backend, "capabilities", None),
                "supports_node_placement",
                False,
            )
        )

        # Per-pod GPU count: the planner's resources.gpus.count is a per-task
        # total; split it evenly across the assigned nodes (one pod per node).
        total_gpus = int(gpu_count) if gpu_count else 0
        if total_gpus and total_gpus % self._node_count != 0:
            raise ValueError(
                f"k8s operator '{self.config.name}': resources.gpus.count="
                f"{total_gpus} is not divisible by the {self._node_count} assigned "
                "node(s); request a multiple of the node count (each node's pod "
                "gets count/nodes GPUs, bounded by the backend's gpus_per_node)."
            )
        self._per_pod_gpus = (total_gpus // self._node_count) if total_gpus else 0
        self._cuda_visible_devices = cuda_visible_devices

        # Real node IPs (for multi-node leader/peer env wiring), discovered at
        # allocation time and carried on the backend allocation.
        self._assigned_node_ips = []
        alloc = getattr(backend, "allocation", None)
        self._allocation_id = getattr(alloc, "allocation_id", None) if alloc else None
        if alloc is not None:
            by_name = {n.name: n.ip_address for n in alloc.nodes}
            self._assigned_node_ips = [
                by_name.get(name, "") for name in self._assigned_node_names
            ]

        # Create-before-destroy handoff: the assigned node(s)' placeholder pods to
        # delete AFTER applying the (Pending) task pod. GPU tasks only -- CPU-only
        # tasks keep the placeholder (and the node's GPUs) reserved for the GPU
        # workloads that overlap on the same node.
        self._handoff_pods = []
        if self._node_placement and self._per_pod_gpus > 0:
            resolver = getattr(backend, "reservation_pod_for_node", None)
            if callable(resolver):
                for node_name in self._assigned_node_names:
                    pod = resolver(node_name)
                    if pod:
                        self._handoff_pods.append(pod)
        # Handoff order: destroy-before-create (delete placeholder BEFORE applying the task
        # pod) when the backend detected a GPU ResourceQuota, else create-before-destroy.
        self._handoff_destroy_first = bool(
            getattr(backend, "handoff_destroy_first", False)
        )

        # NVLink (MNNVL / IMEX): pods claim a ComputeDomain channel when the backend
        # created one. Independent of GPU scheduling -- device_plugin GPU pods can
        # also claim an IMEX channel (DRA ComputeDomain-only driver).
        self._compute_domain_channel = getattr(
            backend, "compute_domain_channel", None
        )

        # CLI-level kube access flags, applied to every kubectl call in the wrapper.
        self._kubectl_global_args = list(
            getattr(backend, "kubectl_global_args", []) or []
        )
        self._kubectl_apply_args = list(
            getattr(backend, "kubectl_apply_args", []) or []
        )

    def _effective_tolerations(self) -> list[dict[str, Any]]:
        if self._tolerations is not None:
            return [dict(t) for t in self._tolerations]
        return [dict(DEFAULT_GPU_TOLERATION)]

    def _pin_node(self, index: int) -> str | None:
        """Hostname to pin pod ``index`` onto, or None when placement is off."""
        if self._node_placement and index < len(self._assigned_node_names):
            return self._assigned_node_names[index]
        return None

    def _node_local_gpu_slot_start(self, replica_index: int) -> int:
        """First GPU index of this pod's node-local slot.

        The planner reserves each pod a contiguous GPU interval on its node and
        encodes it in ``cuda_visible_devices`` (e.g. ``"4,5,6,7"``) -- even on
        k8s, where the device plugin/DRA picks the physical GPUs, the interval is
        still computed for packing correctness. Its first index is the pod's
        node-local slot start (what disjoint packing is built on). Falls back to
        ``replica_index * per_pod_gpus`` when no slot was reserved (e.g. a dry-run
        without an allocation), preserving same-task replica separation.
        """
        slots = parse_cuda_visible_devices(self._cuda_visible_devices)
        if slots:
            return min(slots)
        return replica_index * self._per_pod_gpus

    def _rdma_pod_nics(
        self, gpu_slot_start: int, *, gpus: int | None = None
    ) -> tuple[list[str], list[str]]:
        """Per-pod RDMA NIC slice as ``(resource_names, hca_names)``.

        Assigns ``per_pod_gpus`` of the node's RDMA NIC *resources* starting at this
        pod's node-local GPU slot (``gpu_slot_start``), so the requested NIC window
        lines up with the GPUs the pod occupies (scheduling: co-located pods packed
        onto disjoint GPU slots get disjoint NIC windows). The ``hca_names`` are
        informational only (mirrored into ``SFLOW_RDMA_HCAS``); NIC *selection* is
        left entirely to NCCL/gIB + UCX -- sflow never pins ``NCCL_IB_HCA``. Returns
        ``([], [])`` when the RDMA fast path is off or the task requests no GPUs.
        Extended-resource names are de-duped and empty ones
        (host-device provider, which grants access via a device mount not a
        resource) are dropped, so the returned resource list is what the pod
        actually requests as limits. ``gpus`` overrides the slice size (merge-pod
        mode passes the union GPU count for its single container).
        """
        nics = len(self._rdma_nic_specs)
        gpus = self._per_pod_gpus if gpus is None else gpus
        if not (self._rdma_enabled and gpus > 0 and nics):
            return [], []
        if gpus >= nics:
            chosen = list(range(nics))
        else:
            offset = gpu_slot_start % nics
            chosen = [(offset + k) % nics for k in range(gpus)]
        specs = [self._rdma_nic_specs[j] for j in chosen]
        resources: list[str] = []
        seen: set[str] = set()
        for res, _hca in specs:
            if res and res not in seen:
                seen.add(res)
                resources.append(res)
        return resources, [hca for _res, hca in specs]

    def _rdma_all_nics(self) -> tuple[list[str], list[str]]:
        """Every node RDMA NIC as ``(resource_names, hca_names)`` -- for a pod that
        owns its node (a merged co-located pod).

        The per-pod NIC window (``_rdma_pod_nics``) exists only to keep several pods
        sharing one node off each other's NICs. A merged pod is the single pod for
        its co-located members, so there is nothing to carve against: expose all of
        the node's NICs and let NCCL/UCX select. Resource names are de-duped and
        empty ones (host-device provider grants verbs via a device mount, not a
        resource) dropped; returns ``([], [])`` when the RDMA fast path is off.
        """
        if not self._rdma_enabled:
            return [], []
        resources: list[str] = []
        seen: set[str] = set()
        for res, _hca in self._rdma_nic_specs:
            if res and res not in seen:
                seen.add(res)
                resources.append(res)
        return resources, [hca for _res, hca in self._rdma_nic_specs]

    def _gpu_driver_preamble(self, gpus: int) -> list[str]:
        """Shell lines making the node's NVIDIA driver loadable in the container.

        Device-plugin / NVIDIA-container-runtime setups (notably GKE) bind-mount
        the host driver into ``/usr/local/nvidia`` but do NOT add it to the loader
        path. Images that don't bake ``/usr/local/nvidia/lib64`` into
        ``LD_LIBRARY_PATH`` then can't resolve ``libcuda.so.1`` -- vLLM/torch fail
        with "Failed to infer device type" and ``nvidia-smi`` can't find
        ``libnvidia-ml.so``. Every GPU pod needs this (single-node included), so it
        is applied whenever the pod holds a GPU; the dir is prepended, preserving
        the image's own entries. No-op for GPU-less pods (frontend, etcd, nats).
        """
        if gpus <= 0:
            return []
        # Guard each prepend on the dir actually existing on the node: the host
        # driver mount at /usr/local/nvidia is a convention (nvidia-container-runtime
        # / GKE bind-mount), not guaranteed on every cluster. A missing dir on the
        # loader path is a silent no-op, but guarding keeps the env clean and mirrors
        # _gib_preamble's existence check (a cluster mounting the driver elsewhere
        # then simply relies on the image's own baked-in paths).
        return [
            "if [ -d /usr/local/nvidia/lib64 ]; then",
            "  export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}",
            "fi",
            "if [ -d /usr/local/nvidia/bin ]; then",
            "  export PATH=/usr/local/nvidia/bin:${PATH:-}",
            "fi",
        ]

    def _gib_preamble(self) -> list[str]:
        """Shell lines wiring GKE gIB NCCL config for a GPU pod on a gIB cluster.

        gIB is **workload-agnostic cluster infra**: whenever the installer is present
        (``set_nccl_env.sh`` exists) its net + tuner plugins auto-load into every GPU
        pod from the device-plugin driver dir and ABORT NCCL init unless
        ``NCCL_CONF_FILE`` is set. So this sources ``set_nccl_env.sh`` (which sets it)
        on EVERY GPU pod. When gIB is absent, emit NOTHING (NCCL uses its built-in
        transport + auto-selection).

        NIC selection is left ENTIRELY to NCCL/gIB (topology-aware, GPU-local) and UCX
        -- sflow never pins ``NCCL_IB_HCA``. Correct GPU<->NIC pairing comes from the
        pod owning the whole node (merged / full-node pods): every GPU's PCIe-local NIC
        is then present, so the library pairs them without sflow's help. (A partial-node
        pod cannot guarantee this -- the GPU device plugin picks the physical GPUs
        independently of the NIC grant -- so co-located GPU tasks should be *merged*
        into one full-node pod.) The NVIDIA driver libs go on ``LD_LIBRARY_PATH`` via
        ``_gpu_driver_preamble``, not here.
        """
        script = self._rdma_nccl_env_script
        if not script:
            return []
        return [f"if [ -f {script} ]; then", f"  source {script}", "fi"]

    def _extra_entrypoint_preamble(self) -> list[str]:
        """Extra bash lines prepended to the task entrypoint (subclass hook).

        Injected right before the user script -- after the gpu-driver / RDMA / gIB
        preambles -- so a subclass can add setup that must see those env vars and
        run before the workload. Base class: nothing. The ``k8s_mpi`` operator
        overrides this to inject its MPI bootstrap (sshd/keypair/hostfile/wait +
        the transparent ``mpirun`` wrapper) for the pods route.
        """
        return []

    def _covering_pvc(self, path_str: str) -> dict[str, Any] | None:
        """The configured PVC whose mount_path is ``path_str`` or a parent of it."""
        p = Path(path_str)
        for vol in self._pvc_volumes:
            # emptyDir volumes are ephemeral scratch, not a data source -- they
            # must NOT suppress the hostPath fallback for an artifact path.
            if vol.get("empty_dir") is not None:
                continue
            mount_path = vol.get("mount_path")
            if not mount_path:
                continue
            mp = Path(str(mount_path))
            if p == mp or mp in p.parents:
                return vol
        return None

    @staticmethod
    def _host_path_type(path_str: str) -> str:
        """Best-effort hostPath ``type`` for an fs:// path, via a controller stat.

        If the controller can see the path (typical when it lives on a shared
        filesystem mounted on both controller and nodes), pin the type so the
        kubelet rejects the pod with a clear error when a node lacks it -- instead
        of silently creating an empty dir. When the controller can't see it (a
        node-only path, or an output dir created at runtime), return "" so the
        kubelet stays lenient.
        """
        try:
            p = Path(path_str)
            if p.is_dir():
                return "Directory"
            if p.is_file():
                return "File"
        except OSError:
            pass
        return ""

    def _artifact_injection(
        self,
    ) -> tuple[
        dict[str, str],
        list[tuple[str, str]],
        list[tuple[str, str]],
        list[dict[str, Any]],
    ]:
        """K8s-native artifact wiring for a task's pod(s), from resolved artifacts.

        Returns ``(configmap_data, file_mounts, host_path_mounts, pvc_mounts)``:

        * ``configmap_data`` -- ``{key: inline_content}`` for ``file://`` artifacts
          declared with inline content; these become one ConfigMap mounted into the
          pod (so the content lives in the cluster, not on the controller's disk).
        * ``file_mounts`` -- ``[(in_pod_path, key)]`` subPath mounts for the above,
          placing each file at its resolved ``${{ artifacts.NAME.path }}`` location.
        * ``host_path_mounts`` -- ``[(node_path, hostpath_type)]`` for ``fs://``
          (and non-inline ``file://``) artifacts NOT served by a PVC, hostPath-
          mounted at the same path so a shared / node-local location is visible.
        * ``pvc_mounts`` -- the declared backend PVCs (deduped) to mount.

        ``artifacts`` is a workflow-level field, so like the Slurm path (whose
        ``local_artifact_mounts`` auto-mounts every artifact into every task) every
        workflow artifact is GLOBAL: each ``file://`` inline artifact (ConfigMap) and
        each ``fs://`` path (PVC or hostPath), plus the declared backend ``volumes:``
        PVCs, is mounted into EVERY task pod regardless of whether the task's script
        names it -- e.g. the dynamo frontend needs the model dir even though its
        script never references the path (it loads the model card from the path it
        discovers via etcd).
        """
        cm_data: dict[str, str] = {}
        file_mounts: list[tuple[str, str]] = []
        host_path_mounts: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        # Declared PVCs mount into every pod (shared workflow storage), deduped.
        pvc_by_name: dict[str, dict[str, Any]] = {}
        for vol in self._pvc_volumes:
            vol_name = sanitize_name(str(vol["name"]))
            pvc_by_name.setdefault(vol_name, {**vol, "name": vol_name})
        for art in self._artifacts:
            uri = str(getattr(art, "uri", "") or "")
            scheme = urlparse(uri).scheme.lower()
            if scheme not in ("file", "fs"):
                continue
            path = getattr(art, "path", None)
            if path is None:
                continue
            path_str = str(path)
            name = str(getattr(art, "name", "") or "")
            content = getattr(art, "content", None)
            if scheme == "file" and content is not None:
                key = configmap_data_key(name)
                cm_data[key] = content
                file_mounts.append((path_str, key))
                continue
            # A path served by a declared PVC needs no hostPath (the PVC provides it).
            if self._covering_pvc(path_str) is not None:
                continue
            if path_str not in seen_paths:
                seen_paths.add(path_str)
                host_path_mounts.append((path_str, self._host_path_type(path_str)))
        return cm_data, file_mounts, host_path_mounts, list(pvc_by_name.values())

    def _persist_rendered_manifest(
        self, manifest: dict, *, task_name: str, envs: Mapping[str, str]
    ) -> None:
        """Write the rendered List manifest to ``<task>.k8s.yaml`` for auditability.

        Best-effort and actual-run-only: only writes when ``SFLOW_TASK_OUTPUT_DIR``
        exists (created at launch, absent in dry-run). Never raises -- a debug
        artifact must not break a task launch. The env Secret is created separately
        (``kubectl create secret``) and is not part of this manifest, so no secret
        values are written here.
        """
        task_out = envs.get("SFLOW_TASK_OUTPUT_DIR")
        if not task_out or not os.path.isdir(task_out):
            return
        try:
            path = os.path.join(task_out, f"{task_name}.k8s.yaml")
            header = (
                "# Auto-generated by sflow: the manifest applied via "
                "`kubectl apply -f -` for this task.\n"
                "# The env Secret is created separately and is not included here.\n"
            )
            body = yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(header + body)
        except Exception as exc:  # never break a launch over a debug artifact
            _logger.debug(
                f"could not persist k8s manifest for '{task_name}': {exc}"
            )

    def _scoped_base(self, task_name: str) -> str:
        """Allocation-scoped DNS-1123 base for a task's pod + its objects
        (cfg/env/gpu/artifacts).

        Parallel runs share a namespace, so a name derived from the task name alone
        (e.g. ``frontend-server``) collides across runs and one run's ``kubectl
        apply`` clobbers/juggles another's pods. Appending the per-run allocation id
        (the same id the backend puts on its own ``sflow-res-...-<alloc>`` pods and
        the ``sflow.ai/allocation`` label) makes every object name unique per run.
        Falls back to the bare sanitized name when there is no allocation id (e.g.
        dry-run, where nothing is applied to a live cluster).
        """
        alloc = sanitize_name(self._allocation_id) if self._allocation_id else ""
        if not alloc:
            return sanitize_name(task_name)
        # Truncate the task portion so the "-<alloc>" scope AND the longest object
        # suffix ("-artifacts") always fit the 253-char DNS name budget -- the id
        # must survive so uniqueness holds even for very long task names.
        reserve = 1 + len(alloc) + len("-artifacts")
        head = sanitize_name(task_name, max_length=max(1, 253 - reserve))
        return f"{head}-{alloc}"

    def _merged_pod_base(self) -> str:
        """DNS-1123 base name for the merged pod + its objects (cfg/secrets/RCT).

        A merge pod runs several tasks, so naming it after the leader alone (e.g.
        ``decode-server-0``) is misleading in ``kubectl get pods``. Instead build
        ``merged-<distinct member base names>-<hash>`` -- the distinct base names
        (replicas collapse to their config task name, e.g. ``decode-server``) show
        what the pod runs, and a short stable hash of the concrete member set keeps
        it unique across nodes when the same task types are merged per node.
        """
        bases: list[str] = []
        seen: set[str] = set()
        for m in self._merge_members:
            raw = str(getattr(m, "base_name", None) or getattr(m, "name", ""))
            b = sanitize_name(raw)
            if b and b not in seen:
                seen.add(b)
                bases.append(b)
        label = "-".join(bases)[:48].strip("-") or "tasks"
        names = sorted(str(getattr(m, "name", "")) for m in self._merge_members)
        digest = hashlib.sha1("\n".join(names).encode()).hexdigest()[:6]
        # Allocation-scoped so parallel runs never collide on the merged pod name.
        return self._scoped_base(f"merged-{label}-{digest}")

    def _pod_cpu_request(self, pod_gpus: int) -> str | None:
        """CPU request (a Kubernetes quantity string) for a pod with ``pod_gpus`` GPUs.

        Requests-only policy, OPT-IN: the operator ``cpu`` override wins; otherwise a
        request is injected ONLY when the backend explicitly sets the relevant knob --
        a GPU pod uses ``cpu_per_gpu * pod_gpus`` (when ``cpu_per_gpu`` is set) and a
        pod without GPUs uses ``cpu_request`` (when set). When nothing is set, returns
        ``None`` so NO cpu request is emitted and the pod is unconstrained (BestEffort)
        by default. A knob set to 0 also yields ``None`` (explicit opt-out) -- this
        applies to the operator ``cpu`` override too, so ``cpu: 0`` opts out
        consistently with the backend ``cpu_per_gpu`` / ``cpu_request`` knobs.
        """
        if self._res_cpu is not None:
            text = str(self._res_cpu).strip()
            # An explicit 0 (or empty) opts out -- no cpu request -- matching the
            # backend knobs; any other value (cores or a quantity like "500m") stays.
            return None if text in ("", "0") else text
        if pod_gpus and int(pod_gpus) > 0:
            cores = (
                None
                if self._cpu_per_gpu is None
                else self._cpu_per_gpu * int(pod_gpus)
            )
        else:
            cores = self._cpu_request_default
        return str(cores) if cores and cores > 0 else None

    def _mnnvl_env_defaults(
        self, channel: str | None, envs: Mapping[str, str]
    ) -> dict[str, str]:
        """MNNVL (multi-node NVLink) transport enables for a pod joining a channel.

        A pod that claims an IMEX ComputeDomain channel (``channel``) sits on the
        rack's NVLink fabric, so both transports that can ride it should use it:

        * ``NCCL_MNNVL_ENABLE=1``         -- NCCL cross-node collectives over MNNVL
          (else NCCL falls back to intra-node NVLink + the network transport, and
          on a cluster with IB down that is slow TCP).
        * ``UCX_CUDA_IPC_ENABLE_MNNVL=y`` -- UCX cuda_ipc over MNNVL, i.e. NIXL
          KV-cache transfer (disaggregated serving) and MPI GPU transfers.

        Returned only when the pod actually joins a channel, and only for keys the
        workflow has NOT already set: any value already in ``envs`` (the recipe's
        ``UCX_CUDA_IPC_ENABLE_MNNVL`` variable, a ``-s NCCL_MNNVL_ENABLE=0``
        override, ...) lives in the env Secret and must win, so we skip it here and
        never clobber an explicit choice -- these only fill the gap when nothing set
        them. NOTE: enabling the transport is necessary but not sufficient; the
        framework's comm/KV buffers must also be fabric/VMM-capable for cross-node
        NVLink to actually carry the data.
        """
        if not channel:
            return {}
        return {
            k: v
            for k, v in (
                ("NCCL_MNNVL_ENABLE", "1"),
                ("UCX_CUDA_IPC_ENABLE_MNNVL", "y"),
            )
            if k not in envs
        }

    def _extra_pod_env_defaults(self) -> dict[str, str]:
        """Operator-set inline pod env defaults applied to every task pod this operator
        renders. The base sets none; subclasses (e.g. k8s_mpi) override to inject defaults
        such as OMP_NUM_THREADS. Separate from _mnnvl_env_defaults so it also applies when
        the pod claims no ComputeDomain channel."""
        return {}

    def _build_merged_execution_plan(
        self, *, task_name: str, envs: Mapping[str, str]
    ) -> _K8sExecPlan:
        """Build the plan for a merge-pod leader: one pod, one container, N members.

        The single container requests the union of the members' GPUs (so every
        member process sees every GPU -> NVLink/cuda_ipc) and runs the merged
        launcher, which starts each member's script as a background process with its
        packed ``CUDA_VISIBLE_DEVICES`` and its own sourced env file. Each member's
        output is tagged so the driver demuxes the single container log into per-task
        ``<task>.log`` files (``merge_tag_paths``). Member env is created as one
        Secret file per member from a prefix-namespaced launcher env
        (``merge_launcher_env``), avoiding a container-wide ``envFrom`` collision.
        """
        if not self._image:
            raise ValueError(
                f"k8s operator '{self.config.name}' has no image configured; set "
                "'image' on the operator (the kubernetes backend has no image)."
            )
        c = self.config
        # Name the pod (+ its ConfigMap/Secrets/RCT) after the merged members, not
        # the leader alone, so `kubectl get pods` isn't misleading. The <task>.log /
        # <task>.k8s.yaml still live under the leader task's output dir (task_name).
        base = self._merged_pod_base()
        members = self._merge_members
        union_gpus = int(self._merge_union_gpus)
        configmap_name = sanitize_name(f"{base}-cfg")
        rct_name = (
            sanitize_name(f"{base}-gpu")
            if self._scheduling == "dra" and union_gpus > 0
            else None
        )
        tolerations = self._effective_tolerations()

        # A merged pod is the only GPU pod on its node, so expose EVERY node NIC
        # (no per-pod window to carve out) and let NCCL/UCX pick -- see _rdma_all_nics.
        rdma_nic_resources, rdma_hcas = self._rdma_all_nics()
        dra_coalloc = bool(
            self._scheduling == "dra"
            and self._dra_rdma_device_class
            and union_gpus > 0
        )
        runtime_affinity = (
            bool(rdma_hcas) and self._rdma_runtime_affinity
        ) or dra_coalloc
        # These run once in the launcher's parent shell (env inherited by every member
        # subshell): first make the node's NVIDIA driver loadable (libcuda.so.1) for
        # the union GPUs, then -- if the gIB installer is present (workload-agnostic
        # infra; see the single-pod path) -- source set_nccl_env.sh so the auto-loaded
        # gIB plugins are configured. A merged pod is always single-node, so no
        # NCCL_IB_HCA pin. Then the RDMA affinity setup (non-GKE).
        preamble: list[str] = self._gpu_driver_preamble(union_gpus)
        merged_rdma_lib_mounts: list[tuple[str, str]] = []
        if union_gpus > 0 and self._rdma_lib_mounts:
            merged_rdma_lib_mounts = self._rdma_lib_mounts
            preamble = preamble + self._gib_preamble()
        if runtime_affinity:
            preamble = preamble + build_rdma_affinity_preamble(
                self._network_env.get("SFLOW_PRIMARY_IFACE", "")
            )

        # Per-member scripts, env Secrets, packed CVD, and demux targets -- read
        # live from the member Task objects (script/envs finalized by assembly +
        # configure_task_runtime).
        cm_data: dict[str, str] = {}
        launcher_members: list[tuple[str, str, str, str, str]] = []
        env_file_secrets: list[tuple[str, str]] = []
        member_env_secrets: list[tuple[str, list[tuple[str, str]]]] = []
        combined_env: dict[str, str] = {}
        merge_tag_paths: dict[str, str] = {}
        cleanup_secret_refs: list[str] = []
        for i, m in enumerate(members):
            m_name = str(getattr(m, "name"))
            # runnable_script carries the fail-fast prelude (set -e) for
            # shell members so a failed command in a merged member fails the pod;
            # falls back to raw script for anything not exposing the property.
            m_script = list(
                getattr(m, "runnable_script", None) or getattr(m, "script", []) or []
            )
            m_envs = {
                str(k): str(v)
                for k, v in dict(getattr(m, "envs", {}) or {}).items()
            }
            # K8s: a writable emptyDir is mounted at SFLOW_OUTPUT_DIR (below), so this
            # member's driver-host SFLOW_* paths are valid + writable in the pod (no env
            # remap). Just create this member's per-task subdir before its script runs
            # (shared with the single-pod path; a no-op when SFLOW_TASK_OUTPUT_DIR unset).
            m_script = _sflow_pod_mkdir_preamble(m_envs) + m_script
            cvd = str(getattr(m, "merge_cuda_visible_devices", "") or "")
            script_key = configmap_data_key(f"merge_{m_name}.sh")
            cm_data[script_key] = "\n".join(m_script)
            script_path = f"{SFLOW_SCRIPT_DIR}/{script_key}"
            env_path = f"{SFLOW_SCRIPT_DIR}/menv/{sanitize_name(m_name)}/envsh"
            gate = " ".join(
                str(g) for g in (getattr(m, "merge_gate_after", None) or [])
            )
            launcher_members.append((m_name, cvd, script_path, env_path, gate))
            prefix = f"SFMERGE{i}__"
            key_pairs = [(k, f"{prefix}{k}") for k in m_envs]
            if key_pairs:
                secret_name = sanitize_name(f"{base}-menv-{i}")
                member_env_secrets.append((secret_name, key_pairs))
                env_file_secrets.append((secret_name, env_path))
                cleanup_secret_refs.append(f"secret/{secret_name}")
                for k, v in m_envs.items():
                    combined_env[f"{prefix}{k}"] = v
            out = m_envs.get("SFLOW_TASK_OUTPUT_DIR")
            if out:
                merge_tag_paths[m_name] = os.path.join(out, f"{m_name}.log")

        launcher = merged_launcher_lines(launcher_members, preamble_lines=preamble)

        cm_art_data, file_mounts, host_path_mounts, pvc_mounts = (
            self._artifact_injection()
        )
        artifacts_cm_name = sanitize_name(f"{base}-artifacts") if cm_art_data else None

        items: list[dict[str, Any]] = [
            render_configmap(
                name=configmap_name,
                namespace=self._namespace,
                data={SFLOW_ENTRYPOINT_FILE: "\n".join(launcher), **cm_data},
                task_label=base,
                allocation_id=self._allocation_id,
            )
        ]
        if artifacts_cm_name is not None:
            items.append(
                render_configmap(
                    name=artifacts_cm_name,
                    namespace=self._namespace,
                    data=cm_art_data,
                    task_label=base,
                    allocation_id=self._allocation_id,
                )
            )
        if rct_name is not None:
            items.append(
                render_resource_claim_template(
                    name=rct_name,
                    namespace=self._namespace,
                    device_class=self._gpu_device_class,
                    count=union_gpus,
                    selectors=self._device_selectors,
                    task_label=base,
                    allocation_id=self._allocation_id,
                    nic_device_class=(
                        self._dra_rdma_device_class if dra_coalloc else None
                    ),
                    nic_count=union_gpus if dra_coalloc else None,
                    match_attribute=self._dra_rdma_match_attribute,
                )
            )

        extra_env: dict[str, str] = {}
        extra_env.update(self._network_env)
        # A merged pod is granted EVERY node NIC (see _rdma_all_nics), so there is no
        # per-pod window to pin: leave NCCL_IB_HCA unset and let NCCL/UCX auto-select
        # across all exposed NICs (matching how UCX device selection is always left
        # to the library).
        cd_channel = self._compute_domain_channel if union_gpus > 0 else None
        extra_env.update(self._mnnvl_env_defaults(cd_channel, envs))
        extra_env.update(self._extra_pod_env_defaults())

        conflicts: list[str] = []
        items.append(
            render_task_pod(
                pod_name=base,
                image=self._image,
                configmap_name=configmap_name,
                namespace=self._namespace,
                image_pull_policy=c.image_pull_policy,
                restart_policy=c.restart,
                env_secret_name=None,
                scheduling=self._scheduling,
                gpu_resource_name=self._gpu_resource_name,
                per_pod_gpus=union_gpus,
                resource_claim_name=rct_name,
                host_network=self._host_network,
                host_ipc=self._host_ipc,
                node_selector=self._node_selector,
                assigned_node=self._pin_node(0),
                tolerations=tolerations,
                extra_env=extra_env or None,
                compute_domain_channel=cd_channel,
                task_label=base,
                allocation_id=self._allocation_id,
                artifacts_configmap_name=artifacts_cm_name,
                file_artifact_mounts=file_mounts,
                host_path_mounts=host_path_mounts,
                pvc_mounts=pvc_mounts,
                shm_size=self._shm_size,
                run_as_root=self._run_as_root,
                sflow_scratch_dir=_sflow_pod_output_dir(envs),
                env_vars=self._env_vars,
                working_dir=self._working_dir,
                security_context=self._security_context,
                ports=self._ports,
                lifecycle=self._lifecycle,
                image_pull_secrets=self._image_pull_secrets,
                service_account=self._service_account,
                runtime_class=self._runtime_class,
                priority_class=self._priority_class,
                termination_grace_period=self._termination_grace_period,
                pod_labels=self._labels,
                pod_annotations=self._annotations,
                pod_security_context=self._pod_security_context,
                container_overrides=self._container_overrides,
                pod_overrides=self._pod_overrides,
                conflicts=conflicts,
                cpu_request=self._pod_cpu_request(union_gpus),
                cpu_limit=self._cpu_limit,
                memory_request=self._memory_request,
                memory_limit=self._memory_limit,
                rdma_nic_resources=rdma_nic_resources,
                rdma_ipc_lock=(bool(rdma_hcas) and self._rdma_ipc_lock) or dra_coalloc,
                rdma_host_device_paths=(
                    self._rdma_host_device_paths if rdma_hcas else []
                ),
                rdma_lib_mounts=merged_rdma_lib_mounts,
                env_file_secrets=env_file_secrets,
            )
        )
        _warn_manifest_overrides(task_name, conflicts)

        manifest = {"apiVersion": "v1", "kind": "List", "items": items}
        self._persist_rendered_manifest(manifest, task_name=task_name, envs=envs)

        ns_seg = namespace_segment(self._namespace)
        ns_args = ["--namespace", self._namespace] if self._namespace else []
        global_args = list(self._kubectl_global_args)
        pod_refs = [f"pod/{base}"]

        task_out = envs.get("SFLOW_TASK_OUTPUT_DIR")
        task_log_path = (
            os.path.join(task_out, f"{task_name}.log") if task_out else None
        )

        cleanup_refs = [*pod_refs, f"configmap/{configmap_name}"]
        if artifacts_cm_name is not None:
            cleanup_refs.append(f"configmap/{artifacts_cm_name}")
        cleanup_refs.extend(cleanup_secret_refs)
        if rct_name is not None:
            cleanup_refs.append(f"resourceclaimtemplate.resource.k8s.io/{rct_name}")

        apply_command = build_merged_apply_command(
            manifest_json=json.dumps(manifest, separators=(",", ":")),
            ns_seg=ns_seg,
            pod_name=base,
            member_env_secrets=member_env_secrets,
            handoff_delete_pods=self._handoff_pods,
            handoff_before_apply=self._handoff_destroy_first,
            kubectl_global_args=global_args,
            kubectl_apply_args=list(self._kubectl_apply_args),
            allocation_id=self._allocation_id,
        )
        log_stream_commands = [
            build_log_stream_command(
                pod_refs[0],
                ns_args=ns_args,
                kubectl_global_args=global_args,
                prefix=False,
            )
        ]
        return _K8sExecPlan(
            apply_command=apply_command,
            pod_refs=pod_refs,
            log_stream_commands=log_stream_commands,
            task_log_path=task_log_path,
            cleanup_refs=cleanup_refs,
            global_args=global_args,
            ns_args=ns_args,
            merge_tag_paths=merge_tag_paths or None,
            merge_launcher_env=combined_env,
            merge_console_label=task_name,
        )

    def _build_execution_plan(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> _K8sExecPlan:
        # Merge-pod leader: one pod runs every member's script (union GPUs).
        if self._merge_members:
            return self._build_merged_execution_plan(task_name=task_name, envs=envs)
        if not self._image:
            raise ValueError(
                f"k8s operator '{self.config.name}' has no image configured; set "
                "'image' on the operator (the kubernetes backend has no image)."
            )
        c = self.config
        # Allocation-scoped so parallel runs in one namespace never collide (the
        # pod NAME and its cfg/env/gpu/artifacts objects all derive from `base`).
        base = self._scoped_base(task_name)
        n = self._node_count
        pod_names = [base] if n == 1 else [f"{base}-{i}" for i in range(n)]
        configmap_name = sanitize_name(f"{base}-cfg")
        secret_name = sanitize_name(f"{base}-env")
        use_secret = bool(c.pass_envs and envs)
        rct_name = (
            sanitize_name(f"{base}-gpu")
            if self._scheduling == "dra" and self._per_pod_gpus > 0
            else None
        )
        tolerations = self._effective_tolerations()

        # Per-pod RDMA NIC slice (scoped device resources + matching UCX/NCCL env),
        # sized to the pod's GPU count and aligned to its node-local GPU slot so
        # co-located pods (e.g. prefill + decode packed onto one node) never
        # collide on the node's NICs. Empty unless the backend enabled RDMA.
        try:
            replica_index = int(envs.get("SFLOW_REPLICA_INDEX", "0") or 0)
        except (TypeError, ValueError):
            replica_index = 0
        gpu_slot_start = self._node_local_gpu_slot_start(replica_index)
        rdma_nic_resources, rdma_hcas = self._rdma_pod_nics(gpu_slot_start)
        # DRA GPU<->NIC topology co-allocation (opt-in): the task GPU claim also
        # requests a NIC on the same PCIe root, and the pod needs CAP_IPC_LOCK to
        # pin memory for verbs. Independent of the device-plugin `rdma` providers.
        dra_coalloc = bool(
            self._scheduling == "dra"
            and self._dra_rdma_device_class
            and self._per_pod_gpus > 0
        )
        # Runtime NIC handling: the physical GPU is chosen by the device plugin/DRA,
        # so for providers where the pod sees every node HCA (host-device / shared
        # plugin) or a DRA co-allocated NIC, we defer NIC selection to the in-pod
        # preamble. By default it exposes all NICs and lets NCCL/UCX auto-select the
        # GPU-closest device (SFLOW_RDMA_AFFINITY=auto); it verifies RDMA is usable
        # and falls back to TCP otherwise (never forcing a dead HCA -> NIXL_ERR_BACKEND).
        # GKE is excluded (allow_runtime_affinity=False): it grants a fixed per-pod
        # NIC subset, so it keeps the build-time pin + gIB preamble below.
        runtime_affinity = (
            bool(rdma_hcas) and self._rdma_runtime_affinity
        ) or dra_coalloc
        # gIB setup is WORKLOAD-AGNOSTIC cluster infra: when the installer is present
        # (the RDMA plan carries lib mounts + the NCCL env script), the gIB net +
        # tuner plugins auto-load into EVERY GPU pod from the device-plugin driver dir
        # (/usr/local/nvidia/lib64) and ABORT NCCL init if unconfigured. So mount the
        # gIB config and source set_nccl_env.sh (-> NCCL_CONF_FILE) on every GPU pod,
        # regardless of node count. NIC selection is left to NCCL/gIB + UCX (see
        # _gib_preamble) -- sflow never pins NCCL_IB_HCA.
        script = list(script)
        # Subclass hook (k8s_mpi pods route): the MPI bootstrap must sit right
        # before the user script, so it runs AFTER the gpu-driver/RDMA/gIB
        # preambles prepended below (prepending it first puts it last of the three).
        mpi_preamble = self._extra_entrypoint_preamble()
        if mpi_preamble:
            script = list(mpi_preamble) + script
        rdma_lib_mounts: list[tuple[str, str]] = []
        if rdma_hcas and self._rdma_lib_mounts:
            rdma_lib_mounts = self._rdma_lib_mounts
            script = self._gib_preamble() + script
        elif runtime_affinity:
            # Prepend the verified, topology-aware NIC selection (sets NCCL_IB_HCA
            # in-pod when explicit; on a dead HCA it only HINTS the socket-forcing
            # envs, never sets them, so the rack-NVLink/MNNVL path is preserved).
            # Takes over from the build-time per-pod NIC pin below. UCX is left unset.
            script = (
                build_rdma_affinity_preamble(
                    self._network_env.get("SFLOW_PRIMARY_IFACE", "")
                )
                + script
            )
        # Make the node's NVIDIA driver loadable (libcuda.so.1) on every GPU pod,
        # not just multi-node RDMA ones. Runs first so it precedes any RDMA/NCCL
        # preamble above and the task script below.
        script = self._gpu_driver_preamble(self._per_pod_gpus) + script

        # K8s: the driver's host output dir isn't reachable inside the pod, so a writable
        # emptyDir is mounted at the resolved SFLOW_OUTPUT_DIR (render arg below) and the
        # entrypoint mkdir -p's the per-task subdir before the user script (env unchanged).
        # Output collection registers an EXIT trap (BEFORE the user script) that, on exit,
        # stages the pod's whole SFLOW_WORKFLOW_OUTPUT_DIR subtree into a tar.gz and keeps
        # the container alive until the driver ``kubectl cp``s it back and extracts the
        # files missing on the host workflow dir (files over the size cap are skipped with
        # a warning). The trap -- not a trailing command -- is what makes it run even when
        # the recipe calls ``exit`` (e.g. the harness's ``exit 0``). Only
        # when the env Secret (which carries the SFLOW_* paths) is injected;
        # ``collect_output`` tells ``execute`` to run the driver-side cp-collector
        # concurrently with the pod-status watch.
        collect_output = False
        if use_secret:
            preamble = _sflow_pod_mkdir_preamble(envs)
            # kubectl-cp collection is single-pod only: the driver copies from ONE pod,
            # so arming the wait-trap on a multi-node pod-set would make the workers
            # (never copied) idle for the full grace before exiting.
            # ``collect_grace_seconds: 0`` means the pod does not wait for the driver
            # at all, so there is no window in which a copy could ever land. Arming the
            # trap anyway would stage an archive nobody collects and hand the driver a
            # 0s copy budget (an instant timeout reported as "did not finish within
            # 0s", which reads like a slow copy rather than a disabled feature). Treat
            # it as the opt-out it is: skip the collect on both sides.
            # ``collect_node_local_output: false`` on the backend turns the whole
            # mechanism off -- no EXIT trap in the pod, no driver-side collector. Task
            # completion then depends only on pod status, readiness probes and the
            # merge-pod done-marker, with nothing exec'ing into a running pod.
            if (
                n == 1
                and self._collect_enabled
                and self._collect_max_file_bytes > 0
                and self._collect_grace_seconds > 0
                and envs.get("SFLOW_OUTPUT_DIR")
            ):
                # Exclude injected file:// artifacts (shared by every pod, already on the
                # driver) from the scan so they aren't re-tarred/​re-copied per task.
                exclude_rel = _collect_exclude_rel(
                    self._artifacts, envs.get("SFLOW_WORKFLOW_OUTPUT_DIR")
                )
                preamble = preamble + [
                    _sflow_output_collect_trap(
                        self._collect_max_file_bytes,
                        self._collect_grace_seconds,
                        exclude_rel,
                    )
                ]
                collect_output = True
            # KNOWN LIMITATION when the collect trap is NOT armed (`collect_node_local
            # _output: false`, `collect_grace_seconds: 0`, multi-node pod-sets): nothing
            # emits the newline that flushes a dangling partial line, so a task ending
            # mid-progress-bar leaves its tail undelivered until the stream closes.
            # Deliberately NOT fixed by injecting a flush trap into every task: task
            # completion in those configurations is driven by pod status rather than the
            # log, so the practical cost is a late console line, and the fix would add an
            # EXIT trap (silently superseded by any trap the user's own script registers)
            # to every rendered manifest.
            script = preamble + script

        # K8s-native artifact injection (file:// -> ConfigMap, fs:// -> PVC/hostPath).
        cm_data, file_mounts, host_path_mounts, pvc_mounts = self._artifact_injection()
        artifacts_cm_name = sanitize_name(f"{base}-artifacts") if cm_data else None

        items: list[dict[str, Any]] = [
            render_configmap(
                name=configmap_name,
                namespace=self._namespace,
                data={SFLOW_ENTRYPOINT_FILE: "\n".join(script)},
                task_label=base,
                allocation_id=self._allocation_id,
            )
        ]
        if artifacts_cm_name is not None:
            items.append(
                render_configmap(
                    name=artifacts_cm_name,
                    namespace=self._namespace,
                    data=cm_data,
                    task_label=base,
                    allocation_id=self._allocation_id,
                )
            )
        if rct_name is not None:
            items.append(
                render_resource_claim_template(
                    name=rct_name,
                    namespace=self._namespace,
                    device_class=self._gpu_device_class,
                    count=self._per_pod_gpus,
                    selectors=self._device_selectors,
                    task_label=base,
                    allocation_id=self._allocation_id,
                    # Co-allocate a PCIe-root-aligned NIC in the same claim (opt-in).
                    nic_device_class=(
                        self._dra_rdma_device_class if dra_coalloc else None
                    ),
                    nic_count=self._per_pod_gpus if dra_coalloc else None,
                    match_attribute=self._dra_rdma_match_attribute,
                )
            )
        conflicts: list[str] = []
        for i, pod_name in enumerate(pod_names):
            extra_env: dict[str, str] = {}
            # RDMA fast path: steer IB/NCCL/UCX/NIXL onto the detected RDMA NICs +
            # control interface (and expose SFLOW_* mirrors for explicit use). The
            # task script can still override any of these via `export`.
            extra_env.update(self._network_env)
            # Only GPU pods join the NVLink (IMEX) ComputeDomain. The driver
            # publishes ONE channel per node, single-allocation (one channel-claiming
            # pod per node per ComputeDomain), so CPU-only infra pods
            # (nats/etcd/frontend/download) must NOT claim it -- otherwise co-located
            # pods contend for the node's single channel and all but the first fail
            # scheduling ("cannot allocate all claims"). GPU workers still need a
            # one-pod-per-node topology.
            cd_channel = (
                self._compute_domain_channel if self._per_pod_gpus > 0 else None
            )
            extra_env.update(self._mnnvl_env_defaults(cd_channel, envs))
            extra_env.update(self._extra_pod_env_defaults())
            # NIC selection (NCCL_IB_HCA) is deliberately NOT pinned by sflow: it is
            # left to NCCL/gIB (topology-aware, GPU-local) and UCX, matching how UCX
            # device selection is always left to the library. A grant-based pin can
            # misalign a pod's GPUs onto far NICs (the GPU device plugin picks GPUs
            # independently of the rdma-N grant), so correct pairing instead comes from
            # owning the whole node (merged / full-node pods). See _gib_preamble.
            if n > 1:
                extra_env["SFLOW_TASK_NODE_INDEX"] = str(i)
                if self._assigned_node_ips:
                    extra_env["SFLOW_LEADER_ADDRESS"] = self._assigned_node_ips[0]
            items.append(
                render_task_pod(
                    pod_name=pod_name,
                    image=self._image,
                    configmap_name=configmap_name,
                    namespace=self._namespace,
                    image_pull_policy=c.image_pull_policy,
                    restart_policy=c.restart,
                    env_secret_name=secret_name if use_secret else None,
                    scheduling=self._scheduling,
                    gpu_resource_name=self._gpu_resource_name,
                    per_pod_gpus=self._per_pod_gpus,
                    resource_claim_name=rct_name,
                    host_network=self._host_network,
                    host_ipc=self._host_ipc,
                    node_selector=self._node_selector,
                    assigned_node=self._pin_node(i),
                    tolerations=tolerations,
                    extra_env=extra_env or None,
                    compute_domain_channel=cd_channel,
                    task_label=base,
                    allocation_id=self._allocation_id,
                    artifacts_configmap_name=artifacts_cm_name,
                    file_artifact_mounts=file_mounts,
                    host_path_mounts=host_path_mounts,
                    pvc_mounts=pvc_mounts,
                    shm_size=self._shm_size,
                    run_as_root=self._run_as_root,
                    tty=self._tty,
                    env_vars=self._env_vars,
                    working_dir=self._working_dir,
                    security_context=self._security_context,
                    ports=self._ports,
                    lifecycle=self._lifecycle,
                    image_pull_secrets=self._image_pull_secrets,
                    service_account=self._service_account,
                    runtime_class=self._runtime_class,
                    priority_class=self._priority_class,
                    termination_grace_period=self._termination_grace_period,
                    pod_labels=self._labels,
                    pod_annotations=self._annotations,
                    pod_security_context=self._pod_security_context,
                    container_overrides=self._container_overrides,
                    pod_overrides=self._pod_overrides,
                    conflicts=conflicts,
                    sflow_scratch_dir=(
                        _sflow_pod_output_dir(envs) if use_secret else None
                    ),
                    cpu_request=self._pod_cpu_request(self._per_pod_gpus),
                    cpu_limit=self._cpu_limit,
                    memory_request=self._memory_request,
                    memory_limit=self._memory_limit,
                    rdma_nic_resources=rdma_nic_resources,
                    # IPC_LOCK + host device mounts apply to pods that got an RDMA
                    # NIC slice (device-plugin/host-device providers) or a DRA
                    # co-allocated NIC -- both need CAP_IPC_LOCK to pin verbs memory.
                    rdma_ipc_lock=(bool(rdma_hcas) and self._rdma_ipc_lock)
                    or dra_coalloc,
                    rdma_host_device_paths=(
                        self._rdma_host_device_paths if rdma_hcas else []
                    ),
                    rdma_lib_mounts=rdma_lib_mounts,
                )
            )
        _warn_manifest_overrides(task_name, conflicts)

        manifest = {"apiVersion": "v1", "kind": "List", "items": items}
        self._persist_rendered_manifest(manifest, task_name=task_name, envs=envs)

        ns_seg = namespace_segment(self._namespace)
        ns_args = ["--namespace", self._namespace] if self._namespace else []
        global_args = list(self._kubectl_global_args)
        pod_refs = [f"pod/{p}" for p in pod_names]

        # The per-task log file each pod's `kubectl logs -f` is redirected to
        # (offload -- the driver never processes these lines). One file per task;
        # multi-pod pods append (O_APPEND) with their `[pod/...]` prefix. None when
        # the task output dir is unknown (dry-run).
        task_out = envs.get("SFLOW_TASK_OUTPUT_DIR")
        task_log_path = os.path.join(task_out, f"{task_name}.log") if task_out else None

        # Objects this task owns, deleted by name when it ends (the kubernetes
        # backend's allocation-label sweep is the backstop on Ctrl+C / crash).
        cleanup_refs = [*pod_refs, f"configmap/{configmap_name}"]
        if artifacts_cm_name is not None:
            cleanup_refs.append(f"configmap/{artifacts_cm_name}")
        if use_secret:
            cleanup_refs.append(f"secret/{secret_name}")
        if rct_name is not None:
            cleanup_refs.append(
                f"resourceclaimtemplate.resource.k8s.io/{rct_name}"
            )

        apply_command = build_apply_command(
            manifest_json=json.dumps(manifest, separators=(",", ":")),
            ns_seg=ns_seg,
            pod_names=pod_names,
            secret_name=secret_name if use_secret else None,
            envs=envs,
            handoff_delete_pods=self._handoff_pods,
            handoff_before_apply=self._handoff_destroy_first,
            kubectl_global_args=global_args,
            kubectl_apply_args=list(self._kubectl_apply_args),
            allocation_id=self._allocation_id,
        )
        log_stream_commands = [
            build_log_stream_command(
                ref, ns_args=ns_args, kubectl_global_args=global_args
            )
            for ref in pod_refs
        ]
        return _K8sExecPlan(
            apply_command=apply_command,
            pod_refs=pod_refs,
            log_stream_commands=log_stream_commands,
            task_log_path=task_log_path,
            cleanup_refs=cleanup_refs,
            global_args=global_args,
            ns_args=ns_args,
            collect_output=collect_output,
        )

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        """The ``kubectl apply`` step (start the pod).

        Used for dry-run display and ``<task>.k8s.yaml`` persistence. The full run
        (apply -> stream -> status-watch -> stop) is driven by :meth:`execute`,
        since this operator returns True from :meth:`manages_own_execution`.
        """
        return self._build_execution_plan(
            task_name=task_name, script=script, envs=envs
        ).apply_command

    def manages_own_execution(self) -> bool:
        return True

    async def _status_note_for_pods(
        self, pod_refs: Sequence[str], *, global_args: Sequence[str], ns_args: Sequence[str]
    ) -> str | None:
        """Aggregate the not-yet-started pods' phase/reason into one sub-status.

        ``None`` when every pod is running (nothing to annotate). For a multi-pod
        (multi-node) task each still-starting pod is labelled by its pod name.
        """
        notes: list[str] = []
        for ref in pod_refs:
            note = await k8s_lifecycle.format_pod_start_note(
                ref, global_args=global_args, ns_args=ns_args
            )
            if note:
                label = ref.split("/", 1)[-1]
                notes.append(f"{label}: {note}" if len(pod_refs) > 1 else note)
        return "; ".join(notes) if notes else None

    async def _poll_status_note(
        self,
        pod_refs: Sequence[str],
        *,
        global_args: Sequence[str],
        ns_args: Sequence[str],
        status_note: Callable[[str | None], None],
        interval: float = 3.0,
    ) -> None:
        """Push the pods' live start phase/reason to ``status_note`` until cancelled.

        Display-only and best-effort: any polling error is swallowed so the sub-
        status watcher can never fail the task. The caller cancels it once the
        startup wait is over.
        """
        try:
            while True:
                try:
                    status_note(
                        await self._status_note_for_pods(
                            pod_refs, global_args=global_args, ns_args=ns_args
                        )
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:  # never let a display poll break the task
                    pass
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise

    async def execute(
        self,
        *,
        launcher: Any,
        output_logger: Any,
        env: Mapping[str, str],
        task_name: str,
        script: Sequence[str],
        status_note: Callable[[str | None], None] | None = None,
    ) -> int:
        """Driver-managed run: apply the pod(s), then stream + watch each pod.

        K8s is async: ``kubectl apply`` starts the pod and the bash apply step
        returns once it is scheduled/started. Each pod's ``kubectl logs -f`` is then
        OFFLOADED -- redirected straight to ``<task>.log`` (no sflow-driver per-line
        processing) -- while a single decoupled tailer echoes that file to the
        console (TTY). The pod STATUS is the authoritative completion signal
        (``watch_until_terminal``); the log stream is a side channel that is
        interrupted the instant the pod is terminal (or the workflow ends and this
        coroutine is cancelled). Any failed pod fails the whole task (a multi-node
        task is one unit); otherwise the leader pod's exit code is the task's.

        Once the pod watches have resolved (every pod terminal, or one pod failed and
        its still-running peers were cancelled), each terminal pod's follow is DRAINED
        so ``<task>.log`` -- which ``kubectl logs -f`` has been writing live -- ends
        complete, and is then finalized in place (TTY-sanitize only) before the
        orchestrator runs probes + output/result parsing on it.

        Multi-pod note: every pod's follow APPENDS to the same ``<task>.log``, so the
        file is ordered CHRONOLOGICALLY (interleaved across pods), not grouped per pod
        as the deleted re-fetch used to rebuild it. ``kubectl logs --prefix`` tags every
        line with its pod, so lines stay attributable; anything parsing this file must
        not assume per-pod contiguity.
        The console/TUI stream is only for live observation and may be cut early;
        the disk log is ground truth. The ``finally`` stops the tailer and deletes
        the task's objects either way.
        """
        plan = self._build_execution_plan(
            task_name=task_name, script=list(script), envs=dict(env)
        )
        tailers: list[asyncio.Future] = []
        # Finalize <task>.log exactly once: normally on the terminal path below, else
        # (cancelled first) in the ``finally`` before the pods are deleted.
        finalized = False
        # Single-pod node-local output collector (kubectl cp), run concurrently with the
        # pod-status watch below; None unless this task stages output for collection.
        collector: asyncio.Future | None = None
        # Live sub-status: while the apply step waits for the pod(s) to start, poll
        # their phase/reason and surface it (e.g. "Pending: Unschedulable") next to
        # the RUNNING task status. Cancelled once startup is over (below).
        note_poller: asyncio.Future | None = None
        if status_note is not None:
            note_poller = asyncio.ensure_future(
                self._poll_status_note(
                    plan.pod_refs,
                    global_args=plan.global_args,
                    ns_args=plan.ns_args,
                    status_note=status_note,
                )
            )
        # Merge-pod: the apply command builds one env Secret per member from a
        # prefix-namespaced launcher env (avoids envFrom collisions in the shared
        # container); pass that instead of the leader's plain env.
        apply_env = plan.merge_launcher_env if plan.merge_launcher_env is not None else env
        # Post-terminal stage clock. Armed only once the pods are terminal, because
        # only from then on is the DAG blocked on this epilogue rather than on the
        # workload. Defined out here so the `finally` teardown is timed too -- a stall
        # in the pod delete is just as invisible as one in the log finalize is.
        _stage_t0: float | None = None

        def _stage(name: str) -> None:
            nonlocal _stage_t0
            if _stage_t0 is None:
                return  # pods still running: this time belongs to the workload
            dt = time.monotonic() - _stage_t0
            _stage_t0 = time.monotonic()
            if dt >= _EPILOGUE_WARN_S:
                _logger.warning(
                    f"[{task_name}] post-terminal {name} took {dt:.1f}s -- the pod "
                    "had already finished, so the workflow was blocked on this"
                )

        try:
            rc = await launcher.run_async(
                plan.apply_command,
                output_logger=output_logger,
                env=apply_env,
                task_name=task_name,
            )
            # Startup wait is over (pod running, or apply failed fast): stop
            # annotating and clear the sub-status so RUNNING shows clean.
            if note_poller is not None:
                note_poller.cancel()
                await asyncio.gather(note_poller, return_exceptions=True)
                note_poller = None
            if status_note is not None:
                status_note(None)
            if rc != 0:
                return rc
            # Bytes already in <task>.log after apply == the driver/apply
            # diagnostics the launcher flushed (before any pod log). We preserve
            # this prefix when we rebuild the file from the complete pod logs.
            # Decoupled console tailer(s), reading the offloaded log file(s) from
            # disk (TTY only) -- independent of the file writers so a chatty pod can
            # never re-saturate the event loop via the console path. A single pod
            # gets one tailer on <task>.log (kubectl --prefix already tags each
            # line). A merge-pod gets one tailer per member's <task>.log tagged with
            # the member name (the offloaded demux strips the mux tag and writes no
            # kubectl prefix), plus the leader/default log for any pod-level lines.
            if plan.merge_tag_paths:
                tailed: set[str] = set()
                for member_name, member_path in plan.merge_tag_paths.items():
                    tailed.add(member_path)
                    tailers.append(
                        asyncio.ensure_future(
                            k8s_lifecycle.tail_file_to_console(
                                member_path,
                                task_name=member_name,
                                line_prefix=f"[{member_name}] ",
                            )
                        )
                    )
                if plan.task_log_path and plan.task_log_path not in tailed:
                    label = plan.merge_console_label or task_name
                    tailers.append(
                        asyncio.ensure_future(
                            k8s_lifecycle.tail_file_to_console(
                                plan.task_log_path,
                                task_name=label,
                                line_prefix=f"[{label}] ",
                            )
                        )
                    )
            elif plan.task_log_path:
                tailers.append(
                    asyncio.ensure_future(
                        k8s_lifecycle.tail_file_to_console(
                            plan.task_log_path, task_name=task_name
                        )
                    )
                )
            # Node-local output collection (single-pod K8s): the entrypoint stages the
            # pod's whole workflow-output subtree and holds the container open; this
            # concurrent task waits for the readiness marker, kubectl cp's the archive
            # back, and extracts into the driver's workflow dir the files missing on the
            # host. It MUST run alongside the status watch: the pod stays Running until
            # it's released, so the watch would otherwise never see it go terminal.
            output_dir = env.get("SFLOW_OUTPUT_DIR")
            workflow_dir = env.get("SFLOW_WORKFLOW_OUTPUT_DIR")
            if (
                plan.collect_output
                and not plan.merge_tag_paths
                and plan.task_log_path
                and output_dir
                and workflow_dir
                and plan.pod_refs
            ):
                collector = asyncio.ensure_future(
                    self._collect_via_cp(
                        task_name=task_name,
                        pod_ref=plan.pod_refs[0],
                        output_dir=str(output_dir),
                        dest_dir=str(workflow_dir),
                        log_path=plan.task_log_path,
                        global_args=plan.global_args,
                        ns_args=plan.ns_args,
                    )
                )
            # Watch every pod, but fail the task as a whole the instant any pod
            # dies: a multi-node task is one logical unit (one pod per node), so a
            # dead sub-pod must not leave the task blocked on a still-running peer
            # (e.g. a worker idling on `sleep 3600` after the leader crashed).
            results = await k8s_lifecycle.gather_pods_fail_fast(
                [
                    self._run_pod_stream(plan=plan, index=i)
                    for i in range(len(plan.pod_refs))
                ],
                mpi_world_group=plan.mpi_world_group,
            )
            # From here the pods are done and the DAG is waiting on this epilogue alone:
            # arm the stage clock so a stall names itself instead of going silent.
            _stage_t0 = time.monotonic()
            # The pod is terminal now. If it produced output, the collector already
            # copied it back and released the pod (that's WHY it went terminal); if it
            # produced none, no marker ever came -- so stop the (still-scanning) collector.
            if collector is not None:
                if not collector.done():
                    # Still scanning => NEITHER sentinel arrived. A task with nothing
                    # to collect announces that too, so this is not the ordinary
                    # no-output case -- it means the trap's lines never reached
                    # <task>.log at all: the container died before running the trap
                    # (OOM-kill, SIGKILL, eviction), or `kubectl logs -f` had not
                    # delivered them yet (observed 20 minutes behind, which cost a real
                    # run its whole output tree).
                    _logger.warning(
                        f"[{task_name}] pod reached a terminal phase without either "
                        "collect sentinel appearing in "
                        f"{plan.task_log_path}: node-local output was NOT collected. "
                        "The container either died before its EXIT trap ran, or the "
                        "trap's output was never delivered -- compare the heartbeat "
                        "lines above to see whether the log was still growing. Raise "
                        "`collect_grace_seconds`, or set `collect_node_local_output: "
                        "false` to drop the mechanism."
                    )
                    collector.cancel()
                await asyncio.gather(collector, return_exceptions=True)
                collector = None
                _stage("output-collector join")
            # All pods are terminal and their live streams have been drained. Stop
            # the console tailer(s), then make <task>.log final -- it is the streamed
            # file itself, so this is just the TTY-sanitize pass. It must still happen
            # BEFORE the orchestrator's probes + output/result parsing, which scan it.
            if tailers:
                for t in tailers:
                    t.cancel()
                await asyncio.gather(*tailers, return_exceptions=True)
                tailers = []
                _stage("log-tailer join")
            # <task>.log is the streamed file and each terminal pod's follow has been
            # drained into it, so finalizing is just the TTY-sanitize pass. A peer
            # cancelled by fail-fast / world-group resolution had its stream cut while
            # still Running -- its log simply ends there, which is what actually
            # happened to it.
            await self._finalize_task_log(plan)
            _stage("log finalize")
            finalized = True
            # Any failed pod fails the whole task; otherwise the leader's code.
            return k8s_lifecycle.task_exit_code(results)
        finally:
            if note_poller is not None:
                note_poller.cancel()
                await asyncio.gather(note_poller, return_exceptions=True)
            # Stop the output collector on any exit path (e.g. teardown of a long-lived
            # task whose pod never staged output, so it was still scanning the log).
            if collector is not None:
                if not collector.done():
                    collector.cancel()
                await asyncio.gather(collector, return_exceptions=True)
            for t in tailers:
                if not t.done():
                    t.cancel()
            if tailers:
                await asyncio.gather(*tailers, return_exceptions=True)
            # Cancelled before the normal finalize: still clean the streamed log before
            # teardown. Bounded + best-effort so teardown always reaches the delete and
            # the cancel survives.
            _cancelled: asyncio.CancelledError | None = None
            if not finalized and plan.task_log_path:
                try:
                    await asyncio.wait_for(
                        self._finalize_task_log(plan),
                        timeout=_SFLOW_FINALIZE_ON_CANCEL_TIMEOUT,
                    )
                except asyncio.CancelledError as exc:
                    # Preserve cancellation: still delete below, then re-raise -- don't
                    # let a cancel during finalize masquerade as a completed task.
                    _cancelled = exc
                except Exception:
                    pass  # best-effort finalize -- teardown must still reach the delete
            await k8s_lifecycle.delete_objects(
                plan.cleanup_refs, global_args=plan.global_args, ns_args=plan.ns_args
            )
            _stage("pod delete")
            if _cancelled is not None:
                raise _cancelled

    async def _collect_via_cp(
        self,
        *,
        task_name: str,
        pod_ref: str,
        output_dir: str,
        dest_dir: str,
        log_path: str,
        global_args: Sequence[str],
        ns_args: Sequence[str],
    ) -> None:
        """Copy the pod's staged node-local output back to the driver via ``kubectl cp``.

        Runs concurrently with the pod-status watch. Waits (scanning the offloaded log,
        no kubectl on the healthy path) for the entrypoint's readiness marker -- emitted
        repeatedly once it has staged
        ``$SFLOW_OUTPUT_DIR/<tgz>`` and is holding the container open -- then copies that
        one archive out and extracts it into the driver's WORKFLOW output dir (``dest_dir``)
        WITHOUT overwriting existing files, so both per-task outputs and workflow-level
        files a task wrote (e.g. aiperf's ``aiperf_concurrency_*``) land on the host while
        driver-authoritative files (each task's ``<task>.log``, sflow logs, artifacts) are
        kept. Then it ``touch``es the done-sentinel so the pod stops waiting and exits with
        its real code. The pod's own grace timeout guarantees it can never hang if this
        never runs (e.g. the task produced no output and exited immediately). Best-effort:
        any failure is logged, and the done-sentinel is still set so the pod is released.
        """
        name = pod_ref.split("/", 1)[-1]
        tgz_remote = f"{output_dir.rstrip('/')}/{_SFLOW_COLLECT_TGZ}"
        done_remote = f"{output_dir.rstrip('/')}/{_SFLOW_COLLECT_DONE}"
        # Wait for the pod's readiness marker in the offloaded <task>.log (local reads,
        # no kubectl). Deliberately NOT a pod probe: polling the pod with `kubectl exec`
        # put 42 execs into a readiness-probed TRT-LLM server during GPU autotuning
        # before it died with SIGTERM. The cost of that choice is a handshake only as
        # reliable as log delivery -- `kubectl logs -f` once took 20 minutes to surface
        # these lines, and a container-log ROTATION discards them for good -- which is
        # why the pod REPEATS the marker while it waits (see the trap) rather than
        # announcing once. Raise `collect_grace_seconds`, or set
        # `collect_node_local_output: false` to remove the mechanism entirely, if that
        # trade is wrong for a workload.
        staged = await _wait_for_marker(
            log_path,
            _SFLOW_COLLECT_READY_MARKER.encode(),
            task_name=task_name,
            stop_marker=_SFLOW_COLLECT_NONE_MARKER.encode(),
        )
        if not staged:
            return  # the task announced it had nothing to collect -- not a problem
        # The pod only holds itself open for ``collect_grace_seconds``; if we lost that
        # race it has already exited and its container is gone. Both the cp and the
        # done-sentinel exec run commands INSIDE that container, so against a terminal
        # pod they do not fail fast -- kubectl blocks until the API server eventually
        # answers "cannot exec into a container in a completed pod" (~20 min observed),
        # and ``execute`` cannot return until this collector settles, hanging the whole
        # workflow. Nothing is recoverable at this point (the staged archive died with
        # the pod), so say so loudly instead of stalling, and never touch the pod again.
        phase = await k8s_lifecycle.get_pod_phase(
            pod_ref,
            global_args=global_args,
            ns_args=ns_args,
            timeout=_SFLOW_COLLECT_PHASE_TIMEOUT,
        )
        if phase in k8s_lifecycle.TERMINAL_PHASES:
            _logger.warning(
                f"[{task_name}] pod is already {phase}: its output-collect window "
                f"({self._collect_grace_seconds}s) expired before the driver copied, so "
                "the staged node-local output was DISCARDED with the pod. Raise "
                "`collect_grace_seconds` on the kubernetes backend for tasks with large "
                "outputs."
            )
            return
        local_tgz = os.path.join(dest_dir, _SFLOW_COLLECT_TGZ)
        try:
            os.makedirs(dest_dir, exist_ok=True)
            # Bounded by the pod's own grace window: it stops waiting at that point, so a
            # cp still running past it can only be copying out of a dying container.
            rc, _, err = await k8s_lifecycle.run_kubectl(
                ["cp", f"{name}:{tgz_remote}", local_tgz, *[str(a) for a in ns_args]],
                global_args=global_args,
                timeout=float(self._collect_grace_seconds),
            )
            if rc == 0 and os.path.exists(local_tgz):
                with open(local_tgz, "rb") as fh:
                    blob = fh.read()
                extracted, skipped = _unpack_collected_tar(blob, dest_dir)
                _logger.info(f"[{task_name}] {_collect_summary_line(dest_dir, extracted, skipped)}")
            elif rc == k8s_lifecycle.KUBECTL_TIMEOUT_RC:
                # The copy outlived the pod's patience: the pod stops waiting after
                # collect_grace_seconds and exits MID-COPY, so finishing is impossible
                # (and continuing would block on a dead container). The archive is
                # simply bigger than this window allows -- `kubectl cp` streams a tar
                # through the API server's exec channel at only a few MB/s -- so the
                # one actionable fix is a longer window, which widens BOTH the pod's
                # wait and this timeout (they share collect_grace_seconds).
                _logger.warning(
                    f"[{task_name}] node-local output collection ABANDONED: the copy did "
                    f"not finish within collect_grace_seconds ({self._collect_grace_seconds}s), "
                    "so the pod stopped waiting and its staged output was discarded. "
                    "Raise `collect_grace_seconds` on the kubernetes backend (it bounds both "
                    "the pod's wait and this copy) for tasks with large outputs."
                )
            else:
                _logger.warning(
                    f"[{task_name}] kubectl cp of node-local output failed "
                    f"(rc={rc}): {err or 'see above'}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning(f"[{task_name}] node-local output collection failed: {exc}")
        finally:
            try:
                if os.path.exists(local_tgz):
                    os.remove(local_tgz)
            except OSError:
                pass
            # Release the pod (it exits with the user script's code) even if cp failed.
            # Bounded: if the pod went terminal while the cp ran, this exec would
            # otherwise block on a completed container and wedge the driver -- and the
            # pod needs no releasing once it has already exited.
            await k8s_lifecycle.run_kubectl(
                [
                    "exec", name, *[str(a) for a in ns_args],
                    "--", "sh", "-c", f"touch {shlex.quote(done_remote)}",
                ],
                global_args=global_args,
                timeout=k8s_lifecycle.POLL_KUBECTL_TIMEOUT,
            )

    async def _finalize_task_log(self, plan: _K8sExecPlan) -> None:
        """Make ``<task>.log`` final: strip TTY control bytes from the streamed file.

        ``<task>.log`` IS the log -- ``kubectl logs -f`` writes it live, and a terminal
        pod's follow is DRAINED (:func:`k8s_lifecycle.drain_log_stream`) so it ends
        complete. sflow used to rebuild this file from a one-shot ``kubectl logs``
        re-fetch; that was removed because the kubelet rotates container logs, so the
        re-fetch returned only the last window and replaced an hour of streamed output
        with its final 11 seconds. All that survives of the rebuild is the pass that
        cleaned TTY control bytes.

        Run off the event loop: the sanitize walks the whole file line-at-a-time
        (~266 MB/s measured), so a multi-GB log would otherwise stall the driver during
        teardown -- the exact class of silent block this epilogue had before.

        NOTE: ``asyncio.to_thread`` is NOT cancellable. The cancel path bounds its
        ``await`` (``_SFLOW_FINALIZE_ON_CANCEL_TIMEOUT``) so teardown always reaches the
        pod delete, but the worker thread runs to completion in the background and can
        briefly delay interpreter exit on a very large log. That is deliberate: the
        alternative is either blocking teardown or leaving the log unsanitized.
        """
        if not plan.task_log_path:
            return
        # For a merge-pod the leader's path is ALSO one of the member paths, so de-dup
        # before handing them over (the second pass would only re-clean clean output).
        paths = list(dict.fromkeys(
            [plan.task_log_path, *(plan.merge_tag_paths or {}).values()]
        ))
        await asyncio.to_thread(k8s_lifecycle.sanitize_streamed_logs, paths)

    async def _run_pod_stream(
        self, *, plan: _K8sExecPlan, index: int
    ) -> tuple[int, str]:
        """Offload one pod's log to ``<task>.log`` while its STATUS drives completion.

        The pod phase (``watch_until_terminal``) is authoritative -- it, not the log
        stream, decides when the task is done -- so the DAG/status stays the single
        source of truth and a dropped stream never completes or fails a running task.
        The log stream is OFFLOADED to a child shell (a plain redirect for a single
        pod; ``kubectl | demuxer`` tag-demux for a merge-pod), so the driver's event
        loop is never in the per-line path and its status watches stay responsive. The
        stream is interrupted the moment the pod is terminal, or on teardown
        (cancellation) via the ``finally``. For a long-lived READY service the watch
        never returns, so the task stays alive until the orchestrator cancels it at
        workflow end.

        Returns ``(exit_code, final_phase)``; the caller (``execute``) uses the phases
        to re-fetch the complete on-disk log once all pods are terminal.
        """
        pod_ref = plan.pod_refs[index]
        is_merge = bool(plan.merge_tag_paths and plan.task_log_path)
        stream_proc = None
        if is_merge:
            # Merge-pod: one container carries several members' logs; the tag demux
            # runs in the offloaded child process (python demuxer), splitting the stream
            # into each member's <task>.log on disk -- the driver stays out of the
            # per-line path exactly as the single-pod redirect does.
            stream_proc = await k8s_lifecycle.start_pod_log_demux_stream(
                plan.log_stream_commands[index],
                tag_paths=plan.merge_tag_paths,
                default_path=plan.task_log_path,
            )
        elif plan.task_log_path:
            stream_proc = await k8s_lifecycle.start_pod_log_file_stream(
                plan.log_stream_commands[index], plan.task_log_path
            )
        final_phase = ""
        try:
            final_phase = await k8s_lifecycle.watch_until_terminal(
                pod_ref, global_args=plan.global_args, ns_args=plan.ns_args
            )
        finally:
            # The merge pipeline is a process group (kubectl + demuxer) -> signal the
            # whole group.
            if stream_proc is not None:
                # Terminal pod -> drain the follow so it delivers its tail (killing it
                # mid-stream is what used to truncate <task>.log). Still-Running pod at
                # teardown -> cut it, since its follow would never end. Either way the
                # stream process does not outlive this task.
                await k8s_lifecycle.stop_log_stream(
                    stream_proc,
                    terminal=final_phase in k8s_lifecycle.TERMINAL_PHASES,
                    kill_group=is_merge,
                )
        exit_code = await k8s_lifecycle.pod_exit_code(
            pod_ref,
            global_args=plan.global_args,
            ns_args=plan.ns_args,
            phase=final_phase,
        )
        return exit_code, final_phase

    def network_fallback_status(self, task: Any) -> RdmaRuntimeStatus | None:
        """Report whether this task's pod(s) hit an unusable RDMA NIC / slow transport.

        The in-pod RDMA preamble prints a ``[sflow-rdma]`` decision marker to the
        offloaded ``<task>.log`` (the driver is never in the per-line path), so
        this reads that log's startup prefix and parses the marker -- letting the
        orchestrator surface an unusable RDMA NIC (sflow does not force a fallback;
        the libraries auto-select, which may be NVLink/MNNVL or TCP). It also scans
        UCX debug lines for intra-node TCP, which means cuda_ipc/NVLink was not
        selected. Best-effort: returns ``None`` when the log is unavailable or has
        no relevant marker.
        """
        envs = getattr(task, "envs", None)
        task_out = envs.get("SFLOW_TASK_OUTPUT_DIR") if isinstance(envs, dict) else None
        if not task_out:
            return None
        log_path = os.path.join(task_out, f"{task.name}.log")
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                text = fh.read(_RDMA_SCAN_MAX_BYTES)
        except OSError:
            return None
        status = parse_rdma_runtime_status(text)
        # An unusable IB/RoCE NIC is not a perf problem when the pod is in a
        # rack-scale NVLink (MNNVL) ComputeDomain: NCCL cross-node rides NVLink (see
        # _mnnvl_env_defaults' NCCL_MNNVL_ENABLE), so tell the orchestrator to skip
        # the "slow TCP" warning.
        if status is not None and self._compute_domain_channel:
            status = replace(status, mnnvl_crossnode=True)
        return status

    def writes_own_task_log(self) -> bool:
        """K8s always offloads: the pod log is written straight to ``<task>.log``.

        Returning True makes the app skip attaching a live ``CoalescingFileHandler``
        (it uses a ``DeferredTaskLogHandler`` for driver-side diagnostics instead),
        so the only live writer of ``<task>.log`` is ``kubectl logs -f`` redirected
        to it (see ``execute`` / ``_run_pod_stream``). This keeps the sflow driver's
        event loop out of the per-line byte path entirely; a decoupled tailer
        handles console/TUI visibility.
        """
        return True
