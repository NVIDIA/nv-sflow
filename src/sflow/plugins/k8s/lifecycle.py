# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver-side (Python) Kubernetes pod lifecycle helpers for the k8s operator.

Kubernetes is async: ``kubectl apply`` starts a pod and returns, and log delivery
lags pod execution. So the operator runs the pod as decoupled steps managed by
the sflow driver (see ``K8sContainerOperator.execute``):

* a bash *apply* command starts the pod + waits-ready;
* the pod log is **offloaded**: ``kubectl logs -f`` is redirected straight to
  ``<task>.log`` (``start_pod_log_file_stream``), so the sflow driver's event loop
  is never in the per-line byte path -- it can't be saturated by a chatty server;
* a *decoupled*, bounded tailer (``tail_file_to_console``) reads ``<task>.log`` and
  echoes it to the console in TTY mode, so file-write and console are independent;
* the pod **status** is authoritative (``watch_until_terminal``): the moment it is
  terminal (or the workflow ends and the coroutine is cancelled) the log stream is
  interrupted (``terminate_process``) -- the log is a side channel, not a gate.

All kubectl status calls are async ``create_subprocess_exec`` (mirroring the
kubernetes backend's ``_kubectl``) and carry the CLI-level global flags.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
from collections.abc import Awaitable, Iterator, Mapping, Sequence

from sflow.core.command import Command
from sflow.core.launcher import _strip_ansi
from sflow.logging import SFLOW_TASK_STREAM_ATTR, get_logger

_logger = get_logger(__name__)

# Seconds between pod-phase polls (the authoritative completion signal).
PHASE_POLL_INTERVAL = 2.0
# terminated.exitCode lags container exit; poll a short budget before falling
# back to a phase-derived code.
EXIT_CODE_RETRIES = 30
# Two consecutive "not found" polls => the pod was deleted (out from under us).
_GONE_POLLS = 2
_TERMINAL_PHASES = ("Succeeded", "Failed")
# Console tailer cadence + per-tick line cap. The cap keeps a chatty log from
# re-saturating the event loop via the console path; dropped lines stay in the
# file (which has everything). The tailer is fully decoupled from the file write.
_TAIL_POLL_INTERVAL = 0.3
_TAIL_MAX_LINES_PER_TICK = 400


async def run_kubectl(
    args: Sequence[str], *, global_args: Sequence[str] = ()
) -> tuple[int, str, str]:
    """Run ``kubectl <global_args> <args>`` async; return ``(rc, stdout, stderr)``."""
    proc = await asyncio.create_subprocess_exec(
        "kubectl",
        *[str(a) for a in global_args],
        *[str(a) for a in args],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return (
        proc.returncode if proc.returncode is not None else 1,
        out.decode(errors="replace").strip(),
        err.decode(errors="replace").strip(),
    )


async def get_pod_phase(
    pod_ref: str, *, global_args: Sequence[str], ns_args: Sequence[str]
) -> str:
    """Return ``.status.phase`` (``""`` when the pod is gone / unreadable)."""
    rc, out, _ = await run_kubectl(
        ["get", pod_ref, *ns_args, "-o", "jsonpath={.status.phase}"],
        global_args=global_args,
    )
    return out if rc == 0 else ""


def _stderr_says_not_found(stderr: str) -> bool:
    """True when kubectl's stderr means the object genuinely does not exist.

    A deleted pod makes ``kubectl get`` exit non-zero with
    ``Error from server (NotFound): pods "..." not found``. A TRANSIENT
    control-plane error (API throttling/429, request timeout, TLS handshake blip,
    ``Unable to connect to the server``) ALSO exits non-zero but means the opposite
    -- the pod's existence is simply unknown this instant. Only the former may count
    toward "the pod was deleted out from under us"; conflating the two lets a healthy
    long-lived service pod be declared gone during an API hiccup (common when many
    tasks poll in parallel), which fails the task and tears the whole workflow down.
    """
    s = stderr.lower()
    return "notfound" in s.replace(" ", "") or "(notfound)" in s.replace(" ", "")


async def _pod_terminal_status(
    pod_ref: str, *, global_args: Sequence[str], ns_args: Sequence[str]
) -> tuple[str, bool, bool, bool]:
    """One kubectl call -> ``(phase, container_done, container_failed, not_found)``.

    ``container_done`` is True once the pod's app container(s) have all TERMINATED
    (none still ``running`` or ``waiting``) -- the TRUE "work finished" signal, which
    can PRECEDE the pod ``.status.phase`` flip because kubelet/API status propagation
    lags container exit. ``container_failed`` is True when any terminated container
    exited non-zero. ``not_found`` is True ONLY when kubectl confirmed the pod does
    not exist (a real deletion), NOT on a transient API error -- so the caller does
    not mistake an API hiccup for a deleted pod. ``phase`` is ``""`` (and the bool
    flags False) when the pod is gone or unreadable. The status fields come back
    ``|``-separated in a single query so a watch tick stays one kubectl call.
    """
    jsonpath = (
        "{.status.phase}|"
        "{.status.containerStatuses[*].state.running.startedAt}|"
        "{.status.containerStatuses[*].state.waiting.reason}|"
        "{.status.containerStatuses[*].state.terminated.exitCode}"
    )
    rc, out, err = await run_kubectl(
        ["get", pod_ref, *ns_args, "-o", f"jsonpath={jsonpath}"],
        global_args=global_args,
    )
    if rc != 0:
        # Distinguish a genuinely deleted pod (kubectl prints NotFound) from a
        # transient control-plane error, so only a real deletion advances the
        # gone-streak in watch_until_terminal.
        return "", False, False, _stderr_says_not_found(err)
    phase, running, waiting, exit_codes = (
        p.strip() for p in (out.split("|") + ["", "", "", ""])[:4]
    )
    codes = exit_codes.split()
    # sflow pods are single-container, so "no container still running/waiting" is a
    # safe "work finished" signal. If a sidecar/init container is ever added, this
    # would need to key on the main container's status instead of the pod-wide set.
    container_done = bool(codes) and not running and not waiting
    container_failed = any(c != "0" for c in codes)
    return phase, container_done, container_failed, False


async def format_pod_start_note(
    pod_ref: str, *, global_args: Sequence[str], ns_args: Sequence[str]
) -> str | None:
    """A short ``"<phase>[: <reason>]"`` for a pod that has not started yet.

    Returns ``None`` once the pod is ``Running``/``Succeeded`` (nothing to
    annotate) or is gone/unreadable. ``<reason>`` is the container
    ``waiting.reason`` if present (e.g. ``ImagePullBackOff``), else the
    ``PodScheduled`` condition reason (e.g. ``Unschedulable``). Used to surface a
    live task sub-status (``RUNNING (Pending: Unschedulable)``) while the task
    shows RUNNING but its pod is still scheduling / pulling its image.
    """
    phase = await get_pod_phase(pod_ref, global_args=global_args, ns_args=ns_args)
    if not phase or phase in ("Running", "Succeeded"):
        return None
    reason = ""
    rc, out, _ = await run_kubectl(
        [
            "get", pod_ref, *ns_args, "-o",
            "jsonpath={.status.containerStatuses[*].state.waiting.reason}",
        ],
        global_args=global_args,
    )
    if rc == 0 and out:
        reason = out.split()[0]
    if not reason:
        rc, out, _ = await run_kubectl(
            [
                "get", pod_ref, *ns_args, "-o",
                'jsonpath={.status.conditions[?(@.type=="PodScheduled")].reason}',
            ],
            global_args=global_args,
        )
        if rc == 0 and out:
            reason = out.strip()
    return f"{phase}: {reason}" if reason else phase


async def watch_until_terminal(
    pod_ref: str,
    *,
    global_args: Sequence[str],
    ns_args: Sequence[str],
    interval: float = PHASE_POLL_INTERVAL,
) -> str:
    """Poll until the pod is terminal or gone; return the final (real/derived) phase.

    Returns ``Succeeded``/``Failed`` once the pod reaches it, or ``""`` if the pod
    disappears (deleted out from under us) for two consecutive polls. This is what
    lets the driver break the lagging log stream the instant the pod is done.

    The pod ``.status.phase`` flip can LAG the container actually exiting (kubelet /
    API status propagation), so this ALSO treats the pod as terminal the moment its
    container(s) have terminated -- deriving the phase from the container exit code --
    instead of waiting the phase out. That keeps completion (and the workflow
    teardown that stops sibling services) aligned to the TRUE container status rather
    than the laggy phase. Pods are ``restartPolicy: Never``, so a terminated
    container stays terminated (no restart flap to race).
    """
    missing = 0
    while True:
        phase, container_done, container_failed, not_found = await _pod_terminal_status(
            pod_ref, global_args=global_args, ns_args=ns_args
        )
        if phase in _TERMINAL_PHASES:
            return phase
        if container_done:
            derived = "Failed" if container_failed else "Succeeded"
            # Container(s) exited but the pod phase has not flipped yet: complete now
            # on the true status instead of waiting out the phase-propagation lag.
            _logger.info(
                f"{pod_ref}: container terminated (pod phase={phase or '?'}); "
                f"completing as {derived} without waiting for the phase to flip"
            )
            return derived
        # Only a CONFIRMED NotFound (real deletion) advances the gone-streak. A
        # transient API error (throttle/timeout/TLS blip) -- or a pod that simply has
        # no phase yet -- returns not_found=False and RESETS the streak, so a healthy
        # long-lived service pod is never declared gone during an API hiccup (which
        # would fail the task and fail-fast the whole workflow).
        if not_found:
            missing += 1
            if missing >= _GONE_POLLS:
                _logger.warning(
                    f"{pod_ref}: not found for {missing} consecutive polls "
                    "-- treating as deleted out from under us."
                )
                return ""
        else:
            missing = 0
        await asyncio.sleep(interval)


async def pod_exit_code(
    pod_ref: str,
    *,
    global_args: Sequence[str],
    ns_args: Sequence[str],
    phase: str = "",
) -> int:
    """Best-effort container exit code, falling back to a phase-derived code.

    ``terminated.exitCode`` lags container exit, so poll a short budget; if it
    never appears (e.g. the pod was already deleted), derive from ``phase``
    (``Succeeded`` -> 0, anything else -> 1).
    """
    for _ in range(EXIT_CODE_RETRIES):
        rc, out, _ = await run_kubectl(
            [
                "get",
                pod_ref,
                *ns_args,
                "-o",
                "jsonpath={.status.containerStatuses[0].state.terminated.exitCode}",
            ],
            global_args=global_args,
        )
        if rc == 0 and out:
            try:
                return int(out)
            except ValueError:
                break
        await asyncio.sleep(1)
    return 0 if phase == "Succeeded" else 1


async def gather_pods_fail_fast(
    watchers: Sequence[Awaitable[tuple[int, str]]],
    *,
    mpi_world_group: bool = False,
) -> list[tuple[int, str] | None]:
    """Await one ``(exit_code, phase)`` watcher per pod, failing fast on any death.

    A multi-node task is one logical unit split into one pod per node (leader =
    index 0). If ANY pod reaches a non-zero exit / ``Failed`` phase (or its watcher
    raises), the remaining watchers are cancelled and the partial result list is
    returned at once -- otherwise a still-running peer (e.g. a worker pod idling on
    ``sleep 3600`` after the leader's engine has already crashed) would keep the
    whole task blocked until it is force-killed. When no pod fails, every watcher is
    awaited: a run-to-completion multi-node task needs all pods to finish, and a
    healthy long-lived service -- whose watchers never return -- blocks until the
    caller is cancelled at workflow teardown.

    ``mpi_world_group`` (multi-node MPI): the pods are one MPI ``COMM_WORLD`` -- a
    healthy run keeps every rank pod alive, so the FIRST pod to go terminal (success
    OR failure, ANY rank -- leader or worker) means the group is finished/broken.
    Resolve at once and cancel the survivors instead of blocking on them (idle worker
    pods whose watchers never return, or a hung leader). ``task_exit_code`` is
    leader-index-0 authoritative, so a masked-0 exit (``Succeeded``) still surfaces --
    via the orchestrator's exited-before-ready check -- rather than hanging the run.

    Returns a per-pod ``(exit_code, phase)`` list aligned to ``watchers``; slots for
    pods still running when a peer went terminal (hence cancelled) are ``None``.
    """
    tasks = [asyncio.ensure_future(w) for w in watchers]
    results: list[tuple[int, str] | None] = [None] * len(tasks)
    try:
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            failed = False
            for finished in done:
                idx = tasks.index(finished)
                if finished.cancelled():
                    continue
                if finished.exception() is not None:
                    # A watcher blew up (e.g. kubectl error) -> treat as a failed pod.
                    results[idx] = (1, "Failed")
                    failed = True
                    continue
                rc, phase = finished.result()
                results[idx] = (rc, phase)
                if rc != 0 or phase == "Failed":
                    failed = True
            # Break on any pod failure (existing), or -- for an MPI world group -- the
            # instant ANY pod goes terminal (success or fail), since one rank ending
            # breaks the group; cancel the survivors rather than block on them.
            if failed or (mpi_world_group and any(r is not None for r in results)):
                break
        return results
    finally:
        # A peer failed (or we're being torn down): cancel the still-running pod
        # watchers so we never block on them; their own `finally` cuts the streams.
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def task_exit_code(results: Sequence[tuple[int, str] | None]) -> int:
    """Collapse per-pod ``(exit_code, phase)`` results into the task's exit code.

    A multi-node task is one unit: if ANY pod failed (non-zero exit or ``Failed``
    phase) the task fails with that code (``1`` when the phase is ``Failed`` but the
    numeric code is 0/unknown). Otherwise the leader pod (index 0) carries the
    task's exit code -- unchanged single-pod / all-succeeded behaviour.
    """
    if not results:
        return 0
    for r in results:
        if r is not None and (r[0] != 0 or r[1] == "Failed"):
            return r[0] or 1
    leader = results[0]
    return leader[0] if leader is not None else 0


async def start_pod_log_file_stream(
    log_command: Command, dest_path: str
) -> asyncio.subprocess.Process:
    """Start ``kubectl logs -f <pod> ...`` writing STRAIGHT to ``dest_path`` (append).

    The redirect happens in the child shell, so the sflow driver never reads the
    stream line by line -- its event loop stays free for the status watches and
    the orchestrator's DAG poll. ``exec`` replaces the shell with kubectl so
    terminating the returned process stops kubectl itself. Best-effort file mkdir.
    """
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    argv = " ".join(shlex.quote(str(a)) for a in log_command.as_list())
    shell = f"exec {argv} >> {shlex.quote(dest_path)} 2>/dev/null"
    return await asyncio.create_subprocess_exec("bash", "-c", shell)


def _demux_command(tag_paths: Mapping[str, str], default_path: str) -> list[str]:
    """argv for the streaming stdin demuxer that splits a merged pod's log.

    Routes each ``[[sflow-mux:<task>]] `` line to that member's ``<task>.log`` (tag
    stripped, appended) and every other (pod-level / unknown-tag) line VERBATIM to
    ``default_path`` (the leader's log), reading from stdin -- the live
    ``kubectl logs -f`` pipe, or the post-terminal complete dump.

    Replaces the former ``awk`` program: the default ``awk`` on Debian/Ubuntu is
    ``mawk``, which block-buffers stdin, so a quiet member's tail never reached disk
    until the stream got chattier (the "3 of 4 tasks update, 1 stale" symptom). The
    Python reader (:mod:`sflow.plugins.k8s.log_demux`) uses ``os.read`` + a periodic
    flush so quiet and chatty members land promptly, while still running as its own
    process so the sflow driver's event loop is never in the per-line byte path.
    Member names and paths are passed as argv (never spliced into a shell/awk
    program). ``sys.executable`` runs the child on the same interpreter -- and thus
    the same importable ``sflow`` package -- as the driver.
    """
    argv = [
        sys.executable,
        "-m",
        "sflow.plugins.k8s.log_demux",
        "--default",
        str(default_path),
    ]
    for name, path in tag_paths.items():
        argv += ["--route", f"{name}={path}"]
    return argv


def _demux_shell(tag_paths: Mapping[str, str], default_path: str) -> str:
    """Shell-quoted :func:`_demux_command`, ready to splice into a ``bash -c`` line.

    Both offload paths (the live ``kubectl ... | <demuxer>`` pipeline and the
    post-terminal ``<demuxer> < dump`` rebuild) run the demuxer through a shell, so
    they share this one quoting step rather than repeating it.
    """
    return " ".join(shlex.quote(a) for a in _demux_command(tag_paths, default_path))


async def start_pod_log_demux_stream(
    log_command: Command,
    *,
    tag_paths: Mapping[str, str],
    default_path: str,
) -> asyncio.subprocess.Process:
    """Offload a merge-pod's ``kubectl logs -f`` + tag demux into a child shell.

    A merge-pod's one container carries several members' logs, each line tagged
    ``[[sflow-mux:<task>]] `` by the launcher. Unlike a single pod (a plain shell
    redirect), the merged stream must be split per member -- that split is done by
    the streaming demuxer (see :func:`_demux_command`) piped after ``kubectl``, so
    the sflow driver's event loop is STILL never in the per-line byte path, mirroring
    :func:`start_pod_log_file_stream`. This is what keeps the pod-status watches (and
    hence the early cut-over on terminal) responsive under a chatty merged server --
    the driver reading every line itself is what previously starved them. A decoupled
    tailer echoes each member's ``<task>.log`` to the console.

    The pipeline runs in a new session (``start_new_session=True``) so the whole
    thing (kubectl + demuxer) can be stopped as one process group on terminal /
    teardown (see :func:`terminate_process` ``kill_group``). Best-effort dir creation.
    """
    for path in (default_path, *tag_paths.values()):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    kubectl_argv = " ".join(shlex.quote(str(a)) for a in log_command.as_list())
    demux_argv = _demux_shell(tag_paths, default_path)
    # kubectl's own stderr (e.g. "pod deleted" on teardown) is noise -> /dev/null;
    # the demuxer writes straight to the per-task files, so the pipeline's stdout is
    # empty. The follow feeds the demuxer's stdin, which it reads promptly (os.read).
    shell = f"{kubectl_argv} 2>/dev/null | {demux_argv}"
    return await asyncio.create_subprocess_exec(
        "bash", "-c", shell, start_new_session=True
    )


async def terminate_process(
    proc: asyncio.subprocess.Process, *, kill_group: bool = False
) -> None:
    """Best-effort stop a side-channel subprocess (the log stream), SIGTERM->SIGKILL.

    ``kill_group`` signals the whole process group instead of just ``proc`` -- used
    for the merge-pod demux pipeline (``kubectl | demuxer``), started in its own
    session, so one signal stops both stages. The single-pod redirect ``exec``s
    kubectl in place (one process), so its default single-process terminate is enough.
    """
    if proc.returncode is not None:
        return

    def _signal(sig: int) -> None:
        # Group signal targets the pipeline's session leader group (kubectl + demuxer);
        # os.getpgid == proc.pid here since it was started in a new session.
        if kill_group:
            os.killpg(os.getpgid(proc.pid), sig)
        elif sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()

    try:
        _signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
        return
    except Exception:
        pass
    try:
        _signal(signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        return
    try:
        await proc.wait()
    except Exception:
        return


async def _dump_pod_log(
    pod_ref: str,
    dest_path: str,
    *,
    global_args: Sequence[str],
    ns_args: Sequence[str],
) -> bool:
    """One-shot ``kubectl logs <pod>`` (no ``-f``) -> ``dest_path``; True on success.

    No ``-f``: reads the node's whole log file, so it's complete and not behind
    like the interrupted follow. Returns False if kubectl exits non-zero (e.g. the
    pod was deleted out from under us) so the caller can keep the live content.
    """
    with open(dest_path, "wb") as fh:
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            *[str(a) for a in global_args],
            "logs",
            pod_ref,
            *[str(a) for a in ns_args],
            "--all-containers",
            "--prefix",
            stdout=fh,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()
    return rc == 0


def _sanitize_log_line(body: str) -> str:
    """Clean ONE log line for the rebuilt on-disk log: collapse ``\\r`` redraws to
    the last non-empty frame, then strip ANSI escape sequences. ``body`` has no
    trailing newline.

    Used ONLY by the post-run finalize rebuild (never the live offload path), so a
    TTY task's color/cursor/progress control bytes (aiperf/rich, ``pip``) don't
    pollute the persisted ``<task>.log`` while live streaming stays untouched.
    """
    if "\r" in body:
        frames = body.split("\r")
        body = next((f for f in reversed(frames) if f), frames[-1])
    return _strip_ansi(body)


def _iter_sanitized_lines(src_path: str) -> Iterator[tuple[str, bool]]:
    """Yield ``(sanitized_body, had_trailing_newline)`` for each line of ``src_path``.

    Shared core of :func:`_copy_sanitized` and :func:`_sanitize_file_inplace`
    (bounded memory -- one line at a time). ``newline="\\n"`` keeps in-line ``\\r``
    frames intact so :func:`_sanitize_log_line` can collapse them (not split into
    separate lines) and prevents any newline translation.
    """
    with open(src_path, "r", encoding="utf-8", errors="replace", newline="\n") as src:
        for line in src:
            nl = line.endswith("\n")
            yield _sanitize_log_line(line[:-1] if nl else line), nl


def _copy_sanitized(src_path: str, out) -> None:
    """Append ``src_path`` to the open binary file ``out`` line by line, cleaning
    each line via :func:`_sanitize_log_line`."""
    for body, nl in _iter_sanitized_lines(src_path):
        out.write(body.encode("utf-8"))
        if nl:
            out.write(b"\n")


def _sanitize_file_inplace(path: str) -> None:
    """Rewrite ``path`` with each line cleaned via :func:`_sanitize_log_line`."""
    tmp = f"{path}.san"
    with open(tmp, "w", encoding="utf-8", newline="") as out:
        for body, nl in _iter_sanitized_lines(path):
            out.write(body)
            if nl:
                out.write("\n")
    os.replace(tmp, path)


async def finalize_complete_log(
    pod_refs: Sequence[str],
    dest_path: str,
    *,
    prefix_size: int,
    phases: Sequence[str],
    global_args: Sequence[str],
    ns_args: Sequence[str],
    force: bool = False,
) -> None:
    """Rebuild ``dest_path`` as ``[preserved prefix] + [each pod's COMPLETE log]``.

    Because the live ``kubectl logs -f`` streams are interrupted the moment the
    pods are terminal, ``<task>.log`` may be missing the tail the follow had not
    yet delivered. This re-fetches every pod's whole container log with a one-shot
    ``kubectl logs`` (fast, not behind) and rebuilds the on-disk file as the
    preserved apply/driver-diagnostics prefix followed by each pod's complete log
    (grouped per pod), then atomically renames it into place -- so post-run readers
    (probes + output/result parsing, which scan the whole file) get the complete
    log with no duplication, for both single- and multi-pod tasks.

    Only runs when EVERY pod reached a real terminal phase (so each is
    re-fetchable); if any pod is gone/unknown, or any re-fetch fails, the
    live-streamed content is kept as-is (never wiped for a partial rebuild).
    ``force`` re-fetches irrespective of phase (the cancel/teardown path can't know
    the phases, and a fast-failing pod's log must be saved before the pod is
    deleted); a re-fetch that fails on a gone pod still keeps the live content.
    Best-effort: never raises.
    """
    if not pod_refs:
        return
    if not force and not all(p in ("Succeeded", "Failed") for p in phases):
        return
    tmp_files = [f"{dest_path}.complete.{i}" for i in range(len(pod_refs))]
    final = dest_path + ".final"
    try:
        for pod_ref, tmp in zip(pod_refs, tmp_files):
            ok = await _dump_pod_log(
                pod_ref, tmp, global_args=global_args, ns_args=ns_args
            )
            if not ok:
                return  # keep the live content rather than a partial rebuild
        prefix = b""
        if prefix_size > 0:
            try:
                with open(dest_path, "rb") as f:
                    prefix = f.read(prefix_size)
            except OSError:
                prefix = b""
        with open(final, "wb") as out:
            if prefix:
                out.write(prefix)
            for tmp in tmp_files:
                # Strip ANSI/control bytes (and collapse \r redraws) from the
                # re-fetched pod log as it is spliced in -- keeps the persisted
                # <task>.log clean for TTY tasks. The apply-diagnostics ``prefix``
                # is sflow's own output and is kept verbatim.
                _copy_sanitized(tmp, out)
        os.replace(final, dest_path)
    except Exception:
        pass
    finally:
        for path in (*tmp_files, final):
            try:
                os.remove(path)
            except OSError:
                pass


async def _dump_pod_log_raw(
    pod_ref: str,
    dest_path: str,
    *,
    global_args: Sequence[str],
    ns_args: Sequence[str],
) -> bool:
    """One-shot ``kubectl logs <pod>`` (no ``-f``, NO ``--prefix``) -> ``dest_path``.

    The merge-pod re-fetch: like :func:`_dump_pod_log` it reads the node's whole
    container log (complete, not behind the interrupted follow) with
    ``--all-containers`` (matching the live stream in :func:`build_log_stream_command`
    so the rebuilt log covers exactly what the follow saw), but WITHOUT ``--prefix``
    so each line still begins with its ``[[sflow-mux:<task>]] `` tag -- the splitter
    keys on that tag to rebuild the per-member files. Returns False on non-zero exit
    (e.g. the pod was deleted) so the caller keeps the live content.
    """
    with open(dest_path, "wb") as fh:
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            *[str(a) for a in global_args],
            "logs",
            pod_ref,
            *[str(a) for a in ns_args],
            "--all-containers",
            stdout=fh,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()
    return rc == 0


async def finalize_merged_complete_log(
    pod_ref: str,
    *,
    tag_paths: Mapping[str, str],
    default_path: str,
    prefix_size: int,
    phase: str,
    global_args: Sequence[str],
    ns_args: Sequence[str],
    force: bool = False,
) -> None:
    """Rebuild every merged member's ``<task>.log`` from the pod's COMPLETE log.

    The merge-pod analogue of :func:`finalize_complete_log`: the offloaded live
    demux (``kubectl logs -f | demuxer``) is interrupted the instant the pod is
    terminal, so the per-member files may be missing the tail the follow had not yet
    delivered.
    This re-fetches the pod's whole container log with a one-shot ``kubectl logs``
    (complete, not behind) and re-runs the SAME demuxer over it (single source
    of truth for the routing) into temp files -- the leader/default file seeded with
    its preserved apply-diagnostics prefix -- then atomically renames each into place.
    Post-run readers (probes + output/result parsing) therefore get each member's
    complete log with no duplication, matching the single-pod guarantee.

    Only runs when the pod reached a real terminal phase (so it is re-fetchable); if
    the re-fetch or the re-demux fails, the live-streamed content is kept as-is (never
    wiped for a partial rebuild). ``force`` re-fetches irrespective of phase (the
    cancel/teardown path can't know it). Best-effort: never raises.
    """
    if not force and phase not in ("Succeeded", "Failed"):
        return
    dump = f"{default_path}.merged.dump"
    tmp_tag_paths = {name: f"{p}.merged.final" for name, p in tag_paths.items()}
    tmp_default = f"{default_path}.merged.final"
    temps = set(tmp_tag_paths.values()) | {tmp_default}
    try:
        if not await _dump_pod_log_raw(
            pod_ref, dump, global_args=global_args, ns_args=ns_args
        ):
            return  # keep the live content rather than a partial rebuild
        # Seed the leader/default temp with the preserved apply-diagnostics prefix,
        # and start every member temp fresh, so the demuxer (which appends)
        # rebuilds each file exactly once with no stale/duplicated content.
        prefix = b""
        if prefix_size > 0:
            try:
                with open(default_path, "rb") as f:
                    prefix = f.read(prefix_size)
            except OSError:
                prefix = b""
        with open(tmp_default, "wb") as f:
            f.write(prefix)
        for tmp in tmp_tag_paths.values():
            if tmp != tmp_default:
                open(tmp, "wb").close()
        # Re-demux the COMPLETE dump with the SAME splitter (single source of truth
        # for routing), reading it via stdin and appending into the temps. No console
        # echo -- this is disk only.
        cmd = f"{_demux_shell(tmp_tag_paths, tmp_default)} < {shlex.quote(dump)}"
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await proc.wait() != 0:
            return  # keep the live content
        # Strip ANSI/control bytes (and collapse \r redraws) from each rebuilt
        # per-member file before publishing it. Runs after the demux so the
        # ``[[sflow-mux:<task>]]`` routing tags are already consumed and every line
        # is pure body -- collapsing \r can never drop a tag.
        _sanitize_file_inplace(tmp_default)
        os.replace(tmp_default, default_path)
        for name, tmp in tmp_tag_paths.items():
            if tmp != tmp_default:
                _sanitize_file_inplace(tmp)
                os.replace(tmp, tag_paths[name])
    except Exception:
        pass
    finally:
        for path in (dump, *temps):
            try:
                os.remove(path)
            except OSError:
                pass


def _console_active() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


async def tail_file_to_console(
    path: str, *, task_name: str, line_prefix: str = ""
) -> None:
    """Echo new lines of ``path`` to the console (TTY only), decoupled from the writer.

    Reads only content appended after it starts (so it doesn't re-print the apply
    diagnostics already shown live), emits at most ``_TAIL_MAX_LINES_PER_TICK`` per
    tick with a bounded sleep in between -- so a high-volume log can't re-saturate
    the event loop through the console path (dropped lines remain in the file).
    Runs until cancelled (on pod terminal or workflow end).

    ``line_prefix`` is prepended to each echoed line. It is empty for a single pod
    (``kubectl logs --prefix`` already tags every line with its
    ``[pod/<pod>/<container>]`` source, so adding one would double up); a merge-pod
    member passes ``"[<task>] "`` because its demuxed ``<task>.log`` carries no
    kubectl prefix, so the tag is what keeps the interleaved members attributable."""
    if not _console_active():
        return
    # Start at the current end of file: apply diagnostics were already streamed to
    # the console live by the launcher; only tail the pod logs appended from here.
    try:
        pos = os.path.getsize(path)
    except OSError:
        pos = 0
    partial = b""
    while True:
        data = b""
        try:
            with open(path, "rb") as fh:
                fh.seek(pos)
                data = fh.read()
                pos = fh.tell()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        if data:
            partial += data
            parts = partial.split(b"\n")
            partial = parts.pop()  # keep the incomplete trailing line
            lines = parts
            dropped = 0
            if len(lines) > _TAIL_MAX_LINES_PER_TICK:
                dropped = len(lines) - _TAIL_MAX_LINES_PER_TICK
                lines = lines[-_TAIL_MAX_LINES_PER_TICK:]
            for raw in lines:
                text = _strip_ansi(raw.decode("utf-8", errors="replace")).rstrip()
                if text:
                    _logger.info(
                        f"{line_prefix}{text}", extra={SFLOW_TASK_STREAM_ATTR: True}
                    )
            if dropped:
                _logger.info(
                    f"[{task_name}] ... ({dropped} console lines omitted; full log "
                    f"in {task_name}.log) ...",
                    extra={SFLOW_TASK_STREAM_ATTR: True},
                )
        await asyncio.sleep(_TAIL_POLL_INTERVAL)


async def delete_objects(
    refs: Sequence[str], *, global_args: Sequence[str], ns_args: Sequence[str]
) -> None:
    """Best-effort, non-blocking delete of the task's objects (by name).

    ``--wait=false`` so teardown does not block on the pod's termination grace
    period; the kubernetes backend's allocation-label sweep (and atexit) is the
    backstop for anything left behind.
    """
    if not refs:
        return
    try:
        await run_kubectl(
            ["delete", *refs, *ns_args, "--ignore-not-found", "--wait=false"],
            global_args=global_args,
        )
    except Exception:
        pass
