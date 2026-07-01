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
import shutil
import sys
from collections.abc import Awaitable, Sequence

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
    """Poll the pod phase until it is terminal or gone; return the final phase.

    Returns ``Succeeded``/``Failed`` once the pod reaches it, or ``""`` if the pod
    disappears (deleted out from under us) for two consecutive polls. This is what
    lets the driver break the lagging log stream the instant the pod is done.
    """
    missing = 0
    while True:
        phase = await get_pod_phase(pod_ref, global_args=global_args, ns_args=ns_args)
        if phase in _TERMINAL_PHASES:
            return phase
        if not phase:
            missing += 1
            if missing >= _GONE_POLLS:
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

    Returns a per-pod ``(exit_code, phase)`` list aligned to ``watchers``; slots for
    pods still running when a peer failed (hence cancelled) are ``None``.
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
            if failed:
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


async def terminate_process(proc: asyncio.subprocess.Process) -> None:
    """Best-effort stop a side-channel subprocess (the log stream), SIGTERM->SIGKILL."""
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
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
        proc.kill()
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


async def finalize_complete_log(
    pod_refs: Sequence[str],
    dest_path: str,
    *,
    prefix_size: int,
    phases: Sequence[str],
    global_args: Sequence[str],
    ns_args: Sequence[str],
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
    Best-effort: never raises.
    """
    if not pod_refs or not all(p in ("Succeeded", "Failed") for p in phases):
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
                with open(tmp, "rb") as src:
                    shutil.copyfileobj(src, out)
        os.replace(final, dest_path)
    except Exception:
        pass
    finally:
        for path in (*tmp_files, final):
            try:
                os.remove(path)
            except OSError:
                pass


def _console_active() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


async def tail_file_to_console(path: str, *, task_name: str) -> None:
    """Echo new lines of ``path`` to the console (TTY only), decoupled from the writer.

    Reads only content appended after it starts (so it doesn't re-print the apply
    diagnostics already shown live), emits at most ``_TAIL_MAX_LINES_PER_TICK`` per
    tick with a bounded sleep in between -- so a high-volume log can't re-saturate
    the event loop through the console path (dropped lines remain in the file).
    Runs until cancelled (on pod terminal or workflow end).

    Lines are echoed as-is: ``kubectl logs --prefix`` already tags every line with
    its ``[pod/<pod>/<container>]`` source, so the tailer does NOT add sflow's
    ``[task]`` console prefix (that would double up and lengthen every line)."""
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
                    # No [task] prefix: the line already carries its [pod/...] tag.
                    _logger.info(text, extra={SFLOW_TASK_STREAM_ATTR: True})
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
