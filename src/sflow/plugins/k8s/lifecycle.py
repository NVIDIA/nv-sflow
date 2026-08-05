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
* the pod **status** is authoritative (``watch_until_terminal``): the log is a side
  channel, not a gate. Once the pod is terminal the follow is DRAINED
  (``drain_log_stream``) so it delivers its tail and exits; a pod still RUNNING at
  teardown has its follow cut instead (``terminate_process``), since it would never end.

``<task>.log`` IS the log -- there is no post-run rebuild. sflow used to re-fetch each
pod's whole container log with a one-shot ``kubectl logs`` and replace the streamed
file with it; that was removed because the kubelet rotates container logs, so the
re-fetch returned only the last window and a one-hour server persisted as its final
11 seconds. See the TODO on ``drain_log_stream`` before reinstating anything like it.

All kubectl status calls are async ``create_subprocess_exec`` (mirroring the
kubernetes backend's ``_kubectl``) and carry the CLI-level global flags.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
import time
from collections.abc import Awaitable, Iterator, Mapping, Sequence

from sflow.core.command import Command
from sflow.core.command_trace import get_command_trace
from sflow.utils.console_text import (
    CONSOLE_LINE_CHAR_CAP,
    clamp_for_console,
    frame_in_progress,
    last_visible_frame,
    rejoin_carried_frame,
    strip_ansi,
)
from sflow.logging import SFLOW_TASK_STREAM_ATTR, get_logger

_logger = get_logger(__name__)

# Seconds between pod-phase polls (the authoritative completion signal).
PHASE_POLL_INTERVAL = 2.0
# terminated.exitCode lags container exit; poll a short budget before falling
# back to a phase-derived code.
EXIT_CODE_RETRIES = 30
# Two consecutive "not found" polls => the pod was deleted (out from under us).
_GONE_POLLS = 2
TERMINAL_PHASES = ("Succeeded", "Failed")
# rc reported by ``run_kubectl`` when its optional timeout fires (mirrors the shell's
# 124 for `timeout`), so callers can tell "kubectl said no" from "kubectl never answered".
KUBECTL_TIMEOUT_RC = 124
# Ceiling for the SHORT, idempotent status queries in the poll loops below (one
# ``kubectl get`` each). Without it, a silently-dead TCP connection to the API server
# blocks the call until the kernel stops retransmitting (``tcp_retries2=15`` ~= 15-20
# min), which wedges the whole driver: the status watch never ticks, so the
# orchestrator cannot notice a finished task, and NOTHING is logged for the duration.
# Bounding turns that into a retry on the next tick. Generous enough that a slow but
# healthy control plane still answers. Deliberately NOT applied to ``kubectl logs -f``,
# which is a long-lived stream and must not be cut.
POLL_KUBECTL_TIMEOUT = 30.0
# Ceiling for a delete that WAITS for the objects to go away. Unlike the task-teardown
# delete (which passes ``--wait=false`` and returns in ~0.2s), the allocation label
# sweep blocks until the pods actually terminate -- a real run was measured at 28.4s,
# and a pod with a long terminationGracePeriodSeconds takes longer still. Bounding
# these at POLL_KUBECTL_TIMEOUT would abort legitimate cleanup and leak objects, so
# they get their own, far more generous ceiling: enough headroom for any realistic
# grace period, while still preventing a wedged connection from hanging teardown (or,
# for the atexit sweep, interpreter shutdown) forever.
DELETE_KUBECTL_TIMEOUT = 300.0
# How long a batched pod-status sample stays usable. The status poll is ~90% of all
# kubectl traffic (862 of 957 calls in a measured 7-pod run), because every watcher
# polls its own pod every PHASE_POLL_INTERVAL. Watchers whose ticks land inside this
# window share ONE `kubectl get pod a b c ...` instead of issuing one call each. Set
# just under the poll interval: long enough to coalesce most ticks, short enough that
# a sample is never more than this stale, so terminal detection slips by at most this
# much. At 19 concurrent recipes the poll traffic self-congests (mean `get pod` rose
# 0.2s -> 1.3s), so collapsing it also relieves the control plane.
_BATCH_FRESH_S = PHASE_POLL_INTERVAL * 0.75
# How long to let a terminal pod's `kubectl logs -f` finish delivering before giving up
# on it. Once the container terminates the API server closes the stream, so a healthy
# follow reaches EOF and exits on its own in well under a second; this only has to cover
# a slow last flush. It MUST stay bounded: the historical kubectl failure mode is a
# follow that never exits after pod completion, and the epilogue is on the DAG's
# critical path. Overrun -> cut it and keep whatever it had already written.
STREAM_DRAIN_TIMEOUT = 15.0
# Consecutive status-poll failures per WATCH TARGET, so one control-plane outage logs a
# single WARNING (then debug) instead of one per task per tick. Cleared on the first
# successful poll, which also logs the recovery. Keyed by pod ref for the single-pod
# query and by ``_batch_streak_key(...)`` for a shared batch -- the two namespaces are
# disjoint, so a pod can never be confused with the batch that covers it.
_timeout_streak: dict[str, int] = {}
# Console tailer cadence + per-tick line cap. The cap keeps a chatty log from
# re-saturating the event loop via the console path; dropped lines stay in the
# file (which has everything). The tailer is fully decoupled from the file write.
_TAIL_POLL_INTERVAL = 0.3
_TAIL_MAX_LINES_PER_TICK = 400
# Max characters of ONE line echoed to the console. A line cap is not the same
# guard as the line COUNT cap above, and only this one bounds THE post-run hang: a
# task whose output is an unterminated `\r` progress bar produces a SINGLE line, so
# it passes the count cap untouched no matter how large it is. `kubectl logs -f`
# withholds that line until it terminates (it reads to `\n`), so an hour of bar
# arrives as one multi-megabyte line the moment the task ends.
#
# Rendering that through the rich console handler is what freezes the driver: measured
# at ~6.3us and ~300 BYTES OF RSS PER CHARACTER, so a 48 MB line is ~5 CPU-minutes and
# ~14 GB -- reproduced here as a 35-minute machine-wide stall, matching the ~20-minute
# production hang where py-spy caught the driver at 100% CPU under
# `tail_file_to_console`. It runs on the event-loop thread, so nothing else in the
# driver ticks meanwhile: no pod-status watch, no DAG poll.
#
# The VALUE is the shared console clamp, not a k8s-specific number: the same unbounded
# render is reachable from the launcher's console path on every other backend, so both
# use one constant and one helper (`clamp_for_console`) and cannot drift apart. The
# alias is kept because the tick budget below is expressed in terms of it.
_TAIL_MAX_LINE_CHARS = CONSOLE_LINE_CHAR_CAP
# Max characters echoed across ALL lines of one tick. The two caps above still
# compose into a stall: 400 lines x 2000 chars is 800 KB, ~5s of blocked event loop
# at the ~6.3us/char measured above -- smaller than the post-run hang, but the same
# failure in miniature, and it repeats every tick for as long as a task stays that
# chatty. This is the backstop that makes a tick's render cost bounded outright
# rather than bounded per line. ~0.4s worst case; the remainder waits in the file
# and is reported as omitted, exactly like the line-count cap.
_TAIL_MAX_CHARS_PER_TICK = 65536


def _trace_op(args: Sequence[str]) -> str:
    """A short, REDACTED label for a kubectl invocation, e.g. ``get pod/server-0``.

    Only the verb and its first non-flag operand are kept. Never the full argv: an
    ``exec ... -- sh -c <script>`` carries arbitrary payload, and flag VALUES can
    carry cluster/user identifiers -- none of which belong in a health report that
    gets read (and pasted) during debugging.
    """
    parts = [str(a) for a in args]
    if not parts:
        return "?"
    verb = parts[0]
    # Collect POSITIONALS only: skip flags and the token that follows a value-taking
    # flag, and stop at ``--`` (everything past it is payload). Doing this FIRST is
    # what keeps flag values out of the label -- scanning raw argv for a "/" made
    # ``delete pod -l sflow.ai/allocation=<id>`` label itself with the selector, and
    # ``delete --namespace ns pod/y`` label itself with the namespace.
    positionals: list[str] = []
    skip_next = False
    for p in parts[1:]:
        if skip_next:
            skip_next = False
            continue
        if p == "--":
            break
        if p.startswith("-"):
            skip_next = "=" not in p
            continue
        positionals.append(p)
    # Among the positionals, prefer kubectl's ``type/name`` reference (``pod/x``,
    # ``mpijob/j``, ``name:/path`` for cp); otherwise the first positional (the kind).
    target = next((p for p in positionals if "/" in p or ":" in p), "")
    if not target and positionals:
        target = positionals[0]
    return f"{verb} {target}".strip()[:60]


def run_kubectl_sync(
    args: Sequence[str],
    *,
    global_args: Sequence[str] = (),
    request_timeout: str = "10s",
) -> tuple[int, str, str]:
    """Blocking ``kubectl`` for the PREFLIGHT phase, before the event loop exists.

    The async :func:`run_kubectl` is the entry point everywhere else; this exists only
    because preflight runs synchronously. Both live here so a single module owns "how
    sflow runs kubectl" -- and, crucially, so both feed the same health trace. Bounded
    by kubectl's own ``--request-timeout`` (it cannot use ``asyncio.wait_for``).
    """
    import subprocess

    op = _trace_op(args)
    started_wall = time.time()
    started = time.monotonic()
    argv = [
        "kubectl",
        *[str(a) for a in global_args],
        *[str(a) for a in args],
        f"--request-timeout={request_timeout}",
    ]
    try:
        result = subprocess.run(argv, capture_output=True, text=True)
        rc, out, err = (
            result.returncode,
            (result.stdout or "").strip(),
            (result.stderr or "").strip(),
        )
    except Exception as exc:  # kubectl missing / spawn failure
        rc, out, err = 1, "", str(exc)
    get_command_trace().record(
        "kubectl", op, time.monotonic() - started, rc, started_at=started_wall
    )
    return rc, out, err


async def run_kubectl(
    args: Sequence[str],
    *,
    global_args: Sequence[str] = (),
    timeout: float | None = None,
    input: bytes | None = None,
) -> tuple[int, str, str]:
    """Run ``kubectl <global_args> <args>`` async; return ``(rc, stdout, stderr)``.

    ``timeout`` (seconds) bounds the call. It is OPTIONAL and defaults to ``None`` =
    wait forever, which is the historical behavior every existing caller relies on
    (some kubectl calls are legitimately long: ``cp`` of a big archive, ``logs`` of a
    huge log), so this stays a drop-in for them. Pass it where an unbounded wait can
    wedge the driver -- notably any ``cp``/``exec`` into a pod that may already be
    terminal, where kubectl can block for ~20 minutes before the API server answers
    ``cannot exec into a container in a completed pod``.

    ``input`` (bytes) is piped to stdin, for the ``apply -f -`` style calls that feed
    a manifest in rather than naming a file.

    On timeout the child is killed (never leaked) and the call reports failure as a
    normal ``(rc, out, err)`` triple -- rc ``KUBECTL_TIMEOUT_RC`` -- rather than
    raising, so callers keep their existing rc-checking shape.

    Every invocation -- here and in :func:`run_kubectl_sync` -- is recorded to the
    external-command health trace, so "was it kubectl or was it sflow?" has one
    answer covering both the preflight/allocation phase and the task phase.
    """
    op = _trace_op(args)
    started = time.monotonic()
    # Wall clock too: durations come from the monotonic clock (immune to steps), but
    # the trace timestamp must be a real time of day so it lines up with the workflow
    # Timeline and with cluster-side events.
    started_wall = time.time()
    proc = await asyncio.create_subprocess_exec(
        "kubectl",
        *[str(a) for a in global_args],
        *[str(a) for a in args],
        stdin=asyncio.subprocess.PIPE if input is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if timeout is None:
        out, err = await proc.communicate(input)
    else:
        try:
            out, err = await asyncio.wait_for(proc.communicate(input), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            # Kill the hung child so it cannot outlive the driver's interest in it.
            try:
                proc.kill()
            except (ProcessLookupError, OSError):  # already gone
                pass
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except (asyncio.TimeoutError, TimeoutError, ProcessLookupError, OSError):
                pass
            get_command_trace().record(
                "kubectl", op, time.monotonic() - started, KUBECTL_TIMEOUT_RC,
                timed_out=True, started_at=started_wall,
            )
            return (
                KUBECTL_TIMEOUT_RC,
                "",
                f"kubectl {op} timed out after {timeout}s",
            )
    rc = proc.returncode if proc.returncode is not None else 1
    get_command_trace().record(
        "kubectl", op, time.monotonic() - started, rc, started_at=started_wall
    )
    return (
        rc,
        out.decode(errors="replace").strip(),
        err.decode(errors="replace").strip(),
    )


async def get_pod_phase(
    pod_ref: str,
    *,
    global_args: Sequence[str],
    ns_args: Sequence[str],
    timeout: float | None = None,
) -> str:
    """Return ``.status.phase`` (``""`` when the pod is gone / unreadable).

    ``timeout`` is overridable because one caller runs this INSIDE a deadline it
    must not eat: the output collector checks the phase while the pod is counting
    down ``collect_grace_seconds``, so a full-length stall here would burn the copy
    window before the copy even starts. Callers on no deadline keep the default.
    """
    rc, out, _ = await run_kubectl(
        ["get", pod_ref, *ns_args, "-o", "jsonpath={.status.phase}"],
        global_args=global_args,
        timeout=POLL_KUBECTL_TIMEOUT if timeout is None else timeout,
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


_STATUS_JSONPATH_FIELDS = (
    "{.status.phase}|"
    "{.status.containerStatuses[*].state.running.startedAt}|"
    "{.status.containerStatuses[*].state.waiting.reason}|"
    "{.status.containerStatuses[*].state.terminated.exitCode}"
)


def _parse_status_fields(raw: str) -> tuple[str, bool, bool, bool]:
    """``phase|running|waiting|exitcodes`` -> the 4-tuple contract."""
    phase, running, waiting, exit_codes = (
        p.strip() for p in (raw.split("|") + ["", "", "", ""])[:4]
    )
    codes = exit_codes.split()
    # sflow pods are single-container, so "no container still running/waiting" is a
    # safe "work finished" signal. If a sidecar/init container is ever added, this
    # would need to key on the main container's status instead of the pod-wide set.
    container_done = bool(codes) and not running and not waiting
    container_failed = any(c != "0" for c in codes)
    return phase, container_done, container_failed, False


class _PodStatusBatcher:
    """Coalesces concurrent pod-status polls into one kubectl call.

    Watchers register while they are watching; when two or more share a kubectl
    context, a tick refreshes ALL of them in a single ``kubectl get pod a b c``.
    Deliberately transparent: with fewer than two active watchers it falls straight
    through to the original one-pod query, so single-pod callers (the MPI launcher
    watch, every existing test) keep byte-identical behaviour.
    """

    def __init__(self) -> None:
        self._active: dict[tuple, set[str]] = {}
        self._cache: dict[tuple, tuple[float, dict[str, tuple[str, bool, bool, bool]]]] = {}
        self._locks: dict[tuple, asyncio.Lock] = {}

    @staticmethod
    def _key(global_args: Sequence[str], ns_args: Sequence[str]) -> tuple:
        return (tuple(str(a) for a in global_args), tuple(str(a) for a in ns_args))

    def register(self, pod_ref: str, *, global_args: Sequence[str], ns_args: Sequence[str]) -> None:
        self._active.setdefault(self._key(global_args, ns_args), set()).add(pod_ref)

    def unregister(self, pod_ref: str, *, global_args: Sequence[str], ns_args: Sequence[str]) -> None:
        key = self._key(global_args, ns_args)
        refs = self._active.get(key)
        if refs:
            refs.discard(pod_ref)
            if not refs:
                self._active.pop(key, None)
                self._cache.pop(key, None)
                self._locks.pop(key, None)

    def active_count(self, *, global_args: Sequence[str], ns_args: Sequence[str]) -> int:
        return len(self._active.get(self._key(global_args, ns_args), ()))

    async def status(
        self, pod_ref: str, *, global_args: Sequence[str], ns_args: Sequence[str]
    ) -> tuple[str, bool, bool, bool]:
        key = self._key(global_args, ns_args)
        refs = self._active.get(key, set())
        if len(refs) < 2:
            return await _single_pod_terminal_status(
                pod_ref, global_args=global_args, ns_args=ns_args
            )
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            ts, data = self._cache.get(key, (0.0, {}))
            if (time.monotonic() - ts) >= _BATCH_FRESH_S:
                # Re-read the active set HERE rather than reusing the local bound before
                # the lock: ``refs`` aliases the live set, and peers unregistering while
                # we waited would otherwise have us query a stale membership.
                data = await _batched_pod_status(
                    sorted(self._active.get(key, ())),
                    global_args=global_args,
                    ns_args=ns_args,
                )
                self._cache[key] = (time.monotonic(), data)
        # A successful batch carries an entry for every ref it was asked about
        # (absent-from-output is recorded there as not_found=True). Falling back to the
        # default therefore means the batch FAILED, or this ref joined after it was
        # built -- both transient, never "pod gone".
        return data.get(pod_ref, ("", False, False, False))


_POD_STATUS_BATCHER = _PodStatusBatcher()


def _batch_streak_key(ns_args: Sequence[str]) -> str:
    """Stall-streak key for a shared batch: stable as its membership changes.

    Deliberately NOT derived from the pod list -- pods join and leave the batch every
    time a task starts or finishes, and a key that moved with them would restart the
    streak mid-outage and re-warn on every tick, which is the flooding the streak
    exists to prevent. The ``[batch]`` prefix keeps this out of the per-pod keyspace.
    """
    return "[batch]" + " ".join(str(a) for a in ns_args)


def _note_poll_failure(key: str, *, subject: str, rc: int, err: str) -> None:
    """Warn ONCE per stall episode for ``key``, then drop to debug.

    Every watcher hits the same outage on every tick, so logging unconditionally would
    emit one line per pod per tick for as long as it lasts -- but logging NOTHING is
    worse, and was the actual regression this exists to prevent: a stalled control
    plane leaves the DAG apparently frozen, and the event-loop watchdog cannot see it
    (the poll is an ordinary await -- the loop is idle, not blocked). ``command_trace
    .jsonl`` records the call either way, but a post-mortem artifact is not a
    substitute for saying so while it is happening.
    """
    n = _timeout_streak.get(key, 0) + 1
    _timeout_streak[key] = n
    if rc == KUBECTL_TIMEOUT_RC:
        detail = (
            f"timed out after {POLL_KUBECTL_TIMEOUT}s "
            "(control-plane unreachable or connection stalled)"
        )
    else:
        detail = f"failed (rc={rc}): {err.strip() or 'no stderr'}"
    log = _logger.warning if n == 1 else _logger.debug
    log(f"{subject}: status poll {detail}; retrying [consecutive={n}]")


def _note_poll_recovered(key: str, *, subject: str, also_clear: Sequence[str] = ()) -> None:
    """Clear ``key``'s stall streak, announcing the recovery if there was one.

    ``also_clear`` drops the per-pod streaks a batch supersedes: a pod that warned on
    the single-pod path before joining a batch would otherwise keep its entry forever
    (nothing else clears it), so its next lone timeout would log at debug and be
    invisible. Cleared silently -- the batch-level recovery line already covers them.
    """
    recovered = bool(_timeout_streak.pop(key, 0))
    for extra in also_clear:
        _timeout_streak.pop(extra, None)
    if recovered:
        _logger.info(f"{subject}: status poll recovered; control plane answering again")


async def _batched_pod_status(
    pod_refs: Sequence[str], *, global_args: Sequence[str], ns_args: Sequence[str]
) -> dict[str, tuple[str, bool, bool, bool]]:
    """One ``kubectl get`` for many pods -> ``{pod_ref: status-tuple}``.

    ``--ignore-not-found`` so a deleted pod does not fail the whole batch: it is
    simply absent from the output, which is how a genuine deletion is detected. On a
    non-zero rc nothing is trusted -- an empty map is returned and every caller reads
    it as a transient error, never as "pod gone".

    That "treat every failure as transient" rule is why EVERY non-zero rc is reported
    here, not just a timeout (the single-pod path can leave rc=1 quiet because there it
    means NotFound -- a real answer). A batch never sees NotFound: ``--ignore-not-found``
    turns a missing pod into rc=0 with the row absent, so a non-zero rc is always a
    genuine error, and an unreported one would loop the whole batch silently forever.
    """
    # The LIST-shaped query below is only valid for TWO OR MORE named resources: kubectl
    # returns a bare Pod (no ``.items``) for a single one, so ``{range .items[*]}`` yields
    # NOTHING at rc=0 -- and the "absent from a successful batch" rule below would then
    # report a healthy Running pod as deleted. This is reachable: ``status`` snapshots the
    # active set only when it issues the query, and peers unregistering (fail-fast cancels
    # them together) can shrink it to one in between. Delegate instead of guessing.
    if len(pod_refs) < 2:
        return {
            ref: await _single_pod_terminal_status(
                ref, global_args=global_args, ns_args=ns_args
            )
            for ref in pod_refs
        }
    # The row separator must reach kubectl as the two characters ``\`` + ``n`` -- its
    # jsonpath parser reads the escape itself. A REAL newline byte here (plain "\n" in
    # the Python literal) makes kubectl exit 1 with "error parsing jsonpath", which this
    # function reports as an empty map == "transient error" -- so every watcher in the
    # batch would poll forever, silently, and no multi-pod task could ever finish.
    tmpl = (
        "{range .items[*]}{.metadata.name}|" + _STATUS_JSONPATH_FIELDS + r'{"\n"}{end}'
    )
    rc, out, err = await run_kubectl(
        ["get", *pod_refs, *ns_args, "--ignore-not-found", "-o", f"jsonpath={tmpl}"],
        global_args=global_args,
        timeout=POLL_KUBECTL_TIMEOUT,
    )
    streak_key = _batch_streak_key(ns_args)
    subject = f"batched status poll ({len(pod_refs)} pods)"
    if rc != 0:
        _note_poll_failure(streak_key, subject=subject, rc=rc, err=err)
        return {}
    _note_poll_recovered(streak_key, subject=subject, also_clear=pod_refs)
    by_name: dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        name, _, rest = line.partition("|")
        by_name[name.strip()] = rest
    result: dict[str, tuple[str, bool, bool, bool]] = {}
    for ref in pod_refs:
        name = ref.split("/", 1)[-1]
        result[ref] = (
            _parse_status_fields(by_name[name])
            if name in by_name
            else ("", False, False, True)  # in a successful batch: confirmed absent
        )
    return result


async def _single_pod_terminal_status(
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
    rc, out, err = await run_kubectl(
        ["get", pod_ref, *ns_args, "-o", f"jsonpath={_STATUS_JSONPATH_FIELDS}"],
        global_args=global_args,
        timeout=POLL_KUBECTL_TIMEOUT,
    )
    if rc != 0:
        if rc == KUBECTL_TIMEOUT_RC:
            # Surface the stall -- a wedged connection used to produce NO output at all
            # for as long as it lasted, which is what made the hang undiagnosable. But
            # EVERY polling task hits this every POLL_KUBECTL_TIMEOUT during one
            # outage, so an 18-task workflow would emit 18 lines per tick: warn ONCE
            # per stall episode per pod, then drop to debug until it recovers.
            #
            # ONLY the timeout is reported here. Unlike the batched query, this one
            # does not pass --ignore-not-found, so rc=1 is kubectl's normal way of
            # saying NotFound -- a real answer, handled just below, not a failure.
            _note_poll_failure(pod_ref, subject=pod_ref, rc=rc, err=err)
        # Distinguish a genuinely deleted pod (kubectl prints NotFound) from a
        # transient control-plane error, so only a real deletion advances the
        # gone-streak in watch_until_terminal. A timeout is transient by definition.
        return "", False, False, _stderr_says_not_found(err)
    # Recovered: say so, so a stall episode has a visible end as well as a start.
    _note_poll_recovered(pod_ref, subject=pod_ref)
    return _parse_status_fields(out)


async def _pod_terminal_status(
    pod_ref: str, *, global_args: Sequence[str], ns_args: Sequence[str]
) -> tuple[str, bool, bool, bool]:
    """One kubectl call -> ``(phase, container_done, container_failed, not_found)``.

    Unchanged contract. Internally this now shares one ``kubectl get`` with the other
    pods being watched in the same kubectl context (see :class:`_PodStatusBatcher`) --
    the status poll was ~90% of all kubectl traffic. With fewer than two active
    watchers it falls through to the original single-pod query, so nothing about a
    lone watcher's behaviour changes.
    """
    return await _POD_STATUS_BATCHER.status(
        pod_ref, global_args=global_args, ns_args=ns_args
    )


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
        timeout=POLL_KUBECTL_TIMEOUT,
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
            timeout=POLL_KUBECTL_TIMEOUT,
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
    # Join the batch: while >=2 pods in this kubectl context are watched, their polls
    # collapse into one kubectl call per tick.
    _POD_STATUS_BATCHER.register(pod_ref, global_args=global_args, ns_args=ns_args)
    try:
        return await _watch_until_terminal_loop(
            pod_ref, global_args=global_args, ns_args=ns_args, interval=interval
        )
    finally:
        _POD_STATUS_BATCHER.unregister(
            pod_ref, global_args=global_args, ns_args=ns_args
        )


async def _watch_until_terminal_loop(
    pod_ref: str,
    *,
    global_args: Sequence[str],
    ns_args: Sequence[str],
    interval: float,
) -> str:
    missing = 0
    while True:
        phase, container_done, container_failed, not_found = await _pod_terminal_status(
            pod_ref, global_args=global_args, ns_args=ns_args
        )
        if phase in TERMINAL_PHASES:
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
            timeout=POLL_KUBECTL_TIMEOUT,
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


async def drain_log_stream(
    proc: asyncio.subprocess.Process,
    *,
    kill_group: bool = False,
    timeout: float | None = None,
) -> bool:
    """Let a TERMINAL pod's ``kubectl logs -f`` finish on its own; True if it did.

    sflow used to SIGKILL the follow the instant the pod went terminal and then rebuild
    ``<task>.log`` with a one-shot ``kubectl logs`` re-fetch. The missing tail that
    repaired was self-inflicted (we cut our own stream), and the cure was worse than the
    disease: the kubelet ROTATES container logs, so the re-fetch returned only the last
    window and REPLACED the complete streamed copy with it -- a one-hour trtllm server
    persisted as its final 11 seconds. Draining instead means the stream is already
    complete and there is nothing to rebuild.

    Bounded, because the historical kubectl failure is a follow that never exits after
    pod completion. On overrun the process is killed and False is returned; the log then
    simply ends where the stream did.

    TODO(k8s-log-completeness): with the re-fetch gone, ``<task>.log`` is exactly what
    ``kubectl logs -f`` delivered. If a real run is ever found MISSING output that the
    pod definitely produced -- a truncated tail on a task that ended normally, not a
    task cancelled mid-run -- that is a genuine follow-delivery gap and this is where to
    fix it. Prefer repairing it HERE (e.g. a longer drain, or a targeted
    ``kubectl logs --since-time`` for just the gap) over reinstating the unconditional
    full re-fetch, which is what caused the rotation data loss in the first place.

    Only valid once the pod is TERMINAL. Against a still-running pod the follow never
    ends, so teardown must keep using :func:`terminate_process`.
    """
    if proc.returncode is not None:
        return True
    budget = STREAM_DRAIN_TIMEOUT if timeout is None else timeout
    try:
        await asyncio.wait_for(proc.wait(), timeout=budget)
        return True
    except (asyncio.TimeoutError, TimeoutError):
        _logger.warning(
            f"log stream did not finish within {budget}s of the pod going terminal; "
            "cutting it -- the log ends where the stream did"
        )
    except Exception:
        return False
    await terminate_process(proc, kill_group=kill_group)
    return False


async def stop_log_stream(
    proc: asyncio.subprocess.Process,
    *,
    terminal: bool,
    kill_group: bool = False,
) -> bool:
    """End a pod's log follow the right way for its state; True if it drained.

    The single entry point for "this task is done with its log stream", so the two
    callers (:meth:`K8sContainerOperator._run_pod_stream` and the MPI launcher watch)
    cannot drift apart on the ``terminal`` / ``kill_group`` handling.

    ``terminal=True`` means the pod reached a terminal phase, so the API server has
    closed the stream and the follow will hit EOF -- DRAIN it so it delivers its tail.
    ``terminal=False`` is a pod still RUNNING at teardown (fail-fast peer, SIGINT); its
    follow would never end, so it must be cut.

    The reap runs in a ``finally`` because the drain is a cancellable await on the
    teardown path: a cancel landing inside it would otherwise propagate out and leave
    the kubectl child orphaned. :func:`terminate_process` signals SIGTERM before its
    first await, so the signal lands even if the cleanup itself is then cancelled.
    """
    if not terminal:
        await terminate_process(proc, kill_group=kill_group)
        return False
    try:
        return await drain_log_stream(proc, kill_group=kill_group)
    finally:
        # No-op once the follow has exited (terminate_process returns immediately when
        # returncode is set); this only matters on the cancel/overrun paths.
        await terminate_process(proc, kill_group=kill_group)


def sanitize_streamed_logs(paths: Sequence[str]) -> None:
    """Clean ``\\r`` redraws / ANSI out of logs written by the live stream.

    ``<task>.log`` is now always the streamed file, and the deleted re-fetch was also
    the pass that stripped TTY control bytes -- so do it directly on the streamed file. Streaming and
    line-at-a-time, so a multi-GB log costs no memory. Best-effort: a log that cannot
    be rewritten is left exactly as streamed.
    """
    for path in dict.fromkeys(p for p in paths if p):
        try:
            if os.path.getsize(path) == 0:
                continue
            _sanitize_file_inplace(path)
        except OSError:
            continue


def _sanitize_log_line(body: str) -> str:
    """Clean ONE log line for the rebuilt on-disk log: collapse ``\\r`` redraws to
    the last non-empty frame, then strip ANSI escape sequences. ``body`` has no
    trailing newline.

    Used ONLY by the post-run finalize (:func:`sanitize_streamed_logs`), which runs
    after the pod's follow has ended -- never on the live offload path. So a TTY task's
    color/cursor/progress control bytes (aiperf/rich, ``pip``) don't pollute the
    persisted ``<task>.log``, while live streaming stays untouched.
    """
    return strip_ansi(last_visible_frame(body))


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


def _sanitize_file_inplace(path: str) -> None:
    """Rewrite ``path`` with each line cleaned via :func:`_sanitize_log_line`."""
    tmp = f"{path}.san"
    with open(tmp, "w", encoding="utf-8", newline="") as out:
        for body, nl in _iter_sanitized_lines(path):
            out.write(body)
            if nl:
                out.write("\n")
    os.replace(tmp, path)


def _console_active() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _console_line(raw: bytes, *, task_name: str) -> str:
    """One raw log line -> the bounded text to echo (``""`` when there is nothing to show).

    Collapses ``\\r`` redraws to the only frame a terminal ever displayed, strips ANSI, and
    caps the length. One unbounded line is enough to freeze the driver here, so the cap is
    not cosmetic -- see :data:`_TAIL_MAX_LINE_CHARS`.
    """
    text = strip_ansi(
        last_visible_frame(raw).decode("utf-8", errors="replace")
    ).rstrip()
    # A long line with no `\r` to collapse (a JSON/base64 dump) is capped by the shared
    # console clamp -- the same one every other backend's console path uses, so the two
    # cannot drift on how much of a line the console is allowed to render.
    return clamp_for_console(
        text, cap=_TAIL_MAX_LINE_CHARS, source=f"{task_name}.log"
    )


async def tail_file_to_console(
    path: str, *, task_name: str, line_prefix: str = ""
) -> None:
    """Echo new lines of ``path`` to the console (TTY only), decoupled from the writer.

    Reads only content appended after it starts (so it doesn't re-print the apply
    diagnostics already shown live), emits at most ``_TAIL_MAX_LINES_PER_TICK`` per
    tick with a bounded sleep in between -- so a high-volume log can't re-saturate
    the event loop through the console path (dropped lines remain in the file).
    Runs until cancelled (on pod terminal or workflow end).

    Every echoed line is bounded in LENGTH as well as count: ``\\r`` redraws collapse
    to their last non-empty frame and anything still over ``_TAIL_MAX_LINE_CHARS`` is
    truncated. One unbounded line is enough to freeze the whole driver here -- see that
    constant. A tick's TOTAL is capped too (``_TAIL_MAX_CHARS_PER_TICK``), since the
    per-line and per-count caps still multiply into a multi-second stall.

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
    # The frame the terminal is still showing for the unterminated line, which
    # ``partial`` no longer holds because that line ended on a ``\r`` (so
    # ``frame_in_progress`` collapsed it to nothing). Held separately and put back with
    # ``rejoin_carried_frame`` -- exactly as ``SubprocessLauncher`` does -- because plain
    # concatenation would splice a superseded frame onto the bytes that overwrite it.
    # Without it a bar whose last redraw ends in ``\r`` shows NOTHING on the console.
    carried = b""

    def _read_new() -> bytes:
        """Bytes appended since the last read (``b""`` if the file is unreadable)."""
        nonlocal pos
        try:
            with open(path, "rb") as fh:
                fh.seek(pos)
                data = fh.read()
                pos = fh.tell()
                return data
        except OSError:  # includes FileNotFoundError: the writer may not have started
            return b""

    def _echo(text: str) -> None:
        _logger.info(f"{line_prefix}{text}", extra={SFLOW_TASK_STREAM_ATTR: True})

    def _consume(data: bytes) -> None:
        """Emit the complete lines in ``data``, retaining the unterminated remainder."""
        nonlocal partial, carried
        parts = rejoin_carried_frame(carried, partial + data).split(b"\n")
        # Keep the incomplete trailing line, collapsing its in-place `\r` redraws now so
        # a still-running progress bar cannot grow this buffer without bound (mirrors
        # the launcher's own read loop).
        tail = parts.pop()
        partial = frame_in_progress(tail)
        carried = b"" if partial else last_visible_frame(tail)
        lines = parts
        dropped = 0
        if len(lines) > _TAIL_MAX_LINES_PER_TICK:
            dropped = len(lines) - _TAIL_MAX_LINES_PER_TICK
            lines = lines[-_TAIL_MAX_LINES_PER_TICK:]
        # A line is emitted whole or not at all, so the budget must never be able to
        # reject the FIRST line of a tick -- that would silence the console
        # permanently, one tick at a time. Guaranteeing forward progress directly
        # (always emit one line, then spend the budget) is what actually holds that
        # invariant: comparing the two constants does not, because a clamped line is
        # `_TAIL_MAX_LINE_CHARS` PLUS the truncation notice, whose length grows with
        # the task name. This way the caps stay independently tunable and no arithmetic
        # relationship between them has to be maintained.
        budget = _TAIL_MAX_CHARS_PER_TICK
        emitted = False
        for idx, raw in enumerate(lines):
            text = _console_line(raw, task_name=task_name)
            if not text:
                continue
            if emitted and len(text) > budget:
                # Out of render budget for this tick. Stop rather than block the loop;
                # the rest stays in the file and the next tick starts fresh.
                dropped += len(lines) - idx
                break
            budget -= len(text)
            emitted = True
            _echo(text)
        if dropped:
            _logger.info(
                f"[{task_name}] ... ({dropped} console lines omitted; full log "
                f"in {task_name}.log) ...",
                extra={SFLOW_TASK_STREAM_ATTR: True},
            )

    try:
        while True:
            data = _read_new()
            if data:
                _consume(data)
            await asyncio.sleep(_TAIL_POLL_INTERVAL)
    finally:
        # Cancelled (pod terminal / workflow end). Two things are still outstanding.
        #
        # First, one last read: ``execute`` cancels this tailer immediately after
        # ``drain_log_stream`` delivered the pod's tail, so the most interesting bytes
        # of the whole run can land inside the final poll gap and would otherwise never
        # be echoed. The same caps apply, so a burst arriving here still cannot flood.
        #
        # Second, the unterminated line -- typically a progress bar that never printed
        # its newline, which is precisely the shape this path exists for. Echo its final
        # frame so the last thing the task displayed is not the one thing missing.
        # Mirrors ``SubprocessLauncher._flush_tail`` for every other backend.
        #
        # Best-effort throughout: the console is observability only, and <task>.log on
        # disk already holds every byte either way, so a failure here is swallowed
        # rather than allowed to disturb teardown.
        try:
            _consume(_read_new())
            if partial or carried:
                tail_text = _console_line(
                    rejoin_carried_frame(carried, partial), task_name=task_name
                )
                if tail_text:
                    _echo(tail_text)
        except Exception:  # pragma: no cover - console echo must never break teardown
            pass


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
        rc, _out, err = await run_kubectl(
            ["delete", *refs, *ns_args, "--ignore-not-found", "--wait=false"],
            global_args=global_args,
            timeout=POLL_KUBECTL_TIMEOUT,
        )
        if rc != 0:
            # Say it out loud. ``run_kubectl`` RETURNS a non-zero rc (it does not
            # raise), so without this the except below never fires and a failed or
            # timed-out teardown delete is completely silent -- the same invisibility
            # this timeout work exists to remove. Not fatal: the backend's
            # allocation-label sweep still reclaims whatever is left behind.
            reason = (
                f"timed out after {POLL_KUBECTL_TIMEOUT}s"
                if rc == KUBECTL_TIMEOUT_RC
                else f"rc={rc}"
            )
            _logger.warning(
                f"teardown delete of {len(refs)} object(s) did not complete ({reason}): "
                f"{err or 'see above'}; the allocation-label sweep will reclaim them"
            )
    except Exception:
        pass
