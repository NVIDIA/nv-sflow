# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine-local, cross-process GPU reservation for intra-node backends.

Multiple ``sflow run`` processes on the same host otherwise each pack GPUs from
device 0 independently (the in-process planner has no idea another run exists),
so concurrent runs collide on the same physical GPUs. This module gives them a
shared, file-locked reservation registry:

    try_reserve_gpus(count, run_id) -> [GpuHandle, ...]  # claim, or raise
    release_gpus(run_id)                                 # give them back

The claim never blocks: callers that want to wait retry on their own clock, so
the waiting stays interruptible (see Operator.acquire_resources).

A GPU is considered free only when it is (a) not reserved by another sflow run
and (b) not busy with any *foreign* workload -- another user's job, a bare
``docker run``, a notebook -- detected via nvidia-smi compute processes and
memory usage. So sflow never steals a GPU that something else is already using.

Each run writes one JSON record naming the GPU UUIDs it holds. The claim step
(read registry -> pick from the free set -> write own record) runs under a short
exclusive :func:`fcntl.flock`, so concurrent runs serialize only on that
millisecond-scale critical section and then execute in parallel on disjoint
GPUs. Records whose owning process is dead are reclaimed automatically, so a
crashed run never leaks its GPUs.

Platform note: the registry needs POSIX ``fcntl``. This module must stay
importable everywhere regardless (``sflow.cli.run`` imports :data:`WAIT_FOR_GPUS_ENV`
from it), so ``fcntl`` is imported lazily and :func:`reservation_enabled` reports
``False`` when it is unavailable -- the same guard ``sflow.core.launcher`` uses
for ``pty``/``termios``.

Layering note: this lives in ``utils`` rather than ``core`` because it models
nothing about sflow's domain -- it is host-level machinery (file locks, procfs,
``nvidia-smi``) that happens to be consumed by the docker backend and operator,
plus one env-var constant read by the CLI. Keeping it here lets ``cli`` import it
without reaching into ``core``, and leaves it available to any future intra-node
backend without moving again. It sits next to :mod:`sflow.utils.gpu`, which owns
the ``CUDA_VISIBLE_DEVICES`` parsing this module's callers pair it with.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from sflow.logging import get_logger

_logger = get_logger(__name__)

try:  # POSIX only -- absent on Windows, where reservation is simply disabled.
    import fcntl
except ImportError:  # pragma: no cover - exercised via the import-guard test
    fcntl = None  # type: ignore[assignment]

# Env override for the registry directory. Default lives under the machine-local
# temp dir (NOT $HOME, which may be NFS-shared across hosts and would let records
# from other machines masquerade as local reservations).
_REGISTRY_DIR_ENV = "SFLOW_GPU_RESERVATION_DIR"
_DEFAULT_DIR_NAME = "sflow-gpu-reservations"

# A GPU with at least this much memory already in use is treated as busy even if
# no compute process is visible (e.g. processes owned by other users that
# nvidia-smi hides). Tunable to ignore driver/context overhead.
_BUSY_MEM_MIB_ENV = "SFLOW_GPU_BUSY_MEM_MIB"
_DEFAULT_BUSY_MEM_MIB = 512
# Set to "1" to ignore foreign usage entirely (sflow owns the whole box).
_IGNORE_FOREIGN_ENV = "SFLOW_GPU_IGNORE_FOREIGN"

_NVIDIA_SMI_TIMEOUT_S = 15
DEFAULT_POLL_INTERVAL_S = 5.0

# Upper bound on how long a claim may wait for the registry lock. The critical
# section is milliseconds of local file I/O, so anything near this means the
# holder is wedged (SIGSTOPped, or a hung NFS/FUSE registry dir). Bounded rather
# than blocking because this runs in an uncancellable worker thread: a plain
# blocking flock there would outlive the task and hang the driver at exit.
_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.05

# Run-time toggles set by the CLI (in-process env), read by the backend.
GPU_RESERVATION_ENV = "SFLOW_GPU_RESERVATION"
WAIT_FOR_GPUS_ENV = "SFLOW_WAIT_FOR_GPUS"

# Appended to insufficient-GPU errors so the message itself tells the user how to
# proceed instead of leaving them to find the env knobs in the docs.
_ESCAPE_HATCH_HINT = (
    f"Wait for GPUs instead of failing with --wait-for-gpus <seconds>; if the "
    f"'busy' GPUs are yours to use anyway, set {_IGNORE_FOREIGN_ENV}=1 (ignore "
    f"foreign workloads) or raise {_BUSY_MEM_MIB_ENV} (default "
    f"{_DEFAULT_BUSY_MEM_MIB} MiB); {GPU_RESERVATION_ENV}=0 disables reservation "
    f"entirely."
)


class RegistryLockBusy(RuntimeError):
    """The registry lock could not be taken within :data:`_LOCK_TIMEOUT_S`.

    Transient by nature (a wedged holder, a stalled registry filesystem), so
    callers that are allowed to wait should retry rather than fail the task.
    """


class GpuProbeError(RuntimeError):
    """``nvidia-smi`` exists but could not be queried *this time*.

    Distinct from "no GPUs on this host": a timeout under load, a transient
    non-zero exit during an ECC/Xid event, or an OS error are all recoverable,
    and must NOT be read as an empty GPU inventory -- doing so would drop the
    reservation entirely and let concurrent runs collide on device 0.
    """


class InsufficientGpusError(RuntimeError):
    """Raised when fewer free GPUs are available than the run requested."""

    def __init__(
        self,
        *,
        requested: int,
        free: int,
        total: int,
        detail: str = "",
    ):
        self.requested = requested
        self.free = free
        self.total = total
        if total == 0:
            message = (
                f"requested {requested} GPU(s) but nvidia-smi reports no GPUs on "
                f"this host"
            )
        else:
            message = (
                f"requested {requested} GPU(s) but only {free} of {total} are free "
                f"on this host (the rest are reserved by other sflow runs or busy "
                f"with foreign workloads)"
            )
            if detail:
                message = f"{message}. {detail}"
            message = f"{message}. {_ESCAPE_HATCH_HINT}"
        super().__init__(message)


@dataclass(frozen=True)
class GpuHandle:
    """A physical GPU as reported by nvidia-smi."""

    index: int
    uuid: str
    memory_used_mib: int = 0


def flock_available() -> bool:
    """Whether the POSIX file locking the registry needs is importable."""
    return fcntl is not None


def registry_dir() -> Path:
    """The machine-local directory holding one JSON record per active run."""
    override = os.environ.get(_REGISTRY_DIR_ENV)
    if override:
        return Path(override)
    import tempfile

    return Path(tempfile.gettempdir()) / _DEFAULT_DIR_NAME


def _ensure_registry_dir() -> Path:
    d = registry_dir()
    # World-usable so a second concurrent run (same or different user) can read
    # and add its own record; individual records still name their owning PID.
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o1777)
    except OSError:
        pass
    return d


@contextmanager
def _registry_lock() -> Iterator[None]:
    """Hold an exclusive flock over the registry for a read-modify-write.

    Held only for the claim/release bookkeeping (milliseconds), never for the
    lifetime of a run and never across an ``nvidia-smi`` call.
    """
    if fcntl is None:  # pragma: no cover - callers gate on reservation_enabled()
        raise RuntimeError("GPU reservation requires POSIX fcntl (not available)")
    d = _ensure_registry_dir()
    lock_path = d / ".lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        # os.open() applies the process umask, so the mode above typically lands
        # as 0o644/0o664 and a *different* user then cannot open the lock O_RDWR
        # -- which would break the cross-user concurrency this registry exists
        # for. Force the mode explicitly (best-effort: only the owner may chmod).
        try:
            os.fchmod(fd, 0o666)
        except OSError:
            pass
        # Bounded non-blocking acquire. A plain blocking LOCK_EX here would park
        # an uncancellable worker thread forever if the holder is wedged, and the
        # interpreter joins executor threads on the way out -- so the driver could
        # not be killed with Ctrl-C. Spin briefly instead and give up loudly.
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RegistryLockBusy(
                        f"GPU reservation registry at {lock_path} stayed locked for "
                        f"{_LOCK_TIMEOUT_S:.0f}s; another process may be stuck or the "
                        f"registry filesystem may be unresponsive"
                    ) from None
                time.sleep(_LOCK_POLL_S)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _busy_mem_threshold_mib() -> int:
    try:
        return int(os.environ.get(_BUSY_MEM_MIB_ENV, _DEFAULT_BUSY_MEM_MIB))
    except ValueError:
        return _DEFAULT_BUSY_MEM_MIB


def discover_gpus() -> list[GpuHandle]:
    """Physical GPUs on this host via nvidia-smi, or ``[]`` if none/unavailable."""
    out = _run_nvidia_smi(
        ["--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]
    )
    if out is None:
        return []
    gpus: list[GpuHandle] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 3 or not parts[0]:
            continue
        try:
            gpus.append(
                GpuHandle(
                    index=int(parts[0]),
                    uuid=parts[1],
                    memory_used_mib=int(float(parts[2])),
                )
            )
        except ValueError:
            continue
    return gpus


def _foreign_busy_uuids(gpus: list[GpuHandle]) -> set[str]:
    """UUIDs of GPUs occupied by non-sflow work: any live compute process, or
    memory usage above the threshold. Empty when foreign detection is disabled."""
    if os.environ.get(_IGNORE_FOREIGN_ENV) == "1":
        return set()

    busy: set[str] = set()
    threshold = _busy_mem_threshold_mib()
    for g in gpus:
        if g.memory_used_mib >= threshold:
            busy.add(g.uuid)

    # A failure here only costs us the compute-process signal; the memory
    # threshold above still stands. Degrade instead of failing the claim, since
    # some drivers/MIG setups reject this query outright.
    try:
        out = _run_nvidia_smi(
            ["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
        )
    except GpuProbeError as e:
        _logger.warning(
            f"Could not list GPU compute processes ({e}); foreign-workload "
            f"detection is falling back to memory usage alone."
        )
        out = None
    if out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.strip().split(",")]
            if parts and parts[0].startswith("GPU-"):
                busy.add(parts[0])
    return busy


def _run_nvidia_smi(args: list[str]) -> str | None:
    """stdout of ``nvidia-smi <args>``, or ``None`` when nvidia-smi is absent.

    Raises :class:`GpuProbeError` when nvidia-smi *exists* but this query failed
    (timeout under load, transient non-zero exit, OS error). Only a missing
    binary means "this host has no GPU tooling"; conflating the two would report
    an empty inventory and silently drop the reservation.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except FileNotFoundError:
        _logger.debug("nvidia-smi is not installed on this host")
        return None
    except (subprocess.TimeoutExpired, OSError) as e:
        raise GpuProbeError(
            f"nvidia-smi did not answer within {_NVIDIA_SMI_TIMEOUT_S}s "
            f"({type(e).__name__})"
        ) from e
    if result.returncode != 0:
        raise GpuProbeError(
            f"nvidia-smi exited {result.returncode}: "
            f"{(result.stderr or '').strip()[:200]}"
        )
    return result.stdout


def process_start_ticks(pid: int) -> int | None:
    """The process' start time in clock ticks since boot, or ``None``.

    Reads field 22 of ``/proc/<pid>/stat``. Recorded alongside the PID so a
    *recycled* PID cannot make a dead run's reservation look alive forever.
    Returns ``None`` where procfs is unavailable, in which case the PID alone is
    used (the pre-existing behavior).
    """
    if pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return None
    # The comm field (2) is parenthesized and may itself contain spaces and
    # parentheses, so split after its final ')'.
    close = raw.rfind(")")
    if close == -1:
        return None
    fields = raw[close + 2 :].split()
    # After comm, field 3 is state; starttime is field 22 overall -> index 19.
    if len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def _pid_alive(pid: int, start_ticks: int | None = None) -> bool:
    """Whether ``pid`` is still running (and, if known, is the *same* process).

    ``start_ticks`` is the value :func:`process_start_ticks` returned when the
    reservation was written. When both it and the current reading are available
    they must match, otherwise the PID has been recycled by an unrelated process
    and the original owner is gone.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user (still a live holder of its GPUs).
        pass
    except OSError:
        return False
    if start_ticks is not None:
        current = process_start_ticks(pid)
        if current is not None and current != start_ticks:
            return False
    return True


def _live_reservations(d: Path) -> list[dict]:
    """Reservation records with a living owner. Reclaims dead-owner records by
    deleting their files (callers already hold the registry lock)."""
    live: list[dict] = []
    for path in sorted(d.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            # Unreadable/corrupt record: drop it so it can't wedge the registry.
            _unlink_quietly(path)
            continue
        try:
            pid = int(record.get("pid", -1))
        except (TypeError, ValueError):
            pid = -1
        start_ticks = record.get("pid_start_ticks")
        if not isinstance(start_ticks, int):
            start_ticks = None
        host = record.get("hostname")
        if host is not None and host != socket.gethostname():
            # Registry shared across hosts (SFLOW_GPU_RESERVATION_DIR pointed at
            # a network path): this machine's process table says nothing about
            # another machine's PIDs, so treat a foreign host's record as live.
            # Reclaiming it would hand its still-held GPUs to a second run.
            live.append(record)
            continue
        if _pid_alive(pid, start_ticks):
            live.append(record)
        else:
            _logger.debug(
                f"Reclaiming GPU reservation from dead pid {pid}: {path.name}"
            )
            _unlink_quietly(path)
    return live


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _record_path(d: Path, run_id: str) -> Path:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in run_id)
    if safe != run_id:
        # Sanitizing is lossy -- task names "a/b" and "a-b" both collapse to
        # "a-b", so two live tasks would share one record file and the second
        # write would erase the first task's claim. Disambiguate with a short
        # digest of the original id; ids needing no sanitizing keep their plain
        # name, so existing records and their filenames are unchanged.
        digest = hashlib.sha1(run_id.encode("utf-8", "replace")).hexdigest()[:8]
        safe = f"{safe}-{digest}"
    return d / f"{safe or 'run'}.json"


def _registry_uuid_sets() -> tuple[set[str], set[str], set[str]]:
    """``(reserved, owned, handed_back)`` UUID sets (caller must hold the lock).

    ``reserved`` are devices this process may not take. ``owned`` is every device
    held by *any* live sflow record, including ones handed back for reuse
    (``gpus.release_after: task_ready``). ``owned`` exists purely to suppress
    foreign-busy detection: the workload sitting on those GPUs is sflow's own, and
    without the distinction the still-serving task would be read as a stranger and
    re-block the very GPU the planner scheduled its successor onto.

    A lent-or-handed-over record is claimable **only by the run that owns it**. That hand-back
    is a within-workflow contract -- the planner packed *this* run's successor onto
    those devices -- so another ``sflow run`` must still see them as taken. Letting
    it claim them would co-locate a second workload on a GPU whose server is very
    much alive, and because ``owned`` also disables foreign-busy detection for
    exactly those devices, nothing downstream would catch it.

    ``handed_back`` is that same set from this process's side: devices THIS run
    released for reuse and may claim again. :func:`try_reserve_gpus` prefers them,
    which is what turns the planner's packing into the placement that actually
    happens -- see the selection comment there.
    """
    reserved: set[str] = set()
    owned: set[str] = set()
    handed_back: set[str] = set()
    me = os.getpid()
    hostname = socket.gethostname()
    for record in _live_reservations(registry_dir()):
        uuids = record.get("gpu_uuids") or []
        owned.update(uuids)
        # Hostname is part of the identity: with a shared registry dir, another
        # machine's record can carry the same PID number as ours.
        mine = record.get("pid") == me and record.get("hostname") == hostname
        if mine and record.get("state") in _CLAIMABLE:
            handed_back.update(uuids)
        else:
            reserved.update(uuids)
    return reserved, owned, handed_back


def _busy_detail(
    gpus: list[GpuHandle], reserved: set[str], foreign: set[str]
) -> str:
    """One-line summary of *why* each unavailable GPU is unavailable."""
    parts: list[str] = []
    for g in gpus:
        if g.uuid in reserved:
            parts.append(f"GPU {g.index}: reserved by another sflow run")
        elif g.uuid in foreign:
            parts.append(f"GPU {g.index}: foreign workload ({g.memory_used_mib} MiB)")
    return ("Busy: " + "; ".join(parts)) if parts else ""


def try_reserve_gpus(count: int, run_id: str) -> list[GpuHandle]:
    """One non-blocking attempt to claim ``count`` free GPUs for ``run_id``.

    Returns the reserved handles, or raises :class:`InsufficientGpusError` when
    not enough are free *right now*. Never sleeps, so a caller that wants to wait
    can do so on its own terms -- on an event loop, under a task timeout, and
    interruptible by Ctrl-C. :func:`reserve_gpus` is the blocking convenience
    wrapper around this for synchronous callers.
    """
    if count <= 0:
        return []

    # nvidia-smi (up to two subprocess calls, seconds each under load) runs
    # OUTSIDE the lock: holding an exclusive flock across it would serialize
    # every concurrent run behind the slowest driver query. Foreign-busy
    # detection is inherently a snapshot anyway; only the sflow-vs-sflow
    # reservation read/write below needs to be atomic.
    gpus = discover_gpus()
    total = len(gpus)
    foreign = _foreign_busy_uuids(gpus)

    with _registry_lock():
        reserved, owned, handed_back = _registry_uuid_sets()
        # Subtract sflow's own devices from the foreign set: a task that released
        # at READY for reuse is still running on its GPU, and must not be
        # mistaken for a stranger's workload sitting on it.
        foreign = foreign - owned
        unavailable = reserved | foreign
        free = [g for g in gpus if g.uuid not in unavailable]

        if len(free) >= count:
            # Take devices THIS run handed back at READY before any other free
            # one. Both are legal -- everything in `free` is claimable -- but only
            # this order matches the plan, and picking by lowest index instead
            # goes wrong in two ways at once:
            #
            #   * the successor the planner packed onto its predecessor's devices
            #     lands somewhere else entirely whenever another run happens to
            #     free a lower-numbered GPU first, which is pure timing; and
            #   * the predecessor's hand-back is then never superseded, so its
            #     record keeps those devices blocked from every OTHER
            #     run (that is the point of the record) while this run has quietly
            #     gone and used different ones. The devices sit idle until the
            #     driver exits.
            #
            # Stable sort, so within each group the nvidia-smi index order that
            # made packing deterministic in the first place is preserved.
            free.sort(key=lambda g: (g.uuid not in handed_back, g.index))
            chosen = free[:count]
            # Establish the new claim BEFORE releasing the old one. Both orders
            # look fine until a write fails (full disk, unwritable registry):
            # dropping first would leave the devices claimed by nobody while this
            # run still intends to use them, so another process could take them.
            # This way a failure leaves the predecessor's hand-over standing --
            # over-holding, which is the recoverable direction.
            _write_record(run_id, [g.uuid for g in chosen])
            _drop_superseded_claims({g.uuid for g in chosen}, run_id)
            _logger.info(
                f"Reserved {count} GPU(s) for run '{run_id}': "
                f"devices {[g.index for g in chosen]} "
                f"({len(free) - count} of {total} still free)"
            )
            return chosen

    raise InsufficientGpusError(
        requested=count,
        free=len(free),
        total=total,
        detail=_busy_detail(gpus, reserved, foreign),
    )


def _drop_superseded_claims(taken: set[str], new_run_id: str) -> None:
    """Transfer ownership: forget devices this run is handing to a new claim.

    A predecessor that handed back (``lent``/``handover``) keeps its record so no OTHER
    run can take its devices before this run's successor arrives. Once the
    successor claims them, that record is stale -- and leaving it would be a slow
    leak: it keeps naming devices this run no longer holds, so they would stay
    blocked for every other process until the driver exits, exactly the
    over-holding the per-task model exists to avoid.

    Except when the predecessor is still RUNNING on them (the ``task_ready``
    hand-back): it gave away its claim, not its occupancy, and its record is the
    only thing keeping another run off a live server's GPU -- dropping it here
    published them the moment the (usually shorter-lived) successor exited. So
    those are merely *marked* superseded; :func:`release_gpus` forgets them when
    the owner itself exits, which is when they are genuinely free.

    Only this process's own lent/handed-over records are touched, and only the devices
    actually taken; a record still naming other devices keeps them. Caller holds
    the registry lock.
    """
    d = registry_dir()
    if not d.exists():
        return
    me = os.getpid()
    hostname = socket.gethostname()
    new_path = _record_path(d, new_run_id)
    for path in sorted(d.glob("*.json")):
        if path == new_path:
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("state") not in _CLAIMABLE:
            continue  # a live claim, not a hand-over -- never ours to drop
        if record.get("pid") != me or record.get("hostname") != hostname:
            continue  # another run's; it alone decides when to let go
        held = record.get("gpu_uuids") or []
        if record.get("state") == _LENT:
            previous = set(record.get("superseded") or [])
            superseded = previous | (taken & set(held))
            if superseded == previous:
                continue
            record["superseded"] = sorted(superseded)
            _replace_record(path, record)
            continue
        remaining = [u for u in held if u not in taken]
        if len(remaining) == len(held):
            continue
        if remaining:
            record["gpu_uuids"] = remaining
            _replace_record(path, record)
        else:
            _unlink_quietly(path)


def _replace_record(path: Path, record: dict) -> None:
    """Publish a record atomically. Caller holds the registry lock."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    os.replace(tmp, path)  # atomic publish


def release_all_for_pid(pid: int | None = None) -> int:
    """Drop every reservation record owned by ``pid`` (default: this process).

    Run-end backstop: a task cancelled at the exact moment its reservation was
    being written can leave a record behind that its own release never saw. The
    owning process is the authority on what it still holds, so on the way out it
    clears anything left in its name. Returns how many records were removed.
    """
    target = os.getpid() if pid is None else pid
    removed = 0
    if not registry_dir().exists():  # never reserved -> nothing to sweep
        return 0
    try:
        with _registry_lock():
            for path in sorted(registry_dir().glob("*.json")):
                try:
                    record = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if record.get("pid") == target:
                    _unlink_quietly(path)
                    removed += 1
    except (OSError, RuntimeError) as e:  # fcntl missing / unwritable registry
        _logger.debug(f"GPU registry sweep for pid {target} failed: {e}")
        return 0
    if removed:
        _logger.debug(f"Released {removed} leftover GPU reservation(s) for pid {target}")
    return removed


# A record is in exactly one of these. Kept as one field rather than a pair of
# booleans: the three states are mutually exclusive, and "reusable and not
# owner_running" is a puzzle where "handover" is a statement.
_HELD = "held"          # the task is running and owns these devices outright
_LENT = "lent"          # READY hand-back: still running, but this run may reuse them
_HANDOVER = "handover"  # the task has exited; a successor of this run is planned here
# The two states in which the devices are claimable again -- by THIS run only.
_CLAIMABLE = (_LENT, _HANDOVER)


def _write_record(run_id: str, gpu_uuids: list[str]) -> None:
    d = _ensure_registry_dir()
    pid = os.getpid()
    record = {
        "run_id": run_id,
        "pid": pid,
        # Guards against PID reuse: a recycled PID has a different start time.
        "pid_start_ticks": process_start_ticks(pid),
        "hostname": socket.gethostname(),
        "gpu_uuids": gpu_uuids,
        "state": _HELD,
        "created_at": time.time(),
    }
    _replace_record(_record_path(d, run_id), record)


def release_gpus(
    run_id: str, *, reusable: bool = False, handover: bool = False
) -> None:
    """Release ``run_id``'s reservation. Idempotent; safe if never reserved.

    ``reusable=True`` (``gpus.release_after: task_ready``) is the READY hand-back:
    the task keeps running. The record stays and turns claimable, so the next task
    **of this same run** can take the devices while the server serves, and the
    workload on them stays attributable to sflow rather than looking foreign. To
    every other run the devices remain taken -- the hand-back is a within-workflow
    contract and the server is very much alive on them.

    ``handover=True`` is the task's *exit* when the planner packed a successor of
    this run onto its devices. The task occupies nothing any more, so this drops
    every device the successor already took and keeps only the rest -- those still
    need holding across the gap until the successor is submitted, or a concurrent
    run would take a device this workflow is counting on.

    With neither flag set this is the ordinary hard release: the record is
    deleted and the devices go back to the whole host.
    """
    # Taking the lock would CREATE the registry directory, so a run that never
    # reserved anything (CPU-only task, reservation disabled) would litter the
    # host with an empty world-writable dir. No registry -> nothing to release.
    if not registry_dir().exists():
        return
    try:
        with _registry_lock():
            path = _record_path(registry_dir(), run_id)
            if not (reusable or handover):
                _unlink_quietly(path)
                return
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                return  # never reserved, already gone, or unreadable
            if reusable:
                record["state"] = _LENT
                _replace_record(path, record)
                return
            # The owner has exited. Devices its successor already claimed are
            # accounted for by that claim (and were only still named here because
            # the owner was live on them); what is left is the hand-over gap.
            superseded = set(record.get("superseded") or [])
            remaining = [
                u for u in (record.get("gpu_uuids") or []) if u not in superseded
            ]
            if not remaining:
                _unlink_quietly(path)
                return
            record["gpu_uuids"] = remaining
            record["state"] = _HANDOVER
            _replace_record(path, record)
    except (OSError, RuntimeError) as e:
        _logger.debug(f"GPU release for '{run_id}' failed: {e}")


def reservation_enabled() -> bool:
    """True unless disabled via ``SFLOW_GPU_RESERVATION=0`` or fcntl is missing."""
    if not flock_available():
        return False
    return os.environ.get(GPU_RESERVATION_ENV, "1") != "0"


def validate_wait_value(raw: str) -> None:
    """Validate a ``--wait-for-gpus`` value, raising ``ValueError`` if malformed.

    Called by the CLI at parse time so a typo like ``--wait-for-gpus 600s`` fails
    immediately rather than at the first GPU task of a long run.
    """
    _parse_wait_env(raw)


def _parse_wait_env(raw: str) -> tuple[bool, float | None]:
    """``(wait, timeout)`` for a raw ``SFLOW_WAIT_FOR_GPUS`` value."""
    raw = raw.strip()
    if not raw:
        return (True, None)
    try:
        timeout = float(raw)
    except ValueError as e:
        # A typo like `--wait-for-gpus 600s` must not silently become "wait
        # forever" -- fail loudly, matching the config path's validation.
        raise ValueError(
            f"{WAIT_FOR_GPUS_ENV} must be empty or a non-negative number, got {raw!r}"
        ) from e
    if timeout < 0:
        raise ValueError(f"{WAIT_FOR_GPUS_ENV} must be >= 0, got {raw!r}")
    # 0 means "wait indefinitely", the same as the recipe field. These two are an
    # override pair (--wait-for-gpus overrides wait_for_gpus), so the identical
    # literal must not mean opposite things on the two surfaces. Failing fast is
    # still the default -- reached by not setting either one.
    return (True, None if timeout == 0 else timeout)


def validate_config_wait_for_gpus(value: int, *, where: str) -> int:
    """Validate a recipe-level ``wait_for_gpus`` (``>= 0``); return it unchanged.

    Shared by the docker backend's config resolution and :func:`wait_options` so
    the same rule is stated once. ``where`` names the source for the message.
    """
    n = int(value)
    if n < 0:
        raise ValueError(f"{where} wait_for_gpus must be >= 0, got {n}".lstrip())
    return n


def wait_options(config_value: int | None = None) -> tuple[bool, float | None]:
    """``(wait, timeout)`` for GPU reservation.

    The ``SFLOW_WAIT_FOR_GPUS`` env (set by ``--wait-for-gpus``) wins: empty ->
    wait forever, a number -> that many seconds. Absent, the backend's
    ``wait_for_gpus`` ``config_value`` applies: ``None`` -> fail fast, ``0`` ->
    wait forever, ``N`` -> wait up to N seconds.
    """
    raw = os.environ.get(WAIT_FOR_GPUS_ENV)
    if raw is None:
        if config_value is None:
            return (False, None)
        n = validate_config_wait_for_gpus(config_value, where="")
        return (True, None if n == 0 else float(n))
    return _parse_wait_env(raw)


def make_run_id(task_name: str) -> str:
    """A registry-unique id for one task's reservation: ``<pid>-<task>``.

    Unique among all live reservations -- the PID differs across processes, and
    task names are unique within one workflow. The PID (plus its start time) also
    drives dead-owner reclaim.
    """
    return f"{os.getpid()}-{task_name}"
