# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the cross-process GPU reservation registry.

Real nvidia-smi and real sleeping are stubbed; the flock registry runs against a
temp directory so the read-modify-write / stale-reclaim logic is exercised for
real without a GPU.
"""

import fcntl
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

import sflow.utils.gpu_reservation as gr
from sflow.utils.gpu_reservation import GpuHandle, InsufficientGpusError


@pytest.fixture(autouse=True)
def _registry(tmp_path, monkeypatch):
    monkeypatch.setenv("SFLOW_GPU_RESERVATION_DIR", str(tmp_path / "reg"))
    # Foreign detection off by default; individual tests opt back in.
    monkeypatch.setenv("SFLOW_GPU_IGNORE_FOREIGN", "1")
    return tmp_path


def _fake_gpus(n, used_mib=0):
    return [GpuHandle(index=i, uuid=f"GPU-{i:04d}", memory_used_mib=used_mib) for i in range(n)]


def _patch_gpus(monkeypatch, gpus):
    monkeypatch.setattr(gr, "discover_gpus", lambda: list(gpus))


def _uuids(handles):
    """reserve_gpus returns GpuHandles (index + uuid); compare on the uuids."""
    return [h.uuid for h in handles]


# ---------------------------------------------------------------------------
# basic reserve / release
# ---------------------------------------------------------------------------


def test_reserve_returns_requested_uuids(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(8))
    handles = gr.try_reserve_gpus(2, "run-a")
    assert _uuids(handles) == ["GPU-0000", "GPU-0001"]
    # The physical device indices come back too, for run reporting.
    assert [h.index for h in handles] == [0, 1]


def test_reserve_zero_is_noop(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(8))
    assert gr.try_reserve_gpus(0, "run-a") == []
    # no record written
    assert list((gr.registry_dir()).glob("*.json")) == []


def test_two_runs_get_disjoint_gpus(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(8))
    a = _uuids(gr.try_reserve_gpus(2, "run-a"))
    b = _uuids(gr.try_reserve_gpus(2, "run-b"))
    assert set(a).isdisjoint(b)
    assert a == ["GPU-0000", "GPU-0001"]
    assert b == ["GPU-0002", "GPU-0003"]


def test_release_frees_gpus_for_next_run(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(4))
    gr.try_reserve_gpus(4, "run-a")
    with pytest.raises(InsufficientGpusError):
        gr.try_reserve_gpus(1, "run-b")
    gr.release_gpus("run-a")
    # now the whole board is free again
    assert _uuids(gr.try_reserve_gpus(4, "run-b")) == [
        "GPU-0000",
        "GPU-0001",
        "GPU-0002",
        "GPU-0003",
    ]


def test_release_is_idempotent(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(2))
    gr.release_gpus("never-reserved")  # must not raise
    gr.try_reserve_gpus(1, "run-a")
    gr.release_gpus("run-a")
    gr.release_gpus("run-a")  # double release ok


# ---------------------------------------------------------------------------
# insufficient GPUs
#
# try_reserve_gpus never waits -- callers that want to retry do it on their own
# clock so the waiting stays cancellable. That retry policy is covered where it
# lives: tests/unit/test_docker_gpu_reservation.py (operator budget) and
# tests/unit/test_core_orchestrator_container_teardown.py (loop, cancel, timeout).
# ---------------------------------------------------------------------------


def test_fail_fast_when_not_enough_free(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(2))
    with pytest.raises(InsufficientGpusError) as ei:
        gr.try_reserve_gpus(4, "run-a")
    assert ei.value.requested == 4
    assert ei.value.free == 2
    assert ei.value.total == 2






# ---------------------------------------------------------------------------
# stale (dead-PID) reclaim
# ---------------------------------------------------------------------------


def test_dead_pid_reservation_is_reclaimed(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(2))
    # Hand-write a record owned by a dead pid holding both GPUs.
    d = gr._ensure_registry_dir()
    (d / "ghost.json").write_text(
        json.dumps({"run_id": "ghost", "pid": 999999999, "gpu_uuids": ["GPU-0000", "GPU-0001"]})
    )
    monkeypatch.setattr(gr, "_pid_alive", lambda pid, start=None: pid != 999999999)

    # The dead holder's GPUs are reclaimed, so a new run can take them.
    assert _uuids(gr.try_reserve_gpus(2, "run-a")) == ["GPU-0000", "GPU-0001"]
    assert not (d / "ghost.json").exists()


def test_live_pid_reservation_is_respected(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(2))
    d = gr._ensure_registry_dir()
    (d / "live.json").write_text(
        json.dumps({"run_id": "live", "pid": 4242, "gpu_uuids": ["GPU-0000"]})
    )
    monkeypatch.setattr(gr, "_pid_alive", lambda pid, start=None: True)
    # Only GPU-0001 is free.
    assert _uuids(gr.try_reserve_gpus(1, "run-a")) == ["GPU-0001"]
    with pytest.raises(InsufficientGpusError):
        gr.try_reserve_gpus(2, "run-b")


def test_corrupt_record_is_dropped(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(1))
    d = gr._ensure_registry_dir()
    (d / "bad.json").write_text("{not json")
    # Corrupt record must not wedge the registry.
    assert _uuids(gr.try_reserve_gpus(1, "run-a")) == ["GPU-0000"]
    assert not (d / "bad.json").exists()


# ---------------------------------------------------------------------------
# foreign (non-sflow) GPU usage
# ---------------------------------------------------------------------------


def test_foreign_gpu_with_process_is_skipped(monkeypatch):
    monkeypatch.delenv("SFLOW_GPU_IGNORE_FOREIGN", raising=False)
    _patch_gpus(monkeypatch, _fake_gpus(2))

    # GPU-0000 has a foreign compute process; GPU-0001 is idle.
    def _fake_smi(args):
        if any("compute-apps" in a for a in args):
            return "GPU-0000, 12345\n"
        return ""

    monkeypatch.setattr(gr, "_run_nvidia_smi", _fake_smi)
    assert _uuids(gr.try_reserve_gpus(1, "run-a")) == ["GPU-0001"]
    with pytest.raises(InsufficientGpusError):
        gr.try_reserve_gpus(2, "run-b")


def test_foreign_gpu_with_high_memory_is_skipped(monkeypatch):
    monkeypatch.delenv("SFLOW_GPU_IGNORE_FOREIGN", raising=False)
    monkeypatch.setenv("SFLOW_GPU_BUSY_MEM_MIB", "500")
    # GPU-0000 has 1024 MiB used (busy), GPU-0001 has 1024 too -> both busy.
    gpus = [
        GpuHandle(index=0, uuid="GPU-0000", memory_used_mib=1024),
        GpuHandle(index=1, uuid="GPU-0001", memory_used_mib=10),
    ]
    _patch_gpus(monkeypatch, gpus)
    monkeypatch.setattr(gr, "_run_nvidia_smi", lambda args: "")
    # Only the low-memory GPU-0001 is claimable.
    assert _uuids(gr.try_reserve_gpus(1, "run-a")) == ["GPU-0001"]


def test_ignore_foreign_env_disables_detection(monkeypatch):
    monkeypatch.setenv("SFLOW_GPU_IGNORE_FOREIGN", "1")
    gpus = [GpuHandle(index=0, uuid="GPU-0000", memory_used_mib=40000)]
    _patch_gpus(monkeypatch, gpus)
    # Even though the GPU is heavily used, foreign detection is off -> claimable.
    assert _uuids(gr.try_reserve_gpus(1, "run-a")) == ["GPU-0000"]


# ---------------------------------------------------------------------------
# no GPUs on host
# ---------------------------------------------------------------------------


def test_no_gpus_available_raises_for_positive_request(monkeypatch):
    _patch_gpus(monkeypatch, [])
    with pytest.raises(InsufficientGpusError) as ei:
        gr.try_reserve_gpus(1, "run-a")
    assert ei.value.total == 0


# ---------------------------------------------------------------------------
# discover_gpus parsing
# ---------------------------------------------------------------------------


def test_discover_gpus_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(
        gr,
        "_run_nvidia_smi",
        lambda args: "0, GPU-aaaa, 0\n1, GPU-bbbb, 1234\n\n",
    )
    gpus = gr.discover_gpus()
    assert [(g.index, g.uuid, g.memory_used_mib) for g in gpus] == [
        (0, "GPU-aaaa", 0),
        (1, "GPU-bbbb", 1234),
    ]


def test_discover_gpus_empty_when_no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gr, "_run_nvidia_smi", lambda args: None)
    assert gr.discover_gpus() == []


# ---------------------------------------------------------------------------
# cross-user / registry hygiene
# ---------------------------------------------------------------------------


def test_lock_file_is_writable_by_other_users(monkeypatch):
    # os.open() applies the umask, so the 0o666 mode alone typically lands as
    # 0o644 and a *second user* could not open the lock O_RDWR -- breaking the
    # cross-user concurrency this registry exists for. The mode must be forced.
    prev = gr.os.umask(0o022)  # the common default that used to strip the bits
    try:
        with gr._registry_lock():
            pass
    finally:
        gr.os.umask(prev)
    mode = (gr.registry_dir() / ".lock").stat().st_mode & 0o777
    assert mode & 0o020, f"group-writable bit missing from lock mode {oct(mode)}"
    assert mode & 0o002, f"other-writable bit missing from lock mode {oct(mode)}"


def test_registry_dir_is_world_usable(monkeypatch):
    d = gr._ensure_registry_dir()
    mode = d.stat().st_mode & 0o7777
    assert mode & 0o777 == 0o777
    assert mode & 0o1000, "sticky bit keeps users from deleting each other's records"


def test_recycled_pid_does_not_keep_a_dead_reservation_alive(monkeypatch):
    # A dead run's PID can be handed to an unrelated process; the recorded start
    # time distinguishes them, so the stale reservation is still reclaimed.
    _patch_gpus(monkeypatch, _fake_gpus(2))
    d = gr._ensure_registry_dir()
    (d / "ghost.json").write_text(
        json.dumps(
            {
                "run_id": "ghost",
                "pid": 4242,
                "pid_start_ticks": 111,
                "gpu_uuids": ["GPU-0000", "GPU-0001"],
            }
        )
    )
    # PID 4242 exists again, but as a process that started later.
    monkeypatch.setattr(gr.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(gr, "process_start_ticks", lambda pid: 999)

    assert _uuids(gr.try_reserve_gpus(2, "run-a")) == ["GPU-0000", "GPU-0001"]
    assert not (d / "ghost.json").exists()


def test_same_pid_and_start_time_is_still_respected(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(2))
    d = gr._ensure_registry_dir()
    (d / "live.json").write_text(
        json.dumps(
            {
                "run_id": "live",
                "pid": 4242,
                "pid_start_ticks": 111,
                "gpu_uuids": ["GPU-0000"],
            }
        )
    )
    monkeypatch.setattr(gr.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(gr, "process_start_ticks", lambda pid: 111)
    assert _uuids(gr.try_reserve_gpus(1, "run-a")) == ["GPU-0001"]


def test_records_carry_the_owning_process_start_time(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "run-a")
    record = json.loads(next(gr.registry_dir().glob("*.json")).read_text())
    assert record["pid"] == gr.os.getpid()
    # On Linux this is a real tick count; elsewhere None (PID-only, as before).
    assert record["pid_start_ticks"] == gr.process_start_ticks(gr.os.getpid())


def test_other_users_process_counts_as_alive():
    # kill -0 raises EPERM for a process owned by another user; that means the
    # process EXISTS, so its reservation must be respected, not reclaimed.
    import os as _os

    def _eperm(pid, sig):
        raise PermissionError(1, "Operation not permitted")

    real_kill = _os.kill
    _os.kill = _eperm
    try:
        assert gr._pid_alive(4242) is True
    finally:
        _os.kill = real_kill


# ---------------------------------------------------------------------------
# error reporting + no-GPU short circuit
# ---------------------------------------------------------------------------


def test_no_gpus_on_the_host_is_reported_distinctly(monkeypatch):
    # "no GPUs at all" is not the same as "all busy": it can never improve, so
    # callers must be able to tell the two apart and not retry this one.
    # The waiting side of that rule lives with the retrying caller, see
    # test_a_host_with_no_gpus_is_never_retried in test_docker_gpu_reservation.py.
    _patch_gpus(monkeypatch, [])
    with pytest.raises(InsufficientGpusError) as ei:
        gr.try_reserve_gpus(1, "run-a")
    assert ei.value.total == 0
    assert "no GPUs on this host" in str(ei.value)


def test_error_names_the_escape_hatches_and_why_each_gpu_is_busy(monkeypatch):
    monkeypatch.delenv("SFLOW_GPU_IGNORE_FOREIGN", raising=False)
    gpus = [
        GpuHandle(index=0, uuid="GPU-0000", memory_used_mib=0),
        GpuHandle(index=1, uuid="GPU-0001", memory_used_mib=4096),
    ]
    _patch_gpus(monkeypatch, gpus)
    monkeypatch.setattr(gr, "_run_nvidia_smi", lambda args: "")
    gr.try_reserve_gpus(1, "holder")  # takes GPU-0000

    with pytest.raises(InsufficientGpusError) as ei:
        gr.try_reserve_gpus(1, "run-a")
    message = str(ei.value)
    # Says which GPU is unavailable and why...
    assert "GPU 0: reserved by another sflow run" in message
    assert "GPU 1: foreign workload (4096 MiB)" in message
    # ...and how to proceed, so the user isn't sent hunting through docs.
    assert "--wait-for-gpus" in message
    assert "SFLOW_GPU_IGNORE_FOREIGN=1" in message
    assert "SFLOW_GPU_BUSY_MEM_MIB" in message


def test_nvidia_smi_is_not_called_while_holding_the_lock(monkeypatch):
    # Holding the exclusive flock across nvidia-smi (up to 15s each) would
    # serialize every concurrent run behind the slowest driver query.
    _patch_gpus(monkeypatch, _fake_gpus(2))
    seen: list[bool] = []
    real_discover = gr.discover_gpus

    def _spy():
        # Try to take the lock non-blocking from this same process: flock is
        # per-fd, so success proves the caller is not inside _registry_lock.
        d = gr._ensure_registry_dir()
        fd = gr.os.open(str(d / ".lock"), gr.os.O_RDWR | gr.os.O_CREAT, 0o666)
        try:
            gr.fcntl.flock(fd, gr.fcntl.LOCK_EX | gr.fcntl.LOCK_NB)
            seen.append(True)
            gr.fcntl.flock(fd, gr.fcntl.LOCK_UN)
        except OSError:
            seen.append(False)
        finally:
            gr.os.close(fd)
        return real_discover()

    monkeypatch.setattr(gr, "discover_gpus", _spy)
    gr.try_reserve_gpus(1, "run-a")
    assert seen == [True]


# ---------------------------------------------------------------------------
# process_start_ticks: /proc/<pid>/stat parsing (the PID-reuse guard)
# ---------------------------------------------------------------------------


def test_process_start_ticks_reads_our_own_process():
    from pathlib import Path

    ticks = gr.process_start_ticks(gr.os.getpid())
    if not Path("/proc/self/stat").exists():  # pragma: no cover - non-Linux
        assert ticks is None
        return
    assert isinstance(ticks, int) and ticks > 0
    # Stable across calls -- it is a boot-relative start time, not a clock read.
    assert gr.process_start_ticks(gr.os.getpid()) == ticks


def test_process_start_ticks_survives_a_comm_containing_spaces_and_parens(monkeypatch):
    # Field 2 of /proc/<pid>/stat is the executable name in parentheses and may
    # itself contain ')' and spaces, so the parser must split after the LAST ')'.
    # starttime is overall field 22; fields 1-2 are pid and comm, so it sits at
    # index 19 of what remains once the comm is stripped.
    after_comm = ["S"] + [f"f{i}" for i in range(4, 53)]
    after_comm[19] = "987654"  # overall field 22 = starttime
    stat = "4242 (weird )name (x)) " + " ".join(after_comm)

    class _FakePath:
        def __init__(self, _p):
            pass

        def read_text(self):
            return stat

    monkeypatch.setattr(gr, "Path", _FakePath)
    assert gr.process_start_ticks(4242) == 987654


def test_process_start_ticks_returns_none_for_unreadable_or_odd_input(monkeypatch):
    assert gr.process_start_ticks(0) is None
    assert gr.process_start_ticks(-1) is None
    # A pid that cannot exist -> no procfs entry -> None, not an exception.
    assert gr.process_start_ticks(999999999) is None

    class _Truncated:
        def __init__(self, _p):
            pass

        def read_text(self):
            return "4242 (bash) S 1 2 3"

    monkeypatch.setattr(gr, "Path", _Truncated)
    assert gr.process_start_ticks(4242) is None

    class _NoParen:
        def __init__(self, _p):
            pass

        def read_text(self):
            return "garbage without a paren"

    monkeypatch.setattr(gr, "Path", _NoParen)
    assert gr.process_start_ticks(4242) is None


def test_missing_start_ticks_in_a_record_falls_back_to_pid_only(monkeypatch):
    # Records written before this field existed (or on non-procfs hosts) must
    # still be honored using the PID alone.
    _patch_gpus(monkeypatch, _fake_gpus(2))
    d = gr._ensure_registry_dir()
    (d / "old.json").write_text(
        json.dumps({"run_id": "old", "pid": gr.os.getpid(), "gpu_uuids": ["GPU-0000"]})
    )
    assert _uuids(gr.try_reserve_gpus(1, "run-a")) == ["GPU-0001"]


# ---------------------------------------------------------------------------
# release_after: task_ready -- handing a GPU on while still running on it
# ---------------------------------------------------------------------------


def _foreign_detection_on(monkeypatch):
    monkeypatch.delenv("SFLOW_GPU_IGNORE_FOREIGN", raising=False)
    # No compute-process rows; the memory threshold is what marks a GPU busy.
    monkeypatch.setattr(gr, "_run_nvidia_smi", lambda args: "")


def test_reusable_release_hands_the_gpu_on_while_the_task_still_serves(monkeypatch):
    """``gpus.release_after: task_ready`` must survive foreign-busy detection.

    The server keeps running -- and keeps its memory allocated -- after handing
    the claim back, so a foreign check that cannot tell it is sflow's own work
    would re-block the very GPU the planner packed its successor onto.
    """
    _foreign_detection_on(monkeypatch)
    gpus = [GpuHandle(index=0, uuid="GPU-0000", memory_used_mib=0)]
    _patch_gpus(monkeypatch, gpus)

    assert _uuids(gr.try_reserve_gpus(1, "server")) == ["GPU-0000"]
    # The server has come up and is now sitting on the device.
    gpus[0] = GpuHandle(index=0, uuid="GPU-0000", memory_used_mib=40_000)
    gr.release_gpus("server", reusable=True)

    assert _uuids(gr.try_reserve_gpus(1, "client")) == ["GPU-0000"]


def test_a_genuinely_foreign_busy_gpu_is_still_refused(monkeypatch):
    """The counterpart: once sflow no longer owns it, busy means hands off."""
    _foreign_detection_on(monkeypatch)
    gpus = [GpuHandle(index=0, uuid="GPU-0000", memory_used_mib=0)]
    _patch_gpus(monkeypatch, gpus)
    gr.try_reserve_gpus(1, "server")
    gpus[0] = GpuHandle(index=0, uuid="GPU-0000", memory_used_mib=40_000)
    gr.release_gpus("server")  # hard release -> not sflow's any more

    with pytest.raises(InsufficientGpusError):
        gr.try_reserve_gpus(1, "client")


def test_another_run_cannot_claim_gpus_handed_back_for_in_run_reuse(monkeypatch):
    """The ``task_ready`` hand-back is scoped to the run that made it.

    Run A's server marks its GPU reusable and keeps serving on it. That is a
    within-workflow contract: A's planner packed A's *own* successor onto that
    device. A different `sflow run` must still see the GPU as taken -- claiming it
    would put a second workload on a live server's GPU, and because the record
    also suppresses foreign-busy detection for that device, nothing else would
    notice.
    """
    _foreign_detection_on(monkeypatch)
    gpus = [GpuHandle(index=0, uuid="GPU-0000", memory_used_mib=0)]
    _patch_gpus(monkeypatch, gpus)

    gr.try_reserve_gpus(1, "server")
    gpus[0] = GpuHandle(index=0, uuid="GPU-0000", memory_used_mib=40_000)
    gr.release_gpus("server", reusable=True)

    # A second driver process on the same host.
    other_pid = os.getpid() + 1000
    monkeypatch.setattr(os, "getpid", lambda: other_pid)
    with pytest.raises(InsufficientGpusError) as excinfo:
        gr.try_reserve_gpus(1, "other-run")
    assert "reserved by another sflow run" in str(excinfo.value)


def test_a_foreign_hosts_record_is_never_mistaken_for_our_own_reuse(monkeypatch):
    """PID alone is not identity when the registry dir is shared across hosts.

    Another machine can run a driver with the same PID number; its reusable
    record must not become claimable here just because the integers match.
    """
    _foreign_detection_on(monkeypatch)
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "server")
    gr.release_gpus("server", reusable=True)

    record_path = next(gr.registry_dir().glob("*.json"))
    record = json.loads(record_path.read_text())
    record["hostname"] = "some-other-host"  # same pid, different machine
    record_path.write_text(json.dumps(record))

    with pytest.raises(InsufficientGpusError):
        gr.try_reserve_gpus(1, "client")


def test_hard_release_after_a_reusable_one_clears_the_record(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "server")
    gr.release_gpus("server", reusable=True)
    assert list(gr.registry_dir().glob("*.json")), "reusable keeps the record"
    gr.release_gpus("server")
    assert list(gr.registry_dir().glob("*.json")) == []


# ---------------------------------------------------------------------------
# nvidia-smi: "cannot tell" must never be read as "no GPUs"
# ---------------------------------------------------------------------------


def test_nvidia_smi_timeout_raises_instead_of_reporting_an_empty_board(monkeypatch):
    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=15)

    monkeypatch.setattr(gr.subprocess, "run", _timeout)
    with pytest.raises(gr.GpuProbeError):
        gr.discover_gpus()


def test_nvidia_smi_nonzero_exit_raises(monkeypatch):
    monkeypatch.setattr(
        gr.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=9, stdout="", stderr="Xid"),
    )
    with pytest.raises(gr.GpuProbeError):
        gr.discover_gpus()


def test_absent_nvidia_smi_really_does_mean_no_gpus(monkeypatch):
    def _missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gr.subprocess, "run", _missing)
    assert gr.discover_gpus() == []


# ---------------------------------------------------------------------------
# registry robustness
# ---------------------------------------------------------------------------


def test_wait_value_zero_means_forever_on_both_surfaces(monkeypatch):
    monkeypatch.delenv(gr.WAIT_FOR_GPUS_ENV, raising=False)
    assert gr.wait_options(None) == (False, None)  # neither set -> fail fast
    assert gr.wait_options(0) == (True, None)  # recipe field
    monkeypatch.setenv(gr.WAIT_FOR_GPUS_ENV, "0")
    assert gr.wait_options(None) == (True, None)  # --wait-for-gpus 0
    monkeypatch.setenv(gr.WAIT_FOR_GPUS_ENV, "30")
    assert gr.wait_options(None) == (True, 30.0)


def test_task_names_that_sanitize_alike_keep_separate_records(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(4))
    a = _uuids(gr.try_reserve_gpus(1, "a/b"))
    b = _uuids(gr.try_reserve_gpus(1, "a-b"))
    assert set(a).isdisjoint(b), "one record must not overwrite the other"
    assert len(list(gr.registry_dir().glob("*.json"))) == 2
    # Ids needing no sanitizing keep their plain filename (existing records).
    assert gr._record_path(gr.registry_dir(), "1234-server").name == "1234-server.json"


def test_records_owned_by_another_host_are_never_reclaimed(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(1))
    d = gr._ensure_registry_dir()
    (d / "other.json").write_text(
        json.dumps(
            {
                "run_id": "other",
                "pid": 999_999_000,  # not a live pid *here*
                "hostname": "some-other-host",
                "gpu_uuids": ["GPU-0000"],
            }
        )
    )
    # A shared registry dir must not let this host free another host's GPUs.
    with pytest.raises(InsufficientGpusError):
        gr.try_reserve_gpus(1, "mine")
    assert (d / "other.json").exists()


def test_registry_lock_gives_up_rather_than_blocking_forever(monkeypatch):
    """A wedged holder must not park the uncancellable acquire thread."""
    monkeypatch.setattr(gr, "_LOCK_TIMEOUT_S", 0.1)
    d = gr._ensure_registry_dir()
    fd = os.open(str(d / ".lock"), os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(gr.RegistryLockBusy):
            with gr._registry_lock():
                pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------
# Hand-over on completion: the successor's GPU must not be published early
# ---------------------------------------------------------------------------


def test_a_handed_over_device_is_blocked_for_other_runs_but_free_for_this_one(
    monkeypatch,
):
    """The gap this closes: `taskx` completes and `tasky` is planned onto its GPU,
    but is not submitted until the next poll tick. Deleting the record in that gap
    publishes the device, and a waiting `sflow run` takes it -- failing tasky with
    "0 free" on a placement that was perfectly valid.
    """
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "taskx")
    # taskx completes, but the planner says a later task of THIS run wants it.
    gr.release_gpus("taskx", handover=True)

    other_pid = os.getpid() + 1000
    monkeypatch.setattr(os, "getpid", lambda: other_pid)
    with pytest.raises(InsufficientGpusError):
        gr.try_reserve_gpus(1, "outsider")


def test_the_successor_can_still_claim_the_handed_over_device(monkeypatch):
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "taskx")
    gr.release_gpus("taskx", handover=True)

    assert _uuids(gr.try_reserve_gpus(1, "tasky")) == ["GPU-0000"]


def test_claiming_a_handed_over_device_transfers_ownership(monkeypatch):
    """The predecessor's record must not linger naming a device it no longer holds:
    it would keep that GPU blocked for every other process until the driver exits,
    which is exactly the over-holding the per-task model exists to avoid."""
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "taskx")
    gr.release_gpus("taskx", handover=True)
    gr.try_reserve_gpus(1, "tasky")

    names = sorted(p.name for p in gr.registry_dir().glob("*.json"))
    assert names == ["tasky.json"], f"stale predecessor record left behind: {names}"


def test_the_successor_prefers_its_predecessors_devices_over_lower_indexed_ones(
    monkeypatch,
):
    """The successor must land where the planner packed it, not on whatever has the
    lowest index at that instant.

    Regression for a real e2e flake: on a shared board, run A finishing frees its
    low-numbered GPUs, and a plain `free[:count]` then hands THOSE to run B's
    consumer instead of the pair run B's own server just handed back. Both are
    claimable, so nothing fails loudly -- the consumer simply runs somewhere the
    plan never said, and the assertion that it reuses its predecessor's devices
    fails on timing alone.
    """
    _patch_gpus(monkeypatch, _fake_gpus(4))
    # Something else holds the LOW pair; our server_a ends up on the HIGH pair.
    gr.try_reserve_gpus(2, "neighbour")  # GPU-0000/0001
    gr.try_reserve_gpus(2, "server_a")  # GPU-0002/0003
    gr.release_gpus("server_a", reusable=True)  # hand-back at READY
    gr.release_gpus("neighbour")  # neighbour finishes -> low pair goes free

    # Everything is claimable now, and plain index order would take the low pair.
    assert _uuids(gr.try_reserve_gpus(2, "consumer")) == ["GPU-0002", "GPU-0003"]


def test_a_hand_back_is_never_stranded_when_the_successor_could_use_it(monkeypatch):
    """The other half of the same bug: taking lower-indexed devices instead leaves
    the predecessor's reusable record naming GPUs nobody will ever supersede, so
    they stay blocked for every OTHER run while this run quietly uses different
    ones -- idle until the driver exits.

    Checked once server_a has exited: until then its record legitimately stands,
    because it is still serving on the devices it lent out.
    """
    _patch_gpus(monkeypatch, _fake_gpus(4))
    gr.try_reserve_gpus(2, "neighbour")  # GPU-0000/0001
    gr.try_reserve_gpus(2, "server_a")  # GPU-0002/0003
    gr.release_gpus("server_a", reusable=True)
    gr.release_gpus("neighbour")
    gr.try_reserve_gpus(2, "consumer")
    gr.release_gpus("server_a", handover=True)

    # Ownership transferred; no reusable leftovers keeping devices off the market.
    names = sorted(p.name for p in gr.registry_dir().glob("*.json"))
    assert names == ["consumer.json"], f"stranded hand-back record(s): {names}"


def test_a_live_servers_devices_stay_held_after_its_successor_finishes(monkeypatch):
    """A READY hand-back gives away the CLAIM, never the occupancy.

    Regression for a real e2e failure: server_a hands its devices back at READY
    and keeps serving on them, the consumer the planner packed there claims them,
    runs, and exits. With server_a's record dropped the moment the consumer
    superseded it, nothing named those devices any more -- so the consumer's
    ordinary hard release published GPUs a live server was still sitting on, and a
    concurrent run walked straight onto them.
    """
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "server_a")
    gr.release_gpus("server_a", reusable=True)  # READY: still serving
    gr.try_reserve_gpus(1, "consumer")
    gr.release_gpus("consumer")  # the consumer is done; server_a is not

    other_pid = os.getpid() + 1000
    monkeypatch.setattr(os, "getpid", lambda: other_pid)
    with pytest.raises(InsufficientGpusError):
        gr.try_reserve_gpus(1, "outsider")


def test_the_servers_own_exit_publishes_the_devices_its_successor_already_used(
    monkeypatch,
):
    """...and the holding ends when the server does, not when the run does."""
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "server_a")
    gr.release_gpus("server_a", reusable=True)
    gr.try_reserve_gpus(1, "consumer")
    gr.release_gpus("consumer")
    gr.release_gpus("server_a", handover=True)  # the server finally exits

    assert list(gr.registry_dir().glob("*.json")) == []


def test_a_server_exiting_before_its_successor_claims_still_holds_the_gap(monkeypatch):
    """A server can outlive its usefulness but not its successor's claim.

    When the exit lands first, the devices are not free yet -- the successor the
    planner packed onto them has not been submitted. The record stays as an
    ordinary hand-over: ours to take, nobody else's.
    """
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "server_a")
    gr.release_gpus("server_a", reusable=True)  # READY
    gr.release_gpus("server_a", handover=True)  # exits before the consumer runs

    other_pid = os.getpid() + 1000
    with monkeypatch.context() as outsider:
        outsider.setattr(os, "getpid", lambda: other_pid)
        with pytest.raises(InsufficientGpusError):
            gr.try_reserve_gpus(1, "outsider")

    assert _uuids(gr.try_reserve_gpus(1, "consumer")) == ["GPU-0000"]


def test_preference_still_falls_back_to_other_free_devices(monkeypatch):
    """A hand-back smaller than the request tops up from the ordinary free pool
    rather than failing -- the preference is an ordering, not a restriction."""
    _patch_gpus(monkeypatch, _fake_gpus(4))
    gr.try_reserve_gpus(2, "neighbour")  # GPU-0000/0001
    gr.try_reserve_gpus(1, "server_a")  # GPU-0002
    gr.release_gpus("server_a", reusable=True)
    gr.release_gpus("neighbour")

    # Wants 3: its own handed-back GPU-0002 first, then the rest by index.
    assert _uuids(gr.try_reserve_gpus(3, "consumer")) == [
        "GPU-0002",
        "GPU-0000",
        "GPU-0001",
    ]


def test_the_preference_never_reaches_another_runs_hand_back(monkeypatch):
    """Preference applies only to devices THIS process handed back. Another run's
    reusable record still reads as reserved, so the ordering can never be a way in."""
    _patch_gpus(monkeypatch, _fake_gpus(4))
    gr.try_reserve_gpus(2, "their_server")  # GPU-0000/0001
    gr.release_gpus("their_server", reusable=True)

    other_pid = os.getpid() + 1000
    monkeypatch.setattr(os, "getpid", lambda: other_pid)
    # Only GPU-0002/0003 are available to us; their hand-back is off limits.
    assert _uuids(gr.try_reserve_gpus(2, "ours")) == ["GPU-0002", "GPU-0003"]


def test_the_last_user_completing_frees_the_device_for_everyone(monkeypatch):
    """End of the chain: tasky is not flagged, so its completion is a hard release
    and the GPU goes back to the whole host."""
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "taskx")
    gr.release_gpus("taskx", handover=True)
    gr.try_reserve_gpus(1, "tasky")
    gr.release_gpus("tasky")  # not flagged -> hard release

    assert list(gr.registry_dir().glob("*.json")) == []
    other_pid = os.getpid() + 1000
    monkeypatch.setattr(os, "getpid", lambda: other_pid)
    assert _uuids(gr.try_reserve_gpus(1, "outsider")) == ["GPU-0000"]


def test_ownership_transfer_keeps_devices_the_successor_did_not_take(monkeypatch):
    """A 4-GPU predecessor handing to a 1-GPU successor still holds the other 3."""
    _patch_gpus(monkeypatch, _fake_gpus(4))
    gr.try_reserve_gpus(4, "big")
    gr.release_gpus("big", handover=True)
    gr.try_reserve_gpus(1, "small")

    record = json.loads((gr.registry_dir() / "big.json").read_text())
    assert sorted(record["gpu_uuids"]) == ["GPU-0001", "GPU-0002", "GPU-0003"]
    # ...and those three are still off-limits to anyone else.
    other_pid = os.getpid() + 1000
    monkeypatch.setattr(os, "getpid", lambda: other_pid)
    with pytest.raises(InsufficientGpusError):
        gr.try_reserve_gpus(1, "outsider")


def test_ownership_transfer_never_touches_another_run_s_records(monkeypatch):
    """Only the owning driver decides when to let go of its own hand-over."""
    _patch_gpus(monkeypatch, _fake_gpus(2))
    d = gr._ensure_registry_dir()
    (d / "theirs.json").write_text(
        json.dumps({
            "run_id": "theirs", "pid": 999999998,
            "hostname": gr.socket.gethostname(),
            "gpu_uuids": ["GPU-0000"], "state": gr._HANDOVER,
        })
    )
    monkeypatch.setattr(gr, "_pid_alive", lambda pid, start_ticks=None: True)
    gr.try_reserve_gpus(1, "mine")  # must take GPU-0001, not their handed-over one

    assert (d / "theirs.json").exists(), "swept another run's hand-over record"
    record = json.loads((d / "theirs.json").read_text())
    assert record["gpu_uuids"] == ["GPU-0000"]


def test_ownership_transfer_leaves_untaken_hand_over_records_alone(monkeypatch):
    """A hand-over covering devices the successor did NOT take must survive whole.

    Reached here with two hand-backs and a successor small enough to need only part
    of one: the record it drew from keeps its remaining device, and the record it
    never touched is left byte-for-byte alone. Dropping either would publish a GPU
    this run is still holding for a later task -- the very bug the hand-over exists
    to prevent.

    (Before hand-backs were preferred over other free devices this was reached a
    different way -- the successor packing onto a lower-indexed device than the one
    on offer. That path is gone by design: a successor now takes its predecessor's
    devices whenever it can. The behavior under test is unchanged.)
    """
    _patch_gpus(monkeypatch, _fake_gpus(3))
    gr.try_reserve_gpus(2, "server_a")  # GPU-0000/0001
    gr.try_reserve_gpus(1, "server_b")  # GPU-0002
    gr.release_gpus("server_a", handover=True)
    gr.release_gpus("server_b", handover=True)

    # Needs one device, so it takes the first handed-back one and no more.
    assert _uuids(gr.try_reserve_gpus(1, "successor")) == ["GPU-0000"]

    partly_taken = json.loads((gr.registry_dir() / "server_a.json").read_text())
    assert partly_taken["gpu_uuids"] == ["GPU-0001"], "untaken device was published"
    assert partly_taken["state"] == gr._HANDOVER

    untouched = json.loads((gr.registry_dir() / "server_b.json").read_text())
    assert untouched["gpu_uuids"] == ["GPU-0002"], "untaken hand-over was disturbed"
    assert untouched["state"] == gr._HANDOVER


def test_a_failed_claim_write_leaves_the_hand_over_standing(monkeypatch):
    """Ordering guard: the new claim is written BEFORE the old one is dropped.

    If the drop came first and the write then failed, the devices would be claimed
    by nobody while this run still intended to use them -- and a concurrent run
    could take them. Over-holding is the recoverable direction.
    """
    _patch_gpus(monkeypatch, _fake_gpus(1))
    gr.try_reserve_gpus(1, "predecessor")
    gr.release_gpus("predecessor", reusable=True)

    def _boom(*a, **k):
        raise OSError("registry full")

    monkeypatch.setattr(gr, "_write_record", _boom)
    with pytest.raises(OSError):
        gr.try_reserve_gpus(1, "successor")

    record = json.loads((gr.registry_dir() / "predecessor.json").read_text())
    assert record["gpu_uuids"] == ["GPU-0000"], (
        "the hand-over was dropped even though the new claim never landed"
    )
