# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runtime GPU<->RDMA NIC affinity preamble (``_k8s_rdma_preamble``).

String-shape checks cover the generated bash; ``bash -n`` validates syntax; and
functional tests exercise the critical TCP fallback path (the one that prevents a
dead HCA from breaking NIXL/UCX) deterministically via the ``SFLOW_*_DIR`` test
hooks -- no real cluster needed.
"""

import shutil
import subprocess

import pytest

from sflow.plugins.operators._k8s_rdma_preamble import (
    build_rdma_affinity_preamble,
    parse_rdma_runtime_status,
)


def _script(primary_iface="eth0", trailer=""):
    return "\n".join(build_rdma_affinity_preamble(primary_iface)) + trailer


# ---------------------------------------------------------------------------
# String shape
# ---------------------------------------------------------------------------


def test_returns_lines_and_bakes_in_iface():
    lines = build_rdma_affinity_preamble("enP5p9s0")
    assert isinstance(lines, list) and len(lines) > 10
    text = "\n".join(lines)
    assert "__SFLOW_PRIMARY_IFACE__" not in text  # token no longer injected
    assert "export UCX_NET_DEVICES" not in text  # sflow never sets UCX_NET_DEVICES
    assert text.rstrip().endswith("# --- end sflow RDMA affinity ---")


def test_covers_modes_verify_and_fallback():
    text = "\n".join(build_rdma_affinity_preamble("eth0"))
    # Mode selector with auto/explicit/off.
    assert "SFLOW_RDMA_AFFINITY" in text
    # Usability gate (the fix): rdmacm + uverbs + an ACTIVE port, else TCP.
    assert "rdma_cm" in text and "uverbs" in text and "ACTIVE" in text
    # auto (default): let NCCL/UCX auto-select; single closest rail when RDMA used.
    assert "UCX_MAX_RNDV_RAILS" in text
    assert "letting NCCL/UCX auto-select" in text
    # explicit: nvidia-smi -> PCIe-root topology match, pins NCCL_IB_HCA / UCX.
    assert "nvidia-smi" in text and "pci.bus_id" in text
    assert "export NCCL_IB_HCA=" in text and "_sflow_pcie_root" in text
    # Fallback disables NCCL IB so it uses sockets instead of a dead HCA.
    assert "NCCL_IB_DISABLE=1" in text


# ---------------------------------------------------------------------------
# Real bash: syntax + behavior
# ---------------------------------------------------------------------------

_BASH = shutil.which("bash")
_needs_bash = pytest.mark.skipif(_BASH is None, reason="bash not available")
_TRAILER = (
    '\necho "RESULT ucx=[${UCX_NET_DEVICES:-}] rails=[${UCX_MAX_RNDV_RAILS:-}]'
    ' ibhca=[${NCCL_IB_HCA:-}] dis=[${NCCL_IB_DISABLE:-}]"'
)


def _run_bash(fake_process, args, *, env=None, stdin=None):
    # The autouse fake_process fixture blocks unregistered subprocesses; allow the
    # real bash through so we can validate/execute the generated preamble.
    fake_process.allow_unregistered(True)
    return subprocess.run(args, capture_output=True, text=True, env=env, input=stdin)


def _usable_dirs(tmp_path, *, active=True, with_device=False):
    """A fake /dev/infiniband + /sys/class/infiniband exposing one mlx5 device."""
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "rdma_cm").write_text("")
    (dev / "uverbs0").write_text("")
    port = tmp_path / "ib" / "mlx5_0" / "ports" / "1"
    port.mkdir(parents=True)
    (port / "state").write_text("4: ACTIVE\n" if active else "1: DOWN\n")
    if with_device:
        (tmp_path / "ib" / "mlx5_0" / "device").write_text("")
    return {
        "SFLOW_RDMA_DEV_DIR": str(dev),
        "SFLOW_IB_SYS_DIR": str(tmp_path / "ib"),
        "PATH": "/usr/bin:/bin",
    }


@_needs_bash
def test_preamble_is_valid_bash(fake_process):
    # `bash -n` parses without executing: catches quoting/heredoc/syntax errors.
    proc = _run_bash(fake_process, [_BASH, "-n"], stdin=_script())
    assert proc.returncode == 0, proc.stderr


@_needs_bash
def test_opt_out_leaves_env_untouched(fake_process):
    proc = _run_bash(
        fake_process,
        [_BASH, "-c", _script("eth0", _TRAILER)],
        env={"SFLOW_RDMA_AFFINITY": "off", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "disabled via SFLOW_RDMA_AFFINITY" in proc.stdout
    assert "RESULT ucx=[] rails=[] ibhca=[] dis=[]" in proc.stdout  # nothing set


@_needs_bash
def test_auto_mode_lets_ucx_auto_select_when_usable(fake_process, tmp_path):
    # Default (auto) + usable RDMA -> NCCL_IB_HCA unset, UCX_NET_DEVICES unset.
    proc = _run_bash(
        fake_process,
        [_BASH, "-c", _script("enP5p9s0", _TRAILER)],
        env=_usable_dirs(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert "letting NCCL/UCX auto-select" in proc.stdout
    assert "RESULT ucx=[] rails=[1] ibhca=[] dis=[]" in proc.stdout


@_needs_bash
def test_auto_mode_preserves_user_ucx_net_devices(fake_process, tmp_path):
    # A user/recipe-provided UCX_NET_DEVICES is never overwritten by sflow.
    env = _usable_dirs(tmp_path)
    env["UCX_NET_DEVICES"] = "mlx5_2:1"
    proc = _run_bash(
        fake_process, [_BASH, "-c", _script("enP5p9s0", _TRAILER)], env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "RESULT ucx=[mlx5_2:1] rails=[1]" in proc.stdout


@_needs_bash
def test_explicit_mode_pins_active_hcas(fake_process, tmp_path):
    # explicit + usable RDMA, no nvidia-smi (PATH restricted) -> pin the ACTIVE HCA.
    env = _usable_dirs(tmp_path, with_device=True)
    env["SFLOW_RDMA_AFFINITY"] = "explicit"
    proc = _run_bash(
        fake_process, [_BASH, "-c", _script("enP5p9s0", _TRAILER)], env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "RESULT ucx=[] rails=[] ibhca=[mlx5_0] dis=[]" in proc.stdout


@_needs_bash
def test_falls_back_to_tcp_when_rdmacm_missing(fake_process, tmp_path):
    # Empty device dir -> no rdma_cm -> NCCL_IB_DISABLE=1; UCX left unset.
    proc = _run_bash(
        fake_process,
        [_BASH, "-c", _script("enP5p9s0", _TRAILER)],
        env={
            "SFLOW_RDMA_DEV_DIR": str(tmp_path / "nope"),
            "SFLOW_IB_SYS_DIR": str(tmp_path / "nope"),
            "PATH": "/usr/bin:/bin",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "ucx=[]" in proc.stdout and "dis=[1]" in proc.stdout
    assert "using TCP" in proc.stdout


@_needs_bash
def test_falls_back_to_tcp_when_no_active_port(fake_process, tmp_path):
    # rdmacm + uverbs present but no IB device has an ACTIVE port -> still TCP.
    env = _usable_dirs(tmp_path, active=False)
    proc = _run_bash(
        fake_process, [_BASH, "-c", _script("enP5p9s0", _TRAILER)], env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "ucx=[]" in proc.stdout and "dis=[1]" in proc.stdout


@_needs_bash
def test_usable_rdma_reports_missing_gpudirect_driver(fake_process, tmp_path):
    env = _usable_dirs(tmp_path)
    env["SFLOW_GPU_DIRECT_SYS_MODULE_DIR"] = str(tmp_path / "modules")
    proc = _run_bash(
        fake_process, [_BASH, "-c", _script("enP5p9s0")], env=env
    )
    assert proc.returncode == 0, proc.stderr
    status = parse_rdma_runtime_status(proc.stdout)
    assert status is not None
    assert status.gpudirect_rdma_unavailable is True
    assert "nvidia_peermem" in status.gpudirect_rdma_reason


@_needs_bash
def test_usable_rdma_reports_gpudirect_driver_ready(fake_process, tmp_path):
    env = _usable_dirs(tmp_path)
    module_dir = tmp_path / "modules"
    (module_dir / "nvidia_peermem").mkdir(parents=True)
    env["SFLOW_GPU_DIRECT_SYS_MODULE_DIR"] = str(module_dir)
    proc = _run_bash(
        fake_process, [_BASH, "-c", _script("enP5p9s0")], env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "GPUDirect RDMA peer-memory driver visible" in proc.stdout
    status = parse_rdma_runtime_status(proc.stdout)
    assert status is not None
    assert status.gpudirect_rdma_unavailable is False


@_needs_bash
def test_tcp_fallback_states_all_ports_down_reason(fake_process, tmp_path):
    # The fallback marker must name the specific cause so users can triage; the
    # parser must then recover that reason from the emitted line (round-trip).
    env = _usable_dirs(tmp_path, active=False)
    proc = _run_bash(
        fake_process, [_BASH, "-c", _script("enP5p9s0")], env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "WARNING" in proc.stdout and "all ports DOWN" in proc.stdout
    status = parse_rdma_runtime_status(proc.stdout)
    assert status is not None and status.degraded_to_tcp
    assert "all ports DOWN" in status.reason


@_needs_bash
def test_tcp_fallback_states_missing_rdma_cm_reason(fake_process, tmp_path):
    proc = _run_bash(
        fake_process,
        [_BASH, "-c", _script("enP5p9s0")],
        env={
            "SFLOW_RDMA_DEV_DIR": str(tmp_path / "nope"),
            "SFLOW_IB_SYS_DIR": str(tmp_path / "nope"),
            "PATH": "/usr/bin:/bin",
        },
    )
    assert proc.returncode == 0, proc.stderr
    status = parse_rdma_runtime_status(proc.stdout)
    assert status is not None and status.degraded_to_tcp
    assert "rdma_cm" in status.reason


# ---------------------------------------------------------------------------
# Runtime status parser (used by sflow to surface the in-pod TCP fallback)
# ---------------------------------------------------------------------------


def test_parse_detects_tcp_fallback_with_reason():
    log = (
        "[pod/decode-server-0/decode-server-0] [sflow-rdma] WARNING rdma requested "
        "but unusable: no InfiniBand port is ACTIVE (all ports DOWN): using TCP "
        "for NCCL (NCCL_IB_DISABLE=1); UCX device selection left to the library\n"
    )
    status = parse_rdma_runtime_status(log)
    assert status is not None
    assert status.degraded_to_tcp is True
    assert "ACTIVE" in status.reason and "DOWN" in status.reason
    assert status.pods_degraded == 1
    assert status.pods_total == 1


def test_parse_reason_keeps_internal_colons():
    log = "[sflow-rdma] a: b: c: using TCP (UCX_NET_DEVICES=eth0)\n"
    status = parse_rdma_runtime_status(log)
    assert status is not None and status.reason == "a: b: c"


def test_parse_rdma_active_is_not_degraded():
    log = (
        "[pod/x/x] [sflow-rdma] auto: letting NCCL/UCX auto-select devices "
        "(UCX_MAX_RNDV_RAILS=1)\n"
    )
    status = parse_rdma_runtime_status(log)
    assert status is not None
    assert status.degraded_to_tcp is False
    assert status.pods_degraded == 0 and status.pods_total == 1


def test_parse_explicit_pin_is_not_degraded():
    log = "[sflow-rdma] explicit: NCCL_IB_HCA=mlx5_0 (UCX device selection left to the library)\n"
    status = parse_rdma_runtime_status(log)
    assert status is not None and status.degraded_to_tcp is False


def test_parse_returns_none_without_marker():
    assert parse_rdma_runtime_status("some unrelated\nvllm log lines\n") is None


def test_parse_disabled_is_not_degraded():
    log = "[sflow-rdma] disabled via SFLOW_RDMA_AFFINITY\n"
    status = parse_rdma_runtime_status(log)
    assert status is not None and status.degraded_to_tcp is False


def test_parse_counts_multiple_pods_mixed():
    log = (
        "[pod/a/a] [sflow-rdma] rc: using TCP for NCCL (NCCL_IB_DISABLE=1)\n"
        "[pod/b/b] [sflow-rdma] auto: letting NCCL/UCX auto-select devices\n"
        "[pod/c/c] [sflow-rdma] rc2: using TCP for NCCL (NCCL_IB_DISABLE=1)\n"
    )
    status = parse_rdma_runtime_status(log)
    assert status is not None
    assert status.degraded_to_tcp is True
    assert status.pods_degraded == 2
    assert status.pods_total == 3


def test_parse_detects_ucx_intra_node_tcp_transport():
    log = (
        "[pod/decode/decode] [1783051354.882948] [node:646:0] "
        "ucp_worker.c:1912 UCX  INFO    ucp_context_0 intra-node cfg#1 "
        "rma_am(tcp/enP5p9s0) amo_am(tcp/enP5p9s0) "
        "am(tcp/enP5p9s0 cma/memory) ka(tcp/enP5p9s0)\n"
    )
    status = parse_rdma_runtime_status(log)
    assert status is not None
    assert status.ucx_intra_node_tcp is True
    assert "tcp/enP5p9s0" in status.ucx_transport


def test_parse_detects_gpudirect_rdma_warning():
    log = (
        "[pod/decode/decode] [sflow-rdma] WARNING GPUDirect RDMA unavailable: "
        "nvidia_peermem/nv_peer_mem kernel module not visible in pod\n"
    )
    status = parse_rdma_runtime_status(log)
    assert status is not None
    assert status.gpudirect_rdma_unavailable is True
    assert "kernel module" in status.gpudirect_rdma_reason


def test_parse_ignores_ucx_intra_node_cuda_ipc_transport():
    log = (
        "[pod/decode/decode] ucp_worker.c:1912 UCX  INFO    "
        "ucp_context_0 intra-node cfg#1 rma_am(cuda_ipc/cuda0) "
        "am(cuda_ipc/cuda0 tcp/enP5p9s0)\n"
    )
    status = parse_rdma_runtime_status(log)
    assert status is None


def test_parse_ignores_intra_node_tcp_when_cuda_ipc_in_a_later_config():
    # UCX's startup config (cfg#1) is tcp-only, but once a peer registers (cfg#2)
    # cuda_ipc is available and carries the KV transfer over NVLink -- the init
    # tables label rma_am "tcp" in BOTH the cuda_ipc-working and TCP-only cases, so
    # firing on the transient cfg#1 is a false positive. cuda_ipc appearing anywhere
    # intra-node => NVLink available => do not warn.
    log = (
        "[pod/d/d] ucp_worker.c:1912 UCX  INFO    ucp_context_0 intra-node cfg#1 "
        "rma_am(tcp/enP5p9s0) amo_am(tcp/enP5p9s0) am(tcp/enP5p9s0 cma/memory) "
        "ka(tcp/enP5p9s0)\n"
        "[pod/d/d] ucp_worker.c:1912 UCX  INFO    ucp_context_0 intra-node cfg#2 "
        "rma_am(tcp/enP5p9s0) amo_am(tcp/enP5p9s0) "
        "am(tcp/enP5p9s0 cma/memory cuda_ipc/cuda) ka(tcp/enP5p9s0)\n"
    )
    status = parse_rdma_runtime_status(log)
    assert status is None


def test_parse_flags_intra_node_tcp_when_cuda_ipc_never_appears():
    # No cuda_ipc anywhere (e.g. cross-pod GPU isolation) -> NVLink genuinely
    # unavailable -> the warning should still fire, across multiple configs.
    log = (
        "[pod/d/d] ucp_worker.c:1912 UCX  INFO    ucp_context_0 intra-node cfg#1 "
        "rma_am(tcp/enP5p9s0) am(tcp/enP5p9s0 cma/memory) ka(tcp/enP5p9s0)\n"
        "[pod/d/d] ucp_worker.c:1912 UCX  INFO    ucp_context_0 intra-node cfg#2 "
        "rma_am(tcp/enP5p9s0) am(tcp/enP5p9s0 cma/memory) ka(tcp/enP5p9s0)\n"
    )
    status = parse_rdma_runtime_status(log)
    assert status is not None
    assert status.ucx_intra_node_tcp is True
    assert "tcp/enP5p9s0" in status.ucx_transport
