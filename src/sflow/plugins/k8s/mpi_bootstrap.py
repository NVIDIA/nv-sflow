# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver-side helpers for multi-node MPI on Kubernetes (the ``k8s_mpi`` operator).

Two execution routes share these builders so one recipe -- a plain
``mpirun -np N ... <workload>`` with no SSH glue -- runs on either:

* **pods route** -- N plain pods (the existing ``k8s`` render). This module
  supplies the bash *bootstrap preamble* (ensure ``sshd``, install the
  driver-injected keypair, leader hostfile) and a *launch gate* inserted right
  before the recipe's ``mpirun`` line: the leader waits for the workers then
  launches, while a worker brings up ``sshd`` and idles. The recipe's setup runs on
  EVERY pod (worker parity: ``mkdir``/file writes land on every node); only the
  final ``mpirun`` is leader-only. A transparent ``mpirun`` wrapper injects the
  hostfile + SSH transport + ``-x`` env forwarding. The recipe never sees any of it.
* **operator route** -- one Kubeflow ``MPIJob`` CR. This module supplies the
  manifest builder that wraps a rendered pod spec into ``Launcher``/``Worker``
  templates; the mpi-operator owns the keypair/hostfile/``sshd``. The launcher
  installs the ``mpirun`` wrapper (for ``-x`` forwarding); the Worker runs the
  recipe's non-``mpirun`` setup (parity) then ``exec``s ``sshd`` -- a readiness
  probe on the ssh port makes ``WaitForWorkersReady`` gate on setup completion.

Env crosses to worker ranks only via ``mpirun -x`` (the ranks are ``orted`` children
of a fresh SSH session that does not inherit the pod/entrypoint env). The wrapper is
a PATH-shadowing executable (not a shell function) so the recipe's ``mpirun`` launch
resolves to it (the operator auto-``exec``s that launch -- see
``K8sMpiOperator._auto_exec_launch`` -- so the recipe writes a plain ``mpirun``). It
scans the environment at *invocation* time and forwards, in addition to the merged
prefix set, every var the recipe itself ``export``ed -- diffed against a pre-recipe
snapshot -- so a plain ``export FOO=bar`` reaches workers without any prefix config.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from sflow.plugins.k8s.render import (
    HOSTNAME_LABEL,
    SFLOW_ALLOC_LABEL,
    SFLOW_TASK_LABEL,
)

# --- Kubeflow MPI Operator (v2beta1) constants -------------------------------
MPIJOB_API_VERSION = "kubeflow.org/v2beta1"
MPIJOB_KIND = "MPIJob"
MPIJOB_CRD = "mpijobs.kubeflow.org"
# Labels the mpi-operator stamps on the pods it creates. Used to discover the
# launcher pod (rank 0 / the HTTP server) for log offload + exit mapping.
MPI_JOB_NAME_LABEL = "training.kubeflow.org/job-name"
MPI_JOB_ROLE_LABEL = "training.kubeflow.org/job-role"
MPI_LAUNCHER_ROLE = "launcher"

# The mpi-operator defaults workers to sshd on :2222 (avoids the node's :22 under
# hostNetwork); we default to the same so the operator's generated ssh config and
# the pods-route sshd agree without extra wiring.
DEFAULT_SSH_PORT = 2222
DEFAULT_SSH_AUTH_MOUNT_PATH = "/root/.ssh"
DEFAULT_LAUNCHER_CREATION_POLICY = "WaitForWorkersReady"
DEFAULT_MPI_IMPLEMENTATION = "OpenMPI"

# The /sflow ConfigMap mount is read-only, so the mpirun shim + hostfile live in a
# writable pod-local dir instead.
SFLOW_MPI_DIR = "/tmp/sflow-mpi"
SFLOW_MPI_BIN_DIR = f"{SFLOW_MPI_DIR}/bin"
SFLOW_MPI_HOSTFILE = f"{SFLOW_MPI_DIR}/hostfile"
SFLOW_MPI_WRAPPER_PATH = f"{SFLOW_MPI_BIN_DIR}/mpirun"
# Atomic once-per-pod lock (an `mkdir`) guarding the wrapper's role barrier so it runs
# exactly once -- the recipe's real mpirun -- and nested/re-entrant mpirun calls skip it.
SFLOW_MPI_BARRIER_LOCK = f"{SFLOW_MPI_DIR}/.barrier.lock"
# Exported env var names snapshotted just before the recipe runs, so the mpirun shim
# can ``-x``-forward whatever the recipe itself exported (diffed at launch) without
# the user having to enumerate ``forward_env_prefixes``. Path is overridable via the
# ``SFLOW_MPI_ENV_SNAPSHOT`` env (used by tests).
SFLOW_MPI_ENV_SNAPSHOT = f"{SFLOW_MPI_DIR}/env.snapshot"

# Ephemeral per-run keypair carried to every server pod via the task env Secret
# (base64). The bootstrap preamble decodes it into ``/root/.ssh`` -- no shared PVC
# and no in-pod keygen/exchange (pods route only; the operator route owns keys).
SSH_PRIVATE_KEY_ENV = "SFLOW_MPI_SSH_PRIVATE_KEY_B64"
SSH_PUBLIC_KEY_ENV = "SFLOW_MPI_SSH_PUBLIC_KEY_B64"

# Transport/system env sflow ALWAYS forwards to remote ranks (they are required or
# sflow-injected -- the gIB/RDMA preamble sets NCCL_NET/NCCL_CONF_FILE/... and the
# user does not know they exist). The user's ``forward_env_prefixes`` is merged on
# top (additive; never shrinks this set). See the plan's "Env forwarding" section.
_BUILTIN_FORWARD_PREFIXES: tuple[str, ...] = (
    "NCCL_",
    "UCX_",
    "GLOO_",
    "NVSHMEM_",
    "OMPI_MCA_",
    "SFLOW_",
)
# Whole-name vars (not prefixes) always forwarded so remote ranks resolve the
# driver libs + binaries the gIB hostPath mounts add to the launcher's env, plus
# OMP_NUM_THREADS (the k8s_mpi operator injects it as pod env; it must reach remote
# ranks too, or co-located ranks over-thread and crash at model load).
_BUILTIN_FORWARD_VARS: tuple[str, ...] = ("PATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS")


def merged_forward_prefixes(user_prefixes: Sequence[str] | None) -> list[str]:
    """Built-in transport/system prefixes UNION the user's additive list (deduped)."""
    out = list(_BUILTIN_FORWARD_PREFIXES)
    for raw in user_prefixes or []:
        p = str(raw).strip()
        if p and p not in out:
            out.append(p)
    return out


def _forward_grep_pattern(prefixes: Sequence[str]) -> str:
    """An ERE matching ``NAME`` of any env var whose name starts with a prefix."""
    alts = "|".join(re.escape(p) for p in prefixes)
    return f"^({alts})[A-Za-z0-9_]*"


def _cpu_bind_flags(cpu_bind: str, slots: int) -> str:
    """OpenMPI ``--map-by``/``--bind-to`` flags for a rank-binding mode ("" if off).

    ``core`` (default): bind each rank to ``PE`` cores, where ``PE`` is computed at
    launch into ``${_sflow_pe}`` by :func:`_cpu_bind_block` -- physical cores per rank,
    clamped to the pod's real cpuset and capped at ``cpu_bind_cores_per_rank``.
    ``numa``: one rank per NUMA domain, round-robin -- caps the LLVM/OpenMP
    ``hardware_concurrency`` each rank sees to a NUMA node's cores (only reduces it
    when the node is NPS>1; on NPS1 the whole node is one domain)."""
    if cpu_bind == "numa":
        return "--map-by numa --bind-to numa"
    if cpu_bind == "core":
        return f"--map-by ppr:{int(slots)}:node:PE=${{_sflow_pe}} --bind-to core"
    return ""


def _cpu_bind_block(cpu_bind: str, slots: int, cpu_bind_cores_per_rank: int = 0) -> str:
    """Wrapper bash that appends the binding flags to ``_sflow_opts`` -- unless the
    recipe already bound (its own ``--bind-to``/``--map-by`` in ``"$@"`` wins). Empty
    (no-op) when binding is off: mode ``none`` or a single-rank pod (``slots <= 1``,
    which never contends), so the launch path is byte-for-byte unchanged there.

    ``cpu_bind_cores_per_rank`` caps ``PE`` (``<= 0`` = uncapped): binding every core
    the pod owns is needlessly aggressive on a fat node and is what over-asks when the
    core count is misread, so take a modest slice per rank and leave the rest free."""
    s = int(slots)
    cap = int(cpu_bind_cores_per_rank)
    flags = _cpu_bind_flags(cpu_bind, s) if s > 1 else ""
    if not flags:
        return ""
    # `core` needs PE = cores/rank computed at launch, so it emits its own (guarded)
    # append; `numa` just appends the static flags. For `core`: count the pod's CPUs
    # with OMP_NUM_THREADS/OMP_THREAD_LIMIT UNSET -- GNU `nproc` otherwise honors them,
    # and sflow sets OMP_NUM_THREADS itself, so a plain `$(nproc)` returns that cap
    # (e.g. 8), not the real CPU count, and PE collapses. CLAMP to the installed CPUs
    # ($(nproc --all)) so a bogus/inflated count can't make PE huge and fail the launch.
    #
    # Then convert CPUs -> CORES: `PE=N` asks for N *cores* per rank, but `nproc` counts
    # logical CPUs (SMT threads). On an SMT-2 host, PE = nproc/slots asks for twice the
    # cores that exist and OpenMPI aborts "A request was made to bind to that would
    # result in binding more processes than cpus on a resource" (e.g. 224 threads / 8
    # ranks -> PE=28, but the node has only 112 cores). So divide by threads-per-core,
    # from `lscpu` and falling back to cpu0's sysfs SMT siblings. This is per-CPU-family,
    # not per-arch: x86 Intel Xeon and AMD EPYC are SMT-2 when hyperthreading/SMT is on,
    # Grace (Neoverse V2) is SMT-1, and Vera (Olympus) is SMT-2 again via spatial
    # multithreading -- so ARM cannot be assumed single-threaded. When neither source
    # answers we assume 1: PE then over-asks by the SMT factor, which the cap below is
    # what actually keeps safe. On a hybrid P/E-core CPU lscpu reports the max (2), so
    # cores are UNDER-counted and PE comes out small -- the safe direction.
    #
    # Finally CAP at `cpu_bind_cores_per_rank`: a rank needs only enough cores to keep
    # its host-side work fed, and a smaller slice is exactly what shrinks the LLVM/OpenMP
    # pools we are binding for. Uncapped, PE grows with the node and re-creates the
    # over-ask above whenever the measured core count is optimistic.
    #
    # If the cpuset has FEWER cores than ranks (`cores < slots`), `--bind-to core` can't
    # be satisfied (PE floors to 1 but slots*1 > cores -> OpenMPI aborts), so DEGRADE to
    # no binding instead of hard-failing a launch that worked before core became the
    # default. Floor PE at 1 otherwise.
    if cpu_bind == "core":
        bind_lines = [
            "  _sflow_ncpu=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc "
            "2>/dev/null || nproc 2>/dev/null || echo 1)",
            '  _sflow_nall=$(nproc --all 2>/dev/null || echo "${_sflow_ncpu}")',
            '  [ "${_sflow_ncpu}" -gt "${_sflow_nall}" ] 2>/dev/null && _sflow_ncpu=${_sflow_nall}',
            # threads-per-core: PE counts cores, nproc counts SMT threads.
            "  _sflow_smt=$(LC_ALL=C lscpu 2>/dev/null | "
            "awk -F: '/^Thread\\(s\\) per core/ {gsub(/[^0-9]/, \"\", $2); print $2; exit}')",
            '  case "${_sflow_smt}" in ""|*[!0-9]*) _sflow_smt=0 ;; esac',
            # lscpu missing (slim images): count cpu0's SMT siblings in sysfs, which is
            # either a list ("0,112" -- x86) or a range ("0-1").
            '  if [ "${_sflow_smt}" -lt 1 ] 2>/dev/null; then',
            "    _sflow_smt=$(awk -F, '{n=0; for (i=1; i<=NF; i++) "
            '{ if ($i ~ /-/) { split($i, r, "-"); n += r[2] - r[1] + 1 } else { n++ } } '
            "print n; exit}' /sys/devices/system/cpu/cpu0/topology/thread_siblings_list "
            "2>/dev/null)",
            '    case "${_sflow_smt}" in ""|*[!0-9]*) _sflow_smt=1 ;; esac',
            '    [ "${_sflow_smt}" -ge 1 ] 2>/dev/null || _sflow_smt=1',
            "  fi",
            '  _sflow_cores=$(( _sflow_ncpu / _sflow_smt )); '
            '[ "${_sflow_cores}" -ge 1 ] || _sflow_cores=1',
            f'  if [ "${{_sflow_cores}}" -lt {s} ] 2>/dev/null; then',
            f'    echo "sflow: pod cpuset has ${{_sflow_cores}} core(s) < {s} rank(s); '
            'skipping --bind-to core (would abort mpirun)" >&2',
            "  else",
            f"    _sflow_pe=$(( _sflow_cores / {s} )); "
            '[ "${_sflow_pe}" -ge 1 ] || _sflow_pe=1',
            *(
                [f'    [ "${{_sflow_pe}}" -gt {cap} ] && _sflow_pe={cap}']
                if cap > 0
                else []
            ),
            f'    echo "sflow: binding {s} rank(s) to ${{_sflow_pe}} core(s) each '
            '(${_sflow_ncpu} cpu(s) / ${_sflow_smt} thread(s) per core'
            + (f", cap {cap}" if cap > 0 else "")
            + ')" >&2',
            f"    _sflow_opts+=({flags})",
            "  fi",
        ]
    else:
        bind_lines = [f"  _sflow_opts+=({flags})"]
    return (
        "\n".join(
            [
                "# sflow CPU binding (multi-rank-per-pod): bind each rank within the",
                "# pod's cpuset so LLVM/OpenMP thread pools size to the binding, not the",
                "# whole node -- prevents pid/thread exhaustion when ranks share a pod.",
                "# Skipped if the recipe already binds (--bind-to/--map-by -> user wins).",
                "_sflow_bound=0",
                'for _a in "$@"; do',
                '  case "${_a}" in',
                "    --bind-to|--bind-to=*|--map-by|--map-by=*) _sflow_bound=1; break ;;",
                "  esac",
                "done",
                'if [ "${_sflow_bound}" = "0" ]; then',
                *bind_lines,
                "fi",
            ]
        )
        + "\n"
    )


def _role_barrier_block(*, ssh_port: int, ssh_dir: str, ensure_sshd: bool) -> str:
    """Wrapper bash (pods route, multi-node) that runs the MPI role barrier ONCE per
    pod -- guarded by an atomic ``mkdir`` lock, so it fires on the recipe's real
    ``mpirun`` and nested/re-entrant ``mpirun`` calls skip it. A worker brings up
    ``sshd`` and idles (``sleep infinity`` -- the wrapper never execs ``mpirun`` there);
    the leader brings up ``sshd`` then blocks until every worker's ``sshd`` (its setup
    done) before falling through to the launch. Reads ``SFLOW_MPI_N_NODES`` /
    ``SFLOW_MPI_NODE_INDEX`` / ``SFLOW_MPI_NODE_IPS`` exported by the bootstrap. Because
    this lives in the wrapper (invoked at the real ``mpirun`` call), it engages no
    matter how the recipe writes its launch line -- no source-line surgery needed."""
    inner: list[str] = []
    if ensure_sshd:
        inner += _ensure_sshd_lines()
    inner += _sshd_prepare_lines()
    inner += [
        f'"${{SFLOW_SSHD_BIN}}" -p {int(ssh_port)} '
        f'&& echo "[sflow-mpi] sshd up on :{int(ssh_port)}"',
        'if [ "${SFLOW_MPI_NODE_INDEX:-0}" != "0" ]; then',
        '  echo "[sflow-mpi] worker ${SFLOW_MPI_NODE_INDEX}: setup done; '
        'idle (leader drives mpirun)"',
        "  sleep infinity",
        "fi",
    ]
    inner += build_leader_wait_lines(ssh_port=ssh_port, ssh_dir=ssh_dir)
    lines = [
        "# sflow k8s_mpi role barrier (pods route, multi-node): a worker brings up sshd",
        "# + idles, the leader waits for every worker's sshd -- run ONCE per pod via an",
        "# atomic mkdir lock (nested/re-entrant mpirun skips it).",
        f'if [ "${{SFLOW_MPI_N_NODES:-1}}" -gt 1 ] '
        f'&& mkdir "{SFLOW_MPI_BARRIER_LOCK}" 2>/dev/null; then',
    ]
    lines += [f"  {ln}" for ln in inner]
    lines += ["fi"]
    return "\n".join(lines) + "\n"


def build_mpirun_wrapper_script(
    *,
    ssh_port: int = DEFAULT_SSH_PORT,
    key_path: str = f"{DEFAULT_SSH_AUTH_MOUNT_PATH}/id_rsa",
    hostfile_path: str = SFLOW_MPI_HOSTFILE,
    forward_prefixes: Sequence[str] | None = None,
    inject_hostfile: bool = True,
    cpu_bind: str = "none",
    cpu_bind_cores_per_rank: int = 0,
    slots: int = 1,
    role_barrier: bool = False,
    ssh_dir: str = DEFAULT_SSH_AUTH_MOUNT_PATH,
    ensure_sshd: bool = True,
) -> str:
    """Return the text of the transparent ``mpirun`` shim (a standalone script).

    The shim is placed first on ``PATH`` (so the auto-``exec``ed ``mpirun`` launch
    hits it), prepends the OpenMPI transport (``--hostfile`` + ``--mca plm_rsh_*``) when
    ``inject_hostfile`` and a hostfile exists, optionally prepends CPU-binding flags
    (``cpu_bind`` for a multi-rank pod, unless the recipe already binds), appends
    ``-x VAR`` for the forward set, then ``exec``s the real ``mpirun`` (found via
    ``SFLOW_REAL_MPIRUN``, captured by the preamble before it shadowed ``PATH``).
    ``inject_hostfile=False`` on the operator route, where the mpi-operator supplies
    the hostfile via ``OMPI_MCA_*``.

    ``cpu_bind`` (``numa``/``core``/``none``) binds each rank when ``slots > 1`` so a
    rank's LLVM/OpenMP thread pools size to the binding, not the whole node -- see
    :func:`_cpu_bind_block`, which also caps ``core``'s cores-per-rank at
    ``cpu_bind_cores_per_rank`` (``<= 0`` = uncapped). A single-rank pod
    (``slots <= 1``) or ``none`` is an exact no-op, keeping the launch path unchanged.

    ``role_barrier`` (pods route) embeds the per-role SSH barrier -- worker idles, leader
    waits for workers -- INTO the shim (see :func:`_role_barrier_block`), so it engages at
    the real ``mpirun`` call no matter how the recipe writes the launch line, instead of
    a fragile source-line gate. Off on the operator route (the mpi-operator owns it).

    The forward set (deduped) is the union of three sources, scanned at call time so
    late ``export``s in the recipe are captured: the always-on whole-name vars
    (``PATH``/``LD_LIBRARY_PATH``); names matching the merged prefix set; and -- via a
    diff against the pre-recipe snapshot (:data:`SFLOW_MPI_ENV_SNAPSHOT`, written by
    :func:`_env_snapshot_lines`) -- every var the recipe itself exported. The last
    source makes a plain ``export FOO=bar`` reach remote ranks without listing it in
    ``forward_env_prefixes``.
    """
    prefixes = merged_forward_prefixes(forward_prefixes)
    grep_pat = _forward_grep_pattern(prefixes)
    explicit_vars = " ".join(_BUILTIN_FORWARD_VARS)
    tmpl = r"""#!/usr/bin/env bash
# sflow transparent mpirun wrapper (k8s_mpi). Injects hostfile/SSH transport and
# -x env forwarding so the recipe keeps a plain `mpirun -np N ... <workload>`.
set -u
# Resolve the REAL mpirun. The preamble captured it in SFLOW_REAL_MPIRUN before
# prepending this shim's dir to PATH; fall back to scanning PATH for a non-self.
_sflow_real="${SFLOW_REAL_MPIRUN:-}"
if [ -z "${_sflow_real}" ]; then
  for _c in $(type -ap mpirun 2>/dev/null); do
    if [ "${_c}" != "__WRAPPER_PATH__" ]; then _sflow_real="${_c}"; break; fi
  done
fi
[ -n "${_sflow_real}" ] || _sflow_real="mpirun"

__ROLE_BARRIER__
_sflow_opts=()
# OpenMPI transport (pods route only): a bare `mpirun` finds neither the hostfile
# nor the alt SSH port, so inject both -- as ONE plm_rsh_args argument.
if [ "__INJECT_HOSTFILE__" = "1" ] && [ -f "__HOSTFILE__" ]; then
  _sflow_opts+=(--hostfile "__HOSTFILE__" \
    --mca plm_rsh_agent ssh \
    --mca plm_rsh_args "-p __SSH_PORT__ -i __KEY_PATH__ -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10")
fi
__BINDING_BLOCK__
# Forward env to remote ranks (sshd gives them a bare env), deduped, from three
# sources: the always-on whole-name vars, the merged prefix set (scanned NOW so late
# `export`s are captured), and every var the recipe itself exported -- the diff of
# the current exports against the pre-recipe snapshot -- so a plain `export FOO=bar`
# reaches workers without listing it in forward_env_prefixes.
_sflow_snap="${SFLOW_MPI_ENV_SNAPSHOT:-__ENV_SNAPSHOT__}"
_sflow_names="$(
  for _v in __EXPLICIT_VARS__; do
    eval "[ -n \"\${${_v}+x}\" ]" && printf '%s\n' "${_v}"
  done
  env | grep -oE '__GREP_PAT__'
  [ -f "${_sflow_snap}" ] && comm -13 "${_sflow_snap}" <(compgen -e | sort -u)
)"
for _v in $(printf '%s\n' "${_sflow_names}" | sort -u | sed '/^$/d'); do
  _sflow_opts+=(-x "${_v}")
done
# Surface the fully-resolved launch -- the real mpirun + every sflow-injected opt
# (hostfile, SSH transport, CPU binding, env forwarding) + the recipe's own args -- so
# the actual command that ran is visible in the task log (the shim otherwise exec's
# silently). Only the leader reaches this line (workers idle in the role barrier above).
echo "[sflow-mpi] launch: ${_sflow_real} ${_sflow_opts[*]} $*"
exec "${_sflow_real}" "${_sflow_opts[@]}" "$@"
"""
    barrier = (
        _role_barrier_block(ssh_port=ssh_port, ssh_dir=ssh_dir, ensure_sshd=ensure_sshd)
        if role_barrier
        else ""
    )
    return (
        tmpl.replace("__WRAPPER_PATH__", SFLOW_MPI_WRAPPER_PATH)
        .replace("__INJECT_HOSTFILE__", "1" if inject_hostfile else "0")
        .replace("__HOSTFILE__", hostfile_path)
        .replace("__SSH_PORT__", str(int(ssh_port)))
        .replace("__KEY_PATH__", key_path)
        .replace("__ROLE_BARRIER__\n", barrier)
        .replace(
            "__BINDING_BLOCK__\n",
            _cpu_bind_block(cpu_bind, slots, cpu_bind_cores_per_rank),
        )
        .replace("__EXPLICIT_VARS__", explicit_vars)
        .replace("__GREP_PAT__", grep_pat)
        .replace("__ENV_SNAPSHOT__", SFLOW_MPI_ENV_SNAPSHOT)
    )


def _install_wrapper_lines(wrapper_script: str) -> list[str]:
    """Bash lines that write the ``mpirun`` shim and put it first on ``PATH``.

    ``SFLOW_REAL_MPIRUN`` is captured BEFORE the shim's dir is prepended, so the
    shim can ``exec`` the genuine binary without recursing into itself.
    """
    return [
        "# --- sflow: transparent mpirun wrapper ---",
        f"mkdir -p {SFLOW_MPI_BIN_DIR}",
        f"cat > {SFLOW_MPI_WRAPPER_PATH} <<'SFLOW_MPIRUN_WRAPPER_EOF'",
        wrapper_script,
        "SFLOW_MPIRUN_WRAPPER_EOF",
        f"chmod +x {SFLOW_MPI_WRAPPER_PATH}",
        'export SFLOW_REAL_MPIRUN="${SFLOW_REAL_MPIRUN:-$(command -v mpirun 2>/dev/null || true)}"',
        f"export PATH={SFLOW_MPI_BIN_DIR}:${{PATH}}",
    ]


def _env_snapshot_lines() -> list[str]:
    """Bash lines that snapshot the exported env var names just before the recipe.

    Emitted as the last preamble line so the diff taken by the mpirun shim at launch
    (``compgen -e`` vs this snapshot) yields exactly the vars the recipe exported,
    which are then ``-x``-forwarded to remote ranks. The path is overridable via the
    ``SFLOW_MPI_ENV_SNAPSHOT`` env so tests can point it at a temp file.
    """
    snap = f'"${{SFLOW_MPI_ENV_SNAPSHOT:-{SFLOW_MPI_ENV_SNAPSHOT}}}"'
    return [
        "# --- sflow: snapshot exported env (pre-recipe) for mpirun -x auto-forward ---",
        f"mkdir -p {SFLOW_MPI_DIR} 2>/dev/null || true",
        f"compgen -e 2>/dev/null | sort -u > {snap} 2>/dev/null || true",
    ]


def _ensure_sshd_lines() -> list[str]:
    """Bash lines that install ``openssh-server`` if the image lacks ``sshd``.

    Mirrors the hand-rolled recipe: the TRT-LLM/Dynamo images ship ssh/mpirun but
    not sshd, and the apt cache is empty, so refresh first. No-op when sshd exists.
    """
    return [
        "if [ ! -x /usr/sbin/sshd ] && ! command -v sshd >/dev/null 2>&1; then",
        '  echo "[sflow-mpi] installing openssh-server"',
        "  export DEBIAN_FRONTEND=noninteractive",
        "  (apt-get update -qq && apt-get install -y -qq --no-install-recommends openssh-server) "
        '|| echo "[sflow-mpi] WARNING: could not install openssh-server"',
        "fi",
    ]


def _install_keypair_lines(ssh_dir: str) -> list[str]:
    """Bash lines that decode the driver-injected keypair into ``ssh_dir``.

    The private/public keys arrive base64 in the task env Secret (same for every
    pod), so all pods trust the same per-run key without a shared PVC or in-pod
    exchange. sshd rejects a group/world-writable privsep dir, so perms are forced.
    """
    return [
        f"mkdir -p /run/sshd {ssh_dir} && chmod 755 /run/sshd && chmod 700 {ssh_dir}",
        "ssh-keygen -A >/dev/null 2>&1 || true",
        f'if [ -n "${{{SSH_PRIVATE_KEY_ENV}:-}}" ]; then',
        f'  printf %s "${{{SSH_PRIVATE_KEY_ENV}}}" | base64 -d > {ssh_dir}/id_rsa',
        f'  printf %s "${{{SSH_PUBLIC_KEY_ENV}:-}}" | base64 -d > {ssh_dir}/id_rsa.pub',
        f"  cp {ssh_dir}/id_rsa.pub {ssh_dir}/authorized_keys",
        f"  chmod 600 {ssh_dir}/id_rsa {ssh_dir}/authorized_keys {ssh_dir}/id_rsa.pub",
        f"  printf 'Host *\\n  StrictHostKeyChecking no\\n  UserKnownHostsFile /dev/null\\n' > {ssh_dir}/config",
        f"  chmod 600 {ssh_dir}/config",
        "else",
        '  echo "[sflow-mpi] FATAL: no injected SSH keypair; cannot bootstrap MPI over SSH"',
        "  exit 1",
        "fi",
    ]


def _sshd_prepare_lines() -> list[str]:
    """Common lines to prepare + locate ``sshd`` (privsep dir, host keys, binary)."""
    return [
        "mkdir -p /run/sshd && chmod 755 /run/sshd",
        "ssh-keygen -A >/dev/null 2>&1 || true",
        'SFLOW_SSHD_BIN="$( [ -x /usr/sbin/sshd ] && echo /usr/sbin/sshd || command -v sshd || true )"',
        'if [ -z "${SFLOW_SSHD_BIN}" ]; then echo "[sflow-mpi] FATAL: sshd unavailable"; exit 1; fi',
    ]


def build_worker_idle_tail(
    *, ssh_port: int = DEFAULT_SSH_PORT, ensure_sshd: bool = True
) -> list[str]:
    """Bring up ``sshd`` in the FOREGROUND (``exec``) so the pod stays alive and the
    launcher / mpi-operator can ``ssh`` in to start ``orted``.

    Used as the tail of an operator-route Worker command, after the recipe setup.
    Assumes the keypair is already in place (the mpi-operator mounts it). Because it
    ``exec``s, nothing after it runs -- the worker never reaches the recipe's launch.
    """
    lines: list[str] = []
    if ensure_sshd:
        lines += _ensure_sshd_lines()
    lines += _sshd_prepare_lines()
    lines += [
        f'echo "[sflow-mpi] worker sshd on :{int(ssh_port)}"',
        f'exec "${{SFLOW_SSHD_BIN}}" -D -e -p {int(ssh_port)}',
    ]
    return lines


def build_leader_wait_lines(
    *, ssh_port: int = DEFAULT_SSH_PORT, ssh_dir: str = DEFAULT_SSH_AUTH_MOUNT_PATH
) -> list[str]:
    """Leader-side wait: block until every peer's ``sshd`` accepts key-auth.

    Because a worker brings up ``sshd`` only AFTER its recipe setup finished, this
    doubles as the "worker setup done" barrier -- the leader never launches a rank on
    a worker whose ``mkdir`` / file setup has not completed.
    """
    return [
        f'SFLOW_SSH_OPTS="-p {int(ssh_port)} -i {ssh_dir}/id_rsa -o StrictHostKeyChecking=no '
        '-o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o BatchMode=yes"',
        'for ip in $(printf \'%s\' "${SFLOW_MPI_NODE_IPS}" | tr \',\' \' \'); do',
        '  echo "[sflow-mpi] waiting for sshd on ${ip}:%d"' % int(ssh_port),
        "  for _ in $(seq 1 240); do",
        '    if ssh ${SFLOW_SSH_OPTS} "${ip}" true 2>/dev/null; then echo "[sflow-mpi] ${ip} ssh ok"; break; fi',
        "    sleep 2",
        "  done",
        "done",
    ]


def build_mpi_bootstrap_preamble(
    *,
    ssh_port: int = DEFAULT_SSH_PORT,
    gpus_per_node: int = 1,
    ensure_sshd: bool = True,
    ssh_dir: str = DEFAULT_SSH_AUTH_MOUNT_PATH,
    forward_prefixes: Sequence[str] | None = None,
    cpu_bind: str = "none",
    cpu_bind_cores_per_rank: int = 0,
) -> list[str]:
    """Bash preamble for the **pods route**, prepended to the task entrypoint.

    On every pod it installs the ``mpirun`` shim and (multi-node) ensures ``sshd`` is
    available and decodes the injected keypair; the leader also builds the hostfile
    from ``SFLOW_TASK_ASSIGNED_NODE_IPS``. It then snapshots the exported env (for the
    shim's ``-x`` auto-forward) and hands control to the recipe, which runs on EVERY
    pod. The per-role split -- worker ``sshd`` + idle vs leader wait-for-workers -- runs
    INSIDE the shim (:func:`_role_barrier_block`) at the real ``mpirun`` call, so the
    recipe setup runs on every node first (filesystem parity), the leader's wait is a
    real barrier (a worker's ``sshd`` comes up only once its setup finished), and it
    engages no matter how the recipe writes its launch line. Single-node tasks get only
    the shim + snapshot (a bare local ``mpirun``); the barrier self-skips (N_NODES=1).
    """
    wrapper = build_mpirun_wrapper_script(
        ssh_port=ssh_port,
        key_path=f"{ssh_dir}/id_rsa",
        forward_prefixes=forward_prefixes,
        inject_hostfile=True,
        cpu_bind=cpu_bind,
        cpu_bind_cores_per_rank=cpu_bind_cores_per_rank,
        # Ranks per node = the hostfile slots this route writes (gpus_per_node), so a
        # `core` PE = cores/gpus_per_node matches the actual rank density.
        slots=gpus_per_node,
        # The role barrier (worker idle / leader wait) lives in the shim so it engages
        # at the real mpirun call regardless of the recipe's launch-line syntax.
        role_barrier=True,
        ssh_dir=ssh_dir,
        ensure_sshd=ensure_sshd,
    )
    lines: list[str] = ["# ===== sflow k8s_mpi bootstrap (pods route) ====="]
    lines += _install_wrapper_lines(wrapper)
    # Node role + peer IPs, EXPORTED so the mpirun shim's role barrier (a subprocess of
    # the entrypoint) sees them. SFLOW_TASK_NODE_INDEX is only set for multi-node tasks.
    lines += [
        'export SFLOW_MPI_NODE_INDEX="${SFLOW_TASK_NODE_INDEX:-0}"',
        'export SFLOW_MPI_NODE_IPS="${SFLOW_TASK_ASSIGNED_NODE_IPS:-}"',
        "export SFLOW_MPI_N_NODES=$(printf '%s' \"${SFLOW_MPI_NODE_IPS}\" | tr ',' '\\n' | grep -c '[^[:space:]]' || true)",
        'if [ "${SFLOW_MPI_N_NODES:-0}" -gt 1 ]; then',
        '  echo "[sflow-mpi] node ${SFLOW_MPI_NODE_INDEX}/${SFLOW_MPI_N_NODES}: preparing SSH/MPI"',
    ]
    if ensure_sshd:
        lines += [f"  {ln}" for ln in _ensure_sshd_lines()]
    lines += [f"  {ln}" for ln in _install_keypair_lines(ssh_dir)]
    # Leader builds the hostfile now (before the recipe); sshd start + worker idle +
    # leader wait run later inside the mpirun shim (so the recipe runs on all pods first).
    lines += [
        '  if [ "${SFLOW_MPI_NODE_INDEX:-0}" = "0" ]; then',
        f"    mkdir -p {SFLOW_MPI_DIR}",
        "    printf '%s' \"${SFLOW_MPI_NODE_IPS}\" | tr ',' '\\n' | sed '/^[[:space:]]*$/d' "
        f"| awk '{{print $1\" slots={int(gpus_per_node)}\"}}' > {SFLOW_MPI_HOSTFILE}",
        f'    echo "[sflow-mpi] hostfile:"; cat {SFLOW_MPI_HOSTFILE}',
        "  fi",
        'fi',
        "# ===== end sflow k8s_mpi bootstrap =====",
    ]
    lines += _env_snapshot_lines()
    return lines


def build_launcher_preamble(
    *,
    forward_prefixes: Sequence[str] | None = None,
    cpu_bind: str = "none",
    cpu_bind_cores_per_rank: int = 0,
    slots: int = 1,
) -> list[str]:
    """Bash preamble for the **operator route** launcher: install the shim only.

    The mpi-operator supplies the hostfile (via ``OMPI_MCA_*``) and SSH transport,
    so the launcher shim adds only ``-x`` env forwarding (``inject_hostfile=0``) plus,
    for a multi-rank pod, the ``cpu_bind`` flags (the operator binds nothing itself).
    ``slots`` is the MPIJob ``slotsPerWorker`` (ranks per node).
    """
    wrapper = build_mpirun_wrapper_script(
        forward_prefixes=forward_prefixes,
        inject_hostfile=False,
        cpu_bind=cpu_bind,
        cpu_bind_cores_per_rank=cpu_bind_cores_per_rank,
        slots=slots,
    )
    lines = ["# ===== sflow k8s_mpi launcher (operator route) ====="]
    lines += _install_wrapper_lines(wrapper)
    lines += _env_snapshot_lines()
    lines += ["# ===== end sflow k8s_mpi launcher ====="]
    return lines


def build_worker_command(
    *,
    ssh_port: int = DEFAULT_SSH_PORT,
    ensure_sshd: bool = True,
    setup_lines: Sequence[str] | None = None,
) -> list[str]:
    """Container command for an operator-route ``Worker``.

    Runs the recipe's non-``mpirun`` setup lines first (``setup_lines``, for
    filesystem/side-effect parity with the launcher), then ``exec``s ``sshd`` in the
    foreground. The mpi-operator mounts the keypair at ``sshAuthMountPath`` and
    ``ssh``es in to start ``orted``; because ``sshd`` comes up only after the setup, a
    readiness probe on ``ssh_port`` makes the operator's ``WaitForWorkersReady`` gate
    on setup completion. With no ``setup_lines`` this is the original sshd-only worker.
    The setup runs the same way as on the launcher (no forced ``set -e``; the recipe
    controls its own error handling). Returns a ``["bash","-lc", "..."]`` command.
    """
    script_lines: list[str] = []
    if setup_lines:
        script_lines += list(setup_lines)
    else:
        script_lines.append("set -e")
    script_lines += build_worker_idle_tail(ssh_port=ssh_port, ensure_sshd=ensure_sshd)
    return ["bash", "-lc", "\n".join(script_lines)]


_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _is_mpirun_launch(stripped: str) -> bool:
    """True when the command's effective program is ``mpirun``.

    Matches a bare ``mpirun ...`` and an ``mpirun`` reached through the transparent
    prefixes recipes put in front of it: ``exec``, env assignments (``FOO=bar``), and
    shell variables (``${NSYS_CMD}``/``$NSYS`` -- typically an optional profiler that
    is empty at runtime). Every token before ``mpirun`` must be such a prefix, so a
    line that merely names ``mpirun`` as an argument (``echo ... mpirun``,
    ``python drive.py --launcher mpirun``) is NOT a launch and the user's process
    model is left untouched.
    """
    for tok in stripped.split():
        if tok == "mpirun":
            return True
        if tok == "exec" or tok.startswith("$") or _ENV_ASSIGN_RE.match(tok):
            continue  # transparent prefix; keep scanning for mpirun
        return False  # a real, non-mpirun program -> not our launch
    return False


def final_launch_index(script: Sequence[str]) -> int | None:
    """Index of the recipe's final ``mpirun`` launch line, or ``None``.

    The single source of truth for "where is the workload launch?", shared by the
    three transforms that key off it: :func:`strip_final_launch` (drop it),
    ``K8sMpiOperator._auto_exec_launch`` (prepend ``exec``), and
    ``K8sMpiOperator._insert_pods_launch_gate`` (insert the role gate before it).

    Walks backwards to the last real command (trailing blank/comment lines skipped)
    and returns its index only when that command is an ``mpirun`` launch -- see
    :func:`_is_mpirun_launch`, which accepts a bare ``mpirun``, an ``exec mpirun``, and
    an ``mpirun`` wrapped by transparent prefixes (``${NSYS_CMD} mpirun``,
    ``FOO=1 mpirun``). Any other final command (a user wrapper or a post-launch step)
    yields ``None`` so callers leave the user's own process model untouched.
    """
    lines = list(script)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        return i if _is_mpirun_launch(stripped) else None
    return None


def strip_final_launch(script: Sequence[str]) -> list[str]:
    """Return ``script`` without its final ``mpirun`` launch line.

    The operator-route Worker runs the recipe's setup (for filesystem parity) but not
    the launch itself. The launch is located by :func:`final_launch_index`; when the
    final command is not an ``mpirun`` launch the script is returned unchanged.
    """
    lines = list(script)
    idx = final_launch_index(lines)
    if idx is None:
        return lines
    return lines[:idx] + lines[idx + 1 :]


def _labels(task_label: str | None, allocation_id: str | None) -> dict[str, str]:
    labels: dict[str, str] = {}
    if task_label:
        labels[SFLOW_TASK_LABEL] = task_label
    if allocation_id:
        labels[SFLOW_ALLOC_LABEL] = allocation_id
    return labels


def inject_mpi_scheduling(
    pod_spec: dict[str, Any],
    *,
    job_name: str,
    include_hostnames: Sequence[str] = (),
    strip_hostname_pin: bool = False,
    one_pod_per_node: bool = True,
) -> dict[str, Any]:
    """Steer an MPIJob template's pods onto the reserved nodes (mutates ``pod_spec``).

    The mpi-operator -- not sflow -- creates the launcher/worker pods, so sflow
    layers scheduling onto the templates:

    * ``include_hostnames`` -> a ``kubernetes.io/hostname`` ``In`` nodeAffinity
      restricting the pod(s) to the reserved nodes (used for Workers, which span
      several nodes and so cannot use the single-hostname nodeSelector pin).
    * ``one_pod_per_node`` -> a podAntiAffinity on the operator's ``job-name``
      label (``topologyKey`` hostname) so launcher + workers spread one per node.
    * ``strip_hostname_pin`` -> drop the single-hostname nodeSelector that
      ``render_task_pod`` added (Workers), keeping any label-based node_selector.
      The Launcher keeps its pin (node 0) so the HTTP server is deterministic.
    """
    affinity = dict(pod_spec.get("affinity") or {})
    if include_hostnames:
        affinity["nodeAffinity"] = {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": HOSTNAME_LABEL,
                                "operator": "In",
                                "values": [str(h) for h in include_hostnames],
                            }
                        ]
                    }
                ]
            }
        }
    if one_pod_per_node:
        affinity["podAntiAffinity"] = {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {"matchLabels": {MPI_JOB_NAME_LABEL: job_name}},
                    "topologyKey": HOSTNAME_LABEL,
                }
            ]
        }
    if affinity:
        pod_spec["affinity"] = affinity
    if strip_hostname_pin:
        ns = pod_spec.get("nodeSelector")
        if isinstance(ns, dict) and HOSTNAME_LABEL in ns:
            ns = {k: v for k, v in ns.items() if k != HOSTNAME_LABEL}
            if ns:
                pod_spec["nodeSelector"] = ns
            else:
                pod_spec.pop("nodeSelector", None)
    return pod_spec


def build_mpijob_manifest(
    *,
    name: str,
    launcher_pod_spec: Mapping[str, Any],
    worker_pod_spec: Mapping[str, Any],
    worker_replicas: int,
    slots_per_worker: int,
    namespace: str | None = None,
    run_launcher_as_worker: bool = True,
    launcher_creation_policy: str = DEFAULT_LAUNCHER_CREATION_POLICY,
    mpi_implementation: str = DEFAULT_MPI_IMPLEMENTATION,
    ssh_auth_mount_path: str = DEFAULT_SSH_AUTH_MOUNT_PATH,
    clean_pod_policy: str = "Running",
    task_label: str | None = None,
    allocation_id: str | None = None,
) -> dict[str, Any]:
    """Render a Kubeflow ``MPIJob`` (``kubeflow.org/v2beta1``).

    ``launcher_pod_spec`` / ``worker_pod_spec`` are full pod ``spec`` dicts (from
    ``render_task_pod(...)["spec"]`` with the container command + affinity already
    set). ``mpi.*`` fields map 1:1 onto ``MPIJobSpec``. ``cleanPodPolicy: Running``
    keeps worker pods around while the launcher runs (a long-lived server), and
    the controller GCs all children via ownerRefs when the CR is deleted.
    """
    metadata: dict[str, Any] = {"name": name}
    if namespace:
        metadata["namespace"] = namespace
    labels = _labels(task_label, allocation_id)
    if labels:
        metadata["labels"] = labels
    spec: dict[str, Any] = {
        "slotsPerWorker": int(slots_per_worker),
        "runLauncherAsWorker": bool(run_launcher_as_worker),
        "launcherCreationPolicy": launcher_creation_policy,
        "sshAuthMountPath": ssh_auth_mount_path,
        "mpiImplementation": mpi_implementation,
        "runPolicy": {"cleanPodPolicy": clean_pod_policy},
        "mpiReplicaSpecs": {
            "Launcher": {
                "replicas": 1,
                "restartPolicy": "Never",
                "template": {
                    "metadata": {"labels": labels} if labels else {},
                    "spec": dict(launcher_pod_spec),
                },
            },
            "Worker": {
                "replicas": int(worker_replicas),
                "restartPolicy": "Never",
                "template": {
                    "metadata": {"labels": labels} if labels else {},
                    "spec": dict(worker_pod_spec),
                },
            },
        },
    }
    return {
        "apiVersion": MPIJOB_API_VERSION,
        "kind": MPIJOB_KIND,
        "metadata": metadata,
        "spec": spec,
    }
