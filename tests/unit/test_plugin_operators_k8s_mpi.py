# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified ``k8s_mpi`` operator: route resolution, the pods-route
MPI bootstrap preamble + transparent ``mpirun`` wrapper, the operator-route MPIJob
CR render (Launcher/Worker templates, scheduling), env forwarding, and the
backend's ``mpijobs.kubeflow.org`` RBAC gating."""

import asyncio
import json
import os
import shutil
import subprocess

import pytest

from sflow.core.backend import Allocation
from sflow.core.compute_node import ComputeNode
from sflow.core.operator_registry import (
    ensure_builtin_operators_registered,
    get_operator_registry,
    operator_config_type_adapter,
)
from sflow.plugins.backends.kubernetes import KubernetesBackend, KubernetesBackendConfig
from sflow.plugins.k8s import mpi_bootstrap as mpi_boot
from sflow.plugins.k8s.capabilities import CapabilityState
from sflow.plugins.operators.k8s_mpi import (
    K8sMpiOperator,
    K8sMpiOperatorConfig,
    MpiConfig,
)

_MARK = "SFLOW_K8S_MANIFEST"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _backend(scheduling="device_plugin", gpus_per_node=8, nodes=3, namespace="ns",
             cpu_per_gpu=None, cpu_request=None):
    backend = KubernetesBackend(
        KubernetesBackendConfig(
            name="k8s",
            type="kubernetes",
            namespace=namespace,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            scheduling=scheduling,
            host_network=True,
            # CPU policy is opt-in and UNSET by default in the product, so the fixture
            # mirrors that. The MPI CPU-request test passes cpu_per_gpu explicitly.
            cpu_per_gpu=cpu_per_gpu,
            cpu_request=cpu_request,
        )
    )
    backend.allocation = Allocation(
        allocation_id="abc",
        nodes=[
            ComputeNode(name=f"node-{i}", ip_address=f"10.0.0.{i + 1}", index=i,
                        num_gpus=gpus_per_node)
            for i in range(nodes)
        ],
        owned=True,
    )
    backend._node_to_resv_pod = {f"node-{i}": f"res-{i}" for i in range(nodes)}
    return backend


def _op(backend, assigned_nodes, *, gpu_count, route="auto", has_mpi_operator=None,
        mpi_extra=None, monkeypatch=None):
    mpi = {"route": route, **(mpi_extra or {})}
    op = K8sMpiOperator(
        K8sMpiOperatorConfig(name="mpi_server", image="img:1", run_as_root=True, mpi=mpi)
    )
    op.apply_backend_context(
        backend=backend, assigned_nodes=list(assigned_nodes), artifacts=[],
        gpu_count=gpu_count,
    )
    # Map the legacy bool hint onto the capability state: True -> USABLE (installed +
    # permitted), False -> ABSENT, None -> leave as detected (UNKNOWN in these tests).
    if has_mpi_operator is True:
        op._mpi_state = CapabilityState.USABLE
    elif has_mpi_operator is False:
        op._mpi_state = CapabilityState.ABSENT
    if monkeypatch is not None:
        monkeypatch.setattr(
            "sflow.plugins.operators.k8s_mpi._generate_ssh_keypair_b64",
            lambda: ("PRIVB64", "PUBB64"),
        )
    return op


def _manifest_from_plan(plan):
    shell = plan.apply_command.as_list()[-1]
    after = shell.split(_MARK, 1)[1]
    body = after.split("\n" + _MARK, 1)[0]
    return json.loads(body.split("\n", 1)[1]), shell


def _entrypoint(manifest):
    for item in manifest["items"]:
        if item["kind"] == "ConfigMap" and "entrypoint.sh" in item.get("data", {}):
            return item["data"]["entrypoint.sh"]
    return ""


# ---------------------------------------------------------------------------
# config + registration
# ---------------------------------------------------------------------------
def test_operator_is_registered():
    ensure_builtin_operators_registered()
    assert "k8s_mpi" in get_operator_registry()


def test_config_defaults():
    c = MpiConfig()
    assert c.route == "auto"
    assert c.run_launcher_as_worker is True
    assert c.launcher_creation_policy == "WaitForWorkersReady"
    assert c.ssh_port == 2222
    assert c.mpi_implementation == "OpenMPI"
    assert c.ensure_sshd is True
    assert c.cpu_bind == "core"
    assert c.cpu_bind_cores_per_rank == 8
    assert c.forward_env_prefixes == []
    # Setup budget defaults to ~15 min (900s -> failureThreshold 180 at a fixed 5s poll).
    assert c.worker_setup_timeout_seconds == 900
    assert c.launcher_discovery_timeout == 600


def test_config_validates_via_discriminated_union_with_raw_expression():
    # Load-time validation runs on the RAW (unresolved) recipe, so slots_per_worker
    # must accept a ${{ }} expression string.
    adapter = operator_config_type_adapter()
    v = adapter.validate_python(
        {
            "name": "s",
            "type": "k8s_mpi",
            "image": "r/i:1",
            "mpi": {"route": "operator", "slots_per_worker": "${{ variables.GPN }}",
                    "forward_env_prefixes": ["TRTLLM_"]},
        }
    )
    assert v.type == "k8s_mpi"
    assert v.mpi.slots_per_worker == "${{ variables.GPN }}"
    assert v.mpi.forward_env_prefixes == ["TRTLLM_"]


def test_config_rejects_unknown_mpi_key():
    with pytest.raises(Exception):
        K8sMpiOperatorConfig(name="s", image="i:1", mpi={"bogus": 1})


# ---------------------------------------------------------------------------
# route resolution
# ---------------------------------------------------------------------------
def test_route_pods_forced():
    op = K8sMpiOperator(K8sMpiOperatorConfig(name="s", image="i:1", mpi={"route": "pods"}))
    op._mpi_state = CapabilityState.USABLE  # ignored when forced
    assert op._resolved_route() == "pods"


def test_route_auto_falls_back_to_pods_when_not_usable():
    op = K8sMpiOperator(K8sMpiOperatorConfig(name="s", image="i:1", mpi={"route": "auto"}))
    op._mpi_state = CapabilityState.UNKNOWN  # dry-run / not detected
    assert op._resolved_route() == "pods"
    op._mpi_state = CapabilityState.ABSENT  # not installed
    assert op._resolved_route() == "pods"


def test_route_auto_uses_operator_when_usable():
    op = K8sMpiOperator(K8sMpiOperatorConfig(name="s", image="i:1", mpi={"route": "auto"}))
    op._mpi_state = CapabilityState.USABLE
    assert op._resolved_route() == "operator"


def test_route_auto_falls_back_when_installed_but_not_usable():
    # The operator is installed cluster-wide, but the current creds cannot drive
    # MPIJobs in this namespace (INSTALLED, not USABLE) -> pods (no hard failure).
    op = K8sMpiOperator(K8sMpiOperatorConfig(name="s", image="i:1", mpi={"route": "auto"}))
    op._mpi_state = CapabilityState.INSTALLED
    assert op._resolved_route() == "pods"


def test_route_operator_errors_when_absent():
    op = K8sMpiOperator(
        K8sMpiOperatorConfig(name="s", image="i:1", mpi={"route": "operator"})
    )
    op._mpi_state = CapabilityState.ABSENT
    with pytest.raises(ValueError, match="mpijobs.kubeflow.org"):
        op._resolved_route()


def test_route_operator_proceeds_when_installed_or_unknown():
    # route:operator proceeds on INSTALLED (its RBAC is hard-checked in preflight,
    # not here) and on UNKNOWN (dry-run: detection has not run) -- only ABSENT errors.
    op = K8sMpiOperator(
        K8sMpiOperatorConfig(name="s", image="i:1", mpi={"route": "operator"})
    )
    op._mpi_state = CapabilityState.INSTALLED
    assert op._resolved_route() == "operator"
    op._mpi_state = CapabilityState.UNKNOWN
    assert op._resolved_route() == "operator"


# ---------------------------------------------------------------------------
# env forwarding (built-in UNION additive)
# ---------------------------------------------------------------------------
def test_merged_forward_prefixes_are_builtin_union_user():
    merged = mpi_boot.merged_forward_prefixes(["TRTLLM_", "NCCL_"])  # NCCL_ is builtin
    assert merged[: len(mpi_boot._BUILTIN_FORWARD_PREFIXES)] == list(
        mpi_boot._BUILTIN_FORWARD_PREFIXES
    )
    assert "TRTLLM_" in merged
    assert merged.count("NCCL_") == 1  # dedup


def test_mpirun_wrapper_forwards_and_execs_real_binary():
    script = mpi_boot.build_mpirun_wrapper_script(
        ssh_port=2222, forward_prefixes=["TRTLLM_"], inject_hostfile=True
    )
    # -x forwarding scans the env for the merged prefixes + explicit vars.
    assert "NCCL_" in script and "TRTLLM_" in script
    assert "PATH LD_LIBRARY_PATH" in script
    # exec the REAL mpirun (captured by the preamble), never itself (recursion).
    assert 'exec "${_sflow_real}"' in script
    assert "--hostfile" in script and "plm_rsh_args" in script


def test_mpirun_wrapper_echoes_resolved_launch_line():
    # The shim echoes the fully-resolved launch (real mpirun + sflow-injected opts +
    # recipe args) so the actual command that ran is visible in the task log, THEN exec's
    # it -- otherwise the injected hostfile/-x/binding are hidden from the user.
    script = mpi_boot.build_mpirun_wrapper_script(cpu_bind="core", slots=4)
    assert 'echo "[sflow-mpi] launch: ${_sflow_real} ${_sflow_opts[*]} $*"' in script
    assert script.index("[sflow-mpi] launch:") < script.index('exec "${_sflow_real}"')


def test_mpirun_wrapper_launch_echo_goes_to_stderr(tmp_path, fake_process):
    """The `[sflow-mpi] launch:` diagnostic must go to STDERR, not stdout, so it never
    pollutes a result-parsed launcher's stdout (or a tee'd rank capture). Task logs
    capture stderr too, so it stays visible."""
    fake_process.allow_unregistered(True)
    real = tmp_path / "real_mpirun"
    real.write_text('#!/usr/bin/env bash\nprintf "WORKLOAD-STDOUT\\n"\n')
    real.chmod(0o755)
    wrapper = tmp_path / "mpirun"
    wrapper.write_text(mpi_boot.build_mpirun_wrapper_script(inject_hostfile=False))
    wrapper.chmod(0o755)
    out = subprocess.run(
        [str(wrapper), "-np", "1", "app"],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "SFLOW_REAL_MPIRUN": str(real)},
    )
    assert out.returncode == 0, out.stderr
    assert "[sflow-mpi] launch:" in out.stderr  # diagnostic on stderr
    assert "[sflow-mpi] launch:" not in out.stdout  # stdout stays clean for result parsing
    assert "WORKLOAD-STDOUT" in out.stdout


def test_mpirun_wrapper_operator_route_omits_hostfile():
    # The operator route gets the hostfile from the mpi-operator (OMPI_MCA_*), so
    # the launcher wrapper only forwards env -- it must not inject --hostfile.
    script = mpi_boot.build_mpirun_wrapper_script(inject_hostfile=False)
    assert '"__INJECT_HOSTFILE__"' not in script  # token substituted
    assert '[ "0" = "1" ]' in script  # inject gate is off


def test_mpirun_wrapper_forwards_recipe_exports_via_snapshot():
    # The shim also -x-forwards whatever the recipe exported (diff vs the pre-recipe
    # snapshot), so a plain `export FOO=bar` reaches workers without forward_env_prefixes.
    script = mpi_boot.build_mpirun_wrapper_script(forward_prefixes=["TRTLLM_"])
    assert "comm -13" in script and "compgen -e" in script
    assert mpi_boot.SFLOW_MPI_ENV_SNAPSHOT in script
    # regression: the explicit + prefix forwarding is preserved.
    assert "PATH LD_LIBRARY_PATH" in script
    assert "NCCL_" in script and "TRTLLM_" in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_mpirun_wrapper_snapshot_diff_forwards_recipe_export_end_to_end(
    tmp_path, fake_process
):
    """Executed end to end: a var exported AFTER the snapshot is `-x`-forwarded; a var
    already present at snapshot time (non-prefixed) is not; passthrough args survive."""
    # The autouse fake_process fixture blocks real subprocesses; run real bash here.
    fake_process.allow_unregistered(True)
    real = tmp_path / "real_mpirun"
    real.write_text('#!/usr/bin/env bash\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    real.chmod(0o755)
    wrapper = tmp_path / "mpirun"
    wrapper.write_text(mpi_boot.build_mpirun_wrapper_script(inject_hostfile=False))
    wrapper.chmod(0o755)
    snap = tmp_path / "env.snapshot"

    base_env = {
        "PATH": os.environ["PATH"],
        "SFLOW_MPI_ENV_SNAPSHOT": str(snap),
        "SFLOW_REAL_MPIRUN": str(real),
        "BASELINE_VAR": "1",  # present at snapshot time -> must NOT be forwarded
    }
    # Snapshot the exports BEFORE the recipe exports FOO.
    subprocess.run(
        f"compgen -e | sort -u > {snap}",
        shell=True, executable="/bin/bash", env=base_env, check=True,
    )
    run_env = {**base_env, "FOO": "bar"}  # the recipe's own export (post-snapshot)
    out = subprocess.run(
        [str(wrapper), "-np", "2", "hostname"],
        capture_output=True, text=True, env=run_env,
    )
    assert out.returncode == 0, out.stderr
    args = out.stdout.split("\n")
    assert "FOO" in args  # recipe export forwarded despite no matching prefix
    assert "BASELINE_VAR" not in args  # snapshotted non-prefixed var not force-forwarded
    assert "hostname" in args and "-np" in args  # passthrough args reach real mpirun


# ---------------------------------------------------------------------------
# CPU binding (mpi.cpu_bind) -- bind each rank so LLVM/OpenMP thread pools size to
# the binding, not the whole node (prevents pid/thread exhaustion when ranks share a pod)
# ---------------------------------------------------------------------------
def test_mpirun_wrapper_numa_binding_default():
    # Multi-rank pod + numa: inject `--map-by numa --bind-to numa`, guarded by an
    # arg-scan so a recipe that already binds wins.
    script = mpi_boot.build_mpirun_wrapper_script(cpu_bind="numa", slots=4)
    assert "--map-by numa --bind-to numa" in script
    # the user-binding arg-scan is present (respect a recipe's own --bind-to/--map-by).
    assert "--bind-to|--bind-to=*|--map-by|--map-by=*" in script


def test_mpirun_wrapper_core_binding_uses_slots_and_clamped_nproc():
    # core: PE = cores/rank computed at launch into ${_sflow_pe}. The CPU count is read
    # with OMP_NUM_THREADS/OMP_THREAD_LIMIT UNSET (GNU nproc honors them, and sflow sets
    # OMP_NUM_THREADS -- else nproc returns that cap, not the real CPUs, collapsing PE),
    # CLAMPED to the installed CPUs ($(nproc --all)) so a bogus/inflated count can't
    # over-bind and fail the launch, and floored at 1.
    script = mpi_boot.build_mpirun_wrapper_script(cpu_bind="core", slots=4)
    assert "--map-by ppr:4:node:PE=${_sflow_pe} --bind-to core" in script
    assert "env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc" in script  # ignore OMP cap
    assert "nproc --all" in script  # clamp source (installed CPUs)
    assert "_sflow_pe=$(( _sflow_cores / 4 ))" in script
    assert "-ge 1 ] || _sflow_pe=1" in script  # floor at 1


def test_mpirun_wrapper_core_binding_converts_cpus_to_cores():
    # `PE=N` asks for N *cores* per rank but nproc counts logical CPUs (SMT threads), so
    # PE = nproc/slots over-asks by the SMT factor and OpenMPI aborts "binding more
    # processes than cpus on a resource". The wrapper divides by threads-per-core.
    script = mpi_boot.build_mpirun_wrapper_script(cpu_bind="core", slots=4)
    assert "lscpu" in script and "Thread\\(s\\) per core" in script
    assert "_sflow_cores=$(( _sflow_ncpu / _sflow_smt ))" in script
    assert "_sflow_smt=1" in script  # unknown/garbage threads-per-core -> assume 1


def test_mpirun_wrapper_core_binding_caps_cores_per_rank():
    # PE is capped so a rank takes a modest slice instead of every core it can see --
    # a big affinity mask is the thread explosion the binding exists to prevent, and an
    # uncapped PE is what over-asks on a fat node. 0 = uncapped (cores/slots).
    script = mpi_boot.build_mpirun_wrapper_script(
        cpu_bind="core", slots=4, cpu_bind_cores_per_rank=8
    )
    assert '[ "${_sflow_pe}" -gt 8 ] && _sflow_pe=8' in script
    uncapped = mpi_boot.build_mpirun_wrapper_script(
        cpu_bind="core", slots=4, cpu_bind_cores_per_rank=0
    )
    assert "-gt 8 ] && _sflow_pe=" not in uncapped
    assert "_sflow_pe=$(( _sflow_cores / 4 ))" in uncapped


def test_mpirun_wrapper_core_binding_skips_when_cpuset_smaller_than_ranks():
    # Guard: a pod cpuset with FEWER cores than ranks-per-pod can't satisfy `--bind-to
    # core` (PE floors to 1 but slots*1 > cores -> OpenMPI aborts "not enough processing
    # elements"). Since core became the default bind, the wrapper must DEGRADE to no
    # binding at launch instead of hard-failing a launch that worked before. The check
    # is a runtime `_sflow_cores -lt slots` guard around the bind-flag append.
    script = mpi_boot.build_mpirun_wrapper_script(cpu_bind="core", slots=4)
    assert '"${_sflow_cores}" -lt 4 ]' in script  # runtime guard: cores vs rank count
    assert "skipping --bind-to core" in script  # degrade-to-no-bind path present


def test_mpirun_wrapper_no_binding_for_single_rank_pod():
    # slots=1: one rank per pod never contends -> no binding injected (any mode).
    script = mpi_boot.build_mpirun_wrapper_script(cpu_bind="numa", slots=1)
    assert "--bind-to" not in script
    assert "_sflow_bound" not in script


def test_mpirun_wrapper_no_binding_when_none():
    # cpu_bind=none is an exact no-op even for a multi-rank pod (today's behaviour).
    script = mpi_boot.build_mpirun_wrapper_script(cpu_bind="none", slots=8)
    assert "--bind-to" not in script
    assert "_sflow_bound" not in script


def test_mpirun_wrapper_binding_defaults_off_for_existing_callers():
    # Regression: callers that don't ask for binding (default) get today's wrapper --
    # no map/bind flags -- so the HIGH-blast-radius launch path is unchanged.
    script = mpi_boot.build_mpirun_wrapper_script(inject_hostfile=True)
    assert "--bind-to" not in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_mpirun_wrapper_binding_respects_user_provided_binding(tmp_path, fake_process):
    """Executed end to end: numa flags are injected when the recipe doesn't bind, and
    SUPPRESSED when the recipe passes its own --bind-to (user wins)."""
    fake_process.allow_unregistered(True)
    real = tmp_path / "real_mpirun"
    real.write_text('#!/usr/bin/env bash\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    real.chmod(0o755)
    wrapper = tmp_path / "mpirun"
    wrapper.write_text(
        mpi_boot.build_mpirun_wrapper_script(inject_hostfile=False, cpu_bind="numa", slots=4)
    )
    wrapper.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "SFLOW_REAL_MPIRUN": str(real)}

    # The real mpirun prints each arg on its own line; drop the `[sflow-mpi] launch:`
    # echo line (which re-prints the whole command) so we inspect the REAL argv.
    def _argv(stdout):
        return [ln for ln in stdout.split("\n") if not ln.startswith("[sflow-mpi]")]

    # No recipe binding -> sflow injects `--map-by numa --bind-to numa`.
    out = subprocess.run([str(wrapper), "-np", "4", "app"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    args = _argv(out.stdout)
    assert "--bind-to" in args and "numa" in args and "--map-by" in args

    # Recipe already binds -> sflow injects NOTHING (its --bind-to must not appear twice).
    out = subprocess.run([str(wrapper), "-np", "4", "--bind-to", "core", "app"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    args = _argv(out.stdout)
    assert args.count("--bind-to") == 1  # only the recipe's, none injected
    assert "--map-by" not in args


def test_mpirun_wrapper_binding_respects_rankfile(tmp_path, fake_process):
    """A recipe that binds via --rankfile (not --bind-to/--map-by) must ALSO suppress
    sflow's default core-binding injection -- else OpenMPI rejects the conflicting
    directives now that `core` is the default. A fat cpuset ensures binding WOULD be
    injected absent the fix (so this is a real regression guard, not a no-op)."""
    fake_process.allow_unregistered(True)
    fake = _fake_cpu_topology(tmp_path, nproc=64, threads_per_core=2)  # 32 cores >> 4 ranks
    real = tmp_path / "real_mpirun"
    real.write_text('#!/usr/bin/env bash\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    real.chmod(0o755)
    wrapper = tmp_path / "mpirun"
    wrapper.write_text(
        mpi_boot.build_mpirun_wrapper_script(
            inject_hostfile=False, cpu_bind="core", slots=4
        )
    )
    wrapper.chmod(0o755)
    out = subprocess.run(
        [str(wrapper), "-np", "4", "--rankfile", "rf", "app"],
        capture_output=True,
        text=True,
        env={"PATH": f"{fake}:{os.environ['PATH']}", "SFLOW_REAL_MPIRUN": str(real)},
    )
    assert out.returncode == 0, out.stderr
    argv = [ln for ln in out.stdout.split("\n") if not ln.startswith("[sflow-mpi]")]
    assert "--map-by" not in argv  # sflow injected nothing (recipe's rankfile wins)
    assert "--bind-to" not in argv
    assert "--rankfile" in argv  # the recipe's own binding survives


def test_mpi_config_rejects_negative_cores_per_rank():
    """cpu_bind_cores_per_rank must be >= 0 (0 == uncapped); a negative typo must not
    silently disable the cap."""
    from pydantic import ValidationError

    from sflow.plugins.operators.k8s_mpi import MpiConfig

    MpiConfig(cpu_bind_cores_per_rank=0)  # uncapped -- allowed
    MpiConfig(cpu_bind_cores_per_rank=16)  # allowed
    with pytest.raises(ValidationError):
        MpiConfig(cpu_bind_cores_per_rank=-1)


def _fake_cpu_topology(tmp_path, *, nproc, threads_per_core, lscpu=True):
    """A bin dir (to prepend to PATH) whose `nproc`/`lscpu` report a chosen topology,
    so the wrapper's launch-time core math can be exercised for real. ``lscpu=False``
    shadows lscpu with a failing stub to simulate a slim image (no util-linux), which
    sends the wrapper down its cpu0-sysfs SMT-sibling fallback."""
    bindir = tmp_path / f"fakebin_{nproc}_{threads_per_core}_{int(lscpu)}"
    bindir.mkdir(exist_ok=True)
    (bindir / "nproc").write_text(f"#!/usr/bin/env bash\necho {nproc}\n")
    (bindir / "lscpu").write_text(
        f'#!/usr/bin/env bash\necho "Thread(s) per core:  {threads_per_core}"\n'
        if lscpu
        else "#!/usr/bin/env bash\nexit 127\n"
    )
    for f in ("nproc", "lscpu"):
        (bindir / f).chmod(0o755)
    return bindir


def _run_wrapper(tmp_path, fake_bin, fake_process, *, slots, cores_per_rank, label=""):
    """Run the built shim with a stub `mpirun` that echoes its argv; return (argv, stderr)."""
    fake_process.allow_unregistered(True)  # this one really shells out
    real = tmp_path / "real_mpirun"
    real.write_text('#!/usr/bin/env bash\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    real.chmod(0o755)
    wrapper = tmp_path / f"mpirun_{label}_{slots}_{cores_per_rank}"
    wrapper.write_text(
        mpi_boot.build_mpirun_wrapper_script(
            inject_hostfile=False, cpu_bind="core", slots=slots,
            cpu_bind_cores_per_rank=cores_per_rank,
        )
    )
    wrapper.chmod(0o755)
    out = subprocess.run(
        [str(wrapper), "-np", str(slots), "app"], capture_output=True, text=True,
        env={"PATH": f"{fake_bin}:{os.environ['PATH']}", "SFLOW_REAL_MPIRUN": str(real)},
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.split("\n") if not ln.startswith("[sflow-mpi]")], out.stderr


# Real host shapes sflow lands on. `PE` counts CORES while `nproc` counts logical CPUs,
# so threads-per-core differs by CPU FAMILY, not by arch: x86 Intel/AMD run SMT-2 with
# hyperthreading/SMT enabled, Grace (Neoverse V2) is SMT-1, and Vera (Olympus) is SMT-2
# again via spatial multithreading. The invariant every case must hold is
# `PE * ranks <= physical cores` -- violating it is exactly the OpenMPI abort
# "binding more processes than cpus on a resource".
_CPU_FAMILIES = [
    # (label, nproc/cpuset threads, threads-per-core, ranks-per-pod, expected PE @cap 8)
    ("intel-xeon-8xb200", 224, 2, 8, 8),    # GKE a4-highgpu-8g: 112 cores -> 14, capped
    ("amd-epyc-turin-2s", 512, 2, 8, 8),    # 256 cores -> 32, capped
    ("grace-gb200", 72, 1, 4, 8),           # 72 cores -> 18, capped
    ("grace-gb300-starved", 8, 1, 4, 2),    # tiny cpuset: 8 cores -> 2, cap not reached
    ("vera-1s", 176, 2, 8, 8),              # 88 Olympus cores -> 11, capped
    ("vera-2s", 352, 2, 16, 8),             # 176 cores -> 11, capped
    ("intel-no-smt", 96, 1, 8, 8),          # hyperthreading disabled: 96 cores -> 12
]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
@pytest.mark.parametrize("label,ncpu,smt,slots,want_pe", _CPU_FAMILIES)
def test_mpirun_wrapper_core_binding_pe_math_per_cpu_family(
    tmp_path, fake_process, label, ncpu, smt, slots, want_pe
):
    """Executed end to end per CPU family. The bug this guards: on SMT-2 hosts the old
    `PE = nproc/slots` asked for the THREAD count per rank (224/8 = 28 cores a rank =
    224 of the node's 112 cores) and OpenMPI aborted the launch."""
    fake = _fake_cpu_topology(tmp_path, nproc=ncpu, threads_per_core=smt)
    argv, stderr = _run_wrapper(tmp_path, fake, fake_process, slots=slots, cores_per_rank=8, label=label)
    assert f"ppr:{slots}:node:PE={want_pe}" in argv
    assert f"{slots} rank(s) to {want_pe} core(s) each" in stderr  # decision is logged
    # The invariant OpenMPI enforces: the request must fit in the physical cores.
    assert want_pe * slots <= ncpu // smt


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_mpirun_wrapper_core_binding_cap_is_what_bounds_pe(tmp_path, fake_process):
    """Uncapped (cores_per_rank=0) keeps the old cores/ranks math -- it fits, but hands
    each rank every core it can see, which is the affinity mask the binding exists to
    shrink. The cap is what bounds it: 112 cores / 8 ranks = 14 -> 8."""
    fake = _fake_cpu_topology(tmp_path, nproc=224, threads_per_core=2)
    argv, _ = _run_wrapper(tmp_path, fake, fake_process, slots=8, cores_per_rank=0, label="uncapped")
    assert "ppr:8:node:PE=14" in argv
    argv, _ = _run_wrapper(tmp_path, fake, fake_process, slots=8, cores_per_rank=8, label="capped")
    assert "ppr:8:node:PE=8" in argv


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_mpirun_wrapper_core_binding_smt_fallback_without_lscpu(tmp_path, fake_process):
    """Slim images ship no `lscpu`: threads-per-core then comes from cpu0's sysfs SMT
    siblings, and if that is unreadable too the wrapper assumes 1 (the cap keeps the
    request survivable). Either way the launch must proceed, never abort."""
    fake = _fake_cpu_topology(tmp_path, nproc=224, threads_per_core=2, lscpu=False)
    argv, stderr = _run_wrapper(tmp_path, fake, fake_process, slots=8, cores_per_rank=8, label="nolscpu")
    # This host's own /sys answers (SMT-2 -> PE 8; SMT-1 -> 224/8 = 28 capped to 8),
    # so either way the cap holds the request to 8 cores per rank.
    assert "ppr:8:node:PE=8" in argv
    assert "--bind-to" in argv and "core" in argv


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_mpirun_wrapper_core_binding_degrades_on_tiny_cpuset(tmp_path, fake_process):
    """8 logical CPUs, SMT-2 (= 4 cores) for 8 ranks: `--bind-to core` is unsatisfiable,
    so the wrapper degrades to NO binding rather than aborting the launch."""
    fake = _fake_cpu_topology(tmp_path, nproc=8, threads_per_core=2)
    argv, stderr = _run_wrapper(tmp_path, fake, fake_process, slots=8, cores_per_rank=8, label="tiny")
    assert "--bind-to" not in argv and "--map-by" not in argv
    assert "4 core(s) < 8 rank(s)" in stderr


def test_bootstrap_preamble_injects_cpu_binding_pods_route():
    # Pods route: the bootstrap-built wrapper carries the binding for a multi-rank pod.
    lines = mpi_boot.build_mpi_bootstrap_preamble(gpus_per_node=8, cpu_bind="numa")
    assert "--map-by numa --bind-to numa" in "\n".join(lines)


def test_mpirun_wrapper_role_barrier_worker_idles_leader_waits():
    # Barrier-in-wrapper (pods route, multi-node): the wrapper runs the role barrier
    # ONCE (atomic mkdir lock) -- worker brings up sshd + idles, leader waits for every
    # worker's sshd -- BEFORE injecting/exec'ing, so it engages regardless of how the
    # recipe writes its mpirun line (`fi; ${NSYS_CMD} mpirun`, compound, etc.).
    script = mpi_boot.build_mpirun_wrapper_script(role_barrier=True, ssh_port=2222)
    assert "mkdir" in script and ".barrier.lock" in script
    assert 'if [ "${SFLOW_MPI_N_NODES:-1}" -gt 1 ]' in script  # multi-node gate
    assert "sleep infinity" in script  # worker idles
    assert "waiting for sshd" in script  # leader waits
    # the barrier runs before the exec of the real mpirun
    assert script.index("sleep infinity") < script.index('exec "${_sflow_real}"')


def test_mpirun_wrapper_role_barrier_fails_fast_on_sshd_failure():
    # sshd start failure must ABORT the wrapper (exit non-zero), not let a worker idle
    # forever / the leader fall through to a doomed launch.
    script = mpi_boot.build_mpirun_wrapper_script(role_barrier=True, ssh_port=2222)
    assert "sshd failed to start" in script  # sshd start failure is surfaced
    assert "exit 1" in script  # ...and aborts the wrapper


def test_build_leader_wait_lines_fails_fast_on_unreachable_worker():
    # The leader's wait loop must fail fast when a peer's sshd never becomes reachable,
    # not fall through to launch (which would fail late with "Connection refused").
    from sflow.plugins.k8s.mpi_bootstrap import build_leader_wait_lines

    text = "\n".join(build_leader_wait_lines(ssh_port=2222))
    assert "never became reachable" in text  # exhaustion is surfaced
    assert "exit 1" in text  # ...and aborts


def test_mpirun_wrapper_no_role_barrier_by_default():
    # Operator route / default: NO barrier -- the mpi-operator owns sshd + wait.
    script = mpi_boot.build_mpirun_wrapper_script(inject_hostfile=False)
    assert "sleep infinity" not in script
    assert ".barrier.lock" not in script


def test_launcher_preamble_injects_cpu_binding_operator_route():
    # Operator route: the launcher wrapper carries the binding too (the mpi-operator
    # supplies the hostfile but not CPU binding).
    lines = mpi_boot.build_launcher_preamble(
        cpu_bind="core", slots=2, cpu_bind_cores_per_rank=8
    )
    text = "\n".join(lines)
    assert "--map-by ppr:2:node:PE=${_sflow_pe} --bind-to core" in text
    assert "_sflow_pe=$(( _sflow_cores / 2 ))" in text  # clamped cores/slots
    assert '[ "${_sflow_pe}" -gt 8 ] && _sflow_pe=8' in text  # cap threaded through


# ---------------------------------------------------------------------------
# pods-route bootstrap preamble
# ---------------------------------------------------------------------------
def test_bootstrap_preamble_installs_wrapper_and_branches():
    # The bootstrap installs the shim + keypair and (leader-only) builds the hostfile,
    # exports the SFLOW_MPI_* role vars (so the shim's barrier sees them), then snapshots
    # the env before the recipe. The worker idle + leader wait now live INSIDE the shim
    # (role barrier), which the bootstrap installs -- so they appear in this text.
    lines = mpi_boot.build_mpi_bootstrap_preamble(ssh_port=2222, gpus_per_node=8)
    text = "\n".join(lines)
    # transparent mpirun shim: written to a WRITABLE dir + PATH-prepended, and the
    # real binary captured BEFORE the shadow so `exec mpirun` resolves to the shim.
    assert mpi_boot.SFLOW_MPI_WRAPPER_PATH in text
    assert "SFLOW_REAL_MPIRUN" in text
    assert f"export PATH={mpi_boot.SFLOW_MPI_BIN_DIR}:" in text
    # multi-node branch: keypair install + leader-only hostfile.
    assert "SFLOW_TASK_NODE_INDEX" in text
    assert mpi_boot.SSH_PRIVATE_KEY_ENV in text
    assert 'if [ "${SFLOW_MPI_NODE_INDEX:-0}" = "0" ]; then' in text  # hostfile leader-guarded
    assert "slots=8" in text
    # role vars are EXPORTED so the shim subprocess (barrier) can read them.
    assert 'export SFLOW_MPI_NODE_INDEX=' in text
    assert 'export SFLOW_MPI_N_NODES=' in text and 'export SFLOW_MPI_NODE_IPS=' in text
    # env snapshot (for the shim's -x auto-forward) is emitted before the recipe.
    assert "compgen -e" in text and "env.snapshot" in text
    # the role barrier (worker idle + leader wait) is baked into the installed shim.
    assert "sleep infinity" in text
    assert "waiting for sshd" in text
    assert mpi_boot.SFLOW_MPI_BARRIER_LOCK in text


def test_mpirun_shim_barrier_runs_once_and_gates_on_multinode():
    # The role barrier is in the shim, guarded by an atomic mkdir lock (once per pod)
    # and the multi-node gate, and runs before the exec of the real mpirun.
    wrapper = mpi_boot.build_mpirun_wrapper_script(role_barrier=True, ssh_port=2222)
    assert f'mkdir "{mpi_boot.SFLOW_MPI_BARRIER_LOCK}"' in wrapper
    assert 'if [ "${SFLOW_MPI_N_NODES:-1}" -gt 1 ]' in wrapper
    assert "sshd up on :2222" in wrapper
    assert 'if [ "${SFLOW_MPI_NODE_INDEX:-0}" != "0" ]; then' in wrapper
    assert wrapper.index("sleep infinity") < wrapper.index('exec "${_sflow_real}"')


# ---------------------------------------------------------------------------
# worker command (operator route)
# ---------------------------------------------------------------------------
def test_worker_command_runs_sshd():
    cmd = mpi_boot.build_worker_command(ssh_port=2222)
    assert cmd[:2] == ["bash", "-lc"]
    assert "-D -e -p 2222" in cmd[2]


# ---------------------------------------------------------------------------
# MPIJob manifest builder
# ---------------------------------------------------------------------------
def test_mpijob_manifest_shape():
    m = mpi_boot.build_mpijob_manifest(
        name="server",
        launcher_pod_spec={"containers": [{"name": "l"}]},
        worker_pod_spec={"containers": [{"name": "w"}]},
        worker_replicas=1,
        slots_per_worker=8,
    )
    assert m["apiVersion"] == "kubeflow.org/v2beta1"
    assert m["kind"] == "MPIJob"
    spec = m["spec"]
    assert spec["slotsPerWorker"] == 8
    assert spec["runLauncherAsWorker"] is True
    assert spec["launcherCreationPolicy"] == "WaitForWorkersReady"
    assert spec["sshAuthMountPath"] == "/root/.ssh"
    assert spec["mpiReplicaSpecs"]["Launcher"]["replicas"] == 1
    assert spec["mpiReplicaSpecs"]["Worker"]["replicas"] == 1


def test_inject_scheduling_worker_gets_affinity_and_strips_pin():
    spec = {"nodeSelector": {"kubernetes.io/hostname": "node-0", "tenant": "gpu"}}
    mpi_boot.inject_mpi_scheduling(
        spec, job_name="server", include_hostnames=["node-0", "node-1"],
        strip_hostname_pin=True,
    )
    # hostname pin dropped, label selector preserved.
    assert spec["nodeSelector"] == {"tenant": "gpu"}
    aff = spec["affinity"]
    vals = aff["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"
    ][0]["matchExpressions"][0]["values"]
    assert vals == ["node-0", "node-1"]
    anti = aff["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
    assert anti["topologyKey"] == "kubernetes.io/hostname"
    assert anti["labelSelector"]["matchLabels"] == {
        mpi_boot.MPI_JOB_NAME_LABEL: "server"
    }


def test_inject_scheduling_launcher_keeps_pin():
    spec = {"nodeSelector": {"kubernetes.io/hostname": "node-0"}}
    mpi_boot.inject_mpi_scheduling(spec, job_name="server", one_pod_per_node=True)
    assert spec["nodeSelector"] == {"kubernetes.io/hostname": "node-0"}
    assert "podAntiAffinity" in spec["affinity"]
    assert "nodeAffinity" not in spec["affinity"]


# ---------------------------------------------------------------------------
# pods-route end-to-end render
# ---------------------------------------------------------------------------
def test_pods_route_multinode_render(monkeypatch):
    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="pods",
             monkeypatch=monkeypatch)
    # execute() delivers the keypair via the task env (see _inject_pods_keypair_env);
    # simulate that here since we build the plan directly.
    envs = op._inject_pods_keypair_env(
        {"SFLOW_TASK_ASSIGNED_NODE_IPS": "10.0.0.1,10.0.0.2"}
    )
    plan = op._build_execution_plan(
        task_name="server", script=["exec mpirun -np 16 trtllm-serve /m"], envs=envs,
    )
    manifest, shell = _manifest_from_plan(plan)
    # one pod per node (leader index 0); names allocation-scoped ("abc" in tests).
    assert plan.pod_refs == ["pod/server-abc-0", "pod/server-abc-1"]
    entry = _entrypoint(manifest)
    # bootstrap preamble injected BEFORE the user script, AFTER the gpu-driver line.
    assert "sflow k8s_mpi bootstrap" in entry
    assert entry.index("LD_LIBRARY_PATH=/usr/local/nvidia") < entry.index(
        "sflow k8s_mpi bootstrap"
    ) < entry.index("exec mpirun -np 16 trtllm-serve /m")
    # ephemeral keypair injected into the env Secret (base64), shared by both pods.
    # The secret is created before the manifest heredoc; the value is read from the
    # process env at apply time (never inlined into the manifest/command).
    pre_manifest = shell.split(_MARK, 1)[0]
    assert mpi_boot.SSH_PRIVATE_KEY_ENV in pre_manifest
    assert "PRIVB64" not in shell


def test_pods_route_multinode_runs_setup_on_workers_and_barrier_in_shim(monkeypatch):
    # Worker parity: the recipe setup (exports, mkdir) runs on EVERY pod, unchanged and
    # NOT wrapped in a role gate. The per-role split (worker idle / leader wait) lives in
    # the installed mpirun shim's barrier, which fires at the real mpirun call.
    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="pods",
             monkeypatch=monkeypatch)
    envs = op._inject_pods_keypair_env(
        {"SFLOW_TASK_ASSIGNED_NODE_IPS": "10.0.0.1,10.0.0.2"}
    )
    plan = op._build_execution_plan(
        task_name="server",
        script=["export FOO=1", "mkdir -p /tmp/cache", "mpirun -np 16 trtllm-serve /m"],
        envs=envs,
    )
    entry = _entrypoint(_manifest_from_plan(plan)[0])
    # No source-line gate is inserted; the recipe setup + launch are untouched.
    assert "launch gate (pods route)" not in entry
    assert "export FOO=1" in entry and "mkdir -p /tmp/cache" in entry
    assert "exec mpirun -np 16 trtllm-serve /m" in entry
    # The role split lives in the installed shim's barrier (guarded by the once-lock).
    assert "sleep infinity" in entry
    assert "waiting for sshd" in entry
    assert mpi_boot.SFLOW_MPI_BARRIER_LOCK in entry


def test_pods_route_single_node_has_no_launch_gate(monkeypatch):
    # A single-node task needs no cross-node barrier; the shim's barrier self-skips
    # (N_NODES=1) so the bare local mpirun runs directly.
    op = _op(_backend(nodes=1, gpus_per_node=8), ["node-0"], gpu_count=8, route="pods",
             monkeypatch=monkeypatch)
    plan = op._build_execution_plan(
        task_name="server", script=["export FOO=1", "mpirun -np 8 x"], envs={},
    )
    entry = _entrypoint(_manifest_from_plan(plan)[0])
    assert "launch gate (pods route)" not in entry
    assert "exec mpirun -np 8 x" in entry


def test_pods_route_multinode_plan_is_mpi_world_group(monkeypatch):
    # The multi-node MPI pods plan marks itself an MPI world group so execute()
    # resolves when ANY rank pod terminates instead of blocking on the survivors.
    op = _op(_backend(nodes=3, gpus_per_node=8), ["node-0", "node-1"], gpu_count=16,
             route="pods", monkeypatch=monkeypatch)
    envs = op._inject_pods_keypair_env(
        {"SFLOW_TASK_ASSIGNED_NODE_IPS": "10.0.0.1,10.0.0.2"}
    )
    plan = op._build_execution_plan(
        task_name="server", script=["mpirun -np 16 trtllm-serve /m"], envs=envs,
    )
    assert plan.mpi_world_group is True


def test_pods_route_single_node_plan_not_mpi_world_group(monkeypatch):
    # A single-node MPI task has no peer pods to hang on -- keep the default (await
    # the one pod), so world-group early resolution is off.
    op = _op(_backend(nodes=1, gpus_per_node=8), ["node-0"], gpu_count=8, route="pods",
             monkeypatch=monkeypatch)
    plan = op._build_execution_plan(
        task_name="server", script=["mpirun -np 8 x"], envs={},
    )
    assert plan.mpi_world_group is False


# ---------------------------------------------------------------------------
# prefixed mpirun launch detection (final_launch_index): real recipes wrap the
# launch (`${NSYS_CMD} mpirun ...`) -- it must still engage the gate + auto-exec
# ---------------------------------------------------------------------------
def test_final_launch_index_matches_var_prefixed_mpirun():
    # `${NSYS_CMD} mpirun ...` -- a variable prefix (often empty at runtime) in front
    # of mpirun -- is still THE workload launch that needs sflow's MPI handling.
    assert mpi_boot.final_launch_index(["setup", "${NSYS_CMD} mpirun -np 16 x"]) == 1


def test_final_launch_index_matches_env_and_exec_prefixed_mpirun():
    assert mpi_boot.final_launch_index(["FOO=1 BAR=2 mpirun -np 8 x"]) == 0
    assert mpi_boot.final_launch_index(["exec ${NSYS_CMD} mpirun -np 8 x"]) == 0


def test_final_launch_index_still_matches_bare_and_exec_mpirun():
    # Regression: the original bare / exec forms still resolve.
    assert mpi_boot.final_launch_index(["mpirun -np 8 x"]) == 0
    assert mpi_boot.final_launch_index(["exec mpirun -np 8 x"]) == 0


def test_final_launch_index_ignores_mpirun_only_as_argument():
    # Conservative: a real command that merely mentions mpirun as an ARGUMENT (not the
    # program being run) is NOT a launch -- sflow leaves the user's process model alone.
    assert mpi_boot.final_launch_index(["echo run mpirun by hand"]) is None
    assert mpi_boot.final_launch_index(["python drive.py --launcher mpirun"]) is None


def test_pods_route_barrier_engages_for_compound_mpirun_line(monkeypatch):
    # The real mlperf recipe glues the launch onto an if-block's `fi;`
    # (`fi; ${NSYS_CMD} mpirun ...`), which a source-line gate can't touch. The barrier
    # lives in the mpirun shim, so it engages at the real mpirun call regardless of the
    # launch-line syntax -- fixing the ssh-refused race where the leader launched before
    # the workers' sshd came up.
    op = _op(_backend(nodes=3, gpus_per_node=8), ["node-0", "node-1"], gpu_count=16,
             route="pods", monkeypatch=monkeypatch)
    envs = op._inject_pods_keypair_env(
        {"SFLOW_TASK_ASSIGNED_NODE_IPS": "10.0.0.1,10.0.0.2"}
    )
    plan = op._build_execution_plan(
        task_name="server",
        script=[
            'NSYS_CMD=""; if [ "0" = "1" ]; then',
            '  NSYS_CMD="nsys profile -o x"',
            "fi; ${NSYS_CMD} mpirun -np 16 trtllm-serve /m",
        ],
        envs=envs,
    )
    entry = _entrypoint(_manifest_from_plan(plan)[0])
    # No source-line gate (can't insert into a compound `fi; ... mpirun` line), and the
    # compound launch line is preserved verbatim ...
    assert "launch gate (pods route)" not in entry
    assert "fi; ${NSYS_CMD} mpirun -np 16 trtllm-serve /m" in entry
    # ... but the shim's barrier (leader-wait / worker-idle) IS present and fires at the
    # real mpirun call.
    assert "sleep infinity" in entry and "waiting for sshd" in entry
    assert mpi_boot.SFLOW_MPI_BARRIER_LOCK in entry


# ---------------------------------------------------------------------------
# auto-exec of the mpirun launch (recipe writes a plain `mpirun ...`)
# ---------------------------------------------------------------------------
def test_bare_mpirun_launch_auto_execd_pods_route():
    # The recipe writes a plain `mpirun ...`; sflow auto-`exec`s it so mpirun becomes
    # the container's main process (clean SIGTERM on pod delete). Pods route.
    op = _op(_backend(nodes=1, gpus_per_node=8), ["node-0"], gpu_count=8, route="pods")
    plan = op._build_execution_plan(
        task_name="server",
        script=["export FOO=1", "mpirun -np 8 trtllm-serve /m"],
        envs={},
    )
    entry = _entrypoint(_manifest_from_plan(plan)[0])
    assert "exec mpirun -np 8 trtllm-serve /m" in entry
    # setup line untouched; the launch is neither left bare nor double-`exec`ed.
    assert "export FOO=1" in entry
    assert "\nmpirun -np 8" not in entry
    assert "exec exec" not in entry


def test_bare_mpirun_launch_auto_execd_operator_route():
    op = _op(_backend(nodes=3, gpus_per_node=8), ["node-0", "node-1"], gpu_count=16,
             route="operator", has_mpi_operator=True)
    plan = op._build_execution_plan(
        task_name="server", script=["mpirun -np 16 trtllm-serve /m"], envs={},
    )
    entry = _entrypoint(_manifest_from_plan(plan)[0])
    assert "exec mpirun -np 16 trtllm-serve /m" in entry


def test_explicit_exec_mpirun_not_double_execd():
    # Backward compat: a recipe that still writes `exec mpirun` is left as-is.
    op = _op(_backend(nodes=1, gpus_per_node=8), ["node-0"], gpu_count=8, route="pods")
    plan = op._build_execution_plan(
        task_name="server", script=["exec mpirun -np 8 x"], envs={},
    )
    entry = _entrypoint(_manifest_from_plan(plan)[0])
    assert "exec mpirun -np 8 x" in entry
    assert "exec exec mpirun" not in entry


def test_non_mpirun_final_command_not_auto_execd():
    # Conservative: only a FINAL `mpirun` launch is rewritten. A trailing non-mpirun
    # command means the user has their own process model -> nothing is `exec`ed.
    op = _op(_backend(nodes=1, gpus_per_node=8), ["node-0"], gpu_count=8, route="pods")
    plan = op._build_execution_plan(
        task_name="server", script=["mpirun -np 8 x", "echo done"], envs={},
    )
    entry = _entrypoint(_manifest_from_plan(plan)[0])
    assert "exec mpirun" not in entry
    assert "exec echo" not in entry


def test_inject_pods_keypair_env_delivers_via_task_env(monkeypatch):
    # The keypair must ride the task env so the apply command's runtime Secret
    # creation reads the real value (empty otherwise -> preamble aborts, exit 1).
    monkeypatch.setattr(
        "sflow.plugins.operators.k8s_mpi._generate_ssh_keypair_b64",
        lambda: ("PRIVB64", "PUBB64"),
    )
    backend = _backend(nodes=2, gpus_per_node=4)
    # multi-node pods route -> keypair injected, existing env preserved.
    op = _op(backend, ["node-0", "node-1"], gpu_count=8, route="pods")
    out = op._inject_pods_keypair_env({"A": "1"})
    assert out["A"] == "1"
    assert out[mpi_boot.SSH_PRIVATE_KEY_ENV] == "PRIVB64"
    assert out[mpi_boot.SSH_PUBLIC_KEY_ENV] == "PUBB64"
    # single-node pods route -> no SSH needed, no keypair.
    op1 = _op(backend, ["node-0"], gpu_count=4, route="pods")
    assert mpi_boot.SSH_PRIVATE_KEY_ENV not in op1._inject_pods_keypair_env({})
    # operator route -> the mpi-operator owns the keys, sflow injects none.
    opo = _op(backend, ["node-0", "node-1"], gpu_count=8, route="operator",
              has_mpi_operator=True)
    assert mpi_boot.SSH_PRIVATE_KEY_ENV not in opo._inject_pods_keypair_env({})


def test_pods_route_single_node_skips_keypair(monkeypatch):
    backend = _backend(nodes=1, gpus_per_node=8)
    op = _op(backend, ["node-0"], gpu_count=8, route="pods", monkeypatch=monkeypatch)
    plan = op._build_execution_plan(
        task_name="server", script=["exec mpirun -np 8 x"], envs={}
    )
    _manifest, shell = _manifest_from_plan(plan)
    # single-node: no peers, so the keypair is NOT injected into the env Secret
    # (the entrypoint preamble still references it, guarded at runtime by N>1).
    pre_manifest = shell.split(_MARK, 1)[0]
    assert mpi_boot.SSH_PRIVATE_KEY_ENV not in pre_manifest


# ---------------------------------------------------------------------------
# operator-route end-to-end render
# ---------------------------------------------------------------------------
def test_operator_route_renders_mpijob():
    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="operator",
             has_mpi_operator=True)
    plan = op._build_mpijob_execution_plan(
        task_name="server", script=["exec mpirun -np 16 trtllm-serve /m"], envs={},
    )
    manifest, shell = _manifest_from_plan(plan)
    kinds = [i["kind"] for i in manifest["items"]]
    assert "ConfigMap" in kinds and "MPIJob" in kinds
    mpijob = next(i for i in manifest["items"] if i["kind"] == "MPIJob")
    spec = mpijob["spec"]
    assert spec["slotsPerWorker"] == 8  # per-node GPUs (16 / 2 nodes)
    assert spec["mpiReplicaSpecs"]["Worker"]["replicas"] == 1  # nodes - launcher
    # launcher pinned to reserved node 0 (deterministic HTTP server addressing).
    lspec = spec["mpiReplicaSpecs"]["Launcher"]["template"]["spec"]
    assert lspec["nodeSelector"]["kubernetes.io/hostname"] == "node-0"
    # worker runs sshd + nodeAffinity to the reserved server nodes.
    wspec = spec["mpiReplicaSpecs"]["Worker"]["template"]["spec"]
    assert "-D -e -p 2222" in wspec["containers"][0]["command"][2]
    wvals = wspec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]
    assert wvals == ["node-0", "node-1"]
    # GPU handoff: the CR is applied, then the server nodes' placeholders deleted.
    assert "res-0" in shell and "res-1" in shell
    # MPIJob/pod names are allocation-scoped ("abc" in tests) so parallel runs
    # never collide on the CR name.
    assert plan.mpijob_name == "server-abc"
    assert "mpijob.kubeflow.org/server-abc" in plan.cleanup_refs


def test_operator_route_worker_runs_setup_and_has_readiness_probe():
    # Worker parity (operator route): the Worker runs the recipe's non-mpirun setup
    # (filesystem parity) then execs sshd; a readiness probe on the ssh port makes
    # WaitForWorkersReady gate on setup completion.
    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="operator",
             has_mpi_operator=True)
    plan = op._build_mpijob_execution_plan(
        task_name="server",
        script=["export FOO=1", "mkdir -p /tmp/cache",
                "exec mpirun -np 16 trtllm-serve /m"],
        envs={},
    )
    manifest, _ = _manifest_from_plan(plan)
    mpijob = next(i for i in manifest["items"] if i["kind"] == "MPIJob")
    wspec = mpijob["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"]
    container = wspec["containers"][0]
    wcmd = container["command"][2]
    # worker runs the recipe setup (minus the launch) then execs sshd.
    assert "export FOO=1" in wcmd and "mkdir -p /tmp/cache" in wcmd
    assert "-D -e -p 2222" in wcmd
    # the actual launch is NOT run on the worker.
    assert "trtllm-serve" not in wcmd and "-np 16" not in wcmd
    # readiness probe gates WaitForWorkersReady on sshd (== setup done).
    assert container["readinessProbe"]["tcpSocket"]["port"] == 2222
    # default probe budget is ~15 min (180 x 5s).
    assert container["readinessProbe"]["failureThreshold"] == 180
    assert container["readinessProbe"]["periodSeconds"] == 5


def test_operator_route_worker_readiness_probe_budget_configurable():
    # The worker setup budget is configurable via the single mpi.worker_setup_timeout_seconds
    # knob; it flows into the rendered probe as failureThreshold = ceil(timeout / 5s), with a
    # fixed 5s poll (non-divisible timeouts round up).
    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(
        backend, ["node-0", "node-1"], gpu_count=16, route="operator",
        has_mpi_operator=True,
        mpi_extra={"worker_setup_timeout_seconds": 7201},  # ceil(7201 / 5) = 1441
    )
    plan = op._build_mpijob_execution_plan(
        task_name="server", script=["exec mpirun -np 16 trtllm-serve /m"], envs={},
    )
    manifest, _ = _manifest_from_plan(plan)
    mpijob = next(i for i in manifest["items"] if i["kind"] == "MPIJob")
    probe = mpijob["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"][
        "containers"
    ][0]["readinessProbe"]
    assert probe["failureThreshold"] == 1441  # ceil rounding
    assert probe["periodSeconds"] == 5  # fixed poll granularity
    # untouched probe fields keep their defaults.
    assert probe["initialDelaySeconds"] == 5
    assert probe["timeoutSeconds"] == 5


def test_operator_route_worker_readiness_probe_accepts_expression_default():
    # A raw ${{ }} expression (unresolved in this test path) must not crash render; the
    # resolver-style helper falls back to the historical default rather than raising.
    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(
        backend, ["node-0", "node-1"], gpu_count=16, route="operator",
        has_mpi_operator=True,
        mpi_extra={"worker_setup_timeout_seconds": "${{ variables.T }}"},
    )
    plan = op._build_mpijob_execution_plan(
        task_name="server", script=["exec mpirun -np 16 trtllm-serve /m"], envs={},
    )
    manifest, _ = _manifest_from_plan(plan)
    mpijob = next(i for i in manifest["items"] if i["kind"] == "MPIJob")
    probe = mpijob["spec"]["mpiReplicaSpecs"]["Worker"]["template"]["spec"][
        "containers"
    ][0]["readinessProbe"]
    assert probe["failureThreshold"] == 180  # unresolvable -> default (~15 min)


def test_operator_route_compute_domain_defaults_mnnvl_env():
    # agg-style MPIJob on a rack-NVLink cluster: pods that join the IMEX
    # ComputeDomain get BOTH MNNVL transport enables (NCCL + UCX cuda_ipc) on the
    # launcher AND worker containers, so cross-node collectives / KV transfer ride
    # the NVLink fabric rather than intra-node NVLink + (possibly slow-TCP) network.
    backend = _backend(nodes=3, gpus_per_node=8)
    backend._compute_domain_channel = "cd-chan"
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="operator",
             has_mpi_operator=True)
    plan = op._build_mpijob_execution_plan(
        task_name="server", script=["exec mpirun -np 16 trtllm-serve /m"], envs={},
    )
    manifest, _ = _manifest_from_plan(plan)
    mpijob = next(i for i in manifest["items"] if i["kind"] == "MPIJob")
    specs = mpijob["spec"]["mpiReplicaSpecs"]
    for role in ("Launcher", "Worker"):
        container = specs[role]["template"]["spec"]["containers"][0]
        env = {e["name"]: e["value"] for e in container.get("env", [])}
        assert env.get("NCCL_MNNVL_ENABLE") == "1", role
        assert env.get("UCX_CUDA_IPC_ENABLE_MNNVL") == "y", role


def test_operator_route_cpu_request_scales_with_gpus():
    # MPIJob launcher AND worker templates get requests.cpu = cpu_per_gpu (8) x
    # per-pod GPUs (16 GPUs / 2 nodes = 8 -> 64), so ranks never run BestEffort.
    backend = _backend(nodes=3, gpus_per_node=8, cpu_per_gpu=8)
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="operator",
             has_mpi_operator=True)
    plan = op._build_mpijob_execution_plan(
        task_name="server", script=["exec mpirun -np 16 trtllm-serve /m"], envs={},
    )
    manifest, _ = _manifest_from_plan(plan)
    mpijob = next(i for i in manifest["items"] if i["kind"] == "MPIJob")
    specs = mpijob["spec"]["mpiReplicaSpecs"]
    for role in ("Launcher", "Worker"):
        res = specs[role]["template"]["spec"]["containers"][0]["resources"]
        assert res["requests"]["cpu"] == "64", role
        assert "cpu" not in res.get("limits", {}), role  # requests-only default


def test_operator_route_mounts_output_emptydir_on_both_templates():
    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="operator",
             has_mpi_operator=True)
    envs = {
        "SFLOW_OUTPUT_DIR": "/host/out",
        "SFLOW_TASK_OUTPUT_DIR": "/host/out/run-1/server",
        "SFLOW_WORKFLOW_OUTPUT_DIR": "/host/out/run-1",
    }
    plan = op._build_mpijob_execution_plan(
        task_name="server", script=["exec mpirun -np 16 trtllm-serve /m"], envs=envs,
    )
    manifest, _ = _manifest_from_plan(plan)
    cm = next(
        i for i in manifest["items"]
        if i["kind"] == "ConfigMap" and "entrypoint.sh" in i.get("data", {})
    )
    ep = cm["data"]["entrypoint.sh"]
    # Env is NOT remapped; the launcher just mkdir's the per-task dir (mpirun -x forwards
    # the driver-host SFLOW_* to workers, which get the same emptyDir mount).
    assert 'mkdir -p "$SFLOW_TASK_OUTPUT_DIR"' in ep
    assert "export SFLOW_TASK_OUTPUT_DIR" not in ep
    # Both launcher AND worker pods get a writable emptyDir at the resolved output dir.
    mpijob = next(i for i in manifest["items"] if i["kind"] == "MPIJob")
    specs = mpijob["spec"]["mpiReplicaSpecs"]
    for role in ("Launcher", "Worker"):
        mounts = specs[role]["template"]["spec"]["containers"][0]["volumeMounts"]
        assert any(
            m["mountPath"] == "/host/out" and m["name"] == "sflow-scratch"
            for m in mounts
        ), role


def test_operator_route_single_node_drops_worker_spec():
    backend = _backend(nodes=2, gpus_per_node=8)
    op = _op(backend, ["node-0"], gpu_count=8, route="operator", has_mpi_operator=True)
    plan = op._build_mpijob_execution_plan(
        task_name="server", script=["exec mpirun -np 8 x"], envs={},
    )
    manifest, _ = _manifest_from_plan(plan)
    mpijob = next(i for i in manifest["items"] if i["kind"] == "MPIJob")
    # launcher-only MPIJob (no workers) has no Worker replica spec.
    assert "Worker" not in mpijob["spec"]["mpiReplicaSpecs"]


def test_operator_route_launcher_command_has_wrapper_and_user_script():
    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="operator",
             has_mpi_operator=True)
    plan = op._build_mpijob_execution_plan(
        task_name="server", script=["exec mpirun -np 16 trtllm-serve /m"], envs={},
    )
    manifest, _ = _manifest_from_plan(plan)
    entry = _entrypoint(manifest)
    assert "sflow k8s_mpi launcher" in entry
    assert "exec mpirun -np 16 trtllm-serve /m" in entry


# ---------------------------------------------------------------------------
# operator-route launcher-discovery timeout (FIX)
# ---------------------------------------------------------------------------
class _FakeLauncher:
    """Minimal launcher stub: its apply command 'succeeds' (rc 0)."""

    async def run_async(self, *args, **kwargs):
        return 0


def _drive_execute_mpijob(op, monkeypatch, *, env=None):
    """Run op._execute_mpijob with all kubectl-touching lifecycle calls stubbed out,
    returning the ``timeout`` captured by the patched discover_launcher_pod."""
    from sflow.plugins.operators import k8s_mpi as k8s_mpi_mod

    captured: dict = {}

    async def _fake_discover(job_name, *, global_args, ns_args, timeout, **kw):
        captured["timeout"] = timeout
        return None  # no launcher pod -> skips log streaming/exit-code lookup

    async def _fake_watch(*args, **kwargs):
        return "Succeeded"

    async def _fake_delete(*args, **kwargs):
        return None

    monkeypatch.setattr(
        k8s_mpi_mod.mpi_lifecycle, "discover_launcher_pod", _fake_discover
    )
    monkeypatch.setattr(
        k8s_mpi_mod.mpi_lifecycle, "watch_mpijob_until_terminal", _fake_watch
    )
    monkeypatch.setattr(k8s_mpi_mod.k8s_lifecycle, "delete_objects", _fake_delete)

    rc = asyncio.run(
        op._execute_mpijob(
            launcher=_FakeLauncher(),
            output_logger=None,
            env=env or {},
            task_name="server",
            script=["exec mpirun -np 16 trtllm-serve /m"],
            status_note=None,
        )
    )
    return rc, captured


def test_operator_route_launcher_discovery_timeout_default_passed_through(monkeypatch):
    # Regression for the previous hardcode: the callsite now passes a timeout at all, and
    # the default equals the module-level _LAUNCHER_DISCOVERY_TIMEOUT (600.0).
    from sflow.plugins.k8s import mpi_lifecycle

    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="operator",
             has_mpi_operator=True)
    rc, captured = _drive_execute_mpijob(op, monkeypatch)
    assert rc == 0
    assert captured["timeout"] == mpi_lifecycle._LAUNCHER_DISCOVERY_TIMEOUT == 600.0


def test_operator_route_launcher_discovery_timeout_override_passed_through(monkeypatch):
    # FIX: mpi.launcher_discovery_timeout overrides the discover_launcher_pod timeout.
    backend = _backend(nodes=3, gpus_per_node=8)
    op = _op(backend, ["node-0", "node-1"], gpu_count=16, route="operator",
             has_mpi_operator=True,
             mpi_extra={"launcher_discovery_timeout": 1800})
    rc, captured = _drive_execute_mpijob(op, monkeypatch)
    assert rc == 0
    assert captured["timeout"] == 1800.0


# ---------------------------------------------------------------------------
# backend RBAC gating
# ---------------------------------------------------------------------------
# Only route:operator HARD-requires mpijobs RBAC (explicit choice -> clear error).
# route:auto never does: missing RBAC means "installed but not usable" and the
# operator falls back to pods (gated by the usability probe), so it must not be in
# the required set regardless of whether the CRD was detected.
@pytest.mark.parametrize(
    "routes,expected",
    [
        (["pods"], False),
        (["auto"], False),
        (["operator"], True),
        ([], False),
    ],
)
def test_backend_mpijobs_rbac_gating(routes, expected):
    backend = _backend()
    backend.note_mpi_operator_routes(routes)
    has = any(r[1] == "mpijobs.kubeflow.org" for r in backend._required_permissions())
    assert has is expected


# ---------------------------------------------------------------------------
# MPIJob launcher log path (drain + finalize) -- previously untested
# ---------------------------------------------------------------------------


def _drive_mpi_execute(monkeypatch, tmp_path, *, phase="Succeeded"):
    """Run K8sMpiOperator.execute on the operator route with everything faked."""
    from sflow.plugins.k8s import lifecycle as k8s_life
    from sflow.plugins.k8s import mpi_lifecycle as mpi_life

    events: list = []

    class _Proc:
        returncode = None

        async def wait(self):
            return 0

    class _Launcher:
        async def run_async(self, command, **kw):
            events.append(("apply",))
            return 0

    async def fake_discover(job, **kw):
        return "pod/launcher-0"

    async def fake_stream(log_command, dest_path):
        events.append(("stream", dest_path))
        return _Proc()

    async def fake_tail(path, *, task_name):
        await asyncio.sleep(3600)

    async def fake_watch(mpijob_ref, launcher_ref, **kw):
        events.append(("watch",))
        return phase

    async def fake_stop(proc, *, terminal, kill_group=False):
        events.append(("stop", terminal))
        return terminal

    def fake_sanitize(paths):
        events.append(("sanitize", tuple(paths)))

    async def fake_exit(pod_ref, *, phase="", **kw):
        return 0

    async def fake_delete(refs, **kw):
        events.append(("delete",))

    monkeypatch.setattr(mpi_life, "discover_launcher_pod", fake_discover)
    monkeypatch.setattr(mpi_life, "watch_mpijob_until_terminal", fake_watch)
    monkeypatch.setattr(k8s_life, "start_pod_log_file_stream", fake_stream)
    monkeypatch.setattr(k8s_life, "tail_file_to_console", fake_tail)
    monkeypatch.setattr(k8s_life, "stop_log_stream", fake_stop)
    monkeypatch.setattr(k8s_life, "sanitize_streamed_logs", fake_sanitize)
    monkeypatch.setattr(k8s_life, "pod_exit_code", fake_exit)
    monkeypatch.setattr(k8s_life, "delete_objects", fake_delete)

    op = _op(_backend(nodes=2), ["node-0", "node-1"], gpu_count=16,
             route="operator", has_mpi_operator=True)
    rc = asyncio.run(
        op.execute(
            launcher=_Launcher(), output_logger=None,
            env={"SFLOW_TASK_OUTPUT_DIR": str(tmp_path)},
            task_name="mpi_server", script=["run"],
        )
    )
    return rc, events


def test_mpi_launcher_log_is_drained_then_finalized_before_delete(monkeypatch, tmp_path):
    """The MPIJob launcher's follow must drain, not be killed mid-stream.

    Killing it is what truncated <task>.log and forced the (now removed) one-shot
    re-fetch. This path had NO test coverage when it was changed.
    """
    rc, events = _drive_mpi_execute(monkeypatch, tmp_path)
    assert rc == 0
    kinds = [e[0] for e in events]
    assert ("stop", True) in events, f"terminal MPIJob must DRAIN its follow: {events}"
    assert any(e[0] == "sanitize" for e in events), "the streamed log must be finalized"
    # Ordering: watch -> stop the stream -> finalize the log -> delete the CR.
    assert (kinds.index("watch") < kinds.index("stop")
            < kinds.index("sanitize") < kinds.index("delete")), kinds
    sanitized = next(e[1] for e in events if e[0] == "sanitize")
    assert sanitized == (str(tmp_path / "mpi_server.log"),), sanitized


def test_mpi_non_terminal_job_cuts_the_stream_instead_of_draining(monkeypatch, tmp_path):
    """A launcher that never reached a terminal phase has no EOF to wait for."""
    rc, events = _drive_mpi_execute(monkeypatch, tmp_path, phase="")
    assert ("stop", False) in events, f"non-terminal must CUT, not drain: {events}"
