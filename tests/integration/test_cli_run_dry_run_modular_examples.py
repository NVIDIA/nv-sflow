# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run the backend-agnostic modular samples (``examples/modular``) end to end.

Each case composes a backend fragment + workload(s) + the shared benchmark and
proves that: (1) the same workload runs on Slurm / plain K8s / K8s-MPI unchanged,
(2) the shared backend name ``cluster`` and logical operators resolve, and
(3) ``required_by: [benchmark]`` on the server folds into the benchmark's
``depends_on`` (no ``--missable-tasks``).
"""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sflow.cli import app

_MODULAR = (
    Path(__file__).resolve().parents[2] / "examples" / "modular" / "backend_agnostic"
)


# name -> (relative file list, extra -s overrides)
_CASES = {
    "trtllm_k8s": (
        ["backends/k8s.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_trtllm.yaml", "benchmark.yaml"],
        [],
    ),
    "trtllm_k8s_mpi": (
        ["backends/k8s_mpi.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_trtllm.yaml", "benchmark.yaml"],
        ["NUM_NODES=2", "SERVER_GPUS=16"],
    ),
    "vllm_slurm": (
        ["backends/slurm.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_vllm.yaml", "benchmark.yaml"],
        [],
    ),
    "sglang_k8s": (
        ["backends/k8s.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_sglang.yaml", "benchmark.yaml"],
        [],
    ),
    "serve_k8s": (
        ["backends/k8s.yaml", "workloads/trtllm_serve.yaml", "benchmark.yaml"],
        [],
    ),
    # TRT-LLM takes the MPI fragment: it wants one rank per GPU. Pairing it with
    # backends/slurm.yaml would give it a single rank.
    "serve_slurm_mpi": (
        ["backends/slurm_mpi.yaml", "workloads/trtllm_serve.yaml", "benchmark.yaml"],
        [],
    ),
    "trtllm_slurm_mpi": (
        ["backends/slurm_mpi.yaml", "workloads/dynamo_common.yaml",
         "workloads/dynamo_trtllm.yaml", "benchmark.yaml"],
        [],
    ),
    "vllm_serve_k8s": (
        ["backends/k8s.yaml", "workloads/vllm_serve.yaml", "benchmark.yaml"],
        [],
    ),
    "vllm_serve_slurm": (
        ["backends/slurm.yaml", "workloads/vllm_serve.yaml", "benchmark.yaml"],
        [],
    ),
    "sglang_serve_k8s": (
        ["backends/k8s.yaml", "workloads/sglang_serve.yaml", "benchmark.yaml"],
        [],
    ),
    "sglang_serve_slurm": (
        ["backends/slurm.yaml", "workloads/sglang_serve.yaml", "benchmark.yaml"],
        [],
    ),
}


def _dry_run(files, overrides, tmp_path, verbose=False):
    args = ["run"]
    for f in files:
        cfg = _MODULAR / f
        assert cfg.exists(), cfg
        args += ["-f", str(cfg)]
    for o in overrides:
        args += ["-s", o]
    args += [
        "--workspace-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"),
        "--dry-run",
    ]
    # Per-task operator settings (ntasks / ntasks_per_node / mpi) only reach the
    # plan output under --verbose.
    if verbose:
        args.append("--verbose")
    return CliRunner().invoke(app, args, catch_exceptions=False)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_modular_composition_dry_runs(name, tmp_path):
    files, overrides = _CASES[name]
    result = _dry_run(files, overrides, tmp_path)

    assert result.exit_code == 0, result.output
    # All backend fragments name their backend `cluster`.
    assert "backend 'cluster'" in result.output, result.output
    assert "server" in result.output and "benchmark" in result.output
    # required_by folded: the benchmark hub gained a depends_on the server.
    assert "depends_on=['server']" in result.output, result.output


@pytest.mark.parametrize(
    "name,workload",
    [
        ("vllm_serve_slurm", "vLLM"),
        ("sglang_serve_slurm", "SGLang"),
        ("vllm_slurm", "Dynamo vLLM"),
    ],
)
def test_slurm_fragment_runs_single_process_servers(name, workload, tmp_path):
    """vLLM and SGLang shard the model inside ONE process.

    backends/slurm.yaml used to launch every server as SERVER_GPUS srun ranks,
    which for these frameworks starts that many independent copies of the server
    competing for the same port -- the task then fails for a reason that has
    nothing to do with the model. One task per node is what prevents that; the
    fragments are single-node (NUM_NODES=1), so that is one server.
    """
    files, overrides = _CASES[name]
    result = _dry_run(files, [*overrides, "SERVER_GPUS=4"], tmp_path, verbose=True)

    assert result.exit_code == 0, result.output
    assert "ntasks_per_node: 1" in result.output, f"{workload} must run one per node"
    # The bug being guarded: never one rank per GPU for these frameworks.
    assert "ntasks: 4" not in result.output, result.output
    assert "ntasks_per_node: 4" not in result.output, result.output


@pytest.mark.parametrize("name", ["serve_slurm_mpi", "trtllm_slurm_mpi"])
def test_slurm_mpi_fragment_runs_one_rank_per_gpu(name, tmp_path):
    """TRT-LLM is the other shape: one MPI rank per GPU, capped at the node."""
    files, overrides = _CASES[name]
    result = _dry_run(
        files, [*overrides, "SERVER_GPUS=4", "GPUS_PER_NODE=8"], tmp_path, verbose=True
    )

    assert result.exit_code == 0, result.output
    assert "ntasks: 4" in result.output, result.output
    # Capped at the node's GPU count so a wider server spreads instead of
    # over-subscribing the first node.
    assert "ntasks_per_node: 4" in result.output, result.output
    assert "mpi: pmix" in result.output, result.output


def test_slurm_mounts_the_model_artifact_into_the_container(tmp_path):
    """The weights must be MOUNTED, not merely named.

    These fragments used to carry a plain ``MODEL_PATH`` variable, which named a
    path and left mounting to whoever ran it -- so a server started, found
    nothing at that path, and failed for a reason unrelated to the model. The
    backend now declares an ``fs://`` artifact and sflow mounts it at the same
    path inside the container.
    """
    files, overrides = _CASES["vllm_serve_slurm"]
    result = _dry_run(files, overrides, tmp_path, verbose=True)

    assert result.exit_code == 0, result.output
    # The sample's placeholder path does not exist on this host, so sflow mounts
    # the nearest real ancestor; either way the weights land at the same path
    # inside the container as outside, which is the property under test.
    assert "/models:/models" in result.output, result.output
    # And the server is pointed at the artifact, not a bare variable.
    assert "/models/Qwen3-0.6B" in result.output, result.output


def test_model_artifact_override_moves_the_mount(tmp_path):
    """``-a LOCAL_MODEL_PATH=fs://...`` must retarget the mount AND the command.

    The two have to move together: overriding the artifact but leaving the
    server pointed at the old path is exactly the silent failure the artifact
    exists to prevent.
    """
    model = tmp_path / "weights"
    model.mkdir()
    files, _ = _CASES["vllm_serve_slurm"]
    args_files = []
    for f in files:
        args_files += ["-f", str(_MODULAR / f)]
    result = CliRunner().invoke(
        app,
        ["run", *args_files,
         "-a", f"LOCAL_MODEL_PATH=fs://{model}",
         "--workspace-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
         "--dry-run", "--verbose"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    # rich wraps long paths across lines, so compare with whitespace collapsed.
    flat = "".join(result.output.split())
    assert f"{model}:{model}" in flat, "mount must follow the override"
    assert "/models/Qwen3-0.6B" not in flat, "stale default must be gone"


@pytest.mark.parametrize(
    "name",
    ["trtllm_k8s", "vllm_slurm", "sglang_k8s", "trtllm_slurm_mpi"],
)
def test_dynamo_endpoints_are_declared_for_every_backend(name, tmp_path):
    """Dynamo finds NATS and etcd through the ENVIRONMENT, not CLI flags.

    ``python3 -m dynamo.*`` reads ETCD_ENDPOINTS and NATS_SERVER; sflow exports
    workflow variables into each task's environment. dynamo_common.yaml declared
    only HEAD_NODE_IP, so the frontend and every worker fell back to localhost --
    which happens to work while all tasks share one network namespace, and
    silently fails to register workers as soon as one lands elsewhere. The
    shipped self-contained recipes declare both; the modular fragments must too.
    """
    files, overrides = _CASES[name]
    result = _dry_run(files, overrides, tmp_path, verbose=True)

    assert result.exit_code == 0, result.output
    assert "ETCD_ENDPOINTS" in result.output, result.output
    assert "NATS_SERVER" in result.output, result.output


def test_dynamo_endpoint_ports_match_their_readiness_probes(tmp_path):
    """The URL a client dials and the port the probe waits on must be one value.

    Both are now driven by ETCD_PORT / NATS_PORT, so a port change cannot leave
    the probe reporting "up" on a port nothing serves.
    """
    common = (_MODULAR / "workloads" / "dynamo_common.yaml").read_text()

    assert "${{ variables.ETCD_PORT }}" in common
    assert "${{ variables.NATS_PORT }}" in common
    # No bare ports left in the endpoint URLs or the probes.
    for literal in (":2379", ":4222", "port: 2379", "port: 4222"):
        assert literal not in common, f"hard-coded {literal} should use the variable"
    # Peer URLs keep their own port, also declared.
    assert "${{ variables.ETCD_PEER_PORT }}" in common
    assert ":2380" not in common


@pytest.mark.parametrize(
    "workload",
    ["dynamo_vllm_disagg.yaml", "dynamo_sglang_disagg.yaml", "dynamo_trtllm_disagg.yaml"],
)
def test_disagg_readiness_uses_a_marker_every_worker_actually_logs(workload):
    """One verified marker for both workers, on every framework.

    ``Registering model with endpoint types`` is logged only by the worker that
    serves the OpenAI API. Sharing it across both workers meant prefill waited
    1200s for a line it never prints and failed the workflow with a healthy
    engine -- on vLLM (ssh-slurm-20260908-105547-a11a2507) and again on TRT-LLM
    (ssh-slurm-mpi-20260909-040120-eef79546). It also fires ~4.5 min too EARLY on
    TRT-LLM decode, before the engine has loaded.

    ``Registered endpoint 'generate'`` -- the runtime accepting work -- was
    verified to appear in each worker's OWN prefill_server.log/decode_server.log
    (and never in frontend_server.log) across all six real logs from the three
    frameworks. That matters because ``log_watch`` reads only the task's own
    ``<task>.log`` unless ``logger_task_name`` redirects it.
    """
    text = (_MODULAR / "workloads" / workload).read_text()
    config = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )

    assert config.count("""match_pattern: "Registered endpoint 'generate'\"""") == 2, (
        "both prefill and decode must watch the same verified marker"
    )
    assert "Registering model with endpoint types" not in config, (
        "the OpenAI-API-only line is not a readiness signal for a disagg worker"
    )
    # The marker is emitted by the WORKER, so the probe must keep reading the
    # worker's own log. Pointing it at another task (e.g. frontend_server) would
    # silently watch a file the line never appears in.
    assert "logger_task_name" not in config, (
        "disagg readiness must watch the worker's own log, not another task's"
    )


def test_aiperf_tokenizer_defaults_to_the_mounted_weights():
    """A bare model name makes aiperf try HuggingFace and die at startup.

    TOKENIZER defaulted to SERVED_MODEL_NAME, so aiperf got ``--tokenizer
    Qwen3-0.6B`` and failed with ``OSError("Qwen3-0.6B is not a local folder and
    is not a valid model identifier ...")`` before issuing a single request
    (sflow_output/ssh-slurm-20260909-024428-8a7a7607). The weights are already
    mounted for the server, so the tokenizer points at the same artifact.
    """
    text = (_MODULAR / "benchmark.yaml").read_text()
    config = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "value: ${{ artifacts.LOCAL_MODEL_PATH.path }}" in config
    assert "value: ${{ variables.SERVED_MODEL_NAME }}" not in config


def test_aiperf_writes_results_where_they_are_collected_and_gates_on_them():
    """aiperf exits 0 even when it measured nothing, so the recipe must check.

    Two failures in one run: without ``--artifact-dir`` the exports went to a
    container-local path nothing collects, and because aiperf still exited 0 the
    task went COMPLETED and the workflow reported success with no benchmark at
    all. The CSV must land under the task output dir and carry a metric row.
    """
    text = (_MODULAR / "benchmark.yaml").read_text()

    assert "--artifact-dir" in text, "results must be written where they are collected"
    # Same location the self-contained Slurm recipes use.
    assert "SFLOW_WORKFLOW_OUTPUT_DIR" in text
    # The gate: aiperf's own output decides, not its exit status.
    assert "profile_export_aiperf.csv" in text
    assert "exit 1" in text, "a benchmark that measured nothing must fail the task"


def test_modular_aiperf_flags_match_the_self_contained_slurm_recipes():
    """A number measured here must be comparable with one measured there.

    The modular hub had drifted to a bare-bones invocation -- no stddev pinning,
    no ``--extra-inputs`` (so no ``ignore_eos``/``min_tokens``, meaning outputs
    could stop early and skew throughput), no ``--random-seed``, and a different
    ``--ui``/``--record-processors``. Same workload, different measurement.
    """
    ref_path = (
        Path(__file__).resolve().parents[2]
        / "examples" / "self_contained" / "slurm" / "dynamo_vllm_disagg.yaml"
    )

    def flags(text: str) -> set[str]:
        block = text.split("aiperf profile", 1)[1].split("Benchmarking finished", 1)[0]
        return set(re.findall(r"--[a-z-]+", block))

    modular = flags((_MODULAR / "benchmark.yaml").read_text())
    reference = flags(ref_path.read_text())

    assert modular == reference, (
        f"only in modular: {sorted(modular - reference)}; "
        f"only in reference: {sorted(reference - modular)}"
    )


@pytest.mark.parametrize(
    "workload", ["dynamo_sglang.yaml", "dynamo_sglang_disagg.yaml"]
)
def test_sglang_workers_bind_all_interfaces(workload):
    """SGLang binds its servers to localhost unless told otherwise.

    Dynamo advertises the auto-detected LAN IP to discovery, so without
    ``--host 0.0.0.0`` prefill published ``10.85.54.9:8998`` while its bootstrap
    server was listening on loopback. Decode then failed the KV handshake for
    every request with ``Errno 111 Connection refused`` -- the servers looked
    healthy and READY, aiperf reported "512 valid requests and 0 errors", and not
    one token was generated (sflow_output/ssh-slurm-20260909-034509-95fcddb3).

    Only the SGLang workloads need this; the vLLM and TRT-LLM dynamo workers do
    not take ``--host``, matching examples/self_contained/slurm/.
    """
    text = (_MODULAR / "workloads" / workload).read_text()
    servers = text.count("python3 -m dynamo.sglang")
    assert servers >= 1
    assert text.count("--host 0.0.0.0") == servers, (
        "every SGLang worker must bind all interfaces, not loopback"
    )


def test_aiperf_gate_rejects_a_run_that_generated_no_tokens():
    """Counting CSV rows is not enough to prove a benchmark happened.

    When the server accepts requests but streams back nothing, aiperf still exits
    0, still reports "N valid requests and 0 errors", and still writes its
    AGGREGATE rows -- only the per-request latency section is empty. The gate has
    to require a metric that exists only once tokens were generated.
    """
    text = (_MODULAR / "benchmark.yaml").read_text()
    assert "Time to First Token" in text, (
        "gate must require a per-request generation metric, not just row count"
    )
    assert text.count("exit 1") >= 2, "both no-CSV and no-generation must fail"


def test_modular_dynamo_dag_chains_through_infra(tmp_path):
    """The Dynamo chain wires nats/etcd -> frontend -> server -> benchmark."""
    files, overrides = _CASES["trtllm_k8s"]
    result = _dry_run(files, overrides, tmp_path)
    assert result.exit_code == 0, result.output
    for task in ("nats_server", "etcd_server", "frontend_server", "server", "benchmark"):
        assert task in result.output, result.output
    # server depends on the frontend (from dynamo_common) ...
    assert "depends_on=['frontend_server']" in result.output
    # ... and the benchmark depends on the server (from required_by).
    assert "depends_on=['server']" in result.output
