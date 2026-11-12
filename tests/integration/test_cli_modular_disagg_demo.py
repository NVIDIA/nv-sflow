"""The generated modular bundle serves the same asymmetric DAG on all backends."""

import asyncio
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sflow.cli import app
from sflow.config.loader import safe_load
from sflow.core.probe import ProbeType
from sflow.plugins.probes.log_watch import LogWatchProbe


def test_k8s_demo_uses_cli_selected_read_only_model_pvc(tmp_path):
    root = Path(__file__).resolve().parents[2] / "examples/modular/dynamo_vllm"
    output = tmp_path / "pvc.yaml"
    result = CliRunner().invoke(
        app,
        [
            "compose",
            "-f",
            str(root / "dynamo_vllm_workload.yaml"),
            "-f",
            str(root / "k8s.yaml"),
            "--resolve",
            "-o",
            str(output),
            "--set",
            "MODEL_PVC_CLAIM=demo-model-cache",
            "--set",
            "K8S_NAMESPACE=demo-namespace",
            "--artifact",
            "LOCAL_MODEL_PATH=fs:///models/snapshots/demo-model",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    config = safe_load(output.read_text())
    backend = config["backends"][0]
    assert backend["namespace"] == "demo-namespace"
    volume = next(v for v in backend["volumes"] if v["claim"] == "demo-model-cache")
    assert volume["read_only"] is True
    artifact = next(a for a in config["artifacts"] if a["name"] == "LOCAL_MODEL_PATH")
    assert artifact["uri"].startswith(f"fs://{volume['mount_path']}/")


@pytest.mark.parametrize(
    "role,replica,endpoint,other_endpoint",
    [
        ("prefill", 0, "dynamo.prefill", "dynamo.backend"),
        ("prefill", 1, "dynamo.prefill", "dynamo.backend"),
        ("decode", 0, "dynamo.backend", "dynamo.prefill"),
    ],
)
def test_demo_readiness_matches_dynamo_121_registration_logs(
    role, replica, endpoint, other_endpoint, tmp_path
):
    root = Path(__file__).resolve().parents[2] / "examples/modular/dynamo_vllm"
    config = safe_load((root / "dynamo_vllm_workload.yaml").read_text())
    task_config = next(
        t for t in config["workflow"]["tasks"] if t["name"] == f"{role}_server"
    )
    readiness = task_config["probes"]["readiness"]
    assert any("http_post" in p for p in readiness)
    watch = next(p["log_watch"] for p in readiness if "log_watch" in p)
    name = f"{role}_server_{replica}"
    task = SimpleNamespace(name=name, envs={"SFLOW_WORKFLOW_OUTPUT_DIR": str(tmp_path)})
    log = tmp_path / name / f"{name}.log"
    log.parent.mkdir()
    probe = LogWatchProbe(
        regex_pattern=watch["match_pattern"],
        match_count=watch["match_count"],
        type=ProbeType.READINESS,
    )

    # Replay the Dynamo 1.2.1 worker registration format.
    def registration(component, method):
        return f"INFO shared_tcp_endpoint: Registered endpoint '{component}.{method}' with shared TCP server\n"

    log.write_text(
        registration(endpoint, "get_perf_metrics")
        + registration(other_endpoint, "generate")
    )
    assert asyncio.run(probe.check(task)) is False
    with log.open("a") as stream:
        stream.write(registration(endpoint, "generate"))
    assert asyncio.run(probe.check(task)) is True


@pytest.mark.parametrize("backend", ["k8s", "slurm", "docker"])
def test_demo_infrastructure_uses_shared_dynamo_mode_flags(backend, tmp_path):
    root = Path(__file__).resolve().parents[2] / "examples/modular/dynamo_vllm"
    output = tmp_path / "resolved.yaml"
    result = CliRunner().invoke(
        app,
        [
            "compose",
            "-f",
            str(root / "dynamo_vllm_workload.yaml"),
            "-f",
            str(root / f"{backend}.yaml"),
            "--resolve",
            "-o",
            str(output),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    config = safe_load(output.read_text())
    workload = safe_load((root / "dynamo_vllm_workload.yaml").read_text())
    infra = safe_load((root / f"{backend}.yaml").read_text())
    assert "SERVER_IMAGE" not in infra.get("variables", {})
    operators = {operator["name"]: operator for operator in config["operators"]}
    for name in ("server", "helper", "client"):
        variable = "CLIENT_IMAGE" if name == "client" else "SERVER_IMAGE"
        operator = operators[name]
        assert (
            operator.get("container_image", operator.get("image"))
            == workload["variables"][variable]["value"]
        )
    tasks = {task["name"]: task for task in config["workflow"]["tasks"]}
    for role in ("prefill", "decode"):
        launch = next(
            line
            for line in tasks[f"{role}_server"]["script"]
            if "-m dynamo.vllm" in line
        )
        args = shlex.split(launch)
        assert "--is-prefill-worker" not in args
        assert args[args.index("--disaggregation-mode") + 1] == role
        assert "--kv-transfer-config" in args


@pytest.mark.parametrize(
    "backend,decode_gpus", [("k8s", 2), ("slurm", 4), ("docker", 4)]
)
def test_generated_modular_demo_placement(backend, decode_gpus, tmp_path):
    bundle = tmp_path / "bundle"
    runner = CliRunner()
    generated = runner.invoke(app, ["sample", "modular/dynamo_vllm", "-o", str(bundle)])
    assert generated.exit_code == 0, generated.output

    args = [
        "-f",
        str(bundle / "dynamo_vllm_workload.yaml"),
        "-f",
        str(bundle / f"{backend}.yaml"),
    ]
    result = runner.invoke(
        app, ["run", *args, "--dry-run", "--verbose"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    allocation = result.output.split("── Allocation map", 1)[1].split(
        "── Resource Occupancy", 1
    )[0]
    nodes = allocation.split("├─ Tasks:")
    assert len(nodes) == 3, allocation
    assert "GPU 0: prefill_server_0" in nodes[1], allocation
    assert "GPU 1: prefill_server_1" in nodes[1], allocation
    assert "decode_server_0" not in nodes[1], allocation
    for gpu in range(decode_gpus):
        assert f"GPU {gpu}: decode_server_0" in nodes[2], allocation
    assert "prefill_server_2" not in allocation
    assert "decode_server_1" not in allocation
    assert (
        "depends_on: ['prefill_server_0', 'prefill_server_1', 'decode_server_0']"
        in result.output
    )
