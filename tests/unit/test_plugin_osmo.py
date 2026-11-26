# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from sflow.plugins.backends.osmo import OsmoBackend, OsmoBackendConfig
from sflow.plugins.operators import osmo as osmo_mod
from sflow.plugins.operators import osmo_logs
from sflow.plugins.operators.osmo import (
    OsmoOperator,
    OsmoOperatorConfig,
    exit_code_from_query,
    osmo_name,
    render_spec,
)


def _query(status, exit_code=None, task_status=None):
    task = {"name": "main", "exit_code": exit_code, "status": task_status}
    return {"status": status, "groups": [{"tasks": [task]}]}


STARTED = (0, json.dumps(_query("RUNNING")))


def _script_body(spec):
    task = yaml.safe_load(spec)["workflow"]["tasks"][0]
    assert task["args"] == [task["files"][0]["path"]]
    return base64.b64decode(task["files"][0]["contents"]).decode()


def test_render_spec_is_block_yaml_without_template_markers():
    spec = render_spec(
        workflow_name="sflow-t",
        image="ubuntu:24.04",
        script=["echo '{{ x }} {% y %} {# z #}'"],
        envs={"A": "it's", "my-var": "x"},
        resources={"cpu": 1},
    )
    # OSMO Jinja-renders specs with these markers and splits sections on ^workflow:
    assert not any(m in spec for m in ("{{", "{%", "{#"))
    assert spec.startswith("workflow:\n")
    # Non-identifier env names would break `export`, so they are skipped.
    assert (
        _script_body(spec) == "export A='it'\"'\"'s'\necho '{{ x }} {% y %} {# z #}'\n"
    )


def test_render_spec_creates_sflow_output_dirs():
    spec = render_spec(
        workflow_name="w",
        image="i",
        script=["echo ok > $SFLOW_TASK_OUTPUT_DIR/r.txt"],
        envs={"SFLOW_TASK_OUTPUT_DIR": "/out/t"},
        resources={},
    )
    assert 'mkdir -p "$SFLOW_TASK_OUTPUT_DIR"' in _script_body(spec)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_query("PENDING"), None),
        (_query("RUNNING"), None),
        (_query("COMPLETED", 0), 0),
        (_query("FAILED", 3), 3),
        (_query("FAILED", 0), 1),  # failed after the process exited 0
        (_query("FAILED_CANCELED"), 1),
        (_query("FAILED_IMAGE_PULL"), 1),
    ],
)
def test_exit_code_from_query(result, expected):
    assert exit_code_from_query(result) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sflow-my_task", "sflow-my_task"),
        ("sflow-a.b/c", "sflow-a-b-c"),
        ("1abc", "sflow-1abc"),
    ],
)
def test_osmo_name(raw, expected):
    assert osmo_name(raw) == expected


def _operator(gpus=0):
    op = OsmoOperator(
        OsmoOperatorConfig(name="op", image="ubuntu:24.04", poll_interval=0.01)
    )
    backend = OsmoBackend(OsmoBackendConfig(name="osmo", pool="p1", platform="ovx-l40"))
    op.apply_backend_context(
        backend=backend, assigned_nodes=[], artifacts=[], gpu_count=gpus
    )
    return op


def test_resources_and_submit_args_use_backend_context():
    op = _operator(gpus=2)
    assert op.resources() == {
        "cpu": 1,
        "memory": "4Gi",
        "storage": "10Gi",
        "gpu": 2,
        "platform": "ovx-l40",
    }
    assert op.submit_args("w.yaml") == [
        "workflow",
        "submit",
        "w.yaml",
        "-t",
        "json",
        "--pool",
        "p1",
    ]


def test_operator_rejects_non_osmo_backend():
    op = OsmoOperator(OsmoOperatorConfig(name="op", image="i"))
    local = SimpleNamespace(name="local", config=SimpleNamespace(type="local"))
    with pytest.raises(ValueError, match="only on an 'osmo' backend"):
        op.apply_backend_context(backend=local, assigned_nodes=[], artifacts=[])


def test_operator_rejects_multi_node_tasks():
    op = OsmoOperator(OsmoOperatorConfig(name="op", image="i"))
    backend = OsmoBackend(OsmoBackendConfig(name="osmo"))
    with pytest.raises(ValueError, match="single-node tasks only"):
        op.apply_backend_context(
            backend=backend, assigned_nodes=["n0", "n1"], artifacts=[]
        )


def test_poll_interval_must_be_positive():
    with pytest.raises(ValueError):
        OsmoOperatorConfig(name="op", image="i", poll_interval=0)


@pytest.fixture(autouse=True)
def _reset_label_probe(monkeypatch):
    monkeypatch.setattr(osmo_mod, "_submit_has_label", None)


@pytest.mark.parametrize(
    ("help_text", "label"),
    [("[--priority P] [--label KEY=VALUE]", "sflow-task=t"), ("[--priority P]", None)],
)
def test_submit_adds_label_only_when_cli_supports_it(monkeypatch, help_text, label):
    fake = _FakeOsmo([(0, json.dumps(_query("COMPLETED", 0)))], help=help_text)
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    asyncio.run(_run(_operator(), _Launcher()))
    submit = fake.calls[0]
    assert ("--label" in submit) == bool(label)
    if label:
        assert submit[submit.index("--label") + 1] == label


class _FakeOsmo:
    """Replaces ``_osmo`` with scripted replies per subcommand."""

    def __init__(
        self, queries=(), submit_delay=0.0, submit_reply=None, help="", queued=()
    ):
        self.help = help
        self.calls = []
        # The queue wait queries first: ``queued`` replies, then one that has started.
        self.queries = [*queued, STARTED, *queries]
        self.submit_delay = submit_delay
        self.submit_reply = submit_reply or (0, json.dumps({"name": "sflow-t-1"}))

    async def __call__(self, *args):
        if "--help" in args:  # label-support probe; not a workflow call
            return 0, self.help
        self.calls.append(args)
        if args[1] == "submit":
            await asyncio.sleep(self.submit_delay)
            return self.submit_reply
        if args[1] == "query":
            return self.queries.pop(0)
        return 0, "{}"

    def subcommands(self):
        return [c[1] for c in self.calls]


class _Launcher:
    def __init__(self, block=False, codes=()):
        self.block = block
        self.codes = list(codes)
        self.commands = []

    async def run_async(self, command, **_):
        self.commands.append(command)
        if self.block:
            await asyncio.Event().wait()
        return self.codes.pop(0) if self.codes else 0


def _run(op, launcher):
    return op.execute(
        launcher=launcher,
        output_logger=logging.getLogger("t"),
        env={"K": "v"},
        task_name="t",
        script=["true"],
    )


def test_execute_streams_logs_then_returns_task_exit_code(monkeypatch):
    fake = _FakeOsmo(
        [(0, json.dumps(_query("RUNNING"))), (0, json.dumps(_query("FAILED", 3)))]
    )
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    launcher = _Launcher()
    assert asyncio.run(_run(_operator(), launcher)) == 3
    assert launcher.commands[0][-3:-1] == ["sflow-t-1", "main"]
    assert fake.subcommands() == ["submit", "query", "query", "query"]  # no cancel


def test_execute_waits_in_queue_before_attaching_the_log_stream(monkeypatch, caplog):
    queued = (0, json.dumps(_query("PENDING", task_status="SCHEDULING")))
    fake = _FakeOsmo([(0, json.dumps(_query("COMPLETED", 0)))], queued=[queued] * 3)
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    launcher, notes = _Launcher(), []
    caplog.set_level(logging.INFO)
    code = asyncio.run(
        _operator().execute(
            launcher=launcher,
            output_logger=logging.getLogger("t"),
            env={},
            task_name="t",
            script=["true"],
            status_note=notes.append,
        )
    )
    assert code == 0
    assert len(launcher.commands) == 1  # attached once, after the queue
    assert fake.subcommands() == ["submit", *["query"] * 5]
    assert "OSMO sflow-t-1 queued in p1: SCHEDULING" in notes
    assert caplog.text.count("queued in p1: SCHEDULING") == 1  # logged on change


def test_execute_waits_while_requeued_and_logs_the_state_periodically(
    monkeypatch, caplog
):
    queued = (0, json.dumps(_query("WAITING", task_status="SCHEDULING")))
    fake = _FakeOsmo([(0, json.dumps(_query("COMPLETED", 0)))], queued=[queued] * 3)
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    monkeypatch.setattr(osmo_mod, "_QUEUE_LOG_EVERY_S", 0.0)
    launcher = _Launcher()
    caplog.set_level(logging.INFO)
    assert asyncio.run(_run(_operator(), launcher)) == 0
    assert len(launcher.commands) == 1
    assert caplog.text.count("queued in p1: SCHEDULING") == 3  # one per poll


def test_execute_cancels_when_queries_fail_while_queued(monkeypatch):
    fake = _FakeOsmo(queued=[(1, "boom")] * osmo_mod._MAX_QUERY_FAILURES)
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    launcher = _Launcher()
    assert asyncio.run(_run(_operator(), launcher)) == 1
    assert launcher.commands == []  # never attached
    assert fake.subcommands()[-1] == "cancel"


def test_execute_cancels_when_queries_keep_failing(monkeypatch):
    fake = _FakeOsmo([(1, "boom")] * osmo_mod._MAX_QUERY_FAILURES)
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    assert asyncio.run(_run(_operator(), _Launcher())) == 1
    assert fake.subcommands()[-1] == "cancel"


def test_execute_cancels_the_workflow_on_teardown(monkeypatch):
    fake = _FakeOsmo()
    monkeypatch.setattr(osmo_mod, "_osmo", fake)

    async def scenario():
        task = asyncio.ensure_future(_run(_operator(), _Launcher(block=True)))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert fake.calls[-1][:3] == ("workflow", "cancel", "sflow-t-1")


def test_execute_cancels_a_workflow_created_while_cancelling_submit(monkeypatch):
    fake = _FakeOsmo(submit_delay=0.1)
    monkeypatch.setattr(osmo_mod, "_osmo", fake)

    async def scenario():
        task = asyncio.ensure_future(_run(_operator(), _Launcher()))
        await asyncio.sleep(0.02)  # mid-submit
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert fake.calls[-1][:3] == ("workflow", "cancel", "sflow-t-1")


@pytest.mark.parametrize("reply", [(2, "bad spec"), (0, "not json")])
def test_execute_fails_when_submit_does_not_return_an_id(monkeypatch, reply):
    fake = _FakeOsmo(submit_reply=reply)
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    assert asyncio.run(_run(_operator(), _Launcher())) == (reply[0] or 1)
    assert fake.subcommands() == ["submit"]


def _fake_osmo_logs(tmp_path, lines):
    fake = tmp_path / "osmo"
    fake.write_text(
        "#!/bin/sh\n" + "".join(f"printf '%s\\n' '{line}'\n" for line in lines)
    )
    fake.chmod(0o755)


def _stream(tmp_path):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            osmo_logs.__name__,
            "wf-1",
            "main",
            str(tmp_path / "state"),
        ],
        capture_output=True,
        text=True,
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
    )


HEADER = "Workflow wf-1 has logs:"
OSMO_LINE = "2026/11/17 16:23:45 [main][osmo] All Inputs Gathered"
WORK_LINE = "2026/11/17 16:23:45 [main] TTFT: 12.5 ms"
CUT_MSG = "Log stream has timed out or failed. Please run the command again."


def test_operator_and_log_stream_agree_on_the_cut_code():
    assert osmo_mod._LOG_STREAM_CUT == osmo_logs.CUT
    assert osmo_mod._LOG_STREAM_MODULE == osmo_logs.__name__


def test_log_stream_strips_osmo_header_and_workload_prefix(tmp_path, fake_process):
    # The autouse fake_process blocks real subprocesses; run the module for real.
    fake_process.allow_unregistered(True)
    _fake_osmo_logs(tmp_path, [HEADER, OSMO_LINE, WORK_LINE])
    r = _stream(tmp_path)
    assert r.returncode == 0
    assert r.stdout == f"{OSMO_LINE}\nTTFT: 12.5 ms\n"


def test_log_stream_reattach_skips_lines_already_written(tmp_path, fake_process):
    fake_process.allow_unregistered(True)
    _fake_osmo_logs(tmp_path, [HEADER, OSMO_LINE, "", CUT_MSG])
    first = _stream(tmp_path)
    assert first.returncode == osmo_logs.CUT
    assert first.stdout == f"{OSMO_LINE}\n"  # the CLI's blank line + message dropped
    # OSMO replays the log from the start on the next attach.
    _fake_osmo_logs(tmp_path, [HEADER, OSMO_LINE, "", WORK_LINE])
    second = _stream(tmp_path)
    assert second.returncode == 0
    assert second.stdout == "\nTTFT: 12.5 ms\n"


def test_execute_reattaches_after_a_cut_stream(monkeypatch):
    fake = _FakeOsmo(
        [(0, json.dumps(_query("RUNNING"))), (0, json.dumps(_query("COMPLETED", 0)))]
    )
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    launcher = _Launcher(codes=[osmo_logs.CUT, 0])
    assert asyncio.run(_run(_operator(), launcher)) == 0
    assert len(launcher.commands) == 2
    assert fake.subcommands() == ["submit", "query", "query", "query"]


def test_execute_reattaches_after_a_stream_error(monkeypatch):
    fake = _FakeOsmo(
        [(0, json.dumps(_query("RUNNING"))), (0, json.dumps(_query("COMPLETED", 0)))]
    )
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    launcher = _Launcher(codes=[1, 0])  # e.g. connection reset, then a clean stream
    assert asyncio.run(_run(_operator(), launcher)) == 0
    assert len(launcher.commands) == 2
    assert fake.subcommands() == ["submit", "query", "query", "query"]


def test_execute_fetches_last_lines_when_task_ends_during_a_cut(monkeypatch):
    fake = _FakeOsmo([(0, json.dumps(_query("COMPLETED", 0)))])
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    launcher = _Launcher(codes=[osmo_logs.CUT, 0])
    assert asyncio.run(_run(_operator(), launcher)) == 0
    assert len(launcher.commands) == 2  # cut attach + final attach
    assert fake.subcommands() == ["submit", "query", "query"]


def test_execute_cancels_when_queries_fail_after_cut_streams(monkeypatch):
    fake = _FakeOsmo([(1, "boom")] * osmo_mod._MAX_QUERY_FAILURES)
    monkeypatch.setattr(osmo_mod, "_osmo", fake)
    launcher = _Launcher(codes=[osmo_logs.CUT] * osmo_mod._MAX_QUERY_FAILURES)
    assert asyncio.run(_run(_operator(), launcher)) == 1
    assert fake.subcommands()[-1] == "cancel"


def test_backend_preflight_requires_osmo_cli(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    backend = OsmoBackend(OsmoBackendConfig(name="osmo"))
    with pytest.raises(RuntimeError, match="not on PATH"):
        backend.preflight_validate()


def test_backend_preflight_reports_probe_start_failure(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/osmo")

    def boom(*args, **kwargs):
        raise PermissionError("not executable")

    monkeypatch.setattr("subprocess.run", boom)
    with pytest.raises(RuntimeError, match="could not run 'osmo workflow list'"):
        OsmoBackend(OsmoBackendConfig(name="osmo")).preflight_validate()


def test_osmo_cli_spawn_failure_returns_an_error_code(monkeypatch):
    async def no_spawn(*args, **kwargs):
        raise FileNotFoundError("osmo")

    monkeypatch.setattr(osmo_mod.asyncio, "create_subprocess_exec", no_spawn)
    rc, out = asyncio.run(osmo_mod._osmo("workflow", "query", "wf-1"))
    assert rc == 127
    assert "could not start osmo workflow query" in out


def test_backend_null_pool_falls_back_to_osmo_profile():
    resolver = SimpleNamespace(resolve=lambda value, ctx: None)
    conf = OsmoBackend.resolve_config(
        OsmoBackendConfig(name="osmo", pool="${{ variables.P }}"),
        resolver=resolver,
        ctx={},
        workflow_name="w",
    )
    assert conf.pool is None


def test_backend_default_operator_requires_explicit_osmo_operator():
    with pytest.raises(ValueError, match="explicit 'osmo' operator"):
        OsmoBackend(OsmoBackendConfig(name="osmo")).default_operator(name="t")


def test_log_stream_drops_the_cli_update_banner_but_keeps_other_stderr(
    tmp_path, fake_process
):
    fake_process.allow_unregistered(True)
    banner = [
        "WARNING: New client 6.4.0 available.",
        "Current version: 6.3.1.210ffea0.",
        "Please run the following command:",
        "curl -fsSL https://osmo.example.com/cli/install.sh | bash",
    ]
    fake = tmp_path / "osmo"
    fake.write_text(
        "#!/bin/sh\n"
        + "".join(f"printf '%s\\n' '{line}' >&2\n" for line in banner)
        + "printf '%s\\n' 'real error' >&2\n"
        + f"printf '%s\\n' '{HEADER}' '{WORK_LINE}'\n"
    )
    fake.chmod(0o755)
    run = _stream(tmp_path)
    assert run.returncode == 0
    assert run.stdout == "TTFT: 12.5 ms\n"
    assert run.stderr == "real error\n"
