# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from io import BytesIO
from pathlib import Path

import pytest

from sflow.app.assembly import build_state, resolve_artifacts
from sflow.config.schema import SflowConfig, TaskConfig, VariableConfig, WorkflowConfig
from sflow.core.state import SflowState
from sflow.core.task_graph import TaskGraph
from sflow.core.workflow import Workflow


def _empty_state() -> SflowState:
    return SflowState(workflow=Workflow(name="wf", task_graph=TaskGraph()))


def test_resolve_artifacts_file_scheme_resolves_relative_path_against_workspace(
    tmp_path: Path,
):
    state = _empty_state()
    cfg = SflowConfig(
        version="0.1",
        artifacts=[{"name": "A", "uri": "file://data.txt"}],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )

    out_dir = tmp_path / "output"
    state = resolve_artifacts(cfg, state, workspace_dir=tmp_path, output_dir=out_dir, materialize=False)
    a = state.artifacts["A"]
    assert a.path is not None
    assert a.path == tmp_path / "data.txt"


def test_resolve_artifacts_fs_remote_filesystem_passes_through_missing_path(
    tmp_path: Path,
):
    # Off-host backends (e.g. Kubernetes): a missing fs:// path is a remote path and
    # must NOT be created/validated on the controller, even when materializing.
    missing = tmp_path / "remote" / "model"
    cfg = SflowConfig(
        version="0.1",
        artifacts=[{"name": "MODEL", "uri": f"fs://{missing}"}],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )
    state = resolve_artifacts(
        cfg,
        _empty_state(),
        workspace_dir=tmp_path,
        output_dir=tmp_path / "out",
        materialize=True,
        remote_filesystem=True,
    )
    assert state.artifacts["MODEL"].path == missing
    assert not missing.exists()


def test_resolve_artifacts_fs_local_creates_missing_path(tmp_path: Path):
    # Host-FS backends keep the convenience of creating a missing fs:// output dir.
    missing = tmp_path / "local" / "out"
    cfg = SflowConfig(
        version="0.1",
        artifacts=[{"name": "OUT", "uri": f"fs://{missing}"}],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )
    resolve_artifacts(
        cfg,
        _empty_state(),
        workspace_dir=tmp_path,
        output_dir=tmp_path / "out",
        materialize=True,
        remote_filesystem=False,
    )
    assert missing.exists()


def test_backends_execute_offhost_detects_offhost_backends():
    from types import SimpleNamespace

    from sflow.app.run_support import backends_execute_offhost

    def _be(supports_host_path_mounts: bool):
        return SimpleNamespace(
            capabilities=SimpleNamespace(
                supports_host_path_mounts=supports_host_path_mounts
            )
        )

    state = _empty_state()
    state.backends = {"local": _be(True)}
    assert backends_execute_offhost(state) is False
    state.backends = {"local": _be(True), "k8s": _be(False)}
    assert backends_execute_offhost(state) is True


def test_resolve_artifacts_inline_file_content_materializes_only_when_enabled(
    tmp_path: Path,
):
    state = _empty_state()
    cfg = SflowConfig(
        version="0.1",
        artifacts=[{"name": "INLINE", "uri": "file://inline.txt", "content": "hello"}],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )

    out_dir = tmp_path / "output"
    # Planning / dry-run path should not write files.
    state = resolve_artifacts(cfg, state, workspace_dir=tmp_path, output_dir=out_dir, materialize=False)
    p = state.artifacts["INLINE"].path
    assert p == out_dir / "inline.txt"
    assert p.exists() is False

    # Materialize should write the content under the output dir.
    state = resolve_artifacts(cfg, state, workspace_dir=tmp_path, output_dir=out_dir, materialize=True)
    assert (out_dir / "inline.txt").read_text() == "hello"

    # Second resolve with different content should overwrite (output dir is per-run).
    cfg2 = SflowConfig(
        version="0.1",
        artifacts=[
            {"name": "INLINE", "uri": "file://inline.txt", "content": "different"}
        ],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )
    out_dir2 = tmp_path / "output2"
    state2 = resolve_artifacts(cfg2, _empty_state(), workspace_dir=tmp_path, output_dir=out_dir2, materialize=True)
    p2 = state2.artifacts["INLINE"].path
    assert p2 is not None
    assert p2 == out_dir2 / "inline.txt"
    assert p2.read_text() == "different"


def test_resolve_artifacts_inline_file_content_remote_filesystem_writes_controller_copy(
    tmp_path: Path,
):
    # Off-host backends (e.g. Kubernetes) inject the content natively into the pod
    # (ConfigMap), but a RELATIVE file:// copy is STILL written to the workflow output
    # folder on the controller -- parity with the slurm/local backends -- so the
    # generated file is inspectable on the host.
    state = _empty_state()
    cfg = SflowConfig(
        version="0.1",
        artifacts=[{"name": "INLINE", "uri": "file://inline.txt", "content": "hello"}],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )

    out_dir = tmp_path / "output"
    state = resolve_artifacts(
        cfg,
        state,
        workspace_dir=tmp_path,
        output_dir=out_dir,
        materialize=True,
        remote_filesystem=True,
    )
    art = state.artifacts["INLINE"]
    # Path is resolved under the output dir (scripts reference it; the pod mounts the
    # file there) AND a controller-side copy is written with the same content.
    assert art.path == out_dir / "inline.txt"
    assert (out_dir / "inline.txt").read_text() == "hello"
    # Inline content is still carried for K8s-native (ConfigMap) injection.
    assert art.content == "hello"


def test_resolve_artifacts_inline_file_content_remote_filesystem_absolute_not_written(
    tmp_path: Path,
):
    # An ABSOLUTE file:// on an off-host backend is an in-pod path -- it must NOT be
    # created on the controller; only native injection carries the content.
    state = _empty_state()
    abs_path = tmp_path / "pod_only" / "inline.txt"
    cfg = SflowConfig(
        version="0.1",
        artifacts=[
            {"name": "INLINE", "uri": f"file://{abs_path}", "content": "hello"}
        ],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )

    out_dir = tmp_path / "output"
    state = resolve_artifacts(
        cfg,
        state,
        workspace_dir=tmp_path,
        output_dir=out_dir,
        materialize=True,
        remote_filesystem=True,
    )
    art = state.artifacts["INLINE"]
    assert art.path == abs_path
    assert not abs_path.exists()  # absolute in-pod path not created on the controller
    assert art.content == "hello"


def test_resolve_artifacts_http_downloads_into_workspace_cache(
    tmp_path: Path, monkeypatch
):
    import sflow.plugins.artifacts.http as http_mod

    calls: list[str] = []

    class _Resp:
        def __init__(self, data: bytes):
            self._bio = BytesIO(data)

        def read(self) -> bytes:
            return self._bio.read()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(url: str):
        calls.append(url)
        return _Resp(b"payload")

    monkeypatch.setattr(http_mod, "urlopen", _fake_urlopen)

    state = _empty_state()
    cfg = SflowConfig(
        version="0.1",
        artifacts=[{"name": "H", "uri": "https://example.com/data.bin"}],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )

    out_dir = tmp_path / "output"
    state = resolve_artifacts(cfg, state, workspace_dir=tmp_path, output_dir=out_dir, materialize=True)
    p = state.artifacts["H"].path
    assert p is not None
    assert p.exists() is True
    assert p.read_bytes() == b"payload"
    assert calls == ["https://example.com/data.bin"]

    # Second call should hit cache and not call urlopen again.
    state = resolve_artifacts(
        cfg, _empty_state(), workspace_dir=tmp_path, output_dir=out_dir, materialize=True
    )
    assert state.artifacts["H"].path == p
    assert calls == ["https://example.com/data.bin"]


def test_build_state_injects_artifact_paths_into_task_envs(tmp_path: Path):
    # Create a file inside workspace.
    (tmp_path / "model.bin").write_text("x")

    cfg = SflowConfig(
        version="0.1",
        artifacts=[{"name": "MODEL", "uri": "file://model.bin"}],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo ${MODEL}"])]
        ),
    )

    state = asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))
    t1 = state.workflow.task_graph.get_task("t1")
    assert t1.envs["MODEL"] == str(tmp_path / "model.bin")


def test_workflow_variables_can_reference_artifacts_ctx(tmp_path: Path):
    cfg = SflowConfig(
        version="0.1",
        artifacts=[{"name": "A", "uri": "file://data.txt"}],
        workflow=WorkflowConfig(
            name="wf",
            variables=[VariableConfig(name="P", value="${{ artifacts.A.path }}")],
            tasks=[TaskConfig(name="t1", script=["echo ${{ variables.P }}"])],
        ),
    )

    state = asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))
    assert state.variables["P"].value == str(tmp_path / "data.txt")


def test_global_variables_can_defer_artifacts_ctx(tmp_path: Path):
    cfg = SflowConfig(
        version="0.1",
        variables=[
            {
                "name": "P",
                "value": "${{ artifacts.A.path }}",
                "type": "string",
            }
        ],
        artifacts=[{"name": "A", "uri": "file://data.txt"}],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[
                TaskConfig(
                    name="t1",
                    script=[
                        "echo jinja=${{ variables.P }}",
                        "echo shell=${P}",
                    ],
                )
            ],
        ),
    )

    state = asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))
    t1 = state.workflow.task_graph.get_task("t1")

    assert state.variables["P"].value == str(tmp_path / "data.txt")
    assert t1.script[0] == f"echo jinja={tmp_path / 'data.txt'}"
    assert t1.envs["P"] == str(tmp_path / "data.txt")


def test_artifacts_can_reference_backend_deferred_global_variable(tmp_path: Path):
    cfg = SflowConfig(
        version="0.1",
        variables=[
            {
                "name": "HEAD_NODE_IP",
                "value": "${{ backends.slurm_cluster.nodes[0].ip_address }}",
                "type": "string",
            }
        ],
        backends=[
            {
                "name": "slurm_cluster",
                "type": "slurm",
                "default": True,
                "account": "acct",
                "partition": "batch",
                "time": "00:10:00",
                "nodes": 2,
                "gpus_per_node": 4,
            }
        ],
        artifacts=[
            {
                "name": "SERVER_CONFIG",
                "uri": "file://server_config.yaml",
                "content": "hostname: ${{ variables.HEAD_NODE_IP }}",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    state = asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))

    assert state.variables["HEAD_NODE_IP"].value == "0.0.0.1"


def test_artifacts_reject_artifact_deferred_global_variable_reference(tmp_path: Path):
    cfg = SflowConfig(
        version="0.1",
        variables=[
            {
                "name": "ARTIFACT_PATH",
                "value": "${{ artifacts.A.path }}",
                "type": "string",
            }
        ],
        artifacts=[
            {
                "name": "A",
                "uri": "file://${{ variables.ARTIFACT_PATH }}",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))

    message = str(exc_info.value)
    assert "Deferred global variable 'ARTIFACT_PATH'" in message
    assert "artifact definition 'artifacts.A'" in message
    assert "cannot be used while resolving artifacts" in message


def test_artifacts_reject_get_alias_artifact_deferred_global_variable_reference(
    tmp_path: Path,
):
    cfg = SflowConfig(
        version="0.1",
        variables=[
            {
                "name": "ARTIFACT_PATH",
                "value": "${{ artifacts.A.path }}",
                "type": "string",
            },
            {
                "name": "ARTIFACT_ALIAS",
                "value": "${{ variables.get('ARTIFACT_PATH') }}",
                "type": "string",
            },
        ],
        artifacts=[
            {
                "name": "A",
                "uri": "file://${{ variables.ARTIFACT_ALIAS }}",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))

    message = str(exc_info.value)
    assert "Deferred global variable 'ARTIFACT_ALIAS'" in message
    assert "artifact definition 'artifacts.A'" in message


def test_global_variables_defer_artifact_root_until_artifacts_exist(tmp_path: Path):
    cfg = SflowConfig(
        version="0.1",
        variables=[
            {
                "name": "ARTIFACT_COUNT",
                "value": "${{ artifacts | length }}",
                "type": "integer",
            }
        ],
        artifacts=[{"name": "A", "uri": "file://data.txt"}],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo ${ARTIFACT_COUNT}"])],
        ),
    )

    state = asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))

    assert state.variables["ARTIFACT_COUNT"].value == 1


def test_global_variables_can_defer_mixed_backend_and_artifact_contexts(
    tmp_path: Path,
):
    cfg = SflowConfig(
        version="0.1",
        variables=[
            {
                "name": "ENDPOINT",
                "value": "${{ backends.slurm_cluster.nodes[0].ip_address }}:${{ artifacts.A.path }}",
                "type": "string",
            }
        ],
        backends=[
            {
                "name": "slurm_cluster",
                "type": "slurm",
                "default": True,
                "account": "acct",
                "partition": "batch",
                "time": "00:10:00",
                "nodes": 2,
                "gpus_per_node": 4,
            }
        ],
        artifacts=[{"name": "A", "uri": "file://data.txt"}],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo ${{ variables.ENDPOINT }}"])],
        ),
    )

    state = asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))
    t1 = state.workflow.task_graph.get_task("t1")
    expected = f"0.0.0.1:{tmp_path / 'data.txt'}"

    assert state.variables["ENDPOINT"].value == expected
    assert t1.script[0] == f"echo {expected}"


def test_resolve_artifacts_with_backend_expression_in_content(tmp_path: Path):
    """
    Artifact content can reference backend node info after allocation.
    This tests that ${{ backends.<name>.nodes[0].ip_address }} is resolved.
    """
    cfg = SflowConfig(
        version="0.1",
        backends=[
            {
                "name": "slurm_cluster",
                "type": "slurm",
                "default": True,
                "account": "acct",
                "partition": "batch",
                "time": "00:10:00",
                "nodes": 2,
                "gpus_per_node": 4,
            }
        ],
        artifacts=[
            {
                "name": "SERVER_CONFIG",
                "uri": "file://server_config.yaml",
                "content": "hostname: ${{ backends.slurm_cluster.nodes[0].ip_address }}\nport: 8000",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    # With allocate=False, placeholder IPs are used (0.0.0.1 for first node).
    state = asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))
    a = state.artifacts["SERVER_CONFIG"]
    assert a.path is not None

    # The content should be resolved but file not written (materialize=False when allocate=False).
    # Re-resolve with materialize=True to write and check.
    from sflow.app.assembly import resolve_artifacts

    out_dir = tmp_path / "output"
    state = resolve_artifacts(cfg, state, workspace_dir=tmp_path, output_dir=out_dir, materialize=True)
    p = state.artifacts["SERVER_CONFIG"].path
    assert p is not None
    content = p.read_text()
    assert "hostname: 0.0.0.1" in content
    assert "port: 8000" in content


def test_resolve_artifacts_with_backend_expression_in_uri(tmp_path: Path):
    """
    Artifact URI can reference backend node info after allocation.
    This tests that ${{ backends.<name>.nodes[0].ip_address }} is resolved in URIs.
    """
    cfg = SflowConfig(
        version="0.1",
        backends=[
            {
                "name": "slurm_cluster",
                "type": "slurm",
                "default": True,
                "account": "acct",
                "partition": "batch",
                "time": "00:10:00",
                "nodes": 2,
                "gpus_per_node": 4,
            }
        ],
        artifacts=[
            {
                "name": "REMOTE_MODEL",
                "uri": "http://${{ backends.slurm_cluster.nodes[0].ip_address }}:8000/model.bin",
            }
        ],
        workflow=WorkflowConfig(
            name="wf",
            tasks=[TaskConfig(name="t1", script=["echo hi"])],
        ),
    )

    # With allocate=False, placeholder IPs are used (0.0.0.1 for first node).
    state = asyncio.run(build_state(cfg, allocate=False, workspace_dir=tmp_path))
    a = state.artifacts["REMOTE_MODEL"]
    # The URI should have the placeholder IP resolved.
    assert a.uri == "http://0.0.0.1:8000/model.bin"


# ---------------------------------------------------------------------------
# Preflight artifact path validation tests
# ---------------------------------------------------------------------------


def test_preflight_fs_artifact_with_existing_path_passes(tmp_path: Path):
    """fs:// artifact pointing to an existing path should pass dry-run."""
    from sflow.app.sflow import SflowApp

    model_dir = tmp_path / "models" / "test-model"
    model_dir.mkdir(parents=True)

    wf = tmp_path / "wf.yaml"
    wf.write_text(
        f'version: "0.1"\n'
        f"artifacts:\n"
        f"  - name: MODEL\n"
        f"    uri: fs://{model_dir}\n"
        f"workflow:\n"
        f"  name: wf\n"
        f"  tasks:\n"
        f"    - name: t1\n"
        f"      script:\n"
        f"        - echo hi\n"
    )
    result = SflowApp().run(file=wf, dry_run=True)
    assert result is None


def test_preflight_fs_artifact_with_missing_path_warns_on_dry_run(tmp_path: Path):
    """fs:// artifact pointing to a non-existent path should warn (not fail) during dry-run."""
    from sflow.app.sflow import SflowApp

    wf = tmp_path / "wf.yaml"
    wf.write_text(
        'version: "0.1"\n'
        "artifacts:\n"
        "  - name: MODEL\n"
        "    uri: fs:///nonexistent/path/to/model\n"
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo hi\n"
    )
    result = SflowApp().run(file=wf, dry_run=True)
    assert result is None


def test_preflight_fs_artifact_with_variable_expression_resolved(tmp_path: Path):
    """fs:// artifact URI using ${{ variables.X }} should resolve and validate the path."""
    from sflow.app.sflow import SflowApp

    model_dir = tmp_path / "models" / "test-model"
    model_dir.mkdir(parents=True)

    wf = tmp_path / "wf.yaml"
    wf.write_text(
        f'version: "0.1"\n'
        f"variables:\n"
        f"  - name: MODEL_DIR\n"
        f"    value: {model_dir}\n"
        f"artifacts:\n"
        f"  - name: MODEL\n"
        f'    uri: "fs://${{{{ variables.MODEL_DIR }}}}"\n'
        f"workflow:\n"
        f"  name: wf\n"
        f"  tasks:\n"
        f"    - name: t1\n"
        f"      script:\n"
        f"        - echo hi\n"
    )
    result = SflowApp().run(file=wf, dry_run=True)
    assert result is None


def test_preflight_fs_artifact_with_variable_expression_missing_path_warns_on_dry_run(tmp_path: Path):
    """fs:// artifact URI resolved from variable to a missing path should warn during dry-run."""
    from sflow.app.sflow import SflowApp

    wf = tmp_path / "wf.yaml"
    wf.write_text(
        'version: "0.1"\n'
        "variables:\n"
        "  - name: MODEL_DIR\n"
        "    value: /nonexistent/variable/path\n"
        "artifacts:\n"
        "  - name: MODEL\n"
        '    uri: "fs://${{ variables.MODEL_DIR }}"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo hi\n"
    )
    result = SflowApp().run(file=wf, dry_run=True)
    assert result is None


def test_preflight_fs_artifact_with_unresolvable_expression_skipped(tmp_path: Path):
    """fs:// artifact URI with unresolvable expression should pass preflight
    (the expression is skipped in path validation). The downstream resolver
    will raise on the undefined variable, which is expected."""
    from sflow.app.sflow import SflowApp

    wf = tmp_path / "wf.yaml"
    wf.write_text(
        'version: "0.1"\n'
        "artifacts:\n"
        "  - name: MODEL\n"
        '    uri: "fs://${{ variables.UNDEFINED_VAR }}"\n'
        "workflow:\n"
        "  name: wf\n"
        "  tasks:\n"
        "    - name: t1\n"
        "      script:\n"
        "        - echo hi\n"
    )
    with pytest.raises(ValueError, match="Undefined variable"):
        SflowApp().run(file=wf, dry_run=True)


# ---------------------------------------------------------------------------
# inline-content artifacts must not escape the run's output directory
# ---------------------------------------------------------------------------


def _inline_cfg(uri: str) -> SflowConfig:
    return SflowConfig(
        version="0.1",
        artifacts=[{"name": "gen", "uri": uri, "content": "payload\n"}],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )


@pytest.mark.parametrize(
    "uri",
    [
        "file://../escape.txt",
        "file://../../escape.txt",
        "file://sub/../../escape.txt",
        "file://./../escape.txt",
    ],
)
def test_inline_content_cannot_escape_the_output_dir(tmp_path: Path, uri: str):
    """A relative file:// URI with `..` must be rejected, not written outside.

    ``output_dir / raw`` does not collapse `..`, so these previously resolved outside the
    run directory and ``path.write_text(content)`` wrote there -- an arbitrary-file-write
    primitive for a workflow from an untrusted source, and a correctness bug regardless:
    files that escape the run dir are never cleaned up and can clobber the user's own.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="escapes the workflow output directory"):
        resolve_artifacts(
            _inline_cfg(uri),
            _empty_state(),
            workspace_dir=tmp_path,
            output_dir=out_dir,
            materialize=True,
        )
    assert not (tmp_path / "escape.txt").exists()
    assert not list(tmp_path.glob("**/escape.txt"))


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("file://gen.txt", "gen.txt"),
        ("file://scripts/helper.sh", "scripts/helper.sh"),
        # `..` that stays INSIDE must still normalize and be allowed -- the guard is a
        # containment check, not a blanket ban on the characters.
        ("file://a/../b/keep.txt", "b/keep.txt"),
    ],
)
def test_inline_content_inside_the_output_dir_still_works(
    tmp_path: Path, uri: str, expected: str
):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    state = resolve_artifacts(
        _inline_cfg(uri),
        _empty_state(),
        workspace_dir=tmp_path,
        output_dir=out_dir,
        materialize=True,
    )
    written = out_dir / expected
    assert written.read_text() == "payload\n"
    assert state.artifacts["gen"].path == written.resolve()


def test_fs_uri_outside_the_workspace_is_still_allowed(tmp_path: Path):
    """The containment check must NOT be generalised to artifact resolution at large.

    Every production recipe mounts its model from an absolute path far outside any
    workspace (e.g. ``fs:///mnt/lustre01/models/deepseek-r1-...``). Confining artifact
    resolution to workspace_dir -- the remediation an external report suggested -- would
    break all of them, which is why the guard is scoped to generated inline content only.
    """
    model = tmp_path / "elsewhere" / "model"
    model.mkdir(parents=True)
    cfg = SflowConfig(
        version="0.1",
        artifacts=[{"name": "m", "uri": f"fs://{model}"}],
        workflow=WorkflowConfig(
            name="wf", tasks=[TaskConfig(name="t1", script=["echo hi"])]
        ),
    )
    state = resolve_artifacts(
        cfg,
        _empty_state(),
        workspace_dir=tmp_path / "ws",
        output_dir=tmp_path / "out",
        materialize=True,
    )
    assert state.artifacts["m"].path == model


def test_traversal_is_rejected_before_anything_is_written(tmp_path: Path):
    """Rejection must happen instead of the write, not after it."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("original\n")
    with pytest.raises(ValueError):
        resolve_artifacts(
            _inline_cfg("file://../victim.txt"),
            _empty_state(),
            workspace_dir=tmp_path,
            output_dir=out_dir,
            materialize=True,
        )
    assert victim.read_text() == "original\n", "the existing file was overwritten"
