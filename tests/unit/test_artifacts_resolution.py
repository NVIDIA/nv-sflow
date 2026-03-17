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


def test_preflight_fs_artifact_with_missing_path_fails(tmp_path: Path):
    """fs:// artifact pointing to a non-existent path should fail dry-run."""
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
    with pytest.raises(ValueError, match="does not exist"):
        SflowApp().run(file=wf, dry_run=True)


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


def test_preflight_fs_artifact_with_variable_expression_missing_path_fails(tmp_path: Path):
    """fs:// artifact URI resolved from variable to a missing path should fail."""
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
    with pytest.raises(ValueError, match="does not exist"):
        SflowApp().run(file=wf, dry_run=True)


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
