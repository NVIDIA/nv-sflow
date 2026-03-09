# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sflow.app.sflow import SflowApp


def test_visualize_mermaid_includes_nodes_and_edges(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
variables:
  MSG:
    value: hi
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo ${{ variables.MSG }}
    - name: t2
      depends_on: [t1]
      script:
        - echo done
""".lstrip()
    )

    out = tmp_path / "dag.mmd"
    SflowApp().visualize(
        file=Path(p), output_path=out, format="mermaid", show_variables=True
    )

    txt = out.read_text()
    assert "graph TD" in txt
    assert 't1["t1' in txt
    assert 't2["t2' in txt
    assert "t1 -- Completed --> t2" in txt
    assert "%% var MSG" in txt


def test_visualize_default_output_goes_to_output_dir_run_id_folder(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""".lstrip()
    )

    out_dir = tmp_path / "out"
    res = SflowApp().visualize(
        file=Path(p),
        output_path=None,
        format="mermaid",
        output_dir=out_dir,
    )

    assert res.saved_path is not None
    saved = Path(res.saved_path)
    assert saved.name == "wf.mmd"
    assert saved.parent.parent == out_dir


def test_visualize_groups_replicas_using_subgraph(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
workflow:
  name: wf
  tasks:
    - name: t1
      replicas:
        count: 3
        policy: parallel
      script:
        - echo hi
    - name: t2
      depends_on: [t1]
      script:
        - echo done
""".lstrip()
    )

    out = tmp_path / "dag.mmd"
    SflowApp().visualize(file=Path(p), output_path=out, format="mermaid")
    txt = out.read_text()

    # SRD-style grouping: subgraph with concrete replica nodes.
    assert 'subgraph "t1"' in txt
    assert "t1_0" in txt
    assert "t1_1" in txt
    assert "t1_2" in txt
    # Dependency edges come from each replica to t2 (task graph semantics).
    assert "t1_0 -- Completed --> t2" in txt
    assert "t1_1 -- Completed --> t2" in txt
    assert "t1_2 -- Completed --> t2" in txt
