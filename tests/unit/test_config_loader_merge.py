# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-YAML merge (load_configs / merge_config_dicts)."""

import pytest

from sflow.config.loader import ConfigLoader, merge_config_dicts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vars_to_map(config) -> dict[str, object]:
    if not config.variables:
        return {}
    return {v.name: v.value for v in config.variables}


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text.lstrip())
    return p


# ---------------------------------------------------------------------------
# merge_config_dicts unit tests
# ---------------------------------------------------------------------------

class TestMergeConfigDicts:
    def test_single_dict_returned_unchanged(self):
        d = {"version": "0.1", "workflow": {"name": "wf", "tasks": [{"name": "t1", "script": ["echo"]}]}}
        assert merge_config_dicts([d]) is d

    _WF = {"workflow": {"name": "wf", "tasks": [{"name": "t", "script": ["echo"]}]}}

    def test_merge_variables_by_name(self):
        a = {"version": "0.1", "variables": {"X": {"value": 1}}, **self._WF}
        b = {"variables": {"Y": {"value": 2}}}
        merged = merge_config_dicts([a, b])
        assert merged["variables"]["X"]["value"] == 1
        assert merged["variables"]["Y"]["value"] == 2

    def test_later_variable_overrides_earlier(self):
        a = {"version": "0.1", "variables": {"X": {"value": 1}}, **self._WF}
        b = {"variables": {"X": {"value": 99}}}
        merged = merge_config_dicts([a, b])
        assert merged["variables"]["X"]["value"] == 99

    def test_merge_variables_list_and_dict_formats(self):
        a = {"version": "0.1", "variables": [{"name": "A", "value": 10}], **self._WF}
        b = {"variables": {"B": {"value": 20}}}
        merged = merge_config_dicts([a, b])
        assert merged["variables"]["A"]["value"] == 10
        assert merged["variables"]["B"]["value"] == 20

    def test_merge_artifacts(self):
        a = {"version": "0.1", "artifacts": [{"name": "M", "uri": "fs:///old"}], **self._WF}
        b = {"artifacts": [{"name": "N", "uri": "fs:///new"}]}
        merged = merge_config_dicts([a, b])
        assert "M" in merged["artifacts"]
        assert "N" in merged["artifacts"]

    def test_merge_backends_operators(self):
        a = {
            "version": "0.1",
            "backends": [{"name": "b1", "type": "slurm", "default": True}],
            "operators": [{"name": "op1", "type": "srun"}],
            **self._WF,
        }
        b = {
            "backends": [{"name": "b2", "type": "slurm"}],
            "operators": [{"name": "op2", "type": "srun"}],
        }
        merged = merge_config_dicts([a, b])
        assert "b1" in merged["backends"]
        assert "b2" in merged["backends"]
        assert "op1" in merged["operators"]
        assert "op2" in merged["operators"]

    def test_workflow_tasks_concatenated(self):
        a = {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t1", "script": ["echo t1"]}],
            },
        }
        b = {
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t2", "script": ["echo t2"]}],
            },
        }
        merged = merge_config_dicts([a, b])
        names = [t["name"] for t in merged["workflow"]["tasks"]]
        assert names == ["t1", "t2"]

    def test_workflow_name_last_wins(self):
        """The same-workflow-name restriction is gone; last non-null name wins."""
        a = {"version": "0.1", "workflow": {"name": "wf1", "tasks": [{"name": "t", "script": ["echo"]}]}}
        b = {"workflow": {"name": "wf2", "tasks": [{"name": "t2", "script": ["echo"]}]}}
        warns: list[str] = []
        merged = merge_config_dicts([a, b], source_labels=["a.yaml", "b.yaml"], override_warnings=warns)
        assert merged["workflow"]["name"] == "wf2"
        assert any("workflow.name" in w for w in warns)

    def test_workflow_name_from_fragment_without_name(self):
        """A fragment omitting the name inherits the earlier file's name."""
        a = {"version": "0.1", "workflow": {"name": "wf1", "tasks": [{"name": "t", "script": ["echo"]}]}}
        b = {"workflow": {"tasks": [{"name": "t2", "script": ["echo"]}]}}
        merged = merge_config_dicts([a, b])
        assert merged["workflow"]["name"] == "wf1"

    def test_version_must_be_consistent(self):
        a = {"version": "0.1"}
        b = {"version": "0.2"}
        with pytest.raises(ValueError, match="Version conflict"):
            merge_config_dicts([a, b])

    def test_missing_workflow_raises(self):
        a = {"version": "0.1", "variables": {"X": {"value": 1}}}
        b = {"variables": {"Y": {"value": 2}}}
        with pytest.raises(ValueError, match="No 'workflow' section"):
            merge_config_dicts([a, b])

    def test_missing_version_is_allowed(self):
        """`version` is optional; the schema defaults it to "0.1".

        Behavior change: this previously raised "No 'version' field found in any
        input file". Only "0.1" has ever existed, so requiring it bought nothing.
        A conflict between files is still an error -- see
        ``test_version_must_be_consistent``.
        """
        a = {"workflow": {"name": "wf", "tasks": [{"name": "t", "script": ["echo"]}]}}
        b = {"variables": {"X": {"value": 1}}}

        merged = merge_config_dicts([a, b])

        assert "version" not in merged
        assert merged["workflow"]["name"] == "wf"

    def test_missing_tasks_raises(self):
        a = {"version": "0.1", "workflow": {"name": "wf"}}
        b = {"variables": {"X": {"value": 1}}}
        with pytest.raises(ValueError, match="No tasks found"):
            merge_config_dicts([a, b])

    def test_workflow_timeout_last_wins(self):
        a = {"version": "0.1", "workflow": {"name": "wf", "timeout": "30m", "tasks": [{"name": "t", "script": ["echo"]}]}}
        b = {"workflow": {"name": "wf", "timeout": "60m"}}
        merged = merge_config_dicts([a, b])
        assert merged["workflow"]["timeout"] == "60m"

    def test_workflow_variables_merged(self):
        a = {"version": "0.1", "workflow": {"name": "wf", "variables": {"WV1": {"value": "a"}}, "tasks": [{"name": "t", "script": ["echo"]}]}}
        b = {"workflow": {"name": "wf", "variables": {"WV2": {"value": "b"}}}}
        merged = merge_config_dicts([a, b])
        assert "WV1" in merged["workflow"]["variables"]
        assert "WV2" in merged["workflow"]["variables"]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No configuration data"):
            merge_config_dicts([])

    def test_three_way_merge(self):
        a = {"version": "0.1", "variables": {"A": {"value": 1}}}
        b = {"variables": {"B": {"value": 2}}, "workflow": {"name": "wf", "tasks": [{"name": "t1", "script": ["echo t1"]}]}}
        c = {"variables": {"C": {"value": 3}}, "workflow": {"name": "wf", "tasks": [{"name": "t2", "script": ["echo t2"]}]}}
        merged = merge_config_dicts([a, b, c])
        assert merged["variables"]["A"]["value"] == 1
        assert merged["variables"]["B"]["value"] == 2
        assert merged["variables"]["C"]["value"] == 3
        task_names = [t["name"] for t in merged["workflow"]["tasks"]]
        assert task_names == ["t1", "t2"]

    def test_version_only_in_first_file_ok(self):
        a = {"version": "0.1", "workflow": {"name": "wf", "tasks": [{"name": "t", "script": ["echo"]}]}}
        b = {"variables": {"X": {"value": 1}}}
        merged = merge_config_dicts([a, b])
        assert merged["version"] == "0.1"

    def test_override_warnings_variable_value_changed(self):
        a = {"version": "0.1", "variables": {"X": {"value": 1}}, **self._WF}
        b = {"variables": {"X": {"value": 99}}}
        warns: list[str] = []
        merge_config_dicts([a, b], source_labels=["a.yaml", "b.yaml"], override_warnings=warns)
        assert len(warns) == 1
        assert "variables.X" in warns[0]
        assert "overridden by b.yaml" in warns[0]
        assert "previously from a.yaml" in warns[0]
        assert "1" in warns[0] and "99" in warns[0]

    def test_override_warnings_variable_same_value_is_silent(self):
        """Identical redefinition is not a conflict -> no warning (deep merge)."""
        a = {"version": "0.1", "variables": {"X": {"value": 1}}, **self._WF}
        b = {"variables": {"X": {"value": 1}}}
        warns: list[str] = []
        merge_config_dicts([a, b], source_labels=["a.yaml", "b.yaml"], override_warnings=warns)
        assert warns == []

    def test_override_warnings_timeout(self):
        a = {"version": "0.1", "workflow": {"name": "wf", "timeout": "30m", "tasks": [{"name": "t", "script": ["echo"]}]}}
        b = {"workflow": {"name": "wf", "timeout": "60m"}}
        warns: list[str] = []
        merge_config_dicts([a, b], source_labels=["a.yaml", "b.yaml"], override_warnings=warns)
        assert any("workflow.timeout" in w and "30m" in w and "60m" in w for w in warns)

    def test_override_warnings_workflow_variable(self):
        a = {"version": "0.1", "workflow": {"name": "wf", "variables": {"WV": {"value": "old"}}, "tasks": [{"name": "t", "script": ["echo"]}]}}
        b = {"workflow": {"name": "wf", "variables": {"WV": {"value": "new"}}}}
        warns: list[str] = []
        merge_config_dicts([a, b], source_labels=["a.yaml", "b.yaml"], override_warnings=warns)
        assert any("workflow.variables.WV" in w and "'old'" in w and "'new'" in w for w in warns)

    def test_no_warnings_when_no_conflicts(self):
        a = {"version": "0.1", "variables": {"X": {"value": 1}}, **self._WF}
        b = {"variables": {"Y": {"value": 2}}}
        warns: list[str] = []
        merge_config_dicts([a, b], source_labels=["a.yaml", "b.yaml"], override_warnings=warns)
        assert warns == []

    def test_override_warnings_artifacts(self):
        a = {"version": "0.1", "artifacts": [{"name": "M", "uri": "fs:///old"}], **self._WF}
        b = {"artifacts": [{"name": "M", "uri": "fs:///new"}]}
        warns: list[str] = []
        merge_config_dicts([a, b], source_labels=["a.yaml", "b.yaml"], override_warnings=warns)
        assert len(warns) == 1
        assert "artifacts.M" in warns[0]
        assert "fs:///old" in warns[0] and "fs:///new" in warns[0]


# ---------------------------------------------------------------------------
# ConfigLoader.load_configs integration tests
# ---------------------------------------------------------------------------

class TestLoadConfigs:
    def test_single_file(self, tmp_path):
        p = _write(tmp_path, "sflow.yaml", """
version: "0.1"
variables:
  X:
    value: 1
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""")
        config = ConfigLoader().load_configs([p])
        assert config.workflow.name == "wf"
        assert _vars_to_map(config)["X"] == 1

    def test_two_files_merged(self, tmp_path):
        base = _write(tmp_path, "base.yaml", """
version: "0.1"
variables:
  X:
    value: 1
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo t1
""")
        extra = _write(tmp_path, "extra.yaml", """
version: "0.1"
variables:
  Y:
    value: 2
workflow:
  name: wf
  tasks:
    - name: t2
      script:
        - echo t2
""")
        config = ConfigLoader().load_configs([base, extra])
        assert config.workflow.name == "wf"
        var_map = _vars_to_map(config)
        assert var_map["X"] == 1
        assert var_map["Y"] == 2
        task_names = [t.name for t in config.workflow.tasks]
        assert task_names == ["t1", "t2"]

    def test_variable_overrides_on_merged(self, tmp_path):
        a = _write(tmp_path, "a.yaml", """
version: "0.1"
variables:
  X:
    value: 1
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo t1
""")
        b = _write(tmp_path, "b.yaml", """
variables:
  Y:
    value: 2
workflow:
  name: wf
  tasks:
    - name: t2
      script:
        - echo t2
""")
        config = ConfigLoader().load_configs([a, b], variable_overrides=["X=99", "Y=88"])
        var_map = _vars_to_map(config)
        assert var_map["X"] == 99
        assert var_map["Y"] == 88

    def test_file_not_found_raises(self, tmp_path):
        exists = _write(tmp_path, "a.yaml", """
version: "0.1"
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""")
        missing = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="missing.yaml"):
            ConfigLoader().load_configs([exists, missing])

    def test_empty_file_raises(self, tmp_path):
        a = _write(tmp_path, "a.yaml", """
version: "0.1"
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""")
        b = _write(tmp_path, "b.yaml", "")
        with pytest.raises(ValueError, match="empty"):
            ConfigLoader().load_configs([a, b])

    def test_empty_paths_raises(self):
        with pytest.raises(ValueError, match="No configuration file"):
            ConfigLoader().load_configs([])

    def test_same_task_name_across_files_scatter_merges(self, tmp_path):
        """Same task name across files is now deep-merged (scattered definition):
        the script comes from one file, the operator from another."""
        a = _write(tmp_path, "a.yaml", """
version: "0.1"
operators:
  op1:
    type: srun
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo a
""")
        b = _write(tmp_path, "b.yaml", """
workflow:
  name: wf
  tasks:
    - name: t1
      operator: op1
""")
        config = ConfigLoader().load_configs([a, b])
        task_names = [t.name for t in config.workflow.tasks]
        assert task_names == ["t1"]
        t1 = config.workflow.tasks[0]
        assert t1.script == ["echo a"]
        assert t1.operator == "op1"

    def test_missing_dependency_across_files_raises(self, tmp_path):
        a = _write(tmp_path, "a.yaml", """
version: "0.1"
workflow:
  name: wf
  tasks:
    - name: t2
      script:
        - echo t2
      depends_on:
        - t1
""")
        with pytest.raises(ValueError, match="depends on unknown task"):
            ConfigLoader().load_configs([a])

    def test_dependency_resolved_across_files(self, tmp_path):
        a = _write(tmp_path, "a.yaml", """
version: "0.1"
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo t1
""")
        b = _write(tmp_path, "b.yaml", """
workflow:
  name: wf
  tasks:
    - name: t2
      script:
        - echo t2
      depends_on:
        - t1
""")
        config = ConfigLoader().load_configs([a, b])
        assert config.workflow.tasks[1].depends_on == ["t1"]


# ---------------------------------------------------------------------------
# Scattered / recursive deep-merge behavior
# ---------------------------------------------------------------------------

class TestScatteredDeepMerge:
    def test_task_resources_deep_merge_preserves_siblings(self):
        """Different nested keys of the same task combine rather than replace."""
        a = {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {"name": "t", "script": ["echo"], "resources": {"nodes": {"count": 1}}}
                ],
            },
        }
        b = {"workflow": {"tasks": [{"name": "t", "resources": {"gpus": {"count": 2}}}]}}
        merged = merge_config_dicts([a, b])
        task = merged["workflow"]["tasks"][0]
        assert task["resources"]["nodes"]["count"] == 1
        assert task["resources"]["gpus"]["count"] == 2

    def test_leaf_conflict_last_wins_with_warning(self):
        a = {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [
                    {"name": "t", "script": ["echo"], "resources": {"gpus": {"count": 1}}}
                ],
            },
        }
        b = {"workflow": {"tasks": [{"name": "t", "resources": {"gpus": {"count": 8}}}]}}
        warns: list[str] = []
        merged = merge_config_dicts([a, b], source_labels=["a.yaml", "b.yaml"], override_warnings=warns)
        assert merged["workflow"]["tasks"][0]["resources"]["gpus"]["count"] == 8
        assert any("workflow.tasks.t.resources.gpus.count" in w for w in warns)
        assert any("previously from a.yaml" in w for w in warns)

    def test_value_list_last_wins(self):
        """A leaf list (e.g. script) is replaced wholesale, not concatenated."""
        a = {
            "version": "0.1",
            "workflow": {"name": "wf", "tasks": [{"name": "t", "script": ["one"]}]},
        }
        b = {"workflow": {"tasks": [{"name": "t", "script": ["two", "three"]}]}}
        merged = merge_config_dicts([a, b])
        assert merged["workflow"]["tasks"][0]["script"] == ["two", "three"]

    def test_backend_scatter_merge(self):
        """A backend can be defined type-first in one file and tuned in another."""
        a = {
            "version": "0.1",
            "backends": {"k8s": {"type": "kubernetes", "namespace": "ns"}},
            **{"workflow": {"name": "wf", "tasks": [{"name": "t", "script": ["echo"]}]}},
        }
        b = {"backends": {"k8s": {"host_network": True}}}
        merged = merge_config_dicts([a, b])
        assert merged["backends"]["k8s"]["type"] == "kubernetes"
        assert merged["backends"]["k8s"]["namespace"] == "ns"
        assert merged["backends"]["k8s"]["host_network"] is True

    def test_storage_section_merged(self):
        """`storage` is now merged across files (previously dropped after file 1)."""
        a = {
            "version": "0.1",
            "storage": {"s3main": {"type": "s3", "bucket": "b"}},
            "workflow": {"name": "wf", "tasks": [{"name": "t", "script": ["echo"]}]},
        }
        b = {"storage": {"s3alt": {"type": "s3", "bucket": "c"}}}
        merged = merge_config_dicts([a, b])
        assert "s3main" in merged["storage"]
        assert "s3alt" in merged["storage"]

    def test_workflow_monitor_merged(self):
        """`workflow.monitor` is now deep-merged (previously dropped)."""
        a = {
            "version": "0.1",
            "workflow": {
                "name": "wf",
                "tasks": [{"name": "t", "script": ["echo"]}],
            },
        }
        b = {"workflow": {"monitor": {"interval": 1000}}}
        merged = merge_config_dicts([a, b])
        assert merged["workflow"]["monitor"]["interval"] == 1000

    def test_unnamed_task_preserved_not_silently_dropped(self):
        """A task without a `name` cannot be merged by name, but must be KEPT (so
        schema validation raises a clear "name required" error) rather than silently
        dropped from the merged result."""
        a = {
            "version": "0.1",
            "workflow": {"name": "wf", "tasks": [{"name": "t", "script": ["echo"]}]},
        }
        b = {"workflow": {"tasks": [{"script": ["echo nameless"]}]}}
        merged = merge_config_dicts([a, b])
        tasks = merged["workflow"]["tasks"]
        assert len(tasks) == 2
        assert [t.get("name") for t in tasks] == ["t", None]
