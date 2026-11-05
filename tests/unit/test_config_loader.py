# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import yaml

from sflow.config.loader import (
    _FLOAT_WITHOUT_SEXAGESIMAL,
    _INT_WITHOUT_SEXAGESIMAL,
    ConfigLoader,
    safe_load,
)


def _vars_to_map(config) -> dict[str, object]:
    # SflowConfig.variables is normalized to a list; convert to name->value for assertions.
    if not config.variables:
        return {}
    return {v.name: v.value for v in config.variables}


def test_load_config_basic(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
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
""".lstrip()
    )

    loader = ConfigLoader()
    config = loader.load_config(p)

    assert config.version == "0.1"
    assert _vars_to_map(config)["X"] == 1
    assert config.workflow.name == "wf"
    assert config.workflow.tasks[0].name == "t1"


def test_load_config_preserves_plain_script_command_with_colon(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo "My GPUs: $CUDA_VISIBLE_DEVICES"
""".lstrip()
    )

    config = ConfigLoader().load_config(p)

    assert config.workflow.tasks[0].script == [
        'echo "My GPUs: $CUDA_VISIBLE_DEVICES"'
    ]


def test_load_config_preserves_plain_script_command_with_colon_in_task_map(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
workflow:
  name: wf
  tasks:
    t1:
      script:
        - echo "My GPUs: $CUDA_VISIBLE_DEVICES"
""".lstrip()
    )

    config = ConfigLoader().load_config(p)

    assert config.workflow.tasks[0].script == [
        'echo "My GPUs: $CUDA_VISIBLE_DEVICES"'
    ]


def test_load_config_applies_variable_overrides(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
variables:
  X:
    value: 1
  FLAG:
    value: false
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""".lstrip()
    )

    loader = ConfigLoader()
    config = loader.load_config(p, variable_overrides=["X=42", "FLAG=true"])

    assert _vars_to_map(config)["X"] == 42
    assert _vars_to_map(config)["FLAG"] is True


def test_load_config_applies_artifact_overrides(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
artifacts:
  - name: MODEL
    uri: fs:///old
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""".lstrip()
    )

    loader = ConfigLoader()
    config = loader.load_config(p, artifact_overrides=["MODEL=fs:///new"])

    assert config.artifacts[0].name == "MODEL"
    assert config.artifacts[0].uri == "fs:///new"


def test_variable_override_unknown_key_raises(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
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
""".lstrip()
    )

    loader = ConfigLoader()
    with pytest.raises(ValueError, match="is not defined in the configuration"):
        loader.load_config(p, variable_overrides=["Y=2"])


def test_variable_override_invalid_format_raises(tmp_path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
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
""".lstrip()
    )

    loader = ConfigLoader()
    with pytest.raises(ValueError, match="Invalid variable override format"):
        loader.load_config(p, variable_overrides=["X"])


# ---------------------------------------------------------------------------
# strip_missable_tasks tests
# ---------------------------------------------------------------------------

from sflow.config.loader import strip_missable_tasks


def test_strip_missable_removes_absent_depends_on():
    config = {
        "workflow": {
            "tasks": [
                {"name": "t1", "script": ["echo"]},
                {"name": "t2", "depends_on": ["t1", "missing"], "script": ["echo"]},
            ]
        }
    }
    strip_missable_tasks(config, ["missing"])
    assert config["workflow"]["tasks"][1].get("depends_on") == ["t1"]


def test_strip_missable_keeps_present_tasks():
    config = {
        "workflow": {
            "tasks": [
                {"name": "t1", "script": ["echo"]},
                {"name": "t2", "depends_on": ["t1"], "script": ["echo"]},
            ]
        }
    }
    strip_missable_tasks(config, ["t1"])
    assert config["workflow"]["tasks"][1]["depends_on"] == ["t1"]


def test_strip_missable_glob_pattern():
    config = {
        "workflow": {
            "tasks": [
                {"name": "t1", "script": ["echo"]},
                {
                    "name": "bench",
                    "depends_on": ["t1", "prefill_server", "decode_server"],
                    "script": ["echo"],
                },
            ]
        }
    }
    strip_missable_tasks(config, ["prefill_*", "decode_*"])
    assert config["workflow"]["tasks"][1]["depends_on"] == ["t1"]


def test_strip_missable_removes_empty_depends_on():
    config = {
        "workflow": {
            "tasks": [
                {"name": "t1", "depends_on": ["missing"], "script": ["echo"]},
            ]
        }
    }
    strip_missable_tasks(config, ["missing"])
    assert "depends_on" not in config["workflow"]["tasks"][0]


def test_strip_missable_removes_probe_logger():
    config = {
        "workflow": {
            "tasks": [
                {
                    "name": "t1",
                    "script": ["echo"],
                    "probes": {
                        "readiness": {
                            "log_watch": {
                                "regex_pattern": "ready",
                                "logger": "missing_task",
                            }
                        }
                    },
                },
            ]
        }
    }
    strip_missable_tasks(config, ["missing_task"])
    lw = config["workflow"]["tasks"][0]["probes"]["readiness"]["log_watch"]
    assert "logger" not in lw


def test_strip_missable_noop_without_workflow():
    config = {"version": "0.1"}
    stripped = strip_missable_tasks(config, ["anything"])
    assert stripped == []
    assert config == {"version": "0.1"}


def test_load_config_warns_that_timeout_is_not_enforced(tmp_path, caplog):
    """`timeout:` is accepted but nothing reads it, so a recipe that sets one
    LOOKS bounded and is not. The field cannot simply be removed (these models
    forbid extra keys, so every config setting it would stop loading), so the
    warning is what stops it lying."""
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
workflow:
  name: wf
  timeout: 115m
  tasks:
    - name: t1
      timeout: 30m
      script:
        - echo hi
    - name: t2
      script:
        - echo hi
""".lstrip()
    )

    with caplog.at_level("WARNING"):
        ConfigLoader().load_config(p)

    msg = "\n".join(r.message for r in caplog.records)
    assert "does not enforce it" in msg
    assert "workflow" in msg and "t1" in msg
    # t2 sets no timeout, so it must not be named.
    assert "t2" not in msg


def test_load_config_is_quiet_when_no_timeout_is_set(tmp_path, caplog):
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

    with caplog.at_level("WARNING"):
        ConfigLoader().load_config(p)

    assert "does not enforce it" not in "\n".join(r.message for r in caplog.records)


def test_unquoted_slurm_walltime_is_not_read_as_base_60(tmp_path):
    """`time: 10:00:00` must reach the backend as a string, not YAML 1.1 base-60.

    PyYAML resolves `10:00:00` to the integer 36000, which Slurm reads as 36000
    *minutes*. Guards the ConfigLoader wiring, not just the loader helper.
    """
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
backends:
  - name: slurm
    type: slurm
    default: true
    partition: debug
    account: test
    nodes: 1
    gpus_per_node: 1
    time: 10:00:00
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""".lstrip()
    )

    config = ConfigLoader().load_config(p)

    assert config.backends[0].time == "10:00:00"


def test_integer_walltime_still_means_minutes(tmp_path):
    """A genuine integer must keep working — Slurm reads it as minutes."""
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "0.1"
backends:
  - name: slurm
    type: slurm
    default: true
    partition: debug
    account: test
    nodes: 1
    gpus_per_node: 1
    time: 5400
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""".lstrip()
    )

    config = ConfigLoader().load_config(p)

    assert config.backends[0].time == 5400

@pytest.mark.parametrize(
    "text",
    [
        "10:00:00",  # the reported case: 10 hours, not 36000
        "9:00:00",
        "1:30:00",
        "10:00",
        "01:00:00",  # leading zero already survived under YAML 1.1
        "00:10:00",
        "1-00:00:00",  # Slurm day-hour form
        "10:00:00.5",  # sexagesimal *float* — used to fail validation outright
    ],
)
def test_walltime_stays_a_string(text: str) -> None:
    assert safe_load(f"time: {text}") == {"time": text}


def test_only_sexagesimal_scalars_differ_from_pyyaml() -> None:
    """Dropping base-60 must not disturb any other scalar type.

    `017` is YAML 1.1 octal (15); `1e10` stays a string because PyYAML requires a
    signed exponent. Both quirks are deliberately preserved.
    """
    document = (
        "a: 7\nb: -7\nc: 0\nd: 0x1f\ne: 017\nf: 1.5\ng: -0.25\n"
        "h: true\ni: null\nj: slurm\nk: [1, 2]\nl: {x: 1}\nm: 1e10\n"
    )
    assert safe_load(document) == yaml.safe_load(document)


def test_loader_does_not_mutate_pyyaml_global_state() -> None:
    """Also pins the upstream behavior this loader exists to neutralize."""
    safe_load("time: 10:00:00")
    assert yaml.safe_load("time: 10:00:00") == {"time": 36000}


def _without_sexagesimal(pattern: str) -> str:
    """Delete the base-60 alternative from a PyYAML resolver pattern."""
    collapsed = re.sub(r"\s+", "", pattern)
    return re.sub(r"\|[^|]*?\(\?::\[0-5\]\?\[0-9\]\)\+[^|)]*", "", collapsed)


@pytest.mark.parametrize(
    ("tag", "ours"),
    [
        ("tag:yaml.org,2002:int", _INT_WITHOUT_SEXAGESIMAL),
        ("tag:yaml.org,2002:float", _FLOAT_WITHOUT_SEXAGESIMAL),
    ],
)
def test_patterns_stay_in_sync_with_pyyaml(tag: str, ours) -> None:
    """Fail loudly if PyYAML edits the patterns we copied.

    Our resolvers are PyYAML's own, minus the base-60 branch. That copy would
    otherwise rot silently on a PyYAML upgrade, so pin the relationship rather
    than the literal text: re-derive ours from upstream and compare.
    """
    upstream = next(
        pattern
        for resolvers in yaml.SafeLoader.yaml_implicit_resolvers.values()
        for resolver_tag, pattern in resolvers
        if resolver_tag == tag
    )
    assert "(?::[0-5]?[0-9])+" in upstream.pattern, (
        f"PyYAML no longer resolves {tag} as base-60; this workaround may be obsolete"
    )
    assert _without_sexagesimal(upstream.pattern) == re.sub(r"\s+", "", ours.pattern)


def test_config_without_version_defaults_to_0_1(tmp_path):
    """`version` is optional; omitting it means "0.1"."""
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""".lstrip()
    )

    config = ConfigLoader().load_config(p)

    assert config.version == "0.1"


def test_explicit_unsupported_version_still_rejected(tmp_path):
    """Optional does not mean unvalidated: a declared bad version still fails."""
    p = tmp_path / "sflow.yaml"
    p.write_text(
        """
version: "9.9"
workflow:
  name: wf
  tasks:
    - name: t1
      script:
        - echo hi
""".lstrip()
    )

    with pytest.raises(Exception, match="9.9"):
        ConfigLoader().load_config(p)
