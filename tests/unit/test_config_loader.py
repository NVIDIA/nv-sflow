# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sflow.config.loader import ConfigLoader


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
