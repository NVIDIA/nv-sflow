# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from sflow.config.loader import ConfigLoader


def test_config_loader_fails_fast_on_invalid_expression_syntax(tmp_path: Path):
    p = tmp_path / "sflow.yaml"
    p.write_text(
        "\n".join(
            [
                'version: "0.1"',
                "variables:",
                "  X:",
                '    value: "${{ 1 + }}"',
                "backends:",
                "  - name: local",
                "    type: local",
                "    default: true",
                "workflow:",
                "  name: wf",
                "  tasks:",
                "    - name: t1",
                "      script:",
                "        - echo hi",
                "",
            ]
        )
    )

    with pytest.raises(ValueError, match="expression syntax validation failed"):
        ConfigLoader().load_config(p)
