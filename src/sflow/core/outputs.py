# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sflow.logging import get_logger
from sflow.utils.parser import LinesParser

from .task import OutputSpec, Task

_logger = get_logger(__name__)


def task_log_path(task: Task) -> Path | None:
    """
    Best-effort locate the task log file for output parsing.

    Current layout (SRD REQ-4.4 / app.sflow):
      <SFLOW_TASK_OUTPUT_DIR>/<task_name>.log
    """
    task_out = task.envs.get("SFLOW_TASK_OUTPUT_DIR")
    if task_out:
        return Path(task_out) / f"{task.name}.log"

    wf_out = task.envs.get("SFLOW_WORKFLOW_OUTPUT_DIR")
    if wf_out:
        return Path(wf_out) / task.name / f"{task.name}.log"

    return None


def task_outputs_path(task: Task) -> Path | None:
    task_out = task.envs.get("SFLOW_TASK_OUTPUT_DIR")
    if task_out:
        return Path(task_out) / "outputs.json"
    return None


def parse_outputs_from_text(text: str, specs: list[OutputSpec]) -> dict[str, Any]:
    """
    Parse task outputs from log text using parse-style patterns.

    Semantics (MVP):
    - Apply all patterns line-by-line over the log text.
    - Collect named fields across all matches.
    - If a key has a single match -> scalar, otherwise -> list.
    """
    if not specs:
        return {}
    patterns = [s.pattern for s in specs if s and s.pattern]
    if not patterns:
        return {}
    # Logs are written via logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    # so the actual user output is the final "message" segment after 3 separators.
    # Parse against the extracted message to keep patterns simple (SRD-style).
    lines = []
    for line in text.splitlines():
        # Split at most 3 times: ts - logger - level - message
        parts = line.split(" - ", 3)
        lines.append(parts[-1] if parts else line)

    parser = LinesParser(patterns)
    parser.add_lines(lines)
    return dict(parser.parsed_dict())


async def collect_task_outputs(task: Task) -> dict[str, Any]:
    """
    Parse outputs for a task from its log file and persist outputs.json.

    This is intentionally best-effort for MVP:
    - Missing log file -> {} (no error)
    - Parse errors -> {} (logged at warning)
    """
    if not task.output_specs:
        task.outputs = {}
        return {}

    log_path = task_log_path(task)
    if log_path is None:
        task.outputs = {}
        return {}

    def _read_text() -> str | None:
        try:
            return log_path.read_text(errors="ignore")
        except FileNotFoundError:
            return None
        except Exception as e:
            _logger.warning(f"Failed to read log for outputs parsing: {log_path}: {e}")
            return None

    text = await asyncio.to_thread(_read_text)
    if not text:
        task.outputs = {}
        return {}

    try:
        parsed = parse_outputs_from_text(text, list(task.output_specs))
    except Exception as e:
        _logger.warning(f"Failed to parse outputs for task {task.name}: {e}")
        parsed = {}

    task.outputs = parsed

    out_path = task_outputs_path(task)
    if out_path is not None:
        payload = {
            "task": task.name,
            "specs": [asdict(s) for s in (task.output_specs or [])],
            "outputs": parsed,
        }
        try:
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except Exception as e:
            _logger.warning(f"Failed to write outputs.json for task {task.name}: {e}")

    return parsed
