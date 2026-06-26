# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-driven monitor injection: turn ``--enable-*-monitor`` flags into config.

The single home for the logic that injects default hardware monitors requested
via CLI flags, shared by two call sites that operate on different representations:

* the app layer (:meth:`sflow.app.sflow.SflowApp.run`) injects into the validated
  :class:`~sflow.config.schema.SflowConfig` pydantic model
  (:func:`apply_monitor_cli_overrides`);
* the compose layer (the run/batch YAML snapshot) injects into the raw merged
  config dict before validation (:func:`inject_cli_monitors_into_dict`).

Both paths share the SAME default (:func:`default_monitor_dict`) and the SAME
target validation (:func:`validate_monitor_target_tasks`), so the two
representations cannot drift. A monitor already declared in the recipe is always
left untouched -- the CLI only fills gaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from sflow.logging import get_logger

if TYPE_CHECKING:
    from sflow.config.schema import SflowConfig

_logger = get_logger(__name__)


def default_monitor_dict() -> Dict[str, Any]:
    """Default monitor injected by the ``--enable-*-monitor`` CLI flags.

    All hardware scopes (no ``scopes`` block) plus the detailed report -- the
    single source of truth for both the pydantic and dict injection paths.
    """
    return {"report": {"enabled": True}}


def validate_monitor_target_tasks(
    known_task_names: Iterable[str], requested: Iterable[str]
) -> None:
    """Raise ``ValueError`` if any ``--enable-task-monitor`` target is unknown."""
    known = {str(n) for n in known_task_names if n}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(
            f"--enable-task-monitor refers to unknown task(s): {sorted(set(unknown))}. "
            f"Known tasks: {sorted(known)}"
        )


def apply_monitor_cli_overrides(
    config: "SflowConfig",
    *,
    enable_workflow_monitor: bool,
    enable_task_monitors: Optional[List[str]],
) -> "SflowConfig":
    """Inject default monitors requested via CLI flags into a validated config.

    Each injected monitor uses the OOTB defaults (all hardware scopes, detailed
    CSV+SVG report). Monitors already declared in the recipe are left untouched so
    app-specific designs win; the CLI only fills gaps.
    """
    from sflow.config.schema import MonitorConfig

    enable_task_monitors = enable_task_monitors or []
    if not enable_workflow_monitor and not enable_task_monitors:
        return config

    def _default_monitor() -> "MonitorConfig":
        # No scopes -> all built-ins; enable the detailed report (csv+svg by default).
        return MonitorConfig.model_validate(default_monitor_dict())

    updates: Dict[str, Any] = {}
    workflow = config.workflow

    if enable_workflow_monitor:
        if workflow.monitor is not None:
            _logger.info(
                "workflow.monitor already defined in the recipe; "
                "--enable-workflow-monitor is a no-op."
            )
        else:
            workflow = workflow.model_copy(update={"monitor": _default_monitor()})
            _logger.info("Enabled default workflow monitor (all hardware scopes).")

    if enable_task_monitors:
        validate_monitor_target_tasks(
            (t.name for t in workflow.tasks), enable_task_monitors
        )
        targets = set(enable_task_monitors)
        new_tasks = []
        for task in workflow.tasks:
            if task.name in targets:
                if task.monitor is not None:
                    _logger.info(
                        f"Task '{task.name}' already defines a monitor in the recipe; "
                        "--enable-task-monitor is a no-op for it."
                    )
                else:
                    task = task.model_copy(update={"monitor": _default_monitor()})
                    _logger.info(
                        f"Enabled default task monitor for '{task.name}' (all hardware scopes)."
                    )
            new_tasks.append(task)
        workflow = workflow.model_copy(update={"tasks": new_tasks})

    if workflow is not config.workflow:
        updates["workflow"] = workflow
    return config.model_copy(update=updates) if updates else config


def inject_cli_monitors_into_dict(
    merged: Dict[str, Any],
    *,
    enable_workflow_monitor: bool,
    enable_task_monitors: Optional[List[str]],
) -> None:
    """Inject default monitor sections into a raw merged config dict (pre-validation).

    The dict-level twin of :func:`apply_monitor_cli_overrides`, used by the compose
    snapshot so a saved/re-run config reproduces the same CLI-enabled monitoring.
    Existing monitor sections from the recipe are left untouched.
    """
    enable_task_monitors = enable_task_monitors or []
    if not enable_workflow_monitor and not enable_task_monitors:
        return
    wf = merged.get("workflow")
    if not isinstance(wf, dict):
        return

    # Treat a present-but-null ``monitor`` the same as a missing one (``None`` ->
    # inject the default), matching the model path's ``monitor is not None`` check
    # so the two representations cannot drift.
    if enable_workflow_monitor and wf.get("monitor") is None:
        wf["monitor"] = default_monitor_dict()

    if not enable_task_monitors:
        return

    tasks = wf.get("tasks")
    targets = set(enable_task_monitors)
    if isinstance(tasks, list):
        names = {t.get("name") for t in tasks if isinstance(t, dict)}
        validate_monitor_target_tasks(names, enable_task_monitors)
        for task in tasks:
            if isinstance(task, dict) and task.get("name") in targets and task.get("monitor") is None:
                task["monitor"] = default_monitor_dict()
    elif isinstance(tasks, dict):
        validate_monitor_target_tasks(tasks.keys(), enable_task_monitors)
        for name in targets:
            spec = tasks.get(name)
            if isinstance(spec, dict) and spec.get("monitor") is None:
                spec["monitor"] = default_monitor_dict()
