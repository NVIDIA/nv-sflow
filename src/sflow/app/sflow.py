# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import deque
import os
from pathlib import Path
from typing import Any

from sflow.app.assembly import build_state, release_backends
from sflow.config.loader import ConfigLoader
from sflow.logging import add_log_file, get_logger

_logger = get_logger(__name__)


def extract_container_mounts_from_extra_args(extra_args: list[str]) -> list[str]:
    """
    Extract --container-mounts values from extra_args.
    
    Handles both formats:
    - --container-mounts /path1:/path2
    - --container-mounts=/path1:/path2
    
    Multiple comma-separated mounts are split into individual entries.
    """
    mounts: list[str] = []
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg == "--container-mounts" and i + 1 < len(extra_args):
            mounts.extend(extra_args[i + 1].split(","))
            i += 2
        elif arg.startswith("--container-mounts="):
            mounts.extend(arg.split("=", 1)[1].split(","))
            i += 1
        else:
            i += 1
    return mounts


class SflowApp:
    """
    Application facade used by CLI/UI integrations.
    """

    def __init__(self) -> None:
        # Stores the workflow output directory from the last run (even if failed)
        self.last_workflow_output_dir: Path | None = None

    def run(
        self,
        *,
        file: Path,
        dry_run: bool = False,
        resume: str | None = None,
        variable_overrides: list[str] | None = None,
        artifact_overrides: list[str] | None = None,
        workspace_dir: Path | None = None,
        output_dir: Path | None = None,
        tui: bool = False,
        tui_log_buffer: deque[logging.LogRecord] | None = None,
        tui_refresh_per_second: int | None = None,
    ) -> Path | None:
        """
        Run the workflow and return the workflow output directory path.
        
        Returns:
            Path to the workflow output directory (e.g., sflow_output/<workflow-name>-<timestamp>-<id>),
            or None for dry-run mode.
        """
        import asyncio
        import secrets
        from datetime import datetime

        if resume is not None:
            raise NotImplementedError("--resume is not implemented yet")

        # Reset from previous runs
        self.last_workflow_output_dir = None

        # load the config
        config = ConfigLoader().load_config(
            file, variable_overrides, artifact_overrides
        )

        async def _run_async() -> Path | None:
            import atexit
            import contextlib
            import signal
            import subprocess
            from contextlib import suppress

            from sflow.core.orchestrator import Orchestrator
            from sflow.ui.rich_tui import RichTui, RichTuiConfig

            ui: RichTui | None = None
            ui_task: asyncio.Task | None = None
            orch: Orchestrator | None = None
            received_signal: signal.Signals | None = None
            atexit_cleaned = False
            owned_allocation_ids: list[str] = []

            async def _ui_loop() -> None:
                # Refresh at a higher rate than Orchestrator poll_interval so logs feel like tail -f.
                # Use the same rate as Live refresh to avoid over-refreshing.
                refresh_hz = 10
                if ui is not None:
                    try:
                        refresh_hz = int(getattr(getattr(ui, "_config", None), "refresh_per_second", 10))
                    except Exception:
                        refresh_hz = 10
                sleep_s = 0.1 if refresh_hz <= 0 else max(0.01, 1.0 / float(refresh_hz))
                while True:
                    if ui is not None:
                        ui.refresh()
                        if ui.workflow is not None and ui.workflow.is_finished():
                            return
                    await asyncio.sleep(sleep_s)

            # Start UI as early as possible (before build_state / backend allocation).
            if tui:
                cfg = RichTuiConfig()
                if tui_refresh_per_second is not None:
                    cfg.refresh_per_second = int(tui_refresh_per_second)
                ui = RichTui(
                    workflow=None,
                    workflow_name=config.workflow.name,
                    config=cfg,
                    log_buffer=tui_log_buffer,
                    attach_log_handler=False if tui_log_buffer is not None else True,
                )

            ui_cm = ui if ui is not None else contextlib.nullcontext()
            with ui_cm:
                if ui is not None:
                    ui.refresh()
                    ui_task = asyncio.create_task(_ui_loop())

                # Workspace/output dirs are needed early for artifact resolution.
                ws_dir = (
                    Path(workspace_dir) if workspace_dir is not None else Path.cwd()
                )
                out_dir = (
                    Path(output_dir)
                    if output_dir is not None
                    else ws_dir / "sflow_output"
                )

                # Compute workflow_out_dir early so inline file:// artifacts are
                # written under the run output folder rather than the workspace.
                if dry_run:
                    workflow_out_dir = out_dir / "_dry_run" / config.workflow.name
                else:
                    run_id = f"{config.workflow.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
                    slurm_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
                    if slurm_job_id:
                        run_id = f"{slurm_job_id}-{run_id}"
                    workflow_out_dir = out_dir / run_id
                    self.last_workflow_output_dir = workflow_out_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    workflow_out_dir.mkdir(parents=True, exist_ok=True)

                # -----------------------------------------------------------------
                # Signal handling (SRD REQ-5.3)
                # -----------------------------------------------------------------
                loop = asyncio.get_running_loop()
                main_task = asyncio.current_task()

                def _on_signal(sig: signal.Signals) -> None:
                    nonlocal received_signal
                    received_signal = sig
                    _logger.warning(f"Received {sig.name}; requesting shutdown...")
                    if orch is not None:
                        orch.request_stop(sig.name)
                        # When orchestrator is running, let it unwind via request_stop so we don't
                        # turn a user cancel into a noisy CancelledError.
                        return
                    # If we are mid-planning/allocation, cancel the main task so awaits can unwind.
                    if main_task is not None and not main_task.done():
                        main_task.cancel()

                installed_signals: list[signal.Signals] = []
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.add_signal_handler(sig, _on_signal, sig)
                        installed_signals.append(sig)
                    except (NotImplementedError, RuntimeError):
                        # Not supported on some platforms / threads.
                        pass

                # build the state:
                # - dry-run: never allocates
                allocate = not dry_run
                state = None
                try:
                    build_kw: dict[str, Any] = {
                        "allocate": allocate,
                        "output_dir": workflow_out_dir,
                    }
                    if workspace_dir is not None:
                        build_kw["workspace_dir"] = ws_dir
                    state = await build_state(config, **build_kw)
                finally:
                    # If build_state was cancelled and did not return a state, there's nothing we can do here.
                    # build_state itself is responsible for releasing partial allocations on failure/cancel.
                    pass

                assert state is not None
                if ui is not None:
                    ui.set_workflow(state.workflow)
                    ui.refresh()

                # Atexit fallback cleanup for owned allocations (best-effort).
                # This will not run on SIGKILL or hard crashes; it's a last resort.
                if allocate:
                    try:
                        for b in state.backends.values():
                            alloc = getattr(b, "allocation", None)
                            if (
                                alloc
                                and getattr(alloc, "owned", True)
                                and getattr(alloc, "allocation_id", None)
                            ):
                                owned_allocation_ids.append(str(alloc.allocation_id))
                    except Exception:
                        owned_allocation_ids = []

                    def _atexit_cleanup() -> None:
                        nonlocal atexit_cleaned
                        if atexit_cleaned:
                            return
                        import shutil

                        scancel_bin = shutil.which("scancel")
                        if not scancel_bin:
                            return
                        for alloc_id in owned_allocation_ids:
                            try:
                                subprocess.run(
                                    [scancel_bin, alloc_id],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    check=False,
                                )
                            except Exception:
                                pass

                    atexit.register(_atexit_cleanup)

                # -----------------------------------------------------------------
                # Output directory structure + built-in envs (SRD REQ-1.4 / REQ-4.4)
                # -----------------------------------------------------------------
                tg = state.workflow.task_graph
                # -----------------------------------------------------------------
                # Container mount hygiene for Slurm/Pyxis runs
                # -----------------------------------------------------------------
                # When using Pyxis containers, tasks typically run inside a container filesystem.
                # Our built-in SFLOW_* env vars point at host paths (workspace/output dirs), so
                # we must mount those host paths into the container at the same absolute paths.
                #
                # This is intentionally done here (after we compute SFLOW_* dirs) rather than in
                # assembly, because assembly does not know the final output directory layout.
                def _ensure_sflow_dir_mounts_for_srun_container(
                    *,
                    task: Any,
                    ws_dir: Path,
                    out_dir: Path,
                    workflow_out_dir: Path,
                    task_out_dir: Path,
                ) -> None:
                    try:
                        op = getattr(task, "operator", None)
                        op_conf = getattr(op, "config", None)
                        if op_conf is None:
                            return
                        if getattr(op_conf, "type", None) != "srun":
                            return
                        # Only relevant when using a container (Pyxis flags).
                        if not (
                            getattr(op_conf, "container_image", None)
                            or getattr(op_conf, "container_name", None)
                        ):
                            return

                        def _mount_key(mount: str) -> tuple[str, str] | None:
                            parts = str(mount).split(":", 2)
                            if len(parts) < 2:
                                return None
                            return (parts[0], parts[1])

                        existing_mounts = list(
                            getattr(op_conf, "container_mounts", None) or []
                        )
                        existing_keys: set[tuple[str, str]] = set()
                        for m in existing_mounts:
                            k = _mount_key(m)
                            if k is not None:
                                existing_keys.add(k)

                        auto_mounts: list[str] = []
                        for p in (ws_dir, out_dir, workflow_out_dir, task_out_dir):
                            src = str(p)
                            dst = src  # preserve absolute path inside container
                            key = (src, dst)
                            if key in existing_keys:
                                continue
                            auto_mounts.append(f"{src}:{dst}:rw")
                            existing_keys.add(key)

                        if auto_mounts:
                            setattr(op_conf, "container_mounts", existing_mounts + auto_mounts)
                    except Exception:
                        # Best-effort only; do not break planning/execution due to mount inference.
                        return

                if not dry_run:
                    # Add a global sflow log file under the workflow output dir.
                    add_log_file(str(workflow_out_dir / "sflow.log"))

                for t in tg.get_tasks():
                    task_out_dir = workflow_out_dir / t.name
                    if not dry_run:
                        task_out_dir.mkdir(parents=True, exist_ok=True)

                    t.envs.setdefault("SFLOW_WORKSPACE_DIR", str(ws_dir))
                    t.envs.setdefault("SFLOW_OUTPUT_DIR", str(out_dir))
                    t.envs.setdefault(
                        "SFLOW_WORKFLOW_OUTPUT_DIR", str(workflow_out_dir)
                    )
                    t.envs.setdefault("SFLOW_TASK_OUTPUT_DIR", str(task_out_dir))
                    _ensure_sflow_dir_mounts_for_srun_container(
                        task=t,
                        ws_dir=ws_dir,
                        out_dir=out_dir,
                        workflow_out_dir=workflow_out_dir,
                        task_out_dir=task_out_dir,
                    )

                    if not dry_run:
                        log_path = task_out_dir / f"{t.name}.log"
                        existing = [
                            h
                            for h in t.logger.handlers
                            if isinstance(h, logging.FileHandler)
                            and getattr(h, "baseFilename", None) == str(log_path)
                        ]
                        if not existing:
                            fh = logging.FileHandler(log_path)
                            fh.setFormatter(
                                logging.Formatter(
                                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                                )
                            )
                            t.logger.addHandler(fh)
                        t.logger.setLevel(logging.INFO)
                        t.logger.propagate = False

                if dry_run:
                    plan_tasks = tg.get_tasks()
                    order = tg.dag.topological_sort()

                    # Validate artifact paths (fs:// and file:// schemes)
                    def _validate_artifact_paths(
                        artifacts: dict,
                    ) -> list[str]:
                        """Check if artifact filesystem paths exist."""
                        from urllib.parse import urlparse

                        warnings: list[str] = []
                        for name, art in (artifacts or {}).items():
                            uri = getattr(art, "uri", "") or ""
                            try:
                                scheme = (urlparse(str(uri)).scheme or "").lower()
                            except Exception:
                                scheme = ""
                            # Only validate local filesystem artifacts
                            if scheme not in {"fs", "file"}:
                                continue
                            art_path = getattr(art, "path", None)
                            if art_path is None:
                                continue
                            # Skip paths with unresolved variables
                            path_str = str(art_path)
                            if "$" in path_str or "{" in path_str:
                                continue
                            if not Path(art_path).exists():
                                warnings.append(
                                    f"Artifact '{name}' (type: '{scheme}') path does not exist: {art_path}"
                                )
                                if scheme == "file":
                                    warnings.append(
                                        f"\tBut you can ignore this warning since 'file' type artifact will be created at runtime."
                                    )
                        return warnings

                    artifact_warnings = _validate_artifact_paths(state.artifacts)
                    if artifact_warnings:
                        _logger.warning("Artifact path validation warnings:")
                        for w in artifact_warnings:
                            _logger.warning(f"  ⚠ {w}")
                    
                    # Validate container mount paths (REQ: warn users about invalid mounts)
                    def _validate_container_mounts(
                        tasks: list, *, sflow_output_dir: Path
                    ) -> list[str]:
                        """Check if container mount source paths exist on the filesystem."""
                        warnings: list[str] = []
                        sflow_out_str = str(sflow_output_dir)
                        for task in tasks:
                            op = getattr(task, "operator", None)
                            op_conf = getattr(op, "config", None)
                            if op_conf is None:
                                continue

                            # Check srun container_mounts
                            mounts = getattr(op_conf, "container_mounts", None) or []
                            # Also check docker mounts
                            if not mounts:
                                mounts = getattr(op_conf, "mounts", None) or []

                            for mount_spec in mounts:
                                # Mount format: /host/path:/container/path[:options]
                                parts = str(mount_spec).split(":", 2)
                                if len(parts) < 2:
                                    continue
                                host_path = parts[0]
                                if not host_path:
                                    continue
                                # Skip environment variable references (not resolvable at plan time)
                                if "$" in host_path or "{" in host_path:
                                    continue
                                # Skip auto-generated sflow output directories (created at runtime)
                                if host_path.startswith(sflow_out_str):
                                    continue
                                if not Path(host_path).exists():
                                    warnings.append(
                                        f"Task '{task.name}': mount source path does not exist: {host_path}"
                                    )
                        return warnings

                    mount_warnings = _validate_container_mounts(
                        plan_tasks, sflow_output_dir=out_dir
                    )
                    if mount_warnings:
                        _logger.warning("Container mount path validation warnings:")
                        for w in mount_warnings:
                            _logger.warning(f"  ⚠ {w}")

                    # Backends summary
                    used_backends = sorted(
                        {
                            t.backend_name
                            for t in plan_tasks
                            if getattr(t, "backend_name", None)
                        }
                    )
                    _logger.info("Dry-run plan:")
                    _logger.info(f"- workspace_dir: {ws_dir}")
                    _logger.info(f"- output_dir: {out_dir}")
                    _logger.info(f"- workflow_output_dir (planned): {workflow_out_dir}")

                    # Print overridden variables and artifacts
                    if variable_overrides:
                        _logger.info("- variable overrides:")
                        for var_override in variable_overrides:
                            if "=" in var_override:
                                key, value = var_override.split("=", 1)
                                value_stripped = value.strip()
                                if value_stripped.startswith("[") and value_stripped.endswith("]"):
                                    # Domain override (list)
                                    _logger.info(f"    {key} = {value}  (domain sweep)")
                                else:
                                    # Single value override
                                    _logger.info(f"    {key} = {value}")
                            else:
                                _logger.info(f"    {var_override}")
                    if artifact_overrides:
                        _logger.info("- artifact overrides:")
                        for art_override in artifact_overrides:
                            _logger.info(f"    {art_override}")
                    _logger.info(
                        f"- tasks: {len(plan_tasks)} (topological order: {', '.join(order)})"
                    )
                    _logger.info(
                        f"- backends defined: {', '.join(sorted(state.backends.keys()))}"
                    )
                    if used_backends:
                        _logger.info(
                            f"- backends used by tasks: {', '.join(used_backends)}"
                        )

                    for b_name, backend in state.backends.items():
                        b_type = backend.__class__.__name__
                        alloc = backend.allocation
                        if alloc is None:
                            _logger.info(
                                f"  - backend {b_name}: type={b_type}, allocated=no (dry-run)"
                            )
                        else:
                            nodes = [n.name for n in alloc.nodes]
                            _logger.info(
                                f"  - backend {b_name}: type={b_type}, allocated=yes, allocation_id={alloc.allocation_id}, nodes={nodes}"
                            )

                    # Collect and print all container mounts across tasks
                    # Skip sflow auto-generated output directory mounts
                    # Also extract --container-mounts from extra_args
                    all_mounts: set[str] = set()
                    for task_name in order:
                        task = tg.get_task(task_name)
                        task_op_conf = getattr(getattr(task, "operator", None), "config", None)
                        if task_op_conf is not None:
                            # Get mounts from container_mounts field
                            mounts = getattr(task_op_conf, "container_mounts", None)
                            if mounts:
                                for mount in mounts:
                                    # Skip sflow output directory mounts (auto-generated)
                                    if "sflow_output" in mount.lower():
                                        continue
                                    all_mounts.add(mount)
                            # Also get mounts from extra_args
                            extra_args = getattr(task_op_conf, "extra_args", None)
                            if extra_args:
                                for mount in extract_container_mounts_from_extra_args(list(extra_args)):
                                    if "sflow_output" in mount.lower():
                                        continue
                                    all_mounts.add(mount)
                    if all_mounts:
                        _logger.info("")
                        _logger.info("Container mounts (aggregated from all tasks):")
                        for mount in sorted(all_mounts):
                            _logger.info(f"  - {mount}")

                    # Tasks detail
                    _logger.info("")
                    _logger.info("=" * 60)
                    _logger.info("Tasks:")
                    _logger.info("=" * 60)
                    for idx, name in enumerate(order, 1):
                        t = tg.get_task(name)
                        deps = tg.dag.get_dependencies(name)
                        op_conf = getattr(getattr(t, "operator", None), "config", None)
                        op_type_str = getattr(op_conf, "type", None) or "unknown"

                        # Resources are best-effort inferred from runtime/envs for display.
                        nodelist = getattr(op_conf, "nodelist", None) or []
                        cuda_visible = t.envs.get("CUDA_VISIBLE_DEVICES")
                        task_out_dir = t.envs.get("SFLOW_TASK_OUTPUT_DIR")
                        retry = getattr(t, "retries", None)
                        retry_str = (
                            f"{retry.count}x, interval={retry.interval}, backoff={retry.backoff}"
                            if retry is not None
                            else "none"
                        )

                        # Task header
                        _logger.info("")
                        _logger.info(f"  [{idx}] {t.name}")
                        _logger.info(f"      ├─ backend: {getattr(t, 'backend_name', None)}")
                        _logger.info(f"      ├─ operator: {op_type_str}")
                        _logger.info(f"      ├─ deps: {list(deps) if deps else '[]'}")
                        _logger.info(f"      ├─ nodelist: {nodelist}")
                        if cuda_visible:
                            _logger.info(f"      ├─ CUDA_VISIBLE_DEVICES: {cuda_visible}")
                        _logger.info(f"      ├─ task_output_dir: {task_out_dir}")
                        _logger.info(f"      ├─ retries: {retry_str}")

                        # Sweep variables
                        if t.sweep_variables:
                            sweep_vals = {k: t.envs.get(k, "") for k in t.sweep_variables}
                            sweep_items = ", ".join(f"{k}={v}" for k, v in sweep_vals.items())
                            _logger.info(f"      ├─ sweep_vars: {{{sweep_items}}}")

                        # Operator config details
                        if op_conf is not None:
                            op_details: list[tuple[str, str]] = []
                            # Common srun operator fields
                            if getattr(op_conf, "nodes", None) is not None:
                                op_details.append(("nodes", str(op_conf.nodes)))
                            if getattr(op_conf, "ntasks", None) is not None:
                                op_details.append(("ntasks", str(op_conf.ntasks)))
                            if getattr(op_conf, "ntasks_per_node", None) is not None:
                                op_details.append(("ntasks_per_node", str(op_conf.ntasks_per_node)))
                            if getattr(op_conf, "cpus_per_task", None) is not None:
                                op_details.append(("cpus_per_task", str(op_conf.cpus_per_task)))
                            if getattr(op_conf, "gpus", None) is not None:
                                op_details.append(("gpus", str(op_conf.gpus)))
                            if getattr(op_conf, "gpus_per_task", None) is not None:
                                op_details.append(("gpus_per_task", str(op_conf.gpus_per_task)))
                            if getattr(op_conf, "container_image", None) is not None:
                                op_details.append(("container_image", op_conf.container_image))
                            if getattr(op_conf, "container_name", None) is not None:
                                op_details.append(("container_name", op_conf.container_name))
                            if getattr(op_conf, "container_mounts", None):
                                mounts = op_conf.container_mounts
                                if len(mounts) <= 3:
                                    op_details.append(("container_mounts", str(mounts)))
                                else:
                                    op_details.append(("container_mounts", f"[{len(mounts)} mounts]"))
                            if getattr(op_conf, "mpi", None) is not None:
                                op_details.append(("mpi", op_conf.mpi))
                            if getattr(op_conf, "job_id", None) is not None and getattr(op_conf, "job_id", None) != "0":
                                op_details.append(("job_id", str(op_conf.job_id)))
                            if getattr(op_conf, "extra_args", None):
                                extra_args_list = list(op_conf.extra_args)
                                if len(extra_args_list) <= 5:
                                    op_details.append(("extra_args", str(extra_args_list)))
                                else:
                                    op_details.append(("extra_args", f"[{len(extra_args_list)} args]"))

                            if op_details:
                                _logger.info("      └─ operator config:")
                                for i, (key, val) in enumerate(op_details):
                                    prefix = "└─" if i == len(op_details) - 1 else "├─"
                                    _logger.info(f"         {prefix} {key}: {val}")
                            else:
                                _logger.info("      └─ operator config: (default)")
                    _logger.info("")
                    _logger.info("=" * 60)
                    return None  # dry-run: no actual output directory created

                # run the workflow and always release backend allocations
                try:
                    orch = Orchestrator(workflow=state.workflow, poll_interval=1)
                    await orch.run()

                    # Determine overall success based on final task statuses (not just "orchestrator returned").
                    from sflow.core.task import TaskStatus

                    tasks = tg.get_tasks()
                    failed = [
                        t
                        for t in tasks
                        if t.status in {TaskStatus.FAILED, TaskStatus.TIMEOUT}
                    ]
                    cancelled = [t for t in tasks if t.status == TaskStatus.CANCELLED]
                    if failed:
                        names = ", ".join(t.name for t in failed)
                        raise RuntimeError(
                            f"Workflow '{config.workflow.name}' failed: {len(failed)} task(s) failed ({names})"
                        )
                    # Treat cancellations as non-success (covers fail-fast dependents and future user-cancel).
                    if cancelled:
                        names = ", ".join(t.name for t in cancelled)
                        raise RuntimeError(
                            f"Workflow '{config.workflow.name}' cancelled: {len(cancelled)} task(s) cancelled ({names})"
                        )
                finally:
                    # Always attempt to release owned backend allocations.
                    try:
                        await release_backends(state)
                    finally:
                        atexit_cleaned = True
                        # Remove signal handlers (SflowApp can be reused in-process in tests).
                        for sig in installed_signals:
                            with suppress(Exception):
                                loop.remove_signal_handler(sig)
                    if ui_task is not None:
                        ui_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await ui_task
                    if ui is not None:
                        ui.refresh()

                # If we were interrupted by a signal, propagate a meaningful exit status.
                if received_signal is not None:
                    if received_signal == signal.SIGINT:
                        raise KeyboardInterrupt()
                    raise SystemExit(128 + int(received_signal))

                return workflow_out_dir

        return asyncio.run(_run_async())

    def visualize(
        self,
        *,
        file: Path,
        output_path: Path | None = None,
        format: str = "mermaid",
        show_variables: bool = False,
        workspace_dir: Path | None = None,
        output_dir: Path | None = None,
    ):
        """
        Generate a workflow DAG visualization.

        Supported formats:
        - mermaid: writes Mermaid graph text to output_path
        - dot: writes Graphviz DOT to output_path
        - png/svg/pdf: renders DOT via `dot` binary

        Notes:
        - The visualization includes explicit Start/End nodes connected to all entry/exit tasks.
        """
        import asyncio
        import secrets
        import shutil
        import subprocess
        from dataclasses import dataclass
        from datetime import datetime

        from sflow.core.task_graph import TaskGraph

        @dataclass
        class VisualizeResult:
            task_count: int
            topo_order: list[str]
            saved_path: str | None = None
            format: str | None = None

        config = ConfigLoader().load_config(file)
        ws_dir = Path(workspace_dir) if workspace_dir is not None else Path.cwd()
        state = asyncio.run(build_state(config, allocate=False, workspace_dir=ws_dir))
        tg: TaskGraph = state.workflow.task_graph

        order = tg.dag.topological_sort()
        tasks = tg.get_tasks()
        fmt = format.lower()

        out_dir = (
            Path(output_dir) if output_dir is not None else ws_dir / "sflow_output"
        )

        ext_by_format: dict[str, str] = {
            "mermaid": ".mmd",
            "dot": ".dot",
            "png": ".png",
            "svg": ".svg",
            "pdf": ".pdf",
        }
        if fmt not in ext_by_format:
            raise ValueError(f"Unsupported format: {format}")

        if output_path is None:
            run_id = f"{config.workflow.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
            workflow_out_dir = out_dir / run_id
            workflow_out_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                workflow_out_dir / f"{config.workflow.name}{ext_by_format[fmt]}"
            )

        def _gpu_label(task) -> str:
            cuda = task.envs.get("CUDA_VISIBLE_DEVICES")
            if not cuda:
                return ""
            # "0,1,2" -> 3
            try:
                n = len([x for x in cuda.split(",") if x.strip() != ""])
            except Exception:
                n = 0
            return f" gpus={n}"

        # -----------------------------------------------------------------
        # REQ-6.9: Visualization grouping (replicas) - SRD style
        # Use Mermaid/DOT subgraphs to group replicas (e.g. t1_0..t1_N) under base task "t1".
        # This matches SRD Appendix D.
        # -----------------------------------------------------------------
        replica_bases: set[str] = set()
        try:
            for t_conf in config.workflow.tasks:
                if getattr(t_conf, "replicas", None) is not None:
                    replica_bases.add(str(t_conf.name))
        except Exception:
            replica_bases = set()

        def _is_replica_node(base: str, node_name: str) -> bool:
            prefix = base + "_"
            if not node_name.startswith(prefix):
                return False
            suffix = node_name[len(prefix) :]
            return suffix.isdigit()

        # base -> concrete node names for that base in topo order
        base_groups: dict[str, list[str]] = {b: [] for b in replica_bases}
        for n in order:
            for b in replica_bases:
                if _is_replica_node(b, n):
                    base_groups[b].append(n)
                    break

        def _is_readiness_task(node_name: str) -> bool:
            try:
                from sflow.core.probe import ProbeType

                t = tg.get_task(node_name)
                return any(
                    getattr(p, "type", None) == ProbeType.READINESS
                    for p in (t.probes or [])
                )
            except Exception:
                return False

        def _edge_label(from_node: str, to_node: str) -> str:
            # Replica chaining edges represent sequencing; label them Completed.
            for b in replica_bases:
                if _is_replica_node(b, from_node) and _is_replica_node(b, to_node):
                    return "Completed"
            return "Ready" if _is_readiness_task(from_node) else "Completed"

        def _unique_node_id(base: str) -> str:
            candidate = base
            i = 2
            while candidate in tg.dag.nodes:
                candidate = f"{base}_{i}"
                i += 1
            return candidate

        start_id = "start" if "start" not in tg.dag.nodes else _unique_node_id("start")
        stop_id = "stop" if "stop" not in tg.dag.nodes else _unique_node_id("stop")

        # For display order of subgraphs: keep the base order as config order.
        replica_base_order = [
            t.name for t in config.workflow.tasks if str(t.name) in replica_bases
        ]
        replica_base_order = [str(x) for x in replica_base_order]
        # Non-replica nodes in topo order
        non_replica_nodes = [
            n for n in order if not any(_is_replica_node(b, n) for b in replica_bases)
        ]

        def _mermaid() -> str:
            lines: list[str] = ["graph TD"]
            if show_variables and state.variables:
                for k, v in sorted(state.variables.items()):
                    lines.append(f"%% var {k}={v.value!r}")
            # SRD-like start/stop nodes
            lines.append(f"  {start_id}((start))")
            lines.append(f"  {stop_id}(((stop)))")
            lines.append("")

            # Subgraphs for replica bases
            for base in replica_base_order:
                members = base_groups.get(base, [])
                if not members:
                    continue
                lines.append(f'  subgraph "{base}"')
                for m in members:
                    lines.append(f"    {m}")
                # If there are explicit sequencing edges between replicas, render them inside the subgraph.
                for to_node in members:
                    for from_node in tg.dag.get_dependencies(to_node):
                        if from_node in members:
                            label = _edge_label(from_node, to_node)
                            lines.append(f"    {from_node} -- {label} --> {to_node}")
                lines.append("  end")
            if replica_base_order:
                lines.append("")

            # Regular (non-replica) task nodes with labels
            for node in non_replica_nodes:
                t = tg.get_task(node)
                label = f"{t.name}{_gpu_label(t)}"
                lines.append(f'  {t.name}["{label}"]')
            lines.append("")

            # Connect start -> sources (nodes with zero deps)
            sources = [n for n in order if len(tg.dag.get_dependencies(n)) == 0]
            for n in sources:
                lines.append(f"  {start_id} --> {n}")
            lines.append("")

            # Main dependency edges (exclude replica-internal edges already shown in subgraphs)
            for to_node in order:
                for from_node in tg.dag.get_dependencies(to_node):
                    # Skip internal edges if both in same replica group; already drawn above.
                    if any(
                        _is_replica_node(b, from_node) and _is_replica_node(b, to_node)
                        for b in replica_bases
                    ):
                        continue
                    label = _edge_label(from_node, to_node)
                    lines.append(f"  {from_node} -- {label} --> {to_node}")

            lines.append("")
            # Connect sinks -> stop
            sinks = [n for n in order if len(tg.dag.get_dependents(n)) == 0]
            for n in sinks:
                lines.append(f"  {n} -- Completed --> {stop_id}")
            lines.append("")
            return "\n".join(lines)

        def _dot() -> str:
            lines: list[str] = ['digraph "workflow" {', "  rankdir=LR;"]
            if show_variables and state.variables:
                lines.append('  subgraph "cluster_vars" {')
                lines.append('    label="variables";')
                for k, v in sorted(state.variables.items()):
                    safe = f"var_{k}"
                    lines.append(f'    {safe} [shape=note,label="{k}={v.value}"];')
                lines.append("  }")
            # Start/stop nodes
            lines.append(f'  "{start_id}" [shape=circle,label="start"];')
            lines.append(f'  "{stop_id}" [shape=doublecircle,label="stop"];')

            # Replica clusters (Graphviz subgraph clusters)
            for base in replica_base_order:
                members = base_groups.get(base, [])
                if not members:
                    continue
                lines.append(f'  subgraph "cluster_{base}" {{')
                lines.append(f'    label="{base}";')
                for m in members:
                    lines.append(f'    "{m}";')
                lines.append("  }")

            # Regular nodes
            for node in non_replica_nodes:
                t = tg.get_task(node)
                label = f"{t.name}{_gpu_label(t)}"
                lines.append(f'  "{t.name}" [label="{label}"];')

            # start -> sources
            sources = [n for n in order if len(tg.dag.get_dependencies(n)) == 0]
            for n in sources:
                lines.append(f'  "{start_id}" -> "{n}";')

            # dependency edges with labels
            for to_node in order:
                for from_node in tg.dag.get_dependencies(to_node):
                    label = _edge_label(from_node, to_node)
                    lines.append(f'  "{from_node}" -> "{to_node}" [label="{label}"];')

            # sinks -> stop
            sinks = [n for n in order if len(tg.dag.get_dependents(n)) == 0]
            for n in sinks:
                lines.append(f'  "{n}" -> "{stop_id}" [label="Completed"];')

            lines.append("}")
            lines.append("")
            return "\n".join(lines)

        # Write or render
        if fmt == "mermaid":
            output_path.write_text(_mermaid())
            return VisualizeResult(
                task_count=len(tasks),
                topo_order=order,
                saved_path=str(output_path),
                format="mermaid",
            )

        dot_text = _dot()
        if fmt == "dot":
            output_path.write_text(dot_text)
            return VisualizeResult(
                task_count=len(tasks),
                topo_order=order,
                saved_path=str(output_path),
                format="dot",
            )

        if fmt in {"png", "svg", "pdf"}:
            dot_bin = shutil.which("dot")
            if not dot_bin:
                raise ValueError(
                    "Graphviz `dot` is required for png/svg/pdf output. "
                    "Either install graphviz or use --format mermaid."
                )
            # Render via dot
            proc = subprocess.run(
                [dot_bin, f"-T{fmt}", "-o", str(output_path)],
                input=dot_text.encode("utf-8"),
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"dot failed with exit code {proc.returncode}")
            return VisualizeResult(
                task_count=len(tasks),
                topo_order=order,
                saved_path=str(output_path),
                format=fmt,
            )
        raise ValueError(f"Unsupported format: {format}")
