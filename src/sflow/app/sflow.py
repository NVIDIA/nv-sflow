# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sflow.app.assembly import build_state, release_backends
from sflow.app.monitor_cli import apply_monitor_cli_overrides
from sflow.app.monitor_planner import run_monitor_postprocess
from sflow.app.run_support import (
    build_run_paths,
    collect_operator_runtime_warnings,
    config_uses_offhost_backend,
    configure_task_runtime,
    find_reserved_env_collisions,
    preflight_validate_artifacts,
    validate_container_mounts,
)
from sflow.config.loader import ConfigLoader
from sflow.core.command_log import (
    CommandLogRouter,
    reset_active_command_log_router,
    set_active_command_log_router,
)
from sflow.core.execution_summary import SflowSummaryWriter
from sflow.core.uploads import UploadResult
from sflow.logging import (
    CoalescingFileHandler,
    DeferredTaskLogHandler,
    add_log_file,
    enable_console_logging,
    get_logger,
)
from sflow.runtime_info import format_runtime_info
from sflow.utils.container import collect_container_mounts
from sflow.utils.logging import (
    build_allocation_map_lines,
    build_resource_rehearsal_lines,
    log_dry_run_envelope,
    log_dry_run_section,
)

if TYPE_CHECKING:
    from sflow.config.schema import SflowConfig

_logger = get_logger(__name__)


def _merge_backend_extra_args(
    config: SflowConfig,
    extra_args_by_type: dict[str, list[str]],
) -> SflowConfig:
    """Merge CLI-provided extra args into matching backends, keyed by backend type.

    ``extra_args_by_type`` maps a backend ``type`` (e.g. ``"slurm"`` / ``"docker"``)
    to the CLI args for that backend kind, so ``--extra-salloc-args`` only touches
    Slurm backends and ``--extra-docker-args`` only docker backends. Each backend
    de-dups the args by option against its own config ``extra_args`` (CLI wins).
    """
    if not config.backends or not extra_args_by_type:
        return config

    updated_backends = []
    merged_any = False
    for b in config.backends:
        extra = extra_args_by_type.get(getattr(b, "type", None))
        merge_extra_args = getattr(b, "merge_extra_args", None)
        if extra and callable(merge_extra_args):
            updated = merge_extra_args(extra)
        else:
            updated = b
        updated_backends.append(updated)
        if updated is not b:
            merged_any = True

    if merged_any:
        _logger.info(
            f"Merged CLI extra args into backend(s) by type: {extra_args_by_type}"
        )
        return config.model_copy(update={"backends": updated_backends})
    return config


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
        file: Path | list[Path],
        dry_run: bool = False,
        quiet: bool = False,
        # Dry-run only: when False (default) the Tasks section prints a compact
        # one-line summary per task; when True it prints the full per-task detail
        # (nodelist, output dir, probes, operator config, ...).
        verbose: bool = False,
        resume: str | None = None,
        variable_overrides: list[str] | None = None,
        artifact_overrides: list[str] | None = None,
        missable_tasks: list[str] | None = None,
        # CLI-provided backend extra args keyed by backend type:
        # {"slurm": [...salloc args...], "docker": [...docker run args...]}.
        backend_extra_args_by_type: dict[str, list[str]] | None = None,
        # CLI-level Kubernetes access (KubectlConfig) from `sflow run` flags;
        # applied to kubernetes backends so recipes stay cluster-agnostic.
        kubectl_config: Any | None = None,
        # Enable default hardware monitors via CLI without editing the recipe.
        enable_workflow_monitor: bool = False,
        enable_task_monitors: list[str] | None = None,
        workspace_dir: Path | None = None,
        output_dir: Path | None = None,
        # Slurm sbatch stdout/stderr paths, only set when invoked via `sflow batch`.
        # Surfaced in the dry-run Plan so users can see where the job's logs will land.
        sbatch_output: str | None = None,
        sbatch_error: str | None = None,
        tui: bool = False,
        tui_log_buffer: deque[logging.LogRecord] | None = None,
        tui_log_lock: threading.Lock | None = None,
        tui_refresh_per_second: int | None = None,
    ) -> Path | None:
        """
        Run the workflow and return the workflow output directory path.

        Returns:
            Path to the workflow output directory (e.g., sflow_output/<workflow-name>-<timestamp>-<id>),
            or None for dry-run mode.
        """
        import asyncio
        if resume is not None:
            raise NotImplementedError("--resume is not implemented yet")

        # Reset from previous runs
        self.last_workflow_output_dir = None

        # load the config (supports single path or multiple paths for merging)
        files = [file] if isinstance(file, Path) else list(file)
        _loader = ConfigLoader()
        config = _loader.load_configs(
            files, variable_overrides, artifact_overrides, missable_tasks
        )
        _missable_stripped = _loader.missable_stripped

        if backend_extra_args_by_type:
            config = _merge_backend_extra_args(config, backend_extra_args_by_type)

        # Export KUBECONFIG so kubectl invoked directly inside user task scripts
        # (not just sflow's own calls) also targets the selected cluster. sflow's
        # own calls additionally pass --kubeconfig/--context explicitly.
        if kubectl_config is not None and getattr(kubectl_config, "kubeconfig", None):
            os.environ["KUBECONFIG"] = str(kubectl_config.kubeconfig)

        config = apply_monitor_cli_overrides(
            config,
            enable_workflow_monitor=enable_workflow_monitor,
            enable_task_monitors=enable_task_monitors,
        )

        async def _run_async() -> Path | None:
            import atexit
            import contextlib
            import signal
            from contextlib import suppress

            from sflow.core.orchestrator import Orchestrator
            from sflow.ui.rich_tui import RichTui, RichTuiConfig

            ui: RichTui | None = None
            ui_task: asyncio.Task | None = None
            ui_torn_down = False
            orch: Orchestrator | None = None
            received_signal: signal.Signals | None = None
            atexit_cleaned = False
            owned_backend_allocations: list[tuple[Any, Any]] = []
            summary_writer: SflowSummaryWriter | None = None
            command_log_router: CommandLogRouter | None = None
            command_log_token = None

            async def _ui_loop() -> None:
                # Refresh at a higher rate than Orchestrator poll_interval so logs feel like tail -f.
                # Use the same rate as Live refresh to avoid over-refreshing.
                refresh_hz = 10
                if ui is not None:
                    try:
                        refresh_hz = int(
                            getattr(
                                getattr(ui, "_config", None), "refresh_per_second", 10
                            )
                        )
                    except Exception:
                        refresh_hz = 10
                sleep_s = 0.1 if refresh_hz <= 0 else max(0.01, 1.0 / float(refresh_hz))
                while True:
                    if ui is not None:
                        ui.refresh()
                        if ui.workflow is not None and ui.workflow.is_finished():
                            return
                    await asyncio.sleep(sleep_s)

            async def _teardown_ui() -> None:
                # Idempotently stop the live TUI (refresh loop + Textual app) and
                # resume console logging. Called BEFORE the deferred monitor
                # post-process so its progress hint + report path surface on the
                # plain terminal (just before the CLI's final line), and again via
                # the cleanup stack as a backstop. The TUI runs with the console
                # handler disabled, so without re-enabling it those logs would be
                # invisible. A no-op when no TUI is active.
                nonlocal ui_torn_down
                if ui is None or ui_torn_down:
                    return
                ui_torn_down = True
                if ui_task is not None:
                    ui_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ui_task
                with contextlib.suppress(Exception):
                    await ui.stop_async()
                with contextlib.suppress(Exception):
                    enable_console_logging()

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
                    log_lock=tui_log_lock,
                    attach_log_handler=False if tui_log_buffer is not None else True,
                )

            async with contextlib.AsyncExitStack() as cleanup_stack:
                if ui is not None:
                    await ui.start_async()
                    ui.refresh()
                    ui_task = asyncio.create_task(_ui_loop())
                    # One idempotent teardown handles the refresh loop, the Textual
                    # app, and resuming console logging.
                    cleanup_stack.push_async_callback(_teardown_ui)

                # Workspace/output dirs are needed early for artifact resolution.
                run_paths = build_run_paths(
                    workflow_name=config.workflow.name,
                    dry_run=dry_run,
                    workspace_dir=workspace_dir,
                    output_dir=output_dir,
                    run_id_prefix=os.environ.get("SFLOW_RUN_ID_PREFIX")
                    or os.environ.get("SLURM_JOB_ID")
                    or os.environ.get("SLURM_JOBID"),
                )
                ws_dir = run_paths.workspace_dir
                out_dir = run_paths.output_dir
                workflow_out_dir = run_paths.workflow_output_dir

                # Inline file:// artifacts are written under workflow_out_dir
                # rather than the workspace when this is a real run.
                if not dry_run:
                    self.last_workflow_output_dir = workflow_out_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    workflow_out_dir.mkdir(parents=True, exist_ok=True)
                    command_log_router = CommandLogRouter(workflow_out_dir)
                    command_log_token = set_active_command_log_router(
                        command_log_router
                    )
                    cleanup_stack.callback(
                        reset_active_command_log_router,
                        command_log_token,
                    )

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
                if ui is not None:
                    ui.set_interrupt_handler(lambda: _on_signal(signal.SIGINT))

                # Pre-flight: validate artifact paths before allocation.
                # fs:// artifacts must already exist; fail early so we never waste
                # backend resources on a missing path -- except for off-host backends
                # (e.g. Kubernetes), where fs:// paths live on the remote cluster/image
                # and are therefore only warned about, not validated locally.
                _artifact_warnings = preflight_validate_artifacts(
                    config.artifacts,
                    config.variables,
                    workspace_dir=ws_dir,
                    dry_run=dry_run,
                    skip_local_fs_validation=config_uses_offhost_backend(config),
                )

                # build the state:
                # - dry-run: never allocates
                allocate = not dry_run
                state = None
                try:
                    build_kw: dict[str, Any] = {
                        "allocate": allocate,
                        "output_dir": workflow_out_dir,
                        "source_files": files,
                        "kubectl_config": kubectl_config,
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
                                owned_backend_allocations.append((b, alloc))
                    except Exception:
                        owned_backend_allocations = []

                    def _atexit_cleanup() -> None:
                        nonlocal atexit_cleaned
                        if atexit_cleaned:
                            return
                        for backend, allocation in owned_backend_allocations:
                            try:
                                backend.emergency_release(allocation)
                            except Exception:
                                pass
                        atexit_cleaned = True

                    atexit.register(_atexit_cleanup)

                # -----------------------------------------------------------------
                # Output directory structure + built-in envs (SRD REQ-1.4 / REQ-4.4)
                # -----------------------------------------------------------------
                tg = state.workflow.task_graph

                if not dry_run:
                    # Add a global sflow log file under the workflow output dir.
                    add_log_file(str(workflow_out_dir / "sflow.log"))

                for t in tg.get_tasks():
                    task_out_dir = configure_task_runtime(
                        t,
                        ws_dir=ws_dir,
                        out_dir=out_dir,
                        workflow_out_dir=workflow_out_dir,
                        dry_run=dry_run,
                    )

                    if not dry_run:
                        # The per-task <task>.log is the single source of truth
                        # for a task's output. In offload mode the operator writes
                        # it directly (srun --output / shell redirect), so sflow
                        # must not open the same file concurrently (single-writer):
                        # instead it buffers the launcher's captured driver-side
                        # diagnostics and appends them to the SAME <task>.log once
                        # the operator releases it (DeferredTaskLogHandler) -- no
                        # scattered <task>.orchestration.log sidecar.
                        offload = False
                        try:
                            offload = bool(t.operator.writes_own_task_log())
                        except Exception:
                            offload = False
                        log_path = task_out_dir / f"{t.name}.log"
                        already = any(
                            getattr(h, "baseFilename", None) == str(log_path)
                            for h in t.logger.handlers
                            if isinstance(
                                h, (logging.FileHandler, DeferredTaskLogHandler)
                            )
                        )
                        if not already:
                            handler: logging.Handler = (
                                DeferredTaskLogHandler(str(log_path))
                                if offload
                                else CoalescingFileHandler(log_path)
                            )
                            handler.setFormatter(
                                logging.Formatter(
                                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                                )
                            )
                            t.logger.addHandler(handler)
                        t.logger.setLevel(logging.INFO)
                        t.logger.propagate = False

                if dry_run:
                    plan_tasks = tg.get_tasks()
                    order = tg.dag.topological_sort()

                    # Validate container mount paths (REQ: warn users about invalid mounts)
                    mount_warnings = validate_container_mounts(
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
                    log_dry_run_envelope(f"Dry-run: {config.workflow.name}")

                    if len(files) > 1 and _loader.file_contributions:
                        log_dry_run_section(
                            f"Input files ({len(files)} → merged workflow)"
                        )
                        for i, contrib in enumerate(_loader.file_contributions):
                            is_last = i == len(_loader.file_contributions) - 1
                            fname = contrib["path"].name
                            parent = contrib["path"].parent.name
                            label = (
                                f"{parent}/{fname}"
                                if parent and parent != "."
                                else fname
                            )
                            connector = "└─" if is_last else "├─"
                            arrow = " ──►" if is_last else ""
                            _logger.info(
                                f"  {connector} {label}{arrow} {config.workflow.name}"
                                if is_last
                                else f"  {connector} {label}"
                            )
                            for sec_name, sec_items in contrib["sections"]:
                                branch = " " if is_last else "│"
                                items_str = ", ".join(sec_items[:5])
                                if len(sec_items) > 5:
                                    items_str += f", … (+{len(sec_items) - 5})"
                                _logger.info(f"  {branch}    {sec_name}: [{items_str}]")
                        _logger.info("")

                    if _missable_stripped:
                        _logger.warning(
                            f"Missable tasks: removed {len(_missable_stripped)} reference(s) to absent tasks:"
                        )
                        for _ms in _missable_stripped:
                            _logger.warning(f"  ⚠ {_ms}")
                    log_dry_run_section("Plan")
                    pad = 21
                    _logger.info(f"  {'workspace_dir:':<{pad}}{ws_dir}")
                    _logger.info(f"  {'output_dir:':<{pad}}{out_dir}")
                    _logger.info(f"  {'workflow_output_dir:':<{pad}}{workflow_out_dir}")
                    if sbatch_output is not None:
                        _logger.info(f"  {'sbatch out:':<{pad}}{sbatch_output}")
                    if sbatch_error is not None:
                        _logger.info(f"  {'sbatch err:':<{pad}}{sbatch_error}")
                    _logger.info(
                        f"  {'tasks:':<{pad}}{len(plan_tasks)} "
                        f"(order: {', '.join(order)})"
                    )
                    _logger.info(
                        f"  {'backends defined:':<{pad}}"
                        f"{', '.join(sorted(state.backends.keys()))}"
                    )
                    if used_backends:
                        _logger.info(
                            f"  {'backends used:':<{pad}}{', '.join(used_backends)}"
                        )

                    # Variable / artifact overrides (CLI --set / --artifact) get their
                    # own section so the Plan stays a fixed set of run facts.
                    if variable_overrides or artifact_overrides:
                        log_dry_run_section("Overrides")
                        if variable_overrides:
                            _logger.info("  variables:")
                            for var_override in variable_overrides:
                                if "=" in var_override:
                                    key, value = var_override.split("=", 1)
                                    value_stripped = value.strip()
                                    if value_stripped.startswith(
                                        "["
                                    ) and value_stripped.endswith("]"):
                                        # Domain override (list)
                                        _logger.info(
                                            f"    {key} = {value}  (domain sweep)"
                                        )
                                    else:
                                        # Single value override
                                        _logger.info(f"    {key} = {value}")
                                else:
                                    _logger.info(f"    {var_override}")
                        if artifact_overrides:
                            _logger.info("  artifacts:")
                            for art_override in artifact_overrides:
                                _logger.info(f"    {art_override}")

                    # Warn when user-declared variables reuse a reserved sflow env
                    # var name: sflow injects/owns these at launch, so the collision
                    # leads to undefined behavior (the variable or sflow's value may
                    # win depending on the var). Surface it during dry-run validation.
                    reserved_collisions = find_reserved_env_collisions(
                        state.variables or {}
                    )
                    if reserved_collisions:
                        log_dry_run_section("Reserved env collisions")
                        _logger.warning(
                            "User-defined variables reuse reserved sflow env var "
                            "names; rename them to avoid undefined behavior at task "
                            "launch:"
                        )
                        for name in reserved_collisions:
                            _logger.warning(f"  ⚠ {name}")

                    log_dry_run_section("Backends")
                    for b_name, backend in state.backends.items():
                        b_type = backend.__class__.__name__
                        alloc = backend.allocation
                        if alloc is None:
                            _logger.info(
                                f"  [{b_name}] type={b_type}, allocated=no (dry-run)"
                            )
                        else:
                            nodes = [n.name for n in alloc.nodes]
                            _logger.info(
                                f"  [{b_name}] type={b_type}, allocated=yes, "
                                f"id={alloc.allocation_id}, nodes={nodes}"
                            )

                        details = backend.dry_run_details()
                        if details:
                            for i, (key, val) in enumerate(details):
                                prefix = "└─" if i == len(details) - 1 else "├─"
                                _logger.info(f"      {prefix} {key}: {val}")

                    allocation_map_lines = build_allocation_map_lines(
                        plan_tasks, state.backends
                    )
                    if allocation_map_lines:
                        log_dry_run_section("Allocation map")
                        for line in allocation_map_lines:
                            _logger.info(line)

                    resource_rehearsal_lines = build_resource_rehearsal_lines(
                        [tg.get_task(name) for name in order]
                    )
                    if resource_rehearsal_lines:
                        log_dry_run_section("Resource Occupancy")
                        for line in resource_rehearsal_lines:
                            _logger.info(line)

                    all_mounts: set[str] = set()
                    for task_name in order:
                        task = tg.get_task(task_name)
                        task_op_conf = getattr(
                            getattr(task, "operator", None), "config", None
                        )
                        if task_op_conf is not None:
                            for mount in collect_container_mounts(task_op_conf):
                                if "sflow_output" in mount.lower():
                                    continue
                                all_mounts.add(mount)
                    if all_mounts:
                        log_dry_run_section("Container mounts")
                        for mount in sorted(all_mounts):
                            _logger.info(f"  - {mount}")

                    log_dry_run_section("Workflow DAG")
                    dag_lines = tg.dag.render_ascii()
                    for dag_line in dag_lines:
                        _logger.info(f"  {dag_line}")

                    if not quiet:
                        log_dry_run_section("Tasks")
                        for idx, name in enumerate(order, 1):
                            t = tg.get_task(name)
                            deps = tg.dag.get_dependencies(name)
                            op_conf = getattr(
                                getattr(t, "operator", None), "config", None
                            )
                            op_type_str = getattr(op_conf, "type", None) or "unknown"

                            nodelist = getattr(op_conf, "nodelist", None) or []
                            cuda_visible = t.envs.get("CUDA_VISIBLE_DEVICES")
                            task_out_dir = t.envs.get("SFLOW_TASK_OUTPUT_DIR")
                            retry = getattr(t, "retries", None)
                            retry_str = (
                                f"{retry.count}x, interval={retry.interval}, backoff={retry.backoff}"
                                if retry is not None
                                else "none"
                            )

                            # Default (non-verbose): one compact line per task. Extra
                            # facts (retries/probes/sweep) are appended only when set,
                            # so simple tasks stay terse. Full detail is gated on
                            # --verbose below.
                            if not verbose:
                                short_parts = [
                                    f"backend={getattr(t, 'backend_name', None)}",
                                    f"operator={op_type_str}",
                                    f"depends_on={list(deps) if deps else '[]'}",
                                ]
                                if retry is not None:
                                    short_parts.append(f"retries={retry.count}x")
                                if t.probes:
                                    short_parts.append(f"probes={len(t.probes)}")
                                if t.sweep_variables:
                                    # Show the swept variable(s) and this replica's
                                    # value(s) — that's what makes the replica distinct
                                    # (e.g. benchmark_64 → CONCURRENCY=64) — rather than
                                    # a bare count. Braces group multiple sweep vars.
                                    sweep_vals = ", ".join(
                                        f"{k}={t.envs.get(k, '')}"
                                        for k in t.sweep_variables
                                    )
                                    short_parts.append(f"sweep={{{sweep_vals}}}")
                                _logger.info(
                                    f"  [{idx}] {t.name}  ({', '.join(short_parts)})"
                                )
                                continue

                            _logger.info("")
                            _logger.info(f"  [{idx}] {t.name}")
                            _logger.info(
                                f"      ├─ backend: {getattr(t, 'backend_name', None)}"
                            )
                            _logger.info(f"      ├─ operator: {op_type_str}")
                            _logger.info(
                                f"      ├─ depends_on: {list(deps) if deps else '[]'}"
                            )
                            _logger.info(f"      ├─ nodelist: {nodelist}")
                            if cuda_visible:
                                _logger.info(
                                    f"      ├─ CUDA_VISIBLE_DEVICES: {cuda_visible}"
                                )
                            _logger.info(f"      ├─ task_output_dir: {task_out_dir}")
                            _logger.info(f"      ├─ retries: {retry_str}")

                            if t.sweep_variables:
                                sweep_vals = {
                                    k: t.envs.get(k, "") for k in t.sweep_variables
                                }
                                sweep_items = ", ".join(
                                    f"{k}={v}" for k, v in sweep_vals.items()
                                )
                                _logger.info(f"      ├─ sweep_vars: {{{sweep_items}}}")

                            if t.probes:
                                _logger.info("      ├─ probes:")
                                for pi, probe in enumerate(t.probes):
                                    is_last_probe = pi == len(t.probes) - 1
                                    p_prefix = "└─" if is_last_probe else "├─"
                                    probe_type = str(probe.type)
                                    cls_name = probe.__class__.__name__
                                    details: list[str] = []
                                    if hasattr(probe, "_host") and hasattr(
                                        probe, "_port"
                                    ):
                                        kind = "tcp_port"
                                        details.append(
                                            f"host={probe._host} (fake ip when dry-run, real ip when running)"
                                        )
                                        details.append(f"port={probe._port}")
                                        on_node = getattr(probe, "_on_node", None)
                                        if on_node:
                                            details.append(f"on_node={on_node}")
                                    elif hasattr(probe, "_url"):
                                        kind = (
                                            "http_get"
                                            if "Get" in cls_name
                                            else "http_post"
                                        )
                                        details.append(
                                            f"url={probe._url} (fake ip when dry-run, real ip when running)"
                                        )
                                    elif hasattr(probe, "_regex"):
                                        kind = "log_watch"
                                        pat = (
                                            getattr(probe, "_pattern_display", None)
                                            or probe._regex.pattern
                                        )
                                        details.append(f"pattern={pat}")
                                        mc = getattr(probe, "_match_count", 1)
                                        if mc != 1:
                                            details.append(f"match_count={mc}")
                                        logger_name = getattr(
                                            probe, "_logger_task_name", None
                                        )
                                        if logger_name:
                                            details.append(f"logger={logger_name}")
                                    else:
                                        kind = cls_name
                                    detail_str = (
                                        f" ({', '.join(details)})" if details else ""
                                    )
                                    timing = (
                                        f"delay={probe.delay}s, timeout={probe.timeout}s, "
                                        f"interval={probe.interval}s"
                                    )
                                    connector = "   " if is_last_probe else "│  "
                                    _logger.info(
                                        f"         {p_prefix} {probe_type}: {kind}{detail_str}"
                                    )
                                    _logger.info(f"         {connector}  {timing}")

                            if op_conf is not None:
                                op_details: list[tuple[str, str]] = []
                                if getattr(op_conf, "nodes", None) is not None:
                                    op_details.append(("nodes", str(op_conf.nodes)))
                                if getattr(op_conf, "ntasks", None) is not None:
                                    op_details.append(("ntasks", str(op_conf.ntasks)))
                                if (
                                    getattr(op_conf, "ntasks_per_node", None)
                                    is not None
                                ):
                                    op_details.append(
                                        (
                                            "ntasks_per_node",
                                            str(op_conf.ntasks_per_node),
                                        )
                                    )
                                if getattr(op_conf, "cpus_per_task", None) is not None:
                                    op_details.append(
                                        ("cpus_per_task", str(op_conf.cpus_per_task))
                                    )
                                if getattr(op_conf, "gpus", None) is not None:
                                    op_details.append(("gpus", str(op_conf.gpus)))
                                if getattr(op_conf, "gpus_per_task", None) is not None:
                                    op_details.append(
                                        ("gpus_per_task", str(op_conf.gpus_per_task))
                                    )
                                if (
                                    getattr(op_conf, "container_image", None)
                                    is not None
                                ):
                                    op_details.append(
                                        ("container_image", op_conf.container_image)
                                    )
                                if getattr(op_conf, "container_name", None) is not None:
                                    op_details.append(
                                        ("container_name", op_conf.container_name)
                                    )
                                mounts = collect_container_mounts(op_conf)
                                if mounts:
                                    if len(mounts) <= 3:
                                        op_details.append(
                                            ("mounts", str(mounts))
                                        )
                                    else:
                                        op_details.append(
                                            (
                                                "mounts",
                                                f"[{len(mounts)} mounts]",
                                            )
                                        )
                                if getattr(op_conf, "mpi", None) is not None:
                                    op_details.append(("mpi", op_conf.mpi))
                                if (
                                    getattr(op_conf, "job_id", None) is not None
                                    and getattr(op_conf, "job_id", None) != "0"
                                ):
                                    op_details.append(("job_id", str(op_conf.job_id)))
                                if getattr(op_conf, "extra_args", None):
                                    extra_args_list = list(op_conf.extra_args)
                                    if len(extra_args_list) <= 5:
                                        op_details.append(
                                            ("extra_args", str(extra_args_list))
                                        )
                                    else:
                                        op_details.append(
                                            (
                                                "extra_args",
                                                f"[{len(extra_args_list)} args]",
                                            )
                                        )

                                if op_details:
                                    _logger.info("      └─ operator config:")
                                    for i, (key, val) in enumerate(op_details):
                                        prefix = (
                                            "└─" if i == len(op_details) - 1 else "├─"
                                        )
                                        _logger.info(f"         {prefix} {key}: {val}")
                                else:
                                    _logger.info("      └─ operator config: (default)")

                        if not verbose:
                            _logger.info("")
                            _logger.info("  (use --verbose for full per-task details)")

                    for warning in collect_operator_runtime_warnings(plan_tasks):
                        _logger.warning(f"  ⚠ {warning}")

                    if _artifact_warnings:
                        _logger.warning("")
                        _logger.warning(
                            "Artifact path warnings (non-existent fs:// / file:// paths):"
                        )
                        for w in _artifact_warnings:
                            _logger.warning(f"  ⚠ {w}")
                        _logger.warning(
                            "These paths must exist before the workflow is run."
                        )

                    # Storage targets + planned uploads (dry-run only).
                    if state.storage_targets:
                        log_dry_run_section("Storage targets")
                        for name, target in state.storage_targets.items():
                            _logger.info(f"  [{name}] {type(target).__name__}")
                            # Surface offline credential/SDK warnings (e.g. S3 with
                            # no boto3 or no AWS credentials) before a real run.
                            for w in target.dry_run_warnings():
                                _logger.warning(f"    ⚠ {w}")
                        upload_tasks = [
                            t for t in plan_tasks if getattr(t, "uploads", None)
                        ]
                        if upload_tasks:
                            log_dry_run_section("Planned uploads")
                            for t in upload_tasks:
                                _logger.info(f"  {t.name}:")
                                for i, u in enumerate(t.uploads):
                                    to_desc = u.to_expr or "<basename>"
                                    meta = f"on_error={u.on_error}"
                                    if getattr(u, "disambiguate_with", None):
                                        meta += ", auto-renamed per replica"
                                    _logger.info(f"    [{i}] {u.from_expr}")
                                    _logger.info(
                                        f"         → {u.target}:{to_desc}  ({meta})"
                                    )

                        if state.workflow_upload is not None:
                            wu = state.workflow_upload
                            to_desc = wu.to_expr or "<run_id>.zip"
                            log_dry_run_section("Planned workflow upload")
                            _logger.info(
                                f"  → {wu.target}:{to_desc}  (on_error={wu.on_error})"
                            )

                    if state.monitor_registry is not None:
                        reg = state.monitor_registry
                        log_dry_run_section("Planned monitors")
                        _logger.info(f"  {'output dir:':<21}{reg.out_dir}")
                        _logger.info(
                            f"  {'node collectors:':<21}{reg.collector_count} "
                            f"(deduped, one per node)"
                        )
                        for consumer in reg.consumers:
                            gpu_str = (
                                "all"
                                if consumer.gpus is None
                                else ",".join(str(g) for g in consumer.gpus)
                            )
                            _logger.info(f"  [{consumer.name}] ({consumer.owner})")
                            _logger.info(
                                f"      nodes={', '.join(consumer.nodes) or 'none'} "
                                f"gpus={gpu_str} scopes={','.join(consumer.scopes) or 'all'} "
                                f"report={'yes' if consumer.report else 'no'}"
                            )

                    log_dry_run_envelope(f"Dry-run complete: {config.workflow.name}")

                    return None  # dry-run: no actual output directory created

                # run the workflow and always release backend allocations
                monitor_postprocessed = False
                try:
                    if not dry_run:
                        command_log_paths = (
                            command_log_router.planned_paths()
                            if command_log_router is not None
                            else {}
                        )
                        summary_writer = SflowSummaryWriter(
                            workflow_out_dir / "sflow_summary.log"
                        )
                        summary_writer.start(
                            workflow=state.workflow,
                            output_dir=workflow_out_dir,
                            runtime_info_text=format_runtime_info(),
                            command_log_paths=command_log_paths,
                        )

                    orch = Orchestrator(
                        workflow=state.workflow,
                        poll_interval=1,
                        execution_summary=summary_writer,
                        storage_targets=state.storage_targets,
                        monitor_registry=state.monitor_registry,
                    )
                    # Start the workflow-level hardware monitor (covers the whole
                    # pool for the workflow lifetime). Task monitors are fired by
                    # the orchestrator. Singleton-per-node dedup is handled by the
                    # registry refcount.
                    if (
                        state.monitor_registry is not None
                        and state.workflow_monitor is not None
                    ):
                        await state.monitor_registry.acquire(state.workflow_monitor)

                    await orch.run()
                    # Give any loop-level signal callback queued during the final
                    # orchestrator turn a chance to set received_signal before
                    # task-status post-processing.
                    await asyncio.sleep(0)

                    # Tear the live TUI down now that the workload is finished, so the
                    # deferred monitor post-process (and its progress hint) render on
                    # the plain terminal -- right before the CLI's final completion
                    # line -- instead of being hidden behind / lost with the TUI.
                    await _teardown_ui()

                    # Stop all hardware monitor collectors (workflow + any lingering
                    # task monitors) now that the workload is done, then run the
                    # single deferred post-process pass (overview + per-consumer
                    # reports) BEFORE upload_all so reports ship with the archive.
                    if state.monitor_registry is not None:
                        if state.workflow_monitor is not None:
                            await state.monitor_registry.release(state.workflow_monitor)
                        await state.monitor_registry.shutdown()
                        run_monitor_postprocess(state.monitor_registry)
                        monitor_postprocessed = True

                    if received_signal is not None:
                        if received_signal == signal.SIGINT:
                            raise KeyboardInterrupt()
                        raise SystemExit(128 + int(received_signal))

                    # Workflow-level upload (zip the whole output dir). Runs regardless
                    # of task success/failure so partial results can still be shipped;
                    # individual on_error governs whether a failed upload propagates.
                    wf_upload_results: list[UploadResult] = []
                    if state.workflow_upload is not None:
                        from sflow.core.uploads import run_workflow_upload
                        from sflow.core.variable import build_variables_ctx

                        ok = await run_workflow_upload(
                            state.workflow_upload,
                            workflow_name=config.workflow.name,
                            workflow_out_dir=workflow_out_dir,
                            storage_targets=state.storage_targets,
                            variables_ctx=build_variables_ctx(state.variables),
                            results=wf_upload_results,
                        )
                        if summary_writer is not None:
                            summary_writer.record_uploads(wf_upload_results)
                        if not ok:
                            detail = (
                                f"Workflow '{config.workflow.name}' upload_all failed"
                            )
                            if summary_writer is not None:
                                summary_writer.workflow_finished(
                                    status="FAILED",
                                    detail=detail,
                                )
                                summary_writer.flush()
                            _logger.error(
                                "workflow.upload_all failed (on_error=fail); "
                                "treating workflow run as failed."
                            )
                            raise RuntimeError(detail)

                    # Consolidated end-of-run upload report: per-task uploads were
                    # recorded by the orchestrator, workflow upload just above.
                    # Re-flush so the summary file's Uploads section includes the
                    # workflow-level archive; the CLI prints that section in its
                    # final artifact block so upload details appear at the end.
                    if summary_writer is not None:
                        summary_writer.flush()

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
                        detail = (
                            f"Workflow '{config.workflow.name}' failed: "
                            f"{len(failed)} task(s) failed ({names})"
                        )
                        if summary_writer is not None:
                            summary_writer.workflow_finished(
                                status="FAILED", detail=detail
                            )
                        raise RuntimeError(
                            detail
                        )
                    # Treat cancellations as non-success (covers fail-fast dependents and future user-cancel).
                    if cancelled:
                        names = ", ".join(t.name for t in cancelled)
                        detail = (
                            f"Workflow '{config.workflow.name}' cancelled: "
                            f"{len(cancelled)} task(s) cancelled ({names})"
                        )
                        if summary_writer is not None:
                            summary_writer.workflow_finished(
                                status="CANCELLED", detail=detail
                            )
                        if received_signal is not None:
                            if received_signal == signal.SIGINT:
                                raise KeyboardInterrupt()
                            raise SystemExit(128 + int(received_signal))
                        raise RuntimeError(
                            detail
                        )
                finally:
                    # Stop the TUI (and resume console logging) before any deferred
                    # monitor post-process below, so its output is visible even when
                    # run() raised. Idempotent with the success-path call above.
                    await _teardown_ui()
                    # Defensive: ensure no monitor collectors linger, and ALWAYS emit
                    # the monitor overview -- even if run() raised before the normal
                    # post-process above -- so monitor reporting is independent of the
                    # workflow's finish status. Best-effort; never masks the real error.
                    if (
                        state is not None
                        and getattr(state, "monitor_registry", None) is not None
                    ):
                        with suppress(Exception):
                            if state.workflow_monitor is not None:
                                await state.monitor_registry.release(
                                    state.workflow_monitor
                                )
                            await state.monitor_registry.shutdown()
                        if not monitor_postprocessed:
                            with suppress(Exception):
                                run_monitor_postprocess(state.monitor_registry)
                                monitor_postprocessed = True
                    # Always attempt to release owned backend allocations.
                    try:
                        await release_backends(state)
                    finally:
                        atexit_cleaned = True
                        # Remove signal handlers (SflowApp can be reused in-process in tests).
                        for sig in installed_signals:
                            with suppress(Exception):
                                loop.remove_signal_handler(sig)
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
        file: Path | list[Path],
        output_path: Path | None = None,
        format: str = "mermaid",
        show_variables: bool = False,
        variable_overrides: list[str] | None = None,
        artifact_overrides: list[str] | None = None,
        missable_tasks: list[str] | None = None,
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

        files = [file] if isinstance(file, Path) else list(file)
        config = ConfigLoader().load_configs(
            files, variable_overrides, artifact_overrides, missable_tasks
        )
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
