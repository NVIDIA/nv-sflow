# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one sflow task as one OSMO workflow through the ``osmo`` CLI.

Flow: submit -> ``osmo workflow logs --task`` into the task log -> ``query`` for
the exit code -> ``cancel`` if sflow tears the task down first. The CLI owns
auth (``osmo login``), so sflow stores no OSMO credentials.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shlex
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, Field

from sflow.core.command import Command
from sflow.core.operator import Operator, OperatorConfig
from sflow.core.operator_registry import register_operator
from sflow.logging import get_logger

_logger = get_logger(__name__)

OSMO_TASK = "main"
_SCRIPT_PATH = "/tmp/sflow_task.sh"
_UNSAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]+")
_MAX_QUERY_FAILURES = 5
# Run as ``python -m``; not imported here, so runpy does not warn that the module
# is already loaded. Keep _LOG_STREAM_CUT equal to osmo_logs.CUT.
_LOG_STREAM_MODULE = "sflow.plugins.operators.osmo_logs"
_LOG_STREAM_CUT = 75
# Bound on every osmo CLI call, so a hung endpoint cannot stall a task or teardown.
_CLI_TIMEOUT_S = 120.0
# Workflow statuses that mean the task waits in the queue; see _wait_while_queued.
_QUEUED = ("PENDING", "WAITING")
# While queued, log the state on change and at least this often.
_QUEUE_LOG_EVERY_S = 60.0
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Driver-host paths that scripts write into; create them in the container like the
# k8s operator does. ponytail: files written there stay in OSMO; collecting them is
# PR 2 (map to {{output}}).
_OUTPUT_DIR_VARS = (
    "SFLOW_OUTPUT_DIR",
    "SFLOW_WORKFLOW_OUTPUT_DIR",
    "SFLOW_TASK_OUTPUT_DIR",
)


class OsmoOperatorConfig(OperatorConfig):
    """Workload settings for one OSMO task. Pool and platform come from the backend."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["osmo"] = "osmo"
    image: str
    cpu: int = 1
    memory: str = "4Gi"
    storage: str = "10Gi"
    # Overrides the backend's platform (OSMO node type, e.g. ``ovx-l40``).
    platform: str | None = None
    # Seconds between ``osmo workflow query`` calls while waiting for the result.
    poll_interval: float = Field(default=10.0, gt=0)

    def container_images(self) -> list[str]:
        """Return the workload image, so pre-flight validates the reference."""
        return [self.image]


def osmo_name(raw: str) -> str:
    """OSMO names must match ``[a-zA-Z]([a-zA-Z0-9_-]*[a-zA-Z0-9])?``."""
    name = _UNSAFE_NAME.sub("-", raw).strip("-_")
    if not name or not name[0].isalpha():
        name = f"sflow-{name}".rstrip("-")
    return name[:63].rstrip("-_")


def render_spec(
    *,
    workflow_name: str,
    image: str,
    script: Sequence[str],
    envs: Mapping[str, str],
    resources: Mapping[str, Any],
) -> str:
    """Render the OSMO workflow spec as block YAML.

    The script and env ship as one base64 file: OSMO Jinja-renders any spec
    containing ``{{``, ``{%`` or ``{#``, and base64 has none of them. The CLI also
    rejects JSON-style specs (it splits sections on ``^workflow:``).
    """
    skipped = [k for k in envs if not _ENV_KEY.fullmatch(k)]
    if skipped:
        _logger.warning(
            f"OSMO task: env names that are not shell names are skipped: {skipped}"
        )
    exports = [
        f"export {k}={shlex.quote(str(v))}"
        for k, v in envs.items()
        if _ENV_KEY.fullmatch(k)
    ]
    dirs = [f'"${v}"' for v in _OUTPUT_DIR_VARS if envs.get(v)]
    mkdir = [f"mkdir -p {' '.join(dirs)} 2>/dev/null || true"] if dirs else []
    body = "\n".join([*exports, *mkdir, *script]) + "\n"
    task = {
        "name": OSMO_TASK,
        "image": image,
        "command": ["bash"],
        "args": [_SCRIPT_PATH],
        "files": [
            {
                "path": _SCRIPT_PATH,
                "base64": True,
                "contents": base64.b64encode(body.encode()).decode(),
            }
        ],
    }
    spec = {
        "workflow": {
            "name": workflow_name,
            "resources": {"default": dict(resources)},
            "tasks": [task],
        }
    }
    return yaml.safe_dump(spec, sort_keys=False, width=10**9)


def exit_code_from_query(result: Mapping[str, Any]) -> int | None:
    """Map ``osmo workflow query -t json`` to an exit code; None while not terminal."""
    status = str(result.get("status") or "")
    if status != "COMPLETED" and not status.startswith("FAILED"):
        return None
    code = _main_task(result).get("exit_code")
    if status == "COMPLETED":
        return code if isinstance(code, int) else 0
    # A FAILED* workflow never counts as success, even if the process exited 0
    # (e.g. the output upload failed). Cancel, image pull, eviction: no exit code.
    return code if isinstance(code, int) and code != 0 else 1


# Cached per process: None until probed.
_submit_has_label: bool | None = None


async def _submit_supports_label() -> bool:
    """``osmo workflow submit --label`` arrived in OSMO CLI 6.4; 6.3 rejects it."""
    global _submit_has_label
    if _submit_has_label is None:
        rc, out = await _osmo("workflow", "submit", "--help")
        _submit_has_label = rc == 0 and "--label" in out
    return _submit_has_label


async def _osmo(*args: str, timeout: float = _CLI_TIMEOUT_S) -> tuple[int, str]:
    """Run one ``osmo`` CLI command with a time bound.

    Returns the exit code and stdout, or stdout plus stderr when it failed, so a
    warning on stderr cannot break JSON parsing of a successful call. A timeout
    kills the command and returns 124.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "osmo",
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:  # CLI removed or not executable after preflight
        return 127, f"could not start osmo {' '.join(args[:2])}: {exc}"
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"osmo {' '.join(args[:2])} timed out after {timeout:.0f}s"
    except asyncio.CancelledError:
        proc.kill()
        raise
    rc = proc.returncode or 0
    text = out.decode(errors="replace")
    return rc, text if rc == 0 else text + err.decode(errors="replace")


@register_operator("osmo", OsmoOperatorConfig)
class OsmoOperator(Operator):
    """Run the task script in ``image`` as a single-task OSMO workflow."""

    def __init__(self, config: OsmoOperatorConfig):
        """Store the config; pool, platform and GPUs arrive with the backend context."""
        super().__init__(config)
        self.config: OsmoOperatorConfig
        self._pool: str | None = None
        self._platform: str | None = None
        self._gpus = 0

    def apply_backend_context(
        self,
        *,
        backend: Any,
        assigned_nodes: Sequence[str],
        artifacts: Sequence[Any],
        cuda_visible_devices: str | None = None,
        gpu_count: int | None = None,
    ) -> None:
        """Take the pool and platform from the osmo backend, and the task's GPU count."""
        bconf = getattr(backend, "config", None)
        if getattr(bconf, "type", None) != "osmo":
            raise ValueError(
                f"Operator '{self.config.name}' (type osmo) runs only on an 'osmo' "
                f"backend, but the task is on backend '{getattr(backend, 'name', '?')}'. "
                "Set the task's 'backend:' to the osmo backend."
            )
        if len(assigned_nodes) > 1:
            raise ValueError(
                f"Operator '{self.config.name}' (type osmo) runs single-node tasks "
                f"only, but the task asks for {len(assigned_nodes)} nodes. "
                "Multi-node OSMO tasks are not supported yet."
            )
        self._pool = bconf.pool
        self._platform = bconf.platform
        self._gpus = int(gpu_count or 0)

    def manages_own_execution(self) -> bool:
        """Run through :meth:`execute`: submit, stream, query, cancel."""
        return True

    def resources(self) -> dict[str, Any]:
        """Return the OSMO resource spec for this task."""
        c = self.config
        res: dict[str, Any] = {"cpu": c.cpu, "memory": c.memory, "storage": c.storage}
        if self._gpus:
            res["gpu"] = self._gpus
        platform = c.platform or self._platform
        if platform:
            res["platform"] = platform
        return res

    def submit_args(self, spec_path: str, label: str | None = None) -> list[str]:
        """Return the ``osmo workflow submit`` arguments for a rendered spec."""
        args = ["workflow", "submit", spec_path, "-t", "json"]
        if self._pool:
            args += ["--pool", self._pool]
        if label:
            args += ["--label", label]
        return args

    def build_command(
        self,
        *,
        task_name: str,
        script: Sequence[str],
        envs: Mapping[str, str],
    ) -> Command:
        """Required by :class:`Operator`; unused, because :meth:`execute` runs the task."""
        return Command(exec="osmo").add_arg("workflow").add_arg("submit")

    async def _submit(self, spec: str, task_name: str) -> tuple[int, str]:
        """Submit the spec. If sflow cancels meanwhile, cancel the new workflow too."""
        with tempfile.TemporaryDirectory(prefix="sflow-osmo-") as tmp:
            spec_path = Path(tmp) / "workflow.yaml"
            spec_path.write_text(spec)
            label = (
                f"sflow-task={osmo_name(task_name)}"
                if await _submit_supports_label()
                else None
            )
            submit = asyncio.ensure_future(
                _osmo(*self.submit_args(str(spec_path), label))
            )
            try:
                return await asyncio.shield(submit)
            except asyncio.CancelledError:
                # The submit may still create the workflow: let it finish, then cancel it.
                rc, out = await submit
                wf_id = _workflow_id(out) if rc == 0 else None
                if wf_id:
                    await _cancel(wf_id, task_name)
                raise

    async def _poll(
        self, wf_id: str, output_logger: Any, check: Callable[[dict], Any]
    ) -> Any:
        """Query until ``check(result)`` is not None and return it; None if the query keeps failing."""
        failures = 0
        while True:
            rc, out = await _osmo("workflow", "query", wf_id, "-t", "json")
            if rc == 0:
                failures = 0
                value = check(json.loads(out))
                if value is not None:
                    return value
            else:
                failures += 1
                if failures >= _MAX_QUERY_FAILURES:
                    output_logger.error(
                        f"osmo workflow query {wf_id} keeps failing:\n{out}"
                    )
                    return None
            await asyncio.sleep(self.config.poll_interval)

    async def _wait_for_exit_code(self, wf_id: str, output_logger: Any) -> int | None:
        """Query until the workflow is terminal; None if the query keeps failing."""
        return await self._poll(wf_id, output_logger, exit_code_from_query)

    async def _wait_while_queued(
        self,
        wf_id: str,
        output_logger: Any,
        task_name: str,
        status_note: Callable[[str | None], None] | None,
    ) -> bool:
        """Poll while the workflow waits in the OSMO queue, logging its state.

        PENDING: not started yet. WAITING: started, but the task waits to be
        (re)scheduled, e.g. after a preemption. A log stream attached in either
        state is cut every ~60 s, so attach only after this returns. False: the
        status query keeps failing.
        """
        start = time.monotonic()
        last_state, last_log = None, 0.0

        def check(result: dict) -> bool | None:
            nonlocal last_state, last_log
            status = result.get("status")
            if status not in _QUEUED:
                return True
            state = _main_task(result).get("status") or status
            waited = time.monotonic() - start
            note = f"OSMO {wf_id} queued in {self._pool or 'default pool'}: {state}"
            if state != last_state or waited - last_log >= _QUEUE_LOG_EVERY_S:
                _logger.info(f"[{task_name}] {note} ({waited / 60:.0f}m)")
                last_state, last_log = state, waited
            if status_note is not None:
                status_note(note)
            return None

        ok = await self._poll(wf_id, output_logger, check) is not None
        if status_note is not None:
            status_note(f"OSMO {wf_id}")
        return ok

    async def _stream_and_wait(
        self,
        launcher: Any,
        output_logger: Any,
        wf_id: str,
        task_name: str,
        status_note: Callable[[str | None], None] | None = None,
    ) -> int | None:
        """Stream the task log into ``<task>.log``, then return the exit code.

        Attaches only after the workflow leaves the queue (see
        :meth:`_wait_while_queued`). A proxy can still cut the stream while the task
        is quiet (seen after ~60 s idle), and
        the CLI can fail on other stream errors, so re-attach after any non-zero
        exit until OSMO reports a terminal status; ``osmo_logs`` skips the lines
        earlier attaches already wrote. None: the status query keeps failing.
        """
        if not await self._wait_while_queued(
            wf_id, output_logger, task_name, status_note
        ):
            return None
        with tempfile.TemporaryDirectory(prefix="sflow-osmo-logs-") as tmp:
            stream = [sys.executable, "-u", "-m", _LOG_STREAM_MODULE, wf_id, OSMO_TASK]
            stream.append(str(Path(tmp) / "lines"))

            def attach() -> Any:
                return launcher.run_async(
                    stream, output_logger=output_logger, task_name=task_name
                )

            failures = 0
            while True:
                rc = await attach()
                if rc == 0:
                    return await self._wait_for_exit_code(wf_id, output_logger)
                q_rc, out = await _osmo("workflow", "query", wf_id, "-t", "json")
                if q_rc != 0:
                    failures += 1
                    if failures >= _MAX_QUERY_FAILURES:
                        output_logger.error(
                            f"osmo workflow query {wf_id} keeps failing:\n{out}"
                        )
                        return None
                    continue
                failures = 0
                code = exit_code_from_query(json.loads(out))
                if code is not None:
                    # The task ended while the stream was cut: one more bounded
                    # attach writes its last lines (results, probe markers).
                    try:
                        await asyncio.wait_for(attach(), _CLI_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        _logger.warning(f"[{task_name}] final OSMO log fetch timed out")
                    return code
                if rc == _LOG_STREAM_CUT:
                    _logger.info(
                        f"[{task_name}] OSMO log stream cut; re-attaching to {wf_id}"
                    )
                else:  # stream error, not an idle cut: back off before re-attaching
                    _logger.warning(
                        f"[{task_name}] OSMO log stream exited {rc}; "
                        f"re-attaching to {wf_id}"
                    )
                    await asyncio.sleep(self.config.poll_interval)

    async def execute(
        self,
        *,
        launcher: Any,
        output_logger: Any,
        env: Mapping[str, str],
        task_name: str,
        script: Sequence[str],
        status_note: Callable[[str | None], None] | None = None,
    ) -> int:
        """Submit the task as an OSMO workflow, stream its logs, and return its exit code.

        Any exit before the workflow is terminal (cancel, error, failing queries)
        cancels the workflow, so nothing keeps running on the pool.
        """
        spec = render_spec(
            workflow_name=osmo_name(f"sflow-{task_name}"),
            image=self.config.image,
            script=script,
            envs=env,
            resources=self.resources(),
        )
        rc, out = await self._submit(spec, task_name)
        wf_id = _workflow_id(out) if rc == 0 else None
        if wf_id is None:
            output_logger.error(f"osmo workflow submit failed (rc={rc}):\n{out}")
            return rc or 1
        _logger.info(f"[{task_name}] OSMO workflow {wf_id}")
        if status_note is not None:
            status_note(f"OSMO {wf_id}")

        terminal = False
        try:
            code = await self._stream_and_wait(
                launcher, output_logger, wf_id, task_name, status_note
            )
            terminal = code is not None
            return 1 if code is None else code
        finally:
            if not terminal:
                await asyncio.shield(_cancel(wf_id, task_name))
            if status_note is not None:
                status_note(None)


def _main_task(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the sflow task from a query result (first task as fallback), or {}."""
    tasks = [t for g in result.get("groups") or [] for t in g.get("tasks") or []]
    return next(
        (t for t in tasks if t.get("name") == OSMO_TASK), tasks[0] if tasks else {}
    )


def _workflow_id(submit_output: str) -> str | None:
    """Return the workflow ID from ``osmo workflow submit -t json``, or None."""
    try:
        return str(json.loads(submit_output)["name"])
    except (ValueError, KeyError, TypeError):
        return None


async def _cancel(wf_id: str, task_name: str) -> None:
    """Cancel a workflow, best effort: log a failure instead of raising."""
    rc, out = await _osmo("workflow", "cancel", wf_id, "-m", "sflow teardown")
    if rc != 0:
        _logger.warning(f"[{task_name}] osmo workflow cancel {wf_id} failed: {out}")
