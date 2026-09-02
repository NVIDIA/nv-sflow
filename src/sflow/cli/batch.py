# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI command for generating sbatch scripts to run sflow in batch mode.
"""

import csv
import json
import os
import shlex
import tempfile
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Annotated, Any, List, Optional, Protocol
from urllib.parse import urlparse
from urllib.request import url2pathname

import typer
import yaml as _yaml

from sflow.app.sflow import SflowApp
from sflow.config.loader import safe_load
from sflow.cli import DOCS_URL, app
from sflow.cli._args import (  # split_list_arg re-exported for back-compat
    EnableTaskMonitorOption,
    EnableWorkflowMonitorOption,
    ExcludeNodesOption,
    IncludeNodesOption,
    split_list_arg,
)
from sflow.core.log_offload import OFFLOAD_TASK_LOGS_ENV
from sflow.logging import configure_logging, get_logger
from sflow.resolution import enrich_error_with_location
from sflow.runtime_info import log_runtime_info
from sflow.utils.extra_args import dedup_merge_extra_args
from sflow.utils.install import (
    DEFAULT_SFLOW_GIT_URL,
    sflow_git_install_url,
    sflow_index_url_error,
    sflow_pypi_requirement,
    sflow_version_error,
)
from sflow.utils.slurm import emit_gpus_per_node_semantics_warning

_logger = get_logger(__name__)

_sflow_app = SflowApp()
_DEFAULT_SFLOW_GIT_URL = DEFAULT_SFLOW_GIT_URL


def _detect_slurm_account() -> str | None:
    """Try to detect the current user's default Slurm account.

    Queries ``sacctmgr`` for the associations of the current OS user.
    Returns the first account found, or None.
    """
    import os
    import subprocess

    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user:
        try:
            user = subprocess.check_output(["whoami"], text=True).strip()
        except Exception:
            return None
    try:
        out = subprocess.check_output(
            [
                "sacctmgr",
                "show",
                "assoc",
                f"user={user}",
                "format=Account%30",
                "--noheader",
                "--parsable2",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        for line in out.strip().splitlines():
            acct = line.strip()
            if acct:
                return acct
    except Exception:
        pass
    return None


def _detect_slurm_partition() -> str | None:
    """Try to detect the default Slurm partition.

    Looks for a partition marked as default (``*``) in ``sinfo`` output.
    Falls back to the first available partition.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["sinfo", "--noheader", "--format=%P"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        first: str | None = None
        for line in out.strip().splitlines():
            name = line.strip()
            if not name:
                continue
            if first is None:
                first = name.rstrip("*")
            if name.endswith("*"):
                return name.rstrip("*")
        return first
    except Exception:
        pass
    return None


def _resolve_slurm_defaults(
    partition: str | None,
    account: str | None,
) -> tuple[str, str]:
    """Resolve partition and account, auto-detecting from Slurm when not provided.

    Emits warnings for auto-detected values. Raises ``typer.BadParameter``
    if a value cannot be determined.
    """
    if partition is None:
        partition = _detect_slurm_partition()
        if partition:
            typer.echo(
                f"  Warning: --partition not specified, auto-detected: {partition}",
                err=True,
            )
        else:
            raise typer.BadParameter(
                "Could not auto-detect a Slurm partition. Please specify --partition / -p explicitly."
            )

    if account is None:
        account = _detect_slurm_account()
        if account:
            typer.echo(
                f"  Warning: --account not specified, auto-detected: {account}",
                err=True,
            )
        else:
            raise typer.BadParameter(
                "Could not auto-detect a Slurm account. Please specify --account / -A explicitly."
            )

    return partition, account


def _git_current_ref(repo_path: Path) -> str | None:
    """Return the current git branch, or detached HEAD commit if needed."""
    import subprocess

    try:
        branch = subprocess.check_output(
            ["git", "-C", str(repo_path), "symbolic-ref", "--quiet", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if branch:
            return branch
    except Exception:
        pass

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if commit:
            return commit
    except Exception:
        pass

    return None


def _repo_path_from_direct_url(url: str) -> Path | None:
    """Resolve a local repo path from a PEP 610 direct_url entry."""
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None

    raw_path = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        raw_path = f"//{parsed.netloc}{raw_path}"

    repo_path = Path(raw_path)
    if repo_path.exists():
        return repo_path
    return None


def _resolve_effective_sflow_version(sflow_version: str | None) -> str | None:
    """Resolve the git ref/version that generated batch scripts should install."""
    if sflow_version:
        return sflow_version

    try:
        dist = importlib_metadata.distribution("sflow")
    except importlib_metadata.PackageNotFoundError:
        return None

    try:
        direct_url_text = dist.read_text("direct_url.json")
    except Exception:
        direct_url_text = None

    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            direct_url = {}

        vcs_info = direct_url.get("vcs_info") or {}
        requested_revision = vcs_info.get("requested_revision")
        if requested_revision:
            return str(requested_revision)

        repo_url = direct_url.get("url")
        if isinstance(repo_url, str):
            repo_path = _repo_path_from_direct_url(repo_url)
            if repo_path:
                repo_ref = _git_current_ref(repo_path)
                if repo_ref:
                    return repo_ref

    version = getattr(dist, "version", None)
    if version:
        return str(version)

    try:
        return importlib_metadata.version("sflow")
    except importlib_metadata.PackageNotFoundError:
        return None


# Install-spec helpers are shared with `sflow upgrade`; see sflow.utils.install.
# The module-private aliases are kept so existing call sites (and tests) that
# reference them by their original names keep working.
_sflow_git_install_url = sflow_git_install_url
_sflow_pypi_requirement = sflow_pypi_requirement
_sflow_version_error = sflow_version_error


def _resolve_sbatch_extra_args(
    extra_args: list[str],
    config_files: list[Path],
    set_var: list[str] | None,
) -> list[str]:
    """Resolve ``${{ }}`` expressions in sbatch extra args.

    Supports both ``${{ variables.SLURM_NODES }}`` (full path) and
    ``${{ SLURM_NODES }}`` (shorthand).  Builds a variable context from the
    config YAML files (defaults) with ``set_var`` overrides applied on top,
    then resolves any Jinja2 expressions found in the extra args.

    Variable values are wrapped in :class:`VariableValue` so that
    ``${{ variables.X.domain }}`` is accessible.
    """
    if not any("${{" in arg for arg in extra_args):
        return list(extra_args)

    from sflow.resolution import ExpressionResolver
    from sflow.core.variable import build_variables_ctx_from_raw, extract_domains_from_raw_config

    var_map: dict[str, Any] = {}
    domain_map: dict[str, list[Any]] = {}
    for cfg_path in config_files:
        try:

            with open(cfg_path) as fh:
                data = safe_load(fh)
            if data:
                var_map.update(_build_var_map(data))
                domain_map.update(extract_domains_from_raw_config(data))
        except Exception:
            pass

    if set_var:
        for override in set_var:
            if "=" in override:
                k, v = override.split("=", 1)
                var_map[k] = v

    wrapped = build_variables_ctx_from_raw(var_map, domain_map)
    ctx: dict[str, Any] = {"variables": wrapped}
    ctx.update(wrapped)
    resolver = ExpressionResolver()

    resolved: list[str] = []
    for arg in extra_args:
        if "${{" in arg:
            try:
                resolved.append(str(resolver.resolve(arg, ctx)))
            except Exception:
                resolved.append(arg)
        else:
            resolved.append(arg)
    return resolved


@dataclass(frozen=True)
class _ResolvedSlurmBackend:
    """Resolved Slurm backend fields used for multi-backend sbatch generation."""

    name: str
    partition: str
    account: str
    nodes: int
    time: str | None
    extra_args: list[str]
    gpus_per_node: int | None = None


def _resolve_batch_backends(
    config_files: list[Path],
    set_var: list[str] | None,
) -> list[Any] | None:
    """Load + resolve a (composed) config's backend objects via the standard pipeline.

    Runs ``ConfigLoader`` (merge + ``--set`` overrides) followed by
    ``resolve_global_variables`` / ``resolve_backends`` so ``sflow batch`` classifies
    backends exactly as ``sflow run`` does, instead of re-parsing YAML with a bespoke
    resolver. Returns the resolved backend objects (``state.backends`` values), or
    ``None`` when the config cannot be loaded/resolved -- callers fall back to their
    own handling and the subsequent dry-run validation surfaces the underlying error.
    """
    try:
        from sflow.app.assembly import resolve_backends, resolve_global_variables
        from sflow.config.loader import ConfigLoader
        from sflow.core.state import SflowState
        from sflow.core.task_graph import TaskGraph
        from sflow.core.workflow import Workflow

        config = ConfigLoader().load_configs(list(config_files), set_var, None, None)
        state = SflowState(
            workflow=Workflow(name=config.workflow.name, task_graph=TaskGraph())
        )
        state = resolve_global_variables(config, state)
        state = resolve_backends(config, state)
    except Exception:
        return None
    return list(state.backends.values())


def _resolve_slurm_backends(
    config_files: list[Path],
    set_var: list[str] | None,
) -> list[_ResolvedSlurmBackend]:
    """Resolve the Slurm backends declared by a (composed) config.

    This reuses the standard config pipeline -- ``ConfigLoader`` (merge + ``--set``
    overrides) followed by ``resolve_global_variables`` / ``resolve_backends`` --
    so ``sflow batch`` sees exactly the same resolved backend objects as
    ``sflow run``, instead of re-parsing YAML with a bespoke resolver. Only Slurm
    backends are returned, in config-declaration order; non-Slurm backends
    (docker/kubernetes/local) are ignored here and never receive Slurm-specific
    handling.

    The result drives whether ``sflow batch`` emits a multi-backend driver job:
    a config with >=2 Slurm backends becomes one driver sbatch (sized to the
    leader backend) and each backend runs its own salloc at runtime. Returns an
    empty list when the config cannot be loaded/resolved; the subsequent dry-run
    validation surfaces the underlying error with full context.
    """
    from sflow.plugins.backends.slurm import SlurmBackend

    backends = _resolve_batch_backends(config_files, set_var)
    if backends is None:
        return []

    resolved: list[_ResolvedSlurmBackend] = []
    for backend in backends:
        if not isinstance(backend, SlurmBackend):
            continue
        conf = backend.config
        try:
            nodes_int = int(conf.nodes)
        except (TypeError, ValueError):
            continue
        try:
            gpus_int: int | None = int(conf.gpus_per_node)
        except (TypeError, ValueError):
            gpus_int = None
        resolved.append(
            _ResolvedSlurmBackend(
                name=str(conf.name),
                partition=str(conf.partition),
                account=str(conf.account),
                nodes=nodes_int,
                time=None if conf.time is None else str(conf.time),
                extra_args=[str(a) for a in (conf.extra_args or [])],
                gpus_per_node=gpus_int,
            )
        )
    return resolved


def _kubernetes_backend_names(
    config_files: list[Path],
    set_var: list[str] | None,
) -> list[str]:
    """Names of the Kubernetes backends declared by the (composed) config.

    Resolved through the same pipeline as :func:`_resolve_slurm_backends`. Returns an
    empty list when the config cannot be loaded/resolved -- the dry-run validation
    (or the single-backend planner) then surfaces the real error with full context.
    """
    from sflow.plugins.backends.kubernetes import KubernetesBackend

    backends = _resolve_batch_backends(config_files, set_var)
    if backends is None:
        return []
    return [
        str(backend.name)
        for backend in backends
        if isinstance(backend, KubernetesBackend)
    ]


def _reject_kubernetes_batch(
    config_files: list[Path],
    set_var: list[str] | None,
) -> None:
    """Fail fast when a config routed to ``sflow batch`` uses a Kubernetes backend.

    ``sflow batch`` exists to generate a Slurm ``sbatch`` script; Kubernetes schedules
    its own pods and has no batch/sbatch step, so a k8s workflow must run via
    ``sflow run`` (which drives ``kubectl`` directly, optionally with ``--kube-*``
    flags). Raises :class:`ValueError` naming the offending backend(s); the caller
    turns it into a CLI error message + non-zero exit.
    """
    k8s_names = _kubernetes_backend_names(config_files, set_var)
    if not k8s_names:
        return
    joined = ", ".join(sorted(k8s_names))
    raise ValueError(
        f"'sflow batch' does not support the Kubernetes backend "
        f"(backend(s): {joined}). Kubernetes schedules its own pods, so there is no "
        f"Slurm sbatch job to generate. Run the workflow directly with 'sflow run' "
        f"instead (e.g. 'sflow run <config> [--kube-* ...]')."
    )


def _select_wrapper_backend(
    slurm_backends: list[_ResolvedSlurmBackend],
) -> _ResolvedSlurmBackend:
    """Pick the most resource-heavy Slurm backend to own the driver sbatch.

    The driver sbatch reserves this backend's nodes up front via normal Slurm
    scheduling, and the lighter backends ``salloc`` nested inside it. Reserving
    the largest footprint as the batch job (rather than nesting it) makes the
    whole multi-backend allocation easier to schedule. Heaviness is compared by
    node count, then total GPUs (``nodes * gpus_per_node``), then GPUs per node;
    ties keep config-declaration order.
    """

    def weight(b: _ResolvedSlurmBackend) -> tuple[int, int, int]:
        gpn = b.gpus_per_node or 0
        return (b.nodes, b.nodes * gpn, gpn)

    return max(slurm_backends, key=weight)


def _build_multi_backend_driver_directives(
    slurm_backends: list[_ResolvedSlurmBackend],
    *,
    job_name: str,
    sbatch_output: str,
    sbatch_error: str,
    time: str | None,
    leader_extra_args: list[str],
    include_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Build #SBATCH directives + markers for a multi-backend driver job.

    A config with >=2 Slurm backends is submitted as a single driver sbatch
    sized to the **most resource-heavy** backend (see :func:`_select_wrapper_backend`).
    Inside, ``sflow run`` lets that backend reuse this allocation while every
    other backend runs its own ``salloc`` (so each backend gets a distinct Slurm
    job id, which pyxis/enroot need for their per-job runtime dir to match the
    node that provisioned it). Wrapping the heaviest backend in the batch job and
    nesting the lighter ``salloc``s makes the whole allocation easier to schedule
    and avoids the Slurm heterogeneous-job model, where all components share the
    leader job id and pyxis fails on non-leader partitions.

    Returns ``(directives, exports)`` where ``exports`` carry the
    per-backend-salloc markers read by :meth:`SlurmBackend.allocate`.
    """
    leader = _select_wrapper_backend(slurm_backends)
    directives: list[str] = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={sbatch_output}",
        f"#SBATCH --error={sbatch_error}",
        "#SBATCH --mem=0",
        f"#SBATCH --partition={leader.partition}",
        f"#SBATCH --account={leader.account}",
        f"#SBATCH --nodes={leader.nodes}",
    ]
    component_time = leader.time or time
    if component_time:
        directives.append(f"#SBATCH --time={component_time}")
    # Steer the driver allocation with the node filters (before extra_args wins).
    if include_nodes:
        directives.append(f"#SBATCH --nodelist={','.join(include_nodes)}")
    if exclude_nodes:
        directives.append(f"#SBATCH --exclude={','.join(exclude_nodes)}")
    # #SBATCH directives carry only the LEADER backend's merged set (its own
    # extra_args + the CLI --sbatch-extra-args, de-duped by option, CLI wins).
    # The non-leader backends pick the CLI args up at runtime: the generated
    # `sflow run` is invoked with `--extra-salloc-args`, which merges them into
    # every backend's salloc independently (see _generate_sbatch_script).
    for extra in dedup_merge_extra_args(leader.extra_args, leader_extra_args):
        directives.append(f"#SBATCH {extra}")
    exports: list[str] = [
        "export SFLOW_SLURM_MULTI_BACKEND_SALLOC=1",
        f"export SFLOW_SLURM_WRAPPER_BACKEND={shlex.quote(leader.name)}",
    ]
    return directives, exports


def _generate_sbatch_script(
    *,
    files: list[Path],
    set_var: list[str] | None,
    artifact: list[str] | None,
    missable_tasks: list[str] | None = None,
    log_level: str,
    workspace_dir: Path | None,
    output_dir: Path | None,
    job_name: str,
    sbatch_output: str,
    sbatch_error: str,
    partition: str,
    account: str,
    time: str | None,
    nodes: int | None,
    gpus_per_node: int | None,
    sbatch_extra_args: list[str] | None,
    sflow_venv_path: Path | None,
    sflow_version: str | None,
    sflow_source_path: Path | None = None,
    sflow_index_url: str | None = None,
    enable_workflow_monitor: bool = False,
    enable_task_monitors: list[str] | None = None,
    include_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    skip_artifact_check: bool = False,
) -> str:
    """Generate the content of an sbatch script that wraps ``sflow run``.

    When the (composed) config declares >=2 Slurm backends, a multi-backend
    driver script is emitted (sized to the leader backend); inside, each backend
    runs its own salloc so it binds to its own allocation/partition. Otherwise a
    single-allocation script is generated.
    """
    sflow_cmd_parts = ["sflow", "run"]
    for f in files:
        sflow_cmd_parts.extend(["--file", shlex.quote(str(f))])

    if set_var:
        for var in set_var:
            sflow_cmd_parts.extend(["--set", shlex.quote(var)])

    if artifact:
        for art in artifact:
            sflow_cmd_parts.extend(["--artifact", shlex.quote(art)])

    if missable_tasks:
        for mt in missable_tasks:
            sflow_cmd_parts.extend(["--missable-tasks", shlex.quote(mt)])

    # Forwarded to the inner `sflow run`: batch's own preflight is a dry run (which
    # already only warns), so the hard failure this suppresses happens in the job.
    if skip_artifact_check:
        sflow_cmd_parts.append("--skip-artifact-check")

    if enable_workflow_monitor:
        sflow_cmd_parts.append("--enable-workflow-monitor")

    if enable_task_monitors:
        for task_name in enable_task_monitors:
            sflow_cmd_parts.extend(["--enable-task-monitor", shlex.quote(task_name)])

    if log_level != "info":
        sflow_cmd_parts.extend(["--log-level", log_level])

    if workspace_dir:
        sflow_cmd_parts.extend(["--workspace-dir", shlex.quote(str(workspace_dir))])

    if output_dir:
        sflow_cmd_parts.extend(["--output-dir", shlex.quote(str(output_dir))])

    # Forward node include/exclude to the inner `sflow run` so every backend
    # applies them (the leader backend that reuses this allocation post-filters;
    # backends that salloc themselves get --nodelist/--exclude at runtime).
    for host in include_nodes or []:
        sflow_cmd_parts.extend(["--include-nodes", shlex.quote(host)])
    for host in exclude_nodes or []:
        sflow_cmd_parts.extend(["--exclude-nodes", shlex.quote(host)])

    resolved_extra_args = (
        _resolve_sbatch_extra_args(sbatch_extra_args, files, set_var)
        if sbatch_extra_args
        else []
    )

    # Thread the CLI extra args into the inner `sflow run` as --extra-salloc-args
    # so each backend merges them into its OWN salloc at runtime (deduped by
    # option, CLI wins). The leader backend reuses the driver allocation (no
    # salloc), so this only takes effect for backends that salloc themselves.
    for extra_arg in resolved_extra_args:
        sflow_cmd_parts.extend(["--extra-salloc-args", shlex.quote(extra_arg)])

    # A config with >=2 Slurm backends is submitted as a single driver sbatch
    # sized to the leader backend; inside, each backend runs its own salloc (the
    # leader reuses this allocation) so every backend gets a distinct Slurm job
    # id. Single-backend configs keep the original single-allocation script.
    slurm_backends = _resolve_slurm_backends(files, set_var)
    multi_backend_exports: list[str] = []
    if len(slurm_backends) >= 2:
        sbatch_directives, multi_backend_exports = (
            _build_multi_backend_driver_directives(
                slurm_backends,
                job_name=job_name,
                sbatch_output=sbatch_output,
                sbatch_error=sbatch_error,
                time=time,
                leader_extra_args=resolved_extra_args,
                include_nodes=include_nodes,
                exclude_nodes=exclude_nodes,
            )
        )
        generated_by = (
            f"# Generated by: sflow batch "
            f"(multi-backend: {len(slurm_backends)} backends, per-backend salloc)"
        )
    else:
        sbatch_directives = [
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={sbatch_output}",
            f"#SBATCH --error={sbatch_error}",
            "#SBATCH --mem=0",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --account={account}",
        ]
        if nodes is not None:
            sbatch_directives.append(f"#SBATCH --nodes={nodes}")
        if time:
            sbatch_directives.append(f"#SBATCH --time={time}")
        # Steer the driver allocation with the node filters (before extra_args so an
        # explicit --sbatch-extra-args --nodelist/--exclude still wins).
        if include_nodes:
            sbatch_directives.append(f"#SBATCH --nodelist={','.join(include_nodes)}")
        if exclude_nodes:
            sbatch_directives.append(f"#SBATCH --exclude={','.join(exclude_nodes)}")
        # Include the (single) slurm backend's extra_args, merged with the CLI
        # --sbatch-extra-args and de-duped by option (CLI wins on conflict).
        backend_extra_args = slurm_backends[0].extra_args if slurm_backends else []
        for extra_arg in dedup_merge_extra_args(backend_extra_args, resolved_extra_args):
            sbatch_directives.append(f"#SBATCH {extra_arg}")
        generated_by = "# Generated by: sflow batch"

    script_lines = [
        "#!/bin/bash",
        "#",
        generated_by,
        f"# Workflow file(s): {', '.join(str(f) for f in files)}",
        "#",
        "",
        *sbatch_directives,
        "",
        "set -x",
        "",
        "# sbatch's default --export=ALL leaks the submitter's env into the job.",
        "# Drop VIRTUAL_ENV/PATH (a caller venv may be the wrong arch -> 'Exec format",
        "# error') and PYTHONPATH (an sflow src tree on it would shadow the per-job",
        "# editable install), so the job uses only its fresh per-job venv.",
        'if [ -n "${VIRTUAL_ENV:-}" ]; then',
        "    PATH=$(printf '%s' \"$PATH\" | tr ':' '\\n' | grep -vxF \"$VIRTUAL_ENV/bin\" | paste -sd ':' -)",
        "    export PATH",
        "fi",
        "unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT PYTHONHOME PYTHONPATH",
        "",
    ]

    if sflow_venv_path:
        # Explicit parent: bake the resolved absolute path.
        venv_parent_str = shlex.quote(str(Path(sflow_venv_path).resolve()))
    else:
        # No explicit parent: resolve compute-node-local scratch at RUNTIME so it
        # honors the cluster's per-node $TMPDIR (often a per-job dir that is
        # auto-cleaned) and falls back to /tmp. Node-local scratch is the right
        # home for a disposable, job-id-keyed venv -- fast small-file I/O, no
        # shared-FS metadata contention, and no home/project quota use. The quotes
        # are emitted literally so $TMPDIR expands on the compute node.
        venv_parent_str = '"${TMPDIR:-/tmp}/sflow_compute_node_venv"'

    # How sflow is installed into the fresh per-job venv. Two main routes share
    # --sflow-version, plus a dev-only source override; all mutually exclusive
    # (enforced + sanity-checked at the CLI layer):
    #   - Route 1 (git, the default): install the resolved git ref from the repo.
    #   - Route 2 (--sflow-index-url): install a released wheel from a private PyPI
    #     index, with --sflow-version as the version/specifier.
    #   - --sflow-source-path (dev): editable install from a per-job copy of a
    #     local checkout.
    if sflow_source_path is not None:
        source_path_str = shlex.quote(str(Path(sflow_source_path).resolve()))
        # A single shared source tree cannot be reused across concurrent jobs: an
        # editable build rewrites setuptools-scm's _version.py and the *.egg-info
        # back into the tree (a warm uv cache does NOT skip these writes), so
        # concurrent installs would race on those files. Give every job its own
        # copy instead -- copy the checkout into a per-job dir (heavy/generated
        # paths excluded; .git kept so setuptools-scm can resolve the version) and
        # install editable from there. Fully isolated, so it needs no lock.
        #
        # The per-job venv/source dirs (.sflow_venv*, .sflow_src*) MUST be excluded
        # for correctness, not just size: when --sflow-venv-path is the source
        # checkout itself (the under-dev e2e passes --sflow-venv-path and
        # --sflow-source-path both = $REPO_DIR), SFLOW_SRC_DIR lands *inside* the
        # copy source. Without these excludes the copy would recurse into its own
        # destination and into every concurrent job's growing copy -- a runaway
        # that fills the filesystem and never finishes. The remaining excludes are
        # size/speed only. The same --exclude=PATTERN syntax works for both rsync
        # and the tar fallback, so one list drives both copy paths.
        copy_excludes = [
            ".venv",
            "venv",
            ".sflow_venv*",
            ".sflow_src*",
            "sflow_compute_node_venv",
            "sflow_output",
            "build",
            "dist",
            "*.egg-info",
            "__pycache__",
            # NOT just size: pip/uv write partial `*.tmp` files under .cache while
            # other jobs are still bootstrapping, and rsync exits 24 ("some files
            # vanished") when one disappears mid-transfer -- which the bootstrap
            # treats as fatal, so the whole Slurm job dies seconds in with no output
            # directory at all. A shared, concurrently-written cache must never be
            # part of the copy source.
            ".cache",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".tox",
            "node_modules",
            "docs-site",
            "htmlcov",
            ".coverage",
            "coverage.xml",
            ".gitnexus",
        ]
        copy_exclude_args = " ".join(
            f"--exclude={shlex.quote(p)}" for p in copy_excludes
        )
        sflow_install_lines = [
            "# Per-job copy of the local checkout (rsync, or tar when rsync is",
            "# absent), then editable install from the copy.",
            'SFLOW_SRC_DIR="$SFLOW_VENV_PARENT/.sflow_src-${SLURM_JOB_ID:-$$}"',
            'rm -rf "$SFLOW_SRC_DIR"',
            'mkdir -p "$SFLOW_SRC_DIR"',
            "if command -v rsync >/dev/null 2>&1; then",
            f'    rsync -a {copy_exclude_args} {source_path_str}/ "$SFLOW_SRC_DIR/"',
            "else",
            f'    tar -C {source_path_str} {copy_exclude_args} -cf - . | tar -C "$SFLOW_SRC_DIR" -xf -',
            "fi",
            'cd "$SFLOW_SRC_DIR"',
            '"$VIRTUAL_ENV/bin/uv" pip install -e ".[dev]"',
        ]
    elif sflow_index_url is not None:
        # Route 2 -- PyPI private index: install a released sflow wheel
        # (--sflow-version is the version/specifier, validated at the CLI layer).
        # --extra-index-url keeps the default index available for sflow's deps,
        # since the private repo typically holds only sflow. Credentials come from
        # the compute node (~/.netrc or a credential helper); the URL is checked
        # for embedded credentials at the CLI layer.
        requirement = _sflow_pypi_requirement(sflow_version)
        sflow_install_cmd = (
            f"{shlex.quote(requirement)} "
            f"--extra-index-url {shlex.quote(sflow_index_url)} "
            "--prerelease=allow"
        )
        sflow_install_lines = [
            "set +x",
            f'"$VIRTUAL_ENV/bin/uv" pip install {sflow_install_cmd}',
            "set -x",
        ]
    else:
        # Route 1 -- git: install the resolved ref (--sflow-version, or the running
        # env's ref / 'main' when omitted) from the sflow git repo.
        effective_sflow_version = _resolve_effective_sflow_version(sflow_version)
        sflow_install_cmd = (
            f"{shlex.quote(f'sflow @ {_sflow_git_install_url(effective_sflow_version)}')} "
            "--prerelease=allow"
        )
        sflow_install_lines = [
            f'"$VIRTUAL_ENV/bin/uv" pip install {sflow_install_cmd}',
        ]

    effective_output_dir = (
        shlex.quote(str(output_dir))
        if output_dir
        else shlex.quote(
            str(workspace_dir / "sflow_output")
            if workspace_dir
            else str(Path.cwd() / "sflow_output")
        )
    )
    # cp lines (run inside the on-exit finalize fn) for each config file, copied
    # into the resolved workflow output dir. Indented for the shell function body.
    config_copy_lines = [
        f'    cp {shlex.quote(str(f))} "$SFLOW_WF_DIR/" 2>/dev/null || true'
        for f in files
    ]

    script_lines.extend(
        [
            f"SFLOW_VENV_PARENT={venv_parent_str}",
            'mkdir -p "$SFLOW_VENV_PARENT"',
            "",
            "# Fresh per-job venv keyed on the Slurm job id (PID fallback off-Slurm),",
            "# so concurrent jobs never collide -- no shared venv, no flock.",
            'SFLOW_VENV_DIR="$SFLOW_VENV_PARENT/.sflow_venv-${SLURM_JOB_ID:-$$}"',
            "",
            "# Run on exit AND on the signals Slurm uses for timeout/cancel/preempt",
            "# (a bare EXIT trap does not fire on an untrapped signal). It ALWAYS",
            "# copies the sbatch .out/.err and config(s) into the workflow output dir",
            "# -- using a <job id>-sflow-submit dir if the run never created one -- so",
            "# a failed bootstrap or run still leaves a full debug picture, then",
            "# removes the disposable per-job venv/source copy.",
            "_sflow_finalize() {",
            "    # Best-effort: capture the incoming rc FIRST (before any other command",
            "    # changes $?), then disarm all traps so a signal arriving mid-cleanup --",
            "    # or the EXIT trap firing after a signal-triggered run -- cannot re-enter",
            "    # this handler. Then disable errexit and guard every step so a failed",
            "    # copy/cleanup never changes the job's exit status.",
            "    _sflow_rc=$?",
            "    trap - EXIT INT TERM HUP",
            "    set +e",
            f"    SBATCH_OUT_PATTERN={shlex.quote(sbatch_output)}",
            f"    SBATCH_ERR_PATTERN={shlex.quote(sbatch_error)}",
            '    SBATCH_OUT="${SBATCH_OUT_PATTERN//%j/$SLURM_JOB_ID}"',
            '    SBATCH_ERR="${SBATCH_ERR_PATTERN//%j/$SLURM_JOB_ID}"',
            f'    SFLOW_WF_DIR=$(find {effective_output_dir} -maxdepth 1 -type d -name "${{SLURM_JOB_ID}}-*" 2>/dev/null | head -1)',
            f'    [ -n "$SFLOW_WF_DIR" ] || SFLOW_WF_DIR={effective_output_dir}/"${{SLURM_JOB_ID}}-sflow-submit"',
            '    mkdir -p "$SFLOW_WF_DIR" 2>/dev/null || true',
            '    cp "$SBATCH_OUT" "$SFLOW_WF_DIR/" 2>/dev/null || true',
            '    cp "$SBATCH_ERR" "$SFLOW_WF_DIR/" 2>/dev/null || true',
            *config_copy_lines,
            '    rm -rf "$SFLOW_VENV_DIR" ${SFLOW_SRC_DIR:+"$SFLOW_SRC_DIR"} 2>/dev/null || true',
            "    # exit (not return): on a trapped signal this terminates the job instead",
            "    # of resuming the interrupted bootstrap/run; traps are disarmed above so",
            "    # this exit cannot re-enter the handler.",
            '    exit "$_sflow_rc"',
            "}",
            "trap _sflow_finalize EXIT INT TERM HUP",
            "",
            "# Fail fast during bootstrap so we never run sflow from a half-built venv;",
            "# re-disabled once ready so post-run steps (e.g. log copy) still execute.",
            "set -e",
            "",
            "# Resolve a real system python3 for venv creation: well-known absolute",
            "# locations first, then PATH (already cleaned of the caller venv).",
            'SFLOW_BOOTSTRAP_PYTHON=""',
            'for _candidate in /usr/bin/python3 /usr/local/bin/python3 "$(command -v python3 || true)"; do',
            '    if [ -n "$_candidate" ] && [ -x "$_candidate" ]; then',
            '        SFLOW_BOOTSTRAP_PYTHON="$_candidate"',
            "        break",
            "    fi",
            "done",
            'if [ -z "$SFLOW_BOOTSTRAP_PYTHON" ]; then',
            '    echo "ERROR: could not locate a system python3 to bootstrap the sflow venv" >&2',
            "    exit 1",
            "fi",
            "",
            "# Create the fresh per-job venv and install sflow into it.",
            '"$SFLOW_BOOTSTRAP_PYTHON" -m venv "$SFLOW_VENV_DIR"',
            'source "$SFLOW_VENV_DIR/bin/activate"',
            '"$VIRTUAL_ENV/bin/pip" install uv',
            *sflow_install_lines,
            '"$VIRTUAL_ENV/bin/sflow" --help',
            "set +e",
            "",
        ]
    )

    run_prelude = [
        f"cd {shlex.quote(str(workspace_dir))}",
        "",
        'export SFLOW_RUN_ID_PREFIX="$SLURM_JOB_ID"',
        "",
    ]
    # Forward the per-task log offload decision (set by --offload-task-logs or the
    # environment) into the job so the inner `sflow run` sees it even under
    # `--export=NONE`.
    _offload_env = os.environ.get(OFFLOAD_TASK_LOGS_ENV)
    if _offload_env is not None:
        run_prelude.append(
            f"export {OFFLOAD_TASK_LOGS_ENV}={shlex.quote(_offload_env)}"
        )
        run_prelude.append("")
    if multi_backend_exports:
        run_prelude.append(
            "# Multi-backend: the leader reuses this allocation; other backends each"
        )
        run_prelude.append("# run their own salloc (see SlurmBackend.allocate).")
        run_prelude.extend(multi_backend_exports)
        run_prelude.append("")

    script_lines.extend(
        [
            *run_prelude,
            "# Run sflow workflow",
            '"$SFLOW_VENV_DIR/bin/sflow" ' + " ".join(sflow_cmd_parts[1:]),
            "SFLOW_RUN_RC=$?",
            "",
            "# Exit with the workflow's status (set -e was disabled above) so Slurm",
            "# and downstream --dependency=afterok see a failed run. The finalize trap",
            "# copies the .out/.err + config(s) and cleans up regardless of this rc.",
            'exit "$SFLOW_RUN_RC"',
            "",
        ]
    )

    return "\n".join(script_lines)


@dataclass(frozen=True)
class BatchLauncherRequest:
    """Inputs needed to generate a persistent launcher artifact."""

    files: list[Path]
    set_var: list[str] | None
    artifact: list[str] | None
    missable_tasks: list[str] | None
    log_level: str
    workspace_dir: Path | None
    output_dir: Path | None
    job_name: str
    sbatch_output: str
    sbatch_error: str
    partition: str
    account: str
    time: str | None
    nodes: int | None
    gpus_per_node: int | None
    sbatch_extra_args: list[str] | None
    sflow_venv_path: Path | None
    sflow_version: str | None
    sflow_source_path: Path | None = None
    sflow_index_url: str | None = None
    enable_workflow_monitor: bool = False
    enable_task_monitors: list[str] | None = None
    include_nodes: list[str] | None = None
    exclude_nodes: list[str] | None = None
    skip_artifact_check: bool = False


@dataclass(frozen=True)
class BatchPlanRequest:
    """Inputs the launch strategy needs to plan a batch submission."""

    files: list[Path]
    set_var: list[str] | None
    cli_nodes: int | None
    cli_gpus_per_node: int | None


@dataclass(frozen=True)
class BatchPlan:
    """Strategy-produced plan for a batch submission.

    The launch strategy owns all backend-specific decisions (e.g. Slurm
    heterogeneous jobs and node/gpu derivation). The ``batch`` entry point only
    echoes ``messages``, aborts on ``error``, and feeds the resolved values back
    into the launcher request (``nodes``/``gpus_per_node``) and the dry-run
    inputs (``dry_run_nodes``/``dry_run_gpus_per_node``).
    """

    messages: list[str]
    error: str | None
    nodes: int | None
    gpus_per_node: int | None
    dry_run_nodes: int | None
    dry_run_gpus_per_node: int | None


class BatchLaunchStrategy(Protocol):
    backend_type: str

    def plan(self, request: BatchPlanRequest) -> BatchPlan:
        """Resolve backend-specific node/gpu/multi-backend planning."""
        ...

    def generate(self, request: BatchLauncherRequest) -> str:
        """Generate a persistent launcher that eventually invokes ``sflow run``."""
        ...

    def submit(self, script_path: Path) -> str:
        """Submit the generated launcher and return backend submission output."""
        ...


class SlurmBatchLaunchStrategy:
    backend_type = "slurm"

    def plan(self, request: BatchPlanRequest) -> BatchPlan:
        """Plan a Slurm batch submission (single-allocation vs multi-backend driver).

        A config with >=2 Slurm backends is submitted as a single driver sbatch
        (sized to the leader backend); each backend runs its own salloc, so
        nodes/gpus/partition are per-backend rather than a single value: report
        the per-backend plan and do not apply (or require) the single CLI
        ``-N``/``-G``. A single Slurm backend keeps the original behavior,
        deriving nodes/gpus from the config when they are not supplied on the CLI.
        """
        messages: list[str] = []
        slurm_backends = _resolve_slurm_backends(request.files, request.set_var)

        if len(slurm_backends) >= 2:
            wrapper = _select_wrapper_backend(slurm_backends)
            messages.append(
                f"  Info: {len(slurm_backends)} Slurm backends detected -> generating a "
                "multi-backend driver job (driver wraps the heaviest backend; the rest salloc):"
            )
            for b in slurm_backends:
                gpn = "unset" if b.gpus_per_node is None else str(b.gpus_per_node)
                role = (
                    "driver/leader, reuses sbatch allocation"
                    if b.name == wrapper.name
                    else "own salloc at runtime"
                )
                messages.append(
                    f"          [{b.name}] partition={b.partition}, "
                    f"nodes={b.nodes}, gpus_per_node={gpn} ({role})"
                )
            messages.append(
                "        The driver sbatch is sized to the leader backend; other backends "
                "salloc at runtime. CLI -p/--partition and -N/--nodes single values are not applied."
            )
            if request.cli_nodes is not None or request.cli_gpus_per_node is not None:
                messages.append(
                    "  Warning: --nodes/--gpus-per-node are ignored for multi-backend "
                    "jobs; each backend uses its own config values."
                )
            # gpus_per_node is planning-only regardless of het; warn once per
            # distinct configured value so the note isn't tied to a single
            # misleading number.
            for gpn_value in sorted(
                {b.gpus_per_node for b in slurm_backends if b.gpus_per_node}
            ):
                emit_gpus_per_node_semantics_warning(
                    gpn_value, messages.append, prefix="  Warning: "
                )
            return BatchPlan(
                messages=messages,
                error=None,
                nodes=request.cli_nodes,
                gpus_per_node=request.cli_gpus_per_node,
                dry_run_nodes=None,
                dry_run_gpus_per_node=None,
            )

        nodes = request.cli_nodes
        if nodes is None:
            nodes = _derive_nodes(request.files, cli_overrides=request.set_var)
            if nodes is not None:
                messages.append(
                    f"  Info: --nodes not specified, derived from config: {nodes}"
                )
            else:
                return BatchPlan(
                    messages=messages,
                    error="--nodes not specified and could not be derived from config backends.",
                    nodes=None,
                    gpus_per_node=request.cli_gpus_per_node,
                    dry_run_nodes=request.cli_nodes,
                    dry_run_gpus_per_node=request.cli_gpus_per_node,
                )
        else:
            # -N was given, so nothing above derived the config's own number -- but
            # `slurm_backends` was already resolved at the top of this method, so
            # comparing the two costs nothing and catches a mismatch that is otherwise
            # invisible until the job is allocated.
            conflict = _node_count_conflict(
                cli_nodes=nodes,
                config_nodes=slurm_backends[0].nodes if slurm_backends else None,
                origin=request.files[0].name if request.files else "the config",
            )
            if conflict is not None:
                messages.append(conflict)

        gpus_per_node = request.cli_gpus_per_node
        if gpus_per_node is None:
            gpus_per_node = _derive_gpus_per_node(
                request.files, cli_overrides=request.set_var
            )
            if gpus_per_node is not None:
                messages.append(
                    f"  Info: --gpus-per-node not specified, derived from config: {gpus_per_node}"
                )
        emit_gpus_per_node_semantics_warning(
            gpus_per_node, messages.append, prefix="  Warning: "
        )
        return BatchPlan(
            messages=messages,
            error=None,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            dry_run_nodes=request.cli_nodes,
            dry_run_gpus_per_node=request.cli_gpus_per_node,
        )

    def generate(self, request: BatchLauncherRequest) -> str:
        return _generate_sbatch_script(
            files=request.files,
            set_var=request.set_var,
            artifact=request.artifact,
            skip_artifact_check=request.skip_artifact_check,
            missable_tasks=request.missable_tasks,
            log_level=request.log_level,
            workspace_dir=request.workspace_dir,
            output_dir=request.output_dir,
            job_name=request.job_name,
            sbatch_output=request.sbatch_output,
            sbatch_error=request.sbatch_error,
            partition=request.partition,
            account=request.account,
            time=request.time,
            nodes=request.nodes,
            gpus_per_node=request.gpus_per_node,
            sbatch_extra_args=request.sbatch_extra_args,
            sflow_venv_path=request.sflow_venv_path,
            sflow_version=request.sflow_version,
            sflow_source_path=request.sflow_source_path,
            sflow_index_url=request.sflow_index_url,
            enable_workflow_monitor=request.enable_workflow_monitor,
            enable_task_monitors=request.enable_task_monitors,
            include_nodes=request.include_nodes,
            exclude_nodes=request.exclude_nodes,
        )

    def submit(self, script_path: Path) -> str:
        return _submit_sbatch(script_path)


def _batch_launch_strategy(backend_type: str = "slurm") -> BatchLaunchStrategy:
    if backend_type == "slurm":
        return SlurmBatchLaunchStrategy()
    if backend_type in {"docker", "kubernetes"}:
        raise NotImplementedError(
            f"{backend_type} persistent launcher strategy is defined by the batch "
            "interface but not implemented yet"
        )
    raise ValueError(f"Unknown batch backend strategy: {backend_type}")


def _write_slurm_dry_run_config(
    *,
    files: list[Path],
    variable_overrides: list[str] | None,
    artifact_overrides: list[str] | None,
    missable_tasks: list[str] | None,
    nodes: int | None,
    gpus_per_node: int | None,
    directory: Path,
) -> Path | None:
    """Write a composed dry-run config with Slurm batch overrides applied."""
    if nodes is None and gpus_per_node is None:
        return None

    from sflow.config.loader import ConfigLoader

    config = ConfigLoader().load_configs(
        files,
        variable_overrides,
        artifact_overrides,
        missable_tasks,
    )
    updated_backends = []
    changed = False
    for backend_conf in config.backends or []:
        if getattr(backend_conf, "type", None) != "slurm":
            updated_backends.append(backend_conf)
            continue
        updates: dict[str, int] = {}
        if nodes is not None:
            updates["nodes"] = nodes
        if gpus_per_node is not None:
            updates["gpus_per_node"] = gpus_per_node
        updated_backends.append(backend_conf.model_copy(update=updates))
        changed = True

    if not changed:
        return None

    dry_run_config = config.model_copy(update={"backends": updated_backends})
    data = dry_run_config.model_dump(mode="json", exclude_none=True)
    # `backends` is typed as the base BackendConfig, so the model-level dump above
    # serializes each backend with only base fields and silently drops
    # subclass-specific ones (e.g. slurm account/partition/time/nodes), producing
    # a config that then fails re-validation. Re-dump each backend via its
    # concrete type to preserve all of its fields.
    data["backends"] = [
        backend.model_dump(mode="json", exclude_none=True)
        for backend in updated_backends
    ]
    # `operators` is likewise typed as the base OperatorConfig, so its subclass
    # fields (e.g. srun name/container_image) are dropped by the model-level dump;
    # re-dump each operator via its concrete type as well.
    operators = getattr(dry_run_config, "operators", None)
    if operators:
        data["operators"] = [
            operator.model_dump(mode="json", exclude_none=True)
            for operator in operators
        ]
    path = directory / "sflow-batch-dry-run.yaml"
    path.write_text(_yaml.safe_dump(data, sort_keys=False))
    return path


def _slurm_dry_run_inputs(
    *,
    files: list[Path],
    variable_overrides: list[str] | None,
    artifact_overrides: list[str] | None,
    missable_tasks: list[str] | None,
    nodes: int | None,
    gpus_per_node: int | None,
    directory: Path,
) -> tuple[list[Path], list[str] | None, list[str] | None, list[str] | None]:
    override_file = _write_slurm_dry_run_config(
        files=files,
        variable_overrides=variable_overrides,
        artifact_overrides=artifact_overrides,
        missable_tasks=missable_tasks,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        directory=directory,
    )
    if override_file is None:
        return files, variable_overrides, artifact_overrides, missable_tasks
    return [override_file], None, None, None


_RESERVED_CSV_COLUMNS = frozenset({"sflow_config_file", "job_name", "missable_tasks"})
_NODE_COLUMN_NAMES = frozenset({"SLURM_NODES", "NUM_SLURM_NODES", "NUM_NODES"})


def parse_row_selector(values: list[str], *, n_rows: int | None = None) -> list[int]:
    """Parse ``--row`` values into a flat sorted list of 1-based row indices.

    Supported formats (all 1-based; slice end is **exclusive** like Python):

    * Single int:        ``--row 1``
    * Negative int:      ``--row -1``       →  last row
    * Comma-separated:   ``--row 1,3,5``   or   ``--row [1,3,5]``
    * Slice:             ``--row 1:4``      →  rows 1, 2, 3
    * Slice with step:   ``--row 1:6:2``    →  rows 1, 3, 5
    * Open-ended slice:  ``--row 3:``       →  row 3 to last (needs *n_rows*)
    * Negative slice:    ``--row -3:``      →  last 3 rows   (needs *n_rows*)
    * Brackets optional: ``--row [1:4]``    same as ``--row 1:4``

    Multiple ``--row`` flags are combined:  ``--row 1:3 --row 7``  →  [1, 2, 7]

    Negative indices follow Python semantics: ``-1`` is the last row, ``-2``
    is second-to-last, etc.  When *n_rows* is ``None``, negative indices and
    open-ended slices are kept as-is (callers must resolve them later via
    :func:`resolve_row_indices`).
    """
    indices: set[int] = set()
    for raw in values:
        token = raw.strip().strip("[]")
        if not token:
            continue
        if "," in token:
            for part in token.split(","):
                part = part.strip()
                if part:
                    indices.update(_parse_single_or_slice(part, n_rows=n_rows))
        else:
            indices.update(_parse_single_or_slice(token, n_rows=n_rows))
    result = sorted(indices, key=lambda x: (x < 0, x))
    if n_rows is not None:
        result = resolve_row_indices(result, n_rows)
    return result


def resolve_row_indices(indices: list[int], n_rows: int) -> list[int]:
    """Resolve negative 1-based row indices to positive ones.

    Negative indices map like Python: ``-1 → n_rows``, ``-2 → n_rows - 1``, etc.
    After resolution, indices outside ``[1, n_rows]`` are dropped with a warning.
    """
    resolved: set[int] = set()
    for idx in indices:
        pos = n_rows + 1 + idx if idx < 0 else idx
        if 1 <= pos <= n_rows:
            resolved.add(pos)
        else:
            typer.echo(
                f"  Warning: row index {idx} (resolved to {pos}) "
                f"is out of range [1, {n_rows}]; skipping.",
                err=True,
            )
    return sorted(resolved)


def _parse_single_or_slice(token: str, *, n_rows: int | None = None) -> list[int]:
    """Parse a single int or a start:stop[:step] slice into 1-based indices.

    Open-ended slices (``3:``, ``:-2``) require *n_rows* to resolve the missing
    bound.  When *n_rows* is ``None`` and the slice is open-ended, a
    :class:`typer.BadParameter` is raised.
    """
    if ":" in token:
        parts = token.split(":")
        if len(parts) == 2:
            start_s, stop_s = parts
            step = 1
        elif len(parts) == 3:
            start_s, stop_s, step_s = parts
            step = int(step_s) if step_s else 1
        else:
            raise typer.BadParameter(
                f"Invalid slice: '{token}' (expected start:stop or start:stop:step)"
            )
        if step == 0:
            raise typer.BadParameter("Slice step cannot be zero")

        has_open_end = not start_s or not stop_s
        if has_open_end and n_rows is None:
            raise typer.BadParameter(
                f"Open-ended slice '{token}' requires known row count. "
                f"This will be resolved automatically when used with --bulk-input."
            )

        if not start_s:
            start = 1
        else:
            start = int(start_s)
            if start < 0 and n_rows is not None:
                start = n_rows + 1 + start

        if not stop_s:
            stop = n_rows + 1  # type: ignore[operator]
        else:
            stop = int(stop_s)
            if stop < 0 and n_rows is not None:
                stop = n_rows + 1 + stop

        return list(range(start, stop, step))
    return [int(token)]


_MAX_NAME_LEN = 30


def _sanitize_name(name: str, max_len: int = _MAX_NAME_LEN) -> str:
    """Sanitize a name for use as a filename / Slurm job name.

    Replaces non-alphanumeric characters (except ``_`` and ``-``) with ``_``,
    collapses consecutive ``_``, strips leading/trailing ``_``, and truncates.
    """
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned[:max_len].rstrip("_") if cleaned else "row"


def _dedup_words(name: str) -> str:
    """Remove duplicate words from an underscore-separated name, preserving order.

    ``trtllm_prefill_trtllm_decode`` → ``trtllm_prefill_decode``
    """
    seen: set[str] = set()
    out: list[str] = []
    for word in name.split("_"):
        if word and word not in seen:
            seen.add(word)
            out.append(word)
    return "_".join(out)


def _normalize_col_value(value: str) -> str | None:
    """Normalize a CSV column value for inclusion in a derived name.

    Returns ``None`` to skip the value entirely, or a shortened string:

    * Path URIs (``fs:///…``, ``s3://…``) → ``None``
    * Absolute paths (``/…``) → ``None``
    * Container images (``registry/image:tag``, e.g. ``nvcr.io/…:0.8.0``) → ``None``
    * Everything else → returned as-is
    """
    if "://" in value or value.startswith("/") or "/" in value:
        return None
    return value


def _path_to_stem(raw: str) -> str:
    """Convert a config file path to a descriptive stem for naming.

    Joins directory components and the file stem with ``_`` so that relative
    paths like ``trtllm/prefill.yaml`` become ``trtllm_prefill`` instead of
    just ``prefill``.  Absolute paths and bare filenames fall back to the
    plain stem.
    """
    p = Path(raw.strip())
    if p.is_absolute():
        return p.stem
    parts = list(p.parent.parts) + [p.stem]
    parts = [part for part in parts if part not in (".", "..")]
    return "_".join(parts) if parts else p.stem


class _RowNamingCtx:
    """Precomputed context for deriving row names from a CSV.

    Built once from all rows, then passed to each ``_derive_row_name`` call
    so that common-stem detection and differing-column detection are O(1) per row
    instead of O(R) per row (eliminating the O(R²) total cost).
    """

    __slots__ = (
        "common_stems",
        "differing_cols",
        "cli_nodes",
        "fallback_base",
        "cli_var_overrides",
    )

    def __init__(
        self,
        all_rows: list[dict[str, str]],
        fallback_base: str = "sflow",
        cli_nodes: int | None = None,
        cli_var_overrides: dict[str, str] | None = None,
    ) -> None:
        self.fallback_base = fallback_base
        self.cli_nodes = cli_nodes
        # Global ``--set`` overrides (shared by every row); a node-count override here
        # wins over a row's CSV node cell so the derived name matches the allocation.
        self.cli_var_overrides = cli_var_overrides or {}

        all_stem_sets = [
            {_path_to_stem(p) for p in r["sflow_config_file"].split()} for r in all_rows
        ]
        self.common_stems: set[str] = (
            set.intersection(*all_stem_sets) if all_stem_sets else set()
        )

        skip_cols = _RESERVED_CSV_COLUMNS | _NODE_COLUMN_NAMES
        self.differing_cols: list[str] = []
        if all_rows:
            candidate_cols = [c for c in all_rows[0] if c not in skip_cols]
            for col in candidate_cols:
                all_vals = {(r.get(col) or "").strip() for r in all_rows}
                if len(all_vals) > 1:
                    self.differing_cols.append(col)


def build_row_naming_ctx(
    all_rows: list[dict[str, str]],
    fallback_base: str = "sflow",
    cli_nodes: int | None = None,
    cli_var_overrides: dict[str, str] | None = None,
) -> _RowNamingCtx:
    """Build the shared naming context once before iterating rows."""
    return _RowNamingCtx(
        all_rows,
        fallback_base=fallback_base,
        cli_nodes=cli_nodes,
        cli_var_overrides=cli_var_overrides,
    )


def _node_count_column(source: "dict[str, Any]") -> "tuple[str, int] | None":
    """First parseable node-count column as ``(name, value)``, or None.

    Scans ``_NODE_COLUMN_NAMES`` in SORTED order. Iterating the frozenset directly made
    the answer depend on PYTHONHASHSEED: a CSV carrying two node columns picked a
    different one per process, so the count a run was submitted with was a coin flip,
    and the column named by the conflict message below could disagree with the value it
    was complaining about.

    ``source`` is a merged override map (e.g. ``all_overrides`` / a var map) so a CLI
    ``--set`` value already wins over the CSV cell. A present-but-non-numeric value is
    SKIPPED (the scan continues) rather than aborting, so one malformed column can't
    mask a good one. Single source of truth for the node-column peek used by the naming,
    both bulk paths, and the conflict checks.
    """
    for col in sorted(_NODE_COLUMN_NAMES):
        raw = source.get(col)
        val = raw.strip() if isinstance(raw, str) else raw
        if val in (None, ""):
            continue
        try:
            return col, int(val)
        except (ValueError, TypeError):
            continue
    return None


def _first_node_column_int(source: "dict[str, Any]") -> int | None:
    """Just the count from :func:`_node_count_column`."""
    found = _node_count_column(source)
    return found[1] if found is not None else None


def _node_count_conflict_message(*, nodes: int, source: str, listed: str) -> str:
    """Why two node counts cannot both be right. Shared by every path that has both.

    ``--nodes`` sizes the sbatch allocation. The config's own number sizes the *recipe*:
    the backend's node count and the ``match_count`` of readiness probes. The same number
    in both is fine and is the normal way to run -- but when they differ, the quiet
    precedence rule (``--nodes`` wins for sbatch, the config keeps its own value)
    allocates one number of nodes and plans the workflow for another: probes wait on a
    node that was never allocated, or the job holds nodes nothing will ever use.
    """
    return (
        f"--nodes={nodes} does not match the node count this run would use: {listed} "
        f"(from {source}). --nodes sizes the sbatch allocation, while that number sizes "
        f"the workflow inside it -- the backend's node count and the match_count of "
        f"readiness probes -- so two different numbers means the job is allocated one "
        f"size and the recipe plans for another. Use the same number in both, or drop "
        f"--nodes and let {source} size both."
    )


def _reject_conflicting_node_counts(
    *,
    rows: "list[dict[str, str]]",
    row_indices: "set[int] | None",
    nodes: int | None,
    cli_var_map: dict[str, str],
    csv_path: Path,
) -> None:
    """Refuse a bulk run whose two node counts disagree.

    See :func:`_node_count_conflict_message` for why they cannot both be right. This is
    the ``--bulk-input`` path, where the CSV states the size per row, so a disagreement
    is unambiguously a mistake and is refused outright. The config-driven paths only warn
    -- see :func:`_node_count_conflict`.

    Only the rows this run actually submits are checked, and a ``--set`` of the node
    variable counts as the row's value, since it overrides the CSV cell.
    """
    if nodes is None:
        return

    conflicts: list[tuple[int, str, int]] = []
    for idx, row in enumerate(rows, start=1):
        if row_indices is not None and idx not in row_indices:
            continue
        # CLI --set wins over the CSV cell, the same merge the row overrides use.
        found = _node_count_column({**row, **cli_var_map})
        if found is not None and found[1] != nodes:
            conflicts.append((idx, found[0], found[1]))

    if not conflicts:
        return

    # Name the column the conflicting value actually came from, rather than re-scanning
    # for one: with two node columns present those are not necessarily the same column.
    column = conflicts[0][1]
    source = (
        f"--set {column}"
        if column in cli_var_map
        else f"the {column} column of {csv_path.name}"
    )
    listed = ", ".join(f"row {idx} says {value}" for idx, _col, value in conflicts[:5])
    if len(conflicts) > 5:
        listed += f", and {len(conflicts) - 5} more"

    raise ValueError(
        _node_count_conflict_message(nodes=nodes, source=source, listed=listed)
    )


def _node_count_conflict(
    *, cli_nodes: int | None, config_nodes: "int | None", origin: str
) -> str | None:
    """The warning for a ``--nodes`` that disagrees with the config's own node count.

    ``--bulk-input`` REFUSES its version of this (:func:`_reject_conflicting_node_counts`):
    a CSV node column is that row's declared size, so a mismatch there is a mistake. Here
    the number lives in the config, where ``--nodes`` has always been allowed to size the
    allocation on its own -- so this only says so, and the caller decides where to print
    it (the plan's message list, or straight to stderr in a bulk loop).

    Worth saying at all because the mismatch is otherwise invisible until the job is
    already allocated: the slurm backend logs ``config_nodes=... env_nodes=...`` and
    continues, then srun asks for more tasks than there are nodes and readiness probes
    wait on a ``match_count`` that can never arrive.
    """
    if cli_nodes is None or config_nodes is None or int(config_nodes) == cli_nodes:
        return None
    return "  Warning: " + _node_count_conflict_message(
        nodes=cli_nodes,
        source=f"the node count in {origin}",
        listed=f"{origin} says {config_nodes}",
    )


def _resolve_node_count(
    row: dict[str, str],
    cli_nodes: int | None,
    cli_var_overrides: dict[str, str] | None = None,
) -> str | None:
    """Return the node count for a row as ``<N>n``, or None if unknown.

    Precedence mirrors the sbatch node sizing so the name tracks the allocation:
    an explicit ``--nodes`` wins, then a global ``--set`` override of a node-count
    variable (not the stale CSV cell), then the row's CSV node column.
    """
    if cli_nodes is not None:
        return f"{cli_nodes}n"
    # CLI --set wins over the CSV cell per node-count column, then a malformed value is
    # skipped instead of masking a good column (see _first_node_column_int).
    n = _first_node_column_int({**row, **(cli_var_overrides or {})})
    return f"{n}n" if n is not None else None


def _derive_row_name(
    row: dict[str, str],
    idx: int,
    ctx: _RowNamingCtx,
) -> str:
    """Derive a descriptive name for a bulk-input CSV row.

    Uses a precomputed ``_RowNamingCtx`` so each call is O(F + C) instead of
    O(R * (F + C)), keeping total batch naming O(R * (F + C)) instead of O(R²).

    Priority:
    1. Explicit ``job_name`` column value (if present and non-empty).
    2. Auto-derived from multiple sources:
       a. Unique config file stems (stems not shared by every row).
          Handles relative paths (``../../dir/file.yaml`` → stem ``file``).
       b. Node count (always included as ``<N>n``, from CLI --nodes or CSV column).
       c. Unique column values (short, non-path values that differ across rows;
          node columns are excluded since they are already handled above).
    3. Falls back to *fallback_base* when nothing distinguishes the row.

    A post-processing step deduplicates repeated words
    (e.g. ``trtllm_prefill_trtllm_decode`` → ``trtllm_prefill_decode``).
    The row index ``_{idx:03d}`` is always appended for guaranteed uniqueness.
    The base name is capped at 30 characters before the suffix.
    """
    explicit = (row.get("job_name") or "").strip()
    if explicit:
        return f"{_sanitize_name(explicit)}_{idx:03d}"

    parts: list[str] = []

    row_stems = [_path_to_stem(p) for p in row["sflow_config_file"].split()]
    parts.extend(s for s in row_stems if s not in ctx.common_stems)

    node_label = _resolve_node_count(row, ctx.cli_nodes, ctx.cli_var_overrides)
    if node_label:
        parts.append(node_label)

    for col in ctx.differing_cols:
        val = (row.get(col) or "").strip()
        if not val:
            continue
        normed = _normalize_col_value(val)
        if normed is None:
            continue
        parts.append(normed)

    base = "_".join(parts) if parts else ctx.fallback_base
    base = _dedup_words(base)
    return f"{_sanitize_name(base)}_{idx:03d}"


def _submit_sbatch(script_path: Path) -> str:
    """Submit an sbatch script and return the stdout message (e.g. job id)."""
    import subprocess

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _build_var_map(
    data: dict, cli_overrides: list[str] | None = None
) -> dict[str, Any]:
    """Build a variable name->value map from raw sflow YAML data.

    Handles both variable formats:
    - dict-of-dict: ``{KEY: {description: …, value: …}}``
    - list-of-dict: ``[{name: KEY, value: …}]``
    - simple dict:  ``{KEY: scalar_value}``

    Also includes workflow-level variables.  CLI ``--set`` overrides
    (``KEY=VALUE`` strings) are applied last with highest priority.
    """
    var_map: dict[str, Any] = {}
    raw_vars = data.get("variables") or []
    if isinstance(raw_vars, dict):
        for k, v in raw_vars.items():
            if isinstance(v, dict):
                var_map[k] = v.get("value")
            else:
                var_map[k] = v
    elif isinstance(raw_vars, list):
        for v in raw_vars:
            if isinstance(v, dict) and "name" in v:
                var_map[v["name"]] = v.get("value")

    wf = data.get("workflow")
    if isinstance(wf, dict):
        wf_vars = wf.get("variables") or []
        if isinstance(wf_vars, dict):
            for k, v in wf_vars.items():
                if isinstance(v, dict):
                    var_map[k] = v.get("value")
                else:
                    var_map[k] = v
        elif isinstance(wf_vars, list):
            for v in wf_vars:
                if isinstance(v, dict) and "name" in v:
                    var_map[v["name"]] = v.get("value")

    for entry in cli_overrides or []:
        if "=" in entry:
            k, v_str = entry.split("=", 1)
            var_map[k] = v_str

    return var_map


def _resolve_backend_int_field(
    data: dict, field: str, var_map: dict[str, Any]
) -> int | None:
    """Resolve an integer field from the first slurm backend definition.

    If the value is a ``${{ variables.X }}`` expression, look it up in *var_map*.
    Returns the resolved integer, or ``None`` if the field is absent or unresolvable.
    """
    import re as _re

    for b in data.get("backends") or []:
        if not isinstance(b, dict) or b.get("type") != "slurm":
            continue
        if field not in b:
            continue
        val = b[field]
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            m = _re.search(r"\$\{\{\s*variables\.(\w+)\s*\}\}", val)
            if m:
                ref_val = var_map.get(m.group(1))
                if ref_val is not None:
                    try:
                        return int(ref_val)
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
    return None


def _derive_backend_int(
    config_files: list[Path],
    field: str,
    cli_overrides: list[str] | None = None,
) -> int | None:
    """Resolve an int Slurm-backend field (``nodes`` / ``gpus_per_node``) from the config.

    Fast path: a lightweight per-file regex peek (handles ``<field>: <int>`` and a bare
    ``${{ variables.X }}``). If that can't resolve the value (e.g. a compound expression
    like ``${{ variables.NUM_NODES * 2 }}``), fall back to the SAME full resolution
    pipeline as the dry-run (:func:`_resolve_slurm_backends`) so the generated sbatch
    matches ``sflow run`` -- no dry-run/generated-script divergence. The fallback only
    runs when the regex returns None, so currently-resolving configs are unchanged and
    partial fragments (which the pipeline can't validate) keep their regex result.
    """

    merged_var_map: dict[str, Any] = {}
    all_data: list[dict] = []

    for f in config_files:
        try:
            with open(f) as fh:
                raw = safe_load(fh)
            if isinstance(raw, dict):
                all_data.append(raw)
                merged_var_map.update(_build_var_map(raw))
        except Exception:
            continue

    for entry in cli_overrides or []:
        if "=" in entry:
            k, v = entry.split("=", 1)
            merged_var_map[k] = v

    for d in all_data:
        result = _resolve_backend_int_field(d, field, merged_var_map)
        if result is not None:
            return result

    for backend in _resolve_slurm_backends(config_files, cli_overrides):
        val = getattr(backend, field, None)
        if val is not None:
            return int(val)
    return None


def _derive_gpus_per_node(
    config_files: list[Path],
    cli_overrides: list[str] | None = None,
) -> int | None:
    """Derive ``gpus_per_node`` from the config's Slurm backend (see
    :func:`_derive_backend_int` for the regex-fast-path + full-pipeline-fallback)."""
    return _derive_backend_int(config_files, "gpus_per_node", cli_overrides)


def _derive_nodes(
    config_files: list[Path],
    cli_overrides: list[str] | None = None,
) -> int | None:
    """Derive ``nodes`` from the config's Slurm backend (see :func:`_derive_backend_int`
    for the regex-fast-path + full-pipeline-fallback)."""
    return _derive_backend_int(config_files, "nodes", cli_overrides)


def _classify_csv_columns(
    columns: list[str],
    row_configs: list[tuple[list[Path], list[str] | None]],
) -> tuple[set[str], set[str]]:
    """Classify CSV column names as variable overrides or artifact overrides.

    Checks columns against ALL config file sets (one per CSV row).
    Each entry is ``(config_files, per_row_missable_tasks)``.
    A column is valid if it matches a variable or artifact in ANY row's config.

    Returns (var_columns, art_columns).
    Raises ValueError if a column matches neither in any config set.
    """
    from sflow.config.loader import ConfigLoader

    var_names: set[str] = set()
    art_names: set[str] = set()
    seen: set[tuple[str, ...]] = set()
    load_errors: list[tuple[tuple[str, ...], Exception]] = []
    loaded_count = 0

    for config_files, row_missable in row_configs:
        key = tuple(str(f) for f in config_files)
        if key in seen:
            continue
        seen.add(key)
        try:
            config = ConfigLoader().load_configs(
                config_files, missable_tasks=row_missable
            )
        except Exception as exc:
            load_errors.append((key, exc))
            continue
        loaded_count += 1
        for v in config.variables or []:
            var_names.add(v.name)
        wf = config.workflow
        if wf and wf.variables:
            for v in wf.variables:
                var_names.add(v.name)
        for a in config.artifacts or []:
            art_names.add(a.name)

    if load_errors:
        _logger.warning(
            f"{len(load_errors)} config file set(s) failed to load "
            f"({loaded_count} succeeded):"
        )
        for files, exc in load_errors:
            file_list = " + ".join(files)
            _logger.warning(f"  ⚠ [{file_list}]: {exc}")
        if loaded_count == 0:
            _logger.warning(
                "  No config sets loaded successfully. If tasks from one file "
                "reference tasks in another, consider adding --missable-tasks / -M "
                "or a 'missable_tasks' CSV column."
            )

    var_cols: set[str] = set()
    art_cols: set[str] = set()
    for col in columns:
        if col in _RESERVED_CSV_COLUMNS:
            continue
        if col in var_names:
            var_cols.add(col)
        elif col in art_names:
            art_cols.add(col)
        else:
            msg = (
                f"CSV column '{col}' is not a variable or artifact "
                f"defined in any of the config file sets"
            )
            if load_errors and loaded_count == 0:
                msg += (
                    f". Note: all {len(load_errors)} config set(s) failed to load"
                    f" — the root cause is likely a config loading error above, "
                    f"not a missing variable. Common fix: add --missable-tasks / -M "
                    f"for tasks referenced in depends_on that don't exist in "
                    f"all files, or add a 'missable_tasks' column to the CSV."
                )
            raise ValueError(msg)
    return var_cols, art_cols


def read_bulk_csv(csv_path: Path) -> tuple[list[str], list[dict]]:
    """Read and validate a bulk-input CSV file.

    Returns (columns, rows).
    Raises ValueError if the file is empty, lacks the ``sflow_config_file`` column, has
    no data rows, or has a row that leaves that required column blank.
    """
    import csv

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {csv_path}")
        columns = list(reader.fieldnames)
        if "sflow_config_file" not in columns:
            raise ValueError(
                f"CSV must contain a 'sflow_config_file' column. Found: {columns}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file has no data rows: {csv_path}")
    # A present HEADER is not the same as a present VALUE. ``csv.DictReader`` pads a row
    # that has fewer fields than the header with None, so a short row yields
    # ``{"sflow_config_file": None}`` and sails past the column check above -- then every
    # downstream ``row["sflow_config_file"].split()`` raises AttributeError, which escapes
    # typer as a raw traceback instead of a CLI error. Validating here, at the one place
    # every caller (batch, compose, run) reads a bulk CSV, fixes all of those at once and
    # keeps the guard from having to be re-derived at each dereference.
    #
    # Rejecting rather than coercing to "" is deliberate: an empty file list is not a
    # runnable row, so coercion would only defer the failure to a less obvious place.
    for number, row in enumerate(rows, start=1):
        if not (row.get("sflow_config_file") or "").strip():
            raise ValueError(
                f"CSV data row {number} has no value for the required "
                f"'sflow_config_file' column: {csv_path}. Every row must name at least "
                "one YAML config file (space-separate several to merge them). A row with "
                "fewer fields than the header will do this -- check for a missing comma."
            )
    return columns, rows


def resolve_row_files(
    row: dict, csv_dir: Path, resolved_cli_files: list[Path],
) -> list[Path]:
    """Resolve and dedup config file paths for a single CSV row.

    CLI files are prepended; CSV paths are resolved relative to *csv_dir*.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    for p in resolved_cli_files + [(csv_dir / x).resolve() for x in row["sflow_config_file"].split()]:
        if p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


def row_missable(row: dict, cli_missable: list[str] | None) -> list[str] | None:
    """Merge CLI and CSV ``missable_tasks`` for a single row."""
    m = list(cli_missable) if cli_missable else []
    csv_m = (row.get("missable_tasks") or "").strip()
    if csv_m:
        m.extend(csv_m.split())
    return m or None


def build_all_row_configs(
    rows: list[dict],
    csv_dir: Path,
    resolved_cli_files: list[Path],
    cli_missable: list[str] | None,
) -> list[tuple[list[Path], list[str] | None]]:
    """Build (config_files, missable) tuples for all rows, for column classification."""
    return [
        (resolve_row_files(r, csv_dir, resolved_cli_files), row_missable(r, cli_missable))
        for r in rows
    ]


def _parse_kv_list(entries: list[str] | None) -> dict[str, str]:
    """Parse a list of 'KEY=VALUE' strings into a dict."""
    result: dict[str, str] = {}
    for entry in entries or []:
        if "=" in entry:
            k, v = entry.split("=", 1)
            result[k] = v
    return result


def merge_row_overrides(
    row: dict,
    var_cols: set[str],
    art_cols: set[str],
    cli_var_map: dict[str, str],
    cli_art_map: dict[str, str],
) -> tuple[list[str] | None, list[str] | None]:
    """Merge CLI and CSV overrides for a single row.

    For variables, CLI ``--set`` takes precedence over CSV values.
    For artifacts, CLI ``--artifact`` takes precedence over CSV values.

    Returns (set_var_list, artifact_list).
    """
    merged_vars: dict[str, str] = {}
    for col in var_cols:
        if row.get(col):
            merged_vars[col] = row[col]
    merged_vars.update(cli_var_map)
    set_var = [f"{k}={v}" for k, v in merged_vars.items()] or None

    merged_arts: dict[str, str] = {}
    for col in art_cols:
        if row.get(col):
            merged_arts[col] = row[col]
    merged_arts.update(cli_art_map)
    artifacts = [f"{k}={v}" for k, v in merged_arts.items()] or None

    return set_var, artifacts


def resolve_csv_row(
    csv_path: Path,
    row_idx: int,
    cli_files: list[Path] | None = None,
    cli_set_var: list[str] | None = None,
    cli_artifact: list[str] | None = None,
    cli_missable: list[str] | None = None,
) -> tuple[list[Path], list[str] | None, list[str] | None, list[str] | None]:
    """Resolve a single CSV row into (config_files, set_var, artifact, missable_tasks).

    High-level convenience that reads the CSV, classifies columns, and merges
    overrides for the selected row (1-based index).
    Used by ``sflow run --bulk-input``.
    """
    columns, rows = read_bulk_csv(csv_path)
    if row_idx < 0:
        row_idx = len(rows) + 1 + row_idx
    if row_idx < 1 or row_idx > len(rows):
        raise IndexError(f"Row {row_idx} out of range (CSV has {len(rows)} rows)")

    csv_dir = csv_path.parent
    resolved_cli = [fp.resolve() for fp in (cli_files or [])]

    all_row_configs = build_all_row_configs(rows, csv_dir, resolved_cli, cli_missable)
    var_cols, art_cols = _classify_csv_columns(columns, all_row_configs)

    row = rows[row_idx - 1]
    config_files = resolve_row_files(row, csv_dir, resolved_cli)
    missable = row_missable(row, cli_missable)

    cli_var_map = _parse_kv_list(cli_set_var)
    cli_art_map = _parse_kv_list(cli_artifact)
    set_var, artifacts = merge_row_overrides(row, var_cols, art_cols, cli_var_map, cli_art_map)

    return config_files, set_var, artifacts, missable


def _scan_sflow_yamls(paths: list[Path]) -> list[Path]:
    """Scan file paths, directories, and glob patterns for valid sflow YAML configs.

    A valid sflow YAML is a ``*.yaml`` or ``*.yml`` file whose top-level
    mapping contains a ``version`` key.

    Supports:
    - Explicit file paths (``workflow.yaml``)
    - Directories (scanned for ``*.yaml`` / ``*.yml``)
    - Glob patterns (``examples/self_contained/slurm/*``, ``configs/**/*.yaml``)
    """
    import glob as _glob


    candidates: list[Path] = []
    for p in paths:
        if p.is_dir():
            candidates.extend(sorted(p.glob("*.yaml")))
            candidates.extend(sorted(p.glob("*.yml")))
        elif p.is_file():
            if p.suffix in (".yaml", ".yml"):
                candidates.append(p)
        else:
            expanded = sorted(Path(m) for m in _glob.glob(str(p)))
            if expanded:
                for ep in expanded:
                    if ep.is_file() and ep.suffix in (".yaml", ".yml"):
                        candidates.append(ep)
                    elif ep.is_dir():
                        candidates.extend(sorted(ep.glob("*.yaml")))
                        candidates.extend(sorted(ep.glob("*.yml")))

    valid: list[Path] = []
    for f in candidates:
        try:
            with open(f) as fh:
                data = safe_load(fh)
            if isinstance(data, dict) and "workflow" in data:
                valid.append(f.resolve())
        except Exception:
            continue
    return sorted(set(valid))


def _failure_detail(error: Exception, indent: str = "      ") -> str:
    """Render an exception for the end-of-run failure block, reason included.

    The failure list used to keep ``str(e).split("\\n")[0]``, which is wrong for
    exactly the errors that need explaining: a pydantic ``ValidationError`` renders as
    a header line followed by one stanza per problem, and the config loader wraps it as
    ``"Configuration validation failed:\\n<detail>"``. The first line is the header, so
    a bad config was reported as ``Configuration validation failed:`` and nothing else
    -- the run knew which field was wrong and threw that away.

    So the whole message is kept: first line inline with the caller's ``[idx]`` prefix,
    the rest indented under it. Not truncated -- an error long enough to be a problem
    to read has not turned up, and a cap that hides the tail of a validation error
    would reintroduce the bug this exists to fix.
    """
    lines = str(error).splitlines() or [repr(error)]
    return "\n".join([lines[0]] + [f"{indent}{line}" for line in lines[1:]])


def _run_bulk_submit(
    *,
    yaml_files: list[Path],
    cli_set_var: list[str] | None,
    cli_artifact: list[str] | None,
    log_level: str,
    workspace_dir: Path | None,
    output_dir: Path | None,
    sbatch_output: str,
    sbatch_error: str,
    partition: str,
    account: str,
    time: str | None,
    nodes: int | None,
    gpus_per_node: int | None,
    sbatch_extra_args: list[str] | None,
    sflow_venv_path: Path | None,
    sflow_version: str | None,
    sflow_source_path: Path | None = None,
    sflow_index_url: str | None = None,
    submit: bool,
    missable_tasks: list[str] | None = None,
    resolve: bool = False,
    enable_workflow_monitor: bool = False,
    enable_task_monitors: list[str] | None = None,
    include_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    skip_artifact_check: bool = False,
) -> None:
    """Process multiple self-contained sflow YAML configs as individual batch jobs."""
    import re as _re
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    effective_output = output_dir or Path.cwd() / "sflow_output"
    bulk_dir = effective_output / f"bulk_submit_{stamp}"
    bulk_dir.mkdir(parents=True, exist_ok=True)

    _SBATCH_JOB_ID_RE = _re.compile(r"Submitted batch job (\d+)")

    cli_var_keys: set[str] = set()
    for entry in cli_set_var or []:
        if "=" in entry:
            cli_var_keys.add(entry.split("=", 1)[0])

    summary: list[str] = []
    failures: list[str] = []
    failed_count = 0
    result_rows: list[dict[str, str]] = []

    for idx, yaml_file in enumerate(yaml_files, start=1):
        job_name = _sanitize_name(yaml_file.stem)
        typer.echo(f"\n[{idx}/{len(yaml_files)}] Processing {yaml_file.name} ...")

        # Warn about CLI variable overrides
        if cli_var_keys:
            try:

                with open(yaml_file) as fh:
                    data = safe_load(fh)
                config_var_names: set[str] = set()
                raw_vars = data.get("variables") or []
                if isinstance(raw_vars, dict):
                    config_var_names.update(raw_vars.keys())
                elif isinstance(raw_vars, list):
                    for v in raw_vars:
                        if isinstance(v, dict) and "name" in v:
                            config_var_names.add(v["name"])
                        elif isinstance(v, str):
                            config_var_names.add(v)
                wf = data.get("workflow")
                if isinstance(wf, dict):
                    wf_vars = wf.get("variables") or []
                    if isinstance(wf_vars, dict):
                        config_var_names.update(wf_vars.keys())
                    elif isinstance(wf_vars, list):
                        for v in wf_vars:
                            if isinstance(v, dict) and "name" in v:
                                config_var_names.add(v["name"])
                overlap = cli_var_keys & config_var_names
                for name in sorted(overlap):
                    typer.echo(
                        f"  Warning: variable '{name}' in {yaml_file.name} overridden by --set",
                        err=True,
                    )
            except Exception:
                pass

        # Derive gpus_per_node: config value wins over CLI
        config_gpus = _derive_gpus_per_node([yaml_file], cli_overrides=cli_set_var)
        row_gpus = config_gpus if config_gpus is not None else gpus_per_node
        emit_gpus_per_node_semantics_warning(
            row_gpus,
            lambda message: typer.echo(message, err=True),
            prefix="  Warning: ",
        )
        if (
            gpus_per_node is not None
            and config_gpus is not None
            and gpus_per_node != config_gpus
        ):
            typer.echo(
                f"  Warning: --gpus-per-node={gpus_per_node} overridden by "
                f"{yaml_file.name} config value ({config_gpus})",
                err=True,
            )

        # Dry-run validation
        try:
            # sflow batch is Slurm-only; a Kubernetes-backed config has no sbatch
            # step. Reject it here so it is reported as a per-config failure (use
            # `sflow run` for k8s) instead of emitting a meaningless script.
            _reject_kubernetes_batch(
                [yaml_file], list(cli_set_var) if cli_set_var else None
            )
            with tempfile.TemporaryDirectory(prefix="sflow-batch-dry-run-") as tmp:
                dry_files, dry_vars, dry_artifacts, dry_missable = _slurm_dry_run_inputs(
                    files=[yaml_file],
                    variable_overrides=list(cli_set_var) if cli_set_var else None,
                    artifact_overrides=list(cli_artifact) if cli_artifact else None,
                    missable_tasks=missable_tasks,
                    nodes=nodes,
                    gpus_per_node=None if config_gpus is not None else gpus_per_node,
                    directory=Path(tmp),
                )
                _sflow_app.run(
                    file=dry_files,
                    dry_run=True,
                    variable_overrides=dry_vars,
                    artifact_overrides=dry_artifacts,
                    missable_tasks=dry_missable,
                    workspace_dir=workspace_dir,
                    output_dir=output_dir,
                    sbatch_output=sbatch_output,
                    sbatch_error=sbatch_error,
                    enable_workflow_monitor=enable_workflow_monitor,
                    enable_task_monitors=enable_task_monitors,
                    include_nodes=include_nodes,
                    exclude_nodes=exclude_nodes,
                )
        except Exception as e:
            failed_count += 1
            summary.append(f"  [{idx}] {yaml_file.name}: SKIPPED (dry-run failed)")
            failures.append(f"  [{idx}] {yaml_file.name}: {_failure_detail(e)}")
            fail_row: dict[str, str] = {
                "sflow_config_file": str(yaml_file),
                "job_name": job_name,
                "slurm_job_id": "FAILED",
                "sflow_output_dir": "",
                "sflow_batch_dir": bulk_dir.name,
                "status": "dry-run failed",
            }
            if resolve:
                fail_row["composed_sflow_config"] = ""
            result_rows.append(fail_row)
            continue

        # Determine node count from config if not given via CLI
        row_nodes = nodes
        if row_nodes is None:
            try:
                # Regex-fast-path + full-pipeline-fallback (matches the dry-run) for a
                # compound ``${{ }}`` node expression. Only peek the node-count columns
                # (loading the YAML) when the backend has no resolvable ``nodes`` field.
                row_nodes = _derive_nodes([yaml_file], cli_overrides=cli_set_var)
                if row_nodes is None:

                    with open(yaml_file) as fh:
                        data = safe_load(fh)
                    row_nodes = _first_node_column_int(
                        _build_var_map(data, cli_overrides=cli_set_var)
                    )
            except Exception:
                pass
        else:
            # Both numbers exist on this path too: --nodes for the allocation, the
            # config's own for the recipe. This loop does not go through the strategy's
            # plan(), so derive the config's number the same way plan() would.
            conflict = _node_count_conflict(
                cli_nodes=row_nodes,
                config_nodes=_derive_nodes([yaml_file], cli_overrides=cli_set_var),
                origin=yaml_file.name,
            )
            if conflict is not None:
                typer.echo(conflict, err=True)

        script = _batch_launch_strategy("slurm").generate(
            BatchLauncherRequest(
                files=[yaml_file],
                set_var=cli_set_var,
                artifact=cli_artifact,
                skip_artifact_check=skip_artifact_check,
                missable_tasks=missable_tasks,
                log_level=log_level,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                job_name=job_name,
                sbatch_output=sbatch_output,
                sbatch_error=sbatch_error,
                partition=partition,
                account=account,
                time=time,
                nodes=row_nodes,
                gpus_per_node=row_gpus,
                sbatch_extra_args=sbatch_extra_args,
                sflow_venv_path=sflow_venv_path,
                sflow_version=sflow_version,
                sflow_source_path=sflow_source_path,
                sflow_index_url=sflow_index_url,
                enable_workflow_monitor=enable_workflow_monitor,
                enable_task_monitors=enable_task_monitors,
                include_nodes=include_nodes,
                exclude_nodes=exclude_nodes,
            )
        )
        script_path = bulk_dir / f"{job_name}.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)

        # Generate composed/resolved YAML alongside the sbatch script
        composed_yaml_path: str = ""
        try:
            from sflow.cli.compose import _compose_files

            yaml_output = _compose_files(
                [yaml_file],
                cli_set_var or None,
                cli_artifact or None,
                log_level,
                resolve=resolve,
                missable_tasks=missable_tasks,
                quiet_missable=True,
                enable_workflow_monitor=enable_workflow_monitor,
                enable_task_monitors=enable_task_monitors,
            )
            yaml_path = bulk_dir / f"{job_name}.yaml"
            yaml_path.write_text(yaml_output)
            composed_yaml_path = str(yaml_path)
        except Exception as e:
            typer.echo(
                f"  Warning: could not generate composed config for {yaml_file.name}: {e}",
                err=True,
            )

        status = "saved"
        job_id = ""
        if submit:
            try:
                msg = _batch_launch_strategy("slurm").submit(script_path)
                status = msg
                m = _SBATCH_JOB_ID_RE.search(msg)
                if m:
                    job_id = m.group(1)
            except RuntimeError as e:
                status = f"FAILED ({e})"

        sflow_output_dir = f"{effective_output}/{job_id}-*" if job_id else ""
        summary.append(f"  [{idx}] {script_path.name}: {yaml_file.name} -> {status}")
        success_row: dict[str, str] = {
            "sflow_config_file": str(yaml_file),
            "job_name": job_name,
            "slurm_job_id": job_id
            if job_id
            else ("not submitted" if not submit else "FAILED"),
            "sflow_output_dir": sflow_output_dir
            if sflow_output_dir
            else ("not submitted" if not submit else ""),
            "sflow_batch_dir": bulk_dir.name,
            "status": status,
        }
        if resolve:
            success_row["composed_sflow_config"] = composed_yaml_path
        result_rows.append(success_row)

    generated = len(yaml_files) - failed_count
    typer.echo(
        f"\nBulk submit: {generated}/{len(yaml_files)} configs processed"
        + (f" ({failed_count} failed validation)" if failed_count else "")
    )
    for line in summary:
        typer.echo(line)
    if failures:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"ERRORS: {len(failures)} config(s) failed dry-run validation:")
        typer.echo(f"{'=' * 60}")
        for f in failures:
            typer.echo(f)
        typer.echo(f"{'=' * 60}")

    if result_rows:
        import csv

        results_csv = bulk_dir / "results.csv"
        fieldnames = [
            "sflow_config_file",
            "job_name",
            "slurm_job_id",
            "backend_job_id",
            "sflow_output_dir",
            "sflow_batch_dir",
            "status",
        ]
        if resolve:
            fieldnames.append("composed_sflow_config")
        for rr in result_rows:
            rr["backend_job_id"] = rr.get("slurm_job_id", "")
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result_rows)
        typer.echo(f"\nResults CSV: {results_csv}")

    typer.echo(f"Scripts directory: {bulk_dir}")

    if not submit and generated > 0:
        typer.echo("\n(Scripts generated but not submitted. To submit, add: --submit)")


def _run_bulk_edit(
    *,
    csv_path: Path,
    cli_files: list[Path] | None,
    cli_set_var: list[str] | None,
    cli_artifact: list[str] | None,
    log_level: str,
    workspace_dir: Path | None,
    output_dir: Path | None,
    job_name: str,
    sbatch_output: str,
    sbatch_error: str,
    partition: str,
    account: str,
    time: str | None,
    nodes: int | None,
    gpus_per_node: int | None,
    sbatch_extra_args: list[str] | None,
    sflow_venv_path: Path | None,
    sflow_version: str | None,
    sflow_source_path: Path | None = None,
    sflow_index_url: str | None = None,
    submit: bool,
    row_selectors: list[str] | None = None,
    resolve: bool = False,
    missable_tasks: list[str] | None = None,
    enable_workflow_monitor: bool = False,
    enable_task_monitors: list[str] | None = None,
    include_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    skip_artifact_check: bool = False,
) -> None:
    """Generate (and optionally submit) one sbatch job per CSV row.

    CLI ``--set`` and ``--artifact`` flags provide baseline overrides.
    CSV columns override those baselines per row (with a warning).
    """
    cli_var_map = _parse_kv_list(cli_set_var)
    cli_art_map = _parse_kv_list(cli_artifact)
    columns, rows = read_bulk_csv(csv_path)

    if nodes is None and not (_NODE_COLUMN_NAMES & set(columns)):
        raise ValueError(
            f"--nodes was not provided and the CSV does not contain a node-count column. "
            f"Either pass --nodes or add one of these columns to the CSV: "
            f"{', '.join(sorted(_NODE_COLUMN_NAMES))}"
        )

    csv_dir = csv_path.parent
    resolved_cli_files = [p.resolve() for p in (cli_files or [])]

    row_configs = build_all_row_configs(
        rows,
        csv_dir,
        resolved_cli_files,
        missable_tasks,
    )
    var_cols, art_cols = _classify_csv_columns(columns, row_configs)

    csv_var_names = var_cols
    csv_art_names = art_cols
    overlap_vars = set(cli_var_map.keys()) & csv_var_names
    overlap_arts = set(cli_art_map.keys()) & csv_art_names
    for name in sorted(overlap_vars):
        typer.echo(
            f"  Warning: variable '{name}' specified via --set and also in CSV; "
            f"CLI --set value will take precedence over CSV.",
            err=True,
        )
    for name in sorted(overlap_arts):
        typer.echo(
            f"  Warning: artifact '{name}' specified via --artifact and also in CSV; "
            f"CLI --artifact value will take precedence.",
            err=True,
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bulk_dir = Path(output_dir or Path.cwd() / "sflow_output") / f"bulk_input_{stamp}"
    bulk_dir.mkdir(parents=True, exist_ok=True)

    import re as _re

    _SBATCH_JOB_ID_RE = _re.compile(r"Submitted batch job (\d+)")

    summary: list[str] = []
    dry_run_failures: list[str] = []
    failed_count = 0
    result_rows: list[dict[str, str]] = []
    effective_output_dir = output_dir or Path.cwd() / "sflow_output"

    row_indices: set[int] | None = None
    if row_selectors:
        row_indices = set(parse_row_selector(row_selectors, n_rows=len(rows)))
    _reject_conflicting_node_counts(
        rows=rows,
        row_indices=row_indices,
        nodes=nodes,
        cli_var_map=cli_var_map,
        csv_path=csv_path,
    )
    naming_ctx = build_row_naming_ctx(
        rows, fallback_base=job_name, cli_nodes=nodes, cli_var_overrides=cli_var_map
    )

    for idx, row in enumerate(rows, start=1):
        if row_indices is not None and idx not in row_indices:
            continue
        config_files = resolve_row_files(row, csv_dir, resolved_cli_files)

        set_var_opt, artifacts_opt = merge_row_overrides(
            row, csv_var_names, csv_art_names, cli_var_map, cli_art_map
        )
        set_var = set_var_opt or []
        artifacts = artifacts_opt or []

        all_overrides: dict[str, str] = {}
        for col in columns:
            if col not in _RESERVED_CSV_COLUMNS and row.get(col):
                all_overrides[col] = row[col]
        all_overrides.update(cli_var_map)
        all_overrides.update(cli_art_map)
        overrides_desc = ", ".join(f"{k}={v}" for k, v in all_overrides.items())

        result_row = dict(row)
        for name, value in {**cli_var_map, **cli_art_map}.items():
            if name in result_row:
                result_row[name] = value
        effective_missable = row_missable(row, missable_tasks)

        # Derive gpus_per_node: config/CSV value wins over CLI
        config_gpus = _derive_gpus_per_node(config_files, cli_overrides=set_var)
        row_gpus = config_gpus if config_gpus is not None else gpus_per_node
        emit_gpus_per_node_semantics_warning(
            row_gpus,
            lambda message: typer.echo(message, err=True),
            prefix="  Warning: ",
        )
        if (
            gpus_per_node is not None
            and config_gpus is not None
            and gpus_per_node != config_gpus
        ):
            typer.echo(
                f"  Warning: --gpus-per-node={gpus_per_node} overridden by "
                f"config value ({config_gpus}) for row {idx}",
                err=True,
            )

        try:
            # sflow batch is Slurm-only; a Kubernetes-backed config has no sbatch
            # step. Reject it here so it is reported as a per-row failure (use
            # `sflow run` for k8s) instead of emitting a meaningless script.
            _reject_kubernetes_batch(config_files, set_var or None)
            with tempfile.TemporaryDirectory(prefix="sflow-batch-dry-run-") as tmp:
                dry_files, dry_vars, dry_artifacts, dry_missable = _slurm_dry_run_inputs(
                    files=config_files,
                    variable_overrides=set_var or None,
                    artifact_overrides=artifacts or None,
                    missable_tasks=effective_missable,
                    nodes=nodes,
                    gpus_per_node=None if config_gpus is not None else gpus_per_node,
                    directory=Path(tmp),
                )
                _sflow_app.run(
                    file=dry_files,
                    dry_run=True,
                    variable_overrides=dry_vars,
                    artifact_overrides=dry_artifacts,
                    missable_tasks=dry_missable,
                    workspace_dir=workspace_dir,
                    output_dir=output_dir,
                    sbatch_output=sbatch_output,
                    sbatch_error=sbatch_error,
                    enable_workflow_monitor=enable_workflow_monitor,
                    enable_task_monitors=enable_task_monitors,
                    include_nodes=include_nodes,
                    exclude_nodes=exclude_nodes,
                )
        except Exception as e:
            failed_count += 1
            summary.append(f"  [{idx}] SKIPPED: ({overrides_desc})")
            dry_run_failures.append(f"  [{idx}] {_failure_detail(e)}")
            result_row["slurm_job_id"] = "FAILED"
            result_row["sflow_output_dir"] = ""
            result_row["sflow_batch_dir"] = bulk_dir.name
            if resolve:
                result_row["composed_sflow_config"] = ""
            result_rows.append(result_row)
            continue

        # Generate merged sflow config YAML alongside the sbatch script
        try:
            from sflow.cli.compose import _compose_files

            yaml_output = _compose_files(
                config_files,
                set_var or None,
                artifacts or None,
                log_level,
                resolve=resolve,
                missable_tasks=effective_missable,
                quiet_missable=True,
                enable_workflow_monitor=enable_workflow_monitor,
                enable_task_monitors=enable_task_monitors,
            )
        except Exception as e:
            typer.echo(
                f"  Warning: could not generate merged config for row {idx}: {e}",
                err=True,
            )
            yaml_output = None

        row_nodes = nodes
        if row_nodes is None:
            # Resolve the backend node count from the config with this row's MERGED
            # overrides (CLI --set winning over CSV) -- the SAME inputs the dry-run
            # above and the gpus_per_node derivation use -- so a `--set` node override
            # reaches the generated ``#SBATCH --nodes`` directive, not just the dry-run.
            # Fall back to a node-count CSV column (also --set-overridden, via
            # ``all_overrides``) when the backend has no resolvable ``nodes`` field.
            row_nodes = _derive_nodes(config_files, cli_overrides=set_var)
            if row_nodes is None:
                row_nodes = _first_node_column_int(all_overrides)

        row_name = _derive_row_name(row, idx, naming_ctx)

        composed_config_path = ""
        if yaml_output:
            merged_yaml_path = bulk_dir / f"{row_name}.yaml"
            merged_yaml_path.write_text(yaml_output)
            composed_config_path = str(merged_yaml_path)

        script_path = bulk_dir / f"{row_name}.sh"
        script = _batch_launch_strategy("slurm").generate(
            BatchLauncherRequest(
                files=config_files,
                set_var=set_var or None,
                artifact=artifacts or None,
                skip_artifact_check=skip_artifact_check,
                missable_tasks=effective_missable,
                log_level=log_level,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                job_name=row_name,
                sbatch_output=sbatch_output,
                sbatch_error=sbatch_error,
                partition=partition,
                account=account,
                time=time,
                nodes=row_nodes,
                gpus_per_node=row_gpus,
                sbatch_extra_args=sbatch_extra_args,
                sflow_venv_path=sflow_venv_path,
                sflow_version=sflow_version,
                sflow_source_path=sflow_source_path,
                sflow_index_url=sflow_index_url,
                enable_workflow_monitor=enable_workflow_monitor,
                enable_task_monitors=enable_task_monitors,
                include_nodes=include_nodes,
                exclude_nodes=exclude_nodes,
            )
        )
        script_path.write_text(script)
        script_path.chmod(0o755)

        status = "saved"
        job_id = ""
        if submit:
            try:
                msg = _batch_launch_strategy("slurm").submit(script_path)
                status = msg
                m = _SBATCH_JOB_ID_RE.search(msg)
                if m:
                    job_id = m.group(1)
            except RuntimeError as e:
                status = f"FAILED ({e})"

        result_row["slurm_job_id"] = job_id
        result_row["sflow_output_dir"] = (
            f"{effective_output_dir}/{job_id}-*" if job_id else ""
        )
        result_row["sflow_batch_dir"] = bulk_dir.name
        if resolve:
            result_row["composed_sflow_config"] = composed_config_path
        result_rows.append(result_row)
        summary.append(f"  [{idx}] {script_path.name}: ({overrides_desc}) -> {status}")

    processed = len(summary)
    generated = processed - failed_count
    row_info = (
        f" (rows: {','.join(str(r) for r in sorted(row_indices))})"
        if row_indices
        else ""
    )
    typer.echo(
        f"\nBulk input: {generated}/{processed} jobs generated from {csv_path.name}{row_info}"
        + (f" ({failed_count} failed dry-run)" if failed_count else "")
    )
    typer.echo(f"Scripts directory: {bulk_dir}")
    for line in summary:
        typer.echo(line)
    if dry_run_failures:
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"ERRORS: {len(dry_run_failures)} row(s) failed dry-run validation:")
        typer.echo(f"{'=' * 60}")
        for f in dry_run_failures:
            typer.echo(f)
        typer.echo(f"{'=' * 60}")

    if result_rows:
        results_csv = bulk_dir / "results.csv"
        result_columns = columns + [
            "slurm_job_id",
            "backend_job_id",
            "sflow_output_dir",
            "sflow_batch_dir",
        ]
        if resolve:
            result_columns.append("composed_sflow_config")
        for rr in result_rows:
            if not rr.get("slurm_job_id"):
                rr["slurm_job_id"] = "not submitted" if not submit else ""
            rr["backend_job_id"] = rr.get("slurm_job_id", "")
            if not rr.get("sflow_output_dir"):
                rr["sflow_output_dir"] = "not submitted" if not submit else ""
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_columns)
            writer.writeheader()
            writer.writerows(result_rows)
        typer.echo(f"\nResults CSV: {results_csv}")
    typer.echo(f"Scripts directory: {bulk_dir}")

    if not submit and generated > 0:
        typer.echo("\n(Scripts generated but not submitted. To submit, add: --submit)")


@app.command(epilog=f"Documentation: {DOCS_URL}")
def batch(
    src_files: Annotated[
        Optional[List[Path]],
        typer.Argument(
            help="Workflow YAML file(s). Multiple files are merged into a single workflow.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    file: Annotated[
        Optional[List[Path]],
        typer.Option(
            "-f",
            "--file",
            help="Path to sflow YAML workflow file(s). Can be specified multiple times to merge configs.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    set_var: Annotated[
        Optional[List[str]],
        typer.Option(
            "--set",
            "-s",
            help="Override variable value or domain (format: KEY=VALUE or KEY=[1,2,3] for domain). Can be used multiple times.",
        ),
    ] = None,
    artifact: Annotated[
        Optional[List[str]],
        typer.Option(
            "--artifact",
            "-a",
            help="Override artifact URI (format: NAME=URI, can be used multiple times)",
        ),
    ] = None,
    enable_workflow_monitor: EnableWorkflowMonitorOption = False,
    enable_task_monitor: EnableTaskMonitorOption = None,
    include_nodes: IncludeNodesOption = None,
    exclude_nodes: ExcludeNodesOption = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level for sflow run (debug, info, warning, error, critical). Default: info.",
        ),
    ] = "info",
    workspace_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--workspace-dir",
            help="Workspace root directory. Default: current working directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            help="Global output root directory for sflow. Default: <workspace-dir>/sflow_output",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path.cwd() / "sflow_output",
    # sbatch options
    job_name: Annotated[
        str,
        typer.Option(
            "--job-name",
            "-J",
            help="Slurm job name",
        ),
    ] = "sflow",
    sbatch_output: Annotated[
        str,
        typer.Option(
            "--sbatch-output",
            "-O",
            help="Slurm output file pattern. Default: sflow_output/%j-sflow-submit.out",
        ),
    ] = str(Path.cwd() / "sflow_output" / "%j-sflow-submit.out"),
    sbatch_error: Annotated[
        str,
        typer.Option(
            "--sbatch-error",
            "-E",
            help="Slurm error file pattern. Default: sflow_output/%j-sflow-submit.err",
        ),
    ] = str(Path.cwd() / "sflow_output" / "%j-sflow-submit.err"),
    partition: Annotated[
        Optional[str],
        typer.Option(
            "--partition",
            "-p",
            help="Slurm partition. Auto-detected from the cluster if not specified.",
        ),
    ] = None,
    account: Annotated[
        Optional[str],
        typer.Option(
            "--account",
            "-A",
            help="Slurm account. Auto-detected from the current user's associations if not specified.",
        ),
    ] = None,
    time: Annotated[
        Optional[str],
        typer.Option(
            "--time",
            help="Slurm time limit (e.g., 01:00:00)",
        ),
    ] = None,
    nodes: Annotated[
        Optional[int],
        typer.Option(
            "--nodes",
            "-N",
            help="Number of nodes for sbatch. If omitted in single-job mode, derived from the config's backends[].nodes field. "
            "With --bulk-input, optional if the CSV contains a SLURM_NODES, NUM_SLURM_NODES, or NUM_NODES column.",
        ),
    ] = None,
    gpus_per_node: Annotated[
        Optional[int],
        typer.Option(
            "--gpus-per-node",
            "-G",
            help="Number of GPUs per node for cluster topology. If not set, derived from the sflow config's backend definition. "
            "Applied to slurm backend config for resource planning. "
            "This does NOT add a sbatch directive; use -e '--gpus-per-node=N' if your cluster requires it.",
        ),
    ] = None,
    sbatch_extra_args: Annotated[
        Optional[List[str]],
        typer.Option(
            "--sbatch-extra-args",
            "-e",
            help="Additional sbatch directives to append as '#SBATCH' lines. "
            "Supports ${{ variables.X }} or ${{ X }} expressions resolved from the sflow config "
            "(e.g., -e '--segment=${{ SLURM_NODES }}'). "
            "Variable values from --set overrides and CSV bulk-input columns are applied "
            "before resolution. Use single quotes to prevent shell expansion. "
            "Can be used multiple times.",
        ),
    ] = None,
    # runtime options
    sflow_venv_path: Annotated[
        Optional[Path],
        typer.Option(
            "--sflow-venv-path",
            "-v",
            help="Parent directory in which each Slurm job creates its OWN fresh, "
            "disposable per-job virtualenv (.sflow_venv-<job id>/) and installs sflow "
            "into it -- from the git ref resolved via --sflow-version, or editable from "
            "--sflow-source-path. The venv is built on the compute node with a resolved "
            "system python3, so it always matches the node's architecture (x86/arm), and "
            "is removed when the job exits. This is the venv parent dir, NOT an existing "
            "venv to reuse. Defaults to compute-node-local scratch resolved at run time "
            "(${TMPDIR:-/tmp}/sflow_compute_node_venv); pass a (shared-filesystem) path "
            "to override.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    sflow_version: Annotated[
        Optional[str],
        typer.Option(
            "--sflow-version",
            help="Git ref (branch or tag) to install from the GitHub repo (e.g., 'main', 'v0.1.0'), "
            "or a repository URL with an @ref suffix "
            "(e.g., 'https://git.example.com/example/sflow.git@develop'). "
            "If not specified, generated scripts default to the currently executing sflow environment's "
            "installed git ref when available, otherwise the installed package version, and only fall back "
            "to 'main' when neither can be determined. Mutually exclusive with --sflow-source-path. "
            "When --sflow-index-url is set, this is instead interpreted as a PyPI version "
            "specifier (e.g. '0.2.1' or '>=0.2,<0.3'); see --sflow-index-url.",
        ),
    ] = None,
    sflow_index_url: Annotated[
        Optional[str],
        typer.Option(
            "--sflow-index-url",
            help="Install sflow from a private PyPI index (e.g. an Artifactory registry "
            "such as https://<host>/artifactory/api/pypi/<repo>/simple) instead of "
            "from git. When set, --sflow-version is treated as a PyPI version specifier: a "
            "bare version is pinned ('0.2.1' -> 'sflow==0.2.1'), an operator spec is passed "
            "through ('>=0.2,<0.3'), and omitting it installs the latest available. The index "
            "is added via uv's --extra-index-url so dependencies still resolve from the "
            "default index. Credentials must be available on the compute node (e.g. ~/.netrc) "
            "or via a credential helper; URLs containing embedded credentials are rejected. "
            "Mutually exclusive with --sflow-source-path.",
        ),
    ] = None,
    sflow_source_path: Annotated[
        Optional[Path],
        typer.Option(
            "--sflow-source-path",
            help="Path to a local sflow source checkout. When set, each job copies this "
            "checkout (via rsync) into its own per-job dir and installs sflow editable "
            "from the copy (`uv pip install -e \".[dev]\"`, dev extras) instead of from "
            "the git repo; the per-job copy keeps concurrent jobs from racing on the "
            "setuptools-scm build artifacts written during an editable build. The path "
            "must be readable from the compute node, which must have rsync. Mutually "
            "exclusive with --sflow-version.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    skip_artifact_check: Annotated[
        bool,
        typer.Option(
            "--skip-artifact-check",
            help="Do not fail when an fs:// artifact path does not exist locally; "
            "warn and continue. Forwarded to the `sflow run` inside the submitted "
            "job, for paths that only exist on the compute nodes.",
        ),
    ] = False,
    missable_tasks: Annotated[
        Optional[List[str]],
        typer.Option(
            "--missable-tasks",
            "-M",
            help="Task names or glob patterns (e.g. 'prefill_*') that may be absent when composing "
            "modular configs from multiple files. Absent missable tasks are removed from depends_on "
            "and probes with a warning. Only valid with multiple input files or --bulk-input/--bulk-submit. Repeatable.",
        ),
    ] = None,
    # output options
    sbatch_path: Annotated[
        Optional[Path],
        typer.Option(
            "--sbatch-path",
            "-o",
            help="Write the sbatch script to this file instead of stdout",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            help="Submit the job immediately after generating the script",
        ),
    ] = False,
    bulk_input: Annotated[
        Optional[Path],
        typer.Option(
            "--bulk-input",
            "-b",
            help="CSV file for bulk job generation. "
            "Reserved columns: 'sflow_config_file' (required; to merge multiple YAML files into "
            "one workflow, list them space-separated in a single cell, e.g. 'backend.yaml workflow.yaml'), "
            "'job_name' (optional, explicit name for the generated script and Slurm job), "
            "'missable_tasks' (optional, space-separated task names/globs). "
            "All other columns are matched to variable or artifact names as overrides. "
            "When 'job_name' is absent, names are auto-derived from unique config file stems. "
            "If --nodes is not provided, the CSV must contain one of: SLURM_NODES, NUM_SLURM_NODES, or NUM_NODES.",
        ),
    ] = None,
    row: Annotated[
        Optional[List[str]],
        typer.Option(
            "--row",
            help="Only process specific CSV row(s) by 1-based index. "
            "Supports: single (--row 1), negative (--row=-1 → last row), "
            "multiple (--row 1 --row 3), "
            "comma-separated (--row 1,3,5), Python-style slices with exclusive end "
            "(--row 1:4 → rows 1,2,3; --row 1:6:2 → rows 1,3,5; --row [1:4]), "
            "and open-ended/negative slices (--row=-3: → last 3 rows; --row 3: → row 3 to end). "
            "Negative indices use --row=N syntax to avoid flag ambiguity. "
            "Requires --bulk-input.",
        ),
    ] = None,
    resolve: Annotated[
        bool,
        typer.Option(
            "-r",
            "--resolve",
            help="Resolve all resolvable variables to literal values in the generated YAML config "
            "(same as sflow compose --resolve). Works with single-job, --bulk-input, and --bulk-submit modes.",
        ),
    ] = False,
    bulk_submit: Annotated[
        Optional[List[Path]],
        typer.Option(
            "--bulk-submit",
            "-B",
            help="File path(s), folder(s), or glob pattern(s) of self-contained sflow YAML configs. "
            "Each valid YAML is processed as a standalone batch job (no merging). "
            "Folders are scanned for *.yaml/*.yml files. Glob patterns (e.g. 'examples/self_contained/slurm/*') are expanded. "
            "CLI flags (--set, --artifact, --partition, etc.) are applied to every config.",
        ),
    ] = None,
    offload_task_logs: Annotated[
        Optional[bool],
        typer.Option(
            "--offload-task-logs/--no-offload-task-logs",
            help="Have each task write its own log via the operator (srun --output, "
            "or a host-side shell redirect for local/docker) through a compute-side "
            "prefixer, instead of streaming it through the sflow driver inside the job. "
            "ON by default for local/docker/slurm in non-interactive runs; pass "
            "--no-offload-task-logs to force streaming. Overrides the backend's "
            "offload_task_logs; no-op for k8s/ssh.",
        ),
    ] = None,
):
    """
    Generate an sbatch script for running sflow in Slurm batch mode.

    This command creates a bash script with sbatch directives that wraps
    the 'sflow run' command for headless execution on a Slurm cluster.

    Three modes are supported:

    1. Single-job mode (default): generate one sbatch script from one or more
       YAML config files (merged into a single workflow). Requires --nodes.

    2. Bulk-input mode (--bulk-input): read a CSV file where each row defines
       a job with its own config files and variable/artifact overrides.

    3. Bulk-submit mode (--bulk-submit): pass file paths or folder paths of
       self-contained sflow YAML configs. Each valid YAML is processed as a
       standalone batch job (no merging). Folders are scanned for *.yaml/*.yml
       files. CLI flags (--set, --artifact, etc.) are applied to all configs.
       Warns when --set overrides a variable already defined in a config.

    ┌─────────────────────────────────────────────────────────────┐
    │  --bulk-input vs --bulk-submit                              │
    ├──────────────────────┬──────────────────────────────────────┤
    │                      │                                      │
    │  --bulk-input (-b)   │  --bulk-submit (-B)                  │
    │                      │                                      │
    │  CSV-driven          │  File/folder-driven                  │
    │                      │                                      │
    │  jobs.csv            │  ./examples/                         │
    │   ├─ row 1 ──┐      │   ├─ sglang_agg.yaml ──→ job 1      │
    │   ├─ row 2 ──┤      │   ├─ vllm_agg.yaml   ──→ job 2      │
    │   └─ row 3 ──┘      │   └─ trtllm_agg.yaml ──→ job 3      │
    │                      │                                      │
    │  Each row can:       │  Each YAML is:                       │
    │  · merge N files     │  · self-contained (no merging)       │
    │  · override vars     │  · CLI --set applied uniformly       │
    │  · override arts     │  · nodes from config or CLI          │
    │                      │                                      │
    │  Use when configs    │  Use when each YAML is a complete    │
    │  are modular and     │  standalone workflow ready to run     │
    │  need per-row        │  as-is                               │
    │  customization       │                                      │
    └──────────────────────┴──────────────────────────────────────┘

    Sbatch stdout/stderr logs are automatically copied into the sflow workflow
    output directory at the end of each generated script.

    CSV format for --bulk-input:

        sflow_config_file              SLURM_NODES  MODEL_PATH
        backend.yaml sglang/agg.yaml   2            /models/llama
        backend.yaml trtllm/agg.yaml   4            /models/llama

        The 'sflow_config_file' column is required. To merge multiple YAML
        files into one workflow, list them space-separated in a single cell.
        Other columns are matched to variable or artifact names as overrides.

    Examples:
        # Generate sbatch script to stdout
        sflow batch workflow.yaml

        # Merge multiple config files
        sflow batch backends.yaml workflow.yaml tasks.yaml

        # Generate and save to file
        sflow batch workflow.yaml --sbatch-path run_workflow.sh

        # Generate with Slurm options
        sflow batch workflow.yaml --partition gpu --time 02:00:00 --account myaccount

        # Generate with GPU allocation
        sflow batch workflow.yaml --nodes 2 --gpus-per-node 8

        # Generate and submit immediately
        sflow batch workflow.yaml --partition gpu --submit

        # With variable overrides
        sflow batch workflow.yaml --set NUM_GPUS=8 --set MODEL=llama

        # With custom virtual environment
        sflow batch workflow.yaml --sflow-venv-path /path/to/.venv

        # With extra sbatch directives (supports ${{ variables.X }} expressions)
        sflow batch workflow.yaml -e "--exclusive" -e "--segment=${{ variables.SLURM_NODES }}"

        # Bulk input: generate one job per CSV row (--nodes not required)
        sflow batch --bulk-input jobs.csv --partition gpu --account myaccount

        # Bulk submit: process all YAML files in a folder as standalone workflows
        sflow batch --bulk-submit ./examples/ --partition gpu --account myaccount --submit

        # Bulk submit: specific files
        sflow batch -B sglang_agg.yaml -B vllm_agg.yaml --partition gpu --submit

        # Bulk submit: with variable overrides applied to all configs
        sflow batch -B ./examples/ --set SLURM_NODES=2 --partition gpu --submit
    """
    configure_logging(level=log_level, console=True)
    log_runtime_info()

    # Accept comma- and/or whitespace-separated task names (and repeated flags).
    enable_task_monitor = split_list_arg(enable_task_monitor)
    include_nodes = split_list_arg(include_nodes)
    exclude_nodes = split_list_arg(exclude_nodes)

    # Per-invocation override for the per-task log offload. Setting it here makes
    # the in-process dry-run plan reflect the choice; _generate_sbatch_script also
    # exports it into the generated job so the inner `sflow run` applies it.
    if offload_task_logs is not None:
        os.environ[OFFLOAD_TASK_LOGS_ENV] = "1" if offload_task_logs else "0"

    # sflow batch only targets Slurm. For the single-job path, reject a
    # Kubernetes-backed config NOW -- before Slurm partition/account auto-detection --
    # so a k8s recipe gets the clear "use sflow run" hint instead of a confusing
    # "could not auto-detect a Slurm partition" error. The bulk paths reject k8s per
    # row/config inside their runners (where each row's files are known).
    if bulk_input is None and bulk_submit is None:
        single_job_files = list(src_files or []) + list(file or [])
        if not single_job_files:
            single_job_files = [Path("sflow.yaml").resolve()]
        try:
            _reject_kubernetes_batch(single_job_files, set_var)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    partition, account = _resolve_slurm_defaults(partition, account)

    if row and bulk_input is None:
        typer.echo("Error: --row requires --bulk-input.", err=True)
        raise typer.Exit(code=1)

    if sflow_version is not None and sflow_source_path is not None:
        typer.echo(
            "Error: --sflow-version and --sflow-source-path are mutually exclusive; "
            "pass at most one.",
            err=True,
        )
        raise typer.Exit(code=1)

    if sflow_index_url is not None and sflow_source_path is not None:
        typer.echo(
            "Error: --sflow-index-url and --sflow-source-path are mutually exclusive; "
            "pass at most one.",
            err=True,
        )
        raise typer.Exit(code=1)

    index_url_error = sflow_index_url_error(sflow_index_url)
    if index_url_error:
        typer.echo(f"Error: {index_url_error}", err=True)
        raise typer.Exit(code=1)

    # Sanity-check --sflow-version for whichever install route is active: a git
    # ref/URL by default, or a PyPI version/specifier when --sflow-index-url is
    # set. This is the single source of truth -- script generation trusts it.
    version_error = _sflow_version_error(
        sflow_version, registry=sflow_index_url is not None
    )
    if version_error is not None:
        typer.echo(f"Error: {version_error}", err=True)
        raise typer.Exit(code=1)

    # --- Bulk-edit mode ---
    if bulk_input is not None:
        try:
            _run_bulk_edit(
                csv_path=bulk_input,
                cli_files=list(src_files or []) + list(file or []),
                cli_set_var=set_var,
                cli_artifact=artifact,
                skip_artifact_check=skip_artifact_check,
                log_level=log_level,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                job_name=job_name,
                sbatch_output=sbatch_output,
                sbatch_error=sbatch_error,
                partition=partition,
                account=account,
                time=time,
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                sbatch_extra_args=sbatch_extra_args,
                sflow_venv_path=sflow_venv_path,
                sflow_version=sflow_version,
                sflow_source_path=sflow_source_path,
                sflow_index_url=sflow_index_url,
                submit=submit,
                row_selectors=row,
                resolve=resolve,
                missable_tasks=missable_tasks,
                enable_workflow_monitor=enable_workflow_monitor,
                enable_task_monitors=enable_task_monitor,
                include_nodes=include_nodes,
                exclude_nodes=exclude_nodes,
            )
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        return

    # --- Bulk-submit mode ---
    if bulk_submit is not None:
        all_paths = list(bulk_submit)
        if src_files:
            all_paths.extend(src_files)
        if file:
            all_paths.extend(file)

        csv_in_bulk_submit = [p for p in all_paths if p.is_file() and p.suffix.lower() == ".csv"]
        if csv_in_bulk_submit:
            names = ", ".join(str(f) for f in csv_in_bulk_submit)
            typer.echo(
                f"Error: CSV file(s) detected in --bulk-submit input: {names}\n"
                f"  --bulk-submit expects sflow YAML files or directories, not CSV.\n"
                f"  Did you mean to use --bulk-input (-b)?\n"
                f"  Example: sflow batch --bulk-input {csv_in_bulk_submit[0]}",
                err=True,
            )
            raise typer.Exit(code=1)

        yaml_files = _scan_sflow_yamls(all_paths)
        if not yaml_files:
            typer.echo(
                "Error: no valid sflow YAML files found in the provided paths.",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"Found {len(yaml_files)} sflow YAML config(s):")
        for yf in yaml_files:
            typer.echo(f"  - {yf.name}")
        try:
            _run_bulk_submit(
                yaml_files=yaml_files,
                cli_set_var=set_var,
                cli_artifact=artifact,
                skip_artifact_check=skip_artifact_check,
                log_level=log_level,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                sbatch_output=sbatch_output,
                sbatch_error=sbatch_error,
                partition=partition,
                account=account,
                time=time,
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                sbatch_extra_args=sbatch_extra_args,
                sflow_venv_path=sflow_venv_path,
                sflow_version=sflow_version,
                sflow_source_path=sflow_source_path,
                sflow_index_url=sflow_index_url,
                submit=submit,
                missable_tasks=missable_tasks,
                resolve=resolve,
                enable_workflow_monitor=enable_workflow_monitor,
                enable_task_monitors=enable_task_monitor,
                include_nodes=include_nodes,
                exclude_nodes=exclude_nodes,
            )
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        return

    # --- Single-job mode ---
    files = list(src_files or []) + list(file or [])
    if not files:
        files = [Path("sflow.yaml").resolve()]

    csv_files = [f for f in files if f.suffix.lower() == ".csv"]
    if csv_files:
        names = ", ".join(str(f) for f in csv_files)
        typer.echo(
            f"Error: CSV file(s) detected in input: {names}\n"
            f"  CSV files cannot be used as workflow YAML inputs directly.\n"
            f"  Did you mean to use --bulk-input (-b)?\n"
            f"  Example: sflow batch --bulk-input {csv_files[0]}",
            err=True,
        )
        raise typer.Exit(code=1)

    if missable_tasks and len(files) < 2:
        typer.echo(
            "Error: --missable-tasks is only valid with multiple input files (modular configs).",
            err=True,
        )
        raise typer.Exit(code=1)

    cli_nodes = nodes
    cli_gpus_per_node = gpus_per_node

    # Backend-specific planning -- single-allocation vs multi-backend driver, plus
    # node/gpu derivation -- is owned by the launch strategy so this entry point
    # stays backend-type agnostic and does not reason about Slurm specifics itself.
    strategy = _batch_launch_strategy("slurm")
    batch_plan = strategy.plan(
        BatchPlanRequest(
            files=files,
            set_var=set_var,
            cli_nodes=cli_nodes,
            cli_gpus_per_node=cli_gpus_per_node,
        )
    )
    for message in batch_plan.messages:
        typer.echo(message, err=True)
    if batch_plan.error is not None:
        typer.echo(f"Error: {batch_plan.error}", err=True)
        raise typer.Exit(code=1)
    nodes = batch_plan.nodes
    gpus_per_node = batch_plan.gpus_per_node

    # Run dry-run validation before generating sbatch script
    typer.echo("Running dry-run validation before generating sbatch script...")
    try:
        with tempfile.TemporaryDirectory(prefix="sflow-batch-dry-run-") as tmp:
            # Honor the strategy's plan: a multi-backend job keeps each backend's
            # own nodes/gpus, so the single CLI -N/-G must not override them here.
            dry_files, dry_vars, dry_artifacts, dry_missable = _slurm_dry_run_inputs(
                files=files,
                variable_overrides=set_var,
                artifact_overrides=artifact,
                missable_tasks=missable_tasks,
                nodes=batch_plan.dry_run_nodes,
                gpus_per_node=batch_plan.dry_run_gpus_per_node,
                directory=Path(tmp),
            )
            _sflow_app.run(
                file=dry_files,
                dry_run=True,
                variable_overrides=dry_vars,
                artifact_overrides=dry_artifacts,
                missable_tasks=dry_missable,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                sbatch_output=sbatch_output,
                sbatch_error=sbatch_error,
                enable_workflow_monitor=enable_workflow_monitor,
                enable_task_monitors=enable_task_monitor,
                include_nodes=include_nodes,
                exclude_nodes=exclude_nodes,
            )
        typer.echo("✓ Dry-run validation passed\n")
    except ValueError as e:
        msg = enrich_error_with_location(str(e), files)
        typer.echo(f"✗ Configuration error: {msg}", err=True)
        typer.echo("Aborting sbatch generation due to configuration errors.", err=True)
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        typer.echo(f"✗ File not found: {e}", err=True)
        typer.echo("Aborting sbatch generation due to missing files.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"✗ Dry-run validation failed: {e}", err=True)
        typer.echo("Aborting sbatch generation due to validation errors.", err=True)
        raise typer.Exit(code=1)

    script_content = strategy.generate(
        BatchLauncherRequest(
            files=files,
            set_var=set_var,
            artifact=artifact,
            skip_artifact_check=skip_artifact_check,
            missable_tasks=missable_tasks,
            log_level=log_level,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            job_name=job_name,
            sbatch_output=sbatch_output,
            sbatch_error=sbatch_error,
            partition=partition,
            account=account,
            time=time,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            sbatch_extra_args=sbatch_extra_args,
            sflow_venv_path=sflow_venv_path,
            sflow_version=sflow_version,
            sflow_source_path=sflow_source_path,
            sflow_index_url=sflow_index_url,
            enable_workflow_monitor=enable_workflow_monitor,
            enable_task_monitors=enable_task_monitor,
            include_nodes=include_nodes,
            exclude_nodes=exclude_nodes,
        )
    )

    # Generate composed/resolved YAML alongside the sbatch script
    if sbatch_path:
        try:
            from sflow.cli.compose import _compose_files

            yaml_output = _compose_files(
                files,
                set_var or None,
                artifact or None,
                log_level,
                resolve=resolve,
                missable_tasks=missable_tasks,
                quiet_missable=True,
                enable_workflow_monitor=enable_workflow_monitor,
                enable_task_monitors=enable_task_monitor,
            )
            yaml_path = sbatch_path.with_suffix(".yaml")
            yaml_path.write_text(yaml_output)
            typer.echo(f"✓ Composed config written to: {yaml_path}")
        except Exception as e:
            typer.echo(f"  Warning: could not generate composed config: {e}", err=True)

    if sbatch_path:
        sbatch_path.write_text(script_content)
        sbatch_path.chmod(0o755)
        typer.echo(script_content)
        typer.echo(f"✓ Sbatch script written to: {sbatch_path}")

        if submit:
            try:
                msg = strategy.submit(sbatch_path)
                typer.echo(f"✓ Job submitted: {msg}")
            except RuntimeError as e:
                typer.echo(f"✗ {e}", err=True)
                raise typer.Exit(code=1)
    else:
        typer.echo(script_content)
        typer.echo(
            "\n# (stdout only — to save as a file, add: -o <path>.sh)",
            err=True,
        )

        if submit:
            typer.echo(
                "⚠ Cannot submit without --sbatch-path / -o. "
                "Please specify a file to save the script first.",
                err=True,
            )
            raise typer.Exit(code=1)
