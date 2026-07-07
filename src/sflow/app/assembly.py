# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Application composition root.

This module is the single place where we wire together:
- validated config DTOs (sflow.config.schema)
- core runtime objects (sflow.core.*)
- concrete plugin implementations (sflow.plugins.*)
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any, Literal

from sflow.resolution import (
    ExpressionResolver,
    artifacts_ctx as _artifacts_ctx_impl,
    maybe_int as _maybe_int,
    resolve_and_update_variables as _resolve_and_update_variables_impl,
    resolve_artifacts as _resolve_artifacts_impl,
    resolve_deferred_global_variables as _resolve_deferred_global_variables_impl,
    resolve_global_variables as _resolve_global_variables_impl,
    resolve_workflow_variables as _resolve_workflow_variables_impl,
)
from sflow.app.backend_lifecycle import (
    allocate_backends as _allocate_backends_impl,
    preflight_validate_backends as _preflight_validate_backends_impl,
    preflight_validate_container_images as _preflight_validate_container_images_impl,
    release_backends as _release_backends_impl,
    resolve_backends as _resolve_backends_impl,
    seed_placeholder_backend_allocations as _seed_placeholder_backend_allocations_impl,
)
from sflow.app.resource_planner import plan_resource_placements
from sflow.app.run_support import backends_execute_offhost
from sflow.app.task_context import (
    build_task_info,
    build_task_expression_hint,
    build_tasks_ctx,
    compute_task_service,
    extract_task_expressions,
)
from sflow.config.schema import SflowConfig
from sflow.core.state import SflowState
from sflow.core.task import (
    OutputSpec,
    ResolvedUpload,
    ResultConfigRuntime,
    ResultSpec,
    RetryPolicy,
    Task,
    TaskPort,
    TaskStatus,
)
from sflow.core.task_graph import TaskGraph
from sflow.core.variable import build_variables_ctx
from sflow.core.workflow import Workflow
from sflow.logging import get_logger

resolver = ExpressionResolver()

_logger = get_logger(__name__)


def _build_result_config_runtime(r_conf: Any) -> ResultConfigRuntime:
    """
    Convert the validated ``ResultConfig`` schema object into the runtime
    ``ResultConfigRuntime`` used by the orchestrator and result parser.

    Notes:
    - ``r_conf.patterns`` and ``r_conf.file`` are mutually exclusive (enforced by schema).
    - Per-pattern ``source`` falls back to the parent ``source`` when not set.
    """
    specs: list[ResultSpec] = []
    parent_source = getattr(r_conf, "source", "log") or "log"
    for p in list(getattr(r_conf, "patterns", None) or []):
        per_source = getattr(p, "source", None) or parent_source
        specs.append(
            ResultSpec(
                name=p.name,
                regex=p.regex,
                engine="regex",
                source=per_source,
                type=getattr(p, "type", "auto"),
                unit=getattr(p, "unit", None),
                aggregate=getattr(p, "aggregate", "last"),
                required=bool(getattr(p, "required", False)),
                group=getattr(p, "group", None),
            )
        )
    return ResultConfigRuntime(
        specs=specs,
        file=getattr(r_conf, "file", None),
        source=parent_source,
    )


# Backward-compatible private aliases for existing tests/callers during extraction.
_build_task_info = build_task_info
_build_tasks_ctx = build_tasks_ctx


def preflight_validate_backends(state: SflowState) -> None:
    return _preflight_validate_backends_impl(state)


def preflight_validate_container_images(config: SflowConfig, state: SflowState) -> None:
    return _preflight_validate_container_images_impl(
        config,
        state,
        resolver=resolver,
    )


def preflight_validate_task_graph(
    config: SflowConfig,
    state: SflowState,
    *,
    workspace_dir: Any | None = None,
    output_dir: Any | None = None,
) -> None:
    """
    Validate task planning against placeholder backend allocations before real allocation.

    This reuses the normal graph-building and GPU-packing logic with deterministic placeholder
    nodes so capacity/configuration errors surface before real backend resources are allocated.
    """
    planning_state = SflowState(
        workflow=Workflow(name=config.workflow.name, task_graph=TaskGraph()),
        variables=dict(state.variables),
        artifacts=dict(state.artifacts),
        backends=dict(state.backends),
        default_backend=state.default_backend,
    )
    original_allocations = {
        name: backend.allocation for name, backend in planning_state.backends.items()
    }
    try:
        planning_state = _seed_placeholder_backend_allocations(planning_state)
        # Backend-derived globals are available before artifacts; artifact-derived
        # globals stay deferred until artifact resolution has populated artifacts.*.
        planning_state = resolve_deferred_global_variables(
            config,
            planning_state,
            available_contexts=frozenset({"backends"}),
            defer_contexts=frozenset({"artifacts"}),
        )
        planning_state = resolve_artifacts(
            config,
            planning_state,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            materialize=False,
        )
        planning_state = resolve_deferred_global_variables(config, planning_state)
        planning_state = resolve_workflow_variables(
            config,
            planning_state,
            workspace_dir=workspace_dir,
        )
        build_task_graph(config, planning_state, workspace_dir=workspace_dir)
    finally:
        for name, backend in planning_state.backends.items():
            backend.allocation = original_allocations.get(name)


def _artifacts_ctx(
    state: SflowState,
) -> dict[str, Any]:
    return _artifacts_ctx_impl(state)


def resolve_artifacts(
    config: SflowConfig,
    state: SflowState,
    *,
    workspace_dir: Any | None = None,
    output_dir: Any | None = None,
    materialize: bool = False,
    remote_filesystem: bool = False,
) -> SflowState:
    return _resolve_artifacts_impl(
        config,
        state,
        resolver=resolver,
        workspace_dir=workspace_dir,
        output_dir=output_dir,
        materialize=materialize,
        remote_filesystem=remote_filesystem,
    )


def _seed_placeholder_backend_allocations(state: SflowState) -> SflowState:
    return _seed_placeholder_backend_allocations_impl(state)


def _resolve_and_update_variables(
    *,
    state: SflowState,
    variable_confs: list[Any],
    collision: Literal["overwrite", "error"] = "overwrite",
    extra_ctx: dict[str, Any] | None = None,
) -> SflowState:
    return _resolve_and_update_variables_impl(
        state=state,
        variable_confs=variable_confs,
        resolver=resolver,
        collision=collision,
        extra_ctx=extra_ctx,
    )


def resolve_global_variables(config: SflowConfig, state: SflowState) -> SflowState:
    return _resolve_global_variables_impl(config, state, resolver=resolver)


def resolve_deferred_global_variables(
    config: SflowConfig,
    state: SflowState,
    *,
    available_contexts: frozenset[str] | None = None,
    defer_contexts: frozenset[str] | None = None,
) -> SflowState:
    return _resolve_deferred_global_variables_impl(
        config,
        state,
        resolver=resolver,
        available_contexts=available_contexts,
        defer_contexts=defer_contexts,
    )


def resolve_backends(
    config: SflowConfig, state: SflowState, *, kubectl_config: Any | None = None
) -> SflowState:
    return _resolve_backends_impl(
        config, state, resolver=resolver, kubectl_config=kubectl_config
    )


async def allocate_backends(state: SflowState) -> SflowState:
    return await _allocate_backends_impl(state)


async def release_backends(state: SflowState) -> SflowState:
    return await _release_backends_impl(state)


def resolve_workflow_variables(
    config: SflowConfig, state: SflowState, *, workspace_dir: Any | None = None
) -> SflowState:
    return _resolve_workflow_variables_impl(
        config,
        state,
        resolver=resolver,
        workspace_dir=workspace_dir,
    )


def resolve_storage_targets(config: SflowConfig, state: SflowState) -> SflowState:
    """
    Resolve `${{ ... }}` expressions inside `config.storage` entries and instantiate
    concrete `StorageTarget` plugins into `state.storage_targets`.

    Storage targets may reference resolved variables, backends, and artifacts in
    fields like `prefix`, `bucket`, `region`, etc.
    """
    if not config.storage:
        return state

    from sflow.core.storage_registry import (
        ensure_builtin_storage_registered,
        get_storage_class,
        storage_config_type_adapter,
    )

    ensure_builtin_storage_registered()
    adapter = storage_config_type_adapter()

    backends_ctx: dict[str, Any] = {
        name: b.to_dict() for name, b in (state.backends or {}).items()
    }
    variables_ctx = build_variables_ctx(state.variables)
    artifacts_ctx = _artifacts_ctx(state)
    ctx: dict[str, Any] = {
        "variables": variables_ctx,
        "backends": backends_ctx,
        "artifacts": artifacts_ctx,
        **variables_ctx,
    }

    for s_conf in config.storage:
        # Resolve expressions in the config's dict form, then re-validate so we
        # end up with a typed, expression-free config to hand to the plugin.
        as_dict = s_conf.model_dump(by_alias=False)
        resolved_dict = resolver.resolve(as_dict, ctx)
        resolved_conf = adapter.validate_python(resolved_dict)

        target_cls = get_storage_class(resolved_conf.type)
        target = target_cls(resolved_conf)  # type: ignore[call-arg]
        state.add_storage_target(target)

    return state


def _plan_merge_groups(
    task_graph: TaskGraph, resource_placements: "Mapping[str, Any]"
) -> None:
    """Bundle single-node GPU tasks co-located on one node into one merged pod.

    Runs at assembly time after resource placement. For each backend that opts in
    via ``merge_colocated_gpu_pods``, tasks with >0 GPUs assigned to exactly one
    physical node are grouped by ``(backend, node)``. Groups with >=2 members
    become one merged pod owned by a deterministic leader (first member by name):

    * every member sees ALL union GPUs (``CUDA_VISIBLE_DEVICES``), with its own
      packed slice listed FIRST so the workload uses it as ``cuda:0`` -- exposing
      the peers' GPUs is what lets cross-member ``cuda_ipc``/NVLink P2P work (a
      single-GPU-per-member visibility hides the peers and forces UCX/NIXL to TCP);
    * the leader is made to depend on the union of the members' external deps so
      the whole group starts together (followers are not launched on their own);
    * the leader's operator is handed the ordered member Task objects + union GPU
      count via ``apply_merge_group`` (duck-typed; only the k8s operator uses it).

    Merged members may not depend on one another (they run concurrently in one
    pod), so an intra-group completion dependency raises ``ValueError``. Multi-node
    GPU tasks and CPU-only tasks are never merged.
    """
    groups: dict[tuple[str, str], list[str]] = {}
    gpu_counts: dict[str, int] = {}
    for name, placement in resource_placements.items():
        backend = getattr(placement, "backend", None)
        if backend is None or not getattr(
            backend, "merge_colocated_gpu_pods", False
        ):
            continue
        try:
            gpu_count = int(getattr(placement, "gpu_count", 0) or 0)
        except (TypeError, ValueError):
            gpu_count = 0
        assigned = list(getattr(placement, "assigned_nodes", None) or [])
        # Only single-node GPU tasks merge; multi-node tasks and CPU-only infra
        # (etcd/nats/frontend) keep their own pods.
        if gpu_count <= 0 or len(assigned) != 1:
            continue
        groups.setdefault((str(backend.name), assigned[0]), []).append(name)
        gpu_counts[name] = gpu_count

    for (backend_name, node), members in groups.items():
        if len(members) < 2:
            continue
        members = sorted(members)
        member_set = set(members)
        # Concurrent-only: a member depending on another member (completion-
        # before-start) can't be honored inside one concurrent pod.
        for m in members:
            for dep in task_graph.dag.get_dependencies(m):
                if dep in member_set:
                    raise ValueError(
                        f"Cannot merge co-located tasks {members} on node "
                        f"'{node}': '{m}' depends on '{dep}' in the same merge "
                        "group, but merged tasks run concurrently in one pod. "
                        "Remove the intra-group dependency or disable "
                        "merge_colocated_gpu_pods on the backend."
                    )
        leader_name = members[0]
        group_id = f"{backend_name}:{node}"
        # Every member's container sees ALL union GPUs, but with its OWN packed
        # slice FIRST so the workload uses that as cuda:0 / its local rank. Exposing
        # the peers' GPUs (not just its own) is what lets cross-member cuda_ipc /
        # NVLink P2P work -- a single-GPU-per-member CUDA_VISIBLE_DEVICES hides the
        # peers and forces UCX/NIXL onto TCP, defeating the point of merging.
        union_gpus = sum(gpu_counts[m] for m in members)
        start = 0
        member_tasks: list[Task] = []
        for m in members:
            t = task_graph.get_task(m)
            count = gpu_counts[m]
            own = list(range(start, start + count))
            others = [i for i in range(union_gpus) if i not in own]
            t.merge_cuda_visible_devices = ",".join(str(i) for i in own + others)
            t.merge_group_id = group_id
            start += count
            member_tasks.append(t)
        leader = task_graph.get_task(leader_name)
        leader.merge_members = list(members)
        for m in members:
            if m != leader_name:
                task_graph.get_task(m).merge_leader = leader_name
        # The leader waits for the union of every member's external deps so the
        # group starts together (followers never launch on their own).
        for m in members:
            for dep in task_graph.dag.get_dependencies(m):
                if dep not in member_set and dep != leader_name:
                    task_graph.dag.add_edge(dep, leader_name)
        # Hand the leader's operator the ordered member tasks + union GPU count so
        # it renders the single merged pod. Duck-typed: only the k8s container
        # operator implements apply_merge_group; other operators simply lack it.
        apply_merge_group = getattr(leader.operator, "apply_merge_group", None)
        if callable(apply_merge_group):
            apply_merge_group(members=member_tasks, union_gpus=union_gpus)
        # Transparent log: merging changes the pod topology (sflow-owned), so make
        # it visible which co-located GPU tasks now share one pod/node.
        _logger.info(
            "Kubernetes backend '%s': merged %d co-located GPU tasks %s on node "
            "'%s' into one pod (%d GPUs; shared NVLink/cuda_ipc, one IMEX channel "
            "claim per node).",
            backend_name,
            len(members),
            members,
            node,
            union_gpus,
        )


def _warn_channel_contention(resource_placements: "Mapping[str, Any]") -> None:
    """Hard-warn when a ComputeDomain channel claim contends with pod topology.

    The NVIDIA DRA driver publishes ONE IMEX channel per node per ComputeDomain, so
    a channel claim needs at most one channel-claiming (GPU) pod per node. Merge
    (default ``auto``) guarantees that. But when a backend has a channel configured
    AND ``merge_colocated_gpu_pods`` is off, two+ GPU pods can land on one node and
    all but one fail scheduling ("cannot allocate all claims"). Warn once per
    backend at plan time (recommend ``merge: auto`` or one GPU pod per node); the
    run still proceeds. Duck-typed on the backend (only the k8s backend exposes a
    ``compute_domain_channel``); other backends simply never trigger it.
    """
    per_node: dict[tuple[str, str], int] = {}
    backends_by_name: dict[str, Any] = {}
    for _name, placement in resource_placements.items():
        backend = getattr(placement, "backend", None)
        if backend is None:
            continue
        try:
            gpu_count = int(getattr(placement, "gpu_count", 0) or 0)
        except (TypeError, ValueError):
            gpu_count = 0
        if gpu_count <= 0:
            continue  # only GPU pods claim the channel
        backends_by_name[str(backend.name)] = backend
        for node in getattr(placement, "assigned_nodes", None) or []:
            per_node[(str(backend.name), str(node))] = (
                per_node.get((str(backend.name), str(node)), 0) + 1
            )

    warned: set[str] = set()
    for (backend_name, node), count in sorted(per_node.items()):
        if count < 2 or backend_name in warned:
            continue
        backend = backends_by_name[backend_name]
        channel = getattr(backend, "compute_domain_channel", None)
        merge = getattr(backend, "merge_colocated_gpu_pods", False)
        if channel and not merge:
            warned.add(backend_name)
            _logger.warning(
                "Kubernetes backend '%s': a ComputeDomain channel ('%s') is claimed "
                "by every GPU pod, but merge_colocated_gpu_pods is off and %d GPU "
                "tasks are placed on node '%s'. The NVIDIA driver publishes ONE IMEX "
                "channel per node, so all but one channel-claiming pod on that node "
                "will fail scheduling ('cannot allocate all claims'). Set "
                "merge_colocated_gpu_pods: auto (recommended) or place one GPU pod "
                "per node.",
                backend_name,
                channel,
                count,
                node,
            )


def _warn_interconnect_hints(resource_placements: "Mapping[str, Any]") -> None:
    """App-agnostic interconnect hints from cross-node GPU placement + scope + IB.

    The interconnect priority is NVLink -> IB/RDMA -> TCP. When a backend's GPU pods
    land on >=2 distinct nodes (cross-node GPU communication is possible) and the
    fast tier is not reachable, hint the framework/admin-owned piece sflow does not
    own (it never pins the transport):

    * node-scoped NVLink + IB down -> no fast cross-node path; co-locate
      prefill+decode per node (intra-node NVLink) or enable IB.
    * rack-scoped NVLink + no IMEX channel -> provide
      ``dra.use_compute_domain_channel`` (a name or ``auto``) for MNNVL; otherwise
      cross-node KV falls back to IB (if up) or slow TCP.

    Duck-typed on the backend (only the k8s backend exposes ``nvlink_domain_scope``
    / ``rdma_enabled`` / ``compute_domain_channel``); warn-only, once per backend.
    Scope ``None`` (undetected, e.g. dry-run) or ``off`` -> no hint (stay quiet when
    unsure). The vLLM knob is named only as an example.
    """
    gpu_nodes: dict[str, set[str]] = {}
    backends_by_name: dict[str, Any] = {}
    for _name, placement in resource_placements.items():
        backend = getattr(placement, "backend", None)
        if backend is None:
            continue
        try:
            gpu_count = int(getattr(placement, "gpu_count", 0) or 0)
        except (TypeError, ValueError):
            gpu_count = 0
        if gpu_count <= 0:
            continue
        backends_by_name[str(backend.name)] = backend
        nodes = gpu_nodes.setdefault(str(backend.name), set())
        for node in getattr(placement, "assigned_nodes", None) or []:
            nodes.add(str(node))

    for backend_name, nodes in sorted(gpu_nodes.items()):
        if len(nodes) < 2:
            continue  # single-node GPU placement: intra-node NVLink handles it
        backend = backends_by_name[backend_name]
        scope = getattr(backend, "nvlink_domain_scope", None)
        rdma_enabled = bool(getattr(backend, "rdma_enabled", False))
        channel = getattr(backend, "compute_domain_channel", None)
        if scope == "node" and not rdma_enabled:
            _logger.warning(
                "Kubernetes backend '%s': GPU tasks span %d nodes but the NVLink "
                "domain is node-scoped and IB/RDMA is down -- there is no fast "
                "cross-node interconnect, so cross-node KV transfer will use slow "
                "TCP. Co-locate prefill+decode per node (intra-node NVLink) or "
                "enable IB/RDMA.",
                backend_name,
                len(nodes),
            )
        elif scope == "rack" and not channel:
            _logger.warning(
                "Kubernetes backend '%s': GPU tasks span %d nodes on a rack-scoped "
                "(MNNVL) cluster but no IMEX ComputeDomain channel is configured. "
                "Set dra.use_compute_domain_channel (a channel name or 'auto') so "
                "cross-node KV rides NVLink/MNNVL; otherwise it falls back to IB (if "
                "up) or slow TCP. Cross-node MNNVL also needs the framework's "
                "fabric/VMM KV memory (e.g. vLLM --enable-sleep-mode) + "
                "UCX_CUDA_IPC_ENABLE_MNNVL=y (recipe-owned).",
                backend_name,
                len(nodes),
            )


def build_task_graph(
    config: SflowConfig, state: SflowState, *, workspace_dir: Any | None = None
) -> TaskGraph:
    """
    Build a TaskGraph from workflow task configs using the current (resolved) state.

    Requirements / assumptions:
    - `state.backends` must be populated (via `resolve_backends`).
    - Tasks are launched via Operators (operator-only execution model).
    - Backend/allocation-specific launch context is applied through
      `Operator.apply_backend_context`.
    """

    # Import here to avoid plugin imports in core modules.
    from sflow.core.operator_registry import (
        ensure_builtin_operators_registered,
        get_operator_class,
        operator_config_type_adapter,
    )
    from sflow.core.probe import ProbeType
    from sflow.plugins.probes import (
        HttpGetProbe,
        HttpPostProbe,
        LogWatchProbe,
        TcpPortProbe,
    )

    if not state.backends:
        raise ValueError(
            "No backends are available in state; call resolve_backends first"
        )

    operator_confs = {o.name: o for o in (config.operators or [])}

    # Always enable operator mode: runtime abstraction is removed; operators are the only launch mechanism.
    ensure_builtin_operators_registered()
    operator_adapter = operator_config_type_adapter()

    # Context for resolving expressions (scripts/resources/etc.)
    variables_ctx = build_variables_ctx(state.variables)
    if (not state.artifacts) and (config.artifacts):
        state = resolve_artifacts(
            config, state, workspace_dir=workspace_dir, materialize=False
        )
    artifacts_ctx = _artifacts_ctx(state)
    backends_ctx: dict[str, Any] = {
        name: b.to_dict() for name, b in (state.backends or {}).items()
    }
    ctx: dict[str, Any] = {
        "variables": variables_ctx,
        "artifacts": artifacts_ctx,
        "backends": backends_ctx,
        "workflow": {"name": config.workflow.name},
        **variables_ctx,
    }

    def _resolve_value(v: Any) -> Any:
        """
        Resolve `${{ ... }}` expressions inside an arbitrary python value using the current ctx.

        We use this for operator configs since operator models may contain expression strings
        (e.g., container_image="${{ variables.IMG }}", extra_args=["--foo=${{ BAR }}"]).
        """
        if resolver.has_expression(v):
            return resolver.resolve(v, ctx)
        if isinstance(v, list):
            return [_resolve_value(x) for x in v]
        if isinstance(v, dict):
            return {k: _resolve_value(val) for k, val in v.items()}
        return v

    def _resolve_replica_count(task_name: str, count_raw: Any) -> int:
        if count_raw is None:
            raise ValueError(f"Task '{task_name}' replicas.count is None")
        resolved = (
            resolver.resolve(count_raw, ctx)
            if resolver.has_expression(count_raw)
            else count_raw
        )
        resolved = _maybe_int(resolved)
        if isinstance(resolved, bool):
            raise ValueError(
                f"Task '{task_name}' replicas.count must resolve to int, got boolean {resolved!r}"
            )
        if isinstance(resolved, int):
            if resolved <= 0:
                raise ValueError(
                    f"Task '{task_name}' replicas.count must be > 0, got {resolved}"
                )
            return resolved
        if isinstance(resolved, str):
            try:
                v = int(resolved)
            except ValueError as e:
                raise ValueError(
                    f"Task '{task_name}' replicas.count must resolve to int, got {resolved!r}"
                ) from e
            if v <= 0:
                raise ValueError(
                    f"Task '{task_name}' replicas.count must be > 0, got {v}"
                )
            return v
        raise ValueError(
            f"Task '{task_name}' replicas.count must resolve to int, got {type(resolved).__name__}"
        )

    def _resolve_replica_policy(task_name: str, policy_raw: Any) -> str:
        """
        Resolve replicas.policy which may be a concrete ReplicaPolicy/str or an expression string.
        """
        resolved = (
            resolver.resolve(policy_raw, ctx)
            if resolver.has_expression(policy_raw)
            else policy_raw
        )
        # Normalize enum-ish values to string
        if hasattr(resolved, "value"):
            resolved = getattr(resolved, "value")
        policy = str(resolved).strip().lower()
        if policy not in {"parallel", "sequential"}:
            raise ValueError(
                f"Task '{task_name}' replicas.policy must be 'parallel' or 'sequential', got {policy!r}"
            )
        return policy

    def _replica_sweep_instances(
        task_name: str, var_names: list[str]
    ) -> list[dict[str, Any]]:
        """
        Expand a replica sweep into per-replica variable assignments.

        Today we implement a simple cartesian product across domains:
        - variables: ["A", "B"] with domains [1,2] and ["x","y"] -> 4 replicas
        """
        if not var_names:
            return [{}]

        domains: list[list[Any]] = []
        for vn in var_names:
            v = (state.variables or {}).get(vn)
            if v is None:
                raise ValueError(
                    f"Task '{task_name}' replicas.variables references unknown variable '{vn}'"
                )
            if not v.domain:
                raise ValueError(
                    f"Task '{task_name}' replicas.variables requires variable '{vn}' to define a non-empty domain"
                )
            domains.append(list(v.domain))

        instances: list[dict[str, Any]] = []
        for combo in itertools.product(*domains):
            instances.append(
                {vn: val for vn, val in zip(var_names, combo, strict=True)}
            )
        return instances

    def _resolve_int(task_name: str, *, field: str, value: Any) -> int:
        resolved = (
            resolver.resolve(value, ctx) if resolver.has_expression(value) else value
        )
        resolved = _maybe_int(resolved)
        if isinstance(resolved, bool):
            raise ValueError(
                f"Task '{task_name}' {field} must resolve to int, got boolean {resolved!r}"
            )
        if isinstance(resolved, int):
            return resolved
        if isinstance(resolved, str):
            try:
                return int(resolved)
            except ValueError as e:
                raise ValueError(
                    f"Task '{task_name}' {field} must resolve to int, got {resolved!r}"
                ) from e
        raise ValueError(
            f"Task '{task_name}' {field} must resolve to int, got {type(resolved).__name__}"
        )

    def _is_http_probe_config(p_conf: Any) -> bool:
        """Return True if the probe config uses http_get or http_post."""
        return (
            getattr(p_conf, "http_get", None) is not None
            or getattr(p_conf, "http_post", None) is not None
        )

    def _http_probe_references_vars(p_conf: Any, var_names: list[str]) -> bool:
        """Check if an HTTP probe config's URL or body references any of the given variable names.

        Inspects the raw (pre-resolved) strings so per-replica variable references like
        ``${{ variables.CONCURRENCY }}``, ``${CONCURRENCY}``, or ``${SFLOW_REPLICA_INDEX}``
        are detected.  ``var_names`` should include both user-declared sweep variables and
        reserved replica variables (e.g. ``SFLOW_REPLICA_INDEX``).
        """
        if not var_names:
            return False
        texts: list[str] = []
        http = getattr(p_conf, "http_get", None) or getattr(p_conf, "http_post", None)
        if http is None:
            return False
        texts.append(str(http.url))
        body = getattr(http, "body", None)
        if body is not None:
            texts.append(str(body))
        combined = " ".join(texts)
        return any(var_name in combined for var_name in var_names)

    def _probe_config_list(p_conf: Any) -> list[Any]:
        if p_conf is None:
            return []
        return p_conf if isinstance(p_conf, list) else [p_conf]

    def _build_probe(
        task_name: str,
        *,
        p_conf: Any,
        p_type: ProbeType,
        default_host: str | None = None,
    ):
        """
        Convert a ProbeConfig (from config schema) into a concrete Probe instance.
        """
        delay = int(getattr(p_conf, "delay", 0))
        timeout = _resolve_int(
            task_name, field=f"probes.{p_type}.timeout", value=p_conf.timeout
        )
        each_check_timeout = _resolve_int(
            task_name, field=f"probes.{p_type}.each_check_timeout", value=p_conf.each_check_timeout
        )
        interval = _resolve_int(
            task_name, field=f"probes.{p_type}.interval", value=p_conf.interval
        )
        success_threshold = _resolve_int(
            task_name,
            field=f"probes.{p_type}.success_threshold",
            value=p_conf.success_threshold,
        )
        failure_threshold = _resolve_int(
            task_name,
            field=f"probes.{p_type}.failure_threshold",
            value=p_conf.failure_threshold,
        )

        if delay < 0:
            raise ValueError(f"Task '{task_name}' probes.{p_type}.delay must be >= 0")
        if timeout < 0:
            raise ValueError(f"Task '{task_name}' probes.{p_type}.timeout must be >= 0")
        if each_check_timeout < 0:
            raise ValueError(f"Task '{task_name}' probes.{p_type}.each_check_timeout must be >= 0")
        if interval < 0:
            raise ValueError(
                f"Task '{task_name}' probes.{p_type}.interval must be >= 0"
            )
        if success_threshold <= 0:
            raise ValueError(
                f"Task '{task_name}' probes.{p_type}.success_threshold must be > 0"
            )
        if failure_threshold <= 0:
            raise ValueError(
                f"Task '{task_name}' probes.{p_type}.failure_threshold must be > 0"
            )

        common = dict(
            type=p_type,
            delay=delay,
            timeout=timeout,
            each_check_timeout=each_check_timeout,
            interval=interval,
            success_threshold=success_threshold,
            failure_threshold=failure_threshold,
        )

        if getattr(p_conf, "tcp_port", None) is not None:
            tcp = p_conf.tcp_port
            port = _resolve_int(
                task_name, field=f"probes.{p_type}.tcp_port.port", value=tcp.port
            )
            host_raw = getattr(tcp, "host", None)
            host = (
                str(resolver.resolve(host_raw, ctx))
                if host_raw is not None and resolver.has_expression(host_raw)
                else (
                    str(host_raw)
                    if host_raw is not None
                    else (default_host if default_host is not None else "127.0.0.1")
                )
            )
            on_node = getattr(tcp, "on_node", "first")
            return TcpPortProbe(host=host, port=port, on_node=on_node, **common)

        if getattr(p_conf, "http_get", None) is not None:
            http = p_conf.http_get
            url_raw = str(http.url)
            url = (
                str(resolver.resolve(url_raw, ctx))
                if resolver.has_expression(url_raw)
                else url_raw
            )
            return HttpGetProbe(
                url=url, headers=getattr(http, "headers", None), **common
            )

        if getattr(p_conf, "http_post", None) is not None:
            http = p_conf.http_post
            url_raw = str(http.url)
            url = (
                str(resolver.resolve(url_raw, ctx))
                if resolver.has_expression(url_raw)
                else url_raw
            )
            body_raw = getattr(http, "body", None)
            body = (
                str(resolver.resolve(body_raw, ctx))
                if body_raw is not None and resolver.has_expression(body_raw)
                else body_raw
            )
            return HttpPostProbe(
                url=url,
                headers=getattr(http, "headers", None),
                body=body,
                **common,
            )

        if getattr(p_conf, "log_watch", None) is not None:
            lw = p_conf.log_watch
            match_count = (
                _resolve_int(
                    task_name,
                    field=f"probes.{p_type}.log_watch.match_count",
                    value=lw.match_count,
                )
                if getattr(lw, "match_count", None) is not None
                else 1
            )
            return LogWatchProbe(
                regex_pattern=str(lw.regex_pattern),
                logger_task_name=getattr(lw, "logger", None),
                match_count=match_count,
                **common,
            )

        raise ValueError(
            f"Task '{task_name}' probes.{p_type} has no probe type configured"
        )

    task_graph = TaskGraph()

    # Track variable names already flagged as overridden by backend runtime env,
    # so we warn at most once per name across the whole graph build.
    warned_env_overrides: set[str] = set()

    # ---------------------------------------------------------------------
    # Replica planning: expand base tasks into concrete DAG nodes
    # ---------------------------------------------------------------------
    # base task name -> list of concrete node names (replicas)
    replica_names_by_base: dict[str, list[str]] = {}
    # concrete node name -> per-replica env mappings (stringified)
    replica_envs: dict[str, dict[str, str]] = {}
    # base task name -> replica policy ("parallel" / "sequential")
    replica_policy_by_base: dict[str, str] = {}

    for t_conf in config.workflow.tasks:
        if not t_conf.replicas:
            replica_names_by_base[t_conf.name] = [t_conf.name]
            replica_envs[t_conf.name] = {"SFLOW_REPLICA_INDEX": "0"}
            replica_policy_by_base[t_conf.name] = "parallel"
            continue

        r = t_conf.replicas
        policy = _resolve_replica_policy(t_conf.name, r.policy)
        replica_policy_by_base[t_conf.name] = policy

        sweep_vars = list(r.variables or [])
        instances = (
            _replica_sweep_instances(t_conf.name, sweep_vars) if sweep_vars else []
        )

        if r.count is not None:
            count = _resolve_replica_count(t_conf.name, r.count)
            if instances and count != len(instances):
                raise ValueError(
                    f"Task '{t_conf.name}' replicas.count={count} does not match sweep size {len(instances)} "
                    f"derived from replicas.variables={sweep_vars}"
                )
            if not instances:
                instances = [{} for _ in range(count)]
        else:
            # If count is omitted and no sweep vars are specified, default to 1 replica.
            if not instances:
                instances = [{}]

        # Generate replica names based on sweep variable values or numeric index
        def _make_replica_name(
            base_name: str, idx: int, instance: dict[str, Any], sweep_vars: list[str]
        ) -> str:
            """Generate replica name from variable values (if sweep) or numeric index."""
            if sweep_vars and instance:
                # Use variable values in the order they appear in sweep_vars
                value_parts = []
                for var_name in sweep_vars:
                    if var_name in instance:
                        val = instance[var_name]
                        # Sanitize value for use in task name (replace problematic chars)
                        val_str = (
                            str(val)
                            .replace(".", "_")
                            .replace("-", "_")
                            .replace(" ", "_")
                        )
                        value_parts.append(val_str)
                if value_parts:
                    return f"{base_name}_{'_'.join(value_parts)}"
            # Fallback to numeric index when no sweep variables
            return f"{base_name}_{idx}"

        concrete_names = [
            _make_replica_name(t_conf.name, i, instances[i], sweep_vars)
            for i in range(len(instances))
        ]
        replica_names_by_base[t_conf.name] = concrete_names
        for i, node_name in enumerate(concrete_names):
            env: dict[str, str] = {"SFLOW_REPLICA_INDEX": str(i)}
            for k, v in instances[i].items():
                env[k] = str(v)
            replica_envs[node_name] = env

    # Track sweep variable names per replica (for dry-run display)
    # concrete node name -> list of sweep variable names
    replica_sweep_vars: dict[str, list[str]] = {}
    for t_conf in config.workflow.tasks:
        if t_conf.replicas and t_conf.replicas.variables:
            sweep_var_names = list(t_conf.replicas.variables)
            for node_name in replica_names_by_base.get(t_conf.name, []):
                replica_sweep_vars[node_name] = sweep_var_names

    resource_placements = plan_resource_placements(
        config,
        state,
        resolver=resolver,
        ctx=ctx,
        replica_names_by_base=replica_names_by_base,
        replica_policy_by_base=replica_policy_by_base,
    )

    # Resolve service ports up front so services_by_node (below) can use them.
    ports_by_node: dict[str, list[TaskPort]] = {}
    for node_name, placement in resource_placements.items():
        resolved_ports: list[TaskPort] = []
        for p in list(getattr(placement.task_config, "ports", None) or []):
            port_val = _resolve_int(node_name, field="ports.port", value=p.port)
            if not (1 <= port_val <= 65535):
                raise ValueError(
                    f"Task '{node_name}' ports.port must be in 1..65535, got {port_val}"
                )
            resolved_ports.append(TaskPort(port=port_val, name=p.name))
        ports_by_node[node_name] = resolved_ports

    services_by_node = {
        node_name: compute_task_service(
            backend=placement.backend,
            assigned_nodes=placement.assigned_nodes,
            ports=ports_by_node.get(node_name, []),
        )
        for node_name, placement in resource_placements.items()
    }
    # task.<name>.service for probes (additive; _build_probe reads ctx at call time).
    ctx["task"] = {
        node_name: {"service": service}
        for node_name, service in services_by_node.items()
    }

    # First pass: add task nodes in planner order so dry-run and real run share
    # the same resource placement and conflict-checking source of truth.
    for node_name, placement in resource_placements.items():
        t_conf = placement.task_config
        idx = placement.replica_index
        replica_policy = placement.replica_policy
        base = t_conf.name
        concrete_nodes = replica_names_by_base.get(base, [base])
        if not concrete_nodes:
            raise ValueError(f"Task '{base}' produced zero replicas")
        backend = placement.backend
        assigned_nodes = placement.assigned_nodes

        # Resolve operator (operator-only execution model).
        task_operator = None
        operator_name: str | None = None
        operator_overrides: dict[str, Any] = {}
        if t_conf.operator:
            if isinstance(t_conf.operator, str):
                operator_name = t_conf.operator
            else:
                operator_name = t_conf.operator.name
                operator_overrides = dict(
                    t_conf.operator.model_dump(exclude={"name"}, exclude_none=True)
                )

            base_op = operator_confs.get(operator_name)
            if base_op is None:
                raise ValueError(
                    f"Task '{t_conf.name}' references unknown operator '{operator_name}'"
                )

            merged = base_op.model_dump()
            if "extra_args" in operator_overrides and merged.get("extra_args"):
                operator_overrides["extra_args"] = list(
                    merged["extra_args"]
                ) + list(operator_overrides["extra_args"])
            merged.update(operator_overrides)
            merged["name"] = operator_name
            merged = _resolve_value(merged)
            op_conf = operator_adapter.validate_python(merged)
        else:
            # Default operator is backend-owned behavior.
            operator_name = f"default_{backend.name}"
            task_operator = backend.default_operator(
                name=operator_name,
                assigned_nodes=assigned_nodes,
            )
            op_conf = task_operator.config

        if task_operator is None:
            operator_cls = get_operator_class(op_conf.type)
            task_operator = operator_cls(op_conf)  # type: ignore[arg-type]

        cuda_visible = placement.cuda_visible_devices

        task_operator.apply_backend_context(
            backend=backend,
            assigned_nodes=assigned_nodes,
            artifacts=list((state.artifacts or {}).values()),
            cuda_visible_devices=cuda_visible,
            gpu_count=placement.gpu_count,
        )

        task_logger = get_logger(f"sflow.task.{node_name}")
        task_logger.propagate = False

        # Resolve `${{ ... }}` expressions inside task scripts using the current context.
        # For replicas with sweep variables, overlay per-replica values so that
        # ${{ variables.CONCURRENCY }} resolves to the replica-specific value.
        # Note: `${{ task.* }}` expressions are resolved in a second pass after
        # all tasks are built (see below).
        replica_env = replica_envs.get(node_name, {})
        if replica_env:
            from sflow.core.variable import VariableValue

            replica_ctx = dict(ctx)
            replica_variables = dict(ctx.get("variables", {}))
            for k, v in replica_env.items():
                if k == "SFLOW_REPLICA_INDEX":
                    continue
                existing = replica_variables.get(k)
                domain = existing.domain if isinstance(existing, VariableValue) else None
                typed_v = _maybe_int(v)
                wrapped = VariableValue(typed_v, domain=domain)
                replica_variables[k] = wrapped
                replica_ctx[k] = wrapped
            replica_ctx["variables"] = replica_variables
            resolve_ctx = replica_ctx
        else:
            resolve_ctx = ctx

        script = [
            str(resolver.resolve(line, resolve_ctx))
            if resolver.has_expression(line) and "task." not in line
            else line
            for line in list(t_conf.script)
        ]
        task = Task(
            name=node_name,
            logger=task_logger,
            operator=task_operator,
            status=TaskStatus.INITIATED,
            script=script,
        )
        task.assigned_nodes = list(assigned_nodes or [])
        # The planner computes this slice uniformly for every backend; it is only
        # injected into the execution env via backend.resource_env (skipped for
        # Kubernetes). Carried on the task so the dry-run allocation map is
        # consistent across backends.
        task.cuda_visible_devices = cuda_visible
        task.operator_name = operator_name
        # Authoritative base->replica link (config task name) for plan-time consumers
        # (e.g. the monitor planner) so they never re-derive it from the name string.
        task.base_name = t_conf.name
        task.sweep_variables = replica_sweep_vars.get(node_name, [])

        # Build SFLOW_TASK_ASSIGNED_NODE_NAMES and SFLOW_TASK_ASSIGNED_NODE_IPS env vars
        # These provide easy access to the task's assigned nodes in scripts
        if assigned_nodes and backend and backend.allocation:
            has_runtime_node_addresses = getattr(
                getattr(backend, "capabilities", None),
                "has_runtime_node_addresses",
                True,
            )
            alloc_nodes_by_name = {n.name: n for n in backend.allocation.nodes}
            node_names: list[str] = []
            node_ips: list[str] = []
            for n_name in assigned_nodes:
                node_names.append(n_name)
                if has_runtime_node_addresses:
                    node_obj = alloc_nodes_by_name.get(n_name)
                    if node_obj:
                        node_ips.append(node_obj.ip_address)
                    else:
                        node_ips.append("")  # Fallback if node not found
            task.envs["SFLOW_TASK_ASSIGNED_NODE_NAMES"] = ",".join(node_names)
            if has_runtime_node_addresses:
                task.envs["SFLOW_TASK_ASSIGNED_NODE_IPS"] = ",".join(node_ips)
        # Outputs (MVP): store parse patterns to be evaluated from the task log after completion.
        if getattr(t_conf, "outputs", None):
            for o in list(t_conf.outputs or []):
                # source is kept for schema parity; MVP parses from merged log file.
                task.output_specs.append(
                    OutputSpec(
                        pattern=str(o.pattern),
                        source=str(getattr(o, "source", "stdout")),
                    )
                )

        # Consolidated result parsing (new contract).
        # See docs/developer/dev-notes/result-parsing.md.
        if getattr(t_conf, "result", None) is not None:
            task.result_config = _build_result_config_runtime(t_conf.result)

        # ports resolved up front; attach to the task.
        task.ports = list(ports_by_node.get(node_name, []))

        # Uploads: attach unresolved expression strings; ExpressionResolver runs
        # at task-completion time (in core.uploads.run_task_uploads) so that
        # ${{ task.output_dir }} and other task-scoped refs have values.
        if getattr(t_conf, "uploads", None):
            # When a task is replicated, every replica shares the same upload `to:`
            # and would silently overwrite the others on the storage target. By
            # default sflow auto-disambiguates by inserting the replica name into
            # the uploaded filename (e.g. results.csv -> results_<replica>.csv).
            # Users opt out / control the layout by referencing ${{ task.name }}
            # in `to:`, in which case we leave the destination untouched.
            multiple_replicas = len(concrete_nodes) > 1
            for u in list(t_conf.uploads or []):
                user_named_replica = resolver.references_attribute(
                    u.to, "task", "name"
                )
                disambiguate = multiple_replicas and not user_named_replica
                # Warn once per upload spec (on the first replica) that sflow is
                # auto-renaming to avoid cross-replica overwrites.
                if disambiguate and idx == 0:
                    dest = u.to or "<filename>"
                    _logger.warning(
                        f"Task '{t_conf.name}' is replicated and upload to "
                        f"'{dest}' does not reference ${{{{ task.name }}}}; sflow "
                        f"will auto-rename each replica's upload (inserting "
                        f"'_<replica>' before the extension, e.g. 'results.csv' -> "
                        f"'results_{t_conf.name}_0.csv') so replicas don't overwrite "
                        f"each other. Add ${{{{ task.name }}}} to 'to:' to control "
                        f"the layout and silence this warning."
                    )
                task.uploads.append(
                    ResolvedUpload(
                        target=u.target,
                        from_expr=u.from_,
                        to_expr=u.to,
                        on_error=u.on_error,
                        disambiguate_with=task.name if disambiguate else None,
                    )
                )
        # Probes (readiness/failure)
        if t_conf.probes:
            # Default probe host: use the task's assigned node IP (not localhost),
            # so probes can reach services running on remote backend nodes.
            default_probe_host: str | None = None
            try:
                alloc = getattr(backend, "allocation", None)
                has_runtime_node_addresses = getattr(
                    getattr(backend, "capabilities", None),
                    "has_runtime_node_addresses",
                    True,
                )
                if (
                    has_runtime_node_addresses
                    and alloc
                    and getattr(alloc, "nodes", None)
                ):
                    by_name = {n.name: n.ip_address for n in alloc.nodes}
                    if assigned_nodes:
                        default_probe_host = by_name.get(assigned_nodes[0])
                    if default_probe_host is None:
                        default_probe_host = alloc.nodes[0].ip_address
            except Exception:
                default_probe_host = None

            # For parallel replicated tasks, skip HTTP probes on non-first
            # replicas when the probe URL/body don't reference any per-replica
            # variables — the probes would send identical requests, creating
            # unnecessary duplicate load.  Per-replica variables include
            # user-declared sweep variables and reserved variables like
            # SFLOW_REPLICA_INDEX.
            #
            # Sequential replicas always get their own probe because each
            # replica runs at a different time and needs an independent
            # timeout deadline.
            replica_var_names: list[str] = []
            if t_conf.replicas and len(concrete_nodes) > 1:
                per_replica_env = replica_envs.get(node_name, {})
                replica_var_names = list(per_replica_env.keys())
            is_non_first_replica = idx > 0 and len(concrete_nodes) > 1
            can_share_probe = (
                is_non_first_replica and replica_policy == "parallel"
            )

            readiness_probe_configs = _probe_config_list(t_conf.probes.readiness)
            if readiness_probe_configs:
                skip = (
                    can_share_probe
                    and all(
                        _is_http_probe_config(p_conf)
                        and not _http_probe_references_vars(
                            p_conf, replica_var_names
                        )
                        for p_conf in readiness_probe_configs
                    )
                )
                if skip:
                    _logger.debug(
                        "Skipping readiness HTTP probe on parallel replica '%s' "
                        "(identical to first replica)",
                        node_name,
                    )
                    first_task = task_graph.get_task(concrete_nodes[0])
                    if first_task is not None:
                        first_task.readiness_followers.append(node_name)
                else:
                    for p_conf in readiness_probe_configs:
                        task.probes.append(
                            _build_probe(
                                node_name,
                                p_conf=p_conf,
                                p_type=ProbeType.READINESS,
                                default_host=default_probe_host,
                            )
                        )
            failure_probe_configs = _probe_config_list(t_conf.probes.failure)
            if failure_probe_configs:
                skip = (
                    can_share_probe
                    and all(
                        _is_http_probe_config(p_conf)
                        and not _http_probe_references_vars(
                            p_conf, replica_var_names
                        )
                        for p_conf in failure_probe_configs
                    )
                )
                if skip:
                    _logger.debug(
                        "Skipping failure HTTP probe on parallel replica '%s' "
                        "(identical to first replica)",
                        node_name,
                    )
                    first_task = task_graph.get_task(concrete_nodes[0])
                    if first_task is not None:
                        first_task.failure_followers.append(node_name)
                else:
                    for p_conf in failure_probe_configs:
                        task.probes.append(
                            _build_probe(
                                node_name,
                                p_conf=p_conf,
                                p_type=ProbeType.FAILURE,
                                default_host=default_probe_host,
                            )
                        )
        task.backend_name = backend.name
        task.resource_release_after.update(placement.resource_release_after)
        # Optional retry policy (REQ-3.6).
        if t_conf.retries:
            retry_count = _resolve_int(
                node_name, field="retries.count", value=t_conf.retries.count
            )
            retry_interval = _resolve_int(
                node_name, field="retries.interval", value=t_conf.retries.interval
            )
            retry_backoff = _resolve_int(
                node_name,
                field="retries.backoff",
                value=t_conf.retries.backoff,
            )
            if retry_count < 0:
                raise ValueError(
                    f"Task '{node_name}' retries.count must be >= 0, got {retry_count}"
                )
            if retry_interval < 0:
                raise ValueError(
                    f"Task '{node_name}' retries.interval must be >= 0, got {retry_interval}"
                )
            if retry_backoff < 1:
                raise ValueError(
                    f"Task '{node_name}' retries.backoff must be >= 1, got {retry_backoff}"
                )
            task.retries = RetryPolicy(
                count=int(retry_count),
                interval=float(retry_interval),
                backoff=float(retry_backoff),
            )
        # Inject all resolved variables into task env by default (SRD intent).
        # Replica envs (including sweep vars) override global variables.
        task.envs.update(
            {k: str(v.value) for k, v in (state.variables or {}).items()}
        )
        # Also inject artifact paths as env vars (SRD REQ-1.5: `${NAME}` convenience).
        for aname, ainfo in (artifacts_ctx or {}).items():
            apath = ainfo.get("path")
            if apath is not None:
                task.envs.setdefault(aname, str(apath))
        task.envs.update(replica_envs.get(node_name, {}))
        backend_env = backend.resource_env(cuda_visible_devices=cuda_visible)
        # The backend's runtime env wins over same-named workflow variables (it
        # reflects allocation truth, e.g. SLURM_*). Surface this so a user-declared
        # variable being shadowed is not silent.
        user_var_names = set(state.variables or {})
        for env_key, env_val in backend_env.items():
            if (
                env_key in user_var_names
                and task.envs.get(env_key) not in (None, env_val)
                and env_key not in warned_env_overrides
            ):
                _logger.warning(
                    f"Backend '{backend.name}' runtime env '{env_key}' overrides "
                    f"workflow variable of the same name; the backend value is used "
                    f"for task launch."
                )
                warned_env_overrides.add(env_key)
        task.envs.update(backend_env)
        task_graph.dag.add_node(node_name, task)

        # If the task is replicated sequentially, enforce replica order by chaining edges.
        if (
            t_conf.replicas
            and replica_policy_by_base.get(base) == "sequential"
            and idx > 0
        ):
            task_graph.dag.add_edge(concrete_nodes[idx - 1], node_name)

    # Second pass: add edges
    for t_conf in config.workflow.tasks:
        if t_conf.depends_on:
            for dep in t_conf.depends_on:
                dep_replicas = replica_names_by_base.get(dep, [dep])
                # If the dependency is sequentially replicated, depending on the last replica is sufficient.
                if replica_policy_by_base.get(dep) == "sequential" and dep_replicas:
                    dep_nodes = [dep_replicas[-1]]
                else:
                    dep_nodes = dep_replicas

                # If the *target* task is sequentially replicated, only the first replica
                # needs to depend on upstream tasks; later replicas depend on the chain.
                target_nodes = replica_names_by_base.get(t_conf.name, [t_conf.name])
                if (
                    replica_policy_by_base.get(t_conf.name) == "sequential"
                    and target_nodes
                ):
                    target_nodes = [target_nodes[0]]

                for node_name in target_nodes:
                    for dep_node in dep_nodes:
                        task_graph.dag.add_edge(dep_node, node_name)

    # Third pass: resolve `${{ task.* }}` expressions in task scripts now that all tasks are built.
    # This enables referencing other tasks' assigned nodes and GPUs.
    tasks_ctx = build_tasks_ctx(
        task_graph, state.backends or {}, replica_names_by_base
    )
    task_ctx: dict[str, Any] = {
        "task": tasks_ctx,
        "variables": variables_ctx,
        "artifacts": artifacts_ctx,
        "backends": backends_ctx,
        "workflow": {"name": config.workflow.name},
        **variables_ctx,
    }

    for task in task_graph.get_tasks():
        new_script: list[str] = []
        for line in task.script:
            if resolver.has_expression(line) and "task." in line:
                try:
                    resolved = str(resolver.resolve(line, task_ctx))
                    new_script.append(resolved)
                except Exception as e:
                    task_exprs = extract_task_expressions(line)
                    hint = build_task_expression_hint(
                        task_exprs, tasks_ctx, replica_names_by_base
                    )
                    exprs_display = ", ".join(task_exprs) if task_exprs else "(unknown)"
                    location = resolver._find_expression_in_sources(
                        task_exprs[0] if task_exprs else line
                    )
                    msg = (
                        f"Failed to resolve task expression in "
                        f"'{task.name}' script{location}: {exprs_display}"
                    )
                    if hint:
                        msg += f"\n  Hint: {hint}"
                    raise ValueError(msg) from e
            else:
                new_script.append(line)
        task.script = new_script

    # Merge-pod mode (default auto per Kubernetes backend): bundle single-node GPU
    # tasks the planner co-located on one node into one pod so they share
    # NVLink/cuda_ipc (and one IMEX channel claim per node).
    _plan_merge_groups(task_graph, resource_placements)
    # Guard: a claimed ComputeDomain channel + merge off + >1 GPU pod/node contends
    # on the node's single IMEX channel -> hard-warn (still attempts).
    _warn_channel_contention(resource_placements)
    # App-agnostic interconnect hints (cross-node GPU placement + scope + IB +
    # channel) for the framework/admin-owned pieces sflow does not own.
    _warn_interconnect_hints(resource_placements)

    return task_graph


async def build_state(
    config: SflowConfig,
    *,
    allocate: bool = True,
    workspace_dir: Any | None = None,
    output_dir: Any | None = None,
    source_files: list[Any] | None = None,
    kubectl_config: Any | None = None,
) -> SflowState:
    """
    Build runtime state from configuration (composition root).

    This is intentionally kept out of core to avoid core importing plugins.
    """
    from pathlib import Path

    if source_files:
        resolver.source_files = [Path(f) for f in source_files]

    # Seed an empty workflow/state; we will populate task graph after resolution/allocation.
    wf = Workflow(name=config.workflow.name, task_graph=TaskGraph())
    state = SflowState(workflow=wf)

    # Resolve global variables and backends.
    state = resolve_global_variables(config, state)
    state = resolve_backends(config, state, kubectl_config=kubectl_config)

    # Allocate resources (unless dry-run).
    if allocate:
        # REQ-5.1: fail fast before consuming cluster resources.
        preflight_validate_backends(state)
        preflight_validate_container_images(config, state)
        preflight_validate_task_graph(
            config,
            state,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
        )
        state = await allocate_backends(state)
    else:
        # Populate placeholder allocations (for any remaining unallocated backends)
        # so workflow variables can reference backends.*
        state = _seed_placeholder_backend_allocations(state)

    try:
        # Backend-derived globals are available before artifacts; artifact-derived
        # globals stay deferred until artifact resolution has populated artifacts.*.
        state = resolve_deferred_global_variables(
            config,
            state,
            available_contexts=frozenset({"backends"}),
            defer_contexts=frozenset({"artifacts"}),
        )

        # Resolve artifacts after allocation so they can reference backend info
        # (e.g. ${{ backends.<name>.nodes[0].ip_address }} when runtime addresses exist).
        # Off-host backends (e.g. Kubernetes) execute remotely, so local fs:// paths
        # are passed through (not validated/created on the controller).
        state = resolve_artifacts(
            config,
            state,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            materialize=allocate,
            remote_filesystem=backends_execute_offhost(state),
        )

        # Top-level variables may defer references to backends.* and artifacts.*
        # until those contexts exist.
        state = resolve_deferred_global_variables(config, state)

        # Workflow variables may reference backend allocations (e.g. backends.<name>.nodes[0].ip_address),
        # so resolve them after allocation (real or placeholder).
        state = resolve_workflow_variables(config, state)

        # Instantiate storage targets (S3, etc.) after variables/artifacts/backends are
        # known, so target fields like `prefix:` can reference any of them.
        state = resolve_storage_targets(config, state)

        # Capture workflow-level upload_all spec (resolved when the run finishes,
        # against `workflow.output_dir` / `workflow.run_id`, so it stays unresolved here).
        if config.workflow.upload_all is not None:
            from sflow.core.uploads import ResolvedWorkflowUpload

            ua = config.workflow.upload_all
            state.workflow_upload = ResolvedWorkflowUpload(
                target=ua.target,
                to_expr=ua.to,
                on_error=ua.on_error,
            )

        # Build the task graph (uses allocation info if present).
        tg = build_task_graph(config, state)
        state.workflow = Workflow(name=config.workflow.name, task_graph=tg)

        # Build the hardware monitor schedule (plan time). Resolves nodes/GPUs,
        # dedups to per-node collectors, and attaches consumers to state/tasks.
        # Collector scripts + raw dir are materialized only on real runs.
        from sflow.app.monitor_planner import build_monitor_registry

        state.monitor_registry = build_monitor_registry(
            config, state, output_dir=output_dir, materialize=allocate
        )
        return state
    except BaseException:
        # If we allocated real resources, make sure we release them even if planning fails
        # (e.g., GPU planning/validation raises during build_task_graph).
        if allocate:
            try:
                await release_backends(state)
            except Exception as e:
                _logger.error(
                    f"Failed to release backends after build_state failure: {e}"
                )
        raise
