# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sflow.config.schema import SflowConfig
from sflow.core.backend import Backend
from sflow.core.compute_node import ComputeNode
from sflow.core.dag import DAG
from sflow.core.state import SflowState


GpuReservation = tuple[int, int, str, str]
NodeReservation = tuple[str, str]


@dataclass(frozen=True)
class ResourcePlacement:
    """Final resource placement for one concrete task replica."""

    task_config: Any
    replica_index: int
    replica_policy: str
    backend: Backend
    assigned_nodes: list[str]
    cuda_visible_devices: str | None
    resource_release_after: dict[str, str]
    nodes_inferred_from_gpus: bool = False


class ReservationPolicyPlanner:
    """Shared release policy checks for resource reservations."""

    def __init__(self) -> None:
        self.ancestors: dict[str, set[str]] = {}
        self.has_readiness: dict[str, bool] = {}
        self._release_after: dict[str, str] = {}
        self._release_explicit: dict[str, bool] = {}

    @staticmethod
    def release_after_value(value: Any) -> str:
        return str(getattr(value, "value", value))

    def effective_release_after(self, resource_conf: Any, *, has_readiness: bool) -> str:
        if "release_after" in getattr(resource_conf, "model_fields_set", set()):
            return self.release_after_value(resource_conf.release_after)
        return "workflow_completion" if has_readiness else "task_completion"

    def set_task_policy(
        self, task_name: str, *, has_readiness: bool, resource_conf: Any | None
    ) -> None:
        self.has_readiness[task_name] = has_readiness
        if resource_conf is None:
            return
        self._release_after[task_name] = self.effective_release_after(
            resource_conf,
            has_readiness=has_readiness,
        )
        self._release_explicit[task_name] = "release_after" in getattr(
            resource_conf, "model_fields_set", set()
        )

    def set_ancestors(self, ancestors: dict[str, set[str]]) -> None:
        self.ancestors = ancestors

    def reservation_reuse_result(
        self,
        *,
        owner_task: str,
        current_task: str,
        release_after: str,
    ) -> tuple[bool, str]:
        if owner_task == current_task:
            return False, "same task"
        if owner_task not in self.ancestors.get(current_task, set()):
            return False, f"not an upstream dependency of {current_task}"
        if release_after == "task_ready":
            return True, "released after task readiness"
        if release_after == "task_completion":
            if self.has_readiness.get(owner_task, False):
                return (
                    False,
                    "has readiness probes; task_completion is not released at READY",
                )
            return True, "released after task completion"
        if release_after == "workflow_completion":
            if (
                not self._release_explicit.get(owner_task, False)
                and self.has_readiness.get(owner_task, False)
            ):
                return (
                    False,
                    "inferred workflow_completion because task has readiness probes and may still be running after READY",
                )
            return False, "held until workflow completion"
        return False, f"unknown release_after policy {release_after!r}"

    def reservation_is_reusable(
        self,
        *,
        owner_task: str,
        current_task: str,
        release_after: str,
    ) -> bool:
        reusable, _reason = self.reservation_reuse_result(
            owner_task=owner_task,
            current_task=current_task,
            release_after=release_after,
        )
        return reusable


class GpuReservationPlanner(ReservationPolicyPlanner):
    """Plan GPU reservations against concrete DAG lifetime information."""

    def __init__(self) -> None:
        super().__init__()
        # (backend_name, node_name) -> [(start, end_exclusive, owner_task, release_after)]
        self.reservations: dict[tuple[str, str], list[GpuReservation]] = {}
        self.task_stages: dict[str, int] = {}

    @property
    def gpu_release_after(self) -> dict[str, str]:
        return self._release_after

    @gpu_release_after.setter
    def gpu_release_after(self, value: dict[str, str]) -> None:
        self._release_after = value

    @property
    def gpu_release_explicit(self) -> dict[str, bool]:
        return self._release_explicit

    @gpu_release_explicit.setter
    def gpu_release_explicit(self, value: dict[str, bool]) -> None:
        self._release_explicit = value

    def set_concrete_task(
        self,
        task_name: str,
        *,
        has_readiness: bool,
        gpu_resource_conf: Any | None,
    ) -> None:
        self.set_task_policy(
            task_name,
            has_readiness=has_readiness,
            resource_conf=gpu_resource_conf,
        )

    def set_task_stages(self, stages: dict[str, int]) -> None:
        self.task_stages = stages

    def conflicting_reservations(
        self,
        cursor_key: tuple[str, str],
        task_name: str,
    ) -> list[GpuReservation]:
        return [
            reservation
            for reservation in self.reservations.get(cursor_key, [])
            if not self.reservation_is_reusable(
                owner_task=reservation[2],
                current_task=task_name,
                release_after=reservation[3],
            )
        ]

    def format_blockers(
        self,
        *,
        backend_name: str,
        node_name: str,
        task_name: str,
    ) -> str:
        reservations = list(self.reservations.get((backend_name, node_name), []))
        if not reservations:
            return ""

        by_gpu: dict[int, list[tuple[int, int, str, str, bool, str]]] = {}
        for start, end, owner, release_after in reservations:
            reusable, reason = self.reservation_reuse_result(
                owner_task=owner,
                current_task=task_name,
                release_after=release_after,
            )
            for gpu_idx in range(start, end):
                by_gpu.setdefault(gpu_idx, []).append(
                    (start, end, owner, release_after, reusable, reason)
                )

        lines = [
            f"\n  GPU reservation graph on node '{node_name}' while planning task '{task_name}':"
        ]
        timeline_lines = self._format_timeline(
            node_name=node_name,
            task_name=task_name,
            by_gpu=by_gpu,
        )
        if timeline_lines:
            lines.append("    Timeline:")
            lines.extend(timeline_lines)
        for gpu_idx in sorted(by_gpu):
            entries = by_gpu[gpu_idx]
            chain = " -> ".join(entry[2] for entry in entries)
            lines.append(f"    GPU {gpu_idx}: {chain}")
            blocking = [entry for entry in entries if not entry[4]]
            reusable = [entry for entry in entries if entry[4]]
            if blocking:
                lines.append("      blocking:")
                for _start, _end, owner, release_after, _reusable, reason in blocking:
                    lines.append(
                        f"        - {owner}: release_after={release_after}; {reason}"
                    )
            if reusable:
                lines.append("      reusable:")
                for _start, _end, owner, release_after, _reusable, reason in reusable:
                    lines.append(
                        f"        - {owner}: release_after={release_after}; {reason}"
                    )
        return "\n".join(lines) + "\n"

    def _format_timeline(
        self,
        *,
        node_name: str,
        task_name: str,
        by_gpu: dict[int, list[tuple[int, int, str, str, bool, str]]],
    ) -> list[str]:
        timeline_groups: dict[tuple[int, str, str], list[int]] = {}
        first_blocker_by_gpu: dict[int, str] = {}
        for gpu_idx in sorted(by_gpu):
            for index, (_start, _end, owner, _release_after, reusable, _reason) in enumerate(
                by_gpu[gpu_idx]
            ):
                stage = self.task_stages.get(owner, -1)
                verb = "uses" if index == 0 else "reuses"
                timeline_groups.setdefault((stage, owner, verb), []).append(gpu_idx)
                if not reusable and gpu_idx not in first_blocker_by_gpu:
                    first_blocker_by_gpu[gpu_idx] = owner

        def _gpu_label(gpu_indices: list[int]) -> str:
            ordered = sorted(set(gpu_indices))
            if not ordered:
                return "GPUs"
            ranges: list[str] = []
            start = prev = ordered[0]
            for idx in ordered[1:]:
                if idx == prev + 1:
                    prev = idx
                    continue
                ranges.append(str(start) if start == prev else f"{start}-{prev}")
                start = prev = idx
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            prefix = "GPU" if len(ordered) == 1 else "GPUs"
            return f"{prefix} {', '.join(ranges)}"

        lines = []
        for stage, owner, verb in sorted(timeline_groups):
            lines.append(
                f"      Stage {stage}: {owner} {verb} {_gpu_label(timeline_groups[(stage, owner, verb)])}"
            )
        if first_blocker_by_gpu:
            for gpu_idx, owner in sorted(first_blocker_by_gpu.items()):
                lines.append(
                    f"      Failed placement: {task_name} needs GPU {gpu_idx}, but it is blocked by {owner}"
                )
        return lines

    def first_available_start(
        self,
        *,
        backend_name: str,
        node_name: str,
        task_name: str,
        count: int,
        capacity: int,
    ) -> int | None:
        start = 0
        reservations = sorted(
            self.conflicting_reservations((backend_name, node_name), task_name),
            key=lambda r: (r[0], r[1]),
        )
        for res_start, res_end, _owner, _release_after in reservations:
            if start + count <= res_start:
                return start
            if start < res_end:
                start = res_end
        if start + count <= capacity:
            return start
        return None

    def reserve(
        self,
        *,
        backend_name: str,
        node_name: str,
        task_name: str,
        start: int,
        count: int,
    ) -> None:
        release_after = self.gpu_release_after.get(task_name, "workflow_completion")
        self.reservations.setdefault((backend_name, node_name), []).append(
            (start, start + count, task_name, release_after)
        )


class NodeReservationPlanner(ReservationPolicyPlanner):
    """Plan whole-node reservations against concrete DAG lifetime information."""

    def __init__(self) -> None:
        super().__init__()
        # (backend_name, node_name) -> [(owner_task, release_after)]
        self.reservations: dict[tuple[str, str], list[NodeReservation]] = {}

    @property
    def node_release_after(self) -> dict[str, str]:
        return self._release_after

    @node_release_after.setter
    def node_release_after(self, value: dict[str, str]) -> None:
        self._release_after = value

    @property
    def node_release_explicit(self) -> dict[str, bool]:
        return self._release_explicit

    @node_release_explicit.setter
    def node_release_explicit(self, value: dict[str, bool]) -> None:
        self._release_explicit = value

    def set_concrete_task(
        self,
        task_name: str,
        *,
        has_readiness: bool,
        node_resource_conf: Any | None,
    ) -> None:
        self.set_task_policy(
            task_name,
            has_readiness=has_readiness,
            resource_conf=node_resource_conf,
        )

    def conflicting_reservations(
        self,
        cursor_key: tuple[str, str],
        task_name: str,
    ) -> list[NodeReservation]:
        return [
            reservation
            for reservation in self.reservations.get(cursor_key, [])
            if not self.reservation_is_reusable(
                owner_task=reservation[0],
                current_task=task_name,
                release_after=reservation[1],
            )
        ]

    def first_available_nodes(
        self,
        *,
        backend_name: str,
        task_name: str,
        candidate_nodes: list[str],
        count: int,
    ) -> list[str] | None:
        available = [
            node_name
            for node_name in candidate_nodes
            if not self.conflicting_reservations((backend_name, node_name), task_name)
        ]
        if len(available) < count:
            return None
        return available[:count]

    def format_blockers(
        self,
        *,
        backend_name: str,
        candidate_nodes: list[str],
        task_name: str,
    ) -> str:
        lines: list[str] = []
        for node_name in candidate_nodes:
            reservations = self.reservations.get((backend_name, node_name), [])
            if not reservations:
                continue
            lines.append(
                f"\n  Node reservation graph on node '{node_name}' while planning task '{task_name}':"
            )
            for owner, release_after in reservations:
                reusable, reason = self.reservation_reuse_result(
                    owner_task=owner,
                    current_task=task_name,
                    release_after=release_after,
                )
                status = "reusable" if reusable else "blocking"
                lines.append(
                    f"    - {owner}: release_after={release_after}; {status}; {reason}"
                )
        return "\n".join(lines) + ("\n" if lines else "")

    def reserve(
        self,
        *,
        backend_name: str,
        node_name: str,
        task_name: str,
    ) -> None:
        release_after = self.node_release_after.get(task_name, "workflow_completion")
        self.reservations.setdefault((backend_name, node_name), []).append(
            (task_name, release_after)
        )


def _maybe_int(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


class ResourcePlacementPlanner:
    """Compute task resource placement and conflict checks for a workflow DAG."""

    def __init__(
        self,
        config: SflowConfig,
        state: SflowState,
        *,
        resolver: Any,
        ctx: dict[str, Any],
        replica_names_by_base: dict[str, list[str]],
        replica_policy_by_base: dict[str, str],
    ) -> None:
        self.config = config
        self.state = state
        self.resolver = resolver
        self.ctx = ctx
        self.replica_names_by_base = replica_names_by_base
        self.replica_policy_by_base = replica_policy_by_base
        self.gpu_planner = GpuReservationPlanner()
        self.node_planner = NodeReservationPlanner()
        # Reserved for future default placement policy. Empty preserves today's
        # behavior: tasks without explicit resources use all allocation nodes.
        self.default_task_nodes: dict[tuple[str, str], list[str]] = {}

    def plan(self) -> dict[str, ResourcePlacement]:
        concrete_dag, concrete_task_meta = self._build_concrete_dag()
        concrete_order = concrete_dag.topological_sort()
        ancestors = {
            node_name: self._collect_ancestors(concrete_dag, node_name)
            for node_name in concrete_order
        }
        self.gpu_planner.set_ancestors(ancestors)
        self.node_planner.set_ancestors(ancestors)
        self.gpu_planner.set_task_stages(
            self._task_stages(concrete_dag, concrete_order)
        )

        placements: dict[str, ResourcePlacement] = {}
        for concrete_node_name in concrete_order:
            t_conf, replica_index = concrete_task_meta[concrete_node_name]
            base_task_name = t_conf.name
            replica_policy = self.replica_policy_by_base.get(base_task_name, "parallel")
            backend = self._resolve_backend(t_conf)

            resources = t_conf.resources
            nodes_resource_conf = resources.nodes if resources and resources.nodes else None
            gpus_resource_conf = resources.gpus if resources and resources.gpus else None
            nodes_reservation_enabled = (
                nodes_resource_conf is not None
                and "release_after" in getattr(nodes_resource_conf, "model_fields_set", set())
            )

            assigned_nodes, nodes_inferred_from_gpus = self._assigned_nodelist(
                task_name=concrete_node_name,
                base_task_name=base_task_name,
                runtime_backend=backend,
                replica_index=replica_index,
                replica_policy=replica_policy,
                nodes_resource_present=nodes_resource_conf is not None,
                nodes_reservation_enabled=nodes_reservation_enabled,
                nodes_indices_raw=getattr(nodes_resource_conf, "indices", None),
                nodes_count_raw=getattr(nodes_resource_conf, "count", None),
                nodes_exclude_raw=getattr(nodes_resource_conf, "exclude", None),
                gpus_count_raw=getattr(gpus_resource_conf, "count", None),
            )
            cuda_visible = self._cuda_visible_devices(
                task_name=concrete_node_name,
                runtime_backend=backend,
                assigned_nodes=assigned_nodes,
                replica_index=replica_index,
                gpus_count_raw=getattr(gpus_resource_conf, "count", None),
            )

            release_after: dict[str, str] = {}
            has_readiness = self.gpu_planner.has_readiness.get(concrete_node_name, False)
            if nodes_resource_conf is not None and "release_after" in getattr(
                nodes_resource_conf, "model_fields_set", set()
            ):
                release_after["nodes"] = self.node_planner.effective_release_after(
                    nodes_resource_conf,
                    has_readiness=has_readiness,
                )
            if gpus_resource_conf is not None:
                release_after["gpus"] = self.gpu_planner.effective_release_after(
                    gpus_resource_conf,
                    has_readiness=has_readiness,
                )

            placements[concrete_node_name] = ResourcePlacement(
                task_config=t_conf,
                replica_index=replica_index,
                replica_policy=replica_policy,
                backend=backend,
                assigned_nodes=assigned_nodes,
                cuda_visible_devices=cuda_visible,
                resource_release_after=release_after,
                nodes_inferred_from_gpus=nodes_inferred_from_gpus,
            )
        return placements

    def _build_concrete_dag(self) -> tuple[DAG, dict[str, tuple[Any, int]]]:
        concrete_dag = DAG(name=self.config.workflow.name)
        concrete_task_meta: dict[str, tuple[Any, int]] = {}
        for t_conf in self.config.workflow.tasks:
            for idx, node_name in enumerate(
                self.replica_names_by_base.get(t_conf.name, [t_conf.name])
            ):
                concrete_dag.add_node(node_name)
                concrete_task_meta[node_name] = (t_conf, idx)
                has_readiness = bool(
                    t_conf.probes and t_conf.probes.readiness is not None
                )
                gpu_resource_conf = (
                    t_conf.resources.gpus
                    if t_conf.resources and t_conf.resources.gpus
                    else None
                )
                node_resource_conf = (
                    t_conf.resources.nodes
                    if t_conf.resources and t_conf.resources.nodes
                    else None
                )
                self.gpu_planner.set_concrete_task(
                    node_name,
                    has_readiness=has_readiness,
                    gpu_resource_conf=gpu_resource_conf,
                )
                self.node_planner.set_concrete_task(
                    node_name,
                    has_readiness=has_readiness,
                    node_resource_conf=node_resource_conf,
                )

        for t_conf in self.config.workflow.tasks:
            concrete_nodes = self.replica_names_by_base.get(t_conf.name, [t_conf.name])
            if (
                t_conf.replicas
                and self.replica_policy_by_base.get(t_conf.name) == "sequential"
            ):
                for idx in range(1, len(concrete_nodes)):
                    concrete_dag.add_edge(concrete_nodes[idx - 1], concrete_nodes[idx])

            if t_conf.depends_on:
                for dep in t_conf.depends_on:
                    dep_replicas = self.replica_names_by_base.get(dep, [dep])
                    if self.replica_policy_by_base.get(dep) == "sequential" and dep_replicas:
                        dep_nodes = [dep_replicas[-1]]
                    else:
                        dep_nodes = dep_replicas

                    target_nodes = list(concrete_nodes)
                    if (
                        self.replica_policy_by_base.get(t_conf.name) == "sequential"
                        and target_nodes
                    ):
                        target_nodes = [target_nodes[0]]

                    for node_name in target_nodes:
                        for dep_node in dep_nodes:
                            concrete_dag.add_edge(dep_node, node_name)

        return concrete_dag, concrete_task_meta

    def _collect_ancestors(
        self, concrete_dag: DAG, node_name: str, seen: set[str] | None = None
    ) -> set[str]:
        if seen is None:
            seen = set()
        for dep in concrete_dag.get_dependencies(node_name):
            if dep in seen:
                continue
            seen.add(dep)
            seen.update(self._collect_ancestors(concrete_dag, dep, seen))
        return seen

    def _task_stages(self, concrete_dag: DAG, concrete_order: list[str]) -> dict[str, int]:
        stages: dict[str, int] = {}
        for node_name in concrete_order:
            deps = concrete_dag.get_dependencies(node_name)
            stages[node_name] = 0 if not deps else max(stages[dep] for dep in deps) + 1
        return stages

    def _resolve_backend(self, t_conf: Any) -> Backend:
        if isinstance(t_conf.backend, str):
            backend = (self.state.backends or {}).get(t_conf.backend)
        elif t_conf.backend is None:
            backend = self.state.default_backend or next(iter((self.state.backends or {}).values()))
        else:
            raise NotImplementedError(
                f"Inline backend overrides are not supported yet for task '{t_conf.name}'"
            )
        if backend is None:
            raise ValueError(f"Task '{t_conf.name}' references unknown backend")
        return backend

    def _resolve_int(self, task_name: str, *, field: str, value: Any) -> int:
        resolved = (
            self.resolver.resolve(value, self.ctx)
            if self.resolver.has_expression(value)
            else value
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

    def _resolve_int_list(self, task_name: str, *, field: str, values: Any) -> list[int]:
        resolved_values = (
            self.resolver.resolve(values, self.ctx)
            if self.resolver.has_expression(values)
            else values
        )
        if isinstance(resolved_values, str):
            import json

            try:
                resolved_values = json.loads(resolved_values)
            except json.JSONDecodeError:
                pass
        if not isinstance(resolved_values, list):
            resolved_values = [resolved_values]
        return [
            self._resolve_int(task_name, field=f"{field}[{i}]", value=value)
            for i, value in enumerate(resolved_values)
        ]

    def _assigned_nodelist(
        self,
        *,
        task_name: str,
        base_task_name: str,
        runtime_backend: Backend,
        replica_index: int,
        replica_policy: str,
        nodes_resource_present: bool,
        nodes_reservation_enabled: bool,
        nodes_indices_raw: list[Any] | None,
        nodes_count_raw: Any | None,
        nodes_exclude_raw: Any | None,
        gpus_count_raw: Any | None,
    ) -> tuple[list[str], bool]:
        if runtime_backend.allocation is None:
            return [], False

        alloc_nodes = list(runtime_backend.allocation.nodes)
        if not alloc_nodes:
            return [], False

        def _reserve_nodes_if_needed(selected_node_names: list[str]) -> None:
            if not nodes_reservation_enabled:
                return
            for node_name in selected_node_names:
                self.node_planner.reserve(
                    backend_name=runtime_backend.name,
                    node_name=node_name,
                    task_name=task_name,
                )

        def _raise_if_nodes_unavailable(selected_node_names: list[str]) -> None:
            if not nodes_resource_present:
                return
            unavailable = [
                node_name
                for node_name in selected_node_names
                if self.node_planner.conflicting_reservations(
                    (runtime_backend.name, node_name),
                    task_name,
                )
            ]
            if unavailable:
                blockers = self.node_planner.format_blockers(
                    backend_name=runtime_backend.name,
                    candidate_nodes=selected_node_names,
                    task_name=task_name,
                )
                raise ValueError(
                    f"Task '{task_name}' requests node(s) {selected_node_names}, "
                    f"but {unavailable} do not remain available."
                    f"{blockers}"
                )

        if nodes_exclude_raw is not None:
            raw = (
                nodes_exclude_raw
                if isinstance(nodes_exclude_raw, list)
                or self.resolver.has_expression(nodes_exclude_raw)
                else [nodes_exclude_raw]
            )
            n = len(alloc_nodes)
            resolved_exclude: set[int] = set()
            for idx in self._resolve_int_list(
                task_name,
                field="resources.nodes.exclude",
                values=raw,
            ):
                resolved_idx = idx if idx >= 0 else idx + n
                if resolved_idx < 0 or resolved_idx >= n:
                    raise ValueError(
                        f"Task '{task_name}' resources.nodes.exclude contains index {idx} "
                        f"out of range for {n} allocated node(s) "
                        f"(valid: {-n}..{n - 1})"
                    )
                resolved_exclude.add(resolved_idx)
            alloc_nodes = [
                node for i, node in enumerate(alloc_nodes) if i not in resolved_exclude
            ]
            if not alloc_nodes:
                raise ValueError(
                    f"Task '{task_name}' resources.nodes.exclude removed all nodes from the pool"
                )

        selected_nodes = alloc_nodes
        if nodes_indices_raw is not None:
            n = len(alloc_nodes)
            chosen_nodes: list[ComputeNode] = []
            for idx in self._resolve_int_list(
                task_name,
                field="resources.nodes.indices",
                values=nodes_indices_raw,
            ):
                resolved_idx = idx if idx >= 0 else idx + n
                if resolved_idx < 0 or resolved_idx >= n:
                    raise ValueError(
                        f"Task '{task_name}' resources.nodes.indices contains out-of-range index {idx}; "
                        f"allocation has {n} nodes (valid: {-n}..{n - 1})"
                    )
                chosen_nodes.append(alloc_nodes[resolved_idx])
            selected_nodes = chosen_nodes
            if nodes_count_raw is None:
                selected_node_names = [node.name for node in selected_nodes]
                _raise_if_nodes_unavailable(selected_node_names)
                _reserve_nodes_if_needed(selected_node_names)
                return selected_node_names, False

        if nodes_count_raw is not None:
            count = self._resolve_int(
                task_name,
                field="resources.nodes.count",
                value=nodes_count_raw,
            )
            if count <= 0:
                raise ValueError(
                    f"Task '{task_name}' resources.nodes.count must be > 0, got {count}"
                )
            candidate_node_names = [node.name for node in selected_nodes]
            has_node_reservation_conflicts = any(
                self.node_planner.conflicting_reservations(
                    (runtime_backend.name, node_name),
                    task_name,
                )
                for node_name in candidate_node_names
            )

            if nodes_reservation_enabled or has_node_reservation_conflicts:
                selected_node_names = self.node_planner.first_available_nodes(
                    backend_name=runtime_backend.name,
                    task_name=task_name,
                    candidate_nodes=candidate_node_names,
                    count=count,
                )
                if selected_node_names is None:
                    available = [
                        node_name
                        for node_name in candidate_node_names
                        if not self.node_planner.conflicting_reservations(
                            (runtime_backend.name, node_name),
                            task_name,
                        )
                    ]
                    blockers = self.node_planner.format_blockers(
                        backend_name=runtime_backend.name,
                        candidate_nodes=candidate_node_names,
                        task_name=task_name,
                    )
                    raise ValueError(
                        f"Task '{task_name}' needs {count} node(s) "
                        f"(replica_index={replica_index}, policy={replica_policy}), "
                        f"but only {len(available)} node(s) remain available from "
                        f"{len(candidate_node_names)} candidate node(s)."
                        f"{blockers}"
                    )
            else:
                start = 0 if replica_policy == "sequential" else replica_index * count
                end = start + count
                if end > len(selected_nodes):
                    raise ValueError(
                        f"Task '{task_name}' needs {count} nodes (replica_index={replica_index}, policy={replica_policy}), "
                        f"but allocation has only {len(selected_nodes)} nodes"
                    )
                selected_node_names = [node.name for node in selected_nodes[start:end]]

            _reserve_nodes_if_needed(selected_node_names)
            return selected_node_names, False

        if gpus_count_raw is not None and runtime_backend.allocation is not None:
            gpus_needed = self._resolve_int(
                task_name,
                field="resources.gpus.count",
                value=gpus_count_raw,
            )
            if gpus_needed <= 0:
                raise ValueError(
                    f"Task '{task_name}' resources.gpus.count must be > 0, got {gpus_needed}"
                )

            alloc_nodes_by_name = {n.name: n for n in runtime_backend.allocation.nodes}
            for node in alloc_nodes:
                n = alloc_nodes_by_name.get(node.name)
                if n is None or getattr(n, "num_gpus", None) is None:
                    continue
                try:
                    cap = int(n.num_gpus)
                except Exception:
                    continue
                if cap <= 0:
                    continue

                start = self.gpu_planner.first_available_start(
                    backend_name=runtime_backend.name,
                    node_name=node.name,
                    task_name=task_name,
                    count=gpus_needed,
                    capacity=cap,
                )
                if start is not None:
                    node_names = [node.name]
                    _raise_if_nodes_unavailable(node_names)
                    _reserve_nodes_if_needed(node_names)
                    return node_names, True

        if gpus_count_raw is not None and runtime_backend.allocation is not None:
            alloc_nodes_by_name = list(runtime_backend.allocation.nodes)
            caps = [
                int(n.num_gpus)
                for n in alloc_nodes_by_name
                if getattr(n, "num_gpus", None) is not None
            ]
            if caps:
                gpus_needed = self._resolve_int(
                    task_name,
                    field="resources.gpus.count",
                    value=gpus_count_raw,
                )
                per_node_cap = min(caps)
                if per_node_cap > 0 and gpus_needed > per_node_cap:
                    if gpus_needed % per_node_cap != 0:
                        raise ValueError(
                            f"Task '{task_name}' requests {gpus_needed} GPUs; automatic multi-node expansion requires "
                            f"gpus.count to be a multiple of per-node GPU capacity ({per_node_cap}). "
                            f"Set resources.nodes.count/indices explicitly to override."
                        )
                    nodes_needed = math.ceil(gpus_needed / per_node_cap)
                    alloc_nodes_map = {n.name: n for n in alloc_nodes_by_name}
                    candidates: list[tuple[str, int, int]] = []
                    for node in alloc_nodes:
                        n = alloc_nodes_map.get(node.name)
                        if n is None or getattr(n, "num_gpus", None) is None:
                            continue
                        try:
                            cap = int(n.num_gpus)
                        except Exception:
                            continue
                        if cap <= 0:
                            continue
                        start = self.gpu_planner.first_available_start(
                            backend_name=runtime_backend.name,
                            node_name=node.name,
                            task_name=task_name,
                            count=per_node_cap,
                            capacity=cap,
                        )
                        if start is not None:
                            candidates.append((node.name, cap, start))

                    if candidates:
                        for cursor in sorted({c for _, _, c in candidates}):
                            names = [
                                n for (n, _cap, cur) in candidates if cur == cursor
                            ]
                            if len(names) >= nodes_needed:
                                node_names = names[:nodes_needed]
                                _raise_if_nodes_unavailable(node_names)
                                _reserve_nodes_if_needed(node_names)
                                return node_names, True

                    start = 0 if replica_policy == "sequential" else replica_index * nodes_needed
                    end = start + nodes_needed
                    if end > len(alloc_nodes):
                        raise ValueError(
                            f"Task '{task_name}' requests {gpus_needed} GPUs which requires {nodes_needed} nodes "
                            f"(replica_index={replica_index}, policy={replica_policy}), "
                            f"but allocation has only {len(alloc_nodes)} nodes"
                        )
                    node_names = [n.name for n in alloc_nodes[start:end]]
                    _raise_if_nodes_unavailable(node_names)
                    _reserve_nodes_if_needed(node_names)
                    return node_names, True

        default_nodes = self.default_task_nodes.get((runtime_backend.name, base_task_name))
        if default_nodes is not None:
            node_names = list(default_nodes)
            _raise_if_nodes_unavailable(node_names)
            _reserve_nodes_if_needed(node_names)
            return node_names, False

        node_names = [n.name for n in alloc_nodes]
        _raise_if_nodes_unavailable(node_names)
        _reserve_nodes_if_needed(node_names)
        return node_names, False

    def _cuda_visible_devices(
        self,
        *,
        task_name: str,
        runtime_backend: Backend,
        assigned_nodes: list[str],
        replica_index: int,
        gpus_count_raw: Any | None,
    ) -> str | None:
        if gpus_count_raw is None:
            return None
        count = self._resolve_int(
            task_name,
            field="resources.gpus.count",
            value=gpus_count_raw,
        )
        if count <= 0:
            raise ValueError(
                f"Task '{task_name}' resources.gpus.count must be > 0, got {count}"
            )

        def _backend_gpu_state_summary() -> str:
            if not runtime_backend.allocation:
                return ""
            gpu_nodes: list[tuple[str, int]] = []
            for node in runtime_backend.allocation.nodes:
                num_gpus = getattr(node, "num_gpus", None)
                if num_gpus is None:
                    continue
                try:
                    gpu_nodes.append((node.name, int(num_gpus)))
                except Exception:
                    continue
            if not gpu_nodes:
                return ""

            caps = [cap for _name, cap in gpu_nodes]
            total_capacity = sum(caps)
            total_allocated = sum(
                min(
                    max(
                        (
                            end
                            for _start, end, _owner, _release_after in self.gpu_planner.conflicting_reservations(
                                (runtime_backend.name, n_name), task_name
                            )
                        ),
                        default=0,
                    ),
                    cap,
                )
                for n_name, cap in gpu_nodes
            )
            total_remaining = total_capacity - total_allocated
            per_node_str = (
                f"gpus_per_node={caps[0]}"
                if len(set(caps)) == 1
                else f"per_node_capacities={caps}"
            )
            return (
                f"backend_gpu_state=(nodes={len(gpu_nodes)}, {per_node_str}, "
                f"total_capacity={total_capacity}, already_allocated={total_allocated}, "
                f"remaining={total_remaining})"
            )

        if runtime_backend.allocation and assigned_nodes and len(assigned_nodes) == 1:
            n_name = assigned_nodes[0]
            alloc_nodes_by_name = {n.name: n for n in runtime_backend.allocation.nodes}
            n = alloc_nodes_by_name.get(n_name)
            if n is not None and getattr(n, "num_gpus", None) is not None:
                cap = int(n.num_gpus)
                if cap <= 0:
                    raise ValueError(
                        f"Task '{task_name}' cannot allocate GPUs on node '{n_name}' with non-positive capacity {cap}"
                    )

                start = self.gpu_planner.first_available_start(
                    backend_name=runtime_backend.name,
                    node_name=n_name,
                    task_name=task_name,
                    count=count,
                    capacity=cap,
                )
                if start is None:
                    occupied_until = max(
                        (
                            end
                            for _start, end, _owner, _release_after in self.gpu_planner.conflicting_reservations(
                                (runtime_backend.name, n_name), task_name
                            )
                        ),
                        default=0,
                    )
                    available = cap - occupied_until
                    still_needed = count - available
                    backend_gpu_state = _backend_gpu_state_summary()
                    blockers = self.gpu_planner.format_blockers(
                        backend_name=runtime_backend.name,
                        node_name=n_name,
                        task_name=task_name,
                    )
                    raise ValueError(
                        f"Task '{task_name}' requests {count} GPUs on node '{n_name}', but only {available} GPUs "
                        f"remain available (total_capacity={cap}, already_allocated={occupied_until}, "
                        f"still_needed={still_needed})"
                        f"{', ' + backend_gpu_state if backend_gpu_state else ''}."
                        f"{blockers}"
                        f"Consider increasing backend nodes or reducing concurrent GPU requests."
                    )

                slice_str = ",".join(str(i) for i in range(start, start + count))
                self.gpu_planner.reserve(
                    backend_name=runtime_backend.name,
                    node_name=n_name,
                    task_name=task_name,
                    start=start,
                    count=count,
                )
                return slice_str

        if runtime_backend.allocation and assigned_nodes and len(assigned_nodes) > 1:
            alloc_nodes_by_name = {n.name: n for n in runtime_backend.allocation.nodes}
            caps: list[int] = []
            for n_name in assigned_nodes:
                n = alloc_nodes_by_name.get(n_name)
                if n is None or getattr(n, "num_gpus", None) is None:
                    caps = []
                    break
                caps.append(int(n.num_gpus))

            if caps:
                total_cap = sum(caps)
                if count > total_cap:
                    backend_gpu_state = _backend_gpu_state_summary()
                    raise ValueError(
                        f"Task '{task_name}' requests {count} GPUs but assigned nodes have only {total_cap} GPUs total"
                        f"{', ' + backend_gpu_state if backend_gpu_state else ''}"
                    )
                per_node = min(min(caps), math.ceil(count / len(assigned_nodes)))
                if per_node <= 0:
                    raise ValueError(
                        f"Task '{task_name}' resources.gpus.count must be > 0, got {count}"
                    )

                starts: list[int | None] = [
                    self.gpu_planner.first_available_start(
                        backend_name=runtime_backend.name,
                        node_name=n_name,
                        task_name=task_name,
                        count=per_node,
                        capacity=cap,
                    )
                    for n_name, cap in zip(assigned_nodes, caps, strict=True)
                ]
                if any(start is None for start in starts):
                    backend_gpu_state = _backend_gpu_state_summary()
                    blocked_node = next(
                        n_name
                        for n_name, start in zip(assigned_nodes, starts, strict=True)
                        if start is None
                    )
                    blockers = self.gpu_planner.format_blockers(
                        backend_name=runtime_backend.name,
                        node_name=blocked_node,
                        task_name=task_name,
                    )
                    raise ValueError(
                        f"Task '{task_name}' requests {per_node} GPUs per node across {assigned_nodes}, "
                        f"but at least one node lacks enough non-conflicting GPUs"
                        f"{', ' + backend_gpu_state if backend_gpu_state else ''}."
                        f"{blockers if blockers else ''}"
                    )
                if len(set(starts)) != 1:
                    raise ValueError(
                        f"Task '{task_name}' requests GPUs across multiple nodes, but the nodes have different "
                        f"already-allocated GPU cursors {starts}. Pin nodes explicitly to avoid ambiguity."
                    )
                start0 = int(starts[0])
                for n_name, cap in zip(assigned_nodes, caps, strict=True):
                    if start0 + per_node > cap:
                        available = cap - start0
                        still_needed = per_node - available
                        backend_gpu_state = _backend_gpu_state_summary()
                        raise ValueError(
                            f"Task '{task_name}' requests {per_node} GPUs per node starting at {start0} on node "
                            f"'{n_name}', but only {available} GPUs remain available "
                            f"(total_capacity={cap}, already_allocated={start0}, still_needed={still_needed})"
                            f"{', ' + backend_gpu_state if backend_gpu_state else ''}."
                        )
                for n_name in assigned_nodes:
                    self.gpu_planner.reserve(
                        backend_name=runtime_backend.name,
                        node_name=n_name,
                        task_name=task_name,
                        start=start0,
                        count=per_node,
                    )
                return ",".join(str(i) for i in range(start0, start0 + per_node))

        start = replica_index * count
        if runtime_backend.allocation and assigned_nodes:
            alloc_nodes_by_name = {n.name: n for n in runtime_backend.allocation.nodes}
            for n_name in assigned_nodes:
                n = alloc_nodes_by_name.get(n_name)
                if n is None or getattr(n, "num_gpus", None) is None:
                    continue
                if start + count > int(n.num_gpus):
                    backend_gpu_state = _backend_gpu_state_summary()
                    raise ValueError(
                        f"Task '{task_name}' requests GPUs [{start}..{start + count - 1}] "
                        f"but node '{n_name}' has only {n.num_gpus} GPUs"
                        f"{', ' + backend_gpu_state if backend_gpu_state else ''}."
                    )

        return ",".join(str(i) for i in range(start, start + count))


def plan_resource_placements(
    config: SflowConfig,
    state: SflowState,
    *,
    resolver: Any,
    ctx: dict[str, Any],
    replica_names_by_base: dict[str, list[str]],
    replica_policy_by_base: dict[str, str],
) -> dict[str, ResourcePlacement]:
    """Return final resource placement and conflicts for every concrete task."""
    return ResourcePlacementPlanner(
        config,
        state,
        resolver=resolver,
        ctx=ctx,
        replica_names_by_base=replica_names_by_base,
        replica_policy_by_base=replica_policy_by_base,
    ).plan()
