#!/usr/bin/env python3
"""Compute and display the GPU allocation plan from sflow YAML configs.

Usage:
    python check_gpu_plan.py <yaml_file> [<yaml_file> ...]

Shows:
    - Per-task GPU allocation (count, replicas, total)
    - Backend capacity (nodes * gpus_per_node)
    - Oversubscription warnings
    - Node-pinned tasks vs floating tasks
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

EXPRESSION_PATTERN = re.compile(r"\$\{\{(.+?)\}\}", re.DOTALL)


def _load_and_merge(filepaths: list[str]) -> dict:
    """Load one or more YAML files and merge them (simplified merge)."""
    merged: dict = {}
    for fp in filepaths:
        path = Path(fp)
        if not path.exists():
            print(f"Warning: file not found: {fp}", file=sys.stderr)
            continue
        with open(fp) as f:
            config = yaml.safe_load(f.read())
        if not isinstance(config, dict):
            continue
        if not merged:
            merged = config
            continue
        for key in ("variables", "artifacts", "backends", "operators"):
            existing = merged.get(key, {} if key == "variables" else [])
            incoming = config.get(key, {} if key == "variables" else [])
            if isinstance(existing, dict) and isinstance(incoming, dict):
                existing.update(incoming)
                merged[key] = existing
            elif isinstance(existing, list) and isinstance(incoming, list):
                names_seen = {
                    item["name"]
                    for item in existing
                    if isinstance(item, dict) and "name" in item
                }
                for item in incoming:
                    if isinstance(item, dict) and item.get("name") in names_seen:
                        existing = [
                            item
                            if isinstance(e, dict) and e.get("name") == item.get("name")
                            else e
                            for e in existing
                        ]
                    else:
                        existing.append(item)
                merged[key] = existing
        if "workflow" in config:
            if "workflow" not in merged:
                merged["workflow"] = config["workflow"]
            else:
                wf = merged["workflow"]
                inc_wf = config["workflow"]
                if isinstance(inc_wf, dict):
                    wf_tasks = wf.get("tasks", [])
                    inc_tasks = inc_wf.get("tasks", [])
                    if isinstance(wf_tasks, list) and isinstance(inc_tasks, list):
                        wf["tasks"] = wf_tasks + inc_tasks
                    wf_vars = wf.get("variables", {})
                    inc_vars = inc_wf.get("variables", {})
                    if isinstance(wf_vars, dict) and isinstance(inc_vars, dict):
                        wf_vars.update(inc_vars)
                        wf["variables"] = wf_vars
    return merged


def _resolve_value(val, variables: dict) -> int | str | None:
    """Attempt to resolve a value that may contain a simple expression."""
    if isinstance(val, (int, float)):
        return int(val)
    if not isinstance(val, str):
        return None
    if "${{" not in val:
        try:
            return int(val)
        except (ValueError, TypeError):
            return val

    inner = val.strip()
    m = re.match(r"^\$\{\{\s*(.+?)\s*\}\}$", inner)
    if not m:
        return None
    expr = m.group(1)

    var_pattern = re.compile(r"variables\.(\w+)")
    resolved_expr = expr
    for vm in var_pattern.finditer(expr):
        vname = vm.group(1)
        vdef = variables.get(vname)
        if isinstance(vdef, dict):
            vval = vdef.get("value")
        else:
            vval = vdef
        if vval is None or (isinstance(vval, str) and "${{" in vval):
            return None
        resolved_expr = resolved_expr.replace(f"variables.{vname}", str(vval))

    try:
        result = eval(resolved_expr, {"__builtins__": {}}, {})  # noqa: S307
        return int(result)
    except Exception:
        return None


def _get_backend_capacity(config: dict, variables: dict) -> list[dict]:
    """Extract backend node/GPU capacity."""
    backends = config.get("backends", [])
    if not isinstance(backends, list):
        return []
    results = []
    for b in backends:
        if not isinstance(b, dict):
            continue
        name = b.get("name", "<unnamed>")
        btype = b.get("type", "local")
        nodes = _resolve_value(b.get("nodes", 1), variables)
        gpn = _resolve_value(b.get("gpus_per_node", 0), variables)
        is_default = b.get("default", False)
        results.append(
            {
                "name": name,
                "type": btype,
                "nodes": nodes,
                "gpus_per_node": gpn,
                "total_gpus": (nodes or 0) * (gpn or 0) if nodes and gpn else None,
                "default": is_default,
            }
        )
    return results


def _get_task_gpu_plan(config: dict, variables: dict) -> list[dict]:
    """Extract per-task GPU allocation plan."""
    wf = config.get("workflow", {})
    tasks = wf.get("tasks", []) if isinstance(wf, dict) else []
    if not isinstance(tasks, list):
        return []

    wf_vars = wf.get("variables", {}) if isinstance(wf, dict) else {}
    all_vars = dict(variables)
    if isinstance(wf_vars, dict):
        all_vars.update(wf_vars)

    results = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        name = task.get("name", "<unnamed>")
        resources = task.get("resources", {})
        replicas = task.get("replicas", {})

        gpu_count: int | None = 0
        gpu_unresolved = False
        node_indices = None
        node_count = None

        if isinstance(resources, dict):
            gpus = resources.get("gpus", {})
            if isinstance(gpus, dict):
                raw_gc = gpus.get("count", 0)
                gc = _resolve_value(raw_gc, all_vars)
                if gc is not None:
                    gpu_count = gc
                elif raw_gc != 0:
                    gpu_count = 0
                    gpu_unresolved = True
            nodes = resources.get("nodes", {})
            if isinstance(nodes, dict):
                idx = nodes.get("indices")
                if isinstance(idx, list):
                    node_indices = idx
                nc = nodes.get("count")
                if nc is not None:
                    node_count = _resolve_value(nc, all_vars)

        replica_count = 1
        sweep_vars = []
        policy = "parallel"
        if isinstance(replicas, dict):
            rc = _resolve_value(replicas.get("count", 1), all_vars)
            replica_count = rc if rc is not None else 1
            policy = replicas.get("policy", "parallel")
            sv = replicas.get("variables", [])
            if isinstance(sv, list):
                sweep_vars = sv
                if sweep_vars and replica_count == 1:
                    domain_size = 1
                    for svname in sweep_vars:
                        vdef = all_vars.get(svname, {})
                        if isinstance(vdef, dict):
                            domain = vdef.get("domain", [])
                            if isinstance(domain, list) and domain:
                                domain_size *= len(domain)
                    replica_count = domain_size

        total_gpus = gpu_count * replica_count

        results.append(
            {
                "name": name,
                "gpus_per_replica": gpu_count,
                "gpu_unresolved": gpu_unresolved,
                "replicas": replica_count,
                "policy": policy,
                "total_gpus": total_gpus,
                "node_indices": node_indices,
                "node_count": node_count,
                "sweep_vars": sweep_vars,
            }
        )

    return results


def print_plan(filepaths: list[str]) -> int:
    config = _load_and_merge(filepaths)
    if not config:
        print("Error: no valid config loaded", file=sys.stderr)
        return 1

    variables = config.get("variables", {})
    if not isinstance(variables, dict):
        variables = {}

    backends = _get_backend_capacity(config, variables)
    tasks = _get_task_gpu_plan(config, variables)

    print(f"\n{'=' * 70}")
    print("  GPU Allocation Plan")
    print(f"  Files: {', '.join(filepaths)}")
    print(f"{'=' * 70}")

    if backends:
        print("\n  Backends:")
        print(
            f"  {'Name':<20} {'Type':<8} {'Nodes':<7} {'GPUs/Node':<10} {'Total GPUs':<10} {'Default'}"
        )
        print(f"  {'-' * 20} {'-' * 8} {'-' * 7} {'-' * 10} {'-' * 10} {'-' * 7}")
        for b in backends:
            nodes_str = str(b["nodes"]) if b["nodes"] is not None else "?"
            gpn_str = str(b["gpus_per_node"]) if b["gpus_per_node"] is not None else "?"
            total_str = str(b["total_gpus"]) if b["total_gpus"] is not None else "?"
            default_str = "yes" if b["default"] else ""
            print(
                f"  {b['name']:<20} {b['type']:<8} {nodes_str:<7} {gpn_str:<10} {total_str:<10} {default_str}"
            )

    if tasks:
        print("\n  Tasks:")
        print(
            f"  {'Name':<30} {'GPUs':<6} {'Replicas':<10} {'Total GPUs':<11} {'Policy':<12} {'Pinned'}"
        )
        print(f"  {'-' * 30} {'-' * 6} {'-' * 10} {'-' * 11} {'-' * 12} {'-' * 10}")

        total_gpu_demand = 0
        has_unresolved = False
        pinned_tasks = []
        floating_tasks = []

        for t in tasks:
            pin = ""
            if t["node_indices"] is not None:
                pin = f"nodes {t['node_indices']}"
                pinned_tasks.append(t)
            elif t["node_count"] is not None:
                pin = f"{t['node_count']} node(s)"
            if t["total_gpus"] > 0:
                floating_tasks.append(t)

            policy_str = t["policy"] if t["replicas"] > 1 else ""
            gpu_str = "?" if t["gpu_unresolved"] else str(t["gpus_per_replica"])
            total_str = "?" if t["gpu_unresolved"] else str(t["total_gpus"])
            print(
                f"  {t['name']:<30} {gpu_str:<6} "
                f"{t['replicas']:<10} {total_str:<11} "
                f"{policy_str:<12} {pin}"
            )

            if not t["gpu_unresolved"]:
                if t["policy"] == "parallel" or t["replicas"] == 1:
                    total_gpu_demand += t["total_gpus"]
                else:
                    total_gpu_demand += t["gpus_per_replica"]
            else:
                has_unresolved = True

        demand_str = f"{total_gpu_demand}+" if has_unresolved else str(total_gpu_demand)
        print(f"\n  Peak concurrent GPU demand: {demand_str}")
        if has_unresolved:
            print(
                "  Note: some GPU counts contain unresolved expressions (shown as '?')"
            )
            print("        Use sflow compose --resolve to get exact values")

        default_backend = next(
            (b for b in backends if b["default"]), backends[0] if backends else None
        )
        if default_backend and default_backend["total_gpus"] is not None:
            capacity = default_backend["total_gpus"]
            print(
                f"  Backend capacity:           {capacity} ({default_backend['name']})"
            )
            if total_gpu_demand > capacity:
                print(
                    f"\n  *** WARNING: GPU demand ({total_gpu_demand}) exceeds capacity ({capacity})! ***"
                )
                print("  *** Increase nodes or reduce GPU counts / replicas. ***")
            elif total_gpu_demand == capacity:
                print("  Status: Fully utilized (demand == capacity)")
            else:
                headroom = capacity - total_gpu_demand
                print(f"  Status: OK ({headroom} GPU(s) headroom)")

    print()
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <yaml_file> [<yaml_file> ...]", file=sys.stderr)
        return 2
    return print_plan(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
