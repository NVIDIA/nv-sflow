# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any


class DAG:
    def __init__(self, name: str | None = None):
        self.nodes = {}
        self.edges = {}
        self.in_degree = {}
        self.out_degree = {}
        self.name = name

    def __getitem__(self, node_id: str) -> Any:
        """Allow [] access to node data by id."""
        return self.nodes[node_id]

    def __setitem__(self, node_id: str, node_data: Any) -> None:
        """Allow [] assignment to node data by id."""
        if node_id in self.nodes:
            self.nodes[node_id] = node_data
        else:
            self.add_node(node_id, node_data)

    def add_node(self, node_id: str, node_data: Any = None):
        """Add a node to the DAG."""
        if node_id not in self.nodes:
            self.nodes[node_id] = node_data
            self.edges[node_id] = {}
            self.in_degree[node_id] = 0
            self.out_degree[node_id] = 0

    def add_edge(self, from_node: str, to_node: str, data: Any = None):
        """Add a directed edge from from_node to to_node with optional data."""
        if from_node not in self.nodes:
            self.add_node(from_node)
        if to_node not in self.nodes:
            self.add_node(to_node)

        if to_node not in self.edges[from_node]:
            self.out_degree[from_node] += 1
            self.in_degree[to_node] += 1

        self.edges[from_node][to_node] = data

    def get_edge_data(self, from_node: str, to_node: str) -> Any:
        """Get data associated with an edge."""
        if from_node in self.edges and to_node in self.edges[from_node]:
            return self.edges[from_node][to_node]
        return None

    def get_dependencies(self, node_id: str) -> list[str]:
        """Get all nodes that the given node depends on."""
        dependencies = []
        for node, edges in self.edges.items():
            if node_id in edges:
                dependencies.append(node)
        return dependencies

    def get_dependents(self, node_id: str) -> list[str]:
        """Get all nodes that depend on the given node."""
        return list(self.edges.get(node_id, {}).keys())

    def topological_sort(self) -> list[str]:
        """Return a topological ordering of the nodes."""
        in_degree_copy = self.in_degree.copy()
        queue = [node for node, degree in in_degree_copy.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for dependent in self.edges[node]:
                in_degree_copy[dependent] -= 1
                if in_degree_copy[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def has_cycle(self) -> bool:
        """Check if the DAG contains a cycle."""
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    def render_ascii(self, max_width: int | None = None) -> list[str]:
        """Render the DAG as an ASCII art graph for terminal display.

        Uses a layered layout (Sugiyama-style) where nodes are placed at
        levels determined by the longest path from root nodes. Connections
        are drawn using Unicode box-drawing characters.

        When the graph is too wide for horizontal box layout, falls back to
        a compact vertical format that lists nodes within each layer.
        """
        from collections import defaultdict

        if max_width is None:
            try:
                import shutil

                max_width = max(shutil.get_terminal_size((120, 24)).columns - 32, 60)
            except Exception:
                max_width = 88

        try:
            order = self.topological_sort()
        except ValueError:
            return ["(graph contains a cycle — cannot render)"]

        if not order:
            return ["(empty graph)"]

        node_layer: dict[str, int] = {}
        for node in order:
            deps = self.get_dependencies(node)
            node_layer[node] = 0 if not deps else max(node_layer[d] for d in deps) + 1

        layer_nodes: dict[int, list[str]] = defaultdict(list)
        for node in order:
            layer_nodes[node_layer[node]].append(node)

        num_layers = max(node_layer.values()) + 1

        max_name_len = max(len(n) for n in order)
        box_inner = max(max_name_len, 4)
        box_w = box_inner + 4  # "│ " + name + " │"
        h_gap = 3
        max_per_layer = max(len(layer_nodes[layer]) for layer in range(num_layers))
        canvas_w = max(max_per_layer * box_w + (max_per_layer - 1) * h_gap, box_w)

        if canvas_w > max_width:
            return self._render_compact(layer_nodes, num_layers, max_width)

        node_cx: dict[str, int] = {}
        for level in range(num_layers):
            nodes = layer_nodes[level]
            n = len(nodes)
            layer_w = n * box_w + (n - 1) * h_gap
            offset = (canvas_w - layer_w) // 2
            for i, node in enumerate(nodes):
                node_cx[node] = offset + i * (box_w + h_gap) + box_w // 2

        # Reorder nodes within each layer by average parent position to reduce edge crossings
        for level in range(1, num_layers):
            nodes = layer_nodes[level]
            if len(nodes) <= 1:
                continue

            def _parent_center(nd: str) -> float:
                parents = self.get_dependencies(nd)
                positions = [node_cx[p] for p in parents if p in node_cx]
                return sum(positions) / len(positions) if positions else 0.0

            nodes.sort(key=_parent_center)
            layer_nodes[level] = nodes
            n = len(nodes)
            layer_w = n * box_w + (n - 1) * h_gap
            offset = (canvas_w - layer_w) // 2
            for i, node in enumerate(nodes):
                node_cx[node] = offset + i * (box_w + h_gap) + box_w // 2

        all_edges = [(fn, tn) for fn in order for tn in self.get_dependents(fn)]

        _CHAR = {
            (True, True, True, True): "┼",
            (True, True, True, False): "┤",
            (True, True, False, True): "├",
            (True, True, False, False): "│",
            (True, False, True, True): "┴",
            (True, False, True, False): "┘",
            (True, False, False, True): "└",
            (True, False, False, False): "│",
            (False, True, True, True): "┬",
            (False, True, True, False): "┐",
            (False, True, False, True): "┌",
            (False, True, False, False): "│",
            (False, False, True, True): "─",
            (False, False, True, False): "─",
            (False, False, False, True): "─",
            (False, False, False, False): " ",
        }

        def _new_row() -> list[str]:
            return [" "] * canvas_w

        def _put(row: list[str], col: int, s: str) -> None:
            for i, ch in enumerate(s):
                if 0 <= col + i < len(row):
                    row[col + i] = ch

        def _passthrough_xs(layer: int) -> set[int]:
            pts: set[int] = set()
            for fn, tn in all_edges:
                if node_layer[fn] < layer < node_layer[tn]:
                    pts.add(node_cx[tn])
            return pts

        def _box_ranges(layer: int) -> list[tuple[int, int]]:
            return [
                (node_cx[n] - box_w // 2, node_cx[n] - box_w // 2 + box_w)
                for n in layer_nodes[layer]
            ]

        def _overlaps_box(x: int, ranges: list[tuple[int, int]]) -> bool:
            return any(left <= x < right for left, right in ranges)

        rows: list[list[str]] = []

        for level in range(num_layers):
            passthrough = _passthrough_xs(level)
            box_ranges = _box_ranges(level)

            for row_type in ("top", "mid", "bot"):
                row = _new_row()
                for node in layer_nodes[level]:
                    cx = node_cx[node]
                    left = cx - box_w // 2
                    if row_type == "top":
                        _put(row, left, "┌" + "─" * (box_w - 2) + "┐")
                    elif row_type == "mid":
                        _put(row, left, "│" + node.center(box_w - 2) + "│")
                    else:
                        _put(row, left, "└" + "─" * (box_w - 2) + "┘")
                for pt_x in passthrough:
                    if not _overlaps_box(pt_x, box_ranges) and 0 <= pt_x < canvas_w:
                        row[pt_x] = "│"
                rows.append(row)

            if level >= num_layers - 1:
                continue

            conn_edges: set[tuple[int, int]] = set()
            for fn, tn in all_edges:
                fl, tl = node_layer[fn], node_layer[tn]
                if fl <= level and tl >= level + 1:
                    px = node_cx[fn] if fl == level else node_cx[tn]
                    conn_edges.add((px, node_cx[tn]))

            if not conn_edges:
                rows.append(_new_row())
                continue

            parent_set = {px for px, _ in conn_edges}
            child_set = {cx for _, cx in conn_edges}
            spans = [(min(px, cx), max(px, cx)) for px, cx in conn_edges]

            r1 = _new_row()
            for px in parent_set:
                if 0 <= px < canvas_w:
                    r1[px] = "│"

            r2 = _new_row()
            touched: set[int] = set()
            for mn, mx in spans:
                touched.update(range(mn, mx + 1))
            for x in sorted(touched):
                up = x in parent_set
                down = x in child_set
                left = any(mn < x for mn, mx in spans if mn <= x <= mx)
                right = any(mx > x for mn, mx in spans if mn <= x <= mx)
                r2[x] = _CHAR[(up, down, left, right)]

            r3 = _new_row()
            for cx in child_set:
                if 0 <= cx < canvas_w:
                    r3[cx] = "▼"

            rows.extend([r1, r2, r3])

        return ["".join(r).rstrip() for r in rows]

    def _render_compact(
        self,
        layer_nodes: dict[int, list[str]],
        num_layers: int,
        max_width: int,
    ) -> list[str]:
        """Compact vertical rendering for DAGs too wide for horizontal box layout.

        Each layer is rendered as a single box listing all nodes vertically.
        """
        max_content = 0
        for level in range(num_layers):
            for node in layer_nodes[level]:
                dependents = self.get_dependents(node)
                line_len = len(node) + (
                    len(f"  → {', '.join(dependents)}") if dependents else 0
                )
                max_content = max(max_content, line_len)
        header_len = len(f" Layer {num_layers - 1} ")
        inner_w = max(max_content + 2, header_len + 2)
        inner_w = min(inner_w, max_width - 2)
        box_w = inner_w + 2
        center = box_w // 2

        lines: list[str] = []
        for level in range(num_layers):
            nodes = layer_nodes[level]
            header = f" Layer {level} "
            right_fill = inner_w - len(header) - 1
            lines.append(f"┌─{header}{'─' * right_fill}┐")
            for node in nodes:
                dependents = self.get_dependents(node)
                if dependents:
                    dep_names = ", ".join(dependents)
                    suffix = f"  → {dep_names}"
                    available = inner_w - len(node) - 2
                    if len(suffix) > available:
                        suffix = suffix[: max(available - 1, 0)] + "…"
                    lines.append(f"│ {node}{suffix:<{inner_w - len(node) - 1}}│")
                else:
                    lines.append(f"│ {node:<{inner_w - 1}}│")
            if level < num_layers - 1:
                lines.append(f"└{'─' * (center - 1)}┬{'─' * (box_w - center - 1)}┘")
                lines.append(f"{' ' * center}│")
                lines.append(f"{' ' * center}▼")
            else:
                lines.append(f"└{'─' * inner_w}┘")
        return lines
