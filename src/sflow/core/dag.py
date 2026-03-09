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
