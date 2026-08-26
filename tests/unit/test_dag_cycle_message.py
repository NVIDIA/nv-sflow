# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a cycle error tells the person who has to fix it.

A workflow is routinely merged from half a dozen files, each contributing a task or
two, so a dependency loop appears without any single file looking wrong. "Graph
contains a cycle" then leaves the reader to rebuild the loop by hand from the composed
config. These tests pin the parts that make the message actionable: which tasks are in
the loop, which direction each edge runs, and which tasks are merely stuck behind it.
"""

import pytest

from sflow.core.dag import DAG


def build(edges, extra_nodes=()):
    """A DAG from ``(dependency, dependent)`` pairs -- the order add_edge takes."""
    dag = DAG(name="test")
    for node in extra_nodes:
        dag.add_node(node)
    for dependency, dependent in edges:
        dag.add_edge(dependency, dependent)
    return dag


# ----------------------------------------------------------------------------------
# find_cycle
# ----------------------------------------------------------------------------------


def test_an_acyclic_graph_has_no_cycle():
    assert build([("a", "b"), ("b", "c")]).find_cycle() == []
    assert build([("a", "b"), ("b", "c")]).topological_sort() == ["a", "b", "c"]


def test_a_cycle_is_returned_in_edge_order():
    cycle = build([("a", "b"), ("b", "c"), ("c", "a")]).find_cycle()

    assert sorted(cycle) == ["a", "b", "c"]
    # [x, y, z] means x -> y -> z -> x, so each node is a dependency of the next.
    assert cycle[1] in build([("a", "b"), ("b", "c"), ("c", "a")]).edges[cycle[0]]


def test_a_task_that_depends_on_itself_is_a_cycle_of_one():
    assert build([("a", "a")]).find_cycle() == ["a"]


def test_the_cycle_is_found_even_when_reached_through_clean_tasks():
    """The loop rarely sits at the root; it is reached from tasks that are fine."""
    dag = build([("root", "a"), ("a", "b"), ("b", "c"), ("c", "a")])

    assert sorted(dag.find_cycle()) == ["a", "b", "c"]


def test_a_long_chain_does_not_blow_the_stack():
    """Replica fan-out makes these chains long; a RecursionError while reporting a
    cycle would replace one confusing error with a worse one."""
    chain = [(f"task_{i}", f"task_{i + 1}") for i in range(2000)]

    assert build(chain).find_cycle() == []


# ----------------------------------------------------------------------------------
# The message
# ----------------------------------------------------------------------------------


def test_the_message_names_every_edge_of_the_loop():
    """This is the real shape from a disaggregated serving recipe: the load balancer
    was ordered before the frontends while also waiting on the workers, which wait on
    the frontends."""
    dag = build(
        [
            ("nginx_server", "frontend_server_0"),
            ("frontend_server_0", "prefill_server_0"),
            ("prefill_server_0", "nginx_server"),
        ]
    )

    with pytest.raises(ValueError) as raised:
        dag.topological_sort()

    message = str(raised.value)
    assert "3 task(s) depend on each other in a loop" in message
    assert "frontend_server_0 depends on nginx_server" in message
    assert "prefill_server_0 depends on frontend_server_0" in message
    assert "nginx_server depends on prefill_server_0" in message
    assert "Remove one of those 'depends_on' entries" in message


def test_tasks_stuck_behind_the_loop_are_listed_apart_from_it():
    """They are not the bug. Folded in with the loop, every task downstream of it
    reads like a participant."""
    dag = build([("a", "b"), ("b", "a"), ("b", "benchmark"), ("benchmark", "report")])

    with pytest.raises(ValueError) as raised:
        dag.topological_sort()

    message = str(raised.value)
    loop_lines = [line for line in message.splitlines() if " depends on " in line]
    assert sorted(loop_lines) == ["  a depends on b", "  b depends on a"]
    assert "Waiting behind it: benchmark, report" in message


def test_a_long_blocked_list_is_capped():
    edges = [("a", "b"), ("b", "a")] + [("b", f"task_{i:02d}") for i in range(12)]

    with pytest.raises(ValueError) as raised:
        build(edges).topological_sort()

    message = str(raised.value)
    assert "and 4 more" in message
    assert "task_11" not in message


def test_tasks_outside_the_loop_are_still_reported_as_resolvable():
    """A clean task that never touches the loop must not appear anywhere in the error."""
    dag = build([("a", "b"), ("b", "a")], extra_nodes=["unrelated"])

    with pytest.raises(ValueError) as raised:
        dag.topological_sort()

    assert "unrelated" not in str(raised.value)


def test_has_cycle_still_answers_with_a_bool():
    """Its contract is the exception *type*, which the richer message must not change."""
    assert build([("a", "b"), ("b", "a")]).has_cycle() is True
    assert build([("a", "b")]).has_cycle() is False
