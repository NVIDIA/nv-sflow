# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from sflow.resolution import ExpressionResolver
from sflow.core.variable import VariableValue


@pytest.fixture
def resolver() -> ExpressionResolver:
    return ExpressionResolver()


def test_has_expression_detects_in_str_list_and_dict(resolver: ExpressionResolver):
    assert resolver.has_expression("nope") is False
    assert resolver.has_expression("${{ 1 + 1 }}") is True
    assert resolver.has_expression(["a", "${{ x }}"]) is True
    assert resolver.has_expression({"a": "b", "c": "${{ x }}"}) is True
    assert resolver.has_expression({"a": ["b", {"c": "d"}]}) is False


def test_validate_syntax_valid_and_invalid_with_locations(resolver: ExpressionResolver):
    ok = resolver.validate_syntax(
        {"a": "${{ 1 + 2 }}", "b": ["x", "${{ foo }}"]}, location="root"
    )
    assert ok.valid is True
    assert ok.errors == []

    bad = resolver.validate_syntax(
        {"a": "${{ 1 + }}", "b": ["x", "${{ foo }}"]}, location="root"
    )
    assert bad.valid is False
    # error should point to the exact nested location
    assert any(e.location == "root.a" for e in bad.errors)
    assert any("${{ 1 + }}" in e.expression for e in bad.errors)


def test_extract_references_returns_undeclared_names(resolver: ExpressionResolver):
    # jinja2's undeclared vars are top-level names (not dotted paths)
    refs = resolver.extract_references(
        {
            "a": "${{ foo + bar }}",
            "b": ["${{ variables.MY_VAR }}", "nope"],
        }
    )
    assert refs == {"foo", "bar", "variables"}


def test_resolve_string_strict_undefined_raises_value_error(
    resolver: ExpressionResolver,
):
    with pytest.raises(ValueError, match="Undefined variable"):
        resolver.resolve("${{ missing }}", context={})


def test_resolve_string_returns_rendered_value(resolver: ExpressionResolver):
    assert resolver.resolve("x${{ foo }}y", context={"foo": "Z"}) == "xZy"
    # jinja renders results as strings
    assert resolver.resolve("${{ 1 + 2 }}", context={}) == "3"


def test_resolve_recurses_lists_and_dicts(resolver: ExpressionResolver):
    value = {
        "a": ["${{ x }}", 1, {"b": "${{ y }}"}],
        "c": "noexpr",
    }
    out = resolver.resolve(value, context={"x": "X", "y": "Y"})
    assert out == {"a": ["X", 1, {"b": "Y"}], "c": "noexpr"}


def test_resolve_with_partial_context_ignore_undefined_keeps_expression(resolver):
    value = ["${{ known }}", "${{ unknown }}", {"x": "${{ known }}-${{ unknown }}"}]
    out = resolver.resolve_with_partial_context(
        value, context={"known": "K"}, ignore_undefined=True
    )
    assert out[0] == "K"
    # permissive Jinja renders undefined variables as empty strings
    assert out[1] == ""
    assert out[2]["x"] == "K-"


def test_resolve_with_partial_context_without_ignore_undefined_errors(resolver):
    with pytest.raises(ValueError, match="Undefined variable"):
        resolver.resolve_with_partial_context(
            "${{ unknown }}", context={}, ignore_undefined=False
        )


# -- VariableValue in expression context -------------------------------------


class TestVariableValueInExpressions:
    """Verify VariableValue works seamlessly as a Jinja context value."""

    def test_domain_access(self, resolver):
        ctx = {"variables": {"CONC": VariableValue(16, domain=[1, 4, 16, 64])}}
        result = resolver.resolve("${{ variables.CONC.domain }}", ctx)
        assert result == "[1, 4, 16, 64]"

    def test_domain_empty_when_not_set(self, resolver):
        ctx = {"variables": {"X": VariableValue("hello")}}
        result = resolver.resolve("${{ variables.CONC.domain }}", {**ctx, "variables": {"CONC": VariableValue(1)}})
        assert result == "[]"

    def test_renders_as_value(self, resolver):
        ctx = {"variables": {"ISL": VariableValue(1024, domain=[1024, 8192])}}
        assert resolver.resolve("${{ variables.ISL }}", ctx) == "1024"

    def test_string_value_renders(self, resolver):
        ctx = {"variables": {"IMG": VariableValue("nginx:latest")}}
        assert resolver.resolve("${{ variables.IMG }}", ctx) == "nginx:latest"

    def test_arithmetic(self, resolver):
        ctx = {"variables": {"ISL": VariableValue(1024)}}
        assert resolver.resolve("${{ variables.ISL * 5 }}", ctx) == "5120"

    def test_arithmetic_between_two_variables(self, resolver):
        ctx = {"variables": {"A": VariableValue(10), "B": VariableValue(3)}}
        assert resolver.resolve("${{ variables.A + variables.B }}", ctx) == "13"

    def test_comparison(self, resolver):
        ctx = {"variables": {"N": VariableValue(4)}}
        result = resolver.resolve("${{ 'yes' if variables.N > 2 else 'no' }}", ctx)
        assert result == "yes"

    def test_shorthand_access(self, resolver):
        """${{ X }} shorthand (via **variables_ctx spread) works with VariableValue."""
        vv = VariableValue(42, domain=[1, 2, 42])
        ctx = {"variables": {"X": vv}, "X": vv}
        assert resolver.resolve("${{ X }}", ctx) == "42"
        assert resolver.resolve("${{ X.domain }}", ctx) == "[1, 2, 42]"

    def test_string_concatenation(self, resolver):
        ctx = {"variables": {"HOST": VariableValue("10.0.0.1"), "PORT": VariableValue(8080)}}
        result = resolver.resolve("${{ variables.HOST }}:${{ variables.PORT }}", ctx)
        assert result == "10.0.0.1:8080"

    def test_range_list_expression(self, resolver):
        ctx = {
            "variables": {
                "INFRA_NODE_INDEX": VariableValue(0),
                "NUM_FRONTENDS": VariableValue(2),
            }
        }
        result = resolver.resolve(
            "${{ range(variables.INFRA_NODE_INDEX, variables.INFRA_NODE_INDEX + variables.NUM_FRONTENDS) | list }}",
            ctx,
        )
        assert result == "[0, 1]"
