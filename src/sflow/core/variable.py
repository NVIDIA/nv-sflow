# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class VariableType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"

    def __str__(self) -> str:
        return self.value


class Variable(BaseModel):
    name: str
    value: Any
    description: str | None = None
    type: VariableType = VariableType.STRING
    domain: list[Any] | None = None


class VariableValue:
    """Wraps a variable's resolved value with metadata accessible in expressions.

    Allows ``${{ variables.X }}`` to render as the value (backward-compatible),
    while ``${{ variables.X.domain }}`` exposes the variable's domain list.

    Arithmetic, comparison, and container operations delegate to the underlying
    value so that expressions like ``${{ variables.ISL * 5 }}`` keep working.
    """

    __slots__ = ("_value", "domain")

    def __init__(self, value: Any, *, domain: list[Any] | None = None) -> None:
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "domain", domain if domain is not None else [])

    @property
    def value(self) -> Any:
        return self._value

    # -- String representation (used by Jinja2 template rendering) -----------

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return repr(self._value)

    def __format__(self, format_spec: str) -> str:
        return format(self._value, format_spec)

    # -- Type coercion -------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def __float__(self) -> float:
        return float(self._value)

    # -- Hashing & equality --------------------------------------------------

    def __hash__(self) -> int:
        return hash(self._value)

    def _unwrap(self, other: Any) -> Any:
        return other._value if isinstance(other, VariableValue) else other

    def __eq__(self, other: object) -> bool:
        return self._value == self._unwrap(other)

    def __ne__(self, other: object) -> bool:
        return self._value != self._unwrap(other)

    def __lt__(self, other: Any) -> bool:
        return self._value < self._unwrap(other)

    def __le__(self, other: Any) -> bool:
        return self._value <= self._unwrap(other)

    def __gt__(self, other: Any) -> bool:
        return self._value > self._unwrap(other)

    def __ge__(self, other: Any) -> bool:
        return self._value >= self._unwrap(other)

    # -- Arithmetic ----------------------------------------------------------

    def __add__(self, other: Any) -> Any:
        return self._value + self._unwrap(other)

    def __radd__(self, other: Any) -> Any:
        return other + self._value

    def __sub__(self, other: Any) -> Any:
        return self._value - self._unwrap(other)

    def __rsub__(self, other: Any) -> Any:
        return other - self._value

    def __mul__(self, other: Any) -> Any:
        return self._value * self._unwrap(other)

    def __rmul__(self, other: Any) -> Any:
        return other * self._value

    def __truediv__(self, other: Any) -> Any:
        return self._value / self._unwrap(other)

    def __rtruediv__(self, other: Any) -> Any:
        return other / self._value

    def __floordiv__(self, other: Any) -> Any:
        return self._value // self._unwrap(other)

    def __rfloordiv__(self, other: Any) -> Any:
        return other // self._value

    def __mod__(self, other: Any) -> Any:
        return self._value % self._unwrap(other)

    def __rmod__(self, other: Any) -> Any:
        return other % self._value

    def __neg__(self) -> Any:
        return -self._value

    def __pos__(self) -> Any:
        return +self._value

    def __abs__(self) -> Any:
        return abs(self._value)

    # -- Container protocol (for list/dict/string values) --------------------

    def __len__(self) -> int:
        return len(self._value)

    def __iter__(self):  # type: ignore[override]
        return iter(self._value)

    def __contains__(self, item: Any) -> bool:
        return item in self._value

    def __getitem__(self, key: Any) -> Any:
        return self._value[key]


# ---------------------------------------------------------------------------
# Context builders — single ground-truth for wrapping variables for Jinja
# ---------------------------------------------------------------------------


def build_variables_ctx(
    variables: dict[str, Variable] | None,
) -> dict[str, VariableValue]:
    """Build a Jinja-friendly variables context from resolved :class:`Variable` objects.

    Used by ``assembly.py`` where the full ``Variable`` model is available.
    """
    return {
        name: VariableValue(var.value, domain=var.domain)
        for name, var in (variables or {}).items()
    }


def build_variables_ctx_from_raw(
    var_map: dict[str, Any],
    domain_map: dict[str, list[Any]] | None = None,
) -> dict[str, VariableValue]:
    """Build a Jinja-friendly variables context from plain value/domain dicts.

    Used by CLI entry points (``batch``, ``compose``) that operate on raw YAML
    dicts rather than :class:`Variable` objects.
    """
    dm = domain_map or {}
    return {
        name: VariableValue(val, domain=dm.get(name))
        for name, val in var_map.items()
    }


def extract_domains_from_raw_config(data: dict[str, Any]) -> dict[str, list[Any]]:
    """Extract ``{name: domain_list}`` from raw sflow YAML data.

    Handles all variable formats used in sflow configs:
    - dict-of-dict:  ``variables: {KEY: {value: …, domain: […]}}``
    - list-of-dict:  ``variables: [{name: KEY, domain: […]}]``

    Scans both top-level ``variables`` and ``workflow.variables``.
    """
    domain_map: dict[str, list[Any]] = {}

    for section in (_get_var_section(data), _get_wf_var_section(data)):
        if section is None:
            continue
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(v, dict) and "domain" in v:
                    domain_map[k] = v["domain"]
        elif isinstance(section, list):
            for v in section:
                if isinstance(v, dict) and "name" in v and "domain" in v:
                    domain_map[v["name"]] = v["domain"]

    return domain_map


def _get_var_section(data: dict[str, Any]) -> Any:
    return data.get("variables")


def _get_wf_var_section(data: dict[str, Any]) -> Any:
    wf = data.get("workflow")
    return wf.get("variables") if isinstance(wf, dict) else None
