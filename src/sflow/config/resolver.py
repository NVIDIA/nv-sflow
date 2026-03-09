# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from jinja2 import StrictUndefined, TemplateSyntaxError, UndefinedError
from jinja2 import meta as jinja2_meta
from jinja2.sandbox import SandboxedEnvironment


@dataclass
class ExpressionValidationError:
    """Represents a validation error for an expression."""

    expression: str
    error: str
    location: Optional[str] = (
        None  # e.g., "variables.MY_VAR" or "workflow.tasks.etcd.script[0]"
    )

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"Expression error{loc}: {self.error} in '{self.expression}'"


@dataclass
class ValidationResult:
    """Result of expression validation."""

    valid: bool
    errors: List[ExpressionValidationError] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


class ExpressionResolver:
    """
    Uses Jinja2 engine to evaluate expressions and handles complex constraint logic.
    Supports ${{ ... }} syntax for expressions.
    """

    # Matches ${{ ... }}
    VARIABLE_PATTERN = re.compile(r"\$\{\{(.+?)\}\}")

    def __init__(self):
        # SandboxedEnvironment reduces the impact of template injection if a config file
        # comes from an untrusted source. We do NOT enable autoescape because these
        # expressions are used for general config values (not HTML rendering).
        self._env = SandboxedEnvironment(
            undefined=StrictUndefined,  # Raise error on undefined variables
            autoescape=False,
            variable_start_string="${{",
            variable_end_string="}}",
        )

    def has_expression(self, value: Any) -> bool:
        """Check if a value contains any ${{ }} expressions."""
        if isinstance(value, str):
            return "${{" in value
        elif isinstance(value, list):
            return any(self.has_expression(item) for item in value)
        elif isinstance(value, dict):
            return any(self.has_expression(v) for v in value.values())
        return False

    def validate_syntax(
        self, value: Any, location: Optional[str] = None
    ) -> ValidationResult:
        """
        Validates expression syntax without resolving values.
        Use this at config load time to catch syntax errors early.
        """
        errors: List[ExpressionValidationError] = []
        self._validate_recursive(value, location, errors)
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _validate_recursive(
        self,
        value: Any,
        location: Optional[str],
        errors: List[ExpressionValidationError],
    ):
        """Recursively validate expressions in a value."""
        if isinstance(value, str):
            if "${{" in value:
                try:
                    # Just parse, don't render - this validates syntax
                    self._env.parse(value)
                except TemplateSyntaxError as e:
                    errors.append(
                        ExpressionValidationError(
                            expression=value, error=str(e), location=location
                        )
                    )
        elif isinstance(value, list):
            for i, item in enumerate(value):
                item_loc = f"{location}[{i}]" if location else f"[{i}]"
                self._validate_recursive(item, item_loc, errors)
        elif isinstance(value, dict):
            for k, v in value.items():
                item_loc = f"{location}.{k}" if location else k
                self._validate_recursive(v, item_loc, errors)

    def extract_references(self, value: Any) -> Set[str]:
        """
        Extract all variable references from expressions.
        Useful for dependency analysis (e.g., finding which task outputs are needed).
        """
        references: Set[str] = set()
        self._extract_recursive(value, references)
        return references

    def _extract_recursive(self, value: Any, references: Set[str]):
        """Recursively extract references from a value."""
        if isinstance(value, str):
            if "${{" in value:
                try:
                    ast = self._env.parse(value)
                    # Get undeclared variables from the AST
                    undeclared = jinja2_meta.find_undeclared_variables(ast)
                    references.update(undeclared)
                except TemplateSyntaxError:
                    pass  # Ignore syntax errors here, they'll be caught by validate
        elif isinstance(value, list):
            for item in value:
                self._extract_recursive(item, references)
        elif isinstance(value, dict):
            for v in value.values():
                self._extract_recursive(v, references)

    def resolve(self, value: Any, context: Dict[str, Any]) -> Any:
        """
        Resolves a single value (string or structure) using the given context.
        Recursive for lists and dictionaries.
        """
        if isinstance(value, str):
            if "${{" in value:
                return self._resolve_string(value, context)
            return value
        elif isinstance(value, list):
            return [self.resolve(item, context) for item in value]
        elif isinstance(value, dict):
            return {k: self.resolve(v, context) for k, v in value.items()}
        else:
            return value

    def _resolve_string(self, value: str, context: Dict[str, Any]) -> Any:
        try:
            template = self._env.from_string(value)
            resolved = template.render(**context)
            return resolved
        except UndefinedError as e:
            raise ValueError(f"Undefined variable in expression '{value}': {e}")
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid expression syntax in '{value}': {e}")
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{value}': {e}")

    def resolve_with_partial_context(
        self, value: Any, context: Dict[str, Any], ignore_undefined: bool = False
    ) -> Any:
        """
        Resolve expressions, optionally ignoring undefined variables.
        Useful for partial resolution when not all context is available yet.
        """
        if not ignore_undefined:
            return self.resolve(value, context)

        permissive_env = SandboxedEnvironment(
            autoescape=False,
            variable_start_string="${{",
            variable_end_string="}}",
        )

        def resolve_partial(v: Any) -> Any:
            if isinstance(v, str):
                if "${{" not in v:
                    return v
                try:
                    template = permissive_env.from_string(v)
                    return template.render(**context)
                except UndefinedError:
                    return v
            elif isinstance(v, list):
                return [resolve_partial(item) for item in v]
            elif isinstance(v, dict):
                return {k: resolve_partial(val) for k, val in v.items()}
            return v

        return resolve_partial(value)
