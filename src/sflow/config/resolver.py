"""Backward-compatible imports for expression resolution APIs."""

from sflow.resolution import (
    ExpressionResolver,
    ExpressionValidationError,
    ValidationResult,
    enrich_error_with_location,
    find_lines_in_files,
)

__all__ = [
    "ExpressionResolver",
    "ExpressionValidationError",
    "ValidationResult",
    "enrich_error_with_location",
    "find_lines_in_files",
]
