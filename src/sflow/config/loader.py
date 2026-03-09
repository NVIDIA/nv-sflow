# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from sflow.config.resolver import ExpressionResolver

from .schema import SflowConfig


class ConfigLoader:
    """
    Parses YAML configuration, validates against the config schema, and applies CLI overrides.

    The loading process:
    1. Parse YAML to dict
    2. Apply CLI overrides (--set, --artifact)
    """

    def __init__(self):
        self._resolver = ExpressionResolver()
        self.config: Optional[SflowConfig] = None

    def load_config(
        self,
        path: Path,
        variable_overrides: Optional[List[str]] = None,
        artifact_overrides: Optional[List[str]] = None,
    ) -> SflowConfig:
        """
        Loads the configuration from a file, applying CLI overrides.

        Args:
            path (Path): Path to the configuration file.
            variable_overrides (Optional[List[str]]): List of variable overrides (key=value).
            artifact_overrides (Optional[List[str]]): List of artifact overrides (key=uri).

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If parsing fails or validation errors occur.

        Returns:
            SflowConfig: The validated configuration.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

        if config_data is None:
            raise ValueError(f"Configuration file is empty: {path}")

        syntax = self._resolver.validate_syntax(config_data, location=str(path))
        if not syntax.valid:
            details = "\n".join(str(e) for e in syntax.errors)
            raise ValueError(
                f"Configuration expression syntax validation failed:\n{details}"
            )

        # Apply CLI overrides to the raw dict
        if variable_overrides:
            self._apply_variable_overrides(config_data, variable_overrides)

        if artifact_overrides:
            self._apply_artifact_overrides(config_data, artifact_overrides)

        try:
            config = SflowConfig.model_validate(config_data)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed:\n{e}")

        self.config = config
        return config

    def _apply_variable_overrides(
        self, config_data: Dict[str, Any], overrides: List[str]
    ):
        """
        Applies CLI variable overrides to the configuration dictionary.
        Format: KEY=VALUE
        """
        if "variables" not in config_data or config_data["variables"] is None:
            # Initialize as dict to make lookups easier during override application
            # The schema validation will normalize it to list later if needed
            config_data["variables"] = {}

        variables = config_data["variables"]

        # If variables is a list, convert to dict temporarily for easier override handling
        is_list = isinstance(variables, list)
        if is_list:
            var_map = {v.get("name"): v for v in variables}
        else:
            var_map = variables

        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid variable override format: '{override}'. Expected KEY=VALUE."
                )
            key, value = override.split("=", 1)

            # Helper to infer type (int, float, bool, list, or string)
            typed_value = self._infer_type(value)

            # Update or create the variable entry
            if key in var_map:
                entry = var_map[key]
                # If it's a dict (VariableConfig structure), update appropriately
                if isinstance(entry, dict):
                    if isinstance(typed_value, list):
                        # If the value is a list, treat it as a domain for replica sweeps
                        # Set domain to the list and value to the first element
                        entry["domain"] = typed_value
                        if typed_value:
                            entry["value"] = typed_value[0]
                    else:
                        entry["value"] = typed_value
                else:
                    # Fallback or scalar case (if supported in future)
                    # For now, if we converted from list, entry is a dict
                    # If it was a dict, entry might be just the value? No, assumed dict structure.
                    var_map[key] = {"value": typed_value}
            else:
                # Raise error if variable not found in config
                raise ValueError(
                    f"Variable '{key}' specified in overrides is not defined in the configuration."
                )

        # If we converted from list, we need to update the original list
        if is_list:
            # We updated the dict objects in var_map, which are references to items in the list?
            # Yes, if list contains dicts, var_map contains refs to those dicts.
            # But if we assigned new dict to var_map[key], the list is not updated.
            # So we must reconstruct the list or update carefully.
            # Simple approach: Since we only modified mutable dicts in place (entry["value"] = ...),
            # the original list items are updated.
            # EXCEPT if we did var_map[key] = ... (assignment), which we don't do for existing keys above.
            pass

    def _apply_artifact_overrides(
        self, config_data: Dict[str, Any], overrides: List[str]
    ):
        """
        Applies CLI artifact overrides to the configuration dictionary.
        Format: NAME=URI

        When changing from 'file://' to any other URI scheme, the 'content' field is
        automatically removed to prevent validation errors (inline content is only valid for 'file://').
        """
        if "artifacts" not in config_data or config_data["artifacts"] is None:
            config_data["artifacts"] = []

        artifacts = config_data["artifacts"]

        # Handle both list and dict formats
        is_list = isinstance(artifacts, list)
        if is_list:
            artifact_map = {a.get("name"): a for a in artifacts}
        else:
            # Dict format: key is name
            artifact_map = artifacts

        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid artifact override format: '{override}'. Expected NAME=URI."
                )
            name, uri = override.split("=", 1)

            if name in artifact_map:
                artifact_entry = artifact_map[name]
                old_uri = artifact_entry.get("uri", "")

                # Update the URI
                artifact_entry["uri"] = uri

                # Handle type changes: if changing from file:// to any non-file:// scheme, remove content
                # because inline content is only valid for 'file://' URIs
                old_is_file = old_uri.startswith("file://")
                new_is_non_file = not uri.startswith("file://")

                if old_is_file and new_is_non_file:
                    # Remove content field when switching from file:// to fs://
                    if "content" in artifact_entry:
                        del artifact_entry["content"]
            else:
                # Raise error if artifact not found in config
                raise ValueError(
                    f"Artifact '{name}' specified in overrides is not defined in the configuration."
                )

    def _infer_type(self, value: str) -> Any:
        """
        Infers the type of the value string (int, float, bool, list, or str).

        Supports JSON-like list syntax: [1, 2, 3] or ["a", "b", "c"]
        """
        import json

        # Try JSON parsing first (handles lists, nested structures)
        # This catches: [1,2,3], ["a","b"], {"key": "value"}, etc.
        value_stripped = value.strip()
        if value_stripped.startswith("[") or value_stripped.startswith("{"):
            try:
                return json.loads(value_stripped)
            except json.JSONDecodeError:
                pass  # Fall through to other type inference

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try bool
        lower_val = value.lower()
        if lower_val in ("true", "yes", "on"):
            return True
        if lower_val in ("false", "no", "off"):
            return False

        # Default string
        return value
