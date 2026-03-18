# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypeVar, Union
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PositiveInt,
    field_validator,
    model_validator,
)

from sflow.core.backend_registry import (
    backend_config_type_adapter,
    ensure_builtin_backends_registered,
)
from sflow.core.operator import OperatorConfig
from sflow.core.operator_registry import (
    ensure_builtin_operators_registered,
    operator_config_type_adapter,
)

# -----------------------------------------------------------------------------
# Resolvable Type Support
# -----------------------------------------------------------------------------
# Fields that can contain ${{ }} expressions need to accept strings at parse time.
# The actual type validation happens after expression resolution.

T = TypeVar("T")

# Type alias for fields that can be either the target type OR an expression string
# Usage: nodes: Resolvable[int] means it can be `4` or `"${{ variables.NODE_COUNT }}"`
Resolvable = Union[T, str]


def is_expression(value: Any) -> bool:
    """Check if a value is an unresolved expression string."""
    return isinstance(value, str) and "${{" in value


def _normalize_to_list(v: Any) -> Any:
    """Normalizes a dict of items to a list of items with 'name' injected."""
    if isinstance(v, dict):
        return [
            {**value, "name": key}
            if isinstance(value, dict)
            else {"name": key, "value": value}
            for key, value in v.items()
        ]
    return v


class StrictBaseModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class VariableConfig(StrictBaseModel):
    """Configuration for a variable."""

    name: str
    description: Optional[str] = None
    value: Any
    domain: Optional[List[Any]] = None
    type: str = "string"

    @model_validator(mode="after")
    def check_value_in_domain(self) -> "VariableConfig":
        if self.domain is not None and self.value not in self.domain:
            raise ValueError(
                f"Value '{self.value}' is not in the allowed domain: {self.domain}"
            )
        return self


class ArtifactConfig(StrictBaseModel):
    """Configuration for an artifact."""

    name: str
    description: Optional[str] = None
    uri: str
    content: Optional[str] = None  # For inline file artifacts

    @field_validator("uri")
    @classmethod
    def uri_must_be_valid(cls, v: str) -> str:
        # Skip validation if the URI contains template expressions (e.g. ${{ ... }})
        # These will be resolved later and validated at resolution time.
        if "${{" in v:
            return v
        try:
            parsed = urlparse(v)
            if not parsed.scheme:
                raise ValueError(
                    "URI must have a scheme (e.g., http://, s3://, file://)"
                )
            if not (parsed.netloc or parsed.path):
                raise ValueError("URI must have a network location or path")
        except ValueError as e:
            # Re-raise ValueErrors as-is
            raise e
        except Exception as e:
            raise ValueError(f"Invalid URI: {v}. Error: {e}")
        return v

    @model_validator(mode="after")
    def check_inline_content(self) -> "ArtifactConfig":
        if self.content is not None and not self.uri.startswith("file://"):
            raise ValueError("Inline content is only supported for 'file://' URIs")
        return self


class BackendConfig(BaseModel):
    """Configuration for a compute backend."""

    name: str
    type: str
    default: bool = False
    # If set, this value will be used to populate ComputeNode.num_gpus for all nodes
    # returned by this backend allocation. This enables better GPU packing/validation.
    gpus_per_node: Optional[Resolvable[int]] = None

    @field_validator("gpus_per_node")
    @classmethod
    def gpu_per_node_must_be_positive_if_concrete(cls, v: Any) -> Any:
        # Allow unresolved expressions; validate concrete ints only.
        if v is None or is_expression(v):
            return v
        try:
            iv = int(v)
        except Exception as e:
            raise ValueError(
                f"gpus_per_node must be an int or expression, got {v!r}"
            ) from e
        if iv <= 0:
            raise ValueError(f"gpus_per_node must be > 0, got {iv}")
        return iv


class TaskOperatorOverrideConfig(BaseModel):
    """
    Task-level operator reference with optional per-task overrides.

    Mirrors the runtime override pattern:
      operator: "op_name"
      operator:
        name: op_name
        (any operator-specific overrides...)
    """

    model_config = ConfigDict(extra="allow")

    name: str


class TcpPortProbeConfig(StrictBaseModel):
    """TCP port probe configuration."""

    port: Resolvable[int]  # Can be int or expression
    host: Optional[Resolvable[str]] = None
    on_node: Literal["first", "each"] = "first"


class HttpProbeConfig(StrictBaseModel):
    url: Resolvable[str]
    headers: Optional[Dict[str, str]] = None
    body: Optional[str] = None


class LogWatchProbeConfig(StrictBaseModel):
    regex_pattern: Optional[str] = None
    match_pattern: Optional[str] = None
    logger: Optional[str] = None
    match_count: Optional[Resolvable[int]] = 1

    @model_validator(mode="before")
    @classmethod
    def check_pattern_exclusivity(cls, data: Any) -> Any:
        if isinstance(data, dict):
            has_regex = data.get("regex_pattern") is not None
            has_match = data.get("match_pattern") is not None
            if has_regex and has_match:
                raise ValueError(
                    "Only one of 'regex_pattern' or 'match_pattern' may be set, not both"
                )
            if not has_regex and not has_match:
                raise ValueError(
                    "Either 'regex_pattern' or 'match_pattern' must be set"
                )
        return data

    @model_validator(mode="after")
    def normalize_pattern(self) -> "LogWatchProbeConfig":
        if self.regex_pattern is None and self.match_pattern is not None:
            self.regex_pattern = self.match_pattern
        return self


class ProbeConfig(StrictBaseModel):
    """Configuration for a single probe check."""

    # One of these must be set
    tcp_port: Optional[TcpPortProbeConfig] = None
    http_get: Optional[HttpProbeConfig] = None
    http_post: Optional[HttpProbeConfig] = None
    log_watch: Optional[LogWatchProbeConfig] = None

    @model_validator(mode="after")
    def check_one_probe_type(self) -> "ProbeConfig":
        probes = [self.tcp_port, self.http_get, self.http_post, self.log_watch]
        set_probes = [p for p in probes if p is not None]
        if len(set_probes) != 1:
            raise ValueError("Exactly one probe type must be specified")
        return self

    # Common settings (can be expressions)
    delay: Resolvable[int] = 0
    timeout: Resolvable[int] = 60
    interval: Resolvable[int] = 5
    success_threshold: Resolvable[int] = 1
    failure_threshold: Resolvable[int] = 3


class ProbesConfig(StrictBaseModel):
    """Configuration for task probes."""

    readiness: Optional[ProbeConfig] = None
    failure: Optional[ProbeConfig] = None


class OutputMetricConfig(StrictBaseModel):
    description: Optional[str] = None
    type: Optional[str] = None
    aggregate: Optional[str] = None


class OutputConfig(StrictBaseModel):
    pattern: str
    source: str = "stdout"
    metrics: Optional[Dict[str, OutputMetricConfig]] = None


class NodeResourceConfig(StrictBaseModel):
    """Node resource configuration for a task."""

    indices: Optional[List[Resolvable[int]]] = None  # Can be [0, 1] or ["${{ ... }}"]
    count: Optional[Resolvable[int]] = None  # Can be int or expression
    exclude: Optional[Union[List[Resolvable[int]], Resolvable[int]]] = None


class GpuResourceConfig(StrictBaseModel):
    """GPU resource configuration for a task."""

    count: Resolvable[int]  # Can be int or expression like "${{ variables.GPU_COUNT }}"


class ResourcesConfig(StrictBaseModel):
    nodes: Optional[NodeResourceConfig] = None
    gpus: Optional[GpuResourceConfig] = None


class RetryConfig(StrictBaseModel):
    """Retry configuration for task failures."""

    count: Resolvable[int]
    interval: Resolvable[int]
    backoff: Resolvable[int] = 1


class ReplicaPolicy(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ReplicaConfig(StrictBaseModel):
    count: Optional[Union[PositiveInt, str]] = (
        None  # Can be an int or a variable expression
    )
    # Allow ${{ ... }} expressions to decide the policy at runtime (resolved during assembly).
    policy: Union[ReplicaPolicy, str] = ReplicaPolicy.PARALLEL
    variables: Optional[List[str]] = None  # List of variables to sweep/distribute

    @field_validator("policy")
    @classmethod
    def policy_must_be_valid_if_concrete(cls, v: Any) -> Any:
        # Allow unresolved expressions; validate concrete values only.
        if v is None or is_expression(v):
            return v
        if isinstance(v, ReplicaPolicy):
            return v
        if isinstance(v, str):
            try:
                return ReplicaPolicy(v)
            except Exception as e:
                raise ValueError(
                    f"replicas.policy must be 'parallel' or 'sequential' (or an expression), got {v!r}"
                ) from e
        raise ValueError(
            f"replicas.policy must be 'parallel' or 'sequential' (or an expression), got {type(v).__name__}"
        )


class TaskConfig(StrictBaseModel):
    """Configuration for a single task."""

    name: str
    operator: Optional[Union[str, TaskOperatorOverrideConfig]] = None
    backend: Optional[Union[str, Dict[str, Any]]] = None  # Name or inline override
    script: List[str]
    resources: Optional[ResourcesConfig] = None

    @field_validator("script")
    @classmethod
    def script_must_not_be_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Script list cannot be empty")
        return v

    probes: Optional[ProbesConfig] = None
    outputs: Optional[List[OutputConfig]] = None
    depends_on: Optional[List[str]] = None
    replicas: Optional[ReplicaConfig] = None
    retries: Optional[RetryConfig] = None
    timeout: Optional[Union[int, str]] = None
    variables: Optional[
        Annotated[List[VariableConfig], BeforeValidator(_normalize_to_list)]
    ] = None


class WorkflowConfig(StrictBaseModel):
    """Configuration for the workflow execution."""

    name: str
    timeout: Optional[Union[str, int]] = None
    variables: Optional[
        Annotated[List[VariableConfig], BeforeValidator(_normalize_to_list)]
    ] = None
    tasks: Annotated[List[TaskConfig], BeforeValidator(_normalize_to_list)]

    @field_validator("tasks")
    @classmethod
    def tasks_must_not_be_empty(cls, v: List[TaskConfig]) -> List[TaskConfig]:
        if not v:
            raise ValueError("Tasks list cannot be empty")
        return v

    @model_validator(mode="after")
    def check_dependencies(self) -> "WorkflowConfig":
        task_names = {t.name for t in self.tasks}

        # Check task name uniqueness
        if len(task_names) != len(self.tasks):
            seen = set()
            duplicates = set()
            for t in self.tasks:
                if t.name in seen:
                    duplicates.add(t.name)
                seen.add(t.name)
            raise ValueError(f"Duplicate task names found: {duplicates}")

        for task in self.tasks:
            # Check explicit dependencies
            if task.depends_on:
                for dep in task.depends_on:
                    if dep not in task_names:
                        raise ValueError(
                            f"Task '{task.name}' depends on unknown task '{dep}'"
                        )

            # Check probe log watchers
            if task.probes:
                for probe_type in ["readiness", "failure"]:
                    probe = getattr(task.probes, probe_type)
                    if probe and probe.log_watch and probe.log_watch.logger:
                        if probe.log_watch.logger not in task_names:
                            raise ValueError(
                                f"Task '{task.name}' {probe_type} probe refers to unknown task '{probe.log_watch.logger}'"
                            )
        return self


class SflowConfig(StrictBaseModel):
    """
    Main configuration model for Sflow.
    """

    version: str
    variables: Optional[
        Annotated[List[VariableConfig], BeforeValidator(_normalize_to_list)]
    ] = None
    artifacts: Optional[
        Annotated[List[ArtifactConfig], BeforeValidator(_normalize_to_list)]
    ] = None
    backends: Optional[
        Annotated[List[BackendConfig], BeforeValidator(_normalize_to_list)]
    ] = None
    operators: Optional[
        Annotated[List[OperatorConfig], BeforeValidator(_normalize_to_list)]
    ] = None
    workflow: WorkflowConfig

    @model_validator(mode="after")
    def check_backends(self) -> "SflowConfig":
        if self.backends:
            defaults = [b for b in self.backends if b.default]
            if len(defaults) > 1:
                raise ValueError(
                    "Multiple default backends found. Only one backend can be set as default."
                )
        return self

    @field_validator("backends", mode="before")
    @classmethod
    def backends_must_match_registered_configs(cls, v: Any) -> Any:
        if v is None:
            return None

        v = _normalize_to_list(v)
        ensure_builtin_backends_registered()
        adapter = backend_config_type_adapter()

        if not isinstance(v, list):
            raise TypeError("backends must be a list or dict")

        out = []
        for item in v:
            # If user passed a model instance, validate its dumped dict so discriminator works.
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            out.append(adapter.validate_python(item))
        return out

    @field_validator("version")
    @classmethod
    def version_must_be_valid(cls, v: str) -> str:
        if v not in ["0.1"]:
            raise ValueError(f"Version {v} is not supported")
        return v

    @field_validator("operators", mode="before")
    @classmethod
    def operators_must_match_registered_configs(cls, v: Any) -> Any:
        if v is None:
            return None

        # Support both list and dict forms via the same normalizer used elsewhere.
        v = _normalize_to_list(v)

        # Populate registry for built-in operators before validating.
        ensure_builtin_operators_registered()
        adapter = operator_config_type_adapter()

        if not isinstance(v, list):
            raise TypeError("operators must be a list or dict")

        out: list[OperatorConfig] = []
        for item in v:
            out.append(adapter.validate_python(item))
        return out


def validate_node_exclude_indices(config: SflowConfig) -> None:
    """Validate resources.nodes.exclude indices against backend node count.

    Must be called **after** variable overrides are applied so that the
    resolved node count reflects ``--set`` / CSV overrides.  Resolves
    ``${{ variables.X }}`` expressions using the config's own variable
    definitions.  Skips values that cannot be resolved statically.
    """
    import re as _re

    if not config.backends:
        return

    var_map: dict[str, Any] = {}
    for v in config.variables or []:
        var_map[v.name] = v.value
    for v in config.workflow.variables or []:
        var_map[v.name] = v.value

    _VAR_RE = _re.compile(r"^\$\{\{\s*variables\.(\w+)\s*\}\}$")

    def _try_resolve_int(val: Any) -> int | None:
        if val is None:
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            m = _VAR_RE.match(val)
            if m:
                ref = var_map.get(m.group(1))
                if ref is not None and not is_expression(ref):
                    try:
                        return int(ref)
                    except (ValueError, TypeError):
                        return None
                return None
            if is_expression(val):
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None
        return None

    backend_map: dict[str, Any] = {}
    default_backend: Any = None
    for b in config.backends:
        backend_map[b.name] = b
        if b.default:
            default_backend = b

    for task in config.workflow.tasks:
        if not task.resources or not task.resources.nodes:
            continue
        exclude = task.resources.nodes.exclude
        if exclude is None:
            continue

        backend = default_backend
        if task.backend is not None:
            if isinstance(task.backend, str):
                backend = backend_map.get(task.backend)
            else:
                continue

        if backend is None:
            continue

        total_nodes = _try_resolve_int(getattr(backend, "nodes", None))
        if total_nodes is None:
            continue

        raw_list = exclude if isinstance(exclude, list) else [exclude]
        for idx_val in raw_list:
            idx = _try_resolve_int(idx_val)
            if idx is None:
                continue
            if idx < 0 or idx >= total_nodes:
                raise ValueError(
                    f"Task '{task.name}' resources.nodes.exclude contains index "
                    f"{idx} out of range for {total_nodes} allocated node(s) "
                    f"(valid: 0..{total_nodes - 1})"
                )
