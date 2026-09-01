# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypeVar, Union
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
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
from sflow.core.storage_registry import (
    ensure_builtin_storage_registered,
    storage_config_type_adapter,
)
from sflow.logging import get_logger

_logger = get_logger(__name__)

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


class StorageConfig(BaseModel):
    """Base configuration for a post-execution storage target (S3, GCS, ...).

    Concrete plugins (e.g. `S3StorageConfig`) subclass this and redeclare
    `type: Literal["s3"] = "s3"` so Pydantic can discriminate the union
    built in `sflow.core.storage_registry.storage_config_type_adapter()`.
    """

    name: str
    type: str


class BackendConfig(BaseModel):
    """Configuration for a compute backend."""

    name: str
    type: str
    default: bool = False
    # If set, this value will be used to populate ComputeNode.num_gpus for all nodes
    # returned by this backend allocation. This enables better GPU packing/validation.
    # `0` is allowed and means "CPU-only" (tasks requesting GPUs will be rejected
    # downstream with a clear error).
    gpus_per_node: Optional[Resolvable[int]] = None
    # Backend-agnostic node include/exclude host lists. Also settable via the
    # ``--include-nodes`` / ``--exclude-nodes`` CLI flags (which union over these).
    # Each backend translates them to its native node selection: Slurm
    # ``--nodelist`` / ``--exclude``, Kubernetes ``nodeAffinity`` In/NotIn, Docker
    # host-pool filtering. Entries may be ``${{ }}`` expressions and may be comma-
    # or whitespace-joined (backends normalize after resolution).
    include_nodes: Optional[List[Resolvable[str]]] = None
    exclude_nodes: Optional[List[Resolvable[str]]] = None

    def container_images(self) -> list[str]:
        """Return backend-owned default operator image references, if any."""
        return []

    def planning_node_count(self) -> Any | None:
        """Return the statically configured planning node count, if available."""
        return None

    def merge_extra_args(self, extra_args: list[str]) -> "BackendConfig":
        """Return a config copy with CLI-provided backend args merged, if supported.

        De-dups by option (shared with ``sflow batch``): a CLI ``--gres=gpu:4``
        overrides a recipe ``--gres=gpu:8`` rather than both being kept, with the
        CLI winning on a conflicting option. Repeatable ``key=value`` flags (e.g.
        ``--env=FOO=1`` / ``--env=BAR=2``) are preserved as distinct entries.
        """
        if not extra_args or not hasattr(self, "extra_args"):
            return self
        from sflow.utils.extra_args import dedup_merge_extra_args

        existing = [str(arg) for arg in (getattr(self, "extra_args", None) or [])]
        merged = dedup_merge_extra_args(existing, [str(arg) for arg in extra_args])
        if merged == existing:
            return self
        return self.model_copy(update={"extra_args": merged})

    def merge_node_filters(
        self,
        include_nodes: list[str] | None,
        exclude_nodes: list[str] | None,
    ) -> "BackendConfig":
        """Return a copy with CLI include/exclude node lists unioned over YAML.

        CLI-provided hosts are appended to any recipe ``include_nodes`` /
        ``exclude_nodes`` (order preserved, deduped by exact string). Raises if the
        merge makes a concrete host appear in both lists.
        """
        from sflow.utils.node_filters import find_node_filter_overlap, merge_node_lists

        update: dict[str, Any] = {}
        if include_nodes:
            update["include_nodes"] = merge_node_lists(self.include_nodes, include_nodes)
        if exclude_nodes:
            update["exclude_nodes"] = merge_node_lists(self.exclude_nodes, exclude_nodes)
        if not update:
            return self
        merged = self.model_copy(update=update)
        overlap = find_node_filter_overlap(
            [n for n in (merged.include_nodes or []) if not is_expression(n)],
            [n for n in (merged.exclude_nodes or []) if not is_expression(n)],
        )
        if overlap:
            raise ValueError(
                f"backend '{self.name}': node(s) {overlap} appear in both "
                "include_nodes and exclude_nodes after merging CLI flags"
            )
        return merged

    @model_validator(mode="after")
    def _node_filters_must_not_overlap(self) -> "BackendConfig":
        from sflow.utils.node_filters import find_node_filter_overlap

        overlap = find_node_filter_overlap(
            [n for n in (self.include_nodes or []) if not is_expression(n)],
            [n for n in (self.exclude_nodes or []) if not is_expression(n)],
        )
        if overlap:
            raise ValueError(
                f"backend '{self.name}': node(s) {overlap} appear in both "
                "include_nodes and exclude_nodes"
            )
        return self

    @field_validator("gpus_per_node")
    @classmethod
    def gpu_per_node_must_be_non_negative_if_concrete(cls, v: Any) -> Any:
        # Allow unresolved expressions; validate concrete ints only.
        if v is None or is_expression(v):
            return v
        try:
            iv = int(v)
        except Exception as e:
            raise ValueError(
                f"gpus_per_node must be an int or expression, got {v!r}"
            ) from e
        if iv < 0:
            raise ValueError(f"gpus_per_node must be >= 0, got {iv}")
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
        """Reject a probe that names two *different* patterns, or none at all.

        The two fields are one setting under two spellings, so a config that carries
        both with the same value is not a mistake -- it is what this model produces:
        :meth:`normalize_pattern` copies ``match_pattern`` into ``regex_pattern``, and
        anything that dumps a validated config and loads it again (``sflow compose``,
        and the temp config ``sflow batch`` writes for its dry run when ``--nodes`` or
        ``--gpus-per-node`` re-plans the backends) hands both of them straight back.
        Rejecting that round trip made every recipe whose readiness probe is written
        with ``match_pattern`` fail its dry run with 'Only one of ... not both'.
        """
        if isinstance(data, dict):
            regex = data.get("regex_pattern")
            match = data.get("match_pattern")
            if regex is not None and match is not None and regex != match:
                raise ValueError(
                    "Only one of 'regex_pattern' or 'match_pattern' may be set, not both "
                    f"with different values (got {regex!r} and {match!r})"
                )
            if regex is None and match is None:
                raise ValueError(
                    "Either 'regex_pattern' or 'match_pattern' must be set"
                )
        return data

    @model_validator(mode="after")
    def normalize_pattern(self) -> "LogWatchProbeConfig":
        """Give consumers one field to read, whichever spelling was written.

        Both spellings mean the same thing at runtime: the log_watch probe treats the
        value as a literal unless it carries an ``re:`` / ``regex:`` prefix. Keeping
        ``match_pattern`` as written rather than clearing it means a dumped config
        still reads the way its author wrote it -- which is why the check above has to
        tolerate the pair.
        """
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
    timeout: Resolvable[int] = 1200
    each_check_timeout: Resolvable[int] = 30
    interval: Resolvable[int] = 5
    success_threshold: Resolvable[int] = 1
    failure_threshold: Resolvable[int] = 3


class ProbesConfig(StrictBaseModel):
    """Configuration for task probes."""

    readiness: Optional[Union[ProbeConfig, List[ProbeConfig]]] = None
    failure: Optional[Union[ProbeConfig, List[ProbeConfig]]] = None

    @field_validator("readiness")
    @classmethod
    def readiness_list_must_not_be_empty(cls, v: Any) -> Any:
        if isinstance(v, list) and not v:
            raise ValueError("readiness probe list cannot be empty")
        return v

    @field_validator("failure")
    @classmethod
    def failure_list_must_not_be_empty(cls, v: Any) -> Any:
        if isinstance(v, list) and not v:
            raise ValueError("failure probe list cannot be empty")
        return v


class OutputMetricConfig(StrictBaseModel):
    description: Optional[str] = None
    type: Optional[str] = None
    aggregate: Optional[str] = None


class OutputConfig(StrictBaseModel):
    pattern: str
    source: str = "stdout"
    metrics: Optional[Dict[str, OutputMetricConfig]] = None


# ---------------------------------------------------------------------------
# Result parsing schema (see docs/developer/dev-notes/result-parsing.md)
# ---------------------------------------------------------------------------

# Allowed values for the advanced ``result.patterns[*].aggregate`` field.
RESULT_AGGREGATES = (
    "first",
    "last",
    "list",
    "count",
    "min",
    "max",
    "avg",
    "sum",
)

# Allowed values for the advanced ``result.patterns[*].type`` field.
RESULT_TYPES = ("auto", "string", "int", "float", "bool", "json")

# Allowed values for the ``source`` selector. Initially only ``log`` is implemented.
RESULT_SOURCES = ("log",)


class ResultPatternConfig(StrictBaseModel):
    """
    Advanced regex pattern entry for ``result.patterns[*]``.

    See ``docs/developer/dev-notes/result-parsing.md`` "Advanced regex patterns".
    """

    name: str
    regex: str
    type: str = "auto"
    unit: Optional[str] = None
    aggregate: str = "last"
    required: bool = False
    source: Optional[str] = None
    group: Optional[Union[str, int]] = None

    @field_validator("aggregate")
    @classmethod
    def aggregate_must_be_valid(cls, v: str) -> str:
        if v not in RESULT_AGGREGATES:
            raise ValueError(
                f"result.patterns[*].aggregate must be one of "
                f"{list(RESULT_AGGREGATES)}, got {v!r}"
            )
        return v

    @field_validator("type")
    @classmethod
    def type_must_be_valid(cls, v: str) -> str:
        if v not in RESULT_TYPES:
            raise ValueError(
                f"result.patterns[*].type must be one of {list(RESULT_TYPES)}, got {v!r}"
            )
        return v

    @field_validator("source")
    @classmethod
    def source_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in RESULT_SOURCES:
            raise ValueError(
                f"result.patterns[*].source must be one of {list(RESULT_SOURCES)} (initial release), got {v!r}"
            )
        return v


def _normalize_result_entry(v: Any) -> Any:
    """
    Normalize the ``result`` task entry.

    Accepted shapes:
    - ``{ "metric_name": "regex", ... }``  (simple map -> patterns)
    - ``{ "patterns": [...] }``            (advanced; patterns is a list)
    - ``{ "file": "..." }``                (file source; file is a string)
    - ``{ "patterns": [...], "source": "log" }``

    Detection rules (in order):
    1. If the dict has ``patterns`` as a list, treat it as the advanced form verbatim.
    2. If the dict is only ``file`` (+ optional ``source``), treat it as file-source
       form and let ``ResultConfig`` validate that the value is a JSON source path.
       A metric literally named ``file`` should use the advanced ``patterns`` form.
    3. Otherwise, if every value is a string, treat it as a simple regex map.
    """
    if not isinstance(v, dict):
        return v

    # Advanced form: `patterns` is explicitly a list.
    if isinstance(v.get("patterns"), list):
        return v

    # File-source form: only `file` (+ optional `source`) is present, file is a string.
    keys = set(v.keys())
    if (
        isinstance(v.get("file"), str)
        and keys.issubset({"file", "source"})
    ):
        return v

    # Otherwise, simple regex map — every value must be a string.
    if all(isinstance(val, str) for val in v.values()):
        patterns = [{"name": name, "regex": regex} for name, regex in v.items()]
        return {"patterns": patterns}

    # Fall through: hand the dict to Pydantic; it will produce a clear error.
    return v


class ResultConfig(StrictBaseModel):
    """
    Schema for the consolidated ``result`` task entry.

    Validation rules (initial implementation):
    - At least one of ``patterns`` or ``file`` must be set.
    - ``patterns`` and ``file`` must not be set together (initial release).
    - ``source`` must be one of ``RESULT_SOURCES``.
    """

    patterns: Optional[List[ResultPatternConfig]] = None
    file: Optional[str] = None
    source: str = "log"

    @field_validator("file")
    @classmethod
    def file_must_be_json_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not v.lower().endswith(".json"):
            raise ValueError(
                "result.file must point to a JSON source path ending in '.json'; "
                "for a metric named 'file', use result.patterns"
            )
        return v

    @field_validator("source")
    @classmethod
    def source_must_be_valid(cls, v: str) -> str:
        if v not in RESULT_SOURCES:
            raise ValueError(
                f"result.source must be one of {list(RESULT_SOURCES)} (initial release), got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def check_patterns_or_file(self) -> "ResultConfig":
        if not self.patterns and not self.file:
            raise ValueError(
                "result must contain either 'patterns', 'file', or simple "
                "'name: regex' map entries"
            )
        if self.patterns and self.file:
            raise ValueError(
                "result.patterns and result.file are mutually exclusive in this release"
            )
        if self.patterns:
            seen: set[str] = set()
            duplicates: set[str] = set()
            for pattern in self.patterns:
                if pattern.name in seen:
                    duplicates.add(pattern.name)
                seen.add(pattern.name)
            if duplicates:
                names = ", ".join(sorted(repr(name) for name in duplicates))
                raise ValueError(f"duplicate result.patterns name(s): {names}")
        return self


class ResourceReleaseAfter(str, Enum):
    """When a task-level resource reservation can be reused."""

    WORKFLOW_COMPLETION = "workflow_completion"
    TASK_READY = "task_ready"
    TASK_COMPLETION = "task_completion"


class UploadSpec(BaseModel):
    """Per-task upload specification: copy a local file (or glob) to a named storage target."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    target: str
    # `from` is a Python keyword, so use an alias.
    from_: str = Field(alias="from", min_length=1)
    to: Optional[str] = Field(default=None, min_length=1)
    on_error: Literal["warn", "fail"] = "warn"

    @model_validator(mode="after")
    def check_glob_to_compat(self) -> "UploadSpec":
        # If `from` is a glob and a literal `to:` is given, require trailing `/`.
        # A literal `to:` against multiple matches would silently re-namespace
        # output as `to/basename` on one run vs `to` on another.
        # Skip when `from` is an unresolved expression (resolution happens later).
        if is_expression(self.from_) or self.to is None:
            return self
        has_glob = any(ch in self.from_ for ch in ("*", "?", "["))
        if has_glob and not self.to.endswith("/"):
            raise ValueError(
                "upload 'to' must end with '/' when 'from' contains a glob "
                f"pattern (got from='{self.from_}', to='{self.to}'). "
                "A literal 'to' against multiple glob matches has run-to-run "
                "ambiguous semantics; use a trailing '/' to anchor the "
                "destination as a directory."
            )
        return self


class WorkflowUploadConfig(StrictBaseModel):
    """Workflow-level upload: bundle the entire workflow output dir as a zip and upload."""

    target: str
    # Remote key for the uploaded zip. May contain ${{ workflow.* }} / ${{ variables.* }}.
    # Defaults to `${{ workflow.run_id }}.zip` (resolved at run end).
    to: Optional[str] = Field(default=None, min_length=1)
    on_error: Literal["warn", "fail"] = "warn"


class NodeResourceConfig(StrictBaseModel):
    """Node resource configuration for a task."""

    indices: Optional[Union[List[Resolvable[int]], str]] = None  # Can be [0, 1], ["${{ ... }}"], or "${{ ... }}" resolving to a list
    count: Optional[Resolvable[int]] = None  # Can be int or expression
    exclude: Optional[Union[List[Resolvable[int]], Resolvable[int]]] = None
    #: ``None`` means "placement only": the task is put on these nodes but does not
    #: claim them, so other tasks may share the node. Naming a policy is what turns
    #: the node into a reservation that blocks others -- see
    #: :class:`~sflow.app.resource_planner.NodeReservationPlanner`.
    #:
    #: The absence of a policy has to be representable *in the data*, not merely as
    #: "the field was never assigned": a config that is dumped and loaded again --
    #: which ``sflow compose`` does, and ``sflow batch`` does for its dry run when
    #: ``--nodes`` / ``--gpus-per-node`` re-plan the backends -- would otherwise come
    #: back with every default made explicit, turning every node-pinned task into a
    #: claimant and failing plans that are perfectly valid from source.
    release_after: Optional[ResourceReleaseAfter] = None

    @field_validator("indices")
    @classmethod
    def indices_must_be_list_or_expression(cls, v: Any) -> Any:
        if isinstance(v, str) and not is_expression(v):
            raise ValueError(
                "resources.nodes.indices must be a list or an expression that resolves to a list"
            )
        return v


class GpuResourceConfig(StrictBaseModel):
    """GPU resource configuration for a task.

    ``count`` alone is index-agnostic: the planner packs the task into any
    contiguous idle GPU run on a single node. ``indices`` pins the task to
    specific device indices instead, and the two combine -- ``count`` is then the
    total across nodes and ``indices`` the per-node slice, so a task fans out
    over ``count / len(indices)`` nodes.

    Unlike ``resources.nodes.indices``, GPU indices must be non-negative:
    ``CUDA_VISIBLE_DEVICES`` is a literal device list, and ``-1`` would mean a
    different physical device on nodes with different ``gpus_per_node``.
    """

    count: Optional[Resolvable[int]] = None  # int or "${{ variables.GPU_COUNT }}"
    indices: Optional[Union[List[Resolvable[int]], str]] = None  # [0, 1] or "${{ ... }}"
    #: ``None`` means "no policy named": GPUs are always reserved (they cannot be
    #: shared), so the planner infers the lifetime from the task instead --
    #: ``workflow_completion`` for a task with readiness probes, which may still be
    #: running after it reports READY, and ``task_completion`` otherwise. Carried in
    #: the data rather than in ``model_fields_set`` for the same reason as
    #: :attr:`NodeResourceConfig.release_after`.
    release_after: Optional[ResourceReleaseAfter] = None

    # mode="before" so bools are still bools here: Pydantic would otherwise
    # coerce them to 0/1 and the check below could never fire.
    @field_validator("indices", mode="before")
    @classmethod
    def indices_must_be_list_or_expression(cls, v: Any) -> Any:
        if isinstance(v, str) and not is_expression(v):
            raise ValueError(
                "resources.gpus.indices must be a list or an expression that resolves to a list"
            )
        if isinstance(v, list):
            if not v:
                raise ValueError("resources.gpus.indices must not be empty")
            # Reject bools before Pydantic coerces them to 0/1: YAML reads
            # `yes`/`no`/`on`/`off` as booleans, so a typo would silently become
            # a valid-looking GPU index.
            bools = [i for i in v if isinstance(i, bool)]
            if bools:
                raise ValueError(
                    f"resources.gpus.indices must be integers, got bool(s) {bools}; "
                    "quote YAML values like 'on'/'no' if you meant a GPU index"
                )
            concrete = [i for i in v if isinstance(i, int)]
            negative = [i for i in concrete if i < 0]
            if negative:
                raise ValueError(
                    f"resources.gpus.indices must be non-negative, got {negative}; "
                    "GPU indices are absolute device ids, not Python-style offsets"
                )
            if len(set(concrete)) != len(concrete):
                raise ValueError(
                    f"resources.gpus.indices contains duplicate index/indices: {v}"
                )
        return v

    @model_validator(mode="after")
    def require_count_or_indices(self) -> "GpuResourceConfig":
        if self.count is None and self.indices is None:
            raise ValueError(
                "resources.gpus requires 'count', 'indices', or both"
            )
        return self


class ResourcesConfig(StrictBaseModel):
    nodes: Optional[NodeResourceConfig] = None
    gpus: Optional[GpuResourceConfig] = None


# ---------------------------------------------------------------------------
# Monitor (hardware resource monitoring) schema
# ---------------------------------------------------------------------------

# Built-in hardware scopes the collector knows how to sample.
MONITOR_BUILTIN_SCOPES = ("cpu", "gpu", "memory", "disk", "network")

# Default sampling interval (ms) when neither monitor nor scope sets one.
DEFAULT_MONITOR_INTERVAL_MS = 5000
# Floor for any sampling interval (ms). Below this the collector spins near-busy,
# burning CPU on the monitored nodes and producing huge logs for little signal.
MIN_MONITOR_INTERVAL_MS = 100
# Default detailed-report formats (both pure stdlib; ``png`` needs matplotlib).
DEFAULT_MONITOR_REPORT_FORMATS: tuple[str, ...] = ("csv", "svg")


def _validate_monitor_interval_ms(value: Optional[int]) -> Optional[int]:
    """Reject sub-``MIN_MONITOR_INTERVAL_MS`` sampling intervals.

    Shared by ``monitor.interval`` and each per-scope ``interval`` override so a
    too-small (or non-positive) value is caught at validation time instead of
    silently clamping to a hot-spin loop at plan/run time.
    """
    if value is not None and value < MIN_MONITOR_INTERVAL_MS:
        raise ValueError(
            f"monitor interval must be >= {MIN_MONITOR_INTERVAL_MS}ms (got {value}); "
            "sub-100ms sampling spins the collector hot and bloats the logs"
        )
    return value


class MonitorScopeConfig(StrictBaseModel):
    """Base config for one built-in monitor scope.

    ``interval`` overrides the parent ``monitor.interval`` for this scope (ms).

    Note: monitor intervals are concrete ints, not ``${{ }}`` expressions --
    monitor fields are not run through the expression resolver.
    """

    enabled: bool = True
    interval: Optional[int] = None

    @field_validator("interval")
    @classmethod
    def interval_must_meet_floor(cls, v: Optional[int]) -> Optional[int]:
        return _validate_monitor_interval_ms(v)


class MonitorGpuScopeConfig(MonitorScopeConfig):
    # Override the `nvidia-smi --query-gpu` field list for the gpu scope. The
    # first field MUST be ``index`` so per-GPU rows key on the integer GPU id the
    # report filters on, and the post-processor maps every field to a metric.
    fields: Optional[str] = None

    @field_validator("fields")
    @classmethod
    def fields_must_lead_with_index(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        tokens = [t.strip() for t in v.split(",") if t.strip()]
        if not tokens:
            raise ValueError("monitor.scopes.gpu.fields cannot be empty")
        if tokens[0].lower() != "index":
            raise ValueError(
                "monitor.scopes.gpu.fields must start with 'index' (the GPU id the "
                f"report keys on); got {tokens[0]!r}"
            )
        return v


class MonitorCustomScopeConfig(StrictBaseModel):
    """User-defined monitor commands (Jinja2-resolvable like ``tasks.script``)."""

    script: List[str]

    @field_validator("script")
    @classmethod
    def script_must_not_be_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("monitor.scopes.custom.script cannot be empty")
        return v


class MonitorScopesConfig(StrictBaseModel):
    """Which hardware scopes a monitor collects.

    When the whole ``scopes`` block is omitted, all built-in scopes are active.
    When provided, only the listed scopes (plus ``custom``) are active.
    """

    cpu: Optional[MonitorScopeConfig] = None
    gpu: Optional[MonitorGpuScopeConfig] = None
    memory: Optional[MonitorScopeConfig] = None
    disk: Optional[MonitorScopeConfig] = None
    network: Optional[MonitorScopeConfig] = None
    custom: Optional[MonitorCustomScopeConfig] = None

    def active_builtin_scopes(self) -> list[str]:
        """Return the built-in scopes explicitly enabled in this block."""
        active: list[str] = []
        for name in MONITOR_BUILTIN_SCOPES:
            scope = getattr(self, name)
            if scope is not None and scope.enabled:
                active.append(name)
        return active

    def has_any_active(self) -> bool:
        return bool(self.active_builtin_scopes()) or self.custom is not None


class MonitorResourcesConfig(StrictBaseModel):
    """Which hardware a monitor targets.

    Same slicing logic as ``tasks.resources`` (filter by claimed nodes, then by
    GPU count), plus ``used_by_tasks`` to monitor the resources assigned to other
    workflow tasks (merged across all referenced tasks).
    """

    nodes: Optional[NodeResourceConfig] = None
    gpus: Optional[GpuResourceConfig] = None
    used_by_tasks: Optional[List[str]] = None


class MonitorReportConfig(StrictBaseModel):
    """Detailed post-run report for a monitor consumer. On by default.

    Defaults to CSV + SVG, both produced with only the Python standard library
    (no third-party deps). ``png`` is also available but requires matplotlib
    (the optional ``sflow[monitor]`` extra).

    Set ``enabled: false`` to opt OUT. Worth doing on large fan-outs: a report
    folder is a per-view copy of the samples, so the cost scales with
    ``samples x views``, not with samples.
    """

    enabled: bool = True
    format: List[Literal["csv", "svg", "png"]] = Field(
        default_factory=lambda: list(DEFAULT_MONITOR_REPORT_FORMATS)
    )


def _normalize_monitor_log_boundary(value: Any) -> Any:
    return {"pattern": value} if isinstance(value, str) else value


class MonitorLogBoundaryConfig(StrictBaseModel):
    pattern: Union[str, List[str]]
    select: Optional[Literal["first", "last"]] = None

    @field_validator("pattern")
    @classmethod
    def pattern_must_be_valid(
        cls, value: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        patterns = [value] if isinstance(value, str) else value
        if not patterns:
            raise ValueError("monitor.window pattern cannot be empty")
        for pattern in patterns:
            body = pattern
            for prefix in ("regex:", "re:"):
                if pattern.startswith(prefix):
                    body = pattern[len(prefix) :]
                    try:
                        re.compile(body)
                    except re.error as exc:
                        raise ValueError(
                            f"invalid monitor.window regex {pattern!r}: {exc}"
                        ) from exc
                    break
            # Blank (or blank-bodied "re:") matches every line, which silently
            # selects an arbitrary boundary instead of failing.
            if not body.strip():
                raise ValueError("monitor.window pattern cannot be empty")
        return value


class MonitorWindowConfig(StrictBaseModel):
    """Task-log markers bounding the reported window."""

    start: Annotated[
        MonitorLogBoundaryConfig, BeforeValidator(_normalize_monitor_log_boundary)
    ]
    end: Annotated[
        MonitorLogBoundaryConfig, BeforeValidator(_normalize_monitor_log_boundary)
    ]


class MonitorConfig(StrictBaseModel):
    """A hardware monitor bound to the workflow or a task."""

    resources: Optional[MonitorResourcesConfig] = None
    scopes: Optional[MonitorScopesConfig] = None
    # Default sampling interval (ms) for built-in scopes without their own interval.
    # Concrete int only (monitor fields are not expression-resolved).
    interval: int = DEFAULT_MONITOR_INTERVAL_MS
    # Always present: declaring `monitor:` means you want its report. Opt out
    # with `report: {enabled: false}` (raw samples + the overview are written
    # either way).
    report: MonitorReportConfig = Field(default_factory=MonitorReportConfig)
    window: Optional[MonitorWindowConfig] = None

    @field_validator("interval")
    @classmethod
    def interval_must_meet_floor(cls, v: int) -> int:
        return _validate_monitor_interval_ms(v)  # type: ignore[return-value]

    @model_validator(mode="after")
    def check_scopes_active(self) -> "MonitorConfig":
        if self.scopes is not None and not self.scopes.has_any_active():
            raise ValueError(
                "monitor.scopes is set but no scope is active; enable at least one "
                f"built-in scope ({', '.join(MONITOR_BUILTIN_SCOPES)}) or add scopes.custom"
            )
        if self.window is not None and not self.report.enabled:
            raise ValueError("monitor.window requires report.enabled: true")
        return self


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


class TaskPortConfig(StrictBaseModel):
    """A service port a task exposes."""

    port: Resolvable[int]
    name: Optional[str] = None


class TaskConfig(StrictBaseModel):
    """Configuration for a single task."""

    name: str
    operator: Optional[Union[str, TaskOperatorOverrideConfig]] = None
    backend: Optional[Union[str, Dict[str, Any]]] = None  # Name or inline override
    script: List[str]
    resources: Optional[ResourcesConfig] = None
    ports: Optional[List[TaskPortConfig]] = None

    @field_validator("script")
    @classmethod
    def script_must_not_be_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Script list cannot be empty")
        return v

    # Fail-fast for the task's shell script. When true, sflow prepends ``set -e`` so a
    # failed command (a failed ``pip install``, a benchmark that errored, a server that
    # never launched) exits the task non-zero instead of being masked by a later
    # successful command (classically a trailing ``echo "done"`` -> exit 0). Applies to
    # shell operators only (never ``python``, whose script is Python source).
    # ``None`` (unset, the default) takes the BACKEND's default: **true on Kubernetes**
    # (a failed command in a pod should fail the task) and **false everywhere else**
    # (Slurm/local/docker unchanged). Set explicitly (``fail_fast: true``/``false``) to
    # override the backend default per task.
    fail_fast: Optional[bool] = None

    probes: Optional[ProbesConfig] = None
    outputs: Optional[List[OutputConfig]] = None
    # New consolidated result entry; accepts either:
    # - a simple map (str -> regex)
    # - a ResultConfig object with `patterns` and/or `file`
    # See docs/developer/dev-notes/result-parsing.md for details.
    result: Optional[
        Annotated[ResultConfig, BeforeValidator(_normalize_result_entry)]
    ] = None
    uploads: Optional[List[UploadSpec]] = None
    depends_on: Optional[List[str]] = None
    # Reverse dependency pointer: names of downstream tasks that must run AFTER
    # this one. `A required_by [B]` is equivalent to `B depends_on [A]` and is
    # folded into the targets' `depends_on` at load time. Targets absent from the
    # merged workflow are skipped, so modular fragments compose without cross-file
    # references (no --missable-tasks needed in the forward direction).
    required_by: Optional[List[str]] = None
    replicas: Optional[ReplicaConfig] = None
    retries: Optional[RetryConfig] = None
    # NOT ENFORCED. Accepted so existing recipes keep loading (this model forbids
    # extra keys, so removing it would reject every config that sets it) and so
    # multi-file merge can carry it, but nothing reads it: no code path passes it
    # to Orchestrator._launch_task_with_timeout, and TaskStatus.TIMEOUT is never
    # assigned. Bound a task with the backend's own limit (Slurm `--time`) until
    # this is wired. load_config WARNs when it is set.
    timeout: Optional[Union[int, str]] = None
    variables: Optional[
        Annotated[List[VariableConfig], BeforeValidator(_normalize_to_list)]
    ] = None
    # Hardware monitor bound to this task (starts with the task, stops when the
    # task's process exits / at workflow teardown).
    monitor: Optional[MonitorConfig] = None


class WorkflowConfig(StrictBaseModel):
    """Configuration for the workflow execution."""

    name: str
    # NOT ENFORCED -- see TaskConfig.timeout above. A workflow declaring
    # `timeout: 115m` today runs unbounded.
    timeout: Optional[Union[str, int]] = None
    variables: Optional[
        Annotated[List[VariableConfig], BeforeValidator(_normalize_to_list)]
    ] = None
    tasks: Annotated[List[TaskConfig], BeforeValidator(_normalize_to_list)]
    # Optional: zip the entire workflow output dir and upload to a storage target
    # after the orchestrator finishes (independent of per-task `uploads:`).
    upload_all: Optional[WorkflowUploadConfig] = None
    # Workflow-level hardware monitor (covers the whole pool by default; runs for
    # the full workflow lifetime).
    monitor: Optional[MonitorConfig] = None

    @field_validator("monitor", mode="before")
    @classmethod
    def monitor_must_not_use_log_window(cls, value: Any) -> Any:
        window = (
            value.get("window")
            if isinstance(value, dict)
            else getattr(value, "window", None)
        )
        if window is not None:
            raise ValueError(
                "workflow.monitor.window is not supported; configure it on a task monitor"
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def _fold_required_by(cls, data: Any) -> Any:
        """Fold each task's ``required_by`` into the targets' ``depends_on``.

        ``A required_by [B]`` means B runs after A -- i.e. it is equivalent to
        ``B depends_on [A]``. Targets absent from the (already-merged) task set
        are skipped, so modular fragments compose without cross-file references.
        Runs before ``check_dependencies`` (mode="after"), which then validates
        the resulting ``depends_on``.
        """
        if not isinstance(data, dict):
            return data
        raw_tasks = _normalize_to_list(data.get("tasks"))
        if not isinstance(raw_tasks, list) or not raw_tasks:
            return data

        # Operate on shallow copies so we never mutate the caller's task dicts: this
        # validator's input is the merged config dict, which may be reused elsewhere
        # (e.g. `sflow compose` serializes the merged dict). Only the top-level
        # ``depends_on`` list is rewritten (always to a fresh list), so a shallow copy
        # per task is sufficient.
        tasks = [dict(t) if isinstance(t, dict) else t for t in raw_tasks]

        by_name: Dict[str, Any] = {
            t["name"]: t
            for t in tasks
            if isinstance(t, dict) and t.get("name") is not None
        }

        for t in tasks:
            if not isinstance(t, dict):
                continue
            src = t.get("name")
            required_by = t.get("required_by")
            if not src or not required_by:
                continue
            for target in required_by:
                if target == src:
                    continue  # self-reference would create a cycle
                target_task = by_name.get(target)
                if target_task is None:
                    _logger.debug(
                        "required_by: task '%s' lists absent target '%s'; skipping",
                        src,
                        target,
                    )
                    continue
                deps = list(target_task.get("depends_on") or [])
                if src not in deps:
                    deps.append(src)
                target_task["depends_on"] = deps

        data = dict(data)
        data["tasks"] = tasks
        return data

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
                    probes = getattr(task.probes, probe_type)
                    if probes is None:
                        continue
                    probe_list = probes if isinstance(probes, list) else [probes]
                    for probe in probe_list:
                        if (
                            probe.log_watch
                            and probe.log_watch.logger
                            and probe.log_watch.logger not in task_names
                        ):
                            raise ValueError(
                                f"Task '{task.name}' {probe_type} probe refers to unknown task '{probe.log_watch.logger}'"
                            )

            # Check task-level monitor used_by_tasks references.
            if task.monitor and task.monitor.resources:
                for ref in task.monitor.resources.used_by_tasks or []:
                    if ref not in task_names:
                        raise ValueError(
                            f"Task '{task.name}' monitor.resources.used_by_tasks "
                            f"refers to unknown task '{ref}'"
                        )

        # Check workflow-level monitor used_by_tasks references.
        if self.monitor and self.monitor.resources:
            for ref in self.monitor.resources.used_by_tasks or []:
                if ref not in task_names:
                    raise ValueError(
                        f"workflow.monitor.resources.used_by_tasks refers to "
                        f"unknown task '{ref}'"
                    )
        return self


class SflowConfig(StrictBaseModel):
    """
    Main configuration model for Sflow.
    """

    # Optional. Only "0.1" has ever existed, so requiring it bought nothing but a
    # line of boilerplate at the top of every recipe. Omitting it means "0.1".
    version: str = "0.1"
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
    storage: Optional[
        Annotated[List[StorageConfig], BeforeValidator(_normalize_to_list)]
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

    @model_validator(mode="after")
    def check_storage(self) -> "SflowConfig":
        """Validate storage target names and upload references."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for storage in self.storage or []:
            if storage.name in seen:
                duplicates.add(storage.name)
            seen.add(storage.name)
        if duplicates:
            raise ValueError(
                f"Duplicate storage target names found: {sorted(duplicates)}"
            )

        target_names = {s.name for s in (self.storage or [])}
        for task in self.workflow.tasks:
            if not task.uploads:
                continue
            for spec in task.uploads:
                if spec.target not in target_names:
                    raise ValueError(
                        f"Task '{task.name}' upload references unknown storage target "
                        f"'{spec.target}'. Declared targets: {sorted(target_names) or 'none'}"
                    )
        if self.workflow.upload_all is not None:
            if self.workflow.upload_all.target not in target_names:
                raise ValueError(
                    f"workflow.upload_all references unknown storage target "
                    f"'{self.workflow.upload_all.target}'. "
                    f"Declared targets: {sorted(target_names) or 'none'}"
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

    @field_validator("storage", mode="before")
    @classmethod
    def storage_must_match_registered_configs(cls, v: Any) -> Any:
        if v is None:
            return None

        v = _normalize_to_list(v)
        ensure_builtin_storage_registered()
        adapter = storage_config_type_adapter()

        if not isinstance(v, list):
            raise TypeError("storage must be a list or dict")

        out = []
        for item in v:
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
            item_type = (
                item.get("type")
                if isinstance(item, dict)
                else getattr(item, "type", None)
            )
            if item_type == "docker":
                raise ValueError(
                    "Operator type 'docker' is not valid; use `type: docker_run` for the "
                    "Docker launch operator. (The Docker backend is `type: docker`, but its "
                    "operator is `type: docker_run`.)"
                )
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

        planning_node_count = getattr(backend, "planning_node_count", None)
        total_nodes = _try_resolve_int(
            planning_node_count() if callable(planning_node_count) else None
        )
        if total_nodes is None:
            continue

        raw_list = exclude if isinstance(exclude, list) else [exclude]
        for idx_val in raw_list:
            idx = _try_resolve_int(idx_val)
            if idx is None:
                continue
            resolved_idx = idx if idx >= 0 else idx + total_nodes
            if resolved_idx < 0 or resolved_idx >= total_nodes:
                raise ValueError(
                    f"Task '{task.name}' resources.nodes.exclude contains index "
                    f"{idx} out of range for {total_nodes} allocated node(s) "
                    f"(valid: {-total_nodes}..{total_nodes - 1})"
                )
