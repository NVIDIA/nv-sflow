# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-scoped default for a task's shell fail-fast (``set -e``).

An explicit ``fail_fast`` in the task YAML always wins. When omitted, the default
follows the backend: True on Kubernetes (a failed command in a pod should fail the
task), False elsewhere (Slurm/local/docker unchanged).
"""

from __future__ import annotations

from sflow.app.assembly import _resolve_fail_fast
from sflow.config.schema import TaskConfig
from sflow.core.backend import Backend
from sflow.plugins.backends.kubernetes import KubernetesBackend
from sflow.plugins.backends.slurm import SlurmBackend


def test_task_config_fail_fast_defaults_to_none():
    # None (unset) is what lets the resolver apply the backend default; an explicit
    # bool in YAML is preserved as-is.
    assert TaskConfig(name="t", script=["echo 1"]).fail_fast is None
    assert TaskConfig(name="t", script=["echo 1"], fail_fast=False).fail_fast is False
    assert TaskConfig(name="t", script=["echo 1"], fail_fast=True).fail_fast is True


def test_backend_default_fail_fast_by_type():
    assert Backend.default_fail_fast is False           # base: unchanged default
    assert KubernetesBackend.default_fail_fast is True   # a failed pod command fails
    assert SlurmBackend.default_fail_fast is False        # slurm behavior unchanged


class _Be:
    def __init__(self, default_fail_fast):
        self.default_fail_fast = default_fail_fast


def test_resolve_fail_fast_honors_explicit_value():
    # Explicit YAML value wins regardless of the backend default.
    assert _resolve_fail_fast(True, _Be(False)) is True
    assert _resolve_fail_fast(False, _Be(True)) is False


def test_resolve_fail_fast_falls_back_to_backend_default_when_unset():
    assert _resolve_fail_fast(None, _Be(True)) is True
    assert _resolve_fail_fast(None, _Be(False)) is False


def test_resolve_fail_fast_defaults_false_when_no_backend():
    # A task with no placement/backend -> unchanged (False).
    assert _resolve_fail_fast(None, None) is False
