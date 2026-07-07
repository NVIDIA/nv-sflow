# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sflow.core.kubectl_config import KubectlConfig


def test_empty_config_has_no_global_args():
    cfg = KubectlConfig()
    assert cfg.global_args() == []
    assert cfg.is_empty() is True


def test_global_args_maps_kubeconfig_context_and_passthrough():
    cfg = KubectlConfig(
        kubeconfig="/home/me/.kube/prod",
        context="prod-east",
        extra_args=["--insecure-skip-tls-verify", "--request-timeout=30s"],
    )
    assert cfg.global_args() == [
        "--kubeconfig",
        "/home/me/.kube/prod",
        "--context",
        "prod-east",
        "--insecure-skip-tls-verify",
        "--request-timeout=30s",
    ]
    assert cfg.is_empty() is False


def test_namespace_excluded_from_global_args():
    # namespace overrides the backend default and is threaded via --namespace
    # separately, so it must not appear in the global kubectl flags.
    cfg = KubectlConfig(namespace="team-ns")
    assert cfg.global_args() == []
    assert cfg.is_empty() is False


def test_node_filters_are_not_kubectl_config_fields():
    # Node include/exclude now flow through backend config (BackendConfig fields /
    # the --include-nodes / --exclude-nodes flags), not KubectlConfig.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(KubectlConfig)}
    assert "exclude_nodes" not in field_names
    assert "include_nodes" not in field_names
