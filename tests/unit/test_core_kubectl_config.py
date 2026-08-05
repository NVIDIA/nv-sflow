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


def test_compute_domain_overrides_are_not_global_args_but_mark_non_empty():
    # --kube-compute-domain-* override backend config, not kubectl global flags, so
    # they stay out of global_args() but still make the config non-empty (so it is
    # applied). Tri-state: create=False is a real override, not "unset".
    cfg = KubectlConfig(compute_domain_channel="auto")
    assert cfg.global_args() == []
    assert cfg.is_empty() is False

    cfg_create_off = KubectlConfig(compute_domain_create=False)
    assert cfg_create_off.global_args() == []
    assert cfg_create_off.is_empty() is False


def test_skip_pvc_is_not_a_global_arg_but_marks_non_empty():
    # --kube-skip-pvc drops PVC-backed backend volumes for the run; it is a backend
    # override, not a kubectl global flag, so it stays out of global_args() but still
    # makes the config non-empty (so it is applied).
    cfg = KubectlConfig(skip_pvc=True)
    assert cfg.global_args() == []
    assert cfg.is_empty() is False


def test_node_filters_are_not_kubectl_config_fields():
    # Node include/exclude now flow through backend config (BackendConfig fields /
    # the --include-nodes / --exclude-nodes flags), not KubectlConfig.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(KubectlConfig)}
    assert "exclude_nodes" not in field_names
    assert "include_nodes" not in field_names


def test_apply_args_are_not_global_args_but_mark_non_empty():
    # --extra-kubectl-apply-args are flags for the `apply` SUBCOMMAND. kubectl only
    # accepts them AFTER the verb, so they must stay out of global_args() (where they
    # would be rejected as unknown flags on every call) while still making the config
    # non-empty so it gets applied to the backend.
    cfg = KubectlConfig(apply_args=["--validate=false"])
    assert cfg.global_args() == []
    assert cfg.is_empty() is False
    assert cfg.kubectl_apply_args() == ["--validate=false"]


def test_apply_args_are_shell_split_like_other_extra_args():
    # A bundled entry must not reach kubectl as one unparsable argv token.
    cfg = KubectlConfig(apply_args=["--validate=false --server-side"])
    assert cfg.kubectl_apply_args() == ["--validate=false", "--server-side"]


def test_apply_args_and_global_args_stay_separate():
    cfg = KubectlConfig(extra_args=["--insecure-skip-tls-verify"], apply_args=["--validate=false"])
    assert cfg.global_args() == ["--insecure-skip-tls-verify"]
    assert cfg.kubectl_apply_args() == ["--validate=false"]
