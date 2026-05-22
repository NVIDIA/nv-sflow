# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sflow.resolution import resolve_variables_inline


def test_resolve_variables_inline_partially_resolves_deferred_backend_expression():
    merged = {
        "version": "0.1",
        "variables": [
            {"name": "INFRA_NODE_INDEX", "value": 0},
            {"name": "MODEL_NAME", "value": "llama"},
        ],
        "workflow": {
            "name": "wf",
            "variables": [
                {
                    "name": "HEAD_NODE_IP",
                    "value": "${{ backends.slurm_cluster.nodes[0].ip_address if variables.INFRA_NODE_INDEX == 0 else backends.slurm_cluster.nodes[-1].ip_address }}",
                },
            ],
            "tasks": [
                {
                    "name": "server",
                    "script": [
                        "serve --model ${MODEL_NAME}",
                        "echo ${{ variables.MODEL_NAME }}",
                        "echo ${CUDA_VISIBLE_DEVICES}",
                    ],
                },
            ],
        },
    }

    resolved = resolve_variables_inline(merged)

    assert "variables" not in resolved
    wf_vars = {
        entry["name"]: entry["value"] for entry in resolved["workflow"]["variables"]
    }
    assert wf_vars["HEAD_NODE_IP"] == (
        "${{ backends.slurm_cluster.nodes[0].ip_address if 0 == 0 else "
        "backends.slurm_cluster.nodes[-1].ip_address }}"
    )
    assert resolved["workflow"]["tasks"][0]["script"] == [
        "serve --model llama",
        "echo llama",
        "echo ${CUDA_VISIBLE_DEVICES}",
    ]


def test_resolve_variables_inline_keeps_replica_variables_and_resolves_domains():
    merged = {
        "version": "0.1",
        "variables": [
            {"name": "CONCURRENCY", "value": 64, "domain": [64, 128]},
            {"name": "OTHER", "value": 99},
        ],
        "workflow": {
            "name": "wf",
            "tasks": [
                {
                    "name": "bench",
                    "script": [
                        "run --concurrency ${CONCURRENCY}",
                        "domain=${{ variables.CONCURRENCY.domain }}",
                        "other=${{ variables.OTHER }}",
                    ],
                    "replicas": {
                        "variables": ["CONCURRENCY"],
                        "policy": "sequential",
                    },
                },
            ],
        },
    }

    resolved = resolve_variables_inline(merged)

    var_names = [entry["name"] for entry in resolved["variables"]]
    assert var_names == ["CONCURRENCY"]
    assert resolved["workflow"]["tasks"][0]["script"] == [
        "run --concurrency ${CONCURRENCY}",
        "domain=[64, 128]",
        "other=99",
    ]
