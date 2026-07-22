# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Kubernetes support primitives for sflow.

These modules are the backend-agnostic building blocks used by *both* the
Kubernetes backend (``plugins/backends/kubernetes.py``) and the K8s operators
(``plugins/operators/k8s.py``, ``k8s_mpi.py``): declarative manifest rendering,
client-side shell builders, driver-side pod/MPIJob lifecycle, cluster capability
detection, RDMA detection + runtime affinity, and the in-cluster probe transport.

Keeping them here (rather than under ``operators``) removes the former
backend->operators cross-import that only existed to share these helpers.
"""
