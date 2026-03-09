---
title: Operators
sidebar_position: 6
---

An `operator` defines **how a task is launched** (locally, via `srun`, inside a container, etc.).

If a task does not set `operator: ...`, `sflow` chooses a backend-specific default:

- local backend → `bash`
- slurm backend → `srun`

## Define an operator and reference it from tasks

This is based on `tests/integration/guide/sflow_http_echo_slurm_container.yaml`:

```yaml
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: ${{ variables.SLURM_ACCOUNT }}
    partition: ${{ variables.SLURM_PARTITION }}
    time: ${{ variables.SLURM_TIME }}
    nodes: ${{ variables.SLURM_NODES }}

operators:
  - name: slurm_container_py
    type: srun
    container_image: python:3.13-slim
    container_name: slurm_container_py
    container_writable: true
    container_mount_home: false

workflow:
  name: http_echo_slurm_container
  tasks:
    - name: echo_server
      operator: slurm_container_py
      script:
        - python -c 'print("server")'
```

## Task-level operator overrides (deeper)

You can also use the object form to override operator settings for a single task:

```yaml
operator:
  name: slurm_container_py
  # operator-specific overrides go here (for srun: ntasks, nodes, extra_args, ...)
```
