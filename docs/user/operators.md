---
title: Operators
sidebar_position: 6
---

An `operator` defines **how a task is launched** (locally, via `srun`, inside a container, etc.).

If a task does not set `operator: ...`, `sflow` chooses a backend-specific default:

- local backend → `bash`
- slurm backend → `srun`
- docker backend → `docker_run`
- kubernetes backend → no default; declare an explicit `k8s` operator

:::note Docker operator rename
The Docker launch operator is `type: docker_run`. Older YAML that declared an
operator with `type: docker` should be updated to `type: docker_run`. Docker
backend configs still use `type: docker`.
:::

## Define an operator and reference it from tasks

Example with a containerized srun operator:

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

## Slurm runtime environment aliases

The `srun` operator uses `--export=ALL` by default, so task processes inherit
the environment prepared by sflow and by the Slurm controller. For portability
across backends, it also maps common Slurm allocation and rank variables to
`SFLOW_*` aliases inside the launched shell:

| Slurm variable | SFLOW alias |
|----------------|-------------|
| `SLURM_JOB_ID` / `SLURM_JOBID` | `SFLOW_BACKEND_JOB_ID` |
| `SLURM_JOB_NODELIST` / `SLURM_NODELIST` | `SFLOW_BACKEND_NODELIST` |
| `SLURM_NNODES` | `SFLOW_BACKEND_NUM_NODES` |
| `SLURM_STEP_ID` | `SFLOW_BACKEND_STEP_ID` |
| `SLURMD_NODENAME` | `SFLOW_TASK_NODE_NAME` |
| `SLURM_NODEID` | `SFLOW_TASK_NODE_INDEX` |
| `SLURM_PROCID` | `SFLOW_TASK_PROCESS_ID` |
| `SLURM_LOCALID` | `SFLOW_TASK_LOCAL_PROCESS_ID` |
| `SLURM_NTASKS` | `SFLOW_TASK_NUM_PROCESSES` |

Prefer the `SFLOW_*` aliases in task scripts when the script should stay
backend-agnostic. Use raw `SLURM_*` values only for Slurm-specific logic.

## Task-level operator overrides (deeper)

You can also use the object form to override operator settings for a single task:

```yaml
operator:
  name: slurm_container_py
  # operator-specific overrides go here (for srun: ntasks, nodes, extra_args, ...)
```
