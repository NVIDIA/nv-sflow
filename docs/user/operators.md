---
title: Operators
sidebar_position: 6
---

An `operator` defines **how a task is launched** — locally via `bash`, on Slurm via
`srun`, inside a container via `docker_run`, over SSH via `ssh`, through a Python
interpreter via `python`, or on Kubernetes via `k8s` / `k8s_mpi`.

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

## Operator matrix

| Operator | Default for backend | How the script runs | Task log handling |
|----------|---------------------|---------------------|-------------------|
| `bash` | `local` | bash shell | offload to `<task>.log` (default) or stream |
| `srun` | `slurm` | bash shell on the compute node | offload to `<task>.log` (default) or stream |
| `docker_run` | `docker` | bash shell inside the container | offload to `<task>.log` (default) or stream |
| `k8s` | — (declare explicitly) | bash shell inside the pod | streamed from the pod via `kubectl` |
| `k8s_mpi` | — (declare explicitly) | `mpirun` launcher inside the pod | streamed from the pod via `kubectl` |
| `ssh` | — | bash shell on the remote host | streamed through the driver |
| `python` | — | Python **source** via `python -c` (not a shell) | streamed through the driver |

**Log offload** (bash/srun/docker_run) writes each task's `<task>.log` on the compute
side and reproduces sflow's log format, keeping large log volumes off the driver. It is on
by default (backend field `offload_task_logs: true`, or per-operator `log_to_file`) and
auto-falls back to live streaming on an interactive TTY or under `--tui`. `ssh` and
`python` do not support offload — they always stream through the driver.

:::caution `python` runs source, not a shell
The `python` operator passes your `script:` lines to `python -c` as Python source, so
shell syntax (pipes, `&&`, env-var expansion) does **not** apply. Use `bash` for shell
scripts.
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

## Other operator types

The `srun` example above shows the container form. The remaining operator types are
summarized below; see the [Quick Reference](./quick-reference.md) for the full field
tables.

### bash

Runs the task script locally through `bash`. It is the default operator for the `local`
backend and needs no configuration.

```yaml
operators:
  - name: local_bash
    type: bash
```

### docker_run

Runs the task in a container via `docker run`. It is the default operator for the
`docker` backend. Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `image` (required) | — | Container image to run. |
| `workdir` | — | Working directory inside the container. |
| `mounts` | — | Extra bind mounts (`host:container[:ro]`). |
| `gpus` | — | GPU request (`all`, `device=0,1`, ...). **Overridden** by the backend's planned CUDA slice when the task declares `resources.gpus` (sflow sets `gpus=device=<assigned>`). |
| `pass_envs` | — | Environment variables to forward into the container. |
| `extra_args` | — | Extra raw `docker run` args. |
| `auto_mount_runtime_dirs` | `true` | Auto-mount the workspace/output dirs into the container. Automatically disabled for remote `docker_host`/`context` hosts (the paths don't exist there). |
| `log_to_file` | `true` | Offload this task's log to `<task>.log` on the container side (see the [Operator matrix](#operator-matrix)). |

```yaml
operators:
  - name: containerized
    type: docker_run
    image: nvcr.io/nvidia/pytorch:24.12-py3
    gpus: all
    mounts:
      - "/data:/data:ro"
```

### python

Runs the task script through a Python interpreter. Key fields: `python_exec` (default
`python`), `extra_args`.

```yaml
operators:
  - name: py
    type: python
    python_exec: python3
```

### ssh

Runs the task script on a remote host over SSH. Key fields: `host` (required), `user`,
`port`, `identity_file`, `extra_args`.

```yaml
operators:
  - name: remote
    type: ssh
    host: gpu-01.example.com
    user: ubuntu
    identity_file: ~/.ssh/id_ed25519
```

### k8s

Runs the task as a Kubernetes pod. The workload **`image` lives on the operator** — the
`kubernetes` backend carries no image of its own. The same operator handles single- and
multi-node tasks: a single-node task is one pod, and a multi-node task (`resources.nodes`)
becomes **one pod per reserved node** (leader = index 0). Common fields: `image`
(required), `image_pull_policy`, `host_network`, `node_selector`, `tty`, `env`,
`security_context`, and `cpu` / `memory` / `cpu_limit` / `memory_limit`. See
[Backends](./backends.md#operator-k8s) for the full pod/container field table.

```yaml
version: "0.1"

backends:
  - name: k8s
    type: kubernetes
    default: true
    namespace: ml-team
    nodes: 1
    gpus_per_node: 8
    scheduling: device_plugin   # default (nvidia.com/gpu); dra is opt-in/WIP

operators:
  - name: server_op
    type: k8s
    image: my/server:1.0

workflow:
  name: k8s_demo
  tasks:
    - name: train
      operator: server_op
      resources: {gpus: {count: 2}}
      script:
        - nvidia-smi -L
```

### k8s_mpi

Runs an **MPI** job on Kubernetes. **Use `k8s_mpi` for any workload launched with `mpirun`**
(TRT-LLM `trtllm-llmapi-launch`, multi-GPU tensor/pipeline-parallel serving, etc.), and write
the `mpirun -np N ... <workload>` command **explicitly** in the task script. From that command
k8s_mpi sets up the correct MPI world for **both single-node and multi-node** — a lone launcher
pod when the job fits one node, or launcher + worker pods across nodes — so the same recipe
scales from `-np 8` (1 node) to `-np 16` (2 nodes) with no SSH/hostfile glue. sflow injects the
SSH keypair, hostfile, and `sshd`/launcher wiring (via the Kubeflow MPI Operator when present,
else plain pods). It inherits every `k8s` operator field and adds an `mpi:` block (`route`,
`slots_per_worker`, `ssh_port`, ...). Reserve plain `k8s` for non-MPI, single-pod workloads.
See [Backends](./backends.md#kubernetes-backend).

```yaml
operators:
  - name: mpi_op
    type: k8s_mpi
    image: my/trtllm:1.0

workflow:
  name: k8s_mpi_demo
  tasks:
    - name: train
      operator: mpi_op
      resources:
        nodes: {count: 2}
        gpus: {count: 16}
      script:
        - mpirun -np 16 python train.py
```

## Task-level operator overrides (deeper)

You can also use the object form to override operator settings for a single task:

```yaml
operator:
  name: slurm_container_py
  # operator-specific overrides go here (for srun: ntasks, nodes, extra_args, ...)
```

Override keys are **strict**: each must be a real field of the operator this task resolves
to. An unknown key — a typo, or a setting meant for a different backend's operator (e.g. an
srun flag on a Kubernetes operator) — fails the run with a clear error rather than being
silently ignored. If a setting should only apply under one backend, put it on that
operator in the backend fragment (operators deep-merge by name per backend); the task
override then stays backend-agnostic.
