---
title: Configuration
sidebar_position: 10
---

`sflow` uses a YAML config file (default name: `sflow.yaml`). Top-level structure:

```yaml
version: "0.1"
variables: ...
artifacts: ...
backends: ...
operators: ...
storage: ...     # optional: post-run upload targets (S3), paired with task `uploads:`
workflow: ...
```

This page follows the **current schema in code** (`src/sflow/config/schema.py`).

:::tip
Looking for a quick lookup of all config fields? See the [Quick Reference](./quick-reference.md) table.
:::

## version

Optional. Omit it and sflow uses `"0.1"`, the only value that has ever existed:

```yaml
version: "0.1"   # optional
```

If you do declare it, it must be `"0.1"` — any other value is rejected. When
several files are merged, they must not declare conflicting versions.

## variables

Variables can be written as a **dict** or a **list** (they are normalized internally).

Dict form (recommended):

```yaml
variables:
  SLURM_PARTITION:
    description: "Slurm partition"
    value: debug
```

List form:

```yaml
variables:
  - name: SLURM_PARTITION
    description: "Slurm partition"
    value: debug
```

How to use:

- In YAML expressions: `${{ variables.SLURM_PARTITION }}`
- In task scripts (as env var): `${SLURM_PARTITION}`

### Override variables via CLI (--set)

```bash
sflow run --file sflow.yaml --set SLURM_PARTITION=debug --set NUM_GPUS=4
```

Notes:

- `--set` can **only override variables that already exist** in the config; otherwise it errors.
- Values use simple type inference (int/float/bool/list/string).
- List values set the variable domain for replica sweeps, and the variable value becomes the first item.

You can also read a variable's domain inside expressions:

```yaml
script:
  - echo "all concurrencies=${{ variables.CONCURRENCY.domain }}"
  - echo "max concurrency=${{ variables.CONCURRENCY.domain | max }}"
```

## artifacts

`artifacts` are “named resources” you can reference via `${{ artifacts.NAME.path }}` in expressions.

The `uri` scheme determines how an artifact resolves to `path`:

- `fs://<path>` / `file://<path>`: resolved to a local filesystem path (relative paths are relative to the workspace).
- `http://` / `https://`: **downloaded to a local cache**, and `path` is set to the cached file.
- `s3://` (and any other unregistered scheme): passthrough — no local materialization, so `${{ artifacts.NAME.path }}` falls back to the `uri` string itself.
- `hf://` / `huggingface://` and `docker://`: the scheme is registered, but materialization is **not yet implemented** and raises `NotImplementedError` (see [Known Limitations](./intro.md#known-limitations)). Reference a local `file://`/`fs://` path, or pull the model/image inside your task for now.

Example:

```yaml
artifacts:
  model_dir:
    uri: fs://./models/qwen
```

### Override artifacts via CLI (--artifact)

```bash
sflow run --file sflow.yaml --artifact model_dir=fs:///mnt/models/qwen
```

Same requirement: the artifact must already be defined in `artifacts`, otherwise it errors.

## backends

sflow ships four backend types — `local`, `slurm`, `docker`, and `kubernetes`. Every
backend shares a few common fields, including `gpus_per_node` (GPU capacity per node for
planning/packing) and the node filters `include_nodes` / `exclude_nodes` (also settable
via the `--include-nodes` / `--exclude-nodes` CLI flags). This page is a summary; see
[Backends](./backends.md) for the full field reference for each type.

### local backend

```yaml
backends:
  local:
    type: local
    default: true
    nodes: 1
```

### slurm backend

```yaml
backends:
  slurm_cluster:
    type: slurm
    default: true
    account: ${{ variables.SLURM_ACCOUNT }}
    partition: ${{ variables.SLURM_PARTITION }}
    time: 00:30:00
    nodes: 2
    gpus_per_node: 8        # sflow planning only
    extra_args:
      - "--gpus-per-node=8" # passed to salloc
```

`gpus_per_node` tells sflow how many GPU indices each node has for planning and
packing. It does not add `--gpus-per-node` to `salloc`; include that flag in
`extra_args` when your cluster requires it.

The Slurm backend requires `account`, `partition`, `time`, `nodes`, and `gpus_per_node`
(set `gpus_per_node: 0` for CPU-only partitions).

If you are already inside a Slurm allocation (e.g. via `salloc` or `sbatch`), you can use:

```bash
sflow run --file sflow.yaml
```

This skips `salloc` and attempts to infer node info from the current environment (`SLURM_JOB_ID/SLURM_JOB_NODELIST`).

### docker backend

The `docker` backend runs each task in a container via the `docker_run` operator. Set
`image` and `gpus_per_node`; an optional `hosts:` pool spreads tasks across remote Docker
daemons. See [Backends](./backends.md#docker-backend).

### kubernetes backend

The `kubernetes` backend reserves real nodes and runs each task as scheduler-placed
pod(s). The backend carries cluster/access config (`namespace`, `nodes`, `gpus_per_node`,
`scheduling`); the workload **image lives on a `k8s` operator**, not the backend. Cluster
selection and credentials are `sflow run` CLI flags, not YAML. See
[Backends](./backends.md#kubernetes-backend).

## operators

An operator defines how a task is launched. sflow ships seven operator types:

- `bash`: run locally via bash
- `srun`: run via Slurm `srun` (supports common Pyxis `--container-*` flags)
- `docker_run`: run inside a container via `docker run`
- `python`: run the script through a Python interpreter
- `ssh`: run on a remote host over SSH
- `k8s`: run as a Kubernetes pod (the workload `image` lives on this operator)
- `k8s_mpi`: run any MPI (`mpirun`) job on Kubernetes — single- or multi-node (write `mpirun` explicitly)

If a task does not set `operator:`, the backend picks a default: local → `bash`,
slurm → `srun`, docker → `docker_run`, kubernetes → none (you must declare a `k8s`
operator). Note the Docker naming: the *backend* is `type: docker`, but its launch
*operator* is `type: docker_run`. See
[Operators](./operators.md) for per-type fields and examples.

Example (bash):

```yaml
operators:
  local_bash:
    type: bash
```

Example (srun + container):

```yaml
operators:
  runtime:
    type: srun
    container_image: nvcr.io/xxx/yyy:tag
    container_mount_home: false
    container_mounts:
      - "/mnt:/mnt:rw"
    extra_args:
      - "--shm-size=64g"
```

## storage

Optional. Declares named post-execution upload targets (S3 today). Each task can then attach
`uploads:` specs, and `workflow.upload_all` can zip and ship the whole run. Credentials come
from the boto3 default chain — never put secrets in YAML (needs the `sflow[s3]` extra).

```yaml
storage:
  - name: results_bucket
    type: s3
    bucket: my-bench-results
    prefix: "runs/${{ variables.RUN_ID }}/"   # optional
```

See [Uploads](./uploads.md) for the full field reference and per-task `uploads:` usage.

## workflow / tasks

Minimal task:

```yaml
workflow:
  name: demo
  tasks:
    - name: step1
      script:
        - echo "hello"
```

### depends_on

```yaml
- name: step2
  depends_on: [step1]
  script:
    - echo "step2"
```

### resources (nodes / GPUs)

```yaml
- name: server
  resources:
    nodes:
      indices: [0]
    gpus:
      count: 4
  script:
    - echo "server on node0, 4 gpus"
```

`resources.nodes` supports `indices`, `count`, and `exclude`. `exclude` removes nodes from the allocation before `indices`, `count`, or GPU packing are applied. `exclude` accepts either a single index or a list, and supports negative indices (counting from the end, e.g. `-1` is the last node); indices are validated against the allocation range (`-total_nodes`..`total_nodes - 1`):

```yaml
- name: workers
  resources:
    nodes:
      exclude: [0]
      count: 2
```

`resources.gpus` takes `count`, `indices`, or both (at least one is required). `count` alone is index-agnostic and packs the task into any contiguous idle GPU run on a single node. `indices` pins specific device ids on the first node where they are all free. With both, `count` is the total across nodes and `indices` the per-node slice, so the task spans `count / len(indices)` nodes using the same slots on each — see [Resources](./resources.md#pin-specific-gpus-with-indices):

```yaml
- name: pinned
  resources:
    gpus:
      count: 8         # total across nodes
      indices: [0, 1]  # -> 4 nodes x GPUs 0,1 on 4-GPU nodes
```

`resources.nodes.release_after` and `resources.gpus.release_after` control when that resource kind can be reused by later tasks in the DAG. Node reservations are only exclusive when `resources.nodes.release_after` is explicitly set; when omitted, both `resources.nodes.indices` and `resources.nodes.count` are placement constraints and may overlap with other planned tasks. GPU reservations infer the safe behavior when omitted: tasks without readiness probes release GPUs after task completion for downstream dependents, while tasks with readiness probes keep GPUs until workflow completion because they may still be running after they become ready. Use `task_ready` when a task can release a resource after its readiness probe succeeds, `task_completion` when the resource can be reused after the task reaches any terminal status (`COMPLETED`, `FAILED`, `TIMEOUT`, or `CANCELLED`), or `workflow_completion` to explicitly reserve it for the whole workflow:

```yaml
- name: check_entire_env
  resources:
    gpus:
      count: 8
  script:
    - nvidia-smi

- name: worker
  depends_on: [check_entire_env]
  replicas:
    count: 4
    policy: parallel
  resources:
    gpus:
      count: 2
```

Policies are independent per resource kind, so a task can release GPUs at readiness while keeping a node reservation until workflow completion. Dry-run performs a scheduler rehearsal for these lifetimes; it validates temporal resource usage rather than only summing all declared resources.

### replicas

Run multiple instances of a task in parallel or sequentially:

```yaml
- name: worker
  replicas:
    count: 4
    policy: parallel
  script:
    - echo "replica $SFLOW_REPLICA_INDEX"
```

See [Replicas](./replicas.md) for details on policies, variable sweeps, and GPU allocation.

### probes (readiness / failure)

Probes are useful for service-style tasks (e.g. start a server, then run a client):

```yaml
- name: api_server
  script:
    - python -m http.server 8000
  probes:
    readiness:
      tcp_port:
        port: 8000
```

`probes.readiness` and `probes.failure` may each be a single probe or a list of probes; when a list is given, every readiness probe must trigger before the task is ready. Probe types include `tcp_port`, `http_get`, `http_post`, and `log_watch`. `probes.failure` marks a running task as failed when its condition is detected, which fail-fast uses to cancel downstream work. `log_watch` patterns match **literally** by default; prefix a pattern with `re:` or `regex:` for a regular expression (see [Probes](./probes.md)).

### Other task fields

Beyond `depends_on`, `resources`, `replicas`, and `probes`, a task supports:

- `operator` / `backend`: name (or inline override object) of the operator or backend for this task.
- `ports`: service ports the task exposes (each with `port` and an optional `name`).
- `timeout`: **accepted but not enforced.** Both the per-task and the workflow `timeout` are parsed and merged, but nothing reads them — a task that sets one runs unbounded, and `sflow run` logs a warning naming every task that does. Bound the run with the backend's own limit instead (Slurm `--time`, a Kubernetes `activeDeadlineSeconds`). Kept so existing recipes keep loading.
- `fail_fast`: bool, **default depends on the backend** — `true` on Kubernetes (a failed command in a pod should fail the task), `false` on Slurm/local/docker (shell default: only the last command's exit code counts). Leave unset to take the backend default, or set explicitly (`true`/`false`) to override per task. When effective-true, sflow prepends `set -e` to shell-operator scripts so any failed command fails the task. Applies to shell operators only (never `python`, whose script is Python source).
- `variables`: task-scoped variables (same format as top-level `variables`).
- `retries`: retry policy (`count`, `interval`, `backoff`) for a failed task.
- `result`: consolidated result parsing that writes `result.json` — see [Results](./results.md).
- `uploads`: copy task outputs to an S3 (or other) storage target — see [Uploads](./uploads.md).
- `monitor`: a hardware monitor bound to the task — see [Monitor](./monitor.md).
- `required_by`: reverse of `depends_on` (`A required_by [B]` ≡ `B depends_on [A]`), handy for modular fragments — see [Modular Workflows](./modular-workflows.md).
