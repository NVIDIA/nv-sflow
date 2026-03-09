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
workflow: ...
```

This page follows the **current schema in code** (`src/sflow/config/schema.py`).

## version

Currently supported:

```yaml
version: "0.1"
```

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
- Values use simple type inference (int/float/bool/string).

## artifacts

`artifacts` are “named resources” you can reference via `${{ artifacts.NAME.path }}` in expressions.

In v0.1, the code resolves `uri` to a local `path` only for:

- `fs://<path>`
- `file://<path>`

Other schemes are kept as-is (`path` remains the URI string). **No automatic download/pull** happens today.

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
    extra_args:
      - "--gpus-per-node=8"
```

If you are already inside a Slurm allocation (e.g. via `salloc` or `sbatch`), you can use:

```bash
sflow run --file sflow.yaml
```

This skips `salloc` and attempts to infer node info from the current environment (`SLURM_JOB_ID/SLURM_JOB_NODELIST`).

## operators

An operator defines how a task is launched. Two common ones:

- `bash`: run locally via bash
- `srun`: run via Slurm `srun` (supports common Pyxis `--container-*` flags)

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
