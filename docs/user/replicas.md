---
title: Replicas
sidebar_position: 7.5
---

`replicas` allows you to run multiple instances of the same task, either in parallel or sequentially. This is useful for scaling workloads, running parameter sweeps, or executing benchmarks with different configurations.

## Basic Usage

To create multiple identical copies of a task:

```yaml
workflow:
  name: demo
  tasks:
    - name: worker
      replicas:
        count: 4
        policy: parallel
      script:
        - echo "I am replica $SFLOW_REPLICA_INDEX"
```

This creates 4 parallel workers (`worker_0`, `worker_1`, `worker_2`, `worker_3`), each running the same script.

## Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `count` | positive int or `${{ }}` expression | `null` | Number of replicas to create. Has no numeric default — when omitted, the count is taken from the `variables` sweep size (below). `0`/negative are rejected. |
| `policy` | `parallel` \| `sequential` \| `${{ }}` expression | `parallel` | How replicas are scheduled. May be an expression that resolves to `parallel`/`sequential`. |
| `variables` | list of variable names | none | Variables to sweep over (creates a Cartesian product) |

:::note `count` and `variables`
Provide `count`, `variables`, or both. With `variables` only, the replica count equals
the size of the swept `domain`(s). With both, the sizes must match (see [Combining Count
and Variables](#combining-count-and-variables)).
:::

## Replica Naming

Replica task names are generated based on the configuration:

- **With `count` only**: Numeric indices are used: `task_0`, `task_1`, `task_2`, etc.
- **With `variables` sweep**: Variable values are used: `task_16`, `task_32` for single variable, or `task_0_001_32` for multiple variables (values joined with `_`).

Special characters in variable values (`.`, `-`, spaces) are replaced with `_` in task names.

## Environment Variables

Each replica receives these environment variables:

- **`SFLOW_REPLICA_INDEX`**: Zero-based index of the replica (0, 1, 2, ...)
- Any swept variables from the `variables` list

## Replica Fan-Out Summary

| Mode | Config | Resulting replica names | Per-replica env | `CUDA_VISIBLE_DEVICES` |
|------|--------|-------------------------|-----------------|------------------------|
| Count only | `count: 4` | `t_0`, `t_1`, `t_2`, `t_3` | `SFLOW_REPLICA_INDEX` | each replica gets its own contiguous slice of size `resources.gpus.count` |
| Single-var sweep | `variables: [BATCH_SIZE]` with `domain: [16, 32]` | `t_16`, `t_32` | `SFLOW_REPLICA_INDEX` + `BATCH_SIZE` | same per-replica slicing |
| Multi-var Cartesian | `variables: [LR, BATCH_SIZE]` | `t_0_001_32`, `t_0_001_64`, … (values joined with `_`) | `SFLOW_REPLICA_INDEX` + every swept variable | same per-replica slicing |

Per-replica `CUDA_VISIBLE_DEVICES` slicing applies on the **local**, **slurm**, and **docker** backends — replica *N* gets the *N*-th contiguous GPU slice (e.g. `count: 2` → replica 0 sees `0,1`, replica 1 sees `2,3`). On **Kubernetes**, each replica is normally its own pod and the device plugin / DRA assigns its GPUs, so `CUDA_VISIBLE_DEVICES` is left unset — unless sflow merges co-located GPU tasks into one shared pod, where it sets a per-member `CUDA_VISIBLE_DEVICES` to pin each one's slice.

## Replica Policies

### Parallel Policy (default)

All replicas start at the same time and run concurrently:

```yaml
- name: prefill_server
  replicas:
    count: 2
    policy: parallel
  resources:
    gpus:
      count: 1
  script:
    - python -m server --role prefill
```

This launches 2 prefill servers simultaneously, each getting its own GPU slice.

**Dependency behavior with parallel replicas:**
- Downstream tasks wait for **ALL** replicas to complete before starting
- All replicas independently depend on upstream tasks

### Sequential Policy

Replicas run one after another in order. Replica N waits for replica N-1 to complete:

```yaml
- name: benchmark
  replicas:
    count: 3
    policy: sequential
  script:
    - echo "Running benchmark iteration $SFLOW_REPLICA_INDEX"
```

This runs benchmarks one at a time: `benchmark_0` → `benchmark_1` → `benchmark_2`.

**Dependency behavior with sequential replicas:**
- Downstream tasks only wait for the **LAST** replica to complete
- Only the first replica depends on upstream tasks; subsequent replicas form a chain

## Variable Sweeps

The `variables` field enables parameter sweeps. Each variable must have a `domain` defined, and sflow creates replicas for the Cartesian product of all domain values.

:::note `value` must be in `domain`
When a variable declares a `domain`, its `value` must be one of the domain members —
sflow rejects the config otherwise. The `value` acts as the default (e.g. for a non-swept
run or `--set` override); the full `domain` drives the sweep.
:::

### Single Variable Sweep

```yaml
variables:
  BATCH_SIZE:
    value: 32
    domain: [16, 32, 64, 128]

workflow:
  name: sweep
  tasks:
    - name: train
      replicas:
        variables:
          - BATCH_SIZE
        policy: sequential
      script:
        - echo "Training with batch size $BATCH_SIZE"
```

This creates 4 replicas (`train_16`, `train_32`, `train_64`, `train_128`), each with a different `BATCH_SIZE` value.

### Multi-Variable Sweep (Cartesian Product)

```yaml
variables:
  LEARNING_RATE:
    value: 0.001
    domain: [0.001, 0.01]
  BATCH_SIZE:
    value: 32
    domain: [32, 64]

workflow:
  name: hyperparam_sweep
  tasks:
    - name: experiment
      replicas:
        variables:
          - LEARNING_RATE
          - BATCH_SIZE
        policy: parallel
      script:
        - echo "LR=$LEARNING_RATE BS=$BATCH_SIZE"
```

This creates 4 replicas (2×2 Cartesian product), named using the variable values:
- `experiment_0_001_32`: LR=0.001, BS=32
- `experiment_0_001_64`: LR=0.001, BS=64
- `experiment_0_01_32`: LR=0.01, BS=32
- `experiment_0_01_64`: LR=0.01, BS=64

### Combining Count and Variables

If you specify both `count` and `variables`, they must match:

```yaml
replicas:
  count: 4
  variables:
    - CONCURRENCY
```

If `CONCURRENCY` has `domain: [8, 16, 32, 64]` (4 values), this works. If the domain size doesn't match `count`, sflow raises an error.

## GPU Allocation with Replicas

When combining replicas with GPU resources, sflow automatically slices `CUDA_VISIBLE_DEVICES` for each replica:

```yaml
backends:
  - name: cluster
    type: slurm
    gpus_per_node: 8

workflow:
  tasks:
    - name: worker
      replicas:
        count: 4
        policy: parallel
      resources:
        gpus:
          count: 2
      script:
        - echo "My GPUs: $CUDA_VISIBLE_DEVICES"
```

Each replica gets 2 GPUs:
- `worker_0`: CUDA_VISIBLE_DEVICES=0,1
- `worker_1`: CUDA_VISIBLE_DEVICES=2,3
- `worker_2`: CUDA_VISIBLE_DEVICES=4,5
- `worker_3`: CUDA_VISIBLE_DEVICES=6,7

:::note Kubernetes
Per-replica `CUDA_VISIBLE_DEVICES` slicing applies to the **local**, **slurm**, and
**docker** backends. On **Kubernetes**, each replica normally gets its own pod and DRA / the
device plugin assigns its GPUs, so `CUDA_VISIBLE_DEVICES` is left unset — the exception is
co-located GPU tasks that sflow merges into one shared pod, where it sets a per-member
`CUDA_VISIBLE_DEVICES` to pin each task's slice. See
[Resources: GPUs on Kubernetes](./resources.md#gpus-on-kubernetes).
:::

## Real-World Examples

### Disaggregated Inference (Prefill/Decode Servers)

```yaml
variables:
  NUM_PREFILL_SERVERS:
    value: 2
  NUM_DECODE_SERVERS:
    value: 2

workflow:
  tasks:
    - name: prefill_server
      replicas:
        count: ${{ variables.NUM_PREFILL_SERVERS }}
        policy: parallel
      resources:
        gpus:
          count: 1
      script:
        - python -m server --mode prefill

    - name: decode_server
      replicas:
        count: ${{ variables.NUM_DECODE_SERVERS }}
        policy: parallel
      resources:
        gpus:
          count: 1
      script:
        - python -m server --mode decode
```

### Benchmark Sweep with Sequential Runs

```yaml
variables:
  CONCURRENCY:
    value: 16
    domain: [16, 32, 64, 128]

workflow:
  tasks:
    - name: server
      script:
        - python -m http.server 8000
      probes:
        readiness:
          tcp_port:
            port: 8000

    - name: benchmark
      replicas:
        variables:
          - CONCURRENCY
        policy: sequential
      depends_on:
        - server
      script:
        - echo "Running benchmark with concurrency=$CONCURRENCY"
        - benchmark --concurrency $CONCURRENCY
```

The benchmarks run one at a time (`benchmark_0` → `benchmark_1` → ...) to avoid interference, while the server runs continuously.

## Common Patterns

### Parallel Workers

Use `policy: parallel` when replicas are independent and can run concurrently:
- Multiple inference servers
- Data parallel training workers
- Concurrent data processing pipelines

### Sequential Iterations

Use `policy: sequential` when replicas should run one after another:
- Benchmark sweeps (to avoid resource contention)
- Iterative experiments that build on previous results
- Rate-limited API calls

### Dynamic Replica Count

Use expressions to set replica count from variables:

```yaml
variables:
  NUM_WORKERS:
    value: 4

workflow:
  tasks:
    - name: worker
      replicas:
        count: ${{ variables.NUM_WORKERS }}
        policy: parallel
      script:
        - echo "Worker $SFLOW_REPLICA_INDEX of $NUM_WORKERS"
```

This allows overriding replica count via CLI:

```bash
sflow run --set NUM_WORKERS=8
```
