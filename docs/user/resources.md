---
title: Resources
sidebar_position: 7
---

`resources` lets you constrain where a task runs (which nodes) and how many GPUs it should get.

## GPUs: `CUDA_VISIBLE_DEVICES` slicing (local / Slurm / Docker)

On the **local**, **slurm**, and **docker** backends sflow slices GPU indices per task and
replica and exports them as `CUDA_VISIBLE_DEVICES`. On **Kubernetes** GPUs are assigned by
the device plugin (default) or DRA instead — see [GPUs on Kubernetes](#gpus-on-kubernetes) below.

GPU resource example:

Key idea:

- Set `backends.<name>.gpus_per_node` so sflow can **pack and slice** GPU indices per task/replica.
- Set `task.resources.gpus.count` to request GPUs for that task.

Minimal example:

```yaml
version: "0.1"

variables:
  SLURM_ACCOUNT: { value: your_slurm_account }
  SLURM_PARTITION: { value: your_slurm_partition }
  SLURM_TIME: { value: "00:05:00" }
  SLURM_NODES: { value: 1 }
  GPUS_PER_NODE: { value: 4 }

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    gpus_per_node: ${{ variables.GPUS_PER_NODE }}
    account: ${{ variables.SLURM_ACCOUNT }}
    partition: ${{ variables.SLURM_PARTITION }}
    time: ${{ variables.SLURM_TIME }}
    nodes: ${{ variables.SLURM_NODES }}

workflow:
  name: slurm_gpu_cuda_visible
  tasks:
    - name: t2
      replicas:
        count: 2
        policy: parallel
      resources:
        gpus:
          count: 2
      script:
        - echo "replica=$SFLOW_REPLICA_INDEX CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

## Nodes: pin tasks to specific nodes

Use `resources.nodes` to select which allocated nodes a task may use.

- `indices`: explicit node positions from the allocation
- `count`: first N nodes from the selected pool
- `exclude`: node positions to remove before applying `indices`, `count`, or GPU packing

Indices are 0-based positions into the node list after any `exclude` filtering.

**Negative indices** work like Python: `-1` is the last node, `-2` is second-to-last, etc.

If a Slurm task does not set `resources.nodes`, sflow passes the full backend allocation to `srun`.

### Pin server and client to the same node

Useful for "server + client" style workflows where `127.0.0.1` must work:

```yaml
workflow:
  name: wf
  tasks:
    - name: server
      resources:
        nodes:
          indices: [0]
      script: ["python -m http.server 8000"]
    - name: client
      depends_on: [server]
      resources:
        nodes:
          indices: [0]
      script: ["curl -sf http://127.0.0.1:8000/ > /dev/null"]
```

### Run a task on the last allocated node

Useful when the benchmark client should run on a dedicated node separate from the serving nodes:

```yaml
workflow:
  name: wf
  tasks:
    - name: serving
      resources:
        nodes:
          exclude: [-1]   # all nodes except the last
      script: ["start_server.sh"]
    - name: benchmark
      depends_on: [serving]
      resources:
        nodes:
          indices: [-1]   # last node only
      script: ["run_benchmark.sh"]
```

### Exclude nodes before placement

`exclude` removes nodes from the available pool. This is useful when a shared service must stay on the head node and the rest of the workflow should avoid it:

```yaml
workflow:
  name: wf
  tasks:
    - name: control_plane
      resources:
        nodes:
          indices: [0]
      script: ["start_control_plane.sh"]
    - name: workers
      depends_on: [control_plane]
      resources:
        nodes:
          exclude: [0]
          count: 2
      script: ["start_workers.sh"]
```

`count` slices the filtered pool in order. In the example above, if the allocation is `[n1, n2, n3, n4]`, the `workers` task uses `[n2, n3]`.

`exclude` accepts a single index, a list of indices, or an expression that resolves to either:

```yaml
resources:
  nodes:
    exclude: "${{ range(0, 2) | list }}"  # removes nodes 0 and 1
```

Negative indices in `indices` are resolved after `exclude`. For example, `exclude: [3]` and `indices: [-1]` on a four-node allocation selects node 2, because node 3 is removed first.

:::important Node resources overlap by default
`resources.nodes.indices` and `resources.nodes.count` are placement constraints unless you also set `resources.nodes.release_after`.

That means two tasks can select the same node by default. This is intentional for common server/client or colocated workload patterns. Add `resources.nodes.release_after` only when the selected node should be treated as an exclusive reservation with a lifecycle.
:::

## GPU packing

Set `resources.gpus.count` to reserve GPU IDs and set `CUDA_VISIBLE_DEVICES` for the task. sflow packs GPU requests onto the selected node pool and advances to later nodes when earlier nodes are full.

```yaml
workflow:
  name: wf
  tasks:
    - name: prefill
      resources:
        nodes:
          exclude: [-1]
        gpus:
          count: 4
      script: ["start_prefill.sh"]
    - name: benchmark
      depends_on: [prefill]
      resources:
        nodes:
          indices: [-1]
      script: ["run_benchmark.sh"]
```

A `count`-only request is **index-agnostic**: sflow gives the task any contiguous run of idle GPUs. A run never straddles a node boundary — if a 4-GPU task finds only GPUs 2,3 free on the first node, it skips that node entirely and starts at GPU 0 of the next one.

If a GPU request cannot fit on one node but is an exact multiple of `backends.<name>.gpus_per_node`, sflow can expand the task across multiple nodes. If the request is not a valid multiple or the selected pool is too small, validation fails before execution.

When you pin the node count yourself with `resources.nodes`, `count` is still the
**total** over those nodes and every node takes the same slice — so `count` must be a
positive multiple of the number of assigned nodes, and `count / nodes` must fit on the
smallest of them. `nodes.count: 2` with `gpus.count: 1` is rejected: one GPU cannot be
split across two nodes. Write `gpus.count: 2` for one GPU on each of two nodes.

This `CUDA_VISIBLE_DEVICES` packing applies to the **local**, **slurm**, and **docker** backends. Kubernetes assigns GPUs differently — see below.

### Pin specific GPUs with `indices`

Use `resources.gpus.indices` when a task must land on particular device IDs — NUMA/NVLink affinity, reproducing a vendor benchmark topology, or steering around a known-bad device.

Indices are 0-based device IDs on each node and must be **non-negative** (unlike `resources.nodes.indices`, which allows Python-style negatives). `CUDA_VISIBLE_DEVICES` preserves the order you write, so `indices: [1, 0]` exports `1,0`.

**`indices` alone** — the task takes exactly those devices on the **first node where they are all free**. The scan restarts at node 0 for every task, so low-numbered nodes stay reachable for later tasks:

```yaml
workflow:
  name: wf
  tasks:
    - name: a
      resources: {gpus: {indices: [0, 1], release_after: workflow_completion}}
      script: ["run.sh"]
    - name: b                     # node0's 0,1 are taken -> b lands on node1
      depends_on: [a]
      resources: {gpus: {indices: [0, 1], release_after: workflow_completion}}
      script: ["run.sh"]
    - name: c                     # node0's 2,3 are still idle -> c backfills node0
      depends_on: [b]
      resources: {gpus: {indices: [2, 3], release_after: workflow_completion}}
      script: ["run.sh"]
```

**`count` + `indices`** — `count` is the **total** across nodes and `indices` is the per-node slice, so the task fans out over `count / len(indices)` nodes and uses the same slots on each. With 4-GPU nodes, `count: 8` and `indices: [0, 1]` gives GPUs 0,1 on four nodes:

```yaml
resources:
  gpus:
    count: 8
    indices: [0, 1]
```

`count` must be a positive multiple of `len(indices)`. If the task also pins nodes via `resources.nodes`, those nodes win and `count` must equal `len(indices) × (number of assigned nodes)`.

Indices may also come from an expression, like `indices: "${{ variables.GPU_IDS }}"`.

At least one of `count` or `indices` is required under `resources.gpus`. Non-contiguous sets such as `indices: [0, 2]` are allowed, and a later task can take the gaps.

:::note Not available on Kubernetes
`resources.gpus.indices` is rejected at plan time on the Kubernetes backend: the device plugin or DRA picks the physical devices, so sflow cannot honor a pin. Use `resources.gpus.count` there.
:::

## GPUs on Kubernetes

The Kubernetes backend does **not** slice `CUDA_VISIBLE_DEVICES` for ordinary pods.
Instead, the `device_plugin` (an `nvidia.com/gpu` limit, **the default**) or DRA (a
`ResourceClaimTemplate`, opt-in / WIP) assigns physical devices to the pod, so for a
normal single- or multi-node pod `CUDA_VISIBLE_DEVICES` is left unset and the container
sees exactly the GPUs it was granted.

> **Exception — merged co-located pods.** When sflow co-locates compatible same-node
> tasks into one pod (`merge_colocated_gpu_pods: auto`, the default), the merged launcher
> **does** set a per-member `CUDA_VISIBLE_DEVICES` so each task sees its own GPUs first
> (as `cuda:0`). See [merge-pod (`merge_colocated_gpu_pods`)](./backends.md#kubernetes-backend) in Backends.

`resources.gpus.count` is a **per-task total** that is split evenly across the task's
assigned nodes, so it must be a multiple of the node count (each pod gets
`count / nodes` GPUs) — the same rule every backend applies, enforced at plan time.
A multi-node task becomes **one pod per reserved node**. See
[Backends: GPU requests](./backends.md#gpu-requests) for details.

## Resource reuse with `release_after`

`resources.nodes.release_after` and `resources.gpus.release_after` control when a task-level reservation can be reused by later tasks in the DAG.

Supported values:

- `workflow_completion`: hold the reservation until the whole workflow finishes
- `task_ready`: release after the task's readiness probe succeeds
- `task_completion`: release after the task reaches a terminal state (`COMPLETED`, `FAILED`, `TIMEOUT`, or `CANCELLED`)

GPU reservations infer a safe default when `release_after` is omitted:

- tasks without readiness probes release GPUs after task completion for downstream dependents
- tasks with readiness probes hold GPUs until workflow completion, because they may still be serving after becoming `READY`

Node placement behaves differently from GPU placement: **node selections can overlap by default**. `resources.nodes.indices` and `resources.nodes.count` only constrain where a task may run. They do not reserve those nodes exclusively unless `resources.nodes.release_after` is explicitly set. Add `resources.nodes.release_after` when you want an explicit exclusive node reservation with a lifecycle.

Example: a one-time environment check can release all GPUs after it completes, allowing downstream workers to reuse them:

```yaml
workflow:
  name: release_after_check
  tasks:
    - name: check_entire_node
      resources:
        gpus:
          count: 8
          release_after: task_completion
      script:
        - nvidia-smi

    - name: worker
      depends_on: [check_entire_node]
      replicas:
        count: 4
        policy: parallel
      resources:
        gpus:
          count: 2
      script:
        - echo "worker GPUs=${CUDA_VISIBLE_DEVICES}"
```

Example: a setup service can release an explicit node reservation after readiness if it no longer needs exclusive placement once clients start:

```yaml
workflow:
  name: release_after_ready
  tasks:
    - name: bootstrap
      resources:
        nodes:
          indices: [0]
          release_after: task_ready
      script:
        - python -m http.server 8000
      probes:
        readiness:
          tcp_port:
            port: 8000

    - name: client
      depends_on: [bootstrap]
      resources:
        nodes:
          indices: [0]
      script:
        - curl -sf http://127.0.0.1:8000/ > /dev/null
```

Dry-run rehearses these lifetimes across the DAG, so oversubscription errors include the tasks and release policies that block placement.
