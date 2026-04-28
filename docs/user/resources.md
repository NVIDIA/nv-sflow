---
title: Resources
sidebar_position: 7
---

`resources` lets you constrain where a task runs (which nodes) and how many GPUs it should get.

## GPUs: `CUDA_VISIBLE_DEVICES` slicing (Slurm)

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

If a GPU request cannot fit on one node but is an exact multiple of `backends.<name>.gpus_per_node`, sflow can expand the task across multiple nodes. If the request is not a valid multiple or the selected pool is too small, validation fails before execution.
