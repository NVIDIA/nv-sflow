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

Use `resources.nodes.indices` to select specific nodes from the allocation. Indices are 0-based
positions into the node list (after any `exclude` filtering).

**Negative indices** work like Python: `-1` is the last node, `-2` is second-to-last, etc.

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
