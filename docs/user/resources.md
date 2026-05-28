---
title: Resources
sidebar_position: 7
---

`resources` lets you constrain where a task runs (which nodes) and how many GPUs it should get.

## GPUs: `NVIDIA_VISIBLE_DEVICES` slicing (Slurm)

GPU resource example:

Key idea:

- Set `backends.<name>.gpus_per_node` so sflow can **pack and slice** GPU indices per task/replica.
- Set `task.resources.gpus.count` to request GPUs for that task. SFlow exports
  the resulting slice as `NVIDIA_VISIBLE_DEVICES` before launching `srun`, so
  NVIDIA container runtimes see the intended GPU subset. SFlow does not set
  `CUDA_VISIBLE_DEVICES`; task scripts may set it themselves if a framework
  specifically needs it.

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
  name: slurm_gpu_nvidia_visible
  tasks:
    - name: t2
      replicas:
        count: 2
        policy: parallel
      resources:
        gpus:
          count: 2
      script:
        - echo "replica=$SFLOW_REPLICA_INDEX NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
```

## Nodes: pin tasks to the same node

This is useful for “server + client” style workflows where `127.0.0.1` must work.

Example pattern:

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
