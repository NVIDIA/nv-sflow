---
title: Introduction
sidebar_position: 1
---

`sflow` is a **workflow orchestrator**: you describe _what to run_ in a `sflow.yaml` (tasks, dependencies, how to launch each task, and required resources). `sflow` executes the DAG in order, collects logs, and organizes outputs into a consistent directory structure.

![sflow TUI](/img/sflow_tui.gif)

## Use Cases

### Complex Slurm Workflows

sflow streamlines orchestration within Slurm clusters with built-in support for:

- Automatic hostname/IP detection after allocation
- Workload distribution across nodes and GPUs
- Runtime readiness and failure checks (probes)
- Replica scaling (parallel workers, sweeps)

Define what you want to run — no more hand-crafted bash scripts to manage resource placement or ensure processes land on the right nodes and GPUs. Below is an example DAG for a Dynamo PD disaggregated LLM inference service:

```mermaid
graph TD
  start((start))
  stop(((stop)))

  subgraph "prefill_server"
    prefill_server_0
    prefill_server_1
  end
  subgraph "decode_server"
    decode_server_0
    decode_server_1
  end
  subgraph "benchmark"
    benchmark_0
    benchmark_1
    benchmark_0 -- Completed --> benchmark_1
  end

  gpu_monitor["gpu_monitor"]
  nats_server["nats_server"]
  etcd_server["etcd_server"]
  frontend_server["frontend_server"]

  start --> gpu_monitor
  start --> nats_server
  start --> etcd_server

  nats_server -- Ready --> frontend_server
  etcd_server -- Ready --> frontend_server
  frontend_server -- Ready --> prefill_server_0
  frontend_server -- Ready --> prefill_server_1
  frontend_server -- Ready --> decode_server_0
  frontend_server -- Ready --> decode_server_1
  frontend_server -- Ready --> benchmark_0
  prefill_server_0 -- Ready --> benchmark_0
  prefill_server_1 -- Ready --> benchmark_0
  decode_server_0 -- Ready --> benchmark_0
  decode_server_1 -- Ready --> benchmark_0

  gpu_monitor -- Completed --> stop
  benchmark_1 -- Completed --> stop

```

### Cross-Environment Orchestration

Codify startup order, replica scale, readiness probes, and log capture in YAML — then run the same file locally or on a cluster by switching the backend.

### Benchmarking & Experiment Automation

Standardize how you launch runs, capture logs/artifacts, and structure outputs so results are reproducible across teams and machines.

### Local Development & Testing

Use the `local` backend with the `bash` operator to validate your DAG and scripts on your laptop before moving to a Slurm cluster.

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Workflow** | A set of tasks wired into a DAG via `depends_on`. |
| **Task** | An executable unit. The key field is `script` — a list of lines joined into a bash script. |
| **Backend** | Where compute comes from. Built-ins: `slurm` (allocates via `salloc`) and `local` (simulates nodes on the local machine). |
| **Operator** | How a task is launched. Built-ins: `bash` and `srun`. Named operators let you preset flags and reuse them across tasks. |
| **Variable** | A named value referenced as `${{ variables.NAME }}` in YAML or `${NAME}` in scripts. Override from the CLI with `--set`. |
| **Expression** | `${{ ... }}` syntax inside YAML to reference variables, backend info, task metadata, and more (e.g. `${{ backends.slurm.nodes[0].ip_address }}`). |

## Architecture

```mermaid
graph TB
  subgraph CLI["CLI Layer"]
    run["sflow run"]
    batch["sflow batch"]
    sample["sflow sample"]
    visualize["sflow visualize"]
  end

  subgraph App["Application"]
    sflowapp["SflowApp"]
    assembly["Assembly Pipeline"]
  end

  subgraph Config["Configuration"]
    loader["Config Loader\n(YAML + Pydantic)"]
    resolver["Expression Resolver\n(Jinja2 ${{ }})"]
    schema["Schema Models"]
  end

  subgraph Plugins["Plugins (Pluggable)"]
    subgraph Backends
      local_be["local"]
      slurm_be["slurm"]
    end
    subgraph Operators
      bash_op["bash"]
      srun_op["srun"]
      docker_op["docker"]
      ssh_op["ssh"]
    end
    subgraph Probes
      tcp["TCP Port"]
      http["HTTP Get/Post"]
      logwatch["Log Watch"]
    end
    subgraph Artifacts
      fs_art["fs://"]
      file_art["file://"]
      http_art["http://"]
      hf_art["hf://"]
    end
  end

  subgraph Core["Core Engine"]
    state["SflowState\n(variables, backends,\nartifacts, workflow)"]
    taskgraph["Task Graph (DAG)"]
    orchestrator["Orchestrator\n(poll loop)"]
    launcher["Subprocess Launcher"]
  end

  subgraph Output["Output"]
    tui["TUI (Rich)"]
    logs["Logs & Outputs\nsflow_output/"]
  end

  CLI --> sflowapp
  sflowapp --> assembly
  assembly --> loader
  assembly --> resolver
  loader --> schema
  assembly --> Backends
  assembly --> Operators
  assembly --> Artifacts
  assembly --> state
  state --> taskgraph
  taskgraph --> orchestrator
  orchestrator --> launcher
  orchestrator --> Probes
  launcher --> logs
  sflowapp --> tui
```

## How to Use sflow (General Workflow)

```mermaid
flowchart TD
  A["1. Write sflow.yaml\n(variables, backends,\noperators, tasks)"] --> B["2. Validate\nsflow run -f sflow.yaml --dry-run"]
  B --> C{Errors?}
  C -- Yes --> A
  C -- No --> D{Environment?}

  D -- Local testing --> E["3a. Run locally\nsflow run -f sflow.yaml --tui"]
  D -- Slurm interactive --> F["3b. Run on Slurm\nsflow run -f sflow.yaml --tui"]
  D -- Slurm production --> G["3c. Generate sbatch\nsflow batch -f sflow.yaml\n-o run.sh --submit"]

  E --> H["4. sflow resolves variables\nand builds task graph"]
  F --> H
  G --> H

  H --> I["5. Backend allocates resources\n(salloc for Slurm,\nsynthetic nodes for local)"]
  I --> J["6. Orchestrator executes DAG\n• Launches tasks via operators\n• Monitors probes\n• Handles retries"]
  J --> K["7. Collect outputs & logs\nsflow_output/<run_id>/"]

  K --> L{All tasks\npassed?}
  L -- Yes --> M(("Done ✓"))
  L -- No --> N["Check logs, fix config,\nre-run"]
  N --> A

  style A fill:#4a9eff,color:#fff
  style M fill:#22c55e,color:#fff
  style C fill:#f59e0b,color:#fff
  style L fill:#f59e0b,color:#fff
```

## Known Limitations

The following features are **not yet implemented** in the current release:

- `sflow run --resume` — raises `NotImplementedError`
- `sflow run --task` — raises `BadParameter`

This user guide reflects actual code behavior. Not all planned features may be available yet.

## Next Steps

| Topic | Page |
|-------|------|
| Run a minimal example | [Quickstart](./quickstart.md) |
| Variables, expressions, env injection | [Variables](./variables.md) |
| Named inputs (paths, images, etc.) | [Artifacts](./artifacts.md) |
| Compute backends (local, Slurm) | [Backends](./backends.md) |
| Task launch methods (bash, srun, containers) | [Operators](./operators.md) |
| Node/GPU placement, CUDA_VISIBLE_DEVICES | [Resources](./resources.md) |
| Readiness/failure gates for services | [Probes](./probes.md) |
| Log and output directory structure | [Outputs & Logs](./outputs.md) |
| Full sflow.yaml schema | [Configuration](./configuration.md) |
| CLI options | [CLI Reference](./cli.md) |
