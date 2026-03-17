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
| **Operator** | How a task is launched. Built-ins: `bash`, `srun`, `docker`, `ssh`, `python`. Named operators let you preset flags and reuse them across tasks. |
| **Variable** | A named value referenced as `${{ variables.NAME }}` in YAML or `${NAME}` in scripts. Override from the CLI with `--set`. |
| **Expression** | Jinja2-based `${{ ... }}` syntax inside YAML to reference variables, backend info, task metadata, and more (e.g. `${{ backends.slurm.nodes[0].ip_address }}`). Supports filters (`${{ [a, b] \| min }}`), conditionals, and list indexing. |
| **Artifact** | A named external resource (model, config, dataset) referenced by URI and resolved to a local path at runtime. |
| **Probe** | A health-check gate. Readiness probes block dependents until a service is live; failure probes terminate the workflow when a fatal condition is detected. |
| **Replica** | A task can be replicated N times (parallel or sequential) with per-replica variable overrides for sweeps. |

For detailed architecture diagrams, execution flow, assembly pipeline, orchestrator internals, plugin reference, and output structure, see [Architecture](./architecture.md).

## How to Use sflow (General Workflow)

```mermaid
flowchart TD
  write["1. Write sflow.yaml"] --> validate["2. Validate (--dry-run)"]
  validate --> errCheck{Errors?}
  errCheck -- Yes --> write
  errCheck -- No --> envChoice{Environment?}

  envChoice -- Local --> runLocal["3a. sflow run --tui"]
  envChoice -- Slurm interactive --> runSlurm["3b. sflow run --tui"]
  envChoice -- Slurm production --> runBatch["3c. sflow batch --submit"]

  runLocal --> resolve["4. Resolve variables\nbuild task graph"]
  runSlurm --> resolve
  runBatch --> resolve

  resolve --> allocate["5. Allocate resources"]
  allocate --> execute["6. Execute DAG\n(operators + probes)"]
  execute --> collect["7. Collect outputs & logs"]

  collect --> passCheck{All tasks passed?}
  passCheck -- Yes --> done(("Done"))
  passCheck -- No --> fix["Check logs & re-run"]
  fix --> write
```

## Modular Workflow

For larger projects, split config into composable modules and pass them directly to `sflow run` or `sflow batch` -- no separate compose step required. This enables framework swapping, benchmark mixing, and CSV-driven parameter sweeps. See [Modular Workflows](./modular-workflows.md) for details.

```mermaid
flowchart TD
  modules["1. Write modular YAMLs\n(base, servers, benchmark)"] --> validate["2. Validate (--dry-run)\nsflow run -f a.yaml -f b.yaml --dry-run"]
  validate --> errCheck{Errors?}
  errCheck -- Yes --> modules
  errCheck -- No --> runChoice{Run mode?}

  runChoice -- Single run --> run["3a. sflow run\n-f a.yaml -f b.yaml --tui"]
  runChoice -- Batch submit --> batch["3b. sflow batch\n-f a.yaml -f b.yaml --submit"]
  runChoice -- Parameter sweep --> bulk["3c. sflow batch\n--bulk-input sweep.csv"]

  run --> done(("Done"))
  batch --> done
  bulk --> done
```

### Config Merging Rules

When multiple YAML files are provided:

| Section | Merge Strategy |
|---------|---------------|
| `version` | Must match across all files |
| `variables` | Merge by name (later overrides earlier) |
| `artifacts` | Merge by name |
| `backends` | Merge by name |
| `operators` | Merge by name |
| `workflow.tasks` | Concatenated (later files append tasks) |
| `workflow.name` | Last non-null wins |

## Expression System

The `${{ ... }}` expression syntax (powered by Jinja2) provides access to the full runtime context:

| Namespace | Example | Description |
|-----------|---------|-------------|
| `variables` | `${{ variables.MODEL_NAME }}` | Resolved variable value |
| `artifacts` | `${{ artifacts.MODEL.path }}` | Artifact local path |
| `backends` | `${{ backends.slurm.nodes[0].ip_address }}` | Backend node info |
| `task` | `${{ task.assigned_nodes }}` | Current task's node assignment |
| Filters | `${{ [a, b] \| min }}` | Jinja2 filters |

Expressions are resolved in phases — variables first, then backends, then artifacts, then task-level — so later phases can reference earlier results.

## Known Limitations

The following features are **not yet implemented** in the current release:

- `sflow run --resume` — raises `NotImplementedError`
- `sflow run --task` — raises `BadParameter`
- `hf://` and `docker://` artifact materialization — raises `NotImplementedError`

This user guide reflects actual code behavior. Not all planned features may be available yet.

## Next Steps

| Topic | Page |
|-------|------|
| Architecture, execution flow, plugins | [Architecture](./architecture.md) |
| Run a minimal example | [Quickstart](./quickstart.md) |
| Variables, expressions, env injection | [Variables](./variables.md) |
| Named inputs (paths, images, etc.) | [Artifacts](./artifacts.md) |
| Compute backends (local, Slurm) | [Backends](./backends.md) |
| Task launch methods (bash, srun, containers) | [Operators](./operators.md) |
| Node/GPU placement, CUDA_VISIBLE_DEVICES | [Resources](./resources.md) |
| Parallel/sequential replicas, sweeps | [Replicas](./replicas.md) |
| Composable configs, sweeps, missable tasks | [Modular Workflows](./modular-workflows.md) |
| Readiness/failure gates for services | [Probes](./probes.md) |
| Log and output directory structure | [Outputs & Logs](./outputs.md) |
| Full sflow.yaml schema | [Configuration](./configuration.md) |
| CLI options | [CLI Reference](./cli.md) |
| Frequently asked questions | [FAQ](./faq.md) |
