---
title: Modular Workflows
sidebar_position: 6
---

Modular workflows split a single monolithic YAML config into multiple composable files. Instead of one large file containing everything, each concern is separated into its own file that can be mixed, matched, and reused.

## Why modular?

### The problem with monolithic configs

A typical disaggregated inference workflow needs: Slurm backend config, shared infrastructure tasks (NATS, etcd, frontend), framework-specific server tasks (prefill, decode), and benchmark tasks. Putting everything in one file leads to:

- **Duplication**: The same Slurm config and infrastructure tasks are copied across sglang, vllm, and trtllm variants
- **Maintenance burden**: Changing a shared component (e.g. frontend args) requires editing every variant
- **Large files**: 300-500+ lines per variant, hard to review and diff
- **Inflexible scaling**: Can't easily swap benchmarks or mix frameworks

### The modular approach

Split the workflow into logical building blocks:

```
inference_x_v2/
├── slurm_config.yaml          # Slurm backend (shared)
├── common_workflow.yaml       # Infrastructure tasks (shared)
├── benchmark_aiperf.yaml      # AIPerf benchmark (swappable)
├── benchmark_infmax.yaml      # InfMax benchmark (swappable)
├── bulk_input.csv             # Batch job definitions
├── sglang/
│   ├── prefill.yaml           # SGLang prefill server
│   └── decode.yaml            # SGLang decode server
├── vllm/
│   ├── prefill.yaml           # vLLM prefill server
│   └── decode.yaml            # vLLM decode server
└── trtllm/
    ├── prefill.yaml           # TRT-LLM prefill server
    └── decode.yaml            # TRT-LLM decode server
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Reuse** | Shared components (slurm_config, common_workflow) are written once and used by all variants |
| **Swap** | Change the benchmark by swapping `benchmark_aiperf.yaml` for `benchmark_infmax.yaml` |
| **Mix frameworks** | Combine any prefill/decode pair: sglang + aiperf, vllm + infmax, trtllm + aiperf, etc. |
| **Smaller diffs** | Changes to one component only touch one file |
| **Bulk testing** | Define many combinations in a CSV and generate/submit all at once |
| **Computed variables** | Each module defines only the variables it needs; computed variables (e.g. `GPUS_PER_WORKER`) chain across modules |

## How it works

### Composing files

When multiple YAML files are passed to `sflow compose` or `sflow run`, they are merged using these rules:

- **`version`**: must be consistent across files
- **`variables`**: merged by name (later file wins on conflict)
- **`artifacts`**: merged by name (later file wins)
- **`backends`**: merged by name (later file wins)
- **`operators`**: merged by name (later file wins)
- **`workflow.name`**: must be consistent across files
- **`workflow.variables`**: merged by name (later file wins)
- **`workflow.tasks`**: concatenated in file order

This means you compose a workflow by listing the files in order:

```bash
sflow compose slurm_config.yaml common_workflow.yaml \
              trtllm/prefill.yaml trtllm/decode.yaml \
              benchmark_aiperf.yaml -o composed.yaml
```

The result is a single valid YAML with all components merged.

### File order matters

Tasks are concatenated in file order. Since tasks use `depends_on` to define the DAG, the order doesn't affect execution -- but it does affect readability. A recommended convention:

1. Backend/infrastructure config first (`slurm_config.yaml`)
2. Shared workflow and tasks (`common_workflow.yaml`)
3. Framework-specific tasks (`trtllm/prefill.yaml`, `trtllm/decode.yaml`)
4. Benchmark tasks last (`benchmark_aiperf.yaml`)

### Variables across modules

Each module defines only the variables it needs. When composed, variables from all files are merged:

- `slurm_config.yaml` defines `SLURM_NODES`, `GPUS_PER_NODE`, etc.
- `trtllm/prefill.yaml` defines `CTX_TP_SIZE`, `CTX_GPUS_PER_WORKER`, etc.
- `benchmark_aiperf.yaml` defines `CONCURRENCY`, `ISL`, `OSL`, etc.

Computed variables can reference variables from other modules:

```yaml
# In trtllm/prefill.yaml
CTX_GPUS_PER_WORKER:
  type: integer
  value: ${{ variables.CTX_TP_SIZE * variables.CTX_DP_SIZE * variables.CTX_PP_SIZE }}

CTX_NODES_PER_WORKER:
  type: integer
  value: ${{ [variables.CTX_GPUS_PER_WORKER // variables.GPUS_PER_NODE, 1] | max }}
```

Here `GPUS_PER_NODE` comes from `slurm_config.yaml`, and `CTX_GPUS_PER_WORKER` is computed within the same module. The chained reference works because sflow resolves variables iteratively across all modules.

## Handling missing tasks (`--missable-tasks`)

When composing modular files, a task in one file may declare `depends_on` referencing a task from another file that wasn't included. A common example is switching between **disaggregated** (separate prefill + decode servers) and **aggregated** (single server) inference modes. Each mode defines different tasks, but the benchmark may reference tasks from both:

- `disagg/prefill.yaml` defines `prefill_server`, `disagg/decode.yaml` defines `decode_server`
- `agg/agg.yaml` defines `agg_server`
- `benchmark_aiperf.yaml` has `depends_on` referencing all three

When composing for disaggregated mode, `agg_server` doesn't exist. When composing for aggregated mode, `prefill_server` and `decode_server` don't exist. Use `--missable-tasks` to declare which tasks are allowed to be absent:

```bash
# Disaggregated mode: agg_server doesn't exist, mark it missable
sflow compose base.yaml disagg/prefill.yaml disagg/decode.yaml \
              benchmark_aiperf.yaml \
              --missable-tasks agg_server \
              -o disagg.yaml

# Aggregated mode: prefill/decode servers don't exist, mark them missable
sflow compose base.yaml agg/agg.yaml \
              benchmark_aiperf.yaml \
              --missable-tasks prefill_server \
              --missable-tasks decode_server \
              -o agg.yaml
```

Glob patterns are supported: `--missable-tasks 'prefill_*'` matches `prefill_server`, `prefill_server_0`, etc.

When a missable task is absent, sflow:

- Removes it from `depends_on` lists of other tasks
- Removes probe `logger` references pointing to it
- Logs which references were removed

This is useful when:

- Switching between aggregated and disaggregated inference modes
- Building partial workflows for testing a single component
- Composing subsets of a modular config for quick iteration

### Per-row missable tasks in CSV (`missable_tasks` column)

When using `--bulk-input` with a CSV, you can specify missable tasks per row using the reserved `missable_tasks` column. This allows mixing disaggregated and aggregated configurations in the same CSV:

```csv
sflow_config_file,NUM_CTX_SERVERS,CTX_TP_SIZE,NUM_AGG_SERVERS,AGG_TP_SIZE,missable_tasks
base.yaml disagg/prefill.yaml disagg/decode.yaml bench.yaml,2,2,,,agg_server
base.yaml agg/agg.yaml bench.yaml,,,,4,prefill_server decode_server
```

- **Disagg rows**: set `missable_tasks=agg_server` because `agg_server` doesn't exist in disagg configs
- **Agg rows**: set `missable_tasks=prefill_server decode_server` because those tasks don't exist in agg configs

Each row only applies its own `missable_tasks` -- not the union of all rows. This ensures the correct tasks are stripped per row.

CSV columns that only exist in some row configs (e.g. `NUM_AGG_SERVERS` for agg rows, `NUM_CTX_SERVERS` for disagg rows) are automatically handled. sflow validates columns against ALL row configs, so a column is valid if it matches a variable in any row's config.

The `missable_tasks` column values are merged with CLI `--missable-tasks` if both are provided.

## Usage patterns

You do **not** need to run `sflow compose` before `sflow run` or `sflow batch`. Both commands accept multiple `-f` flags and merge the files automatically. `sflow compose` is only needed when you want to inspect or save the merged result.

### Pattern 1: Direct run (no compose step)

Pass multiple files directly to `sflow run` or `sflow batch` -- they are merged on the fly:

```bash
# Validate
sflow run -f slurm_config.yaml -f common_workflow.yaml \
          -f sglang/prefill.yaml -f sglang/decode.yaml \
          -f benchmark_aiperf.yaml --dry-run

# Run interactively with TUI
sflow run -f slurm_config.yaml -f common_workflow.yaml \
          -f sglang/prefill.yaml -f sglang/decode.yaml \
          -f benchmark_aiperf.yaml --tui

# Or submit to Slurm directly
sflow batch -f slurm_config.yaml -f common_workflow.yaml \
            -f trtllm/prefill.yaml -f trtllm/decode.yaml \
            -f benchmark_aiperf.yaml \
            -N 1 -G 4 -p gpu -A myaccount -o run.sh --submit
```

### Pattern 2: Compose for inspection

Use `sflow compose` when you want to review or share the merged result:

```bash
# Merge into a single file for inspection
sflow compose slurm_config.yaml common_workflow.yaml \
              trtllm/prefill.yaml trtllm/decode.yaml \
              benchmark_aiperf.yaml -o composed.yaml

# Then run or submit the composed file
sflow run -f composed.yaml --tui
```

### Pattern 3: Bulk input (sweep)

For running many combinations defined in a CSV:

```csv
sflow_config_file,SLURM_NODES,NUM_CTX_SERVERS,CTX_TP_SIZE,NUM_GEN_SERVERS,GEN_TP_SIZE,DYNAMO_IMAGE
slurm_config.yaml common_workflow.yaml trtllm/prefill.yaml trtllm/decode.yaml benchmark_infmax.yaml,1,1,2,1,2,nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.0
slurm_config.yaml common_workflow.yaml sglang/prefill.yaml sglang/decode.yaml benchmark_infmax.yaml,1,2,1,1,2,nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.0
slurm_config.yaml common_workflow.yaml vllm/prefill.yaml vllm/decode.yaml benchmark_infmax.yaml,2,2,1,3,2,nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0
```

```bash
sflow batch --bulk-input bulk_input.csv \
            -a LOCAL_MODEL_PATH=fs:///path/to/model \
            -G 4 -p gpu -A myaccount --submit
```

Each row generates a separate Slurm job with different framework, node count, and parallelism settings.

### Pattern 4: Resolve for inspection

Use `--resolve` to see what the final config looks like with all variables inlined:

```bash
sflow compose slurm_config.yaml common_workflow.yaml \
              trtllm/prefill.yaml trtllm/decode.yaml \
              benchmark_aiperf.yaml --resolve -o resolved.yaml
```

This produces a plain-text YAML with no `${{ }}` expressions (except replica sweep variables and runtime-dependent values). Useful for:

- Reviewing the exact values that will be used
- Sharing configs with others who don't need the flexibility of variables
- Debugging expression resolution issues

## Modular vs self-contained

| Aspect | Self-contained | Modular |
|--------|---------------|---------|
| Files | 1 YAML per workflow | N files composed together |
| Reuse | Copy-paste shared sections | Import shared modules |
| Scaling | Edit each variant separately | Change one file, affects all |
| Batch | `--bulk-submit` (one job per YAML) | `--bulk-input` (CSV-driven) |
| Best for | Simple workflows, quick experiments | Production sweeps, multi-framework testing |

Both approaches are fully supported. Start with self-contained configs for simplicity, and move to modular when you need to scale across frameworks or run parameter sweeps.
