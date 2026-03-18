---
title: CLI reference
sidebar_position: 11
---

`sflow` currently exposes these CLI commands:

- `sflow run` – Run a workflow
- `sflow compose` – Compose multiple YAML files into a single config
- `sflow batch` – Generate sbatch script for Slurm batch mode
- `sflow visualize` – Visualize workflow DAG
- `sflow sample` – List and copy sample workflows

> Note: `--resume` / `--task` are currently marked as not implemented in code and will error immediately.

## sflow run

```bash
sflow run --file sflow.yaml
```

Common options:

- `--file, -f <path>`: config file path (default: `sflow.yaml`)
- `--dry-run`: validate + print execution plan, without running tasks
- `--tui`: enable Rich TUI (left: tasks + backends, right: auto-tail logs)
- `--set, -s KEY=VALUE`: override variables (repeatable); variable must already exist in `variables`
- `--artifact, -a NAME=URI`: override artifacts (repeatable); artifact must already exist in `artifacts`
- `--missable-tasks, -M <pattern>`: task names or glob patterns (e.g. `prefill_*`) that may be absent when composing multiple files. Missing missable tasks are removed from `depends_on` and probes with a warning. Only valid with multiple input files. Repeatable.
- `--workspace-dir <dir>`: workspace root directory (default: current directory)
- `--output-dir <dir>`: output root directory (default: `<workspace-dir>/sflow_output`)
- `--log-level <level>`: `debug|info|warning|error|critical` (default: `info`)

Notes:

- `--tui` is ignored in `--dry-run` mode.
- In `--tui` mode, logs are captured and rendered in the right pane (to avoid interleaving console logs with the live UI).

Output structure (non dry-run):

- `<output-dir>/<run_id>/sflow.log`: global log
- `<output-dir>/<run_id>/<task>/<task>.log`: per-task log

## sflow visualize

```bash
sflow visualize --file sflow.yaml --format mermaid
```

Common options:

- `--file, -f <path>`: config file path
- `--format <fmt>`: `mermaid|dot|png|svg|pdf`
- `--output, -o <path>`: output file path; if omitted, writes to `<output-dir>/<run_id>/<workflow>.<ext>`
- `--show-variables`: include variables in output (as comments)
- `--workspace-dir <dir>` / `--output-dir <dir>`: same as `run`

Notes:

- `png/svg/pdf` output requires Graphviz `dot`. Otherwise use `--format mermaid` or `--format dot`.

## sflow compose

Compose multiple sflow YAML files into a single valid workflow config. Supports single-file passthrough or multi-file merging.

```bash
# Compose and print to stdout
sflow compose backends.yaml workflow.yaml tasks.yaml

# Compose and write to file
sflow compose -f backends.yaml -f tasks.yaml -o merged.yaml

# Compose with variable overrides
sflow compose backends.yaml tasks.yaml --set SLURM_NODES=4 -o merged.yaml

# Compose and resolve all variables to literal values
sflow compose backends.yaml tasks.yaml --resolve -o resolved.yaml

# Bulk compose: generate one composed YAML per CSV row
sflow compose --bulk-input jobs.csv -o output_dir

# Bulk compose with validation
sflow compose --bulk-input jobs.csv --validate -o output_dir
```

Common options:

- File inputs (positional or `--file, -f`): workflow YAML files to merge
- `--output, -o <path>`: output file path (default: stdout)
- `--set, -s KEY=VALUE`: override variable values (repeatable)
- `--artifact, -a NAME=URI`: override artifact URIs (repeatable)
- `--resolve, -r`: resolve all resolvable variables to literal values inline and remove them from the variables section. Without this flag, variables are kept as `${{ }}` expressions for flexibility.
- `--validate, -vl`: run dry-run validation on each composed config to check for resource issues (e.g. GPU over-subscription)
- `--missable-tasks, -M <pattern>`: task names or glob patterns that may be absent when composing multiple files (repeatable). Missing references are removed with a warning. Only valid with multiple input files or `--bulk-input`.
- `--bulk-input, -b <csv>`: CSV file for bulk compose (one YAML per row). Supports a `missable_tasks` column for per-row missable task patterns.
- `--row`: process specific CSV rows (supports ranges, e.g. `--row 1:4`)
- `--log-level`: logging level (default: `info`)
- `--verbose, -v`: enable verbose output

Notes:

- A single file is accepted (useful for `--resolve` to inline variables)
- The composed config is validated against the sflow schema before output
- Variable expressions (`${{ }}`) support chained references (e.g. `NODES_PER_WORKER` can reference `GPUS_PER_WORKER`)
- `--resolve` preserves variables used by `replicas.variables` (sweep variables) since their value changes per replica. Runtime-dependent expressions (e.g. backend node IPs) are also kept.

## sflow batch

Generate sbatch scripts for running sflow in Slurm batch mode. Supports three modes:

1. **Single-job mode** (default): generate one sbatch script from config files
2. **Bulk-input mode** (`--bulk-input`): CSV-driven, one job per row with per-row overrides
3. **Bulk-submit mode** (`--bulk-submit`): file/folder-driven, each YAML is a standalone job

```bash
# Single-job mode
sflow batch workflow.yaml -N 2 -G 4 -p gpu -A myaccount -o run.sh --submit

# Bulk-input mode (CSV-driven)
sflow batch --bulk-input jobs.csv -G 4 -p gpu -A myaccount --submit

# Bulk-submit mode (folder of self-contained configs)
sflow batch --bulk-submit ./examples/ -G 4 -p gpu -A myaccount --submit

# Bulk-submit with specific files
sflow batch -B sglang_agg.yaml -B vllm_agg.yaml -G 4 -p gpu --submit

# Bulk-submit with glob pattern
sflow batch --bulk-submit 'examples/slurm_*' -G 4 -p gpu -A myaccount --submit
```

Common options:

- `--file, -f <path>`: config file path (default: `sflow.yaml`)
- `--sbatch-path, -o <path>`: write sbatch script to file (required for `--submit` in single-job mode)
- `--submit`: submit the job immediately after generating the script
- `--partition, -p <name>`: Slurm partition (auto-detected if not specified)
- `--account, -A <name>`: Slurm account (auto-detected if not specified)
- `--time <limit>`: time limit (e.g., `02:00:00`)
- `--nodes, -N <count>`: number of nodes. Required for single-job mode. For bulk modes, auto-detected from the config's slurm backend `nodes` field.
- `--gpus-per-node, -G <count>`: number of GPUs per node for cluster topology (default: 4). Applied to slurm backend config, not as a sbatch directive. Use `-e '--gpus-per-node=N'` if your cluster requires the sbatch directive.
- `--job-name, -J <name>`: Slurm job name (default: `sflow`)
- `--set, -s KEY=VALUE`: override variables (repeatable)
- `--artifact, -a NAME=URI`: override artifacts (repeatable)
- `--missable-tasks, -M <pattern>`: task names or glob patterns that may be absent when composing modular configs (repeatable). Missing references are removed with a warning. Only valid with multiple input files or `--bulk-input`/`--bulk-submit`.
- `--sflow-venv-path <path>`: path to existing Python venv for compute nodes
- `--sbatch-extra-args, -e <arg>`: additional `#SBATCH` directives (repeatable)
- `--sbatch-output, -O <pattern>`: Slurm stdout pattern (default: `sflow_output/%j-sflow-submit.out`)
- `--sbatch-error, -E <pattern>`: Slurm stderr pattern (default: `sflow_output/%j-sflow-submit.err`)

### Bulk-input mode (`--bulk-input`)

- `--bulk-input, -b <csv>`: CSV file with a required `sflow_config_file` column and optional `job_name` column. All other columns are matched to variable or artifact names.
- `--row`: process specific rows (e.g. `--row 1:4`, `--row 1,3,5`)
- `--resolve, -r`: resolve variables in the generated merged YAML configs (same as `sflow compose --resolve`)
- Override precedence: for variables, CSV values override CLI `--set`. For artifacts, CLI `--artifact` overrides CSV values.
- Generates both `.sh` (sbatch script) and `.yaml` (merged config) files per row.
- Always writes a `results.csv` with job IDs, output directories, and status.
- Reserved CSV column `missable_tasks`: space-separated task names or glob patterns per row. Merged with CLI `--missable-tasks`. Allows mixed disagg/agg rows in the same CSV where different rows have different absent tasks. Columns that only exist in some row configs (e.g. `NUM_AGG_SERVERS` for agg rows, `NUM_CTX_SERVERS` for disagg rows) are automatically handled.

### Bulk-submit mode (`--bulk-submit`)

- `--bulk-submit, -B <path>`: file paths, folder paths, or glob patterns. Folders are scanned for `*.yaml`/`*.yml` files with a `version` key.
- Each YAML is processed as a self-contained workflow (no merging).
- CLI flags (`--set`, `--artifact`, etc.) are applied to every config. Warns when `--set` overrides a variable already defined in a config.
- Node count is auto-detected from the config's slurm backend.
- Always writes a `results.csv` with job IDs and status.

### Notes

- A dry-run validation is performed before generating each sbatch script. CLI `--nodes` and `--gpus-per-node` are applied directly to the slurm backend during validation.
- Sbatch stdout/stderr logs are automatically copied into the sflow workflow output directory at the end of each generated script.
- Without `--submit`, a hint is shown to remind you to add `--submit` for actual submission.

## sflow sample

List available sample workflows or copy a sample to your project. Supports both single-file samples and modular folder samples.

```bash
# List all available samples (includes modular folders)
sflow sample --list
sflow sample

# Copy a self-contained sample
sflow sample slurm_dynamo_trtllm_agg

# Copy a modular sample folder
sflow sample inference_x_v2

# Copy with custom output path
sflow sample local_hello_world --output my_workflow.yaml

# Overwrite existing file/folder
sflow sample inference_x_v2 --force
```

Available sample categories:

- **Local**: `local_hello_world` – minimal local workflow
- **Slurm (self-contained)**: `slurm_dynamo_sglang_agg`, `slurm_dynamo_vllm_agg`, `slurm_dynamo_trtllm_agg`, etc. – complete workflows in a single YAML
- **Modular**: `inference_x_v2/` – folder with composable YAML files (slurm_config, common_workflow, framework-specific prefill/decode, benchmarks)

### Modular samples

Modular samples are folders containing multiple YAML files designed to be composed together. When you copy a modular sample, the entire folder is copied:

```bash
sflow sample inference_x_v2
```

After copying, you get usage hints showing two workflows:

- **Option A (Bulk batch)**: Use `sflow batch --bulk-input <folder>/bulk_input.csv` to generate and submit jobs from a CSV
- **Option B (Compose + Submit)**: Use `sflow compose` to merge files into a complete config, then `sflow run` or `sflow batch` to execute

Common options:

- `<name>`: sample name or folder name
- `--output, -o <path>`: output path (default: `./<sample_name>`)
- `--force, -f`: overwrite existing file/folder if it exists
- `--list, -l`: list all available samples
