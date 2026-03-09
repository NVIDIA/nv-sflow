---
title: CLI reference
sidebar_position: 11
---

`sflow` currently exposes these CLI commands:

- `sflow run` – Run a workflow
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

## sflow batch

Generate an sbatch script for running sflow in Slurm batch mode (fire-and-forget).

```bash
sflow batch --file workflow.yaml --sbatch-path run.sh --submit
```

Common options:

- `--file, -f <path>`: config file path (default: `sflow.yaml`)
- `--sbatch-path, -o <path>`: write sbatch script to file (required for `--submit`)
- `--submit`: submit the job immediately after generating the script
- `--partition, -p <name>`: Slurm partition
- `--account, -A <name>`: Slurm account
- `--time <limit>`: time limit (e.g., `02:00:00`)
- `--nodes, -N <count>`: number of nodes for the sbatch job
- `--gpus-per-node, -G <count>`: number of GPUs per node
- `--job-name, -J <name>`: Slurm job name (default: `sflow`)
- `--set, -s KEY=VALUE`: override variables (repeatable)
- `--artifact, -a NAME=URI`: override artifacts (repeatable)
- `--sflow-venv-path <path>`: path to existing Python venv for compute nodes

Notes:

- The generated script includes automatic venv setup for compute nodes
- A dry-run validation is performed before the actual run to catch configuration errors early

## sflow sample

List available sample workflows or copy a sample to your project.

```bash
# List all available samples
sflow sample --list
sflow sample

# Copy a sample to current directory
sflow sample local_hello_world

# Copy with custom output path
sflow sample local_hello_world --output my_workflow.yaml

# Overwrite existing file
sflow sample local_hello_world --force
```

Available sample categories:

- **Local**: `local_hello_world` – minimal local workflow
- **Slurm**: `slurm_sglang_server_client` – Slurm-based inference workflow
- **Dynamo**: `slurm_dynamo_sglang_agg`, `slurm_dynamo_vllm_agg`, `slurm_dynamo_trtllm_agg`, etc. – disaggregated inference workflows

Common options:

- `<name>`: sample name (e.g., `local_hello_world` or `local_hello_world.yaml`)
- `--output, -o <path>`: output path for the sample file (default: `./<sample_name>`)
- `--force, -f`: overwrite existing file if it exists
- `--list, -l`: list all available samples
