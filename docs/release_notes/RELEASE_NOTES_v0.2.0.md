# sflow v0.2.0 Release Notes

**Release date:** March 2026  
**Previous release:** [v0.1.0](https://github.com/NVIDIA/nv-sflow/releases/tag/v0.1.0) (February 2, 2026)

---

## Highlights

sflow v0.2.0 is a major feature release that introduces **modular multi-file composition**, a **bulk-input batch system**, **AI agent skills**, and significantly expands framework support to cover **SGLang, vLLM, and TRT-LLM** across both aggregated and disaggregated inference modes.

```
v0.1.0                                    v0.2.0
┌──────────────────────┐           ┌──────────────────────────────────────┐
│  Single YAML file    │           │  Multi-file modular composition      │
│  Single-job batch    │    ──>    │  Bulk-input CSV batch (parallel)     │
│  TRT-LLM disagg only │           │  SGLang + vLLM + TRT-LLM (agg+PD)    │
│  Basic probes        │           │  Multi-node probes + preflight       │
│  No agent skills     │           │  AI agent skills (sflow skill)       │
└──────────────────────┘           └──────────────────────────────────────┘
```

---

## New Features

### 1. Modular Multi-File Composition (`sflow compose`)

Split workflows into reusable YAML fragments and merge them at runtime. This is the recommended pattern for production workflows.

```
┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌───────────┐
│ slurm_config    │  │ common_workflow  │  │ sglang/      │  │ benchmark │
│   .yaml         │  │   .yaml          │  │  prefill.yaml│  │   .yaml   │
│                 │  │                  │  │  decode.yaml │  │           │
│ backend +       │  │ infra tasks:     │  │  agg.yaml    │  │ aiperf /  │
│ SLURM vars      │  │ etcd, nats,      │  │              │  │ infmax    │
│                 │  │ frontend, nginx  │  │ server tasks │  │           │
└────────┬────────┘  └────────┬─────────┘  └──────┬───────┘  └─────┬─────┘
         │                    │                   │                │
         └────────────────────┴───────────────────┴────────────────┘
                                     │
                              sflow compose / run -f
                                     │
                              ┌──────▼──────┐
                              │ Merged YAML │
                              └─────────────┘
```

- **New CLI command:** `sflow compose` -- merge multiple files, resolve expressions, output a single YAML
- **Multi-file `sflow run -f`:** pass multiple `-f` flags to compose and run in one step
- **`--missable-tasks`:** gracefully handle absent tasks in `depends_on` (e.g., skip `prefill_server` in aggregated mode)
- **Merge rules:** version must match, named items merge by name (later wins), tasks concatenate

### 2. Bulk-Input Batch System

Run many configurations from a CSV file, where each row produces an independent Slurm job.

- **`sflow batch --bulk-input bulk_input.csv`** -- generate and submit jobs from CSV
- **Per-row overrides:** variable values, artifact paths, and `missable_tasks` per row
- **Parallel preflight:** dry-run validation runs in parallel (up to 16 concurrent) for fast feedback
- **`--row` selector:** test individual rows before bulk submission (`--row 0`, `--row 0:5`)
- **Auto-detect Slurm account:** batch mode can detect the user's default Slurm account

### 3. AI Agent Skills (`sflow skill`)

Packaged AI coding agent skills that teach LLM-based coding assistants (Cursor, Copilot, etc.) how to write sflow YAML and debug errors.

- **New CLI command:** `sflow skill` -- export skills into your project's `.cursor/skills/` directory
- **Two skills included:**
  - `writing-sflow-yaml` -- YAML authoring with schema reference, examples, and validation script
  - `sflow-error-analysis` -- error triage with categorized patterns, quick-fix table, and log parser
- **Four utility scripts:**
  - `validate_sflow_yaml.py` -- static validation of YAML configs
  - `check_gpu_plan.py` -- GPU allocation plan and oversubscription warnings
  - `parse_sflow_errors.py` -- categorize errors from log files (text and JSON output)
  - `summarize_run.py` -- summarize task status from an output directory

### 4. Expanded Framework Support (inference_x_v2)

Production-ready modular workflow samples for three LLM inference frameworks:

| Framework | Aggregated | Disaggregated (P/D) | Multi-Node |
|-----------|:----------:|:-------------------:|:----------:|
| SGLang    | Yes        | Yes                 | Yes        |
| vLLM      | Yes        | Yes                 | Yes        |
| TRT-LLM   | Yes        | Yes                 | Yes        |

All three frameworks share a common infrastructure layer (`common_workflow.yaml`) with NATS, etcd, frontend, nginx, and container image loading -- only the server task files differ.

### 5. DAG Visualization Improvements

- **ASCII art rendering:** `DAG.render_ascii()` draws the task graph in the terminal using Unicode box-drawing characters
- **Allocation map:** dry-run output now shows a visual node/GPU allocation summary

---

## Improvements

### Preflight Validation

- **Container image validation:** `srun` operator now validates container image URIs at config time (catches typos before Slurm allocation)
- **Task graph validation:** preflight checks for GPU oversubscription, node index bounds, and dependency cycles
- **Artifact path validation:** `fs://` artifact paths are checked during dry-run
- **Enroot credential check:** warns if `~/.config/enroot/.credentials` is missing for private registries

### Probes

- **Multi-node TCP probe:** `on_node: "each"` option for `tcp_port` probes -- waits for the port to be open on every assigned node, not just the first
- **`match_pattern` alias:** log watch probes now accept `match_pattern` as an alternative to `regex_pattern` for literal string matching
- **`match_count`:** wait for a log pattern to appear a specific number of times before marking ready (e.g., a server that logs "worker initialized" once per GPU needs `match_count: 8` on an 8-GPU node)

### Batch Mode

- **Parallel preflight validation:** bulk-input dry-run validation runs concurrently (default 16 workers)
- **`--artifact` override from CLI and CSV:** artifact URIs can be overridden per-row in CSV or via `--artifact` flag, with CLI taking precedence
- **Improved venv creation:** better fallback logic for creating Python venvs on compute nodes

### Schema

- **Node exclusion:** `resources.nodes.exclude` to exclude specific node indices from task placement
- **Node exclusion validation:** validates exclude indices don't exceed backend node count

### Performance

- **CPU performance improvements** in orchestrator loop and probe polling

---

## Documentation

- **Comprehensive user guide** with 17 pages: [nvidia.github.io/nv-sflow](https://nvidia.github.io/nv-sflow/docs/user/intro)
- **New pages:** [Architecture](https://nvidia.github.io/nv-sflow/docs/user/architecture), [Quick Reference](https://nvidia.github.io/nv-sflow/docs/user/quick-reference), [Modular Workflows](https://nvidia.github.io/nv-sflow/docs/user/modular-workflows)
- **Expanded pages:** [Quickstart](https://nvidia.github.io/nv-sflow/docs/user/quickstart) (Slurm + local sections, batch mode guide), [CLI Reference](https://nvidia.github.io/nv-sflow/docs/user/cli), [Variables](https://nvidia.github.io/nv-sflow/docs/user/variables), [Samples](https://nvidia.github.io/nv-sflow/docs/user/samples)
- **Interactive landing page** with animated sflow introduction

---

## Breaking Changes

- Deprecated standalone config files (`dynamo_gpt_oss.yaml`, `dynamo_sglang_qwen3_32b.yaml`, `dynamo_trtllm_qwen3_32b.yaml`, `dynamo_vllm_qwen3_32b.yaml`, `qwen_2_5_vllm.yaml`) have been removed in favor of modular `inference_x_v2/` samples
- Root-level `decode_config.yaml` and `prefill_config.yaml` removed
- Docs URL updated from internal GitLab to public GitHub Pages

---

## Licensing & Compliance

- Apache 2.0 license added
- SPDX headers added to all source files
- Third-party attribution file (`ATTRIBUTION.md`) included
- `CONTRIBUTING.md` added to clarify contribution policy

---

## Stats

- **231 files changed**, 49,263 insertions, 3,371 deletions
- **6,398 new lines of tests** across 59 test files
- **12 new sample workflows** (SGLang, vLLM, TRT-LLM in agg + disagg modes)
- **2 new CLI commands** (`compose`, `skill`)
- **4 new utility scripts** for AI-assisted workflow development

---

## Upgrade Guide

```bash
# Install v0.2.0
uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@main"

# Export AI agent skills to your project
sflow skill --output .cursor/skills/

# Try the new modular samples
sflow sample inference_x_v2

# Compose and run a modular workflow
sflow run \
  -f slurm_config.yaml -f common_workflow.yaml \
  -f sglang/prefill.yaml -f sglang/decode.yaml -f benchmark_aiperf.yaml \
  --missable-tasks agg_server --tui
```
