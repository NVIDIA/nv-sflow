---
name: sflow-error-analysis
description: >-
  Diagnose and troubleshoot sflow workflow errors from log output, error messages, and
  task failures. Covers config validation errors, expression resolution failures, SLURM
  backend issues, probe timeouts, task crashes, and batch submission problems. Use when
  the user encounters an sflow error, pastes error output, asks to debug a failed workflow,
  or asks about sflow troubleshooting.
---

# sflow Error Analysis

## Triage Flow

1. **Identify category** from the error message (see table below)
2. **Locate source** -- check the file/line hint or task log
3. **Apply fix** from the quick-fix table or [error-catalog.md](error-catalog.md)

## Error Categories

| Category            | Marker                                           | Where to Look             |
|---------------------|--------------------------------------------------|---------------------------|
| Config loading      | `Configuration error:`, `File not found:`        | CLI output / sflow.log    |
| YAML syntax         | `Error parsing YAML`                             | CLI output                |
| Expression          | `Undefined variable`, `Invalid expression syntax`| CLI output                |
| Artifact            | `Artifact path validation failed`                | CLI output / dry-run      |
| Merge conflict      | `Version conflict`, `Workflow name conflict`     | CLI output (multi-file)   |
| SLURM               | `salloc failed`, `sbatch failed`                 | CLI / sbatch stderr       |
| Probe timeout       | Task stuck waiting                               | `<task>/<task>.log`       |
| Task failure        | `Traceback`, non-zero exit                       | `<task>/<task>.log`       |
| Batch/CSV           | `CSV file`, `sflow_config_file column`           | CLI output                |

## Log Locations

```
<output_dir>/<run_id>/
  sflow.log                   # orchestrator log
  <task_name>/<task_name>.log  # task stdout/stderr
  <task_name>_0/               # replica 0
```

Batch jobs: also check the `.sh` script and sbatch stdout/stderr files.

## Quick-Fix Table

| Error Text                                              | Fix                                                |
|---------------------------------------------------------|----------------------------------------------------|
| `Configuration file not found: <path>`                  | Check `-f` path                                    |
| `Error parsing YAML configuration: <detail>`            | Fix YAML syntax at indicated line                  |
| `Configuration validation failed: <detail>`             | Check field types/values (schema-reference.md)     |
| `Variable '<key>' specified in overrides is not defined` | Declare variable in YAML first                    |
| `Undefined variable in expression`                      | Fix spelling; ensure variable is declared          |
| `Invalid expression syntax`                             | Fix `${{ }}` Jinja2 syntax                         |
| `Artifact path validation failed`                       | Fix `fs://` path or create it                      |
| `Version conflict`                                      | All files must use `version: "0.1"`                |
| Probe timeout (task hangs)                              | Check task log; adjust `regex_pattern` / `timeout` |
| `Traceback` in task log                                 | Read traceback for root cause                      |
| `salloc failed`                                         | Check partition/account/availability               |
| `CSV file must contain a 'sflow_config_file' column`   | Add required column to CSV                         |

## Quick Diagnostic Commands

```bash
# Validate config without executing
sflow run -f config.yaml --dry-run

# See resolved expressions
sflow compose file1.yaml file2.yaml --resolve -o merged.yaml

# Check SLURM environment
sinfo -p <partition>
sacctmgr show account <name>

# Parse errors from log
python scripts/parse_sflow_errors.py <sflow.log>
python scripts/parse_sflow_errors.py - < <(sflow run -f config.yaml --dry-run 2>&1)

# Summarize a run's output
python scripts/summarize_run.py <output_dir>/<run_id>/
```

## Common Runtime Issues

- **OOM**: Reduce batch size, TP/DP, or model size
- **Port conflict**: Kill stale processes or adjust port offsets
- **Container not found**: Verify image URI and registry access
- **NCCL errors**: Set `NCCL_SOCKET_IFNAME`, check inter-node network
- **Probe mismatch**: Server output doesn't match `regex_pattern`

## Additional Resources

- Exhaustive error patterns: [error-catalog.md](error-catalog.md)
- Full docs: [faq](https://nvidia.github.io/nv-sflow/docs/user/faq),
  [cli](https://nvidia.github.io/nv-sflow/docs/user/cli),
  [configuration](https://nvidia.github.io/nv-sflow/docs/user/configuration),
  [backends](https://nvidia.github.io/nv-sflow/docs/user/backends)
