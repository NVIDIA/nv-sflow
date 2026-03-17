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

## Triage Workflow

When encountering an sflow error, follow this sequence:

1. **Identify the error category** -- read the error message and match it against the categories below
2. **Locate the source** -- determine which file/line caused the error
3. **Apply the fix** -- use the quick-fix table or the detailed catalog

## Error Categories

| Category              | Marker / Pattern                                    | Where to Look                      |
|-----------------------|-----------------------------------------------------|------------------------------------|
| Config loading        | `Configuration error:`, `File not found:`           | CLI output / sflow.log             |
| YAML syntax           | `Error parsing YAML`                                | CLI output                         |
| Expression resolution | `Undefined variable`, `Invalid expression syntax`   | CLI output with source line hint   |
| Artifact validation   | `Artifact path validation failed`                   | CLI output / dry-run               |
| Merge conflict        | `Version conflict`, `Workflow name conflict`        | CLI output (multi-file compose)    |
| SLURM backend         | `scontrol`, `salloc failed`, `sbatch failed`        | CLI output / sbatch stderr         |
| Probe timeout         | Task stuck waiting, no `readiness` match            | Task log: `<task>/<task>.log`      |
| Task failure          | `Traceback`, non-zero exit code                     | Task log: `<task>/<task>.log`      |
| Batch/CSV errors      | `CSV file`, `sflow_config_file column`              | CLI output                         |

## Log File Locations

sflow output is organized as:

```
<output_dir>/<run_id>/
  sflow.log                    # Orchestrator log (task lifecycle, probes, errors)
  <task_name>/
    <task_name>.log            # Task stdout/stderr
  <task_name>_0/               # Replicated task (replica index 0)
    <task_name>_0.log
```

For batch jobs, also check:
- The generated `.sh` script (sbatch wrapper)
- sbatch stdout/stderr files (`--sbatch-output`, `--sbatch-error`)

## Quick-Fix Table

| Error Text                                                 | Cause                                        | Fix                                                        |
|------------------------------------------------------------|----------------------------------------------|-------------------------------------------------------------|
| `Configuration file not found: <path>`                     | Wrong file path                              | Check the `-f` argument path                                |
| `Configuration file is empty: <path>`                      | Empty YAML file                              | Add content to the file                                     |
| `Error parsing YAML configuration: <detail>`               | YAML syntax error                            | Fix YAML indentation/syntax at the indicated line           |
| `Configuration validation failed: <detail>`                | Schema violation (Pydantic)                  | Check field types and allowed values in schema reference    |
| `Variable '<key>' specified in overrides is not defined`   | `--set` with unknown variable                | Declare the variable in the YAML first                      |
| `Artifact '<name>' specified in overrides is not defined`  | `--artifact` with unknown artifact           | Declare the artifact in the YAML first                      |
| `Undefined variable in expression <expr>`                  | Typo or missing variable                     | Check variable name spelling; ensure it's declared          |
| `Invalid expression syntax in '<value>'`                   | Bad `${{ }}` Jinja2 syntax                   | Fix the expression; check brackets, filters, operators      |
| `Artifact path validation failed: ... does not exist`      | `fs://` points to non-existent path          | Fix the path or create the directory/file                   |
| `Version conflict: '<a>' vs '<b>'`                         | Mismatched `version` across files            | All files must use `version: "0.1"`                         |
| `Merged configuration ... is incomplete`                   | Missing version/workflow/tasks after merge   | Ensure all required sections exist across merged files      |
| Probe timeout (task hangs at readiness check)              | Server didn't emit expected log pattern      | Check task log for startup errors; adjust `regex_pattern`   |
| `Traceback (most recent call last)` in task log            | Python exception in task                     | Read the full traceback in the task log for root cause       |
| `salloc failed with exit code <N>`                         | SLURM allocation failure                     | Check partition/account/node availability                    |
| `sbatch failed: <stderr>`                                  | sbatch submission error                      | Check SLURM args, partition, account                        |
| `CSV file must contain a 'sflow_config_file' column`       | Missing required CSV column                  | Add `sflow_config_file` column to CSV                       |

## Diagnostic Steps by Category

### Config/YAML Errors

1. Run `sflow run -f <file> --dry-run` to validate without executing
2. Look for `Source: line N in file.yaml` hints in the error message
3. Check YAML indentation -- sflow uses strict parsing (`extra="forbid"`)
4. Validate with: `python skills/writing-sflow-yaml/scripts/validate_sflow_yaml.py <file>`

### Expression Errors

1. The error message includes the failing expression and location
2. Common issues:
   - Referencing `task.<name>` in a YAML field (only works in scripts)
   - Misspelled variable name
   - Missing closing `}}`
   - Using Python syntax not supported in Jinja2 sandboxed env
3. Use `sflow compose ... --resolve` to see resolved expressions

### Probe / Runtime Errors

1. Check the task log at `<output_dir>/<run_id>/<task>/<task>.log`
2. For probe timeouts: the server likely crashed before emitting the readiness pattern
3. Search the task log for `Traceback`, `Error`, `CUDA`, `OOM`
4. Common runtime issues:
   - **OOM**: Reduce batch size, TP size, or model size
   - **CUDA errors**: Check GPU availability, driver compatibility
   - **Port conflicts**: Another process using the port; adjust port or kill stale process
   - **Container not found**: Verify container image URI; check registry access
5. For failure probes triggering: the task emitted the failure pattern (usually `Traceback`)

### SLURM Errors

1. Check `scontrol show partition <name>` for partition existence
2. Check `sacctmgr show account <name>` for account validity
3. Verify node availability: `sinfo -p <partition>`
4. For multi-node jobs: ensure requested nodes don't exceed partition limits
5. Check `--extra_args` for invalid sbatch/srun flags

### Batch/CSV Errors

1. Verify CSV has a `sflow_config_file` column
2. Check that all file paths in CSV are accessible
3. Column names must match declared variables or artifacts
4. Use `--row 0` to test a single row before bulk submission

## Automated Error Parsing

Run the error parsing script to categorize errors from a log file:

```bash
python skills/sflow-error-analysis/scripts/parse_sflow_errors.py <sflow.log>

# Or pipe from command output:
sflow run -f config.yaml --dry-run 2>&1 | python skills/sflow-error-analysis/scripts/parse_sflow_errors.py -
```

## Additional Resources

- For exhaustive error catalog, see [error-catalog.md](error-catalog.md)
- If the above don't cover your issue, fetch the relevant page from the online docs
  at https://nvidia.github.io/nv-sflow/docs/user/faq -- also see `cli`, `configuration`,
  `backends` (same URL pattern)
