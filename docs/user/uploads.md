---
title: Uploads
sidebar_position: 9
---

# Uploads (post-execution storage)

sflow can automatically upload selected files to cloud storage (S3 today; GCS/Azure are
planned as additional plugins) as each task completes. You declare named **storage targets**
at the top of `sflow.yaml`, then attach per-task `uploads:` specs.

Uploads run **per task, on `COMPLETED`** — partial results survive a fail-fast workflow
cancellation.

## YAML shape

```yaml
storage:
  - name: results_bucket
    type: s3
    bucket: my-bench-results
    region: us-west-2
    prefix: "runs/${{ variables.RUN_ID }}/"     # optional; prepended to every remote key

workflow:
  tasks:
    - name: benchmark
      script: [...]
      uploads:
        - target: results_bucket
          from: "${{ task.output_dir }}/results.csv"
          to: "main/results.csv"     # optional; relative to the target's prefix
          on_error: fail              # warn (default) | fail

        - target: results_bucket
          from: "${{ task.output_dir }}/*.json"   # glob
          on_error: warn
```

## Field reference

### Storage target

| Field | Required | Description |
|---|---|---|
| `name` | yes | Name referenced by each upload spec's `target:` |
| `type` | yes | Plugin discriminator. `s3` today |
| `bucket` | yes (s3) | S3 bucket name |
| `region` | no | AWS region. Defaults to boto3's chain |
| `prefix` | no | Key prefix prepended to every uploaded object |
| `endpoint_url` | no | For S3-compatible stores (MinIO, Ceph RGW, ...) |
| `addressing_style` | no | `auto`, `virtual`, or `path`. Defaults to `path` when `endpoint_url` is set, else boto3's default. Most S3-compatible endpoints require `path`. |
| `storage_class` | no | S3 storage class (e.g. `STANDARD_IA`) |

### Upload spec

| Field | Required | Description |
|---|---|---|
| `target` | yes | Name of a declared storage entry |
| `from` | yes | Local file path or glob. May contain `${{ }}` expressions |
| `to` | no | Remote key (relative to the target's `prefix`). If omitted, the local basename is used. If `to` ends with `/` it is treated as a directory and the basename is appended. **When `from` is a glob, `to` must end with `/`** (or be omitted) — a literal `to:` against multiple matches is rejected at validation time to avoid silent layout changes between runs. `..` segments are rejected to prevent prefix escape. For replicated tasks, the basename is auto-suffixed with `_${{ task.name }}` unless `to:` already references `${{ task.name }}` (see [Replicated tasks](#replicated-tasks)). |
| `on_error` | no | `warn` (default) logs and continues; `fail` flips the task to `FAILED` (and triggers fail-fast if enabled) |

## Expression resolution timing

| Field | Resolved when | Available context |
|---|---|---|
| Storage target fields (`bucket`, `prefix`, ...) | Assembly (once per run) | `variables.*`, `backends.*`, `artifacts.*` |
| Upload `from` / `to` | Task completion (per upload) | `task.output_dir`, `task.backend`, `task.operator`, `task.assigned_nodes`, plus the task's env vars via `env.*` |

`${{ task.output_dir }}` resolves to the task's `SFLOW_TASK_OUTPUT_DIR` — the per-task subdirectory under `sflow_output/<run_id>/`.

## Replicated tasks

When a task has `replicas:`, each replica runs as `<task>_0`, `<task>_1`, ... Each replica has its own `task.output_dir`, so `from:` works correctly out of the box. The destination, however, is shared: a `to:` of `main/results.csv` (or an omitted/`main/`-style `to:`) would resolve to the same remote key for every replica and silently overwrite — last-writer-wins, with no error.

**By default sflow prevents this by auto-renaming each replica's upload.** It inserts the replica name (`${{ task.name }}`, e.g. `benchmark_0`) into the filename, just before the extension:

| `to:` | Replica `benchmark_0` uploads to |
|---|---|
| `main/results.csv` | `main/results_benchmark_0.csv` |
| `main/` (directory) | `main/results_benchmark_0.csv` |
| *(omitted)* | `results_benchmark_0.csv` |

The rename applies per file, so globs are disambiguated too (`a.json`, `b.json` → `a_benchmark_0.json`, `b_benchmark_0.json`). sflow logs a warning the first time it auto-renames a spec so the behavior is never silent.

**To control the layout yourself**, reference `${{ task.name }}` anywhere in `to:`. sflow then leaves the destination exactly as written (no auto-rename, no warning):

```yaml
- target: results_bucket
  from: "${{ task.output_dir }}/results.csv"
  to: "${{ task.name }}/results.csv"      # -> benchmark_0/results.csv
```

Non-replicated tasks (a single replica or no `replicas:`) are never auto-renamed.

## S3-compatible stores (MinIO, Ceph RGW)

Set `endpoint_url`; sflow defaults `addressing_style` to `path`, which is what most non-AWS deployments require. Example:

```yaml
storage:
  - name: minio
    type: s3
    endpoint_url: "http://minio.local:9000"
    bucket: bench-results
    # addressing_style: path  # default when endpoint_url is set; override if needed
```

## Credentials

S3 uses boto3's default credential chain. **Never put secrets in YAML.** Supported sources:

- Env vars: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- `~/.aws/credentials` (and `AWS_PROFILE` to select a named profile)
- EC2 instance / EKS pod IAM role (no extra config needed)

## Optional dependency

S3 support requires `boto3`. Install via the extra:

```bash
pip install 'sflow[s3]'
```

Using an S3 storage target without `boto3` raises a friendly error during planning, before task execution starts.

## Dry-run

`sflow run --dry-run` lists every planned upload (target, `from`, `to`, `on_error`) without contacting any remote. Use this to verify the upload plan before a live run.

For S3 targets, the dry-run also runs an **offline credential preflight** under the
`Storage targets` section: it warns if `boto3` isn't installed, or if no AWS
credentials are detected (no `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or
`AWS_PROFILE` env vars and no `~/.aws/credentials` file). No network call is made,
so an attached IAM role can't be detected here — the warning is a heads-up, not a
hard failure.

## Failure semantics

| Spec | Outcome of an upload failure |
|---|---|
| `on_error: warn` (default) | A warning is logged. Task stays `COMPLETED`. Workflow continues |
| `on_error: fail` | The task flips to `FAILED`. If `fail_fast` is enabled (default), the rest of the workflow is cancelled |

An upload failure is **terminal** — failed uploads are not retried via the task's `retries:` block. Retries cover the user's script, not the post-execution upload.

## Workflow-level upload (bundle the whole run)

Per-task `uploads:` are great when you know which files matter ahead of time.
For a "just archive everything" workflow — logs, parsed outputs, per-task
artifacts — set `workflow.upload_all` instead.  When configured, sflow zips the
entire `workflow_out_dir` after the orchestrator finishes and uploads it as a
single object to the named storage target.

```yaml
storage:
  - name: results_bucket
    type: s3
    bucket: my-bench-results
    prefix: "runs/"

workflow:
  name: my_workflow
  upload_all:
    target: results_bucket
    to: "archive/${{ workflow.run_id }}.zip"   # optional; default: <run_id>.zip
    on_error: warn                              # warn (default) | fail
  tasks: [...]
```

- The zip is produced from `SFLOW_WORKFLOW_OUTPUT_DIR` (the per-run folder
  under `sflow_output/`). All regular files under that directory are included,
  preserving relative paths.
- The upload runs **after `orch.run()` returns**, regardless of whether some
  tasks failed — so partial results are still shipped. Per-spec `on_error`
  governs how a failure is handled, mirroring per-task uploads. Both the
  zip-packaging step (e.g. disk full) and the remote transfer go through
  `on_error`.
- `to:` is resolved at run-end against `${{ workflow.name }}`,
  `${{ workflow.run_id }}`, `${{ workflow.output_dir }}`, and `variables.*`.
  `..` segments are rejected.

`workflow.upload_all` is independent of per-task `uploads:` — you can use both,
either, or neither.

For a runnable example, see `examples/local_storage_upload_all.yaml` in the repository.

## See also

- [Architecture](./architecture.md) — where storage fits in the plugin model and assembly pipeline
- [Variables](./variables.md), [Artifacts](./artifacts.md) — for the expressions that drive `from`/`to`/`prefix`
