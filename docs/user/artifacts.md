---
title: Artifacts
sidebar_position: 4
---

`artifacts` are “named URIs” you can reference from expressions and task scripts.

Local schemes (`fs://` / `file://`) resolve to a local filesystem path, while
`http(s)://` artifacts **are downloaded and cached** on the controller (SHA256-keyed).
They are mainly used to:

- Normalize local paths (`fs://` / `file://`) so you can reference them consistently
- Download and cache remote files (`http(s)://`) so every task sees the same local copy
- Inject a convenient env var `${NAME}` into every task (the artifact's resolved `path`)
- Make paths available inside containers: bind-mounted when running with containers
  on **slurm** / **docker**; on **kubernetes**, `file://` inline content is injected
  via a ConfigMap and `fs://` paths are treated as remote (see below)

## Minimal example

```yaml
version: "0.1"

artifacts:
  MODEL_DIR:
    uri: fs://./models/qwen

workflow:
  name: wf
  tasks:
    - name: show_paths
      script:
        - echo "env=${MODEL_DIR}"
        - echo "expr=${{ artifacts.MODEL_DIR.path }}"
```

## Supported URI schemes

Artifacts are exposed as:

- `${{ artifacts.NAME.uri }}`
- `${{ artifacts.NAME.path }}`
- `${NAME}` (env var injected into tasks)

| Scheme | Resolves to | Notes |
|--------|-------------|-------|
| `fs://<path>` | Local filesystem path | Relative paths resolve against `--workspace-dir` (default: current directory). |
| `file://<path>` | Local filesystem path | Supports inline `content` (see below). Relative paths resolve against `--workspace-dir`. |
| `http://` / `https://` | Downloaded local copy | The file **is downloaded and cached** on the controller (SHA256-keyed), so `path` points at the cached copy and every task reads the same file. |
| `hf://` / `huggingface://` | — | **Not yet implemented** — raises `NotImplementedError` when materialized. Reference a local `fs://` / `file://` path instead. |
| `docker://` | — | **Not yet implemented** — raises `NotImplementedError` when materialized. |
| `s3://` | Raw URI (passthrough) | Accepted by URI validation but has **no artifact resolver**, so `path` stays the raw `s3://...` string — it is **not** downloaded. (To *upload* results to S3, see [Uploads](./uploads.md).) |

## Override artifacts at runtime (`--artifact`)

```bash
sflow run --file sflow.yaml --artifact MODEL_DIR=fs:///mnt/models/qwen
```

Notes:

- `--artifact` can only override artifacts that already exist in `artifacts:` (otherwise it errors).

## Variable expressions in artifact URIs

Artifact URIs can use `${{ }}` expressions to reference variables:

```yaml
variables:
  MODEL_DIR:
    value: /data/models/Qwen3-8B-FP8

artifacts:
  - name: LOCAL_MODEL_PATH
    uri: fs://${{ variables.MODEL_DIR }}
```

The expression is resolved before the artifact path is validated. This means:

- If the variable resolves to a valid path, the `fs://` path check verifies it exists
- If the variable itself is an unresolved expression (e.g. references another computed variable), the path check is skipped and deferred to runtime
- Shell variable references (e.g. `$HOME`) in URIs are also skipped during validation

### Path validation

`fs://` artifact paths are checked on disk to surface missing paths early (before allocating backend resources). The severity depends on the run mode and backend:

- **`--dry-run`**: a missing `fs://` path is a **warning** — the dry-run still succeeds.
- **Real run on a local / Slurm / Docker backend**: a missing `fs://` path **fails** the run before allocation (`Artifact path validation failed`), so create input paths first.
- **Off-host backends (e.g. Kubernetes)**: `fs://` paths are treated as **remote** paths on the cluster/image, so they are **not validated or created locally** — a missing path just warns. Ensure the path exists inside the pod (baked into the image, or mounted via a PVC/hostPath you configure).
- **`file://` paths with `content`**: skipped (the file is generated at runtime from the inline content).
- **URIs with unresolved expressions**: skipped (validated later during full resolution).

To override an artifact path at runtime:

```bash
sflow run -f workflow.yaml --artifact LOCAL_MODEL_PATH=fs:///actual/path/to/model
```

## Inline `content`

The schema allows:

```yaml
artifacts:
  CONFIG_YAML:
    uri: file://config.yaml
    content: |
      key: value
```

Artifacts with `content` are materialized to disk under the workflow output directory at runtime.
The content can use `${{ }}` expressions that are resolved before writing.
