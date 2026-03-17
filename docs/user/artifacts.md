---
title: Artifacts
sidebar_position: 4
---

`artifacts` are “named URIs” you can reference from expressions and task scripts.

In v0.1, artifacts are **not automatically downloaded/pulled**. They are mainly used to:

- Normalize local paths (`fs://` / `file://`) so you can reference them consistently
- Inject a convenient env var `${NAME}` into every task (the artifact's resolved `path`)
- Automatically mount path when running with containers inside slurm cluster

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

## Supported URI schemes (v0.1)

Artifacts are exposed as:

- `${{ artifacts.NAME.uri }}`
- `${{ artifacts.NAME.path }}`
- `${NAME}` (env var injected into tasks)

Rules in v0.1:

- `fs://<path>` and `file://<path>` become a real local filesystem path
  - relative paths are resolved relative to `--workspace-dir` (default: current directory)
- other schemes keep `path` as the raw URI string (no download/pull yet)

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

`fs://` artifact paths are validated during dry-run to catch missing paths early (before allocating Slurm nodes):

- **`fs://` paths**: must exist on disk. If missing, the dry-run fails with an error.
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
