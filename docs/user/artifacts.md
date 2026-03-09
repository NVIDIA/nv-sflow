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

## Inline `content` (current limitation)

The schema allows:

```yaml
artifacts:
  CONFIG_YAML:
    uri: file://config.yaml
    content: |
      key: value
```

But in v0.1, `content` is only validated (must be `file://...`) and is **not materialized to disk yet**.
If you reference `${CONFIG_YAML}`, make sure the file already exists in your workspace.
