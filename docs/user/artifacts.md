---
title: Artifacts
sidebar_position: 4
---

`artifacts` are “named URIs” you can reference from expressions and task scripts. Every
artifact is exposed three ways — `${{ artifacts.NAME.uri }}`, `${{ artifacts.NAME.path }}`,
and the env var `${NAME}` injected into every task.

An artifact is more than a stored path: it is a *declaration that a path matters*, so sflow
acts on it — it resolves and validates the path, downloads and caches `http(s)://` files,
writes out `file://` inline `content`, and — the key part — **makes the path available to the
task wherever the task actually runs, including inside a container and on a remote host.**
That last property is why you should prefer an artifact over a bare path string (next section).

## Prefer an artifact over a raw path in a variable

You can put a path in a variable and use `${MY_PATH}` in a script — but a variable is just a
string, and sflow does nothing with it: it is not resolved, not validated, not shipped
anywhere, and **not mounted into a container**. Declare the path as an artifact instead and you
get, on every backend:

- **Auto-mount into containers.** When a task runs in a container, sflow makes the artifact's
  path available **inside** the container at the **same absolute path** it has outside — so
  `${{ artifacts.X.path }}` and `${X}` point at the same real file whether or not a container
  is involved, and you never hand-write `container_mounts`, `docker -v`, or k8s `volumes`. A
  raw variable path gets none of this: inside the container it points at a path that usually
  does not exist.
- **Resolution & early validation** — relative paths resolve against the workspace, and a
  missing path fails before you allocate a backend (see [Path validation](#path-validation)).
- **Fetching & materialization** — `http(s)://` files are downloaded and cached; `file://`
  inline `content` is written to disk for you.

**Rule of thumb: if a task reads or writes a path, declare it as an artifact** and reference
`${{ artifacts.X.path }}` / `${X}` — don't pass the path as a plain variable string.

## Artifacts and where the task actually runs

The machine you run `sflow run` on (the controller) and the machine a task runs on are often
**not the same**: a Slurm compute node, a Docker container, or — most starkly — a **remote
Kubernetes pod on another host**. A filesystem path that exists on the controller means nothing
over there, so *how an artifact reaches the task* is the whole point. One general rule holds on
every backend:

> A **`file://` artifact with inline `content` ships itself** to wherever the task runs; an
> **`fs://` artifact names storage the execution host must already be able to see.**

- **`file://` + `content` — portable, self-shipping.** sflow materializes the content and
  places it at the artifact's path wherever the task runs. Because sflow moves the bytes, it
  works identically on local, Docker, Slurm, and Kubernetes with no mount setup from you — the
  right choice for small text you author in the recipe (launch scripts, configs, engine specs).
- **`fs://` — you provide the storage.** sflow resolves and mounts the path, but the bytes must
  exist where the task runs. On a single host that is automatic; across hosts you make the path
  reachable — e.g. **shared storage** (Lustre/GPFS/NFS) on multi-node Slurm. On **Kubernetes**
  the pod is remote and cannot see the controller's disk, so an `fs://` path is treated as
  **remote**: it must exist **inside the image** or be mounted from a **PVC** (declare the PVC
  under the backend's `volumes:` at a `mount_path`, then point the `fs://` artifact at a path
  under it). See [Path validation](#path-validation) for how each case is checked.

So: inline small text as `file://` `content` (it travels with the task), and reserve `fs://`
for large, pre-existing data (models, datasets, checkpoints) on storage the execution host can
see.

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
