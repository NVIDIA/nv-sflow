---
name: writing-sflow-yaml
description: >-
  Write, create, and modify sflow YAML workflow configuration files. Covers schema structure,
  variables, task DAGs, backends (local, slurm, docker, kubernetes), operators, probes,
  replicas, artifacts, result parsing, storage/uploads (S3), hardware monitoring, and modular
  composition.
  Use when the user asks to create an sflow YAML, configure a workflow, set up inference
  serving on Slurm or Kubernetes, or asks about sflow YAML syntax.
---

# Writing sflow YAML Configurations

sflow turns **one portable recipe** — tasks, wiring, resources — into backend-native
execution on `local` / `slurm` / `docker` / `kubernetes`. Write recipes **shape-first**:
get the DAG right, *then* fill in scripts, scaling, resources, and backend specifics.

**Follow the steps below in order.** The most common mistake is jumping straight to GPU
counts and backend flags before the workflow shape is correct. Don't. Design the DAG first;
everything else hangs off it.

> Deep field reference → [schema-reference.md](schema-reference.md). Runnable, annotated
> recipes → [examples.md](examples.md) (cited per step). Full docs →
> [nvidia.github.io/nv-sflow/docs/user/intro](https://nvidia.github.io/nv-sflow/docs/user/intro).

## The shared skeleton

Every recipe has the same top-level shape. You design `workflow.tasks` first; the rest is
optional scaffolding you add only when a step calls for it.

```yaml
version: "0.1"        # schema version (NOT the sflow release) — required in EVERY file
variables: { ... }    # optional — ${{ variables.X }} in YAML, ${X} in scripts
artifacts: [ ... ]    # optional — named paths/URIs, auto-mounted at the same path
backends:  [ ... ]    # optional — omit ⇒ a default `local` backend
operators: [ ... ]    # optional — reusable launch presets
storage:   [ ... ]    # optional — S3 upload targets
workflow:
  name: ...
  tasks: [ ... ]      # THE DAG — design this first (Step 1)
```

Omit `backends`/`operators` and you get `local` + `bash` — enough to prototype a DAG on a
laptop before touching a cluster.

---

## Step 1 — Shape the DAG (do this first)

The DAG *is* the recipe. Decide **what the tasks are and how they connect** before writing
any script body.

1. **List the tasks** — one node per logical step or service (`etcd`, `frontend`,
   `prefill`, `decode`, `benchmark`, …).
2. **Wire them** with `depends_on` (a task lists the upstreams it waits for). A dependent
   starts once every upstream is **READY** (if that upstream has a readiness probe, Step 4)
   or **COMPLETED** (if it has none).
3. **Identify the terminal task** — the one whose completion ends the run (usually the
   client/`benchmark`). Long-running servers never "complete"; sflow stops them when the
   terminal task finishes.
4. **Reusable fragments wire in reverse** with `required_by:` — a module attaches itself to
   a shared hub (e.g. `benchmark`) without editing the hub or adding cross-file `depends_on`.

Canonical serving shape (Dynamo-style — see examples.md Example 3):

```text
infra (nats, etcd) ──▶ frontend ──▶ workers (prefill + decode | agg) ──▶ benchmark ──▶ end
```

Write the wiring skeleton first — **script bodies come in Step 2**:

```yaml
workflow:
  name: serve_and_bench
  tasks:
    - { name: etcd }
    - { name: frontend,  depends_on: [etcd] }
    - { name: worker,    depends_on: [frontend] }
    - { name: benchmark, depends_on: [worker] }        # terminal task ⇒ run ends here
```

Sanity-check the shape now: *does anything start before its inputs exist? is there exactly
one clear end? do "parallel" branches actually need to be concurrent?* Fixing shape is cheap
here and expensive later.

## Step 2 — Write the task scripts

- `script:` is a **list of lines joined into one bash script** (the `python` operator instead
  runs the lines as Python *source* via `python -c`).
- **Don't inline files or hard-code paths.** Any file a script needs — launch/helper script,
  config, model dir, dataset — belongs in an **artifact** (Step 3), referenced as
  `${{ artifacts.X.path }}`. Never embed Python via heredocs or `python3 -c` — quoting breaks
  when YAML → shell → Python nests.
- **Reference values:** `${{ variables.X }}` in YAML (plan-time), `${X}` in scripts (runtime).
  For arithmetic/comparison, set the variable's `type` (`integer`, …) or it stays a string.

```yaml
workflow:
  tasks:
    - name: worker
      depends_on: [frontend]
      script:
        - python3 ${{ artifacts.LAUNCH.path }} ${{ variables.MODEL }}   # LAUNCH: see Step 3
```

See examples.md Examples 1–2.

## Step 3 — Handle paths with artifacts (don't hand-manage paths)

**An artifact is a named path/URI that sflow resolves, validates, and — inside containers —
auto-mounts at the *same absolute path* on every backend.** This is the biggest lever for
portability: declare a path once and you stop caring *where* it lives or *how* to mount it.
`${{ artifacts.X.path }}` (YAML, plan-time) and `${X}` (script, runtime) always agree, sflow
**auto-checks** the reference at preflight, and **auto-mounts** it — so you never hand-write
`container_mounts`, `docker -v`, or k8s `volumes` for it (doing so is redundant — pitfall 13).

Pick the scheme by what the artifact holds:

- **Small code snippets & config files → a `file://` artifact with inline `content:`.** sflow
  writes the content and mounts it into the container at the same path on *any* backend —
  local, docker, slurm, kubernetes — so **you don't manage where it lives or how it gets into
  the container**. Use it for launch/helper scripts, config JSON/YAML, engine specs, etc.
  **Never** inline them via heredocs / `python3 -c` (YAML → shell → Python quoting breaks) or
  bake them into the image.

  ```yaml
  artifacts:
    - name: LAUNCH
      uri: file://launch.py             # relative → resolves under SFLOW_WORKSPACE_DIR
      content: |
        import sys; print("serving", sys.argv[1])
    - name: GEN_CONFIG
      uri: file://gen_config.yaml
      content: |
        max_batch_size: 256
        kv_cache_free_gpu_mem_fraction: 0.9
  workflow:
    tasks:
      - name: worker
        script:
          - python3 ${{ artifacts.LAUNCH.path }} --config ${{ artifacts.GEN_CONFIG.path }}
  ```

- **Large / pre-existing data (models, datasets, checkpoints) → an `fs://` artifact** pointing
  at storage the compute nodes already see. It still resolves + mounts the same path
  everywhere, but *you* supply the bytes:
  - **slurm** → keep it on **shared storage** (Lustre/GPFS/NFS) so the path exists on every node.
  - **kubernetes** → back it with a **PVC**: declare the PVC under backend `volumes:` at a
    `mount_path`, then point the `fs://` artifact at a path *under* that mount. An `fs://` path
    no PVC covers and the node lacks makes the pod fail (pitfall 9).

**Rule of thumb: if a task touches a path, declare it as an artifact** — small text → `file://`
with `content:`, big data → `fs://` on shared storage / a PVC. Artifact schemes & `content:`
rules → schema-reference.md.

## Step 4 — Gate the wiring with probes

`depends_on` alone only waits for a process to *start*. For services, wait for **ready**.

- **Every long-running server gets a `readiness` probe** so dependents wait for real up-ness:
  `log_watch` (server logs a ready line), `tcp_port` (port open), or `http_get`/`http_post`
  (health endpoint).
- **Add a `failure` probe** on servers — `failure.log_watch` for
  `"Traceback (most recent call last)"` fails the run fast instead of hanging until timeout.
- `log_watch` matches **literally** by default (parentheses/dots aren't special); prefix
  `re:` / `regex:` for real regex.
- **Don't probe short-lived one-shot tasks** (benchmarks, setup steps).
- Defaults: `timeout` 1200 s (readiness deadline only), `interval` 5, `each_check_timeout`
  30, `success_threshold` 1, `failure_threshold` 3.

```yaml
- name: frontend
  depends_on: [etcd]
  script: [ "python -m my.frontend --port 8000" ]
  probes:
    readiness: { tcp_port: { port: 8000 } }
    failure:   { log_watch: { regex_pattern: "Traceback (most recent call last)" } }
```

(On **kubernetes**, tcp/http probes run from an in-cluster probe pod automatically — the
driver host usually can't route to the pod network.) See the probes doc.

## Step 5 — Scale with replicas & sweeps

- **Replicate a task:** `replicas: { count: N, policy: parallel|sequential }`.
  `parallel` = all at once (downstream waits for all); `sequential` = one-by-one (sweeps).
- **Parameter sweep:** `replicas.variables: [X]` where `X` declares a `domain:` → one replica
  per value (Cartesian product across multiple swept vars). Each replica gets
  `SFLOW_REPLICA_INDEX` and the swept values in its env; omit `count` and it equals the sweep
  size.

See examples.md Example 5; the replicas doc.

## Step 6 — Place on nodes & GPUs

- **Pin placement** with `resources.nodes` (`indices` / `count` / `exclude`; negative indices
  count from the end). **Request GPUs** with `resources.gpus.count`.
- `gpus.count` is a per-task **total** across the task's nodes, and every node takes the
  same slice — so when you also pin `resources.nodes`, `count` must be a positive multiple
  of the node count. `nodes.count: 2` + `gpus.count: 1` is rejected; write `gpus.count: 2`
  for one GPU on each of two nodes.
- On `local`/`slurm`/`docker`, sflow slices `CUDA_VISIBLE_DEVICES` per task/replica — set the
  backend's `gpus_per_node` so it can pack.
- **GPU planning:** `GPUs/worker = TP × DP × PP` (framework-dependent — some use attention-DP
  / expert parallelism), `total = GPUs/worker × replicas`, `nodes = ceil(total / gpus_per_node)`.
  When unsure of a framework's parallelism, ask the user.
- Use `resources.*.release_after` (`task_ready` / `task_completion` / `workflow_completion`)
  to let later tasks reuse a reservation.

See the resources doc.

## Step 7 — Backend-specific handling

Pick a backend and apply *its* rules. Start `local`, move to a cluster once the DAG is proven.

**`local`** — no config needed (default). `bash` operator. Set `nodes: N` to simulate a
multi-node placement on one machine.

**`slurm`** — default operator `srun`. Backend needs `partition`, `account`, `time`
(HH:MM:SS or int minutes), `nodes`, `gpus_per_node`. Containers: put `container_image` /
`container_name` on an `srun` operator (reuse the image across tasks by `container_name`).
Keep models / workspace / **output dir** on **shared storage** (Lustre/GPFS/NFS) so an
`fs://` artifact auto-mounts at the same path on every node.

```yaml
backends:
  - name: slurm
    type: slurm
    default: true
    nodes: 1
    partition: ${{ variables.PART }}      # block style: ${{ }} values can't sit in a { } flow map
    account: ${{ variables.ACCT }}
    time: "01:00:00"                       # HH:MM:SS or int minutes
    gpus_per_node: 8
operators:
  - { name: ctr, type: srun, container_image: nvcr.io/nvidia/pytorch:24.07-py3,
      container_writable: true, mpi: pmix }
```

**`docker`** — backend is `type: docker`, its launch operator is `type: docker_run` (mind
the naming). The `image` can live on the backend or the operator. Remote/multi-host via a
`hosts:` list.

**`kubernetes`** — the workload **`image` lives on the `k8s` operator**, never the backend
(the backend reserves + pins nodes with placeholder pods). Rules that bite:
- Cluster access is **CLI-only** (`--kubeconfig` / `--kube-context` / `--kube-namespace`) so
  recipes stay cluster-agnostic; `namespace` is a backend field (one namespace per backend).
- `resources.gpus.count` is a per-task **total**, split evenly across the task's nodes → must
  be a multiple of the node count. For an ordinary pod sflow leaves `CUDA_VISIBLE_DEVICES`
  unset (the device-plugin/DRA assigns the GPUs); when co-located GPU tasks are **merged into
  one pod** (`merge_colocated_gpu_pods`, default `auto`) it sets a per-member
  `CUDA_VISIBLE_DEVICES` to pin each task's slice.
- A single-node task → one pod; a `resources.nodes` task → one pod per node (leader = index 0).
- **Cluster data → a PVC**: declare it under backend `volumes:` at a `mount_path`, then point
  an `fs://` artifact at a path *under* that mount. Output dirs are ephemeral `emptyDir` — for
  persistence use a PVC or `uploads:` (Step 8).
- **Any MPI workload** (`mpirun`-launched, e.g. TRT-LLM `trtllm-llmapi-launch`) → `type: k8s_mpi`,
  and write the `mpirun -np N …` **explicitly** in the script: sflow reads it to set up the MPI
  world for **both single- and multi-node** and injects the keypair/hostfile/sshd/`-x` env
  forwarding. Reserve plain `k8s` for non-MPI, single-pod workloads.

```yaml
backends:
  - { name: k8s, type: kubernetes, default: true, namespace: default, nodes: 1,
      gpus_per_node: 8, scheduling: device_plugin }   # device_plugin default; dra is WIP
operators:
  - { name: gpu_op, type: k8s, image: nvcr.io/nvidia/pytorch:24.12-py3 }
```

See the backends doc; examples.md Examples 8 (k8s), 10 (k8s + PVC), 11 (slurm + Lustre).

## Step 8 — Capture outputs

- **Metrics:** add `result:` (a regex map, `patterns:`, or a JSON `file:`) to parse a task's
  log/output into `${SFLOW_TASK_OUTPUT_DIR}/result.json` plus a workflow `results.json`.
  Prefer it over the legacy `outputs:`. A task isn't `COMPLETED` until parsing finishes.
- **Ship files:** name `storage:` S3 targets, then per-task `uploads:` (fire on `COMPLETED`)
  or `workflow.upload_all` (zip the whole run). Credentials come from the **boto3 default
  chain** — never put secrets in YAML; needs the `sflow[s3]` extra.
- **Telemetry:** `monitor:` samples GPU/CPU/mem/net on `local`/`slurm`/`docker` (not k8s yet),
  or enable it without editing the recipe via `--enable-workflow-monitor` /
  `--enable-task-monitor`.

See examples.md Examples 12 (result parsing), 7 (monitor), 9 (uploads); the results / uploads / monitor docs.

## Step 9 — Validate before running

Run the one-command preflight before every real run (static schema + reference checks + GPU
plan + `sflow --dry-run`, with a root-cause explanation on failure). **Exit 0 = ready.**

```bash
python scripts/preflight.py my_workflow.yaml
python scripts/preflight.py slurm.yaml common.yaml sglang/agg.yaml --set TP_SIZE=8   # composed
```

Standalone steps when you want just one:

```bash
python scripts/validate_sflow_yaml.py my_workflow.yaml   # schema, refs, k8s volumes
python scripts/check_gpu_plan.py     my_workflow.yaml    # GPU plan + oversubscription
sflow run -f my_workflow.yaml --dry-run                  # full plan + expression resolution
```

---

## Modular composition (when splitting across files)

Pass fragments directly: `sflow run -f slurm.yaml -f common.yaml -f sglang/agg.yaml`.
Files are combined with a **recursive deep merge keyed on `name`** — `version` must match;
`variables` / `artifacts` / `backends` / `operators` / `storage` and same-name
`workflow.tasks` are deep-merged (one definition can be scattered across files); leaf
scalars/lists are last-file-wins with a warning. Prefer `required_by:` on optional fragments
(absent targets skip silently) over `--missable-tasks`. See examples.md Example 4.

## Pitfalls checklist

1. Missing `version: "0.1"` in **every** file (including fragments).
2. `${{ task.X.nodes }}` in a YAML field — task context works only in scripts.
3. Forgetting `depends_on` — tasks run immediately without it.
4. GPU oversubscription: Σ GPU counts > `nodes × gpus_per_node`.
5. Probing a short-lived one-shot task (don't probe benchmarks).
6. Expecting `log_watch` to be regex — it's **literal** unless prefixed `re:` / `regex:`.
7. Docker: the operator is `type: docker_run` (the backend is `type: docker`).
8. Putting the workload `image` on the **kubernetes backend** — it belongs on the `k8s`/`k8s_mpi` **operator**.
9. K8s: an `fs://` path no `volumes:` PVC covers and the node lacks → the pod fails (hostPath). Use a PVC.
10. Expecting K8s task outputs to persist — the output dir is an ephemeral `emptyDir`; use a PVC or `uploads:`.
11. Multi-node Slurm with models/workspace/output on **node-local** storage — same-path mounts break; use shared storage.
12. S3 secrets in YAML (use the boto3 chain) or forgetting the `sflow[s3]` extra.
13. Hand-writing mounts for a path already declared as an artifact — redundant; artifacts auto-mount same-path.

## Additional resources

- Field reference & merge rules → [schema-reference.md](schema-reference.md)
- 12 annotated, runnable recipes → [examples.md](examples.md)
- Full user docs → [nvidia.github.io/nv-sflow/docs/user/intro](https://nvidia.github.io/nv-sflow/docs/user/intro)
