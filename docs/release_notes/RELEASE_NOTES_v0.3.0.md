# sflow v0.3.0 Release Notes

**Release date:** July 2026  
**Previous release:** [v0.2.1](https://github.com/NVIDIA/nv-sflow/releases/tag/v0.2.1) (April 2026)

---

## Highlights

sflow v0.3.0 adds a **Kubernetes backend**, extending the same portable recipe model to Kubernetes GPU platforms alongside Local, Docker, and Slurm. The same recipe semantics — DAG, resources, probes, artifacts, and result parsing — now map to Kubernetes pods and MPI-style jobs, with sflow rendering the GPU, InfiniBand/RDMA, and NVLink/MNNVL configuration from your recipe. This release also introduces **merge-pod** for same-node GPU fabrics, **recursive deep-merge** for modular composition, **declarative result parsing and S3 uploads**, **hardware monitoring**, and richer self-explaining run summaries.

| Area | v0.2.x | v0.3.0 |
| --- | --- | --- |
| **Backends** | Local · Docker · Slurm | **+ Kubernetes** (`k8s` / `k8s_mpi`) |
| **Composition** | multi-file: later file wins | recursive deep-merge, keyed by name |
| **Outputs** | manual scraping | declarative `result:` parsing **+ S3 uploads** |
| **Monitoring** | — | `monitor:` telemetry + run summaries |
| **Same-node GPUs** | — | **merge-pod**: same-node CUDA IPC / NVLink (Kubernetes) |

---

## New Features

### 1. Kubernetes backend

The same recipe semantics now map to Kubernetes as one execution target alongside Local, Docker, and Slurm:

- **`k8s` and `k8s_mpi` operators** for Kubernetes pods and MPI-style jobs
- **Reserved-node discovery and task-pod pinning**
- **Device-plugin or DRA GPU allocation** (device plugin is the default)
- **In-cluster TCP/HTTP readiness probes**
- **Offloaded task logs, output collection, and pod cleanup**

The recipe stays the stable abstraction — change the backend/operator fragment and the task DAG, scripts, probes, artifacts, and result contract carry over:

```diff
 backends:
-  - type: slurm
+  - type: kubernetes

 operators:
-  - type: srun
+  - type: k8s
```

### 2. Kubernetes fabric configuration rendered from your recipe

On Kubernetes the hard part is getting the pod to actually *use* the fast fabrics. sflow probes the live nodes at run time and renders the Kubernetes-native requests and environment that carry the hardware up through the communication libraries, so NCCL/UCX select NVLink/MNNVL/IB instead of quietly falling back to TCP.

| The Kubernetes question | What sflow does for you |
| --- | --- |
| Expose GPUs to a pod | Renders `nvidia.com/gpu` limits, or DRA resource claims when you set `scheduling: dra` |
| Enable InfiniBand / RDMA | Auto-detects the cluster's RDMA path (GKE RDMA resources, an `rdma/*` device plugin, or host `/dev/infiniband`) and wires resources, `IPC_LOCK`, and NCCL/socket env |
| Same-node NVLink | Merges co-located GPU tasks into one pod and right-sizes `/dev/shm` so CUDA IPC / NVLink P2P is available |
| Multi-node MNNVL | Claims an NVIDIA ComputeDomain (IMEX) channel and sets `NCCL_MNNVL_ENABLE` / `UCX_CUDA_IPC_ENABLE_MNNVL` |
| NCCL/UCX transport flags | Injects the transport env so the libraries select the fast path instead of silently falling back to TCP |

**Where the line sits:** sflow generates and wires the Kubernetes config; it does not install your cluster. The capabilities still come from the platform — NVIDIA GPU Operator or the DRA driver, an RDMA mechanism, the ComputeDomain/IMEX driver for rack-scale NVLink — and sflow detects what is present on each run and requests it correctly. Enabling a transport is necessary but not sufficient: the workload must actually use it, so benchmark representative runs. Cluster-specific CNI networking (for example Multus network attachments) stays out of scope.

### 3. Merge-pod: same-node GPU fabrics

Separate pods on one physical node can hide GPUs from one another and force cooperating workers onto IB or TCP even when NVLink/NVSwitch is available. Merge-pod places compatible same-node GPU tasks into one pod with combined GPU visibility, so CUDA IPC / NVLink / NVSwitch is available. The tasks remain logically separate in YAML. Prefill/decode KV transfer is a representative use case, not the only one.

Merge groups are formed inside one sflow workflow run, and only when tasks:

- come from the same recipe/task graph
- use the same Kubernetes backend and physical node
- use the same container image
- are single-node GPU tasks that can run concurrently
- have no completion dependency on another member of the merge group

CPU-only tasks and multi-node tasks are not merged. The gain matters most when co-located tasks exchange significant GPU-resident data; compute-bound tasks, small transfers, or frameworks without CUDA IPC/P2P support may see little benefit, so benchmark representative workloads. Merge-pod also fits current IMEX guidance of one ComputeDomain/IMEX daemon per node and one workload pod per node within a domain.

References:

- NVIDIA Dynamo, [Disaggregated Inference Communication Guide](https://docs.nvidia.com/dynamo/kubernetes-deployment/operate/disagg-communication)
- NVIDIA CUDA, [Programming Systems with Multiple GPUs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
- NVIDIA, [Enabling Multi-Node NVLink on Kubernetes](https://developer.nvidia.com/blog/enabling-multi-node-nvlink-on-kubernetes-for-gb200-and-beyond/)

### 4. Declarative result parsing and S3 uploads

Turn raw task output into structured data and ship it — no bespoke scraping or upload scripts.

- **`result:` parsing** — declare a simple regex map (`metric: 'regex'`), richer `patterns:` (with `type`, `unit`, `aggregate`, `required`), or read a JSON `file:` the task wrote. sflow writes a per-task `result.json` and updates a workflow-level `results.json` index, exposing both paths as `SFLOW_TASK_RESULT_FILE` / `SFLOW_WORKFLOW_RESULT_FILE`. A task is not marked `COMPLETED` until parsing finishes; parsing is best-effort this release (a missing `required` match sets `ok: false` and logs a warning rather than failing the run).
- **`storage:` + `uploads:` / `upload_all`** — define named `storage:` targets (S3 today; GCS/Azure planned), then `uploads:` the exact files or globs you want per task, or `upload_all:` to zip and ship the whole run directory at the end. Credentials come from the boto3 default chain (env, `~/.aws`, or an instance/pod IAM role) — never in YAML. Install with the `sflow[s3]` extra.

### 5. Hardware monitoring

Add `monitor:` to collect GPU, CPU, memory, disk, and network telemetry from bare nodes. sflow generates CSV/SVG reports plus a readable overview after execution. (Supported on Local, Docker, and Slurm; see [Tested Environments & Current Support](#tested-environments--current-support) for Kubernetes.)

```
Metric Summary
GPU util %        min=42   avg=87   max=99
GPU mem used GiB  min=18   avg=61   max=76
GPU power W       min=310  avg=642  max=718
```

### 6. Self-explaining run summaries

Every run can produce a terminal-friendly summary with task timing, lifecycle events, GPU/node occupancy, command logs, dependencies, and failure hints.

```
Task Duration Chart
prefill    |####..........................|   58.412s  READY
decode     |##########################....|  6m14.882s  READY
benchmark  |..........................####|   47.906s  COMPLETED
```

### 7. Modular composition and CLI control

- **Recursive deep-merge, keyed on name** — scatter one backend or task across files (`type` in one, `host_network` in another) and same-named entries assemble into a single definition. (See [Breaking Changes](#breaking-changes) for the behavior change from v0.2.x.)
- **`required_by`** — a reusable fragment attaches itself to a shared hub task (e.g. `benchmark`) with no edit to the hub and no cross-file `depends_on`; targets missing from the composed run are skipped.
- **Backend-aware CLI overrides** — `--extra-args` reaches whichever backend runs (Slurm `salloc`, `docker run`, or `kubectl`) and de-dups by option with CLI winning (`--gres=gpu:4` overrides a recipe's `gpu:8` instead of passing both). Use `--extra-salloc-args` / `--extra-docker-args` / `--extra-kubectl-args` to target one backend kind.

### 8. New and expanded recipes

v0.3.0 adds and expands examples for:

- backend-agnostic modular workflows
- Kubernetes, Docker, Slurm, and Local execution
- Dynamo aggregated and disaggregated serving
- SGLang, vLLM, and TensorRT-LLM
- monitoring, results, uploads, replicas, and bulk runs

---

## Improvements

- **Richer run outputs** — command logs and runtime provenance, richer execution summaries and TUI visibility, and hardened per-task logging and log-watch behavior.

---

## Tested Environments & Current Support

**Validated setups**

- vanilla bare-metal Kubernetes
- Google Kubernetes Engine (GKE)

**Current limitations / work in progress**

- **Kubernetes monitoring is not fully supported yet.** The built-in bare-node `monitor:` feature is supported on Local, Docker, and Slurm. On Kubernetes, monitor blocks are currently skipped because there is no DCGM/DaemonSet collector; sampling the sflow driver host would produce misleading data.
- **DRA GPU allocation is supported but still WIP.** The path is implemented, but it has not yet been broadly validated across different Kubernetes distributions, versions, and NVIDIA DRA deployments. The widely available device-plugin path remains the default.
- **Kubernetes execution is driver-attached.** Use interactive `sflow run`; detached Kubernetes batch execution is not supported yet.
- Kubernetes environments beyond the tested bare-metal and GKE setups may require cluster-specific validation and tuning.

---

## Documentation

- User guide, including the Kubernetes backend, merge-pod, results/uploads, and monitoring: [nvidia.github.io/nv-sflow](https://nvidia.github.io/nv-sflow/)

---

## Breaking Changes

- **Multi-file recipes now deep-merge instead of concatenating/replacing.** Same-name tasks and named sections (variables/backends/operators/artifacts/storage) across included files merge field-by-field (last non-null wins) instead of raising a duplicate-name error or replacing wholesale; a cross-file `workflow.name` conflict now warns and takes the last value instead of raising. All shipped recipes are unaffected; only inputs that previously errored, or that relied on a later file *fully replacing* an earlier section, change.
  - *Migration:* if you relied on wholesale replacement, note that fields now merge (last non-null wins). Audit accidental duplicate names in multi-file workflows.

**Opt-in (non-breaking):** set `fail_fast: true` on a shell task to prepend `set -e`, so a failed command fails the task instead of a later successful command (e.g. a trailing `echo`) masking it. The default is `false` — existing shell semantics are unchanged.

---

## Upgrade Guide

```bash
# Install v0.3.0
uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@main"

# Validate a Kubernetes recipe without consuming cluster resources
sflow run -f examples/self_contained/kubernetes/hello_world.yaml --dry-run

# Explore portable, backend-agnostic recipe composition
ls examples/modular/backend_agnostic/
```

Documentation: [https://nvidia.github.io/nv-sflow/](https://nvidia.github.io/nv-sflow/)

Repository: [https://github.com/NVIDIA/nv-sflow](https://github.com/NVIDIA/nv-sflow)
