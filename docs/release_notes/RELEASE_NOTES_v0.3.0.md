# sflow v0.3.0 Release Notes

**Release date:** June 2026
**Previous release:** v0.2.2 (May 2026)

---

## Highlights

sflow v0.3.0 brings a major **Kubernetes backend overhaul** and refines how CLI-provided
backend extra args are routed.

- **Kubernetes overhaul.** The four previous k8s operators (`k8s_apply`, `k8s_lws`,
  `k8s_run`, `k8s_reserve`) are consolidated into a single `k8s` operator that renders each
  task into pinned pod(s) — one pod for a single-node task, N pods (leader = index 0) for a
  multi-node task. New capabilities: DRA (`resource.k8s.io` ResourceClaimTemplate) or
  `device_plugin` GPU requests, node reserve+discover+pin with a create-before-destroy GPU
  handoff, RDMA/GKE-gIB fast path for multi-node NCCL/UCX/NIXL, ComputeDomain (Multi-Node
  NVLink), PVC/ConfigMap/hostPath artifact injection, and driver-managed pod execution
  (offloaded log streaming + authoritative pod-status completion). New Dynamo recipes
  (vLLM / SGLang / TRT-LLM, agg + disagg) ship under `examples/`.
- **CLI kube access + extra-arg routing.** New `sflow run` kube flags keep recipes
  cluster-agnostic, and `--extra-args` is now backend-agnostic with backend-specific
  variants, all de-duplicating by option (CLI wins) so a CLI value cleanly overrides a
  recipe default.

---

## User-Facing Changes

### Kubernetes

- **Single `k8s` operator.** Use `type: k8s` (with a workload `image:`) for all Kubernetes
  tasks. It renders one Pod for a single-node task and one Pod per assigned node for a
  multi-node task (`resources.nodes`), requesting GPUs via DRA (default) or the legacy
  `device_plugin` mode (backend `scheduling:` field).
- **New `sflow run` kube flags** (keep volatile, cluster-specific access out of the recipe):
  - `--kubeconfig` — kubeconfig path (also exported as `KUBECONFIG`).
  - `--kube-context` — kubeconfig context.
  - `--kube-namespace` — override the namespace for all kubernetes backends.
  - `--extra-kubectl-args` — extra global kubectl flags applied to every call (repeatable).
- **Backend-agnostic node include/exclude.** New `--include-nodes` / `--exclude-nodes`
  flags on `sflow run` and `sflow batch` (and matching `include_nodes` / `exclude_nodes`
  backend config fields) restrict or steer the candidate node pool across all backends:
  Slurm `--nodelist`/`--exclude`, Kubernetes `hostname` In/NotIn nodeAffinity, and Docker
  host-pool filtering. This replaces the earlier Kubernetes-only `--kube-exclude-node`.
- **New backend fields:** `scheduling` (`dra` | `device_plugin`), `dra` (device class,
  device selectors, `compute_domain`, RDMA co-allocation), `volumes` (pre-existing PVC
  mounts), `rdma`, `tolerations`, `reservation.timeout`. GPU `resources.gpus.count` is a
  per-task total split evenly across the assigned nodes.
- **RDMA/InfiniBand auto-detection now works beyond GKE.** A single `rdma` field
  (`auto` | `off` | `gke` | `shared_device_plugin` | `host_device`) steers it: `auto`
  runs a provider chain -- GKE multi-NIC, `k8s-rdma-shared-dev-plugin` / NVIDIA Network
  Operator (`rdma/*`), and a generic host-device (`/dev/infiniband` + `CAP_IPC_LOCK`)
  fallback -- so on-prem Mellanox clusters get scoped RDMA (UCX/NIXL + NCCL over IB)
  instead of silently falling back to TCP. Each task pod then verifies RDMA is usable
  and, by default, exposes all NICs so NCCL/UCX auto-select each GPU's closest device
  (tunable per pod via the `SFLOW_RDMA_AFFINITY` env: `auto` | `explicit` | `off`); if
  RDMA is unusable it falls back to TCP so the workload still comes up. For DRA
  clusters, `dra.rdma_device_class` co-allocates a PCIe-root-aligned NIC with each GPU.
  This is a behavior change: clusters that previously ran over TCP may now use RDMA on
  `auto`; set `rdma: off` to opt out.

### CLI extra args

- **`--extra-args, -e` is now backend-agnostic.** Its values are forwarded to whichever
  backend the recipe uses: merged into each Slurm backend's `salloc`, each docker
  backend's `docker run`, and every `kubectl` call's global flags. Whichever backend the
  recipe contains picks the args up.
- **New backend-specific flags** for when you want to target one backend kind only:
  - `--extra-salloc-args` — Slurm `salloc` only (e.g. `--gpus-per-node=4`).
  - `--extra-docker-args` — docker `docker run` only (e.g. `--shm-size=16g`).
  - `--extra-kubectl-args` — kubectl global flags only (e.g. `--request-timeout=30s`).
- **De-dup by option (CLI wins).** CLI extra args now override a recipe backend's
  `extra_args` on a conflicting option instead of both being passed (e.g. CLI
  `--gres=gpu:4` overrides a recipe `--gres=gpu:8`). Repeatable `key=value` flags such as
  `--env=FOO=1` / `--env=BAR=2` are preserved as distinct entries. A more specific
  `--extra-<type>-args` wins over the generic `--extra-args` on a conflicting option.

### Breaking changes and migration

- **Kubernetes operators consolidated into one `k8s` operator.** The `k8s_apply`, `k8s_lws`,
  `k8s_run`, and `k8s_reserve` operator types are removed. Why: they duplicated
  reserve/render/launch logic and each covered only part of a run; the single `k8s` operator
  handles single- and multi-node pods, GPU scheduling, and driver-managed execution in one
  place. Migrate every recipe operator to `type: k8s`:
  - `type: k8s_apply` / `k8s_run` / `k8s_lws` → `type: k8s` (multi-node is now driven by the
    task's `resources.nodes` instead of a dedicated LeaderWorkerSet operator).
  - `k8s_reserve` is no longer a separate operator — the kubernetes backend always reserves,
    discovers, and pins nodes itself.
  Tests for the removed operators (`test_plugin_operators_k8s_apply/_lws/_run/_reserve.py`)
  were deleted and replaced by `test_plugin_operators_k8s*.py`; the registry/assembly tests
  were updated to expect the single `k8s` type.
- **The kubernetes backend no longer has an `image` field.** Workload images belong on the
  operator; reservation/placeholder pods use a fixed internal sleeper image. Move any backend
  `image:` onto the `k8s` operator's `image:`. A stray `image:` on the backend is now dropped
  (not validated), so image-preflight no longer fails on it (`test_preflight_validate_container_images.py`
  updated accordingly).
- **`--kubectl-arg` was renamed to `--extra-kubectl-args`.** Update any scripts:
  - `sflow run --kubectl-arg=--request-timeout=30s` → `sflow run --extra-kubectl-args=--request-timeout=30s`
  - or use the generic form: `sflow run --extra-args=--request-timeout=30s`.
- `--extra-args` itself is unchanged in spelling and remains backward compatible; it is now
  generic (applies to Slurm, docker, and kubectl) rather than Slurm-only. For a kubernetes
  workflow, prefer `--extra-kubectl-args` for kubectl-only flags: a generic `--extra-args`
  value that is not a valid kubectl global flag (e.g. a Slurm-ism like `--gpus-per-node`) is
  applied to every kubectl call and now emits a warning.

### Behavior changes

- **GPU slice planning is now uniform across backends.** The planner computes each task's
  `CUDA_VISIBLE_DEVICES` slice even for backends that do not inject it into the env (e.g.
  Kubernetes, where the cluster/DRA assigns physical devices), so multi-task/multi-replica
  node packing and oversubscription checks stay consistent. The slice is still only turned
  into an env var for backends that support it. Dry-run allocation maps for such backends now
  show the computed slice instead of blank.

---

## Documentation Updated

- `docs/user/cli.md`
- `docs/user/backends.md`
- `docs/user/architecture.md`
