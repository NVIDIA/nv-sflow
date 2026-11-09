# Backend-agnostic modular samples

These samples compose a **workload** with a **backend** (and the shared
`benchmark.yaml`) to produce a runnable inference + benchmark workflow. The same
workload YAML runs on Slurm, plain Kubernetes, Kubernetes MPI, or a single machine
with Docker -- without edits.

Two sflow features make this work:

- **Recursive scattered merge** — files are deep-merged by the entry `name`, so a
  task/operator/backend can be defined in one file and extended in another. Every
  backend fragment names its backend `cluster` and defines the logical operators
  `server` / `helper` / `client`, so workloads reference
  `${{ backends.cluster.nodes[0].ip_address }}` and `operator: server` uniformly.
- **`required_by`** — each workload's `server` declares `required_by: [benchmark]`
  (the reverse of `depends_on`). It folds into the benchmark's `depends_on` at
  load time, and is skipped when `benchmark.yaml` is omitted — so no
  `--missable-tasks` is ever needed.

## Layout

```
modular/
├── benchmark.yaml            # shared AIPerf hub (dependency-free; servers point to it)
├── backends/
│   ├── slurm.yaml            # backend `cluster` + srun operators (single-process server)
│   ├── slurm_mpi.yaml        # backend `cluster` + srun server as one MPI rank per GPU
│   ├── k8s.yaml              # backend `cluster` + k8s operators (single-pod)
│   ├── k8s_mpi.yaml          # backend `cluster` + k8s_mpi server, k8s helper/client
│   └── docker.yaml           # backend `cluster` + docker_run operators (one local box)
└── workloads/
    ├── dynamo_common.yaml           # NATS + etcd + frontend infra (Dynamo only)
    ├── dynamo_trtllm.yaml           # Dynamo TRT-LLM aggregated server
    ├── dynamo_trtllm_disagg.yaml    # Dynamo TRT-LLM prefill + decode (disaggregated)
    ├── dynamo_vllm.yaml             # Dynamo vLLM aggregated server
    ├── dynamo_vllm_disagg.yaml      # Dynamo vLLM prefill + decode (disaggregated)
    ├── dynamo_sglang.yaml           # Dynamo SGLang aggregated server
    ├── dynamo_sglang_disagg.yaml    # Dynamo SGLang prefill + decode (disaggregated)
    ├── trtllm_serve.yaml            # raw trtllm-serve      (no Dynamo infra)
    ├── vllm_serve.yaml              # raw vllm serve        (no Dynamo infra)
    └── sglang_serve.yaml            # raw sglang.launch_server (no Dynamo infra)
```

The `dynamo_*_disagg.yaml` workloads split serving into a `prefill_server` (context)
and a `decode_server` (generation) that exchange the KV cache — vLLM/SGLang over
NIXL / the configured transfer backend, TRT-LLM over its UCX cache transceiver.
Both workers share the same `dynamo_common.yaml` infra and each declare
`required_by: [benchmark]`.

## Supported combinations

| Workload | Slurm | Slurm MPI | K8s (plain) | K8s MPI |
|----------|:-----:|:---------:|:-----------:|:-------:|
| `dynamo_trtllm` (+ `dynamo_common`) | — | yes | yes (single-node) | yes (cross-node) |
| `dynamo_vllm` (+ `dynamo_common`)   | yes | — | yes (single-node) | — (vLLM agg is single-worker) |
| `dynamo_sglang` (+ `dynamo_common`) | yes | — | yes (single-node) | — (SGLang agg is single-worker here) |
| `dynamo_trtllm_disagg` (+ `dynamo_common`) | — | yes | yes (both workers on one node) | yes (per-worker cross-node) |
| `dynamo_vllm_disagg` (+ `dynamo_common`)   | yes | — | yes (both workers on one node) | — (single-node workers) |
| `dynamo_sglang_disagg` (+ `dynamo_common`) | yes | — | yes (both workers on one node) | — (single-node workers) |
| `trtllm_serve`                      | — | yes | yes (single-node) | yes (cross-node) |
| `vllm_serve`                        | yes | — | yes (single-node) | — (needs a cross-node Ray cluster) |
| `sglang_serve`                      | yes | — | yes (single-node) | — (needs SGLang multi-node args) |

- **Slurm vs Slurm MPI is the same choice as K8s vs K8s MPI: how the server is
  launched.** `backends/slurm.yaml` runs the server as ONE process (`ntasks: 1`)
  that shards the model itself — what vLLM and SGLang expect.
  `backends/slurm_mpi.yaml` runs one MPI rank per GPU under pmix — what TRT-LLM
  expects. Composing a vLLM/SGLang workload with the MPI fragment starts
  `SERVER_GPUS` independent copies of the server that then race for the same
  port; composing TRT-LLM with the plain fragment gives it a single rank.

- Single-node means `SERVER_GPUS <= GPUS_PER_NODE`.
- The `dynamo_*_disagg` workloads run two workers, so they need `2 * SERVER_GPUS`
  GPUs in total (e.g. the default `SERVER_GPUS=4` needs an 8-GPU node). On plain
  Kubernetes both workers pack onto one node (they merge into a single pod, so the
  KV cache moves over intra-node NVLink); on Slurm they share the allocation.
- `dynamo_*` workloads must also include `workloads/dynamo_common.yaml`; the
  `*_serve` workloads (`trtllm_serve` / `vllm_serve` / `sglang_serve`) must NOT
  (they serve the OpenAI API directly on port 8000).
- Only `dynamo_trtllm` and `trtllm_serve` support cross-node (`backends/k8s_mpi.yaml`)
  out of the box — TRT-LLM drives the cross-node MPI world itself. The vLLM/SGLang
  workloads here are single-node (intra-node tensor parallelism); cross-node vLLM
  (Ray) / SGLang (`--dist-init-addr`/`--nnodes`/`--node-rank`) is left to the user.

## Usage

Always compose backend + workload(s) + benchmark, and dry-run first:

```bash
# Dynamo TRT-LLM on plain Kubernetes (single node)
sflow run -f backends/k8s.yaml \
          -f workloads/dynamo_common.yaml -f workloads/dynamo_trtllm.yaml \
          -f benchmark.yaml --dry-run

# Dynamo TRT-LLM on Kubernetes MPI (2 nodes, 16-GPU tensor parallel)
sflow run -f backends/k8s_mpi.yaml \
          -f workloads/dynamo_common.yaml -f workloads/dynamo_trtllm.yaml \
          -f benchmark.yaml -s NUM_NODES=2 -s SERVER_GPUS=16 --dry-run

# Raw trtllm-serve on Slurm (MPI fragment: TRT-LLM wants one rank per GPU)
sflow run -f backends/slurm_mpi.yaml \
          -f workloads/trtllm_serve.yaml -f benchmark.yaml \
          -s SLURM_ACCOUNT=acct -s SLURM_PARTITION=batch --dry-run

# Dynamo SGLang on plain Kubernetes (single node)
sflow run -f backends/k8s.yaml \
          -f workloads/dynamo_common.yaml -f workloads/dynamo_sglang.yaml \
          -f benchmark.yaml --dry-run

# Dynamo vLLM DISAGGREGATED (prefill + decode) on plain Kubernetes (1 node, 8 GPUs)
sflow run -f backends/k8s.yaml \
          -f workloads/dynamo_common.yaml -f workloads/dynamo_vllm_disagg.yaml \
          -f benchmark.yaml -s SERVER_GPUS=4 --dry-run

# Dynamo TRT-LLM DISAGGREGATED on Slurm (MPI fragment)
sflow run -f backends/slurm_mpi.yaml \
          -f workloads/dynamo_common.yaml -f workloads/dynamo_trtllm_disagg.yaml \
          -f benchmark.yaml -s SLURM_ACCOUNT=acct -s SLURM_PARTITION=batch --dry-run

# Raw vLLM OpenAI server on plain Kubernetes (single node)
sflow run -f backends/k8s.yaml \
          -f workloads/vllm_serve.yaml -f benchmark.yaml --dry-run

# Raw SGLang OpenAI server on Slurm
sflow run -f backends/slurm.yaml \
          -f workloads/sglang_serve.yaml -f benchmark.yaml \
          -s SLURM_ACCOUNT=acct -s SLURM_PARTITION=batch --dry-run
```

## Per-cluster values to override

These samples ship with placeholders — set them with `-s KEY=VALUE` (or a
`variables:` override file):

- Slurm: `SLURM_ACCOUNT`, `SLURM_PARTITION`, `GPUS_PER_NODE`, `NUM_NODES`
- Kubernetes: `K8S_NAMESPACE`, `GPUS_PER_NODE`, `NUM_NODES`
- Workload: `SERVER_IMAGE`, `SERVED_MODEL_NAME`, `SERVER_GPUS`
- Benchmark: `CLIENT_IMAGE`, `ISL`, `OSL`, `CONCURRENCY`

### The model is an artifact, not a variable

Each backend fragment declares the weights as an `fs://` artifact, and every
workload refers to them as `${{ artifacts.LOCAL_MODEL_PATH.path }}`. Override the
location with `-a`, not `-s`:

```bash
-a LOCAL_MODEL_PATH=fs:///lustre/models/Qwen3-8B-FP8
```

The point is that sflow MOUNTS an `fs://` artifact into the task container at the
same path it has on the host — `--container-mounts` on Slurm, a `hostPath` volume
on Kubernetes — so the weights are actually present where the server looks. A
plain variable only named a path and left mounting to you.

Declaring it on the BACKEND is deliberate: making a host path visible inside a
container is a backend concern, so the workloads stay free of it.

Docker needs no special case since v0.3.2: `docker_run` bind-mounts local
`fs://` / `file://` artifacts at the same path inside the container by itself.
`backends/docker.yaml` still declares an explicit `MODEL_HOST_PATH` mount so the
fragment also works on older sflow and for paths that are not artifacts; keep
`LOCAL_MODEL_PATH` somewhere under `MODEL_HOST_PATH` there.

## Running on one machine (Docker)

`backends/docker.yaml` is the single-box counterpart to the cluster fragments, so a
recipe can be proven locally before it goes near a queue:

```bash
sflow run -f backends/docker.yaml -f workloads/sglang_serve.yaml -f benchmark.yaml \
          -s GPUS_PER_NODE=1 -s SERVER_GPUS=1 \
          -s MODEL_HOST_PATH=/data/models \
          -a LOCAL_MODEL_PATH=fs:///data/models/Qwen3-0.6B
```

Needs Docker and, for GPU workloads, the NVIDIA Container Toolkit. Every container
runs with `--network=host`, so `${{ backends.cluster.nodes[0].ip_address }}`
resolves to `127.0.0.1` exactly as the workloads expect, and `MODEL_HOST_PATH` is
bind-mounted at the same path inside the container so the artifact path needs no
rewrite — which is why `LOCAL_MODEL_PATH` must live under `MODEL_HOST_PATH` here.

The server's GPUs come from the workload's `resources.gpus.count`, not from a
hard-coded `gpus: all`: sflow reserves that many free devices from a machine-local
registry and pins the container to them. Several `sflow run` processes can
therefore share one box -- a second recipe takes the next free GPUs, or waits with
`--wait-for-gpus`. CPU-only helpers and the client request no GPUs and see none.
See [GPU reservation](../../../docs/user/backends.md#gpu-reservation-local-concurrent-runs).
