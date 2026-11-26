# Dynamo vLLM workload with swappable infrastructure

`dynamo_vllm_workload.yaml` contains the shared workflow: NATS, etcd, frontend,
prefill/decode workers and the AIPerf benchmark. Compose it with exactly one
infrastructure file: `docker.yaml`, `slurm.yaml` or `k8s.yaml`.

All platforms inherit the same container image,
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1-cuda13`, from the shared workload.
Infrastructure files contain generic defaults; supply your hosts, account,
namespace and model path on the CLI. No credentials belong in the YAML.

```bash
sflow sample modular/dynamo_vllm --output demo_dynamo_vllm
cd demo_dynamo_vllm

# Docker: replace the documentation addresses with your two hosts.
sflow run -f dynamo_vllm_workload.yaml -f docker.yaml \
  --set HOST_0_ADDRESS=192.0.2.10 --set HOST_1_ADDRESS=192.0.2.11 \
  --set SSH_USER=your_user \
  --artifact LOCAL_MODEL_PATH=fs:///models/Qwen3-0.6B --dry-run

# Slurm: weights must be on storage shared by both compute nodes.
sflow run -f dynamo_vllm_workload.yaml -f slurm.yaml \
  --set SLURM_PARTITION=your_partition --set SLURM_ACCOUNT=your_account \
  --artifact LOCAL_MODEL_PATH=fs:///models/Qwen3-0.6B --dry-run

# Kubernetes: the PVC contains Qwen3-0.6B/ at its root and supports both nodes.
sflow run -f dynamo_vllm_workload.yaml -f k8s.yaml \
  --kubeconfig "$HOME/.kube/config" --set K8S_NAMESPACE=default \
  --set MODEL_PVC_CLAIM=your-model-pvc \
  --artifact LOCAL_MODEL_PATH=fs:///models/Qwen3-0.6B --dry-run
```

Run a dry-run first, then remove `--dry-run` to launch. For a remote Slurm login
host, add `--ssh your_user@login.example.org --ssh-fetch all`; SSH keys and
ProxyJump settings stay in your private SSH configuration. For Kubernetes,
select a node pool with `--kube-node-selector your-label=your-value` when needed.

| Infrastructure | Node 1 (index 0) | Node 2 (index 1) | GPUs per node |
|---|---|---|---|
| Docker | 2 × TP1 prefill | 1 × TP4 decode | 8 |
| Slurm | 2 × TP1 prefill | 1 × TP4 decode | 4 |
| Kubernetes | 2 × TP1 prefill | 1 × TP2 decode | 2 |

Each worker stays on its assigned node, with separate GPUs and a unique NIXL
side-channel port. KV transfers cross nodes; tensor parallelism stays within
each worker's node, so no Ray cluster is needed. The benchmark waits for every
worker to pass both its HTTP probe and its own log gate. Prefill watches
`Registered endpoint 'dynamo.prefill.generate'`; decode watches
`Registered endpoint 'dynamo.backend.generate'`. AIPerf output is checked for
generation metrics before the benchmark succeeds.

## Workload overrides

Shared defaults are two prefill workers at TP1, one decode worker at TP4,
ISL 1024, OSL 256, concurrency 4, 16 measured requests and 4 warmups. Kubernetes
overrides decode TP to 2 to fit its two-GPU nodes.

`NUM_PREFILL_WORKERS`, `NUM_DECODE_WORKERS`, `CTX_TP_SIZE`, `GEN_TP_SIZE`,
`EXTRA_PREFILL_ARGS` and `EXTRA_DECODE_ARGS` are CLI-overridable controls. Prefill
workers stay on node 0 and decode workers on node 1; the total GPU demand for
each role must fit that node. Override `GPUS_PER_NODE` to match your hardware.

```bash
sflow run -f dynamo_vllm_workload.yaml -f docker.yaml \
  --set GEN_TP_SIZE=2 --set CONCURRENCY=8 --set MULTI_ROUND=8 \
  --artifact LOCAL_MODEL_PATH=fs:///models/Qwen3-0.6B --dry-run
```

## Infrastructure prerequisites

- Docker needs SSH/Docker access to both hosts, NVIDIA Container Toolkit, and
  peer connectivity for NATS, etcd, frontend and NIXL traffic. Set
  `HOST_0_ADDRESS`, `HOST_1_ADDRESS` and `SSH_USER` for your pool. Model data must
  already exist at the artifact path on both hosts.
- Slurm needs container-enabled `srun` and a valid partition/account. Set
  `SLURM_PARTITION`, `SLURM_ACCOUNT` and `SLURM_TIME` for your cluster. The backend
  requests GPU GRES using `--gpus-per-node` and the configured `GPUS_PER_NODE`.
- Kubernetes needs device-plugin GPUs, valid kubeconfig/RBAC and permission to
  use host networking and the model PVC. Set `K8S_NAMESPACE` and
  `MODEL_PVC_CLAIM` to an existing populated claim in that namespace. The claim
  must support concurrent reads from both nodes (`ReadOnlyMany` or
  `ReadWriteMany`); `ReadWriteOnce` does not support this two-node layout.
  The PVC mounts read-only at `/models`, and `LOCAL_MODEL_PATH` must point
  inside it. No per-node model staging is needed. This infrastructure uses
  plain TCP without RDMA. Keep runtime/JIT caches outside the read-only model
  mount. When using Hugging Face snapshot paths, mount the whole cache so their
  relative blob symlinks remain accessible.

The model path defaults to `fs:///models/Qwen3-0.6B` on every platform. Override
it with `--artifact LOCAL_MODEL_PATH=fs:///your/model/path`; the backend mounts
it automatically. When selecting another checkpoint, also set
`SERVED_MODEL_NAME`. `SERVER_IMAGE` and `CLIENT_IMAGE` are shared workload
settings; use the same version across platforms when comparing results.
Warm caches before presenting: first image pulls and JIT compilation can take
minutes. Regenerate the bundle after updating the sample.
