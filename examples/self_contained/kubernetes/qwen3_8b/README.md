# Qwen3-8B on Kubernetes — DGX vs GB (NVL72), TensorRT-LLM

Self-contained sflow recipes that serve **Qwen3-8B (dense, FP8)** with TensorRT-LLM on
two machine shapes, so you can see exactly what changes between them. The *workload* is
identical; only the **sflow Kubernetes backend fields, a few envs, and the launch layout**
differ — which is the whole point of these examples.

| Machine | GPUs/node | TP=8 lands on | Cross-node transport |
|---------|:---------:|---------------|----------------------|
| **DGX** (e.g. DGX B200) | 8 | ONE node (intra-node NVLink/NVSwitch) | RDMA/InfiniBand (only for disagg KV) |
| **GB** (GB200/GB300 NVL72) | 4 | TWO nodes (cross-node **MNNVL**) | Multi-Node NVLink via an IMEX **ComputeDomain** |

## Files

| Workload | DGX (8 GPU/node) | GB (NVL72, 4 GPU/node) |
|----------|------------------|------------------------|
| Dynamo TRT-LLM, aggregated | `dgx_dynamo_trtllm_agg.yaml` | `gb_dynamo_trtllm_agg.yaml` |
| Dynamo TRT-LLM, disaggregated (prefill+decode) | `dgx_dynamo_trtllm_disagg.yaml` | `gb_dynamo_trtllm_disagg.yaml` |
| Pure `trtllm-serve` (no Dynamo infra) | `dgx_trtllm_serve.yaml` | `gb_trtllm_serve.yaml` |

## What differs, DGX → GB

Everything else (the engine config, the `dynamo.trtllm` / `trtllm-serve` command, the
benchmark) is the same. The diff is concentrated in the backend block, a couple of envs,
and the resulting node layout:

| Dimension | DGX (8 GPU/node) | GB NVL72 (4 GPU/node) |
|-----------|------------------|-----------------------|
| `backends.k8s_cluster.gpus_per_node` | `8` | `4` |
| `compute_domain.channel` | *(absent — not needed)* | set (IMEX MNNVL channel; `auto` or a name) |
| `rdma` | `auto` (cross-node disagg KV over IB) | `disable` (use the MNNVL fabric, not IB) |
| Server env | — | `export UCX_CUDA_IPC_ENABLE_MNNVL=y` (disagg servers) |
| TP=8 worker spans | 1 node | 2 nodes (one cross-node MPI world) |
| Nodes reserved (agg / serve) | 1 | 2 |
| Nodes reserved (disagg) | 2 (prefill 1 + decode 1) | 4 (prefill 2 + decode 2) |

Node counts are derived from `TP_SIZE` and `GPUS_PER_NODE`, so overriding either
re-plans the layout automatically.

> **Why TP=8 on an 8B model?** Qwen3-8B fits on a single GPU, so TP=8 is over-provisioned
> — it is chosen so the DGX↔GB difference is exactly *where the TP world lives* (one node
> vs. two nodes over MNNVL). For real 8B serving drop `TP_SIZE` (e.g. `-s TP_SIZE=1`) and
> scale out with more replicas/clients instead.

## Usage

Always dry-run first. Override the placeholders for your cluster
(`K8S_NAMESPACE`, the model path, and — on GB — the `COMPUTE_DOMAIN_CHANNEL`), and add a
node selector with `--kube-node-selector KEY=VALUE` only if your cluster needs one. Cluster
access is passed as CLI flags, not YAML:

```bash
# DGX B200 — Dynamo TRT-LLM aggregated (1 node, TP=8)
sflow run -f dgx_dynamo_trtllm_agg.yaml \
  --kube-namespace my-ns --kube-node-selector tenant=my-dgx \
  -a LOCAL_MODEL_PATH=fs:///mnt/model-cache/Qwen3-8B-FP8 --dry-run

# GB NVL72 — Dynamo TRT-LLM aggregated (2 nodes, MNNVL + ComputeDomain)
sflow run -f gb_dynamo_trtllm_agg.yaml \
  --kube-namespace my-ns --kube-node-selector tenant=my-gb \
  --kube-compute-domain-channel my-imex-channel \
  -a LOCAL_MODEL_PATH=fs:///mnt/model-cache/Qwen3-8B-FP8 --dry-run

# DGX B200 — disaggregated (prefill node + decode node, KV over IB)
sflow run -f dgx_dynamo_trtllm_disagg.yaml --kube-namespace my-ns \
  --kube-node-selector tenant=my-dgx --dry-run

# GB NVL72 — disaggregated (prefill 2 nodes + decode 2 nodes, KV over MNNVL)
sflow run -f gb_dynamo_trtllm_disagg.yaml --kube-namespace my-ns \
  --kube-node-selector tenant=my-gb --kube-compute-domain-channel my-imex-channel --dry-run

# Pure trtllm-serve (no Dynamo): dgx_trtllm_serve.yaml / gb_trtllm_serve.yaml
sflow run -f dgx_trtllm_serve.yaml --kube-namespace my-ns \
  --kube-node-selector tenant=my-dgx --dry-run
```

Drop `--dry-run` to run for real. `--kube-compute-domain-channel` overrides the recipe's
`COMPUTE_DOMAIN_CHANNEL` per run (GB only); on DGX there is no ComputeDomain to set.

## Notes

- **Dense model:** Qwen3-8B has no experts, so the engine configs use `tensor_parallel_size`
  only — no `moe_expert_parallel_size` / `moe_config` / `enable_attention_dp`, and
  `trtllm-serve` gets `--tp_size` (no `--ep_size`).
- **Model path:** `LOCAL_MODEL_PATH` (`fs://`) must be visible inside the pods — a PVC or
  hostPath mounted at that path (declare a backend `volumes:` entry), or override it with
  `-a LOCAL_MODEL_PATH=...`. On Kubernetes a missing `fs://` path only warns at dry-run
  (it is treated as a remote path).
- **Images:** the Dynamo recipes default to `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime`;
  the pure-serve recipes to `nvcr.io/nvidia/tensorrt-llm/release`. Override with
  `-s DYNAMO_IMAGE=...` / `-s TRTLLM_IMAGE=...`.
- **Kubernetes limitations (v0.3.0):** interactive `sflow run` only, and the built-in
  `monitor:` is skipped on k8s — see `docs/user/backends.md`.
