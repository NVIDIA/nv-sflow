# sflow YAML Examples

Annotated examples covering common workflow patterns.

> For runnable sample workflows: `sflow sample --list` and [samples](https://nvidia.github.io/nv-sflow/docs/user/samples).
> For quickstart tutorial: [quickstart](https://nvidia.github.io/nv-sflow/docs/user/quickstart).

---

## Example 1: Local Hello World

Simplest possible workflow. No backends, no operators -- just runs locally.

```yaml
version: "0.1"

variables:
  WHO:
    description: "Name to greet"
    value: "Nvidia"

workflow:
  name: hello_world
  tasks:
    - name: hello
      script:
        - echo "Hello, ${WHO}!"
```

Run: `sflow run -f hello.yaml --tui`

Override variable: `sflow run -f hello.yaml --set WHO=World`

---

## Example 2: Local DAG with Dependencies

A multi-stage pipeline showing `depends_on` relationships.

```yaml
version: "0.1"

variables:
  DATASET:
    value: cifar10
  EPOCHS:
    type: integer
    value: 10

workflow:
  name: training_pipeline
  tasks:
    - name: prepare_data
      script:
        - echo "Downloading ${DATASET}..."
        - python download_data.py --dataset ${DATASET}

    - name: preprocess
      script:
        - python preprocess.py --input data/${DATASET}
      depends_on:
        - prepare_data

    - name: train
      script:
        - python train.py --epochs ${EPOCHS}
      depends_on:
        - preprocess

    - name: evaluate
      script:
        - python evaluate.py --checkpoint best_model.pt
      depends_on:
        - train

    - name: export_model
      script:
        - python export.py --format onnx
      depends_on:
        - evaluate
```

DAG: `prepare_data -> preprocess -> train -> evaluate -> export_model`

Visualize: `sflow visualize -f pipeline.yaml --format mermaid`

---

## Example 3: SLURM SGLang Disaggregated Inference

Full standalone workflow for disaggregated prefill/decode serving with benchmarking.
This is the most common production pattern.

```yaml
version: "0.1"

variables:
  # SLURM config
  SLURM_ACCOUNT:
    value: my_account
  SLURM_PARTITION:
    value: my_partition
  SLURM_TIMELIMIT:
    value: 120
  GPUS_PER_NODE:
    type: integer
    value: 8
  SLURM_NODES:
    value: 2

  # Model
  SERVED_MODEL_NAME:
    value: Llama-3.1-70B-FP8
  MODEL_NAME:
    value: meta-llama/Llama-3.1-70B-FP8

  # Prefill server
  NUM_CTX_SERVERS:
    type: integer
    value: 1
  CTX_TP_SIZE:
    type: integer
    value: 8
  CTX_DP_SIZE:
    type: integer
    value: 1
  CTX_PP_SIZE:
    type: integer
    value: 1
  CTX_BATCH_SIZE:
    type: integer
    value: 128
  CTX_MAX_NUM_TOKENS:
    type: integer
    value: 8192

  # Decode server
  NUM_GEN_SERVERS:
    type: integer
    value: 1
  GEN_TP_SIZE:
    type: integer
    value: 8
  GEN_DP_SIZE:
    type: integer
    value: 1
  GEN_PP_SIZE:
    type: integer
    value: 1
  GEN_BATCH_SIZE:
    type: integer
    value: 128
  GEN_MAX_NUM_TOKENS:
    type: integer
    value: 8192

  # Benchmark
  ISL:
    value: 1024
  OSL:
    value: 1024
  CONCURRENCY:
    value: 64
    domain: [64]

  # Container
  DYNAMO_IMAGE:
    value: nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.0

artifacts:
  - name: LOCAL_MODEL_PATH
    uri: fs:///models/Llama-3.1-70B-FP8

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    time: ${{ variables.SLURM_TIMELIMIT }}
    nodes: ${{ variables.SLURM_NODES }}
    partition: ${{ variables.SLURM_PARTITION }}
    account: ${{ variables.SLURM_ACCOUNT }}
    gpus_per_node: ${{ variables.GPUS_PER_NODE }}

operators:
  - name: dynamo_sglang
    type: srun
    container_image: ${{ variables.DYNAMO_IMAGE }}
    container_writable: true
    mpi: pmix

workflow:
  name: sglang_disagg
  variables:
    HEAD_NODE_IP:
      value: "${{ backends.slurm_cluster.nodes[0].ip_address }}"
    NATS_SERVER:
      value: "nats://${{ backends.slurm_cluster.nodes[0].ip_address }}:4222"
    ETCD_ENDPOINTS:
      value: "${{ backends.slurm_cluster.nodes[0].ip_address }}:2379"

  tasks:
    # Infrastructure: image pull, NATS, etcd, frontend
    - name: load_image
      operator:
        name: dynamo_sglang
        ntasks: ${{ variables.SLURM_NODES }}
        ntasks_per_node: 1
      script:
        - echo "Image Loaded"
        - sleep 3600
      probes:
        readiness:
          log_watch:
            regex_pattern: "Image Loaded"
            match_count: ${{ variables.SLURM_NODES }}
          timeout: 1200
          interval: 2

    - name: nats_server
      operator: dynamo_sglang
      script:
        - nats-server -js
      resources:
        nodes:
          indices: [0]
      probes:
        readiness:
          tcp_port:
            port: 4222
          timeout: 60
          interval: 2
      depends_on:
        - load_image

    - name: etcd_server
      operator: dynamo_sglang
      script:
        - >
          etcd --listen-client-urls "http://0.0.0.0:2379"
          --advertise-client-urls "http://0.0.0.0:2379"
          --listen-peer-urls "http://0.0.0.0:2380"
          --initial-advertise-peer-urls "http://${HEAD_NODE_IP}:2380"
          --initial-cluster "default=http://${HEAD_NODE_IP}:2380"
          --data-dir /tmp/etcd
      resources:
        nodes:
          indices: [0]
      probes:
        readiness:
          tcp_port:
            port: 2379
          timeout: 60
          interval: 2
      depends_on:
        - load_image

    - name: frontend_server
      operator: dynamo_sglang
      script:
        - python3 -m dynamo.frontend --http-port 8000
      resources:
        nodes:
          indices: [0]
      probes:
        readiness:
          tcp_port:
            port: 8000
          timeout: 120
          interval: 5
      depends_on:
        - nats_server
        - etcd_server

    # Prefill server -- uses replicas for scaling
    - name: prefill_server
      operator:
        name: dynamo_sglang
        ntasks_per_node: 1
      replicas:
        count: ${{ variables.NUM_CTX_SERVERS }}
        policy: "parallel"
      script:
        - set -x
        - export FIRST_CUDA_DEVICE=$(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f1)
        - export VLLM_NIXL_SIDE_CHANNEL_PORT=$((5557 + ${FIRST_CUDA_DEVICE}))
        - export DYN_SYSTEM_PORT=$((8082 + ${FIRST_CUDA_DEVICE}))
        - >
          python3 -m dynamo.sglang
          --model-path ${{ artifacts.LOCAL_MODEL_PATH.path }}
          --served-model-name ${{ variables.SERVED_MODEL_NAME }}
          --tensor-parallel-size ${{ variables.CTX_TP_SIZE }}
          --data-parallel-size ${{ variables.CTX_DP_SIZE }}
          --max-running-requests ${{ variables.CTX_BATCH_SIZE }}
          --max-prefill-tokens ${{ variables.CTX_MAX_NUM_TOKENS }}
          --disaggregation-mode prefill
          --disaggregation-bootstrap-port $((8998 + ${FIRST_CUDA_DEVICE}))
          --disaggregation-transfer-backend nixl
          --load-balance-method round_robin
          --trust-remote-code
          --skip-tokenizer-init
          --host 0.0.0.0
      resources:
        gpus:
          count: ${{ variables.CTX_TP_SIZE * variables.CTX_DP_SIZE * variables.CTX_PP_SIZE }}
      depends_on:
        - frontend_server
      probes:
        readiness:
          log_watch:
            regex_pattern: "orker handler initialized"
          timeout: 600
          interval: 10
        failure:
          log_watch:
            regex_pattern: "Traceback (most recent call last)"
          interval: 10

    # Decode server
    - name: decode_server
      operator:
        name: dynamo_sglang
        ntasks_per_node: 1
      replicas:
        count: ${{ variables.NUM_GEN_SERVERS }}
        policy: "parallel"
      script:
        - set -x
        - export FIRST_CUDA_DEVICE=$(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f1)
        - export VLLM_NIXL_SIDE_CHANNEL_PORT=$((5557 + ${FIRST_CUDA_DEVICE}))
        - export DYN_SYSTEM_PORT=$((8082 + ${FIRST_CUDA_DEVICE}))
        - >
          python3 -m dynamo.sglang
          --model-path ${{ artifacts.LOCAL_MODEL_PATH.path }}
          --served-model-name ${{ variables.SERVED_MODEL_NAME }}
          --tensor-parallel-size ${{ variables.GEN_TP_SIZE }}
          --data-parallel-size ${{ variables.GEN_DP_SIZE }}
          --max-running-requests ${{ variables.GEN_BATCH_SIZE }}
          --max-prefill-tokens ${{ variables.GEN_MAX_NUM_TOKENS }}
          --disaggregation-mode decode
          --disaggregation-bootstrap-port $((8998 + ${FIRST_CUDA_DEVICE}))
          --disaggregation-transfer-backend nixl
          --trust-remote-code
          --skip-tokenizer-init
          --host 0.0.0.0
      resources:
        gpus:
          count: ${{ variables.GEN_TP_SIZE * variables.GEN_DP_SIZE * variables.GEN_PP_SIZE }}
      depends_on:
        - frontend_server
      probes:
        readiness:
          log_watch:
            regex_pattern: "orker handler initialized"
          timeout: 600
          interval: 10
        failure:
          log_watch:
            regex_pattern: "Traceback (most recent call last)"
          interval: 10

    # Benchmark (sequential sweep over CONCURRENCY values)
    - name: benchmark
      operator:
        name: dynamo_sglang
        ntasks: 1
      script:
        - set -x
        - pip install aiperf==0.3.0
        - >
          aiperf profile
          --artifact-dir ${SFLOW_WORKFLOW_OUTPUT_DIR}/aiperf_concurrency_${CONCURRENCY}
          --model ${{ variables.SERVED_MODEL_NAME }}
          --tokenizer ${{ artifacts.LOCAL_MODEL_PATH.path }}
          --endpoint-type chat
          --endpoint /v1/chat/completions
          --streaming
          --url http://${{ variables.HEAD_NODE_IP }}:8000
          --synthetic-input-tokens-mean ${{ variables.ISL }}
          --synthetic-input-tokens-stddev 0
          --output-tokens-mean ${{ variables.OSL }}
          --concurrency ${CONCURRENCY}
          --request-count 512
          --warmup-request-count ${CONCURRENCY}
          --ui simple
      resources:
        nodes:
          indices: [0]
      replicas:
        variables:
          - CONCURRENCY
        policy: sequential
      depends_on:
        - prefill_server
        - decode_server
        - frontend_server
```

Key patterns:
- Infrastructure tasks pinned to node 0 via `nodes.indices: [0]`
- Server tasks use `replicas.count` for scaling
- GPU resources computed from parallelism config: `TP * DP * PP`
- Probes: `tcp_port` for infra, `log_watch` for servers, `failure` probe on Traceback
- Benchmark sweeps via `replicas.variables` with `policy: sequential`

---

## Example 4: Modular Composition (inference_x_v2)

Split a workflow across files for reuse. This is the recommended pattern for production.

**File 1: `slurm_config.yaml`** -- SLURM backend and shared variables:

```yaml
version: "0.1"

variables:
  SLURM_ACCOUNT:
    value: my_account
  SLURM_PARTITION:
    value: my_partition
  SLURM_TIMELIMIT:
    value: 120
  GPUS_PER_NODE:
    type: integer
    value: 8
  SLURM_NODES:
    value: 2

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    time: ${{ variables.SLURM_TIMELIMIT }}
    nodes: ${{ variables.SLURM_NODES }}
    partition: ${{ variables.SLURM_PARTITION }}
    account: ${{ variables.SLURM_ACCOUNT }}
    gpus_per_node: ${{ variables.GPUS_PER_NODE }}
```

**File 2: `common_workflow.yaml`** -- Shared infra tasks (nats, etcd, frontend, etc.)

**File 3: `sglang/prefill.yaml`** -- Prefill server task + variables

**File 4: `sglang/decode.yaml`** -- Decode server task + variables

**File 5: `benchmark_aiperf.yaml`** -- Benchmark task + variables

### Disaggregated mode (prefill + decode):

```bash
sflow compose slurm_config.yaml common_workflow.yaml \
  sglang/prefill.yaml sglang/decode.yaml benchmark_aiperf.yaml \
  --set DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.0 \
  --missable-tasks agg_server \
  -o merged.yaml

sflow run -f merged.yaml --tui
```

### Aggregated mode (single server, no prefill/decode split):

```bash
sflow run \
  -f slurm_config.yaml -f common_workflow.yaml \
  -f sglang/agg.yaml -f benchmark_aiperf.yaml \
  --set DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.0 \
  --missable-tasks prefill_server --missable-tasks decode_server \
  --tui
```

### Batch submission:

```bash
sflow batch \
  -f slurm_config.yaml -f common_workflow.yaml \
  -f sglang/prefill.yaml -f sglang/decode.yaml -f benchmark_aiperf.yaml \
  -p my_partition -A my_account -N 4 --gpus-per-node 8 \
  --set DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.0 \
  --submit
```

---

## Example 5: Variable Sweep with Replicas

Run a benchmark at multiple concurrency levels sequentially:

```yaml
version: "0.1"

variables:
  CONCURRENCY:
    description: "Request concurrency for benchmark"
    value: 16
    domain: [16, 32, 64, 128]

workflow:
  name: sweep_benchmark
  tasks:
    - name: benchmark
      script:
        - echo "Running at concurrency ${CONCURRENCY}"
        - python benchmark.py --concurrency ${CONCURRENCY}
      replicas:
        variables:
          - CONCURRENCY
        policy: sequential
```

This creates 4 replicas: `benchmark_16`, `benchmark_32`, `benchmark_64`, `benchmark_128`.
With `policy: sequential`, each runs only after the previous completes.

For parallel execution of all values, use `policy: "parallel"`.

---

## Example 6: Bulk Input CSV

Run many configurations from a CSV file. Each row produces an independent job.

**bulk_input.csv:**

```csv
sflow_config_file,SLURM_NODES,NUM_CTX_SERVERS,NUM_GEN_SERVERS,CTX_TP_SIZE,GEN_TP_SIZE,CONCURRENCY,missable_tasks
"slurm_config.yaml,common_workflow.yaml,sglang/prefill.yaml,sglang/decode.yaml,benchmark_aiperf.yaml",2,1,1,8,8,"[64,128]",agg_server
"slurm_config.yaml,common_workflow.yaml,sglang/agg.yaml,benchmark_aiperf.yaml",1,,,4,,,"prefill_server,decode_server"
```

```bash
sflow batch --bulk-input bulk_input.csv \
  -p my_partition -A my_account --gpus-per-node 8 \
  --set DYNAMO_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.0 \
  --submit
```

Key CSV columns:
- `sflow_config_file` (required): comma-separated list of YAML files to compose
- Variable columns: override variable values per row
- `missable_tasks`: comma-separated task names to mark as missable
- Empty cells keep the default value from the YAML

## Example 7: Hardware Monitoring

Add a workflow-level monitor (whole pool, full run) and a task-level monitor that
tracks the resources used by the inference servers during the benchmark. Collectors
run on the bare node host; reports are generated once after the run.

```yaml
version: "0.1"
backends:
  - name: slurm_cluster
    type: slurm
    default: true
    nodes: 2
    gpus_per_node: 8
    partition: ${{ variables.SLURM_PARTITION }}
    account: ${{ variables.SLURM_ACCOUNT }}
    time: "01:00:00"
workflow:
  name: serve_and_bench
  monitor:                       # whole pool, full workflow lifetime
    interval: 5000
    report:
      enabled: true              # format defaults to [csv, svg]
  tasks:
    - name: server
      operator: my_container
      script:
        - python -m my.server
      probes:
        readiness:
          tcp_port: { port: 8000 }
    - name: aiperf
      operator: my_container
      depends_on: [server]
      script:
        - aiperf profile ...
      monitor:                   # bound to aiperf; monitors the server's GPUs
        resources:
          used_by_tasks: [server]
        scopes:
          gpu: {}
        report:
          enabled: true
```

After the run, see `<output_dir>/<run_id>/sflow_monitor.log` for the terminal
overview and `<output_dir>/<run_id>/sflow_monitor/` for detailed CSVs.

---

## Example 8: Kubernetes GPU Task

The workload image lives on the **`k8s` operator**; the `kubernetes` backend reserves and
pins nodes (it has no `image`). Cluster access is passed via CLI flags, never YAML.

```yaml
version: "0.1"

backends:
  - name: k8s
    type: kubernetes
    default: true
    namespace: default
    nodes: 1
    gpus_per_node: 8
    scheduling: device_plugin    # default (nvidia.com/gpu) | dra (opt-in, WIP)

operators:
  - name: gpu_op
    type: k8s
    image: nvcr.io/nvidia/pytorch:24.12-py3

workflow:
  name: kubernetes_gpu
  tasks:
    - name: train
      operator: gpu_op
      resources:
        gpus:
          count: 2               # per-task total; on k8s split evenly across assigned nodes
      script:
        - nvidia-smi -L
        - python train.py
```

Run: `sflow run -f k8s_gpu.yaml --kubeconfig ~/.kube/config --kube-context my-ctx --kube-namespace ml-team`

Key patterns:
- Cluster selection is CLI (`--kubeconfig` / `--kube-context` / `--kube-namespace`), keeping
  the recipe cluster-agnostic.
- A multi-node task (`resources.nodes`) becomes one pod per node (leader = index 0), with
  `SFLOW_TASK_NODE_INDEX` / `SFLOW_LEADER_ADDRESS` injected for rendezvous.
- Network probes (`tcp_port` / `http_get`) run from an in-cluster probe pod automatically.
- For any MPI workload (an explicit `mpirun -np N ...`, single- or multi-node) use a
  `type: k8s_mpi` operator; it sets up the MPI world either way. Plain `k8s` is for non-MPI pods.

---

## Example 9: Post-Run S3 Uploads

Declare a `storage:` target and attach per-task `uploads:`. Credentials come from the boto3
default chain — never put secrets in YAML. Install with `pip install 'sflow[s3]'`.

```yaml
version: "0.1"

storage:
  - name: results_bucket
    type: s3
    bucket: my-bench-results
    region: us-west-2
    prefix: "runs/${{ variables.RUN_ID }}/"   # optional; prepended to every key

variables:
  RUN_ID:
    value: "exp-001"

workflow:
  name: bench_and_upload
  # Optional: also zip + upload the entire run output dir at the end.
  upload_all:
    target: results_bucket
    to: "archive/${{ workflow.run_id }}.zip"
    on_error: warn
  tasks:
    - name: benchmark
      script:
        - python benchmark.py --out ${SFLOW_TASK_OUTPUT_DIR}
      uploads:
        - target: results_bucket
          from: "${{ task.output_dir }}/results.csv"
          to: "main/results.csv"          # relative to the target prefix
          on_error: fail                   # warn (default) | fail
        - target: results_bucket
          from: "${{ task.output_dir }}/*.json"   # glob -> `to` must end with '/' or be omitted
          on_error: warn
```

Key patterns:
- `${{ task.output_dir }}` resolves to `SFLOW_TASK_OUTPUT_DIR` at task completion.
- `on_error: fail` flips the task to `FAILED` (triggers fail-fast) — use only when the upload
  IS the deliverable.
- For S3-compatible stores (MinIO, Ceph RGW) set `endpoint_url:` (sflow defaults
  `addressing_style` to `path`).
- `sflow run --dry-run` lists every planned upload and runs an offline credential preflight.

---

## Example 10: Kubernetes with a PVC-Backed Model (shared storage)

A pod's filesystem is not the driver host's disk, so cluster-resident data (a model on shared
storage) must come from a **PersistentVolumeClaim**. Declare it once as a backend `volumes:`
entry, then point an `fs://` artifact at a path *under* its `mount_path`: sflow sees the PVC
covers the path, skips the hostPath fallback, and `${{ artifacts.MODEL.path }}` resolves to the
in-pod path on the PVC. The PVC (and its data) must already exist in the namespace.

```yaml
version: "0.1"

backends:
  - name: k8s
    type: kubernetes
    default: true
    namespace: default
    nodes: 1
    gpus_per_node: 8
    volumes:
      # Read-only shared model store: mounted into EVERY task pod at /models.
      - name: models
        claim: model-store-pvc
        mount_path: /models
        read_only: true
      # Writable PVC to persist run outputs past pod teardown (K8S output dirs are
      # otherwise an ephemeral emptyDir). ensure_writable fixes root-owned subPaths.
      - name: outputs
        claim: run-output-pvc
        mount_path: /sflow_output
        read_only: false
        ensure_writable: true
      # Per-pod ephemeral scratch for a JIT/kernel cache (no PVC needed).
      - name: cache
        empty_dir: { size_limit: 50Gi }
        mount_path: /root/.cache

artifacts:
  # Path is UNDER /models -> served by the PVC (no hostPath). Resolves in-pod to
  # /models/Llama-3.1-70B-FP8 for both YAML and script access.
  - name: MODEL
    uri: fs:///models/Llama-3.1-70B-FP8

operators:
  - name: gpu_op
    type: k8s
    image: nvcr.io/nvidia/pytorch:24.12-py3

workflow:
  name: k8s_pvc_serve
  # Point SFLOW_OUTPUT_DIR at the writable PVC so results/logs persist (K8S output is
  # otherwise emptyDir; small files are copied back, but a PVC keeps everything).
  variables:
    SFLOW_OUTPUT_DIR:
      value: /sflow_output
  tasks:
    - name: serve
      operator: gpu_op
      resources:
        gpus: { count: 8 }
      script:
        - ls ${{ artifacts.MODEL.path }}            # /models/Llama-3.1-70B-FP8 (on the PVC)
        - python -m vllm.entrypoints.openai.api_server --model ${{ artifacts.MODEL.path }}
      probes:
        readiness:
          tcp_port: { port: 8000 }
        failure:
          log_watch: { regex_pattern: "Traceback (most recent call last)" }
```

Run: `sflow run -f k8s_pvc.yaml --kubeconfig ~/.kube/config --kube-namespace ml-team`

Key patterns:
- Backend `volumes:` is workflow-wide storage — mounted into every task pod, even ones whose
  script never names the path (e.g. a frontend that discovers the model dir via etcd).
- `fs://` artifact under a PVC `mount_path` → the PVC serves it (no hostPath); an `fs://` path
  **not** covered by any PVC would instead hostPath-mount at the same path and fail loudly if the
  node lacks it.
- K8S task output dirs are an ephemeral `emptyDir` (10 MiB collect-back to the driver). Mount a
  writable PVC at the output path (as above) or use `uploads:` for anything larger or persistent.
- `--kube-skip-pvc` drops PVC volumes (keeps `empty_dir`) for a quick scheduling check on a
  cluster without the recipe's PVCs.

---

## Example 11: Slurm with Lustre Shared Storage + Artifacts

On Slurm, keep models, datasets, the workspace, and the output dir on the shared filesystem
(Lustre / GPFS / NFS) so every node sees the *same absolute path*. A single `fs://` artifact
then auto-mounts identically into the pyxis container on every node — no per-node mount config,
no manual `container_mounts`. The auto-mount is a same-path bind (`{path}:{path}:rw`), so
`${{ artifacts.X.path }}` is valid on the host and inside the container alike.

```yaml
version: "0.1"

variables:
  SLURM_ACCOUNT: { value: my_account }
  SLURM_PARTITION: { value: my_partition }

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    nodes: 2
    gpus_per_node: 8
    partition: ${{ variables.SLURM_PARTITION }}
    account: ${{ variables.SLURM_ACCOUNT }}
    time: "01:00:00"

artifacts:
  # All on Lustre — same path on every allocated node, so the same-path mount just works.
  - name: MODEL
    uri: fs:///lustre/shared/models/Llama-3.1-70B-FP8
  - name: DATASET
    uri: fs:///lustre/shared/datasets/eval
  # A generated helper: relative file:// -> written under the (shared) output dir, then
  # bind-mounted same-path into the container. Never embed multi-line Python inline.
  - name: EVAL_SCRIPT
    uri: file://eval.py
    content: |
      import sys, json
      print(json.dumps({"model": sys.argv[1], "data": sys.argv[2]}))

operators:
  - name: my_container
    type: srun
    container_image: nvcr.io/nvidia/pytorch:24.12-py3
    container_writable: true
    mpi: pmix
    # container_mounts: only for EXTRA host paths NOT already declared as artifacts, e.g.:
    # container_mounts: ["/lustre/shared/hf_cache:/root/.cache/huggingface:rw"]

workflow:
  name: slurm_lustre_eval
  tasks:
    - name: evaluate
      operator: my_container
      resources:
        gpus: { count: 8 }
      script:
        # All three paths resolve identically on the host and inside the container.
        - python ${{ artifacts.EVAL_SCRIPT.path }} ${{ artifacts.MODEL.path }} ${{ artifacts.DATASET.path }}
        - echo "wrote to ${SFLOW_TASK_OUTPUT_DIR}"   # SFLOW_* dirs auto-mount same-path too
```

Run from a login node whose cwd is on shared storage:
`sflow run -f slurm_lustre.yaml --output-dir /lustre/shared/runs/$USER/sflow_output`

Key patterns:
- Shared storage + same-path mount = one `fs://` artifact works across all nodes with zero
  per-node mount config.
- `--workspace-dir` / `--output-dir` should point at shared storage for multi-node runs, so the
  workspace and auto-mounted `SFLOW_*` output dirs exist at the same path on every node.
- Reach for the srun operator's `container_mounts` only for host paths you did *not* declare as
  artifacts; declared artifacts are already auto-mounted.
- A bare (non-container) srun task accesses shared paths directly — no mount needed at all.

---

## Example 12: Declarative Result Parsing (`result:`)

Turn a task's log/stdout into structured metrics with a `result:` block — sflow writes
`${SFLOW_TASK_OUTPUT_DIR}/result.json` per task and aggregates a workflow-level
`${SFLOW_WORKFLOW_OUTPUT_DIR}/results.json`. Prefer this over the legacy `outputs:`. A task
is **not** `COMPLETED` until parsing finishes (and any `uploads:` run).

```yaml
version: "0.1"
workflow:
  name: bench_and_parse
  tasks:
    # (a) Simple form: a name -> regex map. Each capture group 1 becomes a metric.
    - name: quick_bench
      script:
        - echo "throughput: 376.48 tokens/sec"
        - echo "p50 latency: 5.17 ms"
      result:
        throughput: "throughput:\\s*([0-9.]+)"
        latency_p50: "p50 latency:\\s*([0-9.]+)"

    # (b) Advanced form: typed patterns with aggregation (source defaults to the task log).
    - name: full_bench
      depends_on: [quick_bench]
      script:
        - python benchmark.py     # prints many "iter=.. tok/s=.." lines
      result:
        source: log               # only "log" is implemented today
        patterns:
          - { name: out_tok_s, regex: "tok/s=([0-9.]+)", type: float, aggregate: max, unit: "tok/s" }
          - { name: iters,     regex: "iter=([0-9]+)",   type: int,   aggregate: count }
          - { name: model,     regex: "model=(\\S+)",    required: true }

    # (c) File form: read a JSON the task already wrote (path must end in .json).
    - name: aiperf
      depends_on: [full_bench]
      script:
        - aiperf profile ... --output ${SFLOW_TASK_OUTPUT_DIR}/profile.json
      result:
        file: "${{ task.output_dir }}/profile.json"
```

Key patterns:
- Three shapes, **`patterns` and `file` are mutually exclusive**; the simple map is sugar for
  `patterns`. `aggregate`: `first|last|list|count|min|max|avg|sum` (default `last`);
  `type`: `auto|string|int|float|bool|json`.
- Regexes are YAML strings — escape backslashes (`\\s`, `\\S`, `[0-9.]+`); capture **group 1**
  is the value unless you set `group:`.
- Parsing is **best-effort**: a miss or a failed `required:` spec logs a warning and sets
  `ok=false`/`errors` in `result.json`, but does not by itself fail the task.
- `result.json` keys: `schema, task, status, ok, values, metadata, matches, errors`; the env
  vars `SFLOW_TASK_RESULT_FILE` / `SFLOW_WORKFLOW_RESULT_FILE` point at the two files.
- See the results doc: [nvidia.github.io/nv-sflow/docs/user/results](https://nvidia.github.io/nv-sflow/docs/user/results).
