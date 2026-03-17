# sflow YAML Examples

Annotated examples covering common workflow patterns.

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
