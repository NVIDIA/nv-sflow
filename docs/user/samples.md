---
title: Sample Workflows
sidebar_position: 12
---

This page contains sample workflow configurations that you can use as starting points for your own workflows. You can also access these samples using the `sflow sample` command.

> 📁 **View original sample files**: [src/sflow/samples](https://github.com/NVIDIA/nv-sflow/tree/main/src/sflow/samples)

## Listing Available Samples

```bash
# List all available samples
sflow sample --list

# Copy a sample to your current directory
sflow sample self_contained/local/hello_world

# Copy with custom output path
sflow sample self_contained/local/dag --output my_workflow.yaml
```

> **Agent skills:** beyond samples, `sflow` bundles AI-agent skills for writing and debugging sflow YAML. Copy them into your project with `sflow skill` (`--list` to see them, `-o <dir>` to choose the output directory). See the [CLI reference](./cli.md#sflow-skill).

## Sample catalog

Two families ship with sflow:

- **`self_contained/<backend>/<name>`** — one file, ready to run. Grouped below by backend.
- **`modular/inference_x_v2`** — a composable recipe set (framework fragments + benchmarks + a CSV sweep). See [Modular inference recipe](#modular-inference-recipe-inference_x_v2).

| Backend | `sflow sample <path>` | What it shows |
|---------|-----------------------|---------------|
| local | `self_contained/local/hello_world` | Minimal single task + a variable |
| local | `self_contained/local/dag` | Multi-task DAG with `depends_on` |
| local | `self_contained/local/variable_domain` | Variable `domain` (allowed-value validation) |
| local | `self_contained/local/result_parsing` | Parse metrics from a log/JSON into `result.json` |
| local | `self_contained/local/storage_upload` | Per-task `uploads:` to an S3 target |
| local | `self_contained/local/storage_upload_all` | `upload_all:` — zip the whole run and upload |
| local | `self_contained/local/monitor` | `monitor:` hardware telemetry (GPU/CPU/mem/net) |
| docker | `self_contained/docker/hello_world` | Single container task via `docker_run` |
| docker | `self_contained/docker/multi_node` | Multi-host Docker (`docker_host`/`context`) |
| docker | `self_contained/docker/sglang_qwen3` | SGLang Qwen3 server + client in containers |
| docker | `self_contained/docker/gpu_monitor` | GPU monitor with a log-marker report `window:` — runs a stock CUDA `nbody` container as a deliberate idle/burst/idle square wave, so the lifecycle report and the windowed report visibly differ |
| slurm | `self_contained/slurm/sglang_server_client` | Server + client on Slurm with readiness probes |
| slurm | `self_contained/slurm/aiperf_template` | AIPerf benchmark template |
| slurm | `self_contained/slurm/auto_replica` | Replica fan-out sized from a variable sweep |
| slurm | `self_contained/slurm/resource_release_after` | `release_after` GPU/node lifetimes |
| slurm | `self_contained/slurm/gpu_indices` | `resources.gpus` — `count` vs `indices` vs both, with the 4-node × 4-GPU planner output annotated |
| slurm | `self_contained/slurm/gpu_placement_matrix` | GPU-placement regression matrix: container vs bare step, high slice offset, two tasks sharing a node, multi-node — proves placement by UUID rather than by GPU count |
| slurm | `self_contained/slurm/monitor_mixed` | The broadest single-job regression net: two Slurm pools/operators, replicas with cross-task refs, `release_after` GPU reuse, the placement matrix, and a monitor with a log-marker window. **If you only run one recipe on a new cluster, run this one.** |
| slurm | `self_contained/slurm/multi_backend` | One workflow spanning multiple backends |
| slurm | `self_contained/slurm/trtllm_serve_disagg` | TRT-LLM disaggregated serving |
| slurm | `self_contained/slurm/infmax_v1_ds_r1` | InfMax DeepSeek-R1 benchmark |
| slurm | `self_contained/slurm/dynamo_{sglang,vllm,trtllm}_agg` | Dynamo aggregated inference (3 frameworks) |
| slurm | `self_contained/slurm/dynamo_{sglang,vllm,trtllm}_disagg` | Dynamo prefill/decode disaggregated (3 frameworks) |
| kubernetes | `self_contained/kubernetes/hello_world` | Single pod via the `k8s` operator |
| kubernetes | `self_contained/kubernetes/dynamo_trtllm_disagg` | Multi-node MPI (`k8s_mpi`), RDMA, node reservation |

Run `sflow sample --list` for the live list (it annotates each sample with node/GPU needs).

---

## Local Samples

These samples run locally without requiring a Slurm cluster.

### Hello World

A minimal example that demonstrates basic sflow concepts.

```yaml
version: "0.1"

variables:
  WHO:
    description: "who to greet"
    value: Nvidia

workflow:
  name: local_hello_world
  tasks:
    - name: hello
      script:
        - echo "Hello ${WHO}"
```

**Run it:**

```bash
sflow sample self_contained/local/hello_world
sflow run -f hello_world.yaml
```

---

### DAG Workflow

A multi-task workflow demonstrating task dependencies, data flow between tasks, and parallel execution.

```yaml
version: "0.1"

variables:
  - name: MODEL_NAME
    type: string
    value: tiny-transformer

workflow:
  name: quickstart_dag
  tasks:
    - name: prepare_data
      script:
        - echo "prepare_data start"
        - echo "model(jinja)=${{ variables.MODEL_NAME }}" > ${SFLOW_WORKFLOW_OUTPUT_DIR}/dataset.txt
        - echo "model(shell)=${MODEL_NAME}" >> ${SFLOW_WORKFLOW_OUTPUT_DIR}/dataset.txt

    - name: preprocess
      depends_on: [prepare_data]
      script:
        - test -f ${SFLOW_WORKFLOW_OUTPUT_DIR}/dataset.txt
        - grep -q "model(jinja)=tiny-transformer" ${SFLOW_WORKFLOW_OUTPUT_DIR}/dataset.txt
        - grep -q "model(shell)=tiny-transformer" ${SFLOW_WORKFLOW_OUTPUT_DIR}/dataset.txt
        - echo "encoded_data ok" > ${SFLOW_WORKFLOW_OUTPUT_DIR}/encoded.txt

    - name: train
      depends_on: [preprocess]
      script:
        - test -f ${SFLOW_WORKFLOW_OUTPUT_DIR}/encoded.txt
        - echo "checkpoint for ${MODEL_NAME}" > ${SFLOW_WORKFLOW_OUTPUT_DIR}/checkpoint.pt

    - name: evaluate_on_dataset1
      depends_on: [train]
      script:
        - test -f ${SFLOW_WORKFLOW_OUTPUT_DIR}/checkpoint.pt
        - echo "accuracy=0.99 dataset=dataset1" > ${SFLOW_TASK_OUTPUT_DIR}/metrics.txt

    - name: evaluate_on_dataset2
      depends_on: [train]
      script:
        - test -f ${SFLOW_WORKFLOW_OUTPUT_DIR}/checkpoint.pt
        - echo "accuracy=0.88 dataset=dataset2" > ${SFLOW_TASK_OUTPUT_DIR}/metrics.txt

    - name: export_model
      depends_on: [evaluate_on_dataset1, evaluate_on_dataset2]
      script:
        - test -f ${SFLOW_WORKFLOW_OUTPUT_DIR}/evaluate_on_dataset1/metrics.txt
        - test -f ${SFLOW_WORKFLOW_OUTPUT_DIR}/evaluate_on_dataset2/metrics.txt
        - echo "exported ${MODEL_NAME}" > ${SFLOW_WORKFLOW_OUTPUT_DIR}/model.onnx
```

**Run it:**

```bash
sflow sample self_contained/local/dag
sflow run -f dag.yaml --dry-run  # Validate
sflow run -f dag.yaml            # Execute
```

---

### Result Parsing

Demonstrates the consolidated [`task.result`](./results.md) entry: parsing
metrics from a task log with a regex map, writing JSON directly to
`$SFLOW_TASK_RESULT_FILE`, and a downstream task reading the per-task `result.json`
and workflow-level `results.json` index.

```yaml
version: "0.1"

backends:
  - name: local
    type: local
    default: true
    nodes: 1

workflow:
  name: local_result_parsing
  tasks:
    - name: benchmark_log
      script:
        - |
          echo "TTFT: 40.0 ms"
          echo "TTFT: 42.5 ms"
          echo "tok/s: 123.0"
          echo "p99 latency: 88 ms"
      result:
        ttft: 'TTFT:\s*([0-9.]+)\s*ms'      # last match wins -> 42.5
        tps: 'tok/s:\s*([0-9.]+)'
        latency_p99: 'p99 latency:\s*([0-9.]+)\s*ms'

    - name: benchmark_file
      script:
        - |
          echo '{"throughput": 999.5, "errors": 0}' > "$SFLOW_TASK_RESULT_FILE"
      result:
        file: result.json

    - name: verify
      depends_on: [benchmark_log, benchmark_file]
      script:
        - test -f "$SFLOW_WORKFLOW_OUTPUT_DIR/benchmark_log/result.json"
        - test -f "$SFLOW_WORKFLOW_RESULT_FILE"
```

**Run it:**

```bash
sflow sample self_contained/local/result_parsing
sflow run -f result_parsing.yaml
```

---

### Variable Domain

Sweeps a task across a variable's `domain` values, running one execution per value — a compact way to see replica sweeps without a cluster.

```bash
sflow sample self_contained/local/variable_domain
sflow run -f variable_domain.yaml
```

---

### Storage Upload

Post-execution [uploads](./uploads.md): per-task `uploads:` specs ship logs and result files to a named `storage` target (e.g. S3). `storage_upload` uploads selected files; `storage_upload_all` uses `upload_all` to ship every task's outputs.

```bash
sflow sample self_contained/local/storage_upload
sflow run -f storage_upload.yaml

sflow sample self_contained/local/storage_upload_all
sflow run -f storage_upload_all.yaml
```

> S3 uploads need the S3 extra: `pip install 'sflow[s3]'` (credentials come from the boto3 default chain).

---

### Workflow Monitor

Attaches a hardware-utilization [monitor](./monitor.md) at the workflow level, sampling GPU/CPU usage during the run and (optionally) rendering charts.

```bash
sflow sample self_contained/local/monitor
sflow run -f monitor.yaml
```

> PNG charts need the monitor extra: `pip install 'sflow[monitor]'`.

---

## Docker Samples

These samples use the local **Docker backend** to run containerized workloads on
your workstation. They require Docker on PATH (and, for GPU samples, an NVIDIA GPU
plus the NVIDIA Container Toolkit so `--gpus all` works).

### Hello World (Docker)

The Docker counterpart of the local hello world: runs a one-line task inside a container using the `docker` backend and its default `docker_run` operator.

```bash
sflow sample self_contained/docker/hello_world
sflow run -f hello_world.yaml
```

---

### Multi-Node Docker Hosts

Fans a single task out across an explicit pool of remote Docker hosts (one container per host), demonstrating multi-host placement without Slurm. See [Backends](./backends.md#multi-node-docker-hosts).

```bash
sflow sample self_contained/docker/multi_node
sflow run -f multi_node.yaml
```

---

### SGLang Serving Qwen3-0.6B (Local Container)

`self_contained/docker/sglang_qwen3` is the local-container counterpart of
`self_contained/slurm/sglang_server_client`: it stands up SGLang serving **Qwen3-0.6B** in a
container, runs a tiny stdlib client against the OpenAI-compatible API, then runs
an **AIPerf benchmark** at concurrency 8 — no Slurm required.

Highlights:

- **Local Docker backend** (`type: docker`) with one GPU; the `docker_run`
  operator launches the SGLang image with `--gpus all --network host`.
- A **readiness** `log_watch` probe releases downstream tasks only after SGLang
  prints its "ready to roll" banner, plus a **failure** probe for `Traceback`.
- The client is a `file://` artifact (stdlib `urllib`), so no extra packages are
  installed; it reads the model name / prompt / port from sflow-injected env vars.
- An **AIPerf** task (CPU-only `python:3.12-slim` container) `pip install`s aiperf
  and benchmarks the server (`--concurrency 8`), mirroring the benchmark task in
  `self_contained/slurm/dynamo_trtllm_agg`. Results land under the task's output dir.

```bash
sflow sample self_contained/docker/sglang_qwen3
sflow run -f sglang_qwen3.yaml

# Validate only
sflow run -f sglang_qwen3.yaml --dry-run

# Add hardware monitoring without editing the recipe
sflow run -f sglang_qwen3.yaml --enable-workflow-monitor
```

---

## Slurm Samples

These samples require a Slurm cluster with GPU resources.

### SGLang Server + Benchmark (Single Node)

Deploys an SGLang inference server with AIPerf benchmarking on Slurm.

**Features:**
- SGLang server with FP8 inference
- Hardware monitoring driven from the benchmark task (declarative `monitor` with `used_by_tasks`)
- AIPerf benchmarking client
- Readiness probes for service orchestration

```yaml
version: "0.1"

variables:
  # Slurm Configuration
  SLURM_ACCOUNT:
    description: "SLURM account"
    value: your_account
  SLURM_PARTITION:
    description: "SLURM partition"
    value: your_partition
  SLURM_TIMELIMIT:
    description: "SLURM time limit"
    value: 60
  GPUS_PER_NODE:
    description: "GPUs per node"
    value: 4
  SLURM_NODES:
    description: "Number of nodes"
    value: 1
  
  # Model Configuration
  HF_MODEL_NAME:
    description: "HF model name"
    value: Qwen/Qwen3-0.6B-FP8
  SERVED_MODEL_NAME:
    description: "Served model name"
    value: Qwen3-0-6B-FP8
  LOCAL_MODEL_PATH:
    description: "Local model path"
    value: /tmp/models/Qwen3-0.6B-FP8
  
  # SGLang Server Configuration
  NUM_SERVERS:
    description: "Number of servers"
    value: 1
  TP_SIZE:
    description: "Tensor parallel size"
    value: 4
  MAX_RUNNING_REQUESTS:
    description: "Max running requests"
    value: 32
  
  # Benchmark Configuration
  ISL:
    description: "Input sequence length"
    value: 1024
  OSL:
    description: "Output sequence length"
    value: 1024
  MULTI_ROUND:
    description: "Number of benchmark rounds"
    value: 8
  CONCURRENCY:
    description: "Concurrency"
    value: 32
  
  # Container Images
  SGLANG_IMAGE:
    description: "SGLang image"
    value: "lmsysorg/sglang:v0.5.7-cu130-runtime"
  AIPERF_IMAGE:
    description: "AIPerf container image"
    value: python:3.12-slim

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
  - name: sglang_runtime
    type: srun
    container_name: sglang_runtime
    container_writable: true
    container_mount_home: false
    ntasks_per_node: 1
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.SGLANG_IMAGE }}
  - name: aiperf
    type: srun
    container_name: aiperf
    container_writable: true
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.AIPERF_IMAGE }}

workflow:
  name: sglang_qwen3_0_6b
  timeout: 60m
  variables:
    HEAD_NODE_IP:
      description: "Head node IP"
      value: "${{ backends.slurm_cluster.nodes[0].ip_address }}"
  tasks:
    - name: load_image
      operator: 
        name: sglang_runtime
        ntasks_per_node: 1
      script:
        - echo "Image Loaded"
        - sleep 3600
      probes:
        readiness:
          log_watch:
            regex_pattern: "Image Loaded"
          timeout: 1200
          interval: 2

    - name: install_aiperf
      operator: 
        name: aiperf
        ntasks_per_node: 1
      script:
        - pip install aiperf==0.3.0
        - hf download ${{ variables.HF_MODEL_NAME }} --local-dir ${{ variables.LOCAL_MODEL_PATH }}
        - echo "AIPerf installed"
        - sleep 3600
      probes:
        readiness:
          log_watch:
            regex_pattern: "AIPerf installed"
          timeout: 1200
          interval: 2

    - name: sglang_server
      operator: sglang_runtime
      replicas:
        count: ${{ variables.NUM_SERVERS }}
        policy: parallel
      resources:
        gpus:
          count: ${{ variables.TP_SIZE }}
        nodes:
          indices: [0]
      script:
        - set -x
        - export SGLANG_DISABLE_WATCHDOG=1
        - >
          python -m sglang_router.launch_server --model ${{ variables.HF_MODEL_NAME }} 
          --host 0.0.0.0
          --port 8000
          --fp8-gemm-backend flashinfer_trtllm
          --moe-runner-backend flashinfer_trtllm
          --served-model-name ${{ variables.SERVED_MODEL_NAME }}
          --tensor-parallel-size ${{ variables.TP_SIZE }}
          --trust-remote-code
          --max-running-requests ${{ variables.MAX_RUNNING_REQUESTS }}
      probes:
        readiness:
          log_watch:
            regex_pattern: "Workflow completed"
      depends_on:
        - load_image

    - name: benchmark
      operator:
        name: aiperf
        ntasks: 1
      script:
        - set -x
        - >
          aiperf profile --artifact-dir ${SFLOW_WORKFLOW_OUTPUT_DIR}/aiperf_concurrency_${CONCURRENCY}
          --model ${{ variables.SERVED_MODEL_NAME }}
          --tokenizer ${{ variables.LOCAL_MODEL_PATH }}
          --endpoint-type chat
          --endpoint /v1/chat/completions
          --streaming
          --url http://${{ variables.HEAD_NODE_IP }}:8000
          --synthetic-input-tokens-mean ${{ variables.ISL }}
          --synthetic-input-tokens-stddev 0
          --output-tokens-mean ${{ variables.OSL }}
          --output-tokens-stddev 0
          --extra-inputs "max_tokens:${{ variables.OSL }}"
          --extra-inputs "min_tokens:${{ variables.OSL }}"
          --extra-inputs "ignore_eos:true"
          --concurrency ${CONCURRENCY}
          --request-count $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --warmup-request-count ${CONCURRENCY}
          --num-dataset-entries $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --random-seed 100
          --ui simple
        - echo "Benchmarking finished"
      resources:
        nodes:
          indices: [0]
      # Sample the SERVER's resources from this benchmark task (used_by_tasks),
      # not the benchmark client's own node, so the report captures the server's
      # GPU/CPU usage over the benchmark window.
      monitor:
        resources:
          used_by_tasks:
            - sglang_server
        report:
          enabled: true
      depends_on:
        - sglang_server
        - install_aiperf
```

**Run it:**

```bash
sflow sample self_contained/slurm/sglang_server_client

# Validate configuration
sflow run -f sglang_server_client.yaml \
  --set SLURM_ACCOUNT=your_account \
  --set SLURM_PARTITION=your_partition \
  --dry-run

# Submit to Slurm
sflow batch -f sglang_server_client.yaml \
  -A your_account -p your_partition -N 1 -G 4 \
  --sbatch-path sglang_job.sh --submit
```

---

### Dynamo TRT-LLM Disaggregated Inference (Single Node)

Deploys a disaggregated inference setup with separate prefill and decode servers using NVIDIA Dynamo and TensorRT-LLM.

**Features:**
- Disaggregated prefill/decode architecture
- NATS and etcd for service discovery
- Configurable tensor parallelism
- Sequential benchmark sweeps with variable domains
- Retry policies for server reliability
- File-type artifacts for dynamic configuration

```yaml
version: "0.1"

variables:
  # Slurm Configuration
  SLURM_ACCOUNT:
    description: "SLURM account"
    value: your_account
  SLURM_PARTITION:
    description: "SLURM partition"
    value: your_partition
  SLURM_TIMELIMIT:
    description: "SLURM time limit"
    value: 120
  GPUS_PER_NODE:
    description: "GPUs per node"
    value: 4
  SLURM_NODES:
    description: "Number of nodes"
    value: 1

  # Model Configuration
  SERVED_MODEL_NAME:
    description: "Served model name"
    value: Qwen3-0-6B-FP8
  MODEL_NAME:
    description: "Model path"
    value: Qwen/Qwen3-0.6B-FP8
  LOCAL_MODEL_PATH:
    description: "Local model path"
    value: /tmp/models/Qwen3-0.6B-FP8

  # Prefill Server Configuration
  NUM_CTX_SERVERS:
    description: "Number of context/prefill servers"
    value: 1
  CTX_TP_SIZE:
    description: "Context tensor parallel size"
    value: 2

  # Decode Server Configuration
  NUM_GEN_SERVERS:
    description: "Number of generation/decode servers"
    value: 1
  GEN_TP_SIZE:
    description: "Generation tensor parallel size"
    value: 2

  # Benchmark Configuration with Domain Sweep
  CONCURRENCY:
    description: "Concurrency"
    value: 64
    domain: [64, 128]  # Will create sequential benchmark runs
  
  # Container Images
  DYNAMO_IMAGE:
    description: "Dynamo TRTLLM container image"
    value: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.0

artifacts:
  # File-type artifacts are generated by sflow with dynamic content
  - name: PREFILL_CONFIG
    uri: file://prefill_config.yaml
    content: |
      max_batch_size: 128
      tensor_parallel_size: ${{ variables.CTX_TP_SIZE }}
      # ... additional configuration

  - name: DECODE_CONFIG
    uri: file://decode_config.yaml
    content: |
      tensor_parallel_size: ${{ variables.GEN_TP_SIZE }}
      # ... additional configuration

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
  - name: dynamo_trtllm
    type: srun
    container_name: dynamo_trtllm
    container_writable: true
    container_mount_home: false
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.DYNAMO_IMAGE }}

workflow:
  name: dynamo
  timeout: 115m
  variables:
    HEAD_NODE_IP:
      value: "${{ backends.slurm_cluster.nodes[0].ip_address }}"
    ETCD_ENDPOINTS:
      value: "${{ backends.slurm_cluster.nodes[0].ip_address }}:2379"
    NATS_SERVER:
      value: "nats://${{ backends.slurm_cluster.nodes[0].ip_address }}:4222"

  tasks:
    - name: nats_server
      operator: dynamo_trtllm
      script:
        - nats-server -js
      probes:
        readiness:
          tcp_port:
            port: 4222
          timeout: 60

    - name: etcd_server
      operator: dynamo_trtllm
      script:
        - etcd --listen-client-urls "http://0.0.0.0:2379" ...
      probes:
        readiness:
          tcp_port:
            port: 2379
          timeout: 60

    - name: frontend_server
      operator: dynamo_trtllm
      script:
        - python3 -m dynamo.frontend --http-port 8000
      probes:
        readiness:
          tcp_port:
            port: 8000
            timeout: 120
      depends_on:
        - nats_server
        - etcd_server

    - name: prefill_server
      operator:
        name: dynamo_trtllm
        ntasks: ${{ variables.CTX_TP_SIZE }}
      replicas:
        count: ${{ variables.NUM_CTX_SERVERS }}
        policy: parallel
      script:
        - trtllm-llmapi-launch python3 -m dynamo.trtllm --disaggregation-mode prefill ...
      resources:
        gpus:
          count: ${{ variables.CTX_TP_SIZE }}
      probes:
        readiness:
          log_watch:
            regex_pattern: "Setting PyTorch memory fraction"
          timeout: 600
        failure:
          log_watch:
            regex_pattern: "Traceback (most recent call last)"
      retries:
        count: 3
        interval: 30
        backoff: 2
      depends_on:
        - frontend_server

    - name: decode_server
      operator:
        name: dynamo_trtllm
        ntasks: ${{ variables.GEN_TP_SIZE }}
      replicas:
        count: ${{ variables.NUM_GEN_SERVERS }}
        policy: parallel
      script:
        - trtllm-llmapi-launch python3 -m dynamo.trtllm --disaggregation-mode decode ...
      resources:
        gpus:
          count: ${{ variables.GEN_TP_SIZE }}
      retries:
        count: 3
        interval: 30
        backoff: 2
      depends_on:
        - frontend_server

    - name: benchmark
      operator:
        name: aiperf
        ntasks: 1
      replicas:
        variables:
          - CONCURRENCY  # Sweeps over domain [64, 128]
        policy: sequential
      script:
        - aiperf profile --concurrency ${CONCURRENCY} ...
      depends_on:
        - prefill_server
        - decode_server
        - frontend_server
```

**Run it:**

```bash
sflow sample self_contained/slurm/dynamo_trtllm_disagg

# Validate configuration
sflow run -f dynamo_trtllm_disagg.yaml \
  --set SLURM_ACCOUNT=your_account \
  --set SLURM_PARTITION=your_partition \
  --dry-run

# Submit to Slurm
sflow batch -f dynamo_trtllm_disagg.yaml \
  -A your_account -p your_partition -N 1 -G 4 \
  --sbatch-path dynamo_job.sh --submit
```

---

### TRT-LLM Serve Disaggregated Inference (Single Node)

Deploys a disaggregated inference setup with separate prefill and decode servers using TensorRT-LLM's native `trtllm-serve disaggregated` command.

**Features:**
- Disaggregated prefill/decode architecture with `trtllm-serve`
- Dynamic configuration using file-type artifacts with backend node IP resolution
- Configurable tensor parallelism for prefill and decode servers
- Hardware monitoring driven from the benchmark task (declarative `monitor` with `used_by_tasks`)
- Sequential benchmark sweeps with variable domains
- Failure probes for error detection

```yaml
version: "0.1"

variables:
  # Slurm Configuration
  SLURM_ACCOUNT:
    description: "SLURM account"
    value: your_account
  SLURM_PARTITION:
    description: "SLURM partition"
    value: your_partition
  SLURM_TIMELIMIT:
    description: "SLURM time limit"
    value: 120
  GPUS_PER_NODE:
    description: "GPUs per node"
    value: 4
  SLURM_NODES:
    description: "Number of nodes"
    value: 1

  # Model Configuration
  SERVED_MODEL_NAME:
    description: "Served model name"
    value: Qwen3-0-6B-FP8
  MODEL_NAME:
    description: "Model path"
    value: Qwen/Qwen3-0.6B-FP8
  LOCAL_MODEL_PATH:
    description: "Local model path"
    value: /tmp/models/Qwen3-0.6B-FP8

  # Prefill Server Configuration
  NUM_CTX_SERVERS:
    description: "Number of context/prefill servers"
    value: 1
  CTX_TP_SIZE:
    description: "Context tensor parallel size"
    value: 2

  # Decode Server Configuration
  NUM_GEN_SERVERS:
    description: "Number of generation/decode servers"
    value: 1
  GEN_TP_SIZE:
    description: "Generation tensor parallel size"
    value: 2

  # Benchmark Configuration with Domain Sweep
  CONCURRENCY:
    description: "Concurrency"
    value: 128
    domain: [128, 256]  # Will create sequential benchmark runs

  # Container Images
  TRTLLM_IMAGE:
    description: "TRT-LLM container image"
    value: nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post2
  AIPERF_IMAGE:
    description: "AIPerf container image"
    value: python:3.12-slim

artifacts:
  # File-type artifacts with dynamic backend node IP resolution
  - name: SERVER_CONFIG
    uri: file://server_config.yaml
    content: |
      hostname: ${{ backends.slurm_cluster.nodes[0].ip_address }}
      port: 8000
      backend: pytorch
      context_servers:
        num_instances: ${{ variables.NUM_CTX_SERVERS }}
        urls:
          - ${{ backends.slurm_cluster.nodes[0].ip_address }}:8536
      generation_servers:
        num_instances: ${{ variables.NUM_GEN_SERVERS }}
        urls:
          - ${{ backends.slurm_cluster.nodes[0].ip_address }}:8336

  - name: PREFILL_CONFIG
    uri: file://prefill_config.yaml
    content: |
      max_batch_size: 128
      tensor_parallel_size: ${{ variables.CTX_TP_SIZE }}
      # ... additional configuration

  - name: DECODE_CONFIG
    uri: file://decode_config.yaml
    content: |
      tensor_parallel_size: ${{ variables.GEN_TP_SIZE }}
      # ... additional configuration

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
  - name: trtllm_container
    type: srun
    container_name: trtllm_container
    container_writable: true
    container_mount_home: false
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.TRTLLM_IMAGE }}
  - name: aiperf
    type: srun
    container_name: aiperf
    container_writable: true
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.AIPERF_IMAGE }}

workflow:
  name: trtllm_server_disagg
  timeout: 115m
  variables:
    HEAD_NODE_IP:
      description: "Head node IP (resolved after allocation)"
      value: "${{ backends.slurm_cluster.nodes[0].ip_address }}"

  tasks:
    - name: load_image
      operator:
        name: trtllm_container
        ntasks_per_node: 1
      script:
        - hf download ${{ variables.MODEL_NAME }} --local-dir ${{ variables.LOCAL_MODEL_PATH }}
        - echo "Image Loaded"
        - sleep 3600
      probes:
        readiness:
          log_watch:
            regex_pattern: "Image Loaded"
          timeout: 1200

    - name: frontend_server
      operator: trtllm_container
      script:
        - cat ${{ artifacts.SERVER_CONFIG.path }}
        - trtllm-serve disaggregated -c ${{ artifacts.SERVER_CONFIG.path }} -t 7200 -r 7200
      resources:
        nodes:
          indices: [0]
      probes:
        readiness:
          log_watch:
            regex_pattern: "Application startup complete"
          timeout: 120
      depends_on:
        - prefill_server
        - decode_server

    - name: prefill_server
      operator:
        name: trtllm_container
        ntasks: ${{ variables.CTX_TP_SIZE }}
        ntasks_per_node: ${{ [ variables.CTX_TP_SIZE, variables.GPUS_PER_NODE ] | min }}
      replicas:
        count: ${{ variables.NUM_CTX_SERVERS }}
        policy: parallel
      script:
        - cat ${{ artifacts.PREFILL_CONFIG.path }}
        - >
          trtllm-llmapi-launch trtllm-serve ${LOCAL_MODEL_PATH}
          --host ${HEAD_NODE_IP}
          --port $((8536 + ${SFLOW_REPLICA_INDEX}))
          --extra_llm_api_options ${{ artifacts.PREFILL_CONFIG.path }}
      resources:
        gpus:
          count: ${{ variables.CTX_TP_SIZE }}
      probes:
        readiness:
          log_watch:
            regex_pattern: "Application startup complete"
          timeout: 600
        failure:
          log_watch:
            regex_pattern: "Traceback (most recent call last)"
      depends_on:
        - load_image

    - name: decode_server
      operator:
        name: trtllm_container
        ntasks: ${{ variables.GEN_TP_SIZE }}
        ntasks_per_node: ${{ [ variables.GEN_TP_SIZE, variables.GPUS_PER_NODE ] | min }}
      replicas:
        count: ${{ variables.NUM_GEN_SERVERS }}
        policy: parallel
      script:
        - cat ${{ artifacts.DECODE_CONFIG.path }}
        - >
          trtllm-llmapi-launch trtllm-serve ${LOCAL_MODEL_PATH}
          --host ${HEAD_NODE_IP}
          --port $((8336 + ${SFLOW_REPLICA_INDEX}))
          --extra_llm_api_options ${{ artifacts.DECODE_CONFIG.path }}
      resources:
        gpus:
          count: ${{ variables.GEN_TP_SIZE }}
      probes:
        readiness:
          log_watch:
            regex_pattern: "Application startup complete"
          timeout: 600
        failure:
          log_watch:
            regex_pattern: "Traceback (most recent call last)"
      depends_on:
        - load_image

    - name: benchmark
      operator:
        name: aiperf
        ntasks: 1
      replicas:
        variables:
          - CONCURRENCY  # Sweeps over domain [128, 256]
        policy: sequential
      script:
        - aiperf profile --concurrency ${CONCURRENCY} --url http://${HEAD_NODE_IP}:8000 ...
      # Sample the prefill/decode SERVERS' resources from this benchmark task
      # (used_by_tasks), not the benchmark client's own node, so the report
      # captures the servers' GPU/CPU usage over the benchmark window.
      monitor:
        resources:
          used_by_tasks:
            - prefill_server
            - decode_server
        report:
          enabled: true
      depends_on:
        - prefill_server
        - decode_server
        - frontend_server
```

**Run it:**

```bash
sflow sample self_contained/slurm/trtllm_serve_disagg

# Validate configuration
sflow run -f trtllm_serve_disagg.yaml \
  --set SLURM_ACCOUNT=your_account \
  --set SLURM_PARTITION=your_partition \
  --dry-run

# Submit to Slurm
sflow batch -f trtllm_serve_disagg.yaml \
  -A your_account -p your_partition -N 1 -G 4 \
  --sbatch-path trtllm_disagg_job.sh --submit
```

---

### InfMax Multi-Node Disaggregated Inference (DS-R1)

A production-ready multi-node disaggregated inference setup optimized for large models like DeepSeek-R1 using NVIDIA Dynamo and TensorRT-LLM.

**Features:**
- Multi-node deployment (default 3 nodes with 4 GPUs each)
- Disaggregated prefill/decode architecture with configurable parallelism
- NATS and etcd for service discovery
- Hardware monitoring driven from the benchmark task (declarative `monitor` with `used_by_tasks`)
- MoE (Mixture of Experts) optimization parameters
- Sequential benchmark sweeps with variable domains
- File-type artifacts for dynamic server configuration
- Failure probes for error detection

```yaml
version: "0.1"

variables:
  # Slurm Configuration
  SLURM_ACCOUNT:
    description: "SLURM account"
    value: your_account
  SLURM_PARTITION:
    description: "SLURM partition"
    value: your_partition
  SLURM_TIMELIMIT:
    description: "SLURM time limit"
    value: 120
  GPUS_PER_NODE:
    description: "GPUs per node"
    value: 4
  SLURM_NODES:
    description: "Number of nodes"
    value: 3

  # Model Configuration
  SERVED_MODEL_NAME:
    description: "Served model name"
    value: DS-R1

  # Prefill Server Configuration
  NUM_CTX_SERVERS:
    description: "Number of context/prefill servers"
    value: 1
  CTX_TP_SIZE:
    description: "Context tensor parallel size"
    value: 4
  CTX_BATCH_SIZE:
    description: "Context batch size"
    value: 1
  CTX_MAX_NUM_TOKENS:
    description: "Context max number of tokens"
    value: 8448

  # Decode Server Configuration
  NUM_GEN_SERVERS:
    description: "Number of generation/decode servers"
    value: 1
  GEN_TP_SIZE:
    description: "Generation tensor parallel size"
    value: 8
  GEN_BATCH_SIZE:
    description: "Generation batch size"
    value: 128

  # Benchmark Configuration with Domain Sweep
  CONCURRENCY:
    description: "Concurrency"
    value: 64
    domain: [32, 64]  # Will create sequential benchmark runs

  # Container Images
  DYNAMO_IMAGE:
    description: "Dynamo TRTLLM container image"
    value: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.0

artifacts:
  - name: LOCAL_MODEL_PATH
    uri: fs:///path/to/your/model
  - name: PREFILL_CONFIG
    uri: file://prefill_config.yaml
    content: |
      max_batch_size: ${{ variables.CTX_BATCH_SIZE }}
      tensor_parallel_size: ${{ variables.CTX_TP_SIZE }}
      moe_expert_parallel_size: ${{ variables.CTX_TP_SIZE }}
      # ... additional configuration
  - name: DECODE_CONFIG
    uri: file://decode_config.yaml
    content: |
      tensor_parallel_size: ${{ variables.GEN_TP_SIZE }}
      max_batch_size: ${{ variables.GEN_BATCH_SIZE }}
      # ... additional configuration

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
  - name: dynamo_trtllm
    type: srun
    container_image: ${{ variables.DYNAMO_IMAGE }}
    container_writable: true
    container_mount_home: false
    mpi: pmix

workflow:
  name: infmax
  timeout: 115m
  variables:
    HEAD_NODE_IP:
      value: "${{ backends.slurm_cluster.nodes[0].ip_address }}"
    ETCD_ENDPOINTS:
      value: "${{ backends.slurm_cluster.nodes[0].ip_address }}:2379"
    NATS_SERVER:
      value: "nats://${{ backends.slurm_cluster.nodes[0].ip_address }}:4222"

  tasks:
    - name: load_image
      operator:
        name: dynamo_trtllm
        ntasks: ${{ variables.SLURM_NODES }}
        ntasks_per_node: 1
      script:
        - echo "Image Loaded"
      probes:
        readiness:
          log_watch:
            regex_pattern: "Image Loaded"
          timeout: 1200

    - name: nats_server
      operator: dynamo_trtllm
      script:
        - nats-server -js
      resources:
        nodes:
          indices: [0]
      probes:
        readiness:
          tcp_port:
            port: 4222
      depends_on:
        - load_image

    - name: etcd_server
      operator: dynamo_trtllm
      script:
        - etcd --listen-client-urls "http://0.0.0.0:2379" ...
      resources:
        nodes:
          indices: [0]
      probes:
        readiness:
          tcp_port:
            port: 2379
      depends_on:
        - load_image

    - name: frontend_server
      operator: dynamo_trtllm
      script:
        - python3 -m dynamo.frontend --http-port 8000
      resources:
        nodes:
          indices: [0]
      probes:
        readiness:
          tcp_port:
            port: 8000
      depends_on:
        - nats_server
        - etcd_server

    - name: prefill_server
      operator:
        name: dynamo_trtllm
        ntasks: ${{ variables.CTX_TP_SIZE }}
        ntasks_per_node: ${{ [ variables.CTX_TP_SIZE, variables.GPUS_PER_NODE ] | min }}
      replicas:
        count: ${{ variables.NUM_CTX_SERVERS }}
        policy: parallel
      script:
        - trtllm-llmapi-launch python3 -m dynamo.trtllm --disaggregation-mode prefill ...
      resources:
        gpus:
          count: ${{ variables.CTX_TP_SIZE }}
      probes:
        readiness:
          log_watch:
            regex_pattern: "Setting PyTorch memory fraction"
        failure:
          log_watch:
            regex_pattern: "Traceback (most recent call last)"
      depends_on:
        - frontend_server

    - name: decode_server
      operator:
        name: dynamo_trtllm
        ntasks: ${{ variables.GEN_TP_SIZE }}
        ntasks_per_node: ${{ [ variables.GEN_TP_SIZE, variables.GPUS_PER_NODE ] | min }}
      replicas:
        count: ${{ variables.NUM_GEN_SERVERS }}
        policy: parallel
      script:
        - trtllm-llmapi-launch python3 -m dynamo.trtllm --disaggregation-mode decode ...
      resources:
        gpus:
          count: ${{ variables.GEN_TP_SIZE }}
      probes:
        readiness:
          log_watch:
            regex_pattern: "Setting PyTorch memory fraction"
        failure:
          log_watch:
            regex_pattern: "Traceback (most recent call last)"
      depends_on:
        - frontend_server

    - name: benchmark
      operator:
        name: aiperf
        ntasks: 1
      replicas:
        variables:
          - CONCURRENCY  # Sweeps over domain [32, 64]
        policy: sequential
      script:
        - aiperf profile --concurrency ${CONCURRENCY} ...
      # Sample the prefill/decode SERVERS' resources from this benchmark task
      # (used_by_tasks), not the benchmark client's own node, so the report
      # captures the servers' GPU/CPU usage over the benchmark window.
      monitor:
        resources:
          used_by_tasks:
            - prefill_server
            - decode_server
        report:
          enabled: true
      depends_on:
        - prefill_server
        - decode_server
        - frontend_server
```

**Run it:**

```bash
sflow sample self_contained/slurm/infmax_v1_ds_r1

# Validate configuration
sflow run -f infmax_v1_ds_r1.yaml \
  --set SLURM_ACCOUNT=your_account \
  --set SLURM_PARTITION=your_partition \
  --dry-run

# Submit to Slurm (multi-node)
sflow batch -f infmax_v1_ds_r1.yaml \
  -A your_account -p your_partition -N 3 -G 4 \
  --sbatch-path infmax_job.sh --submit
```

---

### More Slurm Samples

Additional single-file Slurm workflows (copy with `sflow sample self_contained/slurm/<name>`, then `sflow run`/`sflow batch`):

| Sample | Description |
|--------|-------------|
| `dynamo_sglang_agg` | Dynamo + SGLang **aggregated** serving (single server) with NATS/etcd/frontend and an AIPerf benchmark |
| `dynamo_sglang_disagg` | Dynamo + SGLang **disaggregated** prefill/decode serving |
| `dynamo_vllm_agg` | Dynamo + vLLM **aggregated** serving |
| `dynamo_vllm_disagg` | Dynamo + vLLM **disaggregated** prefill/decode serving |
| `dynamo_trtllm_agg` | Dynamo + TensorRT-LLM **aggregated** serving |
| `multi_backend` | A single workflow spanning more than one backend |
| `resource_release_after` | GPU `release_after` semantics for freeing resources between phases |
| `auto_replica` | Auto replica detection with per-replica node/GPU assignment |
| `aiperf_template` | Minimal single-task AIPerf benchmarking template |

```bash
sflow sample self_contained/slurm/dynamo_sglang_agg
sflow run -f dynamo_sglang_agg.yaml \
  --set SLURM_ACCOUNT=your_account --set SLURM_PARTITION=your_partition --dry-run
```

---

## Kubernetes Samples

These samples run on the **Kubernetes backend**, which reserves nodes and schedules each task as pod(s). Cluster selection and credentials are CLI flags (`--kubeconfig`, `--kube-context`, `--kube-namespace`, `--kube-node-selector`, …), so the recipe stays cluster-agnostic. See [Backends](./backends.md#kubernetes-backend).

### Hello World (Kubernetes)

A minimal pod task using a `k8s` operator (the workload image lives on the operator, since the backend has no image of its own).

```bash
sflow sample self_contained/kubernetes/hello_world
sflow run -f hello_world.yaml \
  --kube-namespace my-namespace --kube-node-selector tenant=my-pool
```

### Dynamo TRT-LLM Disaggregated (Kubernetes)

The Kubernetes counterpart of the Slurm Dynamo TRT-LLM disaggregated sample: NATS/etcd/frontend plus prefill/decode servers scheduled as GPU pods, benchmarked with AIPerf.

```bash
sflow sample self_contained/kubernetes/dynamo_trtllm_disagg
sflow run -f dynamo_trtllm_disagg.yaml \
  -a LOCAL_MODEL_PATH=fs:///mnt/model-cache/your-model \
  --kube-namespace my-namespace --kube-node-selector tenant=my-pool
```

---

## Key Concepts Demonstrated

| Sample | Concepts |
|--------|----------|
| `self_contained/local/hello_world` | Variables, basic task execution |
| `self_contained/local/dag` | Task dependencies, parallel execution, built-in env vars |
| `self_contained/local/variable_domain` | Variable `domain` sweeps (one run per value) |
| `self_contained/local/result_parsing` | Regex/JSON result parsing into `result.json` / `results.json` |
| `self_contained/local/storage_upload` | Per-task `uploads:` to a storage target (e.g. S3) |
| `self_contained/local/storage_upload_all` | `upload_all` to ship every task's outputs |
| `self_contained/local/monitor` | Workflow-level hardware monitor and charts |
| `self_contained/docker/hello_world` | Local Docker backend, `docker_run` operator |
| `self_contained/docker/multi_node` | Multi-host Docker pool, one container per host |
| `self_contained/docker/sglang_qwen3` | Local Docker backend, containerized LLM serving (SGLang + Qwen3-0.6B), readiness/failure probes, file:// client script |
| `self_contained/slurm/sglang_server_client` | Slurm backend, operators, probes, replicas, GPU resources |
| `self_contained/slurm/dynamo_trtllm_disagg` | Service discovery (NATS/etcd), retry policies, multi-process tasks |
| `self_contained/slurm/trtllm_serve_disagg` | Artifacts with backend IP resolution, failure probes, variable sweeps |
| `self_contained/slurm/infmax_v1_ds_r1` | Multi-node deployment, MoE optimization, GPU monitoring, file artifacts |
| `self_contained/slurm/auto_replica` | Auto replica detection, task context, node/GPU assignment |
| `self_contained/slurm/aiperf_template` | AIPerf benchmarking template, simple single-task workflow |
| `self_contained/kubernetes/hello_world` | Kubernetes backend, `k8s` operator, pod scheduling |
| `self_contained/kubernetes/dynamo_trtllm_disagg` | Disaggregated inference on Kubernetes GPU pods |

---

## Modular Samples (Folder-based)

Modular samples are folders containing multiple composable YAML files. Instead of one monolithic config, the workflow is split into reusable building blocks.

### Modular inference recipe (inference_x_v2)

A modular inference benchmark setup supporting multiple frameworks (SGLang, vLLM, TensorRT-LLM) with disaggregated prefill/decode servers.

**Structure:**

```
inference_x_v2/
├── slurm_config.yaml          # Slurm backend configuration
├── common_workflow.yaml       # Shared tasks (load_image, nats, etcd, frontend)
├── benchmark_aiperf.yaml      # AIPerf benchmark task
├── benchmark_infmax.yaml      # InfMax benchmark task
├── bulk_input.csv             # CSV for bulk batch jobs (disagg + agg rows)
├── sglang/
│   ├── prefill.yaml           # SGLang prefill server task (disaggregated)
│   ├── decode.yaml            # SGLang decode server task (disaggregated)
│   └── agg.yaml               # SGLang aggregated server task
├── vllm/
│   ├── prefill.yaml           # vLLM prefill server task (disaggregated)
│   ├── decode.yaml            # vLLM decode server task (disaggregated)
│   └── agg.yaml               # vLLM aggregated server task
├── trtllm/
│   ├── prefill.yaml           # TRT-LLM prefill server task (disaggregated)
│   ├── decode.yaml            # TRT-LLM decode server task (disaggregated)
│   └── agg.yaml               # TRT-LLM aggregated server task
└── composed_recipes/          # Pre-composed, ready-to-run single-file recipes
    ├── trtllm_agg_benchmark_aiperf_1n_007.yaml
    ├── trtllm_prefill_decode_benchmar_001.yaml
    ├── vllm_prefill_decode_benchmark_005.yaml
    ├── sglang_agg_benchmark_aiperf_2n_008.yaml
    └── ...                    # one worked example per framework / topology
```

The `composed_recipes/` folder holds fully-merged single-file recipes — the same
artifacts a `--bulk-input` sweep produces from the fragments above. Use them as
copy-paste starting points or to see exactly what deep-merge yields.

The `bulk_input.csv` supports both disaggregated and aggregated workflows using the `missable_tasks` column:
- Disagg rows include `prefill.yaml + decode.yaml` and set `missable_tasks=agg_server`
- Agg rows include `agg.yaml` and set `missable_tasks=prefill_server decode_server`

**Copy the modular sample:**

```bash
sflow sample modular/inference_x_v2
```

**Usage Option A: Bulk batch (CSV-driven)**

Each row in `bulk_input.csv` defines a job with its own config files and variable overrides:

```bash
# Preview (no submission)
sflow batch --bulk-input inference_x_v2/bulk_input.csv \
  -a LOCAL_MODEL_PATH=fs:///path/to/model -G 4 -A ACCOUNT -p PARTITION

# Submit all jobs
sflow batch --bulk-input inference_x_v2/bulk_input.csv \
  -a LOCAL_MODEL_PATH=fs:///path/to/model -G 4 -A ACCOUNT -p PARTITION --submit
```

**Usage Option B: Compose + Submit (step-by-step)**

```bash
# Step 1: Compose modular files into a complete config
sflow compose inference_x_v2/slurm_config.yaml \
              inference_x_v2/common_workflow.yaml \
              inference_x_v2/trtllm/prefill.yaml \
              inference_x_v2/trtllm/decode.yaml \
              inference_x_v2/benchmark_aiperf.yaml \
              -o composed.yaml

# Step 2: Validate, run, or submit
sflow run -f composed.yaml --dry-run                          # validate
sflow run -f composed.yaml --tui                               # run interactively
sflow batch -f composed.yaml -N 1 -G 4 -p PARTITION -A ACCOUNT \
            -o run.sh --submit                                 # submit to Slurm
```

**Computed variables:**

The modular samples use chained computed variables to simplify GPU/node calculations:

```yaml
variables:
  CTX_TP_SIZE:
    type: integer
    value: 2
  CTX_DP_SIZE:
    type: integer
    value: 1
  CTX_PP_SIZE:
    type: integer
    value: 1
  CTX_GPUS_PER_WORKER:
    type: integer
    value: ${{ variables.CTX_TP_SIZE * variables.CTX_DP_SIZE * variables.CTX_PP_SIZE }}
  CTX_NODES_PER_WORKER:
    type: integer
    value: ${{ [variables.CTX_GPUS_PER_WORKER // variables.GPUS_PER_NODE, 1] | max }}
```

---

## Tips

1. **Always validate first**: Use `--dry-run` before actual execution
2. **Override variables**: Use `--set KEY=VALUE` to customize configurations
3. **Override model path**: Use `--artifact LOCAL_MODEL_PATH=fs:///path/to/model` to point to your actual model
4. **Use `--resolve`**: Add `--resolve` to `sflow compose` or `sflow batch --bulk-input` to inline all variables into literal values for a fully-baked config
5. **Check sample source**: Samples are located in `src/sflow/samples/` in the sflow package
