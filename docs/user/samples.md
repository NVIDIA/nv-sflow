---
title: Sample Workflows
sidebar_position: 12
---

This page contains sample workflow configurations that you can use as starting points for your own workflows. You can also access these samples using the `sflow sample` command.

> 📁 **View original sample files**: [src/sflow/samples](../../src/sflow/samples)

## Listing Available Samples

```bash
# List all available samples
sflow sample --list

# Copy a sample to your current directory
sflow sample local_hello_world

# Copy with custom output path
sflow sample local_dag --output my_workflow.yaml
```

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
  name: hello_local
  tasks:
    - name: hello
      script:
        - echo "Hello ${WHO}"
```

**Run it:**

```bash
sflow sample local_hello_world
sflow run -f local_hello_world.yaml
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
sflow sample local_dag
sflow run -f local_dag.yaml --dry-run  # Validate
sflow run -f local_dag.yaml            # Execute
```

---

## Slurm Samples

These samples require a Slurm cluster with GPU resources.

### SGLang Server + Benchmark (Single Node)

Deploys an SGLang inference server with AIPerf benchmarking on Slurm.

**Features:**
- SGLang server with FP8 inference
- GPU monitoring
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

    - name: gpu_monitor
      operator: sglang_runtime
      script:
        - echo "Starting gpu monitor"
        - >
          nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,memory.total,memory.used 
          --format=csv,noheader,nounits -lms 2000 | 
          while IFS= read -r input || [ -n "$input" ] ; 
          do timestamp=$(date +%s%3N); 
          printf "%s.%s,%s\n" "${timestamp:0:10}" "${timestamp:10:3}" "${input}"; 
          done 
          >> ${SFLOW_TASK_OUTPUT_DIR}/gpu_monitor_node_${SLURM_NODEID}_${SLURMD_NODENAME}.log
      probes:
        readiness:
          log_watch:
            regex_pattern: "Starting gpu monitor"
      resources:
        nodes:
          indices: [0]
      depends_on:
        - load_image
        - install_aiperf
    
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
      depends_on:
        - sglang_server
        - install_aiperf
```

**Run it:**

```bash
sflow sample slurm_sglang_server_client

# Validate configuration
sflow run -f slurm_sglang_server_client.yaml \
  --set SLURM_ACCOUNT=your_account \
  --set SLURM_PARTITION=your_partition \
  --dry-run

# Submit to Slurm
sflow batch -f slurm_sglang_server_client.yaml \
  -A your_account -p your_partition -N 1 \
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
sflow sample slurm_dynamo_trtllm_disagg

# Validate configuration
sflow run -f slurm_dynamo_trtllm_disagg.yaml \
  --set SLURM_ACCOUNT=your_account \
  --set SLURM_PARTITION=your_partition \
  --dry-run

# Submit to Slurm
sflow batch -f slurm_dynamo_trtllm_disagg.yaml \
  -A your_account -p your_partition -N 1 \
  --sbatch-path dynamo_job.sh --submit
```

---

### TRT-LLM Serve Disaggregated Inference (Single Node)

Deploys a disaggregated inference setup with separate prefill and decode servers using TensorRT-LLM's native `trtllm-serve disaggregated` command.

**Features:**
- Disaggregated prefill/decode architecture with `trtllm-serve`
- Dynamic configuration using file-type artifacts with backend node IP resolution
- Configurable tensor parallelism for prefill and decode servers
- GPU monitoring task
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
      depends_on:
        - prefill_server
        - decode_server
        - frontend_server
```

**Run it:**

```bash
sflow sample slurm_trtllm_serve_disagg

# Validate configuration
sflow run -f slurm_trtllm_serve_disagg.yaml \
  --set SLURM_ACCOUNT=your_account \
  --set SLURM_PARTITION=your_partition \
  --dry-run

# Submit to Slurm
sflow batch -f slurm_trtllm_serve_disagg.yaml \
  -A your_account -p your_partition -N 1 \
  --sbatch-path trtllm_disagg_job.sh --submit
```

---

### InfMax Multi-Node Disaggregated Inference (DS-R1)

A production-ready multi-node disaggregated inference setup optimized for large models like DeepSeek-R1 using NVIDIA Dynamo and TensorRT-LLM.

**Features:**
- Multi-node deployment (default 3 nodes with 4 GPUs each)
- Disaggregated prefill/decode architecture with configurable parallelism
- NATS and etcd for service discovery
- GPU monitoring across all nodes
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

    - name: gpu_monitor
      operator:
        name: dynamo_trtllm
        ntasks_per_node: 1
      resources:
        nodes:
          count: ${{ variables.SLURM_NODES }}
      script:
        - nvidia-smi monitoring...
      depends_on:
        - load_image

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
      depends_on:
        - prefill_server
        - decode_server
        - frontend_server
```

**Run it:**

```bash
sflow sample slurm_infmax_v1_ds_r1

# Validate configuration
sflow run -f slurm_infmax_v1_ds_r1.yaml \
  --set SLURM_ACCOUNT=your_account \
  --set SLURM_PARTITION=your_partition \
  --dry-run

# Submit to Slurm (multi-node)
sflow batch -f slurm_infmax_v1_ds_r1.yaml \
  -A your_account -p your_partition -N 3 \
  --sbatch-path infmax_job.sh --submit
```

---

## Key Concepts Demonstrated

| Sample | Concepts |
|--------|----------|
| `local_hello_world` | Variables, basic task execution |
| `local_dag` | Task dependencies, parallel execution, built-in env vars |
| `slurm_sglang_server_client` | Slurm backend, operators, probes, replicas, GPU resources |
| `slurm_dynamo_trtllm_disagg` | Service discovery (NATS/etcd), retry policies, multi-process tasks |
| `slurm_trtllm_serve_disagg` | Artifacts with backend IP resolution, failure probes, variable sweeps |
| `slurm_infmax_v1_ds_r1` | Multi-node deployment, MoE optimization, GPU monitoring, file artifacts |
| `slurm_auto_replica` | Auto replica detection, task context, node/GPU assignment |
| `slurm_aiperf_template` | AIPerf benchmarking template, simple single-task workflow |

## Tips

1. **Always validate first**: Use `--dry-run` before actual execution
2. **Override variables**: Use `--set KEY=VALUE` to customize configurations
3. **Check sample source**: Samples are located in `src/sflow/samples/` in the sflow package
