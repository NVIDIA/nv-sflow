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
- **`modular/<set>`** — folders of composable fragments (`sflow compose` / `sflow batch --bulk-input`). See [Modular Samples](#modular-samples-folder-based).

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
| docker | `self_contained/docker/multi_node` | Two-host torchrun rendezvous smoke test: one container + one torchrun process per Docker host, proving node-index / leader-address wiring |
| docker | `self_contained/docker/sglang_qwen3` | SGLang Qwen3-0.6B server + stdlib client + AIPerf, all in local containers |
| docker | `self_contained/docker/gpu_monitor` | GPU monitor with a log-marker report `window:` — runs a stock CUDA `nbody` container as a deliberate idle/burst/idle square wave, so the lifecycle report and the windowed report visibly differ |
| docker | `self_contained/docker/sglang_glm52` | GLM-5.2 (NVFP4) on SGLang, one remote Docker host: a TP8 server plus the InferenceX AgentX benchmark as a reference run invoked it |
| docker | `self_contained/docker/sglang_disagg_glm53` | GLM-5.3-Flash disaggregated prefill/decode on a Docker host pool — pure SGLang (no Dynamo, no NATS/etcd), SGLang's built-in `disagg_router`, prefill and decode on different machines |
| docker | `self_contained/docker/vllm_disagg_glm53` | GLM-5.3-Flash (NVFP4) disaggregated prefill/decode on vLLM — structural twin of `sglang_disagg_glm53` (same hosts, DAG, monitor, aiperf; only image and server commands differ) |
| docker | `self_contained/docker/dynamo_vllm_agg` | One aggregated vLLM worker at TP16 spanning two Docker hosts (a single process group across machines) |
| docker | `self_contained/docker/dynamo_vllm_disagg` | Cross-node prefill/decode disaggregation on a Docker host pool — prefill on one machine, decode on another, so the KV transfer over NIXL is a real network hop |
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
| slurm | `self_contained/slurm/dynamo_sglang_agg` | Dynamo + SGLang aggregated serving |
| slurm | `self_contained/slurm/dynamo_vllm_agg` | Dynamo + vLLM aggregated serving |
| slurm | `self_contained/slurm/dynamo_trtllm_agg` | Dynamo + TensorRT-LLM aggregated serving |
| slurm | `self_contained/slurm/dynamo_sglang_disagg` | Dynamo + SGLang prefill/decode disaggregated |
| slurm | `self_contained/slurm/dynamo_vllm_disagg` | Dynamo + vLLM prefill/decode disaggregated |
| slurm | `self_contained/slurm/dynamo_trtllm_disagg` | Dynamo + TensorRT-LLM prefill/decode disaggregated |
| kubernetes | `self_contained/kubernetes/hello_world` | Single pod via the `k8s` operator |
| kubernetes | `self_contained/kubernetes/multinode` | Multi-node distributed training: one pod per reserved node (leader = index 0) via device-plugin GPU scheduling |
| kubernetes | `self_contained/kubernetes/dynamo_sglang_agg` | Kubernetes port of the Slurm `dynamo_sglang_agg`: a single aggregated SGLang worker serves prefill and decode |
| kubernetes | `self_contained/kubernetes/dynamo_sglang_disagg` | Kubernetes port of the Slurm `dynamo_sglang_disagg`: prefill/decode workers, KV cache transferred over NIXL by default |
| kubernetes | `self_contained/kubernetes/dynamo_sglang_disagg_dsr1` | Same recipe with the KV transfer defaulting to mooncake over NVLink/MNNVL (NVL72 racks) |
| kubernetes | `self_contained/kubernetes/dynamo_trtllm_agg` | Aggregated TRT-LLM + Dynamo; the worker runs on `k8s_mpi` so one recipe covers single-node and cross-node aggregation as one MPI world |
| kubernetes | `self_contained/kubernetes/dynamo_trtllm_disagg` | Kubernetes port of the Slurm `dynamo_trtllm_disagg`: prefill/decode servers on `k8s_mpi`, device-plugin GPUs, node reservation |
| kubernetes | `self_contained/kubernetes/dynamo_vllm_agg` | Aggregated companion of `dynamo_vllm_disagg`: one TP16 worker spanning 2 nodes as a Ray cluster |
| kubernetes | `self_contained/kubernetes/dynamo_vllm_disagg` | Kubernetes port of the Slurm `dynamo_vllm_disagg`; model and JIT cache are `fs://` artifacts on PVCs, overridable at runtime |
| kubernetes | `self_contained/kubernetes/trtllm_serve_agg` | Aggregated bare `trtllm-serve` (no NATS/etcd/frontend) on `k8s_mpi` |
| kubernetes | `self_contained/kubernetes/trtllm_serve_disagg` | Disaggregated bare `trtllm-serve`: separate prefill and decode servers plus a `trtllm-serve disaggregated` router, no Dynamo infra |
| kubernetes | `self_contained/kubernetes/qwen3_8b/dgx_dynamo_trtllm_agg` | Qwen3-8B FP8, Dynamo + TRT-LLM aggregated on a DGX-style 8-GPU node (TP=8 fits one node; no IMEX) |
| kubernetes | `self_contained/kubernetes/qwen3_8b/dgx_dynamo_trtllm_disagg` | Qwen3-8B FP8, Dynamo + TRT-LLM disaggregated on DGX nodes (one 8-GPU node per worker; KV over UCX/InfiniBand) |
| kubernetes | `self_contained/kubernetes/qwen3_8b/dgx_trtllm_serve` | Qwen3-8B FP8, bare `trtllm-serve` on a DGX-style 8-GPU node |
| kubernetes | `self_contained/kubernetes/qwen3_8b/gb_dynamo_trtllm_agg` | Qwen3-8B FP8, Dynamo + TRT-LLM aggregated on a GB NVL72 rack (TP=8 spans two 4-GPU nodes over MNNVL via an IMEX `compute_domain`) |
| kubernetes | `self_contained/kubernetes/qwen3_8b/gb_dynamo_trtllm_disagg` | Qwen3-8B FP8, Dynamo + TRT-LLM disaggregated on a GB NVL72 rack (each worker spans two 4-GPU nodes; KV over MNNVL) |
| kubernetes | `self_contained/kubernetes/qwen3_8b/gb_trtllm_serve` | Qwen3-8B FP8, bare `trtllm-serve` on a GB NVL72 rack (TP=8 across two 4-GPU nodes via IMEX) |
| modular | `modular/backend_agnostic` | Workload × backend composition: the same workload YAML runs on Slurm, Kubernetes, Kubernetes MPI or Docker via scattered merge + `required_by` |
| modular | `modular/inference_x_v2` | Framework fragments + benchmarks + a CSV sweep (see [Modular inference recipe](#modular-inference-recipe-inference_x_v2)) |
| modular | `modular/utils` | `set_up_dynamo.yaml` — idempotent Dynamo bootstrap fragment (`ai-dynamo` libs, `nats-server`, `etcd`) that folds itself into any Dynamo workflow via `required_by` |

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

# This example demonstrates the consolidated `result` task entry.
#
# It covers two common ways to produce small structured task results:
#   1. Parse metrics from the task log with regex patterns.
#   2. Write JSON directly to $SFLOW_TASK_RESULT_FILE and let sflow normalize it.
#
# Run it locally:
#   sflow run examples/self_contained/local/result_parsing.yaml
#
# Key output files:
#   <run_dir>/benchmark_log/result.json
#   <run_dir>/benchmark_file/result.json
#   <run_dir>/results.json

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
        # Block scalar keeps `: ` in echoed benchmark lines from being parsed
        # as YAML mapping syntax.
        - |
          echo "warming up"
          echo "TTFT: 40.0 ms"
          echo "TTFT: 42.5 ms"
          echo "tok/s: 123.0"
          echo "p99 latency: 88 ms"
      result:
        # Simple regex map: each regex is matched against the merged task log.
        # The first capture group becomes the result value; the default
        # aggregate is `last`, so ttft resolves to 42.5.
        ttft: 'TTFT:\s*([0-9.]+)\s*ms'
        tps: 'tok/s:\s*([0-9.]+)'
        latency_p99: 'p99 latency:\s*([0-9.]+)\s*ms'

    - name: benchmark_file
      script:
        # Tasks that already produce structured output can write a plain JSON
        # object directly to $SFLOW_TASK_RESULT_FILE.
        - |
          echo '{"throughput": 999.5, "errors": 0}' > "$SFLOW_TASK_RESULT_FILE"
      result:
        file: result.json

    - name: verify
      depends_on: [benchmark_log, benchmark_file]
      script:
        - |
          set -euo pipefail

          log_result="$SFLOW_WORKFLOW_OUTPUT_DIR/benchmark_log/result.json"
          file_result="$SFLOW_WORKFLOW_OUTPUT_DIR/benchmark_file/result.json"
          index="$SFLOW_WORKFLOW_RESULT_FILE"

          test -f "$log_result"
          test -f "$file_result"
          test -f "$index"

          grep -q '"schema_version": "sflow.result.v1"' "$log_result"
          grep -q '"ttft": 42.5' "$log_result"
          grep -q '"tps": 123.0' "$log_result"
          grep -q '"latency_p99": 88' "$log_result"

          grep -q '"schema_version": "sflow.result.v1"' "$file_result"
          grep -q '"throughput": 999.5' "$file_result"
          grep -q '"errors": 0' "$file_result"

          grep -q '"schema_version": "sflow.results.v1"' "$index"
          grep -q '"benchmark_log"' "$index"
          grep -q '"benchmark_file"' "$index"

          printf 'RESULT_PARSING_SAMPLE_PASS\n' > "$SFLOW_TASK_OUTPUT_DIR/verify.txt"
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

> S3 uploads need the S3 extra: `pip install 'nv-sflow[s3]'` (credentials come from the boto3 default chain).

---

### Workflow Monitor

Attaches a hardware-utilization [monitor](./monitor.md) at the workflow level, sampling GPU/CPU usage during the run and (optionally) rendering charts.

```bash
sflow sample self_contained/local/monitor
sflow run -f monitor.yaml
```

> PNG charts need the monitor extra: `pip install 'nv-sflow[monitor]'`.

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
- Model path as an `fs://` artifact (`LOCAL_MODEL_PATH`), overridden with `-a` at run time

Full file (`sflow sample self_contained/slurm/sglang_server_client`):

```yaml
version: "0.1"

variables:
  # Variables has 2 ways of access, and can be overridden by --set flag in sflow batch / sflow run command
  # 1. In YAML expressions: `${{ variables.NAME }}`, sflow will automatically resolve the expression and inject the value before the workflow is executed, it works across any fields in the yaml file, such as backends, operators, workflow, tasks, scripts, etc.
  # 2. In task scripts (as env var): `${NAME}`, the variable can also be accessed as an environment variable in the task script
  
  # Slurm Configuration - These will be skipped if you use 'sflow batch -f sflow.yaml --submit' to launch the workflow and overwrite the variables in cli args
  SLURM_ACCOUNT:
    description: "SLURM account"
    value: your_slurm_account
  SLURM_PARTITION:
    description: "SLURM partition"
    value: your_slurm_partition
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
    value: 16
  
  # Container Images
  SGLANG_IMAGE:
    description: "SGLang image"
    value: "lmsysorg/sglang:v0.5.7-cu130-runtime"
  AIPERF_IMAGE:
    description: "AIPerf container image"
    value: python:3.12-slim

# If you want to use a local model path, you can define a fs:// type artifact and reference it in the task scripts as `${{ artifacts.NAME.path }}`
artifacts:
  - name: LOCAL_MODEL_PATH
    description: "Qwen3-0.6B-FP8 model path"
    uri: fs:///path/to/your/model 

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    time: ${{ variables.SLURM_TIMELIMIT }}
    nodes: ${{ variables.SLURM_NODES }}
    partition: ${{ variables.SLURM_PARTITION }}
    account: ${{ variables.SLURM_ACCOUNT }}
    gpus_per_node: ${{ variables.GPUS_PER_NODE }}
    # In some cluster like Corewaeve / OCI, you may need to add --gpus-per-node=N directive to the sbatch / salloc command
    # For GB200 / GB300 clusters, you may need to add --segment=N directive to make sure allocated nodes are in same network domain
    # extra_args:
    #   - "--gpus-per-node=${{ variables.GPUS_PER_NODE }}"
    #   - "--segment=${{ variables.SLURM_NODES }}"

operators:
  - name: sglang_runtime
    type: srun
    container_name: sglang_runtime
    container_writable: true
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
        ntasks: ${{ variables.SLURM_NODES }}
        ntasks_per_node: 1
      script:
        - echo "Image Loaded"
        - sleep 3600
      probes:
        readiness:
          log_watch:
            match_pattern: "Image Loaded"
            match_count: ${{ variables.SLURM_NODES }}
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
          --served-model-name ${{ variables.SERVED_MODEL_NAME }}
          --tensor-parallel-size ${{ variables.TP_SIZE }}
          --trust-remote-code
          --max-running-requests ${{ variables.MAX_RUNNING_REQUESTS }}
      probes:
        readiness:
          log_watch:
            match_pattern: "Workflow completed"
      depends_on:
        - load_image

    - name: benchmark
      operator:
        name: aiperf
        ntasks: 1
      script:
        - pip install aiperf==0.3.0
        - set -x
        - export COLUMNS=200
        - >
          aiperf profile --artifact-dir ${SFLOW_WORKFLOW_OUTPUT_DIR}/aiperf_concurrency_${CONCURRENCY}
          --model ${{ variables.SERVED_MODEL_NAME }}
          --tokenizer ${{ artifacts.LOCAL_MODEL_PATH.path }}
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
          --extra-inputs "repetition_penalty:1.0"
          --extra-inputs "temperature: 0.0"
          --concurrency ${CONCURRENCY}
          --request-count $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --warmup-request-count ${CONCURRENCY}
          --num-dataset-entries $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --random-seed 100
          -H 'Authorization: Bearer NOT USED'
          -H 'Accept: text/event-stream'
          --record-processors 8
          --ui simple
        - echo "Benchmarking finished"
      resources:
        nodes:
          indices: [0]
      # Trigger a hardware monitor from this benchmark task but sample the
      # server's resources (used_by_tasks), not the benchmark client's own node
      # -- so the report captures the server's GPU/CPU usage over the benchmark
      # window.
      monitor:
        resources:
          used_by_tasks:
            - sglang_server
        report:
          enabled: true
      depends_on:
        - sglang_server
```

**Run it:**

```bash
sflow sample self_contained/slurm/sglang_server_client

# Validate configuration
sflow run -f sglang_server_client.yaml \
  --set SLURM_ACCOUNT=ACCOUNT \
  --set SLURM_PARTITION=PARTITION \
  --dry-run

# Submit to Slurm
sflow batch -f sglang_server_client.yaml \
  -a LOCAL_MODEL_PATH=fs:///path/to/model \
  -A ACCOUNT -p PARTITION -N 1 -G 4 \
  --sbatch-path sglang_job.sh --submit
```

---

### Dynamo TRT-LLM Disaggregated Inference (Single Node)

Deploys a disaggregated inference setup with separate prefill and decode servers using NVIDIA Dynamo and TensorRT-LLM.

**Features:**
- Disaggregated prefill/decode architecture
- NATS and etcd for service discovery
- Configurable tensor parallelism (`*_GPUS_PER_WORKER` computed from TP × DP × PP sizes the srun `ntasks`/GPU count)
- Sequential benchmark sweeps with variable domains
- HTTP readiness probes against the frontend, log-watch failure probes
- File-type artifacts for dynamic configuration

Excerpt — the `decode_server` task mirrors `prefill_server`; see `sflow sample self_contained/slurm/dynamo_trtllm_disagg` for the full file:

```yaml
  CONCURRENCY:
    description: "Concurrency"
    value: 50
    domain: [50, 100]
  
  # Container Images
  AIPERF_IMAGE:
    description: "AIPerf container image"
    value: python:3.12-slim
  DYNAMO_IMAGE:
    description: "Dynamo TRTLLM container image"
    value: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0

# ... (12 lines omitted)

artifacts:
  # file type artifacts will be generated by sflow with the content defined in this yaml file, accessible as `${{ artifacts.NAME.path }}` in task scripts
  - name: LOCAL_MODEL_PATH
    uri: fs:///path/to/your/model

# ... (59 lines omitted)

operators:
  - name: dynamo
    type: srun
    container_image: ${{ variables.DYNAMO_IMAGE }}
    container_writable: true
    mpi: pmix
  - name: aiperf
    type: srun
    container_image: ${{ variables.AIPERF_IMAGE }}
    container_writable: true
    mpi: pmix

# ... (72 lines omitted)

    - name: frontend_server
      operator: dynamo
      script:
        - python3 -m dynamo.frontend --http-port 8000 ${{ variables.EXTRA_FRONTEND_ARGS }}
      resources:
        nodes:
          indices: [0]
      probes:
        readiness:
          tcp_port:
            port: 8000
          timeout: 300
          interval: 5
      depends_on:
        - nats_server
        - etcd_server

    - name: prefill_server
      operator:
        name: dynamo
        ntasks: ${{ variables.CTX_GPUS_PER_WORKER }}
        ntasks_per_node: ${{ [ variables.CTX_GPUS_PER_WORKER, variables.GPUS_PER_NODE ] | min }}
      replicas:
        count: ${{ variables.NUM_CTX_SERVERS }}
        policy: "parallel"
      script:
        - set -x
        - echo ${CUDA_VISIBLE_DEVICES}
        - export TLLM_LOG_LEVEL=INFO
        - export TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1
        - export TRTLLM_MOE_USE_LOW_PRECISION_COMBINE=1
        - env | grep TLLM
        - env | grep UCX
        - unset UCX_TLS
        - >
          trtllm-llmapi-launch python3 -m dynamo.trtllm
          --model-path ${{ artifacts.LOCAL_MODEL_PATH.path }}
          --served-model-name ${SERVED_MODEL_NAME}
          --disaggregation-mode prefill
          --extra-engine-args ${{ artifacts.PREFILL_CONFIG.path }} ${EXTRA_PREFILL_ARGS}
      resources:
        gpus:
          count: ${{ variables.CTX_GPUS_PER_WORKER }}
      depends_on:
        - frontend_server
      probes:
        readiness:
          http_post:
            url: "http://${{ backends.slurm_cluster.nodes[0].ip_address }}:8000/v1/chat/completions"
            headers:
              Content-Type: "application/json"
            body: '{"model": "${{ variables.SERVED_MODEL_NAME }}", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}'
          timeout: 1200
          interval: 20
        failure:
          log_watch:
            match_pattern: "Traceback (most recent call last)"
            match_count: 5
          interval: 30

# ... (44 lines omitted)

    - name: benchmark
      operator:
        name: aiperf
        ntasks: 1
      script:
        - pip install aiperf==0.3.0
        - set -x
        - export COLUMNS=200
        - >
          aiperf profile --artifact-dir ${SFLOW_WORKFLOW_OUTPUT_DIR}/aiperf_concurrency_${CONCURRENCY}
          --model ${{ variables.SERVED_MODEL_NAME }}
          --tokenizer ${{ artifacts.LOCAL_MODEL_PATH.path }}
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
          --extra-inputs "repetition_penalty:1.0"
          --extra-inputs "temperature: 0.0"
          --concurrency ${CONCURRENCY}
          --request-count $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --warmup-request-count ${CONCURRENCY}
          --num-dataset-entries $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --random-seed 100
          -H 'Authorization: Bearer NOT USED'
          -H 'Accept: text/event-stream'
          --record-processors 8
          --ui simple
        - echo "Benchmarking finished"
      resources:
        nodes:
          indices: [0]
      replicas:
        variables:
          - CONCURRENCY
        policy: sequential
      # Trigger a hardware monitor from this benchmark task but sample the
      # prefill/decode SERVERS' resources (used_by_tasks), not the benchmark
      # client's own node -- so the report captures the servers' GPU/CPU usage
      # over the benchmark window.
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
sflow sample self_contained/slurm/dynamo_trtllm_disagg

# Validate configuration
sflow run -f dynamo_trtllm_disagg.yaml \
  --set SLURM_ACCOUNT=ACCOUNT \
  --set SLURM_PARTITION=PARTITION \
  --dry-run

# Submit to Slurm
sflow batch -f dynamo_trtllm_disagg.yaml \
  -a LOCAL_MODEL_PATH=fs:///path/to/model \
  -A ACCOUNT -p PARTITION -N 1 -G 4 \
  --sbatch-path dynamo_job.sh --submit
```

---

### TRT-LLM Serve Disaggregated Inference (Single Node)

Deploys a disaggregated inference setup with separate prefill and decode servers using TensorRT-LLM's native `trtllm-serve disaggregated` command.

**Features:**
- Disaggregated prefill/decode architecture with `trtllm-serve`
- Dynamic configuration using file-type artifacts: the router config resolves the backend node IP, and each worker appends its own `host:port` to a placeholder URL list
- Configurable tensor parallelism for prefill and decode servers
- Hardware monitoring driven from the benchmark task (declarative `monitor` with `used_by_tasks`)
- Sequential benchmark sweeps with variable domains
- Failure probes for error detection

Excerpt — the `decode_server` task mirrors `prefill_server`; see `sflow sample self_contained/slurm/trtllm_serve_disagg` for the full file:

```yaml
  CONCURRENCY:
    description: "Concurrency"
    value: 20
    domain: [20,40]
  
  # Container Images
  AIPERF_IMAGE:
    description: "AIPerf container image"
    value: python:3.12-slim
  TRTLLM_IMAGE:
    description: "Dynamo TRTLLM container image"
    # value: nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc1 # 19 GB
    value: nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post2 # 15 GB

# ... (12 lines omitted)

artifacts:
  # file type artifacts will be generated by sflow with the content defined in this yaml file, accessible as `${{ artifacts.NAME.path }}` in task scripts
  - name: LOCAL_MODEL_PATH
    uri: fs:///path/to/your/model
  - name: PREFILL_URLS
    # Place holder file for prefill URLs, wil be injected later by each worker
    uri: file://prefill_urls.yaml
    content: |
      
      context_servers:
        num_instances: ${{ variables.NUM_CTX_SERVERS }}
        urls:
  - name: DECODE_URLS
    # Place holder file for decode URLs, wil be injected later by each worker
    uri: file://decode_urls.yaml
    content: |
      
      generation_servers:
        num_instances: ${{ variables.NUM_GEN_SERVERS }}
        urls:
  - name: SERVER_CONFIG
    uri: file://server_config.yaml
    content: |
      hostname: ${{ backends.slurm_cluster.nodes[0].ip_address }}
      port: 8000
      backend: pytorch

# ... (63 lines omitted)

operators:
  - name: trtllm_container
    type: srun
    container_image: ${{ variables.TRTLLM_IMAGE }}
    container_writable: true
    mpi: pmix
  - name: aiperf
    type: srun
    container_name: aiperf
    container_writable: true
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.AIPERF_IMAGE }}

# ... (29 lines omitted)

    - name: frontend_server
      operator: trtllm_container
      script:
        - set -x
        - export TRTLLM_SERVER_DISABLE_GC=1
        - export TRTLLM_WORKER_DISABLE_GC=1
        - cat ${{ artifacts.PREFILL_URLS.path }} >> ${{ artifacts.SERVER_CONFIG.path }}
        - cat ${{ artifacts.DECODE_URLS.path }} >> ${{ artifacts.SERVER_CONFIG.path }}
        - cat ${{ artifacts.SERVER_CONFIG.path }}
        - trtllm-serve disaggregated -c ${{ artifacts.SERVER_CONFIG.path }} -t 7200 -r 7200
      resources:
        nodes:
          indices: [0]
      probes:
        readiness:
          log_watch:
            match_pattern: "Application startup complete"
          timeout: 300
          interval: 2
      depends_on:
        - prefill_server
        - decode_server

    - name: prefill_server
      operator:
        name: trtllm_container
        ntasks: ${{ variables.CTX_GPUS_PER_WORKER }}
        ntasks_per_node: ${{ [ variables.CTX_GPUS_PER_WORKER, variables.GPUS_PER_NODE ] | min }}
      replicas:
        count: ${{ variables.NUM_CTX_SERVERS }}
        policy: "parallel"
      script:
        - set -x
        - echo ${CUDA_VISIBLE_DEVICES}
        # GPU placement probe. sflow plans a per-task slice and re-applies it
        # inside the step; this prints what the container ACTUALLY has next to
        # what was planned, which is the only way to tell the two failure modes
        # apart: if the planned ordinals are absent from `nvidia-smi -L`, the
        # slice never reached the container (carved and renumbered by the
        # runtime); if they are present, placement worked and the launcher chose
        # the wrong device. Cheap, and it makes every future triage one grep.
        - echo "SFLOW_GPU_PROBE task=${SLURM_JOB_NAME:-?} procid=${SLURM_PROCID:-?} localid=${SLURM_LOCALID:-?} planned_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
        - nvidia-smi -L 2>&1 | sed "s/^/SFLOW_GPU_PROBE visible: /" || true
        - export TLLM_LOG_LEVEL=INFO
        - export TRTLLM_SERVER_DISABLE_GC=1
        - export TRTLLM_WORKER_DISABLE_GC=1
        - export NCCL_GRAPH_MIXING_SUPPORT=0
        - export TRTLLM_ENABLE_PDL=1
        - env | grep TLLM
        - env | grep UCX
        - unset UCX_TLS
        - cat ${{ artifacts.PREFILL_CONFIG.path }}
        - export FIRST_CUDA_DEVICE=$(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f1)
        - export CTX_PORT=$((${{ variables.CTX_START_PORT }} + ${FIRST_CUDA_DEVICE}))
        - export FIRST_NODE_IP=$(echo ${SFLOW_TASK_ASSIGNED_NODE_IPS} | cut -d',' -f1)
        - if [ "${SLURM_PROCID:-0}" = "0" ]; then echo -e "\n    - ${FIRST_NODE_IP}:${CTX_PORT}" >> ${{ artifacts.PREFILL_URLS.path }}; fi
        - >
          trtllm-llmapi-launch trtllm-serve ${{ artifacts.LOCAL_MODEL_PATH.path }} 
          --host ${FIRST_NODE_IP} 
          --port ${CTX_PORT} 
          --extra_llm_api_options ${{ artifacts.PREFILL_CONFIG.path }} ${EXTRA_PREFILL_ARGS}
      resources:
        gpus:
          count: ${{ variables.CTX_GPUS_PER_WORKER }}
      depends_on:
        - load_image
      probes:
        readiness:
          log_watch:
            match_pattern: "Application startup complete"
          timeout: 1200
          interval: 20
        failure:
          log_watch:
            match_pattern: "Traceback (most recent call last)"
            match_count: 5
          interval: 30

# ... (56 lines omitted)

    - name: benchmark
      operator:
        name: aiperf
        ntasks: 1
      script:
        - pip install aiperf==0.3.0
        - set -x
        - export COLUMNS=200
        - >
          aiperf profile --artifact-dir ${SFLOW_WORKFLOW_OUTPUT_DIR}/aiperf_concurrency_${CONCURRENCY}
          --model ${{ variables.SERVED_MODEL_NAME }}
          --tokenizer ${{ artifacts.LOCAL_MODEL_PATH.path }}
          --endpoint-type chat
          --endpoint /v1/chat/completions
          --streaming
          --url http://${{ variables.HEAD_NODE_IP }}:8000
          --synthetic-input-tokens-mean ${{ variables.ISL }}
          --synthetic-input-tokens-stddev 0
          --output-tokens-mean ${{ variables.OSL }}
          --output-tokens-stddev 0
          --concurrency ${CONCURRENCY}
          --request-count $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --warmup-request-count ${CONCURRENCY}
          --num-dataset-entries $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --random-seed 100
          -H 'Authorization: Bearer NOT USED'
          -H 'Accept: text/event-stream'
          --record-processors 8
          --ui simple
        - echo "Benchmarking finished"
      resources:
        nodes:
          indices: [0]
      replicas:
        variables:
          - CONCURRENCY
        policy: sequential
      # Trigger a hardware monitor from this benchmark task but sample the
      # prefill/decode SERVERS' resources (used_by_tasks), not the benchmark
      # client's own node -- so the report captures the servers' GPU/CPU usage
      # over the benchmark window.
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
  --set SLURM_ACCOUNT=ACCOUNT \
  --set SLURM_PARTITION=PARTITION \
  --dry-run

# Submit to Slurm
sflow batch -f trtllm_serve_disagg.yaml \
  -a LOCAL_MODEL_PATH=fs:///path/to/model \
  -A ACCOUNT -p PARTITION -N 1 -G 4 \
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

Excerpt — same task DAG as the Dynamo TRT-LLM sample above (`load_image`, `nats_server`, `etcd_server`, `frontend_server`, `prefill_server`, `decode_server`, `benchmark`); see `sflow sample self_contained/slurm/infmax_v1_ds_r1` for the full file:

```yaml
  GPUS_PER_NODE:
    description: "GPUs per node"
    type: integer
    value: 4
  SLURM_NODES:
    description: "Number of nodes"
    value: 3

  # Model Configuration
  SERVED_MODEL_NAME:
    description: "Served model name"
    value: DS-R1

# ... (102 lines omitted)

  CONCURRENCY:
    description: "Concurrency"
    value: 64
    domain: [32, 64]
  
  # Container Images
  AIPERF_IMAGE:
    description: "AIPerf container image"
    value: python:3.12-slim
  DYNAMO_IMAGE:
    description: "Dynamo TRTLLM container image"
    value: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0

# ... (1 lines omitted)

artifacts:
  # file type artifacts will be generated by sflow with the content defined in this yaml file, accessible as `${{ artifacts.NAME.path }}` in task scripts
  - name: LOCAL_MODEL_PATH
    uri: fs:///path/to/your/model

# ... (60 lines omitted)

operators:
  - name: dynamo
    type: srun
    container_image: ${{ variables.DYNAMO_IMAGE }}
    container_writable: true
    mpi: pmix
    # extra_args:
    #   - --container-image=${{ variables.DYNAMO_IMAGE }}
    #   - --container-writable
  - name: aiperf
    type: srun
    container_name: aiperf
    container_writable: true
    mpi: pmix
    extra_args:
      - --container-image=${{ variables.AIPERF_IMAGE }}

# ... (88 lines omitted)

    - name: prefill_server
      operator:
        name: dynamo
        ntasks: ${{ variables.CTX_GPUS_PER_WORKER }}
        ntasks_per_node: ${{ [ variables.CTX_GPUS_PER_WORKER, variables.GPUS_PER_NODE ] | min }}
      replicas:
        count: ${{ variables.NUM_CTX_SERVERS }}
        policy: "parallel"
      script:
        - set -x
        - echo ${CUDA_VISIBLE_DEVICES}
        - export TLLM_LOG_LEVEL=INFO
        - export TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1
        - export TRTLLM_MOE_USE_LOW_PRECISION_COMBINE=1
        - env | grep TLLM
        - env | grep UCX
        - unset UCX_TLS
        - >
          trtllm-llmapi-launch python3 -m dynamo.trtllm
          --model-path ${{ artifacts.LOCAL_MODEL_PATH.path }}
          --served-model-name ${SERVED_MODEL_NAME}
          --disaggregation-mode prefill
          --extra-engine-args ${{ artifacts.PREFILL_CONFIG.path }} ${EXTRA_PREFILL_ARGS}
      resources:
        gpus:
          count: ${{ variables.CTX_GPUS_PER_WORKER }}
      depends_on:
        - frontend_server
      probes:
        readiness:
          http_post:
            url: "http://${{ backends.slurm_cluster.nodes[0].ip_address }}:8000/v1/chat/completions"
            headers:
              Content-Type: "application/json"
            body: '{"model": "${{ variables.SERVED_MODEL_NAME }}", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}'
          timeout: 1200
          interval: 20
        failure:
          log_watch:
            match_pattern: "Traceback (most recent call last)"
            match_count: 5
          interval: 30

# ... (44 lines omitted)

    - name: benchmark
      operator:
        name: aiperf
        ntasks: 1
      script:
        - set -x
        - export COLUMNS=200
        - pip install aiperf==0.3.0
        - ls ${{ artifacts.LOCAL_MODEL_PATH.path }}
        - >
          aiperf profile --artifact-dir ${SFLOW_WORKFLOW_OUTPUT_DIR}/aiperf_concurrency_${CONCURRENCY}
          --model ${{ variables.SERVED_MODEL_NAME }}
          --tokenizer ${{ artifacts.LOCAL_MODEL_PATH.path }}
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
          --extra-inputs "repetition_penalty:1.0"
          --extra-inputs "temperature: 0.0"
          --concurrency ${CONCURRENCY}
          --request-count $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --warmup-request-count ${CONCURRENCY}
          --num-dataset-entries $((${{ variables.MULTI_ROUND }}*${CONCURRENCY}))
          --random-seed 100
          -H 'Authorization: Bearer NOT USED'
          -H 'Accept: text/event-stream'
          --record-processors 8
          --ui simple
        - echo "Benchmarking finished"
      resources:
        nodes:
          indices: [0]
      replicas:
        variables:
          - CONCURRENCY
        policy: sequential
      # Trigger a hardware monitor from this benchmark task but sample the
      # prefill/decode SERVERS' resources (used_by_tasks), not the benchmark
      # client's own node -- so the report captures the servers' GPU/CPU usage
      # over the benchmark window.
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
  --set SLURM_ACCOUNT=ACCOUNT \
  --set SLURM_PARTITION=PARTITION \
  --dry-run

# Submit to Slurm (multi-node)
sflow batch -f infmax_v1_ds_r1.yaml \
  -a LOCAL_MODEL_PATH=fs:///path/to/model \
  -A ACCOUNT -p PARTITION -N 3 -G 4 \
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
  --set SLURM_ACCOUNT=ACCOUNT --set SLURM_PARTITION=PARTITION --dry-run
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
3. **Override model path**: the serving samples declare `LOCAL_MODEL_PATH` as an `fs://` artifact (default `fs:///path/to/your/model`) and reference it as `${{ artifacts.LOCAL_MODEL_PATH.path }}`; point it at your checkpoint with `--artifact LOCAL_MODEL_PATH=fs:///path/to/model` (`-a` for short)
4. **Use `--resolve`**: Add `--resolve` to `sflow compose` or `sflow batch --bulk-input` to inline all variables into literal values for a fully-baked config
5. **Check sample source**: Samples are located in `src/sflow/samples/` in the sflow package
