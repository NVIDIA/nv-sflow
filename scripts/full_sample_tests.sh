#!/bin/bash

set -uo pipefail

TEST_TYPE="a"
SUBMIT=""
PREFLIGHT_ONLY=""
MAX_JOBS=16
CLI_MODEL_PATH=""
CLI_PARTITION=""
CLI_ACCOUNT=""
usage() {
    echo "Usage: $0 [-a|-s|-m|-inf|-smoke] [-S] [-P] [-j N] [-M model_path] [-p partition] [-A account]"
    echo "  -a  all tests (default)"
    echo "  -s  self-contained examples only"
    echo "  -m  modular examples only"
    echo "  -inf  infmax batch suites only"
    echo "  -smoke  curated Slurm smoke subset with broad coverage"
    echo "  -S  submit jobs to Slurm"
    echo "  -P  preflight checks only (skip job submission even if -S is set)"
    echo "  -j  max parallel jobs (default: 16, 0 for unlimited)"
    echo "  -M  model path (default: \$MODEL_PATH or /home/)"
    echo "  -p  Slurm partition (default: dummy_part for preflight, my_partition for e2e)"
    echo "  -A  Slurm account (default: dummy_acct for preflight, user for e2e)"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -a) TEST_TYPE="a" ;;
        -s) TEST_TYPE="s" ;;
        -m) TEST_TYPE="m" ;;
        -inf) TEST_TYPE="inf" ;;
        -smoke) TEST_TYPE="smoke" ;;
        -S) SUBMIT="--submit" ;;
        -P) PREFLIGHT_ONLY="1" ;;
        -j) [ $# -ge 2 ] || { usage; exit 1; }; shift; MAX_JOBS="$1" ;;
        -M) [ $# -ge 2 ] || { usage; exit 1; }; shift; CLI_MODEL_PATH="$1" ;;
        -p) [ $# -ge 2 ] || { usage; exit 1; }; shift; CLI_PARTITION="$1" ;;
        -A) [ $# -ge 2 ] || { usage; exit 1; }; shift; CLI_ACCOUNT="$1" ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$REPO_DIR/examples"
CSV_FILE="$EXAMPLES_DIR/inference_x_v2/bulk_input.csv"
MODEL_PATH="${CLI_MODEL_PATH:-${MODEL_PATH:-/home/}}"
PARTITION="${CLI_PARTITION:-dummy_part}"
ACCOUNT="${CLI_ACCOUNT:-dummy_acct}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
INFMAX_DIR="$REPO_DIR/tests/e2e_tests/infmax"
INFMAX_CSV_NAME="${INFMAX_CSV_NAME:-1k1k_jobs.csv}"
INFMAX_ROW="${INFMAX_ROW:-1}"
INFMAX_RECIPE_REPO="${INFMAX_RECIPE_REPO:-}"
INFMAX_RECIPE_REF="${INFMAX_RECIPE_REF:-}"
INFMAX_RECIPE_SUBDIR="${INFMAX_RECIPE_SUBDIR:-}"
INFMAX_REFRESH_RECIPES="${INFMAX_REFRESH_RECIPES:-}"
INFMAX_BENCH_SERVING_DIR="${INFMAX_BENCH_SERVING_DIR:-$INFMAX_DIR/nvidia_submission/sa-bench}"
PREFLIGHT_SKIP_NOTES=""
INFMAX_RECIPE_SKIP_REASON=""

STAMP=$(date +%Y%m%d-%H%M%S)
PREFLIGHT_DIR="$REPO_DIR/sflow_output/preflight_$STAMP"
mkdir -p "$PREFLIGHT_DIR"

source "$SCRIPT_DIR/use_under_dev_sflow.sh"
RESULTS_DIR=""
cleanup() {
    if [ -n "$RESULTS_DIR" ]; then
        rm -rf "$RESULTS_DIR"
    fi
    cleanup_under_dev_sflow
}
trap cleanup EXIT
setup_under_dev_sflow "$REPO_DIR"

RESULTS_DIR=$(mktemp -d)
TEST_ID=0
EXPECTED_BATCH_SFLOW_VERSION="$SFLOW_UNDER_DEV_REF"

abs_path() {
    local path="$1"
    if [ -d "$path" ]; then
        (cd "$path" && pwd)
        return
    fi
    if [ -e "$path" ]; then
        local dir base
        dir="$(dirname "$path")"
        base="$(basename "$path")"
        printf '%s/%s\n' "$(cd "$dir" && pwd)" "$base"
        return
    fi
    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$PWD" "$path" ;;
    esac
}

throttle() {
    if [ "$MAX_JOBS" -gt 0 ]; then
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 0.1
        done
    fi
}

record_preflight_skip() {
    local note="$1"
    case "$PREFLIGHT_SKIP_NOTES" in
        *"$note"*) ;;
        *) PREFLIGHT_SKIP_NOTES="${PREFLIGHT_SKIP_NOTES}  - ${note}\n" ;;
    esac
}

fetch_infmax_recipes() {
    INFMAX_RECIPE_SKIP_REASON=""
    local need_fetch="$INFMAX_REFRESH_RECIPES"
    local required_path

    for required_path in \
        "$INFMAX_DIR/dsr1-fp8-gb200-multi_node-sglang/$INFMAX_CSV_NAME" \
        "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm/$INFMAX_CSV_NAME" \
        "$INFMAX_DIR/kimik2.5-fp4-gb200-multi_node-vllm/$INFMAX_CSV_NAME" \
        "$INFMAX_BENCH_SERVING_DIR"; do
        if [ ! -e "$required_path" ]; then
            need_fetch="1"
            break
        fi
    done

    if [ -z "$need_fetch" ]; then
        return 0
    fi

    if ! command -v git >/dev/null 2>&1; then
        INFMAX_RECIPE_SKIP_REASON="git is not available"
        echo "WARN: skipping Infmax recipe fetch because git is not available." >&2
        return 1
    fi
    if ! command -v rsync >/dev/null 2>&1; then
        INFMAX_RECIPE_SKIP_REASON="rsync is not available"
        echo "WARN: skipping Infmax recipe fetch because rsync is not available." >&2
        return 1
    fi

    if [ -z "$INFMAX_RECIPE_REPO" ] || [ -z "$INFMAX_RECIPE_REF" ] || [ -z "$INFMAX_RECIPE_SUBDIR" ]; then
        INFMAX_RECIPE_SKIP_REASON="recipe source env is incomplete"
        echo "WARN: skipping Infmax recipe fetch because recipe source env is incomplete." >&2
        echo "      Set INFMAX_RECIPE_REPO, INFMAX_RECIPE_REF, and INFMAX_RECIPE_SUBDIR to enable these checks." >&2
        return 1
    fi

    local tmp_dir
    tmp_dir=$(mktemp -d)

    echo "Fetching infmax recipes ($INFMAX_RECIPE_REF:$INFMAX_RECIPE_SUBDIR) ..."
    if ! GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$INFMAX_RECIPE_REF" "$INFMAX_RECIPE_REPO" "$tmp_dir/repo" >/dev/null 2>&1; then
        rm -rf "$tmp_dir"
        INFMAX_RECIPE_SKIP_REASON="recipe repo clone failed"
        echo "WARN: skipping Infmax checks because recipe repo clone failed." >&2
        return 1
    fi

    local source_dir="$tmp_dir/repo/$INFMAX_RECIPE_SUBDIR"
    if [ ! -d "$source_dir" ]; then
        rm -rf "$tmp_dir"
        INFMAX_RECIPE_SKIP_REASON="recipe subdir was not found"
        echo "WARN: skipping Infmax checks because recipe subdir was not found: $INFMAX_RECIPE_SUBDIR" >&2
        return 1
    fi

    rsync -a --delete --exclude='batch_test.sh' "$source_dir/" "$INFMAX_DIR/"
    rm -rf "$tmp_dir"
    return 0
}

set_infmax_suite_overrides() {
    INFMAX_SUITE_OVERRIDES=(-s "CONCURRENCY=[16,32]")
    case "$1" in
        kimik2.5-fp4-gb200-multi_node-vllm)
            INFMAX_SUITE_OVERRIDES+=(-s "DYNAMO_VERSION=1.0.1" -e "--container-remap-root")
            ;;
    esac
}

run_check() {
    local label="$1"
    shift
    local cmd_str="$*"
    TEST_ID=$((TEST_ID + 1))
    local id
    id=$(printf "%03d" "$TEST_ID")
    local result_file="$RESULTS_DIR/${id}.result"
    local output_file="$RESULTS_DIR/${id}.output"

    # Detect output path from -o / --output-dir / --sbatch-path args
    local out_path=""
    local prev=""
    for arg in "$@"; do
        if [ "$prev" = "-o" ] || [ "$prev" = "--output-dir" ] || [ "$prev" = "--sbatch-path" ]; then
            out_path="$arg"
            break
        fi
        prev="$arg"
    done

    throttle

    (
        local status
        if "$@" >"$output_file" 2>&1; then
            status="OK"
        else
            status="FAIL"
        fi
        {
            echo "STATUS=$status"
            echo "LABEL=$label"
            echo "CMD=$cmd_str"
        } > "$result_file"

        # Save the raw command to the output directory for reference
        if [ -n "$out_path" ]; then
            local cmd_target
            if [ -d "$out_path" ]; then
                cmd_target="$out_path"
            else
                cmd_target=$(dirname "$out_path")
            fi
            if [ -d "$cmd_target" ]; then
                printf '# Test: %s\n# Status: %s\n$ %s\n' "$label" "$status" "$cmd_str" \
                    > "$cmd_target/_command.txt"
            fi
        fi
    ) &
}

# =========================================================================
# Preflight: CLI smoke tests (no jobs submitted)
# =========================================================================
if true; then
    echo ""
    echo "===== Preflight: CLI smoke tests (no Slurm submission) ====="
    echo "===== Running tests in parallel (max_jobs=${MAX_JOBS:-unlimited}) ====="
    echo ""

    VERSION_INFO_LOG="$PREFLIGHT_DIR/sflow_version.log"
    run_check "sflow --version runtime info" \
        bash -c "sflow --version > '$VERSION_INFO_LOG' 2>&1 && \
            grep -F -- 'sflow executable' '$VERSION_INFO_LOG' && \
            grep -F -- 'version :' '$VERSION_INFO_LOG' && \
            grep -F -- 'bin     :' '$VERSION_INFO_LOG' && \
            grep -F -- 'python  :' '$VERSION_INFO_LOG' && \
            grep -F -- 'package :' '$VERSION_INFO_LOG' && \
            grep -F -- 'install :' '$VERSION_INFO_LOG' && \
            ! grep -E '^sflow [0-9]' '$VERSION_INFO_LOG'"

    if [ "$TEST_TYPE" = "inf" ]; then
        INFMAX_BATCH_PREFLIGHT_DIR="$PREFLIGHT_DIR/infmax_batch"
        if fetch_infmax_recipes; then
            INFMAX_TARGET_DIRS=(
                "$INFMAX_DIR/dsr1-fp8-gb200-multi_node-sglang"
                "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm"
                "$INFMAX_DIR/kimik2.5-fp4-gb200-multi_node-vllm"
            )
            for suite_dir in "${INFMAX_TARGET_DIRS[@]}"; do
                suite_name=$(basename "$suite_dir")
                set_infmax_suite_overrides "$suite_name"
                run_check "infmax multi-node batch $suite_name" \
                    sflow batch \
                        --bulk-input "$suite_dir/$INFMAX_CSV_NAME" \
                        --row "$INFMAX_ROW" \
                        -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                        -a "BENCH_SERVING_DIR=fs://$INFMAX_BENCH_SERVING_DIR" \
                        -G "$GPUS_PER_NODE" \
                        -A "$ACCOUNT" \
                        -p "$PARTITION" \
                        --log-level warn \
                        "${INFMAX_SUITE_OVERRIDES[@]}" \
                        --output-dir "$INFMAX_BATCH_PREFLIGHT_DIR/$suite_name"
            done
        else
            record_preflight_skip "Infmax batch preflight checks skipped: ${INFMAX_RECIPE_SKIP_REASON:-recipe source unavailable}"
            echo "  SKIP: Infmax recipe source unavailable; skipping Infmax batch preflight checks."
        fi
    else

    # -- sflow run --dry-run: local examples --
    run_check "local_hello_world" \
        sflow run "$EXAMPLES_DIR/local_hello_world.yaml" --dry-run
    run_check "local_dag" \
        sflow run "$EXAMPLES_DIR/local_dag.yaml" --dry-run
    run_check "local_variable_domain" \
        sflow run "$EXAMPLES_DIR/local_variable_domain.yaml" --dry-run

    # -- sflow run (live): verify replica sweep + domain resolution --
    # Note: may fail in sandboxed environments (pty device limits) with many parallel tasks.
    DOMAIN_RUN_DIR="$PREFLIGHT_DIR/run_variable_domain"
    run_check "run local_variable_domain (live, optional)" \
        sflow run "$EXAMPLES_DIR/local_variable_domain.yaml" \
            --output-dir "$DOMAIN_RUN_DIR"

    # -- sflow run/batch: plain script commands containing ':' must stay strings --
    COLON_SCRIPT_DIR="$PREFLIGHT_DIR/colon_in_task_script"
    COLON_SCRIPT_FIXTURE="$COLON_SCRIPT_DIR/colon_in_task_script.yaml"
    COLON_SCRIPT_DRYRUN_LOG="$COLON_SCRIPT_DIR/dry_run.log"
    COLON_SCRIPT_COMPOSED="$COLON_SCRIPT_DIR/colon_in_task_script_composed.yaml"
    COLON_SCRIPT_BATCH="$COLON_SCRIPT_DIR/colon_in_task_script_batch.sh"
    COLON_SCRIPT_BATCH_CONFIG="$COLON_SCRIPT_DIR/colon_in_task_script_batch.yaml"
    mkdir -p "$COLON_SCRIPT_DIR"
    cat > "$COLON_SCRIPT_FIXTURE" <<'EOF'
version: "0.1"

variables:
  SLURM_ACCOUNT:
    value: dummy_acct
  SLURM_PARTITION:
    value: dummy_part
  SLURM_TIMELIMIT:
    value: "00:10:00"
  SLURM_NODES:
    value: 1
  GPUS_PER_NODE:
    value: 4

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: ${{ variables.SLURM_ACCOUNT }}
    partition: ${{ variables.SLURM_PARTITION }}
    time: ${{ variables.SLURM_TIMELIMIT }}
    nodes: ${{ variables.SLURM_NODES }}
    gpus_per_node: ${{ variables.GPUS_PER_NODE }}

operators:
  - name: srun_no_container
    type: srun
    ntasks_per_node: 1
    mpi: pmix

workflow:
  name: colon_in_task_script
  tasks:
    - name: worker
      operator: srun_no_container
      resources:
        gpus:
          count: 1
      script:
        - echo "My GPUs: $CUDA_VISIBLE_DEVICES"
        - echo "COLON_SCRIPT_E2E_PASS"
EOF
    run_check "run colon in task script (dry-run)" \
        bash -c "sflow run \"$COLON_SCRIPT_FIXTURE\" --dry-run > \"$COLON_SCRIPT_DRYRUN_LOG\" 2>&1"
    run_check "compose colon in task script" \
        sflow compose "$COLON_SCRIPT_FIXTURE" \
            -o "$COLON_SCRIPT_COMPOSED"
    run_check "batch colon in task script" \
        sflow batch -f "$COLON_SCRIPT_FIXTURE" \
            -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
            -o "$COLON_SCRIPT_BATCH"

    if [ "$TEST_TYPE" = "m" ] || [ "$TEST_TYPE" = "a" ]; then
        INFMAX_BATCH_PREFLIGHT_DIR="$PREFLIGHT_DIR/infmax_batch"
        if fetch_infmax_recipes; then
            INFMAX_TARGET_DIRS=(
                "$INFMAX_DIR/dsr1-fp8-gb200-multi_node-sglang"
                "$INFMAX_DIR/dsr1-fp4-gb200-multi_node-trtllm"
                "$INFMAX_DIR/kimik2.5-fp4-gb200-multi_node-vllm"
            )
            for suite_dir in "${INFMAX_TARGET_DIRS[@]}"; do
                suite_name=$(basename "$suite_dir")
                set_infmax_suite_overrides "$suite_name"
                run_check "infmax multi-node batch $suite_name" \
                    sflow batch \
                        --bulk-input "$suite_dir/$INFMAX_CSV_NAME" \
                        --row "$INFMAX_ROW" \
                        -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                        -a "BENCH_SERVING_DIR=fs://$INFMAX_BENCH_SERVING_DIR" \
                        -G "$GPUS_PER_NODE" \
                        -A "$ACCOUNT" \
                        -p "$PARTITION" \
                        --log-level warn \
                        "${INFMAX_SUITE_OVERRIDES[@]}" \
                        --output-dir "$INFMAX_BATCH_PREFLIGHT_DIR/$suite_name"
            done
        else
            record_preflight_skip "Infmax batch preflight checks skipped: ${INFMAX_RECIPE_SKIP_REASON:-recipe source unavailable}"
            echo "  SKIP: Infmax recipe source unavailable; skipping Infmax batch preflight checks."
        fi
    fi

    # -- sflow run/batch/compose: release_after enables dynamic resource rehearsal --
    RELEASE_AFTER_DIR="$PREFLIGHT_DIR/release_after_resource_rehearsal"
    RELEASE_AFTER_FIXTURE="$RELEASE_AFTER_DIR/release_after_inferred_task_completion.yaml"
    RELEASE_AFTER_NEGATIVE_FIXTURE="$RELEASE_AFTER_DIR/release_after_explicit_workflow_completion.yaml"
    RELEASE_AFTER_NODES_OMITTED_FIXTURE="$RELEASE_AFTER_DIR/nodes_release_after_omitted_overlap.yaml"
    RELEASE_AFTER_NODES_COUNT_NEGATIVE_FIXTURE="$RELEASE_AFTER_DIR/nodes_release_after_explicit_count.yaml"
    RELEASE_AFTER_NODES_INDICES_NEGATIVE_FIXTURE="$RELEASE_AFTER_DIR/nodes_release_after_explicit_indices.yaml"
    RELEASE_AFTER_DRYRUN_LOG="$RELEASE_AFTER_DIR/dry_run.log"
    RELEASE_AFTER_NEGATIVE_LOG="$RELEASE_AFTER_DIR/negative_dry_run.log"
    RELEASE_AFTER_NODES_OMITTED_LOG="$RELEASE_AFTER_DIR/nodes_omitted_dry_run.log"
    RELEASE_AFTER_NODES_COUNT_NEGATIVE_LOG="$RELEASE_AFTER_DIR/nodes_count_negative_dry_run.log"
    RELEASE_AFTER_NODES_INDICES_NEGATIVE_LOG="$RELEASE_AFTER_DIR/nodes_indices_negative_dry_run.log"
    RELEASE_AFTER_COMPOSE_LOG="$RELEASE_AFTER_DIR/compose_validate.log"
    RELEASE_AFTER_COMPOSED="$RELEASE_AFTER_DIR/composed.yaml"
    RELEASE_AFTER_BATCH="$RELEASE_AFTER_DIR/release_after_batch.sh"
    mkdir -p "$RELEASE_AFTER_DIR"
    cat > "$RELEASE_AFTER_FIXTURE" <<'EOF'
version: "0.1"

variables:
  SLURM_ACCOUNT:
    value: dummy_acct
  SLURM_PARTITION:
    value: dummy_part

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: ${{ variables.SLURM_ACCOUNT }}
    partition: ${{ variables.SLURM_PARTITION }}
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 8

workflow:
  name: release_after_inferred_task_completion
  tasks:
    - name: check_entire_env
      resources:
        gpus:
          count: 8
      script:
        - echo "$CUDA_VISIBLE_DEVICES"

    - name: worker
      replicas:
        count: 4
        policy: parallel
      resources:
        gpus:
          count: 2
      script:
        - echo "worker GPUs: $CUDA_VISIBLE_DEVICES"
      depends_on:
        - check_entire_env
EOF
    cat > "$RELEASE_AFTER_NEGATIVE_FIXTURE" <<'EOF'
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: dummy_acct
    partition: dummy_part
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 8

workflow:
  name: release_after_explicit_workflow_completion
  tasks:
    - name: check_entire_env
      resources:
        gpus:
          count: 8
          release_after: workflow_completion
      script:
        - echo "$CUDA_VISIBLE_DEVICES"

    - name: worker
      replicas:
        count: 4
        policy: parallel
      resources:
        gpus:
          count: 2
      script:
        - echo "worker GPUs: $CUDA_VISIBLE_DEVICES"
      depends_on:
        - check_entire_env
EOF
    cat > "$RELEASE_AFTER_NODES_OMITTED_FIXTURE" <<'EOF'
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: dummy_acct
    partition: dummy_part
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 0

workflow:
  name: nodes_release_after_omitted_overlap
  tasks:
    - name: count_a
      resources:
        nodes:
          count: 1
      script:
        - echo count_a

    - name: count_b
      resources:
        nodes:
          count: 1
      script:
        - echo count_b

    - name: pinned_a
      resources:
        nodes:
          indices: [0]
      script:
        - echo pinned_a

    - name: pinned_b
      resources:
        nodes:
          indices: [0]
      script:
        - echo pinned_b
EOF
    cat > "$RELEASE_AFTER_NODES_COUNT_NEGATIVE_FIXTURE" <<'EOF'
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: dummy_acct
    partition: dummy_part
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 0

workflow:
  name: nodes_release_after_explicit_count
  tasks:
    - name: exclusive_count
      resources:
        nodes:
          count: 1
          release_after: workflow_completion
      script:
        - echo exclusive_count

    - name: sibling
      resources:
        nodes:
          count: 1
      script:
        - echo sibling
EOF
    cat > "$RELEASE_AFTER_NODES_INDICES_NEGATIVE_FIXTURE" <<'EOF'
version: "0.1"

backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: dummy_acct
    partition: dummy_part
    time: "00:10:00"
    nodes: 1
    gpus_per_node: 0

workflow:
  name: nodes_release_after_explicit_indices
  tasks:
    - name: exclusive_index
      resources:
        nodes:
          indices: [0]
          release_after: workflow_completion
      script:
        - echo exclusive_index

    - name: sibling
      resources:
        nodes:
          count: 1
      script:
        - echo sibling
EOF
    run_check "run release_after resource rehearsal (dry-run)" \
        bash -c "sflow run \"$RELEASE_AFTER_FIXTURE\" --dry-run > \"$RELEASE_AFTER_DRYRUN_LOG\" 2>&1 && grep -q 'Resource release rehearsal' \"$RELEASE_AFTER_DRYRUN_LOG\""
    run_check "batch release_after resource rehearsal" \
        sflow batch -f "$RELEASE_AFTER_FIXTURE" \
            -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
            -o "$RELEASE_AFTER_BATCH"
    run_check "compose release_after resource rehearsal (validate)" \
        bash -c "sflow compose \"$RELEASE_AFTER_FIXTURE\" --validate -o \"$RELEASE_AFTER_COMPOSED\" > \"$RELEASE_AFTER_COMPOSE_LOG\" 2>&1 && grep -q 'Dry-run validation passed' \"$RELEASE_AFTER_COMPOSE_LOG\" && ! grep -q 'WARNING: dry-run validation failed' \"$RELEASE_AFTER_COMPOSE_LOG\""
    run_check "run explicit workflow_completion resource lifetime (expect fail)" \
        bash -c "! sflow run \"$RELEASE_AFTER_NEGATIVE_FIXTURE\" --dry-run > \"$RELEASE_AFTER_NEGATIVE_LOG\" 2>&1 && grep -q 'remain available' \"$RELEASE_AFTER_NEGATIVE_LOG\""
    run_check "run omitted node release_after allows overlap (dry-run)" \
        bash -c "sflow run \"$RELEASE_AFTER_NODES_OMITTED_FIXTURE\" --dry-run > \"$RELEASE_AFTER_NODES_OMITTED_LOG\" 2>&1 && grep -q 'Dry-run complete' \"$RELEASE_AFTER_NODES_OMITTED_LOG\" && ! grep -q 'remain available' \"$RELEASE_AFTER_NODES_OMITTED_LOG\""
    run_check "run explicit node count workflow_completion (expect fail)" \
        bash -c "! sflow run \"$RELEASE_AFTER_NODES_COUNT_NEGATIVE_FIXTURE\" --dry-run > \"$RELEASE_AFTER_NODES_COUNT_NEGATIVE_LOG\" 2>&1 && grep -q 'remain available' \"$RELEASE_AFTER_NODES_COUNT_NEGATIVE_LOG\" && grep -q 'exclusive_count: release_after=workflow_completion' \"$RELEASE_AFTER_NODES_COUNT_NEGATIVE_LOG\""
    run_check "run explicit node indices workflow_completion (expect fail)" \
        bash -c "! sflow run \"$RELEASE_AFTER_NODES_INDICES_NEGATIVE_FIXTURE\" --dry-run > \"$RELEASE_AFTER_NODES_INDICES_NEGATIVE_LOG\" 2>&1 && grep -q 'remain available' \"$RELEASE_AFTER_NODES_INDICES_NEGATIVE_LOG\" && grep -q 'exclusive_index: release_after=workflow_completion' \"$RELEASE_AFTER_NODES_INDICES_NEGATIVE_LOG\""

    # -- sflow run --dry-run: readiness accepts a list and builds multiple readiness probes --
    READINESS_AND_DIR="$PREFLIGHT_DIR/readiness_probe_and"
    READINESS_AND_FIXTURE="$READINESS_AND_DIR/readiness_probe_and.yaml"
    READINESS_AND_DRYRUN_LOG="$READINESS_AND_DIR/dry_run.log"
    READINESS_SINGLE_FIXTURE="$READINESS_AND_DIR/readiness_probe_single.yaml"
    READINESS_SINGLE_DRYRUN_LOG="$READINESS_AND_DIR/single_dry_run.log"
    mkdir -p "$READINESS_AND_DIR"
    cat > "$READINESS_AND_FIXTURE" <<'EOF'
version: "0.1"
workflow:
  name: readiness_probe_and
  tasks:
    - name: service
      script:
        - echo "readiness one"
        - sleep 1
        - echo "readiness two"
        - touch "${SFLOW_WORKFLOW_OUTPUT_DIR}/all_readiness_probes_passed"
        - sleep 2
      probes:
        readiness:
          - log_watch:
              match_pattern: "readiness one"
            interval: 0
            timeout: 10
          - log_watch:
              match_pattern: "readiness two"
            interval: 0
            timeout: 10
    - name: after_ready
      depends_on:
        - service
      script:
        - test -f "${SFLOW_WORKFLOW_OUTPUT_DIR}/all_readiness_probes_passed"
        - echo "after_ready observed all readiness probes"
EOF
    run_check "run readiness probe list (dry-run)" \
        bash -c "sflow run \"$READINESS_AND_FIXTURE\" --dry-run > \"$READINESS_AND_DRYRUN_LOG\" 2>&1"
    cat > "$READINESS_SINGLE_FIXTURE" <<'EOF'
version: "0.1"
workflow:
  name: readiness_probe_single_compat
  tasks:
    - name: service
      script:
        - echo "single readiness"
      probes:
        readiness:
          log_watch:
            match_pattern: "single readiness"
          interval: 0
          timeout: 10
EOF
    run_check "run single readiness probe compatibility (dry-run)" \
        bash -c "sflow run \"$READINESS_SINGLE_FIXTURE\" --dry-run > \"$READINESS_SINGLE_DRYRUN_LOG\" 2>&1"

    # -- sflow run --dry-run: self-contained slurm examples --
    for f in "$EXAMPLES_DIR"/slurm_*.yaml; do
        name=$(basename "$f" .yaml)
        run_check "dry-run $name" \
            sflow run "$f" --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"
    done

    # -- sflow run (fake Slurm): backend gpus_per_node is planning-only; --gpus-per-node is extra_args-only --
    SLURM_GPN_DIR="$PREFLIGHT_DIR/slurm_gpus_per_node_extra_args_only"
    SLURM_GPN_FIXTURE_DIR="$SLURM_GPN_DIR/fixture"
    SLURM_GPN_FAKE_BIN="$SLURM_GPN_DIR/fake_bin"
    SLURM_GPN_LOG_DIR="$SLURM_GPN_DIR/logs"
    mkdir -p "$SLURM_GPN_FIXTURE_DIR" "$SLURM_GPN_FAKE_BIN" "$SLURM_GPN_LOG_DIR"
    cat > "$SLURM_GPN_FAKE_BIN/salloc" <<'EOF'
#!/bin/bash
{
    printf 'salloc'
    for arg in "$@"; do
        printf ' %s' "$arg"
    done
    printf '\n'
} >> "$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args"
echo "salloc: Granted job allocation 424242"
echo "salloc: Nodes fake-node are ready for job"
EOF
    cat > "$SLURM_GPN_FAKE_BIN/scontrol" <<'EOF'
#!/bin/bash
if [ "$1" = "getaddrs" ]; then
    echo "fake-node: 127.0.0.1:123"
    exit 0
fi
exit 1
EOF
    cat > "$SLURM_GPN_FAKE_BIN/srun" <<'EOF'
#!/bin/bash
{
    printf 'srun'
    for arg in "$@"; do
        printf ' %s' "$arg"
    done
    printf '\n'
} >> "$SFLOW_FAKE_SLURM_LOG_DIR/srun.args"
exit 0
EOF
    cat > "$SLURM_GPN_FAKE_BIN/scancel" <<'EOF'
#!/bin/bash
echo "scancel $*" >> "$SFLOW_FAKE_SLURM_LOG_DIR/scancel.args"
exit 0
EOF
    chmod +x "$SLURM_GPN_FAKE_BIN"/salloc "$SLURM_GPN_FAKE_BIN"/scontrol \
        "$SLURM_GPN_FAKE_BIN"/srun "$SLURM_GPN_FAKE_BIN"/scancel
    cat > "$SLURM_GPN_FIXTURE_DIR/no_extra_args.yaml" <<'EOF'
version: "0.1"
backends:
  - name: fake_slurm
    type: slurm
    default: true
    account: acct
    partition: gpu
    time: "00:05:00"
    nodes: 1
    gpus_per_node: 4
workflow:
  name: slurm_no_implicit_gpus_per_node
  tasks:
    - name: worker
      script:
        - echo no implicit gpus-per-node
EOF
    cat > "$SLURM_GPN_FIXTURE_DIR/with_extra_args.yaml" <<'EOF'
version: "0.1"
backends:
  - name: fake_slurm
    type: slurm
    default: true
    account: acct
    partition: gpu
    time: "00:05:00"
    nodes: 1
    gpus_per_node: 4
    extra_args:
      - "--gpus-per-node=4"
workflow:
  name: slurm_explicit_gpus_per_node
  tasks:
    - name: worker
      script:
        - echo explicit gpus-per-node
EOF
    run_check "run fake slurm gpus-per-node extra_args-only" \
        bash -c "set -euo pipefail
            export PATH='$SLURM_GPN_FAKE_BIN':\"\$PATH\"
            export SFLOW_FAKE_SLURM_LOG_DIR='$SLURM_GPN_LOG_DIR'
            rm -f \"\$SFLOW_FAKE_SLURM_LOG_DIR\"/*.args
            no_extra_log='$SLURM_GPN_DIR/no_extra_run.log'
            with_extra_log='$SLURM_GPN_DIR/with_extra_run.log'
            sflow run '$SLURM_GPN_FIXTURE_DIR/no_extra_args.yaml' \
                --output-dir '$SLURM_GPN_DIR/no_extra_run' \
                > \"\$no_extra_log\" 2>&1
            first_salloc=\$(sed -n '1p' \"\$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args\")
            if printf '%s\n' \"\$first_salloc\" | grep -q -- '--gpus-per-node'; then
                echo \"unexpected implicit --gpus-per-node in: \$first_salloc\"
                exit 1
            fi
            grep -F -- 'backend.gpus_per_node=4 is sflow planning only' \"\$no_extra_log\"
            sflow run '$SLURM_GPN_FIXTURE_DIR/with_extra_args.yaml' \
                --output-dir '$SLURM_GPN_DIR/with_extra_run' \
                > \"\$with_extra_log\" 2>&1
            second_salloc=\$(sed -n '2p' \"\$SFLOW_FAKE_SLURM_LOG_DIR/salloc.args\")
            printf '%s\n' \"\$second_salloc\" | grep -F -- '--gpus-per-node=4'
            grep -F -- 'backend.gpus_per_node=4 is sflow planning only' \"\$with_extra_log\""

    # -- sflow run --dry-run: modular (multi-file) --
    SLURM_CFG="$EXAMPLES_DIR/inference_x_v2/slurm_config.yaml"
    COMMON="$EXAMPLES_DIR/inference_x_v2/common_workflow.yaml"
    BENCH_INFMAX="$EXAMPLES_DIR/inference_x_v2/benchmark_infmax.yaml"
    BENCH_AIPERF="$EXAMPLES_DIR/inference_x_v2/benchmark_aiperf.yaml"
    DYNAMO_IMAGE="${DYNAMO_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0}"
    MODULAR_MISSABLE=(-M agg_server -M prefill_server -M decode_server -M benchmark_infmax -M benchmark_aiperf)
    MODULAR_OVERRIDES=(-a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" -s "DYNAMO_IMAGE=$DYNAMO_IMAGE")
    for framework in trtllm sglang vllm; do
        run_check "dry-run modular $framework/disagg" \
            sflow run "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/inference_x_v2/$framework/prefill.yaml" \
                "$EXAMPLES_DIR/inference_x_v2/$framework/decode.yaml" \
                "$BENCH_INFMAX" \
                --dry-run "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}"
        run_check "dry-run modular $framework/agg" \
            sflow run "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/inference_x_v2/$framework/agg.yaml" \
                "$BENCH_AIPERF" \
                --dry-run "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}"
    done

    # -- sflow compose: variable domain access --
    COMPOSE_DOMAIN_DIR="$PREFLIGHT_DIR/compose_domain"
    mkdir -p "$COMPOSE_DOMAIN_DIR"
    run_check "compose variable_domain" \
        sflow compose "$EXAMPLES_DIR/local_variable_domain.yaml" -vl -r \
            -o "$COMPOSE_DOMAIN_DIR/resolved.yaml"

    # -- sflow compose: deferred Jinja should keep backend refs but inline resolved vars --
    COMPOSE_DEFERRED_DIR="$PREFLIGHT_DIR/compose_deferred_jinja"
    COMPOSE_DEFERRED_FIXTURE_DIR="$COMPOSE_DEFERRED_DIR/fixture"
    mkdir -p "$COMPOSE_DEFERRED_FIXTURE_DIR"
    cat > "$COMPOSE_DEFERRED_FIXTURE_DIR/vars.yaml" <<'EOF'
version: "0.1"
variables:
  - name: INFRA_NODE_INDEX
    value: 0
backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: acct
    partition: batch
    time: "00:10:00"
    nodes: 4
    gpus_per_node: 4
EOF
    cat > "$COMPOSE_DEFERRED_FIXTURE_DIR/workflow.yaml" <<'EOF'
version: "0.1"
workflow:
  name: wf
  variables:
    - name: HEAD_NODE_IP
      value: ${{ backends.slurm_cluster.nodes[0].ip_address if variables.INFRA_NODE_INDEX == 0 else backends.slurm_cluster.nodes[-1].ip_address }}
    - name: NATS_SERVER
      value: nats://${{ backends.slurm_cluster.nodes[0].ip_address if variables.INFRA_NODE_INDEX == 0 else backends.slurm_cluster.nodes[-1].ip_address }}:4222
  tasks:
    - name: t1
      script:
        - echo hi
EOF
    run_check "compose deferred_jinja_literal_rewrite" \
        sflow compose "$COMPOSE_DEFERRED_FIXTURE_DIR/vars.yaml" \
            "$COMPOSE_DEFERRED_FIXTURE_DIR/workflow.yaml" -r \
            -o "$COMPOSE_DEFERRED_DIR/resolved.yaml"

    # -- sflow compose: resources.nodes.indices/exclude may be a single expression string resolving to a list --
    COMPOSE_INDICES_DIR="$PREFLIGHT_DIR/compose_indices_expression"
    COMPOSE_INDICES_FIXTURE_DIR="$COMPOSE_INDICES_DIR/fixture"
    COMPOSE_INDICES_DRYRUN_LOG="$COMPOSE_INDICES_DIR/dry_run.log"
    mkdir -p "$COMPOSE_INDICES_FIXTURE_DIR"
    cat > "$COMPOSE_INDICES_FIXTURE_DIR/vars.yaml" <<'EOF'
version: "0.1"
variables:
  - name: INFRA_NODE_INDEX
    value: 0
    type: integer
  - name: NUM_FRONTENDS
    value: 2
    type: integer
backends:
  - name: slurm_cluster
    type: slurm
    default: true
    account: acct
    partition: batch
    time: "00:10:00"
    nodes: 4
    gpus_per_node: 4
EOF
    cat > "$COMPOSE_INDICES_FIXTURE_DIR/workflow.yaml" <<'EOF'
version: "0.1"
workflow:
  name: wf
  tasks:
    - name: frontend_server
      script:
        - echo hi
      resources:
        nodes:
          indices: ${{ range(variables.INFRA_NODE_INDEX, variables.INFRA_NODE_INDEX + variables.NUM_FRONTENDS) | list }}
    - name: worker_server
      script:
        - echo worker
      resources:
        nodes:
          exclude: ${{ range(variables.INFRA_NODE_INDEX, variables.INFRA_NODE_INDEX + variables.NUM_FRONTENDS) | list }}
    - name: ordered_pool
      script:
        - echo ordered
      replicas:
        count: 4
        policy: parallel
      resources:
        nodes:
          indices: [-1, 0, 1, 2]
          count: 1
EOF
    run_check "compose nodes.indices/exclude expression strings resolve to list" \
        sflow compose "$COMPOSE_INDICES_FIXTURE_DIR/vars.yaml" \
            "$COMPOSE_INDICES_FIXTURE_DIR/workflow.yaml" -r \
            -o "$COMPOSE_INDICES_DIR/resolved.yaml"
    run_check "run nodes.indices/exclude expression strings and indices+count ordering (dry-run)" \
        bash -c "sflow run \"$COMPOSE_INDICES_FIXTURE_DIR/vars.yaml\" \"$COMPOSE_INDICES_FIXTURE_DIR/workflow.yaml\" --dry-run > \"$COMPOSE_INDICES_DRYRUN_LOG\" 2>&1"

    # -- sflow compose: single-file self-contained examples --
    COMPOSE_SINGLE_DIR="$PREFLIGHT_DIR/compose_single"
    mkdir -p "$COMPOSE_SINGLE_DIR"
    for f in "$EXAMPLES_DIR"/slurm_*.yaml; do
        name=$(basename "$f" .yaml)
        run_check "compose $name" \
            sflow compose "$f" -vl -r -o "$COMPOSE_SINGLE_DIR/$name.yaml"
    done

    # -- sflow compose: modular (multi-file) --
    COMPOSE_MODULAR_DIR="$PREFLIGHT_DIR/compose_modular"
    mkdir -p "$COMPOSE_MODULAR_DIR"
    for framework in trtllm sglang vllm; do
        run_check "compose modular $framework/disagg" \
            sflow compose "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/inference_x_v2/$framework/prefill.yaml" \
                "$EXAMPLES_DIR/inference_x_v2/$framework/decode.yaml" \
                "$BENCH_INFMAX" \
                "${MODULAR_MISSABLE[@]}" -r -vl \
                -o "$COMPOSE_MODULAR_DIR/${framework}_disagg.yaml"
        run_check "compose modular $framework/agg" \
            sflow compose "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/inference_x_v2/$framework/agg.yaml" \
                "$BENCH_AIPERF" \
                "${MODULAR_MISSABLE[@]}" -r -vl \
                -o "$COMPOSE_MODULAR_DIR/${framework}_agg.yaml"
    done

    # -- sflow compose --bulk-input (CSV) --
    if [ -f "$CSV_FILE" ]; then
        run_check "compose bulk-input all rows" \
            sflow compose -b "$CSV_FILE" -o "$PREFLIGHT_DIR/compose_bulk_input"

        run_check "compose bulk-input single row" \
            sflow compose -b "$CSV_FILE" --row 1 -o "$PREFLIGHT_DIR/compose_bulk_input_row1"

        run_check "compose bulk-input row range" \
            sflow compose -b "$CSV_FILE" --row 7:10 -o "$PREFLIGHT_DIR/compose_bulk_input_multi_rows"

        # -- negative index and open-ended slice tests --
        run_check "compose bulk-input last row (--row=-1)" \
            sflow compose -b "$CSV_FILE" --row=-1 -o "$PREFLIGHT_DIR/compose_bulk_input_last_row"

        run_check "compose bulk-input negative range (--row=-3:)" \
            sflow compose -b "$CSV_FILE" --row=-3: -o "$PREFLIGHT_DIR/compose_bulk_input_last3"

        run_check "compose bulk-input open-end slice (--row 3:)" \
            sflow compose -b "$CSV_FILE" --row=3: -o "$PREFLIGHT_DIR/compose_bulk_input_3_to_end"

        run_check "compose bulk-input negative slice (--row=-3:-1)" \
            sflow compose -b "$CSV_FILE" --row=-3:-1 -o "$PREFLIGHT_DIR/compose_bulk_input_neg_slice"
    else
        echo "  SKIP: CSV not found at $CSV_FILE"
    fi

    # -- sflow batch -f (single file): self-contained examples --
    BATCH_SINGLE_DIR="$PREFLIGHT_DIR/batch_single"
    mkdir -p "$BATCH_SINGLE_DIR"
    for f in "$EXAMPLES_DIR"/slurm_*.yaml; do
        name=$(basename "$f" .yaml)
        run_check "batch single $name" \
            sflow batch -f "$f" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -o "$BATCH_SINGLE_DIR/$name.sh"
    done

    # -- sflow batch -f (multi-file): modular examples --
    BATCH_MODULAR_DIR="$PREFLIGHT_DIR/batch_modular"
    mkdir -p "$BATCH_MODULAR_DIR"
    for framework in trtllm sglang vllm; do
        run_check "batch modular $framework/disagg" \
            sflow batch \
                -f "$SLURM_CFG" -f "$COMMON" \
                -f "$EXAMPLES_DIR/inference_x_v2/$framework/prefill.yaml" \
                -f "$EXAMPLES_DIR/inference_x_v2/$framework/decode.yaml" \
                -f "$BENCH_INFMAX" -r \
                "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -o "$BATCH_MODULAR_DIR/${framework}_disagg.sh"
        run_check "batch modular $framework/agg" \
            sflow batch \
                -f "$SLURM_CFG" -f "$COMMON" \
                -f "$EXAMPLES_DIR/inference_x_v2/$framework/agg.yaml" \
                -f "$BENCH_AIPERF" \
                "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -o "$BATCH_MODULAR_DIR/${framework}_agg.sh"
    done

    # -- sflow batch -e with expression resolution --
    BATCH_EXTRA_ARGS_DIR="$PREFLIGHT_DIR/batch_extra_args_expr"
    mkdir -p "$BATCH_EXTRA_ARGS_DIR"
    EXTRA_ARGS_EXAMPLE="$EXAMPLES_DIR/slurm_dynamo_sglang_disagg.yaml"
    if [ -f "$EXTRA_ARGS_EXAMPLE" ]; then
        run_check "batch -e expression resolution" \
            sflow batch -f "$EXTRA_ARGS_EXAMPLE" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -s "SLURM_NODES=3" \
                -e '--segment=${{ variables.SLURM_NODES }}' \
                -o "$BATCH_EXTRA_ARGS_DIR/expr_test.sh"
        if [ -f "$BATCH_EXTRA_ARGS_DIR/expr_test.sh" ]; then
            if grep -q '#SBATCH --segment=3' "$BATCH_EXTRA_ARGS_DIR/expr_test.sh"; then
                echo "  PASS: -e expression resolved to '--segment=3'"
            else
                echo "  FAIL: -e expression was not resolved (expected '#SBATCH --segment=3')"
                grep '#SBATCH --segment' "$BATCH_EXTRA_ARGS_DIR/expr_test.sh" || echo "    (no --segment directive found)"
            fi
        fi
    fi

    # -- sflow batch default --sflow-version: should follow current execution env --
    BATCH_DEFAULT_VERSION_DIR="$PREFLIGHT_DIR/batch_default_sflow_version"
    mkdir -p "$BATCH_DEFAULT_VERSION_DIR"
    if [ -f "$EXTRA_ARGS_EXAMPLE" ]; then
        run_check "batch default --sflow-version from current env" \
            sflow batch -f "$EXTRA_ARGS_EXAMPLE" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -s "SLURM_NODES=3" \
                -o "$BATCH_DEFAULT_VERSION_DIR/default_version.sh"
    fi

    # -- sflow batch -e with variables.X.domain expression --
    BATCH_DOMAIN_DIR="$PREFLIGHT_DIR/batch_domain_expr"
    mkdir -p "$BATCH_DOMAIN_DIR"
    DOMAIN_EXAMPLE="$EXAMPLES_DIR/local_variable_domain.yaml"
    if [ -f "$DOMAIN_EXAMPLE" ]; then
        run_check "batch -e domain expression" \
            sflow batch -f "$DOMAIN_EXAMPLE" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                --nodes 1 \
                -e '--comment=${{ variables.CONCURRENCY.domain }}' \
                -o "$BATCH_DOMAIN_DIR/domain_test.sh"
    fi

    # -- sflow batch --bulk-submit (no --submit): self-contained --
    run_check "batch bulk-submit (no submit)" \
        sflow batch --bulk-submit "$EXAMPLES_DIR" \
            -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
            -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
            --output-dir "$PREFLIGHT_DIR/batch_bulk_submit"

    # -- sflow batch --bulk-input (no --submit): CSV --
    if [ -f "$CSV_FILE" ]; then
        run_check "batch bulk-input (no submit)" \
            sflow batch --bulk-input "$CSV_FILE" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn -r \
                --output-dir "$PREFLIGHT_DIR/batch_bulk_input"

        # -- verify sflow_batch_dir column in results.csv --
        # -- negative index and open-ended slice tests --
        run_check "batch bulk-input last row (--row=-1)" \
            sflow batch --bulk-input "$CSV_FILE" --row=-1 \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                --output-dir "$PREFLIGHT_DIR/batch_bulk_input_last_row"

        run_check "batch bulk-input last 3 rows (--row=-3:)" \
            sflow batch --bulk-input "$CSV_FILE" --row=-3: \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                --output-dir "$PREFLIGHT_DIR/batch_bulk_input_last3"

        run_check "batch bulk-input open-end (--row=3:)" \
            sflow batch --bulk-input "$CSV_FILE" --row=3: \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                --output-dir "$PREFLIGHT_DIR/batch_bulk_input_3_to_end"
    else
        echo "  SKIP: CSV not found at $CSV_FILE"
    fi

    # -- sflow batch --bulk-input + CLI -f: CLI config must be prepended to every CSV row --
    BATCH_BULK_CLI_FILES_DIR="$PREFLIGHT_DIR/batch_bulk_input_cli_files"
    BATCH_BULK_CLI_FILES_FIXTURE_DIR="$BATCH_BULK_CLI_FILES_DIR/fixture"
    mkdir -p "$BATCH_BULK_CLI_FILES_FIXTURE_DIR"
    cat > "$BATCH_BULK_CLI_FILES_FIXTURE_DIR/common.yaml" <<'EOF'
version: "0.1"
variables:
  - name: SHARED_VALUE
    value: from_common
EOF
    cat > "$BATCH_BULK_CLI_FILES_FIXTURE_DIR/task.yaml" <<'EOF'
version: "0.1"
workflow:
  name: batch_bulk_input_cli_files
  tasks:
    - name: show_shared
      script:
        - echo "${SHARED_VALUE}"
EOF
    cat > "$BATCH_BULK_CLI_FILES_FIXTURE_DIR/jobs.csv" <<EOF
sflow_config_file
$BATCH_BULK_CLI_FILES_FIXTURE_DIR/task.yaml
EOF
    run_check "batch bulk-input with cli -f prepends config" \
        bash -c "set -euo pipefail
            sflow batch -f '$BATCH_BULK_CLI_FILES_FIXTURE_DIR/common.yaml' \
                --bulk-input '$BATCH_BULK_CLI_FILES_FIXTURE_DIR/jobs.csv' \
                -p '$PARTITION' -A '$ACCOUNT' --nodes 1 --log-level warn \
                --output-dir '$BATCH_BULK_CLI_FILES_DIR/out'
            sh_file=\$(find '$BATCH_BULK_CLI_FILES_DIR/out' -name '*.sh' -print -quit)
            test -n \"\$sh_file\"
            common_path=\$(python - '$BATCH_BULK_CLI_FILES_FIXTURE_DIR/common.yaml' <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
            task_path=\$(python - '$BATCH_BULK_CLI_FILES_FIXTURE_DIR/task.yaml' <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
)
            common_arg=\"--file \$common_path\"
            task_arg=\"--file \$task_path\"
            grep -F -- \"\$common_arg\" \"\$sh_file\"
            grep -F -- \"\$task_arg\" \"\$sh_file\"
            python - \"\$sh_file\" \"\$common_arg\" \"\$task_arg\" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
common = sys.argv[2]
task = sys.argv[3]
if text.index(common) > text.index(task):
    raise SystemExit('CLI -f config appears after CSV row config')
PY"

    # -- sflow batch --bulk-input with -e expression: verify per-row resolution --
    if [ -f "$CSV_FILE" ]; then
        BATCH_BULK_EXPR_DIR="$PREFLIGHT_DIR/batch_bulk_input_expr"
        run_check "batch bulk-input -e expression" \
            sflow batch --bulk-input "$CSV_FILE" \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH" \
                -p "$PARTITION" -A "$ACCOUNT" --log-level warn \
                -e '--segment=${{ variables.SLURM_NODES }}' \
                --output-dir "$BATCH_BULK_EXPR_DIR"
        EXPR_FAIL=0
        for sh_file in "$BATCH_BULK_EXPR_DIR"/bulk_input_*/*.sh; do
            [ -f "$sh_file" ] || continue
            if grep -q '#SBATCH --segment=\${{' "$sh_file"; then
                echo "  FAIL: unresolved expression in $(basename "$sh_file")"
                EXPR_FAIL=1
            elif ! grep -q '#SBATCH --segment=[0-9]' "$sh_file"; then
                echo "  FAIL: missing --segment directive in $(basename "$sh_file")"
                EXPR_FAIL=1
            fi
        done
        if [ "$EXPR_FAIL" -eq 0 ]; then
            echo "  PASS: -e expressions resolved per CSV row in bulk-input"
        fi
    fi

    # -- sflow batch --bulk-input with -s overlapping CSV column: CLI --set must win --
    if [ -f "$CSV_FILE" ]; then
        BATCH_BULK_SET_DIR="$PREFLIGHT_DIR/batch_bulk_input_set_precedence"
        BATCH_BULK_SET_OUT="$RESULTS_DIR/batch_bulk_input_set_precedence.stderr"
        run_check "batch bulk-input -s overrides CSV column" \
            bash -c "sflow batch --bulk-input '$CSV_FILE' \
                -a 'LOCAL_MODEL_PATH=fs://$MODEL_PATH' \
                -p '$PARTITION' -A '$ACCOUNT' --log-level warn \
                -s 'GPUS_PER_NODE=77' \
                --output-dir '$BATCH_BULK_SET_DIR' 2> '$BATCH_BULK_SET_OUT'"
    fi

    # -- sflow compose --bulk-input with --set overlapping CSV column: CLI --set must win --
    if [ -f "$CSV_FILE" ]; then
        COMPOSE_BULK_SET_DIR="$PREFLIGHT_DIR/compose_bulk_input_set_precedence"
        COMPOSE_BULK_SET_OUT="$RESULTS_DIR/compose_bulk_input_set_precedence.stderr"
        run_check "compose bulk-input --set overrides CSV column" \
            bash -c "sflow compose --bulk-input '$CSV_FILE' \
                --set 'GPUS_PER_NODE=77' \
                -o '$COMPOSE_BULK_SET_DIR' 2> '$COMPOSE_BULK_SET_OUT'"
    fi

    # -- sflow run --bulk-input --row (dry-run): CSV row execution --
    # Missable tasks are defined in the CSV's missable_tasks column, not via CLI -M.
    if [ -f "$CSV_FILE" ]; then
        run_check "run bulk-input row 1 (dry-run)" \
            sflow run --bulk-input "$CSV_FILE" --row 1 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        run_check "run bulk-input row 3 (dry-run)" \
            sflow run --bulk-input "$CSV_FILE" --row 3 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        run_check "run bulk-input with cli files (dry-run)" \
            sflow run -f "$SLURM_CFG" --bulk-input "$CSV_FILE" --row 1 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        # -- negative index tests for sflow run --
        run_check "run bulk-input last row (--row=-1, dry-run)" \
            sflow run --bulk-input "$CSV_FILE" --row=-1 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        run_check "run bulk-input negative row (--row=-3, dry-run)" \
            sflow run --bulk-input "$CSV_FILE" --row=-3 --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"

        run_check "run bulk-input missing --row (expect fail)" \
            bash -c '! sflow run --bulk-input '"$CSV_FILE"' --dry-run 2>&1'

        run_check "run --row without bulk-input (expect fail)" \
            bash -c '! sflow run --row 1 --dry-run 2>&1'
    else
        echo "  SKIP: CSV not found at $CSV_FILE"
    fi

    # -- sflow visualize --
    run_check "visualize modular vllm/disagg" \
        sflow visualize "$SLURM_CFG" "$COMMON" \
            "$EXAMPLES_DIR/inference_x_v2/vllm/prefill.yaml" \
            "$EXAMPLES_DIR/inference_x_v2/vllm/decode.yaml" \
            "$BENCH_INFMAX" \
            "${MODULAR_MISSABLE[@]}" "${MODULAR_OVERRIDES[@]}" \
            -o "$PREFLIGHT_DIR/visualize_vllm_disagg.png"

    # -- sflow sample --
    run_check "sample list" \
        sflow sample --list
    SAMPLE_SELF_DIR="$PREFLIGHT_DIR/sample_copy_self"
    mkdir -p "$SAMPLE_SELF_DIR"
    run_check "sample copy self-contained" \
        sflow sample local_hello_world \
            --output "$SAMPLE_SELF_DIR/local_hello_world.yaml"
    SAMPLE_MODULAR_DIR="$PREFLIGHT_DIR/sample_copy_modular"
    mkdir -p "$SAMPLE_MODULAR_DIR"
    run_check "sample copy modular" \
        sflow sample inference_x_v2 \
            --output "$SAMPLE_MODULAR_DIR/inference_x_v2"

    fi

    # =====================================================================
    # Wait for all parallel tests and aggregate results
    # =====================================================================
    echo "Launched $TEST_ID tests — waiting for completion..."
    echo ""
    wait

    PASS=0
    FAIL=0
    TOTAL=0
    FAILED_LABELS=""
    FAILED_DETAILS=""
    for result_file in "$RESULTS_DIR"/*.result; do
        [ -f "$result_file" ] || continue
        TOTAL=$((TOTAL + 1))
        id=$(basename "$result_file" .result)
        output_file="$RESULTS_DIR/${id}.output"

        status="" label="" cmd=""
        while IFS='=' read -r key value; do
            case "$key" in
                STATUS) status="$value" ;;
                LABEL)  label="$value" ;;
                CMD)    cmd="$value" ;;
            esac
        done < "$result_file"

        if [ "$status" = "OK" ]; then
            PASS=$((PASS + 1))
            echo "  [$id] $label ... OK"
            echo "       \$ $cmd"
            highlights=$(grep -E 'Output directory:|Scripts directory:|Results CSV:|Bulk (submit|input|compose):|topological order:' "$output_file" 2>/dev/null | head -10 || true)
            if [ -n "$highlights" ]; then
                echo "$highlights" | sed 's/^/       /'
            fi
        else
            FAIL=$((FAIL + 1))
            echo "  [$id] $label ... FAIL"
            echo "       \$ $cmd"
            echo "       Reason (captured output):"
            if [ -s "$output_file" ]; then
                sed -n '1,80p' "$output_file" 2>/dev/null | sed 's/^/       /'
                FAILED_DETAILS="$FAILED_DETAILS\n[$id] $label\n  \$ $cmd\n$(sed -n '1,80p' "$output_file" 2>/dev/null | sed 's/^/  /')\n"
            else
                echo "       (no output captured)"
                FAILED_DETAILS="$FAILED_DETAILS\n[$id] $label\n  \$ $cmd\n  (no output captured)\n"
            fi
            FAILED_LABELS="$FAILED_LABELS  - $label\n"
        fi
    done

    # Save test commands and results to the preflight output directory
    TEST_LOG="$PREFLIGHT_DIR/preflight_test_log.txt"
    {
        echo "# Preflight Test Log"
        echo "# Generated: $(date)"
        echo "# Results: $PASS/$TOTAL passed, $FAIL failed"
        echo ""
    } > "$TEST_LOG"
    for result_file in "$RESULTS_DIR"/*.result; do
        [ -f "$result_file" ] || continue
        id=$(basename "$result_file" .result)
        log_status="" log_label="" log_cmd=""
        while IFS='=' read -r key value; do
            case "$key" in
                STATUS) log_status="$value" ;;
                LABEL)  log_label="$value" ;;
                CMD)    log_cmd="$value" ;;
            esac
        done < "$result_file"
        echo "[$id] $log_status  $log_label" >> "$TEST_LOG"
        echo "  \$ $log_cmd" >> "$TEST_LOG"
        echo "" >> "$TEST_LOG"
    done

    if [ "$TEST_TYPE" != "inf" ]; then

    # -- Post-wait: verify replica sweep resolves per-replica values + domain --
    SFLOW_LOG=$(find "$DOMAIN_RUN_DIR" -name "sflow.log" -print -quit 2>/dev/null)
    if [ -f "$SFLOW_LOG" ]; then
        REPLICA_FAIL=0
        # Verify domain resolved in the command log
        if grep -q 'concurrency_domain=\[1, 4, 16\]' "$SFLOW_LOG"; then
            :  # pass
        else
            echo "  FAIL: sflow.log did not contain resolved concurrency domain list"
            REPLICA_FAIL=1
        fi
        if grep -q 'framework_domain=.*sglang.*vllm.*trtllm' "$SFLOW_LOG"; then
            :  # pass
        else
            echo "  FAIL: sflow.log did not contain resolved framework domain list"
            REPLICA_FAIL=1
        fi
        # Verify per-replica value shift for both sweep variables
        if grep -q "echo concurrency=1$" "$SFLOW_LOG" && grep -q "echo concurrency=16$" "$SFLOW_LOG"; then
            :  # pass
        else
            echo "  FAIL: concurrency replica value shift not found (expected 1 and 16)"
            REPLICA_FAIL=1
        fi
        if grep -q "echo framework=sglang$" "$SFLOW_LOG" && grep -q "echo framework=trtllm$" "$SFLOW_LOG"; then
            :  # pass
        else
            echo "  FAIL: framework replica value shift not found (expected sglang and trtllm)"
            REPLICA_FAIL=1
        fi
        if [ "$REPLICA_FAIL" -eq 0 ]; then
            echo "  PASS: replica sweep resolves per-replica values + domain correctly"
        else
            FAIL=$((FAIL + REPLICA_FAIL))
            TOTAL=$((TOTAL + REPLICA_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - replica sweep value/domain resolution\n"
        fi
    fi

    # -- Post-wait: verify live run summary and command-only command logs --
    SFLOW_SUMMARY_LOG=$(find "$DOMAIN_RUN_DIR" -name "sflow_summary.log" -print -quit 2>/dev/null)
    if [ -f "$SFLOW_SUMMARY_LOG" ]; then
        SUMMARY_FAIL=0
        DOMAIN_WORKFLOW_DIR=$(dirname "$SFLOW_SUMMARY_LOG")
        for summary_pattern in \
            'Sflow Summary' \
            'Runtime' \
            'Workflow DAG' \
            'Timeline' \
            'Task Duration Chart' \
            'Counts       :' \
            'FAILED/CANCELLED Tasks :' \
            'SUBMITTED'; do
            if ! grep -Fq "$summary_pattern" "$SFLOW_SUMMARY_LOG"; then
                echo "  FAIL: sflow_summary.log missing '$summary_pattern'"
                SUMMARY_FAIL=1
            fi
        done
        if ! grep -Eq 'COMPLETED|FAILED|CANCELLED|READY' "$SFLOW_SUMMARY_LOG"; then
            echo "  FAIL: sflow_summary.log missing terminal task event"
            SUMMARY_FAIL=1
        fi
        if grep -Fq 'Traceback' "$SFLOW_SUMMARY_LOG"; then
            echo "  FAIL: sflow_summary.log contains traceback"
            SUMMARY_FAIL=1
        fi

        FIRST_CMD_LOG=$(find "$DOMAIN_WORKFLOW_DIR" -name "*_cmds.log" -print -quit 2>/dev/null)
        if [ -z "$FIRST_CMD_LOG" ]; then
            echo "  FAIL: no command log found under $DOMAIN_WORKFLOW_DIR"
            SUMMARY_FAIL=1
        fi
        BASH_CMD_LOG="$DOMAIN_WORKFLOW_DIR/bash_cmds.log"
        if [ -f "$BASH_CMD_LOG" ]; then
            if ! grep -Fq 'bash -c' "$BASH_CMD_LOG"; then
                echo "  FAIL: bash_cmds.log missing bash -c command"
                SUMMARY_FAIL=1
            fi
            if grep -xFq 'concurrency=1' "$BASH_CMD_LOG" || \
               grep -xFq 'framework=sglang' "$BASH_CMD_LOG"; then
                echo "  FAIL: bash_cmds.log contains raw task output lines"
                SUMMARY_FAIL=1
            fi
        else
            echo "  FAIL: bash_cmds.log missing under $DOMAIN_WORKFLOW_DIR"
            SUMMARY_FAIL=1
        fi

        if [ "$SUMMARY_FAIL" -eq 0 ]; then
            echo "  PASS: sflow_summary.log and command logs generated for live run"
        else
            FAIL=$((FAIL + SUMMARY_FAIL))
            TOTAL=$((TOTAL + SUMMARY_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - sflow summary/command logs\n"
        fi
    else
        echo "  FAIL: sflow_summary.log not found under $DOMAIN_RUN_DIR"
        FAIL=$((FAIL + 1))
        TOTAL=$((TOTAL + 1))
        FAILED_LABELS="$FAILED_LABELS  - sflow summary log missing\n"
    fi

    # -- Post-wait: verify ${{ variables.X.domain }} resolved in batch -e --
    BATCH_DOMAIN_SCRIPT="$BATCH_DOMAIN_DIR/domain_test.sh"
    if [ -f "$BATCH_DOMAIN_SCRIPT" ]; then
        if grep -q '#SBATCH --comment=\[1, 4, 16\]' "$BATCH_DOMAIN_SCRIPT"; then
            echo "  PASS: batch -e variables.X.domain resolved to [1, 4, 16]"
        else
            echo "  FAIL: batch -e variables.X.domain not resolved in sbatch script"
            grep '#SBATCH --comment' "$BATCH_DOMAIN_SCRIPT" || echo "    (no --comment directive found)"
            FAIL=$((FAIL + 1))
            TOTAL=$((TOTAL + 1))
            FAILED_LABELS="$FAILED_LABELS  - batch -e variables.X.domain resolution\n"
        fi
    fi

    # -- Post-wait: verify default batch install version follows current env --
    BATCH_DEFAULT_SCRIPT="$BATCH_DEFAULT_VERSION_DIR/default_version.sh"
    if [ -f "$BATCH_DEFAULT_SCRIPT" ]; then
        expected_ref="git+https://github.com/NVIDIA/nv-sflow.git@$EXPECTED_BATCH_SFLOW_VERSION"
        if grep -Fq "$expected_ref" "$BATCH_DEFAULT_SCRIPT"; then
            echo "  PASS: batch default --sflow-version resolved to $EXPECTED_BATCH_SFLOW_VERSION"
        else
            echo "  FAIL: batch default --sflow-version did not resolve to $EXPECTED_BATCH_SFLOW_VERSION"
            grep -F 'git+https://github.com/NVIDIA/nv-sflow.git@' "$BATCH_DEFAULT_SCRIPT" || \
                echo "    (no nv-sflow install line found)"
            FAIL=$((FAIL + 1))
            TOTAL=$((TOTAL + 1))
            FAILED_LABELS="$FAILED_LABELS  - batch default --sflow-version resolution\n"
        fi
    fi

    # -- Post-wait: verify ${{ variables.X.domain }} resolved correctly --
    DOMAIN_RESOLVED="$COMPOSE_DOMAIN_DIR/resolved.yaml"
    if [ -f "$DOMAIN_RESOLVED" ]; then
        DOMAIN_FAIL=0
        if grep -q '\[1, 4, 16\]' "$DOMAIN_RESOLVED"; then
            echo "  PASS: variables.CONCURRENCY.domain resolved to [1, 4, 16]"
        else
            echo "  FAIL: variables.CONCURRENCY.domain not resolved in compose output"
            DOMAIN_FAIL=1
        fi
        if grep -q "sglang.*vllm.*trtllm" "$DOMAIN_RESOLVED"; then
            echo "  PASS: variables.FRAMEWORK.domain resolved to framework list"
        else
            echo "  FAIL: variables.FRAMEWORK.domain not resolved in compose output"
            DOMAIN_FAIL=1
        fi
        if [ "$DOMAIN_FAIL" -gt 0 ]; then
            FAIL=$((FAIL + DOMAIN_FAIL))
            TOTAL=$((TOTAL + DOMAIN_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - variables.X.domain resolution\n"
        fi
    fi

    # -- Post-wait: verify compose -r rewrites resolved vars inside deferred Jinja --
    COMPOSE_DEFERRED_RESOLVED="$PREFLIGHT_DIR/compose_deferred_jinja/resolved.yaml"
    COMPOSE_DEFERRED_FAIL=0
    if [ ! -f "$COMPOSE_DEFERRED_RESOLVED" ]; then
        echo "  FAIL: compose deferred-Jinja e2e output missing"
        COMPOSE_DEFERRED_FAIL=1
    else
        if grep -q 'variables.INFRA_NODE_INDEX' "$COMPOSE_DEFERRED_RESOLVED"; then
            echo "  FAIL: compose deferred-Jinja output still references variables.INFRA_NODE_INDEX"
            COMPOSE_DEFERRED_FAIL=1
        fi
        if grep -q 'if 0 == 0' "$COMPOSE_DEFERRED_RESOLVED"; then
            echo "  PASS: compose -r rewrote resolved vars inside deferred Jinja"
        else
            echo "  FAIL: compose deferred-Jinja output did not inline the resolved literal"
            COMPOSE_DEFERRED_FAIL=1
        fi
    fi
    if [ "$COMPOSE_DEFERRED_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + COMPOSE_DEFERRED_FAIL))
        TOTAL=$((TOTAL + COMPOSE_DEFERRED_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - compose deferred-Jinja resolution\n"
    fi

    # -- Post-wait: verify compose -r resolves nodes.indices expression strings to a YAML list value --
    COMPOSE_INDICES_RESOLVED="$PREFLIGHT_DIR/compose_indices_expression/resolved.yaml"
    COMPOSE_INDICES_FAIL=0
    if [ ! -f "$COMPOSE_INDICES_RESOLVED" ]; then
        echo "  FAIL: compose nodes.indices e2e output missing"
        COMPOSE_INDICES_FAIL=1
    else
        export COMPOSE_INDICES_RESOLVED
        if python - <<'PY'
import os
from pathlib import Path
import yaml

resolved_path = Path(os.environ["COMPOSE_INDICES_RESOLVED"])
data = yaml.safe_load(resolved_path.read_text())
tasks = {task["name"]: task for task in data["workflow"]["tasks"]}
indices = tasks["frontend_server"]["resources"]["nodes"]["indices"]
exclude = tasks["worker_server"]["resources"]["nodes"]["exclude"]
assert indices in ("[0, 1]", [0, 1]), indices
assert exclude in ("[0, 1]", [0, 1]), exclude
PY
        then
            echo "  PASS: compose -r resolves resources.nodes.indices/exclude expression strings to [0, 1]"
        else
            echo "  FAIL: compose nodes.indices/exclude output did not resolve to [0, 1]"
            COMPOSE_INDICES_FAIL=1
        fi
    fi
    if [ "$COMPOSE_INDICES_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + COMPOSE_INDICES_FAIL))
        TOTAL=$((TOTAL + COMPOSE_INDICES_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - compose nodes.indices/exclude expression resolution\n"
    fi

    # -- Post-wait: verify dry-run assigns nodes from indices+count in the configured order --
    COMPOSE_INDICES_DRYRUN_FAIL=0
    if [ ! -f "$COMPOSE_INDICES_DRYRUN_LOG" ]; then
        echo "  FAIL: dry-run nodes.indices/count log missing"
        COMPOSE_INDICES_DRYRUN_FAIL=1
    else
        export COMPOSE_INDICES_DRYRUN_LOG
        if python - <<'PY'
import os
import re
from pathlib import Path

text = Path(os.environ["COMPOSE_INDICES_DRYRUN_LOG"]).read_text()
expected = {
    "ordered_pool_0": "slurm_cluster-node3",
    "ordered_pool_1": "slurm_cluster-node0",
    "ordered_pool_2": "slurm_cluster-node1",
    "ordered_pool_3": "slurm_cluster-node2",
}
for task_name, node_name in expected.items():
    pattern = rf"\[\d+\]\s+{re.escape(task_name)}.*?nodelist:\s+\['{re.escape(node_name)}'\]"
    assert re.search(pattern, text, re.S), (task_name, node_name)
PY
        then
            echo "  PASS: dry-run assigns indices+count replicas in configured order"
        else
            echo "  FAIL: dry-run did not preserve indices+count ordering"
            COMPOSE_INDICES_DRYRUN_FAIL=1
        fi
    fi
    if [ "$COMPOSE_INDICES_DRYRUN_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + COMPOSE_INDICES_DRYRUN_FAIL))
        TOTAL=$((TOTAL + COMPOSE_INDICES_DRYRUN_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - dry-run nodes.indices+count ordering\n"
    fi

    # -- Post-wait: verify a ':' in an unquoted task script command remains a string --
    COLON_SCRIPT_PREFLIGHT_FAIL=0
    for colon_output in "$COLON_SCRIPT_COMPOSED" "$COLON_SCRIPT_BATCH_CONFIG"; do
        if [ ! -f "$colon_output" ]; then
            echo "  FAIL: colon task script output missing: $colon_output"
            COLON_SCRIPT_PREFLIGHT_FAIL=1
            continue
        fi
        export COLON_SCRIPT_OUTPUT="$colon_output"
        if python - <<'PY'
import os
from pathlib import Path
import yaml

config_path = Path(os.environ["COLON_SCRIPT_OUTPUT"])
data = yaml.safe_load(config_path.read_text())
tasks = {task["name"]: task for task in data["workflow"]["tasks"]}
assert tasks["worker"]["script"][0] == 'echo "My GPUs: $CUDA_VISIBLE_DEVICES"'
PY
        then
            echo "  PASS: colon in task script is preserved as a plain command string in $(basename "$colon_output")"
        else
            echo "  FAIL: colon in task script was not preserved as a command string in $(basename "$colon_output")"
            COLON_SCRIPT_PREFLIGHT_FAIL=1
        fi
    done
    if [ "$COLON_SCRIPT_PREFLIGHT_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + COLON_SCRIPT_PREFLIGHT_FAIL))
        TOTAL=$((TOTAL + COLON_SCRIPT_PREFLIGHT_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - colon in task script preservation\n"
    fi

    # -- Post-wait: verify readiness probe list appears as two readiness checks in dry-run plan --
    READINESS_AND_FAIL=0
    if [ ! -f "$READINESS_AND_DRYRUN_LOG" ]; then
        echo "  FAIL: readiness probe list dry-run log missing"
        READINESS_AND_FAIL=1
    else
        if ! grep -q 'readiness: log_watch (pattern=readiness one)' "$READINESS_AND_DRYRUN_LOG"; then
            echo "  FAIL: first readiness probe missing from dry-run plan"
            READINESS_AND_FAIL=1
        fi
        if ! grep -q 'readiness: log_watch (pattern=readiness two)' "$READINESS_AND_DRYRUN_LOG"; then
            echo "  FAIL: second readiness probe missing from dry-run plan"
            READINESS_AND_FAIL=1
        fi
        if [ "$READINESS_AND_FAIL" -eq 0 ]; then
            echo "  PASS: readiness probe list expands to multiple readiness checks"
        fi
    fi
    if [ "$READINESS_AND_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + READINESS_AND_FAIL))
        TOTAL=$((TOTAL + READINESS_AND_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - readiness probe list dry-run expansion\n"
    fi

    # -- Post-wait: verify old single readiness probe object still works --
    READINESS_SINGLE_FAIL=0
    if [ ! -f "$READINESS_SINGLE_DRYRUN_LOG" ]; then
        echo "  FAIL: single readiness probe dry-run log missing"
        READINESS_SINGLE_FAIL=1
    else
        if ! grep -q 'readiness: log_watch (pattern=single readiness)' "$READINESS_SINGLE_DRYRUN_LOG"; then
            echo "  FAIL: single readiness probe missing from dry-run plan"
            READINESS_SINGLE_FAIL=1
        fi
        if grep -q 'ValidationError\|Traceback' "$READINESS_SINGLE_DRYRUN_LOG"; then
            echo "  FAIL: single readiness probe dry-run emitted validation error"
            READINESS_SINGLE_FAIL=1
        fi
        if [ "$READINESS_SINGLE_FAIL" -eq 0 ]; then
            echo "  PASS: single readiness probe object remains compatible"
        fi
    fi
    if [ "$READINESS_SINGLE_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + READINESS_SINGLE_FAIL))
        TOTAL=$((TOTAL + READINESS_SINGLE_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - single readiness probe compatibility\n"
    fi

    # -- Post-wait: verify sflow sample copy flows --
    SAMPLE_SELF_OUT="$PREFLIGHT_DIR/sample_copy_self/local_hello_world.yaml"
    SAMPLE_MODULAR_OUT="$PREFLIGHT_DIR/sample_copy_modular/inference_x_v2"
    SAMPLE_COPY_FAIL=0
    if [ -s "$SAMPLE_SELF_OUT" ]; then
        echo "  PASS: sample copied self-contained workflow to custom output path"
    else
        echo "  FAIL: sample self-contained copy missing or empty"
        SAMPLE_COPY_FAIL=1
    fi
    if [ -d "$SAMPLE_MODULAR_OUT" ] && \
       [ -f "$SAMPLE_MODULAR_OUT/slurm_config.yaml" ] && \
       [ -f "$SAMPLE_MODULAR_OUT/bulk_input.csv" ]; then
        echo "  PASS: sample copied modular workflow folder with key files"
    else
        echo "  FAIL: sample modular copy missing expected files"
        SAMPLE_COPY_FAIL=1
    fi
    if [ "$SAMPLE_COPY_FAIL" -gt 0 ]; then
        FAIL=$((FAIL + SAMPLE_COPY_FAIL))
        TOTAL=$((TOTAL + SAMPLE_COPY_FAIL))
        FAILED_LABELS="$FAILED_LABELS  - sample copy flows\n"
    fi

    # -- Post-wait: verify CLI --set wins over CSV column in bulk-input --
    # batch: generated sbatch scripts must call `sflow run --set GPUS_PER_NODE=77`
    # and must NOT pass the CSV value (GPUS_PER_NODE=4) for that variable.
    if [ -d "$BATCH_BULK_SET_DIR" ]; then
        SET_FAIL=0
        scripts_found=0
        for sh_file in "$BATCH_BULK_SET_DIR"/bulk_input_*/*.sh; do
            [ -f "$sh_file" ] || continue
            scripts_found=$((scripts_found + 1))
            if ! grep -q -- '--set GPUS_PER_NODE=77' "$sh_file"; then
                echo "  FAIL: CLI --set GPUS_PER_NODE=77 missing in $(basename "$sh_file")"
                SET_FAIL=1
            fi
            if grep -q -- '--set GPUS_PER_NODE=4\b' "$sh_file"; then
                echo "  FAIL: CSV GPUS_PER_NODE=4 not overridden in $(basename "$sh_file")"
                SET_FAIL=1
            fi
        done
        if [ "$scripts_found" -eq 0 ]; then
            echo "  FAIL: no scripts generated in $BATCH_BULK_SET_DIR"
            SET_FAIL=1
        fi
        if [ -f "$BATCH_BULK_SET_OUT" ] && \
           ! grep -q "CLI --set value will take precedence" "$BATCH_BULK_SET_OUT"; then
            echo "  FAIL: expected 'CLI --set value will take precedence' warning (batch)"
            SET_FAIL=1
        fi
        if [ "$SET_FAIL" -eq 0 ]; then
            echo "  PASS: batch bulk-input --set overrides CSV column (CLI wins)"
        else
            FAIL=$((FAIL + SET_FAIL))
            TOTAL=$((TOTAL + SET_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - batch bulk-input --set precedence\n"
        fi
    fi

    # compose: merged YAMLs must carry the CLI value for GPUS_PER_NODE (77), not 4.
    if [ -d "$COMPOSE_BULK_SET_DIR" ]; then
        SET_FAIL=0
        yamls_found=0
        for yaml_file in "$COMPOSE_BULK_SET_DIR"/compose_*/*.yaml; do
            [ -f "$yaml_file" ] || continue
            yamls_found=$((yamls_found + 1))
            # Extract GPUS_PER_NODE variable block: expect value '77' from CLI, not 4 from CSV.
            gpn_value=$(awk '
                /name: GPUS_PER_NODE/ {found=1; next}
                found && /value:/ {
                    sub(/.*value:[[:space:]]*/, "")
                    gsub(/["'\'']/, "")
                    print
                    exit
                }
            ' "$yaml_file")
            if [ "$gpn_value" != "77" ]; then
                echo "  FAIL: GPUS_PER_NODE expected 77 (CLI), got '$gpn_value' in $(basename "$yaml_file")"
                SET_FAIL=1
            fi
        done
        if [ "$yamls_found" -eq 0 ]; then
            echo "  FAIL: no yamls generated in $COMPOSE_BULK_SET_DIR"
            SET_FAIL=1
        fi
        if [ -f "$COMPOSE_BULK_SET_OUT" ] && \
           ! grep -q "CLI --set value will take precedence" "$COMPOSE_BULK_SET_OUT"; then
            echo "  FAIL: expected 'CLI --set value will take precedence' warning (compose)"
            SET_FAIL=1
        fi
        if [ "$SET_FAIL" -eq 0 ]; then
            echo "  PASS: compose bulk-input --set overrides CSV column (CLI wins)"
        else
            FAIL=$((FAIL + SET_FAIL))
            TOTAL=$((TOTAL + SET_FAIL))
            FAILED_LABELS="$FAILED_LABELS  - compose bulk-input --set precedence\n"
        fi
    fi

    # -- Post-wait: verify sflow_batch_dir column in results.csv --
    for mode in batch_bulk_submit batch_bulk_input; do
        csv_file=$(find "$PREFLIGHT_DIR/$mode" -name results.csv -print -quit 2>/dev/null)
        if [ -f "$csv_file" ]; then
            csv_file=$(abs_path "$csv_file")
            if head -1 "$csv_file" | grep -q "sflow_batch_dir"; then
                bulk_dir=$(basename "$(dirname "$csv_file")")
                if grep -q "$bulk_dir" "$csv_file"; then
                    echo "  PASS: sflow_batch_dir column present and correct in $csv_file"
                else
                    echo "  FAIL: sflow_batch_dir value mismatch in $csv_file"
                    FAIL=$((FAIL + 1))
                    TOTAL=$((TOTAL + 1))
                    FAILED_LABELS="$FAILED_LABELS  - sflow_batch_dir value mismatch ($mode)\n"
                fi
            else
                echo "  FAIL: sflow_batch_dir column missing from $csv_file"
                FAIL=$((FAIL + 1))
                TOTAL=$((TOTAL + 1))
                FAILED_LABELS="$FAILED_LABELS  - sflow_batch_dir column missing ($mode)\n"
            fi
        fi
    done

    fi

    echo ""
    echo "===== Preflight Summary: $PASS/$TOTAL passed, $FAIL failed ====="
    echo ""
    echo "===== Results Directory: $PREFLIGHT_DIR ====="
    file_count=$(find "$PREFLIGHT_DIR" -type f | wc -l)
    echo "  ($file_count file(s) total)"
    echo ""

    if [ -n "$PREFLIGHT_SKIP_NOTES" ]; then
        echo "Skipped checks:"
        echo -e "$PREFLIGHT_SKIP_NOTES"
    fi

    if [ "$FAIL" -gt 0 ]; then
        echo "Failed tests:"
        echo -e "$FAILED_LABELS"
        echo "Failure details:"
        echo -e "$FAILED_DETAILS"
        echo "ERROR: $FAIL preflight check(s) failed — aborting before job submission."
        exit 1
    fi

fi

# =========================================================================
# Real e2e tests (submit jobs to Slurm)
# =========================================================================
if [ -n "$SUBMIT" ] && [ -z "$PREFLIGHT_ONLY" ]; then
    echo ""
    echo "===== All preflight checks passed — proceeding to job submission ====="
    echo ""
    set -x
    cd "$SCRIPT_DIR/../tests/e2e_tests"
    E2E_PARTITION="${CLI_PARTITION:-my_partition}"
    E2E_ACCOUNT="${CLI_ACCOUNT:-user}"
    if [ -n "${COLON_SCRIPT_FIXTURE:-}" ] && [ -f "$COLON_SCRIPT_FIXTURE" ]; then
        SFLOW_COLON_SCRIPT_FIXTURE="$COLON_SCRIPT_FIXTURE" \
        SFLOW_COLON_SCRIPT_OUTPUT_DIR="$PWD/sflow_output/colon_in_task_script" \
            ./sample_test.sh -p "$E2E_PARTITION" -A "$E2E_ACCOUNT" -m "$MODEL_PATH" -t "$TEST_TYPE" --submit -- "-e --exclude=gb-nvl-137-compute09,gb-nvl-137-compute16" # 09 has some GPU issues
    else
        ./sample_test.sh -p "$E2E_PARTITION" -A "$E2E_ACCOUNT" -m "$MODEL_PATH" -t "$TEST_TYPE" --submit -- "-e --exclude=gb-nvl-137-compute09,gb-nvl-137-compute16" # 09 has some GPU issues
    fi

    set +x
elif [ -z "$SUBMIT" ]; then
    echo "Preflight only (no -S flag). To submit jobs, re-run with -S."
else
    echo "Preflight only (-P flag). Skipping job submission."
fi
