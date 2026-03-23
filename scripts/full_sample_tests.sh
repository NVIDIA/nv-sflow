#!/bin/bash

set -uo pipefail

TEST_TYPE="a"
SUBMIT=""
PREFLIGHT_ONLY=""
MAX_JOBS=16
CLI_MODEL_PATH=""
CLI_PARTITION=""
CLI_ACCOUNT=""
while getopts "asmSPj:M:p:A:" opt; do
    case "$opt" in
        a) TEST_TYPE="a" ;;
        s) TEST_TYPE="s" ;;
        m) TEST_TYPE="m" ;;
        S) SUBMIT="--submit" ;;
        P) PREFLIGHT_ONLY="1" ;;
        j) MAX_JOBS="$OPTARG" ;;
        M) CLI_MODEL_PATH="$OPTARG" ;;
        p) CLI_PARTITION="$OPTARG" ;;
        A) CLI_ACCOUNT="$OPTARG" ;;
        *) echo "Usage: $0 [-a|-s|-m] [-S] [-P] [-j N] [-M model_path] [-p partition] [-A account]"
           echo "  -a  all tests (default)"
           echo "  -s  self-contained examples only"
           echo "  -m  modular examples only"
           echo "  -S  submit jobs to Slurm"
           echo "  -P  preflight checks only (skip job submission even if -S is set)"
           echo "  -j  max parallel jobs (default: 16, 0 for unlimited)"
           echo "  -M  model path (default: \$MODEL_PATH or /home/)"
           echo "  -p  Slurm partition (default: dummy_part for preflight, my_partition for e2e)"
           echo "  -A  Slurm account (default: dummy_acct for preflight, user for e2e)"
           exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."
EXAMPLES_DIR="$REPO_DIR/examples"
CSV_FILE="$EXAMPLES_DIR/inference_x_v2/bulk_input.csv"
MODEL_PATH="${CLI_MODEL_PATH:-${MODEL_PATH:-/home/}}"
PARTITION="${CLI_PARTITION:-dummy_part}"
ACCOUNT="${CLI_ACCOUNT:-dummy_acct}"

STAMP=$(date +%Y%m%d-%H%M%S)
PREFLIGHT_DIR="$REPO_DIR/sflow_output/preflight_$STAMP"
mkdir -p "$PREFLIGHT_DIR"

RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT
TEST_ID=0

throttle() {
    if [ "$MAX_JOBS" -gt 0 ]; then
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 0.1
        done
    fi
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

    # -- sflow run --dry-run: local examples --
    run_check "local_hello_world" \
        sflow run "$EXAMPLES_DIR/local_hello_world.yaml" --dry-run
    run_check "local_dag" \
        sflow run "$EXAMPLES_DIR/local_dag.yaml" --dry-run

    # -- sflow run --dry-run: self-contained slurm examples --
    for f in "$EXAMPLES_DIR"/slurm_*.yaml; do
        name=$(basename "$f" .yaml)
        run_check "dry-run $name" \
            sflow run "$f" --dry-run \
                -a "LOCAL_MODEL_PATH=fs://$MODEL_PATH"
    done

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
    COMPOSE_MISSABLE=(-M agg_server -M prefill_server -M decode_server)
    for framework in trtllm sglang vllm; do
        run_check "compose modular $framework/disagg" \
            sflow compose "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/inference_x_v2/$framework/prefill.yaml" \
                "$EXAMPLES_DIR/inference_x_v2/$framework/decode.yaml" \
                "${COMPOSE_MISSABLE[@]}" -r -vl \
                -o "$COMPOSE_MODULAR_DIR/${framework}_disagg.yaml"
        run_check "compose modular $framework/agg" \
            sflow compose "$SLURM_CFG" "$COMMON" \
                "$EXAMPLES_DIR/inference_x_v2/$framework/agg.yaml" \
                "${COMPOSE_MISSABLE[@]}" -r -vl \
                -o "$COMPOSE_MODULAR_DIR/${framework}_agg.yaml"
    done

    # -- sflow compose --bulk-input (CSV) --
    if [ -f "$CSV_FILE" ]; then
        run_check "compose bulk-input all rows" \
            sflow compose -b "$CSV_FILE" -o "$PREFLIGHT_DIR/compose_bulk_input"
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
            head -20 "$output_file" 2>/dev/null | sed 's/^/       /'
            FAILED_LABELS="$FAILED_LABELS  - $label\n"
        fi
    done

    echo ""
    echo "===== Preflight Summary: $PASS/$TOTAL passed, $FAIL failed ====="
    echo ""
    echo "===== Results Directory: $PREFLIGHT_DIR ====="
    file_count=$(find "$PREFLIGHT_DIR" -type f | wc -l)
    if command -v tree &>/dev/null; then
        tree --noreport "$PREFLIGHT_DIR" | sed 's/^/  /'
    else
        find "$PREFLIGHT_DIR" -type f | sort | sed "s|^$PREFLIGHT_DIR/|  |"
    fi
    echo "  ($file_count file(s) total)"
    echo ""

    if [ "$FAIL" -gt 0 ]; then
        echo "Failed tests:"
        echo -e "$FAILED_LABELS"
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
    ./sample_test.sh -p "$E2E_PARTITION" -A "$E2E_ACCOUNT" -m "$MODEL_PATH" -t "$TEST_TYPE" --submit -- "-e --exclude=gb-nvl-137-compute09,gb-nvl-137-compute16" # 09 has some GPU issues
elif [ -z "$SUBMIT" ]; then
    echo "Preflight only (no -S flag). To submit jobs, re-run with -S."
else
    echo "Preflight only (-P flag). Skipping job submission."
fi
