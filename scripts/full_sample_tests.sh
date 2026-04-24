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
EXPECTED_BATCH_SFLOW_VERSION=$(python - <<'PY'
from sflow.cli.batch import _resolve_effective_sflow_version

print(_resolve_effective_sflow_version(None) or "main")
PY
)

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
    nodes: 2
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
            if head -1 "$csv_file" | grep -q "sflow_batch_dir"; then
                bulk_dir=$(basename "$(dirname "$csv_file")")
                if grep -q "$bulk_dir" "$csv_file"; then
                    echo "  PASS: sflow_batch_dir column present and correct in $mode/results.csv"
                else
                    echo "  FAIL: sflow_batch_dir value mismatch in $mode/results.csv"
                    FAIL=$((FAIL + 1))
                    TOTAL=$((TOTAL + 1))
                    FAILED_LABELS="$FAILED_LABELS  - sflow_batch_dir value mismatch ($mode)\n"
                fi
            else
                echo "  FAIL: sflow_batch_dir column missing from $mode/results.csv"
                FAIL=$((FAIL + 1))
                TOTAL=$((TOTAL + 1))
                FAILED_LABELS="$FAILED_LABELS  - sflow_batch_dir column missing ($mode)\n"
            fi
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
