#!/bin/bash
# Launch all experimental conditions in parallel across 8 GPUs.
#
# 3 ft methods × 3 models × 4 contamination levels = 36 jobs
# 8 GPUs available → round-robin assignment, wait when full
#
# Usage:
#   cd ~/final_proj && bash scripts/launch_parallel.sh [--ft_methods lora8,lora256,full]

set -e

OUTPUT_DIR="$HOME/final_proj/outputs/gpu_full_run"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Parse optional --ft_methods argument (default: all three)
FT_METHODS="lora8,lora256,full"
while [[ $# -gt 0 ]]; do
    case $1 in
        --ft_methods) FT_METHODS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
IFS=',' read -ra FT_ARRAY <<< "$FT_METHODS"

MODELS=(
    "EleutherAI/pythia-70m"
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-410m"
)
CONTAM_EPOCHS=(0 1 5 10)

TOTAL=$(( ${#FT_ARRAY[@]} * ${#MODELS[@]} * ${#CONTAM_EPOCHS[@]} ))
echo "$(date): Launching $TOTAL conditions across 8 GPUs"
echo "FT methods: ${FT_ARRAY[*]}"
echo "Output: $OUTPUT_DIR"
echo "Logs:   $LOG_DIR"
echo ""

# ── Step 1: Prepare data (MUST complete before parallel jobs) ────
echo "$(date): Preparing shared data splits..."
conda run -n cdd python scripts/prepare_data.py --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/prepare_data.log"
if [ $? -ne 0 ]; then
    echo "ERROR: Data preparation failed! Check $LOG_DIR/prepare_data.log"
    exit 1
fi
echo "$(date): Data preparation complete."
echo ""

# ── Step 2: Launch parallel jobs ─────────────────────────────────
GPU=0
PIDS=()

for FT in "${FT_ARRAY[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        SHORT=$(echo "$MODEL" | sed 's/.*pythia-//' | tr '[:lower:]' '[:upper:]')
        for CE in "${CONTAM_EPOCHS[@]}"; do
            CONDITION="${FT}_${SHORT}_contam${CE}"
            GPU_ID=$((GPU % 8))

            echo "$(date): Launching $CONDITION on GPU $GPU_ID"

            CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n cdd python scripts/run_single_condition.py \
                --model "$MODEL" \
                --contam_epochs "$CE" \
                --ft_method "$FT" \
                --output_dir "$OUTPUT_DIR" \
                > "$LOG_DIR/${CONDITION}.log" 2>&1 &

            PIDS+=($!)
            GPU=$((GPU + 1))

            # If we've filled all 8 GPUs, wait for any one to finish
            if [ ${#PIDS[@]} -ge 8 ]; then
                echo "$(date): 8 GPUs busy, waiting for a slot..."
                wait -n
                NEW_PIDS=()
                for PID in "${PIDS[@]}"; do
                    if kill -0 "$PID" 2>/dev/null; then
                        NEW_PIDS+=("$PID")
                    fi
                done
                PIDS=("${NEW_PIDS[@]}")
            fi
        done
    done
done

echo ""
echo "$(date): All $TOTAL jobs launched. Waiting for completion..."
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/*.log"
echo "  nvidia-smi"
echo ""

# Wait for all remaining jobs
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "$(date): All jobs finished. Failed: $FAILED / $TOTAL"

# Print summary
echo ""
echo "=== SUMMARY ==="
for LOG in "$LOG_DIR"/*.log; do
    COND=$(basename "$LOG" .log)
    [ "$COND" = "prepare_data" ] && continue
    STATUS=$(grep -c "COMPLETE" "$LOG" 2>/dev/null || echo "0")
    if [ "$STATUS" -gt 0 ]; then
        echo "  ✓ $COND"
    else
        echo "  ✗ $COND (check $LOG)"
    fi
done

echo ""
echo "$(date): Done. Run scripts/run_aggregate.py next."
