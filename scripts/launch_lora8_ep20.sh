#!/bin/bash
# Launch 12 conditions for LoRA r=8 with 20 training epochs across 8 GPUs.
# 3 models × 4 contamination levels = 12 jobs
#
# Usage:
#   cd ~/final_proj && nohup bash scripts/launch_lora8_ep20.sh > outputs/gpu_full_run/launch_lora8_ep20.log 2>&1 &

set -e

OUTPUT_DIR="$HOME/final_proj/outputs/gpu_full_run"
LOG_DIR="$OUTPUT_DIR/logs"
FT="lora8"
EPOCHS=20
TAG="lora8_ep20"
mkdir -p "$LOG_DIR"

MODELS=(
    "EleutherAI/pythia-70m"
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-410m"
)
CONTAM_EPOCHS=(0 1 5 10)

TOTAL=$(( ${#MODELS[@]} * ${#CONTAM_EPOCHS[@]} ))
echo "$(date): Launching $TOTAL conditions for $TAG across 8 GPUs"

# Data should already be prepared
if [ ! -d "$OUTPUT_DIR/data/evaluation" ]; then
    echo "$(date): Preparing data..."
    conda run -n cdd python scripts/prepare_data.py --output_dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/prepare_data.log"
fi

GPU=0
PIDS=()

for MODEL in "${MODELS[@]}"; do
    SHORT=$(echo "$MODEL" | sed 's/.*pythia-//' | tr '[:lower:]' '[:upper:]')
    for CE in "${CONTAM_EPOCHS[@]}"; do
        CONDITION="${TAG}_${SHORT}_contam${CE}"
        GPU_ID=$((GPU % 8))

        echo "$(date): Launching $CONDITION on GPU $GPU_ID"

        CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n cdd python scripts/run_single_condition.py \
            --model "$MODEL" \
            --contam_epochs "$CE" \
            --ft_method "$FT" \
            --train_epochs "$EPOCHS" \
            --output_dir "$OUTPUT_DIR" \
            > "$LOG_DIR/${CONDITION}.log" 2>&1 &

        PIDS+=($!)
        GPU=$((GPU + 1))

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

echo ""
echo "$(date): All $TOTAL $TAG jobs launched. Waiting..."

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "$(date): $TAG done. Failed: $FAILED / $TOTAL"
echo "=== SUMMARY ==="
for LOG in "$LOG_DIR"/${TAG}_*.log; do
    COND=$(basename "$LOG" .log)
    STATUS=$(grep -c "COMPLETE" "$LOG" 2>/dev/null || echo "0")
    if [ "$STATUS" -gt 0 ]; then
        echo "  ✓ $COND"
    else
        echo "  ✗ $COND"
    fi
done
