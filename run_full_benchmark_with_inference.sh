#!/bin/bash
# Full benchmark suite: Training + Inference on trained models
# 1. Trains models using PyTorch and JAX
# 2. Saves checkpoints
# 3. Runs inference benchmarks on those checkpoints to measure efficiency

source /home/lsanc68/pvi-ml/.venv/bin/activate
export HF_HOME=/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface
export HF_DATASETS_CACHE=/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface/hub

# Configuration
MODELS="resnet50 vit_b_16 mobilenet_v3_small efficientnet_b0"
FRAMEWORKS="pytorch jax"
BATCH_SIZES="32 64 128"
EPOCHS=2
LR=1e-4
DATASET="clane9/imagenet-100"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"

echo "========================================================================="
echo "FULL BENCHMARK SUITE (TRAINING + INFERENCE)"
echo "========================================================================="
echo "Run ID: $RUN_ID"
echo "Models: $MODELS"
echo "Frameworks: $FRAMEWORKS"
echo "========================================================================="
echo ""

# Create base results directory
mkdir -p results/training/$RUN_ID

total_configs=0
completed_configs=0

# Count total configurations
for framework in $FRAMEWORKS; do
    for model in $MODELS; do
        for batch_size in $BATCH_SIZES; do
            ((total_configs++))
        done
    done
done

echo "Total configurations: $total_configs"
echo ""

for framework in $FRAMEWORKS; do
    for model in $MODELS; do
        for batch_size in $BATCH_SIZES; do
            ((completed_configs++))

            echo "========================================="
            echo "[$completed_configs/$total_configs] Training: $framework / $model / BS=$batch_size"
            echo "========================================="

            # 1. Run Training
            python bench/training_runner.py \
                --framework $framework \
                --model $model \
                --dataset $DATASET \
                --batch-size $batch_size \
                --epochs $EPOCHS \
                --lr $LR \
                --run-id $RUN_ID \
                --save-checkpoints \
                --output-dir results/training

            if [ $? -ne 0 ]; then
                echo "✗ Training failed"
                continue
            fi

            # Find the session directory (it contains the timestamp, so we search for it)
            # Structure: results/training/<run_id>/<model>/<framework>_bs<batch_size>_<timestamp>
            SESSION_DIR=$(find results/training/$RUN_ID/$model -name "${framework}_bs${batch_size}_*" -type d | sort | tail -n 1)
            
            if [ -z "$SESSION_DIR" ]; then
                echo "Could not find session directory for inference."
                continue
            fi

            echo "Session Directory: $SESSION_DIR"

            # Determine checkpoint path
            if [ "$framework" == "pytorch" ]; then
                CHECKPOINT="$SESSION_DIR/checkpoint_epoch_${EPOCHS}.pt"
            else
                CHECKPOINT="$SESSION_DIR/checkpoint_epoch_${EPOCHS}"
            fi

            if [ ! -e "$CHECKPOINT" ] && [ ! -d "$CHECKPOINT" ]; then
                 echo "Checkpoint not found at $CHECKPOINT"
                 continue
            fi

            echo "Found checkpoint: $CHECKPOINT"
            echo "Running Inference Benchmark..."

            # 2. Run Inference on the trained checkpoint
            # We use the same batch size for inference as training for consistency, 
            # but usually inference is done with different batch sizes. 
            # The user asked to "run the models for inference again", implying a sweep or specific test.
            # Let's run a sweep of batch sizes for inference to see efficiency.
            
            python bench/runner.py \
                --framework $framework \
                --model $model \
                --batch-sizes 1 32 128 \
                --checkpoint "$CHECKPOINT" \
                --output-dir "$SESSION_DIR" \
                --verbose

            echo "✓ Completed Training & Inference for $framework / $model / BS=$batch_size"
            echo ""
        done
    done
done

echo "========================================================================="
echo "BENCHMARK SUITE COMPLETE!"
echo "========================================================================="
echo "Results stored under: results/training/$RUN_ID"
