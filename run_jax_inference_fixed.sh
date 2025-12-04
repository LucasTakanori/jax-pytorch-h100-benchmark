#!/bin/bash
#SBATCH --job-name=jax-inference-memory-fix
#SBATCH --output=jax_inference_fixed_%j.out
#SBATCH --error=jax_inference_fixed_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1


# Load required modules
module load CUDA/12.9.0
module load gcc/11.2.0

# Activate virtual environment
source /home/lsanc68/pvi-ml/.venv/bin/activate

# Set environment variables
export HF_HOME=/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface
export HF_DATASETS_CACHE=/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface/hub

# Change to project directory
cd /mmfs1/projects/ece_bst/lsanc68/594Project

echo "========================================================================="
echo "JAX INFERENCE BENCHMARK - MEMORY FIX"
echo "========================================================================="
echo "Fixed bug in utils/memory.py to use peak_bytes_in_use instead of bytes_limit"
echo "Re-running all JAX inference benchmarks to get accurate memory measurements"
echo "========================================================================="
echo ""

MODELS="resnet50 vit_b_16 mobilenet_v3_small efficientnet_b0"
BATCH_SIZES="1 8 32 128"
ITERATIONS=100
WARMUP=20

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/inference/jax_fixed_$TIMESTAMP"
mkdir -p $OUTPUT_DIR

echo "Output directory: $OUTPUT_DIR"
echo "Models: $MODELS"
echo "Batch sizes: $BATCH_SIZES"
echo "Iterations: $ITERATIONS (warmup: $WARMUP)"
echo ""

total_configs=4  # 4 models
completed_configs=0

for model in $MODELS; do
    ((completed_configs++))

    echo "========================================="
    echo "[$completed_configs/$total_configs] JAX / $model"
    echo "========================================="

    # Fix for ViT CUDA_ERROR_ILLEGAL_ADDRESS: Disable XLA autotuning ONLY for ViT
    if [ "$model" == "vit_b_16" ]; then
        export XLA_FLAGS=--xla_gpu_autotune_level=0
        echo "⚠️  Disabling XLA autotuning for ViT to prevent crash"
    else
        unset XLA_FLAGS
    fi

    python bench/runner.py \
        --framework jax \
        --model $model \
        --batch-sizes $BATCH_SIZES \
        --warmup $WARMUP \
        --iterations $ITERATIONS \
        --output-dir $OUTPUT_DIR

    if [ $? -eq 0 ]; then
        echo "✓ Completed successfully"
    else
        echo "✗ Failed - check errors above"
    fi

    echo ""
done

echo "========================================================================="
echo "JAX INFERENCE BENCHMARK COMPLETE!"
echo "========================================================================="
echo "Results saved in: $OUTPUT_DIR"
echo "Completed: $completed_configs/$total_configs models"
echo ""
echo "Memory measurements should now show accurate values (NOT 75GB)"
echo "========================================================================="

deactivate
