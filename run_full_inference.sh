#!/bin/bash
# Full inference benchmark: 4 models × 2 frameworks × 4 batch sizes = 32 configs

source /home/lsanc68/pvi-ml/.venv/bin/activate
export HF_HOME=/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface
export HF_DATASETS_CACHE=/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface/hub

echo "========================================================================="
echo "FULL INFERENCE BENCHMARK SUITE"
echo "========================================================================="
echo "Models: ResNet-50, ViT-Base, MobileNetV3-Small, EfficientNet-B0"
echo "Frameworks: PyTorch, JAX/Flax"
echo "Batch sizes: 1, 8, 32, 128"
echo "Iterations: 100 per configuration"
echo "========================================================================="
echo ""

MODELS="resnet50 vit_b_16 mobilenet_v3_small efficientnet_b0"
FRAMEWORKS="pytorch jax"
BATCH_SIZES="1 8 32 128"
ITERATIONS=100
WARMUP=20

# Create output directories
mkdir -p results/inference/pytorch
mkdir -p results/inference/jax

total_configs=0
completed_configs=0

# Count total configurations
for framework in $FRAMEWORKS; do
    for model in $MODELS; do
        ((total_configs++))
    done
done

echo "Total configurations to benchmark: $total_configs"
echo ""

for framework in $FRAMEWORKS; do
    for model in $MODELS; do
        ((completed_configs++))

        echo "========================================="
        echo "[$completed_configs/$total_configs] $framework / $model"
        echo "========================================="

        python bench/runner.py \
            --framework $framework \
            --model $model \
            --batch-sizes $BATCH_SIZES \
            --warmup $WARMUP \
            --iterations $ITERATIONS \
            --output-dir results/inference/$framework

        if [ $? -eq 0 ]; then
            echo "✓ Completed successfully"
        else
            echo "✗ Failed - check errors above"
        fi

        echo ""
    done
done

echo "========================================================================="
echo "BENCHMARK COMPLETE!"
echo "========================================================================="
echo "Results saved in: results/inference/"
echo "  - PyTorch: results/inference/pytorch/"
echo "  - JAX:     results/inference/jax/"
echo ""
echo "Completed: $completed_configs/$total_configs configurations"
echo "========================================================================="
