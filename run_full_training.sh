#!/bin/bash
# Full training benchmark: 4 models × 1 framework × 3 batch sizes = 12 configs
# (JAX training implementation pending - using PyTorch only)

echo "========================================================================="
echo "FULL TRAINING BENCHMARK SUITE"
echo "========================================================================="
echo "Models: ResNet-50, ViT-Base, MobileNetV3-Small, EfficientNet-B0"
echo "Framework: PyTorch"
echo "Batch sizes: 32, 64, 128"
echo "Epochs: 5 per configuration"
echo "Dataset: ImageNet-100"
echo "========================================================================="
echo ""

MODELS="resnet50 vit_b_16 mobilenet_v3_small efficientnet_b0"
FRAMEWORKS="pytorch jax"
BATCH_SIZES="32 64 128"
EPOCHS=2
LR=1e-4
# Use Hugging Face dataset ID by default (clane9/imagenet-100).
# To use a local ImageFolder, set DATASET to the root directory containing
# 'train' and 'validation' subfolders.
DATASET="clane9/imagenet-100"
DATASET_CACHE_DIR="${HF_DATASETS_CACHE:-/home/lsanc68/ece_bst_link/lsanc68/.cache/huggingface/datasets}"

# Create output directories
mkdir -p results/training/pytorch

if [ -d "$DATASET" ]; then
    echo "Using local dataset at $DATASET"
else
    echo "Using Hugging Face dataset ID: $DATASET"
    echo "Dataset cache directory: $DATASET_CACHE_DIR"
fi

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

echo "Total configurations to benchmark: $total_configs"
echo ""

for framework in $FRAMEWORKS; do
    for model in $MODELS; do
        for batch_size in $BATCH_SIZES; do
            ((completed_configs++))

            echo "========================================="
            echo "[$completed_configs/$total_configs] $framework / $model / BS=$batch_size"
            echo "========================================="

            python bench/training_runner.py \
                --framework $framework \
                --model $model \
                --dataset $DATASET \
                --dataset-cache-dir $DATASET_CACHE_DIR \
                --batch-size $batch_size \
                --epochs $EPOCHS \
                --lr $LR \
                --optimizer adam \
                --scheduler cosine \
                --output-dir results/training/$framework

            if [ $? -eq 0 ]; then
                echo "✓ Completed successfully"
            else
                echo "✗ Failed - check errors above"
            fi

            echo ""
        done
    done
done

echo "========================================================================="
echo "TRAINING BENCHMARK COMPLETE!"
echo "========================================================================="
echo "Results saved in: results/training/pytorch/"
echo ""
echo "Completed: $completed_configs/$total_configs configurations"
echo "========================================================================="
