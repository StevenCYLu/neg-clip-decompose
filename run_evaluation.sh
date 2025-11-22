#!/bin/bash

# Convenience script for running negation decomposition evaluations
# This script provides examples for running MCQ and retrieval evaluations

set -e  # Exit on error

# Configuration
MODEL="ViT-B-32"
PRETRAINED="openai"
NEGBENCH_ROOT="../negbench"
BATCH_SIZE=64
NUM_WORKERS=4

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Check if negbench directory exists
if [ ! -d "$NEGBENCH_ROOT" ]; then
    print_error "negbench directory not found at $NEGBENCH_ROOT"
    print_info "Please update NEGBENCH_ROOT in this script or create a symlink"
    exit 1
fi

# Create results directory
mkdir -p results

# Example 1: MCQ evaluation without decomposition (baseline)
print_header "Example 1: MCQ Baseline (No Decomposition)"
print_info "Running MCQ evaluation on COCO with standard CLIP encoding..."

python eval_mcq.py \
    --model $MODEL \
    --pretrained $PRETRAINED \
    --dataset "COCO_val_mcq_llama3.1_rephrased.csv" \
    --negbench-root $NEGBENCH_ROOT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --output results/mcq_coco_baseline.json

echo ""

# Example 2: MCQ evaluation with decomposition
print_header "Example 2: MCQ with Negation Decomposition"
print_info "Running MCQ evaluation on COCO with negation decomposition (alpha=0.5)..."

python eval_mcq.py \
    --model $MODEL \
    --pretrained $PRETRAINED \
    --dataset "COCO_val_mcq_llama3.1_rephrased.csv" \
    --negbench-root $NEGBENCH_ROOT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --use-decomposition \
    --alpha 0.5 \
    --output results/mcq_coco_decomp_alpha0.5.json

echo ""

# Example 3: MCQ evaluation with different alpha values
print_header "Example 3: MCQ with Different Alpha Values"

for alpha in 0.3 0.5 0.7 1.0; do
    print_info "Alpha = $alpha"
    python eval_mcq.py \
        --model $MODEL \
        --pretrained $PRETRAINED \
        --dataset "COCO_val_mcq_llama3.1_rephrased.csv" \
        --negbench-root $NEGBENCH_ROOT \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --use-decomposition \
        --alpha $alpha \
        --output results/mcq_coco_decomp_alpha${alpha}.json
done

echo ""

# Example 4: Retrieval evaluation without decomposition (baseline)
print_header "Example 4: Retrieval Baseline (No Decomposition)"
print_info "Running retrieval evaluation on COCO negated with standard CLIP encoding..."

python eval_retrieval.py \
    --model $MODEL \
    --pretrained $PRETRAINED \
    --dataset "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv" \
    --negbench-root $NEGBENCH_ROOT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --recall-k 1 5 10 \
    --output results/retrieval_coco_negated_baseline.json

echo ""

# Example 5: Retrieval evaluation with decomposition
print_header "Example 5: Retrieval with Negation Decomposition"
print_info "Running retrieval evaluation on COCO negated with negation decomposition (alpha=0.5)..."

python eval_retrieval.py \
    --model $MODEL \
    --pretrained $PRETRAINED \
    --dataset "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv" \
    --negbench-root $NEGBENCH_ROOT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --use-decomposition \
    --alpha 0.5 \
    --recall-k 1 5 10 \
    --output results/retrieval_coco_negated_decomp_alpha0.5.json

echo ""

# Example 6: Evaluate on VOC2007 MCQ
print_header "Example 6: VOC2007 MCQ"
print_info "Running MCQ evaluation on VOC2007..."

# Baseline
python eval_mcq.py \
    --model $MODEL \
    --pretrained $PRETRAINED \
    --dataset "VOC2007_mcq_llama3.1_rephrased.csv" \
    --negbench-root $NEGBENCH_ROOT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --output results/mcq_voc_baseline.json

# With decomposition
python eval_mcq.py \
    --model $MODEL \
    --pretrained $PRETRAINED \
    --dataset "VOC2007_mcq_llama3.1_rephrased.csv" \
    --negbench-root $NEGBENCH_ROOT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --use-decomposition \
    --alpha 0.5 \
    --output results/mcq_voc_decomp_alpha0.5.json

echo ""

# Example 7: Evaluate on synthetic MCQ
print_header "Example 7: Synthetic MCQ"
print_info "Running MCQ evaluation on synthetic dataset..."

# Baseline
python eval_mcq.py \
    --model $MODEL \
    --pretrained $PRETRAINED \
    --dataset "synthetic_mcq_llama3.1_rephrased.csv" \
    --negbench-root $NEGBENCH_ROOT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --output results/mcq_synthetic_baseline.json

# With decomposition
python eval_mcq.py \
    --model $MODEL \
    --pretrained $PRETRAINED \
    --dataset "synthetic_mcq_llama3.1_rephrased.csv" \
    --negbench-root $NEGBENCH_ROOT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --use-decomposition \
    --alpha 0.5 \
    --output results/mcq_synthetic_decomp_alpha0.5.json

echo ""

print_header "All Evaluations Complete!"
print_info "Results saved in ./results/"
print_info "You can compare baseline vs decomposition results to see the improvement"

# Optional: Print summary
print_header "Summary"
ls -lh results/
