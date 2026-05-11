#!/bin/bash
# ========================================================================
# Full Paper Reproduction Pipeline
# ========================================================================
# This script reproduces ALL paper results in a single command.
# Estimated runtime: 5-7 days on a single NVIDIA RTX 6000 (24 GB).
#
# Prerequisites:
#   1. OncoLung60K downloaded to data/oncolung60k/
#   2. LungHist700 downloaded to data/lunghist700/
#   3. Patient-wise split CSVs in splits/ (provided in repo)
#
# Usage:
#   bash scripts/reproduce_paper.sh
# ========================================================================

set -e  # Exit on first error

DATA_OL="data/oncolung60k"
DATA_LH="data/lunghist700"
RUNS="runs/paper_reproduction"

mkdir -p $RUNS

echo "========================================="
echo "Step 1/4: SOTA benchmark on OncoLung60K"
echo "========================================="
python -m src.benchmark \
    --configs_dir configs/ \
    --data_dir $DATA_OL \
    --output_dir $RUNS/sota_benchmark

echo ""
echo "========================================="
echo "Step 2/4: Extended ablation (8 configs)"
echo "========================================="
for config in configs/ablation/*.yaml; do
    name=$(basename $config .yaml)
    echo "Running: $name"
    python -m src.kfold_cv \
        --config $config \
        --data_dir $DATA_OL \
        --output_dir $RUNS/ablation/$name
done

echo ""
echo "========================================="
echo "Step 3/4: LungHist700 cross-dataset benchmark"
echo "========================================="
python -m src.kfold_cv \
    --config configs/lunghist700_modified_convnext.yaml \
    --data_dir $DATA_LH \
    --output_dir $RUNS/lunghist700

echo ""
echo "========================================="
echo "Step 4/4: Generate Grad-CAM visualizations"
echo "========================================="
python -m src.utils.explainability \
    --weights $RUNS/sota_benchmark/oncolung_modified_convnext/fold0/best.pt \
    --images data/sample_test_images/ \
    --output_dir $RUNS/gradcam_outputs/

echo ""
echo "========================================="
echo "REPRODUCTION COMPLETE"
echo "========================================="
echo ""
echo "Results saved to: $RUNS"
echo "Headline numbers:"
echo "  - SOTA benchmark: $RUNS/sota_benchmark/benchmark_summary.csv"
echo "  - Ablation:       $RUNS/ablation/*/cv_summary.csv"
echo "  - LungHist700:    $RUNS/lunghist700/cv_summary.csv"
