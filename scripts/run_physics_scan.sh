#!/bin/bash

# Physics Scan Script
# ===================
# This script executes the final "Physics Scan" for the Dark Sector Classifier.
# It uses the "Champion" Source Dataset (mDark-5, rinv-0.3, alpha-peak) and the
# optimal adversarial hyperparameters to evaluate generalization across the full
# physics parameter space (all available signal files).

# 1. Setup
# --------
SOURCE_SIGNAL="data/raw/AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak.h5"
OUTPUT_DIR="results/physics_scan"
EPOCHS=50

echo "========================================================"
echo "STARTING PHYSICS SCAN"
echo "========================================================"
echo "Source Signal: $SOURCE_SIGNAL"
echo "Output Directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "========================================================"

# 2. Run Generalization Comparison
# --------------------------------
# This python script handles:
# - Training Standard Model (on Source)
# - Training Adversarial Model (on Source, with optimal params)
# - Cross-evaluating on ALL 6 datasets
# - Calculating Efficiency Ratio and Stability Metrics (KL/JS)
# - Generating Perturbation Analysis Plots

python scripts/run_generalization_comparison.py \
    --source_signal "$SOURCE_SIGNAL" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS

# 3. Completion
# -------------
echo "========================================================"
echo "PHYSICS SCAN COMPLETE"
echo "========================================================"
echo "Results available in: $OUTPUT_DIR"
echo "  - Report: generalization_results.json"
echo "  - Plots:  source_feature_shift.png, etc."
