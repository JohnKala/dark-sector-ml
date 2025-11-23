#!/bin/bash

# Physics Scan Script
# ===================
# This script executes the final "Physics Scan" for the Dark Sector Classifier.
# It uses the directory of the "Champion" Source Dataset (mDark-5, rinv-0.3,
# alpha-peak) to discover all available signal files, and then runs a separate
# generalization study for each one. Each run trains on a different source
# dataset and evaluates generalization across the full physics parameter space.

# 1. Setup
# --------
SOURCE_SIGNAL="data/raw/AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak.h5"
OUTPUT_DIR="results/physics_scan"
EPOCHS=50

echo "========================================================"
echo "STARTING PHYSICS SCAN (MULTI-SOURCE)"
echo "========================================================"
echo "Anchor Source: $SOURCE_SIGNAL"
echo "Base Output Directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "========================================================"

# Determine directory containing the signal files
SOURCE_DIR="$(dirname "$SOURCE_SIGNAL")"

# Collect all signal datasets (exclude NominalSM)
SIGNAL_FILES=()
for f in "$SOURCE_DIR"/*.h5; do
    # Skip non-existent globs
    [ -f "$f" ] || continue

    base_name="$(basename "$f")"
    case "$base_name" in
        *NominalSM*)
            # Skip the Standard Model background file
            continue
            ;;
        *)
            SIGNAL_FILES+=("$f")
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "Found ${#SIGNAL_FILES[@]} signal datasets to use as sources:"
for f in "${SIGNAL_FILES[@]}"; do
    echo "  - $(basename "$f")"
done
echo "========================================================"

# 2. Run Generalization Comparison
# --------------------------------
# This python script handles:
# - Training Standard Model (on each Source)
# - Training Adversarial Model (on each Source, with optimal params)
# - Cross-evaluating on ALL available datasets
# - Calculating Efficiency Ratio and Stability Metrics (KL/JS)
# - Generating Perturbation Analysis Plots

for SIGNAL in "${SIGNAL_FILES[@]}"; do
    BASENAME="$(basename "$SIGNAL" .h5)"
    RUN_DIR="$OUTPUT_DIR/$BASENAME"

    echo "--------------------------------------------------------"
    echo "Running physics scan for source dataset: $BASENAME"
    echo "Output directory: $RUN_DIR"
    echo "--------------------------------------------------------"

    mkdir -p "$RUN_DIR"

    python scripts/run_generalization_comparison.py \
        --source_signal "$SIGNAL" \
        --output_dir "$RUN_DIR" \
        --epochs $EPOCHS
done

# 3. Completion
# -------------
echo "========================================================"
echo "PHYSICS SCAN COMPLETE"
echo "========================================================"
echo "Results available under: $OUTPUT_DIR"
echo "Each source dataset has its own subdirectory containing:"
echo "  - generalization_results.json"
echo "  - source_feature_shift.png, source_prediction_shift.png, etc."
