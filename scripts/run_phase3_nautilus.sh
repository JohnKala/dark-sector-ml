#!/bin/bash
# =============================================================================
# Phase 3: Cross-Source Validation (Nautilus Pod)
# =============================================================================
# Tests if Phase 1 findings hold across all 6 source datasets
# Trains 3 configs (baseline, best_avg, best_min) on each source
# Total: 3 configs × 6 sources = 18 training runs
# =============================================================================

set -e  # Exit on error

# Configuration
OUTPUT_DIR="results/phase3_cross_source"
BATCH_SIZE=256
EPOCHS=50
PATIENCE=10

# Verify environment
echo "=============================================="
echo "Phase 3 Cross-Source Validation - Pre-flight"
echo "=============================================="

# Check GPUs
echo ""
echo "[1/3] Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv || {
    echo "WARNING: nvidia-smi failed. Running on CPU?"
}

# Check data files
echo ""
echo "[2/3] Checking data files..."
cd ~/dark-sector-ml

for file in \
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.2_alpha-peak.h5" \
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5" \
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-low.h5" \
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak.h5" \
    "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.8_alpha-peak.h5" \
    "AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak.h5" \
    "NominalSM.h5"
do
    if [ -f "data/raw/$file" ]; then
        SIZE=$(ls -lh "data/raw/$file" | awk '{print $5}')
        echo "  ✓ $file ($SIZE)"
    else
        echo "  ✗ $file NOT FOUND"
        exit 1
    fi
done

# Check Python environment
echo ""
echo "[3/3] Checking Python environment..."
python3 -c "
import tensorflow as tf
import numpy as np
print(f'✓ TensorFlow {tf.__version__}')
print(f'✓ GPUs available: {len(tf.config.list_physical_devices(\"GPU\"))}')
"

echo ""
echo "=============================================="
echo "Starting Phase 3 Cross-Source Validation"
echo "=============================================="
echo "Output:      $OUTPUT_DIR"
echo "Batch Size:  $BATCH_SIZE"
echo "Epochs:      $EPOCHS"
echo "Configs:     3 (baseline, best_avg, best_min)"
echo "Sources:     6"
echo "Total Runs:  18"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Phase 3
python scripts/run_cross_source_validation.py \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --save_models \
    --eval_robustness

echo ""
echo "=============================================="
echo "Phase 3 Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key outputs:"
echo "  - $OUTPUT_DIR/cross_source_summary.json"
echo "  - $OUTPUT_DIR/cross_source_results.csv"
echo "  - $OUTPUT_DIR/visualizations/"
echo ""
