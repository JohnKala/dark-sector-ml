#!/bin/bash
# =============================================================================
# Phase 1: Grid Search on Single Source (Nautilus Pod)
# =============================================================================
# Grid: alpha=[0.2, 1.0, 5.0] x epsilon=[0.1, 0.5] x iterations=[10, 20]
# Total: 13 configs (1 baseline + 12 adversarial)
# Source: mDark-1_rinv-0.3_alpha-high (200k events)
# =============================================================================

set -e  # Exit on error

# Configuration
SOURCE_SIGNAL="data/raw/AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5"
OUTPUT_DIR="results/phase1_gridsearch"
NUM_GPUS=4          # Adjust based on pod allocation
BATCH_SIZE=256      # Larger batch for 200k events
EPOCHS=50
PATIENCE=10

# Verify environment
echo "=============================================="
echo "Phase 1 Grid Search - Pre-flight Checks"
echo "=============================================="

# Check GPUs
echo ""
echo "[1/4] Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv || {
    echo "WARNING: nvidia-smi failed. Running on CPU?"
    NUM_GPUS=1
}

# Check data files
echo ""
echo "[2/4] Checking data files..."
if [ -f "$SOURCE_SIGNAL" ]; then
    SIZE=$(ls -lh "$SOURCE_SIGNAL" | awk '{print $5}')
    echo "✅ Source file exists: $SOURCE_SIGNAL ($SIZE)"
else
    echo "❌ Source file NOT found: $SOURCE_SIGNAL"
    exit 1
fi

# Check background file
BG_FILE="data/raw/NominalSM.h5"
if [ -f "$BG_FILE" ]; then
    SIZE=$(ls -lh "$BG_FILE" | awk '{print $5}')
    echo "✅ Background file exists: $BG_FILE ($SIZE)"
else
    echo "❌ Background file NOT found: $BG_FILE"
    exit 1
fi

# Check target files (should be 5 others)
echo ""
echo "[3/4] Checking target datasets..."
TARGET_COUNT=$(ls data/raw/AutomatedCMS_*.h5 | grep -v "$(basename $SOURCE_SIGNAL)" | wc -l)
echo "Found $TARGET_COUNT target datasets for cross-evaluation"

# Check Python environment
echo ""
echo "[4/4] Checking Python environment..."
python3 -c "
import tensorflow as tf
import numpy as np
import h5py
import seaborn
print(f'✅ TensorFlow {tf.__version__}')
print(f'✅ NumPy {np.__version__}')
print(f'✅ GPUs available: {len(tf.config.list_physical_devices(\"GPU\"))}')
"

echo ""
echo "=============================================="
echo "Starting Phase 1 Grid Search"
echo "=============================================="
echo "Source:      $SOURCE_SIGNAL"
echo "Output:      $OUTPUT_DIR"
echo "GPUs:        $NUM_GPUS"
echo "Batch Size:  $BATCH_SIZE"
echo "Epochs:      $EPOCHS"
echo "Configs:     13 (1 baseline + 12 adversarial)"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the sweep
python scripts/run_adversarial_generalization_sweep.py \
    --source_signal "$SOURCE_SIGNAL" \
    --output_dir "$OUTPUT_DIR" \
    --num_parallel "$NUM_GPUS" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --save_models \
    --eval_robustness

echo ""
echo "=============================================="
echo "Phase 1 Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key outputs:"
echo "  - $OUTPUT_DIR/sweep_summary.json"
echo "  - $OUTPUT_DIR/sweep_results.csv"
echo "  - $OUTPUT_DIR/visualizations/"
echo ""
