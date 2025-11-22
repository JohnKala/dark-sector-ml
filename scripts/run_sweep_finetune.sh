#!/bin/bash

# ==========================================
# Dark Sector Adversarial Sweep - FINETUNE
# ==========================================

# 1. DATASET CONFIGURATION
SIGNAL_FILE="data/raw/AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak.h5"
OUTPUT_FILE="sweep_results_finetune.csv"

# 2. REFINED HYPERPARAMETER GRID
# Zooming in around Alpha=0.1 (our previous winner)
ALPHAS=(0.01 0.05 0.1 0.2 0.5)

# Focusing on the "Breaking Point" regime
EPSILONS=(0.01 0.05 0.1)

GRAD_ITERS=(10)

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2

# 3. EXECUTION LOOP
echo "Starting Fine-Tuning Sweep..."
echo "Signal File: $SIGNAL_FILE"
echo "Output File: $OUTPUT_FILE"
echo "--------------------------------"

for alpha in "${ALPHAS[@]}"; do
  for eps in "${EPSILONS[@]}"; do
    
    # Calculate eta = eps / 4 (Dynamic Step Size)
    eta=$(python -c "print($eps / 4.0)")
    
    echo "[Running] Alpha=$alpha | Epsilon=$eps | Eta=$eta"
    
    python scripts/run_hyperparameter_sweep.py \
      --signal_path "$SIGNAL_FILE" \
      --output_file "$OUTPUT_FILE" \
      --alpha "$alpha" \
      --epsilon "$eps" \
      --grad_iter 10 \
      --grad_eta "$eta" \
      --epochs 50 \
      --batch_size 256
      
  done
done

echo "--------------------------------"
echo "Fine-Tuning Complete! Results saved to $OUTPUT_FILE"
