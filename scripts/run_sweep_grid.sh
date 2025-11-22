#!/bin/bash

# ==========================================
# Dark Sector Adversarial Sweep Launcher
# ==========================================

# 1. DATASET CONFIGURATION
# Replace this with your actual signal file path on the cluster
SIGNAL_FILE="data/raw/AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak.h5"
OUTPUT_FILE="sweep_results.csv"

# 2. HYPERPARAMETER GRID
# Define the values you want to test
ALPHAS=(0.1 1.0 5.0 10.0)       # Regularization weight
EPSILONS=(0.001 0.01 0.1)       # Perturbation budget (Stronger now!)
GRAD_ITERS=(10)                 # Attack steps (keep fixed usually)

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2

# 3. EXECUTION LOOP
echo "Starting Sweep..."
echo "Signal File: $SIGNAL_FILE"
echo "Output File: $OUTPUT_FILE"
echo "--------------------------------"

for alpha in "${ALPHAS[@]}"; do
  for eps in "${EPSILONS[@]}"; do
    for iter in "${GRAD_ITERS[@]}"; do
      
      echo "[Running] Alpha=$alpha | Epsilon=$eps | Iter=$iter"
      
      # Run the python script
      # Note: background_path is optional (defaults to NominalSM.h5 in same dir)
      python scripts/run_hyperparameter_sweep.py \
        --signal_path "$SIGNAL_FILE" \
        --output_file "$OUTPUT_FILE" \
        --alpha "$alpha" \
        --epsilon "$eps" \
        --grad_iter "$iter" \
        --epochs 50 \
        --batch_size 256
        
    done
  done
done

echo "--------------------------------"
echo "Sweep Complete! Results saved to $OUTPUT_FILE"
