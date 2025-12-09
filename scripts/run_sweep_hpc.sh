#!/bin/bash
#SBATCH --job-name=adv_gen_sweep
#SBATCH --array=0-12                    # 13 configs (0-indexed), adjust based on DEFAULT_SWEEP_CONFIGS
#SBATCH --time=04:00:00                 # 4 hours per config
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                    # Request 1 GPU per job
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
# SBATCH --partition=gpu                # Uncomment and set your GPU partition name
# SBATCH --account=your_account         # Uncomment and set your account if needed

# =============================================================================
# Adversarial Generalization Sweep - HPC Job Array Script
# =============================================================================
# This script runs the adversarial generalization sweep on a SLURM cluster.
# Each array task trains one adversarial configuration and evaluates it on all targets.
#
# Usage:
#   1. Adjust the --array parameter based on number of configs (default: 13)
#   2. Set SOURCE_SIGNAL to your chosen source dataset
#   3. Submit: sbatch scripts/run_sweep_hpc.sh
#   4. After completion: python scripts/aggregate_sweep_results.py --sweep_dir results/adv_gen_sweep/<source_name>
# =============================================================================

# Configuration
SOURCE_SIGNAL="data/raw/AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high.h5"
OUTPUT_DIR="results/adv_gen_sweep"
EPOCHS=50

# Create logs directory
mkdir -p logs

echo "========================================================"
echo "ADVERSARIAL GENERALIZATION SWEEP - JOB ARRAY"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Source Signal: $SOURCE_SIGNAL"
echo "Output Directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "========================================================"

# Activate conda environment (adjust path as needed)
# source ~/.bashrc
# conda activate dark-sector-ml

# Run the sweep for this config index
python scripts/run_adversarial_generalization_sweep.py \
    --source_signal "$SOURCE_SIGNAL" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --single_config_idx $SLURM_ARRAY_TASK_ID \
    --eval_robustness \
    --verbose

echo "========================================================"
echo "JOB COMPLETE"
echo "========================================================"
