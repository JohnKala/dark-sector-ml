"""
Hyperparameter sweep script for adversarial training.

This script is designed for HPC job arrays. It takes a specific configuration
of adversarial hyperparameters, trains a model, evaluates its robustness,
and appends the result to a CSV file.

It correctly handles data mixing by combining a specific Signal dataset
with the Standard Model (Background) dataset.

Usage:
    python run_hyperparameter_sweep.py \
        --signal_path /path/to/signal.h5 \
        --background_path /path/to/NominalSM.h5 \
        --output_file results.csv \
        --alpha 1.0 \
        --epsilon 1e-6 \
        --grad_iter 10 \
        --grad_eta 2e-7 \
        --batch_size 256
"""

import os
import sys
import argparse
import csv
import time
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preparation import create_dataset
from src.data.preprocessor import prepare_ml_dataset, prepare_deepsets_data
from src.training.trainer import train_model
from src.evaluation.robustness import RobustnessEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for adversarial training')
    
    # Data arguments
    parser.add_argument('--signal_path', type=str, required=True, help='Path to Signal H5 dataset (Dark Sector)')
    parser.add_argument('--background_path', type=str, default=None, help='Path to Background H5 dataset (default: NominalSM.h5 in same dir)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output CSV file')
    
    # Adversarial Training Hyperparameters
    parser.add_argument('--alpha', type=float, default=5.0, help='Adversarial loss weight')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='Perturbation budget')
    parser.add_argument('--grad_iter', type=int, default=10, help='Number of attack steps')
    parser.add_argument('--grad_eta', type=float, default=2e-7, help='Attack step size')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--model_type', type=str, default='deepsets', help='Model architecture')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Default background path if not provided
    if args.background_path is None:
        args.background_path = os.path.join(os.path.dirname(args.signal_path), "NominalSM.h5")
    
    print(f"Starting sweep job")
    print(f"Signal: {os.path.basename(args.signal_path)}")
    print(f"Background: {os.path.basename(args.background_path)}")
    print(f"Config: alpha={args.alpha}, eps={args.epsilon}, iter={args.grad_iter}")
    
    # 1. Load and Prepare Data (With Mixing)
    print("Loading and mixing data...")
    
    # Use create_dataset to handle the mixing logic (Signal from Dark file, Background from SM file)
    combined_data = create_dataset(
        [args.signal_path, args.background_path],
        use_scaled=True,            # Standard for this project
        signal_background_mode=True, # ENABLE MIXING
        verbose=False
    )
    
    # Prepare splits
    ml_data = prepare_ml_dataset(
        combined_data, 
        test_size=0.2, 
        val_size=0.25, 
        normalize=True, 
        reshape_3d=True,
        verbose=False
    )
    
    # Format for DeepSets
    prepared_data = prepare_deepsets_data(ml_data, return_masks=True)
    
    # 2. Train Model
    print("Training adversarial model...")
    
    adversarial_config = {
        'alpha': args.alpha,
        'grad_eps': args.epsilon,
        'grad_iter': args.grad_iter,
        'grad_eta': args.grad_eta
    }
    
    # Train
    results = train_model(
        prepared_data=prepared_data,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        adversarial_config=adversarial_config,
        verbose=True,
        save_model=False  # Don't save weights to save disk space during sweeps
    )
    
    model = results['model']
    
    # 3. Evaluate Robustness (The Standardized Exam)
    print("Running standardized robustness evaluation...")
    
    evaluator = RobustnessEvaluator(
        attack_config={
            'grad_eps': args.epsilon,
            'grad_iter': args.grad_iter, # Test on what we trained on
            'grad_eta': args.grad_eta
        },
        batch_size=args.batch_size
    )
    
    eval_metrics = evaluator.evaluate(model, prepared_data)
    
    # 4. Save Results
    print(f"Saving results to {args.output_file}...")
    
    # Check if file exists to write header
    file_exists = os.path.isfile(args.output_file)
    
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            header = [
                'signal_dataset', 'alpha', 'epsilon', 'grad_iter', 'grad_eta', 'batch_size',
                'clean_auc', 'robust_auc', 'robustness_score', 'training_time'
            ]
            writer.writerow(header)
        
        # Write row
        row = [
            os.path.basename(args.signal_path),
            args.alpha,
            args.epsilon,
            args.grad_iter,
            args.grad_eta,
            args.batch_size,
            f"{eval_metrics['clean_auc']:.6f}",
            f"{eval_metrics['robust_auc']:.6f}",
            f"{eval_metrics['robustness_score']:.6f}",
            f"{results['training_time']:.2f}"
        ]
        writer.writerow(row)
        
    print("Job completed successfully.")


if __name__ == "__main__":
    main()
