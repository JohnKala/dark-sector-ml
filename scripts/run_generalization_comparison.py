"""
Generalization Comparison Script.

This script performs a rigorous scientific comparison between Standard and Adversarial Training.
It answers the question: "Does Adversarial Training improve generalization to unseen physics?"

Workflow:
1. Train Standard Model on Source Dataset.
2. Train Adversarial Model on Source Dataset (using optimal hyperparameters).
3. Evaluate BOTH models on ALL available datasets (Source + Unseen Targets).
4. Compute "Smart Metrics":
   - Efficiency Ratio (Signal Eff @ Fixed Background Eff)
   - KL Divergence (Stability of predictions)
5. Save results to JSON for analysis.

Usage:
    python scripts/run_generalization_comparison.py \
        --source_signal data/raw/AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak.h5 \
        --output_dir results/generalization_study
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import DATASET_FILES
from src.data.preparation import create_dataset
from src.data.preprocessor import prepare_ml_dataset, prepare_deepsets_data
from src.training.trainer import train_model
from src.evaluation.metrics import evaluate_model, calculate_efficiency_ratio, calculate_divergence_metrics
from src.visualization.comparison_plots import plot_feature_shift, plot_prediction_shift, plot_delta_heatmap
from src.models.adversarial import AdversarialExampleGenerator

# Champion Hyperparameters (Hardcoded for consistency in this study)
ADV_CONFIG = {
    'alpha': 0.05,
    'grad_eps': 0.1,
    'grad_iter': 10,
    'grad_eta': 0.025  # 2.5 * 0.1 / 10
}
BATCH_SIZE = 128  # Winner of fine-tuning


def parse_args():
    parser = argparse.ArgumentParser(description='Run Generalization Comparison Study')
    parser.add_argument('--source_signal', type=str, required=True, help='Path to Source Signal H5 file')
    parser.add_argument('--background_path', type=str, default=None, help='Path to NominalSM.h5 (optional)')
    parser.add_argument('--output_dir', type=str, default='results/generalization', help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--quick_run', action='store_true', help='Run in fast debug mode (1 epoch, fewer datasets)')
    return parser.parse_args()


def load_and_prepare(signal_path, background_path):
    """Helper to load and prepare data for DeepSets."""
    print(f"Loading: {os.path.basename(signal_path)} + {os.path.basename(background_path)}")
    
    combined_data = create_dataset(
        [signal_path, background_path],
        use_scaled=True,
        signal_background_mode=True,
        verbose=False
    )
    
    ml_data = prepare_ml_dataset(
        combined_data, 
        test_size=0.2, 
        val_size=0.25, 
        normalize=True, 
        reshape_3d=True,
        verbose=False
    )
    
    return prepare_deepsets_data(ml_data, return_masks=True)


def generate_visualizations(model, data, output_dir, prefix=""):
    """Generate perturbation analysis plots."""
    print(f"Generating visualizations for {prefix}...")
    
    # Create generator
    generator = AdversarialExampleGenerator(
        grad_iter=ADV_CONFIG['grad_iter'],
        grad_eps=ADV_CONFIG['grad_eps'],
        grad_eta=ADV_CONFIG['grad_eta']
    )
    
    # Get a batch of data (Signal Only for clearer physics plots)
    features = data['test']['features']
    masks = data['test']['attention_mask']
    labels = data['test']['labels']
    
    # Filter for signal only
    sig_mask = (labels == 1)
    # Take first 1000 signal events
    limit = 1000
    clean_features = features[sig_mask][:limit]
    clean_masks = masks[sig_mask][:limit]
    clean_labels = labels[sig_mask][:limit]
    
    if len(clean_features) == 0:
        print("Warning: No signal events found for visualization.")
        return
        
    # Generate adversarial examples
    # Note: generate_adversarial_examples expects the base model, not the wrapper
    # If 'model' is an AdversarialModelWrapper, we need to access its base_model
    base_model = model.base_model if hasattr(model, 'base_model') else model
    
    perturbed_features = generator.generate_adversarial_examples(
        base_model, clean_features, clean_masks
    )
    
    # 1. Feature Shift
    plot_feature_shift(
        clean_features, 
        perturbed_features.numpy(),
        save_path=os.path.join(output_dir, f"{prefix}_feature_shift.png")
    )
    
    # 2. Prediction Shift
    clean_probs = model.predict([clean_features, clean_masks], verbose=0).ravel()
    pert_probs = model.predict([perturbed_features.numpy(), clean_masks], verbose=0).ravel()
    
    plot_prediction_shift(
        clean_probs, 
        pert_probs,
        save_path=os.path.join(output_dir, f"{prefix}_prediction_shift.png")
    )
    
    # 3. Delta Heatmap
    plot_delta_heatmap(
        clean_features, 
        perturbed_features.numpy(),
        save_path=os.path.join(output_dir, f"{prefix}_delta_heatmap.png")
    )


def main():
    args = parse_args()
    
    # Setup paths
    if args.background_path is None:
        args.background_path = os.path.join(os.path.dirname(args.source_signal), "NominalSM.h5")
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*50)
    print("PHASE 3: GENERALIZATION STUDY")
    print("="*50)
    print(f"Source Signal: {os.path.basename(args.source_signal)}")
    print(f"Adversarial Config: {ADV_CONFIG}")
    if args.quick_run:
        print("!! QUICK RUN MODE ENABLED !!")
        args.epochs = 1
    
    # ---------------------------------------------------------
    # 1. Train Models (Standard vs Adversarial)
    # ---------------------------------------------------------
    
    # Prepare Source Data
    print("\n[1/4] Preparing Source Data...")
    source_data = load_and_prepare(args.source_signal, args.background_path)
    
    # Train Standard Model
    print("\n[2/4] Training Standard Model...")
    std_results = train_model(
        prepared_data=source_data,
        model_type='deepsets',
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        adversarial_config=None,
        verbose=True,
        save_model=False
    )
    std_model = std_results['model']
    
    # Train Adversarial Model
    print("\n[3/4] Training Adversarial Model...")
    adv_results = train_model(
        prepared_data=source_data,
        model_type='deepsets',
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        adversarial_config=ADV_CONFIG,
        verbose=True,
        save_model=False
    )
    adv_model = adv_results['model']
    
    # Generate Visualizations for Source Dataset
    print("\n[3.5/4] Generating Perturbation Analysis Plots...")
    generate_visualizations(adv_model, source_data, args.output_dir, prefix="source")
    
    # ---------------------------------------------------------
    # 2. Cross-Evaluation Loop
    # ---------------------------------------------------------
    print("\n[4/4] Running Cross-Evaluation on ALL datasets...")
    
    comparison_results = []
    
    # Get list of all signal files in the same directory as source
    data_dir = os.path.dirname(args.source_signal)
    # Filter for H5 files that are NOT NominalSM
    target_files = [f for f in os.listdir(data_dir) if f.endswith('.h5') and 'NominalSM' not in f]
    
    if args.quick_run:
        print("Quick Run: Limiting to first 2 target datasets")
        target_files = target_files[:2]
    
    for target_file in target_files:
        target_path = os.path.join(data_dir, target_file)
        print(f"\nEvaluating on: {target_file}")
        
        # Load Target Data
        # Note: We use the SAME background file, but different signal file
        target_data = load_and_prepare(target_path, args.background_path)
        
        # Evaluate Standard Model
        std_eval = evaluate_model(std_model, target_data, verbose=False)
        std_auc = std_eval['metrics']['roc_auc']
        std_probs = std_eval['predictions']['y_pred_proba']
        y_true = std_eval['predictions']['y_true']
        
        # Evaluate Adversarial Model
        adv_eval = evaluate_model(adv_model, target_data, verbose=False)
        adv_auc = adv_eval['metrics']['roc_auc']
        adv_probs = adv_eval['predictions']['y_pred_proba']
        
        # Compute Smart Metrics
        # 1. Efficiency Ratio (at 1% Background Efficiency)
        eff_metrics = calculate_efficiency_ratio(
            y_true, adv_probs, std_probs, target_bg_eff=0.01
        )
        
        # 2. Divergence (How different are the predictions?)
        div_metrics = calculate_divergence_metrics(std_probs, adv_probs)
        
        print(f"  > Std AUC: {std_auc:.4f}")
        print(f"  > Adv AUC: {adv_auc:.4f}")
        print(f"  > Eff Ratio (1%): {eff_metrics['ratio']:.4f} ({'WIN' if eff_metrics['ratio'] > 1 else 'LOSS'})")
        print(f"  > KL Divergence: {div_metrics['kl_divergence']:.4f}")
        
        # Store Result
        comparison_results.append({
            'target_dataset': target_file,
            'is_source': (target_file == os.path.basename(args.source_signal)),
            'std_auc': float(std_auc),
            'adv_auc': float(adv_auc),
            'auc_diff': float(adv_auc - std_auc),
            'efficiency_ratio': eff_metrics['ratio'],
            'std_sig_eff': eff_metrics['sig_eff_b'], # Denominator was std
            'adv_sig_eff': eff_metrics['sig_eff_a'], # Numerator was adv
            'kl_divergence': div_metrics['kl_divergence'],
            'js_divergence': div_metrics['js_divergence']
        })
        
    # ---------------------------------------------------------
    # 3. Save Results
    # ---------------------------------------------------------
    output_file = os.path.join(args.output_dir, 'generalization_results.json')
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=4)
        
    print(f"\nStudy Complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
