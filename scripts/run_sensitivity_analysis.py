#!/usr/bin/env python3
"""
Dark Sector Sensitivity Analysis Pipeline

This script runs a comprehensive sensitivity analysis for dark sector physics,
training models on different physics parameter points and evaluating their
cross-parameter performance to understand model robustness and systematics.

Usage:
    python scripts/run_sensitivity_analysis.py --dataset-dir data/processed --output-dir outputs/sensitivity_analysis

Author: Your Name
Date: 2025-01-10
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import our pipeline functions
from src.data.preparation import create_dataset
from src.data.preprocessor import prepare_ml_dataset, prepare_deepsets_data
from src.data.loader import extract_parameters
from src.training.experiments import train_individual_models, train_leave_one_out_models
from src.evaluation import evaluate_cross_model_performance
from src.visualization import (
    plot_combined_roc_curves, plot_cross_model_heatmap, 
    plot_cross_model_roc_curves, plot_training_history
)
from src.utils import save_results


def match_model_to_dataset(model_name: str, available_datasets: List[str]) -> str:
    """Robustly match model to its corresponding test dataset."""
    model_params = extract_parameters(model_name)
    
    # If no parameters extracted, use first dataset
    if 'mDark' not in model_params:
        return available_datasets[0] if available_datasets else None
    
    # Look for exact parameter match
    for ds_name in available_datasets:
        ds_params = extract_parameters(ds_name)
        
        if ('mDark' in ds_params and 
            ds_params['mDark'] == model_params['mDark'] and
            ds_params['rinv'] == model_params['rinv'] and
            ds_params['alpha'] == model_params['alpha']):
            return ds_name
    
    # Fallback to first dataset
    return available_datasets[0] if available_datasets else None


def prepare_roc_data_from_cross_eval(
    cross_eval_results: Dict[str, Dict[str, Dict[str, Any]]],
    model_dataset_mapping: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """Prepare ROC curve data from cross-evaluation results efficiently."""
    from sklearn.metrics import roc_curve, auc
    
    roc_data = {}
    
    for model_name, model_results in cross_eval_results.items():
        # Find the primary dataset for this model
        target_dataset = model_dataset_mapping.get(model_name)
        
        if target_dataset and target_dataset in model_results:
            ds_results = model_results[target_dataset]
            
            # Calculate ROC curve from stored predictions
            fpr, tpr, _ = roc_curve(ds_results['y_true'], ds_results['y_pred_proba'])
            
            roc_data[model_name] = {
                'roc_curve': {'fpr': fpr, 'tpr': tpr},
                'metrics': {'roc_auc': ds_results['roc_auc']}
            }
    
    return roc_data


def run_sensitivity_analysis(
    dataset_files: List[str],
    output_dir: str = 'outputs/sensitivity_analysis',
    use_scaled: bool = True,
    normalize: bool = True,
    model_type: str = 'deepsets',
    epochs: int = 50,
    batch_size: int = 256,
    run_individual: bool = True,
    run_leave_one_out: bool = True,
    save_models: bool = True,
    verbose: bool = True
) -> None:
    """
    Run comprehensive dark sector sensitivity analysis.
    
    This function trains models on different dark sector parameter points
    and evaluates their performance across the full parameter space to
    understand model robustness and systematic uncertainties.
    
    Parameters:
    -----------
    dataset_files : list
        List of dataset files to process (.h5 files)
    output_dir : str
        Directory to save outputs and visualizations
    use_scaled : bool
        Whether to use scaled particle features (recommended: True)
    normalize : bool
        Whether to normalize features (recommended: True)
    model_type : str
        Model architecture ('dense' or 'deepsets')
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    run_individual : bool
        Whether to run individual model training (one model per parameter point)
    run_leave_one_out : bool
        Whether to run leave-one-out training (test generalization)
    save_models : bool
        Whether to save trained model objects (warning: large files)
    verbose : bool
        Whether to print detailed progress information
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print analysis banner
    if verbose:
        print(f"\n{'='*20} DARK SECTOR SENSITIVITY ANALYSIS {'='*20}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {output_dir}")
        print(f"Model type: {model_type}")
        print(f"Using scaled features: {use_scaled}")
        print(f"Feature normalization: {normalize}")
    
    # Identify SM file vs dark sector files
    sm_file = next((f for f in dataset_files if "NominalSM" in f), None)
    if sm_file is None:
        raise ValueError("No Standard Model file found! Expected file with 'NominalSM' in name.")
    
    dark_files = [f for f in dataset_files if f != sm_file]
    
    if verbose:
        print(f"\nDataset Configuration:")
        print(f"  Standard Model file: {os.path.basename(sm_file)}")
        print(f"  Dark sector files ({len(dark_files)}):")
        for f in dark_files:
            params = extract_parameters(f)
            if 'mDark' in params:
                base_name = os.path.basename(f).replace('.h5', '')
                print(f"    - {base_name} (mDark={params['mDark']}, rinv={params['rinv']}, alpha={params['alpha']})")
            else:
                print(f"    - {os.path.basename(f)}")
    
    # Create test datasets for cross-evaluation
    if verbose:
        print(f"\n{'-'*30}")
        print(f"Preparing test datasets for cross-evaluation...")
    
    test_datasets = {}
    for dark_file in dark_files:
        ds_name = os.path.basename(dark_file).replace('.h5', '')
        
        if verbose:
            print(f"  Processing {ds_name}...")
        
        # Create combined dataset (dark + SM)
        combined_data = create_dataset(
            [dark_file, sm_file],
            use_scaled=use_scaled,
            signal_background_mode=True,
            verbose=False
        )
        
        # Prepare ML dataset
        ml_data = prepare_ml_dataset(
            combined_data,
            normalize=normalize,
            verbose=False
        )
        
        # Prepare for model architecture
        if model_type.lower() == 'deepsets':
            prepared_data = prepare_deepsets_data(ml_data)
        else:
            prepared_data = ml_data
        
        test_datasets[ds_name] = prepared_data
    
    # Results container
    results = {
        'config': {
            'dataset_files': dataset_files,
            'output_dir': output_dir,
            'use_scaled': use_scaled,
            'normalize': normalize,
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Individual model training and evaluation
    if run_individual:
        if verbose:
            print(f"\n{'-'*30}")
            print(f"INDIVIDUAL MODEL TRAINING")
            print(f"Training one model per physics parameter point...")
        
        start_time = time.time()
        
        # Train models
        individual_results = train_individual_models(
            dark_files=dark_files,
            sm_file=sm_file,
            use_scaled=use_scaled,
            normalize=normalize,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        results['individual_models'] = individual_results
        
        # Cross-evaluation
        if verbose:
            print(f"\n{'-'*20}")
            print(f"Cross-evaluating individual models...")
        
        individual_cross_eval = evaluate_cross_model_performance(
            all_models=individual_results,
            all_test_datasets=test_datasets,
            verbose=verbose
        )
        
        results['individual_cross_evaluation'] = individual_cross_eval
        
        # Create model-dataset mapping for ROC plots
        model_dataset_mapping = {}
        for model_name in individual_results.keys():
            model_dataset_mapping[model_name] = match_model_to_dataset(
                model_name, list(test_datasets.keys())
            )
        
        # Prepare ROC data efficiently
        individual_roc_data = prepare_roc_data_from_cross_eval(
            individual_cross_eval, model_dataset_mapping
        )
        
        # Create visualizations
        if verbose:
            print(f"\n{'-'*20}")
            print(f"Creating visualizations for individual models...")
        
        # Combined ROC curves
        plot_combined_roc_curves(
            all_results=individual_roc_data,
            title="ROC Curves: Individual Models on Corresponding Test Sets",
            save_path=f"{output_dir}/individual_combined_roc.png"
        )
        
        # Cross-model performance heatmap
        plot_cross_model_heatmap(
            cross_eval_results=individual_cross_eval,
            metric='roc_auc',
            title="Cross-Model Performance Matrix (ROC AUC)",
            save_path=f"{output_dir}/individual_cross_model_heatmap.png"
        )
        
        # Per-model cross-dataset ROC curves
        for model_name in individual_results.keys():
            plot_cross_model_roc_curves(
                cross_eval_results=individual_cross_eval,
                model_name=model_name,
                title=f"ROC Curves: {model_name} on Different Parameter Points",
                save_path=f"{output_dir}/individual_{model_name}_cross_roc.png"
            )
            
            # Training history
            plot_training_history(
                history=individual_results[model_name]['history'],
                model_name=model_name,
                save_path=f"{output_dir}/individual_{model_name}_training_history.png"
            )
        
        training_time = time.time() - start_time
        if verbose:
            print(f"\nIndividual model analysis completed in {training_time:.2f} seconds")
    
    # Leave-one-out training and evaluation
    if run_leave_one_out:
        if verbose:
            print(f"\n{'-'*30}")
            print(f"LEAVE-ONE-OUT MODEL TRAINING")
            print(f"Training models with one parameter point held out for testing...")
        
        start_time = time.time()
        
        # Train LOO models
        loo_results = train_leave_one_out_models(
            dark_files=dark_files,
            sm_file=sm_file,
            use_scaled=use_scaled,
            normalize=normalize,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        results['leave_one_out_models'] = loo_results
        
        # Cross-evaluation
        if verbose:
            print(f"\n{'-'*20}")
            print(f"Cross-evaluating leave-one-out models...")
        
        loo_cross_eval = evaluate_cross_model_performance(
            all_models=loo_results,
            all_test_datasets=test_datasets,
            verbose=verbose
        )
        
        results['leave_one_out_cross_evaluation'] = loo_cross_eval
        
        # Create model-dataset mapping for LOO models
        loo_model_dataset_mapping = {}
        for model_name in loo_results.keys():
            # For LOO models, map to the left-out dataset
            left_out_file = loo_results[model_name]['left_out_file']
            left_out_name = os.path.basename(left_out_file).replace('.h5', '')
            loo_model_dataset_mapping[model_name] = left_out_name
        
        # Prepare ROC data
        loo_roc_data = prepare_roc_data_from_cross_eval(
            loo_cross_eval, loo_model_dataset_mapping
        )
        
        # Create visualizations
        if verbose:
            print(f"\n{'-'*20}")
            print(f"Creating visualizations for leave-one-out models...")
        
        # Combined ROC curves
        plot_combined_roc_curves(
            all_results=loo_roc_data,
            title="ROC Curves: Leave-One-Out Models on Held-Out Test Sets",
            save_path=f"{output_dir}/loo_combined_roc.png"
        )
        
        # Cross-model performance heatmap
        plot_cross_model_heatmap(
            cross_eval_results=loo_cross_eval,
            metric='roc_auc',
            title="Leave-One-Out Cross-Model Performance Matrix (ROC AUC)",
            save_path=f"{output_dir}/loo_cross_model_heatmap.png"
        )
        
        # Per-model cross-dataset ROC curves
        for model_name in loo_results.keys():
            plot_cross_model_roc_curves(
                cross_eval_results=loo_cross_eval,
                model_name=model_name,
                title=f"ROC Curves: {model_name} Generalization Test",
                save_path=f"{output_dir}/loo_{model_name}_cross_roc.png"
            )
            
            # Training history
            plot_training_history(
                history=loo_results[model_name]['history'],
                model_name=model_name,
                save_path=f"{output_dir}/loo_{model_name}_training_history.png"
            )
        
        training_time = time.time() - start_time
        if verbose:
            print(f"\nLeave-one-out analysis completed in {training_time:.2f} seconds")
    
    # Save all results
    if verbose:
        print(f"\n{'-'*30}")
        print(f"Saving results...")
    
    results_file = f"{output_dir}/sensitivity_analysis_results"
    save_results(results, results_file, save_models=save_models)
    
    # Final summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"DARK SECTOR SENSITIVITY ANALYSIS COMPLETED")
        print(f"{'='*60}")
        print(f"Total runtime: {(time.time() - start_time if 'start_time' in locals() else 0):.2f} seconds")
        print(f"Output directory: {output_dir}")
        print(f"Results saved to: {results_file}.json")
        if save_models:
            print(f"Models saved to: {results_file}.pkl")
        print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")


def main():
    """Command-line interface for sensitivity analysis."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive dark sector sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default parameters
  python scripts/run_sensitivity_analysis.py --dataset-dir data/processed

  # Custom analysis with specific parameters  
  python scripts/run_sensitivity_analysis.py \\
    --dataset-dir data/processed \\
    --output-dir outputs/my_analysis \\
    --model-type deepsets \\
    --epochs 100 \\
    --no-save-models

  # Quick test run (individual models only)
  python scripts/run_sensitivity_analysis.py \\
    --dataset-dir data/processed \\
    --epochs 10 \\
    --no-leave-one-out \\
    --verbose
        """)
    
    parser.add_argument(
        '--dataset-dir', 
        type=str, 
        required=True,
        help='Directory containing .h5 dataset files'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs/sensitivity_analysis',
        help='Output directory for results and plots (default: outputs/sensitivity_analysis)'
    )
    
    parser.add_argument(
        '--model-type', 
        type=str, 
        choices=['dense', 'deepsets'], 
        default='deepsets',
        help='Model architecture (default: deepsets)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Maximum training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=256,
        help='Training batch size (default: 256)'
    )
    
    parser.add_argument(
        '--no-scaled', 
        action='store_true',
        help='Use raw features instead of scaled features'
    )
    
    parser.add_argument(
        '--no-normalize', 
        action='store_true',
        help='Skip feature normalization'
    )
    
    parser.add_argument(
        '--no-individual', 
        action='store_true',
        help='Skip individual model training'
    )
    
    parser.add_argument(
        '--no-leave-one-out', 
        action='store_true',
        help='Skip leave-one-out model training'
    )
    
    parser.add_argument(
        '--no-save-models', 
        action='store_true',
        help='Do not save trained model objects (saves disk space)'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Find dataset files
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    dataset_files = list(dataset_dir.glob("*.h5"))
    if not dataset_files:
        print(f"Error: No .h5 files found in {dataset_dir}")
        sys.exit(1)
    
    dataset_files = [str(f) for f in dataset_files]
    
    # Check for SM file
    sm_files = [f for f in dataset_files if "NominalSM" in f]
    if not sm_files:
        print("Error: No Standard Model file found! Expected file with 'NominalSM' in name.")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Found {len(dataset_files)} dataset files in {dataset_dir}")
    
    # Run analysis
    try:
        run_sensitivity_analysis(
            dataset_files=dataset_files,
            output_dir=args.output_dir,
            use_scaled=not args.no_scaled,
            normalize=not args.no_normalize,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            run_individual=not args.no_individual,
            run_leave_one_out=not args.no_leave_one_out,
            save_models=not args.no_save_models,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
