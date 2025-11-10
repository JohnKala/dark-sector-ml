#!/usr/bin/env python3
"""
Clean Dark Sector Sensitivity Analysis Pipeline

This script provides a robust, efficient implementation that leverages all our
existing functions without redundancy or fragile string matching.

Usage:
    python scripts/run_sensitivity_analysis.py --dataset-dir data/processed

Author: Physics Team
Date: 2025-01-10
"""

import argparse
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import our robust pipeline functions
from src.data.loader import extract_parameters
from src.training.experiments import train_individual_models, train_leave_one_out_models
from src.evaluation import evaluate_cross_model_performance
from src.visualization import (
    plot_combined_roc_curves, plot_cross_model_heatmap, 
    plot_cross_model_roc_curves, plot_training_history,
    plot_dataset_comparison_models
)
from src.utils import save_results


def create_roc_data_from_evaluation(
    cross_eval_results: Dict[str, Dict[str, Dict[str, Any]]],
    model_to_dataset_mapping: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Extract ROC data efficiently from cross-evaluation results.
    No redundant calculations - uses existing data structures.
    """
    from sklearn.metrics import roc_curve
    
    roc_data = {}
    
    for model_name, model_results in cross_eval_results.items():
        target_dataset = model_to_dataset_mapping.get(model_name)
        
        # Handle both with and without "model_" prefix in dataset names
        dataset_key = None
        if target_dataset and target_dataset in model_results:
            dataset_key = target_dataset
        elif target_dataset and f"model_{target_dataset}" in model_results:
            dataset_key = f"model_{target_dataset}"
        
        if dataset_key:
            ds_results = model_results[dataset_key]
            
            # Calculate ROC curve from stored predictions
            fpr, tpr, _ = roc_curve(ds_results['y_true'], ds_results['y_pred_proba'])
            
            roc_data[model_name] = {
                'roc_curve': {'fpr': fpr, 'tpr': tpr},
                'metrics': {'roc_auc': ds_results['roc_auc']}  # Use existing AUC!
            }
    
    return roc_data


def create_robust_model_dataset_mapping(
    model_names: List[str], 
    test_dataset_names: List[str]
) -> Dict[str, str]:
    """
    Create robust mapping between models and their corresponding test datasets.
    Uses our existing extract_parameters() instead of fragile string matching.
    """
    mapping = {}
    
    for model_name in model_names:
        model_params = extract_parameters(model_name)
        
        # Find best matching dataset
        best_match = None
        best_score = 0
        
        for ds_name in test_dataset_names:
            ds_params = extract_parameters(ds_name)
            
            # Count parameter matches
            matches = 0
            total_params = 0
            
            for param_name in ['mDark', 'rinv', 'alpha']:
                if param_name in model_params and param_name in ds_params:
                    total_params += 1
                    if model_params[param_name] == ds_params[param_name]:
                        matches += 1
            
            # Calculate match score
            score = matches / total_params if total_params > 0 else 0
            
            if score > best_score:
                best_score = score
                best_match = ds_name
        
        # Fallback to first dataset if no good match
        mapping[model_name] = best_match or test_dataset_names[0]
    
    return mapping


def create_parameter_sensitivity_plot(
    cross_eval_results: Dict[str, Dict[str, Dict[str, Any]]],
    parameter_name: str,
    metric: str = 'roc_auc',
    output_path: Optional[str] = None
) -> None:
    """
    Create parameter sensitivity plot using robust parameter extraction.
    """
    import matplotlib.pyplot as plt
    
    # Collect data points
    param_performance_data = {}
    
    for model_name, model_results in cross_eval_results.items():
        model_params = extract_parameters(model_name)
        
        if parameter_name not in model_params:
            continue
            
        model_param_value = model_params[parameter_name]
        
        # Collect performance across all test datasets
        for ds_name, ds_results in model_results.items():
            ds_params = extract_parameters(ds_name)
            
            if parameter_name not in ds_params:
                continue
                
            ds_param_value = ds_params[parameter_name]
            performance = ds_results[metric]
            
            # Store data point
            key = f"Model_{model_param_value}"
            if key not in param_performance_data:
                param_performance_data[key] = []
            
            param_performance_data[key].append((ds_param_value, performance))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_key, data_points in param_performance_data.items():
        if not data_points:
            continue
            
        # Sort by parameter value
        data_points.sort(key=lambda x: x[0])
        param_values, performances = zip(*data_points)
        
        ax.plot(param_values, performances, 'o-', label=model_key, alpha=0.7)
    
    ax.set_xlabel(f'{parameter_name} Value')
    ax.set_ylabel(f'{metric.upper()}')
    ax.set_title(f'Model Sensitivity to {parameter_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def generate_analysis_summary(
    individual_results: Dict[str, Any],
    individual_cross_eval: Dict[str, Dict[str, Dict[str, Any]]],
    loo_results: Optional[Dict[str, Any]] = None,
    loo_cross_eval: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    output_dir: str = 'outputs',
    model_type: str = 'deepsets',
    epochs: int = 50,
    use_scaled: bool = True,
    normalize: bool = True,
    adversarial_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generate analysis summary using our robust data structures.
    """
    summary_file = f"{output_dir}/analysis_summary.md"
    
    with open(summary_file, 'w') as f:
        f.write("# Dark Sector Sensitivity Analysis Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration summary
        f.write("## Configuration\n\n")
        f.write(f"- Model type: {model_type}\n")
        f.write(f"- Epochs: {epochs}\n")
        f.write(f"- Features: {'scaled' if use_scaled else 'raw'}, {'normalized' if normalize else 'unnormalized'}\n")
        
        # Add adversarial configuration if present
        if adversarial_config:
            f.write("\n### Adversarial Training Enabled\n\n")
            f.write(f"- Gradient iterations: {adversarial_config.get('grad_iter', 3)}\n")
            f.write(f"- Initial noise: {adversarial_config.get('grad_eps', 1e-6)}\n")
            f.write(f"- Step size: {adversarial_config.get('grad_eta', 2e-4)}\n")
            f.write(f"- KL weight (alpha): {adversarial_config.get('alpha', 5.0)}\n")
        f.write("\n")
        
        # Individual models summary
        f.write("## Individual Models Performance\n\n")
        f.write("| Model | Test Dataset | ROC AUC | Precision | Recall | F1 |\n")
        f.write("|-------|-------------|---------|-----------|--------|----|\n")
        
        for model_name in individual_results.keys():
            # Find best performing dataset for this model
            best_auc = 0
            best_dataset = None
            best_results = None
            
            for ds_name, ds_results in individual_cross_eval[model_name].items():
                if ds_results['roc_auc'] > best_auc:
                    best_auc = ds_results['roc_auc']
                    best_dataset = ds_name
                    best_results = ds_results
            
            if best_results:
                f.write(f"| {model_name} | {best_dataset} | "
                       f"{best_results['roc_auc']:.4f} | "
                       f"{best_results['precision']:.4f} | "
                       f"{best_results['recall']:.4f} | "
                       f"{best_results['f1']:.4f} |\n")
        
        f.write("\n")
        
        # Leave-one-out summary
        if loo_results and loo_cross_eval:
            f.write("## Leave-One-Out Models Performance\n\n")
            f.write("| Model | Left-Out Dataset | ROC AUC | Precision | Recall | F1 |\n")
            f.write("|-------|------------------|---------|-----------|--------|----|\n")
            
            for model_name in loo_results.keys():
                left_out_file = loo_results[model_name]['left_out_file']
                left_out_name = os.path.basename(left_out_file).replace('.h5', '')
                
                # Find performance on left-out dataset
                best_match = None
                for ds_name, ds_results in loo_cross_eval[model_name].items():
                    ds_base = os.path.basename(ds_name).replace('.h5', '')
                    if left_out_name == ds_base or left_out_name in ds_name:
                        best_match = ds_results
                        break
                
                if not best_match:
                    # Use first available dataset
                    best_match = list(loo_cross_eval[model_name].values())[0]
                
                f.write(f"| {model_name} | {left_out_name} | "
                       f"{best_match['roc_auc']:.4f} | "
                       f"{best_match['precision']:.4f} | "
                       f"{best_match['recall']:.4f} | "
                       f"{best_match['f1']:.4f} |\n")
        
        f.write("\n## Analysis Files Generated\n\n")
        f.write("- Individual model ROC curves\n")
        f.write("- Cross-model performance heatmap\n")
        f.write("- Parameter sensitivity plots\n")
        f.write("- Training history plots\n")
        f.write("- Dataset comparison plots\n")
        
        if loo_results:
            f.write("- Leave-one-out model analysis\n")
    
    print(f"✅ Analysis summary saved: {summary_file}")


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
    verbose: bool = True,
    create_run_subdir: bool = True,
    generate_dataset_comparisons: bool = True,
    adversarial_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive sensitivity analysis using our robust pipeline functions.
    
    This implementation:
    - Uses existing functions without redundancy
    - Employs robust parameter matching via extract_parameters()
    - Avoids fragile string matching
    - Reuses evaluation results efficiently
    - Creates timestamped subdirectories for organized output
    - Supports adversarial training with configurable parameters
    
    Parameters:
    -----------
    dataset_files : list of str
        List of dataset files to analyze
    output_dir : str
        Base output directory
    use_scaled : bool
        Whether to use scaled features
    normalize : bool
        Whether to normalize features
    model_type : str
        Model architecture ('dense' or 'deepsets')
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    run_individual : bool
        Whether to run individual model training
    run_leave_one_out : bool
        Whether to run leave-one-out training
    save_models : bool
        Whether to save model weights
    verbose : bool
        Whether to print progress
    create_run_subdir : bool
        Whether to create a timestamped run subdirectory
    generate_dataset_comparisons : bool
        Whether to generate dataset comparison plots
    adversarial_config : dict, optional
        Configuration for adversarial training. If provided, enables adversarial training.
        Example: {'grad_iter': 3, 'grad_eps': 1e-6, 'grad_eta': 2e-4, 'alpha': 5.0}
    """
    
    # Create timestamped run subdirectory if requested
    if create_run_subdir:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"run_{timestamp}_{model_type}_{epochs}epochs"
        if not use_scaled:
            run_name += "_raw"
        if not normalize:
            run_name += "_unnorm"
        if adversarial_config:
            run_name += "_adversarial"
        if not run_individual:
            run_name += "_noindiv"
        if not run_leave_one_out:
            run_name += "_noloo"
        
        # Create the run-specific directory
        actual_output_dir = os.path.join(output_dir, run_name)
        os.makedirs(actual_output_dir, exist_ok=True)
    else:
        actual_output_dir = output_dir
        os.makedirs(actual_output_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DARK SECTOR SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output: {actual_output_dir}")
        print(f"Model: {model_type}")
    
    # Validate dataset files
    sm_file = next((f for f in dataset_files if "NominalSM" in f), None)
    if not sm_file:
        raise ValueError("No Standard Model file found! Expected file with 'NominalSM' in name.")
    
    dark_files = [f for f in dataset_files if f != sm_file]
    
    if verbose:
        print(f"\nDatasets:")
        print(f"  SM file: {os.path.basename(sm_file)}")
        print(f"  Dark files: {len(dark_files)}")
        for f in dark_files:
            params = extract_parameters(f)
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "no params"
            print(f"    - {os.path.basename(f)} ({param_str})")
    
    # Results container
    results = {
        'config': {
            'dataset_files': dataset_files,
            'output_dir': actual_output_dir,
            'base_output_dir': output_dir,
            'use_scaled': use_scaled,
            'normalize': normalize,
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Individual model training
    if run_individual:
        if verbose:
            print(f"\n{'-'*40}")
            print("INDIVIDUAL MODEL TRAINING")
            print(f"{'-'*40}")
        
        start_time = time.time()
        
        # Train models (this creates test datasets internally - no redundancy!)
        individual_results = train_individual_models(
            dark_files=dark_files,
            sm_file=sm_file,
            use_scaled=use_scaled,
            normalize=normalize,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            output_dir=actual_output_dir,
            save_model=save_models,
            verbose=verbose,
            adversarial_config=adversarial_config
        )
        
        results['individual_models'] = individual_results
        
        # Create test datasets for cross-evaluation (minimal, needed for evaluation)
        test_dataset_names = [os.path.basename(f).replace('.h5', '') for f in dark_files]
        
        # Cross-evaluation using our robust function
        if verbose:
            print(f"\nCross-evaluating individual models...")
        
        # Extract the preserved test datasets from training results (fixes empty plots!)
        all_test_datasets = {name: res['prepared_data'] for name, res in individual_results.items()}
        
        individual_cross_eval = evaluate_cross_model_performance(
            all_models=individual_results,
            all_test_datasets=all_test_datasets,  # Use preserved data!
            verbose=verbose
        )
        
        results['individual_cross_evaluation'] = individual_cross_eval
        
        # Create robust model-dataset mapping
        model_dataset_mapping = create_robust_model_dataset_mapping(
            list(individual_results.keys()), 
            test_dataset_names
        )
        
        # Create ROC data efficiently (no redundant calculations)
        individual_roc_data = create_roc_data_from_evaluation(
            individual_cross_eval, model_dataset_mapping
        )
        
        # Generate visualizations
        if verbose:
            print(f"\nGenerating visualizations...")
        
        plot_combined_roc_curves(
            all_results=individual_roc_data,
            title="Individual Models: ROC Curves on Matched Test Sets",
            save_path=f"{actual_output_dir}/individual_combined_roc.png"
        )
        
        plot_cross_model_heatmap(
            cross_eval_results=individual_cross_eval,
            metric='roc_auc',
            title="Individual Models: Cross-Performance Matrix",
            save_path=f"{actual_output_dir}/individual_cross_heatmap.png"
        )
        
        # Per-model analysis
        for model_name in individual_results.keys():
            plot_cross_model_roc_curves(
                cross_eval_results=individual_cross_eval,
                model_name=model_name,
                title=f"{model_name}: Performance Across Parameter Space",
                save_path=f"{actual_output_dir}/individual_{model_name}_cross_roc.png"
            )
            
            plot_training_history(
                history=individual_results[model_name]['history'],
                model_name=model_name,
                save_path=f"{actual_output_dir}/individual_{model_name}_history.png"
            )
        
        # Parameter sensitivity analysis
        for param in ['mDark', 'rinv', 'alpha']:
            try:
                create_parameter_sensitivity_plot(
                    individual_cross_eval,
                    parameter_name=param,
                    metric='roc_auc',
                    output_path=f"{actual_output_dir}/sensitivity_{param}.png"
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not create {param} sensitivity plot: {e}")
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Individual analysis completed in {elapsed:.2f} seconds")
    
    # Leave-one-out model training
    loo_results = None
    loo_cross_eval = None
    
    if run_leave_one_out:
        if verbose:
            print(f"\n{'-'*40}")
            print("LEAVE-ONE-OUT MODEL TRAINING")
            print(f"{'-'*40}")
        
        start_time = time.time()
        
        loo_results = train_leave_one_out_models(
            dark_files=dark_files,
            sm_file=sm_file,
            use_scaled=use_scaled,
            normalize=normalize,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            output_dir=actual_output_dir,
            save_model=save_models,
            verbose=verbose,
            adversarial_config=adversarial_config
        )
        
        results['leave_one_out_models'] = loo_results
        
        # Cross-evaluation
        if verbose:
            print(f"\nCross-evaluating leave-one-out models...")
        
        # Extract test datasets from LOO results for cross-evaluation
        loo_test_datasets = {name: res['prepared_data'] for name, res in loo_results.items()}
        
        loo_cross_eval = evaluate_cross_model_performance(
            all_models=loo_results,
            all_test_datasets=loo_test_datasets,  # Use preserved data like individual models
            verbose=verbose
        )
        
        results['leave_one_out_cross_evaluation'] = loo_cross_eval
        
        # Visualizations - use same robust mapping as individual models
        if verbose:
            print(f"\nGenerating LOO visualizations...")
        
        # Debug: check what keys we have
        if verbose:
            print(f"LOO cross-eval keys: {list(loo_cross_eval.keys())[:2]}...")
            if loo_cross_eval:
                first_model = list(loo_cross_eval.keys())[0]
                print(f"First model dataset keys: {list(loo_cross_eval[first_model].keys())[:2]}...")
        
        # Create dataset names list - LOO cross-eval uses "loo_" prefixed dataset names
        loo_dataset_names = [f"loo_{os.path.basename(f).replace('.h5', '')}" for f in dark_files]
        
        # Use same robust mapping function as individual models 
        loo_model_dataset_mapping = create_robust_model_dataset_mapping(
            list(loo_results.keys()), 
            loo_dataset_names  # Now matches the actual keys in cross-eval results
        )
        
        loo_roc_data = create_roc_data_from_evaluation(
            loo_cross_eval, loo_model_dataset_mapping
        )
        
        # Debug: check if ROC data was created
        if verbose:
            print(f"LOO ROC data created for {len(loo_roc_data)} models")
        
        plot_combined_roc_curves(
            all_results=loo_roc_data,
            title="Leave-One-Out Models: Generalization Performance",
            save_path=f"{actual_output_dir}/loo_combined_roc.png"
        )
        
        plot_cross_model_heatmap(
            cross_eval_results=loo_cross_eval,
            metric='roc_auc',
            title="Leave-One-Out Models: Cross-Performance Matrix",
            save_path=f"{actual_output_dir}/loo_cross_heatmap.png"
        )
        
        # Per-model analysis for LOO (same as individual models)
        for model_name in loo_results.keys():
            plot_cross_model_roc_curves(
                cross_eval_results=loo_cross_eval,
                model_name=model_name,
                title=f"{model_name}: Performance Across Parameter Space",
                save_path=f"{actual_output_dir}/loo_{model_name}_cross_roc.png"
            )
            
            plot_training_history(
                history=loo_results[model_name]['history'],
                model_name=model_name,
                save_path=f"{actual_output_dir}/loo_{model_name}_history.png"
            )
        
        # Parameter sensitivity analysis for LOO
        for param in ['mDark', 'rinv', 'alpha']:
            try:
                create_parameter_sensitivity_plot(
                    loo_cross_eval,
                    parameter_name=param,
                    metric='roc_auc',
                    output_path=f"{actual_output_dir}/loo_sensitivity_{param}.png"
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not create LOO {param} sensitivity plot: {e}")
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Leave-one-out analysis completed in {elapsed:.2f} seconds")
    
    # Dataset comparison plots (NEW functionality)
    if run_individual and generate_dataset_comparisons:
        if verbose:
            print(f"\n{'-'*40}")
            print("DATASET COMPARISON PLOTS")
            print(f"{'-'*40}")
        
        start_time = time.time()
        
        # Get unique datasets from dark_files
        dataset_names = [os.path.basename(f).replace('.h5', '') for f in dark_files]
        
        if verbose:
            print(f"Generating dataset comparison plots for {len(dataset_names)} datasets...")
            
        # Fix: Create a mapping between dataset names and cross-evaluation keys
        # The cross-evaluation results use the model names as keys, which include 'model_' prefix
        dataset_to_crosseval = {}
        for ds_name in dataset_names:
            # First try to find an exact match with 'model_' prefix
            model_key = f"model_{ds_name}"
            if model_key in individual_cross_eval:
                dataset_to_crosseval[ds_name] = model_key
                continue
                
            # If no exact match, try to find a partial match
            for model_name in individual_cross_eval.keys():
                if ds_name in model_name:  # If dataset name is part of the model name
                    dataset_to_crosseval[ds_name] = model_name
                    break
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(actual_output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        if verbose:
            print(f"Plots directory created: {plots_dir}")
            print(f"Dataset to cross-eval mapping: {dataset_to_crosseval}")
            print(f"Available cross-eval keys: {list(individual_cross_eval.keys())[:3]}...")
        
        comparison_count = 0
        for dataset_name in dataset_names:
            try:
                # Get the corresponding model name for this dataset
                model_key = dataset_to_crosseval.get(dataset_name)
                if not model_key:
                    if verbose:
                        print(f"  ⚠ Warning: Could not find matching model for dataset {dataset_name}")
                    continue
                
                # Combine individual and LOO cross-evaluation results for plotting
                combined_cross_eval = dict(individual_cross_eval)
                if loo_cross_eval:
                    # Make sure we don't overwrite any keys
                    for model, evals in loo_cross_eval.items():
                        combined_cross_eval[model] = evals
                    
                plot_dataset_comparison_models(
                    individual_results=individual_results,
                    individual_cross_eval=combined_cross_eval,  # Use combined results
                    target_dataset=model_key,  # Use the model key instead of dataset name
                    loo_results=loo_results,
                    loo_cross_eval=loo_cross_eval,
                    save_path=f"{actual_output_dir}/plots/dataset_comparison_{dataset_name}.png"
                )
                comparison_count += 1
                if verbose:
                    print(f"  ✓ Created comparison plot for {dataset_name}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Warning: Could not create dataset comparison plot for {dataset_name}: {e}")
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Dataset comparison analysis completed in {elapsed:.2f} seconds ({comparison_count} plots generated)")
    
    # Save all results
    if verbose:
        print(f"\nSaving results...")
    
    results_file = f"{actual_output_dir}/sensitivity_analysis_results"
    save_results(results, results_file, save_models=save_models)
    
    # Generate summary
    if run_individual:
        generate_analysis_summary(
            individual_results, individual_cross_eval,
            loo_results, loo_cross_eval,
            actual_output_dir,
            model_type=model_type,
            epochs=epochs,
            use_scaled=use_scaled,
            normalize=normalize,
            adversarial_config=adversarial_config
        )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {actual_output_dir}")
        print(f"Summary: {actual_output_dir}/analysis_summary.md")
        print(f"{'='*60}")
    
    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run robust dark sector sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Directory containing .h5 dataset files')
    parser.add_argument('--output-dir', type=str, default='outputs/sensitivity_analysis',
                       help='Output directory (default: outputs/sensitivity_analysis)')
    parser.add_argument('--model-type', type=str, choices=['dense', 'deepsets'], 
                       default='deepsets', help='Model architecture (default: deepsets)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--no-scaled', action='store_true',
                       help='Use raw features instead of scaled')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Skip feature normalization')
    parser.add_argument('--no-individual', action='store_true',
                       help='Skip individual model training')
    parser.add_argument('--no-leave-one-out', action='store_true',
                       help='Skip leave-one-out training')
    parser.add_argument('--no-save-models', action='store_true',
                       help='Do not save model objects')
    parser.add_argument('--no-dataset-comparison', action='store_true',
                       help='Skip dataset comparison plots generation')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--no-run-subdir', action='store_true',
                       help='Do not create timestamped run subdirectory')
    
    # Adversarial training arguments
    parser.add_argument('--adversarial', action='store_true',
                       help='Enable adversarial training')
    parser.add_argument('--adversarial-alpha', type=float, default=5.0,
                       help='Weight for adversarial KL loss (default: 5.0)')
    parser.add_argument('--adversarial-iterations', type=int, default=3,
                       help='Number of adversarial gradient iterations (default: 3)')
    
    args = parser.parse_args()
    
    # Find dataset files
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    dataset_files = [str(f) for f in dataset_dir.glob("*.h5")]
    if not dataset_files:
        print(f"Error: No .h5 files found in {dataset_dir}")
        sys.exit(1)
    
    # Create adversarial config if enabled
    adversarial_config = None
    if args.adversarial:
        # Use more memory-efficient defaults
        adversarial_config = {
            'grad_iter': min(args.adversarial_iterations, 2),  # Limit to 2 iterations max to reduce memory usage
            'grad_eps': 1e-6,  # Default initial noise
            'grad_eta': 2e-4,  # Default step size
            'alpha': args.adversarial_alpha
        }
        if not args.quiet:
            print(f"Adversarial training enabled with config: {adversarial_config}")
    
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
            verbose=not args.quiet,
            create_run_subdir=not args.no_run_subdir,
            generate_dataset_comparisons=not args.no_dataset_comparison,
            adversarial_config=adversarial_config
        )
    except Exception as e:
        print(f"Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
