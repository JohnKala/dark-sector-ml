#!/usr/bin/env python3
"""
Script to run comparative analysis between standard and adversarial training.

This script provides a command-line interface for running comparative
analysis between standard and adversarial training regimes, including
robustness testing and performance evaluation.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from src modules
from src.evaluation.cross_evaluation import run_comparative_training, evaluate_model_robustness
from src.visualization.comparison_plots import (
    plot_comparative_heatmaps, 
    plot_perturbation_performance,
    plot_robustness_scores,
    plot_training_history_comparison
)


def run_adversarial_comparison(
    dataset_files: List[str],
    output_dir: str,
    model_type: str = 'deepsets',
    epochs: int = 50,
    batch_size: int = 128,
    adversarial_alpha: float = 5.0,
    adversarial_iterations: int = 10,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comparative analysis between standard and adversarial training.
    
    Parameters:
    -----------
    dataset_files : List[str]
        List of dataset files to use
    output_dir : str
        Directory to save outputs
    model_type : str
        Type of model to use ('deepsets' or 'dense')
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    adversarial_alpha : float
        Weight for adversarial KL loss
    adversarial_iterations : int
        Number of adversarial gradient iterations
    random_seed : int
        Random seed for reproducibility
    verbose : bool
        Whether to print verbose output
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results
    """
    start_time = time.time()
    
    # Create adversarial config
    adversarial_config = {
        'grad_iter': adversarial_iterations,
        'grad_eps': 1e-6,
        'grad_eta': 2e-4,
        'alpha': adversarial_alpha
    }
    
    # Run comparative training
    if verbose:
        print(f"Running comparative training with {len(dataset_files)} dataset files...")
        print(f"Adversarial config: {adversarial_config}")
    
    training_results = run_comparative_training(
        dataset_files=dataset_files,
        output_dir=output_dir,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        adversarial_config=adversarial_config,
        random_seed=random_seed,
        verbose=verbose
    )
    
    # Evaluate robustness
    if verbose:
        print("\nEvaluating model robustness...")
    
    standard_model = training_results['standard']['model']
    
    # Use the adapted_data from training_results
    standard_test_data = {
        'test': training_results['standard']['training_results']['test']
    }
    
    standard_robustness = evaluate_model_robustness(
        model=standard_model,
        test_data=standard_test_data,
        model_type=model_type,
        verbose=verbose
    )
    
    adversarial_robustness = None
    if 'adversarial' in training_results:
        adversarial_model = training_results['adversarial']['model']
        
        # Use the adapted_data from training_results
        adversarial_test_data = {
            'test': training_results['adversarial']['training_results']['test']
        }
        
        adversarial_robustness = evaluate_model_robustness(
            model=adversarial_model,
            test_data=adversarial_test_data,
            model_type=model_type,
            verbose=verbose
        )
    
    # Generate visualizations
    if verbose:
        print("\nGenerating visualizations...")
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'comparative', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot training history comparison if adversarial results available
    if 'adversarial' in training_results:
        standard_history = training_results['standard']['training_results']['history']
        adversarial_history = training_results['adversarial']['training_results']['history']
        
        history_fig = plot_training_history_comparison(
            standard_history=standard_history,
            adversarial_history=adversarial_history,
            output_path=os.path.join(vis_dir, 'training_history_comparison.png')
        )
    
    # Plot robustness comparison if both robustness results available
    if adversarial_robustness is not None:
        # Get perturbation strengths
        perturbation_strengths = list(standard_robustness['perturbed'].keys())
        
        # Plot performance under perturbation
        perf_fig = plot_perturbation_performance(
            standard_model_results=standard_robustness,
            adversarial_model_results=adversarial_robustness,
            perturbation_strengths=perturbation_strengths,
            output_path=os.path.join(vis_dir, 'perturbation_performance.png')
        )
        
        # Plot robustness scores
        robust_fig = plot_robustness_scores(
            standard_scores=standard_robustness['robustness_scores'],
            adversarial_scores=adversarial_robustness['robustness_scores'],
            perturbation_strengths=perturbation_strengths,
            output_path=os.path.join(vis_dir, 'robustness_scores.png')
        )
    
    # Calculate total runtime
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\nComparative analysis completed in {total_time:.2f} seconds.")
        print(f"Results saved to {output_dir}")
    
    # Collect all results
    results = {
        'standard': {
            'training': training_results['standard'],
            'robustness': standard_robustness
        }
    }
    
    if 'adversarial' in training_results:
        results['adversarial'] = {
            'training': training_results['adversarial'],
            'robustness': adversarial_robustness
        }
    
    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run comparative analysis between standard and adversarial training."
    )
    
    # Required arguments
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Directory containing dataset files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save outputs')
    
    # Optional arguments
    parser.add_argument('--model-type', type=str, default='deepsets',
                       choices=['deepsets', 'dense'],
                       help='Type of model to use (default: deepsets)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    # Adversarial training arguments
    parser.add_argument('--adversarial-alpha', type=float, default=5.0,
                       help='Weight for adversarial KL loss (default: 5.0)')
    parser.add_argument('--adversarial-iterations', type=int, default=10,
                       help='Number of adversarial gradient iterations (default: 10)')
    
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run comparative analysis
    try:
        run_adversarial_comparison(
            dataset_files=dataset_files,
            output_dir=str(output_dir),
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            adversarial_alpha=args.adversarial_alpha,
            adversarial_iterations=args.adversarial_iterations,
            random_seed=args.random_seed,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
