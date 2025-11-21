"""
Cross-evaluation utilities for comparing standard and adversarial training.

This module provides functions for running comparative analyses between
standard and adversarial training regimes, including robustness testing
and performance evaluation across different parameter points.
"""

import os
import time
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Conditional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Import from other modules
from ..data.loader import load_dataset
from ..data.preparation import create_dataset
from ..data.preprocessor import prepare_ml_dataset
from ..training.trainer import train_model
from ..models.adversarial import AdversarialExampleGenerator
from .metrics import evaluate_model


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Parameters:
    -----------
    seed : int
        Random seed to use
    """
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def run_comparative_training(
    dataset_files: List[str],
    output_dir: str,
    model_type: str = 'deepsets',
    epochs: int = 50,
    batch_size: int = 128,
    adversarial_config: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run both standard and adversarial training with identical settings.
    
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
    adversarial_config : Dict[str, Any], optional
        Configuration for adversarial training
    random_seed : int
        Random seed for reproducibility
    verbose : bool
        Whether to print verbose output
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results for both training regimes
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for comparative training")
    
    # Set random seeds for reproducibility
    set_random_seeds(random_seed)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    standard_dir = os.path.join(output_dir, 'standard')
    adversarial_dir = os.path.join(output_dir, 'adversarial')
    comparative_dir = os.path.join(output_dir, 'comparative')
    
    for directory in [standard_dir, adversarial_dir, comparative_dir]:
        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(directory, 'models'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'results'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'visualizations'), exist_ok=True)
    
    # Load and prepare dataset
    if verbose:
        print("Loading and preparing dataset...")
    
    # Create combined dataset directly from file paths
    combined_dataset = create_dataset(
        file_list=dataset_files,
        use_scaled=True,
        signal_background_mode=True,
        verbose=verbose
    )
    
    # Prepare for ML
    prepared_data = prepare_ml_dataset(
        combined_dataset,
        test_size=0.2,
        val_size=0.2,
        normalize=True,
        reshape_3d=(model_type.lower() == 'deepsets'),  # Reshape for DeepSets models
        random_state=random_seed,
        verbose=verbose
    )
    
    # Save metadata for reproducibility
    np.save(os.path.join(output_dir, 'dataset_metadata.npy'), prepared_data['metadata'])
    
    # Adapt prepared_data for train_model (which expects 'features' instead of 'X' or 'X_norm')
    adapted_data = {}
    for split in ['train', 'val', 'test']:
        adapted_data[split] = {}
        # Use X_norm if available, otherwise X
        if 'X_norm' in prepared_data[split]:
            adapted_data[split]['features'] = prepared_data[split]['X_norm']
        else:
            adapted_data[split]['features'] = prepared_data[split]['X']
        # Copy other keys
        adapted_data[split]['labels'] = prepared_data[split]['y']
        if 'is_valid' in prepared_data[split]:
            adapted_data[split]['attention_mask'] = prepared_data[split]['is_valid']
    
    # Copy metadata
    adapted_data['metadata'] = prepared_data['metadata']
    
    # Train standard model
    if verbose:
        print("\nTraining standard model...")
    
    standard_start_time = time.time()
    standard_results = train_model(
        prepared_data=adapted_data,
        model_type=model_type,
        hidden_units=[128, 64],
        dropout_rate=0.2,
        epochs=epochs,
        batch_size=batch_size,
        model_name="standard_model",
        output_dir=standard_dir,
        save_model=True,
        verbose=verbose
    )
    standard_training_time = time.time() - standard_start_time
    
    # Train adversarial model if config provided
    adversarial_results = None
    if adversarial_config is not None:
        if verbose:
            print("\nTraining adversarial model...")
        
        adversarial_start_time = time.time()
        adversarial_results = train_model(
            prepared_data=adapted_data,
            model_type=model_type,
            hidden_units=[128, 64],
            dropout_rate=0.2,
            epochs=epochs,
            batch_size=batch_size,
            model_name="adversarial_model",
            output_dir=adversarial_dir,
            save_model=True,
            verbose=verbose,
            adversarial_config=adversarial_config
        )
        adversarial_training_time = time.time() - adversarial_start_time
    else:
        adversarial_training_time = 0
    
    # Evaluate models on test data
    if verbose:
        print("\nEvaluating models on test data...")
    
    standard_model = standard_results['model']
    standard_eval = evaluate_model(
        model=standard_model,
        test_data=adapted_data,
        model_type=model_type,
        verbose=verbose
    )
    
    adversarial_eval = None
    if adversarial_results is not None:
        adversarial_model = adversarial_results['model']
        adversarial_eval = evaluate_model(
            model=adversarial_model,
            test_data=adapted_data,
            model_type=model_type,
            verbose=verbose
        )
    
    # Collect results
    results = {
        'standard': {
            'model': standard_model,
            'training_results': standard_results,
            'evaluation': standard_eval,
            'training_time': standard_training_time
        }
    }
    
    if adversarial_results is not None:
        results['adversarial'] = {
            'model': adversarial_model,
            'training_results': adversarial_results,
            'evaluation': adversarial_eval,
            'training_time': adversarial_training_time,
            'config': adversarial_config
        }
    
    # Save comparative results
    import json
    
    # Create serializable results
    serializable_results = {
        'standard': {
            'evaluation': standard_eval,
            'training_time': standard_training_time,
            'history': standard_results['history']
        }
    }
    
    if adversarial_results is not None:
        serializable_results['adversarial'] = {
            'evaluation': adversarial_eval,
            'training_time': adversarial_training_time,
            'history': adversarial_results['history'],
            'config': adversarial_config
        }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy_to_lists(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_lists(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    serializable_results = convert_numpy_to_lists(serializable_results)
    
    with open(os.path.join(comparative_dir, 'results', 'comparative_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    if verbose:
        print("\nComparative training completed.")
        print(f"Standard training time: {standard_training_time:.2f} seconds")
        if adversarial_results is not None:
            print(f"Adversarial training time: {adversarial_training_time:.2f} seconds")
        print(f"Results saved to {output_dir}")
    
    return results


def generate_perturbed_test_data(
    model: 'tf.keras.Model',
    test_features: np.ndarray,
    test_masks: np.ndarray,
    grad_iter: int = 10,
    grad_eps: float = 1e-6,
    grad_eta: float = 2e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate adversarially perturbed test data.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to generate adversarial examples for
    test_features : np.ndarray
        Test features
    test_masks : np.ndarray
        Test masks
    grad_iter : int
        Number of gradient iterations
    grad_eps : float
        Initial perturbation size
    grad_eta : float
        Step size for gradient updates
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Perturbed test features and original masks
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for generating perturbed data")
    
    # Create adversarial example generator
    generator = AdversarialExampleGenerator(
        grad_iter=grad_iter,
        grad_eps=grad_eps,
        grad_eta=grad_eta
    )
    
    # Convert to TensorFlow tensors
    features_tensor = tf.convert_to_tensor(test_features, dtype=tf.float32)
    masks_tensor = tf.convert_to_tensor(test_masks, dtype=tf.bool)
    
    # Generate adversarial examples
    perturbed_features = generator.generate_adversarial_examples(
        model=model,
        features=features_tensor,
        masks=masks_tensor
    )
    
    # Convert back to numpy
    perturbed_features_np = perturbed_features.numpy()
    
    return perturbed_features_np, test_masks


def evaluate_model_robustness(
    model: 'tf.keras.Model',
    test_data: Dict[str, np.ndarray],
    model_type: str = 'deepsets',
    perturbation_configs: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model on both clean and perturbed test data.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to evaluate
    test_data : Dict[str, np.ndarray]
        Test data dictionary
    model_type : str
        Type of model ('dense' or 'deepsets')
    perturbation_configs : List[Dict[str, Any]], optional
        List of perturbation configurations to test
    verbose : bool
        Whether to print evaluation results
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing evaluation results
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for model evaluation")
    
    # Default perturbation configurations if not provided
    if perturbation_configs is None:
        perturbation_configs = [
            {'name': 'weak', 'grad_iter': 5, 'grad_eta': 1e-4, 'grad_eps': 1e-6},
            {'name': 'medium', 'grad_iter': 10, 'grad_eta': 2e-4, 'grad_eps': 1e-6},
            {'name': 'strong', 'grad_iter': 15, 'grad_eta': 3e-4, 'grad_eps': 1e-6}
        ]
    
    # Prepare test data
    if model_type.lower() == 'deepsets':
        test_features = test_data['test']['features']
        test_masks = test_data['test']['attention_mask']
        test_labels = test_data['test']['labels']
    else:
        test_features = test_data['test']['features']
        test_masks = None
        test_labels = test_data['test']['labels']
    
    # Evaluate on clean data
    if verbose:
        print("\nEvaluating on clean test data...")
    
    clean_results = evaluate_model(
        model=model,
        test_data=test_data,
        model_type=model_type,
        verbose=verbose
    )
    
    # Evaluate on perturbed data for each configuration
    perturbed_results = {}
    
    for config in perturbation_configs:
        config_name = config['name']
        
        if verbose:
            print(f"\nGenerating {config_name} perturbations...")
        
        # Generate perturbed test data
        if model_type.lower() == 'deepsets':
            perturbed_features, _ = generate_perturbed_test_data(
                model=model,
                test_features=test_features,
                test_masks=test_masks,
                grad_iter=config['grad_iter'],
                grad_eps=config['grad_eps'],
                grad_eta=config['grad_eta']
            )
            
            # Create perturbed test data dictionary
            perturbed_test_data = {
                'test': {
                    'features': perturbed_features,
                    'attention_mask': test_masks,
                    'labels': test_labels
                }
            }
        else:
            # For dense models, we need a different approach
            # This is a placeholder - implement as needed
            perturbed_test_data = test_data
        
        if verbose:
            print(f"Evaluating on {config_name} perturbed test data...")
        
        # Evaluate on perturbed data
        perturbed_eval = evaluate_model(
            model=model,
            test_data=perturbed_test_data,
            model_type=model_type,
            verbose=verbose
        )
        
        # Calculate perturbation magnitude
        if model_type.lower() == 'deepsets':
            perturbation_magnitude = np.mean(np.abs(perturbed_features - test_features))
        else:
            perturbation_magnitude = 0.0
        
        # Store results
        perturbed_results[config_name] = {
            'evaluation': perturbed_eval,
            'config': config,
            'perturbation_magnitude': perturbation_magnitude
        }
    
    # Calculate robustness scores
    robustness_scores = {}
    for config_name, result in perturbed_results.items():
        clean_auc = clean_results['metrics']['roc_auc']
        perturbed_auc = result['evaluation']['metrics']['roc_auc']
        
        # Robustness score = perturbed_auc / clean_auc (higher is better)
        robustness_score = perturbed_auc / clean_auc
        
        # Performance drop = clean_auc - perturbed_auc (lower is better)
        performance_drop = clean_auc - perturbed_auc
        
        # Normalized performance drop = performance_drop / perturbation_magnitude
        # This measures how much performance drops per unit of perturbation
        if result['perturbation_magnitude'] > 0:
            normalized_drop = performance_drop / result['perturbation_magnitude']
        else:
            normalized_drop = 0.0
        
        robustness_scores[config_name] = {
            'robustness_score': robustness_score,
            'performance_drop': performance_drop,
            'normalized_drop': normalized_drop
        }
    
    # Collect all results
    results = {
        'clean': clean_results,
        'perturbed': perturbed_results,
        'robustness_scores': robustness_scores
    }
    
    # Print summary if verbose
    if verbose:
        print("\n" + "-"*20 + " ROBUSTNESS EVALUATION SUMMARY " + "-"*20)
        print(f"Clean AUC: {clean_results['metrics']['roc_auc']:.4f}")
        
        for config_name, scores in robustness_scores.items():
            perturbed_auc = perturbed_results[config_name]['evaluation']['metrics']['roc_auc']
            print(f"{config_name.capitalize()} perturbation:")
            print(f"  - Perturbed AUC: {perturbed_auc:.4f}")
            print(f"  - Robustness Score: {scores['robustness_score']:.4f}")
            print(f"  - Performance Drop: {scores['performance_drop']:.4f}")
            print(f"  - Normalized Drop: {scores['normalized_drop']:.4f}")
    
    return results