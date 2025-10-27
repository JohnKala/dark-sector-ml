"""
Training experiment utilities for cross-validation and parameter studies.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler

from ..data.preparation import create_dataset
from ..data.preprocessor import prepare_ml_dataset, prepare_deepsets_data
from ..data.loader import extract_parameters

# Conditional imports for TensorFlow-dependent modules
try:
    from .trainer import train_model
    from .ui import print_training_summary
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def train_individual_models(
    dark_files: List[str],
    sm_file: str,
    use_scaled: bool = True,
    normalize: bool = True,
    model_type: str = 'deepsets',
    epochs: int = 50,
    batch_size: int = 256,
    output_dir: str = ".",
    save_model: bool = True,
    verbose: bool = True,
    # NEW: Adversarial training parameters
    adversarial_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train separate models for each dark sector file with SM as background.
    
    Parameters:
    -----------
    dark_files : list of str
        List of dark sector model files
    sm_file : str
        Path to Standard Model file
    use_scaled : bool
        Whether to use scaled features
    normalize : bool
        Whether to normalize features
    model_type : str
        Type of model to train ('dense' or 'deepsets')
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    output_dir : str
        Directory to save model weights
    verbose : bool
        Whether to print progress information
    adversarial_config : dict, optional
        Configuration for adversarial training. If provided, enables adversarial training.
        Example: {'grad_iter': 3, 'grad_eps': 1e-6, 'grad_eta': 2e-4, 'alpha': 5.0}
        
    Returns:
    --------
    dict
        Dictionary containing results for all models
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for training experiments")
    
    all_results = {}
    
    if verbose:
        print(f"\n{'='*20} INDIVIDUAL MODEL TRAINING {'='*20}")
        print(f"Training {len(dark_files)} individual models (one per dark sector file)")
        print(f"Using SM file: {os.path.basename(sm_file)}")
        print(f"Model type: {model_type}")
        print(f"Features: {'scaled' if use_scaled else 'raw'}, {'normalized' if normalize else 'unnormalized'}")
        if adversarial_config:
            print(f"Adversarial training enabled: {adversarial_config}")
    
    # Adjust batch size for adversarial training (performance optimization from Block 1)
    effective_batch_size = batch_size
    if adversarial_config:
        effective_batch_size = max(batch_size * 2, 512)
        if verbose and effective_batch_size != batch_size:
            print(f"Batch size increased to {effective_batch_size} for adversarial training")
    
    for dark_file in dark_files:
        # Extract parameters for model naming
        params = extract_parameters(dark_file)
        model_name = f"model_{os.path.basename(dark_file).replace('.h5', '')}"
        
        if verbose:
            print(f"\n{'-'*30}")
            print(f"Processing: {os.path.basename(dark_file)}")
        
        # Create combined dataset (this dark file + SM background)
        combined_data = create_dataset(
            [dark_file, sm_file],
            use_scaled=use_scaled,
            signal_background_mode=True,
            verbose=verbose
        )
        
        # Prepare ML dataset
        ml_data = prepare_ml_dataset(
            combined_data,
            normalize=normalize,
            verbose=verbose
        )
        
        # Prepare data for DeepSets if needed
        if model_type.lower() == 'deepsets':
            prepared_data = prepare_deepsets_data(ml_data)
        else:
            prepared_data = ml_data
        
        # Train model (with adversarial support if configured)
        results = train_model(
            prepared_data,
            model_type=model_type,
            model_name=model_name,
            epochs=epochs,
            batch_size=effective_batch_size,  # Use optimized batch size
            output_dir=output_dir,
            save_model=save_model,
            verbose=verbose,
            # NEW: Pass adversarial configuration to existing trainer
            adversarial_config=adversarial_config,
            mixed_precision=bool(adversarial_config)  # Auto-enable mixed precision for adversarial
        )
        
        # Store parameters with results
        results['parameters'] = params
        
        # Preserve the prepared data for cross-evaluation (fixes empty plots!)
        results['prepared_data'] = prepared_data
        
        # Add to collection
        all_results[model_name] = results
    
    # Print summary
    if verbose:
        print_training_summary(all_results, "INDIVIDUAL MODELS SUMMARY")
    
    return all_results


def train_leave_one_out_models(
    dark_files: List[str],
    sm_file: str,
    use_scaled: bool = True,
    normalize: bool = True,
    model_type: str = 'deepsets',
    epochs: int = 50,
    batch_size: int = 256,
    output_dir: str = ".",
    save_model: bool = True,
    verbose: bool = True,
    # NEW: Adversarial training parameters
    adversarial_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train models using a leave-one-out approach.
    
    Parameters:
    -----------
    dark_files : list of str
        List of dark sector model files
    sm_file : str
        Path to Standard Model file
    use_scaled : bool
        Whether to use scaled features
    normalize : bool
        Whether to normalize features
    model_type : str
        Type of model to train ('dense' or 'deepsets')
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    output_dir : str
        Directory to save model weights
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    dict
        Dictionary containing results for all leave-one-out models
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for training experiments")
    
    all_results = {}
    
    if verbose:
        print(f"\n{'='*20} LEAVE-ONE-OUT MODEL TRAINING {'='*20}")
        print(f"Training {len(dark_files)} leave-one-out models")
        print(f"Using SM file: {os.path.basename(sm_file)}")
        print(f"Model type: {model_type}")
        print(f"Features: {'scaled' if use_scaled else 'raw'}, {'normalized' if normalize else 'unnormalized'}")
        if adversarial_config:
            print(f"Adversarial training enabled: {adversarial_config}")
    
    # Adjust batch size for adversarial training (performance optimization from Block 1)
    effective_batch_size = batch_size
    if adversarial_config:
        effective_batch_size = max(batch_size * 2, 512)
        if verbose and effective_batch_size != batch_size:
            print(f"Batch size increased to {effective_batch_size} for adversarial training")
    
    # For each file held out
    for i, left_out_file in enumerate(dark_files):
        # Get training files (all except the held-out one)
        training_files = [f for f in dark_files if f != left_out_file] + [sm_file]
        
        model_name = f"loo_{os.path.basename(left_out_file).replace('.h5', '')}"
        
        if verbose:
            print(f"\n{'-'*30}")
            print(f"Leave-out model #{i+1}: Leaving out {os.path.basename(left_out_file)}")
            print(f"Training on {len(training_files)} files")
        
        # Create combined dataset from training files
        combined_data = create_dataset(
            training_files,
            use_scaled=use_scaled,
            signal_background_mode=True,
            verbose=verbose
        )
        
        # Prepare ML dataset with robust normalization
        ml_data = prepare_ml_dataset(
            combined_data,
            normalize=normalize,
            verbose=verbose
        )
        
        # Get normalization parameters for later use with left-out dataset
        norm_params = ml_data.get('norm_params', None)
        
        # Prepare data for DeepSets if needed
        if model_type.lower() == 'deepsets':
            prepared_data = prepare_deepsets_data(ml_data)
        else:
            prepared_data = ml_data
        
        # Train model (with adversarial support if configured)
        train_results = train_model(
            prepared_data,
            model_type=model_type,
            model_name=model_name,
            epochs=epochs,
            batch_size=effective_batch_size,  # Use optimized batch size
            output_dir=output_dir,
            save_model=save_model,
            verbose=verbose,
            # NEW: Pass adversarial configuration to existing trainer
            adversarial_config=adversarial_config,
            mixed_precision=bool(adversarial_config)  # Auto-enable mixed precision for adversarial
        )
        
        # Load left-out dataset for evaluation
        if verbose:
            print(f"Loading left-out dataset for evaluation...")
        
        left_out_data = create_dataset(
            [left_out_file, sm_file],
            use_scaled=use_scaled,
            signal_background_mode=True,
            verbose=verbose
        )
        
        # Prepare left-out dataset WITHOUT normalization first
        left_out_ml_data = prepare_ml_dataset(
            left_out_data,
            normalize=False,  # Don't normalize yet - we'll use training params
            verbose=verbose
        )
        
        # Apply same normalization from training set if needed
        if normalize and norm_params is not None:
            # Apply normalization to left-out data using training parameters
            for split in ['train', 'val', 'test']:
                scaler = StandardScaler()
                scaler.mean_ = np.array(norm_params['mean'])
                scaler.scale_ = np.array(norm_params['scale'])
                scaler.n_features_in_ = len(norm_params['mean'])
                
                # Transform using training set parameters
                left_out_ml_data[split]['X_norm'] = scaler.transform(left_out_ml_data[split]['X'])
            
            # Add norm_params to left_out_data for consistency
            left_out_ml_data['norm_params'] = norm_params
        
        # Prepare left-out data for DeepSets if needed
        if model_type.lower() == 'deepsets':
            left_out_prepared = prepare_deepsets_data(left_out_ml_data)
        else:
            left_out_prepared = left_out_ml_data
        
        # Store results
        all_results[model_name] = {
            'model': train_results['model'],
            'model_name': model_name,
            'history': train_results['history'],
            'training_time': train_results['training_time'],
            'training_files': training_files,
            'left_out_file': left_out_file,
            'left_out_data': left_out_prepared,
            'params': train_results['params'],
            # Preserve the prepared training data for cross-evaluation (consistency with individual models)
            'prepared_data': prepared_data
        }
    
    # Print summary
    if verbose:
        print_training_summary(all_results, "LEAVE-ONE-OUT MODELS SUMMARY")
    
    return all_results
