"""
Model training utilities for dark sector ML models.
"""

import time
from typing import Dict, Any, List, Optional

# Conditional TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..models.factory import create_model
from .ui import ProgressBarCallback


def train_model(
    prepared_data: Dict[str, Any],
    model_type: str = 'deepsets',
    hidden_units: List[int] = [128, 64],
    dropout_rate: float = 0.2,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 10,  # Original value
    lr_factor: float = 0.5,  # Original LR reduction factor
    lr_patience: int = 5,  # Original LR patience
    class_weight: Optional[Dict[int, float]] = {0: 1, 1: 1},  # Original default
    model_name: str = None,
    output_dir: str = ".",
    save_model: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a model on prepared data with organized, clean output.
    
    Parameters:
    -----------
    prepared_data : dict
        Data prepared by prepare_deepsets_data or prepare_ml_dataset
    model_type : str
        Type of model to create ('dense' or 'deepsets')
    hidden_units : list of int
        Sizes of hidden layers
    dropout_rate : float
        Dropout rate for regularization
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Training batch size
    patience : int
        Early stopping patience
    lr_factor : float
        Factor by which to reduce learning rate on plateau
    lr_patience : int
        Number of epochs with no improvement after which learning rate will be reduced
    class_weight : dict or None
        Class weights for training (default: {0: 1, 1: 1} for balanced weighting)
        Set to None for no class weighting
    model_name : str
        Name for model checkpoints (if None, a timestamp is used)
    output_dir : str
        Directory to save model weights (default: current directory)
    save_model : bool
        Whether to save model weights
    verbose : bool
        Whether to print training progress
        
    Returns:
    --------
    dict
        Training results and model, including all parameters for reproducibility
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for model training")

    start_time = time.time()
    
    # Generate model name if not provided
    if model_name is None:
        model_name = f"model_{model_type}_{int(time.time())}"
    
    if verbose:
        print(f"\nInitializing model: {model_name}")
    
    # Determine input shape based on model type
    if model_type.lower() == 'dense':
        # For dense model, we need flat features
        input_shape = (prepared_data['train']['features'].shape[1],)
    else:
        # For DeepSets, we need 3D shape
        input_shape = prepared_data['train']['features'].shape[1:]
    
    # Create model
    model = create_model(
        model_type=model_type,
        input_shape=input_shape,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate
    )
    
    if verbose:
        print(f"Model structure:")
        model.summary(line_length=80)
    
    # Create callbacks
    callbacks = [
        # Early stopping: monitor val_auc (original behavior)
        EarlyStopping(
            monitor='val_auc',
            patience=patience,
            restore_best_weights=True,
            mode='max',  # AUC should be maximized
            verbose=0
        ),
        # LR reduction: restore from original (issue #7)
        ReduceLROnPlateau(
            monitor='val_auc',
            factor=lr_factor,
            patience=lr_patience,
            min_lr=1e-6,
            mode='max',  # AUC should be maximized
            verbose=0
        )
    ]
    
    # Add model checkpoint if saving
    if save_model:
        callbacks.append(
            ModelCheckpoint(
                f"{output_dir}/{model_name}.weights.h5",
                monitor='val_auc',  # Use val_auc to match original
                mode='max',         # AUC should be maximized
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
        )
    
    # Add progress bar if verbose
    if verbose:
        callbacks.append(
            ProgressBarCallback(epochs=epochs, desc=f"Training {model_name}")
        )
    
    # Prepare training data based on model type
    if model_type.lower() == 'deepsets':
        train_inputs = [
            prepared_data['train']['features'],
            prepared_data['train']['attention_mask']
        ]
        val_inputs = [
            prepared_data['val']['features'],
            prepared_data['val']['attention_mask']
        ]
    else:
        train_inputs = prepared_data['train']['features']
        val_inputs = prepared_data['val']['features']
    
    # Train model
    history = model.fit(
        train_inputs,
        prepared_data['train']['labels'],
        validation_data=(
            val_inputs,
            prepared_data['val']['labels']
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,  # Restored class weighting (issue #9)
        verbose=0  # We use our own progress bar
    )
    
    training_time = time.time() - start_time
    
    # Save results - include all parameters for reproducibility
    results = {
        'model': model,
        'model_name': model_name,
        'history': history.history,
        'training_time': training_time,
        'params': {
            'model_type': model_type,
            'hidden_units': hidden_units,
            'dropout_rate': dropout_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'lr_factor': lr_factor,
            'lr_patience': lr_patience,
            'class_weight': class_weight
        }
    }
    
    return results
