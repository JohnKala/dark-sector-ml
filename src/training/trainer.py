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
    verbose: bool = True,
    # NEW ADVERSARIAL PARAMETERS
    adversarial_config: Optional[Dict[str, Any]] = None,
    mixed_precision: bool = False
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
    adversarial_config : dict, optional
        Configuration for adversarial training:
        {
            'grad_iter': 3,      # Number of adversarial gradient iterations
            'grad_eps': 1e-6,    # Initial noise standard deviation  
            'grad_eta': 2e-4,    # Adversarial gradient step size
            'alpha': 5.0         # Weight for adversarial KL loss
        }
    mixed_precision : bool
        Whether to use mixed precision training for performance
        
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
    
    # Create model with adversarial support if requested
    if adversarial_config:
        # Import adversarial components
        from ..models.adversarial import create_adversarial_model
        from ..data.preprocessor import create_optimized_datasets_from_prepared_data
        
        if verbose:
            print(f"Creating adversarial model with config: {adversarial_config}")
        
        # Adjust batch size for adversarial training (performance optimization from Block 1)
        adversarial_batch_size = max(batch_size * 2, 512)
        if verbose and adversarial_batch_size != batch_size:
            print(f"Batch size increased to {adversarial_batch_size} for adversarial training")
        
        # Create adversarial model using our new architecture
        model = create_adversarial_model(
            model_type=model_type,
            input_shape=input_shape,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            adversarial_config=adversarial_config,
            use_mixed_precision=mixed_precision
        )
        
        # Use custom adversarial training loop
        return _train_adversarial_model(
            model, prepared_data, epochs, adversarial_batch_size,
            patience, model_name, output_dir, save_model, verbose
        )
    
    else:
        # Use existing model creation for standard training (unchanged)
        if mixed_precision:
            from ..models.factory import create_model_with_mixed_precision
            model = create_model_with_mixed_precision(
                model_type=model_type,
                input_shape=input_shape,
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                use_mixed_precision=mixed_precision
            )
        else:
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


def _train_adversarial_model(
    adversarial_model: 'AdversarialModelWrapper',
    prepared_data: Dict[str, Any],
    epochs: int,
    batch_size: int,
    patience: int,
    model_name: str,
    output_dir: str,
    save_model: bool,
    verbose: bool
) -> Dict[str, Any]:
    """
    Custom training loop for adversarial models.
    
    This function implements the adversarial training loop from Block 1 
    while reusing existing infrastructure (callbacks, checkpointing, etc.).
    """
    import os
    import numpy as np
    from ..data.preprocessor import create_optimized_datasets_from_prepared_data
    from tensorflow.keras import mixed_precision
    
    start_time = time.time()
    
    if verbose:
        print(f"Starting adversarial training for {model_name}")
        print(f"Adversarial config: {adversarial_model.adversarial_config}")
    
    # Create optimized datasets (from Block 1 integration)
    try:
        datasets = create_optimized_datasets_from_prepared_data(
            prepared_data, batch_size=batch_size, shuffle_train=True
        )
        train_dataset = datasets['train']
        val_dataset = datasets['val'] 
    except ImportError:
        # Fallback if TensorFlow dataset optimization not available
        if verbose:
            print("Warning: Using fallback dataset creation (TensorFlow optimization unavailable)")
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            prepared_data['train']['features'],
            prepared_data['train']['attention_mask'],
            prepared_data['train']['labels']
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            prepared_data['val']['features'], 
            prepared_data['val']['attention_mask'],
            prepared_data['val']['labels']
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Initialize optimizer (from Block 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    # Training history
    history = {
        'loss': [], 
        'ce_loss': [], 
        'kl_loss': [], 
        'val_loss': []
    }
    
    # Create progress bar callback (reuse existing infrastructure)
    if verbose:
        progress_callback = ProgressBarCallback(epochs=epochs, desc=f"Adversarial Training {model_name}")
        progress_callback.on_train_begin()
    
    # Create model checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'model_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Main adversarial training loop (from Block 1, integrated with existing patterns)
    for epoch in range(epochs):
        # Training phase
        epoch_losses = {'loss': [], 'ce_loss': [], 'kl_loss': []}
        
        for features, masks, labels in train_dataset:
            # Convert to float16 if using mixed precision (from Block 1)
            if mixed_precision.global_policy().name == 'mixed_float16':
                features = tf.cast(features, tf.float16)
            
            # Perform adversarial training step
            metrics = adversarial_model.adversarial_train_step(features, masks, labels, optimizer)
            
            # Store metrics
            for key in epoch_losses:
                epoch_losses[key].append(metrics[key].numpy())
        
        # Compute epoch averages
        epoch_metrics = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # Validation phase
        val_losses = []
        for val_features, val_masks, val_labels in val_dataset:
            val_metrics = adversarial_model.validation_step(val_features, val_masks, val_labels)
            val_losses.append(val_metrics['val_loss'].numpy())
        
        val_loss = np.mean(val_losses)
        epoch_metrics['val_loss'] = val_loss
        
        # Store in history
        for key, value in epoch_metrics.items():
            history[key].append(value)
        
        # Update progress bar (reuse existing infrastructure)
        if verbose:
            progress_callback.on_epoch_end(epoch, epoch_metrics)
        
        # Early stopping and checkpointing (from Block 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            if save_model:
                checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.weights.h5")
                adversarial_model.save_weights(checkpoint_path)
        else:
            no_improvement_count += 1
        
        # Early stopping check
        if no_improvement_count >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            # Restore best weights
            if save_model:
                checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.weights.h5")
                adversarial_model.load_weights(checkpoint_path)
            break
    
    # Finish progress bar
    if verbose:
        progress_callback.on_train_end()
    
    # Reset mixed precision policy (from Block 1)
    mixed_precision.set_global_policy('float32')
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Return results in existing format (compatible with existing pipeline)
    results = {
        'model': adversarial_model,
        'model_name': model_name,
        'history': history,
        'training_time': training_time,
        'best_val_loss': best_val_loss,
        'params': {
            'model_type': 'deepsets',
            'adversarial_config': adversarial_model.adversarial_config,
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'training_type': 'adversarial'
        }
    }
    
    if verbose:
        print(f"\nAdversarial training completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    return results
