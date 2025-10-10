"""
Model factory for creating different types of neural network architectures.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from ..config import CONFIG
from typing import List, Tuple, Union


def create_model(
    model_type: str,
    input_shape: Union[Tuple[int], Tuple[int, int, int]],
    hidden_units: List[int] = [128, 64],
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Factory function to create different types of models.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create ('dense' or 'deepsets')
    input_shape : tuple
        Shape of input data (flat for 'dense', 3D for 'deepsets')
    hidden_units : list of int
        Sizes of hidden layers (used in dense model; DeepSets uses fixed architecture)
    dropout_rate : float
        Dropout rate for regularization
        
    Returns:
    --------
    tf.keras.Model
        Compiled model ready for training
    """
    if model_type.lower() == 'dense':
        # Standard dense neural network
        model = Sequential([
            Dense(hidden_units[0], activation='relu', input_shape=input_shape),
            Dropout(dropout_rate),
            Dense(hidden_units[1], activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
    elif model_type.lower() == 'deepsets':
        # DeepSets model for permutation invariance
        from tensorflow.keras import layers, models, backend as K
        
        # Input layer for particle features and attention mask
        particle_inputs = layers.Input(shape=input_shape, name="particle_features")
        mask_inputs = layers.Input(shape=(input_shape[0],), dtype="bool", name="attention_mask") # MIGHT HAVE TO TAKE OUT
        
        # Expand mask dimensions for broadcasting using Lambda layer
        mask_expanded = layers.Lambda(
            lambda x: K.expand_dims(x, axis=-1),
            output_shape=lambda s: (s[0], s[1], 1)  # Add explicit output shape
        )(mask_inputs)
        
        # Phi network (particle-level feature transformation)
        # First apply mask to zero out invalid particles
        masked_inputs = layers.Lambda(
            lambda inputs: inputs[0] * K.cast(inputs[1], K.dtype(inputs[0])),
            output_shape=lambda s: s[0]  # Output shape same as first input
        )([particle_inputs, mask_expanded]) # MIGHT HAVE TO TAKE OUT
        
        # Apply particle-level feature transformation (Phi network)
        x = layers.Conv1D(100, kernel_size=1, activation='relu')(masked_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(100, kernel_size=1, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=1, activation='relu')(x)
        x = layers.BatchNormalization()(x) # MIGHT HAVE TO TAKE OUT
        
        # Apply masking again before pooling to ensure invalid particles don't contribute
        x = layers.Lambda(
            lambda inputs: inputs[0] * K.cast(inputs[1], K.dtype(inputs[0])),
            output_shape=lambda s: s[0]  # Output shape same as first input
        )([x, mask_expanded])
        
        # CHANGED: Custom pooling for true masked average
        # Sum features across particles
        sum_features = layers.Lambda(
            lambda x: K.sum(x, axis=1),
            output_shape=lambda s: (s[0], s[2])  # (batch_size, features)
        )(x)
        
        # Count valid particles (with casting to match feature dtype)
        count_valid = layers.Lambda(
            lambda x: K.sum(K.cast(x, 'float32'), axis=1, keepdims=True),
            output_shape=lambda s: (s[0], 1)  # (batch_size, 1)
        )(mask_inputs)
        
        # Average = sum / count (with epsilon to avoid division by zero)
        x = layers.Lambda(
            lambda inputs: inputs[0] / (inputs[1] + K.epsilon()),
            output_shape=lambda s: s[0]  # Output shape same as first input
        )([sum_features, count_valid])

        # MIGHT HAVE TO REPLACE WITH GLOBALAVGPOOLING
        
        # Rho network (jet-level classification)
        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = models.Model(inputs=[particle_inputs, mask_inputs], outputs=output, name="DeepSets")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'dense', 'deepsets'")
    
    # Added metrics for proper monitoring
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'), # MIGHT HAVE TO REMOVE
            tf.keras.metrics.Precision(name='precision'), # MIGHT HAVE TO REMOVE
            tf.keras.metrics.Recall(name='recall') # MIGHT HAVE TO REMOVE

        ]
    )
    
    return model