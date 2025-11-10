"""
Adversarial model components and utilities for the dark sector ML pipeline.

This module contains the core adversarial training functionality extracted and
integrated from the adversarial training implementation, designed to work with
the existing dark sector ML architecture.
"""

import tensorflow as tf
from typing import Dict, Any, Tuple, Optional
import numpy as np
from tensorflow.keras import mixed_precision

from .factory import create_model_with_mixed_precision


class AdversarialLoss:
    """Computes adversarial losses for model training."""
    
    def __init__(self, alpha: float = 5.0):
        """
        Initialize adversarial loss computation.
        
        Parameters:
        -----------
        alpha : float
            Weight for the KL divergence loss component
        """
        self.alpha = alpha
        self.kl_div = tf.keras.losses.KLDivergence()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
    
    @tf.function(jit_compile=True)
    def compute_loss(
        self,
        pred_original: tf.Tensor,
        pred_adversarial: tf.Tensor,
        labels: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Compute combined adversarial loss.
        
        Parameters:
        -----------
        pred_original : tf.Tensor
            Predictions on original inputs
        pred_adversarial : tf.Tensor
            Predictions on adversarial inputs
        labels : tf.Tensor
            True labels
            
        Returns:
        --------
        dict
            Dictionary containing loss components
        """
        # Cast to float32 for stable computation
        pred_orig_fp32 = tf.cast(pred_original, tf.float32)
        pred_adv_fp32 = tf.cast(pred_adversarial, tf.float32)
        labels_fp32 = tf.cast(labels, tf.float32)
        
        # Cross-entropy loss on original predictions
        ce_loss = self.cross_entropy(labels_fp32, pred_orig_fp32)
        
        # KL divergence between original and adversarial predictions
        # Memory-optimized implementation
        pred_orig_0 = 1-pred_orig_fp32
        pred_orig_1 = pred_orig_fp32
        pred_adv_0 = 1-pred_adv_fp32
        pred_adv_1 = pred_adv_fp32
        
        # Manual KL calculation to avoid large tensor concatenations
        epsilon = 1e-7  # Small constant for numerical stability
        kl_0 = pred_orig_0 * tf.math.log(pred_orig_0 / (pred_adv_0 + epsilon) + epsilon)
        kl_1 = pred_orig_1 * tf.math.log(pred_orig_1 / (pred_adv_1 + epsilon) + epsilon)
        kl_loss = tf.reduce_mean(kl_0 + kl_1)
        
        # Combined loss
        total_loss = ce_loss + self.alpha * kl_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss
        }


class AdversarialExampleGenerator:
    """Generates adversarial examples for any compatible model."""
    
    def __init__(
        self,
        grad_iter: int = 3,
        grad_eps: float = 1e-6,
        grad_eta: float = 2e-4
    ):
        """
        Initialize adversarial example generator.
        
        Parameters:
        -----------
        grad_iter : int
            Number of gradient iterations for adversarial refinement
        grad_eps : float
            Standard deviation for initial noise
        grad_eta : float
            Step size for gradient updates
        """
        self.grad_iter = grad_iter
        self.grad_eps = grad_eps
        self.grad_eta = grad_eta
        self.kl_div = tf.keras.losses.KLDivergence()
    
    @tf.function  # Removed jit_compile=True to reduce memory usage
    def generate_adversarial_examples(
        self,
        model: tf.keras.Model,
        features: tf.Tensor,
        masks: tf.Tensor
    ) -> tf.Tensor:
        """
        Generate adversarial examples using gradient-based perturbation.
        
        This is the core adversarial example generation from the original code,
        optimized and integrated with the existing architecture.
        
        Parameters:
        -----------
        model : tf.keras.Model
            Model to generate adversarial examples for
        features : tf.Tensor
            Original input features
        masks : tf.Tensor
            Attention masks for valid particles
            
        Returns:
        --------
        tf.Tensor
            Adversarial examples
        """
        # Initialize with small noise (from Block 1)
        noise = tf.random.normal(
            shape=tf.shape(features),
            mean=0.0,
            stddev=self.grad_eps,
            dtype=features.dtype
        )
        features_adv = features + noise
        
        # Get original predictions (cached for efficiency - optimization from our analysis)
        output_original = model([features, masks], training=False)
        
        # Iterative adversarial refinement (from Block 1)
        for _ in range(self.grad_iter):
            with tf.GradientTape() as tape:
                tape.watch(features_adv)
                output_adv = model([features_adv, masks], training=False)
                
                # Convert to float32 for KL computation (from Block 1)
                output_orig_fp32 = tf.cast(output_original, tf.float32)
                output_adv_fp32 = tf.cast(output_adv, tf.float32)
                
                # Calculate KL divergence with memory optimization
                # Avoid large concatenations that consume memory
                pred_orig_0 = 1-output_orig_fp32
                pred_orig_1 = output_orig_fp32
                pred_adv_0 = 1-output_adv_fp32
                pred_adv_1 = output_adv_fp32
                
                # Manual KL calculation to avoid large tensor concatenations
                epsilon = 1e-7  # Small constant for numerical stability
                kl_0 = pred_orig_0 * tf.math.log(pred_orig_0 / (pred_adv_0 + epsilon) + epsilon)
                kl_1 = pred_orig_1 * tf.math.log(pred_orig_1 / (pred_adv_1 + epsilon) + epsilon)
                kl_loss = tf.reduce_mean(kl_0 + kl_1)
            
            # Compute gradients and update adversarial examples (from Block 1)
            gradients = tape.gradient(kl_loss, features_adv)
            mask_expanded = tf.cast(tf.expand_dims(masks, axis=-1), features.dtype)
            features_adv = features_adv + self.grad_eta * tf.sign(gradients) * mask_expanded
        
        return features_adv


class AdversarialModelWrapper:
    """Wraps existing models with adversarial training capabilities."""
    
    def __init__(
        self, 
        base_model: tf.keras.Model,
        adversarial_config: Dict[str, Any]
    ):
        """
        Initialize adversarial model wrapper.
        
        Parameters:
        -----------
        base_model : tf.keras.Model
            Base model to wrap with adversarial capabilities
        adversarial_config : dict
            Configuration for adversarial training
        """
        self.base_model = base_model
        self.generator = AdversarialExampleGenerator(
            grad_iter=adversarial_config.get('grad_iter', 3),
            grad_eps=adversarial_config.get('grad_eps', 1e-6),
            grad_eta=adversarial_config.get('grad_eta', 2e-4)
        )
        self.loss_fn = AdversarialLoss(alpha=adversarial_config.get('alpha', 5.0))
        self.adversarial_config = adversarial_config
    
    def __call__(self, inputs, training=False):
        """Forward pass through base model."""
        return self.base_model(inputs, training=training)
    
    def predict(self, inputs, verbose=0):
        """Prediction method for compatibility with Keras API."""
        # Handle both single input and [features, masks] format
        if isinstance(inputs, list):
            return self.base_model.predict(inputs, verbose=verbose)
        else:
            # For compatibility with the evaluation code that might pass just features
            # We need to handle this case by assuming masks are all ones
            # This is a fallback for the cross-evaluation code
            batch_size = inputs.shape[0]
            masks = tf.ones((batch_size, inputs.shape[1] // 3), dtype=tf.float32)
            return self.base_model.predict([inputs, masks], verbose=verbose)
    
    @property
    def trainable_variables(self):
        """Access trainable variables of the base model."""
        return self.base_model.trainable_variables
    
    def save_weights(self, filepath):
        """Save weights of the base model."""
        return self.base_model.save_weights(filepath)
    
    def load_weights(self, filepath):
        """Load weights into the base model."""
        return self.base_model.load_weights(filepath)
    
    def summary(self, **kwargs):
        """Print model summary."""
        return self.base_model.summary(**kwargs)
        
    # Additional compatibility methods for evaluation
    
    @property
    def input(self):
        """Access input of the base model."""
        return self.base_model.input
        
    @property
    def output(self):
        """Access output of the base model."""
        return self.base_model.output
        
    @property
    def input_shape(self):
        """Access input shape of the base model."""
        return self.base_model.input_shape
    
    @tf.function  # Removed jit_compile=True to reduce memory usage
    def adversarial_train_step(
        self, 
        features: tf.Tensor, 
        masks: tf.Tensor, 
        labels: tf.Tensor, 
        optimizer: tf.keras.optimizers.Optimizer
    ) -> Dict[str, tf.Tensor]:
        """
        Perform one adversarial training step.
        
        This integrates the training step logic from Block 1 but uses
        the existing model and optimization infrastructure.
        
        Parameters:
        -----------
        features : tf.Tensor
            Input features
        masks : tf.Tensor
            Attention masks
        labels : tf.Tensor
            Target labels
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer for gradient updates
            
        Returns:
        --------
        dict
            Dictionary containing loss metrics
        """
        # Generate adversarial examples
        features_adv = self.generator.generate_adversarial_examples(
            self.base_model, features, masks
        )
        
        with tf.GradientTape() as tape:
            # Forward pass on both original and adversarial (from Block 1)
            pred_original = self.base_model([features, masks], training=True)
            pred_adversarial = self.base_model([features_adv, masks], training=True)
            
            # Compute adversarial loss
            loss_dict = self.loss_fn.compute_loss(
                pred_original, pred_adversarial, labels
            )
        
        # Apply gradients (from Block 1)
        gradients = tape.gradient(loss_dict['total_loss'], self.base_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
        
        return loss_dict
    
    @tf.function  # Removed jit_compile=True to reduce memory usage
    def validation_step(
        self, 
        features: tf.Tensor, 
        masks: tf.Tensor, 
        labels: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Perform validation step (from Block 1).
        
        Parameters:
        -----------
        features : tf.Tensor
            Input features
        masks : tf.Tensor
            Attention masks  
        labels : tf.Tensor
            Target labels
            
        Returns:
        --------
        dict
            Dictionary containing validation metrics
        """
        # Forward pass in inference mode
        pred = self.base_model([features, masks], training=False)
        
        # Cast to float32 for loss computation
        pred = tf.cast(pred, tf.float32)
        labels_reshaped = tf.cast(labels, dtype=tf.float32)
        
        # Compute validation loss
        ce_loss = tf.keras.losses.BinaryCrossentropy()(labels_reshaped, pred)
        return {'val_loss': ce_loss}


def create_adversarial_model(
    model_type: str,
    input_shape: tuple,
    hidden_units: list,
    dropout_rate: float,
    adversarial_config: Dict[str, Any],
    use_mixed_precision: bool = False
) -> AdversarialModelWrapper:
    """
    Create a model with adversarial training capabilities.
    
    This extends the existing model factory pattern to create adversarial models
    while reusing all existing infrastructure.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('dense' or 'deepsets')
    input_shape : tuple
        Input shape for the model
    hidden_units : list
        Hidden layer sizes
    dropout_rate : float
        Dropout rate
    adversarial_config : dict
        Adversarial training configuration
    use_mixed_precision : bool
        Whether to use mixed precision training
        
    Returns:
    --------
    AdversarialModelWrapper
        Model with adversarial training capabilities
        
    Raises:
    -------
    ValueError
        If adversarial training is requested for unsupported architecture
    """
    
    # Validate model type (from Block 1 - DeepSets only constraint)
    if model_type.lower() != 'deepsets':
        raise ValueError("Adversarial training currently only supports DeepSets architecture")
    
    # Create base model using existing infrastructure
    base_model = create_model_with_mixed_precision(
        model_type, input_shape, hidden_units, dropout_rate, use_mixed_precision
    )
    
    # Wrap with adversarial capabilities
    return AdversarialModelWrapper(base_model, adversarial_config)