"""
Robustness evaluation module for dark sector ML models.

This module provides a standardized framework for evaluating model robustness
against adversarial attacks. It ensures that all models are tested against
the exact same attack protocol for fair comparison.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Union, Tuple, List

from ..models.adversarial import AdversarialExampleGenerator


class RobustnessEvaluator:
    """
    Standardized evaluator for measuring model robustness.
    
    This class implements a consistent "exam" that all models must pass.
    It measures performance on both clean data and data perturbed by a 
    standardized PGD attack.
    """
    
    def __init__(
        self,
        attack_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 256
    ):
        """
        Initialize the robustness evaluator.
        
        Parameters:
        -----------
        attack_config : dict, optional
            Configuration for the standardized attack.
            Defaults to a strong PGD attack:
            - grad_iter: 10
            - grad_eps: 1e-6
            - grad_eta: 2e-7 (eps / 5)
        batch_size : int
            Batch size for evaluation
        """
        # Default "Standardized Exam" configuration
        self.default_config = {
            'grad_iter': 10,
            'grad_eps': 1e-6,
            'grad_eta': 2e-7
        }
        
        self.config = attack_config if attack_config is not None else self.default_config
        self.batch_size = batch_size
        
        # Initialize the generator
        self.generator = AdversarialExampleGenerator(
            grad_iter=self.config['grad_iter'],
            grad_eps=self.config['grad_eps'],
            grad_eta=self.config['grad_eta']
        )
        
        # Metrics
        self.auc_metric = tf.keras.metrics.AUC()
    
    def evaluate(
        self,
        model: tf.keras.Model,
        dataset: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Run the standardized robustness exam on a model.
        
        Parameters:
        -----------
        model : tf.keras.Model
            The model to evaluate
        dataset : dict
            Prepared dataset containing 'test' split with 'features', 'labels', 
            and 'attention_mask'
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics:
            - clean_auc: Performance on original data
            - robust_auc: Performance on perturbed data
            - robustness_score: Ratio (robust_auc / clean_auc)
            - perturbation_budget: The epsilon used
        """
        # Extract test data
        features = dataset['test']['features']
        labels = dataset['test']['labels']
        
        # Handle attention masks
        if 'attention_mask' in dataset['test']:
            masks = dataset['test']['attention_mask']
        else:
            # Create dummy masks if not present (e.g. for Dense models)
            # Assuming features are [N, Particles, Features] or [N, Flat]
            if len(features.shape) == 3:
                masks = np.ones((features.shape[0], features.shape[1]), dtype=bool)
            else:
                # For flat features, we can't easily infer particle count, 
                # but the model likely doesn't use the mask anyway if it's Dense
                masks = np.ones((features.shape[0], 1), dtype=bool)
        
        # Ensure data types
        features = tf.cast(features, tf.float32)
        masks = tf.cast(masks, tf.bool)
        labels = tf.cast(labels, tf.float32)
        
        # Create tf.data.Dataset for batching
        test_ds = tf.data.Dataset.from_tensor_slices((features, masks, labels))
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # 1. Evaluate Clean Performance
        print("Evaluating clean performance...")
        self.auc_metric.reset_state()
        
        for batch_features, batch_masks, batch_labels in test_ds:
            # Model expects [features, masks] list
            preds = model([batch_features, batch_masks], training=False)
            self.auc_metric.update_state(batch_labels, preds)
            
        clean_auc = float(self.auc_metric.result().numpy())
        
        # 2. Evaluate Robust Performance (The Attack)
        print(f"Evaluating robust performance (PGD: eps={self.config['grad_eps']}, iter={self.config['grad_iter']})...")
        self.auc_metric.reset_state()
        
        for batch_features, batch_masks, batch_labels in test_ds:
            # Generate adversarial examples on the fly
            # Note: We attack the model using its own gradients
            adv_features = self.generator.generate_adversarial_examples(
                model, batch_features, batch_masks
            )
            
            # Predict on adversarial examples
            preds_adv = model([adv_features, batch_masks], training=False)
            self.auc_metric.update_state(batch_labels, preds_adv)
            
        robust_auc = float(self.auc_metric.result().numpy())
        
        # 3. Calculate Robustness Score
        # Avoid division by zero
        robustness_score = robust_auc / clean_auc if clean_auc > 0 else 0.0
        
        results = {
            'clean_auc': clean_auc,
            'robust_auc': robust_auc,
            'robustness_score': robustness_score,
            'perturbation_budget': self.config['grad_eps'],
            'attack_steps': self.config['grad_iter']
        }
        
        print(f"Results: Clean AUC={clean_auc:.4f}, Robust AUC={robust_auc:.4f}, Score={robustness_score:.4f}")
        
        return results
