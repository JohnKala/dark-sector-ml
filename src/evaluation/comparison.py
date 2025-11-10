"""
Cross-model performance comparison utilities.
"""

from typing import Dict, Any
import numpy as np

# Conditional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def evaluate_cross_model_performance(
    all_models: Dict[str, Any],
    all_test_datasets: Dict[str, Any],
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate each model on each test dataset to create a cross-performance matrix.
    
    Parameters:
    -----------
    all_models : dict
        Dictionary of trained models and their data
    all_test_datasets : dict
        Dictionary of test datasets
    threshold : float
        Classification threshold
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    dict
        Nested dictionary of evaluation metrics for each model on each dataset
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for cross-model evaluation")
    
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    
    # Initialize results structure
    cross_results = {}
    
    if verbose:
        print(f"\n{'-'*20} CROSS-MODEL EVALUATION {'-'*20}")
    
    # For each model
    for model_name, model_data in all_models.items():
        model = model_data['model']
        model_type = model_data['params']['model_type']
        cross_results[model_name] = {}
        
        if verbose:
            print(f"Evaluating {model_name} across all datasets...")
        
        # For each dataset
        for ds_name, ds_data in all_test_datasets.items():
            # Prepare inputs based on model type
            if model_type.lower() == 'deepsets':
                test_inputs = [
                    ds_data['test']['features'],
                    ds_data['test']['attention_mask']
                ]
            else:
                test_inputs = ds_data['test']['features']
            
            # Get ground truth and predictions
            y_true = ds_data['test']['labels']
            raw_preds = model.predict(test_inputs, verbose=0)
            
            # Check for issues with predictions
            if raw_preds is None or len(raw_preds) == 0:
                print(f"WARNING: Empty predictions for {model_name} on {ds_name}")
                continue
                
            # Check if all predictions are the same value
            y_pred_proba = raw_preds.ravel()
            if len(np.unique(y_pred_proba)) <= 1:
                print(f"WARNING: All predictions are identical ({y_pred_proba[0]}) for {model_name} on {ds_name}")
                # Add a tiny bit of noise to avoid ROC calculation issues
                y_pred_proba = y_pred_proba + np.random.normal(0, 1e-5, size=y_pred_proba.shape)
                y_pred_proba = np.clip(y_pred_proba, 0, 1)
            
            # Check for NaN values
            if np.isnan(y_pred_proba).any():
                print(f"WARNING: NaN predictions for {model_name} on {ds_name}, replacing with 0.5")
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)
            
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            metrics = {
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }
            
            # Store predictions for ROC curves
            metrics['y_pred_proba'] = y_pred_proba
            metrics['y_true'] = y_true
            
            # Store in cross-results
            cross_results[model_name][ds_name] = metrics
    
    return cross_results