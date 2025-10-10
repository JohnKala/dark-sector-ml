"""
Model evaluation metrics and utilities.
"""

from typing import Dict, Any, Optional
import numpy as np

# Conditional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def evaluate_model(
    model: 'tf.keras.Model',
    test_data: Dict[str, Any],
    model_type: str = 'deepsets',
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model to evaluate
    test_data : dict
        Test data prepared for the model
    model_type : str
        Type of model ('dense' or 'deepsets')
    threshold : float
        Classification threshold
    verbose : bool
        Whether to print evaluation results
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and predictions
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for model evaluation")
    
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve,
        precision_score, recall_score, f1_score, confusion_matrix
    )
    
    # Prepare inputs based on model type
    if model_type.lower() == 'deepsets':
        test_inputs = [
            test_data['test']['features'],
            test_data['test']['attention_mask']
        ]
    else:
        test_inputs = test_data['test']['features']
    
    # Get ground truth labels
    y_true = test_data['test']['labels']
    
    # Get model predictions
    y_pred_proba = model.predict(test_inputs, verbose=0).ravel()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Other metrics
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    f1_val = f1_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Collect results
    results = {
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr
        },
        'pr_curve': {
            'precision': precision,
            'recall': recall
        },
        'metrics': {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': precision_val,
            'recall': recall_val,
            'f1': f1_val
        },
        'confusion_matrix': cm,
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    }
    
    # Print results if verbose
    if verbose:
        print(f"\n{'-'*20} EVALUATION RESULTS {'-'*20}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"Precision: {precision_val:.4f}")
        print(f"Recall: {recall_val:.4f}")
        print(f"F1 Score: {f1_val:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"               Neg    Pos")
        print(f"Actual Neg   {cm[0,0]:6d} {cm[0,1]:6d}")
        print(f"      Pos   {cm[1,0]:6d} {cm[1,1]:6d}")
    
    return results


def evaluate_all_models(
    trained_models: Dict[str, Any],
    test_datasets: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate all trained models on their respective test datasets.
    
    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained models from train_individual_models or train_leave_one_out_models
    test_datasets : dict, optional
        Dictionary of additional test datasets to evaluate on
    verbose : bool
        Whether to print evaluation results
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results for all models
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for model evaluation")
    
    eval_results = {}
    
    for model_name, model_data in trained_models.items():
        model = model_data['model']
        model_type = model_data['params']['model_type']
        
        if verbose:
            print(f"\n{'-'*30}")
            print(f"Evaluating model: {model_name}")
        
        # Evaluate on the model's own test dataset
        if 'left_out_data' in model_data:
            # For leave-one-out models, evaluate on left-out dataset
            test_data = model_data['left_out_data']
        else:
            # For individual models, use their own test data
            test_data = model_data
        
        # Evaluate model
        results = evaluate_model(
            model=model,
            test_data=test_data,
            model_type=model_type,
            verbose=verbose
        )
        
        # Store results
        eval_results[model_name] = results
        
        # If additional test datasets provided, evaluate on those too
        if test_datasets is not None:
            for ds_name, ds_data in test_datasets.items():
                if verbose:
                    print(f"\nEvaluating {model_name} on {ds_name}...")
                
                # Evaluate on this dataset
                ds_results = evaluate_model(
                    model=model,
                    test_data=ds_data,
                    model_type=model_type,
                    verbose=verbose
                )
                
                # Store results with dataset name
                eval_results[f"{model_name}_on_{ds_name}"] = ds_results
    
    return eval_results