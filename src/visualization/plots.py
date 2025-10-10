"""
Plotting utilities for model evaluation and comparison.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt

# Import our styling functions
from .styling import get_model_style, format_model_name


def plot_roc_curve(
    evaluation_results: Dict[str, Any],
    model_name: str,
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curve for a single model evaluation.
    
    Parameters:
    -----------
    evaluation_results : dict
        Evaluation results from evaluate_model
    model_name : str
        Name of the model for the legend
    title : str, optional
        Plot title (default: model_name)
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Extract ROC curve data
    fpr = evaluation_results['roc_curve']['fpr']
    tpr = evaluation_results['roc_curve']['tpr']
    roc_auc = evaluation_results['metrics']['roc_auc']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine model color and style based on parameters
    style = get_model_style(model_name)
    
    # Plot ROC curve with consistent styling
    ax.plot(fpr, tpr, 
            color=style["color"], 
            linestyle=style["linestyle"], 
            linewidth=style["linewidth"],
            label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'navy', linestyle='--', lw=1, label='Random')
    
    # Set axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    # Set title
    if title is None:
        title = f'ROC Curve - {model_name}'
    ax.set_title(title)
    
    # Add legend and grid
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_combined_roc_curves(
    all_results: Dict[str, Any],
    title: str = 'Comparison of ROC Curves',
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on one figure with consistent styling.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary of evaluation results for multiple models
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    from sklearn.metrics import roc_curve, auc
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve for each model
    for model_name, results in all_results.items():
        # Get the styling based on model parameters
        style = get_model_style(model_name)
        
        # Check if roc_curve is directly available
        if 'roc_curve' in results:
            fpr = results['roc_curve']['fpr']
            tpr = results['roc_curve']['tpr']
            roc_auc = results['metrics']['roc_auc']
        else:
            # Compute ROC curve from predictions if not available
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = results['roc_auc']
        
        # Format model name for display to match reference image
        display_name = format_model_name(model_name)
        
        # Plot with consistent styling
        ax.plot(
            fpr, tpr,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=f'Tested on {display_name} (AUC = {roc_auc:.4f})'
        )
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1, label='Random')
    
    # Set axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    # Set title
    ax.set_title(title)
    
    # Add legend and grid
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cross_model_roc_curves(
    cross_eval_results: Dict[str, Dict[str, Dict[str, Any]]],
    model_name: str,
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot ROC curves for one model evaluated on multiple datasets with consistent styling.
    
    Parameters:
    -----------
    cross_eval_results : dict
        Results from evaluate_cross_model_performance
    model_name : str
        Name of the model to plot curves for
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    from sklearn.metrics import roc_curve, auc
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get results for this model
    model_results = cross_eval_results[model_name]
    
    # Plot ROC curve for each dataset
    for ds_name, results in model_results.items():
        # Get styling based on dataset parameters
        style = get_model_style(ds_name)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
        roc_auc = results['roc_auc']
        
        # Format dataset name for display
        display_name = format_model_name(ds_name)
        
        # Plot curve with consistent styling
        ax.plot(
            fpr, tpr, 
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=f'Tested on {display_name} (AUC = {roc_auc:.4f})'
        )
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1, label='Random')
    
    # Set axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    # Set title
    if title is None:
        title = f'ROC Curves for Model {model_name} on Different Datasets'
    ax.set_title(title)
    
    # Add legend and grid
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cross_model_heatmap(
    cross_eval_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'roc_auc',
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Create a heatmap showing cross-model performance on different datasets.
    
    Parameters:
    -----------
    cross_eval_results : dict
        Results from evaluate_cross_model_performance
    metric : str
        Metric to display in the heatmap ('roc_auc', 'precision', 'recall', 'f1')
    title : str, optional
        Heatmap title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    cmap : str
        Colormap for the heatmap
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    import pandas as pd
    import seaborn as sns
    
    # Extract model and dataset names
    model_names = list(cross_eval_results.keys())
    dataset_names = list(cross_eval_results[model_names[0]].keys())
    
    # Create matrix for heatmap
    data = np.zeros((len(model_names), len(dataset_names)))
    
    # Fill matrix with metric values
    for i, model_name in enumerate(model_names):
        for j, ds_name in enumerate(dataset_names):
            data[i, j] = cross_eval_results[model_name][ds_name][metric]
    
    # Create dataframe
    df = pd.DataFrame(data, index=model_names, columns=dataset_names)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        df, 
        annot=True, 
        fmt='.3f', 
        cmap=cmap,
        linewidths=0.5, 
        ax=ax
    )
    
    # Set title
    if title is None:
        title = f'Cross-Model {metric.upper()} Performance'
    ax.set_title(title)
    
    # Set axis labels
    ax.set_xlabel('Test Dataset')
    ax.set_ylabel('Model Trained On')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str,
    title: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot training history (loss and accuracy curves).
    
    Parameters:
    -----------
    history : dict
        Training history from model.fit
    model_name : str
        Name of the model
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training & validation loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot training & validation accuracy
    acc_key = 'accuracy' if 'accuracy' in history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
    
    #ax2.plot(history['val_auc'], label='Validation AUC')  
    ax2.plot(history[acc_key], label='Training Accuracy')
    ax2.plot(history[val_acc_key], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Set overall title
    if title is None:
        title = f'Training History - {model_name}'
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
