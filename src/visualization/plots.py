"""
Plotting utilities for model evaluation and comparison.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    
    # Safety check for empty results
    if not model_names:
        print("Warning: No models in cross_eval_results")
        return plt.figure()
    
    dataset_names = list(cross_eval_results[model_names[0]].keys())
    
    # Safety check for empty datasets
    if not dataset_names:
        print("Warning: No datasets in cross_eval_results")
        return plt.figure()
    
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


# ====================================================================
# DATASET COMPARISON VISUALIZATION - NEW FUNCTIONALITY
# ====================================================================


def _adapt_results_for_dataset_comparison(
    individual_results: Dict[str, Any],
    individual_cross_eval: Dict[str, Dict[str, Dict[str, Any]]],
    loo_results: Optional[Dict[str, Any]] = None,
    loo_cross_eval: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
) -> Dict[str, Any]:
    """
    Adapt current results structure to format expected by dataset comparison function.
    
    Parameters:
    -----------
    individual_results : dict
        Individual model training results from current pipeline
    individual_cross_eval : dict
        Individual model cross-evaluation results from current pipeline
    loo_results : dict, optional
        Leave-one-out model training results from current pipeline
    loo_cross_eval : dict, optional
        Leave-one-out model cross-evaluation results from current pipeline
        
    Returns:
    --------
    dict
        Adapted results structure compatible with original notebook function
    """
    adapted = {
        'individual': individual_results,
        'individual_eval': individual_cross_eval
    }
    
    if loo_results and loo_cross_eval:
        adapted['leave_one_out'] = loo_results
        adapted['leave_one_out_eval'] = loo_cross_eval
    
    return adapted


def plot_dataset_comparison_models(
    individual_results: Dict[str, Any],
    individual_cross_eval: Dict[str, Dict[str, Dict[str, Any]]],
    target_dataset: str,
    loo_results: Optional[Dict[str, Any]] = None,
    loo_cross_eval: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create a specialized plot showing multiple models evaluated on one target dataset.
    
    This provides the "dataset-centric" view: How do all our different models 
    perform on this specific parameter point?
    
    Parameters:
    -----------
    individual_results : dict
        Individual model training results from current pipeline
    individual_cross_eval : dict
        Individual model cross-evaluation results from current pipeline
    target_dataset : str
        Name of the target dataset to focus analysis on
    loo_results : dict, optional
        Leave-one-out model training results from current pipeline
    loo_cross_eval : dict, optional
        Leave-one-out model cross-evaluation results from current pipeline
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    sm_accuracy : float
        Standard Model accuracy for reference line
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Adapt data structure to match original notebook expectations
    results = _adapt_results_for_dataset_comparison(
        individual_results, individual_cross_eval, loo_results, loo_cross_eval
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Track already plotted models to avoid duplicates
    plotted_models = set()
    
    # Helper function to extract ROC data and plot a curve
    def plot_model_on_dataset(model_name, dataset_name, eval_results, label_prefix, style=None):
        # Extract dataset results for this model
        model_results = eval_results.get(model_name, {})
        dataset_results = None
        
        # Try to find evaluation results on the target dataset
        # Handle both current structure and any nested structures
        if isinstance(model_results, dict) and dataset_name in model_results:
            dataset_results = model_results[dataset_name]
        elif isinstance(model_results, dict) and 'cross_results' in model_results:
            dataset_results = model_results['cross_results'].get(dataset_name)
        
        # If not found, try with different prefixes/formats
        if not dataset_results and dataset_name.startswith('loo_'):
            # Try without 'loo_' prefix
            alt_name = dataset_name[4:]
            if alt_name in model_results:
                dataset_results = model_results[alt_name]
        
        if not dataset_results:
            print(f"No results found for {model_name} on {dataset_name}")
            return None
        
        # Extract prediction data
        if isinstance(dataset_results, dict) and 'predictions' in dataset_results:
            y_true = dataset_results['predictions']['y_true']
            y_pred = dataset_results['predictions']['y_pred_proba']
        elif isinstance(dataset_results, dict) and 'y_true' in dataset_results:
            y_true = dataset_results['y_true']
            y_pred = dataset_results['y_pred_proba']
        else:
            print(f"Cannot find prediction data for {model_name} on {dataset_name}")
            return None
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Get styling
        if style is None:
            style = get_model_style(model_name)
        
        # Format model name for display
        display_name = format_model_name(model_name)
        
        # Plot with consistent styling
        ax.plot(
            fpr, tpr, 
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=f'{label_prefix} {display_name} (AUC = {roc_auc:.4f})'
        )
        
        # Mark as plotted
        plotted_models.add(model_name)
        return roc_auc
    
    # 1. First plot the model trained directly on this dataset (should be highlighted)
    target_model_name = next((name for name in results['individual'].keys() 
                              if target_dataset in name), None)
    
    if target_model_name:
        # Use a distinctive style for the target model
        target_style = get_model_style(target_dataset)
        target_style["linewidth"] = 3  # Make it slightly thicker
        
        # Plot target model with special prefix
        plot_model_on_dataset(
            target_model_name, 
            target_dataset, 
            results['individual_eval'], 
            "Target model:", 
            target_style
        )
    
    # 2. Plot models trained on other dark datasets
    for model_name in results['individual'].keys():
        if model_name in plotted_models or target_dataset in model_name:
            continue  # Skip already plotted models and target model
        
        plot_model_on_dataset(
            model_name, 
            target_dataset, 
            results['individual_eval'], 
            "Model trained on"
        )
        
    # 3. Plot leave-one-out model if available
    if loo_results and loo_cross_eval:
        # Find the LOO model for this dataset
        target_dataset_base = target_dataset.replace('model_', '')
        
        # Try with 'loo_' prefix first (this is how LOO datasets are named)
        loo_dataset_name = f"loo_{target_dataset_base}"
        loo_model_name = next((name for name in loo_results.keys() 
                              if loo_dataset_name in name), None)
        
        # If not found, try without prefix
        if not loo_model_name:
            loo_model_name = next((name for name in loo_results.keys() 
                                if target_dataset_base in name), None)
        
        if loo_model_name:
            loo_style = {"color": "black", "linestyle": "-.", "linewidth": 2.5}
            
            # Use loo_model_name for lookup in loo_cross_eval
            plot_model_on_dataset(
                loo_model_name, 
                loo_model_name,  # Use the LOO model name as the dataset name
                loo_cross_eval,  # Use LOO cross-eval results
                "Leave-one-out model:", 
                loo_style
            )
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1, label='Random')
    
    # Set axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    # Set title
    display_name = format_model_name(target_dataset)
    title = f'Model Comparison on {display_name} Dataset'
    ax.set_title(title)
    
    # Add legend and grid
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
