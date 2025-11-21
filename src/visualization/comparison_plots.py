"""
Visualization utilities for comparing standard and adversarial training.

This module provides functions for creating comparative visualizations
between standard and adversarial training regimes, including heatmaps,
performance drop plots, and parameter space visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple

from .styling import set_plot_style


def plot_comparative_heatmaps(
    standard_results: Dict[str, Dict[str, float]],
    adversarial_results: Dict[str, Dict[str, float]],
    metric: str = 'roc_auc',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create side-by-side heatmaps comparing standard and adversarial training.
    
    Parameters:
    -----------
    standard_results : Dict[str, Dict[str, float]]
        Results from standard training
    adversarial_results : Dict[str, Dict[str, float]]
        Results from adversarial training
    metric : str
        Metric to visualize
    output_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Set plot style
    set_plot_style()
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get model and dataset names
    model_names = sorted(list(standard_results.keys()))
    dataset_names = []
    for model_name in model_names:
        dataset_names.extend(list(standard_results[model_name].keys()))
    dataset_names = sorted(list(set(dataset_names)))
    
    # Create data matrices
    standard_matrix = np.zeros((len(model_names), len(dataset_names)))
    adversarial_matrix = np.zeros((len(model_names), len(dataset_names)))
    difference_matrix = np.zeros((len(model_names), len(dataset_names)))
    
    # Fill matrices
    for i, model_name in enumerate(model_names):
        for j, dataset_name in enumerate(dataset_names):
            if dataset_name in standard_results[model_name]:
                standard_matrix[i, j] = standard_results[model_name][dataset_name]['metrics'][metric]
            
            if model_name in adversarial_results and dataset_name in adversarial_results[model_name]:
                adversarial_matrix[i, j] = adversarial_results[model_name][dataset_name]['metrics'][metric]
                
            # Calculate difference (improvement)
            difference_matrix[i, j] = adversarial_matrix[i, j] - standard_matrix[i, j]
    
    # Plot standard heatmap
    im0 = axes[0].imshow(standard_matrix, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title(f'Standard Training ({metric})')
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('Model')
    axes[0].set_xticks(np.arange(len(dataset_names)))
    axes[0].set_yticks(np.arange(len(model_names)))
    axes[0].set_xticklabels(dataset_names, rotation=45, ha='right')
    axes[0].set_yticklabels(model_names)
    
    # Plot adversarial heatmap
    im1 = axes[1].imshow(adversarial_matrix, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title(f'Adversarial Training ({metric})')
    axes[1].set_xlabel('Dataset')
    axes[1].set_yticks(np.arange(len(model_names)))
    axes[1].set_xticks(np.arange(len(dataset_names)))
    axes[1].set_xticklabels(dataset_names, rotation=45, ha='right')
    axes[1].set_yticklabels([])
    
    # Plot difference heatmap
    # Use diverging colormap centered at 0
    vmax = max(abs(np.min(difference_matrix)), abs(np.max(difference_matrix)))
    im2 = axes[2].imshow(difference_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Difference (Adversarial - Standard)')
    axes[2].set_xlabel('Dataset')
    axes[2].set_yticks(np.arange(len(model_names)))
    axes[2].set_xticks(np.arange(len(dataset_names)))
    axes[2].set_xticklabels(dataset_names, rotation=45, ha='right')
    axes[2].set_yticklabels([])
    
    # Add colorbars
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_perturbation_performance(
    standard_model_results: Dict[str, Dict[str, float]],
    adversarial_model_results: Dict[str, Dict[str, float]],
    perturbation_strengths: List[str],
    metric: str = 'roc_auc',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot performance vs. perturbation strength for both training regimes.
    
    Parameters:
    -----------
    standard_model_results : Dict[str, Dict[str, float]]
        Results for standard models at different perturbation strengths
    adversarial_model_results : Dict[str, Dict[str, float]]
        Results for adversarial models at different perturbation strengths
    perturbation_strengths : List[str]
        List of perturbation strength names
    metric : str
        Metric to visualize
    output_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Set plot style
    set_plot_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract performance values
    standard_values = [standard_model_results['clean']['metrics'][metric]]
    adversarial_values = [adversarial_model_results['clean']['metrics'][metric]]
    
    for strength in perturbation_strengths:
        if strength in standard_model_results['perturbed']:
            standard_values.append(standard_model_results['perturbed'][strength]['evaluation']['metrics'][metric])
        
        if strength in adversarial_model_results['perturbed']:
            adversarial_values.append(adversarial_model_results['perturbed'][strength]['evaluation']['metrics'][metric])
    
    # Plot performance vs. perturbation strength
    x = np.arange(len(['clean'] + perturbation_strengths))
    width = 0.35
    
    ax.bar(x - width/2, standard_values, width, label='Standard Training')
    ax.bar(x + width/2, adversarial_values, width, label='Adversarial Training')
    
    # Add labels and legend
    ax.set_xlabel('Perturbation Strength')
    ax.set_ylabel(f'{metric.upper()}')
    ax.set_title(f'Performance Under Perturbation ({metric})')
    ax.set_xticks(x)
    ax.set_xticklabels(['clean'] + perturbation_strengths)
    ax.legend()
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_robustness_scores(
    standard_scores: Dict[str, Dict[str, float]],
    adversarial_scores: Dict[str, Dict[str, float]],
    perturbation_strengths: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot robustness scores for both training regimes.
    
    Parameters:
    -----------
    standard_scores : Dict[str, Dict[str, float]]
        Robustness scores for standard models
    adversarial_scores : Dict[str, Dict[str, float]]
        Robustness scores for adversarial models
    perturbation_strengths : List[str]
        List of perturbation strength names
    output_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Set plot style
    set_plot_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract robustness scores
    standard_values = []
    adversarial_values = []
    
    for strength in perturbation_strengths:
        if strength in standard_scores:
            standard_values.append(standard_scores[strength]['robustness_score'])
        
        if strength in adversarial_scores:
            adversarial_values.append(adversarial_scores[strength]['robustness_score'])
    
    # Plot robustness scores
    x = np.arange(len(perturbation_strengths))
    width = 0.35
    
    ax.bar(x - width/2, standard_values, width, label='Standard Training')
    ax.bar(x + width/2, adversarial_values, width, label='Adversarial Training')
    
    # Add labels and legend
    ax.set_xlabel('Perturbation Strength')
    ax.set_ylabel('Robustness Score')
    ax.set_title('Robustness Scores (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(perturbation_strengths)
    ax.legend()
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history_comparison(
    standard_history: Dict[str, List[float]],
    adversarial_history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot training history comparison between standard and adversarial training.
    
    Parameters:
    -----------
    standard_history : Dict[str, List[float]]
        Training history for standard model
    adversarial_history : Dict[str, List[float]]
        Training history for adversarial model
    output_path : str, optional
        Path to save figure
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Set plot style
    set_plot_style()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot training loss
    axes[0, 0].plot(standard_history['loss'], label='Standard')
    if 'total_loss' in adversarial_history:
        axes[0, 0].plot(adversarial_history['total_loss'], label='Adversarial')
    elif 'loss' in adversarial_history:
        axes[0, 0].plot(adversarial_history['loss'], label='Adversarial')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot validation loss
    axes[0, 1].plot(standard_history['val_loss'], label='Standard')
    axes[0, 1].plot(adversarial_history['val_loss'], label='Adversarial')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot adversarial loss components if available
    if 'ce_loss' in adversarial_history and 'kl_loss' in adversarial_history:
        axes[1, 0].plot(adversarial_history['ce_loss'], label='CE Loss')
        axes[1, 0].plot(adversarial_history['kl_loss'], label='KL Loss')
        axes[1, 0].set_title('Adversarial Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # Plot validation metrics if available
    if 'val_auc' in standard_history and 'val_auc' in adversarial_history:
        axes[1, 1].plot(standard_history['val_auc'], label='Standard')
        axes[1, 1].plot(adversarial_history['val_auc'], label='Adversarial')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig