"""
Visualization utilities for comparing Standard vs. Adversarial models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

def plot_feature_shift(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
    feature_names: list = ["pT", "eta", "phi"],
    save_path: Optional[str] = None
):
    """
    Plot histograms of Clean vs. Perturbed features to see what the attack changed.
    
    Parameters:
    -----------
    clean_features : np.ndarray
        Original features (N, num_features) or (N, num_particles, num_features)
    perturbed_features : np.ndarray
        Adversarially perturbed features
    feature_names : list
        Names of the features
    save_path : str, optional
        Path to save the plot
    """
    # Flatten if 3D (DeepSets format: N, P, F) -> (N*P, F)
    if len(clean_features.shape) == 3:
        clean_flat = clean_features.reshape(-1, clean_features.shape[-1])
        pert_flat = perturbed_features.reshape(-1, perturbed_features.shape[-1])
    else:
        clean_flat = clean_features
        pert_flat = perturbed_features
        
    # Remove padding (assuming <= -900 padding for masked particles)
    # A simple heuristic: if pT (index 0) is <= -900, it's padding
    mask = clean_flat[:, 0] > -900
    clean_flat = clean_flat[mask]
    pert_flat = pert_flat[mask]
    
    num_features = min(len(feature_names), clean_flat.shape[1])
    fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 4))
    
    if num_features == 1:
        axes = [axes]
        
    for i in range(num_features):
        ax = axes[i]
        name = feature_names[i]
        
        # Plot histograms
        sns.histplot(clean_flat[:, i], ax=ax, color='blue', alpha=0.5, label='Clean', stat='density', bins=50)
        sns.histplot(pert_flat[:, i], ax=ax, color='red', alpha=0.5, label='Perturbed', stat='density', bins=50)
        
        ax.set_title(f"{name} Distribution")
        ax.set_xlabel(name)
        ax.legend()
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_prediction_shift(
    clean_probs: np.ndarray,
    perturbed_probs: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot histograms of Model Scores (Clean vs. Perturbed) to see if the attack worked.
    
    Parameters:
    -----------
    clean_probs : np.ndarray
        Model probabilities on clean data
    perturbed_probs : np.ndarray
        Model probabilities on perturbed data
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    sns.histplot(clean_probs, color='blue', alpha=0.5, label='Clean Score', stat='density', bins=50)
    sns.histplot(perturbed_probs, color='red', alpha=0.5, label='Perturbed Score', stat='density', bins=50)
    
    plt.title("Prediction Shift Analysis")
    plt.xlabel("Model Probability (Signal)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_delta_heatmap(
    clean_features: np.ndarray,
    perturbed_features: np.ndarray,
    feature_names: list = ["pT", "eta", "phi"],
    save_path: Optional[str] = None
):
    """
    Plot 2D heatmap of feature changes (Deltas).
    Specifically useful for Delta Eta vs Delta Phi.
    
    Parameters:
    -----------
    clean_features : np.ndarray
        Original features
    perturbed_features : np.ndarray
        Perturbed features
    feature_names : list
        Names of features to identify indices
    save_path : str, optional
        Path to save the plot
    """
    # Flatten if 3D
    if len(clean_features.shape) == 3:
        clean_flat = clean_features.reshape(-1, clean_features.shape[-1])
        pert_flat = perturbed_features.reshape(-1, perturbed_features.shape[-1])
    else:
        clean_flat = clean_features
        pert_flat = perturbed_features
        
    # Filter padding (<= -900)
    mask = clean_flat[:, 0] > -900
    clean_flat = clean_flat[mask]
    pert_flat = pert_flat[mask]
    
    # Calculate Deltas
    deltas = pert_flat - clean_flat
    
    # Find indices for eta and phi
    try:
        eta_idx = feature_names.index("eta")
        phi_idx = feature_names.index("phi")
    except ValueError:
        print("Warning: 'eta' or 'phi' not found in feature names. Skipping Delta Heatmap.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist2d(deltas[:, eta_idx], deltas[:, phi_idx], bins=50, cmap='viridis', density=True)
    plt.colorbar(label='Density')
    plt.title("Perturbation Delta: $\Delta\eta$ vs $\Delta\phi$")
    plt.xlabel("$\Delta\eta$")
    plt.ylabel("$\Delta\phi$")
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
