"""
Dataset preparation utilities for combining multiple physics datasets.
"""

import os
import numpy as np
from typing import Dict, List, Any

from .loader import load_dataset


def create_dataset(
    file_list: List[str],
    use_scaled: bool = True,
    signal_background_mode: bool = True,
    dark_jets_only: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Create a combined dataset from multiple files for ML training.
    
    Parameters:
    -----------
    file_list : list of str
        List of file paths to process
    use_scaled : bool
        Whether to use scaled features (True) or raw features (False)
    signal_background_mode : bool
        If True:
        - From dark sector files: keep ONLY dark jets (signal)
        - From SM file: keep ALL jets (background)
    dark_jets_only : bool
        Whether to filter to include only dark jets from all files
    verbose : bool
        Whether to print information during processing
    
    Returns:
    --------
    dict
        Dictionary containing combined dataset ready for ML
    """
    # Check if files exist
    for file_path in file_list:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Lists to collect data from all files
    all_features = []
    all_labels = []
    all_is_dark = []
    all_is_valid = []  # NEW: Keep track of valid particle masks
    file_sources = []  # Track which file each jet came from
    
    for file_idx, file_path in enumerate(file_list):
        # Load data from file
        data = load_dataset(file_path, verbose=verbose)
        
        # Select features based on preference
        features = data["scaled_features"] if use_scaled else data["particle_features"]
        
        # Get labels (is_dark_jet flag)
        labels = data["is_dark_jet"]
        
        # Get valid particle mask
        is_valid = data["is_valid_particle"]  # NEW: Extract valid particle mask
        
        # Handle different dataset construction modes
        if signal_background_mode:
            is_sm_file = "NominalSM" in file_path
            
            if is_sm_file:
                # SM file: all jets are background (label=0)
                processed_labels = np.zeros_like(labels)
                processed_features = features
                processed_is_dark = labels  # Keep original dark flag for analysis
                processed_is_valid = is_valid  # NEW: Keep valid particle mask
                file_src = np.full(len(labels), file_idx)
            else:
                # Dark files: keep only dark jets (label=1)
                dark_mask = labels
                
                # Skip if no dark jets in this file
                if np.sum(dark_mask) == 0:
                    if verbose:
                        print(f"  Skipping {os.path.basename(file_path)} - no dark jets found")
                    continue
                
                processed_features = features[dark_mask]
                processed_labels = np.ones(np.sum(dark_mask))
                processed_is_dark = labels[dark_mask]  # Should be all True
                processed_is_valid = is_valid[dark_mask]  # NEW: Filter valid mask too
                file_src = np.full(np.sum(dark_mask), file_idx)
        else:
            # Simple mode: just use the is_dark_jet flag as labels
            processed_labels = labels.astype(int)
            processed_is_dark = labels
            
            # Filter to dark jets only if requested
            if dark_jets_only:
                dark_mask = labels
                if np.sum(dark_mask) == 0:
                    if verbose:
                        print(f"  Skipping {os.path.basename(file_path)} - no dark jets found")
                    continue
                processed_features = features[dark_mask]
                processed_labels = processed_labels[dark_mask]
                processed_is_dark = processed_is_dark[dark_mask]
                processed_is_valid = is_valid[dark_mask]  # NEW: Filter valid mask too
                file_src = np.full(np.sum(dark_mask), file_idx)
            else:
                processed_features = features
                processed_is_valid = is_valid  # NEW: Keep all valid masks
                file_src = np.full(len(processed_labels), file_idx)
        
        # Add to collection
        all_features.append(processed_features)
        all_labels.append(processed_labels)
        all_is_dark.append(processed_is_dark)
        all_is_valid.append(processed_is_valid)  # NEW: Collect valid masks
        file_sources.append(file_src)
    
    # Combine across files
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    is_dark = np.concatenate(all_is_dark)
    is_valid = np.vstack(all_is_valid)  # NEW: Stack valid masks
    sources = np.concatenate(file_sources)
    
    # Create metadata
    metadata = {
        "total_jets": len(labels),
        "total_dark_jets": int(np.sum(is_dark)),
        "class_balance": {
            "signal": float(np.sum(labels)) / len(labels),
            "background": float(len(labels) - np.sum(labels)) / len(labels)
        },
        "file_list": file_list,
        "source_mapping": {i: os.path.basename(fp) for i, fp in enumerate(file_list)}
    }
    
    # Return combined dataset
    result = {
        "features": features,
        "labels": labels,
        "is_dark": is_dark,
        "is_valid": is_valid,  # NEW: Include valid particle mask
        "sources": sources,
        "metadata": metadata
    }
    
    if verbose:
        print(f"Combined dataset created:")
        print(f"  Total jets: {metadata['total_jets']}")
        print(f"  Signal jets: {int(np.sum(labels))} ({100 * metadata['class_balance']['signal']:.1f}%)")
        print(f"  Background jets: {len(labels) - int(np.sum(labels))} ({100 * metadata['class_balance']['background']:.1f}%)")
        print(f"  Feature shape: {features.shape}")
    
    return result