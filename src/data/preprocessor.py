"""
Data preprocessing utilities for ML model preparation.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import constants from config
from ..config import NUM_PARTICLES, NUM_FEATURES


def prepare_ml_dataset(
    dataset: Dict[str, Any],
    test_size: float = 0.2,
    val_size: float = 0.25,
    normalize: bool = False,  # CHANGED: Default to False to match original behavior
    reshape_3d: bool = False,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Split a dataset into train/val/test and normalize features.
    
    Parameters:
    -----------
    dataset : dict
        Dataset created by create_dataset
    test_size : float
        Fraction of data to use for testing
    val_size : float
        Fraction of remaining data to use for validation
    normalize : bool
        Whether to apply feature normalization (StandardScaler)
    reshape_3d : bool
        Whether to reshape features to 3D format (for DeepSets)
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print information during processing
        
    Returns:
    --------
    dict
        Dictionary containing prepared ML datasets and metadata
    """
    X = dataset["features"]
    y = dataset["labels"]
    is_valid = dataset["is_valid"]  # NEW: Extract valid particle mask
    
    # Split into train/test first, including is_valid mask
    # FIXED: Handle stratification failure with imbalanced classes
    try:
        X_train, X_temp, y_train, y_temp, valid_train, valid_temp = train_test_split(
            X, y, is_valid, test_size=test_size, stratify=y, random_state=random_state
        )
    except ValueError as e:
        if "least populated class" in str(e) or "stratify" in str(e):
            # Fall back to non-stratified split
            X_train, X_temp, y_train, y_temp, valid_train, valid_temp = train_test_split(
                X, y, is_valid, test_size=test_size, random_state=random_state
            )
            if verbose:
                print("Warning: Stratified train/test split failed due to class imbalance, using random split")
        else:
            raise e
    
    # Split temp into val/test, including is_valid mask
    val_ratio = val_size / (1 - test_size)
    try:
        X_val, X_test, y_val, y_test, valid_val, valid_test = train_test_split(
            X_temp, y_temp, valid_temp, test_size=val_ratio, 
            stratify=y_temp, random_state=random_state
        )
    except ValueError as e:
        if "least populated class" in str(e) or "stratify" in str(e):
            # Fall back to non-stratified split
            X_val, X_test, y_val, y_test, valid_val, valid_test = train_test_split(
                X_temp, y_temp, valid_temp, test_size=val_ratio, 
                random_state=random_state
            )
            if verbose:
                print("Warning: Stratified val/test split failed due to class imbalance, using random split")
        else:
            raise e
    
    # Create split dictionaries, including is_valid mask
    splits = {
        "train": {"X": X_train, "y": y_train, "is_valid": valid_train},
        "val": {"X": X_val, "y": y_val, "is_valid": valid_val},
        "test": {"X": X_test, "y": y_test, "is_valid": valid_test}
    }
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        
        # Store original -999 positions
        train_mask = np.isclose(X_train, -999)
        val_mask = np.isclose(X_val, -999)
        test_mask = np.isclose(X_test, -999)
        
        # Fit scaler only on valid (non-padding) values
        # Extract only non-padding values for fitting by masking them as NaN, then use nanmean/nanstd
        X_train_masked = X_train.copy()
        X_train_masked[train_mask] = np.nan
        
        # Compute mean and std per column, ignoring NaN (padding)
        scaler.mean_ = np.nanmean(X_train_masked, axis=0)
        scaler.scale_ = np.nanstd(X_train_masked, axis=0)
        # Avoid division by zero for columns that are all padding
        scaler.scale_[scaler.scale_ == 0] = 1.0
        scaler.n_features_in_ = X_train.shape[1]
        
        # Transform all splits
        norm_train = (X_train - scaler.mean_) / scaler.scale_
        norm_val = (X_val - scaler.mean_) / scaler.scale_
        norm_test = (X_test - scaler.mean_) / scaler.scale_
        
        # Restore -999 in original positions
        norm_train[train_mask] = -999
        norm_val[val_mask] = -999
        norm_test[test_mask] = -999
        
        # Apply 3D reshaping if requested
        if reshape_3d:
            norm_train = norm_train.reshape(X_train.shape[0], NUM_PARTICLES, NUM_FEATURES)
            norm_val = norm_val.reshape(X_val.shape[0], NUM_PARTICLES, NUM_FEATURES)
            norm_test = norm_test.reshape(X_test.shape[0], NUM_PARTICLES, NUM_FEATURES)
            
            # Also reshape valid masks if using 3D
            valid_train = valid_train.reshape(valid_train.shape[0], NUM_PARTICLES)
            valid_val = valid_val.reshape(valid_val.shape[0], NUM_PARTICLES)
            valid_test = valid_test.reshape(valid_test.shape[0], NUM_PARTICLES)
        
        # Collect normalization parameters
        norm_params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }
        
        # Update splits with normalized data
        splits["train"]["X_norm"] = norm_train
        splits["val"]["X_norm"] = norm_val
        splits["test"]["X_norm"] = norm_test
        splits["norm_params"] = norm_params
        
        if verbose:
            print(f"Features normalized" + (" and reshaped to 3D" if reshape_3d else ""))
    elif reshape_3d:
        # If only reshaping without normalization
        splits["train"]["X"] = X_train.reshape(X_train.shape[0], NUM_PARTICLES, NUM_FEATURES)
        splits["val"]["X"] = X_val.reshape(X_val.shape[0], NUM_PARTICLES, NUM_FEATURES)
        splits["test"]["X"] = X_test.reshape(X_test.shape[0], NUM_PARTICLES, NUM_FEATURES)
        
        # Also reshape valid masks
        splits["train"]["is_valid"] = valid_train.reshape(valid_train.shape[0], NUM_PARTICLES)
        splits["val"]["is_valid"] = valid_val.reshape(valid_val.shape[0], NUM_PARTICLES)
        splits["test"]["is_valid"] = valid_test.reshape(valid_test.shape[0], NUM_PARTICLES)
        
        if verbose:
            print("Features reshaped to 3D")
    
    # Add metadata
    splits["metadata"] = {
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "train_balance": {
            "signal": float(np.sum(y_train)) / len(y_train),
            "background": float(len(y_train) - np.sum(y_train)) / len(y_train)
        },
        "original_metadata": dataset["metadata"]
    }
    
    if verbose:
        print(f"Dataset split:")
        print(f"  Train: {len(y_train)} jets ({100 * splits['metadata']['train_balance']['signal']:.1f}% signal)")
        print(f"  Validation: {len(y_val)} jets")
        print(f"  Test: {len(y_test)} jets")
    
    return splits


def prepare_deepsets_data(
    ml_dataset: Dict[str, Any],
    return_masks: bool = True
) -> Dict[str, Any]:
    """
    Prepare data for DeepSets model by reshaping to 3D and creating masks.
    Parameters:
    -----------
    ml_dataset : dict
        Dataset prepared by prepare_ml_dataset
    return_masks : bool
        Whether to generate and return attention masks
    Returns:
    --------
    dict
        Dictionary with data ready for DeepSets training/evaluation
    """
    deepsets_data = {}
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        # Get normalized data if available, otherwise raw data
        if 'X_norm' in ml_dataset[split]:
            X = ml_dataset[split]['X_norm']
        else:
            X = ml_dataset[split]['X']
        y = ml_dataset[split]['y']
        # Get the valid particle mask directly
        mask = ml_dataset[split]['is_valid']  # NEW: Use the propagated mask
        # Reshape to 3D if needed
        if len(X.shape) == 2:
            X_3d = X.reshape(X.shape[0], NUM_PARTICLES, NUM_FEATURES)
            # Reshape mask if it's 2D
            if len(mask.shape) == 2:
                mask = mask.reshape(mask.shape[0], NUM_PARTICLES)
        else:
            X_3d = X
        # Store prepared data
        deepsets_data[split] = {
            'features': X_3d.astype('float32'),
            'labels': y.astype('float32')
        }
        if return_masks:
            deepsets_data[split]['attention_mask'] = mask.astype('bool')
    # Add original metadata
    deepsets_data['metadata'] = ml_dataset['metadata']
    return deepsets_data