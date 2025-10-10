"""
Data loading utilities for dark sector physics datasets.
"""

import os
import numpy as np
import h5py
from typing import Dict, Tuple, List, Optional, Union, Any

# Import constants from config
from ..config import NUM_PARTICLES, NUM_FEATURES


def load_dataset(
    file_path: str,
    extract_params: bool = True,
    create_scaled: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Unified function to load HDF5 particle physics data, extract parameters,
    and create scaled features in one operation.
    
    Parameters:
    -----------
    file_path : str
        Path to the H5 file
    extract_params : bool
        Whether to extract physics parameters from filename
    create_scaled : bool
        Whether to generate scaled features relative to jet properties
    verbose : bool
        Whether to print information during loading
    
    Returns:
    --------
    dict
        A dictionary containing all data structures needed for ML processing
    """
    if verbose:
        print(f"Loading data from {os.path.basename(file_path)}...")
    
    # Initialize result dictionary
    result = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path)
    }
    
    # Extract parameters from filename if requested
    if extract_params:
        result["parameters"] = extract_parameters(file_path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as hf:
            # Check for required datasets
            required_keys = ["particle_features", "jet_pt", "jet_eta", "jet_phi"]
            if not all(key in hf for key in required_keys):
                missing = [key for key in required_keys if key not in hf]
                raise ValueError(f"Missing required datasets in {file_path}: {missing}")
            
            # Load particle features
            particle_features = hf["particle_features"][:]
            
            # Load jet features
            jet_pt = hf["jet_pt"][:]
            jet_eta = hf["jet_eta"][:]
            jet_phi = hf["jet_phi"][:]
            
            # Check if this is a dark sector model
            is_dark_model = "NominalSM" not in file_path
            
            # For dark sector models, try to find jet_is_dark
            if is_dark_model:
                if "jet_is_dark" in hf:
                    jet_is_dark = hf["jet_is_dark"][:].astype(bool)
                else:
                    # If jet_is_dark is missing, assume all jets are NON-dark (original behavior)
                    if verbose:
                        print(f"  Warning: 'jet_is_dark' not found in {file_path}. Assuming all jets are NON-dark.")
                    jet_is_dark = np.zeros(particle_features.shape[0], dtype=bool)
            else:
                # For SM, all jets are non-dark
                jet_is_dark = np.zeros(particle_features.shape[0], dtype=bool)
            
            # Create mask for valid particles (pT >= 0)
            reshaped = particle_features.reshape(particle_features.shape[0], 
                                               NUM_PARTICLES, NUM_FEATURES)
            particle_pt = reshaped[:, :, 0]  # [n_jets, n_particles]
            is_valid_particle = particle_pt >= 0
            
            # Collect jet features in a dictionary
            jet_features = {
                "pt": jet_pt,
                "eta": jet_eta,
                "phi": jet_phi
            }
            
            # Prepare result
            result.update({
                "particle_features": particle_features,
                "jet_features": jet_features,
                "is_dark_jet": jet_is_dark,
                "is_valid_particle": is_valid_particle,
                "n_jets": particle_features.shape[0],
                "n_dark_jets": int(np.sum(jet_is_dark))
            })
            
            # Create scaled features if requested
            if create_scaled:
                result["scaled_features"] = create_scaled_features(
                    particle_features, jet_features, is_valid_particle
                )
            
            if verbose:
                print(f"  Loaded {result['n_jets']} jets ({result['n_dark_jets']} dark jets)")
            
    except Exception as e:
        raise IOError(f"Error processing {file_path}: {str(e)}")
    
    return result


def extract_parameters(file_path: str) -> Dict[str, Union[float, np.float64]]:
    """
    Extract physics parameters from filename.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
        
    Returns:
    --------
    dict
        Dictionary of parameter names and values
    """
    base = os.path.basename(file_path).replace('.h5', '')
    params = {}
    
    for token in base.split('_'):
        if '-' not in token:
            continue
        
        key, val = token.split('-', 1)
        try:
            params[key] = float(val)
        except ValueError:
            # Handle non-numeric values (like 'peak', 'low', 'high')
            # Keep as string
            params[key] = val
    
    return params


def create_scaled_features(
    particle_features: np.ndarray,
    jet_features: Dict[str, np.ndarray],
    is_valid_particle: np.ndarray
) -> np.ndarray:
    """
    Create scaled particle features relative to their jets.
    
    Parameters:
    -----------
    particle_features : ndarray
        Original particle features
    jet_features : dict
        Jet features (pt, eta, phi)
    is_valid_particle : ndarray
        Boolean mask for valid particles
        
    Returns:
    --------
    ndarray
        Scaled particle features with same shape as particle_features
    """
    # Reshape particle features to [n_jets, n_particles, n_features]
    n_jets = particle_features.shape[0]
    reshaped = particle_features.reshape(n_jets, NUM_PARTICLES, NUM_FEATURES)
    
    # Extract individual feature arrays
    particle_pt = reshaped[:, :, 0]  # [n_jets, n_particles]
    particle_eta = reshaped[:, :, 1]  # [n_jets, n_particles]
    particle_phi = reshaped[:, :, 2]  # [n_jets, n_particles]
    
    # Extract jet features and reshape for broadcasting
    jet_pt = jet_features["pt"].reshape(-1, 1)  # [n_jets, 1]
    jet_eta = jet_features["eta"].reshape(-1, 1)  # [n_jets, 1]
    jet_phi = jet_features["phi"].reshape(-1, 1)  # [n_jets, 1]
    
    # Create scaled features (apply scaling only to valid particles)
    scaled_pt = np.zeros_like(particle_pt)
    scaled_eta = np.zeros_like(particle_eta)
    scaled_phi = np.zeros_like(particle_phi)
    
    # Apply scaling where is_valid_particle is True
    scaled_pt[is_valid_particle] = particle_pt[is_valid_particle] / jet_pt.repeat(NUM_PARTICLES, axis=1)[is_valid_particle]
    scaled_eta[is_valid_particle] = particle_eta[is_valid_particle] - jet_eta.repeat(NUM_PARTICLES, axis=1)[is_valid_particle]
    
    # Handle phi periodicity (normalize to [-π, π])
    delta_phi = particle_phi - jet_phi.repeat(NUM_PARTICLES, axis=1)
    delta_phi = np.mod(delta_phi + np.pi, 2 * np.pi) - np.pi
    scaled_phi[is_valid_particle] = delta_phi[is_valid_particle]
    
    # Reshape back to original format
    scaled_features = np.zeros_like(particle_features)
    for jet_idx in range(n_jets):
        for particle_idx in range(NUM_PARTICLES):
            if is_valid_particle[jet_idx, particle_idx]:
                feature_idx = particle_idx * NUM_FEATURES
                scaled_features[jet_idx, feature_idx] = scaled_pt[jet_idx, particle_idx]
                scaled_features[jet_idx, feature_idx + 1] = scaled_eta[jet_idx, particle_idx]
                scaled_features[jet_idx, feature_idx + 2] = scaled_phi[jet_idx, particle_idx]
            else:
                # Keep invalid particles as -999
                feature_idx = particle_idx * NUM_FEATURES
                scaled_features[jet_idx, feature_idx:feature_idx + 3] = -999
    
    return scaled_features