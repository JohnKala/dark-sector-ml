"""
Input/Output utilities for saving and loading results.
"""

import pickle
import json
import os
from typing import Dict, Any
import numpy as np


def save_results(
    results: Dict[str, Any],
    filename: str,
    save_models: bool = False
) -> None:
    """
    Save evaluation results to files.
    
    Parameters:
    -----------
    results : dict
        Dictionary of results to save
    filename : str
        Base filename to use
    save_models : bool
        Whether to save the actual model objects
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Create copy of results for saving (excluding TensorFlow model objects)
    def prepare_for_json(obj):
        if isinstance(obj, dict):
            return {k: prepare_for_json(v) for k, v in obj.items() if k != 'model'}
        elif isinstance(obj, list):
            return [prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle numpy scalar types
        elif np.isscalar(obj) and isinstance(obj, np.generic):
            return obj.item()
        # Handle TensorFlow types
        elif str(type(obj)).startswith("<class 'tensorflow") or 'tensorflow.' in str(type(obj)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            # For objects that can't be serialized directly
            return {"_object_type": obj.__class__.__name__, "_string_repr": str(obj)}
        else:
            return obj
    
    # Clean data for JSON serialization
    save_data = prepare_for_json(results)
    
    # Save as JSON for readability
    try:
        with open(f"{filename}.json", 'w') as f:
            json.dump(save_data, f, indent=2)
    except TypeError as e:
        print(f"Warning: Could not serialize all data to JSON: {e}")
        print("Attempting to save with more aggressive type conversion...")
        
        # More aggressive conversion for problematic types
        def convert_to_basic_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_basic_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_basic_types(item) for item in obj]
            else:
                # Convert anything that's not a basic type to a string
                try:
                    json.dumps(obj)  # Test if it's JSON serializable
                    return obj
                except (TypeError, OverflowError):
                    return str(obj)
        
        # Try again with more aggressive conversion
        save_data_safe = convert_to_basic_types(save_data)
        with open(f"{filename}.json", 'w') as f:
            json.dump(save_data_safe, f, indent=2)
    
    # Save full object with pickle (including models if requested)
    if save_models:
        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump(results, f)
    
    print(f"Results saved to {filename}.json")
    if save_models:
        print(f"Full results with models saved to {filename}.pkl")