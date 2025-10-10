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
        elif hasattr(obj, '__dict__'):
            # For objects that can't be serialized directly
            return {"_object_type": obj.__class__.__name__, "_string_repr": str(obj)}
        else:
            return obj
    
    # Clean data for JSON serialization
    save_data = prepare_for_json(results)
    
    # Save as JSON for readability
    with open(f"{filename}.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Save full object with pickle (including models if requested)
    if save_models:
        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump(results, f)
    
    print(f"Results saved to {filename}.json")
    if save_models:
        print(f"Full results with models saved to {filename}.pkl")