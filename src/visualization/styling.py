"""
Visualization styling utilities for consistent model plotting.
"""

import re
import hashlib
import matplotlib.pyplot as plt
from typing import Dict, Any


def set_plot_style():
    """
    Set consistent plot style for all visualizations.
    """
    # Use a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    # Set figure properties
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    
    # Set line properties
    plt.rcParams['lines.linewidth'] = 2
    
    # Set grid properties
    plt.rcParams['grid.alpha'] = 0.3


def get_model_style(model_name: str) -> Dict[str, Any]:
    """
    Get consistent styling for a model based on its parameters.
    
    Parameters:
    -----------
    model_name : str
        Model or dataset name
        
    Returns:
    --------
    dict
        Styling dictionary with color, linestyle, and linewidth
    """
    # Debug: print the model name we're trying to style
    # print(f"Getting style for: {model_name}")
    
    # Keys are partial strings to match, not exact regex patterns
    style_mapping = {
        "mDark-1_rinv-0.3_alpha-peak": {"color": "blue", "linestyle": "-", "linewidth": 2},
        "mDark-5_rinv-0.3_alpha-peak": {"color": "red", "linestyle": "--", "linewidth": 2},
        "mDark-1_rinv-0.2_alpha-peak": {"color": "green", "linestyle": "-.", "linewidth": 2},
        "mDark-1_rinv-0.8_alpha-peak": {"color": "purple", "linestyle": ":", "linewidth": 2},
        "mDark-1_rinv-0.3_alpha-low": {"color": "orange", "linestyle": "-", "linewidth": 2},
        "mDark-1_rinv-0.3_alpha-high": {"color": "cyan", "linestyle": "--", "linewidth": 2},
        # For model names that include "model_" prefix
        "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak": {"color": "blue", "linestyle": "-", "linewidth": 2},
        "AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak": {"color": "red", "linestyle": "--", "linewidth": 2},
        "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.2_alpha-peak": {"color": "green", "linestyle": "-.", "linewidth": 2},
        "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.8_alpha-peak": {"color": "purple", "linestyle": ":", "linewidth": 2},
        "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-low": {"color": "orange", "linestyle": "-", "linewidth": 2},
        "AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high": {"color": "cyan", "linestyle": "--", "linewidth": 2},
        # Support model names with "model_" prefix
        "model_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak": {"color": "blue", "linestyle": "-", "linewidth": 2},
        "model_AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak": {"color": "red", "linestyle": "--", "linewidth": 2},
        "model_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.2_alpha-peak": {"color": "green", "linestyle": "-.", "linewidth": 2},
        "model_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.8_alpha-peak": {"color": "purple", "linestyle": ":", "linewidth": 2},
        "model_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-low": {"color": "orange", "linestyle": "-", "linewidth": 2},
        "model_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high": {"color": "cyan", "linestyle": "--", "linewidth": 2},
        # LOO model patterns
        "loo_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak": {"color": "blue", "linestyle": "-", "linewidth": 2},
        "loo_AutomatedCMS_mZprime-2000_mDark-5_rinv-0.3_alpha-peak": {"color": "red", "linestyle": "--", "linewidth": 2},
        "loo_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.2_alpha-peak": {"color": "green", "linestyle": "-.", "linewidth": 2},
        "loo_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.8_alpha-peak": {"color": "purple", "linestyle": ":", "linewidth": 2},
        "loo_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-low": {"color": "orange", "linestyle": "-", "linewidth": 2},
        "loo_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-high": {"color": "cyan", "linestyle": "--", "linewidth": 2}
    }
    
    # Match the model/dataset name to a style using partial matching
    for pattern, style in style_mapping.items():
        if pattern in model_name:
            # Debug: print when we find a match
            print(f"  Matched style: {pattern}")
            return style
    
    # If no direct match, try matching by parameters
    mDark_match = re.search(r'mDark-([0-9.]+)', model_name)
    rinv_match = re.search(r'rinv-([0-9.]+)', model_name)
    alpha_match = re.search(r'alpha-([a-z]+)', model_name)
    
    if mDark_match and rinv_match and alpha_match:
        mDark = mDark_match.group(1)
        rinv = rinv_match.group(1)
        alpha = alpha_match.group(1)
        
        # Try to find the closest match
        for pattern, style in style_mapping.items():
            if f"mDark-{mDark}" in pattern and f"rinv-{rinv}" in pattern and f"alpha-{alpha}" in pattern:
                # Debug: print when we find a parameter match
                # print(f"  Parameter matched style: {pattern}")
                return style
    
    # Default style for unmatched models with a variety of colors
    # Generate a deterministic color based on the model name
    hash_val = int(hashlib.md5(model_name.encode()).hexdigest(), 16)
    colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'darkviolet', 'darkcyan', 'darkmagenta']
    styles = ['-', '--', '-.', ':']
    
    color = colors[hash_val % len(colors)]
    style = styles[(hash_val // len(colors)) % len(styles)]
    
    # Debug: print when we use the default style
    # print(f"  Using hash-based style: {color}, {style}")
    return {"color": color, "linestyle": style, "linewidth": 2}


def format_model_name(name: str) -> str:
    """
    Format model name for display in plot legends to match reference.
    
    Example: "model_AutomatedCMS_mZprime-2000_mDark-1_rinv-0.3_alpha-peak"
    becomes "mDark-1.0_rinv-0.3_alpha-peak"
    
    Parameters:
    -----------
    name : str
        Original model or dataset name
        
    Returns:
    --------
    str
        Formatted name for display
    """
    # Extract model parameters using regex
    mDark_match = re.search(r'mDark-([0-9.]+)', name)
    rinv_match = re.search(r'rinv-([0-9.]+)', name)
    alpha_match = re.search(r'alpha-([a-z]+)', name)
    
    # If we can extract all three parameters, format them consistently
    if mDark_match and rinv_match and alpha_match:
        mDark = mDark_match.group(1)
        rinv = rinv_match.group(1)
        alpha = alpha_match.group(1)
        
        # Format mDark with .0 if it's an integer
        if mDark.isdigit():
            mDark = f"{mDark}.0"
            
        return f"mDark-{mDark}_rinv-{rinv}_alpha-{alpha}"
    
    # Try to match complete pattern directly
    complete_pattern = r'(mDark-[0-9.]+_rinv-[0-9.]+_alpha-[a-z]+)'
    complete_match = re.search(complete_pattern, name)
    if complete_match:
        return complete_match.group(1)
    
    # Look for partial parameters
    if mDark_match:
        parts = []
        parts.append(f"mDark-{mDark_match.group(1)}")
        if rinv_match:
            parts.append(f"rinv-{rinv_match.group(1)}")
        if alpha_match:
            parts.append(f"alpha-{alpha_match.group(1)}")
        return "_".join(parts)
    
    # Check for specific patterns in model name
    if "model_" in name:
        return name.replace("model_", "")
    if "loo_" in name:
        return name.replace("loo_", "")
    
    # If no parameter pattern matched, return a simplified name
    # Remove common prefixes
    simplified = name
    prefixes = ["AutomatedCMS_", "model_", "loo_"]
    for prefix in prefixes:
        if simplified.startswith(prefix):
            simplified = simplified.replace(prefix, "", 1)
    
    # If it's the SM file, return a shorter name
    if "NominalSM" in name:
        return "Standard Model"
    
    return simplified
