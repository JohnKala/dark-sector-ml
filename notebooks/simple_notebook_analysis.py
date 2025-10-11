#!/usr/bin/env python3
"""
Simple notebook interface for sensitivity analysis.
Much cleaner than the complex sensitivity_analysis_notebook.py
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sensitivity_analysis import run_sensitivity_analysis


def simple_analysis(
    dataset_dir: str = "data/raw",
    epochs: int = 50,
    model_type: str = 'deepsets',
    show_plots: bool = True
):
    """
    Simple notebook-friendly analysis with minimal complexity.
    
    Parameters:
    -----------
    dataset_dir : str
        Directory containing dataset files
    epochs : int
        Training epochs
    model_type : str
        Model architecture
    show_plots : bool
        Display plots inline after analysis
    """
    
    # Find dataset files
    dataset_files = [str(f) for f in Path(dataset_dir).glob("*.h5")]
    
    if not dataset_files:
        print(f"❌ No .h5 files found in {dataset_dir}")
        return None
    
    # Run the analysis
    print("🚀 Starting Dark Sector Sensitivity Analysis...")
    
    results = run_sensitivity_analysis(
        dataset_files=dataset_files,
        output_dir="outputs/sensitivity_analysis",
        model_type=model_type,
        epochs=epochs,
        batch_size=256,
        run_individual=True,
        run_leave_one_out=True,
        save_models=False,  # Don't save large files in notebooks
        verbose=True,
        create_run_subdir=True
    )
    
    # Show plots inline if requested
    if show_plots:
        display(HTML("<h3>📈 Analysis Results</h3>"))
        
        output_dir = results['config']['output_dir']
        
        # Key plots to show
        plots = [
            'individual_combined_roc.png',
            'individual_cross_heatmap.png',
            'sensitivity_mDark.png'
        ]
        
        for plot_name in plots:
            plot_path = os.path.join(output_dir, plot_name)
            if os.path.exists(plot_path):
                # Display the plot
                img = plt.imread(plot_path)
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title(plot_name.replace('_', ' ').title())
                plt.tight_layout()
                plt.show()
    
    print(f"\n✅ Analysis complete! Results saved to: {results['config']['output_dir']}")
    
    return results


# Even simpler: just import the main function directly
def quick_notebook_run():
    """One-liner for quick testing."""
    return simple_analysis(epochs=10, show_plots=True)


def production_notebook_run():
    """One-liner for production analysis.""" 
    return simple_analysis(epochs=100, show_plots=True)
